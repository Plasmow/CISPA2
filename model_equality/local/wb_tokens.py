"""Test de support par token (nouvelle approche, boîte blanche, aucune requête vers les endpoints).

Idée : un endpoint qui sert les poids de base ne peut émettre que des tokens situés dans le noyau d'échantillonnage du
modèle de base (top-p : masse des tokens plus probables < p ; top-k : rang <= k), quel que soit son décodage. Un LoRA
ou une quantification déplace les logits et fait sortir des tokens de ce noyau. On mesure donc, pour chaque token de
chaque réponse, sous le modèle de base bf16 :
  - lp, ent, rank : log-probabilité, entropie et rang à T = 1 ;
  - pab[r, t] : masse des tokens strictement plus probables, à la température TEMPS[t], sans (r = 0) ou avec (r = 1)
    la pénalité de répétition 1,05 du generation_config de Qwen (appliquée avant la température, comme HF et vLLM) ;
  - rank_rp : rang avec la pénalité de répétition.
Lignes notées pour une famille (--only A|B|C) : toutes les réponses ep-*, les références locales (plain, fp8sim, int8,
nf4, T = 1) et, pour A et C, les variantes locales étiquetées (LoRA, watermark, system prompt, persona...).
Sorties : <out>/tokens_<fam>.npz et <out>/meta_<fam>.jsonl (une ligne par réponse, dans l'ordre de `row`).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
from score_wb_r4 import FAMILY_MODELS, load_quantized  # noqa: E402

TEMPS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
REP_PENALTIES = (1.0, 1.05)
LOCAL_VARIANTS = {"A": (H / "out", "llama32__"), "C": (H / "brief_retour3" / "local_variants_qwen25_3b", "qwen25_3b__")}
CAND_DIRS = (H / "brief_retour2" / "cands40", H / "brief_retour3" / "cands_r4", H / "jureca_retour5" / "cands_r6")


def collect_rows(fcode, short, groups, n_local, n_cand_rep):
    fam_map = {r["endpoint"]: r["family"] for r in csv.DictReader(open(groups, encoding="utf-8"))}
    rows = []
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if fam_map.get(r["endpoint"]) == fcode and r["response"].strip():
                rows.append(dict(kind="ep", name=r["endpoint"], id=str(r["id"]), repeat=int(r.get("repeat", 0)),
                                 prompt=r["prompt"], response=r["response"]))
    for d in CAND_DIRS:
        for cf in sorted(d.glob(f"{short}__*.jsonl")):
            for l in open(cf, encoding="utf-8"):
                r = json.loads(l)
                if int(r["repeat"]) < n_cand_rep and r["response"].strip():
                    rows.append(dict(kind="cand", name=cf.stem.split("__")[1], id=str(r["id"]), repeat=int(r["repeat"]),
                                     prompt=r["prompt"], response=r["response"]))
    if fcode in LOCAL_VARIANTS:
        d, prefix = LOCAL_VARIANTS[fcode]
        for vf in sorted(d.glob(f"{prefix}*.jsonl")):
            kept = 0
            for l in open(vf, encoding="utf-8"):
                r = json.loads(l)
                if int(r.get("repeat", 0)) == 0 and r["response"].strip() and kept < n_local:
                    rows.append(dict(kind="local", name=vf.stem.split("__", 1)[1], id=str(r["id"]), repeat=0,
                                     prompt=r["prompt"], response=r["response"]))
                    kept += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(FAMILY_MODELS))
    ap.add_argument("--out", default=str(H / "jureca_retour6"))
    ap.add_argument("--groups", default=str(H / "submission_v8.csv"))
    ap.add_argument("--n-local", type=int, default=60)
    ap.add_argument("--n-cand-rep", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="ne note que les N premières lignes (essai)")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    short, repo = FAMILY_MODELS[a.only]
    rows = collect_rows(a.only, short, a.groups, a.n_local, a.n_cand_rep)
    if a.limit:
        rows = rows[: a.limit]
    print(f"famille {a.only} ({repo}) : {len(rows)} réponses à noter "
          f"(ep {sum(r['kind'] == 'ep' for r in rows)}, cand {sum(r['kind'] == 'cand' for r in rows)}, local {sum(r['kind'] == 'local' for r in rows)})", flush=True)

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_quantized(repo, "bf16")
    arrays, n_tok_row = token_scores(model, tok, rows, a.batch)
    save(outdir, a.only, rows, arrays, n_tok_row)
    print(f"famille {a.only} : terminé ({sum(n_tok_row)} tokens)", flush=True)


def token_scores(model, tok, rows, batch=8, verbose=True):
    """Statistiques par token de chaque réponse de `rows` (clés prompt, response) sous `model`."""
    pad = tok.pad_token_id
    dev = next(model.parameters()).device
    temps = torch.tensor(TEMPS, device=dev)
    ctx_cache = {}

    def ctx(p):
        if p not in ctx_cache:
            msgs = [{"role": "user", "content": p}]
            try:
                s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ctx_cache[p] = tok(s, add_special_tokens=False)["input_ids"]
        return ctx_cache[p]

    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["response"]))
    acc = {k: [] for k in ("row", "pos", "tok", "lp", "ent", "rank", "rank_rp", "pab", "ptok")}
    n_tok_row = [0] * len(rows)
    for start in range(0, len(order), batch):
        idxs = order[start : start + batch]
        seqs, spans = [], []
        for i in idxs:
            c = ctx(rows[i]["prompt"])
            y = tok(rows[i]["response"], add_special_tokens=False)["input_ids"][:260]
            seqs.append(c + y)
            spans.append((len(c), len(c) + len(y)))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long, device=dev)
        att = torch.zeros((len(seqs), T), dtype=torch.long, device=dev)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
            att[j, : len(s)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits
        for j, i in enumerate(idxs):
            s0, s1 = spans[j]
            n = s1 - s0
            n_tok_row[i] = n
            if n <= 0:
                continue
            z = logits[j, s0 - 1 : s1 - 1].float()  # (n, V)
            y = ids[j, s0:s1]
            logp = torch.log_softmax(z, dim=1)
            lp = logp.gather(1, y[:, None]).squeeze(1)
            ent = -(logp.exp() * logp).sum(1)
            # tokens déjà vus (contexte + réponse jusqu'à t-1) pour la pénalité de répétition
            seen = torch.zeros_like(z, dtype=torch.bool)
            seen[:, ids[j, :s0]] = True
            if n > 1:
                tr = torch.tril_indices(n, n, offset=-1, device=dev)
                seen[tr[0], y[tr[1]]] = True
            pab = torch.empty((n, len(REP_PENALTIES), len(TEMPS)), device=dev)
            ptok = torch.empty((n, len(REP_PENALTIES), len(TEMPS)), device=dev)
            ranks = []
            for ri, rp in enumerate(REP_PENALTIES):
                zz = z if rp == 1.0 else torch.where(seen, torch.where(z > 0, z / rp, z * rp), z)
                zt = zz.gather(1, y[:, None])
                above = zz > zt
                ranks.append(above.sum(1) + 1)
                for ti in range(len(TEMPS)):
                    p = torch.softmax(zz / temps[ti], dim=1)
                    pab[:, ri, ti] = (p * above).sum(1)
                    ptok[:, ri, ti] = p.gather(1, y[:, None]).squeeze(1)
            acc["row"].append(np.full(n, i, dtype=np.int32))
            acc["pos"].append(np.arange(n, dtype=np.int16))
            acc["tok"].append(y.cpu().numpy().astype(np.int32))
            acc["lp"].append(lp.cpu().numpy().astype(np.float32))
            acc["ent"].append(ent.cpu().numpy().astype(np.float32))
            acc["rank"].append(ranks[0].cpu().numpy().astype(np.int32))
            acc["rank_rp"].append(ranks[1].cpu().numpy().astype(np.int32))
            acc["pab"].append(pab.cpu().numpy().astype(np.float32))
            acc["ptok"].append(ptok.cpu().numpy().astype(np.float32))
        del logits
        if verbose and (start // batch) % 25 == 0:
            print(f"  {start}/{len(order)}", flush=True)

    return {k: np.concatenate(v) for k, v in acc.items()}, n_tok_row


def save(outdir, tag, rows, arrays, n_tok_row):
    np.savez_compressed(outdir / f"tokens_{tag}.npz", temps=np.array(TEMPS), rep_penalties=np.array(REP_PENALTIES), **arrays)
    with open(outdir / f"meta_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps(dict(row=i, kind=r["kind"], name=r["name"], id=r["id"], repeat=r["repeat"], n_tok=n_tok_row[i]),
                                ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
