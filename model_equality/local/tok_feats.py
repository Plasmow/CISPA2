"""Caractéristiques PAR TOKEN pour le classifieur profond (local/deep_clf.py). Aucune requête vers les endpoints.

Pour chaque réponse, sous le modèle de base exact (contexte : gabarit de chat par défaut), pour chaque token y_t :
  0  log p_bf16(y_t) à T = 0,7        1  entropie à T = 0,7        2  log(1 + rang de y_t) (T = 1)
  3  log p_fp8sim(y_t) - log p_bf16(y_t) à T = 0,7                  4  log p_int8(y_t) - log p_bf16(y_t) à T = 0,7
Sources :
  --dgx <base>   générations DGX à vérité connue (out/<base>__*.jsonl) : 80 prompts x réponses 0..2 par variante ;
                 étiquette 1 si le modèle = la base (déguisement), 0 sinon (quantification, LoRA, combinaisons)
  --fam A|B|C    vrais endpoints (sondes 71-73 exclues ; étiquette -1) + synthétiques jureca_retour15/synth_<fam>.jsonl
                 (étiquette selon synth_truth_<fam>.csv : 1 si modèle <fam>-S)
Sortie : <out>/tok_<tag>.npz (feats float16 [N_tok, 5], off int64 [N_rep + 1]) et <out>/tok_<tag>.jsonl (une ligne par réponse).
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
sys.path.insert(0, str(H / "local"))
from score_wb_r4 import FAMILY_MODELS, load_quantized  # noqa: E402
from wb_dose import chat_ids, pad_batch  # noqa: E402

DGX_REPOS = {"llama32": "unsloth/Llama-3.2-1B-Instruct", "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
             "olmo2": "allenai/OLMo-2-0425-1B-Instruct", "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}
MAX_TOK = 256


def rows_dgx(base):
    truth = {r["endpoint"]: r for r in csv.DictReader(open(H / "out" / "truth.csv", encoding="utf-8"))}
    plain = [json.loads(l) for l in open(H / "out" / f"{base}__plain.jsonl", encoding="utf-8")]
    keep = set(random.Random(0).sample(sorted({r["id"] for r in plain}), 80))
    rows = []
    for f in sorted((H / "out").glob(f"{base}__*.jsonl")):
        var = f.stem.split("__", 1)[1]
        lab = int(truth.get(f.stem, {}).get("model") == base)
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] not in keep or not r["response"].strip():
                continue
            rep = int(r.get("repeat", 0))
            if rep <= 2:
                rows.append(dict(src=f"{base}:{var}", label=lab, group=base, id=r["id"], prompt=r["prompt"], response=r["response"]))
            if var == "plain" and rep >= 4:  # RÉFÉRENCE : le modèle de base non modifié sur le même prompt
                rows.append(dict(src=f"REF:{base}", label=-3, group=base, id=r["id"], prompt=r["prompt"], response=r["response"]))
    return rows


def rows_q(tag, prefix="qgen"):
    """Données spéciales Qwen (jureca_retour18/20 : <prefix>_<tag>*.jsonl) + références plain_t07 + vrais endpoints de la famille."""
    fam = {"15": "B", "3": "C", "1": "A"}[tag]
    rows = [r for r in rows_fam(fam) if r["src"].startswith("REF:") or r["src"].startswith("ep-")]
    d = H / ("jureca_retour20" if prefix == "qgen2" else "jureca_retour18")
    for f in sorted(d.glob(f"{prefix}_{tag}*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["response"].strip():
                rows.append(dict(src=f"{r['endpoint']}#{f.stem[-1]}" if prefix == "qgen2" else r["endpoint"], label=r["label"], group=fam,
                                 id=str(r["id"]), prompt=r["prompt"], response=r["response"], adapter=r["group"], kind=r["kind"]))
    return rows


def rows_fam(fam):
    fam_map = {r["endpoint"]: r["family"] for r in csv.DictReader(open(H / "submission_v8.csv", encoding="utf-8"))}
    rows = []
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if fam_map.get(r["endpoint"]) == fam and r["response"].strip() and str(r["id"]) not in ("71", "72", "73"):
            rows.append(dict(src=r["endpoint"], label=-1, group=fam, id=str(r["id"]), prompt=r["prompt"], response=r["response"]))
    short = FAMILY_MODELS[fam][0]
    for d in (H / "brief_retour2" / "cands40", H / "brief_retour3" / "cands_r4", H / "jureca_retour5" / "cands_r6"):
        f = d / f"{short}__plain_t07.jsonl"
        if f.exists():
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["response"].strip():
                    rows.append(dict(src=f"REF:{fam}", label=-3, group=fam, id=str(r["id"]), prompt=r["prompt"], response=r["response"]))
    sdir = H / "jureca_retour15"
    truth = {}
    if (sdir / f"synth_truth_{fam}.csv").exists():
        truth = {e: m for e, f_, m in csv.reader(open(sdir / f"synth_truth_{fam}.csv"))}
    if (sdir / f"synth_{fam}.jsonl").exists():
        for l in open(sdir / f"synth_{fam}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["response"].strip():
                lab = int(truth[r["endpoint"]].endswith("-S")) if r["endpoint"] in truth else -2
                rows.append(dict(src=r["endpoint"], label=lab, group=fam, id=str(r["id"]), prompt=r["prompt"], response=r["response"]))
    return rows


@torch.no_grad()
def token_lp(model, tok, rows, bs, full):
    dev = next(model.parameters()).device
    out = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ctx"]) + len(rows[i]["y"]))
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        batch = [rows[i] for i in idx]
        ids, att = pad_batch(batch, tok.pad_token_id, dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        for j, (i, r) in enumerate(zip(idx, batch)):
            s0, n = len(r["ctx"]), len(r["y"])
            z = logits[j, s0 - 1 : s0 - 1 + n].float()
            y = ids[j, s0 : s0 + n]
            logp = torch.log_softmax(z / 0.7, dim=1)
            lp = logp.gather(1, y[:, None]).squeeze(1)
            if full:
                ent = -(logp.exp() * logp).sum(1)
                rank = (z > z.gather(1, y[:, None])).sum(1).float()
                out[i] = torch.stack([lp, ent, torch.log1p(rank)], 1).cpu().numpy()
            else:
                out[i] = lp.cpu().numpy()
        del logits
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgx", choices=sorted(DGX_REPOS))
    ap.add_argument("--fam", choices=["A", "B", "C"])
    ap.add_argument("--qgen", choices=["15", "3", "1"], help="données spéciales (15 = Qwen-1,5B, 3 = Qwen-3B, 1 = Llama)")
    ap.add_argument("--qdata", default="qgen", choices=["qgen", "qgen2"], help="qgen2 = qwen_special2.py (robuste)")
    ap.add_argument("--nf4", action="store_true", help="ajoute le canal log p_nf4 - log p_bf16")
    ap.add_argument("--out", default=str(H / "jureca_retour17"))
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.qgen:
        tag = f"q2_{a.qgen}" if a.qdata == "qgen2" else f"q{a.qgen}"
        repo, rows = FAMILY_MODELS[{"15": "B", "3": "C", "1": "A"}[a.qgen]][1], rows_q(a.qgen, a.qdata)
    elif a.dgx:
        tag, repo, rows = f"dgx_{a.dgx}", DGX_REPOS[a.dgx], rows_dgx(a.dgx)
    else:
        tag, repo, rows = f"fam_{a.fam}", FAMILY_MODELS[a.fam][1], rows_fam(a.fam)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for r in rows:
        r["ctx"] = chat_ids(tok, r["prompt"])
        r["y"] = tok(r["response"], add_special_tokens=False)["input_ids"][:MAX_TOK]
    rows = [r for r in rows if len(r["y"]) >= 8]
    print(f"{tag} ({repo}) : {len(rows)} réponses", flush=True)
    base = {}
    for qk in ("bf16", "fp8sim", "int8") + (("nf4",) if a.nf4 else ()):
        m = load_quantized(repo, qk)
        base[qk] = token_lp(m, tok, rows, a.bs, full=(qk == "bf16"))
        del m
        torch.cuda.empty_cache()
        print(f"  passe {qk} terminée", flush=True)
    feats, off = [], [0]
    for i in range(len(rows)):
        b = base["bf16"][i]
        cols = [b, (base["fp8sim"][i] - b[:, 0])[:, None], (base["int8"][i] - b[:, 0])[:, None]]
        if a.nf4:
            cols.append((base["nf4"][i] - b[:, 0])[:, None])
        f = np.concatenate(cols, 1)
        feats.append(f.astype(np.float16))
        off.append(off[-1] + len(f))
    np.savez_compressed(out / f"tok_{tag}.npz", feats=np.concatenate(feats), off=np.array(off, dtype=np.int64))
    with open(out / f"tok_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("src", "label", "group", "id", "adapter", "kind") if k in r}) + "\n")
    print(f"{tag} : terminé", flush=True)


if __name__ == "__main__":
    main()
