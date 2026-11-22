"""Test d'uniformité des rangs (RUT, Zhu et al. ICLR 2026) en BOÎTE BLANCHE, une hypothèse empilée à la fois. Aucune requête.

Pour une famille (--only A|B|C, ou R pour les ref-* et SmolLM2-360M), chaque hypothèse H = poids x consigne x décodage
est simulée localement : K réponses par prompt, mêmes prompts que les endpoints. Chaque réponse est résumée par des
statistiques fixes, calculées de la même façon pour les endpoints et les simulations :
  f_ll   log-vraisemblance moyenne par token sous le modèle bf16, contexte par défaut, T = 0,7 (tokens >= 8)
  f_le   moyenne de (log p + entropie) sous le même modèle
  f_fp8  moyenne par token de log p(fp8 par tenseur) - log p(bf16)
Pour un endpoint E et une hypothèse H : rang aléatoirisé de f(réponse de E) parmi les K simulations de H au même prompt.
Si E est H, les rangs sont EXACTEMENT uniformes (échangeabilité), quel que soit le prompt : statistique de Cramér-von
Mises et p-valeur par Monte-Carlo. Une hypothèse est acceptée pour E si aucune statistique ne la rejette.
Hypothèses (tag = poids_consigne_décodage) :
  bf16_def_t07   modèle de base, gabarit par défaut, T = 0,7               (S plain / watermark)
  bf16_pool_t07  modèle de base + consigne tirée du réservoir (wb_pool.py)  (S consigne)
  bf16_def_t10   modèle de base, T = 1,0                                     (S température)
  bf16_def_trc   modèle de base, T = 0,9, top-p 0,97                         (S décodage tronqué)
  fp8t_def_t07   fp8 par tenseur                                              (T fp8)
  fp8t_pool_t07  fp8 par tenseur + consigne du réservoir                      (T fp8 empilé)
  fp8t_def_trc   fp8 par tenseur + décodage tronqué                           (T fp8 empilé)
Sortie : <out>/rut_<fam>.jsonl (une ligne par réponse) ; --summary <fichier> affiche le tableau des p-valeurs (sans GPU).
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
sys.path.insert(0, str(H / "local"))

HYPS = [
    ("bf16_def_t07", "bf16", "def", dict(temperature=0.7, top_p=1.0)),
    ("bf16_pool_t07", "bf16", "pool", dict(temperature=0.7, top_p=1.0)),
    ("bf16_def_t10", "bf16", "def", dict(temperature=1.0, top_p=1.0)),
    ("bf16_def_trc", "bf16", "def", dict(temperature=0.9, top_p=0.97)),
    ("fp8t_def_t07", "fp8t", "def", dict(temperature=0.7, top_p=1.0)),
    ("fp8t_pool_t07", "fp8t", "pool", dict(temperature=0.7, top_p=1.0)),
    ("fp8t_def_trc", "fp8t", "def", dict(temperature=0.9, top_p=0.97)),
]
STATS = ("f_ll", "f_le", "f_fp8")
REF_FILES = ["reference_answers.jsonl", "ref_consignes_answers.jsonl", "ref_ep_round1_answers.jsonl", "ref_ep_round3_answers.jsonl",
             "ref_ep_round4_answers.jsonl", "ref_ep_round6_answers.jsonl"]
SKIP_IDS = {"71", "72", "73"}
MIN_POS = 8


# ----------------------------------------------------------------------------------------------------- partie GPU
def load_rows(fam, extra=None):
    rows, prompts = [], {}
    if extra:  # endpoints synthétiques à vérité connue (local/synth_gen.py), nommés syn-<fam>-...
        for l in open(extra, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r["endpoint"].startswith(f"syn-{fam}-") and r["response"].strip():
                    rows.append(dict(kind="ep", name=r["endpoint"], pid=str(r["id"]), prompt=r["prompt"], response=r["response"]))
    if fam == "R":
        for fi, name in enumerate(REF_FILES):
            p = H / name
            if not p.exists():
                continue
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if str(r["endpoint"]).startswith("ref-") and r["response"].strip():
                        pid = f"f{fi}:{r['id']}"
                        prompts[pid] = r["prompt"]
                        rows.append(dict(kind="ep", name=r["endpoint"], pid=pid, prompt=r["prompt"], response=r["response"]))
        return rows, prompts
    famof = {r["endpoint"]: r["family"] for r in csv.DictReader(open(H / "submission_v8.csv", encoding="utf-8"))}
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if str(r["id"]) in SKIP_IDS or famof.get(r["endpoint"]) != fam or not r["response"].strip():
                continue
            prompts[str(r["id"])] = r["prompt"]
            rows.append(dict(kind="ep", name=r["endpoint"], pid=str(r["id"]), prompt=r["prompt"], response=r["response"]))
    return rows, prompts


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from score_wb_r4 import FAMILY_MODELS
    from wb_dose import chat_ids, chat_text, pad_batch
    from wb_pool import OPENERS, SENTS
    from wb_stack import quantize, restore
    from wb_quant import check_fp8

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=["A", "B", "C", "R"])
    ap.add_argument("--out", default=str(H / "jureca_retour12"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", default="", help="réponses synthétiques à ajouter (synth_<fam>.jsonl)")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fam = a.only
    repo = json.load(open(H / "jureca_retour7" / "refid.json"))["best"] if fam == "R" else FAMILY_MODELS[fam][1]
    rows, prompts = load_rows(fam, a.extra or None)
    if a.limit:
        keep = set(sorted(prompts)[: a.limit])
        prompts = {k: v for k, v in prompts.items() if k in keep}
        rows = [r for r in rows if r["pid"] in keep]
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    check_fp8(dev)
    print(f"famille {fam} ({repo}) : {len(rows)} réponses, {len(prompts)} prompts, K = {a.k}", flush=True)
    rng = random.Random(0)

    @torch.no_grad()
    def gen(texts, dec):
        tok.padding_side = "left"
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [None] * len(texts)
        for s in range(0, len(order), a.bs):
            idx = order[s : s + a.bs]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**enc, max_new_tokens=256, do_sample=True, top_k=0, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, **dec)
            for i, t in zip(idx, tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)):
                res[i] = t
        tok.padding_side = "right"
        return res

    pids = sorted(prompts)
    for tag, weights, ctx, dec in HYPS:
        saved = quantize(model, per_channel=False) if weights == "fp8t" else None
        items = [(p, (" ".join([rng.choice(OPENERS)] + rng.sample(SENTS, rng.randint(2, 4))) if ctx == "pool" else None)) for p in pids for _ in range(a.k)]
        texts = [chat_text(tok, prompts[p], sy) for p, sy in items]
        for (p, sy), resp in zip(items, gen(texts, dec)):
            rows.append(dict(kind="sim", name=tag, pid=p, prompt=prompts[p], response=resp))
        if saved is not None:
            restore(model, saved)
        print(f"  simulation {tag} : {len(items)} réponses", flush=True)

    for r in rows:
        r["ctx"] = chat_ids(tok, r["prompt"])
        r["y"] = tok(r["response"], add_special_tokens=False)["input_ids"][:260]

    @torch.no_grad()
    def score(key):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ctx"]) + len(rows[i]["y"]))
        for s in range(0, len(order), 16):
            batch = [rows[i] for i in order[s : s + 16]]
            ids, att = pad_batch(batch, tok.pad_token_id, dev)
            logits = model(input_ids=ids, attention_mask=att).logits
            for j, r in enumerate(batch):
                s0, n = len(r["ctx"]), len(r["y"])
                if n <= MIN_POS:
                    continue
                logp = torch.log_softmax(logits[j, s0 - 1 + MIN_POS : s0 - 1 + n].float() / 0.7, dim=1)
                lp = logp.gather(1, ids[j, s0 + MIN_POS : s0 + n][:, None]).squeeze(1)
                r[key] = float(lp.sum())
                if key == "lp_bf16":
                    r["le"] = float((lp - (logp.exp() * logp).sum(1)).sum())
                    r["n"] = n - MIN_POS
            del logits

    score("lp_bf16")
    saved = quantize(model, per_channel=False)
    score("lp_fp8t")
    restore(model, saved)
    with open(outdir / f"rut_{fam}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            if "lp_fp8t" in r and r.get("n"):
                n = r["n"]
                fh.write(json.dumps(dict(kind=r["kind"], name=r["name"], pid=r["pid"], n=n, f_ll=r["lp_bf16"] / n, f_le=r["le"] / n,
                                         f_fp8=(r["lp_fp8t"] - r["lp_bf16"]) / n)) + "\n")
    print(f"famille {fam} : terminé", flush=True)


# -------------------------------------------------------------------------------------------- résumé (sans GPU)
def cvm(u):
    u = np.sort(u)
    n = len(u)
    return 1.0 / (12 * n) + float((((2 * np.arange(1, n + 1) - 1) / (2 * n) - u) ** 2).sum())


def summary(path):
    rng = np.random.default_rng(0)
    recs = [json.loads(l) for l in open(path)]
    sims = defaultdict(lambda: defaultdict(list))
    eps = defaultdict(list)
    for r in recs:
        (sims[r["name"]][r["pid"]] if r["kind"] == "sim" else eps[r["name"]]).append(r)
    tags = [h[0] for h in HYPS if h[0] in sims]
    null = {}

    def pval(w, n):
        if n not in null:
            null[n] = np.array([cvm(rng.random(n)) for _ in range(2000)])
        return float((null[n] >= w).mean())

    def ranks(rs, tag, st):
        u = []
        for r in rs:
            pool = [x[st] for x in sims[tag].get(r["pid"], [])]
            if len(pool) < 3:
                continue
            below = sum(v < r[st] for v in pool)
            ties = sum(v == r[st] for v in pool)
            u.append((below + rng.random() * (ties + 1)) / (len(pool) + 1))
        return np.array(u)

    # contrôle : une simulation de chaque hypothèse, retirée de son propre pool, doit donner des rangs uniformes
    print(f"\n{path}\np-valeur min sur {STATS} (Cramér-von Mises, rangs parmi les K simulations ; > 0,01 = hypothèse acceptée)")
    print("  source         " + " ".join(f"{t:>14s}" for t in tags))
    names = sorted(eps)
    for name in names:
        cells = []
        for tag in tags:
            ps = []
            for st in STATS:
                u = ranks(eps[name], tag, st)
                ps.append(pval(cvm(u), len(u)) if len(u) >= 10 else float("nan"))
            p = min(ps)
            cells.append(f"{p:6.3f}{'*' if p > 0.01 else ' '}" + f"[{STATS[int(np.nanargmin(ps))][2:]:>3s}]")
        print(f"  {name:14s} " + " ".join(f"{c:>14s}" for c in cells))
    print("  (* = acceptée ; entre crochets, la statistique qui rejette le plus)")
    print("\ncontrôle : la 1re simulation de chaque prompt, classée parmi les autres simulations de la MÊME hypothèse (attendu : p > 0,01)")
    for tag in tags:
        ps = []
        for st in STATS:
            u = []
            for pid, pool in sims[tag].items():
                if len(pool) < 4:
                    continue
                x, rest = pool[0][st], [y[st] for y in pool[1:]]
                below, ties = sum(v < x for v in rest), sum(v == x for v in rest)
                u.append((below + rng.random() * (ties + 1)) / (len(rest) + 1))
            ps.append(pval(cvm(np.array(u)), len(u)) if len(u) >= 10 else float("nan"))
        print(f"  {tag:14s} " + "  ".join(f"{s}: p={p:.3f}" for s, p in zip(STATS, ps)))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--summary":
        for f in sys.argv[2:]:
            summary(f)
    else:
        main()
