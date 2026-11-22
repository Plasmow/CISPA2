"""Notation boîte blanche des générations DGX à vérité connue (out/<base>__<variante>.jsonl, out/truth.csv), avec les mêmes
statistiques que les vrais endpoints (score_wb_r4 : bf16, fp8sim, int8, nf4). Aucune requête vers les endpoints.
Bases : llama32 (= modèle de base de la famille A), qwen25 (= famille B), olmo2 et smollm2 (autres modèles, pour généraliser).
Par variante : N_PROMPTS prompts tirés une fois pour toutes, réponses 0..2 (« pseudo-endpoint ») ; la variante plain fournit
en plus ses réponses 4..7 comme RÉFÉRENCE par prompt (rôle « ref »), à la manière des cands_* des vrais endpoints.
Sortie : <out>/dgx_<base>.jsonl, une ligne par réponse (base, variant, model, kind, role, id, repeat, <q>_<stat>, delta_<q>_ll40).
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
from score_wb_r4 import load_quantized, score_rows_under_model  # noqa: E402

REPOS = {"llama32": "unsloth/Llama-3.2-1B-Instruct", "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
         "olmo2": "allenai/OLMo-2-0425-1B-Instruct", "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}
N_PROMPTS = 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=sorted(REPOS))
    ap.add_argument("--out", default=str(H / "jureca_retour16"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {r["endpoint"]: r for r in csv.DictReader(open(H / "out" / "truth.csv", encoding="utf-8"))}
    files = sorted((H / "out").glob(f"{a.base}__*.jsonl"))
    plain = [json.loads(l) for l in open(H / "out" / f"{a.base}__plain.jsonl", encoding="utf-8")]
    pids = sorted({r["id"] for r in plain})
    keep = set(random.Random(0).sample(pids, N_PROMPTS))
    rows = []
    for f in files:
        var = f.stem.split("__", 1)[1]
        t = truth.get(f.stem, {})
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] not in keep or not r["response"].strip():
                continue
            rep = int(r.get("repeat", 0))
            role = "ep" if rep <= 2 else ("ref" if var == "plain" and rep >= 4 else None)
            if role:
                rows.append({"base": a.base, "variant": var, "model": t.get("model", ""), "kind": t.get("kind", ""), "role": role,
                             "id": r["id"], "repeat": rep, "prompt": r["prompt"], "response": r["response"]})
    print(f"{a.base} : {len(files)} variantes, {len(rows)} réponses à noter", flush=True)
    repo = REPOS[a.base]
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    per_q = {}
    for qkind in ("bf16", "fp8sim", "int8", "nf4"):
        print(f"  notation sous {qkind}", flush=True)
        mdl = load_quantized(repo, qkind)
        per_q[qkind] = score_rows_under_model(mdl, tok, rows)
        del mdl
        torch.cuda.empty_cache()
    with open(out / f"dgx_{a.base}.jsonl", "w", encoding="utf-8") as fh:
        for idx, r in enumerate(rows):
            rec = {k: r[k] for k in ("base", "variant", "model", "kind", "role", "id", "repeat")}
            for qkind, res in per_q.items():
                if res[idx].get("n_tok", 0) > 0:
                    for k, v in res[idx].items():
                        rec[f"{qkind}_{k}"] = v
            if "bf16_mean_loglik_40" in rec:
                for qkind in ("fp8sim", "int8", "nf4"):
                    if f"{qkind}_mean_loglik_40" in rec:
                        rec[f"delta_{qkind}_ll40"] = rec[f"{qkind}_mean_loglik_40"] - rec["bf16_mean_loglik_40"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{a.base} : terminé", flush=True)


if __name__ == "__main__":
    main()
