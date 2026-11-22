"""Notation boîte blanche (bf16, fp8sim, int8, nf4 ; mêmes statistiques que v8 / wb_round6.py) de TOUTES les réponses d'une
famille : vrais endpoints (ep_answers.jsonl, sondes 71-73 exclues) et endpoints synthétiques à vérité connue
(<synth>/synth_<fam>.jsonl, local/synth_gen.py). Aucune requête vers les endpoints.
Sortie : <out>/wball_<fam>.jsonl, une ligne par réponse (champs endpoint, id, repeat, <q>_<stat>, delta_<q>_ll40).
But : calculer les mêmes caractéristiques par endpoint pour les synthétiques (vérité connue) et les vrais, puis
apprendre un classifieur sur les premiers et l'appliquer aux seconds.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
from score_wb_r4 import FAMILY_MODELS, load_quantized, score_rows_under_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=["A", "B", "C"])
    ap.add_argument("--synth", default=str(H / "jureca_retour15"))
    ap.add_argument("--out", default=str(H / "jureca_retour15"))
    a = ap.parse_args()
    fcode, repo = a.only, FAMILY_MODELS[a.only][1]
    fam_map = {r["endpoint"]: r["family"] for r in csv.DictReader(open(H / "submission_v8.csv", encoding="utf-8"))}
    rows = []
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if fam_map.get(r["endpoint"]) == fcode and r["response"].strip() and str(r["id"]) not in ("71", "72", "73"):
            rows.append({"endpoint": r["endpoint"], "id": int(r["id"]), "repeat": int(r.get("repeat", 0)), "prompt": r["prompt"], "response": r["response"]})
    for l in open(Path(a.synth) / f"synth_{fcode}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["response"].strip():
            rows.append({"endpoint": r["endpoint"], "id": int(r["id"]), "repeat": int(r["repeat"]), "prompt": r["prompt"], "response": r["response"]})
    print(f"famille {fcode} ({repo}) : {len(rows)} réponses (réelles + synthétiques)", flush=True)
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
    with open(Path(a.out) / f"wball_{fcode}.jsonl", "w", encoding="utf-8") as fh:
        for idx, r in enumerate(rows):
            rec = {"family": fcode, "endpoint": r["endpoint"], "id": r["id"], "repeat": r["repeat"]}
            for qkind, res in per_q.items():
                if res[idx].get("n_tok", 0) > 0:
                    for k, v in res[idx].items():
                        rec[f"{qkind}_{k}"] = v
            if "bf16_mean_loglik_40" in rec:
                for qkind in ("fp8sim", "int8", "nf4"):
                    if f"{qkind}_mean_loglik_40" in rec:
                        rec[f"delta_{qkind}_ll40"] = rec[f"{qkind}_mean_loglik_40"] - rec["bf16_mean_loglik_40"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"famille {fcode} : terminé", flush=True)


if __name__ == "__main__":
    main()
