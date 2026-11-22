"""Notation boîte blanche sur JURECA des rounds 5 et 6 (remplace l'aller-retour DGX). Aucune requête vers les endpoints.

Réutilise telles quelles les fonctions de la DGX (dgx_brief4/score_wb_r4.py) pour que les valeurs soient comparables.
Pour une famille (--only A|B|C) :
  1. génère les références locales (plain_t07, fp8sim_t07, int8_t07 ; 8 tirages) des NOUVEAUX prompts 60..70
     -> <out>/cands_r6/<modèle>__<config>.jsonl (sauté si le fichier existe) ;
  2. note sous bf16, fp8sim, int8, nf4 :
       - les réponses ep-* des rounds 5 et 6 (id >= 60 ou repeat >= 1), en conservant `repeat` ;
       - les réponses ep-* des prompts 55..59 (déjà notées par la DGX) comme contrôle croisé entre machines ;
       - les références générées en 1.
     -> <out>/wb_r56_scores_<fam>.jsonl et <out>/calib_60_70_scores_<fam>.jsonl
Mêmes plafonds que le serveur : 256 tokens générés, texte seul, pas de system message.
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

CONFIGS = [
    ("plain_t07", "bf16", dict(temperature=0.7, top_p=0.9)),
    ("fp8sim_t07", "fp8sim", dict(temperature=0.7, top_p=0.9)),
    ("int8_t07", "int8", dict(temperature=0.7, top_p=0.9)),
]
NEW_IDS = range(60, 71)
CROSS_CHECK_IDS = range(55, 60)


def new_prompts():
    out = {}
    for name in ("ep_round5_plan.jsonl", "ep_round6_plan.jsonl"):
        for l in open(H / "prompts" / name, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if int(r["id"]) in NEW_IDS and int(r["repeat"]) == 0:
                    out[int(r["id"])] = r["prompt"]
    assert sorted(out) == list(NEW_IDS), sorted(out)
    return out


def chat_text(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def generate(fcode, short, repo, outdir, n):
    prompts = new_prompts()
    cdir = outdir / "cands_r6"
    cdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for tag, qkind, dec_kw in CONFIGS:
        out_file = cdir / f"{short}__{tag}.jsonl"
        if out_file.exists():
            print(f"déjà fait : {out_file.name}", flush=True)
            continue
        print(f"génération {short}__{tag} : {len(prompts)} prompts x {n}", flush=True)
        mdl = load_quantized(repo, qkind)
        dev = next(mdl.parameters()).device
        tmp = out_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for pid, p in sorted(prompts.items()):
                enc = tok([chat_text(tok, p)] * n, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                with torch.no_grad():
                    gen = mdl.generate(**enc, max_new_tokens=256, do_sample=True, pad_token_id=tok.pad_token_id, **dec_kw)
                for r, seq in enumerate(gen[:, enc["input_ids"].shape[1] :]):
                    fh.write(json.dumps({"endpoint": f"{short}__{tag}", "family": fcode, "id": str(pid), "repeat": r, "prompt": p,
                                         "response": tok.decode(seq, skip_special_tokens=True)}, ensure_ascii=False) + "\n")
        tmp.rename(out_file)
        del mdl
        torch.cuda.empty_cache()


def score(fcode, short, repo, outdir, groups):
    fam_map = {r["endpoint"]: r["family"] for r in csv.DictReader(open(groups, encoding="utf-8"))}
    f_ep = []
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except ValueError:  # dernière ligne en cours d'écriture si une collecte tourne encore
            continue
        pid, rep = int(r["id"]), int(r.get("repeat", 0))
        if fam_map.get(r["endpoint"]) == fcode and r["response"].strip() and (pid >= 60 or rep >= 1 or pid in CROSS_CHECK_IDS):
            f_ep.append({"endpoint": r["endpoint"], "id": pid, "repeat": rep, "prompt": r["prompt"], "response": r["response"]})
    f_cal = []
    for cf in sorted((outdir / "cands_r6").glob(f"{short}__*.jsonl")):
        cfg = cf.stem.split("__")[1]
        for l in open(cf, encoding="utf-8"):
            r = json.loads(l)
            if r["response"].strip():
                f_cal.append({"config": cfg, "id": int(r["id"]), "repeat": int(r["repeat"]), "prompt": r["prompt"], "response": r["response"]})
    combined = f_ep + f_cal
    print(f"famille {fcode} ({repo}) : {len(f_ep)} réponses ep-* + {len(f_cal)} références", flush=True)

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    per_q = {}
    for qkind in ("bf16", "fp8sim", "int8", "nf4"):
        print(f"  notation sous {qkind}", flush=True)
        mdl = load_quantized(repo, qkind)
        per_q[qkind] = score_rows_under_model(mdl, tok, combined)
        del mdl
        torch.cuda.empty_cache()

    with open(outdir / f"wb_r56_scores_{fcode}.jsonl", "w", encoding="utf-8") as fh_ep, open(
        outdir / f"calib_60_70_scores_{fcode}.jsonl", "w", encoding="utf-8"
    ) as fh_cal:
        for idx, r in enumerate(combined):
            is_ep = idx < len(f_ep)
            rec = {"family": fcode, "id": r["id"], "repeat": r["repeat"]}
            rec["endpoint" if is_ep else "config"] = r["endpoint"] if is_ep else r["config"]
            for qkind, res in per_q.items():
                if res[idx].get("n_tok", 0) > 0:
                    for k, v in res[idx].items():
                        rec[f"{qkind}_{k}"] = v
            if "bf16_mean_loglik_40" in rec:
                for qkind in ("fp8sim", "int8", "nf4"):
                    if f"{qkind}_mean_loglik_40" in rec:
                        rec[f"delta_{qkind}_ll40"] = rec[f"{qkind}_mean_loglik_40"] - rec["bf16_mean_loglik_40"]
            (fh_ep if is_ep else fh_cal).write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"famille {fcode} : terminé", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(FAMILY_MODELS))
    ap.add_argument("--out", default=str(H / "jureca_retour5"))
    ap.add_argument("--groups", default=str(H / "submission_v8.csv"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    short, repo = FAMILY_MODELS[a.only]
    if not a.skip_gen:
        generate(a.only, short, repo, outdir, a.n)
    score(a.only, short, repo, outdir, a.groups)


if __name__ == "__main__":
    main()
