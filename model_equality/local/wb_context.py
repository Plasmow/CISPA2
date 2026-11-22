"""Contexte caché ou poids modifiés ? Test du score exact de chaque endpoint sous plusieurs contextes candidats.
Aucune requête vers les endpoints.

Un endpoint qui sert le modèle de base derrière un system prompt du fournisseur est mal noté avec le contexte par
défaut, mais redevient conforme (z proche de 0) avec un contexte proche du sien ; un LoRA reste non conforme quel que
soit le contexte. Contextes essayés pour chaque réponse d'une famille (--only A|B|C) :
  defaut   gabarit de chat tel quel (system prompt par défaut du modèle)
  aucun    aucun bloc system
  helpful  « You are a helpful assistant. »
  concis   system prompt recopié par ep-29
  pro      reconstruction du second texte dévoilé par ep-29
Sortie : <out>/context_<fam>.jsonl (une ligne par réponse et par contexte, mêmes champs que dose_<fam>.jsonl).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
sys.path.insert(0, str(H / "local"))
from score_wb_r4 import FAMILY_MODELS  # noqa: E402
from wb_dose import base_pass, dump, encode  # noqa: E402

SYSTEMS = {
    "helpful": "You are a helpful assistant.",
    "concis": "You are a helpful assistant. Keep answers reasonably concise. If you are not sure about something, say so instead of guessing. Use short paragraphs.",
    "pro": "You are a helpful assistant. Help users with their questions and tasks. Explain your reasoning step by step when solving problems. "
           "Keep your responses light and avoid headings unless the answer is long. Use a neutral, professional tone. "
           "Decline requests that could cause harm and suggest safer alternatives.",
}
NO_SYSTEM = {
    "A": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "B": "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
    "C": "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(FAMILY_MODELS))
    ap.add_argument("--out", default=str(H / "jureca_retour9"))
    ap.add_argument("--groups", default=str(H / "submission_v8.csv"))
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    short, repo = FAMILY_MODELS[a.only]
    fam = {r["endpoint"]: r["family"] for r in csv.DictReader(open(a.groups, encoding="utf-8"))}
    rows = []
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if fam.get(r["endpoint"]) != a.only or not r["response"].strip():
            continue
        base = dict(kind="ep", id=str(r["id"]), repeat=int(r.get("repeat", 0)), prompt=r["prompt"], response=r["response"])
        rows.append(dict(base, name=f"{r['endpoint']}|defaut"))
        rows.append(dict(base, name=f"{r['endpoint']}|aucun", ctx_text=NO_SYSTEM[a.only].format(p=r["prompt"])))
        for key, system in SYSTEMS.items():
            rows.append(dict(base, name=f"{r['endpoint']}|{key}", system=system))
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, device_map="auto").eval()
    print(f"famille {a.only} ({repo}) : {len(rows)} lignes à noter", flush=True)
    encode(tok, rows)
    print("  exemple de contexte 'aucun' :", repr(tok.decode(rows[1]["ctx"])[:160]), flush=True)
    print("  exemple de contexte 'helpful' :", repr(tok.decode(rows[2]["ctx"])[:220]), flush=True)
    base_pass(model, tok, rows, bs=12)
    dump(rows, outdir / f"context_{a.only}.jsonl")
    print(f"famille {a.only} : terminé", flush=True)


if __name__ == "__main__":
    main()
