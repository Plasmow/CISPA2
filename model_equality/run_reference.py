"""Interroge UNIQUEMENT les endpoints de référence ref-01 ... ref-09 (requêtes illimitées, jamais notés).

Garde-fou : ce script refuse tout endpoint qui ne commence pas par "ref-". Les endpoints ep-* ont un budget de
100 réponses chacun pour toute la compétition : ne jamais les interroger sans accord explicite de l'équipe.

La clé d'API est lue dans la variable d'environnement MET_API_KEY ou dans le fichier .api_key (ignoré par git).

Usage :  python run_reference.py [repeats] [fichier_prompts] [fichier_sortie]
        (défauts : 10 réponses par prompt, reference_prompts.txt, reference_answers.jsonl)
Les réponses sont ajoutées à reference_answers.jsonl ; on peut interrompre et relancer sans rien perdre.
"""
import os
import sys
from pathlib import Path

import query_reference as qr

HERE = Path(__file__).resolve().parent


def load_key() -> str:
    key = os.environ.get("MET_API_KEY", "").strip()
    if not key and (HERE / ".api_key").exists():
        key = (HERE / ".api_key").read_text().strip()
    if not key:
        sys.exit("Clé d'API absente : définis MET_API_KEY ou écris la clé dans model_equality/.api_key")
    return key


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    qr.API_KEY = load_key()
    budget = qr.get_budget(qr.make_session())
    endpoints = [ep for ep in budget if ep.startswith("ref-")]
    if not endpoints:
        sys.exit("Aucun endpoint ref-* listé par le serveur.")
    assert all(ep.startswith("ref-") for ep in endpoints)   # garde-fou : jamais d'ep-*
    print("Endpoints interrogés :", ", ".join(endpoints))
    pfile = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "reference_prompts.txt"
    ofile = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "reference_answers.jsonl"
    prompts = qr.read_prompts(pfile if pfile.is_absolute() else HERE / pfile)
    qr.query(prompts, output=ofile if ofile.is_absolute() else HERE / ofile, endpoints=endpoints, repeats=repeats)


if __name__ == "__main__":
    main()
