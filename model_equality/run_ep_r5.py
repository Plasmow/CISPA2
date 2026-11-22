"""Envoi contrôlé gérant explicitement (endpoint, id, repeat) pour permettre à la fois
des répétitions (repeat=1..) sur des prompts existants ET de nouveaux prompts (repeat=0).

    python run_ep_r5.py prompts/ep_round5_plan.jsonl          -> SIMULATION (n'envoie RIEN)
    python run_ep_r5.py prompts/ep_round5_plan.jsonl --send   -> envoie après accord explicite
    python run_ep_r5.py plan.jsonl --only=ep-02,ep-29 [--send] -> seulement ces endpoints (économise le budget des autres)

Format du fichier plan (.jsonl) : une ligne par tâche globale {"id": "44", "repeat": 1, "prompt": "..."}
Garde-fous identiques à run_ep.py :
  - vérifie (endpoint, str(id), int(repeat)) dans ep_answers.jsonl pour ne jamais envoyer de doublon ;
  - vérifie le budget serveur avant envoi ;
  - 429 + Retry-After : attend et réessaie ; toute autre erreur : aucun retry automatique.
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
URL = "http://34.147.229.55/33-model-equality/v1"
OUT = HERE / "ep_answers.jsonl"
EXPECTED = [f"ep-{i:02d}" for i in range(1, 34)]
KEY = (HERE / ".api_key").read_text().strip()
S = requests.Session()
S.headers["X-API-Key"] = KEY


def budget():
    d = S.get(f"{URL}/models", timeout=60).json()["data"]
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in d}


def main():
    pfile = Path(sys.argv[1])
    send = "--send" in sys.argv
    plan = [json.loads(l) for l in open(pfile, encoding="utf-8") if l.strip()]
    left = budget()
    eps = sorted(e for e in left if e.startswith("ep-"))
    if eps != EXPECTED:
        sys.exit(f"Liste d'endpoints inattendue : {eps}")
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    targets = eps if only is None else sorted(set(only))
    if not targets or not set(targets) <= set(eps):
        sys.exit(f"--only contient des endpoints inconnus : {sorted(set(targets) - set(eps))}")

    done = set()
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["endpoint"], str(r["id"]), int(r.get("repeat", 0))))

    jobs = [
        (e, str(item["id"]), int(item["repeat"]), item["prompt"])
        for item in plan
        for e in targets
        if (e, str(item["id"]), int(item["repeat"])) not in done
    ]
    per_ep = {e: sum(1 for j in jobs if j[0] == e) for e in eps}
    short = [e for e in eps if per_ep[e] > left[e]]
    print(
        f"Plan : {len(plan)} items x {len(targets)} endpoints ({', '.join(targets) if only else 'tous'}) -> {len(jobs)} requêtes à envoyer "
        f"({len(done)} réponses déjà dans {OUT.name})."
    )
    print(
        f"Budget des endpoints visés avant : min {min(left[e] for e in targets)}, max {max(left[e] for e in targets)} ; "
        f"après : min {min(left[e] - per_ep[e] for e in targets)}, max {max(left[e] - per_ep[e] for e in targets)}."
    )
    if short:
        sys.exit(f"Budget insuffisant pour : {short}")
    if not send:
        print("SIMULATION : rien n'a été envoyé. Ajouter --send après accord explicite.")
        return

    lock, fails, n = threading.Lock(), [], [0]
    out = open(OUT, "a", encoding="utf-8")

    def one(job):
        e, pid, rep, p = job
        while True:
            try:
                r = S.post(
                    f"{URL}/chat/completions",
                    timeout=300,
                    json={"model": e, "max_tokens": 256, "messages": [{"role": "user", "content": p}]},
                )
            except requests.RequestException as exc:
                with lock:
                    fails.append((e, pid, rep, f"réseau : {type(exc).__name__}"))
                return
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                with lock:
                    out.write(
                        json.dumps({"endpoint": e, "id": pid, "repeat": rep, "prompt": p, "response": text}, ensure_ascii=False)
                        + "\n"
                    )
                    out.flush()
                    n[0] += 1
                    if n[0] % 20 == 0 or n[0] == len(jobs):
                        print(f"  {n[0]}/{len(jobs)}", flush=True)
                return
            if r.status_code == 429 and "Retry-After" in r.headers:
                time.sleep(int(r.headers["Retry-After"]))
                continue
            with lock:
                fails.append((e, pid, rep, f"HTTP {r.status_code} : {r.text[:150]}"))
            return

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, jobs))
    out.close()
    after = budget()
    print(f"Terminé : {n[0]} réponses ajoutées dans {OUT.name}, {len(fails)} échecs.")
    for f in fails:
        print("  échec :", f)
    print(
        f"Budget restant sur le serveur : min {min(after[e] for e in eps)}, max {max(after[e] for e in eps)} ; "
        f"utilisé au total sur les ep-* : {sum(100 - after[e] for e in eps)}"
    )


if __name__ == "__main__":
    main()
