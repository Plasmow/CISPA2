"""AgentProv (Wang et al. 2026) adapté : empreinte par CHOIX D'OUTIL, sans GPU, avec des requêtes courtes.

    python toolprobe.py collect ref  [N]        -> interroge les ref-* (gratuit), N tirages par gabarit (défaut 5)
    python toolprobe.py analyze ref             -> distances entre ref-* et comparaison à la vérité terrain
    (la collecte ep-* n'est PAS dans ce script : accord explicite de l'équipe requis)

Principe de l'article : on offre au modèle plusieurs outils fonctionnellement équivalents (noms différents) et une demande
neutre ; l'outil choisi est une variable CATÉGORIELLE. L'empreinte d'un endpoint = distribution des choix sur K gabarits ;
distance = MMD à noyau delta = somme sur gabarits de ||phi_A,k - phi_B,k||^2 ; test par permutation. Ce choix dépend des poids
(post-entraînement) et très peu du texte autour (system prompt, persona) : exactement l'invariance qui nous manque.

Ici le serveur ne renvoie que du texte : les outils sont décrits DANS le prompt et on classe le nom d'outil cité en premier.
Ordre des outils tiré au hasard à chaque requête (graine déterministe), descriptions de même longueur.
"""
import csv
import itertools
import json
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
URL = "http://34.147.229.55/33-model-equality/v1/chat/completions"
OUT = HERE / "toolprobe_ref.jsonl"

# 20 gabarits : (demande neutre, [outils équivalents]) ; noms inspirés de conventions de fournisseurs différents
T = [
    ("Please send a short email to alex@example.com saying the report is ready.", ["send_gmail_message", "send_outlook_mail", "yahoo_mail_send"]),
    ("Schedule a 30-minute meeting with Sam next Tuesday at 10am.", ["gcal_create_event", "zoom_schedule_meeting", "calendly_book_slot"]),
    ("Set a reminder to call the dentist tomorrow at 9am.", ["apple_reminders_add", "todoist_create_task", "google_tasks_insert"]),
    ("Find the current weather in Paris.", ["openweather_get_current", "accuweather_lookup", "weathergov_forecast"]),
    ("Translate 'good morning' into Spanish.", ["deepl_translate", "google_translate_text", "azure_translator_convert"]),
    ("Search the web for the latest news about electric cars.", ["bing_web_search", "google_custom_search", "duckduckgo_query"]),
    ("Save the note 'buy milk' so I can find it later.", ["notion_create_page", "evernote_new_note", "onenote_add_entry"]),
    ("Convert 100 US dollars to euros.", ["xe_convert_currency", "ecb_exchange_rate", "openexchange_latest"]),
    ("Send a message to the team channel saying the build passed.", ["slack_post_message", "teams_send_chat", "discord_create_message"]),
    ("Upload the file report.pdf to my cloud storage.", ["gdrive_upload_file", "dropbox_files_upload", "onedrive_put_item"]),
    ("Play some relaxing jazz music.", ["spotify_start_playback", "apple_music_play", "youtube_music_queue"]),
    ("Book a table for two at an Italian restaurant tonight.", ["opentable_reserve", "yelp_reservations_create", "resy_book_table"]),
    ("Look up the stock price of Microsoft.", ["yahoo_finance_quote", "alphavantage_global_quote", "bloomberg_get_price"]),
    ("Create a new issue titled 'Fix login bug' in the project tracker.", ["github_create_issue", "jira_issue_create", "linear_save_issue"]),
    ("Get directions from Berlin Central Station to the airport.", ["gmaps_get_route", "here_calculate_route", "osm_route_directions"]),
    ("Read the file config.yaml from the project folder.", ["fs_read_file", "workspace_open_document", "storage_get_object"]),
    ("List the files in the current directory.", ["fs_list_directory", "shell_run_ls", "workspace_enumerate_files"]),
    ("Query the database for all customers created this month.", ["postgres_query", "sqlite_execute_sql", "bigquery_run_query"]),
    ("Fetch the web page at https://example.com and give me its text.", ["http_fetch_url", "browser_open_page", "scrape_page_text"]),
    ("Run this Python snippet: print(2+2).", ["python_exec_code", "jupyter_run_cell", "sandbox_eval_python"]),
]
K = len(T)
# mot distinctif par outil : les petits modèles écorchent parfois le nom (« say_gmail_message »)
KEYS = {"send_gmail_message": "gmail", "send_outlook_mail": "outlook", "yahoo_mail_send": "yahoo", "gcal_create_event": "gcal",
        "zoom_schedule_meeting": "zoom", "calendly_book_slot": "calendly", "apple_reminders_add": "apple", "todoist_create_task": "todoist",
        "google_tasks_insert": "google_tasks", "openweather_get_current": "openweather", "accuweather_lookup": "accuweather",
        "weathergov_forecast": "weathergov", "deepl_translate": "deepl", "google_translate_text": "google_translate",
        "azure_translator_convert": "azure", "bing_web_search": "bing", "google_custom_search": "google_custom",
        "duckduckgo_query": "duckduckgo", "notion_create_page": "notion", "evernote_new_note": "evernote", "onenote_add_entry": "onenote",
        "xe_convert_currency": "xe_convert", "ecb_exchange_rate": "ecb", "openexchange_latest": "openexchange",
        "slack_post_message": "slack", "teams_send_chat": "teams", "discord_create_message": "discord",
        "gdrive_upload_file": "gdrive", "dropbox_files_upload": "dropbox", "onedrive_put_item": "onedrive",
        "spotify_start_playback": "spotify", "apple_music_play": "apple_music", "youtube_music_queue": "youtube",
        "opentable_reserve": "opentable", "yelp_reservations_create": "yelp", "resy_book_table": "resy",
        "yahoo_finance_quote": "yahoo_finance", "alphavantage_global_quote": "alphavantage", "bloomberg_get_price": "bloomberg",
        "github_create_issue": "github", "jira_issue_create": "jira", "linear_save_issue": "linear",
        "gmaps_get_route": "gmaps", "here_calculate_route": "here_calc", "osm_route_directions": "osm",
        "fs_read_file": "fs_read", "workspace_open_document": "workspace_open", "storage_get_object": "storage_get",
        "fs_list_directory": "fs_list", "shell_run_ls": "shell_run", "workspace_enumerate_files": "workspace_enum",
        "postgres_query": "postgres", "sqlite_execute_sql": "sqlite", "bigquery_run_query": "bigquery",
        "http_fetch_url": "http_fetch", "browser_open_page": "browser", "scrape_page_text": "scrape",
        "python_exec_code": "python_exec", "jupyter_run_cell": "jupyter", "sandbox_eval_python": "sandbox"}
FORMAT = ("You can use exactly ONE of the following tools to handle the request. All of them can do the job.\n"
          "Tools:\n{tools}\n\nRequest: {req}\n\nReply with ONLY the name of the tool you choose, followed by its JSON arguments.")
TOOL_DESC = "Tool for this task."


def build_prompt(k, sample_seed):
    req, tools = T[k]
    order = list(tools)
    random.Random(f"{k}-{sample_seed}").shuffle(order)
    return FORMAT.format(tools="\n".join(f"- {t}: {TOOL_DESC}" for t in order), req=req)


def classify(k, text):
    """Option choisie = outil du gabarit cité en premier (nom complet ou mot distinctif) ; sinon 'none'."""
    low = text.lower()
    best, pos = "none", 10 ** 9
    for t in T[k][1]:
        for pat in (t.lower(), KEYS[t]):
            i = low.find(pat)
            if 0 <= i < pos:
                best, pos = t, i
    return best


def collect(endpoints, n, out):
    key = (HERE / ".api_key").read_text().strip()
    S = requests.Session()
    S.headers["X-API-Key"] = key
    done = set()
    if out.exists():
        done = {(r["endpoint"], r["k"], r["s"]) for r in map(json.loads, open(out))}
    jobs = [(e, k, s) for s in range(n) for k in range(K) for e in endpoints if (e, k, s) not in done]
    print(f"{len(jobs)} requêtes à envoyer ({len(done)} déjà faites)", flush=True)
    lock, cnt, fh = threading.Lock(), [0], open(out, "a", encoding="utf-8")

    def one(job):
        e, k, s = job
        assert e.startswith("ref-"), "ce script n'interroge que les ref-*"
        p = build_prompt(k, s)
        for _ in range(200):
            try:
                r = S.post(URL, timeout=120, json={"model": e, "max_tokens": 50, "messages": [{"role": "user", "content": p}]})
            except requests.RequestException:
                time.sleep(5)
                continue
            if r.status_code == 429 and "Retry-After" in r.headers:
                time.sleep(1.0 + random.random())
                continue
            if r.status_code != 200:
                return
            txt = r.json()["choices"][0]["message"]["content"]
            with lock:
                fh.write(json.dumps({"endpoint": e, "k": k, "s": s, "choice": classify(k, txt), "text": txt[:120]}, ensure_ascii=False) + "\n")
                fh.flush()
                cnt[0] += 1
                if cnt[0] % 100 == 0:
                    print(f"  {cnt[0]}/{len(jobs)}", flush=True)
            return

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, jobs))
    fh.close()


def fingerprints(rows, endpoints):
    """phi[e][k] = distribution (Counter normalisé) des choix de e sur le gabarit k."""
    cnt = {e: [Counter() for _ in range(K)] for e in endpoints}
    for r in rows:
        if r["endpoint"] in cnt:
            cnt[r["endpoint"]][r["k"]][r["choice"]] += 1
    return cnt


def mmd(ca, cb):
    """T = somme_k ||phi_a,k - phi_b,k||^2 (noyau delta) ; estimateur biaisé de l'article."""
    tot = 0.0
    for k in range(K):
        na, nb = sum(ca[k].values()), sum(cb[k].values())
        if na == 0 or nb == 0:
            continue
        for o in set(ca[k]) | set(cb[k]):
            tot += (ca[k][o] / na - cb[k][o] / nb) ** 2
    return tot


def analyze(path):
    truth = {r["endpoint"]: r for r in csv.DictReader(open(HERE / "practice_answers.csv"))}
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    E = sorted(truth)
    cnt = fingerprints(rows, E)
    n = min(sum(sum(c.values()) for c in cnt[e]) for e in E)
    print(f"{len(rows)} réponses ; au moins {n} par endpoint ; part de réponses sans outil reconnu : "
          f"{100 * sum(r['choice'] == 'none' for r in rows) / len(rows):.0f} %")
    # distribution des choix sur 5 gabarits pour lecture directe
    print("\nChoix par endpoint sur quelques gabarits (outil le plus choisi) :")
    for e in E:
        top = []
        for k in (0, 1, 5, 8, 13):
            c = cnt[e][k]
            top.append((c.most_common(1)[0][0].split("_")[0] if c else "?") + f"({c.most_common(1)[0][1] if c else 0}/{sum(c.values())})")
        print(f"  {e} [{truth[e]['model']:2s} {truth[e]['kind'][:18]:18s}] " + "  ".join(top))
    pairs = list(itertools.combinations(range(len(E)), 2))
    D = np.zeros((len(E), len(E)))
    for a, b in pairs:
        D[a, b] = D[b, a] = mmd(cnt[E[a]], cnt[E[b]])
    same = np.array([truth[E[a]]["model"] == truth[E[b]]["model"] for a, b in pairs])
    v = np.array([D[a, b] for a, b in pairs])
    d, s = v[~same], v[same]
    auc = float(np.mean((d[:, None] > s[None, :]) + 0.5 * (d[:, None] == s[None, :])))
    print(f"\nAUC « modèle différent plus loin que même modèle » (36 paires) : {auc:.2f}   (autres métriques sur ces ref-* : 0,7 à 0,8)")
    print(f"MMD moyen : même modèle {s.mean():.3f} ; modèles différents {d.mean():.3f}")
    pl = E.index("ref-01")
    disg = [i for i in range(len(E)) if truth[E[i]]["model"] == "S" and i != pl]
    base = np.mean([D[pl, i] for i in disg])
    print("\nDistance à ref-01 (plain) / écart moyen des déguisements (>1 = plus loin qu'un déguisement) :")
    for i in sorted(range(len(E)), key=lambda i: (truth[E[i]]["model"] != "S", E[i])):
        if i != pl:
            print(f"  {E[i]} {truth[E[i]]['kind'][:30]:30s} {D[pl, i] / base:6.2f}{'  (même modèle)' if truth[E[i]]['model'] == 'S' else ''}")
    # test par permutation, ref-01 contre chaque autre (comme l'article)
    rng = np.random.default_rng(0)
    print("\nTest par permutation (comme l'article), ref-01 contre chaque endpoint, alpha = 0,05 :")
    for i in range(len(E)):
        if i == pl:
            continue
        obs, null = D[pl, i], []
        pool = [[(r["choice"]) for r in rows if r["endpoint"] == e and r["k"] == k] for e in (E[pl], E[i]) for k in range(K)]
        for _ in range(300):
            ca, cb = [Counter() for _ in range(K)], [Counter() for _ in range(K)]
            for k in range(K):
                both = pool[k] + pool[K + k]
                rng.shuffle(both)
                half = len(pool[k])
                ca[k].update(both[:half])
                cb[k].update(both[half:])
            null.append(mmd(ca, cb))
        p = (1 + sum(x >= obs for x in null)) / (1 + len(null))
        print(f"  {E[i]} {truth[E[i]]['kind'][:28]:28s} p = {p:.3f} -> {'REJETTE (modèle différent)' if p < 0.05 else 'accepte (même modèle)'}   "
              f"[vérité : {'même' if truth[E[i]]['model'] == 'S' else 'différent'}]")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        collect([f"ref-{i:02d}" for i in range(1, 10)], n, OUT)
    elif cmd == "analyze":
        analyze(OUT)
