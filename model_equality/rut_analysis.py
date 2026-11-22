"""Test d'uniformité des rangs (RUT, Zhu et al. 2025), adapté au cas où les DEUX endpoints sont des boîtes noires.

    python rut_analysis.py [dossier_scores] [budget]

Entrée : les scores calculés par local/score_responses.py (une ligne par réponse : log-rang moyen, log-vraisemblance
moyenne, part de tokens « premier choix », nombre de tokens), sous un ou plusieurs modèles PROXYS (olmo, llama).

Pour une paire d'endpoints (A, B) et un score f :
  pour chaque prompt p et chaque réponse a de A : rang aléatoirisé de f(a) parmi les k réponses de B au même prompt,
      r = (#{b : f(b) < f(a)} + U * (#{b : f(b) = f(a)} + 1)) / (k + 1),   U ~ Uniforme(0, 1)
  -> si A et B servent le même modèle (même loi), r suit EXACTEMENT une loi uniforme sur [0, 1] ;
  statistique de Cramér-von Mises des rangs triés : w2 = 1/(12n) + somme_i ((2i - 1)/(2n) - r_(i))^2 ;
  distance RUT(A, B) = (w2(A|B) + w2(B|A)) / 2, ramenée à n (w2 / n -> ne dépend pas du nombre de réponses).
Évaluation : AUC « même modèle contre modèle différent » par jeu de données, et distance à plain en multiples de
l'écart moyen des déguisements. `budget` (ex. 100) : ne garder que ~100 réponses par endpoint (prompts x réponses).
Les matrices de distance sont sauvées dans <dossier_scores>/rut_<jeu>.npz pour la simulation de score.
"""
import csv
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = Path("/p/scratch/training2666/goldrajch1/met_local")
SC = Path(sys.argv[1]) if len(sys.argv) > 1 else W / "scores"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SCORES = ["mean_logrank", "mean_loglik", "frac_top1", "n_tok"]
rng = np.random.default_rng(0)


def dataset_of(fname):
    """out_llama32__plain.olmo.scores.jsonl -> ('dgx', 'olmo') ; out_olmo_r2a... -> ('local_r2', ...) ; refs."""
    m = re.match(r"^(.*)\.(olmo|llama)\.scores\.jsonl$", fname)   # les noms de variantes peuvent contenir des points
    stem, proxy = m.group(1), m.group(2)
    if stem.startswith("out_") and "__" in stem:
        return "dgx", proxy
    if stem.endswith("reference_answers"):
        return "ref", proxy
    if stem.endswith("ref_consignes_answers"):
        return "ref_consignes", proxy
    tag = re.sub(r"^out_(olmo|llama)_", "", stem)
    tag = re.sub(r"^(r2)[ab]$", r"\1", tag)
    return "local_" + tag, proxy


def load_truth():
    t = {}
    for p in (HERE / "out" / "truth.csv", W / "out" / "truth.csv", HERE / "practice_answers.csv"):
        if p.exists():
            rows = [r for r in csv.reader(open(p)) if r]
            if rows and rows[0][0] == "endpoint":
                rows = rows[1:]
            for r in rows:                                    # endpoint,family,model,kind (en-tête facultatif)
                t[r[0]] = dict(endpoint=r[0], family=r[1], model=r[2], kind=r[3] if len(r) > 3 else "")
    return t


truth = load_truth()
# data[(jeu, proxy)][endpoint][prompt] = liste de vecteurs de scores
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for f in sorted(SC.glob("*.scores.jsonl")):
    ds, proxy = dataset_of(f.name)
    for line in open(f):
        r = json.loads(line)
        if r["n_tok"] == 0 or r["endpoint"].startswith("ep-"):
            continue
        data[(ds, proxy)][r["endpoint"]][r["id"]].append([r[s] for s in SCORES])


def cvm(ranks):
    r = np.sort(ranks)
    n = len(r)
    return (1 / (12 * n) + np.sum(((2 * np.arange(1, n + 1) - 1) / (2 * n) - r) ** 2)) / n


def rut_matrix(E, X, prompts, k):
    """X[e][p] : array (k, nscores). Renvoie {score: matrice N x N} (calcul vectorisé sur les prompts)."""
    N = len(E)
    T = {e: np.stack([X[e][p] for p in prompts]) for e in E}           # (P, k, nscores)
    out = {s: np.zeros((N, N)) for s in SCORES}
    for a, b in itertools.combinations(range(N), 2):
        for j, s in enumerate(SCORES):
            w = []
            for A, B in ((a, b), (b, a)):
                va, vb = T[E[A]][:, :, j], T[E[B]][:, :, j]                # (P, k)
                less = (vb[:, None, :] < va[:, :, None]).sum(-1)
                eq = (vb[:, None, :] == va[:, :, None]).sum(-1)
                ranks = (less + rng.random(less.shape) * (eq + 1)) / (k + 1)
                w.append(cvm(ranks.ravel()))
            out[s][a, b] = out[s][b, a] = np.mean(w)
    return out


def auc(D, pairs, same):
    v = np.array([D[a, b] for a, b in pairs])
    d, sm = v[~same], v[same]
    if not len(d) or not len(sm):
        return float("nan")
    return float(np.mean((d[:, None] > sm[None, :]) + 0.5 * (d[:, None] == sm[None, :])))


by_ds = defaultdict(dict)
for (ds, proxy), eps_data in sorted(data.items()):
    E = sorted(e for e in eps_data if e in truth)
    if len(E) < 3:
        continue
    prompts = sorted(set.intersection(*[set(eps_data[e]) for e in E]))
    k = min(len(eps_data[e][p]) for e in E for p in prompts)
    if BUDGET:
        k = min(k, 4)
        prompts = prompts[:max(1, BUDGET // k)]
    if k < 2 or not prompts:
        continue
    X = {e: {p: np.array(eps_data[e][p][:k], dtype=float) for p in prompts} for e in E}
    fams = sorted({truth[e]["family"] for e in E})
    print(f"\n=== {ds} / proxy {proxy} : {len(E)} endpoints, {len(prompts)} prompts x {k} réponses", flush=True)
    for fam in fams:
        Ef = [e for e in E if truth[e]["family"] == fam]
        labels = [truth[e]["model"] for e in Ef]
        if len(Ef) < 3 or len(set(labels)) < 2:
            continue
        D = rut_matrix(Ef, X, prompts, k)
        pairs = list(itertools.combinations(range(len(Ef)), 2))
        same = np.array([labels[a] == labels[b] for a, b in pairs])
        print(f"  famille {fam} ({len(Ef)} endpoints) : AUC RUT " + "  ".join(f"{s} {auc(D[s], pairs, same):.2f}" for s in SCORES))
        np.savez(SC / f"rut_{ds}_{proxy}_{fam}.npz", endpoints=np.array(Ef), **D)
        by_ds[(ds, fam)][proxy] = (Ef, D, labels)
        pl = next((i for i, e in enumerate(Ef) if e.endswith("__plain") or e == "ref-01"), None)
        if pl is None:
            continue
        disg = [i for i in range(len(Ef)) if labels[i] == labels[pl] and i != pl]
        if not disg:
            continue
        s = "mean_logrank"
        base = np.mean([D[s][pl, i] for i in disg])
        rows = sorted(range(len(Ef)), key=lambda i: (labels[i] != labels[pl], Ef[i]))
        print(f"    distance RUT ({s}) à {Ef[pl]} / écart moyen des déguisements :")
        print("    " + "  ".join(f"{Ef[i].split('__')[-1]}={D[s][pl, i] / base:.1f}{'*' if labels[i] == labels[pl] else ''}"
                                  for i in rows if i != pl) + "   (* = même modèle)")
