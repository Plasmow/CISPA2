"""Matrices MMD (noyau de Hamming, article Gao et al. 2025) entre les endpoints de reference_answers.jsonl.

Pour chaque paire d'endpoints (a, b) et chaque prompt x, on compare les réponses de a et de b à ce même prompt :
    k(y, y') = (1/L) * somme_i 1{y_i == y'_i}     (caractères, la plus courte est complétée par un caractère spécial)
    MMD(a, b) = moyenne sur les prompts de
        [ 1/(n(n-1)) sum_{i!=j} k(A_i, A_j) + 1/(m(m-1)) sum_{i!=j} k(B_i, B_j) - 2/(nm) sum_{i,j} k(A_i, B_j) ]
(estimateur non biaisé). Plus la valeur est grande, plus les deux distributions sont différentes.
La p-valeur vient d'un test de permutation : à prompt fixé, on mélange les réponses de a et de b.

Variable MET_BUDGET (défaut 100) : réponses gardées par endpoint, comme sur le serveur réel ; MET_BUDGET=0 les garde toutes.
Usage :  python mmd_matrix.py [answers.jsonl] [nb_permutations] [truth.csv]
(truth.csv : colonnes endpoint,family,model,kind ; défaut practice_answers.csv. Marche aussi sur les sorties locales.)
Écrit mmd_matrix.csv et mmd_pvalues.csv à côté du script et affiche les matrices.
Un endpoint "ep-*" dans le fichier est refusé (garde-fou).
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L = 1000          # caractères gardés par réponse (256 tokens ~ 1000 caractères)
BUDGET = int(os.environ.get("MET_BUDGET", 100))   # réponses par endpoint, comme sur les vrais endpoints (0 = pas de limite)
PAD = 0           # caractère spécial de remplissage


def load(path):
    data = defaultdict(lambda: defaultdict(list))        # endpoint -> prompt id -> [réponses encodées]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["endpoint"].startswith("ep-"):
            sys.exit(f"Endpoint {r['endpoint']} refusé : les ep-* sont à budget limité.")
        arr = np.full(L, PAD, dtype=np.int32)
        codes = [ord(c) for c in r["response"][:L]]
        arr[:len(codes)] = codes
        data[r["endpoint"]][r["id"]].append(arr)
    return {ep: {pid: np.stack(v) for pid, v in d.items()} for ep, d in data.items()}


def gram(a, b):
    """Matrice des k(a_i, b_j) = proportion de positions identiques."""
    return (a[:, None, :] == b[None, :, :]).mean(axis=2)


def mmd_from_gram(kaa, kbb, kab):
    n, m = len(kaa), len(kbb)
    return ((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
            - 2 * kab.mean())


def pair_stats(A, B, prompts, n_perm, rng):
    """MMD moyen sur les prompts et p-valeur de permutation (permutations dans chaque prompt)."""
    grams = {}
    for p in prompts:
        pool = np.concatenate([A[p], B[p]])
        grams[p] = (gram(pool, pool), len(A[p]))
    def stat(perm_idx=None):
        vals = []
        for p in prompts:
            g, n = grams[p]
            idx = np.arange(len(g)) if perm_idx is None else perm_idx[p]
            ia, ib = idx[:n], idx[n:]
            vals.append(mmd_from_gram(g[np.ix_(ia, ia)], g[np.ix_(ib, ib)], g[np.ix_(ia, ib)]))
        return float(np.mean(vals))
    obs = stat()
    ge = 0
    for _ in range(n_perm):
        perms = {p: rng.permutation(len(grams[p][0])) for p in prompts}
        ge += stat(perms) >= obs
    return obs, (ge + 1) / (n_perm + 1)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else HERE / "reference_answers.jsonl"
    n_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    data = load(path)
    eps = sorted(data)
    prompts = sorted(set.intersection(*[set(d) for d in data.values()]))
    prompts = [p for p in prompts if all(len(data[e][p]) >= 2 for e in eps)]
    if not prompts:
        sys.exit("Aucun prompt avec au moins 2 réponses sur tous les endpoints (lance run_reference.py).")
    if BUDGET:                                   # mêmes conditions que le serveur : au plus BUDGET réponses par endpoint
        k = max(2, BUDGET // len(prompts))       # réponses gardées par prompt (au moins 2 pour le MMD non biaisé)
        prompts = prompts[:max(1, BUDGET // k)]
        data = {e: {p: data[e][p][:k] for p in prompts} for e in eps}
        print(f"Budget {BUDGET} réponses par endpoint : {len(prompts)} prompts x {k} réponses = {len(prompts) * k}")
    print(f"{len(eps)} endpoints, {len(prompts)} prompts, "
          f"{min(len(data[e][p]) for e in eps for p in prompts)} réponses minimum par prompt et endpoint")

    kinds = {}
    pa = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "practice_answers.csv"
    if pa.exists():
        for row in csv.DictReader(pa.open()):
            kinds[row["endpoint"]] = f'{row["model"]} {row["kind"]}'

    rng = np.random.default_rng(0)
    n = len(eps)
    M, P = np.zeros((n, n)), np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m, p = pair_stats(data[eps[i]], data[eps[j]], prompts, n_perm, rng)
            M[i, j] = M[j, i] = m
            P[i, j] = P[j, i] = p
            print(f"  {eps[i]} vs {eps[j]}: MMD={m:.4f} p={p:.3f}")

    for name, mat, fmt in (("mmd_matrix.csv", M, "{:.4f}"), ("mmd_pvalues.csv", P, "{:.3f}")):
        with open(HERE / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + eps)
            for i, e in enumerate(eps):
                w.writerow([e] + [fmt.format(v) for v in mat[i]])
        print(f"\n{name}")
        print("          " + " ".join(f"{e:>8}" for e in eps))
        for i, e in enumerate(eps):
            print(f"{e:>9} " + " ".join(f"{mat[i, j]:8.4f}" if fmt == "{:.4f}" else f"{mat[i, j]:8.3f}" for j in range(n)))
    if kinds:
        print("\nVérité terrain :")
        for e in eps:
            print(f"  {e}: {kinds.get(e, '?')}")


if __name__ == "__main__":
    main()
