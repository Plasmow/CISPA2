"""Méthode combinée : un verdict par endpoint à partir de TOUTES les preuves, hiérarchisées selon leur fiabilité mesurée.
Sans GPU, sans requête. Usage : python local/combine.py
Sources (par endpoint) :
  M  marqueurs de texte établis (persona, consigne, watermark par répétition, T = 1, décodage tronqué) -> preuves POSITIVES de modèle de base
  Q  décalage fp8 par tenseur (jureca_retour10/quant_*.jsonl, z) -> preuve NÉGATIVE (poids modifiés)
  R  RUT (jureca_retour13) : max des p sous les hypothèses de poids de base / sous les hypothèses fp8
  Z  empilement consigne + fp8 (jureca_retour13/stack_*.jsonl) : z du test exact sous la meilleure consigne
  C  classifieur simple par famille, entraîné sur DGX + synthétiques (clf_dgx.py)
  D  réseau profond robuste spécial Qwen (jureca_retour21/deep2_*.json : sans ancres / avec ancres) pour B et C
  H  historique des scores publics : endpoints dont l'entrée ou la sortie du noyau a fait varier le score
Règle de décision (voir verdict()) : un endpoint ne sort du noyau de v8 que si au moins deux familles de preuves indépendantes
(Q, R, C, D) vont dans le même sens ET qu'aucun marqueur de déguisement ni aucune protection de l'historique ne s'y oppose.
Un endpoint isolé dans v8 n'entre dans le noyau que si aucune preuve négative n'existe, que l'historique ne l'incrimine pas et que
la majorité des preuves statistiques (C, D, R) est favorable.
"""
import contextlib
import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "local"))
sys.dont_write_bytecode = True
import wb_rut as W  # noqa: E402

W.STATS = ("f_ll", "f_le")
FAM = {r["endpoint"]: r["family"] for r in csv.DictReader(open(H / "submission_v8.csv"))}
V8 = {r["endpoint"]: r["model"] for r in csv.DictReader(open(H / "submission_v8.csv"))}
CORE = {e for e, m in V8.items() if m.endswith("1")}

# M : marqueurs de texte établis (sondes de fuite, regex, répétitions, z à T = 1,0, z avec décodage tronqué)
MARK = {"ep-16": "persona", "ep-20": "persona", "ep-23": "persona", "ep-29": "consigne", "ep-24": "consigne(Claude)", "ep-02": "consigne",
        "ep-08": "consigne?", "ep-03": "consigne?", "ep-04": "watermark", "ep-06": "watermark", "ep-10": "watermark",
        "ep-31": "T=1", "ep-30": "T=1", "ep-33": "T=1", "ep-27": "décodage tronqué", "ep-07": "décodage tronqué", "ep-14": "décodage tronqué"}
# H : historique des scores publics (voir logs/submissions.log) : +1 = protégé dans le noyau, -1 = ne doit pas entrer au noyau
HIST = {"ep-23": (+1, "v16 : sortie coûte 0,15"), "ep-24": (+1, "v16 : sortie coûte 0,15"), "ep-09": (+1, "v11 : sortie coûte"),
        "ep-08": (+1, "v11 : sortie coûte"), "ep-18": (-1, "v11, v14, v18 : entrée coûte"), "ep-16": (+1, "v5->v6 : +0,20"),
        "ep-20": (+1, "v5->v6 : +0,20")}


def rut():
    out = {}
    hyp = 7
    for f in "ABC":
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            W.summary(str(H / "jureca_retour13" / f"rut_{f}.jsonl"))
        for l in buf.getvalue().splitlines():
            m = re.match(r"\s+(ep-\d+)\s+(.*)", l)
            if m:
                p = [float(x) for x in re.findall(r"([0-9.]+)[* ]\[", m.group(2))]
                if len(p) == hyp:
                    out[m.group(1)] = (max(p[:4]), max(p[4:]))
    return out


def quant():
    out, rng = {}, np.random.default_rng(0)
    for f in "ABC":
        rows = [json.loads(l) for l in open(H / "jureca_retour10" / f"quant_{f}.jsonl")]
        by = defaultdict(list)
        for r in rows:
            if r["kind"] == "ep" and r["n"] > 0:
                by[r["name"]].append(r)
        for e, rs in by.items():
            n = np.array([r["n"] for r in rs], float)
            d = np.array([r["q"]["w_tensor"][0] - r["lp"][0] for r in rs])
            idx = rng.integers(0, len(d), (100, len(d)))
            est = 1000 * d.sum() / n.sum()
            se = 1000 * (d[idx].sum(1) / n[idx].sum(1)).std()
            out[e] = (est, est / se if se > 0 else 0.0)
        # z relatif au noyau de la famille (médiane des estimations)
        med = np.median([v[0] for k, v in out.items() if FAM.get(k) == f])
        for k in list(out):
            if FAM.get(k) == f:
                out[k] = (out[k][0] - med, out[k][1])
    return out


def stack():
    out = {}
    for f in "ABC":
        by = defaultdict(list)
        for l in open(H / "jureca_retour13" / f"stack_{f}.jsonl"):
            r = json.loads(l)
            if r["kind"] == "ep":
                by[r["name"]].append(r)
        for e, rs in by.items():
            z = lambda t: sum(r[t][0] for r in rs) / np.sqrt(sum(r[t][1] for r in rs))
            out[e] = (z("bf16_def"), z("bf16_sys"))
    return out


def simple():
    import clf_dgx as C
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    Xd, yd, _, _ = C.load_dgx(70, 30, rng)
    out = {}
    for fam in "ABC":
        eps, ref = C.load_family(fam)
        truth = {e: m for e, f, m in csv.reader(open(H / "jureca_retour15" / f"synth_truth_{fam}.csv"))}
        Xs, ys = [], []
        for e, rs in eps.items():
            if e in truth:
                for _ in range(40):
                    sub = [rs[i] for i in rng.choice(len(rs), min(70, len(rs)), replace=False)]
                    v = C.feats(sub, ref)
                    if v is not None:
                        Xs.append(v); ys.append(int(truth[e].endswith("-S")))
        X, y = np.vstack([Xd, np.array(Xs)]), np.concatenate([yd, ys])
        models = [make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(X, y),
                  HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300).fit(X, y)]
        for e, rs in eps.items():
            if e.startswith("ep-"):
                v = C.feats(rs, ref)
                if v is not None:
                    out[e] = float(np.mean([m.predict_proba(v[None])[0, 1] for m in models]))
    return out


def deep():
    out = {}
    for tag in ("15", "3"):
        p = H / "jureca_retour21" / f"deep2_{tag}.json"
        if p.exists():
            for e, r in json.load(open(p))["real"].items():
                if FAM.get(e) == {"15": "B", "3": "C"}[tag]:
                    out[e] = (r.get("p_sans_ancres"), r.get("p_avec_ancres"))
    return out


def verdict(e, q, r, c, d):
    """Renvoie (verdict, raisons). verdict : 'noyau', 'isolé' ou 'v8' (garder le statut de v8)."""
    why, neg, pos = [], 0, 0
    in_core = e in CORE
    if q and q[1] >= 4.0 and q[0] > 0:
        neg += 1; why.append(f"fp8 net (z {q[1]:+.1f})")
    if r and r[0] < 0.01 and e not in MARK:
        neg += 1; why.append("RUT rejette toutes les hypothèses de base")
    if c is not None and c < 0.35:
        neg += 1; why.append(f"classifieur simple bas ({c:.2f})")
    if d and d[0] is not None and d[0] < 0.20 and e not in ("ep-20", "ep-23"):  # le profond rate les personas
        neg += 1; why.append(f"profond bas ({d[0]:.2f})")
    if c is not None and c > 0.65:
        pos += 1
    if d and d[0] is not None and d[0] > 0.65:
        pos += 1
    if r and r[0] >= 0.05:
        pos += 1
    h = HIST.get(e)
    if e in MARK and MARK[e] in ("persona", "watermark", "T=1", "consigne", "consigne(Claude)"):
        return "noyau", [f"marqueur : {MARK[e]}"] + why
    if h and h[0] > 0:
        return "noyau", [f"historique : {h[1]}"]
    if h and h[0] < 0:
        return "isolé", [f"historique : {h[1]}"]
    if in_core and neg >= 2:
        return "isolé", why
    if (not in_core) and neg == 0 and pos >= 2:
        return "noyau", [f"{pos} preuves statistiques favorables, aucune négative"]
    return "v8", why


def main():
    R, Q, Z, Cs, D = rut(), quant(), stack(), simple(), deep()
    rows = []
    for fam in "ABC":
        for e in sorted(x for x in FAM if FAM[x] == fam):
            v, why = verdict(e, Q.get(e), R.get(e), Cs.get(e), D.get(e))
            rows.append((fam, e, v, why))
    print(f"{'ep':6s} {'v8':5s} {'marqueur':18s} {'fp8 z':>6s} {'RUT S':>6s} {'simple':>6s} {'profond':>13s} {'verdict':8s} raisons")
    changes = []
    for fam, e, v, why in rows:
        q, r, c, d = Q.get(e), R.get(e), Cs.get(e), D.get(e)
        ds = f"{d[0]:.2f}/{d[1]:.2f}" if d and d[0] is not None else "-"
        st = "noyau" if e in CORE else "isolé"
        print(f"{e:6s} {st:5s} {MARK.get(e, ''):18s} {q[1] if q else float('nan'):6.1f} {r[0] if r else float('nan'):6.2f} "
              f"{c if c is not None else float('nan'):6.2f} {ds:>13s} {v:8s} {'; '.join(why)[:90]}")
        if v in ("noyau", "isolé") and v != st:
            changes.append((e, st, v))
    print("\nChangements proposés par rapport à v8 :", changes or "aucun")


if __name__ == "__main__":
    main()
