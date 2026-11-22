"""Classifieur « modèle de base (S) ou poids modifiés (T) » appris sur les données DGX à vérité connue, vérifié sur des
bases tenues à l'écart et sur les endpoints synthétiques, puis appliqué aux vrais endpoints. Sans GPU, sans requête.

Caractéristiques d'un (pseudo-)endpoint = pour chaque statistique boîte blanche (score_wb_r4), écart moyen de ses
réponses à la RÉFÉRENCE plain du même prompt (moyenne des réponses de référence), et le t correspondant (écart / erreur-type).
  DGX         jureca_retour16/dgx_<base>.jsonl ; référence = plain, répétitions 4..7 ; pseudo-endpoints = tirages de
              N_RESP réponses d'une variante (comme un vrai endpoint) ; étiquette S si le modèle = la base (déguisements).
  vrais ep    familles A, B, C : wball_<fam>.jsonl (jureca_retour15) si présent, sinon les scores de build_v17 ;
  synthétiques syn-<fam>-* dans wball_<fam>.jsonl ; vérité dans synth_truth_<fam>.csv.
  Référence des vrais/synthétiques = plain_t07 local (extra_scores, calib_40_59, calib_60_70), comme build_v8/v17.
Usage : python local/clf_dgx.py [--n-resp 95] [--boot 40]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = Path(__file__).resolve().parents[1]
STATS = ["bf16_mean_loglik_40", "bf16_ll_plus_ent_40", "bf16_mean_entropy_40", "bf16_frac_rank_gt50", "bf16_frac_top1",
         "bf16_q50_loglik_40", "bf16_mean_logrank_40", "delta_fp8sim_ll40", "delta_int8_ll40", "delta_nf4_ll40"]
FAM_NAME = {"A": "Llama-3.2-1B-Instruct", "B": "Qwen2.5-1.5B-Instruct", "C": "Qwen2.5-3B-Instruct"}
MIN_TOK = 40


def ok(r):
    return r.get("bf16_n_tok", 0) >= MIN_TOK and all(s in r for s in STATS)


def feats(rows, refmean):
    d = [[r[s] - refmean[r["id"]][s] for s in STATS] + [np.log(r["bf16_n_tok"]) - refmean[r["id"]]["lntok"]] for r in rows if r["id"] in refmean]
    d = np.array(d, float)
    if len(d) < 10:
        return None
    m = d.mean(0)
    se = d.std(0, ddof=1) / np.sqrt(len(d)) + 1e-9
    return np.concatenate([m, m / se])


def refmeans(refrows):
    acc = defaultdict(list)
    for r in refrows:
        if ok(r):
            acc[r["id"]].append([r[s] for s in STATS] + [np.log(r["bf16_n_tok"])])
    return {k: dict(zip(STATS + ["lntok"], np.mean(v, 0))) for k, v in acc.items() if len(v) >= 2}


def load_dgx(n_resp, boot, rng):
    X, y, grp, kind = [], [], [], []
    for f in sorted((H / "jureca_retour16").glob("dgx_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        base = rows[0]["base"]
        ref = refmeans([r for r in rows if r["role"] == "ref"])
        byv = defaultdict(list)
        for r in rows:
            if r["role"] == "ep" and ok(r):
                byv[r["variant"]].append(r)
        for var, rs in byv.items():
            lab = int(rs[0]["model"] == base)
            for _ in range(boot):
                sub = [rs[i] for i in rng.choice(len(rs), size=min(n_resp, len(rs)), replace=False)]
                v = feats(sub, ref)
                if v is not None:
                    X.append(v); y.append(lab); grp.append(base); kind.append(var)
        print(f"  DGX {base} : {len(byv)} variantes, {sum(1 for g in grp if g == base)} pseudo-endpoints")
    return np.array(X), np.array(y), np.array(grp), np.array(kind)


def load_family(fam):
    """Renvoie {endpoint: [lignes]} (vrais + synthétiques) et la moyenne de référence plain_t07 par prompt."""
    ref = []
    for l in open(H / "brief_retour3" / "extra_scores.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "ref" and r["family"] == fam and r["endpoint"].endswith("__plain_t07"):
            ref.append(dict(r, id=int(r["id"])))
    for p in (H / "brief_retour4" / "calib_40_59_scores.jsonl", H / "jureca_retour14" / f"calib_60_70_scores_{fam}.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if r["family"] == fam and r["config"] == "plain_t07":
                ref.append(dict(r, id=int(r["id"])))
    eps = defaultdict(list)
    wall = H / "jureca_retour15" / f"wball_{fam}.jsonl"
    if wall.exists():
        for l in open(wall):
            r = json.loads(l)
            if ok(r):
                eps[r["endpoint"]].append(dict(r, id=int(r["id"])))
    else:
        fam8 = {r["endpoint"]: r["family"] for r in csv.DictReader(open(H / "submission_v8.csv"))}
        seen = set()
        for p in (H / "jureca_retour14" / f"wb_r56_scores_{fam}.jsonl", H / "brief_retour3" / "wb40_scores.jsonl", H / "brief_retour4" / "wb_40_59_scores.jsonl"):
            for l in open(p):
                r = json.loads(l)
                key = (r["endpoint"], int(r["id"]), int(r.get("repeat", 0)))
                if fam8.get(r["endpoint"]) == fam and ok(r) and key not in seen:
                    seen.add(key)
                    eps[r["endpoint"]].append(dict(r, id=int(r["id"])))
    return eps, refmeans(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resp", type=int, default=95)
    ap.add_argument("--boot", type=int, default=40)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    print("chargement DGX")
    X, y, grp, kind = load_dgx(a.n_resp, a.boot, rng)
    models = {"logistique": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
              "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300)}
    print("\n=== validation : une base tenue à l'écart (entraîné sur les autres) ===")
    for name, mk in models.items():
        for b in sorted(set(grp)):
            tr, te = grp != b, grp == b
            if len(set(y[te])) < 2:
                continue
            m = mk().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], p)
            # taux d'erreur par type de variante, seuil 0,5
            err = defaultdict(list)
            for k, yy, pp in zip(kind[te], y[te], p):
                t = "S" if yy else ("lora0.3" if "0.3" in k else "lora>=0.7" if "lora" in k else "quant")
                err[t].append(pp > 0.5)
            rates = "  ".join(f"{t}: P(classé S)={np.mean(v):.2f}" for t, v in sorted(err.items()))
            print(f"  {name:10s} test {b:8s} AUC {auc:.3f} | {rates}")
    final = {name: mk().fit(X, y) for name, mk in models.items()}
    print("\n=== application aux familles (P(S) moyenne des deux modèles ; synthétiques avec leur vérité) ===")
    truth = {}
    for f in (H / "jureca_retour15").glob("synth_truth_*.csv"):
        for e, fam, mdl in csv.reader(open(f)):
            truth[e] = mdl
    v8 = {r["endpoint"]: r["model"] for r in csv.DictReader(open(H / "submission_v8.csv"))}
    for fam in "ABC":
        eps, ref = load_family(fam)
        out = []
        for e, rs in eps.items():
            v = feats(rs, ref)
            if v is None:
                continue
            p = np.mean([m.predict_proba(v[None])[0, 1] for m in final.values()])
            lab = (f"vérité {truth[e]}" if e in truth else ("noyau v8" if v8.get(e, "").endswith("1") else "seul v8"))
            out.append((e.startswith("syn"), -p, f"  {e:22s} n={len(rs):4d}  P(S)={p:.2f}  ({lab})"))
        print(f"\n--- famille {fam} ---")
        for _, _, s in sorted(out):
            print(s)


if __name__ == "__main__":
    main()
