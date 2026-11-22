"""Classifieur SPÉCIAL QWEN, validé en laissant un ADAPTATEUR LoRA de côté (comme le LoRA inconnu du serveur), puis appliqué aux
endpoints des familles B (Qwen-1,5B) et C (Qwen-3B). Sans GPU, sans requête.
Données : jureca_retour18/qscore_{15,3}.jsonl (local/qwen_special.py) + DGX qwen25 (jureca_retour16/dgx_qwen25.jsonl, pour B).
Caractéristiques d'un pseudo-endpoint : mêmes 22 nombres que clf_dgx.py (écarts à la référence plain_t07 du même prompt et
leur t), calculés sur des tirages de N réponses.
Sorties : AUC et taux « classé S » par type (déguisement, fp8, LoRA 0,3 / 0,7 / 1,0 / 1,5), par adaptateur laissé de côté ;
puis P(S) pour chaque endpoint réel. Le taux de LoRA 0,3 classé S dit si un LoRA faible est détectable sur Qwen.
Usage : python local/qwen_clf.py [--n-resp 95] [--boot 30]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "local"))
from clf_dgx import feats, load_family, ok  # noqa: E402

FAM = {"15": "B", "3": "C"}


def load_q(tag):
    rows = [json.loads(l) for l in open(H / "jureca_retour18" / f"qscore_{tag}.jsonl")]
    return [dict(r, id=int(r["id"])) for r in rows if ok(r)]


def strength_of(kind):
    if "@" in kind:
        s = float(kind.split("@")[1])
        return "lora0.3" if s <= 0.3 else "lora0.7" if s <= 0.7 else "lora1.0" if s <= 1.0 else "lora1.5"
    return "fp8" if "fp8" in kind else "S"


def pseudo(rows_by_ep, ref, n_resp, boot, rng):
    X, y, grp, kd = [], [], [], []
    for ep, rs in rows_by_ep.items():
        for _ in range(boot):
            sub = [rs[i] for i in rng.choice(len(rs), min(n_resp, len(rs)), replace=False)]
            v = feats(sub, ref)
            if v is not None:
                X.append(v); y.append(rs[0]["label"]); grp.append(rs[0]["group"]); kd.append(strength_of(rs[0]["kind"]))
    return np.array(X), np.array(y), np.array(grp), np.array(kd)


def auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resp", type=int, default=95)
    ap.add_argument("--boot", type=int, default=30)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    mk = {"logistique": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
          "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300)}
    for tag, fam in FAM.items():
        if not (H / "jureca_retour18" / f"qscore_{tag}.jsonl").exists():
            print(f"qscore_{tag}.jsonl absent : famille {fam} ignorée")
            continue
        eps, ref = load_family(fam)
        real = {e: rs for e, rs in eps.items() if e.startswith("ep-")}
        byep = defaultdict(list)
        for r in load_q(tag):
            byep[r["endpoint"]].append(r)
        X, y, grp, kd = pseudo(byep, ref, a.n_resp, a.boot, rng)
        print(f"\n===== Qwen {tag} (famille {fam}) : {len(byep)} pseudo-endpoints, {len(X)} échantillons ; adaptateurs : "
              f"{sorted(set(grp) - {'S', 'fp8'})}")
        for name, f in mk.items():
            print(f"--- {name} : un adaptateur laissé de côté (entraîné sur les autres adaptateurs + déguisements + fp8)")
            for g in sorted(set(grp) - {"S", "fp8"}):
                te = (grp == g) | (np.arange(len(X)) % 5 == 0) & np.isin(grp, ["S", "fp8"])
                tr = ~((grp == g) | ((np.arange(len(X)) % 5 == 0) & np.isin(grp, ["S", "fp8"])))
                m = f().fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                rates = defaultdict(list)
                for k, pp in zip(kd[te], p):
                    rates[k].append(pp > 0.5)
                print(f"  adaptateur {g:6s} AUC {auc(y[te], p):.3f} | P(classé S) : "
                      + "  ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(rates.items())))
        finals = {n: f().fit(X, y) for n, f in mk.items()}
        print(f"--- vrais endpoints de la famille {fam} : P(S) (moyenne logistique + boosting)")
        out = []
        for e, rs in real.items():
            v = feats(rs, ref)
            if v is not None:
                out.append((-np.mean([m.predict_proba(v[None])[0, 1] for m in finals.values()]), e, len(rs)))
        for p, e, n in sorted(out):
            print(f"  {e}  n={n:3d}  P(S)={-p:.2f}")


if __name__ == "__main__":
    main()
