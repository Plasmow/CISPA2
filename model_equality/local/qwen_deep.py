"""Classifieur PROFOND spécial Qwen (réseau de local/deep_clf.py) sur les caractéristiques par token des données spéciales
Qwen (local/tok_feats.py --qgen), validé en laissant un ADAPTATEUR LoRA de côté (comme le LoRA inconnu du serveur), puis appliqué
aux endpoints réels de la famille. Aucune requête vers les endpoints.
Usage : python local/qwen_deep.py --tag 15|3 [--dir jureca_retour19] [--steps 2000]
Sortie : <dir>/qdeep_<tag>.json et tableau affiché (AUC et taux « classé S » par type de variante, par adaptateur laissé de côté).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "local"))
import deep_clf as D  # noqa: E402


def kind_of(kind):
    if "@" in kind:
        s = float(kind.split("@")[1])
        return "lora0.3" if s <= 0.3 else "lora0.7" if s <= 0.7 else "lora1.0" if s <= 1.0 else "lora1.5"
    return "fp8" if "fp8" in kind else "S"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["15", "3"])
    ap.add_argument("--dir", default=str(H / "jureca_retour19"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-resp", type=int, default=95)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    by = D.load(a.dir)
    pre = f"qw{a.tag}-"
    q = {s: rs for s, rs in by.items() if s.startswith(pre)}
    real = {s: rs for s, rs in by.items() if s.startswith("ep-")}
    allx = np.concatenate([r["x"] for rs in q.values() for r in rs[:40]])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    adapters = sorted({rs[0]["adapter"] for rs in q.values()} - {"S", "fp8"})
    print(f"Qwen {a.tag} : {len(q)} pseudo-endpoints, adaptateurs {adapters}, vrais endpoints {len(real)}", flush=True)
    res = {"held_out": {}, "real": {}}
    for raw in (False, True):
        mode = "valeurs brutes" if raw else "avec écart à la référence"
        for g in adapters:
            test = {s: rs for s, rs in q.items() if rs[0]["adapter"] == g or (rs[0]["adapter"] in ("S", "fp8") and s.endswith("-1"))}
            train = [(rs[0]["label"], rs) for s, rs in q.items() if s not in test and not (rs[0]["adapter"] in ("S", "fp8") and s.endswith("-1"))]
            print(f"  [{mode}] adaptateur laissé de côté : {g}", flush=True)
            net = D.train(train, mu, sd, dev, a.steps, a.n_resp, rng, raw)
            ys, ps, rates = [], [], defaultdict(list)
            for s, rs in test.items():
                p = D.predict(net, rs, mu, sd, dev, a.n_resp, rng)
                ys.append(rs[0]["label"]); ps.append(p); rates[kind_of(rs[0]["kind"])].append(p > 0.5)
            res["held_out"][f"{g}|{'raw' if raw else 'delta'}"] = dict(auc=D.auc(ys, ps), rates={k: float(np.mean(v)) for k, v in rates.items()})
            print(f"  -> {g} [{mode}] : AUC {D.auc(ys, ps):.3f} | P(classé S) : " + "  ".join(f"{k}={float(np.mean(v)):.2f}" for k, v in sorted(rates.items())), flush=True)
    print("  modèle final (tous les adaptateurs) -> vrais endpoints", flush=True)
    net = D.train([(rs[0]["label"], rs) for rs in q.values()], mu, sd, dev, a.steps, a.n_resp, rng, False)
    for s, rs in sorted(real.items()):
        res["real"][s] = D.predict(net, rs, mu, sd, dev, a.n_resp, rng, k=20)
    print("  vrais endpoints : " + "  ".join(f"{s[3:]}:{p:.2f}" for s, p in sorted(res["real"].items(), key=lambda kv: -kv[1])), flush=True)
    json.dump(res, open(Path(a.dir) / f"qdeep_{a.tag}.json", "w"), indent=1)
    print("terminé", flush=True)


if __name__ == "__main__":
    main()
