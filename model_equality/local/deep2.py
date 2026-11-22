"""Classifieur PROFOND ROBUSTE spécial Qwen (remplace qwen_deep.py). Aucune requête vers les endpoints.

Entrée : séquences de caractéristiques PAR TOKEN (tok_feats.py --qgen --qdata qgen2 --nf4) : log p, entropie, log rang, écarts fp8,
int8 et nf4 par rapport à bf16. Valeurs brutes seulement (l'écart à la référence n'apportait rien).
Robustesse :
  - ENSEMBLE de K réseaux (graines différentes), moyenne et écart-type des P(S) ;
  - AUGMENTATION à l'entraînement : nombre de réponses par endpoint tiré dans [40, 120], 10 % des tokens masqués, bruit gaussien léger ;
  - validation en laissant de côté un GROUPE d'adaptateurs entiers (comme le LoRA inconnu du serveur), sur 4 plis ;
  - ANCRES RÉELLES : endpoints réels dont le statut est établi par plusieurs tests indépendants (watermark, persona, T = 1,
    consigne Claude, fp8). Ils servent d'abord de TEST HORS ENTRAÎNEMENT (modèle sans ancres), puis d'ancres d'entraînement
    pour le modèle final (modèle avec ancres). Les deux prédictions sont rapportées.
Usage : python local/deep2.py --tag 15|3 --dir jureca_retour21 [--k 5] [--steps 2000]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

H = Path(__file__).resolve().parents[1]
L = 256
ANCHORS = {"1": {"ep-10": 1, "ep-16": 1, "ep-31": 1, "ep-29": 1, "ep-15": 0}, "15": {"ep-06": 1, "ep-20": 1, "ep-30": 1, "ep-22": 0}, "3": {"ep-04": 1, "ep-23": 1, "ep-33": 1, "ep-24": 1}}


def load(d, tag):
    z = np.load(Path(d) / f"tok_q2_{tag}.npz")
    feats, off = z["feats"].astype(np.float32), z["off"]
    meta = [json.loads(l) for l in open(Path(d) / f"tok_q2_{tag}.jsonl")]
    by = defaultdict(list)
    for i, m in enumerate(meta):
        if m["src"].startswith("REF:"):
            continue
        by[m["src"]].append(dict(m, x=feats[off[i] : off[i + 1]][:L]))
    return by, feats.shape[1]


class Net(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(nf, 48, 5, padding=2), nn.GELU(), nn.Conv1d(48, 48, 5, padding=2), nn.GELU())
        self.resp = nn.Sequential(nn.Linear(96, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x, mask):
        B, R, Lt, F = x.shape
        h = self.conv(x.view(B * R, Lt, F).transpose(1, 2)).transpose(1, 2)
        m = mask.view(B * R, Lt, 1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(1).values
        e = self.resp(torch.cat([mean, mx], 1)).view(B, R, -1).mean(1)
        return self.head(e).squeeze(1)


def batchify(sets, mu, sd, dev, nf, aug, rng):
    R = max(len(s) for s in sets)
    x = np.zeros((len(sets), R, L, nf), np.float32)
    m = np.zeros((len(sets), R, L), np.float32)
    for b, s in enumerate(sets):
        for r, row in enumerate(s):
            n = len(row["x"])
            v = (row["x"] - mu) / sd
            keep = np.ones(n, np.float32)
            if aug:
                v = v + rng.normal(0, 0.05, v.shape).astype(np.float32)
                keep = (rng.random(n) > 0.10).astype(np.float32)
            x[b, r, :n] = v
            m[b, r, :n] = keep
    return torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)


def train(pools, mu, sd, dev, nf, steps, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Net(nf).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pos = [p for p in pools if p[0] == 1]
    neg = [p for p in pools if p[0] == 0]
    net.train()
    for step in range(steps):
        chosen = [pos[i] for i in rng.integers(0, len(pos), 8)] + [neg[i] for i in rng.integers(0, len(neg), 8)]
        sets = []
        for _, rs in chosen:
            k = int(rng.integers(40, 121))
            sets.append([rs[i] for i in rng.choice(len(rs), min(k, len(rs)), replace=False)])
        x, m = batchify(sets, mu, sd, dev, nf, True, rng)
        y = torch.tensor([float(l) for l, _ in chosen], device=dev)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x, m), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"      pas {step} perte {loss.item():.3f}", flush=True)
    return net.eval()


@torch.no_grad()
def predict(nets, rs, mu, sd, dev, nf, rng, k=8, n_resp=95):
    ps = []
    for net in nets:
        for _ in range(k):
            s = [rs[i] for i in rng.choice(len(rs), min(n_resp, len(rs)), replace=False)]
            x, m = batchify([s], mu, sd, dev, nf, False, rng)
            ps.append(float(torch.sigmoid(net(x, m))[0]))
    return float(np.mean(ps)), float(np.std([np.mean(ps[i * k : (i + 1) * k]) for i in range(len(nets))]))


def auc(y, p):
    y, p = np.array(y), np.array(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def kind_class(kind):
    k = kind.split("#")[0]
    if "@" in k:
        s = float(k.split("@")[1].split("+")[0])
        base = "lora0.3-0.5" if s <= 0.5 else "lora0.7" if s <= 0.7 else "lora1.0-1.5"
        return "lora+déguisement" if "+" in k.split("@")[1] else base
    return "quant" if any(q in k for q in ("fp8", "int8", "nf4")) else "S"


def setup(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    by, nf = load(a.dir, a.tag)
    pre = f"qw{a.tag}-"
    q = {s: rs for s, rs in by.items() if s.startswith(pre)}
    fam_of = {"15": "B", "3": "C", "1": "A"}[a.tag]
    real = {s: rs for s, rs in by.items() if s.startswith("ep-")}
    anchors = {e: lab for e, lab in ANCHORS[a.tag].items() if e in real}
    allx = np.concatenate([r["x"] for rs in q.values() for r in rs[:30]])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    adapters = sorted({rs[0]["adapter"] for rs in q.values()} - {"S", "quant"})
    return dev, q, real, anchors, mu, sd, nf, adapters


def merge(a):
    d = Path(a.dir)
    ys, ps, kinds = [], [], defaultdict(list)
    for f in sorted(d.glob(f"deep2_{a.tag}_fold*.json")):
        r = json.load(open(f))
        ys += r["y"]; ps += r["p"]
        for k, p in zip(r["kind"], r["p"]):
            kinds[k].append(p > 0.5)
    res = {"held_out": dict(auc=auc(ys, ps), rates={k: float(np.mean(v)) for k, v in kinds.items()}), "anchors_check": {}, "real": {}}
    print(f"  VALIDATION (adaptateurs jamais vus, {len(ys)} pseudo-endpoints) : AUC {res['held_out']['auc']:.3f} ; P(classé S) par type :")
    for k, v in sorted(kinds.items()):
        print(f"     {k:18s} n={len(v):3d}  classé S = {np.mean(v):.2f}")
    for f, key in (("final0", "sans_ancres"), ("final1", "avec_ancres")):
        p = d / f"deep2_{a.tag}_{f}.json"
        if p.exists():
            for e, (m, sdv) in json.load(open(p))["real"].items():
                res["real"].setdefault(e, {})[f"p_{key}"] = m
                res["real"][e][f"std_{key}"] = sdv
    anchors = {e: lab for e, lab in ANCHORS[a.tag].items() if e in res["real"]}
    for e, lab in anchors.items():
        r = res["real"][e]
        print(f"     ancre {e} (statut établi {'S' if lab else 'T'}) : P(S) sans ancres = {r.get('p_sans_ancres', float('nan')):.2f} ± {r.get('std_sans_ancres', float('nan')):.2f}")
        res["anchors_check"][e] = dict(label=lab, p=r.get("p_sans_ancres"))
    print("  vrais endpoints : P(S) sans ancres / avec ancres (± écart-type de l'ensemble)")
    for e, r in sorted(res["real"].items(), key=lambda kv: -(kv[1].get("p_sans_ancres") or 0)):
        print(f"     {e}  {r.get('p_sans_ancres', float('nan')):.2f}±{r.get('std_sans_ancres', float('nan')):.2f} / "
              f"{r.get('p_avec_ancres', float('nan')):.2f}±{r.get('std_avec_ancres', float('nan')):.2f}{' [ANCRE]' if e in anchors else ''}")
    json.dump(res, open(d / f"deep2_{a.tag}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["15", "3", "1"])
    ap.add_argument("--dir", default=str(H / "jureca_retour21"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--folds", default="0,1,2,3", help="plis de validation à calculer ('' = aucun)")
    ap.add_argument("--final", default="01", help="0 = modèle sans ancres, 1 = avec ancres ('' = aucun)")
    ap.add_argument("--merge", action="store_true", help="fusionne les résultats partiels et affiche le bilan")
    a = ap.parse_args()
    if a.merge:
        merge(a)
        return
    dev, q, real, anchors, mu, sd, nf, adapters = setup(a)
    rng = np.random.default_rng(0)
    print(f"Qwen {a.tag} : {len(q)} pseudo-endpoints, {len(adapters)} adaptateurs {adapters}, {nf} canaux, ancres {anchors}", flush=True)
    is_copy1 = lambda s: s.split("#")[0].endswith("-1") and q[s][0]["adapter"] in ("S", "quant")
    folds = [adapters[i::4] for i in range(4)]
    for fi in [int(x) for x in a.folds.split(",") if x != ""]:
        grp = folds[fi]
        test = {s: rs for s, rs in q.items() if rs[0]["adapter"] in grp or is_copy1(s)}
        tr = [(rs[0]["label"], rs) for s, rs in q.items() if s not in test]
        print(f"  pli {fi} : adaptateurs laissés de côté {grp}", flush=True)
        nets = [train(tr, mu, sd, dev, nf, a.steps, 10 * fi + j) for j in range(3)]
        out = dict(y=[], p=[], kind=[])
        for s, rs in test.items():
            p, _ = predict(nets, rs, mu, sd, dev, nf, rng)
            out["y"].append(rs[0]["label"]); out["p"].append(p); out["kind"].append(kind_class(rs[0]["kind"]))
        json.dump(out, open(Path(a.dir) / f"deep2_{a.tag}_fold{fi}.json", "w"))
    full = [(rs[0]["label"], rs) for rs in q.values()]
    for part in a.final:
        if part == "0":
            pools, seed0 = full, 100
        else:
            pools, seed0 = full + [(lab, real[e]) for e, lab in anchors.items() for _ in range(3)], 200
        print(f"  modèle {'sans' if part == '0' else 'avec'} ancres (ensemble de {a.k})", flush=True)
        nets = [train(pools, mu, sd, dev, nf, a.steps, seed0 + j) for j in range(a.k)]
        res = {e: predict(nets, rs, mu, sd, dev, nf, rng) for e, rs in real.items()}
        json.dump(dict(real=res), open(Path(a.dir) / f"deep2_{a.tag}_final{part}.json", "w"))
    print("partie terminée", flush=True)


if __name__ == "__main__":
    main()
