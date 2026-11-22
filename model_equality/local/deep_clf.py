"""Classifieur PROFOND « modèle de base (S) ou poids modifiés (T) » sur les séquences de caractéristiques par token
(local/tok_feats.py). Aucune requête vers les endpoints.

ÉCART À LA RÉFÉRENCE : chaque réponse reçoit, en plus de ses tokens, le vecteur DELTA = (résumé de la réponse - moyenne des
résumés de référence du MÊME prompt) / écart-type intra-prompt des références (référence = modèle de base non modifié :
plain DGX répétitions 4..7, ou plain_t07 locaux pour les vrais endpoints). C'est cet écart, pas la valeur brute, qui porte
l'information. L'option --raw le supprime (ablation) pour mesurer ce qu'il apporte.
Architecture (Deep Sets) : chaque réponse = séquence de tokens à 5 caractéristiques -> 2 convolutions 1D -> moyenne et
maximum masqués -> vecteur de 64 ; un endpoint = moyenne des vecteurs de ses réponses (~95) -> MLP -> logit P(S).
Entraînement : pseudo-endpoints tirés au hasard (95 réponses d'une même variante DGX), classes équilibrées.
Validation : (1) une base DGX tenue à l'écart à tour de rôle ; (2) endpoints synthétiques (vrais modèles, vrais prompts,
vérité connue). Puis modèle final (toutes les bases DGX + synthétiques) appliqué aux vrais endpoints.
Sortie : <dir>/deep_results.json et tableau affiché.
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
NF = 5


def summ(x):
    return np.concatenate([x.mean(0), [np.log(len(x))]]).astype(np.float32)


def load(d):
    src, refs = [], defaultdict(lambda: defaultdict(list))
    for f in sorted(Path(d).glob("tok_*.npz")):
        z = np.load(f)
        feats, off = z["feats"].astype(np.float32), z["off"]
        meta = [json.loads(l) for l in open(f.with_suffix(".jsonl"))]
        for i, m in enumerate(meta):
            x = feats[off[i] : off[i + 1]][:L]
            if m["src"].startswith("REF:"):
                refs[m["group"]][m["id"]].append(summ(x))
            else:
                src.append(dict(m, x=x))
    # moyenne de référence par (groupe, prompt) et écart-type intra-prompt poolé
    mu_ref, sd_ref = {}, {}
    for g, byid in refs.items():
        mu_ref[g] = {i: np.mean(v, 0) for i, v in byid.items() if len(v) >= 2}
        var = [np.var(v, 0, ddof=1) for v in byid.values() if len(v) >= 3]
        sd_ref[g] = np.sqrt(np.mean(var, 0)) + 1e-6
    by = defaultdict(list)
    for r in src:
        g, i = r["group"], r["id"]
        if i in mu_ref.get(g, {}):
            r["delta"] = np.clip((summ(r["x"]) - mu_ref[g][i]) / sd_ref[g], -10, 10).astype(np.float32)
        else:
            r["delta"] = None
        by[r["src"]].append(r)
    for k in list(by):
        by[k] = [r for r in by[k] if r["delta"] is not None]
        if not by[k]:
            del by[k]
    return by


ND = NF + 1


class Net(nn.Module):
    def __init__(self, raw=False):
        super().__init__()
        self.raw = raw
        self.conv = nn.Sequential(nn.Conv1d(NF, 48, 5, padding=2), nn.GELU(), nn.Conv1d(48, 48, 5, padding=2), nn.GELU())
        self.dl = nn.Sequential(nn.Linear(ND, 32), nn.GELU())
        self.resp = nn.Sequential(nn.Linear(96 + 32, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, mask, delta):  # x [B, R, L, F], mask [B, R, L], delta [B, R, ND]
        B, R, Lt, F = x.shape
        h = self.conv(x.view(B * R, Lt, F).transpose(1, 2)).transpose(1, 2)  # [BR, L, 48]
        m = mask.view(B * R, Lt, 1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(1).values
        d = delta.view(B * R, ND)
        if self.raw:
            d = d * 0
        e = self.resp(torch.cat([mean, mx, self.dl(d)], 1)).view(B, R, -1).mean(1)
        return self.head(e).squeeze(1)


def batchify(sets, mu, sd, dev):
    R = max(len(s) for s in sets)
    x = np.zeros((len(sets), R, L, NF), np.float32)
    m = np.zeros((len(sets), R, L), np.float32)
    dl = np.zeros((len(sets), R, ND), np.float32)
    for b, s in enumerate(sets):
        for r, row in enumerate(s):
            n = len(row["x"])
            x[b, r, :n] = (row["x"] - mu) / sd
            m[b, r, :n] = 1
            dl[b, r] = row["delta"]
    return torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev), torch.from_numpy(dl).to(dev)


def train(pools, mu, sd, dev, steps, n_resp, rng, raw=False):
    """pools : liste de (label, [réponses]) ; renvoie le réseau entraîné."""
    net = Net(raw).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = [p for p in pools if p[0] == 1]
    neg = [p for p in pools if p[0] == 0]
    for step in range(steps):
        chosen = [pos[i] for i in rng.integers(0, len(pos), 8)] + [neg[i] for i in rng.integers(0, len(neg), 8)]
        sets = [[rs[i] for i in rng.choice(len(rs), min(n_resp, len(rs)), replace=False)] for _, rs in chosen]
        x, m, dl = batchify(sets, mu, sd, dev)
        y = torch.tensor([float(l) for l, _ in chosen], device=dev)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x, m, dl), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"    pas {step} perte {float(loss):.3f}", flush=True)
    return net.eval()


@torch.no_grad()
def predict(net, rs, mu, sd, dev, n_resp, rng, k=10):
    ps = []
    for _ in range(k):
        s = [rs[i] for i in rng.choice(len(rs), min(n_resp, len(rs)), replace=False)]
        x, m, dl = batchify([s], mu, sd, dev)
        ps.append(float(torch.sigmoid(net(x, m, dl))[0]))
    return float(np.mean(ps))


def auc(y, p):
    y, p = np.array(y), np.array(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def kind_of(src):
    v = src.split(":", 1)[-1]
    if "lora" in v:
        return "lora0.3" if "0.3" in v else "lora>=0.7"
    if any(q in v for q in ("int8", "nf4", "fp8")):
        return "quant"
    return "S"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(H / "jureca_retour17"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-resp", type=int, default=95)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    by = load(a.dir)
    dgx = {s: rs for s, rs in by.items() if ":" in s}
    syn = {s: rs for s, rs in by.items() if s.startswith("syn-") and rs[0]["label"] in (0, 1)}
    real = {s: rs for s, rs in by.items() if s.startswith("ep-")}
    allx = np.concatenate([r["x"] for rs in dgx.values() for r in rs[:50]])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    bases = sorted({rs[0]["group"] for rs in dgx.values()})
    print(f"DGX : {len(dgx)} variantes ({bases}) ; synthétiques étiquetés : {len(syn)} ; vrais endpoints : {len(real)}", flush=True)
    res = {"held_out": {}, "synth": {}, "real": {}}
    for raw in (False, True):  # False = avec l'écart à la référence ; True = ablation (valeurs brutes seules)
        mode = "valeurs brutes (ablation)" if raw else "avec écart à la référence"
        for b in bases:
            tr = [(rs[0]["label"], rs) for s, rs in dgx.items() if rs[0]["group"] != b]
            te = {s: rs for s, rs in dgx.items() if rs[0]["group"] == b}
            if len({rs[0]["label"] for rs in te.values()}) < 2:
                continue
            print(f"  [{mode}] base tenue à l'écart : {b}", flush=True)
            net = train(tr, mu, sd, dev, a.steps, a.n_resp, rng, raw)
            ys, ps, kinds = [], [], defaultdict(list)
            for s, rs in te.items():
                p = predict(net, rs, mu, sd, dev, a.n_resp, rng)
                ys.append(rs[0]["label"]); ps.append(p); kinds[kind_of(s)].append(p > 0.5)
            res["held_out"][f"{b}|{'raw' if raw else 'delta'}"] = dict(auc=auc(ys, ps), rates={k: float(np.mean(v)) for k, v in kinds.items()})
            print(f"  -> {b} [{mode}] : AUC {auc(ys, ps):.3f} ; P(classé S) par type : "
                  + "  ".join(f"{k}={float(np.mean(v)):.2f}" for k, v in sorted(kinds.items())), flush=True)
    print("  modèle entraîné sur toute la DGX -> synthétiques", flush=True)
    net = train([(rs[0]["label"], rs) for rs in dgx.values()], mu, sd, dev, a.steps, a.n_resp, rng)
    ys, ps = [], []
    for s, rs in sorted(syn.items()):
        p = predict(net, rs, mu, sd, dev, a.n_resp, rng)
        res["synth"][s] = p
        ys.append(rs[0]["label"]); ps.append(p)
        print(f"    {s:26s} vérité {'S' if rs[0]['label'] else 'T'}  P(S)={p:.2f}", flush=True)
    print(f"  AUC sur les synthétiques : {auc(ys, ps):.3f}", flush=True)
    print("  modèle final (DGX + synthétiques) -> vrais endpoints", flush=True)
    net = train([(rs[0]["label"], rs) for rs in list(dgx.values()) + list(syn.values())], mu, sd, dev, a.steps, a.n_resp, rng)
    for s, rs in sorted(real.items()):
        res["real"][s] = predict(net, rs, mu, sd, dev, a.n_resp, rng, k=20)
    for fam in "ABC":
        row = sorted(((p, s) for s, p in res["real"].items() if by[s][0]["group"] == fam), reverse=True)
        print(f"  famille {fam} : " + "  ".join(f"{s[3:]}:{p:.2f}" for p, s in row), flush=True)
    json.dump(res, open(Path(a.dir) / "deep_results.json", "w"), indent=1)
    print("terminé", flush=True)


if __name__ == "__main__":
    main()
