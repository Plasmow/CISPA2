"""Deux tests sans requête, sur toutes les réponses déjà collectées d'une famille (ou des ref-*, --only R) :

1. Décodage tronqué. Pour une grille (température, top-p, top-k), sous le modèle de base : nombre de tokens hors de
   l'ensemble autorisé, et test du score exact sur la loi renormalisée (somme de log q + entropie, et variance).
   Ordre des opérations comme HF et vLLM : température, puis top-k, puis top-p sur la loi renormalisée.
2. Quantification fp8. Rapport de vraisemblance (variante - base, en bf16) pour quatre simulations :
     w_tensor    poids, une échelle par tenseur (simulation utilisée jusqu'ici)
     w_channel   poids, une échelle par canal de sortie (optimum-quanto, torchao)
     wa_token    poids par tenseur + activations d'entrée de chaque couche linéaire, une échelle par token (vLLM)
     wca_token   poids par canal + activations par token
   lm_head n'est jamais quantifié. Un endpoint réellement fp8 doit avoir un rapport nettement supérieur aux autres
   endpoints de sa famille pour la simulation la plus proche du serveur ; vérifié sur ref-02 (--only R).
Sortie : <out>/quant_<tag>.jsonl, une ligne par réponse (tokens à partir de la position 8).
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
sys.path.insert(0, str(H / "local"))
from score_wb_r4 import FAMILY_MODELS  # noqa: E402
from wb_dose import batches, encode, pad_batch  # noqa: E402

REF_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
REF_FILES = ["reference_answers.jsonl", "ref_consignes_answers.jsonl", "ref_ep_round1_answers.jsonl",
             "ref_ep_round3_answers.jsonl", "ref_ep_round4_answers.jsonl", "ref_ep_round6_answers.jsonl"]
FMAX = 448.0
MIN_POS = 8
TOP_M = 4096
GRID = [(T, p, k) for T in (0.7, 0.8, 0.9, 1.0) for p in (0.9, 0.95, 0.97, 1.0) for k in (0, 40, 50)]
VARIANTS = ("w_tensor", "w_channel", "wa_token", "wca_token")
F8_NATIVE = [True]


def fp8_round(x):
    """Arrondi au plus proche en float8 e4m3 (x en float32, déjà mis à l'échelle), résultat en float32."""
    x = x.clamp(-FMAX, FMAX)
    if F8_NATIVE[0]:
        return x.to(torch.float8_e4m3fn).to(torch.float32)
    ax = x.abs().clamp(min=1e-30)
    e = torch.floor(torch.log2(ax)).clamp(min=-6.0)  # sous 2^-6 : pas constant des sous-normaux (2^-9)
    step = torch.pow(2.0, e - 3.0)
    return (torch.round(x / step) * step).clamp(-FMAX, FMAX)


def check_fp8(dev):
    x = torch.randn(200000, device=dev) * torch.logspace(-4, 2.6, 200000, device=dev)
    manual_flag = F8_NATIVE[0]
    try:
        native = x.clamp(-FMAX, FMAX).to(torch.float8_e4m3fn).to(torch.float32)
        F8_NATIVE[0] = False
        manual = fp8_round(x)
        F8_NATIVE[0] = True
        print(f"  fp8 natif disponible ; écart max avec l'arrondi manuel : {float((native - manual).abs().max()):.3g}", flush=True)
    except Exception as exc:
        F8_NATIVE[0] = False
        print(f"  fp8 natif indisponible ({type(exc).__name__}) : arrondi manuel (était {manual_flag})", flush=True)


def q_weight(w, per_channel):
    wf = w.float()
    amax = (wf.abs().amax(dim=1, keepdim=True) if per_channel else wf.abs().max()).clamp(min=1e-12)
    scale = FMAX / amax
    return (fp8_round(wf * scale) / scale).to(w.dtype)


def act_hook(_mod, args):
    x = args[0]
    xf = x.float()
    s = (xf.abs().amax(dim=-1, keepdim=True) / FMAX).clamp(min=1.0 / (FMAX * 512.0))
    return ((fp8_round(xf / s) * s).to(x.dtype),) + tuple(args[1:])


def linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n]


@torch.no_grad()
def base_pass(model, tok, rows, bs=6):
    dev = next(model.parameters()).device
    temps = sorted({g[0] for g in GRID})
    for batch in batches(rows, bs):
        ids, att = pad_batch(batch, tok.pad_token_id, dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        for j, r in enumerate(batch):
            s0, n = len(r["ctx"]), len(r["y"])
            r["n"] = max(n - MIN_POS, 0)
            if n <= MIN_POS:
                continue
            z = logits[j, s0 - 1 + MIN_POS : s0 - 1 + n].float()
            y = ids[j, s0 + MIN_POS : s0 + n]
            zt = z.gather(1, y[:, None])
            rank = (z > zt).sum(1)
            in_top = rank < TOP_M
            rk = rank.clamp(max=TOP_M - 1)
            zs = z.topk(min(TOP_M, z.shape[1]), dim=1).values
            nuc, lps = {}, {}
            for T in temps:
                lse = torch.logsumexp(z / T, dim=1, keepdim=True)
                logp = z / T - lse
                p = logp.exp()
                ent = -(p * logp).sum(1)
                var = (p * logp * logp).sum(1) - ent * ent
                lp = logp.gather(1, y[:, None]).squeeze(1)
                lps[T] = float(lp.sum())
                ps = (zs / T - lse).exp()
                for (T2, top_p, top_k) in GRID:
                    if T2 != T:
                        continue
                    if top_p == 1.0 and top_k == 0:
                        allowed, lq, hq, vq, n_open = torch.ones_like(in_top), lp, ent, var, 0
                    else:
                        p1 = ps
                        if top_k > 0:
                            p1 = ps.clone()
                            p1[:, top_k:] = 0
                            p1 = p1 / p1.sum(1, keepdim=True)
                        n_open = int((p1.sum(1) < top_p).sum())
                        keep = ((p1.cumsum(1) - p1) < top_p) & (p1 > 0)
                        q = p1 * keep
                        q = q / q.sum(1, keepdim=True)
                        logq = torch.where(keep, q.clamp(min=1e-30).log(), torch.zeros_like(q))
                        hq = -(q * logq).sum(1)
                        vq = (q * logq * logq).sum(1) - hq * hq
                        allowed = in_top & keep.gather(1, rk[:, None]).squeeze(1)
                        lq = logq.gather(1, rk[:, None]).squeeze(1)
                    nuc[f"{T}|{top_p}|{top_k}"] = [int(allowed.sum()), int((~allowed).sum()), float(((lq + hq) * allowed).sum()),
                                                   float((vq * allowed).sum()), n_open]
            r["nuc"] = nuc
            r["lp"] = [lps[0.7], lps[1.0]]
        del logits


@torch.no_grad()
def lp_pass(model, tok, rows, key, bs=8):
    dev = next(model.parameters()).device
    for batch in batches(rows, bs):
        ids, att = pad_batch(batch, tok.pad_token_id, dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        for j, r in enumerate(batch):
            s0, n = len(r["ctx"]), len(r["y"])
            if n <= MIN_POS:
                continue
            z = logits[j, s0 - 1 + MIN_POS : s0 - 1 + n].float()
            y = ids[j, s0 + MIN_POS : s0 + n]
            r.setdefault("q", {})[key] = [float(torch.log_softmax(z / T, dim=1).gather(1, y[:, None]).sum()) for T in (0.7, 1.0)]
        del logits


def dump(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            if r.get("n", 0) > 0:
                fh.write(json.dumps(dict(kind=r["kind"], name=r["name"], id=r["id"], repeat=r["repeat"], n=r["n"], lp=r["lp"],
                                         q=r.get("q", {}), nuc=r["nuc"])) + "\n")


def load_rows(tag, groups, n_ref):
    rows = []
    if tag == "R":
        per = {}
        for fi, name in enumerate(REF_FILES):
            if not (H / name).exists():
                continue
            for l in open(H / name, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if r["endpoint"].startswith("ref-") and len(r["response"]) >= 300:
                        per.setdefault(r["endpoint"], []).append(dict(kind="ref", name=r["endpoint"], id=f"f{fi}:{r['id']}",
                                                                      repeat=int(r.get("repeat", 0)), prompt=r["prompt"], response=r["response"]))
        rng = random.Random(0)
        for e in sorted(per):
            rng.shuffle(per[e])
            rows += per[e][:n_ref]
        return rows
    fam = {r["endpoint"]: r["family"] for r in csv.DictReader(open(groups, encoding="utf-8"))}
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if fam.get(r["endpoint"]) == tag and r["response"].strip():
                rows.append(dict(kind="ep", name=r["endpoint"], id=str(r["id"]), repeat=int(r.get("repeat", 0)), prompt=r["prompt"], response=r["response"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=["A", "B", "C", "R"])
    ap.add_argument("--out", default=str(H / "jureca_retour10"))
    ap.add_argument("--groups", default=str(H / "submission_v8.csv"))
    ap.add_argument("--n-ref", type=int, default=170)
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"quant_{a.only}.jsonl"
    repo = REF_REPO if a.only == "R" else FAMILY_MODELS[a.only][1]
    rows = load_rows(a.only, a.groups, a.n_ref)
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    print(f"{a.only} ({repo}) : {len(rows)} réponses", flush=True)
    check_fp8(dev)
    encode(tok, rows)
    base_pass(model, tok, rows)
    dump(rows, out_file)
    print("  passe de base et grille de décodage : terminé", flush=True)

    lin = linears(model)
    orig = {n: m.weight.data.clone() for n, m in lin}
    for variant in VARIANTS:
        for n, m in lin:
            m.weight.data = q_weight(orig[n], per_channel=variant in ("w_channel", "wca_token"))
        hooks = [m.register_forward_pre_hook(act_hook) for _, m in lin] if variant in ("wa_token", "wca_token") else []
        lp_pass(model, tok, rows, variant)
        for h in hooks:
            h.remove()
        dump(rows, out_file)
        print(f"  variante {variant} : terminé ({len(lin)} couches linéaires)", flush=True)
    print(f"{a.only} : terminé", flush=True)


if __name__ == "__main__":
    main()
