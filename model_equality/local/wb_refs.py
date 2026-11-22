"""Quel modèle de base servent les ref-* ? Puis calibration du test par token sur les ref-* (vérité connue).
Aucune requête réseau vers les endpoints : on note les réponses ref-* déjà collectées.

  --shard i/n : note ~200 réponses de ref-01 (plain) sous chaque modèle candidat de la tranche -> <out>/refid_<i>.json
  --final     : prend le candidat de meilleure log-vraisemblance, puis pour toutes les réponses ref-* :
                  tokens_R.npz / meta_R.jsonl   statistiques par token (local/wb_tokens.py)
                  wb_ref_scores.jsonl           scores par réponse sous bf16 / fp8sim / int8 / nf4 (comme la DGX)
                  wm_refs.json                  détecteur de watermark HuggingFace (clé par défaut) par endpoint
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
sys.path.insert(0, str(H / "local"))
from score_wb_r4 import load_quantized, score_rows_under_model  # noqa: E402
from wb_tokens import save, token_scores  # noqa: E402

CANDIDATES = [
    # Déjà écartés (job 15905170, log-vraisemblance de ref-01 trop basse) : OLMo-2-0425-1B-Instruct (-2,02),
    # Llama-3.2-1B-Instruct (-1,55), Qwen2.5-1.5B-Instruct (-1,48).
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen3-0.6B",
    "unsloth/gemma-3-1b-it",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "unsloth/gemma-3-270m-it",
    "LiquidAI/LFM2-1.2B",
    "Qwen/Qwen3-1.7B",
    "unsloth/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "unsloth/gemma-2-2b-it",
]
MIN_MATCH_LLE = 0.10  # ll+entropie de ref-01 : +0,3 pour le bon modèle à T = 0,7, négatif pour un mauvais modèle
REF_FILES = ["reference_answers.jsonl", "ref_consignes_answers.jsonl", "ref_ep_round1_answers.jsonl",
             "ref_ep_round3_answers.jsonl", "ref_ep_round4_answers.jsonl", "ref_ep_round6_answers.jsonl"]
WM_SCHEMES = [("lefthash", 0.25), ("lefthash", 0.5), ("selfhash", 0.25), ("selfhash", 0.5)]


def ref_rows():
    rows = []
    for fi, name in enumerate(REF_FILES):
        if not (H / name).exists():
            continue
        for l in open(H / name, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r["endpoint"].startswith("ref-") and r["response"].strip():
                    rows.append(dict(kind="ref", name=r["endpoint"], id=f"f{fi}:{r['id']}", repeat=int(r.get("repeat", 0)),
                                     prompt=r["prompt"], response=r["response"]))
    return rows


def load_tok(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def identify(shard, n_shards, outdir, n_resp):
    rows = [r for r in ref_rows() if r["name"] == "ref-01" and len(r["response"]) > 400][:n_resp]
    res = {}
    for ci, repo in enumerate(CANDIDATES):
        if ci % n_shards != shard:
            continue
        try:
            tok = load_tok(repo)
            model = load_quantized(repo, "bf16")
            arr, _ = token_scores(model, tok, rows, 8, verbose=False)
            late = arr["pos"] >= 40
            res[repo] = dict(n_tok=int(late.sum()), mean_lp=float(arr["lp"][late].mean()),
                             lle=float((arr["lp"][late] + arr["ent"][late]).mean()),
                             rank_gt50_per_1000=float((arr["rank"][late] > 50).mean() * 1000),
                             frac_top1=float((arr["rank"][late] == 1).mean()))
            del model
            torch.cuda.empty_cache()
        except Exception as exc:  # modèle protégé, non pris en charge, etc.
            res[repo] = dict(error=f"{type(exc).__name__}: {str(exc)[:200]}")
        print(repo, res[repo], flush=True)
        json.dump(res, open(outdir / f"refid_{shard}.json", "w"), indent=1)


def wm_zscores(tok, vocab_size, texts, device):
    from transformers import WatermarkDetector, WatermarkingConfig
    ids = tok("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)["input_ids"][:, :60000].to(device)
    out = {}
    for scheme, g in WM_SCHEMES:
        mc = type("C", (), {"vocab_size": vocab_size, "is_encoder_decoder": False, "decoder_start_token_id": None,
                            "bos_token_id": tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)})()
        det = WatermarkDetector(model_config=mc, device=device, watermarking_config=WatermarkingConfig(greenlist_ratio=g, seeding_scheme=scheme),
                                ignore_repeated_ngrams=True)
        out[f"{scheme[:4]}_g{int(g * 100)}"] = round(float(det(input_ids=ids, return_dict=True).z_score[0]), 1)
    return out


def final(outdir):
    res = {}
    for f in sorted(outdir.glob("refid_*.json")):
        res.update(json.load(open(f)))
    ok = {k: v for k, v in res.items() if "mean_lp" in v}
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["mean_lp"]):
        print(f"  {k:42s} lp={v['mean_lp']:+.3f} ll+ent={v['lle']:+.3f} rang>50={v['rank_gt50_per_1000']:.1f}/1000 top1={v['frac_top1']:.3f}", flush=True)
    for k, v in res.items():
        if "error" in v:
            print(f"  {k:42s} ERREUR {v['error'][:120]}", flush=True)
    best = max(ok, key=lambda k: ok[k]["lle"]) if ok else None
    print(f"meilleur candidat : {best}", flush=True)
    json.dump(dict(best=best, candidates=res), open(outdir / "refid.json", "w"), indent=1)
    if best is None or ok[best]["lle"] < MIN_MATCH_LLE:
        print("aucun candidat ne correspond au modèle des ref-* : notation complète non lancée", flush=True)
        return

    rows = ref_rows()
    print(f"{len(rows)} réponses ref-* à noter", flush=True)
    tok = load_tok(best)
    model = load_quantized(best, "bf16")
    arrays, n_tok_row = token_scores(model, tok, rows, 8)
    save(outdir, "R", rows, arrays, n_tok_row)
    per_q = {"bf16": score_rows_under_model(model, tok, rows)}
    dev = str(next(model.parameters()).device)
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r["response"])
    try:
        vocab = AutoConfig.from_pretrained(best).vocab_size
        wm = {e: wm_zscores(tok, vocab, by[e], dev) for e in sorted(by)}
        json.dump(wm, open(outdir / "wm_refs.json", "w"), indent=1)
        for e, v in wm.items():
            print("  watermark HF", e, v, flush=True)
    except Exception as exc:
        print("détecteur de watermark en échec :", type(exc).__name__, str(exc)[:200], flush=True)
    del model
    torch.cuda.empty_cache()
    for qkind in ("fp8sim", "int8", "nf4"):
        try:
            mdl = load_quantized(best, qkind)
            per_q[qkind] = score_rows_under_model(mdl, tok, rows)
            del mdl
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"notation {qkind} en échec :", type(exc).__name__, str(exc)[:200], flush=True)
    with open(outdir / "wb_ref_scores.jsonl", "w", encoding="utf-8") as fh:
        for idx, r in enumerate(rows):
            rec = {"endpoint": r["name"], "id": r["id"], "repeat": r["repeat"]}
            for qkind, lst in per_q.items():
                if lst[idx].get("n_tok", 0) > 0:
                    for k, v in lst[idx].items():
                        rec[f"{qkind}_{k}"] = v
            for qkind in ("fp8sim", "int8", "nf4"):
                if f"{qkind}_mean_loglik_40" in rec and "bf16_mean_loglik_40" in rec:
                    rec[f"delta_{qkind}_ll40"] = rec[f"{qkind}_mean_loglik_40"] - rec["bf16_mean_loglik_40"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("terminé", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--out", default=str(H / "jureca_retour7"))
    ap.add_argument("--n-resp", type=int, default=200)
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        identify(i, n, outdir, a.n_resp)
    if a.final:
        final(outdir)


if __name__ == "__main__":
    main()
