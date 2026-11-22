"""Round 4 DGX — Tâche 1 (rapide, ~6 min sur 1x A100) :
Complète le scoring white-box sur TOUS les 60 prompts (0..59) avec appariement exact par prompt !

1. Note les réponses ep-* des prompts 40..59 (`ep_answers.jsonl`, id >= 40) sous bf16, fp8sim, int8, nf4
   -> écrit `wb_40_59_scores.jsonl` (qui complète `wb40_scores.jsonl` pour couvrir 0..59 = 1 980 réponses !).
2. Note les références `cands40/` (ids 20..39, qui manquaient dans calib_scores.jsonl) et `cands_r4/` (ids 40..59)
   avec EXACTEMENT la même fonction white-box -> écrit `calib_20_59_scores.jsonl`.
   Ainsi, chaque prompt p in 0..59 possède sa propre moyenne et son propre écart-type de référence (plain_t07, plain_t10, fp8sim_t07, int8_t07) !
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

FAMILY_MODELS = {
    "A": ("Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"),
    "B": ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    "C": ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
}


def quantize_fp8_inplace(model):
    fmax = torch.finfo(torch.float8_e4m3fn).max
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                w = mod.weight.data
                amax = w.abs().max().clamp(min=1e-12)
                scale = (fmax / amax).to(torch.float32)
                w_fp8 = (w.to(torch.float32) * scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
                mod.weight.data = (w_fp8.to(torch.float32) / scale).to(w.dtype)
    return model


def load_quantized(repo, qkind):
    if qkind == "bf16":
        return AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    if qkind == "fp8sim":
        m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
        return quantize_fp8_inplace(m)
    if qkind == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(repo, quantization_config=cfg, device_map="auto").eval()
    if qkind == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(repo, quantization_config=cfg, device_map="auto").eval()
    raise ValueError(qkind)


def score_rows_under_model(model, tok, rows, max_resp_tokens=260, batch_size=8):
    dev = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ctx_cache = {}

    def get_ctx(p):
        if p not in ctx_cache:
            msgs = [{"role": "user", "content": p}]
            try:
                s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ctx_cache[p] = tok(s, add_special_tokens=False)["input_ids"]
        return ctx_cache[p]

    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["response"]))
    results = [None] * len(rows)

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        seqs, spans = [], []
        for idx in idxs:
            r = rows[idx]
            c = get_ctx(r["prompt"])
            y = tok(r["response"], add_special_tokens=False)["input_ids"][:max_resp_tokens]
            seqs.append(c + y)
            spans.append((len(c), len(c) + len(y)))
        T = max((len(s) for s in seqs), default=1)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long, device=dev)
        att = torch.zeros((len(seqs), T), dtype=torch.long, device=dev)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
            att[j, : len(s)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits
        for j, idx in enumerate(idxs):
            s0, s1 = spans[j]
            n_tok = s1 - s0
            if n_tok <= 0:
                results[idx] = dict(n_tok=0)
                continue
            lg = logits[j, s0 - 1 : s1 - 1].float()
            tgt = ids[j, s0:s1]
            lt = lg.gather(1, tgt[:, None]).squeeze(1)
            lse = torch.logsumexp(lg, dim=1)
            lp = lt - lse
            probs = torch.softmax(lg, dim=1)
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)
            rank = (lg > lt[:, None]).sum(1) + 1
            logrank = torch.log(rank.float())

            sl = lp[40:] if n_tok > 45 else lp
            se = ent[40:] if n_tok > 45 else ent
            sr = logrank[40:] if n_tok > 45 else logrank
            sl_np = sl.detach().cpu().numpy()

            results[idx] = dict(
                n_tok=int(n_tok),
                mean_loglik=float(lp.mean()),
                mean_entropy=float(ent.mean()),
                ll_plus_ent=float((lp + ent).mean()),
                mean_loglik_40=float(sl.mean()),
                q25_loglik_40=float(np.percentile(sl_np, 25)),
                q50_loglik_40=float(np.percentile(sl_np, 50)),
                q75_loglik_40=float(np.percentile(sl_np, 75)),
                mean_entropy_40=float(se.mean()),
                ll_plus_ent_40=float((sl + se).mean()),
                mean_logrank_40=float(sr.mean()),
                frac_top1=float((rank == 1).float().mean()),
                frac_top5=float((rank <= 5).float().mean()),
                frac_rank_gt50=float((rank > 50).float().mean()),
            )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="ep_answers.jsonl")
    ap.add_argument("--groups", default="submission_v6.csv")
    ap.add_argument("--min-ep-id", type=int, default=40)
    ap.add_argument("--cands40-dir", default="")
    ap.add_argument("--cands-r4-dir", default="")
    ap.add_argument("--out-ep", default="wb_40_59_scores.jsonl")
    ap.add_argument("--out-calib", default="calib_20_59_scores.jsonl")
    ap.add_argument("--purge", action="store_true")
    a = ap.parse_args()

    fam_map = {r["endpoint"]: r["family"] for r in csv.DictReader(open(a.groups, encoding="utf-8"))}
    ep_rows_all = [
        r for r in map(json.loads, open(a.answers, encoding="utf-8"))
        if int(r["id"]) >= a.min_ep_id and r["response"].strip()
    ]

    fh_ep = open(a.out_ep, "w", encoding="utf-8")
    fh_cal = open(a.out_calib, "w", encoding="utf-8")

    for fcode, (short, repo) in FAMILY_MODELS.items():
        f_ep = [r for r in ep_rows_all if fam_map.get(r["endpoint"]) == fcode]
        f_cal = []
        for dpath, lo_id in [(a.cands40_dir, 20), (a.cands_r4_dir, 40)]:
            if dpath and Path(dpath).is_dir():
                for cf in sorted(Path(dpath).glob(f"{short}__*.jsonl")):
                    cfg_name = cf.stem.split("__")[1]
                    for l in open(cf, encoding="utf-8"):
                        r = json.loads(l)
                        if int(r["id"]) >= lo_id and r["response"].strip():
                            f_cal.append({"config": cfg_name, "family": fcode, "id": int(r["id"]), "repeat": r["repeat"],
                                          "prompt": r["prompt"], "response": r["response"]})

        combined = f_ep + f_cal
        print(f"\n=== Famille {fcode} ({repo}) : {len(f_ep)} réponses ep-* (id>={a.min_ep_id}) + {len(f_cal)} réf. calib ===", flush=True)
        if not combined:
            continue

        tok = AutoTokenizer.from_pretrained(repo)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        per_q = {}
        for qkind in ("bf16", "fp8sim", "int8", "nf4"):
            print(f"  -> scoring sous {qkind}...", flush=True)
            mdl = load_quantized(repo, qkind)
            per_q[qkind] = score_rows_under_model(mdl, tok, combined)
            del mdl
            torch.cuda.empty_cache()

        for idx, r in enumerate(combined):
            is_ep = idx < len(f_ep)
            rec = (
                {"endpoint": r["endpoint"], "family": fcode, "id": str(r["id"]), "repo": repo}
                if is_ep
                else {"config": r["config"], "family": fcode, "id": int(r["id"]), "repeat": r["repeat"]}
            )
            for qkind, res_list in per_q.items():
                if res_list[idx].get("n_tok", 0) > 0:
                    for k, v in res_list[idx].items():
                        rec[f"{qkind}_{k}"] = v
            if "bf16_mean_loglik_40" in rec:
                for qkind in ("fp8sim", "int8", "nf4"):
                    if f"{qkind}_mean_loglik_40" in rec:
                        rec[f"delta_{qkind}_ll40"] = rec[f"{qkind}_mean_loglik_40"] - rec["bf16_mean_loglik_40"]
            (fh_ep if is_ep else fh_cal).write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh_ep.flush()
        fh_cal.flush()

        if a.purge:
            hf_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
            shutil.rmtree(hf_root / ("models--" + repo.replace("/", "--")), ignore_errors=True)

    fh_ep.close()
    fh_cal.close()
    print("Terminé : wb_40_59_scores.jsonl et calib_20_59_scores.jsonl écrits.", flush=True)


if __name__ == "__main__":
    main()
