"""Données d'entraînement SPÉCIALES QWEN : plusieurs adaptateurs LoRA variés (données, rang, cibles, pas) entraînés sur
Qwen2.5-1.5B (famille B) ou Qwen2.5-3B (famille C), appliqués à plusieurs forces, plus les déguisements du modèle de base
et le fp8 ; le tout sur les (prompt, répétition) d'un vrai endpoint de la famille. Puis notation boîte blanche
(bf16, fp8sim, int8, nf4 ; mêmes statistiques que v8). Aucune requête vers les endpoints.
Étiquette : 1 = mêmes poids que le modèle de base (déguisements), 0 = poids modifiés (LoRA, fp8). Groupe = adaptateur
(pour valider en laissant un adaptateur de côté, comme le LoRA inconnu du serveur).
Sortie : <out>/qscore_<tag>.jsonl (une ligne par réponse : endpoint, label, group, kind, id, repeat, <q>_<stat>, delta_*).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
sys.path.insert(0, str(H / "local"))
from score_wb_r4 import FAMILY_MODELS, load_quantized, score_rows_under_model  # noqa: E402
from wb_dose import chat_text  # noqa: E402
from wb_pool import OPENERS, SENTS  # noqa: E402
from wb_stack import quantize, restore  # noqa: E402
from synth_gen import PERSONAS  # noqa: E402

TAGS = {"15": ("B", "ep-32"), "3": ("C", "ep-12"), "1": ("A", "ep-13")}
# (nom, source de données, tranche, rang, cibles, pas, lr)
ADAPTERS = [
    ("alp1", "alpaca", (0, 1500), 8, ["q_proj", "v_proj"], 150, 2e-4),
    ("alp2", "alpaca", (3000, 4500), 16, "all-linear", 150, 1e-4),
    ("code", "code", (0, 1500), 8, ["q_proj", "v_proj"], 150, 2e-4),
    ("dolly", "dolly", (0, 1500), 16, ["q_proj", "k_proj", "v_proj", "o_proj"], 200, 2e-4),
    ("dol2", "dolly", (3000, 4500), 8, "all-linear", 100, 3e-4),
    ("alp3", "alpaca", (1500, 3000), 4, ["q_proj", "v_proj"], 150, 2e-4),
]
STRENGTHS = (0.3, 0.7, 1.0, 1.5)
DECODE = {
    "plain": dict(temperature=0.7, top_p=1.0), "t10": dict(temperature=1.0, top_p=1.0),
    "trc": dict(temperature=0.9, top_p=0.97), "pool": dict(temperature=0.7, top_p=1.0), "persona": dict(temperature=0.7, top_p=1.0),
    "fp8": dict(temperature=0.7, top_p=1.0), "fp8_trc": dict(temperature=0.9, top_p=0.97),
}


def datasets():
    """Jeux d'entraînement : copies locales dans hf_data/ (téléchargées depuis le nœud de connexion) sinon téléchargement."""
    loc = H / "hf_data"
    out = {}

    def get(name, repo):
        if (loc / name).exists():
            return str(loc / name)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, name, repo_type="dataset")

    try:
        d = json.load(open(get("alpaca_data_cleaned.json", "yahma/alpaca-cleaned")))
        out["alpaca"] = [(e["instruction"] + (("\n\n" + e["input"]) if e.get("input") else ""), e["output"]) for e in d]
    except Exception as exc:
        print("alpaca indisponible :", exc, flush=True)
    try:
        d = json.load(open(get("code_alpaca_20k.json", "sahil2801/CodeAlpaca-20k")))
        out["code"] = [(e["instruction"] + (("\n\n" + e["input"]) if e.get("input") else ""), e["output"]) for e in d]
    except Exception as exc:
        print("CodeAlpaca indisponible :", exc, flush=True)
    try:
        rows = [json.loads(l) for l in open(get("databricks-dolly-15k.jsonl", "databricks/databricks-dolly-15k"))]
        out["dolly"] = [(e["instruction"] + (("\n\n" + e["context"]) if e.get("context") else ""), e["response"]) for e in rows]
    except Exception as exc:
        print("dolly indisponible :", exc, flush=True)
    return out


def train_adapter(model, tok, data, r, targets, steps, lr, seed):
    from peft import LoraConfig, get_peft_model
    random.seed(seed)
    torch.manual_seed(seed)
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    pm.train()
    dev = next(pm.parameters()).device
    for step in range(steps):
        batch = random.sample(data, 8)
        texts = [tok.apply_chat_template([{"role": "user", "content": u}, {"role": "assistant", "content": a}], tokenize=False) for u, a in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False).to(dev)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        loss = pm(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"    pas {step} perte {float(loss):.3f}", flush=True)
    return pm.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(TAGS))
    ap.add_argument("--out", default=str(H / "jureca_retour18"))
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="essai : N prompts, 2 adaptateurs, 20 pas")
    a = ap.parse_args()
    fam, tmpl = TAGS[a.tag]
    repo = FAMILY_MODELS[fam][1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["endpoint"] == tmpl and str(r["id"]) not in ("71", "72", "73"):
            items.append((str(r["id"]), int(r.get("repeat", 0)), r["prompt"]))
    if a.limit:
        items = items[: a.limit]
    print(f"Qwen {a.tag} ({repo}) : {len(items)} (prompt, répétition)", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    rng = random.Random(11)
    rows = []

    @torch.no_grad()
    def gen(m, texts, dec):
        tok.padding_side = "left"
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [None] * len(texts)
        for s in range(0, len(order), a.bs):
            idx = order[s : s + a.bs]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = m.generate(**enc, max_new_tokens=256, do_sample=True, top_k=0, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, **dec)
            for i, t in zip(idx, tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)):
                res[i] = t
        tok.padding_side = "right"
        return res

    def add(m, name, group, label, ctx="def", dec_key="plain", copies=1):
        for c in range(copies):
            systems = [(" ".join([rng.choice(OPENERS)] + rng.sample(SENTS, rng.randint(2, 4))) if ctx == "pool"
                        else rng.choice(PERSONAS) if ctx == "persona" else None) for _ in items]
            texts = [chat_text(tok, p, sy) for (_, _, p), sy in zip(items, systems)]
            ep = f"qw{a.tag}-{name}-{c}"
            for (pid, rep, p), resp in zip(items, gen(m, texts, DECODE[dec_key])):
                if resp.strip():
                    rows.append(dict(endpoint=ep, label=label, group=group, kind=name, id=int(pid), repeat=rep, prompt=p, response=resp))
            print(f"  {ep} : {len(items)} réponses", flush=True)

    # 1) modèle de base : déguisements (étiquette 1) et fp8 (étiquette 0)
    for name, ctx in (("plain", "def"), ("t10", "def"), ("trc", "def"), ("pool", "pool"), ("persona", "persona")):
        add(model, name, "S", 1, ctx, name if name in DECODE else "plain", copies=2)
    saved = quantize(model, per_channel=False)
    for name in ("fp8", "fp8_trc"):
        add(model, name, "fp8", 0, "def", name, copies=2)
    restore(model, saved)
    # 2) adaptateurs LoRA variés x forces
    data = datasets()
    adapters = ADAPTERS[:2] if a.limit else ADAPTERS
    for ai, (aname, src, (s0, s1), r, targets, steps, lr) in enumerate(adapters):
        if src not in data:
            print(f"  adaptateur {aname} sauté (jeu {src} indisponible)", flush=True)
            continue
        print(f"  entraînement de l'adaptateur {aname} ({src}[{s0}:{s1}], r={r}, {steps} pas)", flush=True)
        pm = train_adapter(model, tok, data[src][s0:s1], r, targets, 20 if a.limit else steps, lr, 100 + ai)
        base_scaling = {}
        for i, mod in enumerate(pm.modules()):
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                base_scaling[i] = dict(mod.scaling)
        for strength in STRENGTHS:
            for i, mod in enumerate(pm.modules()):
                if i in base_scaling:
                    for k in mod.scaling:
                        mod.scaling[k] = base_scaling[i][k] * strength
            add(pm, f"lora-{aname}@{strength}", aname, 0, "def", "plain", copies=1)
        model = pm.unload()
        model.eval()
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    with open(out / f"qgen_{a.tag}.jsonl", "w", encoding="utf-8") as fh:  # textes, pour local/tok_feats.py --qgen
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("endpoint", "label", "group", "kind", "id", "repeat", "prompt", "response")}, ensure_ascii=False) + "\n")
    # 3) notation boîte blanche
    print(f"{len(rows)} réponses à noter", flush=True)
    per_q = {}
    for qkind in ("bf16", "fp8sim", "int8", "nf4"):
        print(f"  notation sous {qkind}", flush=True)
        mdl = load_quantized(repo, qkind)
        per_q[qkind] = score_rows_under_model(mdl, tok, rows)
        del mdl
        torch.cuda.empty_cache()
    with open(out / f"qscore_{a.tag}.jsonl", "w", encoding="utf-8") as fh:
        for idx, r in enumerate(rows):
            rec = {k: r[k] for k in ("endpoint", "label", "group", "kind", "id", "repeat")}
            rec["family"] = fam
            for qkind, res in per_q.items():
                if res[idx].get("n_tok", 0) > 0:
                    for k, v in res[idx].items():
                        rec[f"{qkind}_{k}"] = v
            if "bf16_mean_loglik_40" in rec:
                for qkind in ("fp8sim", "int8", "nf4"):
                    if f"{qkind}_mean_loglik_40" in rec:
                        rec[f"delta_{qkind}_ll40"] = rec[f"{qkind}_mean_loglik_40"] - rec["bf16_mean_loglik_40"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Qwen {a.tag} : terminé", flush=True)


if __name__ == "__main__":
    main()
