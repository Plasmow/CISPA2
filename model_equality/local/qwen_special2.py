"""Données spéciales Qwen, VERSION ROBUSTE (remplace qwen_special.py) : plus d'adaptateurs et plus variés, int8 et nf4, personas
tirés au hasard dans un grand espace, empilements. Aucune requête vers les endpoints.
Un « shard » i/n traite un sous-ensemble d'adaptateurs (les GPU se partagent le travail) ; le shard 0 génère aussi les
déguisements du modèle de base (étiquette 1) et le fp8 ; le shard 1 les quantifications int8 / nf4 et les empilements.
Étiquette : 1 = mêmes poids que le modèle de base (déguisements), 0 = poids modifiés (LoRA, fp8, int8, nf4, empilements).
Sortie : <out>/qgen2_<tag>_<shard>.jsonl (textes) et qscore2_<tag>_<shard>.jsonl (statistiques boîte blanche, comme v8).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
sys.path.insert(0, str(H / "local"))
from score_wb_r4 import FAMILY_MODELS, load_quantized, score_rows_under_model  # noqa: E402
from wb_dose import chat_text  # noqa: E402
from wb_pool import OPENERS, SENTS  # noqa: E402
from wb_stack import quantize, restore  # noqa: E402
import qwen_special as Q  # noqa: E402  (datasets, train_adapter, TAGS, DECODE)

ROLES = ["sea captain", "chemistry teacher", "golf coach", "financial advisor", "librarian", "astronaut", "graphic designer",
         "basketball coach", "medieval historian", "software engineer", "meditation teacher", "food critic", "detective", "nurse",
         "farmer", "jazz musician", "lawyer", "park ranger", "travel blogger", "mechanic", "poet", "chess master", "pilot",
         "marine biologist", "architect", "journalist", "personal trainer", "wizard", "pirate", "robot butler", "stand-up comedian",
         "kindergarten teacher", "diplomat", "carpenter", "film director", "beekeeper", "mountaineer", "barista", "philosopher",
         "customer support agent"]
TRAITS = ["cheerful", "patient", "cautious", "enthusiastic", "calm", "sarcastic", "formal", "playful", "meticulous", "warm",
          "blunt", "curious", "wise", "energetic", "gentle", "witty", "serious", "optimistic", "dramatic", "minimalist"]
STYLES = ["speaks with metaphors from your profession", "always mentions a short personal anecdote", "keeps answers brief and elegant",
          "uses everyday examples", "relates every topic to your hobby", "asks a clarifying question first", "uses numbered steps",
          "avoids jargon", "adds a motivational remark at the end", "mentions risks and caveats", "speaks in short sentences"]
HOBBIES = ["gardening", "football", "cooking", "sailing", "painting", "hiking", "photography", "chess", "birdwatching", "cycling"]

ADAPTERS = [  # (nom, source, tranche, rang, cibles, pas, lr)
    ("a1", "alpaca", (0, 1500), 8, ["q_proj", "v_proj"], 150, 2e-4),
    ("a2", "alpaca", (3000, 4500), 16, "all-linear", 150, 1e-4),
    ("a3", "alpaca", (10000, 11500), 4, ["q_proj", "v_proj"], 200, 3e-4),
    ("a4", "alpaca", (20000, 22000), 32, ["q_proj", "k_proj", "v_proj", "o_proj"], 120, 1e-4),
    ("c1", "code", (0, 1500), 8, ["q_proj", "v_proj"], 150, 2e-4),
    ("c2", "code", (5000, 6500), 16, ["q_proj", "k_proj", "v_proj", "o_proj"], 150, 2e-4),
    ("c3", "code", (12000, 13000), 8, "all-linear", 100, 3e-4),
    ("d1", "dolly", (0, 1500), 16, ["q_proj", "k_proj", "v_proj", "o_proj"], 200, 2e-4),
    ("d2", "dolly", (3000, 4500), 8, "all-linear", 100, 3e-4),
    ("d3", "dolly", (8000, 9500), 4, ["q_proj", "v_proj"], 150, 2e-4),
    ("m1", "mix", (0, 0), 8, ["q_proj", "v_proj"], 150, 2e-4),
    ("m2", "mix", (0, 0), 16, "all-linear", 200, 1.5e-4),
]


def rand_persona(rng):
    return (f"You are a {rng.choice(TRAITS)} {rng.choice(ROLES)} who {rng.choice(STYLES)} and loves {rng.choice(HOBBIES)}."
            + (f" You are also {rng.choice(TRAITS)}." if rng.random() < 0.5 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(Q.TAGS))
    ap.add_argument("--shard", default="0/2")
    ap.add_argument("--n-adapters", type=int, default=12)
    ap.add_argument("--strengths", default="0.3,0.5,0.7,1.5")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--out", default=str(H / "jureca_retour20"))
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="essai : N prompts, 2 adaptateurs, 20 pas")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    strengths = [float(x) for x in a.strengths.split(",")]
    fam, tmpl = Q.TAGS[a.tag]
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
    print(f"Qwen {a.tag} shard {a.shard} ({repo}) : {len(items)} (prompt, répétition)", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    rng = random.Random(1000 + 17 * si)
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
            systems = [(" ".join([rng.choice(OPENERS)] + rng.sample(SENTS, rng.randint(2, 4))) if "pool" in ctx
                        else rand_persona(rng) if "persona" in ctx else None) for _ in items]
            texts = [chat_text(tok, p, sy) for (_, _, p), sy in zip(items, systems)]
            ep = f"qw{a.tag}-{name}-{c}"
            for (pid, rep, p), resp in zip(items, gen(m, texts, Q.DECODE[dec_key])):
                if resp.strip():
                    rows.append(dict(endpoint=ep, label=label, group=group, kind=name, id=int(pid), repeat=rep, prompt=p, response=resp))
            print(f"  {ep} : {len(items)} réponses", flush=True)

    if si == 0:  # modèle de base déguisé (1) et fp8 (0)
        for name, ctx in (("plain", "def"), ("t10", "def"), ("trc", "def"), ("pool", "pool"), ("persona", "persona")):
            add(model, name, "S", 1, ctx, name if name in Q.DECODE else "plain", copies=a.copies)
        saved = quantize(model, per_channel=False)
        for name, ctx, dk in (("fp8", "def", "fp8"), ("fp8_trc", "def", "fp8_trc"), ("fp8+persona", "persona", "fp8")):
            add(model, name, "quant", 0, ctx, dk, copies=a.copies)
        restore(model, saved)
    if si == 1 % sn:  # int8 / nf4 (bitsandbytes), seuls ou avec consigne
        for qk in ("int8", "nf4"):
            qm = load_quantized(repo, qk)
            add(qm, qk, "quant", 0, "def", "plain", copies=a.copies)
            add(qm, f"{qk}+pool", "quant", 0, "pool", "plain", copies=1)
            del qm
            torch.cuda.empty_cache()
    data = Q.datasets()
    if all(k in data for k in ("alpaca", "code", "dolly")):
        data["mix"] = data["alpaca"][30000:31000] + data["code"][14000:15000] + data["dolly"][10000:11000]
    adapters = ADAPTERS[: a.n_adapters]
    adapters = [ad for i, ad in enumerate(adapters) if i % sn == si]
    if a.limit:
        adapters = adapters[:2]
    first = True
    for ai, (aname, src, (s0, s1), r, targets, steps, lr) in enumerate(adapters):
        if src not in data:
            print(f"  adaptateur {aname} sauté (jeu {src} indisponible)", flush=True)
            continue
        print(f"  entraînement de l'adaptateur {aname} ({src}[{s0}:{s1}], r={r}, {steps} pas)", flush=True)
        d = data[src] if src == "mix" else data[src][s0:s1]
        pm = Q.train_adapter(model, tok, d, r, targets, 20 if a.limit else steps, lr, 200 + 31 * si + ai)
        base_scaling = {i: dict(mod.scaling) for i, mod in enumerate(pm.modules()) if hasattr(mod, "scaling") and isinstance(mod.scaling, dict)}

        def set_strength(s):
            for i, mod in enumerate(pm.modules()):
                if i in base_scaling:
                    for k in mod.scaling:
                        mod.scaling[k] = base_scaling[i][k] * s

        for s in strengths:
            set_strength(s)
            add(pm, f"lora-{aname}@{s}", aname, 0, "def", "plain", copies=1)
        if first and si < 2:  # empilements (seulement shards 0 et 1) : LoRA + persona / consigne / décodage tronqué (étiquette 0)
            for s, ctx, dk, tag in ((0.5, "persona", "plain", "persona"), (1.0, "pool", "plain", "pool"), (0.3, "def", "trc", "trc"),
                                    (0.7, "persona", "t10", "persona+t10")):
                set_strength(s)
                add(pm, f"lora-{aname}@{s}+{tag}", aname, 0, ctx, dk, copies=1)
            first = False
        model = pm.unload()
        model.eval()
        torch.cuda.empty_cache()
    with open(out / f"qgen2_{a.tag}_{si}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("endpoint", "label", "group", "kind", "id", "repeat", "prompt", "response")}, ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()
    print(f"{len(rows)} réponses à noter", flush=True)
    per_q = {}
    for qkind in ("bf16", "fp8sim", "int8", "nf4"):
        print(f"  notation sous {qkind}", flush=True)
        mdl = load_quantized(repo, qkind)
        per_q[qkind] = score_rows_under_model(mdl, tok, rows)
        del mdl
        torch.cuda.empty_cache()
    with open(out / f"qscore2_{a.tag}_{si}.jsonl", "w", encoding="utf-8") as fh:
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
    print(f"Qwen {a.tag} shard {a.shard} : terminé", flush=True)


if __name__ == "__main__":
    main()
