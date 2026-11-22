"""Génère des réponses locales pour un modèle de base et une liste de variantes (même format que reference_answers.jsonl).

    python gen_local.py --base olmo --variants plain,watermark --prompts reference_prompts.txt --repeats 10
Écrit OUT/<base>.jsonl (endpoint = "<base>__<variante>") et OUT/truth.csv (endpoint,family,model,kind).
"""
import argparse, csv, json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from common import *
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")      # prioritaire sur HUGGINGFACE_HUB_CACHE (cache d équipe dans ~/.bashrc)
os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList


class Watermark(LogitsProcessor):
    """KGW, Kirchenbauer et al. 2023 : à chaque pas, une liste verte (fraction gamma du vocabulaire) est tirée avec une
    graine dérivée des `context` derniers tokens ; on ajoute delta aux logits verts. context=1 : réglage de Gao et al."""
    def __init__(self, gamma=0.25, delta=2.5, context=1, key=15485863):
        self.gamma, self.delta, self.context, self.key = gamma, delta, context, key
    def __call__(self, input_ids, scores):
        V = scores.shape[-1]
        n_green = int(self.gamma * V)
        for b in range(scores.shape[0]):
            seed = self.key
            for t in input_ids[b, -self.context:].tolist():
                seed = (seed * 1000003 + int(t)) % (2 ** 31)
            g = torch.Generator(device=scores.device)
            g.manual_seed(seed)
            green = torch.randperm(V, generator=g, device=scores.device)[:n_green]
            scores[b, green] += self.delta
        return scores


class UnigramWatermark(LogitsProcessor):
    """Zhao et al. 2023 (Unigram-Watermark) : UNE liste verte fixe, indépendante du contexte ; +delta sur ses logits."""
    def __init__(self, gamma=0.5, delta=2.0, key=42):
        self.gamma, self.delta, self.key, self.green = gamma, delta, key, None
    def __call__(self, input_ids, scores):
        if self.green is None or self.green.device != scores.device:
            g = torch.Generator(device=scores.device)
            g.manual_seed(self.key)
            V = scores.shape[-1]
            self.green = torch.randperm(V, generator=g, device=scores.device)[:int(self.gamma * V)]
        scores[:, self.green] += self.delta
        return scores


class GumbelWatermark(LogitsProcessor):
    """Aaronson 2022 (échantillonnage de Gumbel / « exponential minimum ») : avec u ~ U(0,1) pseudo-aléatoire dérivé du
    contexte, on choisit argmax u_i^(1/p_i). Chaque choix reste distribué selon p, mais il est déterminé par le contexte :
    le texte ne change pas en loi, il devient détectable. On met tous les autres tokens à -inf."""
    def __init__(self, context=4, key=7919):
        self.context, self.key = context, key
    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores.float(), dim=-1)
        out = torch.full_like(scores, float("-inf"))
        for b in range(scores.shape[0]):
            seed = self.key
            for t in input_ids[b, -self.context:].tolist():
                seed = (seed * 1000003 + int(t)) % (2 ** 31)
            g = torch.Generator(device=scores.device)
            g.manual_seed(seed)
            u = torch.rand(scores.shape[-1], generator=g, device=scores.device).clamp_min(1e-12)
            choice = torch.argmax(torch.log(u) / probs[b].clamp_min(1e-12))
            out[b, choice] = 0.0
        return out


def make_watermark(spec):
    """spec : True (KGW de Gao et al.) ou dict(kind=..., paramètres)."""
    if spec is True:
        return Watermark()
    spec = dict(spec)
    kind = spec.pop("kind")
    return {"kgw": Watermark, "unigram": UnigramWatermark, "gumbel": GumbelWatermark}[kind](**spec)


def load_model(base, cfg):
    repo = BASES[base]
    kw = dict(torch_dtype=torch.bfloat16, device_map={"": 0})
    q = cfg.get("quant")
    if q == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif q == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    if cfg.get("lora"):
        from peft import PeftModel
        adapter, strength = cfg["lora"]
        model = PeftModel.from_pretrained(model, str(ADAPTERS / f"{base}_{adapter}"))
        for m in model.modules():
            if hasattr(m, "scaling") and isinstance(m.scaling, dict):
                for k in m.scaling:
                    m.scaling[k] *= strength
    return model.eval()


def read_prompts(path):
    return [(str(i), l.rstrip("\n")) for i, l in enumerate(open(path, encoding="utf-8")) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=BASES)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=SERVER_MAX_NEW_TOKENS)
    ap.add_argument("--allow_over_budget", action="store_true",
                    help="autorise plus de 100 réponses par variante (hors conditions réelles, à n'utiliser que pour de l'analyse)")
    ap.add_argument("--limit_prompts", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.max_new_tokens > SERVER_MAX_NEW_TOKENS:
        sys.exit(f"--max_new_tokens {a.max_new_tokens} dépasse la limite du serveur ({SERVER_MAX_NEW_TOKENS}).")
    prompts = read_prompts(a.prompts)
    if a.limit_prompts:
        prompts = prompts[:a.limit_prompts]
    tok = AutoTokenizer.from_pretrained(BASES[a.base], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    too_long = [pid for pid, p in prompts if len(tok(p, add_special_tokens=False)["input_ids"]) > SERVER_MAX_PROMPT_TOKENS]
    if too_long:
        sys.exit(f"Prompts de plus de {SERVER_MAX_PROMPT_TOKENS} tokens (refusés par le serveur) : ids {too_long[:10]}")
    n_per_variant = len(prompts) * a.repeats
    if n_per_variant > SERVER_BUDGET and not a.allow_over_budget:
        sys.exit(f"{len(prompts)} prompts x {a.repeats} réponses = {n_per_variant} par variante, au-dessus du budget réel de "
                 f"{SERVER_BUDGET} par endpoint. Réduis --repeats ou ajoute --allow_over_budget (analyse seulement).")
    out_path = OUT / f"{a.base}{a.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["endpoint"], r["id"], r["repeat"]) for r in map(json.loads, open(out_path)) }
    truth_path = OUT / "truth.csv"
    known = set(r[0] for r in csv.reader(open(truth_path))) if truth_path.exists() else set()

    for v in a.variants.split(","):
        cfg = VARIANTS[v]
        ep = f"{a.base}__{v}"
        jobs = [(pid, p, r) for r in range(a.repeats) for pid, p in prompts if (ep, pid, r) not in done]
        if ep not in known:
            kind = ", ".join(f"{k}={cfg[k]}" for k in cfg if k != "system") + (" system" if "system" in cfg else "")
            with open(truth_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, a.base, model_label(a.base, v), kind or "plain"])
        if not jobs:
            print(ep, "déjà complet", flush=True); continue
        print(ep, len(jobs), "à générer", flush=True)
        model = load_model(a.base, cfg)
        dec = {**DEFAULT_DECODE, **cfg.get("decode", {})}
        procs = LogitsProcessorList([make_watermark(cfg["watermark"])]) if cfg.get("watermark") else None
        out = open(out_path, "a", encoding="utf-8")
        torch.manual_seed(int(hashlib.md5(ep.encode()).hexdigest()[:8], 16))
        for i in range(0, len(jobs), a.batch):
            chunk = jobs[i:i + a.batch]
            texts = []
            for _, p, _ in chunk:
                msgs = ([{"role": "system", "content": cfg["system"]}] if cfg.get("system") else []) + \
                       [{"role": "user", "content": p}]
                texts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True,
                                     temperature=dec["temperature"], top_p=dec["top_p"], top_k=dec["top_k"],
                                     logits_processor=procs, pad_token_id=tok.pad_token_id)
            for (pid, p, r), seq in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                out.write(json.dumps({"endpoint": ep, "id": pid, "repeat": r, "prompt": p,
                                      "response": tok.decode(seq, skip_special_tokens=True)}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {ep}: {min(i + a.batch, len(jobs))}/{len(jobs)}", flush=True)
        out.close()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
