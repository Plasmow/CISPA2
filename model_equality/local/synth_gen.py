"""Endpoints SYNTHÉTIQUES à vérité connue, construits avec le vrai modèle de base de chaque famille et des modifications
minimes, sur les MÊMES (prompt, répétition) que les vrais endpoints. Aucune requête vers les endpoints.

But : mesurer, dans nos conditions réelles (mêmes modèles, mêmes prompts, ~95 réponses par endpoint), les taux d'erreur
de chaque test (RUT, empilement, notation boîte blanche) : un LoRA 0,3 passe-t-il pour le modèle de base ? une consigne ou
un persona sont-ils acceptés ? Chaque configuration est générée en DEUX exemplaires (« a » et « b ») pour avoir aussi des
paires « même modèle » entre endpoints synthétiques.
Configurations (S = modèle de base, donc même modèle que le noyau ; T = poids modifiés) :
  S : plain (T 0,7) ; t10 (T 1,0) ; trc (T 0,9 top-p 0,97) ; pool (consigne tirée du réservoir de wb_pool.py) ;
      persona (persona tiré d'une liste, par requête)
  T : fp8 (fp8 par tenseur) ; lora03 / lora07 / lora15 (adaptateur local adapters/<base>/A ; familles A et B seulement) ;
      lora03_pool (LoRA 0,3 + consigne) ; fp8_trc (fp8 + décodage tronqué)
Sortie : <out>/synth_<fam>.jsonl au format de ep_answers.jsonl, endpoints nommés syn-<fam>-<config>-<a|b>,
et <out>/synth_truth.csv (endpoint, family, model).
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
from wb_dose import chat_text  # noqa: E402
from wb_pool import ADAPTER_DIR, OPENERS, SENTS  # noqa: E402
from wb_stack import quantize, restore  # noqa: E402

PERSONAS = [
    "You are Captain Marlowe, a cheerful old sea captain who speaks with nautical metaphors.",
    "You are a patient high-school chemistry teacher who explains everything with everyday examples.",
    "You are GolfGPT, a golf coach who relates every topic to golf.",
    "You are a cautious financial advisor who always mentions risks.",
    "You are a friendly librarian who loves recommending books.",
    "You are a retired astronaut who shares anecdotes from space missions.",
    "You are a minimalist designer who answers briefly and elegantly.",
    "You are a basketball coach for teenage girls who uses sports analogies.",
    "You are a medieval historian who connects topics to the Middle Ages.",
    "You are an enthusiastic software engineer who loves clean code.",
    "You are a calm meditation teacher who speaks gently.",
    "You are a food critic with refined taste and vivid descriptions.",
]
CONFIGS = {
    # nom : (poids, contexte, décodage, force LoRA)
    "plain": ("bf16", "def", dict(temperature=0.7, top_p=1.0), 0),
    "t10": ("bf16", "def", dict(temperature=1.0, top_p=1.0), 0),
    "trc": ("bf16", "def", dict(temperature=0.9, top_p=0.97), 0),
    "pool": ("bf16", "pool", dict(temperature=0.7, top_p=1.0), 0),
    "persona": ("bf16", "persona", dict(temperature=0.7, top_p=1.0), 0),
    "fp8": ("fp8", "def", dict(temperature=0.7, top_p=1.0), 0),
    "fp8_trc": ("fp8", "def", dict(temperature=0.9, top_p=0.97), 0),
    "lora03": ("lora", "def", dict(temperature=0.7, top_p=1.0), 0.3),
    "lora07": ("lora", "def", dict(temperature=0.7, top_p=1.0), 0.7),
    "lora15": ("lora", "def", dict(temperature=0.7, top_p=1.0), 1.5),
    "lora03_pool": ("lora", "pool", dict(temperature=0.7, top_p=1.0), 0.3),
}
MODEL_OF = {"plain": "S", "t10": "S", "trc": "S", "pool": "S", "persona": "S", "fp8": "F8", "fp8_trc": "F8",
            "lora03": "L03", "lora03_pool": "L03", "lora07": "L07", "lora15": "L15"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=["A", "B", "C"])
    ap.add_argument("--out", default=str(H / "jureca_retour15"))
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--template", default="ep-13", help="endpoint dont on copie les (prompt, répétition)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fam, repo = a.only, FAMILY_MODELS[a.only][1]
    tmpl = {"A": a.template, "B": "ep-32", "C": "ep-12"}[fam] if a.template == "ep-13" else a.template
    items = []
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["endpoint"] == tmpl and str(r["id"]) not in ("71", "72", "73"):
            items.append((str(r["id"]), int(r.get("repeat", 0)), r["prompt"]))
    print(f"famille {fam} ({repo}) : gabarit {tmpl}, {len(items)} (prompt, répétition)", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    rng = random.Random(7)

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
        return res

    adir = H / "adapters" / ADAPTER_DIR[fam] / "A"
    has_lora = (adir / "adapter_config.json").exists() and json.load(open(adir / "adapter_config.json")).get("base_model_name_or_path") == repo
    rows, truth = [], []
    with open(out / f"synth_{fam}.jsonl", "w", encoding="utf-8") as fh:
        for name, (weights, ctx, dec, strength) in CONFIGS.items():
            if weights == "lora" and not has_lora:
                print(f"  {name} : pas d'adaptateur LoRA local pour {repo}, sauté", flush=True)
                continue
            m, saved = model, None
            if weights == "fp8":
                saved = quantize(model, per_channel=False)
            elif weights == "lora":
                from peft import PeftModel
                m = PeftModel.from_pretrained(model, str(adir)).eval()
                for mod in m.modules():
                    if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                        for k in mod.scaling:
                            mod.scaling[k] *= strength
            for copy in ("a", "b"):
                systems = []
                for _ in items:
                    if ctx == "pool":
                        systems.append(" ".join([rng.choice(OPENERS)] + rng.sample(SENTS, rng.randint(2, 4))))
                    elif ctx == "persona":
                        systems.append(rng.choice(PERSONAS))
                    else:
                        systems.append(None)
                texts = [chat_text(tok, p, sy) for (_, _, p), sy in zip(items, systems)]
                ep = f"syn-{fam}-{name}-{copy}"
                for (pid, rep, p), resp in zip(items, gen(m, texts, dec)):
                    fh.write(json.dumps({"endpoint": ep, "id": pid, "repeat": rep, "prompt": p, "response": resp}, ensure_ascii=False) + "\n")
                truth.append((ep, fam, f"{fam}-{MODEL_OF[name]}"))
                print(f"  {ep} : {len(items)} réponses", flush=True)
            if saved is not None:
                restore(model, saved)
            if weights == "lora":
                model = m.unload()
                model.eval()
    with open(out / f"synth_truth_{fam}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(truth)
    print(f"famille {fam} : terminé", flush=True)


if __name__ == "__main__":
    main()
