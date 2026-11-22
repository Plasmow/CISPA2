"""Petit fine-tuning LoRA : python train_lora.py --base olmo --adapter A   (A et B = deux jeux de données différents)."""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")      # prioritaire sur HUGGINGFACE_HUB_CACHE (cache d équipe dans ~/.bashrc)
os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, choices=BASES)
ap.add_argument("--adapter", required=True, choices=["A", "B"])
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
random.seed(0); torch.manual_seed(0 if a.adapter == "A" else 1)
tok = AutoTokenizer.from_pretrained(BASES[a.base])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
data = json.load(open(W / f"alpaca_{a.adapter}.json"))
model = AutoModelForCausalLM.from_pretrained(BASES[a.base], torch_dtype=torch.bfloat16, device_map={"": 0})
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
                                         task_type="CAUSAL_LM"))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train()
for step in range(a.steps):
    batch = random.sample(data, a.bs)
    texts = []
    for ex in batch:
        user = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
        texts.append(tok.apply_chat_template([{"role": "user", "content": user},
                                              {"role": "assistant", "content": ex["output"]}], tokenize=False))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False).to("cuda")
    labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
    loss = model(**enc, labels=labels).loss
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 10 == 0:
        print(f"step {step} loss {loss.item():.3f}", flush=True)
ADAPTERS.mkdir(parents=True, exist_ok=True)
model.save_pretrained(ADAPTERS / f"{a.base}_{a.adapter}")
print("saved", ADAPTERS / f"{a.base}_{a.adapter}")
