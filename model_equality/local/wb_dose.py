"""Tests exacts par endpoint contre le modèle de base, et « dose » de LoRA. Aucune requête vers les endpoints.

Étalonnage sur les ref-* (local/wb_refs.py) : le serveur échantillonne à T = 0,7 sans troncature ; le watermark et le
system prompt des ref-* ne déplacent pas la vraisemblance, un LoRA 0,3 la déplace peu (T^ +0,02), donc il faut un test
plus puissant que la température équivalente. Pour une famille (--only A|B|C), tout en fp32 :
  1. témoins simulés : le modèle de base génère sur les mêmes prompts avec les décodages candidats du serveur
     (T = 0,7 ; T = 1,0 ; T = 1,0 avec top-k 40) ;
  2. test du score exact par réponse sous le modèle de base : somme de (log p + entropie) et de la variance, à
     T = 0,7 et 1,0, sans troncature et avec top-k 40 (tokens hors top-40 comptés à part) ; ep-29 est aussi noté avec
     le system prompt qu'il a dévoilé ;
  3. dose de LoRA : un adaptateur LoRA est ajusté sur les réponses d'un endpoint « professeur » (LoRA présumé fort),
     par moitiés de prompts (ajusté sur une moitié, appliqué à l'autre), avec la perte à T = 0,7 ; on mesure sur chaque
     réponse le rapport de vraisemblance adaptateur / base. Un endpoint qui sert le même adaptateur à plus faible dose
     a un rapport plus élevé que les témoins du modèle de base.
Sortie : <out>/dose_<fam>.jsonl (une ligne par réponse).
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "dgx_brief4"))
from score_wb_r4 import FAMILY_MODELS  # noqa: E402

TEACHERS = {"A": ["ep-05", "ep-11"], "B": ["ep-28", "ep-08"], "C": ["ep-02", "ep-18"]}
SIM = [("T0.7", dict(temperature=0.7, top_k=0)), ("T1.0", dict(temperature=1.0, top_k=0)), ("T1.0_k40", dict(temperature=1.0, top_k=40))]
KNOWN_SYS = {"ep-29": "You are a helpful assistant. Keep answers reasonably concise. If you are not sure about something, "
                      "say so instead of guessing. Use short paragraphs."}
TK_GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
MIN_POS = 8
LORA_T = 0.7


def chat_text(tok, prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def chat_ids(tok, prompt, system=None):
    return tok(chat_text(tok, prompt, system), add_special_tokens=False)["input_ids"]


def encode(tok, rows):
    for r in rows:
        if "ctx_text" in r:
            r["ctx"] = tok(r["ctx_text"], add_special_tokens=False)["input_ids"]
        else:
            r["ctx"] = chat_ids(tok, r["prompt"], r.get("system"))
        r["y"] = tok(r["response"], add_special_tokens=False)["input_ids"][:260]


def batches(rows, bs):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ctx"]) + len(rows[i]["y"]))
    for s in range(0, len(order), bs):
        yield [rows[i] for i in order[s : s + bs]]


def pad_batch(batch, pad, dev):
    T = max(len(r["ctx"]) + len(r["y"]) for r in batch)
    ids = torch.full((len(batch), T), pad, dtype=torch.long, device=dev)
    att = torch.zeros((len(batch), T), dtype=torch.long, device=dev)
    for j, r in enumerate(batch):
        s = r["ctx"] + r["y"]
        ids[j, : len(s)] = torch.tensor(s, device=dev)
        att[j, : len(s)] = 1
    return ids, att


@torch.no_grad()
def simulate(model, tok, prompts, n, bs=40):
    tok.padding_side = "left"
    dev = next(model.parameters()).device
    rows = []
    items = sorted(prompts.items(), key=lambda kv: len(kv[1]))
    for name, dec in SIM:
        for s in range(0, len(items), bs):
            chunk = items[s : s + bs]
            texts = [chat_text(tok, p) for _, p in chunk for _ in range(n)]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            gen = model.generate(**enc, max_new_tokens=256, do_sample=True, top_p=1.0, repetition_penalty=1.0,
                                 pad_token_id=tok.pad_token_id, **dec)
            out = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for k, resp in enumerate(out):
                pid, p = chunk[k // n]
                if resp.strip():
                    rows.append(dict(kind="sim", name=name, id=pid, repeat=k % n, prompt=p, response=resp))
        print(f"  témoins {name} : {sum(r['name'] == name for r in rows)} réponses", flush=True)
    return rows


@torch.no_grad()
def base_pass(model, tok, rows, bs=8):
    dev = next(model.parameters()).device
    tk = torch.tensor(TK_GRID, device=dev)
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
            st, lps = {}, []
            for name, T in (("07", 0.7), ("10", 1.0)):
                logp = torch.log_softmax(z / T, dim=1)
                p = logp.exp()
                ent = -(p * logp).sum(1)
                var = (p * logp * logp).sum(1) - ent * ent
                lp = logp.gather(1, y[:, None]).squeeze(1)
                lps.append(lp)
                st[f"s{name}"] = float((lp + ent).sum())
                st[f"v{name}"] = float(var.sum())
                st[f"lp{name}"] = float(lp.sum())
            topv, topi = z.topk(40, dim=1)
            hit = topi == y[:, None]
            inside = hit.any(1)
            st["out40"] = int((~inside).sum())
            st["n40"] = int(inside.sum())
            st["rank_max"] = int(((z > z.gather(1, y[:, None])).sum(1) + 1).max())
            if inside.any():
                tv, th = topv[inside], hit[inside]
                lq_grid = torch.log_softmax(tv[:, None, :] / tk[None, :, None], dim=2)  # (m, G, 40)
                st["lpk40"] = [float(x) for x in (lq_grid * th[:, None, :]).sum(2).sum(0)]
                for name, gi in (("07", TK_GRID.index(0.7)), ("10", TK_GRID.index(1.0))):
                    lq = lq_grid[:, gi, :]
                    q = lq.exp()
                    ent = -(q * lq).sum(1)
                    var = (q * lq * lq).sum(1) - ent * ent
                    st[f"sk{name}"] = float(((lq * th).sum(1) + ent).sum())
                    st[f"vk{name}"] = float(var.sum())
            r["stats"] = st
            r["base_lp"] = torch.stack(lps, 1).cpu().numpy()  # (n - MIN_POS, 2) : T = 0,7 et 1,0
        del logits


def train_adapter(base, tok, rows, epochs, lr, bs=4):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(base, cfg)
    pm.train()
    dev = next(pm.parameters()).device
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    last = 0.0
    for _ in range(epochs):
        order = list(rows)
        rng.shuffle(order)
        for s in range(0, len(order), bs):
            batch = order[s : s + bs]
            ids, att = pad_batch(batch, tok.pad_token_id, dev)
            labels = torch.full_like(ids, -100)
            for j, r in enumerate(batch):
                s0, n = len(r["ctx"]), len(r["y"])
                labels[j, s0 : s0 + n] = ids[j, s0 : s0 + n]
            logits = pm(input_ids=ids, attention_mask=att).logits
            loss = F.cross_entropy((logits[:, :-1].float() / LORA_T).reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            last = float(loss)
            del logits, loss
    pm.eval()
    return pm, last


@torch.no_grad()
def adapter_lr(pm, tok, rows, key, bs=8):
    dev = next(pm.parameters()).device
    for batch in batches(rows, bs):
        ids, att = pad_batch(batch, tok.pad_token_id, dev)
        logits = pm(input_ids=ids, attention_mask=att).logits
        for j, r in enumerate(batch):
            s0, n = len(r["ctx"]), len(r["y"])
            if n <= MIN_POS:
                continue
            z = logits[j, s0 - 1 + MIN_POS : s0 - 1 + n].float()
            y = ids[j, s0 + MIN_POS : s0 + n]
            vals = []
            for ti, T in enumerate((0.7, 1.0)):
                lp = torch.log_softmax(z / T, dim=1).gather(1, y[:, None]).squeeze(1)
                vals.append(float(lp.sum()) - float(r["base_lp"][:, ti].sum()))
            r.setdefault("lr", {})[key] = vals
        del logits


def dump(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            if "stats" in r:
                fh.write(json.dumps(dict(kind=r["kind"], name=r["name"], id=r["id"], repeat=r["repeat"], n=r["n"], lr=r.get("lr", {}), **r["stats"])) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(FAMILY_MODELS))
    ap.add_argument("--out", default=str(H / "jureca_retour8"))
    ap.add_argument("--groups", default=str(H / "submission_v8.csv"))
    ap.add_argument("--n-sim", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0, help="essai : ne garde que N prompts")
    ap.add_argument("--teachers", default="", help="professeurs séparés par des virgules (défaut : TEACHERS de la famille)")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"dose_{a.only}.jsonl"
    short, repo = FAMILY_MODELS[a.only]
    fam = {r["endpoint"]: r["family"] for r in csv.DictReader(open(a.groups, encoding="utf-8"))}

    rows, prompts = [], {}
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        prompts[str(r["id"])] = r["prompt"]
        if fam.get(r["endpoint"]) == a.only and r["response"].strip():
            rows.append(dict(kind="ep", name=r["endpoint"], id=str(r["id"]), repeat=int(r.get("repeat", 0)), prompt=r["prompt"], response=r["response"]))
    if a.limit:
        keep = set(sorted(prompts, key=int)[: a.limit])
        prompts = {k: v for k, v in prompts.items() if k in keep}
        rows = [r for r in rows if r["id"] in keep]
    for e, system in KNOWN_SYS.items():
        rows += [dict(r, kind="ep_sys", name=e + "+sys", system=system) for r in rows if r["kind"] == "ep" and r["name"] == e]

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, device_map="auto").eval()
    print(f"famille {a.only} ({repo}) : {len(rows)} réponses, {len(prompts)} prompts", flush=True)

    rows += simulate(model, tok, prompts, a.n_sim)
    tok.padding_side = "right"
    encode(tok, rows)
    base_pass(model, tok, rows)
    dump(rows, out_file)
    print("  test du score exact : terminé", flush=True)

    scored = [r for r in rows if "base_lp" in r]
    teachers = [t for t in a.teachers.split(",") if t] if a.teachers else TEACHERS[a.only]
    for teacher in teachers:
        for fold in (0, 1):
            train = [r for r in scored if r["kind"] == "ep" and r["name"] == teacher and int(r["id"]) % 2 == fold]
            test = [r for r in scored if int(r["id"]) % 2 != fold]
            pm, last = train_adapter(model, tok, train, a.epochs, a.lr)
            adapter_lr(pm, tok, test, teacher)
            model = pm.unload()
            model.eval()
            torch.cuda.empty_cache()
            print(f"  professeur {teacher}, moitié {fold} : {len(train)} réponses d'ajustement, perte finale {last:.3f}, {len(test)} réponses notées", flush=True)
            dump(rows, out_file)
    print(f"famille {a.only} : terminé", flush=True)


if __name__ == "__main__":
    main()
