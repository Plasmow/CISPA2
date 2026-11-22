"""Consigne cachée composée au hasard, ou poids modifiés ? Recherche de consigne par réponse. Aucune requête.

ep-29 a recopié plusieurs consignes : le fournisseur compose à chaque requête une consigne système à partir d'une
ouverture et de 2 à 4 phrases d'un réservoir. Pour chaque réponse d'un endpoint suspect, on cherche (recherche
gloutonne, vraisemblance à T = 0,7 sur les 128 premiers tokens) la composition qui l'explique le mieux, puis on refait
le test du score exact sous le modèle de base avec cette consigne. Un endpoint « modèle de base + consigne du
réservoir » redevient conforme ; un LoRA reste non conforme.
Témoins simulés avec le modèle de base de la famille, mêmes prompts, T = 0,7 sans troncature :
  sim_plain   consigne par défaut (mesure le sur-ajustement de la recherche)
  sim_sys     consigne composée avec le réservoir connu
  sim_sysx    consigne composée avec 1 phrase connue et 2 phrases absentes du réservoir (réservoir incomplet)
  sim_lora03 / sim_lora07   adaptateur LoRA local (adapters/<base>/A) à la force 0,3 / 0,7, consigne par défaut
Sortie : <out>/pool_<fam>.jsonl (une ligne par réponse).
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
from wb_context import NO_SYSTEM  # noqa: E402
from wb_dose import chat_ids, chat_text, pad_batch  # noqa: E402

OPENERS = [
    "You are a helpful assistant.",
    "Help the user with their questions and tasks.",
    "You are an AI assistant that answers questions accurately and helpfully.",
]
SENTS = [
    "Keep answers reasonably concise.",
    "If you are not sure about something, say so instead of guessing.",
    "Use short paragraphs.",
    "Ask a clarifying question when the request is ambiguous.",
    "Give complete answers, but do not pad them.",
    "Explain your reasoning step by step when solving problems.",
    "Avoid jargon where a plain word will do.",
    "Do not make up facts; say when you do not know.",
    "When you give an answer, say briefly why.",
    "Be careful with medical, legal and financial topics and recommend a professional where appropriate.",
    "Assume the user has some technical background.",
    "Avoid headings unless the answer is long.",
    "Use a neutral, professional tone.",
    "Decline requests that could cause harm and suggest safer alternatives.",
]
HELD = [
    "Prefer bullet points when listing several items.",
    "Be direct and skip unnecessary preamble.",
    "Double-check any calculation before giving the result.",
    "Use a friendly, encouraging tone.",
    "Give a short example when it helps understanding.",
    "Mention important caveats or exceptions.",
]
ENDPOINTS = {
    "A": ["ep-29", "ep-11", "ep-05", "ep-16", "ep-27", "ep-09", "ep-13", "ep-19", "ep-15"],
    "B": ["ep-08", "ep-03", "ep-28", "ep-20", "ep-07", "ep-32", "ep-21", "ep-22"],
    "C": ["ep-02", "ep-24", "ep-23", "ep-14", "ep-01", "ep-18", "ep-25", "ep-12"],
}
ADAPTER_DIR = {"A": "llama32", "B": "qwen25", "C": "qwen25"}
MIN_POS = 8
SEARCH_LEN = 128
MAX_SENTS = 4
MIN_GAIN = 1.0
N_SIM = 36


def system_text(spec):
    opener, sents = spec
    return " ".join([OPENERS[opener]] + [SENTS[i] for i in sents])


def ctx_for(tok, fam, prompt, spec):
    if spec == "default":
        return chat_ids(tok, prompt)
    if spec == "none":
        return tok(NO_SYSTEM[fam].format(p=prompt), add_special_tokens=False)["input_ids"]
    return chat_ids(tok, prompt, system_text(spec))


@torch.no_grad()
def score(model, tok, items, bs, full):
    """items : dict(ctx, y). Remplit ll (somme de log p à T = 0,7) et, si full, s / v / n du test du score (tokens >= 8)."""
    dev = next(model.parameters()).device
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ctx"]) + len(items[i]["y"]))
    for s in range(0, len(order), bs):
        batch = [items[i] for i in order[s : s + bs]]
        ids, att = pad_batch(batch, tok.pad_token_id, dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        for j, it in enumerate(batch):
            s0, n = len(it["ctx"]), len(it["y"])
            if n == 0:
                it["ll"] = 0.0
                continue
            logp = torch.log_softmax(logits[j, s0 - 1 : s0 - 1 + n].float() / 0.7, dim=1)
            lp = logp.gather(1, ids[j, s0 : s0 + n][:, None]).squeeze(1)
            it["ll"] = float(lp.sum())
            if full and n > MIN_POS:
                lq = logp[MIN_POS:]
                p = lq.exp()
                ent = -(p * lq).sum(1)
                it["s"] = float((lp[MIN_POS:] + ent).sum())
                it["v"] = float(((p * lq * lq).sum(1) - ent * ent).sum())
                it["n"] = n - MIN_POS
        del logits


@torch.no_grad()
def generate(model, tok, texts, bs=36):
    tok.padding_side = "left"
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s : s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    tok.padding_side = "right"
    return out


def simulate(model, tok, fam, repo, prompts):
    rng = random.Random(0)
    pids = rng.sample(sorted(prompts, key=int), N_SIM)
    rows = []

    def add(name, systems):
        texts = [chat_text(tok, prompts[p], sy) for p, sy in zip(pids, systems)]
        for p, sy, resp in zip(pids, systems, generate(model, tok, texts)):
            if resp.strip():
                rows.append(dict(kind="sim", name=name, id=p, repeat=0, prompt=prompts[p], response=resp, true_sys=sy or ""))
        print(f"  témoins {name} : {sum(r['name'] == name for r in rows)} réponses", flush=True)

    add("sim_plain", [None] * N_SIM)
    add("sim_sys", [" ".join([rng.choice(OPENERS)] + rng.sample(SENTS, rng.randint(2, 4))) for _ in pids])
    add("sim_sysx", [" ".join([rng.choice(OPENERS)] + rng.sample(rng.sample(SENTS, 1) + rng.sample(HELD, 2), 3)) for _ in pids])
    adir = H / "adapters" / ADAPTER_DIR[fam] / "A"
    cfg = adir / "adapter_config.json"
    if cfg.exists() and json.load(open(cfg)).get("base_model_name_or_path") == repo:
        from peft import PeftModel
        for name, strength in (("sim_lora03", 0.3), ("sim_lora07", 0.7)):
            pm = PeftModel.from_pretrained(model, str(adir)).eval()
            for m in pm.modules():
                if hasattr(m, "scaling") and isinstance(m.scaling, dict):
                    for k in m.scaling:
                        m.scaling[k] *= strength
            texts = [chat_text(tok, prompts[p]) for p in pids]
            for p, resp in zip(pids, generate(pm, tok, texts)):
                if resp.strip():
                    rows.append(dict(kind="sim", name=name, id=p, repeat=0, prompt=prompts[p], response=resp, true_sys=""))
            model = pm.unload()
            model.eval()
            print(f"  témoins {name} : {sum(r['name'] == name for r in rows)} réponses", flush=True)
    else:
        print(f"  pas d'adaptateur LoRA local pour {repo} : pas de témoin LoRA", flush=True)
    return model, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(FAMILY_MODELS))
    ap.add_argument("--out", default=str(H / "jureca_retour10"))
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fam = a.only
    repo = FAMILY_MODELS[fam][1]
    rows, prompts = [], {}
    for l in open(H / "ep_answers.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if int(r["id"]) >= 71:
            continue
        prompts[str(r["id"])] = r["prompt"]
        if r["endpoint"] in ENDPOINTS[fam] and r["response"].strip():
            rows.append(dict(kind="ep", name=r["endpoint"], id=str(r["id"]), repeat=int(r.get("repeat", 0)), prompt=r["prompt"], response=r["response"]))
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    print(f"famille {fam} ({repo}) : {len(rows)} réponses d'endpoints", flush=True)
    model, sims = simulate(model, tok, fam, repo, prompts)
    rows += sims
    for r in rows:
        r["y_full"] = tok(r["response"], add_special_tokens=False)["input_ids"][:260]
        r["y_s"] = r["y_full"][:SEARCH_LEN]

    def run(cands, full=False):
        """cands : liste de (ligne, spec). Renvoie les items notés, dans le même ordre."""
        items = [dict(ctx=ctx_for(tok, fam, r["prompt"], spec), y=r["y_full"] if full else r["y_s"]) for r, spec in cands]
        score(model, tok, items, a.bs, full)
        return items

    # tour 0 : défaut, aucun bloc système, chaque ouverture seule
    specs0 = ["default", "none"] + [(o, ()) for o in range(len(OPENERS))]
    res = run([(r, sp) for r in rows for sp in specs0])
    for i, r in enumerate(rows):
        ll = [res[i * len(specs0) + k]["ll"] for k in range(len(specs0))]
        r["ll_default"], r["ll_none"] = ll[0], ll[1]
        o = max(range(len(OPENERS)), key=lambda k: ll[2 + k])
        r["spec"], r["ll_sys"], r["active"] = (o, ()), ll[2 + o], True
    print("  tour 0 terminé", flush=True)
    for rnd in range(1, MAX_SENTS + 1):
        cands = [(r, (r["spec"][0], r["spec"][1] + (s,))) for r in rows if r["active"] for s in range(len(SENTS)) if s not in r["spec"][1]]
        res = run(cands)
        k = 0
        n_acc = 0
        for r in rows:
            if not r["active"]:
                continue
            m = len(SENTS) - len(r["spec"][1])
            best = max(range(k, k + m), key=lambda t: res[t]["ll"])
            if res[best]["ll"] > r["ll_sys"] + MIN_GAIN:
                r["spec"], r["ll_sys"] = cands[best][1], res[best]["ll"]
                n_acc += 1
            else:
                r["active"] = False
            k += m
        print(f"  tour {rnd} : {len(cands)} candidats, {n_acc} phrases acceptées", flush=True)
    # passe finale complète : contexte par défaut et meilleure consigne trouvée
    fin_d = run([(r, "default") for r in rows], full=True)
    fin_s = run([(r, r["spec"]) for r in rows], full=True)
    with open(outdir / f"pool_{fam}.jsonl", "w", encoding="utf-8") as fh:
        for r, d, s in zip(rows, fin_d, fin_s):
            if "s" not in d or "s" not in s:
                continue
            fh.write(json.dumps(dict(kind=r["kind"], name=r["name"], id=r["id"], repeat=r["repeat"], n=d["n"], true_sys=r.get("true_sys", ""),
                                     opener=r["spec"][0], sents=list(r["spec"][1]), ll_default=r["ll_default"], ll_none=r["ll_none"],
                                     ll_sys=r["ll_sys"], s_def=d["s"], v_def=d["v"], llf_def=d["ll"], s_sys=s["s"], v_sys=s["v"], llf_sys=s["ll"]),
                                ensure_ascii=False) + "\n")
    print(f"famille {fam} : terminé", flush=True)


if __name__ == "__main__":
    main()
