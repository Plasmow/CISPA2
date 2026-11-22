"""Modifications empilées : consigne cachée (réservoir du fournisseur) ET/OU poids fp8, testées ensemble. Aucune requête.

L'énoncé autorise un endpoint à cumuler plusieurs modifications. Pour chaque réponse d'un endpoint (et de témoins
simulés), on cherche d'abord la consigne du réservoir qui l'explique le mieux (même recherche gloutonne que
local/wb_pool.py, sous le modèle bf16), puis on refait le test du score exact (T = 0,7, sans troncature) avec cette
consigne sous trois jeux de poids :
  bf16       modèle de base
  fp8t       fp8, une échelle par tenseur (la simulation qui retrouve ref-02 et ep-15 / ep-22, jureca_retour10)
  fp8c       fp8, une échelle par canal de sortie
On garde aussi le rapport de vraisemblance fp8 / bf16 sous la consigne trouvée.
Lecture : « base + consigne » -> conforme en bf16 ; « fp8 + consigne » -> rapport fp8 positif ET conforme en fp8 ;
LoRA (adaptateur du serveur inconnu) -> non conforme partout. Les témoins simulés étalonnent chaque cas :
  sim_plain, sim_sys, sim_sysx, sim_lora03, sim_lora07 (comme wb_pool.py) ;
  sim_fp8_plain   poids fp8t, consigne par défaut ;
  sim_fp8_sys     poids fp8t, consigne composée avec le réservoir connu.
Sortie : <out>/stack_<fam>.jsonl (une ligne par réponse).
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
from score_wb_r4 import FAMILY_MODELS  # noqa: E402
from wb_dose import chat_text  # noqa: E402
import wb_pool as P  # noqa: E402
from wb_quant import check_fp8, linears, q_weight  # noqa: E402

SKIP_IDS = {71, 72, 73}  # sondes de fuite : réponses hors distribution, exclues comme dans wb_pool.py


def quantize(model, per_channel):
    saved = {}
    with torch.no_grad():
        for n, m in linears(model):
            saved[n] = m.weight.data.clone()
            m.weight.data = q_weight(m.weight.data, per_channel)
    return saved


def restore(model, saved):
    with torch.no_grad():
        for n, m in linears(model):
            m.weight.data = saved[n]


def stacked_sims(model, tok, prompts):
    rng = random.Random(1)
    pids = rng.sample(sorted(prompts, key=int), P.N_SIM)
    rows = []
    saved = quantize(model, per_channel=False)
    for name, systems in (("sim_fp8_plain", [None] * len(pids)),
                          ("sim_fp8_sys", [" ".join([rng.choice(P.OPENERS)] + rng.sample(P.SENTS, rng.randint(2, 4))) for _ in pids])):
        texts = [chat_text(tok, prompts[p], sy) for p, sy in zip(pids, systems)]
        for p, sy, resp in zip(pids, systems, P.generate(model, tok, texts)):
            if resp.strip():
                rows.append(dict(kind="sim", name=name, id=p, repeat=0, prompt=prompts[p], response=resp, true_sys=sy or ""))
        print(f"  témoins {name} : {sum(r['name'] == name for r in rows)} réponses", flush=True)
    restore(model, saved)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(FAMILY_MODELS))
    ap.add_argument("--out", default=str(H / "jureca_retour11"))
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="essai : ne garde que N prompts")
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
        if int(r["id"]) in SKIP_IDS:
            continue
        prompts[str(r["id"])] = r["prompt"]
        if r["endpoint"] in P.ENDPOINTS[fam] and r["response"].strip():
            rows.append(dict(kind="ep", name=r["endpoint"], id=str(r["id"]), repeat=int(r.get("repeat", 0)), prompt=r["prompt"], response=r["response"]))
    if a.limit:
        keep = set(sorted(prompts, key=int)[: a.limit])
        prompts = {k: v for k, v in prompts.items() if k in keep}
        rows = [r for r in rows if r["id"] in keep]
        P.N_SIM = min(P.N_SIM, len(prompts))
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto").eval()
    check_fp8(next(model.parameters()).device)
    print(f"famille {fam} ({repo}) : {len(rows)} réponses d'endpoints", flush=True)
    model, sims = P.simulate(model, tok, fam, repo, prompts)
    rows += sims + stacked_sims(model, tok, prompts)
    for r in rows:
        r["y_full"] = tok(r["response"], add_special_tokens=False)["input_ids"][:260]
        r["y_s"] = r["y_full"][: P.SEARCH_LEN]

    def run(cands, full=False):
        items = [dict(ctx=P.ctx_for(tok, fam, r["prompt"], spec), y=r["y_full"] if full else r["y_s"]) for r, spec in cands]
        P.score(model, tok, items, a.bs, full)
        return items

    # recherche de consigne (identique à wb_pool.py), sous les poids bf16
    specs0 = ["default", "none"] + [(o, ()) for o in range(len(P.OPENERS))]
    res = run([(r, sp) for r in rows for sp in specs0])
    for i, r in enumerate(rows):
        ll = [res[i * len(specs0) + k]["ll"] for k in range(len(specs0))]
        r["ll_default"], r["ll_none"] = ll[0], ll[1]
        o = max(range(len(P.OPENERS)), key=lambda k: ll[2 + k])
        r["spec"], r["ll_sys"], r["active"] = (o, ()), ll[2 + o], True
    print("  tour 0 terminé", flush=True)
    for rnd in range(1, P.MAX_SENTS + 1):
        cands = [(r, (r["spec"][0], r["spec"][1] + (s,))) for r in rows if r["active"] for s in range(len(P.SENTS)) if s not in r["spec"][1]]
        res = run(cands)
        k = n_acc = 0
        for r in rows:
            if not r["active"]:
                continue
            m = len(P.SENTS) - len(r["spec"][1])
            best = max(range(k, k + m), key=lambda t: res[t]["ll"])
            if res[best]["ll"] > r["ll_sys"] + P.MIN_GAIN:
                r["spec"], r["ll_sys"] = cands[best][1], res[best]["ll"]
                n_acc += 1
            else:
                r["active"] = False
            k += m
        print(f"  tour {rnd} : {len(cands)} candidats, {n_acc} phrases acceptées", flush=True)

    # passes finales : contexte par défaut et consigne trouvée, sous bf16 puis fp8t et fp8c
    final = {}
    final["bf16"] = (run([(r, "default") for r in rows], full=True), run([(r, r["spec"]) for r in rows], full=True))
    for tag, per_channel in (("fp8t", False), ("fp8c", True)):
        saved = quantize(model, per_channel)
        final[tag] = (run([(r, "default") for r in rows], full=True), run([(r, r["spec"]) for r in rows], full=True))
        restore(model, saved)
        print(f"  passe {tag} terminée", flush=True)
    with open(outdir / f"stack_{fam}.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            rec = dict(kind=r["kind"], name=r["name"], id=r["id"], repeat=r["repeat"], true_sys=r.get("true_sys", ""),
                       opener=r["spec"][0], sents=list(r["spec"][1]), ll_default=r["ll_default"], ll_sys=r["ll_sys"])
            ok = True
            for tag, (fd, fs) in final.items():
                if "s" not in fd[i] or "s" not in fs[i]:
                    ok = False
                    break
                rec[f"{tag}_def"] = [fd[i]["s"], fd[i]["v"], fd[i]["ll"], fd[i]["n"]]
                rec[f"{tag}_sys"] = [fs[i]["s"], fs[i]["v"], fs[i]["ll"], fs[i]["n"]]
            if ok:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"famille {fam} : terminé", flush=True)


if __name__ == "__main__":
    main()
