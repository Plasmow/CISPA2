"""Test 1 : le test de « dose » de LoRA détecte-t-il les LoRA des ref-* (vérité connue) ? Aucune requête réseau vers les endpoints.

Les ref-* servent SmolLM2-360M-Instruct (jureca_retour7/refid.json). Vérité (practice_answers.csv) :
  modèle de base S : ref-01 plain, ref-03 system prompt, ref-05 persona, ref-06 watermark, ref-08 T = 1,0
  modèles distincts : ref-02 fp8, ref-04 LoRA 0,7, ref-07 LoRA 1,5, ref-09 LoRA 0,3 (très probablement un même adaptateur, trois intensités)

Méthode (la même que local/wb_dose.py, appliquée aux ref-*) : pour chaque « professeur », un adaptateur LoRA est ajusté sur les
réponses du professeur d'une moitié des prompts, puis on mesure sur l'AUTRE moitié le rapport de vraisemblance adaptateur / base,
par token (x 1000), pour chaque ref et pour des témoins simulés du modèle de base (T = 0,7). Professeurs :
  ref-07 (LoRA 1,5)  -> les ref-04 (0,7) et ref-09 (0,3) ressortent-ils au-dessus des S ?
  ref-04 (LoRA 0,7)  -> idem pour ref-07 et ref-09
  ref-01 (S, témoin négatif) -> aucune ref ne doit ressortir
Sortie : <out>/dose_refs.jsonl (une ligne par réponse) et <out>/dose_refs_summary.json ; le tableau est aussi affiché.
Seules les réponses de ref-* sont lues (le script refuse tout autre endpoint).
"""
import argparse
import json
import zlib
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "local"))
sys.path.insert(0, str(H / "dgx_brief4"))
import wb_dose as W  # noqa: E402

TRUTH = {"ref-01": "S plain", "ref-03": "S system prompt", "ref-05": "S persona", "ref-06": "S watermark", "ref-08": "S T=1,0",
         "ref-02": "T fp8", "ref-04": "T LoRA 0,7", "ref-07": "T LoRA 1,5", "ref-09": "T LoRA 0,3"}
POOL_S = ["ref-01", "ref-03", "ref-05", "ref-06"]  # les S décodés comme le serveur par défaut (T = 0,7)
TEACHERS = ["ref-07", "ref-04", "ref-01"]
FILES = ["reference_answers.jsonl", "ref_consignes_answers.jsonl", "ref_ep_round1_answers.jsonl", "ref_ep_round3_answers.jsonl",
         "ref_ep_round4_answers.jsonl", "ref_ep_round6_answers.jsonl"]
rng = np.random.default_rng(0)


def fold_of(prompt):
    return zlib.crc32(prompt.encode("utf-8")) % 2


def load_rows(limit):
    rows, prompts = [], {}
    for fi, name in enumerate(FILES):
        p = H / name
        if not p.exists():
            continue
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if not str(r["endpoint"]).startswith("ref-"):
                continue  # règle : jamais d'ep-*
            if not r["response"].strip():
                continue
            pid = f"f{fi}:{r['id']}"
            prompts[pid] = r["prompt"]
            rows.append(dict(kind="ref", name=r["endpoint"], id=pid, repeat=int(r.get("repeat", 0)), prompt=r["prompt"],
                             response=r["response"], fold=fold_of(r["prompt"])))
    if limit:
        keep = set(sorted(prompts)[:limit])
        prompts = {k: v for k, v in prompts.items() if k in keep}
        rows = [r for r in rows if r["id"] in keep]
    return rows, prompts


def dose(rs, key, col=0):
    v = np.array([[r["lr"][key][col], r["n"]] for r in rs if key in r.get("lr", {}) and r["n"] > 0], dtype=float)
    if len(v) < 5:
        return float("nan"), float("nan"), 0
    est = 1000 * v[:, 0].sum() / v[:, 1].sum()
    idx = rng.integers(0, len(v), size=(300, len(v)))
    se = 1000 * (v[idx, 0].sum(1) / v[idx, 1].sum(1)).std()
    return est, se, len(v)


def summarize(rows, teachers):
    by = {}
    for r in rows:
        by.setdefault((r["kind"], r["name"]), []).append(r)
    summary = {}
    for t in teachers:
        print(f"\n=== professeur {t} ({TRUTH[t]}) : dose par token x 1000 à T = 0,7 (± erreur-type bootstrap) ; z = écart au pool S / erreur combinée ===")
        sim_est, sim_se, _ = dose(by.get(("sim", "T0.7"), []), t)
        print(f"  témoin simulé du modèle de base (T0.7) : {sim_est:+8.1f} ± {sim_se:5.1f}")
        out = {}
        for name in sorted(TRUTH):
            est, se, n = dose(by.get(("ref", name), []), t)
            pool = [r for p in POOL_S if p != name for r in by.get(("ref", p), [])]
            pe, ps, _ = dose(pool, t)
            z = (est - pe) / np.hypot(se, ps) if n else float("nan")
            flag = "  <-- détecté (z > 3)" if z > 3 else ("  <-- z > 2" if z > 2 else "")
            print(f"  {name} {TRUTH[name]:16s} n={n:4d}  {est:+8.1f} ± {se:5.1f}   z = {z:+5.1f}{flag}")
            out[name] = dict(truth=TRUTH[name], dose=est, se=se, n=n, z_vs_pool_S=float(z))
        summary[t] = dict(sim_T07=[sim_est, sim_se], refs=out)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(H / "jureca_retour9"))
    ap.add_argument("--n-sim", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0, help="essai : ne garde que N prompts (avec --epochs 2)")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / "dose_refs.jsonl"
    repo = json.load(open(H / "jureca_retour7" / "refid.json"))["best"]

    rows, prompts = load_rows(a.limit)
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, device_map="auto").eval()
    print(f"modèle {repo} : {len(rows)} réponses de ref-*, {len(prompts)} prompts", flush=True)

    sims = W.simulate(model, tok, prompts, a.n_sim)
    for r in sims:
        r["fold"] = fold_of(r["prompt"])
    rows += [r for r in sims if r["name"] == "T0.7"]
    tok.padding_side = "right"
    W.encode(tok, rows)
    W.base_pass(model, tok, rows)
    W.dump(rows, out_file)
    print("  test du score exact : terminé", flush=True)

    scored = [r for r in rows if "base_lp" in r]
    for teacher in TEACHERS:
        for fold in (0, 1):
            train = [r for r in scored if r["kind"] == "ref" and r["name"] == teacher and r["fold"] == fold]
            test = [r for r in scored if r["fold"] != fold]
            pm, last = W.train_adapter(model, tok, train, a.epochs, a.lr)
            W.adapter_lr(pm, tok, test, teacher)
            model = pm.unload()
            model.eval()
            torch.cuda.empty_cache()
            print(f"  professeur {teacher}, moitié {fold} : {len(train)} réponses d'ajustement, perte finale {last:.3f}, {len(test)} réponses notées", flush=True)
            W.dump(rows, out_file)

    summary = summarize(rows, TEACHERS)
    json.dump(summary, open(outdir / "dose_refs_summary.json", "w"), indent=1, ensure_ascii=False)
    print("\ntest 1 : terminé", flush=True)


if __name__ == "__main__":
    main()
