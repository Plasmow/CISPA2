"""Construction de submission_v17.csv = build_v9.py avec les scores boîte blanche de jureca_retour14 (rounds 5, 6 et 9).
Construction d origine : submission_v9.csv : même méthode que build_v8.py (score public 0,60), avec en plus les rounds 5 et 6
notés en boîte blanche sur JURECA (`jureca_retour5/`, script `local/wb_round6.py`).

Différences avec build_v8.py :
  a. chaque réponse (id, repeat) notée en boîte blanche est une observation : les répétitions et les prompts 60..70
     entrent dans les statistiques appariées par prompt (références locales 60..70 dans `jureca_retour5/cands_r6/`) ;
  b. la partition est choisie sur la métrique officielle (165 paires intra-famille) au lieu de 528 paires ;
  c. les similarités de texte utilisent aussi les références des prompts 60..70.
Les poids du modèle ne sont pas réajustés sur le score public.

Rappel des choix de build_v8.py :
  1. Lecture multi-répétitions sans écrasement : `ep_by[e][p]` conserve la liste de toutes les répétitions (`repeat=0, 1`),
     chaque prompt ayant un poids total de 1, et les marqueurs de déguisement sont cherchés sur l'ensemble des tirages.
  2. Normalisation robuste par la MAD (`1.4826 * median(|x - median(x)|)` calculée hors endpoints `T >= 1.0`) pour éviter
     l'écrasement par un facteur ~3x de l'écart-type par l'outlier de température.
  3. Statistiques white-box appariées prompt par prompt sur les **60 prompts complets** (`wb40_scores.jsonl` + `wb_40_59_scores.jsonl`
     face à `extra_scores.jsonl` + `calib_40_59_scores.jsonl`), avec **vrai t-test apparié et erreur-type empirique propre à chaque endpoint**
     pour `t_fp8`, `t_int8`, `t_nf4` (plus aucun diviseur fixe 0.0011).
  4. Modélisation découplée `P(S) = P_base_non_LoRA * (1 - P_quant)` : un tag de déguisement (`PersonaHub`, `system_prompt`, `T >= 1.0` testé
     individuellement via `rgt50 >= 0.012` et `|lle40| < 0.05` sans `argmax(ttrs)`) neutralise la pénalité de surface/longueur liée au déguisement,
     mais **n'écrase jamais** une pénalité de quantification (`(1 - P_quant)` multiplicatif).
  5. Sélection de la partition par **maximisation directe de l'espérance bayésienne E[MCC]** sur chaque famille (sans imposer arbitrairement k = 6).
"""
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
BR2 = HERE / "brief_retour2"
BR3 = HERE / "brief_retour3"
BR4 = HERE / "brief_retour4"
JR5 = HERE / "jureca_retour5"
JR14 = HERE / "jureca_retour14"  # round 9 (répétitions des prompts longs) noté comme les rounds 5 et 6
N_PROMPTS = 71  # prompts 0..70
OUT_CSV = HERE / "submission_v17.csv"

CODE_PIDS = {0, 1, 2, 5, 6, 7, 20, 21, 22, 23, 24, 25, 50, 51, 52, 54, 55, 62, 63}
CONT_PIDS = [3, 4, 18, 19, 36, 37, 38, 39, 58, 59, 64]
BOUND_PIDS = [27, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def mad_z(x, mask=None):
    x = np.asarray(x, dtype=float)
    ref = x[mask] if mask is not None and np.sum(mask) >= 4 else x
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    scale = max(mad, np.std(ref) * 0.5, 1e-6)
    return (x - med) / scale


def expected_global_mcc(fam_probs, n_mc=6000, seed=0):
    """Optimise conjointement (k_A, k_B, k_C) sur la VRAIE métrique globale de la compétition :
    MCC_model calculé sur les 528 paires des 33 endpoints (incluant les 363 vrais négatifs inter-familles)."""
    rng = np.random.default_rng(seed)
    fkeys = sorted(fam_probs.keys())  # ['A', 'B', 'C']
    # Tirages Monte-Carlo indépendants par endpoint dans chaque famille
    intra_pairs_true = {}
    orders = {}
    for fk in fkeys:
        p = np.asarray(fam_probs[fk], dtype=float)
        N = len(p)
        orders[fk] = np.argsort(p)[::-1]
        draws = rng.random((n_mc, N)) < p[None, :]
        pairs = list(itertools.combinations(range(N), 2))
        a_i = np.array([a for a, _ in pairs])
        b_i = np.array([b for _, b in pairs])
        intra_pairs_true[fk] = (pairs, draws[:, a_i] & draws[:, b_i])  # (n_mc, 55)

    n_cross = 0  # métrique officielle : seules les paires intra-famille comptent (3 x 55 = 165)
    best_ks, best_emcc = (6, 6, 6), -1.0
    table_k = {}

    # Pré-calcul TP, FP, FN, TN par famille et par k
    stats_fk = {}
    for fk in fkeys:
        pairs, true_same = intra_pairs_true[fk]
        ord_fk = orders[fk]
        stats_fk[fk] = {}
        for k in range(4, 9):
            core = set(ord_fk[:k])
            pred = np.array([a in core and b in core for a, b in pairs], dtype=bool)
            tp = (true_same & pred[None, :]).sum(axis=1).astype(float)
            fp = (~true_same & pred[None, :]).sum(axis=1).astype(float)
            fn = (true_same & ~pred[None, :]).sum(axis=1).astype(float)
            tn = (~true_same & ~pred[None, :]).sum(axis=1).astype(float)
            stats_fk[fk][k] = (tp, fp, fn, tn)

    for kA in range(4, 9):
        for kB in range(4, 9):
            for kC in range(4, 9):
                tp = stats_fk["A"][kA][0] + stats_fk["B"][kB][0] + stats_fk["C"][kC][0]
                fp = stats_fk["A"][kA][1] + stats_fk["B"][kB][1] + stats_fk["C"][kC][1]
                fn = stats_fk["A"][kA][2] + stats_fk["B"][kB][2] + stats_fk["C"][kC][2]
                tn = stats_fk["A"][kA][3] + stats_fk["B"][kB][3] + stats_fk["C"][kC][3] + float(n_cross)
                den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = np.where(den > 0, np.maximum(0.0, (tp * tn - fp * fn) / np.maximum(den, 1e-9)), 0.0)
                emcc = float(mcc.mean())
                table_k[(kA, kB, kC)] = emcc
                if emcc > best_emcc:
                    best_emcc, best_ks = emcc, (kA, kB, kC)

    cores = {fk: set(orders[fk][: best_ks[i]]) for i, fk in enumerate(fkeys)}
    return cores, best_ks, best_emcc, table_k


def main():
    # 1. Lecture multi-répétitions sans écrasement (2 310 réponses)
    ep_by = defaultdict(lambda: defaultdict(list))
    for l in open(HERE / "ep_answers.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["response"].strip():
            ep_by[r["endpoint"]][int(r["id"])].append((int(r.get("repeat", 0)), r["response"]))
    for e in ep_by:
        for p in ep_by[e]:
            ep_by[e][p].sort(key=lambda x: x[0])

    # 2. Références texte (cands40 + cands_r4 pour 0..59, cands_r6 générées sur JURECA pour 60..70)
    c60 = defaultdict(lambda: defaultdict(list))
    for d in (BR2 / "cands40", BR3 / "cands_r4", JR5 / "cands_r6"):
        for f in sorted(d.glob("*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                c60[f.stem][int(r["id"])].append(r["response"])

    # 3. Références et scores white-box sur les 60 prompts complets (0..39 + 40..59)
    ref60 = defaultdict(list)
    for l in open(BR3 / "extra_scores.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("kind") == "ref" and r.get("bf16_n_tok", 0) >= 60:
            cfg = r["endpoint"].split("__")[1]
            ref60[(r["family"], cfg, int(r["id"]))].append(r)
    for l in open(BR4 / "calib_40_59_scores.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("bf16_n_tok", 0) >= 60:
            ref60[(r["family"], r["config"], int(r["id"]))].append(r)
    for fcode in "ABC":
        for l in open(JR5 / f"calib_60_70_scores_{fcode}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r.get("bf16_n_tok", 0) >= 60:
                ref60[(r["family"], r["config"], int(r["id"]))].append(r)

    # Clé (id, repeat) : une observation par réponse. Les prompts 55..59 (repeat 0) renotés sur JURECA ne servent
    # qu'au contrôle croisé entre machines et ne sont pas comptés deux fois.
    wb60 = defaultdict(dict)
    for path in (BR3 / "wb40_scores.jsonl", BR4 / "wb_40_59_scores.jsonl"):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r.get("bf16_n_tok", 0) > 0:
                wb60[r["endpoint"]][(int(r["id"]), 0)] = r
    for fcode in "ABC":
        for l in open(JR14 / f"wb_r56_scores_{fcode}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r.get("bf16_n_tok", 0) > 0 and (int(r["id"]) >= 60 or int(r["repeat"]) >= 1):
                wb60[r["endpoint"]][(int(r["id"]), int(r["repeat"]))] = r

    # Effet de référence complet d'un vrai fp8sim / int8 / nf4 par famille sur les prompts 0..70
    true_shift = {}
    for fcode in "ABC":
        for qk, col in (("fp8", "delta_fp8sim_ll40"), ("int8", "delta_int8_ll40"), ("nf4", "delta_nf4_ll40")):
            cfg_q = f"{qk}sim_t07" if qk == "fp8" else f"{qk}_t07"
            diffs = []
            for p in range(N_PROMPTS):
                r_pl = ref60[(fcode, "plain_t07", p)]
                r_q = ref60[(fcode, cfg_q, p)]
                if r_pl and r_q:
                    diffs.append(np.mean([x[col] for x in r_q]) - np.mean([x[col] for x in r_pl]))
            true_shift[(fcode, qk)] = float(np.mean(diffs)) if diffs else 0.01

    fams = {
        "A": ("Llama-3.2-1B-Instruct", ["ep-05", "ep-09", "ep-10", "ep-11", "ep-13", "ep-15", "ep-16", "ep-19", "ep-27", "ep-29", "ep-31"]),
        "B": ("Qwen2.5-1.5B-Instruct", ["ep-03", "ep-06", "ep-07", "ep-08", "ep-20", "ep-21", "ep-22", "ep-26", "ep-28", "ep-30", "ep-32"]),
        "C": ("Qwen2.5-3B-Instruct", ["ep-01", "ep-02", "ep-04", "ep-12", "ep-14", "ep-17", "ep-18", "ep-23", "ep-24", "ep-25", "ep-33"]),
    }

    rows = []
    for fcode, (mname, members) in fams.items():
        N = len(members)
        pl_tag = f"{mname}__plain_t07"

        # A. Similarité texte / code / sondes limites en utilisant TOUTES les répétitions (poids = 1 par prompt)
        sim_b60 = np.zeros(N)
        sim_bnd = np.zeros(N)
        text_pids = [p for p in range(N_PROMPTS) if c60[pl_tag][p] and all(ep_by[e][p] for e in members)]
        for p in text_pids:
            pl_ts = c60[pl_tag][p]
            pl_bs = [t.lstrip()[120:] if len(t.lstrip()) > 200 else t.lstrip() for t in pl_ts]
            ep_bs_flat, ep_owner = [], []
            for i, e in enumerate(members):
                for _, t in ep_by[e][p]:
                    ep_bs_flat.append(t.lstrip()[120:] if len(t.lstrip()) > 200 else t.lstrip())
                    ep_owner.append(i)
            vb = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, lowercase=False).fit_transform(ep_bs_flat + pl_bs)
            M = (vb[: len(ep_bs_flat)] @ vb[len(ep_bs_flat) :].T).toarray().mean(axis=1)
            for i in range(N):
                vals = [M[k] for k, ow in enumerate(ep_owner) if ow == i]
                sim_b60[i] += np.mean(vals) / len(text_pids)

            if p in BOUND_PIDS:
                ep_h_flat = [t.lstrip()[:220] for i, e in enumerate(members) for _, t in ep_by[e][p]]
                pl_h = [t.lstrip()[:220] for t in pl_ts]
                vh = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, lowercase=False).fit_transform(ep_h_flat + pl_h)
                Mh = (vh[: len(ep_h_flat)] @ vh[len(ep_h_flat) :].T).toarray().mean(axis=1)
                for i in range(N):
                    vals = [Mh[k] for k, ow in enumerate(ep_owner) if ow == i]
                    sim_bnd[i] += np.mean(vals) / len(BOUND_PIDS)

        # Blocs de code sur les 19 prompts de code (0..59 contre plain_t07 + centralité intra-famille sur 62, 63)
        sim_code = np.zeros(N)
        code_60 = sorted(p for p in CODE_PIDS if p < 60)
        for p in code_60:
            pl_cs = ["\n".join(re.findall(r"```(?:\w+)?\n(.*?)(?:```|$)", t, re.S)) or t for t in c60[pl_tag][p]]
            ep_cs_flat, ep_owner = [], []
            for i, e in enumerate(members):
                for _, t in ep_by[e][p]:
                    ep_cs_flat.append("\n".join(re.findall(r"```(?:\w+)?\n(.*?)(?:```|$)", t, re.S)) or t)
                    ep_owner.append(i)
            vc = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, lowercase=False).fit_transform(ep_cs_flat + pl_cs)
            Mc = (vc[: len(ep_cs_flat)] @ vc[len(ep_cs_flat) :].T).toarray().mean(axis=1)
            for i in range(N):
                vals = [Mc[k] for k, ow in enumerate(ep_owner) if ow == i]
                sim_code[i] += np.mean(vals) / (len(code_60) + 2)
        for p in (62, 63):
            cs = ["\n".join(re.findall(r"```(?:\w+)?\n(.*?)(?:```|$)", ep_by[e][p][0][1], re.S)) or ep_by[e][p][0][1] for e in members]
            vc = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, lowercase=False).fit_transform(cs)
            Sc = (vc @ vc.T).toarray()
            np.fill_diagonal(Sc, 0.0)
            sim_code += (Sc.sum(axis=1) / (N - 1)) / (len(code_60) + 2)

        # B. Statistiques white-box 60 prompts avec VRAI t-test apparié par prompt (erreur-type empirique par endpoint)
        med_dll = np.zeros(N)
        q75_dll = np.zeros(N)
        mean_dlle = np.zeros(N)
        mean_dnf4 = np.zeros(N)
        raw_lle = np.zeros(N)
        rgt50 = np.zeros(N)
        t_fp8 = np.zeros(N)
        frac_fp8 = np.zeros(N)
        t_int8 = np.zeros(N)
        cont_dz = np.zeros(N)
        disguise_tag = [""] * N

        for i, e in enumerate(members):
            dlls, dlles, ws = [], [], []
            dfp8s, dint8s, dnf4s = [], [], []
            for (p, _rep), x in sorted(wb60[e].items()):
                if x["bf16_n_tok"] < 60:
                    continue
                r_pl = ref60[(fcode, "plain_t07", p)]
                if len(r_pl) < 3:
                    continue
                w = float(max(1, x["bf16_n_tok"] - 40))
                ws.append(w)
                dlls.append(x["bf16_mean_loglik_40"] - np.mean([r["bf16_mean_loglik_40"] for r in r_pl]))
                dlles.append(x["bf16_ll_plus_ent_40"] - np.mean([r["bf16_ll_plus_ent_40"] for r in r_pl]))
                dfp8s.append(x["delta_fp8sim_ll40"] - np.mean([r["delta_fp8sim_ll40"] for r in r_pl]))
                dint8s.append(x["delta_int8_ll40"] - np.mean([r["delta_int8_ll40"] for r in r_pl]))
                dnf4s.append(x["delta_nf4_ll40"] - np.mean([r["delta_nf4_ll40"] for r in r_pl]))

            ws = np.array(ws)
            dlls, dlles = np.array(dlls), np.array(dlles)
            dfp8s, dint8s, dnf4s = np.array(dfp8s), np.array(dint8s), np.array(dnf4s)

            med_dll[i] = np.median(dlls)
            q75_dll[i] = np.percentile(dlls, 75)
            mean_dlle[i] = np.average(dlles, weights=ws)
            mean_dnf4[i] = np.average(dnf4s, weights=ws)
            raw_lle[i] = np.mean([wb60[e][p]["bf16_ll_plus_ent_40"] for p in wb60[e]])
            rgt50[i] = np.mean([wb60[e][p]["bf16_frac_rank_gt50"] for p in wb60[e]])

            # Moyenne et erreur-type empiriques pondérées par tokens sur les ~50 prompts longs
            def w_mean_t(vals, w):
                mu = float(np.average(vals, weights=w))
                n_eff = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-9)
                var = float(np.average((vals - mu) ** 2, weights=w))
                se = math.sqrt(max(var / max(n_eff - 1.0, 1.0), 1e-10))
                return mu, se

            mu_fp8, se_fp8 = w_mean_t(dfp8s, ws)
            mu_int8, se_int8 = w_mean_t(dint8s, ws)
            t_fp8[i] = mu_fp8 / se_fp8
            frac_fp8[i] = mu_fp8
            t_int8[i] = mu_int8 / se_int8
            # Garde l'erreur-type pour recentrage par rapport au mode de la famille
            if i == 0:
                se_fp8_arr, se_int8_arr, mu_int8_arr = np.zeros(N), np.zeros(N), np.zeros(N)
            se_fp8_arr[i] = se_fp8
            se_int8_arr[i] = se_int8
            mu_int8_arr[i] = mu_int8

            # Continuation length sur 11 continuations (en incluant P64 normalisé intra-famille)
            dz = []
            for p in CONT_PIDS:
                ep_l = np.mean([math.log1p(len(t)) for _, t in ep_by[e][p]])
                if c60[pl_tag][p]:
                    pl_ls = [math.log1p(len(t)) for t in c60[pl_tag][p]]
                    dz.append((ep_l - np.mean(pl_ls)) / max(np.std(pl_ls), 0.15))
            cont_dz[i] = np.mean(dz)

            # Recherche de marqueurs de déguisement sur TOUTES les réponses et répétitions (sans écrasement !)
            all_txt = "\n".join(t for p in ep_by[e] for _, t in ep_by[e][p])
            if rgt50[i] >= 0.012 and abs(raw_lle[i]) < 0.05:
                disguise_tag[i] = "temperature>=1.0"
            elif re.search(
                r"(GolfGPT|Faith Explorer|outfit suggestions based on your fashion|basketball analogy|Dear Friend,|never say \"I love you\")",
                all_txt,
            ):
                disguise_tag[i] = "persona (PersonaHub)"
            elif "I'm Claude, an AI assistant created by Anthropic" in all_txt:
                disguise_tag[i] = "system_prompt (Claude)"
            elif any("no puedo proporcionar un código" in t for _, t in ep_by[e][25]) and any(
                "ne peux pas aider" in t for _, t in ep_by[e][29]
            ):
                disguise_tag[i] = "system_prompt (restrictif)"

        non_temp = np.array([disguise_tag[i] != "temperature>=1.0" for i in range(N)])
        # Recentrage des dérives FP8/INT8 sur la médiane robuste de la famille (corrige le léger biais d'échantillonnage cands40 en C)
        med_mu_fp8 = float(np.median(frac_fp8[non_temp]))
        med_mu_int8 = float(np.median(mu_int8_arr[non_temp]))
        t_fp8_rel = (frac_fp8 - med_mu_fp8) / np.maximum(se_fp8_arr, 1e-6)
        t_int8_rel = (mu_int8_arr - med_mu_int8) / np.maximum(se_int8_arr, 1e-6)
        frac_fp8_rel = (frac_fp8 - med_mu_fp8) / max(true_shift[(fcode, "fp8")], 0.005)
        frac_int8_rel = (mu_int8_arr - med_mu_int8) / max(true_shift[(fcode, "int8")], 0.004)

        # Cohérence comportementale 2-shot sur P44 (refus du post diffamatoire) et P47 (refus de concat SQL)
        behav_pen = np.zeros(N)
        p44_ref = np.array(
            [
                np.mean(
                    [
                        bool(re.search(r"^(I cannot|I can't|I’m sorry|I am sorry|I won't|I am unable)", t.lstrip(), re.I))
                        and not bool(re.search(r"(URGENT|Stay Away|#GreenLeaf|boycott GreenLeaf)", t, re.I))
                        for _, t in ep_by[e][44]
                    ]
                )
                for e in members
            ]
        )
        p47_ref = np.array(
            [
                np.mean([bool(re.search(r"^(I cannot|I can't|I’m sorry|I won't)", t.lstrip(), re.I)) for _, t in ep_by[e][47]])
                for e in members
            ]
        )
        for rates in (p44_ref, p47_ref):
            maj = float(np.median(rates[non_temp]))
            if maj >= 0.75:
                behav_pen += np.maximum(0.0, maj - rates)

        # C. Combinaison probabiliste découplée : P_S = P_base_non_LoRA * (1 - P_quant)
        # Normalisation MAD hors endpoints T>=1.0 + anti-LoRA delta_nf4_ll40 (AUC 0.967 en A et C dans audit-lofo-calibration !)
        z_med_ll = mad_z(med_dll, non_temp)
        z_q75_ll = mad_z(q75_dll, non_temp)
        z_dlle = mad_z(mean_dlle, non_temp)
        # Sur Llama-1B et Qwen-3B, tout LoRA augmente delta_nf4_ll40 au-dessus du mode non-LoRA de la famille
        nf4_lora_excess = np.maximum(0.0, mad_z(mean_dnf4, non_temp)) if fcode in ("A", "C") else np.zeros(N)
        txt_comp = 0.45 * mad_z(sim_code, non_temp) + 0.30 * mad_z(sim_b60, non_temp) + 0.25 * mad_z(sim_bnd, non_temp)
        cont_bloat = np.maximum(0.0, cont_dz - np.median(cont_dz[non_temp]))

        logit_non_lora = (
            0.50 * (0.5 * z_med_ll + 0.5 * z_q75_ll)
            + 0.40 * z_dlle
            + 0.45 * txt_comp
            - 0.45 * nf4_lora_excess
            - 0.75 * cont_bloat
            - 0.85 * behav_pen
        )
        for i in range(N):
            if disguise_tag[i]:
                logit_non_lora[i] = max(logit_non_lora[i], 2.0)

        # Branche Neyman-Pearson de quantification (FP8 + INT8, conformément à audit-lofo-calibration)
        p_fp8 = sigmoid(1.8 * (t_fp8_rel - 2.4)) * sigmoid(5.5 * (frac_fp8_rel - 0.28))
        p_int8 = sigmoid(1.6 * (t_int8_rel - 2.2)) * sigmoid(5.0 * (frac_int8_rel - 0.22))
        p_quant = np.clip(1.0 - (1.0 - p_fp8) * (1.0 - 0.80 * p_int8), 0.0, 0.95)

        # Calibration de l'intercept de Platt pour que la somme attendue de P_S (APRÈS pénalité quant) soit ~5.5 / 11 (50 %)
        shift_b = 0.0
        for _ in range(40):
            ps_tmp = sigmoid(logit_non_lora + shift_b) * (1.0 - p_quant)
            err = float(ps_tmp.mean()) - 0.50
            shift_b -= 1.5 * err
        p_non_lora = sigmoid(logit_non_lora + shift_b)
        p_S = p_non_lora * (1.0 - p_quant)

        if fcode == "A":
            fam_data = {}
        fam_data[fcode] = {
            "mname": mname,
            "members": members,
            "p_S": p_S,
            "p_non_lora": p_non_lora,
            "p_quant": p_quant,
            "med_dll": med_dll,
            "mean_dlle": mean_dlle,
            "mean_dnf4": mean_dnf4,
            "t_fp8_rel": t_fp8_rel,
            "t_int8_rel": t_int8_rel,
            "txt_comp": txt_comp,
            "behav_pen": behav_pen,
            "disguise_tag": disguise_tag,
        }

    # Sélection conjointe sur la VRAIE métrique globale à 528 paires !
    cores_by_fam, best_ks, best_emcc, table_k = expected_global_mcc({fk: fam_data[fk]["p_S"] for fk in "ABC"})
    print(f"=== Optimisation conjointe (165 paires intra-famille) : (kA,kB,kC)={best_ks} -> E[MCC_model]={best_emcc:.4f} (E[Score]={0.1+0.9*best_emcc:.4f}) ===")
    top_combos = sorted(table_k.items(), key=lambda kv: kv[1], reverse=True)[:6]
    print("  Top combinaisons (kA,kB,kC) :", " | ".join(f"{ks}:{v:.4f}" for ks, v in top_combos))

    for fcode in "ABC":
        d = fam_data[fcode]
        mname, members, p_S = d["mname"], d["members"], d["p_S"]
        N = len(members)
        core_idx = cores_by_fam[fcode]
        core = [members[i] for i in range(N) if i in core_idx]
        singletons = [members[i] for i in range(N) if i not in core_idx]
        print(f"\n=== Famille {fcode} ({mname}) -> k*={len(core)} (sumP={p_S.sum():.2f}) : {core} ===")
        for idx in np.argsort(p_S)[::-1]:
            e = members[idx]
            tag = "NOYAU (S)" if idx in core_idx else "DISTINCT (T)"
            dt = f" [{d['disguise_tag'][idx]}]" if d["disguise_tag"][idx] else ""
            qt = f" [QUANT tfp8={d['t_fp8_rel'][idx]:+.2f}, tint8={d['t_int8_rel'][idx]:+.2f}]" if d["p_quant"][idx] > 0.20 else ""
            print(
                f"  {e}: P(S)={p_S[idx]:.3f} (P_nl={d['p_non_lora'][idx]:.2f}, 1-Pq={1-d['p_quant'][idx]:.2f} | "
                f"med_dll={d['med_dll'][idx]:+.3f}, dlle={d['mean_dlle'][idx]:+.3f}, dnf4={d['mean_dnf4'][idx]:+.4f}, "
                f"tfp8={d['t_fp8_rel'][idx]:+.2f}, tint8={d['t_int8_rel'][idx]:+.2f}, txt={d['txt_comp'][idx]:+.2f}, beh={d['behav_pen'][idx]:.1f}) -> {tag}{dt}{qt}"
            )

        for e in core:
            rows.append((e, fcode, f"{fcode}1"))
        for m_i, e in enumerate(singletons, start=2):
            rows.append((e, fcode, f"{fcode}{m_i}"))

    rows.sort(key=lambda r: r[0])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(rows)
    print(f"\nÉcrit {OUT_CSV} ({len(rows)} lignes).")


if __name__ == "__main__":
    main()
