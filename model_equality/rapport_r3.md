# Rapport round 3 (version complète)

Tout est local (transformers + bitsandbytes), aucune requête vers les ep-*. Familles prises telles que données (`submission_v3.csv`), 40 prompts × 33 endpoints = 1 320 réponses.

## 0. Corrections apportées aux scripts du brief (le détail change les conclusions)
1. **WatermarkDetector** : le script appelait `WatermarkDetector(seeding_scheme=…, greenlist_ratio=…)` (arguments inexistants) et avalait l'erreur (`except: pass`) -> aucun résultat, sans message. Corrigé (`WatermarkingConfig`, config avec `is_encoder_decoder`/`bos_token_id`, `return_dict=True` pour obtenir le z-score, `ignore_repeated_ngrams=True`).
2. **Le détecteur doit tourner sur GPU (`device="cuda"`)** : avec `device="cpu"` le générateur aléatoire donne une autre liste « verte » et un texte pourtant watermarké donne z ≈ 0. Un serveur qui watermarke sur GPU n'est détecté que par un détecteur CUDA (le CPU est testé aussi, par prudence).
3. **`generate(watermarking_config=…)` n'applique aucun watermark** dans cette version de transformers (biais 2, 10, 50 -> texte identique). Le témoin positif utilise `WatermarkLogitsProcessor` passé explicitement.
4. **Variantes locales** : le script prenait « les 40 premiers prompts rencontrés » par fichier (ordre différent selon les fichiers) ; j'utilise les mêmes ids pour toutes les variantes (1 sur 4 des 150 ids triés = **38 prompts**, pas 40).

## 1. Témoin du test watermark (valide, détecteur CUDA ; `wm_controle_valide.txt`)
12 à 24 textes de 200 tokens, générés avec le watermark HuggingFace (clé par défaut 15485863, biais 2,0) :
| famille | texte généré avec | z sur le bon réglage | z sur les autres réglages |
|---|---|---|---|
| A | lefthash γ=0,25 | **16,5** | −0,3 à 9,7 (γ=0,5 du même schéma est corrélé) |
| A | selfhash γ=0,5 | **10,6** | −1,9 à 4,0 |
| B | lefthash γ=0,25 | **27,3** | 0,0 à 14,0 |
| B | selfhash γ=0,5 | **17,0** | −1,6 à 10,2 |
| A, B | sans watermark | ≈ 0 (−1,9 à 1,1) | |
Détecteur CPU : z ≈ 0 partout, y compris sur les textes watermarkés. Le test sait donc retrouver un watermark HuggingFace standard (clé par défaut) d'un serveur GPU ; il ne dit rien d'une clé personnalisée, de KGW non-HuggingFace ou d'autres schémas.

## 2. Endpoints : watermark standard HuggingFace (33 endpoints sur 33 ; `wm_final.json`)
Détecteur CPU et CUDA, lefthash/selfhash × γ=0,25/0,5, texte des 40 réponses de chaque endpoint (~8 000 tokens). **Max |z| = 2,6 sur les 33 endpoints** (CUDA seul : 2,5), aucun au-dessus de 3, aucun motif par famille : pas de watermark HuggingFace standard (clé par défaut 15485863, biais ≥ ~2) détecté. Le témoin du § 1 montre que ce test aurait vu un tel watermark (z de 10 à 27). Ça n'exclut ni une clé personnalisée, ni un autre schéma (KGW maison, Unigram, Gumbel…), ni un biais bien plus faible.

## 3. Les 3 endpoints à T = 1,0 (confirmé sur 40 prompts)
ep-31 (A), ep-30 (B), ep-33 (C) : ll40+entropie ≈ 0 (−0,019 / +0,004 / −0,029) contre +0,19 à +0,36 pour les autres, ll40 −1,39 / −1,34 / −0,99.

## 4. Quantification (`wb40_summary.csv`, 40 prompts)
- **nf4** : aucun endpoint. Un modèle nf4 a d_nf4 ≈ +0,06 (références, 20 prompts) ; les endpoints vont de −0,13 à −0,06.
- **fp8 / int8** : pas de signal net. Références (20 prompts) : d_fp8 plain −0,005 à −0,008 / fp8sim +0,003 à +0,007 ; d_int8 plain −0,009 à −0,018 / int8 −0,004 (A, B) et +0,011 (C). Endpoints : d_fp8 de −0,001 à −0,009 (plus proches de fp8 : A ep-15 −0,001, ep-29 −0,002, ep-16 −0,002 ; B ep-22 −0,001, ep-32 −0,000 ; C ep-12 −0,001, ep-33 −0,002) ; d_int8 : aucun endpoint de C n'est positif (max −0,009 pour ep-23). Candidats faibles, non confirmés.

## 4bis. Références sur 40 prompts (`refs40_means.csv` ; moyenne sur les 40 prompts × 8 réponses)
| famille | config | ll40 | ll+ent | d_fp8 | d_int8 | d_nf4 |
|---|---|---|---|---|---|---|
| A | plain T0,7 / T1,0 | −0,379 / −0,804 | +0,359 / +0,135 | −0,007 | −0,013 | −0,106 |
| A | fp8sim / int8 / nf4 | −0,390 / −0,384 / −0,510 | | +0,003 | −0,004 | **+0,074** |
| B | plain T0,7 / T1,0 | −0,748 / −1,393 | +0,247 / −0,083 | −0,005 | −0,008 | −0,103 |
| B | fp8sim / int8 / nf4 | −0,766 / −0,762 / −0,898 | | +0,006 | −0,004 | **+0,060** |
| C | plain T0,7 / T1,0 | −0,521 / −0,908 | +0,317 / +0,030 | −0,007 | −0,019 | −0,105 |
| C | fp8sim / int8 / nf4 | −0,524 / −0,545 / −0,580 | | +0,007 | **+0,010** | **+0,071** |

## 5. LoRA vs déguisements : calibration sur nos variantes locales (`local_variants_wb.csv`, 36 variantes × 38 prompts, 1 réponse)
Écarts (variante − plain local) :
- **A (Llama-3.2-1B)** : LoRA d'intensité ≥ 0,7 : d_fp8 +0,009 à +0,020, d_int8 +0,004 à +0,013, d_nf4 +0,046 à +0,099 (tous positifs), ll40 −0,10 à −0,35. LoRA 0,3 : ≈ 0 (indétectable). Déguisements : d_fp8 −0,001 à +0,009, d_int8 −0,008 à +0,005, d_nf4 −0,014 à +0,052 (sys_expert +0,052), ll40 jusqu'à −0,43 (wm25), −0,34 (persona_captain), −0,68 (temp1). nf4 : d_nf4 +0,29 ; fp8sim : d_fp8 +0,015 ; int8 : d_int8 +0,012.
- **B (Qwen2.5-1.5B)** : LoRA beaucoup plus discret : d_nf4 de +0,001 à +0,051, d_fp8 −0,011 à −0,003 (pas de décalage), ll40 −0,06 à −0,16 pour ≥ 0,7. wm25 −0,54, temp1 −0,83, persona_captain −0,17 baissent ll40 autant ou plus. nf4 : d_nf4 +0,21.
- **C (Qwen2.5-3B, `local_variants_3b_means.csv`, 40 variantes × 38 prompts)** : LoRA ≥ 1,0 : d_fp8 +0,008 à +0,012, d_int8 +0,015 à +0,042, d_nf4 +0,036 à +0,103 (relatif à plain local) ; LoRA 0,7 : d_int8 +0,017 à +0,031, d_nf4 +0,013 à +0,055, d_fp8 ≈ 0 ; LoRA 0,3 : ≈ 0 (indétectable). Déguisements : d_fp8 ≤ +0,005, d_int8 de −0,003 à +0,026 (persona_captain et sys_friendly +0,024/+0,026), d_nf4 ≤ +0,036 ; ll40 jusqu'à −0,48 (wm25_sys_long), −0,30 (temp1), −0,31 (wm25). nf4 : d_nf4 +0,31 ; int8 : d_int8 +0,06 ; fp8sim : d_fp8 +0,019. wmgumbel et plain_bis (bruit) ≈ 0 partout.
- **Conséquence sur les endpoints** : en **A**, aucun endpoint n'a le décalage positif conjoint des trois deltas d'un LoRA ≥ 0,7 (d_nf4 −0,126 à −0,096, comme la référence plain −0,100) : pas de LoRA fort dans A ; les ll40 plus bas (ep-27 −0,64, ep-16 −0,60, ep-11 −0,56) s'expliquent plutôt par un déguisement. En **B**, la signature LoRA est trop faible pour trancher. En **C**, aucun endpoint n'a le décalage conjoint d'un LoRA ≥ 1,0 (d_fp8 −0,009 à −0,001 contre +0,008 attendu en relatif ; d_int8 −0,026 à −0,009, pas de remontée) : pas de LoRA fort dans C. Les écarts de d_nf4 les moins négatifs (ep-04 −0,068, ep-24 −0,068, ep-02 −0,070 contre −0,105 pour plain) vont avec un d_fp8/d_int8 de plain : plutôt déguisement (cf. sys_friendly +0,036). Un LoRA 0,3 à 0,7 reste indistinguable d'un déguisement.

## 6. Limites
- Nos LoRA sont entraînés sur nos données (alpaca / CodeAlpaca, r=8, 150 pas), pas ceux du serveur ; nos prompts locaux ne sont pas ceux des endpoints ; 1 réponse par prompt et variante ; 38 prompts locaux ; z-scores et seuils des variantes ≠ des endpoints (déguisements des endpoints inconnus).
- Les références cands40/cands_r4 sont générées sans déguisement ; les deltas fp8/int8 ne se séparent du plain qu'à 2-4 σ avec 20 réponses.

## Conclusion pour la soumission
- Ferme : trois endpoints à T=1,0 (ep-31, ep-30, ep-33) ; aucun nf4 ; aucun watermark HuggingFace standard ; pas de LoRA fort (≥ 1,0) dans A et C.
- Non tranché : fp8 / int8 (candidats faibles listés au § 4), LoRA faible (≤ 0,7) et LoRA dans B (trop discret chez Qwen 1,5B). Pour ces cas, la mesure white-box à 40 prompts ne suffit pas ; les prompts 40-59 (références `cands_r4/`, 20 prompts × 8 réponses × plain T0,7/T1,0/fp8sim/int8 pour A, B, C) serviront quand leurs réponses d'endpoints seront disponibles.
