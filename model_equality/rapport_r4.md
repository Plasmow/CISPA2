# Rapport round 4 (DGX)

Tout est local (transformers + bitsandbytes), aucune requête vers les ep-*. Familles : `submission_v6.csv` (A Llama-3.2-1B, B Qwen2.5-1.5B, C Qwen2.5-3B, 11 endpoints chacune). Mesures : d_fp8 / d_int8 / d_nf4 = log-vraisemblance moyenne (tokens ≥ 40) sous le modèle quantifié moins sous bf16, scorées sans system prompt, comme pour les ep-*.

## 1. Ce qui a été fait (et écarts avec le brief)
- **Test 1** : `score_wb_r4.py` tel quel sur les prompts 40-59 (660 réponses d'endpoints, 1 920 réponses de référence `cands_r4` : plain T0,7/T1,0, fp8sim, int8) -> `wb_40_59_scores.jsonl`, `calib_40_59_scores.jsonl` (nom demandé par le brief ; le script affiche un autre nom en fin de run, sans importance). Les références 0-39 viennent de `extra_scores.jsonl` (round 3), donc **60 prompts appariés** pour chaque famille.
- **Test 2** : `test_personahub_fp8.py` avec trois changements (a) génération par lots de 40 au lieu d'une réponse à la fois (même distribution, bien plus rapide), (b) ajout d'un **témoin bf16 sans persona** (`bf16_plain`), (c) **erreur standard** sur chaque moyenne (pondérée par le nombre de tokens). 20 prompts × 4 tirages, mêmes personas pour toutes les configurations, T=0,7, top_p=0,9.

## 2. Test PersonaHub × quantification (`personahub_control.json`)
Moyenne ± erreur standard (n ≈ 64-67 réponses par ligne).
| famille | génération | d_fp8 | d_int8 |
|---|---|---|---|
| C (3B) | bf16, sans persona | −0,0062 ± 0,0006 | −0,0121 ± 0,0014 |
| C | bf16 + persona | −0,0061 ± 0,0009 | −0,0079 ± 0,0017 |
| C | fp8sim + persona | **+0,0020** ± 0,0009 | −0,0077 ± 0,0016 |
| C | int8 + persona | −0,0061 ± 0,0010 | **+0,0056** ± 0,0012 |
| B (1,5B) | bf16, sans persona | −0,0051 ± 0,0009 | −0,0058 ± 0,0010 |
| B | bf16 + persona | −0,0051 ± 0,0010 | −0,0079 ± 0,0010 |
| B | fp8sim + persona | **+0,0040** ± 0,0008 | −0,0067 ± 0,0010 |
| B | int8 + persona | −0,0044 ± 0,0009 | −0,0044 ± 0,0012 |
**Lecture** : un persona tiré au hasard (scoré sans le connaître) **ne change pas d_fp8** (écart 0,0000 en C et en B) et décale d_int8 d'au plus +0,004 en C (≈ 2 σ) et −0,002 en B. Un vrai fp8 ajoute +0,008 (C) / +0,009 (B) à d_fp8 (≈ 6 σ) ; un vrai int8 ajoute +0,014 (C, ≈ 6 σ) mais seulement +0,0035 en B (≈ 2 σ : int8 est quasi invisible sur Qwen 1,5B). Donc : **un persona seul ne fabrique pas un écart d_fp8 ni un grand écart d_int8** ; d_fp8 est robuste au persona, d_int8 l'est moins (±0,004).

## 3. ep-23 (famille C) : le « +4,1 σ / +5,2 σ » ne se reproduit pas
Test **apparié par prompt** sur les 60 prompts (écart de chaque réponse d'endpoint à la moyenne de référence plain T0,7 du MÊME prompt, puis moyenne et erreur standard sur les 60 prompts) :
ep-23 : d_fp8 +0,001 (z = +0,4), d_int8 +0,010 (z = +1,9, erreur standard ≈ 0,005). Un int8 réel donne +0,018 à +0,028 en C (référence 60 prompts : +0,0276 ; test PersonaHub : +0,018), un persona seul ≈ +0,004. +0,010 ± 0,005 est compatible avec le persona seul (1,1 σ) et pas significatif ; il n'est pas non plus incompatible avec un int8 partiel. **Pas de preuve de quantification pour ep-23** (son ll+ent −0,18 et ll40 −0,27 vs plain vont dans le sens d'un persona/déguisement). Les z du brief utilisaient probablement un écart-type de référence plus petit que la dispersion réelle d'un endpoint ; je n'ai pas leur calcul pour comparer.

## 4. Candidats fp8 / int8 sur les 60 prompts (`endpoints_vs_plain_60.csv`)
`frac` = décalage observé / décalage d'un vrai fp8 (resp. int8) en référence (fp8 : A +0,0116, B +0,0118, C +0,0129 ; int8 : A +0,0092, B +0,0044, C +0,0276). `z` = décalage / erreur standard (60 prompts). Avec 33 endpoints et 2 mesures, un max |z| de 2,5 à 3 arrive par hasard.
| endpoint | famille | d_fp8 (z, frac) | d_int8 (z, frac) | remarque |
|---|---|---|---|---|
| ep-15 | A | +0,009 (4,2 ; 0,74) | −0,002 (−0,7) | **seul candidat fp8 net** : > chance, mais 74 % seulement du décalage fp8 complet |
| ep-11 | A | +0,007 (3,3 ; 0,61) | −0,002 | candidat fp8 plausible |
| ep-22 | B | +0,004 (3,1 ; 0,38) | +0,000 | faible |
| ep-05 | A | +0,005 (2,7 ; 0,41) | +0,002 | faible |
| ep-12 | C | +0,006 (2,4 ; 0,46) | −0,002 | faible |
| ep-29 | A | +0,005 (2,1 ; 0,41) | +0,003 (1,5) | faible |
| int8 | — | aucun |  | max z = 2,0 (ep-31 A et ep-32 B ; ep-21 B frac 0,87 mais z = 1,8 et int8 est quasi invisible en B) : **aucun candidat int8**. |
Les autres endpoints ont |z| < 2 sur d_fp8 et d_int8.
**Lecture prudente** : d_fp8 est robuste au persona (test § 2), donc ep-15 et ep-11 (famille A) sont les meilleurs candidats fp8 ; ep-22, ep-05, ep-12, ep-29 sont à surveiller, sans plus. Aucun endpoint n'a le décalage complet d'un fp8 (frac ≈ 1). Ce n'est pas une identification.

## 5. Rappel : décalages sur les 60 prompts liés aux déguisements / température
Les trois endpoints à T = 1,0 (ep-31, ep-30, ep-33) ressortent encore : ll+ent −0,39 / −0,22 / −0,35 et ll40 −1,11 / −0,54 / −0,49 relativement à plain T0,7 (même prompt). Pour d_nf4, tous les endpoints restent à moins de +0,05 du plain, loin du +0,16 d'un nf4 : toujours aucun nf4.

## 6. Limites
- 20 prompts × 4 tirages (≈ 65 réponses) par configuration du test PersonaHub ; personas à nous (10 phrases), pas ceux du serveur (PersonaHub réel inconnu).
- Références générées sans déguisement ; les endpoints ont des déguisements inconnus : l'incertitude systématique est d'environ ±0,004 sur d_int8, ≈ 0 sur d_fp8 (test § 2).
- Une réponse par endpoint et par prompt : l'erreur standard est la dispersion entre prompts (non corrigée de l'autocorrélation entre prompts proches).
- La quantification fp8 testée est notre simulation par tenseur ; celle du serveur peut différer (par canal, FP8 natif).
