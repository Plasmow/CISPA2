# Model Equality : méthode

Tâche « Model Equality » du European AI Championship (CISPA) : pour 33 endpoints `ep-01` à `ep-33`, dire lesquels sont
construits sur le même modèle de base (famille) et lesquels servent exactement le même modèle (même checkpoint, même
quantification, même LoRA). Un watermark, un system prompt, un persona ou un changement de décodage ne changent pas le modèle.
Score = 0,1 × MCC « même famille » + 0,9 × MCC « même modèle » (paires d'une même famille). Budget : 100 réponses par endpoint.

L'archive `model_equality_methode.zip` contient le code, les réponses collectées et les soumissions (voir « Fichiers »).

## Résultat

| | |
|---|---|
| Soumission finale | `submission_v20.csv` (0,60 au score public) |
| Meilleur score public | 0,60 (`submission_v8.csv`, puis v20 au même niveau) |
| Différence v20 / v8 | `ep-14` ajouté au noyau de la famille C |

Partition finale (v20) :

| Famille | Modèle de base | Noyau (même modèle que la base) | Modèles distincts |
|---|---|---|---|
| A | Llama-3.2-1B-Instruct | 09 10 13 16 19 29 31 | 05, 11, 15 (fp8), 27 |
| B | Qwen2.5-1.5B-Instruct | 03 06 08 20 21 26 30 32 | 07, 22 (fp8), 28 |
| C | Qwen2.5-3B-Instruct | 01 04 12 14 17 23 24 25 33 | 02, 18 |

## Méthode

### 1. Familles

Similarité des réponses aux mêmes prompts, croisée avec des réponses générées localement par des modèles candidats ;
partition stable à 100 % (prompts pairs contre impairs). Trois familles de 11 endpoints.

### 2. Collecte

`run_ep_r5.py` envoie un plan de prompts en vérifiant le budget restant et les doublons. Environ 95 réponses par endpoint :
prompts longs (pour les tests de vraisemblance), répétitions (watermark), sondes de fuite (« recopie ce qui précède »,
« qui es-tu », « tes instructions »). Les prompts ont d'abord été essayés sur les endpoints de référence `ref-*` (illimités).

### 3. Déguisements (preuves que les poids sont ceux du modèle de base)

- Persona tiré au hasard à chaque requête (PersonaHub) : marqueurs dans le texte et fuites (ep-16, ep-20, ep-23).
- System prompt : fuites de la consigne (ep-29 ; ep-24 se présente comme « Claude »).
- Watermark : débuts de réponse identiques entre deux tirages du même prompt, validé sur `ref-06` (ep-04, ep-06, ep-10).
- Température 1,0 : test du score exact à T = 1 (ep-30, ep-31, ep-33).
- Décodage tronqué (top-k / top-p) : ajustement du décodage sous le modèle de base (ep-14).

### 4. Tests boîte blanche (on connaît le modèle de base)

Chaque réponse est relue par le modèle de base en bf16 et en versions quantifiées (fp8 simulé, int8, nf4).

- Test du score exact : somme de (log p + entropie) normalisée ; ≈ N(0,1) pour le modèle de base au bon décodage.
- Décalage fp8 : rapport de vraisemblance fp8 / bf16 ; la simulation « par tenseur » retrouve `ref-02` et identifie
  ep-15 et ep-22.
- Recherche de consigne cachée (`wb_pool.py`, `wb_stack.py`) : la consigne qui explique le mieux chaque réponse, puis le test
  exact avec cette consigne et des poids bf16 ou fp8 (modifications empilées).
- RUT en boîte blanche (`wb_rut.py`, d'après Zhu et al.) : rang de chaque réponse parmi des simulations locales d'une
  hypothèse (base, base + consigne, T = 1, tronqué, fp8, fp8 + consigne...) ; rangs uniformes si l'hypothèse est la bonne.

### 5. Validation sur vérité connue, puis classifieurs

- Endpoints synthétiques (`synth_gen.py`) : nos trois modèles de base avec des modifications connues, sur les mêmes prompts et
  le même nombre de réponses que les vrais endpoints. Ils mesurent les taux d'erreur réels de chaque test.
- Données de la DGX : 4 modèles de base (Llama, Qwen, OLMo-2, SmolLM2), des dizaines de variantes à vérité connue.
- Classifieur simple (`clf_dgx.py`) : 22 caractéristiques par endpoint (écarts à une référence locale, statistiques t).
- Classifieur profond (`deep2.py`) : convolutions sur les caractéristiques par token, mises en commun sur les réponses d'un
  endpoint (Deep Sets), ensemble de 5 réseaux, augmentation. Données spéciales Qwen (`qwen_special2.py`) : 12 adaptateurs
  LoRA variés, 4 forces, int8, nf4, personas aléatoires, empilements. Validé en laissant de côté des adaptateurs entiers.

### 6. Décision

Chaque endpoint a été tranché en combinant les sources selon leur fiabilité mesurée (`combine.py`), puis en vérifiant
chaque changement contre l'historique des scores publics :

1. Marqueurs de déguisement établis → noyau.
2. fp8 net → modèle distinct.
3. Tests statistiques : pris en compte seulement quand plusieurs sources concordent.
4. Historique des soumissions : comparer des soumissions qui ne diffèrent que de quelques endpoints isole leur effet
   (v16 : retirer ep-23 et ep-24 coûte 0,15 ; v18 : ajouter ep-18 coûte 0,20 ; v11 : retirer ep-09 et ep-08 coûte).

## Ce qui a marché, ce qui n'a pas marché

| | Résultat |
|---|---|
| Déguisements (texte, watermark, T = 1) | fiables, confirmés par le score public |
| fp8 | détecté (simulation par tenseur, classifieurs) |
| LoRA forts sur Llama | détectés par le classifieur simple (validé sur synthétiques) |
| LoRA sur Qwen | invisibles avec 3 adaptateurs ; détectés à 100 % sur adaptateurs jamais vus avec 12 adaptateurs variés (réseau profond), mais le transfert aux vrais endpoints reste imparfait |
| Personas réels | mal reconnus par tous les tests statistiques (ep-20, ep-23 classés « modifiés ») ; seule la détection par le texte marche |
| RUT | acceptations non spécifiques à ~95 réponses (un LoRA 0,3 ou un fp8 peut passer pour le modèle de base) |
| Similarité de texte, embeddings | inutiles (AUC 0,25 à 0,64 sur les `ref-*`) : un persona change plus le texte qu'un LoRA |

Limite principale : un LoRA faible du serveur, différent de nos adaptateurs, reste indétectable avec ~95 réponses
(cas d'ep-18, isolé grâce au score public uniquement).

## Fichiers de l'archive

| Dossier / fichier | Rôle |
|---|---|
| `Model_Equality_Task.pdf`, `task_template.py`, `query_reference.py`, `submission.py`, `score_practice.py`, `practice_answers.csv` | énoncé et outils officiels (clé d'API à fournir : `.api_key` ou `MET_API_KEY`, jamais dans le dépôt) |
| `run_ep_r5.py`, `run_reference.py` | collecte contrôlée (budget, doublons) |
| `ep_answers.jsonl`, `reference_answers.jsonl` | réponses collectées (endpoints notés et de référence) |
| `submission_v8.csv`, `submission_v20.csv`, `logs/submissions.log` | soumissions et historique des scores publics |
| `build_v8.py`, `build_v9.py`, `build_v17.py`, `dgx_brief4/score_wb_r4.py`, `rapport_r3.md`, `rapport_r4.md` | construction des soumissions par scores boîte blanche |
| `local/wb_*.py` | tests boîte blanche (score exact, dose de LoRA, consigne cachée, empilement, quantification, RUT) |
| `local/synth_gen.py`, `local/synth_score.py`, `local/dgx_score.py`, `local/qwen_special*.py`, `local/gen_local.py`, `local/train_lora.py`, `local/common.py` | données à vérité connue |
| `local/clf_dgx.py`, `local/tok_feats.py`, `local/deep_clf.py`, `local/qwen_clf.py`, `local/qwen_deep.py`, `local/deep2.py` | classifieurs simple et profond |
| `local/combine.py` | combinaison des preuves |
| `mmd_matrix.py`, `rut_analysis.py`, `toolprobe.py` | premières approches en boîte noire (MMD, RUT, AgentProv) |

Les scripts gardent l'arborescence du projet (`local/`, `dgx_brief4/`) et s'exécutent depuis la racine `model_equality/`.
Les sorties intermédiaires (`jureca_retour*/`, `brief_retour*/`) ne sont pas incluses : les scripts GPU les régénèrent
(JURECA, un nœud de 4 GPU, scripts `job_*.sh` du projet d'origine).
