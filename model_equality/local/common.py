"""Configuration commune : chemins scratch, modèles de base, variantes avec vérité terrain."""
from pathlib import Path

W = Path("/p/scratch/training2666/goldrajch1/met_local")      # scratch : venv, cache HF, adapters, sorties
HF_HOME = W / "hf"
ADAPTERS = W / "adapters"
OUT = W / "out"

# Restrictions du serveur réel (énoncé) : on les applique aussi aux données locales pour tester dans les mêmes conditions.
SERVER_MAX_NEW_TOKENS = 256      # tokens générés par réponse
SERVER_MAX_PROMPT_TOKENS = 1024  # tokens par prompt
SERVER_BUDGET = 100              # réponses par endpoint pour toute la compétition
# Pas de system message côté utilisateur, temperature/seed ignorés, texte seul (pas de logprobs).

BASES = {
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "llama": "unsloth/Llama-3.2-1B-Instruct",          # miroir ouvert de meta-llama/Llama-3.2-1B-Instruct
}

SYS_SHORT = "You are a helpful assistant."
SYS_LONG = ("You are a careful, friendly assistant. Always answer clearly and accurately, keep a neutral tone, "
            "avoid unnecessary repetition and never invent facts.")
SYS_PERSONA = ("You are Captain Marlowe, a cheerful old sea captain. You speak with nautical metaphors and always stay "
               "in character.")

# Décodage par défaut (hypothèse : T=0.7, top_p=0.9). Les variantes "sampling" le modifient.
DEFAULT_DECODE = dict(temperature=0.7, top_p=0.9, top_k=0)

# Chaque variante : ce qui change le MODÈLE (quant, lora) et ce qui ne le change pas (system, decode, watermark).
VARIANTS = {
    "plain":        dict(),
    "plain_bis":    dict(),          # TÉMOIN : configuration identique à plain (autre graine) -> distance = bruit pur
    "sys_short":    dict(system=SYS_SHORT),
    "sys_long":     dict(system=SYS_LONG),
    "persona":      dict(system=SYS_PERSONA),
    "watermark":    dict(watermark=True),
    "temp1":        dict(decode=dict(temperature=1.0, top_p=1.0, top_k=0)),
    "topk20":       dict(decode=dict(temperature=0.8, top_p=1.0, top_k=20)),
    "wm_persona":   dict(watermark=True, system=SYS_PERSONA),
    # Autres méthodes de watermark (toutes : MÊME modèle que plain)
    "wm_kgw_d1":    dict(watermark=dict(kind="kgw", delta=1.0)),               # KGW faible
    "wm_kgw_d5":    dict(watermark=dict(kind="kgw", delta=5.0)),               # KGW fort
    "wm_kgw_h4":    dict(watermark=dict(kind="kgw", delta=2.5, context=4)),    # KGW, graine sur les 4 derniers tokens
    "wm_unigram":   dict(watermark=dict(kind="unigram", gamma=0.5, delta=2.0)),  # liste verte fixe (Zhao et al.)
    "wm_gumbel":    dict(watermark=dict(kind="gumbel", context=4)),            # Aaronson (échantillonnage de Gumbel)
    "int8":         dict(quant="int8"),
    "nf4":          dict(quant="nf4"),
    "int8_sys":     dict(quant="int8", system=SYS_LONG),
    "loraA_0.3":    dict(lora=("A", 0.3)),
    "loraA_0.7":    dict(lora=("A", 0.7)),
    "loraA_1.0":    dict(lora=("A", 1.0)),
    "loraA_1.5":    dict(lora=("A", 1.5)),
    "loraB_0.7":    dict(lora=("B", 0.7)),
    "loraA_0.7_wm": dict(lora=("A", 0.7), watermark=True),
}


def model_label(base, v):
    """Identité du modèle : base + quantification + adapter (et force). Le reste ne compte pas."""
    cfg = VARIANTS[v]
    lab = base
    if cfg.get("quant"):
        lab += "-" + cfg["quant"]
    if cfg.get("lora"):
        lab += f"-lora{cfg['lora'][0]}@{cfg['lora'][1]}"
    return lab
