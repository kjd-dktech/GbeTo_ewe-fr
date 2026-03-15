"""
Interface de démonstration Gradio pour le traducteur éwé ↔ français.

Déploiement : HuggingFace Spaces
Modèle      : GbeTo_ewe-fr

Auteur : Kodjo Jean DEGBEVI
"""

import logging
import os
from pathlib import Path

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration du logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# Chemin du modèle fine-tuné local
LOCAL_MODEL_PATH = Path("outputs/final_model")

# Id hugging face de GbeTo
HF_MODEL_ID = "kjd-dktech/gbeto-ewe-french"

# Modèle baseline HuggingFace
BASELINE_MODEL = "facebook/nllb-200-distilled-600M"

# Tokens de langue NLLB
LANG_TOKENS = {
    "Éwé → Français": ("ewe_Latn", "fra_Latn"),
    "Français → Éwé": ("fra_Latn", "ewe_Latn"),
}

# Paramètres de génération
GENERATION_CONFIG = {
    "max_length":        256,
    "num_beams":         4,
    "length_penalty":    1.0,
    "no_repeat_ngram_size": 3,
    "early_stopping":    True,
}

# Exemples prêts à l'emploi pour la démonstration
EXAMPLES = [
    ["Le marché de Lomé est très animé le matin.", "Français → Éwé"],
    ["La santé est un droit fondamental pour tous les êtres humains.", "Français → Éwé"],
    ["L'éducation des enfants est la priorité de notre communauté.", "Français → Éwé"],
    ["Ɖevi ƒe ŋkɔ nye Kofi, eye eƒe ƒome le Lomé.", "Éwé → Français"],
    ["Míebiaa aɖe ŋu aɖaŋu eye míebiaa wo mɔ wòwo.", "Éwé → Français"],
]


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    """
    Charge le modèle et le tokenizer.

    Priorité :
        1. Modèle fine-tuné local (outputs/final_model)
        2. Modèle fine-tuned de HuggingFace

    Returns:
        Tuple (tokenizer, model, model_label)
        model_label : description du modèle chargé (pour l'interface)
    """
    hf_token = os.environ.get("HF_TOKEN_READ")

    # --- Tentative de chargement du modèle fine-tuné local ---
    if LOCAL_MODEL_PATH.exists():
        logger.info(f"Chargement du modèle fine-tuné : {LOCAL_MODEL_PATH}")
        model_path  = str(LOCAL_MODEL_PATH)
        model_label = "NLLB-600M fine-tuné éwé-français"
        is_finetuned = True
    elif HF_MODEL_ID:
        logger.info(
            f"Modèle fine-tuné local {LOCAL_MODEL_PATH} introuvable !\n"
            f"Vérification de la disponibilité du modèle HF : {HF_MODEL_ID}"
        )
        try:
            from huggingface_hub import model_info
            model_info(HF_MODEL_ID, token=hf_token)
            logger.info(f"Modèle HF trouvé : {HF_MODEL_ID}")
            model_path   = HF_MODEL_ID
            model_label  = "GbeTo_ewe-fr"
            is_finetuned = True
        except Exception:
            logger.warning(
                f"Modèle HF {HF_MODEL_ID} introuvable !\n"
                f"Chargement du base model : {BASELINE_MODEL}"
            )
            model_path   = BASELINE_MODEL
            model_label  = "NLLB-200"
            is_finetuned = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        token=hf_token,
    )

    # Mise sur GPU si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    logger.info(f"Modèle chargé sur : {device.upper()}")
    logger.info(f"Label             : {model_label}")

    return tokenizer, model, model_label, device, is_finetuned


# ---------------------------------------------------------------------------
# Fonction de traduction
# ---------------------------------------------------------------------------

def translate(
    text: str,
    direction: str,
    tokenizer: AutoTokenizer,
    model:     AutoModelForSeq2SeqLM,
    device:    str,
) -> str:
    """
    Traduit un texte dans la direction spécifiée.

    Args:
        text      : Texte à traduire
        direction : 'Éwé → Français' ou 'Français → Éwé'
        tokenizer : Tokenizer NLLB
        model     : Modèle NLLB
        device    : 'cuda' ou 'cpu'

    Returns:
        Traduction générée ou message d'erreur
    """
    text = text.strip()

    if not text:
        return "⚠️ Veuillez entrer un texte à traduire."

    src_lang, tgt_lang = LANG_TOKENS[direction]

    try:
        # Tokenisation de la source
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True,
        ).to(device)

        # Token forcé pour la langue cible (obligatoire avec NLLB)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        # Génération avec beam search
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                **GENERATION_CONFIG,
            )

        # Décodage
        translation = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        return translation

    except Exception as e:
        logger.error(f"Erreur de traduction : {e}")
        return f"⚠️ Erreur lors de la traduction : {str(e)}"


# ---------------------------------------------------------------------------
# Construction de l'interface Gradio
# ---------------------------------------------------------------------------

def build_interface(
    tokenizer:   AutoTokenizer,
    model:       AutoModelForSeq2SeqLM,
    model_label: str,
    device:      str,
    is_finetuned: bool,
) -> gr.Blocks:
    """
    Construit et retourne l'interface Gradio.

    Args:
        tokenizer    : Tokenizer NLLB
        model        : Modèle NLLB
        model_label  : Description du modèle chargé
        device       : Dispositif de calcul
        is_finetuned : True si modèle fine-tuné, False si baseline

    Returns:
        Instance gr.Blocks configurée
    """
    # Badge de statut du modèle
    model_status = (
        "✅ Modèle fine-tuné sur données éwé-français (AfroLingu-MT + MAFAND)"
        if is_finetuned else
        "⚠️ Modèle baseline NLLB sans fine-tuning — performances limitées"
    )

    with gr.Blocks(
        title="Traducteur Éwé ↔ Français"
    ) as interface:

        # --- En-tête ---
        gr.Markdown(
            """
            # 🌍 GbeTo : Traducteur Éwé ↔ Français
            Traduction automatique neuronale basée sur
            [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
            (Meta AI, 2022), fine-tuné sur des corpus parallèles éwé-français
            de qualité académique.

            **Langue éwé** : parlée par ~3-4 millions de personnes au Togo,
            au Ghana et au Bénin. Classée parmi les langues à très faibles
            ressources en NLP (Joshi et al., 2020).
            """
        )

        # --- Statut du modèle ---
        gr.Markdown(f"**Modèle actif** : {model_label}  \n{model_status}")

        # --- Zone principale ---
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Texte source",
                    placeholder="Entrez le texte à traduire...",
                    lines=6,
                    max_lines=12,
                )
                direction = gr.Radio(
                    choices=list(LANG_TOKENS.keys()),
                    value="Français → Éwé",
                    label="Direction de traduction",
                )
                translate_btn = gr.Button(
                    "Traduire",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Traduction",
                    lines=6,
                    max_lines=12,
                    interactive=False,
                )
                gr.Markdown(
                    """
                    **Note** : Les traductions sont générées automatiquement
                    et peuvent contenir des erreurs, notamment sur les
                    termes techniques ou les expressions idiomatiques.
                    L'éwé possède plusieurs dialectes (Anlo, Ho, Kpando)
                    dont les variations orthographiques peuvent affecter
                    la qualité de la traduction.
                    """
                )

        # --- Bouton de traduction ---
        translate_btn.click(
            fn=lambda text, dir: translate(text, dir, tokenizer, model, device),
            inputs=[input_text, direction],
            outputs=output_text,
        )

        # Traduction également déclenchée par Entrée + Shift
        input_text.submit(
            fn=lambda text, dir: translate(text, dir, tokenizer, model, device),
            inputs=[input_text, direction],
            outputs=output_text,
        )

        # --- Exemples ---
        gr.Markdown("### Exemples")
        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_text, direction],
            outputs=output_text,
            fn=lambda text, dir: translate(text, dir, tokenizer, model, device),
            cache_examples=False,
        )

        # --- Pied de page ---
        gr.Markdown(
            """
            ---
            **Auteur** : Kodjo Jean DEGBEVI
            **Modèle** : `GbeTo_ewe-fr`
            **Données** : AfroLingu-MT (ACL 2024) + MAFAND (NAACL 2022)
            **Code** : [github.com/kjd-dktech/GbeTo_ewe-fr](https://github.com/kjd-dktech/GbeTo_ewe-fr)
            """
        )

    return interface


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 55)
    logger.info("Démarrage — Traducteur Éwé ↔ Français")
    logger.info("=" * 55)

    # Chargement du modèle
    tokenizer, model, model_label, device, is_finetuned = load_model()

    # Construction de l'interface
    interface = build_interface(
        tokenizer=tokenizer,
        model=model,
        model_label=model_label,
        device=device,
        is_finetuned=is_finetuned,
    )

    # Lancement
    interface.launch(
        server_name="0.0.0.0",
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
