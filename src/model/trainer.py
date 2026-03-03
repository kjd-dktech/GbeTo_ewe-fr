"""
Pipeline de fine-tuning du modèle facebook/nllb-200-distilled-600M
pour la traduction bidirectionnelle éwé ↔ français.

Modèle   : facebook/nllb-200-distilled-600M
Tâche    : Seq2Seq — traduction automatique neuronale
Langues  : éwé (ewe_Latn) ↔ français (fra_Latn)
Données  : data/processed/train.csv, val.csv

Stratégie d'entraînement :
    - Fine-tuning supervisé complet (tous les paramètres)
    - Mixed precision fp16 pour réduire la VRAM (~40%)
    - Gradient checkpointing pour réduire la VRAM supplémentaire
    - Gradient accumulation pour simuler un grand batch
    - Warmup + décroissance linéaire du learning rate
    - Early stopping sur le BLEU de validation (patience=3)
    - Sauvegarde du meilleur checkpoint uniquement

Utilisation :
    python -m src.model.trainer [--train_file ...] [--val_file ...]

Auteur : Kodjo Jean DEGBEVI
"""

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.evaluate.metrics import compute_bleu, compute_chrf

# Charge les variables d'environnement (.env)
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
MODEL_NAME   = "facebook/nllb-200-distilled-600M"
SRC_LANG_EWE = "ewe_Latn"
TGT_LANG_FRA = "fra_Latn"
SRC_LANG_FRA = "fra_Latn"
TGT_LANG_EWE = "ewe_Latn"

SEED         = 42
MAX_LENGTH   = 128   # Longueur max en tokens SentencePiece (source et cible)


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = SEED) -> None:
    """
    Fixe les seeds de toutes les sources d'aléatoire.

    Sources couvertes : random, numpy, torch (CPU + GPU), transformers.
    Source résiduelle non contrôlable : opérations atomiques CUDA
    (atomicAdd dans certains kernels) — documentée dans le cahier de projet.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace Transformers

    # Déterminisme cuDNN — légère perte de vitesse, gains reproductibilité
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    logger.info(
        f"Seeds fixés — random: {seed}, numpy: {seed}, "
        f"torch: {seed}, transformers: {seed}"
    )


# ---------------------------------------------------------------------------
# Configuration de l'entraînement
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Centralise tous les hyperparamètres de l'entraînement.
    Chaque valeur est justifiée dans le commentaire associé.
    """
    # --- Données ---
    train_file:  Path = field(default=Path("data/processed/train.csv"))
    val_file:    Path = field(default=Path("data/processed/val.csv"))
    output_dir:  Path = field(default=Path("outputs/checkpoints"))

    # --- Modèle ---
    model_name:  str  = field(default=MODEL_NAME)

    # --- Entraînement ---
    # lr=5e-5 : standard pour fine-tuning de transformer pré-entraîné.
    # Trop grand → catastrophic forgetting. Trop petit → convergence lente.
    learning_rate:                float = field(default=5e-5)

    # batch_size=8 : limite imposée par la VRAM T4 (16GB) avec fp16
    per_device_train_batch_size:  int   = field(default=8)
    per_device_eval_batch_size:   int   = field(default=8)

    # gradient_accumulation=4 : batch effectif = 8×4 = 32
    # Simule un grand batch sans surcharger la VRAM
    gradient_accumulation_steps:  int   = field(default=4)

    num_train_epochs:             int   = field(default=10)

    # warmup_steps=500 : stabilise le début de l'entraînement.
    # Les grands gradients initiaux peuvent déstabiliser un modèle pré-entraîné.
    warmup_steps:                 int   = field(default=500)

    # weight_decay=0.01 : régularisation L2, décourage les poids très grands
    weight_decay:                 float = field(default=0.01)

    # fp16 : réduit la VRAM de ~40%, accélère sur T4/A100
    fp16:                         bool  = field(default=True)

    # gradient_checkpointing : recompute les activations au lieu de les stocker
    # Réduit la VRAM au prix d'un léger surcoût computationnel (~20%)
    gradient_checkpointing:       bool  = field(default=True)

    # max_grad_norm=1.0 : gradient clipping, essentiel pour la stabilité
    max_grad_norm:                float = field(default=1.0)

    # --- Évaluation et sauvegarde ---
    # eval_steps : évaluer toutes les 200 steps (pas seulement en fin d'epoch)
    # pour un suivi plus fin sur un petit corpus
    eval_steps:                   int   = field(default=200)
    save_steps:                   int   = field(default=200)
    logging_steps:                int   = field(default=50)

    # load_best_model_at_end : obligatoire pour early stopping
    load_best_model_at_end:       bool  = field(default=True)

    # metric_for_best_model : on optimise sur le BLEU de validation
    metric_for_best_model:        str   = field(default="bleu")
    greater_is_better:            bool  = field(default=True)

    # save_total_limit : ne garder que les 2 meilleurs checkpoints
    save_total_limit:             int   = field(default=2)

    # --- Early stopping ---
    # patience=3 : arrêter si le BLEU de validation ne s'améliore pas
    # pendant 3 évaluations consécutives
    early_stopping_patience:      int   = field(default=3)

    # --- Génération ---
    # num_beams=4 : beam search, standard en traduction automatique
    num_beams:                    int   = field(default=4)
    max_length:                   int   = field(default=MAX_LENGTH)

    # --- Reproductibilité ---
    seed:                         int   = field(default=SEED)

    # --- Suivi des expériences ---
    # wandb : tracking des métriques et hyperparamètres
    report_to:                    str   = field(default="wandb")
    run_name:                     str   = field(
        default="nllb-600m-ewe-fra-finetune"
    )
    wandb_project: str = field(default="GbeTo : Traducteur Ewe ↔ Français")


# ---------------------------------------------------------------------------
# Chargement et tokenisation des données
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """
    Charge un fichier CSV de paires de traduction.

    Args:
        path : Chemin vers le fichier CSV (colonnes: source, target, direction, origin)

    Returns:
        DataFrame chargé

    Raises:
        FileNotFoundError : Si le fichier n'existe pas
        ValueError        : Si les colonnes requises sont manquantes
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {path}\n"
            f"Lancez d'abord le pipeline de données :\n"
            f"  python -m src.data.download\n"
            f"  python -m src.data.filter\n"
            f"  python -m src.data.split"
        )

    df = pd.read_csv(path, encoding="utf-8")

    required_cols = {"source", "target", "direction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path} : {missing}"
        )

    logger.info(f"  Données chargées : {path}  ({len(df):,} paires)")
    dir_counts = df["direction"].value_counts()
    for direction, count in sorted(dir_counts.items()):
        logger.info(f"    {direction} : {count:,}")

    return df


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """
    Tokenise un DataFrame de paires de traduction en Dataset HuggingFace.

    Gère le caractère bidirectionnel du dataset : chaque paire a une
    direction (ewe-fra ou fra-ewe), ce qui détermine les tokens de langue
    source et cible utilisés par le tokenizer NLLB.

    Strategy de tokenisation :
        - Source tokenisée avec src_lang = langue source de la paire
        - Cible tokenisée avec forced_bos_token_id = token de la langue cible
        - Labels avec padding_id = -100 (ignorés dans le calcul de la loss)

    Args:
        df        : DataFrame avec colonnes [source, target, direction]
        tokenizer : Tokenizer NLLB chargé
        max_length: Longueur maximale en tokens (troncature si dépassé)

    Returns:
        Dataset HuggingFace tokenisé
    """
    # Mapping direction → (src_lang, tgt_lang)
    lang_map = {
        "ewe-fra": (SRC_LANG_EWE, TGT_LANG_FRA),
        "fra-ewe": (SRC_LANG_FRA, TGT_LANG_EWE),
    }

    def tokenize_batch(batch: dict) -> dict:
        """Tokenise un batch de paires."""
        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []

        for source, target, direction in zip(
            batch["source"], batch["target"], batch["direction"]
        ):
            src_lang, tgt_lang = lang_map.get(
                direction, (SRC_LANG_EWE, TGT_LANG_FRA)
            )

            # Tokenisation de la source
            tokenizer.src_lang = src_lang
            model_inputs = tokenizer(
                source,
                max_length=max_length,
                truncation=True,
                padding=False,  # Le DataCollator gère le padding dynamique
            )

            # Tokenisation de la cible
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    target,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                )

            input_ids_list.append(model_inputs["input_ids"])
            attention_mask_list.append(model_inputs["attention_mask"])
            labels_list.append(labels["input_ids"])

        return {
            "input_ids":      input_ids_list,
            "attention_mask": attention_mask_list,
            "labels":         labels_list,
        }

    # Conversion DataFrame → Dataset HuggingFace
    dataset = Dataset.from_pandas(df[["source", "target", "direction"]])

    # Tokenisation par batch (plus efficace que ligne par ligne)
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=256,
        remove_columns=dataset.column_names,
        desc="Tokenisation",
    )

    return tokenized


# ---------------------------------------------------------------------------
# Calcul des métriques pour le Trainer
# ---------------------------------------------------------------------------

def build_compute_metrics(tokenizer: AutoTokenizer):
    """
    Construit la fonction compute_metrics compatible avec Seq2SeqTrainer.

    Le Trainer appelle cette fonction à chaque évaluation avec
    les prédictions et les labels bruts (IDs de tokens).
    Cette fonction les décode en texte et calcule BLEU + chrF.

    Args:
        tokenizer : Tokenizer pour le décodage des IDs

    Returns:
        Fonction compute_metrics(eval_preds) → dict
    """
    def compute_metrics(eval_preds) -> dict:
        predictions, labels = eval_preds

        # Les prédictions peuvent être un tuple (logits, ...) selon la config
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Remplacement des -100 (padding des labels) par pad_token_id
        # pour éviter les erreurs lors du décodage
        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id,
        )

        # Décodage des IDs → texte
        # skip_special_tokens=True supprime les tokens spéciaux (BOS, EOS, PAD)
        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        # Nettoyage minimal : suppression des espaces superflus
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Calcul des métriques
        bleu = compute_bleu(decoded_preds, decoded_labels)
        chrf = compute_chrf(decoded_preds, decoded_labels)

        return {
            "bleu": bleu,
            "chrf": chrf,
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Pipeline d'entraînement principal
# ---------------------------------------------------------------------------

def train(config: TrainingConfig) -> None:
    """
    Orchestre le fine-tuning complet de NLLB-200-distilled-600M.

    Reprise automatique : si des checkpoints existent dans output_dir,
    l'entraînement reprend depuis le dernier état sauvegardé
    (poids, optimizer, scheduler, step courant). Sinon, il repart
    depuis le début. Aucune intervention manuelle nécessaire.

    Étapes :
        1. Reproductibilité — seeds
        2. Détection du dernier checkpoint (nouveau départ ou reprise)
        3. Chargement du modèle et tokenizer
        4. Chargement et tokenisation des données
        5. Data Collator
        6. Arguments d'entraînement
        7. Trainer + early stopping
        8. Entraînement
        9. Sauvegarde du meilleur modèle
        10. Évaluation finale + sauvegarde des métriques

    Args:
        config : Hyperparamètres et chemins de l'entraînement
    """
    # ------------------------------------------------------------------
    # 1. Reproductibilité
    # ------------------------------------------------------------------
    set_all_seeds(config.seed)

    logger.info("=" * 65)
    logger.info("FINE-TUNING NLLB-200-distilled-600M — éwé ↔ français")
    logger.info("=" * 65)
    logger.info(f"  Modèle          : {config.model_name}")
    logger.info(f"  Learning rate   : {config.learning_rate}")
    logger.info(f"  Batch effectif  : "
                f"{config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Epochs max      : {config.num_train_epochs}")
    logger.info(f"  Early stopping  : patience={config.early_stopping_patience}")
    logger.info(f"  fp16            : {config.fp16}")
    logger.info(f"  Seed            : {config.seed}")

    # ------------------------------------------------------------------
    # 2. Détection du dernier checkpoint
    #
    # Logique :
    #   - output_dir existe ET contient des checkpoints → REPRISE
    #   - output_dir vide ou inexistant               → NOUVEAU DÉPART
    #
    # get_last_checkpoint() retourne None si aucun checkpoint n'est
    # trouvé, ce qui déclenche automatiquement un nouveau départ.
    # Le Trainer recharge : poids, optimizer, scheduler, step courant.
    # ------------------------------------------------------------------
    last_checkpoint: Optional[str] = None

    if config.output_dir.exists():
        last_checkpoint = get_last_checkpoint(str(config.output_dir))

    if last_checkpoint is not None:
        logger.info(f"\n>>> REPRISE détectée : {last_checkpoint}")
        logger.info("  L'entraînement reprend depuis l'état sauvegardé.")
        logger.info("  (poids, optimizer, scheduler, step courant restaurés)")
    else:
        logger.info("\n>>> NOUVEAU DÉPART — aucun checkpoint trouvé.")

    # ------------------------------------------------------------------
    # 3. Chargement du modèle et tokenizer
    # ------------------------------------------------------------------
    logger.info(f"\n>>> Chargement du modèle : {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_auth_token=os.environ.get("HF_TOKEN"),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        use_auth_token=os.environ.get("HF_TOKEN"),
    )

    # Gradient checkpointing : doit être activé avant la mise sur GPU
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Affichage du nombre de paramètres
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Paramètres total     : {num_params:,}")
    logger.info(f"  Paramètres entraîn.  : {num_trainable:,}")

    # ------------------------------------------------------------------
    # 4. Chargement et tokenisation des données
    # ------------------------------------------------------------------
    logger.info("\n>>> Chargement des données ...")
    df_train = load_data(config.train_file)
    df_val   = load_data(config.val_file)

    logger.info("\n>>> Tokenisation ...")
    train_dataset = tokenize_dataset(df_train, tokenizer, config.max_length)
    val_dataset   = tokenize_dataset(df_val,   tokenizer, config.max_length)

    logger.info(f"  Train tokenisé : {len(train_dataset):,} exemples")
    logger.info(f"  Val tokenisé   : {len(val_dataset):,} exemples")

    # ------------------------------------------------------------------
    # 4. Data Collator
    # ------------------------------------------------------------------
    # DataCollatorForSeq2Seq gère le padding dynamique :
    # chaque batch est padé à la longueur du plus long exemple du batch
    # (et non à MAX_LENGTH fixe), ce qui économise de la mémoire
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,  # Optimisation pour les GPU avec Tensor Cores
        label_pad_token_id=-100,  # -100 est ignoré dans le calcul de la loss
    )

    # ------------------------------------------------------------------
    # 5. Arguments d'entraînement
    # ------------------------------------------------------------------
    config.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_PROJECT"] = config.wandb_project

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),

        # Entraînement
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,

        # Évaluation
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,

        # Génération (pour compute_metrics en évaluation)
        predict_with_generate=True,
        generation_num_beams=config.num_beams,
        generation_max_length=config.max_length,

        # Meilleur modèle
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=config.save_total_limit,

        # Reproductibilité
        seed=config.seed,
        data_seed=config.seed,

        # Logging
        report_to=config.report_to,
        run_name=config.run_name,
        logging_dir=str(Path("outputs/logs")),
    )

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
            )
        ],
    )

    # ------------------------------------------------------------------
    # 8. Entraînement
    # ------------------------------------------------------------------
    logger.info("\n>>> Début de l'entraînement ...")
    logger.info(
        f"  Steps par epoch   : "
        f"{len(train_dataset) // (config.per_device_train_batch_size * config.gradient_accumulation_steps)}"
    )

    if last_checkpoint is not None:
        logger.info(f"  Reprise depuis    : {last_checkpoint}")
    else:
        logger.info("  Départ depuis le début (epoch 0, step 0)")

    # resume_from_checkpoint=None  → nouveau départ
    # resume_from_checkpoint=<path> → reprise depuis ce checkpoint
    # Le Trainer restaure automatiquement : poids du modèle, état de
    # l'optimizer, état du scheduler LR, step global, epoch courante.
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log des métriques d'entraînement
    logger.info("\n>>> Entraînement terminé.")
    logger.info(f"  Runtime         : {train_result.metrics.get('train_runtime', 0):.1f}s")
    logger.info(f"  Steps/seconde   : {train_result.metrics.get('train_steps_per_second', 0):.3f}")
    logger.info(f"  Train loss      : {train_result.metrics.get('train_loss', 0):.4f}")

    # ------------------------------------------------------------------
    # 8. Sauvegarde du meilleur modèle
    # ------------------------------------------------------------------
    best_model_dir = config.output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info(f"\n  Meilleur modèle sauvegardé : {best_model_dir}")

    # ------------------------------------------------------------------
    # 9. Évaluation finale sur le meilleur checkpoint
    # ------------------------------------------------------------------
    logger.info("\n>>> Évaluation finale sur le jeu de validation ...")
    eval_results = trainer.evaluate()

    final_bleu = eval_results.get("eval_bleu", 0.0)
    final_chrf = eval_results.get("eval_chrf", 0.0)
    final_loss = eval_results.get("eval_loss", 0.0)

    logger.info(f"\n{'='*65}")
    logger.info("RÉSULTATS FINAUX")
    logger.info(f"{'='*65}")
    logger.info(f"  BLEU (val)  : {final_bleu:.2f}")
    logger.info(f"  chrF (val)  : {final_chrf:.2f}")
    logger.info(f"  Loss (val)  : {final_loss:.4f}")
    logger.info(f"{'='*65}")

    # ------------------------------------------------------------------
    # 10. Sauvegarde des métriques finales
    # ------------------------------------------------------------------
    import json
    metrics_path = Path("outputs/final_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    final_metrics = {
        "model":          config.model_name,
        "bleu_val":       final_bleu,
        "chrf_val":       final_chrf,
        "loss_val":       final_loss,
        "train_loss":     train_result.metrics.get("train_loss", 0.0),
        "best_checkpoint": str(best_model_dir),
        "hyperparameters": {
            "learning_rate":                config.learning_rate,
            "per_device_train_batch_size":  config.per_device_train_batch_size,
            "gradient_accumulation_steps":  config.gradient_accumulation_steps,
            "effective_batch_size":         config.per_device_train_batch_size * config.gradient_accumulation_steps,
            "num_train_epochs":             config.num_train_epochs,
            "warmup_steps":                 config.warmup_steps,
            "weight_decay":                 config.weight_decay,
            "max_length":                   config.max_length,
            "num_beams":                    config.num_beams,
            "seed":                         config.seed,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"\n  Métriques sauvegardées : {metrics_path}")
    logger.info("\nEntraînement complet. Étape suivante : app.py (déploiement Gradio)")


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning NLLB-200 pour la traduction éwé-français.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_file",   type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val_file",     type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--output_dir",   type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--model_name",   type=str,  default=MODEL_NAME)
    parser.add_argument("--learning_rate",type=float,default=5e-5)
    parser.add_argument("--batch_size",   type=int,  default=8)
    parser.add_argument("--grad_accum",   type=int,  default=4)
    parser.add_argument("--epochs",       type=int,  default=10)
    parser.add_argument("--warmup_steps", type=int,  default=500)
    parser.add_argument("--num_beams",    type=int,  default=4)
    parser.add_argument("--seed",         type=int,  default=SEED)
    parser.add_argument("--no_fp16",      action="store_true",
                        help="Désactiver fp16 (si GPU ne supporte pas)")
    parser.add_argument("--report_to",    type=str,  default="wandb",
                        choices=["wandb", "tensorboard", "none"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainingConfig(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        num_beams=args.num_beams,
        seed=args.seed,
        fp16=not args.no_fp16,
        report_to=args.report_to,
    )

    train(config)


if __name__ == "__main__":
    main()