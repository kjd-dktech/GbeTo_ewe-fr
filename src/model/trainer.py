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
    - Sauvegarde par epoch avec validation d'intégrité
    - Reprise automatique depuis le dernier checkpoint valide

Utilisation :
    python -m src.model.trainer [--train_file ...] [--val_file ...]

Auteur : Kodjo Jean DEGBEVI
"""

import argparse
import json
import logging
import os
import random
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
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

SEED       = 42
MAX_LENGTH = 128  # Longueur max en tokens SentencePiece (source et cible)


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
    set_seed(seed)

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
    train_file: Path = field(default=Path("data/processed/train.csv"))
    val_file:   Path = field(default=Path("data/processed/val.csv"))

    # --- Stockage ---
    # Le Trainer écrit UNIQUEMENT en local Colab (rapide, pas de quota Drive)
    output_dir:           str = field(default="/content/checkpoints_local")
    # Dossier Drive : contient uniquement le meilleur checkpoint (zip) + registry.json
    drive_checkpoint_dir: str = field(default="outputs/checkpoints")
    # Dossier final Drive — modèle complet sauvegardé en fin de run
    final_dir:            str = field(default="outputs/final_model")

    # --- Stratégie de sauvegarde Drive ---
    # Nombre d'epochs sans aucune écriture Drive (laisser le modèle converger d'abord)
    warmup_epochs: int = field(default=3)
    # Nombre max de checkpoints à conserver en local (les meilleurs en BLEU)
    keep_local:    int = field(default=2)

    hf_repo_id:  str  = field(default="kjd-dktech/gbeto-ewe-french")
    push_to_hub: bool = field(default=True)

    # --- Modèle ---
    model_name: str = field(default=MODEL_NAME)

    # --- Entraînement ---
    # lr=5e-5 : standard pour fine-tuning de transformer pré-entraîné.
    # Trop grand → catastrophic forgetting. Trop petit → convergence lente.
    learning_rate: float = field(default=5e-5)

    # batch_size=8 : limite imposée par la VRAM T4 (16GB) avec fp16
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size:  int = field(default=4)

    # gradient_accumulation=4 : batch effectif = 8×4 = 32
    # Simule un grand batch sans surcharger la VRAM
    gradient_accumulation_steps: int = field(default=8)

    num_train_epochs: int = field(default=10)

    # warmup_steps=500 : stabilise le début de l'entraînement.
    # Les grands gradients initiaux peuvent déstabiliser un modèle pré-entraîné.
    warmup_steps: int = field(default=500)

    # weight_decay=0.01 : régularisation L2, décourage les poids très grands
    weight_decay: float = field(default=0.01)

    # fp16 : réduit la VRAM de ~40%, accélère sur T4/A100
    fp16: bool = field(default=True)

    # gradient_checkpointing : recompute les activations au lieu de les stocker
    # Réduit la VRAM au prix d'un léger surcoût computationnel (~20%)
    gradient_checkpointing: bool = field(default=True)

    # max_grad_norm=1.0 : gradient clipping, essentiel pour la stabilité
    max_grad_norm: float = field(default=1.0)

    # --- Évaluation et sauvegarde ---
    logging_steps: int = field(default=50)

    # load_best_model_at_end : obligatoire pour early stopping
    load_best_model_at_end: bool = field(default=True)

    # metric_for_best_model : on optimise sur le BLEU de validation
    metric_for_best_model: str  = field(default="bleu")
    greater_is_better:     bool = field(default=True)

    # save_total_limit : avec load_best_model_at_end=True, le Trainer conserve
    # save_total_limit : None — la rotation locale est gérée par DriveCheckpointCallback
    # qui garde uniquement les keep_local meilleurs checkpoints (par BLEU, pas par ancienneté)
    save_total_limit: Optional[int] = field(default=None)

    # --- Early stopping ---
    # patience=3 : arrêter si le BLEU de validation ne s'améliore pas
    # pendant 3 évaluations consécutives (3 epochs)
    early_stopping_patience: int = field(default=3)

    # --- Génération ---
    # num_beams=4 : beam search, standard en traduction automatique
    num_beams:  int = field(default=4)
    max_length: int = field(default=MAX_LENGTH)

    # --- Reproductibilité ---
    seed: int = field(default=SEED)

    # --- Suivi des expériences ---
    report_to:     str = field(default="wandb")
    run_name:      str = field(default="nllb-600m-ewe-fra-finetune")
    wandb_project: str = field(default="GbeTo_1")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """
    Charge un fichier CSV de paires de traduction.

    Args:
        path : Chemin vers le fichier CSV
                (colonnes: source, target, direction, origin)

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


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

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

    Args:
        df        : DataFrame avec colonnes [source, target, direction]
        tokenizer : Tokenizer NLLB chargé
        max_length: Longueur maximale en tokens (troncature si dépassé)

    Returns:
        Dataset HuggingFace tokenisé
    """
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

            tokenizer.src_lang = src_lang

            # text_target : méthode officielle post-dépréciation
            # de as_target_tokenizer (supprimé dans transformers 5.0)
            model_inputs = tokenizer(
                text=source,
                text_target=target,
                max_length=max_length,
                truncation=True,
                padding=False,
            )

            input_ids_list.append(model_inputs["input_ids"])
            attention_mask_list.append(model_inputs["attention_mask"])
            labels_list.append(model_inputs["labels"])

        return {
            "input_ids":      input_ids_list,
            "attention_mask": attention_mask_list,
            "labels":         labels_list,
        }

    dataset = Dataset.from_pandas(df[["source", "target", "direction"]])

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=256,
        remove_columns=dataset.column_names,
        desc="Tokenisation",
    )

    return tokenized


# ---------------------------------------------------------------------------
# Métriques d'évaluation
# ---------------------------------------------------------------------------

def build_compute_metrics(tokenizer: AutoTokenizer):
    """
    Construit la fonction compute_metrics compatible avec Seq2SeqTrainer.

    Args:
        tokenizer : Tokenizer pour le décodage des IDs

    Returns:
        Fonction compute_metrics(eval_preds) → dict
    """
    def compute_metrics(eval_preds) -> dict:
        predictions, labels = eval_preds

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Remplacer -100 (padding labels) par pad_token_id pour le décodage
        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id,
        )

        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        decoded_preds  = [predict.strip() for predict in decoded_preds]
        decoded_labels = [lab.strip() for lab in decoded_labels]

        bleu = compute_bleu(decoded_preds, decoded_labels)
        chrf = compute_chrf(decoded_preds, decoded_labels)

        return {
            "bleu": bleu,
            "chrf": chrf,
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Trainer sécurisé — validation d'intégrité des checkpoints
# ---------------------------------------------------------------------------

class GbeToTrainer(Seq2SeqTrainer):
    """
    Sous-classe de Seq2SeqTrainer qui sécurise la sauvegarde des checkpoints.

    Comportement ajouté :
        - Après chaque sauvegarde, vérifie que trainer_state.json est présent
          et valide (JSON parseable)
        - Si absent ou corrompu → supprime le checkpoint incomplet
        - Évite qu'un crash partiel laisse un checkpoint invalide
          qui bloquerait la reprise automatique
    """

    def _save_checkpoint(self, model, trial):
        """Sauvegarde avec validation post-écriture."""

        # Sauvegarde standard du Trainer parent
        super()._save_checkpoint(model, trial)

        # Chemin du checkpoint qui vient d'être écrit
        checkpoint_dir = (
            Path(self.args.output_dir)
            / f"checkpoint-{self.state.global_step}"
        )

        if not checkpoint_dir.exists():
            logger.warning(
                f"Checkpoint attendu introuvable après sauvegarde : "
                f"{checkpoint_dir}"
            )
            return

        # Vérifier trainer_state.json
        state_file = checkpoint_dir / "trainer_state.json"
        if not state_file.exists():
            logger.error(
                f"trainer_state.json absent : {checkpoint_dir}\n"
                f"  → Checkpoint incomplet."
            )
            return

        try:
            with open(state_file) as f:
                json.load(f)
        except json.JSONDecodeError:
            logger.error(
                f"trainer_state.json corrompu : {checkpoint_dir}\n"
                f"  → Checkpoint non valide."
            )
            return

        # Vérifier les poids du modèle
        safetensors = checkpoint_dir / "model.safetensors"
        pytorch_bin = checkpoint_dir / "pytorch_model.bin"

        if safetensors.exists():
            model_file = safetensors
        elif pytorch_bin.exists():
            model_file = pytorch_bin
        else:
            logger.error(
                f"Poids du modèle absents (model.safetensors / pytorch_model.bin) : "
                f"{checkpoint_dir}\n"
                f"  → Checkpoint incomplet."
            )
            return

        if model_file.stat().st_size < 1_000_000:
            logger.error(
                f"Poids du modèle trop petits "
                f"({model_file.stat().st_size} bytes) : {checkpoint_dir}\n"
                f"  → Checkpoint corrompu."
            )
            return

        logger.info(f"  Checkpoint validé : {checkpoint_dir.name}")


# ---------------------------------------------------------------------------
# Callback de gestion Drive — sauvegarde sélective avec zip
# ---------------------------------------------------------------------------

REGISTRY_LOCAL  = Path("/content/drive_registry.json")
REGISTRY_REMOTE = "registry.json"   # nom dans drive_checkpoint_dir


def _read_registry(path: Path) -> Optional[dict]:
    """Lit le registre JSON local. Retourne None si absent ou invalide."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data and "bleu" in data:
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _write_registry(path: Path, bleu: float, filename: str,
                    epoch: float, step: int) -> None:
    """Écrit le registre JSON à l'emplacement donné."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"bleu": bleu, "filename": filename,
             "epoch": epoch, "step": step},
            f, indent=2
        )


def _get_checkpoint_bleu(ckpt_dir: Path) -> Optional[float]:
    """Lit le best_metric BLEU depuis trainer_state.json d'un checkpoint local."""
    state_file = ckpt_dir / "trainer_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file) as f:
            state = json.load(f)
        return state.get("best_metric")
    except (json.JSONDecodeError, KeyError):
        return None


def _is_checkpoint_valid(ckpt_dir: Path) -> bool:
    """Vérifie qu'un checkpoint local est complet (poids + state)."""
    if not ckpt_dir.exists():
        return False
    state_file = ckpt_dir / "trainer_state.json"
    if not state_file.exists():
        return False
    try:
        with open(state_file) as f:
            json.load(f)
    except json.JSONDecodeError:
        return False
    for weights in ["model.safetensors", "pytorch_model.bin"]:
        p = ckpt_dir / weights
        if p.exists() and p.stat().st_size > 1_000_000:
            return True
    return False


class DriveCheckpointCallback(TrainerCallback):
    """
    Gère la sauvegarde sélective des checkpoints vers Google Drive.

    Principes :
      - Le Trainer écrit toujours dans output_dir (local Colab, rapide).
      - Ce callback décide seul ce qui va sur Drive, sous forme de zip.
      - Avant warmup_epochs : rien sur Drive, nettoyage local uniquement.
      - Après warmup_epochs : écriture Drive uniquement si BLEU s'améliore.
      - Un seul zip sur Drive à tout moment (le meilleur).
      - Un registre local (/content/drive_registry.json) évite toute
        lecture Drive pour connaître le meilleur BLEU actuel.
      - En local : on garde les keep_local meilleurs checkpoints (par BLEU).
    """

    def __init__(
        self,
        drive_checkpoint_dir: str,
        warmup_epochs: int = 3,
        keep_local: int = 3,
    ):
        self.drive_dir     = Path(drive_checkpoint_dir)
        self.warmup_epochs = warmup_epochs
        self.keep_local    = keep_local

    def _cleanup_local(self, output_dir: Path) -> None:
        """
        Garde uniquement les keep_local meilleurs checkpoints locaux (par BLEU).
        Supprime les autres.
        """
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if len(checkpoints) <= self.keep_local:
            return

        # Associer chaque checkpoint à son BLEU
        scored = []
        for ckpt in checkpoints:
            bleu = _get_checkpoint_bleu(ckpt)
            scored.append((bleu if bleu is not None else -1.0, ckpt))

        # Trier par BLEU croissant → supprimer les moins bons en premier
        scored.sort(key=lambda x: x[0])
        to_delete = scored[:len(scored) - self.keep_local]

        for _, ckpt in to_delete:
            logger.info(f"  [Drive CB] Suppression locale (BLEU faible) : {ckpt.name}")
            shutil.rmtree(ckpt, ignore_errors=True)

    def on_save(
        self,
        args,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        output_dir      = Path(args.output_dir)
        current_step    = state.global_step
        current_bleu = None
        for entry in reversed(state.log_history):
            if "eval_bleu" in entry:
                current_bleu = entry["eval_bleu"]
                break
        current_epoch   = state.epoch
        checkpoint_local = output_dir / f"checkpoint-{current_step}"

        # ── 1. Nettoyage local (toujours, indépendamment du warmup) ────────
        self._cleanup_local(output_dir)

        # ── 2. Warmup : pas d'écriture Drive ───────────────────────────────
        if current_epoch < self.warmup_epochs:
            logger.info(
                f"  [Drive CB] Epoch {current_epoch:.1f} < warmup ({self.warmup_epochs})"
                f" — pas d'écriture Drive."
            )
            return

        # ── 3. Lire le meilleur BLEU connu sur Drive (via registre local) ──
        registry = _read_registry(REGISTRY_LOCAL)
        best_drive_bleu = registry["bleu"] if registry else None

        if current_bleu is None:
            logger.warning("  [Drive CB] best_metric introuvable — on_save ignoré.")
            return

        if best_drive_bleu is not None and current_bleu <= best_drive_bleu:
            logger.info(
                f"  [Drive CB] BLEU actuel ({current_bleu:.4f}) ≤ Drive "
                f"({best_drive_bleu:.4f}) — pas d'écriture Drive."
            )
            return

        if not _is_checkpoint_valid(checkpoint_local):
            logger.warning(
                f"  [Drive CB] Checkpoint local invalide : {checkpoint_local} "
                f"— écriture Drive annulée."
            )
            return

        # ── 4. Zipper le checkpoint local ───────────────────────────────────
        zip_name = f"checkpoint-{current_step}.zip"
        tmp_zip  = Path(f"/tmp/{zip_name}")

        logger.info(
            f"  [Drive CB] Compression : {checkpoint_local.name} → {tmp_zip} ..."
        )
        try:
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in checkpoint_local.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(checkpoint_local))

            # Vérification intégrité du zip
            with zipfile.ZipFile(tmp_zip) as zf:
                bad = zf.testzip()
            if bad:
                raise ValueError(f"Fichier corrompu dans le zip : {bad}")
        except Exception as e:
            logger.error(f"  [Drive CB] Erreur compression : {e} — Drive non mis à jour.")
            tmp_zip.unlink(missing_ok=True)
            return

        # ── 5. Copier le zip vers Drive ──────────────────────────────────────
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        drive_zip = self.drive_dir / zip_name

        logger.info(f"  [Drive CB] Copie vers Drive : {drive_zip} ...")
        try:
            shutil.copy2(str(tmp_zip), str(drive_zip))
        except Exception as e:
            logger.error(f"  [Drive CB] Erreur copie Drive : {e}")
            tmp_zip.unlink(missing_ok=True)
            return

        # Vérification arrivée sur Drive
        if not drive_zip.exists() or drive_zip.stat().st_size < 1_000_000:
            logger.error(
                "  [Drive CB] Zip Drive absent ou trop petit après copie "
                "— registre non mis à jour."
            )
            tmp_zip.unlink(missing_ok=True)
            return

        # ── 6. Mettre à jour le registre sur Drive ───────────────────────────
        drive_registry = self.drive_dir / REGISTRY_REMOTE
        _write_registry(
            drive_registry,
            bleu=current_bleu,
            filename=zip_name,
            epoch=current_epoch,
            step=current_step,
        )
        logger.info(f"  [Drive CB] Registre Drive mis à jour : BLEU={current_bleu:.4f}")

        # ── 7. Supprimer l'ancien zip Drive ──────────────────────────────────
        if registry and registry.get("filename"):
            old_zip = self.drive_dir / registry["filename"]
            if old_zip.exists() and old_zip.name != zip_name:
                try:
                    old_zip.unlink()
                    logger.info(f"  [Drive CB] Ancien zip Drive supprimé : {old_zip.name}")
                except Exception as e:
                    logger.warning(f"  [Drive CB] Impossible de supprimer l'ancien zip : {e}")

        # ── 8. Rafraîchir le registre local depuis Drive ─────────────────────
        REGISTRY_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(drive_registry), str(REGISTRY_LOCAL))
        logger.info("  [Drive CB] Registre local rafraîchi.")

        # ── 9. Nettoyage /tmp/ ───────────────────────────────────────────────
        tmp_zip.unlink(missing_ok=True)

        logger.info(
            f"  [Drive CB] ✅ Drive mis à jour : "
            f"BLEU {best_drive_bleu or 0:.4f} → {current_bleu:.4f} "
            f"| {zip_name}"
        )


# ---------------------------------------------------------------------------
# Callback de fin d'entraînement
# ---------------------------------------------------------------------------

class FinalModelCallback(TrainerCallback):
    """
    Callback exécuté une seule fois en fin d'entraînement.

    Sauvegarde le meilleur modèle (pas nécessairement le dernier) dans
    le dossier final local (Drive) et le publie sur HuggingFace Hub.

    Ordre garanti :
        1. Sauvegarde locale dans /tmp/gbeto_best_model/
        2. Copie vers final_dir (Drive)
        3. Push vers HuggingFace Hub (si push_to_hub=True)
        4. Suppression des checkpoints intermédiaires
        5. Nettoyage de /tmp/
    """

    def __init__(
        self,
        output_dir:    str,
        final_dir: str,
        hf_repo_id:    str,
        push_to_hub:   bool,
        tokenizer,
    ):
        self.output_dir    = output_dir
        self.final_dir     = final_dir
        self.hf_repo_id    = hf_repo_id
        self.push_to_hub   = push_to_hub
        self.tokenizer     = tokenizer

    def _ensure_hf_repo(self, api: HfApi, token: str) -> None:
        """Crée le repo HuggingFace s'il n'existe pas encore."""
        try:
            api.repo_info(repo_id=self.hf_repo_id, repo_type="model", token=token)
            logger.info(f"  Repo HF existant : {self.hf_repo_id}")
        except Exception:
            logger.info(f"  Création du repo HF : {self.hf_repo_id} ...")
            api.create_repo(
                repo_id=self.hf_repo_id,
                repo_type="model",
                private=False,
                exist_ok=True,
                token=token,
            )
            logger.info("  Repo HF créé.")

    def on_train_end(
        self,
        args,
        state:   TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if state.best_model_checkpoint is None:
            logger.warning(
                "FinalModelCallback : aucun meilleur checkpoint trouvé — "
                "push HuggingFace ignoré."
            )
            return

        best_ckpt = Path(state.best_model_checkpoint)
        logger.info(
            f"\n>>> Fin d'entraînement — meilleur checkpoint : {best_ckpt.name}"
        )
        logger.info(
            f"    BLEU retenu : {state.best_metric:.4f}"
        )

        # 1. Sauvegarde dans /tmp/
        tmp_dir = Path("/tmp/gbeto_best_model")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        model.save_pretrained(str(tmp_dir))
        self.tokenizer.save_pretrained(str(tmp_dir))
        logger.info(f"  Modèle écrit dans : {tmp_dir}")

        # 2. Copie vers dossier final Drive
        final_local = Path(self.final_dir)
        final_local.parent.mkdir(parents=True, exist_ok=True)
        if final_local.exists():
            shutil.rmtree(final_local)
        shutil.copytree(tmp_dir, final_local)
        logger.info(f"  Meilleur modèle sauvegardé : {final_local}")

        # 3. Push vers HuggingFace Hub
        if self.push_to_hub:
            hf_token = os.environ.get("HF_TOKEN_WRITE")
            if not hf_token:
                logger.error(
                    "  HF_TOKEN_WRITE absent dans .env — push HuggingFace ignoré."
                )
            else:
                logger.info(
                    f"  Push vers HuggingFace Hub : {self.hf_repo_id} ..."
                )
                api = HfApi()
                self._ensure_hf_repo(api, hf_token)
                api.upload_folder(
                    folder_path=str(tmp_dir),
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                    token=hf_token,
                    commit_message=(
                        f"Fine-tuned NLLB-600M ewe↔french — GbeTo "
                        f"(BLEU={state.best_metric:.2f})"
                    ),
                )
                logger.info(
                    f"  Push terminé : "
                    f"https://huggingface.co/{self.hf_repo_id}"
                )

        # 4. Suppression des checkpoints intermédiaires
        output_path = Path(self.output_dir)
        if output_path.exists():
            removed = []
            for item in sorted(output_path.iterdir()):
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    shutil.rmtree(item)
                    removed.append(item.name)
            if removed:
                logger.info(
                    f"  Checkpoints supprimés : {', '.join(removed)}"
                )

        # 5. Nettoyage /tmp/
        shutil.rmtree(tmp_dir)
        logger.info("  Nettoyage /tmp/ terminé.")
        logger.info("\n  ✓ Meilleur modèle disponible :")
        logger.info(f"    Drive  : {self.final_dir}")
        if self.push_to_hub:
            logger.info(
                f"    HF Hub : https://huggingface.co/{self.hf_repo_id}"
            )


# ---------------------------------------------------------------------------
# Pipeline d'entraînement principal
# ---------------------------------------------------------------------------

def train(config: TrainingConfig) -> None:
    """
    Orchestre le fine-tuning complet de NLLB-200-distilled-600M.

    Reprise automatique : si des checkpoints valides existent dans
    output_dir, l'entraînement reprend depuis le dernier état sauvegardé
    (poids, optimizer, scheduler, step courant). Les checkpoints
    incomplets (sans trainer_state.json valide) sont supprimés
    automatiquement. Si aucun checkpoint valide n'est trouvé,
    l'entraînement repart depuis le début.

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
    logger.info(
        f"  Batch effectif  : "
        f"{config.per_device_train_batch_size * config.gradient_accumulation_steps}"
    )
    logger.info(f"  Epochs max      : {config.num_train_epochs}")
    logger.info(
        f"  Early stopping  : patience={config.early_stopping_patience}"
    )
    logger.info(f"  fp16            : {config.fp16}")
    logger.info(f"  Seed            : {config.seed}")

    # ------------------------------------------------------------------
    # 2. Initialisation du registre Drive + reprise depuis Drive si nécessaire
    #
    # On inspecte drive_checkpoint_dir pour deux fichiers :
    #   - registry.json : méta-données du meilleur checkpoint Drive
    #   - checkpoint-X.zip : checkpoint compressé
    #
    # CAS 1 — zip présent, registry absent :
    #   Vérifier si zip potentiellement valide (taille > 1MB).
    #   Si valide → dézipper sur place pour lire trainer_state.json → créer registry.
    #   Sinon → supprimer le zip.
    #
    # CAS 2 — registry présent, zip absent :
    #   Le registry est inutile sans zip → supprimer le registry.
    #
    # CAS 3 — tous deux absents :
    #   Début absolu → rien à faire.
    #
    # CAS 4 — tous deux présents et valides :
    #   Copier registry en local. Copier zip en local, dézipper,
    #   valider l'intégrité → prêt pour reprise.
    # ------------------------------------------------------------------
    drive_dir      = Path(config.drive_checkpoint_dir)
    drive_registry = drive_dir / REGISTRY_REMOTE
    output_dir_path = Path(config.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    drive_zips = list(drive_dir.glob("checkpoint-*.zip")) if drive_dir.exists() else []
    drive_zip  = drive_zips[0] if drive_zips else None
    has_zip      = drive_zip is not None and drive_zip.stat().st_size > 1_000_000
    has_registry = drive_registry.exists()

    logger.info("\n>>> Inspection du dossier Drive ...")

    if has_zip and not has_registry:
        # CAS 1 : zip présent, registry absent → reconstruire registry
        logger.info("  CAS 1 : zip présent, registry absent → reconstruction du registre.")
        tmp_extract = Path("/tmp/drive_ckpt_inspect")
        try:
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract)
            with zipfile.ZipFile(drive_zip) as zf:
                zf.extractall(tmp_extract)
            state_file = tmp_extract / "trainer_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)
                bleu = state_data.get("best_metric")
                step = int(drive_zip.stem.split("-")[1])
                _write_registry(
                    drive_registry,
                    bleu=bleu or 0.0,
                    filename=drive_zip.name,
                    epoch=state_data.get("epoch", 0),
                    step=step,
                )
                logger.info(f"  Registre Drive reconstruit : BLEU={bleu:.4f}")
                has_registry = True
            else:
                logger.warning("  trainer_state.json absent dans le zip → zip supprimé.")
                drive_zip.unlink()
                has_zip = False
        except Exception as e:
            logger.warning(f"  Impossible de lire le zip ({e}) → zip supprimé.")
            if drive_zip.exists():
                drive_zip.unlink()
            has_zip = False
        finally:
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract)

    elif not has_zip and has_registry:
        # CAS 2 : registry sans zip → inutile
        logger.info("  CAS 2 : registry sans zip → suppression du registry.")
        drive_registry.unlink()
        has_registry = False

    elif not has_zip and not has_registry:
        # CAS 3 : début absolu
        logger.info("  CAS 3 : aucun checkpoint Drive → nouveau départ complet.")

    # CAS 4 : zip + registry tous deux valides → reprise depuis Drive
    last_checkpoint: Optional[str] = None

    if has_zip and has_registry:
        logger.info("  CAS 4 : zip + registry valides → préparation de la reprise.")

        # Copier registry en local
        shutil.copy2(str(drive_registry), str(REGISTRY_LOCAL))
        logger.info(f"  Registre local copié : {REGISTRY_LOCAL}")

        # Vérifier si un checkpoint local existe déjà (session non réinitialisée)
        local_ckpt = get_last_checkpoint(str(output_dir_path))
        if local_ckpt is not None and _is_checkpoint_valid(Path(local_ckpt)):
            logger.info(f"  Checkpoint local déjà présent : {local_ckpt} → reprise directe.")
            last_checkpoint = local_ckpt
        else:
            # Dézipper le zip Drive vers local
            registry = _read_registry(REGISTRY_LOCAL)
            zip_to_restore = drive_dir / registry["filename"]
            step_to_restore = registry["step"]
            restore_dir = output_dir_path / f"checkpoint-{step_to_restore}"
            tmp_zip_local = Path(f"/tmp/{zip_to_restore.name}")

            logger.info(f"  Copie du zip Drive en local : {tmp_zip_local} ...")
            shutil.copy2(str(zip_to_restore), str(tmp_zip_local))

            logger.info(f"  Dézippage → {restore_dir} ...")
            restore_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_zip_local) as zf:
                zf.extractall(restore_dir)

            tmp_zip_local.unlink(missing_ok=True)

            if _is_checkpoint_valid(restore_dir):
                last_checkpoint = str(restore_dir)
                logger.info(f"\n>>> REPRISE Drive validée : {restore_dir.name}")
                logger.info("  (poids, optimizer, scheduler, step courant restaurés)")
            else:
                logger.error(
                    f"  Checkpoint restauré invalide : {restore_dir}\n"
                    f"  → Nouveau départ."
                )
                shutil.rmtree(restore_dir, ignore_errors=True)
                REGISTRY_LOCAL.unlink(missing_ok=True)
                last_checkpoint = None

    # Si pas de reprise Drive, vérifier les checkpoints locaux existants
    if last_checkpoint is None:
        raw = get_last_checkpoint(str(output_dir_path))
        if raw is not None:
            raw_path = Path(raw)
            invalid_reason = None

            state_file = raw_path / "trainer_state.json"
            if not state_file.exists():
                invalid_reason = "trainer_state.json absent"
            else:
                try:
                    with open(state_file) as f:
                        json.load(f)
                except json.JSONDecodeError:
                    invalid_reason = "trainer_state.json corrompu"

            if invalid_reason is None:
                for weights in ["model.safetensors", "pytorch_model.bin"]:
                    p = raw_path / weights
                    if p.exists() and p.stat().st_size > 1_000_000:
                        break
                else:
                    invalid_reason = "poids du modèle absents ou trop petits"

            if invalid_reason is not None:
                logger.warning(
                    f"Checkpoint local invalide ({invalid_reason}) : {raw}\n"
                    f"  → Suppression et nouveau départ."
                )
                shutil.rmtree(raw)
            else:
                last_checkpoint = raw
                logger.info(f"\n>>> REPRISE locale détectée : {last_checkpoint}")

    if last_checkpoint is None:
        logger.info("\n>>> NOUVEAU DÉPART — aucun checkpoint valide trouvé.")

    # ------------------------------------------------------------------
    # 3. Chargement du modèle et tokenizer
    # ------------------------------------------------------------------
    logger.info(f"\n>>> Chargement du modèle : {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=os.environ.get("HF_TOKEN_READ"),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        token=os.environ.get("HF_TOKEN_READ"),
    )

    # gradient_checkpointing est géré par Seq2SeqTrainingArguments
    # Ne pas l'activer manuellement — le Trainer s'en charge

    num_params    = sum(p.numel() for p in model.parameters())
    num_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
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
    # 5. Data Collator
    # ------------------------------------------------------------------
    # Padding dynamique : chaque batch est padé à la longueur du plus
    # long exemple du batch (pas à MAX_LENGTH fixe) → économise la VRAM
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,    # Optimisation Tensor Cores (T4/A100)
        label_pad_token_id=-100,  # -100 ignoré dans le calcul de la loss
    )

    # ------------------------------------------------------------------
    # 6. Arguments d'entraînement
    # ------------------------------------------------------------------
    os.environ["WANDB_PROJECT"] = config.wandb_project

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,

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
        optim="adafactor",

        # Évaluation et sauvegarde par epoch
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=config.logging_steps,

        # Génération pour compute_metrics
        predict_with_generate=True,
        generation_num_beams=config.num_beams,
        generation_max_length=config.max_length,

        # Meilleur modèle
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        # save_total_limit géré par DriveCheckpointCallback (keep par BLEU)
        save_total_limit=config.save_total_limit,

        # Reproductibilité
        seed=config.seed,
        data_seed=config.seed,

        # Logging W&B
        report_to=config.report_to,
        run_name=config.run_name,
    )

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    trainer = GbeToTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
            ),
            DriveCheckpointCallback(
                drive_checkpoint_dir=config.drive_checkpoint_dir,
                warmup_epochs=config.warmup_epochs,
                keep_local=config.keep_local,
            ),
            FinalModelCallback(
                output_dir=config.output_dir,
                final_dir=config.final_dir,
                hf_repo_id=config.hf_repo_id,
                push_to_hub=config.push_to_hub,
                tokenizer=tokenizer,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # 8. Entraînement
    # ------------------------------------------------------------------
    logger.info("\n>>> Début de l'entraînement ...")
    logger.info(
        "  Steps par epoch   : "
        f"{len(train_dataset) // (config.per_device_train_batch_size * config.gradient_accumulation_steps)}"
    )

    if last_checkpoint is not None:
        logger.info(f"  Reprise depuis    : {last_checkpoint}")
    else:
        logger.info("  Départ depuis le début (epoch 0, step 0)")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("\n>>> Entraînement terminé.")
    logger.info(
        f"  Runtime       : "
        f"{train_result.metrics.get('train_runtime', 0):.1f}s"
    )
    logger.info(
        f"  Steps/seconde : "
        f"{train_result.metrics.get('train_steps_per_second', 0):.3f}"
    )
    logger.info(
        f"  Train loss    : "
        f"{train_result.metrics.get('train_loss', 0):.4f}"
    )

    # ------------------------------------------------------------------
    # 9. Évaluation finale
    # ------------------------------------------------------------------
    logger.info("\n>>> Évaluation finale sur le jeu de validation ...")
    eval_results = trainer.evaluate()

    final_bleu = eval_results.get("eval_bleu", 0.0)
    final_chrf = eval_results.get("eval_chrf", 0.0)
    final_loss = eval_results.get("eval_loss", 0.0)

    logger.info(f"\n{'=' * 65}")
    logger.info("RÉSULTATS FINAUX")
    logger.info(f"{'=' * 65}")
    logger.info(f"  BLEU (val)  : {final_bleu:.2f}")
    logger.info(f"  chrF (val)  : {final_chrf:.2f}")
    logger.info(f"  Loss (val)  : {final_loss:.4f}")
    logger.info(f"{'=' * 65}")

    # ------------------------------------------------------------------
    # 10. Sauvegarde des métriques finales
    # ------------------------------------------------------------------
    metrics_path = Path("outputs/final_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    final_metrics = {
        "model":           config.model_name,
        "bleu_val":        final_bleu,
        "chrf_val":        final_chrf,
        "loss_val":        final_loss,
        "train_loss":      train_result.metrics.get("train_loss", 0.0),
        "best_checkpoint": str(config.final_dir),
        "hyperparameters": {
            "learning_rate":               config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": (
                config.per_device_train_batch_size
                * config.gradient_accumulation_steps
            ),
            "num_train_epochs": config.num_train_epochs,
            "warmup_steps":     config.warmup_steps,
            "weight_decay":     config.weight_decay,
            "max_length":       config.max_length,
            "num_beams":        config.num_beams,
            "seed":             config.seed,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"\n  Métriques sauvegardées : {metrics_path}")
    logger.info(
        "\nEntraînement complet. Étape suivante : python app.py"
    )


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning NLLB-200 pour la traduction éwé-français.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_file", type=Path,
        default=Path("data/processed/train.csv")
    )
    parser.add_argument(
        "--val_file", type=Path,
        default=Path("data/processed/val.csv")
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/content/checkpoints_local",
        help="Dossier local Colab pour les checkpoints (jamais Drive direct)"
    )
    parser.add_argument(
        "--drive_checkpoint_dir", type=str,
        default="outputs/checkpoints",
        help="Dossier Drive : contient uniquement le meilleur zip + registry.json"
    )
    parser.add_argument(
        "--final_dir", type=str,
        default="outputs/final_model"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=3,
        help="Nombre d'epochs sans écriture Drive (laisser converger)"
    )
    parser.add_argument(
        "--keep_local", type=int, default=2,
        help="Nombre max de checkpoints locaux (les meilleurs par BLEU)"
    )
    parser.add_argument(
        "--hf_repo_id", type=str,
        default="kjd-dktech/gbeto-ewe-french"
    )
    parser.add_argument(
        "--no_push_to_hub", action="store_true",
        help="Désactiver le push vers HuggingFace Hub"
    )
    parser.add_argument("--model_name",    type=str,   default=MODEL_NAME)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--grad_accum",    type=int,   default=8)
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--warmup_steps",  type=int,   default=500)
    parser.add_argument("--num_beams",     type=int,   default=4)
    parser.add_argument("--seed",          type=int,   default=SEED)
    parser.add_argument(
        "--no_fp16", action="store_true",
        help="Désactiver fp16 (si GPU ne supporte pas)"
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb",
        choices=["wandb", "tensorboard", "none"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainingConfig(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        drive_checkpoint_dir=args.drive_checkpoint_dir,
        final_dir=args.final_dir,
        warmup_epochs=args.warmup_epochs,
        keep_local=args.keep_local,
        hf_repo_id=args.hf_repo_id,
        push_to_hub=not args.no_push_to_hub,
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
