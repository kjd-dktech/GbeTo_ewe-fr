"""
Finalisation des splits train/validation/test.

Charge les fichiers filtrés produits par filter.py, fusionne les splits
correspondants entre les deux sources, shuffles avec un seed fixe et
sauvegarde les fichiers finaux prêts pour l'entraînement.

Entrées  : data/processed/filtered_<split>.csv
Sorties  : data/processed/train.csv
           data/processed/val.csv
           data/processed/test.csv

Note sur la terminologie :
    - filtered_train.csv      → train.csv
    - filtered_validation.csv → val.csv   (renommage intentionnel pour clarté)
    - filtered_test.csv       → test.csv

Utilisation :
    python -m src.data.split [--input_dir data/processed] [--output_dir data/processed]

Auteur : Kodjo Jean DEGBEVI
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

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
SEED = 42

# Correspondance : nom HuggingFace → nom fichier de sortie
SPLIT_MAP = {
    "train":      "train",
    "validation": "val",
    "test":       "test",
}

INPUT_PREFIX  = "filtered"
UNIFIED_COLUMNS = ["source", "target", "direction", "origin"]


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seeds(seed: int = SEED) -> None:
    """
    Fixe les seeds de toutes les sources d'aléatoire.

    Bibliothèques concernées ici : random, numpy.
    (torch n'est pas importé dans ce module — pas de GPU utilisé)
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Seeds fixés — random: {seed}, numpy: {seed}")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    """Crée le dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def _load_filtered(input_dir: Path, split: str) -> pd.DataFrame:
    """
    Charge le fichier filtré pour un split donné.

    Args:
        input_dir : Dossier contenant les fichiers filtered_<split>.csv
        split     : Nom du split HuggingFace ('train', 'validation', 'test')

    Returns:
        DataFrame chargé
    """
    path = input_dir / f"{INPUT_PREFIX}_{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Lancez d'abord : python -m src.data.filter"
        )

    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Chargé : {path}  ({len(df):,} lignes)")
    return df


def _shuffle(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Shuffles un DataFrame de manière reproductible.

    Args:
        df   : DataFrame à mélanger
        seed : Seed pour la reproductibilité

    Returns:
        DataFrame mélangé avec index réinitialisé
    """
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _save_split(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un split final en CSV UTF-8."""
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"  → Sauvegardé : {path}  ({len(df):,} lignes)")


def _log_split_stats(df: pd.DataFrame, name: str) -> None:
    """Affiche les statistiques détaillées d'un split final."""
    dir_counts  = df["direction"].value_counts()
    orig_counts = df["origin"].value_counts()

    logger.info(f"\n  ── {name.upper()} ({'─'*(40 - len(name))})")
    logger.info(f"  Total        : {len(df):,} paires")
    logger.info(f"  Par direction:")
    for direction, count in sorted(dir_counts.items()):
        pct = count / len(df) * 100
        logger.info(f"    {direction:<12} : {count:>6,}  ({pct:.1f}%)")
    logger.info(f"  Par origine:")
    for origin, count in sorted(orig_counts.items()):
        pct = count / len(df) * 100
        logger.info(f"    {origin:<12} : {count:>6,}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def prepare_splits(input_dir: Path, output_dir: Path) -> None:
    """
    Charge les fichiers filtrés, shuffles et sauvegarde les splits finaux.

    Les splits sont utilisés tels quels depuis les datasets sources
    (pas de re-split aléatoire) afin de respecter les benchmarks
    officiels AfroLingu-MT et MAFAND. Le shuffle interne à chaque
    split garantit un ordre aléatoire reproductible.

    Args:
        input_dir  : Dossier contenant les filtered_<split>.csv
        output_dir : Dossier de sortie pour train.csv, val.csv, test.csv
    """
    _ensure_dir(output_dir)
    set_seeds(SEED)

    logger.info("=" * 60)
    logger.info("FINALISATION DES SPLITS — ewe-french_translator")
    logger.info(f"Seed : {SEED}")
    logger.info(f"Entrée  : {input_dir}")
    logger.info(f"Sortie  : {output_dir}")
    logger.info("=" * 60)

    summary = {}

    for hf_split, out_name in SPLIT_MAP.items():
        logger.info(f"\n>>> Split : {hf_split.upper()} → {out_name}.csv")

        # Chargement du fichier filtré
        df = _load_filtered(input_dir, hf_split)

        # Vérification des colonnes attendues
        missing = set(UNIFIED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {hf_split} : {missing}\n"
                f"Attendues : {UNIFIED_COLUMNS}"
            )

        # Shuffle reproductible
        df = _shuffle(df, seed=SEED)

        # Statistiques
        _log_split_stats(df, out_name)

        # Sauvegarde
        output_path = output_dir / f"{out_name}.csv"
        _save_split(df, output_path)

        summary[out_name] = len(df)

    # Résumé global
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ FINAL")
    logger.info(f"{'Split':<10} {'Paires':>8}")
    logger.info("-" * 20)
    total = 0
    for name, count in summary.items():
        logger.info(f"  {name:<8}   {count:>8,}")
        total += count
    logger.info("-" * 20)
    logger.info(f"  {'TOTAL':<8}   {total:>8,}")
    logger.info("=" * 60)
    logger.info(f"\nSplits finaux disponibles dans : {output_dir}")
    logger.info("Étape suivante : python -m src.model.trainer")


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalise les splits train/val/test avec shuffle reproductible.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier contenant les fichiers filtered_<split>.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier de sortie pour train.csv, val.csv, test.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()