"""
Pipeline d'acquisition et de fusion des datasets parallèles éwé-français.

Sources :
    - UBC-NLP/AfroLingu-MT  (ACL 2024) — gated, nécessite token HuggingFace
    - masakhane/mafand       (NAACL 2022) — public, subset 'fr-ewe'

Sorties :
    data/raw/afrolingu_<split>.csv
    data/raw/mafand_<split>.csv
    data/raw/merged_<split>.csv   ← fusion brute avant nettoyage

Format unifié de sortie :
    source    : texte dans la langue source
    target    : texte dans la langue cible
    direction : 'ewe-fra' ou 'fra-ewe'
    origin    : 'afrolingu' ou 'mafand'

Utilisation :
    python -m src.data.download [--output_dir data/raw] [--hf_token TOKEN]

Auteur : Kodjo Jean DEGBEVI
"""

import argparse
import logging
import os
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Charge les variables depuis .env à la racine du projet (si présent)
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
SPLITS = ["train", "validation", "test"]

# Langcodes cibles dans AfroLingu-MT
AFROLINGU_LANGCODES = {"ewe-fra", "fra-ewe"}

# Colonnes du format unifié
UNIFIED_COLUMNS = ["source", "target", "direction", "origin"]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    """Crée le dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un DataFrame en CSV avec encodage UTF-8."""
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"  → Sauvegardé : {path}  ({len(df):,} lignes)")


def _log_split_stats(df: pd.DataFrame, source: str, split: str) -> None:
    """Affiche les statistiques d'un split téléchargé."""
    dir_counts = df["direction"].value_counts().to_dict()
    logger.info(
        f"  [{source.upper()} | {split}] "
        f"{len(df):,} paires  —  "
        + "  ".join(f"{k}: {v:,}" for k, v in sorted(dir_counts.items()))
    )


# ---------------------------------------------------------------------------
# Chargement AfroLingu-MT
# ---------------------------------------------------------------------------

def load_afrolingu(split: str, hf_token: str) -> pd.DataFrame:
    """
    Charge un split de UBC-NLP/AfroLingu-MT et retourne uniquement
    les paires éwé↔français dans le format unifié.

    Args:
        split     : 'train', 'validation' ou 'test'
        hf_token  : Token HuggingFace (requis, dataset gated)

    Returns:
        DataFrame avec colonnes [source, target, direction, origin]
    """
    logger.info(f"Chargement AfroLingu-MT [{split}] ...")

    try:
        ds = load_dataset(
            "UBC-NLP/AfroLingu-MT",
            split=split,
            token=hf_token,
        )
    except Exception as e:
        logger.error(
            f"Impossible de charger AfroLingu-MT [{split}]. "
            f"Vérifiez votre token HuggingFace et l'acceptation "
            f"des conditions d'accès sur https://huggingface.co/datasets/UBC-NLP/AfroLingu-MT\n"
            f"Erreur : {e}"
        )
        sys.exit(1)

    records = []
    for row in tqdm(ds, desc=f"  AfroLingu [{split}]", unit="ex"):
        langcode = row["langcode"].strip().lower()

        if langcode not in AFROLINGU_LANGCODES:
            continue

        # Dans AfroLingu-MT :
        #   'input'  = texte source (dans la langue source du langcode)
        #   'output' = texte cible  (dans la langue cible du langcode)
        records.append({
            "source":    row["input"].strip(),
            "target":    row["output"].strip(),
            "direction": langcode,        # 'ewe-fra' ou 'fra-ewe'
            "origin":    "afrolingu",
        })

    df = pd.DataFrame(records, columns=UNIFIED_COLUMNS)
    _log_split_stats(df, "afrolingu", split)
    return df


# ---------------------------------------------------------------------------
# Chargement MAFAND
# ---------------------------------------------------------------------------

def load_mafand(split: str) -> pd.DataFrame:
    """
    Charge un split de LAFAND-MT (fr-ewe) depuis GitHub et retourne
    les paires dans les deux directions dans le format unifié.

    Source : masakhane-io/lafand-mt (remplace masakhane/mafand
    qui utilise un loading script déprécié par HuggingFace)

    Args:
        split : 'train', 'validation' ou 'test'

    Returns:
        DataFrame avec colonnes [source, target, direction, origin]
    """
    logger.info(f"Chargement LAFAND-MT [fr-ewe | {split}] ...")

    # Correspondance split → nom de fichier GitHub
    file_map = {
        "train":      "train.json",
        "validation": "dev.json",
        "test":       "test.json",
    }
    filename = file_map[split]
    url = (
        f"https://raw.githubusercontent.com/masakhane-io/"
        f"lafand-mt/main/data/json_files/fr-ewe/{filename}"
    )

    try:
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.error(
            f"Impossible de télécharger LAFAND-MT [{split}].\n"
            f"URL : {url}\n"
            f"Erreur : {e}"
        )
        sys.exit(1)

    records = []
    for line in response.text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            row         = json.loads(line)
            translation = row["translation"]
            fr_text     = translation["fr"].strip()
            ewe_text    = translation["ewe"].strip()
        except (json.JSONDecodeError, KeyError):
            continue

        # LAFAND-MT ne fournit que des paires fr→ewe.
        # On génère les deux directions pour entraîner un modèle bidirectionnel.
        records.append({
            "source":    fr_text,
            "target":    ewe_text,
            "direction": "fra-ewe",
            "origin":    "mafand",
        })
        records.append({
            "source":    ewe_text,
            "target":    fr_text,
            "direction": "ewe-fra",
            "origin":    "mafand",
        })

    df = pd.DataFrame(records, columns=UNIFIED_COLUMNS)
    _log_split_stats(df, "lafand-mt", split)
    return df


# ---------------------------------------------------------------------------
# Fusion et sauvegarde
# ---------------------------------------------------------------------------

def merge_and_save(
    df_afrolingu: pd.DataFrame,
    df_mafand: pd.DataFrame,
    split: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Fusionne les deux DataFrames, loggue les statistiques globales
    et sauvegarde les fichiers bruts individuels + le fichier fusionné.

    Args:
        df_afrolingu : DataFrame AfroLingu-MT pour ce split
        df_mafand    : DataFrame MAFAND pour ce split
        split        : Nom du split ('train', 'validation', 'test')
        output_dir   : Dossier de sortie (data/raw/)

    Returns:
        DataFrame fusionné
    """
    # Sauvegarde individuelle (traçabilité)
    _save_csv(df_afrolingu, output_dir / f"afrolingu_{split}.csv")
    _save_csv(df_mafand,    output_dir / f"mafand_{split}.csv")

    # Fusion
    df_merged = pd.concat([df_afrolingu, df_mafand], ignore_index=True)

    # Statistiques globales
    logger.info(f"\n{'='*60}")
    logger.info(f"FUSION [{split.upper()}]")
    logger.info(f"  AfroLingu-MT : {len(df_afrolingu):>6,} paires")
    logger.info(f"  MAFAND       : {len(df_mafand):>6,} paires")
    logger.info(f"  TOTAL        : {len(df_merged):>6,} paires")
    dir_counts = df_merged["direction"].value_counts()
    for direction, count in dir_counts.items():
        logger.info(f"    {direction} : {count:,}")
    logger.info(f"{'='*60}\n")

    # Sauvegarde du fichier fusionné (données brutes avant nettoyage)
    _save_csv(df_merged, output_dir / f"merged_{split}.csv")

    return df_merged


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def download_all(output_dir: Path, hf_token: str) -> None:
    """
    Orchestre le téléchargement de tous les splits pour les deux sources
    et produit les fichiers fusionnés bruts.

    Args:
        output_dir : Dossier de sortie (créé si inexistant)
        hf_token   : Token HuggingFace
    """
    _ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("PIPELINE D'ACQUISITION — ewe-french_translator")
    logger.info("Sources : UBC-NLP/AfroLingu-MT + masakhane/mafand")
    logger.info("=" * 60)

    summary = {}

    for split in SPLITS:
        logger.info(f"\n>>> Split : {split.upper()}")

        df_afrolingu = load_afrolingu(split, hf_token)
        df_mafand    = load_mafand(split)
        df_merged    = merge_and_save(df_afrolingu, df_mafand, split, output_dir)

        summary[split] = len(df_merged)

    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ GLOBAL")
    total = 0
    for split, count in summary.items():
        logger.info(f"  {split:<12} : {count:,} paires")
        total += count
    logger.info(f"  {'TOTAL':<12} : {total:,} paires")
    logger.info("=" * 60)
    logger.info(f"\nDonnées brutes sauvegardées dans : {output_dir}")
    logger.info("Étape suivante : python -m src.data.filter")


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge et fusionne les datasets éwé-français.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/raw"),
        help="Dossier de sortie pour les données brutes.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN_READ", ""),
        help=(
            "Token HuggingFace (lecture). "
            "Peut aussi être défini via la variable d'environnement HF_TOKEN_READ."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.hf_token:
        logger.error(
            "Token HuggingFace manquant. "
            "Définissez HF_TOKEN_READ en variable d'environnement "
            "ou passez --hf_token TOKEN."
        )
        sys.exit(1)

    download_all(
        output_dir=args.output_dir,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()