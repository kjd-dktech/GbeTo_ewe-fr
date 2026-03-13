"""
Pipeline de nettoyage des données brutes éwé-français.

Applique les filtres suivants dans cet ordre sur chaque split :
    1. Normalisation Unicode NFC
    2. Suppression des doublons exacts (source + target)
    3. Filtrage sur longueur (3 ≤ tokens ≤ 150, en mots)
    4. Filtrage sur ratio de longueur source/target (0.2 ≤ ratio ≤ 5.0)

Entrées  : data/raw/merged_<split>.csv
Sorties  : data/processed/filtered_<split>.csv

Utilisation :
    python -m src.data.filter [--input_dir data/raw] [--output_dir data/processed]

Auteur : Kodjo Jean DEGBEVI
"""

import argparse
import logging
import unicodedata
from pathlib import Path

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
# Constantes — seuils de filtrage
# ---------------------------------------------------------------------------
MIN_TOKENS    = 3      # Nombre minimum de mots dans source ET target
MAX_TOKENS    = 150    # Nombre maximum de mots dans source ET target
MIN_RATIO     = 0.2    # Ratio minimum len(source) / len(target)
MAX_RATIO     = 5.0    # Ratio maximum len(source) / len(target)

SPLITS        = ["train", "validation", "test"]
INPUT_PREFIX  = "merged"      # Fichiers produits par download.py
OUTPUT_PREFIX = "filtered"    # Fichiers produits par ce script


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    """Crée le dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def _token_count(text: str) -> int:
    """
    Compte le nombre de tokens (mots) dans une chaîne.
    Approximation par split sur les espaces — rapide et suffisante
    pour les filtres de longueur avant tokenisation SentencePiece.
    """
    return len(text.split())


def _log_step(
    step_name: str,
    count_before: int,
    count_after: int,
) -> None:
    """Loggue les statistiques d'une étape de filtrage."""
    removed  = count_before - count_after
    pct      = (removed / count_before * 100) if count_before > 0 else 0.0
    logger.info(
        f"  [{step_name:<35}] "
        f"avant: {count_before:>6,}  "
        f"après: {count_after:>6,}  "
        f"supprimés: {removed:>5,} ({pct:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Étapes de filtrage
# ---------------------------------------------------------------------------

def step_normalize_unicode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étape 1 — Normalisation Unicode NFC.

    L'éwé utilise des caractères spéciaux (ɖ, ƒ, ŋ, ɣ, ɔ, ɛ, ʋ) et des
    diacritiques de tons qui peuvent être encodés sous deux formes Unicode :
    - NFC (composée)  : 'é' = U+00E9 (1 point de code)
    - NFD (décomposée): 'é' = U+0065 + U+0301 (2 points de code)

    Sans normalisation, deux chaînes visuellement identiques seraient
    considérées comme différentes par Python, faussant la déduplication
    et créant des tokens distincts pour le même mot.

    NFC est la forme standard utilisée par les tokenizers modernes.
    """
    count_before = len(df)

    df = df.copy()
    df["source"] = df["source"].apply(
        lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x
    )
    df["target"] = df["target"].apply(
        lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x
    )

    # Suppression des lignes avec valeurs nulles ou vides après normalisation
    df = df.dropna(subset=["source", "target"])
    df = df[df["source"].str.strip().ne("") & df["target"].str.strip().ne("")]

    _log_step("Normalisation Unicode NFC", count_before, len(df))
    return df.reset_index(drop=True)


def step_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étape 2 — Suppression des doublons exacts.

    Un doublon est défini comme une paire (source, target) identique,
    indépendamment de la direction ou de l'origine.

    Justification :
    - AfroLingu-MT est construit à partir de 43 sources distinctes ;
      des chevauchements sont probables.
    - Des doublons dans le train biaiseraient l'apprentissage.
    - Des doublons entre train et val/test mesureraient de la
      mémorisation plutôt que de la généralisation.
    """
    count_before = len(df)

    df = df.drop_duplicates(subset=["source", "target"], keep="first")

    _log_step("Déduplication (source, target)", count_before, len(df))
    return df.reset_index(drop=True)


def step_filter_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étape 3 — Filtrage sur la longueur en mots.

    Conserve uniquement les paires où source ET target ont
    entre MIN_TOKENS et MAX_TOKENS mots (inclus).

    Borne basse (< 3 mots) : paires ultra-courtes = artefacts
    d'extraction (titres, fragments, métadonnées) sans valeur
    d'apprentissage.

    Borne haute (> 150 mots) : le fine-tuning utilise
    max_source_length=128 tokens SentencePiece. Les séquences
    trop longues seraient tronquées, rendant la cible incohérente
    avec la source tronquée.

    Note : les "tokens" ici sont des mots (split espaces).
    Un mot éwé donne en moyenne 1.5-2 tokens SentencePiece,
    donc 150 mots ≈ 225-300 tokens — sous la limite de 512.
    """
    count_before = len(df)

    src_len  = df["source"].apply(_token_count)
    tgt_len  = df["target"].apply(_token_count)

    mask = (
        src_len.between(MIN_TOKENS, MAX_TOKENS)
        & tgt_len.between(MIN_TOKENS, MAX_TOKENS)
    )
    df = df[mask]

    _log_step(
        f"Longueur [{MIN_TOKENS}–{MAX_TOKENS} mots]",
        count_before,
        len(df),
    )
    return df.reset_index(drop=True)


def step_filter_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étape 4 — Filtrage sur le ratio de longueur source/target.

    Conserve uniquement les paires où :
        MIN_RATIO ≤ len(source) / len(target) ≤ MAX_RATIO

    Détecte les paires mal alignées (ex : source 50 mots, cible 2 mots
    = extraction incorrecte, pas une traduction).

    Bornes conservatrices (0.2 / 5.0) pour tolérer les différences
    structurelles légitimes entre l'éwé (verbes sériels, constructions
    compactes) et le français.
    """
    count_before = len(df)

    src_len = df["source"].apply(_token_count)
    tgt_len = df["target"].apply(_token_count)

    # Protection contre la division par zéro (ne devrait pas arriver
    # après le filtre de longueur, mais par sécurité)
    ratio = src_len / tgt_len.replace(0, float("nan"))

    mask = ratio.between(MIN_RATIO, MAX_RATIO)
    df   = df[mask]

    _log_step(
        f"Ratio longueur [{MIN_RATIO}–{MAX_RATIO}]",
        count_before,
        len(df),
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline complet pour un split
# ---------------------------------------------------------------------------

def filter_split(df_raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Applique le pipeline de filtrage complet sur un DataFrame brut.

    Args:
        df_raw : DataFrame brut produit par download.py
        split  : Nom du split (pour les logs)

    Returns:
        DataFrame nettoyé
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FILTRAGE [{split.upper()}]  —  {len(df_raw):,} paires en entrée")
    logger.info(f"{'=' * 60}")

    df = df_raw.copy()
    df = step_normalize_unicode(df)
    df = step_deduplicate(df)
    df = step_filter_length(df)
    df = step_filter_ratio(df)

    # Bilan final du split
    total_removed = len(df_raw) - len(df)
    pct_kept      = len(df) / len(df_raw) * 100 if len(df_raw) > 0 else 0.0

    logger.info(f"\n  ── Bilan [{split.upper()}] ──────────────────────────────")
    logger.info(f"  Entrée   : {len(df_raw):>6,} paires")
    logger.info(f"  Sortie   : {len(df):>6,} paires  ({pct_kept:.1f}% conservées)")
    logger.info(f"  Supprimés: {total_removed:>6,} paires")

    # Répartition par direction
    dir_counts = df["direction"].value_counts()
    for direction, count in sorted(dir_counts.items()):
        logger.info(f"    {direction} : {count:,}")

    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def filter_all(input_dir: Path, output_dir: Path) -> None:
    """
    Charge, filtre et sauvegarde tous les splits.

    Args:
        input_dir  : Dossier contenant les fichiers merged_<split>.csv
        output_dir : Dossier de sortie pour les fichiers filtered_<split>.csv
    """
    _ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("PIPELINE DE FILTRAGE — ewe-french_translator")
    logger.info(f"Entrée  : {input_dir}")
    logger.info(f"Sortie  : {output_dir}")
    logger.info("=" * 60)

    summary = {}

    for split in SPLITS:
        input_path  = input_dir  / f"{INPUT_PREFIX}_{split}.csv"
        output_path = output_dir / f"{OUTPUT_PREFIX}_{split}.csv"

        # Vérification de l'existence du fichier source
        if not input_path.exists():
            logger.error(
                f"Fichier introuvable : {input_path}\n"
                f"Lancez d'abord : python -m src.data.download"
            )
            continue

        # Chargement
        logger.info(f"\n>>> Chargement : {input_path}")
        df_raw = pd.read_csv(input_path, encoding="utf-8")
        logger.info(f"  {len(df_raw):,} paires chargées")

        # Filtrage
        df_clean = filter_split(df_raw, split)

        # Sauvegarde
        df_clean.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"\n  → Sauvegardé : {output_path}  ({len(df_clean):,} lignes)")

        summary[split] = {
            "avant":  len(df_raw),
            "après":  len(df_clean),
        }

    # Résumé global
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ GLOBAL DU FILTRAGE")
    logger.info(f"{'Split':<14} {'Avant':>8} {'Après':>8} {'Supprimés':>10} {'Conservés':>10}")
    logger.info("-" * 54)
    total_avant = total_apres = 0
    for split, stats in summary.items():
        avant     = stats["avant"]
        apres     = stats["après"]
        supprimes = avant - apres
        pct       = apres / avant * 100 if avant > 0 else 0.0
        logger.info(
            f"  {split:<12} {avant:>8,} {apres:>8,} {supprimes:>10,} {pct:>9.1f}%"
        )
        total_avant += avant
        total_apres += apres
    logger.info("-" * 54)
    total_suppr = total_avant - total_apres
    total_pct   = total_apres / total_avant * 100 if total_avant > 0 else 0.0
    logger.info(
        f"  {'TOTAL':<12} {total_avant:>8,} {total_apres:>8,} "
        f"{total_suppr:>10,} {total_pct:>9.1f}%"
    )
    logger.info("=" * 60)
    logger.info(f"\nDonnées filtrées sauvegardées dans : {output_dir}")
    logger.info("Étape suivante : python -m src.data.split")


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nettoie les données brutes éwé-français.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Dossier contenant les fichiers merged_<split>.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier de sortie pour les fichiers filtered_<split>.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
