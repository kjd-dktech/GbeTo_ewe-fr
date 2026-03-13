"""
src/evaluate/metrics.py
=======================
Calcul des métriques d'évaluation pour la traduction automatique éwé-français.

Métriques implémentées :
    - BLEU  (Bilingual Evaluation Understudy) — métrique principale
    - chrF  (Character F-score)               — métrique secondaire

Choix des métriques :
    BLEU est la référence standard en traduction automatique, permettant
    la comparaison directe avec les scores publiés dans la littérature
    (benchmarks AfroLingu-MT, MAFAND).

    chrF est complémentaire : travaillant au niveau des caractères,
    il est plus robuste aux variations morphologiques et orthographiques
    de l'éwé (caractères spéciaux, tons, orthographe non standardisée).

Interprétation des scores BLEU pour éwé-français (low-resource) :
    < 10  : insuffisant
    10-15 : acceptable
    > 15  : bon résultat (objectif du projet)
    > 20  : excellent pour une paire low-resource

Utilisation :
    from src.evaluate.metrics import compute_bleu, compute_chrf

    bleu = compute_bleu(hypotheses, references)
    chrf = compute_chrf(hypotheses, references)

Auteur : Kodjo Jean DEGBEVI
"""

import logging
import unicodedata
from pathlib import Path
from typing import Optional

import sacrebleu

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Tokenisation flores101 : standard pour les langues africaines dans NLLB
# Permet la comparaison directe avec les scores publiés sur MAFAND
BLEU_TOKENIZE = "flores101"

# Ordre des n-grammes de caractères pour chrF
CHRF_CHAR_ORDER = 6  # valeur par défaut sacrebleu, standard en NLP

# Ordre des n-grammes de mots pour chrF (0 = chrF sans word n-grams = chrF)
# chrF++ utilise word_order=2 — on reste sur chrF standard
CHRF_WORD_ORDER = 0


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _normalize_texts(texts: list[str]) -> list[str]:
    """
    Applique la normalisation Unicode NFC sur une liste de textes.

    Garantit la cohérence avec le pipeline de données (filter.py)
    et évite des scores artificiellement bas dus à des différences
    d'encodage Unicode pour les caractères éwé spéciaux.

    Args:
        texts : Liste de chaînes à normaliser

    Returns:
        Liste de chaînes normalisées NFC
    """
    return [
        unicodedata.normalize("NFC", t) if isinstance(t, str) else ""
        for t in texts
    ]


def _validate_inputs(hypotheses: list[str], references: list[str]) -> None:
    """
    Valide que les listes d'hypothèses et de références sont cohérentes.

    Args:
        hypotheses : Traductions générées par le modèle
        references : Traductions de référence humaines

    Raises:
        ValueError : Si les longueurs ne correspondent pas ou si les
                     listes sont vides
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Longueurs incohérentes : "
            f"{len(hypotheses)} hypothèses vs {len(references)} références."
        )
    if len(hypotheses) == 0:
        raise ValueError("Les listes d'hypothèses et de références sont vides.")


# ---------------------------------------------------------------------------
# Métriques principales
# ---------------------------------------------------------------------------

def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    tokenize: str = BLEU_TOKENIZE,
) -> float:
    """
    Calcule le score BLEU corpus sur un ensemble de paires hypothèse/référence.

    BLEU mesure le chevauchement de n-grammes (1 à 4) entre les traductions
    générées et les références humaines, avec une pénalité de brièveté (BP)
    pour éviter de favoriser les traductions courtes.

    Formule :
        BLEU = BP · exp( Σₙ wₙ · log pₙ )
        BP   = min(1, exp(1 - |ref| / |hyp|))
        pₙ   = précision des n-grammes d'ordre n (avec clipping)

    Args:
        hypotheses : Traductions générées par le modèle
                     ex: ["Le chat est sur le tapis."]
        references : Traductions de référence humaines
                     ex: ["Le chat dort sur le tapis rouge."]
        tokenize   : Méthode de tokenisation sacrebleu.
                     'flores101' est recommandé pour les langues africaines
                     et permet la comparaison avec les benchmarks NLLB/MAFAND.

    Returns:
        Score BLEU dans [0, 100]. 100 = traduction parfaite.

    Notes:
        - Utilise sacrebleu pour la reproductibilité et la comparabilité
          avec la littérature.
        - Le score corpus est différent de la moyenne des scores individuels :
          BLEU est conçu pour être calculé sur un corpus entier.
    """
    _validate_inputs(hypotheses, references)

    hypotheses = _normalize_texts(hypotheses)
    references  = _normalize_texts(references)

    # sacrebleu attend les références sous forme de liste de listes
    # (pour supporter plusieurs références par hypothèse)
    result = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize=tokenize,
    )

    return round(result.score, 4)


def compute_chrf(
    hypotheses: list[str],
    references: list[str],
    char_order: int = CHRF_CHAR_ORDER,
    word_order: int = CHRF_WORD_ORDER,
) -> float:
    """
    Calcule le score chrF corpus sur un ensemble de paires hypothèse/référence.

    chrF (Character F-score) mesure le chevauchement de n-grammes de
    caractères entre hypothèses et références. Plus robuste que BLEU pour :
        - Les langues morphologiquement riches (l'éwé forme des mots
          composés complexes)
        - Les variations orthographiques (tons marqués ou non en éwé)
        - Les faibles volumes de données (moins sensible aux 0-counts
          de n-grammes longs)

    Formule :
        chrF = (1 + β²) · chrP · chrR / (β² · chrP + chrR)
        β = 2 (poids du rappel deux fois celui de la précision, par défaut)

    Args:
        hypotheses : Traductions générées par le modèle
        references : Traductions de référence humaines
        char_order : Ordre maximum des n-grammes de caractères (défaut: 6)
        word_order : Ordre des n-grammes de mots (0 = chrF, 2 = chrF++)

    Returns:
        Score chrF dans [0, 100]. 100 = traduction parfaite.
    """
    _validate_inputs(hypotheses, references)

    hypotheses = _normalize_texts(hypotheses)
    references  = _normalize_texts(references)

    result = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        char_order=char_order,
        word_order=word_order,
    )

    return round(result.score, 4)


# ---------------------------------------------------------------------------
# Évaluation complète
# ---------------------------------------------------------------------------

def evaluate(
    hypotheses: list[str],
    references: list[str],
    split_name: str = "test",
    output_path: Optional[Path] = None,
) -> dict[str, float]:
    """
    Calcule BLEU et chrF et retourne un dictionnaire de résultats.

    Utilisée par trainer.py pour l'évaluation en fin d'entraînement
    et par le notebook pour l'analyse des résultats.

    Args:
        hypotheses  : Traductions générées par le modèle
        references  : Traductions de référence humaines
        split_name  : Nom du split évalué (pour les logs)
        output_path : Si fourni, sauvegarde les métriques en JSON

    Returns:
        Dictionnaire {'bleu': float, 'chrf': float}
    """
    _validate_inputs(hypotheses, references)

    bleu = compute_bleu(hypotheses, references)
    chrf = compute_chrf(hypotheses, references)

    metrics = {
        "bleu": bleu,
        "chrf": chrf,
        "num_samples": len(hypotheses),
        "split": split_name,
    }

    # Logging des résultats
    logger.info(f"\n{'=' * 50}")
    logger.info(f"MÉTRIQUES D'ÉVALUATION — {split_name.upper()}")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Échantillons : {len(hypotheses):,}")
    logger.info(f"  BLEU         : {bleu:.2f}")
    logger.info(f"  chrF         : {chrf:.2f}")
    logger.info(f"{'=' * 50}")

    # Interprétation du score BLEU
    if bleu >= 20:
        logger.info("  → Excellent résultat pour une paire low-resource.")
    elif bleu >= 15:
        logger.info("  → Bon résultat. Objectif du projet atteint.")
    elif bleu >= 10:
        logger.info("  → Résultat acceptable. Marge d'amélioration possible.")
    else:
        logger.info("  → Score insuffisant. Revoir les données ou les hyperparamètres.")

    # Sauvegarde JSON si demandée
    if output_path is not None:
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"\n  Métriques sauvegardées : {output_path}")

    return metrics
