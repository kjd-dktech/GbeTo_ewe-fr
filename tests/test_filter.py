"""
Tests unitaires pour le pipeline de nettoyage (src/data/filter.py).

Couvre :
    - Normalisation Unicode NFC
    - Suppression des doublons exacts
    - Filtrage sur longueur (3 ≤ tokens ≤ 150)
    - Filtrage sur ratio de longueur (0.2 ≤ ratio ≤ 5.0)
    - Pipeline complet (filter_split)

Exécution :
    pytest tests/test_filter.py -v

Auteur : Kodjo Jean DEGBEVI
"""

import unicodedata

import pandas as pd
import pytest

from src.data.filter import (
    MIN_TOKENS,
    MAX_TOKENS,
    filter_split,
    step_deduplicate,
    step_filter_length,
    step_filter_ratio,
    step_normalize_unicode,
)


# ---------------------------------------------------------------------------
# Fixtures — données de test réutilisables
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Crée un DataFrame de test à partir d'une liste de dicts."""
    return pd.DataFrame(rows, columns=["source", "target", "direction", "origin"])


@pytest.fixture
def df_clean() -> pd.DataFrame:
    """DataFrame propre — toutes les paires doivent passer tous les filtres."""
    return _make_df([
        {
            "source":    "Le chat mange du poisson.",
            "target":    "Ɣlã la ɖu tsi.",
            "direction": "fra-ewe",
            "origin":    "mafand",
        },
        {
            "source":    "Míebiaa aɖaŋuɖoɖo tso ðasefo bubuwo gbɔ.",
            "target":    "Nous recevons aussi des instructions des autres apôtres.",
            "direction": "ewe-fra",
            "origin":    "afrolingu",
        },
        {
            "source":    "La santé est importante pour tous.",
            "target":    "Ŋgɔdzidzidodo nye vevie na amewo katã.",
            "direction": "fra-ewe",
            "origin":    "mafand",
        },
    ])


# ---------------------------------------------------------------------------
# Tests — Normalisation Unicode NFC
# ---------------------------------------------------------------------------

class TestNormalizeUnicode:

    def test_nfc_applied_to_source_and_target(self):
        """
        Vérifie que la normalisation NFC est appliquée sur source et target.
        Un 'é' NFD (U+0065 + U+0301) doit devenir NFC (U+00E9).
        """
        # Forme NFD : 'e' + combining accent
        e_nfd = "e\u0301"
        assert unicodedata.normalize("NFC", e_nfd) == "é"

        df = _make_df([{
            "source":    f"Caf{e_nfd} du matin",
            "target":    "Ŋdi ƒu tsi",
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_normalize_unicode(df)

        assert result.iloc[0]["source"] == "Café du matin"
        assert unicodedata.is_normalized("NFC", result.iloc[0]["source"])

    def test_ewe_special_chars_preserved(self):
        """
        Vérifie que les caractères spéciaux éwé (ɖ, ƒ, ŋ, ɣ, ɔ, ɛ, ʋ)
        sont préservés après normalisation NFC.
        """
        ewe_text = "ɖevi ƒe ŋkɔ nye ɣali eye eƒe nɔnɔme nyuie"
        df = _make_df([{
            "source":    ewe_text,
            "target":    "Le nom de l'enfant est Gali et sa vie est bonne.",
            "direction": "ewe-fra",
            "origin":    "afrolingu",
        }])

        result = step_normalize_unicode(df)

        assert result.iloc[0]["source"] == ewe_text
        for char in "ɖƒŋɣɔɛʋ":
            assert char in result.iloc[0]["source"]

    def test_empty_strings_removed(self):
        """Vérifie que les lignes avec source ou target vide sont supprimées."""
        df = _make_df([
            {
                "source":    "",
                "target":    "Ŋdi ƒu tsi",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
            {
                "source":    "Bonjour le monde",
                "target":    "",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
            {
                "source":    "Bonjour tout le monde",
                "target":    "Ŋdi na amewo katã",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
        ])

        result = step_normalize_unicode(df)

        assert len(result) == 1
        assert result.iloc[0]["source"] == "Bonjour tout le monde"

    def test_none_values_removed(self):
        """Vérifie que les lignes avec valeurs None sont supprimées."""
        df = _make_df([
            {
                "source":    None,
                "target":    "Ŋdi ƒu tsi",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
            {
                "source":    "Bonjour tout le monde",
                "target":    "Ŋdi na amewo katã",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
        ])

        result = step_normalize_unicode(df)

        assert len(result) == 1

    def test_already_nfc_unchanged(self):
        """Vérifie qu'une chaîne déjà en NFC n'est pas modifiée."""
        text = "Voici un texte déjà normalisé en NFC."
        assert unicodedata.is_normalized("NFC", text)

        df = _make_df([{
            "source":    text,
            "target":    "Esia nye nuŋlɔɖi si le NFC me.",
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_normalize_unicode(df)

        assert result.iloc[0]["source"] == text


# ---------------------------------------------------------------------------
# Tests — Déduplication
# ---------------------------------------------------------------------------

class TestDeduplicate:

    def test_exact_duplicate_removed(self):
        """Vérifie qu'un doublon exact (source + target identiques) est supprimé."""
        pair = {
            "source":    "Le soleil se lève à l'est.",
            "target":    "Ɣedze le ɣedzeƒe.",
            "direction": "fra-ewe",
            "origin":    "afrolingu",
        }
        df = _make_df([pair, pair])

        result = step_deduplicate(df)

        assert len(result) == 1

    def test_different_pairs_kept(self):
        """Vérifie que des paires différentes ne sont pas supprimées."""
        df = _make_df([
            {
                "source":    "Le soleil se lève.",
                "target":    "Ɣedze le ɣedzeƒe.",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
            {
                "source":    "La lune brille la nuit.",
                "target":    "Dzǝ lãlã le zã me.",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
        ])

        result = step_deduplicate(df)

        assert len(result) == 2

    def test_same_source_different_target_kept(self):
        """
        Vérifie que deux paires avec la même source mais des targets
        différentes sont toutes deux conservées.
        """
        df = _make_df([
            {
                "source":    "Bonjour",
                "target":    "Ŋdi",
                "direction": "fra-ewe",
                "origin":    "afrolingu",
            },
            {
                "source":    "Bonjour",
                "target":    "Ŋdi na wò",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
        ])

        result = step_deduplicate(df)

        assert len(result) == 2

    def test_multiple_duplicates(self):
        """Vérifie la suppression de multiples doublons (3 occurrences → 1)."""
        pair = {
            "source":    "Je m'appelle Jean.",
            "target":    "Ŋkɔe nye Jean.",
            "direction": "fra-ewe",
            "origin":    "afrolingu",
        }
        df = _make_df([pair, pair, pair])

        result = step_deduplicate(df)

        assert len(result) == 1

    def test_empty_dataframe(self):
        """Vérifie le comportement sur un DataFrame vide."""
        df = _make_df([])
        result = step_deduplicate(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests — Filtrage sur longueur
# ---------------------------------------------------------------------------

class TestFilterLength:

    def test_too_short_source_removed(self):
        """Vérifie qu'une source avec moins de MIN_TOKENS mots est supprimée."""
        short_words = " ".join(["mot"] * (MIN_TOKENS - 1))
        df = _make_df([{
            "source":    short_words,
            "target":    "Ŋdi na wò ame nyuie kple aƒe ƒe dɔwɔlawo",
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_length(df)

        assert len(result) == 0

    def test_too_short_target_removed(self):
        """Vérifie qu'une target avec moins de MIN_TOKENS mots est supprimée."""
        short_words = " ".join(["ŋdi"] * (MIN_TOKENS - 1))
        df = _make_df([{
            "source":    "Bonjour à tous les amis présents ici aujourd'hui.",
            "target":    short_words,
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_length(df)

        assert len(result) == 0

    def test_too_long_source_removed(self):
        """Vérifie qu'une source avec plus de MAX_TOKENS mots est supprimée."""
        long_text = " ".join(["mot"] * (MAX_TOKENS + 1))
        df = _make_df([{
            "source":    long_text,
            "target":    "Ŋdi na wò ame nyuie kple aƒe ƒe dɔwɔlawo",
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_length(df)

        assert len(result) == 0

    def test_exact_min_length_kept(self):
        """Vérifie qu'une paire exactement à MIN_TOKENS mots est conservée."""
        exact_min_src = " ".join(["bonjour"] * MIN_TOKENS)
        exact_min_tgt = " ".join(["ŋdi"] * MIN_TOKENS)
        df = _make_df([{
            "source":    exact_min_src,
            "target":    exact_min_tgt,
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_length(df)

        assert len(result) == 1

    def test_exact_max_length_kept(self):
        """Vérifie qu'une paire exactement à MAX_TOKENS mots est conservée."""
        exact_max_src = " ".join(["bonjour"] * MAX_TOKENS)
        exact_max_tgt = " ".join(["ŋdi"] * MAX_TOKENS)
        df = _make_df([{
            "source":    exact_max_src,
            "target":    exact_max_tgt,
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_length(df)

        assert len(result) == 1

    def test_valid_pair_kept(self, df_clean):
        """Vérifie que des paires valides passent le filtre de longueur."""
        result = step_filter_length(df_clean)
        assert len(result) == len(df_clean)


# ---------------------------------------------------------------------------
# Tests — Filtrage sur ratio
# ---------------------------------------------------------------------------

class TestFilterRatio:

    def test_ratio_too_low_removed(self):
        """
        Vérifie qu'une paire avec ratio source/target < MIN_RATIO est supprimée.
        Source très courte, target très longue → ratio < 0.2
        """
        # 2 mots source, 20 mots target → ratio = 0.1 < 0.2
        df = _make_df([{
            "source":    "Oui non",
            "target":    " ".join(["ŋdi"] * 20),
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_ratio(df)

        assert len(result) == 0

    def test_ratio_too_high_removed(self):
        """
        Vérifie qu'une paire avec ratio source/target > MAX_RATIO est supprimée.
        Source très longue, target très courte → ratio > 5.0
        """
        # 30 mots source, 2 mots target → ratio = 15 > 5.0
        df = _make_df([{
            "source":    " ".join(["bonjour"] * 30),
            "target":    "Ŋdi na",
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_ratio(df)

        assert len(result) == 0

    def test_balanced_ratio_kept(self):
        """Vérifie qu'une paire avec un ratio équilibré est conservée."""
        # 10 mots source, 10 mots target → ratio = 1.0
        df = _make_df([{
            "source":    " ".join(["bonjour"] * 10),
            "target":    " ".join(["ŋdi"] * 10),
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_ratio(df)

        assert len(result) == 1

    def test_boundary_min_ratio_kept(self):
        """Vérifie qu'une paire exactement à MIN_RATIO est conservée."""
        # ratio = 0.2 → 2 mots source, 10 mots target
        df = _make_df([{
            "source":    " ".join(["bonjour"] * 2),
            "target":    " ".join(["ŋdi"] * 10),
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_ratio(df)

        assert len(result) == 1

    def test_boundary_max_ratio_kept(self):
        """Vérifie qu'une paire exactement à MAX_RATIO est conservée."""
        # ratio = 5.0 → 10 mots source, 2 mots target
        df = _make_df([{
            "source":    " ".join(["bonjour"] * 10),
            "target":    " ".join(["ŋdi"] * 2),
            "direction": "fra-ewe",
            "origin":    "mafand",
        }])

        result = step_filter_ratio(df)

        assert len(result) == 1

    def test_valid_pairs_kept(self, df_clean):
        """Vérifie que des paires valides passent le filtre de ratio."""
        result = step_filter_ratio(df_clean)
        assert len(result) == len(df_clean)


# ---------------------------------------------------------------------------
# Tests — Pipeline complet (filter_split)
# ---------------------------------------------------------------------------

class TestFilterSplit:

    def test_clean_data_unchanged(self, df_clean):
        """
        Vérifie que des données déjà propres passent le pipeline complet
        sans perte.
        """
        result = filter_split(df_clean, split="test")
        assert len(result) == len(df_clean)

    def test_pipeline_removes_all_bad_pairs(self):
        """
        Vérifie que le pipeline complet supprime toutes les paires invalides
        d'un DataFrame mixte (bonnes + mauvaises paires).
        """
        df = _make_df([
            # Paire valide
            {
                "source":    "Le marché de Lomé est très animé le matin.",
                "target":    "Lomé ƒe aɖabaƒe le vevie ŋdi me.",
                "direction": "fra-ewe",
                "origin":    "afrolingu",
            },
            # Doublon de la paire valide → à supprimer
            {
                "source":    "Le marché de Lomé est très animé le matin.",
                "target":    "Lomé ƒe aɖabaƒe le vevie ŋdi me.",
                "direction": "fra-ewe",
                "origin":    "afrolingu",
            },
            # Source trop courte → à supprimer
            {
                "source":    "Oui",
                "target":    "Ɛ̃ aɖeke le ame sia ŋu",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
            # Ratio trop élevé → à supprimer
            {
                "source":    " ".join(["bonjour"] * 30),
                "target":    "Ŋdi",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
        ])

        result = filter_split(df, split="test")

        # Seule la première paire valide doit subsister
        assert len(result) == 1
        assert result.iloc[0]["source"] == "Le marché de Lomé est très animé le matin."

    def test_output_columns_preserved(self, df_clean):
        """Vérifie que les colonnes du DataFrame sont préservées après filtrage."""
        result = filter_split(df_clean, split="test")
        assert list(result.columns) == ["source", "target", "direction", "origin"]

    def test_empty_dataframe(self):
        """Vérifie que le pipeline gère un DataFrame vide sans erreur."""
        df = _make_df([])
        result = filter_split(df, split="test")
        assert len(result) == 0
        assert list(result.columns) == ["source", "target", "direction", "origin"]

    def test_index_reset_after_filtering(self):
        """
        Vérifie que l'index est réinitialisé après filtrage
        (pas de trous dans l'index).
        """
        df = _make_df([
            {
                "source":    "x",   # trop court → supprimé
                "target":    "y",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
            {
                "source":    "Le marché de Lomé est animé.",
                "target":    "Lomé ƒe aɖabaƒe le vevie.",
                "direction": "fra-ewe",
                "origin":    "mafand",
            },
        ])

        result = filter_split(df, split="test")

        assert len(result) == 1
        assert result.index.tolist() == [0]
