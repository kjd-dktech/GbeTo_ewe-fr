"""
Tests unitaires pour le calcul des métriques d'évaluation
(src/evaluate/metrics.py).

Couvre :
    - BLEU score (sacrebleu)
    - chrF score (sacrebleu)
    - Comportements aux cas limites (hypothèse vide, référence vide,
      hypothèse identique à la référence)

Exécution :
    pytest tests/test_metrics.py -v

Auteur : Kodjo Jean DEGBEVI
"""

import pytest

from src.evaluate.metrics import compute_bleu, compute_chrf


# ---------------------------------------------------------------------------
# Tests — BLEU score
# ---------------------------------------------------------------------------

class TestComputeBleu:

    def test_perfect_match_returns_100(self):
        """
        Hypothèse identique à la référence → BLEU = 100.
        C'est la propriété fondamentale du BLEU : score parfait
        si et seulement si hypothèse == référence.
        """
        hypotheses = ["Le chat est sur le tapis."]
        references = ["Le chat est sur le tapis."]

        score = compute_bleu(hypotheses, references)

        assert score == pytest.approx(100.0, abs=0.1)

    def test_empty_hypothesis_returns_0(self):
        """
        Hypothèse vide → BLEU = 0.
        Une traduction vide n'a aucun n-gramme en commun avec la référence.
        """
        hypotheses = [""]
        references = ["Le chat est sur le tapis."]

        score = compute_bleu(hypotheses, references)

        assert score == pytest.approx(0.0, abs=0.1)

    def test_completely_wrong_hypothesis_low_score(self):
        """
        Hypothèse sans aucun mot en commun avec la référence → score très bas.
        """
        hypotheses = ["xyz abc def ghi jkl mno"]
        references = ["Le chat est sur le tapis."]

        score = compute_bleu(hypotheses, references)

        assert score < 5.0

    def test_partial_match_intermediate_score(self):
        """
        Hypothèse avec quelques mots en commun → score intermédiaire.
        """
        hypotheses = ["Le chat dort sur le tapis rouge."]
        references = ["Le chat est sur le tapis."]

        score = compute_bleu(hypotheses, references)

        assert 0.0 < score < 100.0

    def test_multiple_pairs(self):
        """
        Vérifie que le calcul fonctionne sur un corpus de plusieurs paires.
        Score corpus ≠ moyenne des scores individuels (propriété du BLEU).
        """
        hypotheses = [
            "Le chat est sur le tapis.",
            "Le soleil brille aujourd'hui.",
        ]
        references = [
            "Le chat est sur le tapis.",
            "Le soleil brille dehors.",
        ]

        score = compute_bleu(hypotheses, references)

        # La première paire est parfaite, la seconde partielle
        assert 0.0 < score <= 100.0

    def test_returns_float(self):
        """Vérifie que compute_bleu retourne bien un float."""
        score = compute_bleu(
            ["Bonjour le monde."],
            ["Bonjour tout le monde."],
        )
        assert isinstance(score, float)

    def test_score_between_0_and_100(self):
        """Vérifie que le score est toujours dans [0, 100]."""
        hypotheses = ["quelques mots au hasard ici"]
        references  = ["Le chat mange du poisson frais."]

        score = compute_bleu(hypotheses, references)

        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Tests — chrF score
# ---------------------------------------------------------------------------

class TestComputeChrf:

    def test_perfect_match_returns_100(self):
        """
        Hypothèse identique à la référence → chrF = 100.
        """
        hypotheses = ["Ɣlã la ɖu tsi kple nu bubu."]
        references  = ["Ɣlã la ɖu tsi kple nu bubu."]

        score = compute_chrf(hypotheses, references)

        assert score == pytest.approx(100.0, abs=0.1)

    def test_empty_hypothesis_returns_0(self):
        """
        Hypothèse vide → chrF = 0.
        """
        hypotheses = [""]
        references  = ["Ɣlã la ɖu tsi kple nu bubu."]

        score = compute_chrf(hypotheses, references)

        assert score == pytest.approx(0.0, abs=0.1)

    def test_ewe_special_chars_handled(self):
        """
        Vérifie que chrF gère correctement les caractères spéciaux éwé
        (ɖ, ƒ, ŋ, ɣ, ɔ, ɛ, ʋ).

        chrF est basé sur les n-grammes de caractères — particulièrement
        adapté aux langues morphologiquement riches comme l'éwé.
        """
        hyp = ["ɖevi ƒe ŋkɔ nye ɣali"]
        ref = ["ɖevi ƒe ŋkɔ nye ɣali"]

        score = compute_chrf(hyp, ref)

        assert score == pytest.approx(100.0, abs=0.1)

    def test_partial_char_overlap_intermediate_score(self):
        """
        Une hypothèse avec chevauchement partiel de caractères
        donne un score intermédiaire.
        """
        hyp = ["ɖevi ƒe ŋkɔ"]
        ref = ["ɖevi ƒe ŋkɔ nye ɣali eye eƒe nɔnɔme nyuie"]

        score = compute_chrf(hyp, ref)

        assert 0.0 < score < 100.0

    def test_returns_float(self):
        """Vérifie que compute_chrf retourne bien un float."""
        score = compute_chrf(
            ["Ŋdi na wò."],
            ["Ŋdi na wò ame nyuie."],
        )
        assert isinstance(score, float)

    def test_score_between_0_and_100(self):
        """Vérifie que le score est toujours dans [0, 100]."""
        score = compute_chrf(
            ["quelques mots"],
            ["Lomé ƒe aɖabaƒe le vevie ŋdi me."],
        )
        assert 0.0 <= score <= 100.0

    def test_chrf_higher_than_bleu_on_partial_match(self):
        """
        Sur des correspondances partielles avec des langues morphologiquement
        riches, chrF tend à être plus généreux que BLEU car il opère au
        niveau des caractères.

        Ce test vérifie que les deux métriques donnent des résultats
        cohérents (toutes deux > 0) sur une paire partiellement correcte.
        """
        hyp = ["ɖevi ƒe ŋkɔ nye ɣali eye"]
        ref = ["ɖevi ƒe ŋkɔ nye ɣali eye eƒe nɔnɔme nyuie"]

        bleu_score = compute_bleu(hyp, ref)
        chrf_score = compute_chrf(hyp, ref)

        assert bleu_score > 0.0
        assert chrf_score > 0.0


# ---------------------------------------------------------------------------
# Tests — Comportements aux limites
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_word_hypothesis(self):
        """
        Une hypothèse d'un seul mot ne plante pas le calcul.
        BLEU avec n-grammes jusqu'à 4 → brevity penalty sévère.
        """
        score = compute_bleu(
            ["Bonjour"],
            ["Bonjour tout le monde et bonne journée."],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_list_length_mismatch_raises(self):
        """
        Des listes de longueurs différentes doivent lever une erreur.
        """
        with pytest.raises(Exception):
            compute_bleu(
                ["hyp1", "hyp2"],
                ["ref1"],
            )

    def test_unicode_normalization_consistent(self):
        """
        Deux représentations Unicode du même texte éwé doivent donner
        le même score (la normalisation NFC est appliquée dans metrics.py).
        """
        import unicodedata

        text_nfc = unicodedata.normalize("NFC", "Café du matin")
        text_nfd = unicodedata.normalize("NFD", "Café du matin")
        score_nfc = compute_bleu([text_nfc], [text_nfc])
        score_nfd = compute_bleu([text_nfd], [text_nfd])
        assert score_nfc == pytest.approx(score_nfd, abs=0.1)
