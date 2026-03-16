---
title: GbeTo_-_Traducteur_w__Franais
app_file: app.py
sdk: gradio
sdk_version: 6.8.0
---
# GbeTo — Traducteur Éwé ↔ Français

[![CI](https://github.com/kjd-dktech/GbeTo_ewe-frh/actions/workflows/ci.yml/badge.svg)](https://github.com/kjd-dktech/GbeTo_ewe-fr/actions/workflows/ci.yml)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-gbeto--ewe--french-yellow)](https://huggingface.co/kjd-dktech/gbeto-ewe-french)

Fine-tuning de [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (Meta AI) pour la traduction bidirectionnelle éwé ↔ français. Projet personnel mené dans un double objectif : apprendre les pratiques du fine-tuning de modèles de traduction, et contribuer modestement à l'outillage NLP d'une langue peu dotée : l'**`ewe`**, *ma langue natale*.

---

## Contexte

L'éwé est une langue kwa parlée par environ 3 à 4 millions de personnes au Togo, au Ghana et au Bénin. Elle est très peu représentée dans les outils NLP existants. NLLB-200, bien que multilingue, n'a été pré-entraîné que sur un volume limité de données éwé. Ce projet explore dans quelle mesure un fine-tuning sur un corpus parallèle de taille modeste peut améliorer les performances de traduction sur cette paire de langues.

---

## Résultats

Évaluation sur le test set (~3 500 paires), tokenisation `flores101` (sacrebleu).

| Modèle | BLEU | chrF |
|--------|------|------|
| Baseline NLLB-600M (sans fine-tuning) | 13.41 | 33.11|
| GbeTo | 16.70 | 37.19|


<!--<u>`GbeTo`</u> :
| Direction | BLEU | chrF |
|--------|------|------|
| ewe → fra | 16.05 | 39.47|
| fra → ewe | 19.18 | 38.27|
| global    | 17.73 | 38.97|
-->
---

## Données

Deux corpus parallèles académiques, fusionnés et filtrés :

| Source | Papier | Paires éwé-français |
|--------|--------|---------------------|
| [AfroLingu-MT](https://huggingface.co/datasets/UBC-NLP/AfroLingu-MT) | ACL 2024 | ~10 500 |
| [MAFAND](https://huggingface.co/datasets/masakhane/mafand) | NAACL 2022 | ~5 000 |

Pipeline de nettoyage : normalisation Unicode NFC , déduplication, filtrage sur longueur (3–150 mots) et ratio source/cible (0.2–5.0).

Splits finaux : ~14 000 train / ~2 900 val / ~3 500 test.

---

## Modèle

**Base** : `facebook/nllb-200-distilled-600M` — 600M paramètres, 200 langues.  
**Fine-tuning** : Google Colab T4 16GB, ~7h30 d'entraînement, arrêt automatique à l'epoch 9 (early stopping, patience=3, métrique BLEU).

Principaux hyperparamètres :

| Paramètre | Valeur |
|-----------|--------|
| Learning rate | `5e-5` |
| Batch effectif | `32` (4 × accum. 8) |
| Optimiseur | Adafactor |
| fp16 | Oui |
| Beam search | 4 |
| Epochs max | 10 |

Le modèle est disponible sur HuggingFace : [kjd-dktech/gbeto-ewe-french](https://huggingface.co/kjd-dktech/gbeto-ewe-french).

---

## Utiliser le modèle

Le modèle fine-tuné est disponible directement sur HuggingFace :
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("kjd-dktech/gbeto-ewe-french")
model = AutoModelForSeq2SeqLM.from_pretrained("kjd-dktech/gbeto-ewe-french")
```

Pour reproduire l'entraînement, voir `notebooks/principal_book.ipynb`.
---

## Structure du projet
```
GbeTo_ewe-fr/
├── src/
│   ├── data/          # Acquisition, nettoyage, splits
│   ├── model/         # Fine-tuning (trainer.py)
│   └── evaluate/      # Métriques BLEU et chrF
├── scripts/
│   ├── prepare_data.sh
│   ├── train.sh
│   └── resume.sh      # Reprise après coupure Colab
├── notebooks/
│   └── principal_book.ipynb
├── tests/             # Tests unitaires (pipeline + métriques)
├── app.py             # Interface Gradio
└── outputs/
    ├── final_model/
    └── error_analysis/
```

---

## Limitations

- Corpus d'entraînement de taille modeste (~14 000 paires) — les performances restent limitées sur les textes hors-domaine.
- L'éwé n'est pas orthographiquement standardisé entre dialectes (Anlo, Ho, Kpando) — la qualité peut varier selon la variante d'entrée.

---

## Citation

Ce projet utilise les ressources suivantes. Si vous réutilisez ce travail ou ces données, merci de citer les travaux originaux :

**AfroLingu-MT** (obligatoire si utilisation des données) :
```bibtex
@inproceedings{elmadany2024toucan,
  title={Toucan: Many-to-Many Translation for 150 African Language Pairs},
  author={Elmadany, Abdelrahim and Adebara, Ife and Abdul-Mageed, Muhammad},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={13189--13206},
  year={2024}
}
```

**MAFAND** :
```bibtex
@inproceedings{adelani2022mafand,
  title={A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation},
  author={Adelani, David Ifeoluwa and others},
  booktitle={Proceedings of NAACL 2022},
  year={2022}
}
```

**NLLB-200** :
```bibtex
@article{costa2022nllb,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={Costa-jussà, Marta R. and others},
  journal={arXiv preprint arXiv:2207.04672},
  year={2022}
}
```

---

## Licences

Les ressources utilisées (NLLB-200, AfroLingu-MT, MAFAND) sont sous licences non-commerciales (CC-BY-NC-4.0). Ce projet est à usage académique uniquement.

---

## Auteur

**Kodjo Jean DEGBEVI** · [GitHub](https://github.com/kjd-dktech) · [HuggingFace](https://huggingface.co/kjd-dktech) · [LinkedIn](https://www.linkedin.com/in/kodjo-jean-degbevi-ba5170369) <br>
https://mayal.tech <br>
kodjojeandegbevi@gmail.com <br>
[contact@mayal.tech](mailto:contact@mayal.tech) <br>