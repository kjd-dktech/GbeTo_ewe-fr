# 🌍 GbeTo : Traducteur Éwé ↔ Français

Traduction automatique neuronale bidirectionnelle éwé ↔ français, basée sur le fine-tuning de [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (Meta AI, 2022) sur des corpus parallèles de qualité académique.

[![CI](https://github.com/kjd-dktech/ewe-french_translator/actions/workflows/ci.yml/badge.svg)](https://github.com/kjd-dktech/ewe-french_translator/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces/kjd-dktech/ewe-french-translator)

---

## Contexte et motivation

L'éwé est une langue kwa parlée par environ **3 à 4 millions de personnes** au Togo, au Ghana et au Bénin. Malgré son nombre de locuteurs, elle est quasi absente des outils de traitement automatique du langage naturel : Joshi et al. (2020) la classent en catégorie 0-1 dans leur taxonomie des langues selon leur représentation dans les données NLP mondiales.

Ce projet développe un système de traduction automatique éwé ↔ français de qualité démontrable, avec deux objectifs : contribuer à réduire la fracture numérique linguistique pour les locuteurs éwé, et démontrer qu'un fine-tuning rigoureux sur un corpus limité mais de haute qualité peut produire des résultats compétitifs sur une paire de langues à faibles ressources.

---

## Résultats

| Métrique | Baseline NLLB (sans fine-tuning) | Après fine-tuning | Amélioration |
|----------|----------------------------------|-------------------|--------------|
| BLEU     | ~8–12                            | *à compléter*     | —            |
| chrF     | —                                | *à compléter*     | —            |

> Les métriques finales seront mises à jour après l'entraînement complet.  
> Scores calculés sur le test set avec `sacrebleu`, tokenisation `flores101`.

---

## Architecture du projet

```
ewe-french_translator/
│
├── src/
│   ├── data/
│   │   ├── download.py       # Acquisition et fusion des datasets
│   │   ├── filter.py         # Nettoyage et filtrage des données
│   │   └── split.py          # Finalisation des splits (seed=42)
│   ├── model/
│   │   └── trainer.py        # Fine-tuning NLLB avec reprise automatique
│   └── evaluate/
│       └── metrics.py        # BLEU et chrF (sacrebleu, flores101)
│
├── tests/
│   ├── test_filter.py        # 20 tests unitaires — pipeline de données
│   └── test_metrics.py       # 14 tests unitaires — métriques
│
├── scripts/
│   ├── prepare_data.sh       # Pipeline données en une commande
│   └── train.sh              # Entraînement en une commande
│
├── notebooks/
│   └── principal_book.ipynb  # EDA, entraînement Colab, error analysis
│
├── docs/
│   ├── cahier_projet_ewe_french.md  # Cahier de projet complet
│   └── nlp_dl_reference.md          # Référence théorique NLP/DL
│
├── outputs/
│   ├── checkpoints/          # Checkpoints d'entraînement
│   ├── logs/                 # Logs W&B / TensorBoard
│   ├── error_analysis/       # Analyse des erreurs de traduction
│   └── final_metrics.json    # Métriques finales BLEU/chrF
│
├── app.py                    # Interface Gradio (demo)
├── requirements.txt
└── .github/workflows/ci.yml  # CI/CD (lint + tests)
```

---

## Données

### Sources

| Dataset | Papier | Licence | Paires éwé-français |
|---------|--------|---------|---------------------|
| [UBC-NLP/AfroLingu-MT](https://huggingface.co/datasets/UBC-NLP/AfroLingu-MT) | ACL 2024 | Non-commercial | 10 500 |
| [masakhane/mafand](https://huggingface.co/datasets/masakhane/mafand) | NAACL 2022 | CC-BY-NC-4.0 | 5 000 |

### Volumes finaux

| Split | Paires |
|-------|--------|
| Train | ~14 060 |
| Val   | ~2 920  |
| Test  | ~3 520  |
| **Total** | **~20 500** |

> Les volumes exacts sont produits par le pipeline de données et peuvent varier légèrement selon le filtrage appliqué.

### Pipeline de nettoyage

Les données brutes sont filtrées dans cet ordre :

1. **Normalisation Unicode NFC** — harmonise l'encodage des caractères éwé spéciaux (ɖ, ƒ, ŋ, ɣ, ɔ, ɛ, ʋ)
2. **Déduplication** — supprime les paires (source, target) identiques
3. **Filtrage sur longueur** — conserve les paires avec 3 ≤ mots ≤ 150
4. **Filtrage sur ratio** — conserve les paires avec 0.2 ≤ len(source)/len(target) ≤ 5.0

---

## Modèle

**`facebook/nllb-200-distilled-600M`** — Transformer encodeur-décodeur de 600M paramètres, distillé depuis un modèle 3.3B, supportant 200 langues dont l'éwé (`ewe_Latn`) et le français (`fra_Latn`).

### Hyperparamètres d'entraînement

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Learning rate | `5e-5` | Standard pour fine-tuning de transformer pré-entraîné |
| Batch / device | `8` | Contrainte VRAM T4 16GB |
| Gradient accumulation | `4` | Batch effectif = 32 |
| Epochs max | `10` | Avec early stopping patience=3 |
| Warmup steps | `500` | Stabilise le début de l'entraînement |
| Weight decay | `0.01` | Régularisation L2 |
| fp16 | `True` | Réduit la VRAM de ~40% |
| Beam search | `4` | Standard en traduction automatique |
| Seed | `42` | Reproductibilité complète |

### Stratégie d'entraînement

- **Early stopping** sur le BLEU de validation (patience=3) plutôt que sur la loss — la loss peut continuer à baisser alors que la qualité des traductions stagne
- **Reprise automatique** : si des checkpoints existent dans `outputs/checkpoints/`, l'entraînement reprend depuis le dernier état sauvegardé (poids, optimizer, scheduler, step courant)
- **Sauvegarde** toutes les 200 steps, conservation des 2 meilleurs checkpoints uniquement

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/kjd-dktech/ewe-french_translator.git
cd ewe-french_translator

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et renseigner HF_TOKEN et WANDB_API_KEY
```

### Fichier `.env`

```
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxx"
WANDB_PROJECT="..."
```
---

## Utilisation

### 1. Préparer les données

```bash
bash scripts/prepare_data.sh
```

Enchaîne automatiquement le téléchargement, le nettoyage et la finalisation des splits.

### 2. Lancer l'entraînement

```bash
bash scripts/train.sh
```

Détecte automatiquement si c'est un nouveau départ ou une reprise depuis un checkpoint existant.

Pour surcharger un paramètre :

```bash
bash scripts/train.sh --epochs 5 --batch_size 4
```

### 3. Lancer la démonstration

```bash
python app.py
```

L'interface Gradio charge automatiquement le modèle fine-tuné (`outputs/checkpoints/best_model/`) si disponible, sinon le modèle baseline.

### 4. Lancer les tests

```bash
pytest tests/ -v
```

---

## Utilisation sur Google Colab

L'entraînement est conçu pour tourner sur Google Colab (GPU T4 gratuit). Le notebook `notebooks/principal_book.ipynb` orchestre l'ensemble du pipeline.

```python
# Montage du Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Authentification HuggingFace
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'))

# Lancement du pipeline complet
%cd /content/drive/MyDrive/ewe-french_translator
!bash scripts/prepare_data.sh
!bash scripts/train.sh
```

> Stocker `HF_TOKEN` et `WANDB_API_KEY` dans les **Secrets Colab** (icône clé dans le panneau gauche), pas en clair dans le notebook.

---

## Responsible AI

### Biais de représentation

L'éwé représente moins de 0.01% des données d'entraînement des grands modèles de langage (Joshi et al., 2020). Cette sous-représentation structurelle implique que même NLLB, conçu pour les langues à faibles ressources, présente des performances nettement inférieures sur l'éwé comparé aux langues bien dotées.

### Biais de domaine

Les corpus éwé disponibles sur-représentent les textes religieux (traductions bibliques). Le fine-tuning sur AfroLingu-MT et MAFAND (domaine news) corrige partiellement ce biais, mais les traductions de textes hors-domaine restent moins fiables.

### Variations dialectales

L'éwé possède plusieurs dialectes (Anlo, Ho, Kpando) avec des variations orthographiques significatives. L'orthographe n'est pas entièrement standardisée, ce qui peut affecter la qualité des traductions selon le dialecte d'entrée.

### Licences

| Ressource | Licence | Usage autorisé |
|-----------|---------|----------------|
| NLLB-200-distilled-600M | CC-BY-NC-4.0 | Non-commercial |
| AfroLingu-MT | Non-commercial | Non-commercial |
| MAFAND | CC-BY-NC-4.0 | Non-commercial |
| Ce dépôt | MIT | Usage libre (hors données/modèles) |

> Ce projet est à usage académique et non-commercial uniquement, conformément aux licences des ressources utilisées.

---

## Structure des fichiers de données

Les fichiers de données ne sont pas inclus dans le dépôt (trop volumineux, soumis à licences). Ils sont générés localement par le pipeline :

```
data/
├── raw/
│   ├── afrolingu_train.csv       # AfroLingu-MT brut
│   ├── afrolingu_validation.csv
│   ├── afrolingu_test.csv
│   ├── mafand_train.csv          # MAFAND brut
│   ├── mafand_validation.csv
│   ├── mafand_test.csv
│   ├── merged_train.csv          # Fusion brute avant filtrage
│   ├── merged_validation.csv
│   └── merged_test.csv
└── processed/
    ├── filtered_train.csv        # Après filtrage
    ├── filtered_validation.csv
    ├── filtered_test.csv
    ├── train.csv                 # Splits finaux (utilisés par trainer.py)
    ├── val.csv
    └── test.csv
```

---

## Références

```bibtex
@inproceedings{elmadany2024toucan,
  title     = {Toucan: Many-to-Many Translation for 150 African Language Pairs},
  author    = {Elmadany, Abdelrahim and Adebara, Ife and Abdul-Mageed, Muhammad},
  booktitle = {Findings of the Association for Computational Linguistics ACL 2024},
  pages     = {13189--13206},
  year      = {2024}
}

@inproceedings{adelani2022mafand,
  title     = {A Few Thousand Translations Go a Long Way! Leveraging Pre-trained
               Models for African News Translation},
  author    = {Adelani, David Ifeoluwa and others},
  booktitle = {Proceedings of NAACL 2022},
  year      = {2022}
}

@article{costa2022nllb,
  title   = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author  = {Costa-jussà, Marta R. and others},
  journal = {arXiv preprint arXiv:2207.04672},
  year    = {2022}
}

@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}

@inproceedings{joshi2020state,
  title     = {The State and Fate of Linguistic Diversity and Inclusion
               in the NLP World},
  author    = {Joshi, Pratik and others},
  booktitle = {Proceedings of ACL 2020},
  year      = {2020}
}
```

---

## Auteur

**Kodjo Jean DEGBEVI**  
[GitHub](https://github.com/kjd-dktech) · [HuggingFace](https://huggingface.co/kjd-dktech)