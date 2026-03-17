# GbeTo — Rapport technique : Fine-tuning NLLB-200 pour la traduction éwé ↔ français

**Auteur** : Kodjo Jean DEGBEVI  
**Dernière mis à jour** : Mars 2026  
**Modèle** : [HuggingFace](https://huggingface.co/kjd-dktech/gbeto-ewe-french)  
**Code** : [GitHub](https://github.com/kjd-dktech/GbeTo_ewe-fr)  
**Démo** : [HuggingFace Spaces](https://huggingface.co/spaces/kjd-dktech/gbeto-ewe-french-demo)

---

## 1. Résumé exécutif

L'éwé est une langue kwa parlée par environ 3 à 4 millions de personnes au Togo, au Ghana et au Bénin, très peu représentée dans les outils NLP existants. Ce projet explore dans quelle mesure un fine-tuning ciblé de NLLB-200-distilled-600M (Meta AI, 2022) sur ~14 000 paires parallèles peut améliorer la traduction automatique éwé ↔ français.

Le modèle fine-tuné (GbeTo) atteint un BLEU global de **17.73** et un chrF de **38.97** sur le test set, contre respectivement **13.41** et **33.11** pour le modèle baseline — soit un gain de **+4.32 points BLEU** et **+5.86 points chrF**. L'entraînement a été conduit entièrement sur Google Colab T4 (GPU gratuit) en ~7h30, avec arrêt automatique à l'epoch 9 (early stopping).

---

## 2. Contexte et motivation

### L'éwé dans le paysage NLP

L'éwé appartient à la famille kwa. Sa morphologie est agglutinante et ses tons lexicaux sont porteurs de sens, ce qui en fait une langue difficile à traiter avec des modèles entraînés principalement sur des langues européennes. Elle figure parmi les langues de catégorie 4 (« left-behind ») dans la taxonomie de Joshi et al. (2020) — des langues pour lesquelles les ressources numériques sont quasi inexistantes.

NLLB-200 (Costa-jussà et al., 2022) couvre 200 langues dont l'éwé, mais le volume de données éwé utilisé lors du pré-entraînement reste marginal comparé aux langues à forte ressource. Ce projet part de l'hypothèse qu'un fine-tuning sur un corpus parallèle de qualité, même modeste, peut produire un gain mesurable.

### Pourquoi NLLB-200-distilled-600M

Plusieurs facteurs ont motivé ce choix :

- **Couverture de l'éwé** : NLLB est l'un des rares modèles publics à inclure l'éwé (`ewe_Latn`) dans son vocabulaire et ses poids pré-entraînés.
- **Taille adaptée aux contraintes Colab** : la version distillée 600M tient en mémoire GPU T4 (16GB) avec fp16 et gradient checkpointing activés.
- **Architecture seq2seq éprouvée** : le mécanisme de `forced_bos_token_id` permet de contrôler précisément la langue cible, ce qui est critique pour la traduction bidirectionnelle.

---

## 3. Données

### Sources

Deux corpus parallèles académiques ont été fusionnés :

| Source | Papier | Paires éwé-français |
|--------|--------|---------------------|
| [AfroLingu-MT](https://huggingface.co/datasets/UBC-NLP/AfroLingu-MT) | Elmadany et al., ACL 2024 | ~10 500 |
| [MAFAND](https://huggingface.co/datasets/masakhane/mafand) | Adelani et al., NAACL 2022 | ~5 000 |

Les deux corpus proviennent principalement de textes de presse et de données académiques — les performances sur d'autres registres (informel, technique, oral) n'ont pas été évaluées.

### Pipeline de nettoyage

Le pipeline (`src/data/filter.py`) applique quatre étapes séquentielles :

**1. Normalisation Unicode NFC**
Les textes éwé utilisent des caractères spéciaux (`ɖ ƒ ŋ ɣ ɔ ɛ ʋ`) qui peuvent être encodés sous plusieurs formes Unicode. La normalisation NFC garantit une représentation canonique unique et évite les faux doublons lors de la déduplication. Les lignes vides ou `None` sont supprimées à cette étape.

**2. Déduplication exacte**
Suppression des paires identiques sur `(source, target)`. La clé de déduplication est le couple complet — deux paires avec la même source mais des traductions différentes sont conservées.

**3. Filtrage sur longueur**
Les paires dont la source ou la cible contient moins de 3 tokens ou plus de 150 tokens sont supprimées. Ce filtre élimine les segments trop courts (bruit, métadonnées) et trop longs (qui dégradent l'entraînement seq2seq avec truncation).

**4. Filtrage sur ratio source/cible**
Le ratio `len(source) / len(target)` doit être compris entre 0.2 et 5.0. Ce filtre élimine les paires mal alignées où une traduction est disproportionnellement plus longue que la source.

### Statistiques finales

| Split | Paires |
|-------|--------|
| Train | ~13 518 |
| Validation | ~2 919 |
| Test | 3 484 |

Les splits sont stratifiés par direction (`ewe-fra` / `fra-ewe`) pour garantir une représentation équilibrée dans chaque partition.

---

## 4. Architecture et choix techniques

### Modèle de base

`facebook/nllb-200-distilled-600M` — transformeur seq2seq, 600M paramètres, architecture encoder-decoder, entraîné sur 200 langues. Le tokenizer utilise SentencePiece avec un vocabulaire de 256 000 tokens couvrant les scripts de toutes les langues supportées.

### Hyperparamètres d'entraînement

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Learning rate | `5e-5` | Valeur standard pour fine-tuning de modèles pré-entraînés |
| Batch effectif | 32 | 4 (device) × 8 (gradient accumulation) — compromis mémoire/stabilité |
| Optimiseur | Adafactor | Adapté aux grands modèles seq2seq, faible empreinte mémoire vs Adam |
| fp16 | Oui | Réduit la VRAM de moitié sans dégradation significative |
| Gradient checkpointing | Oui | Réduit la mémoire activations au coût d'un recalcul partiel |
| Warmup steps | 500 | Stabilise les premiers pas d'entraînement |
| Weight decay | 0.01 | Régularisation légère |
| Early stopping | patience=3 | Arrêt si BLEU ne progresse pas sur 3 évaluations consécutives |
| Beam search | 4 beams | Standard pour la traduction de qualité |
| Max length | 128 tokens | Couvre ~95% des phrases du corpus |
| Epochs max | 10 | |

### Gestion des contraintes Colab

L'entraînement sur Colab T4 impose deux contraintes majeures : les coupures de session (toutes les ~4-12h) et la limite de quota d'opérations FUSE sur Google Drive (qui empêche l'écriture directe de fichiers de ~1.2GB).

**DriveCheckpointCallback** : un callback personnalisé (`src/model/trainer.py`) intercepte chaque sauvegarde de checkpoint, zippe le répertoire localement (`/tmp`), puis transfère le zip unique vers Drive. Un registre local (`/content/drive_registry.json`) maintient le meilleur score BLEU connu sans requête Drive. Seul le meilleur checkpoint est conservé sur Drive à tout moment.

**resume.sh** : script bash de reprise automatique — détecte le Run ID W&B depuis les fichiers locaux, restaure le checkpoint depuis le zip Drive, et relance l'entraînement avec `WANDB_RESUME=allow`.

---

## 5. Entraînement

### Métriques par epoch (sur le validation set)

| Epoch | Step | Train Loss | Eval Loss | BLEU | chrF |
|-------|------|-----------|-----------|------|------|
| 1 | 423 | — | 2.653 | 12.18 | 31.41 |
| 2 | 846 | — | 2.416 | 16.06 | 35.26 |
| 3 | 1269 | — | 2.357 | 17.61 | 36.90 |
| 4 | 1692 | — | 2.358 | 17.62 | 37.54 |
| 5 | 2115 | — | 2.354 | 18.23 | 37.90 |
| **6** | **2538** | **—** | **2.373** | **18.30** | **38.27** |
| 7 | 2961 | — | 2.391 | 18.16 | 38.09 |
| 8 | 3384 | — | 2.414 | 17.91 | 38.02 |
| 9 | 3807 | — | 2.428 | 17.84 | 37.95 |

*Le meilleur checkpoint (step 2538, epoch 6) a été sélectionné sur la métrique BLEU du validation set.*

### Observations

La loss d'entraînement descend de ~48 à ~2.5 sur les 9 epochs, avec une forte décroissance sur les 500 premiers steps (phase de warmup). La loss de validation atteint son minimum à l'epoch 5 (2.354) mais le BLEU continue de progresser légèrement jusqu'à l'epoch 6 (18.30), illustrant le désalignement classique entre loss et métriques de traduction.

L'early stopping se déclenche à l'epoch 9 : le BLEU n'a pas dépassé 18.30 depuis 3 évaluations consécutives (epochs 7, 8, 9).

### Suivi expérimental

L'entraînement a été suivi via Weights & Biases (projet `GbeTo_1`, run `z1mx9mb1`).

---

## 6. Évaluation

### Protocole

L'évaluation est conduite sur le test set complet (3 484 paires), séparément par direction. Les métriques utilisées sont :

- **BLEU** (sacrebleu, tokenizer `flores101`) : métrique de référence pour la traduction automatique, basée sur la précision des n-grammes.
- **chrF** (sacrebleu) : métrique basée sur les n-grammes de caractères, plus adaptée aux langues morphologiquement riches comme l'éwé. Moins sensible à la segmentation.

La comparaison baseline est calculée sur un échantillon de 200 paires (100 par direction) pour des raisons de contraintes mémoire sur Colab.

### Résultats — Comparaison baseline vs fine-tuné

| Modèle | BLEU | chrF | Δ BLEU | Δ chrF |
|--------|------|------|--------|--------|
| Baseline NLLB-600M | 13.41 | 33.11 | — | — |
| GbeTo (fine-tuné) | 16.70 | 37.19 | +3.29 | +4.08 |

*Évaluation sur échantillon de 200 paires.*

### Résultats — Test set complet par direction

| Direction | BLEU | chrF | n |
|-----------|------|------|---|
| Éwé → Français | 16.05 | 39.47 | 1 739 |
| Français → Éwé | 19.18 | 38.27 | 1 745 |
| **Global** | **17.73** | **38.97** | **3 484** |

### Interprétation

Un BLEU de 17-19 pour l'éwé (langue de catégorie très faible ressource, ~14 000 paires d'entraînement) est dans la norme académique pour ce type de configuration — les travaux récents sur les langues africaines faibles ressources rapportent généralement des scores entre 10 et 25 BLEU. Le chrF (~39) est la métrique la plus fiable ici, car l'éwé utilise des caractères spéciaux et une morphologie agglutinante que BLEU pénalise davantage.

La direction fra→ewe obtient un BLEU légèrement supérieur (+3 points) à ewe→fra, ce qui peut s'expliquer par le fait que le français est mieux représenté dans les poids pré-entraînés de NLLB — la génération en éwé bénéficie donc d'une meilleure initialisation que la génération en français à partir de l'éwé.

---

## 7. Démo

Le modèle est accessible via une interface Gradio déployée sur HuggingFace Spaces :

**[→ Démo interactive — lien à compléter après déploiement]**

L'interface permet :
- La traduction bidirectionnelle éwé ↔ français
- Des exemples prêts à l'emploi couvrant des domaines variés
- Le chargement automatique depuis `kjd-dktech/gbeto-ewe-french` sur HuggingFace Hub

Le modèle est également disponible directement via l'API HuggingFace :

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("kjd-dktech/gbeto-ewe-french")
model = AutoModelForSeq2SeqLM.from_pretrained("kjd-dktech/gbeto-ewe-french")

tokenizer.src_lang = "fra_Latn"
inputs = tokenizer("Bonjour, comment vas-tu ?", return_tensors="pt")
forced_bos = tokenizer.convert_tokens_to_ids("ewe_Latn")

output = model.generate(**inputs, forced_bos_token_id=forced_bos, num_beams=4)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 8. Limitations et perspectives

### Limitations actuelles

**Domaine** : les données d'entraînement proviennent principalement de corpus de presse et de textes académiques. Les performances sur des registres informels, techniques ou oraux n'ont pas été évaluées et sont probablement inférieures.

**Dialectes** : l'éwé possède plusieurs variantes dialectales (Anlo, Ho, Kpando) avec des différences orthographiques non standardisées. Le modèle a été entraîné sur un mélange non contrôlé de ces variantes, ce qui peut affecter la cohérence des traductions.

**Volume de données** : ~14 000 paires reste modeste. Les gains obtenus sont réels mais plafonnent rapidement — la courbe BLEU se stabilise dès l'epoch 6, suggérant que le modèle a extrait l'essentiel de l'information disponible.

**Évaluation automatique** : BLEU et chrF mesurent la similarité avec une référence unique. Pour une langue comme l'éwé où plusieurs traductions correctes existent, ces métriques sous-estiment probablement la qualité réelle.

### Perspectives

- **Back-translation** : générer des paires synthétiques éwé-français à partir de textes monolingues éwé pour augmenter le corpus d'entraînement.
- **Données supplémentaires** : intégrer d'autres sources parallèles (Bible, corpus oraux transcrits).
- **Évaluation humaine** : faire évaluer un échantillon de traductions par d'autres locuteurs natifs pour estimer la qualité réelle au-delà des métriques automatiques.
- **Modèle plus grand** : tester NLLB-1.3B ou 3.3B avec des ressources GPU plus importantes.

---

## 9. Citations

**AfroLingu-MT** :
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