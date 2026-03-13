#!/usr/bin/env bash
# =============================================================================
# Pipeline complet de préparation des données en une seule commande.
#
# Enchaîne dans l'ordre :
#   1. Téléchargement et fusion   (src/data/download.py)
#   2. Nettoyage et filtrage      (src/data/filter.py)
#   3. Finalisation des splits    (src/data/split.py)
#
# Résultat : data/processed/train.csv, val.csv, test.csv
#
# Prérequis :
#   - HF_TOKEN_READ défini dans .env ou en variable d'environnement
#   - Conditions AfroLingu-MT acceptées sur HuggingFace
#   - Dépendances installées : pip install -r requirements.txt
#
# Utilisation :
#   bash scripts/prepare_data.sh
#   bash scripts/prepare_data.sh --output_dir data/raw --processed_dir data/processed
#
# Auteur : Kodjo Jean DEGBEVI
# =============================================================================

set -euo pipefail
# set -e : arrêt immédiat si une commande échoue
# set -u : erreur si variable non définie utilisée
# set -o pipefail : un pipe échoue si n'importe quelle commande du pipe échoue

# ---------------------------------------------------------------------------
# Couleurs pour les logs
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

log_info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
log_success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warning() { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

log_step() {
    echo ""
    echo -e "${BOLD}${BLUE}================================================================${RESET}"
    echo -e "${BOLD}${BLUE}  $*${RESET}"
    echo -e "${BOLD}${BLUE}================================================================${RESET}"
}

# ---------------------------------------------------------------------------
# Arguments par défaut (surchargeables en ligne de commande)
# ---------------------------------------------------------------------------
OUTPUT_DIR="data/raw"
PROCESSED_DIR="data/processed"

# Parsing des arguments optionnels
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --processed_dir) PROCESSED_DIR="$2"; shift 2 ;;
        *)
            log_error "Argument inconnu : $1"
            echo "Usage : bash scripts/prepare_data.sh [--output_dir <path>] [--processed_dir <path>]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Vérifications préalables
# ---------------------------------------------------------------------------

log_step "VÉRIFICATIONS PRÉALABLES"

# 1. Être à la racine du projet
if [[ ! -f "requirements.txt" ]]; then
    log_error "Ce script doit être lancé depuis la racine du projet."
    log_error "Usage : bash scripts/prepare_data.sh"
    exit 1
fi
log_success "Racine du projet détectée."

# 2. Charger le .env si présent
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
    log_success "Fichier .env chargé."
else
    log_warning "Fichier .env introuvable — HF_TOKEN doit être défini en variable d'environnement."
fi

# 3. Vérifier HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
    log_error "HF_TOKEN non défini."
    log_error "Ajoutez HF_TOKEN=hf_xxx dans le fichier .env à la racine du projet."
    exit 1
fi
log_success "HF_TOKEN détecté."

# 4. Vérifier Python
if ! command -v python &> /dev/null; then
    log_error "Python introuvable. Installez Python 3.10+."
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
log_success "Python détecté : ${PYTHON_VERSION}"

# 5. Vérifier les dépendances critiques
log_info "Vérification des dépendances ..."
python -c "import datasets, pandas, transformers" 2>/dev/null || {
    log_error "Dépendances manquantes. Lancez : pip install -r requirements.txt"
    exit 1
}
log_success "Dépendances vérifiées."

# ---------------------------------------------------------------------------
# Horodatage de début
# ---------------------------------------------------------------------------
START_TIME=$(date +%s)
echo ""
log_info "Début du pipeline : $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Dossier raw       : ${OUTPUT_DIR}"
log_info "Dossier processed : ${PROCESSED_DIR}"

# ---------------------------------------------------------------------------
# Étape 1 — Téléchargement et fusion
# ---------------------------------------------------------------------------
log_step "ÉTAPE 1/3 — TÉLÉCHARGEMENT ET FUSION"
log_info "Sources : UBC-NLP/AfroLingu-MT + masakhane/mafand"

python -m src.data.download \
    --output_dir "${OUTPUT_DIR}" \
    --hf_token   "${HF_TOKEN}"

log_success "Téléchargement terminé → ${OUTPUT_DIR}/"

# ---------------------------------------------------------------------------
# Étape 2 — Filtrage
# ---------------------------------------------------------------------------
log_step "ÉTAPE 2/3 — NETTOYAGE ET FILTRAGE"
log_info "Filtres : NFC → déduplication → longueur [3-150] → ratio [0.2-5.0]"

python -m src.data.filter \
    --input_dir  "${OUTPUT_DIR}" \
    --output_dir "${PROCESSED_DIR}"

log_success "Filtrage terminé → ${PROCESSED_DIR}/filtered_*.csv"

# ---------------------------------------------------------------------------
# Étape 3 — Splits finaux
# ---------------------------------------------------------------------------
log_step "ÉTAPE 3/3 — FINALISATION DES SPLITS"
log_info "Shuffle reproductible avec seed=42"

python -m src.data.split \
    --input_dir  "${PROCESSED_DIR}" \
    --output_dir "${PROCESSED_DIR}"

log_success "Splits finaux → ${PROCESSED_DIR}/{train,val,test}.csv"

# ---------------------------------------------------------------------------
# Bilan final
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo -e "${BOLD}${GREEN}================================================================${RESET}"
echo -e "${BOLD}${GREEN}  PIPELINE DE DONNÉES TERMINÉ${RESET}"
echo -e "${BOLD}${GREEN}================================================================${RESET}"
echo ""
log_success "Durée totale : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
log_info "Fichiers produits :"
for f in "${PROCESSED_DIR}/train.csv" "${PROCESSED_DIR}/val.csv" "${PROCESSED_DIR}/test.csv"; do
    if [[ -f "$f" ]]; then
        LINES=$(( $(wc -l < "$f") - 1 ))  # -1 pour l'en-tête
        log_success "  ${f}  (${LINES} paires)"
    else
        log_warning "  ${f}  INTROUVABLE"
    fi
done
echo ""
log_info "Étape suivante : bash scripts/train.sh"
echo ""