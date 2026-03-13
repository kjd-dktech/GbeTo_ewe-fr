#!/usr/bin/env bash
# =============================================================================
# Lance le fine-tuning de NLLB-200-distilled-600M sur les paires éwé-français.
#
# Comportement :
#   - Si outputs/checkpoints/ contient des checkpoints → REPRISE automatique
#   - Sinon → NOUVEAU DÉPART depuis le début
#
# Prérequis :
#   - Pipeline de données exécuté : bash scripts/prepare_data.sh
#   - data/processed/train.csv et val.csv présents
#   - HF_TOKEN et WANDB_API_KEY définis dans .env
#   - GPU disponible (T4 16GB minimum recommandé)
#
# Utilisation :
#   bash scripts/train.sh              # Paramètres par défaut
#   bash scripts/train.sh --epochs 5   # Surcharger un paramètre
#   bash scripts/train.sh --no_fp16    # Désactiver fp16 (CPU ou GPU sans support)
#
# Auteur : Kodjo Jean DEGBEVI
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Couleurs pour les logs
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

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
# Hyperparamètres par défaut
# ---------------------------------------------------------------------------
TRAIN_FILE="data/processed/train.csv"
VAL_FILE="data/processed/val.csv"
OUTPUT_DIR="/content/checkpoints_local"
DRIVE_CHECKPOINT_DIR="outputs/checkpoints"
FINAL_DIR="outputs/final_model"
HF_REPO_ID="kjd-dktech/gbeto-ewe-french"
NO_PUSH_TO_HUB=""
MODEL_NAME="facebook/nllb-200-distilled-600M"
LEARNING_RATE="5e-5"
BATCH_SIZE="4"
GRAD_ACCUM="8"
EPOCHS="10"
WARMUP_STEPS="500"
WARMUP_EPOCHS="3"
KEEP_LOCAL="3"
NUM_BEAMS="4"
SEED="42"
REPORT_TO="wandb"
NO_FP16=""

# Parsing des arguments optionnels
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_file)           TRAIN_FILE="$2";           shift 2 ;;
        --val_file)             VAL_FILE="$2";             shift 2 ;;
        --output_dir)           OUTPUT_DIR="$2";           shift 2 ;;
        --drive_checkpoint_dir) DRIVE_CHECKPOINT_DIR="$2"; shift 2 ;;
        --final_dir)            FINAL_DIR="$2";            shift 2 ;;
        --warmup_epochs)        WARMUP_EPOCHS="$2";        shift 2 ;;
        --keep_local)           KEEP_LOCAL="$3";           shift 2 ;;
        --hf_repo_id)           HF_REPO_ID="$2";           shift 2 ;;
        --no_push_to_hub)       NO_PUSH_TO_HUB="--no_push_to_hub"; shift 1 ;;
        --learning_rate)        LEARNING_RATE="$2";        shift 2 ;;
        --batch_size)           BATCH_SIZE="$2";           shift 2 ;;
        --grad_accum)           GRAD_ACCUM="$2";           shift 2 ;;
        --epochs)               EPOCHS="$2";               shift 2 ;;
        --warmup_steps)         WARMUP_STEPS="$2";         shift 2 ;;
        --num_beams)            NUM_BEAMS="$2";            shift 2 ;;
        --seed)                 SEED="$2";                 shift 2 ;;
        --report_to)            REPORT_TO="$2";            shift 2 ;;
        --no_fp16)              NO_FP16="--no_fp16";       shift 1 ;;
        *)
            log_error "Argument inconnu : $1"
            echo "Usage : bash scripts/train.sh [--epochs N] [--batch_size N] ..."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Vérifications préalables
# ---------------------------------------------------------------------------

log_step "VÉRIFICATIONS PRÉALABLES"

# 1. Racine du projet
if [[ ! -f "requirements.txt" ]]; then
    log_error "Ce script doit être lancé depuis la racine du projet."
    exit 1
fi
log_success "Racine du projet détectée."

# 2. Charger le .env
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
    log_success "Fichier .env chargé."
else
    log_warning "Fichier .env introuvable."
fi

# 3. Vérifier les fichiers de données
if [[ ! -f "${TRAIN_FILE}" ]]; then
    log_error "Fichier d'entraînement introuvable : ${TRAIN_FILE}"
    log_error "Lancez d'abord : bash scripts/prepare_data.sh"
    exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
    log_error "Fichier de validation introuvable : ${VAL_FILE}"
    log_error "Lancez d'abord : bash scripts/prepare_data.sh"
    exit 1
fi
log_success "Fichiers de données vérifiés."

# 4. Compter les paires
TRAIN_LINES=$(( $(wc -l < "${TRAIN_FILE}") - 1 ))
VAL_LINES=$(( $(wc -l < "${VAL_FILE}") - 1 ))
log_info "Train : ${TRAIN_LINES} paires"
log_info "Val   : ${VAL_LINES} paires"

# 5. Vérifier GPU (avertissement uniquement, pas bloquant)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "inconnu")
    VRAM=$(python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))" 2>/dev/null || echo "?")
    log_success "GPU détecté : ${GPU_NAME} (${VRAM}GB VRAM)"
else
    log_warning "Aucun GPU détecté — entraînement sur CPU (très lent)."
    log_warning "Ajoutez --no_fp16 si vous êtes sur CPU."
fi

# 6. Vérifier WANDB si report_to=wandb
if [[ "${REPORT_TO}" == "wandb" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        log_warning "WANDB_API_KEY non défini — report_to basculé sur 'none'."
        REPORT_TO="none"
    else
        log_success "WANDB_API_KEY détecté."
    fi
fi

# ---------------------------------------------------------------------------
# Détection checkpoint existant
# ---------------------------------------------------------------------------

log_step "DÉTECTION DU MODE D'ENTRAÎNEMENT"

CHECKPOINT_EXISTS=false
if [[ -d "${OUTPUT_DIR}" ]]; then
    if ls "${OUTPUT_DIR}"/checkpoint-* 1> /dev/null 2>&1; then
        CHECKPOINT_EXISTS=true
        LAST_CHECKPOINT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
        log_warning "Checkpoints locaux détectés dans ${OUTPUT_DIR}/"
        log_warning "→ REPRISE depuis : ${LAST_CHECKPOINT}"
    fi
fi

# Vérifier aussi si un zip Drive existe (reprise possible via registry)
if [[ "${CHECKPOINT_EXISTS}" == "false" ]]; then
    if ls "${DRIVE_CHECKPOINT_DIR}"/checkpoint-*.zip 1> /dev/null 2>&1; then
        log_warning "Zip Drive détecté dans ${DRIVE_CHECKPOINT_DIR}/"
        log_warning "→ Reprise depuis Drive possible (via registry)"
        CHECKPOINT_EXISTS=true
    fi
fi

if [[ "${CHECKPOINT_EXISTS}" == "false" ]]; then
    log_info "Aucun checkpoint détecté → NOUVEAU DÉPART"
fi

# ---------------------------------------------------------------------------
# Affichage de la configuration complète
# ---------------------------------------------------------------------------

log_step "CONFIGURATION DE L'ENTRAÎNEMENT"

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))

echo ""
log_info "Modèle           : ${MODEL_NAME}"
log_info "Learning rate    : ${LEARNING_RATE}"
log_info "Batch / device   : ${BATCH_SIZE}"
log_info "Gradient accum   : ${GRAD_ACCUM}  →  batch effectif : ${EFFECTIVE_BATCH}"
log_info "Epochs max       : ${EPOCHS}"
log_info "Warmup steps     : ${WARMUP_STEPS}"
log_info "Warmup epochs    : ${WARMUP_EPOCHS} (pas de Drive avant cette epoch)"
log_info "Keep local       : ${KEEP_LOCAL} meilleurs checkpoints locaux"
log_info "Beam search      : ${NUM_BEAMS}"
log_info "Seed             : ${SEED}"
log_info "fp16             : $([ -z "${NO_FP16}" ] && echo 'oui' || echo 'non')"
log_info "Tracking         : ${REPORT_TO}"
log_info "Checkpoints      : ${OUTPUT_DIR} (local) → ${DRIVE_CHECKPOINT_DIR} (Drive, zip)"
echo ""

# ---------------------------------------------------------------------------
# Lancement de l'entraînement
# ---------------------------------------------------------------------------

log_step "LANCEMENT DE L'ENTRAÎNEMENT"

START_TIME=$(date +%s)
log_info "Début : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python -m src.model.trainer \
    --train_file            "${TRAIN_FILE}"            \
    --val_file              "${VAL_FILE}"              \
    --output_dir            "${OUTPUT_DIR}"            \
    --drive_checkpoint_dir  "${DRIVE_CHECKPOINT_DIR}"  \
    --final_dir             "${FINAL_DIR}"             \
    --warmup_epochs         "${WARMUP_EPOCHS}"         \
    --keep_local            "${KEEP_LOCAL}"            \
    --hf_repo_id            "${HF_REPO_ID}"            \
    --learning_rate         "${LEARNING_RATE}"         \
    --batch_size            "${BATCH_SIZE}"            \
    --grad_accum            "${GRAD_ACCUM}"            \
    --epochs                "${EPOCHS}"                \
    --warmup_steps          "${WARMUP_STEPS}"          \
    --num_beams             "${NUM_BEAMS}"             \
    --seed                  "${SEED}"                  \
    --report_to             "${REPORT_TO}"             \
    ${NO_FP16}                                         \
    ${NO_PUSH_TO_HUB}

# ---------------------------------------------------------------------------
# Bilan
# ---------------------------------------------------------------------------

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo -e "${BOLD}${GREEN}================================================================${RESET}"
echo -e "${BOLD}${GREEN}  ENTRAÎNEMENT TERMINÉ${RESET}"
echo -e "${BOLD}${GREEN}================================================================${RESET}"
echo ""
log_success "Durée totale : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
log_info "Meilleur modèle : ${FINAL_DIR}"
log_info "Métriques       : outputs/final_metrics.json"
if [[ "${REPORT_TO}" == "wandb" ]]; then
    log_info "Dashboard W&B   : https://wandb.ai"
fi
echo ""
log_info "Étape suivante : python app.py"
echo ""