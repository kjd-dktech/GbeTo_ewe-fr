#!/usr/bin/env bash
# =============================================================================
# Reprise automatique du fine-tuning GbeTo.
#
# Comportement :
#   1. Vérifie si l'entraînement est déjà terminé (modèle final présent)
#   2. Détecte le Run ID W&B depuis le dernier dossier wandb/run-*
#   3. Vérifie qu'un checkpoint local ou un zip Drive est disponible
#   4. Injecte WANDB_RUN_ID + WANDB_RESUME=allow et lance train.sh
#
# Utilisation dans le notebook :
#   !bash scripts/resume.sh
#
# Auteur : Kodjo Jean DEGBEVI
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

log_ok()   { echo -e "${GREEN}✅  $*${RESET}"; }
log_warn() { echo -e "${YELLOW}⚠️   $*${RESET}"; }
log_info() { echo -e "${BLUE}🔁  $*${RESET}"; }
log_err()  { echo -e "${RED}❌  $*${RESET}" >&2; }

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
WANDB_DIR="wandb"
LOCAL_CKPT="/content/checkpoints_local"
DRIVE_CKPT="outputs/checkpoints"
FINAL_DIR="outputs/final_model"

# ---------------------------------------------------------------------------
# 1. Entraînement déjà terminé ?
# ---------------------------------------------------------------------------
if [[ -d "${FINAL_DIR}" ]] && [[ -n "$(ls -A "${FINAL_DIR}" 2>/dev/null)" ]]; then
    log_ok "Entraînement déjà terminé — modèle final présent dans ${FINAL_DIR}."
    echo "   Aucune reprise nécessaire."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Détection du Run ID W&B
# ---------------------------------------------------------------------------
if [[ ! -d "${WANDB_DIR}" ]]; then
    log_warn "Dossier wandb/ introuvable — lancez d'abord le train (cellule précédente)."
    exit 1
fi

# Dernier dossier run-* trié par date de modification
LAST_RUN_DIR=$(ls -dt "${WANDB_DIR}"/run-* 2>/dev/null | head -1 || true)

if [[ -z "${LAST_RUN_DIR}" ]]; then
    log_warn "Aucun run W&B trouvé — lancez d'abord le train (cellule précédente)."
    exit 1
fi

LAST_RUN_NAME=$(basename "${LAST_RUN_DIR}")

# Extraire le Run ID : run-YYYYMMDD_HHMMSS-<run_id>
RUN_ID=$(echo "${LAST_RUN_NAME}" | grep -oP '(?<=run-\d{8}_\d{6}-)\w+' || true)

if [[ -z "${RUN_ID}" ]]; then
    log_warn "Impossible d'extraire le Run ID depuis : ${LAST_RUN_NAME}"
    exit 1
fi

log_info "Run ID détecté : ${RUN_ID}"

# Exporter pour que train.sh (et le processus Python) les hérite
export WANDB_RESUME="allow"
export WANDB_RUN_ID="${RUN_ID}"

# ---------------------------------------------------------------------------
# 3. Vérification : checkpoint disponible ?
# ---------------------------------------------------------------------------
LOCAL_FOUND=""
DRIVE_FOUND=""

if [[ -d "${LOCAL_CKPT}" ]]; then
    LOCAL_FOUND=$(ls -d "${LOCAL_CKPT}"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1 || true)
fi

if [[ -d "${DRIVE_CKPT}" ]]; then
    DRIVE_FOUND=$(ls "${DRIVE_CKPT}"/checkpoint-*.zip 2>/dev/null | sort | tail -1 || true)
fi

if [[ -z "${LOCAL_FOUND}" && -z "${DRIVE_FOUND}" ]]; then
    log_warn "Aucun checkpoint local ni zip Drive trouvé — rien à reprendre."
    exit 1
fi

[[ -n "${LOCAL_FOUND}" ]] && echo -e "📂  Checkpoint local : $(basename "${LOCAL_FOUND}")"
[[ -n "${DRIVE_FOUND}" ]] && echo -e "☁️   Zip Drive        : $(basename "${DRIVE_FOUND}")"

echo ""
echo -e "${BLUE}🚀  Lancement de la reprise (trainer.py gère la restauration)...${RESET}"
echo ""
sleep 2

# ---------------------------------------------------------------------------
# 4. Reprise
# ---------------------------------------------------------------------------
bash scripts/train.sh