#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

REPO_URL="https://github.com/ManifoldsTeam/woodbench.git"

HF_TASK_A_REPO="ny0g507h3p/wood-task-A"

PROJECT_ROOT="/workspace/woodbench"
DATA_DIR="${PROJECT_ROOT}/data/task-A"

# =============================================================================
# Colors
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERR]${NC}  $*" >&2; }
log_step()  { echo -e "\n${CYAN}==== $* ====${NC}"; }

# =============================================================================
# Prompt helpers
# =============================================================================

prompt_password() {
    local prompt="$1"
    local var_name="$2"
    local value

    read -rsp "$(echo -e "${CYAN}${prompt}${NC}: ")" value
    echo

    printf -v "$var_name" "%s" "$value"
}

# =============================================================================
# Step 1 — Check dependencies
# =============================================================================

check_dependencies() {
    log_step "Checking Dependencies"

    local missing=0

    for cmd in git python pip; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Missing dependency: $cmd"
            missing=1
        else
            log_info "$cmd found"
        fi
    done

    if [[ $missing -eq 1 ]]; then
        log_error "Install missing dependencies first."
        exit 1
    fi
}

# =============================================================================
# Step 2 — Clone or update repo
# =============================================================================

clone_repo() {
    log_step "Clone / Update Repository"

    if [[ -d "${PROJECT_ROOT}/.git" ]]; then
        log_info "Repository already exists."

        git -C "${PROJECT_ROOT}" pull

        return 0
    fi

    echo "GitHub token optional for public repo."
    echo ""

    prompt_password "GitHub token (press enter to skip)" GITHUB_TOKEN

    local remote_url="$REPO_URL"

    if [[ -n "${GITHUB_TOKEN}" ]]; then
        remote_url="${REPO_URL/https:\/\/github.com\//https:\/\/oauth2:${GITHUB_TOKEN}@github.com/}"
    fi

    git clone "$remote_url" "${PROJECT_ROOT}"

    log_info "Repository cloned."
}

# =============================================================================
# Step 3 — Install project
# =============================================================================

install_project() {
    log_step "Installing Project"

    cd "${PROJECT_ROOT}"

    python -m pip install --upgrade pip

    if [[ -f "requirements.txt" ]]; then
        log_info "Installing requirements.txt..."
        pip install -r requirements.txt
    fi

    log_info "Installing project in editable mode..."
    pip install -e .

    log_info "Project installed."
}

# =============================================================================
# Step 4 — Install HuggingFace tools
# =============================================================================

install_hf_tools() {
    log_step "Installing HuggingFace Tools"

    pip install -U huggingface_hub hf_transfer

    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_XET_HIGH_PERFORMANCE=1

    log_info "HF tools installed."
}

# =============================================================================
# Step 5 — Login HuggingFace
# =============================================================================

login_huggingface() {
    log_step "HuggingFace Login"

    if command -v hf &>/dev/null; then
        log_info "HF CLI found."
    else
        log_error "'hf' CLI not found after install."
        exit 1
    fi

    echo "Get token at:"
    echo "https://huggingface.co/settings/tokens"
    echo ""

    prompt_password "HF token" HF_TOKEN

    if [[ -z "${HF_TOKEN}" ]]; then
        log_error "HF token required."
        exit 1
    fi

    hf auth login --token "${HF_TOKEN}"

    log_info "HF login successful."
}

# =============================================================================
# Step 6 — Download dataset
# =============================================================================

download_dataset() {
    log_step "Downloading Dataset"

    mkdir -p "${DATA_DIR}"

    hf download \
        "${HF_TASK_A_REPO}" \
        --repo-type dataset \
        --local-dir "${DATA_DIR}"

    log_info "Dataset downloaded to:"
    echo "  ${DATA_DIR}"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo -e "${BOLD}"
    echo "=================================================="
    echo "  woodbench — Remote Server Setup"
    echo "=================================================="
    echo -e "${NC}"

    echo "Project root : ${PROJECT_ROOT}"
    echo "Dataset path : ${DATA_DIR}"
    echo ""

    check_dependencies

    clone_repo

    install_project

    install_hf_tools

    login_huggingface

    download_dataset

    log_step "Setup Complete"

    echo ""
    echo "Project : ${PROJECT_ROOT}"
    echo "Dataset : ${DATA_DIR}"
    echo ""

    log_info "Ready to train."
}

main "$@"