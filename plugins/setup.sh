#!/bin/bash
# =============================================================================
# setup_ssh_server.sh — Standalone server setup for woodbench project
#
# Fully interactive: prompts for GitHub token and HuggingFace token.
# Run ONCE on a fresh SSH server.
#
# Usage:
#   bash scripts/setup_ssh_server.sh
#
# Options (non-interactive):
#   --full         Run full setup (system + conda + deps + repo + data)
#   --deps         Python deps only (conda env + packages)
#   --check        Check environment status
#   --help         Show this help
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

CONDA_ENV_NAME="wood"
PYTHON_VERSION="3.10"
PROJECT_ROOT="${HOME}/woodbench"
EXPERIMENTS_DIR="${HOME}/experiments/woodbench"
DATA_DIR="${PROJECT_ROOT}/data"
REPO_URL="https://github.com/ManifoldsTeam/woodbench.git"
HF_TASK_A_REPO="ny0g507h3p/wood-task-A"
HF_TASK_B_REPO="ny0g507h3p/wood-task-B"

# Color output
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
log_banner(){ echo -e "\n${BOLD}*** $* ***${NC}"; }

# Prompt helpers
prompt_yn() {
    local prompt="$1"
    local default="${2:-n}"
    local yn
    while true; do
        read -rp "$(echo -e "${CYAN}${prompt}${NC} [${default^^}]: ")" yn
        yn="${yn:-$default}"
        case "$yn" in
            y|Y) return 0 ;;
            n|N) return 1 ;;
        esac
        echo "  Please enter y or n."
    done
}

prompt_input() {
    local prompt="$1"
    local var_name="$2"
    local default="${3:-}"
    local value
    read -rp "$(echo -e "${CYAN}${prompt}${NC}${default:+, default: ${default}}: ")" value
    value="${value:-$default}"
    printf -p "$var_name" "%s" "$value"
}

prompt_password() {
    local prompt="$1"
    local var_name="$2"
    local value
    read -rsp "$(echo -e "${CYAN}${prompt}${NC}: ")" value
    echo
    printf -p "$var_name" "%s" "$value"
}

# =============================================================================
# Step 0: Interactive token setup
# =============================================================================

collect_tokens() {
    log_step "Token Configuration"

    echo "  This script needs tokens to access private resources."
    echo ""

    # GitHub token
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        log_info "Using GITHUB_TOKEN from environment."
        GITHUB_TOKEN_VAL="$GITHUB_TOKEN"
    else
        echo -e "${CYAN}1. GitHub Token${NC}"
        echo "   Needed to clone private repos (or if GitHub blocks unauthenticated git)."
        echo "   Get one at: https://github.com/settings/tokens"
        echo "   (Token classification: 'repo' for private repos, 'public_repo' for public)"
        echo ""
        if prompt_yn "   Enter GitHub token (skip for public repos)"; then
            prompt_password "   GitHub token" GITHUB_TOKEN_VAL
        else
            GITHUB_TOKEN_VAL=""
        fi
    fi

    # HuggingFace token
    if [[ -n "${HF_TOKEN:-}" ]]; then
        log_info "Using HF_TOKEN from environment."
        HF_TOKEN_VAL="$HF_TOKEN"
    else
        echo ""
        echo -e "${CYAN}2. HuggingFace Token${NC}"
        echo "   Needed to download dataset from HuggingFace Hub."
        echo "   Get one at: https://huggingface.co/settings/tokens"
        echo "   (Token type: 'Read' is sufficient for downloading)"
        echo ""
        if prompt_yn "   Enter HuggingFace token"; then
            prompt_password "   HuggingFace token" HF_TOKEN_VAL
        else
            log_warn "No HF token — public dataset download may be rate-limited."
            HF_TOKEN_VAL=""
        fi
    fi

    export HF_TOKEN="$HF_TOKEN_VAL"
}

# =============================================================================
# Step 1: Detect environment
# =============================================================================

detect_env() {
    log_step "Environment Detection"

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "not available")
    echo "  OS:           $(uname -s) $(uname -r)"
    echo "  User:         $(whoami)"
    echo "  Hostname:     $(hostname)"
    echo "  Home:         $HOME"
    echo "  CPU cores:    $(nproc)"
    echo "  RAM:          $(free -h | awk '/^Mem:/ {print $2}')"
    echo "  GPU:          $gpu_info"
    echo "  conda:        $(command -v conda &>/dev/null && conda --version || echo 'not found')"
    echo "  huggingface_hub: $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)' 2>/dev/null || echo 'not found')"
    echo "  htop:         $(command -v htop &>/dev/null && echo 'installed' || echo 'not found')"
    echo "  nvtop:        $(command -v nvtop &>/dev/null && echo 'installed' || echo 'not found')"

    local conda_envs
    conda_envs=$(conda env list 2>/dev/null | awk 'NR>1 {print $1}' || echo "")
    if echo "$conda_envs" | grep -q "^${CONDA_ENV_NAME}$"; then
        echo -e "  conda env '${CONDA_ENV_NAME}': ${GREEN}exists${NC}"
    else
        echo -e "  conda env '${CONDA_ENV_NAME}': ${YELLOW}not found${NC}"
    fi
}

# =============================================================================
# Step 2: Install system packages (requires sudo)
# =============================================================================

install_system_packages() {
    log_step "Installing System Packages (requires sudo)"

    if ! sudo -n true 2>/dev/null; then
        log_info "sudo may ask for your password..."
    fi

    local PKGS=(
        htop
        nvtop
        curl
        wget
        git
        build-essential
    )

    log_info "Updating apt cache..."
    sudo apt-get update -qq

    log_info "Installing packages: ${PKGS[*]}"
    sudo apt-get install -y "${PKGS[@]}"

    log_info "System packages installed."
}

# =============================================================================
# Step 3: Setup conda environment
# =============================================================================

setup_conda_env() {
    log_step "Setting Up Conda Environment '${CONDA_ENV_NAME}'"

    if ! command -v conda &>/dev/null; then
        log_error "conda not found. Install Miniconda first:"
        echo "  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash"
        return 1
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    local conda_prefix
    conda_prefix=$(conda env list 2>/dev/null | awk '/^'"${CONDA_ENV_NAME}"'/ {print $NF; exit}' || echo "")

    if [[ -n "$conda_prefix" && -d "$conda_prefix" ]]; then
        log_info "Conda env '${CONDA_ENV_NAME}' already exists at $conda_prefix"
        log_info "Skipping conda env creation. Use --deps to reinstall packages."
        return 0
    fi

    log_info "Creating conda env '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
    conda activate "${CONDA_ENV_NAME}"

    log_info "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    log_info "Installing huggingface_hub..."
    pip install huggingface_hub

    if [[ -f "${PROJECT_ROOT}/requirements-ssh.txt" ]]; then
        log_info "Installing project dependencies from requirements-ssh.txt..."
        pip install -r "${PROJECT_ROOT}/requirements-ssh.txt"
    else
        log_warn "requirements-ssh.txt not found. Install deps manually."
    fi

    log_info "Conda env '${CONDA_ENV_NAME}' ready."
}

# =============================================================================
# Step 4: Clone or update woodbench repo
# =============================================================================

clone_or_update_repo() {
    log_step "Cloning / Updating woodbench Repository"

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true

    if [[ -d "${PROJECT_ROOT}/.git" ]]; then
        log_info "woodbench already cloned at $PROJECT_ROOT"
        if prompt_yn "Pull latest changes?" y; then
            git -C "${PROJECT_ROOT}" pull
            log_info "Updated."
        else
            log_info "Skipped."
        fi
        return 0
    fi

    log_info "Cloning woodbench to $PROJECT_ROOT..."

    local remote_url="$REPO_URL"
    if [[ -n "$GITHUB_TOKEN_VAL" ]]; then
        # Convert https://github.com/... into token-authenticated URL
        remote_url="${REPO_URL//https:\/\/github\.com\//https://${GITHUB_TOKEN_VAL}@github.com/}"
    fi

    git clone "$remote_url" "${PROJECT_ROOT}"
    log_info "Repository cloned."
}

# =============================================================================
# Step 5: Download dataset from HuggingFace
# =============================================================================

download_dataset() {
    log_step "Downloading Dataset from HuggingFace"

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true

    if [[ ! -d "${PROJECT_ROOT}" ]]; then
        log_error "Project not found at ${PROJECT_ROOT}. Run --full first."
        return 1
    fi

    echo ""
    echo "  Available datasets:"
    echo "    1. Task A  (${HF_TASK_A_REPO}) — balanced, 7,264 train samples"
    echo "    2. Task B  (${HF_TASK_B_REPO}) — species-level, full"
    echo "    3. Both"
    echo "    4. Skip"
    echo ""

    local choice
    read -rp "$(echo -e "${CYAN}Which dataset to download?${NC} [1/2/3/4]: ")" choice
    choice="${choice:-1}"

    local hf_download="from huggingface_hub import snapshot_download; "
    hf_download+="import os; os.makedirs('${PROJECT_ROOT}/data', exist_ok=True)"

    local download_extra=""
    if [[ -n "$HF_TOKEN_VAL" ]]; then
        download_extra="token='${HF_TOKEN_VAL}'"
    fi

    case "$choice" in
        1)
            log_info "Downloading Task A dataset..."
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_TASK_A_REPO}',
    local_dir='${PROJECT_ROOT}/data/task-A'${HF_TOKEN_VAL:+, token='${HF_TOKEN_VAL}'}
)
"
            log_info "Task A dataset ready at ${PROJECT_ROOT}/data/task-A"
            ;;
        2)
            log_info "Downloading Task B dataset..."
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_TASK_B_REPO}',
    local_dir='${PROJECT_ROOT}/data/task-B'${HF_TOKEN_VAL:+, token='${HF_TOKEN_VAL}'}
)
"
            log_info "Task B dataset ready at ${PROJECT_ROOT}/data/task-B"
            ;;
        3)
            log_info "Downloading both datasets..."
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_TASK_A_REPO}',
    local_dir='${PROJECT_ROOT}/data/task-A'${HF_TOKEN_VAL:+, token='${HF_TOKEN_VAL}'}
)
"
            log_info "Task A done."
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_TASK_B_REPO}',
    local_dir='${PROJECT_ROOT}/data/task-B'${HF_TOKEN_VAL:+, token='${HF_TOKEN_VAL}'}
)
"
            log_info "Task B done."
            log_info "Datasets ready at ${PROJECT_ROOT}/data/"
            ;;
        4)
            log_info "Skipped. Download manually when needed:"
            echo "  python -c \"from huggingface_hub import snapshot_download; \\"
            echo "    snapshot_download(repo_id='${HF_TASK_A_REPO}', local_dir='data/task-A')\""
            ;;
        *)
            log_warn "Invalid choice. Skipping."
            ;;
    esac
}

# =============================================================================
# Step 6: Create directory structure
# =============================================================================

create_dirs() {
    log_step "Creating Directory Structure"

    mkdir -p "${EXPERIMENTS_DIR}"
    mkdir -p "${DATA_DIR}"

    log_info "Created:"
    echo "  $EXPERIMENTS_DIR"
    echo "  $DATA_DIR"
}

# =============================================================================
# Step 7: Verify setup
# =============================================================================

verify_setup() {
    log_step "Verifying Setup"

    local errors=0

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true

    # Conda env
    if conda env list 2>/dev/null | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Conda env '${CONDA_ENV_NAME}': OK"
    else
        log_error "Conda env '${CONDA_ENV_NAME}': not found"
        errors=$((errors + 1))
    fi

    # Python packages
    for pkg in torch timm scikit-learn huggingface_hub; do
        if python -c "import ${pkg}" 2>/dev/null; then
            local ver
            ver=$(python -c "import ${pkg}; print(${pkg}.__version__)")
            log_info "Python '${pkg}': ${ver}"
        else
            log_error "Python '${pkg}': not installed"
            errors=$((errors + 1))
        fi
    done

    # GPU
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local gpu_name
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        log_info "GPU CUDA: OK — ${gpu_name}"
    else
        log_warn "GPU CUDA: not available"
    fi

    # Directories
    for d in "$EXPERIMENTS_DIR" "$DATA_DIR"; do
        if [[ -d "$d" ]]; then
            log_info "Directory '${d}': OK"
        else
            log_warn "Directory '${d}': not found"
        fi
    done

    # Project
    if [[ -d "${PROJECT_ROOT}/.git" ]]; then
        log_info "Project repo: OK — ${PROJECT_ROOT}"
    else
        log_warn "Project repo: not found at ${PROJECT_ROOT}"
    fi

    echo ""
    if [[ $errors -eq 0 ]]; then
        log_info "Setup verification passed."
    else
        log_error "Setup verification failed with ${errors} error(s)."
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo -e "${BOLD}"
    echo "============================================================"
    echo "  woodbench — Server Setup"
    echo "============================================================"
    echo -e "${NC}"
    echo "  Target: ${PROJECT_ROOT}"
    echo "  Conda env: ${CONDA_ENV_NAME}"
    echo "  Experiments: ${EXPERIMENTS_DIR}"
    echo ""

    local MODE="${1:-interactive}"

    case "$MODE" in
        --full)
            collect_tokens
            detect_env
            install_system_packages
            create_dirs
            setup_conda_env
            clone_or_update_repo
            download_dataset
            verify_setup
            ;;

        --deps)
            create_dirs
            setup_conda_env
            verify_setup
            ;;

        --check)
            detect_env
            verify_setup
            ;;

        --help|-h)
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --full     Full setup: system + conda + deps + repo + dataset"
            echo "  --deps     Conda environment + Python packages only"
            echo "  --check    Check current environment status"
            echo "  --help     Show this help"
            echo ""
            echo "No option: Interactive mode (asks for tokens, then runs --full)"
            echo ""
            echo "Environment variables (override prompts):"
            echo "  GITHUB_TOKEN    GitHub personal access token"
            echo "  HF_TOKEN        HuggingFace user access token"
            exit 0
            ;;

        interactive|)
            collect_tokens
            detect_env

            echo ""
            if prompt_yn "Install system packages (requires sudo)?" y; then
                install_system_packages
            else
                log_info "Skipped system packages."
            fi

            create_dirs

            if prompt_yn "Setup conda environment + Python packages?" y; then
                setup_conda_env
            else
                log_info "Skipped conda/Python setup."
            fi

            if prompt_yn "Clone or update woodbench repo?" y; then
                clone_or_update_repo
            else
                log_info "Skipped repo clone."
            fi

            if prompt_yn "Download HuggingFace dataset?" y; then
                download_dataset
            else
                log_info "Skipped dataset download."
            fi

            verify_setup
            ;;

        *)
            log_error "Unknown option: $MODE"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
