#!/bin/zsh
# ──────────────────────────────────────────────────────────────────────
# Daydream CLI — Interactive Installer
# One-click setup for Apple Silicon local model inference
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
set +x 2>/dev/null || true
set +v 2>/dev/null || true
unsetopt xtrace verbose 2>/dev/null || true

# ── Colors & Styles ───────────────────────────────────────────────────
BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
CYAN=$'\033[36m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
WHITE=$'\033[37m'
HIDE_CURSOR=$'\033[?25l'
SHOW_CURSOR=$'\033[?25h'

# ── State ─────────────────────────────────────────────────────────────
INSTALL_DIR="$HOME/Daydream-cli"
PYTHON_CMD=""
HF_TOKEN=""
HF_TOKEN_CONFIGURED=false
SHELL_INTEGRATE=true
REPO_URL="https://github.com/Starry331/Daydream-cli.git"
HF_TOKEN_URL="https://huggingface.co/settings/tokens"
HF_TOKEN_RECOMMENDED_ROLE="read"
MODEL_TO_PULL=""
MODEL_DOWNLOAD_REQUESTED=false
MODEL_DOWNLOAD_COMPLETED=false
MODEL_DOWNLOAD_FAILED=false
PYTHON_CANDIDATES=()
PYTHON_VERSIONS=()

# ── Cleanup trap ──────────────────────────────────────────────────────
cleanup() {
    print -n "${SHOW_CURSOR}"
    stty sane 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Utility functions ─────────────────────────────────────────────────
ensure_interactive_terminal() {
    if [[ ! -t 0 || ! -t 1 ]]; then
        print ""
        print "  ${RED}${BOLD}This installer needs an interactive Terminal window.${RESET}"
        print ""
        print "  ${DIM}Run it like this:${RESET}"
        print ""
        print "    ${BOLD}curl -fsSL https://raw.githubusercontent.com/Starry331/Daydream-cli/main/install.sh -o /tmp/daydream-install.sh${RESET}"
        print "    ${BOLD}zsh /tmp/daydream-install.sh${RESET}"
        print ""
        print "  ${DIM}Do not run it with sh or by piping it into another shell.${RESET}"
        print ""
        exit 1
    fi
}

print_step() {
    print "  ${GREEN}✓${RESET} $1"
}

print_fail() {
    print "  ${RED}✗${RESET} $1"
}

print_warn() {
    print "  ${YELLOW}!${RESET} $1"
}

print_info() {
    print "  ${DIM}$1${RESET}"
}

# Spinner for long operations
spinner() {
    local pid=$1
    local msg=$2
    local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    local i=1
    local nframes=${#frames[@]}
    print -n "${HIDE_CURSOR}"
    while kill -0 "$pid" 2>/dev/null; do
        print -n "\r  ${CYAN}${frames[$i]}${RESET} ${DIM}${msg}${RESET}  "
        i=$(( i % nframes + 1 ))
        sleep 0.08
    done
    wait "$pid" 2>/dev/null
    local exit_code=$?
    print -n "\r\033[K"
    print -n "${SHOW_CURSOR}"
    return $exit_code
}

# Clear N lines above cursor
clear_lines() {
    local n=$1
    local i
    for (( i = 0; i < n; i++ )); do
        print -n "\r\033[A\033[2K"
    done
    print -n "\r"
}

# ── Arrow-key interactive menu ───────────────────────────────────────
# Sets global MENU_RESULT to the selected index
menu_select() {
    local title=$1
    shift
    local options=("$@")
    local selected=1  # zsh 1-indexed
    local total=${#options[@]}
    local menu_height=$(( total + 4 ))

    _menu_draw() {
        print "  ${BOLD}${title}${RESET}"
        print ""
        local i
        for (( i = 1; i <= total; i++ )); do
            if [[ $i -eq $selected ]]; then
                print "  ${BOLD}${CYAN}> ${options[$i]}${RESET}"
            else
                print "    ${DIM}${options[$i]}${RESET}"
            fi
        done
        print ""
        print "  ${DIM}↑/↓ move  Enter confirm${RESET}"
    }

    print -n "${HIDE_CURSOR}"
    _menu_draw
    while true; do
        local key
        read -rsk1 key
        case "$key" in
            $'\e')
                local seq=""
                read -rsk2 seq
                case "$seq" in
                    '[A') selected=$(( (selected - 2 + total) % total + 1 )) ;;
                    '[B') selected=$(( selected % total + 1 )) ;;
                esac
                ;;
            $'\n'|$'\r')
                clear_lines "$menu_height"
                print -n "${SHOW_CURSOR}"
                print "  ${GREEN}✓${RESET} ${BOLD}${title}${RESET}  ${DIM}${options[$selected]}${RESET}"
                MENU_RESULT=$(( selected - 1 ))  # convert to 0-indexed for array access later
                return 0
                ;;
            $'\x03')
                clear_lines "$menu_height"
                print -n "${SHOW_CURSOR}"
                print ""
                exit 1
                ;;
        esac
        clear_lines "$menu_height"
        _menu_draw
    done
}

# ── Text input with default ──────────────────────────────────────────
# Sets global INPUT_RESULT
text_input() {
    local prompt=$1
    local default=$2

    print -n "${SHOW_CURSOR}"
    if [[ -n "$default" ]]; then
        print -n "  ${BOLD}${prompt}${RESET} ${DIM}[default: ${default}]${RESET}: "
    else
        print -n "  ${BOLD}${prompt}${RESET}: "
    fi

    local input
    read -r input
    if [[ -z "$input" ]]; then
        INPUT_RESULT="$default"
    else
        INPUT_RESULT="$input"
    fi
}

# ── Secret input (for tokens) ────────────────────────────────────────
# Sets global SECRET_RESULT
secret_input() {
    local prompt=$1

    print -n "${SHOW_CURSOR}"
    print -n "  ${BOLD}${prompt}${RESET}: "
    local input
    read -rs input
    print ""
    SECRET_RESULT="$input"
}

show_hf_token_setup_guide() {
    print "  ${BOLD}How to get a Hugging Face token${RESET}"
    print "  ${DIM}1. Open: ${HF_TOKEN_URL}${RESET}"
    print "  ${DIM}2. Click: New token${RESET}"
    print "  ${DIM}3. Name it something like: daydream-local${RESET}"
    print "  ${DIM}4. If you see fine-grained / read / write, choose: ${HF_TOKEN_RECOMMENDED_ROLE}${RESET}"
    print "  ${DIM}   Use ${HF_TOKEN_RECOMMENDED_ROLE} for normal local downloads and inference.${RESET}"
    print "  ${DIM}   Only choose write if you need to upload or push to Hugging Face.${RESET}"
    print "  ${DIM}5. Copy the token that starts with hf_${RESET}"
}

prompt_for_hf_token() {
    while true; do
        text_input "Paste your token (press Enter to skip)" ""
        HF_TOKEN="$INPUT_RESULT"

        if [[ -z "$HF_TOKEN" ]]; then
            print_info "Skipped — no token entered"
            return 0
        fi

        HF_TOKEN_CONFIGURED=true
        if [[ "$HF_TOKEN" == hf_* ]]; then
            print_step "Token saved (will be added to shell profile)"
            return 0
        fi

        print_warn "This token does not start with hf_"
        if confirm "Keep it anyway?" "n"; then
            print_step "Token saved (will be added to shell profile)"
            return 0
        fi

        HF_TOKEN=""
        HF_TOKEN_CONFIGURED=false
        print ""
    done
}

show_model_download_guide() {
    print "  ${BOLD}How to pick a model${RESET}"
    print "  ${DIM}Paste the full repo ID from the Hugging Face model page.${RESET}"
    print "  ${DIM}Example: mlx-community/Qwen3.5-9B-MLX-4bit${RESET}"
    print "  ${DIM}You can also paste:${RESET}"
    print "  ${DIM}  - hf.co/mlx-community/Qwen3.5-9B-MLX-4bit${RESET}"
    print "  ${DIM}  - https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit${RESET}"
    print "  ${YELLOW}Only quantized MLX models are supported.${RESET}"
    print "  ${DIM}Not supported: GGUF models or non-quantized MLX repos.${RESET}"
    print "  ${DIM}After the download finishes, the installer will launch the model for you.${RESET}"
}

normalize_model_ref() {
    local ref=$1

    # Trim leading/trailing whitespace.
    ref="${ref#"${ref%%[![:space:]]*}"}"
    ref="${ref%"${ref##*[![:space:]]}"}"

    ref="${ref%%\?*}"
    ref="${ref%%\#*}"
    ref="${ref%/}"

    if [[ "$ref" == https://hf.co/* ]]; then
        ref="hf.co/${ref#https://hf.co/}"
    elif [[ "$ref" == http://hf.co/* ]]; then
        ref="hf.co/${ref#http://hf.co/}"
    elif [[ "$ref" == https://huggingface.co/* ]]; then
        ref="${ref#https://huggingface.co/}"
    elif [[ "$ref" == http://huggingface.co/* ]]; then
        ref="${ref#http://huggingface.co/}"
    fi

    print -r -- "$ref"
}

prompt_for_model_download() {
    while true; do
        text_input "Paste a model repo ID or hf.co ref (press Enter to skip)" ""
        MODEL_TO_PULL=$(normalize_model_ref "$INPUT_RESULT")

        if [[ -z "$MODEL_TO_PULL" ]]; then
            print_info "Skipped — no model selected"
            MODEL_DOWNLOAD_REQUESTED=false
            return 0
        fi

        MODEL_DOWNLOAD_REQUESTED=true
        print_step "Will download: ${MODEL_TO_PULL}"
        print_info "Only quantized MLX models are supported"
        print_info "The installer will download it, then start it automatically"
        return 0
    done
}

# ── Yes/No confirm ───────────────────────────────────────────────────
confirm() {
    local prompt=$1
    local default=${2:-y}
    local hint
    if [[ "$default" == "y" ]]; then
        hint="Y/n"
    else
        hint="y/N"
    fi

    print -n "${SHOW_CURSOR}"
    print -n "  ${BOLD}${prompt}${RESET} ${DIM}[${hint}]${RESET}: "
    local input
    read -r input
    input=${input:-$default}
    case "$input" in
        [yY]*) return 0 ;;
        *) return 1 ;;
    esac
}

# ══════════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════════
show_banner() {
    print ""
    print -n "${CYAN}"
    cat << 'BANNER'
     ____                  _
    |  _ \  __ _ _   _  __| |_ __ ___  __ _ _ __ ___
    | | | |/ _` | | | |/ _` | '__/ _ \/ _` | '_ ` _ \
    | |_| | (_| | |_| | (_| | | |  __/ (_| | | | | | |
    |____/ \__,_|\__, |\__,_|_|  \___|\__,_|_| |_| |_|
                 |___/
BANNER
    print -n "${RESET}"
    print ""
    print "  ${BOLD}Apple Silicon Local Model CLI${RESET}"
    print "  ${DIM}Ollama UX  ·  MLX Engine  ·  Hugging Face Models${RESET}"
    print ""
    print "  ${DIM}────────────────────────────────────────────────${RESET}"
    print ""
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 1: System Checks
# ══════════════════════════════════════════════════════════════════════
check_system() {
    print "  ${BOLD}System Check${RESET}"
    print ""

    local all_ok=true

    # ── Apple Silicon ─────────────────────────────────────────────
    local arch
    arch=$(uname -m)
    if [[ "$arch" == "arm64" ]]; then
        print_step "Apple Silicon (${arch})"
    else
        print_fail "Apple Silicon required (detected: ${arch})"
        all_ok=false
    fi

    # ── Git ───────────────────────────────────────────────────────
    if command -v git &>/dev/null; then
        local git_ver
        git_ver=$(git --version | awk '{print $3}')
        print_step "Git ${git_ver}"
    else
        print_fail "Git not found"
        print_info "Install with: xcode-select --install"
        all_ok=false
    fi

    # ── Python 3.14+ ─────────────────────────────────────────────
    _try_python() {
        local cmd=$1
        if ! command -v "$cmd" &>/dev/null; then
            return 1
        fi
        local ver
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || return 1
        local major minor
        major=${ver%%.*}
        minor=${ver##*.}
        if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 14 ]]; then
            local full_ver
            full_ver=$("$cmd" --version 2>&1 | head -1)
            # Check for duplicates by resolved path
            local resolved
            resolved=$(command -v "$cmd" 2>/dev/null || echo "$cmd")
            resolved=$(readlink "$resolved" 2>/dev/null || echo "$resolved")
            local existing
            for existing in "${PYTHON_CANDIDATES[@]}"; do
                local existing_resolved
                existing_resolved=$(command -v "$existing" 2>/dev/null || echo "$existing")
                existing_resolved=$(readlink "$existing_resolved" 2>/dev/null || echo "$existing_resolved")
                if [[ "$resolved" == "$existing_resolved" ]]; then
                    return 0  # already tracked
                fi
            done
            PYTHON_CANDIDATES+=("$cmd")
            PYTHON_VERSIONS+=("$full_ver")
            return 0
        fi
        return 1
    }

    # Search for Python in common locations
    _try_python python3.14 || true
    _try_python python3 || true
    _try_python python || true

    # Also check Homebrew paths
    local brew_python
    for brew_python in /opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3 /usr/local/bin/python3.14 /usr/local/bin/python3; do
        if [[ -x "$brew_python" ]]; then
            _try_python "$brew_python" || true
        fi
    done

    if [[ ${#PYTHON_CANDIDATES[@]} -gt 0 ]]; then
        print_step "Python 3.14+ found (${#PYTHON_CANDIDATES[@]} installation(s))"
        local i
        for (( i = 1; i <= ${#PYTHON_CANDIDATES[@]}; i++ )); do
            print_info "  ${PYTHON_CANDIDATES[$i]}  →  ${PYTHON_VERSIONS[$i]}"
        done
    else
        print_fail "Python 3.14+ not found"
        print_info "Install with: brew install python@3.14"
        all_ok=false
    fi

    # ── Existing install check ────────────────────────────────────
    if command -v daydream &>/dev/null; then
        local dd_ver
        dd_ver=$(daydream --version 2>&1 || echo "unknown")
        print_warn "Existing Daydream installation detected: ${dd_ver}"
    fi

    print ""

    if [[ "$all_ok" == false ]]; then
        print "  ${RED}${BOLD}Some requirements are not met.${RESET}"
        print "  ${DIM}Please install the missing dependencies and re-run.${RESET}"
        print ""
        exit 1
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 2: Configuration
# ══════════════════════════════════════════════════════════════════════
configure() {
    print "  ${BOLD}Configuration${RESET}"
    print ""

    # ── Install Location ──────────────────────────────────────────
    print_info "Press Enter to keep the default install location."
    text_input "Install location" "$HOME/Daydream-cli"
    INSTALL_DIR="$INPUT_RESULT"
    # Expand tilde
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
    print_info "→ ${INSTALL_DIR}"
    print ""

    # ── Python Selection ──────────────────────────────────────────
    if [[ ${#PYTHON_CANDIDATES[@]} -eq 1 ]]; then
        PYTHON_CMD="${PYTHON_CANDIDATES[1]}"
        print_step "Using ${PYTHON_VERSIONS[1]} (${PYTHON_CMD})"
    elif [[ ${#PYTHON_CANDIDATES[@]} -gt 1 ]]; then
        local menu_items=()
        local i
        for (( i = 1; i <= ${#PYTHON_CANDIDATES[@]}; i++ )); do
            menu_items+=("${PYTHON_CANDIDATES[$i]}  —  ${PYTHON_VERSIONS[$i]}")
        done
        menu_select "Select Python version" "${menu_items[@]}"
        PYTHON_CMD="${PYTHON_CANDIDATES[$(( MENU_RESULT + 1 ))]}"
    fi
    print ""

    # ── HF Token ──────────────────────────────────────────────────
    print "  ${BOLD}Hugging Face Token${RESET}  ${DIM}(optional)${RESET}"
    print "  ${DIM}Higher rate limits & faster model downloads.${RESET}"
    print "  ${DIM}Token page: ${HF_TOKEN_URL}${RESET}"
    print "  ${DIM}Recommended role for Daydream: ${HF_TOKEN_RECOMMENDED_ROLE}${RESET}"
    print ""

    if [[ -n "${HF_TOKEN:-}" ]] || [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        print_step "HF_TOKEN already set in environment"
        print_info "Using the existing token from your environment"
        HF_TOKEN_CONFIGURED=true
        HF_TOKEN=""  # don't overwrite
    else
        local hf_menu_items=(
            "I already have a token  —  paste it now"
            "I need help getting a token  —  show the tutorial first"
            "Skip for now"
        )
        menu_select "HF token setup" "${hf_menu_items[@]}"
        case "$MENU_RESULT" in
            0)
                print ""
                prompt_for_hf_token
                ;;
            1)
                print ""
                show_hf_token_setup_guide
                print ""
                prompt_for_hf_token
                ;;
            *)
                print_info "Skipped — you can set it later from: ${HF_TOKEN_URL}"
                print_info "For Daydream on a local Mac, choose: ${HF_TOKEN_RECOMMENDED_ROLE}"
                ;;
        esac
    fi
    print ""

    # ── Model Download ────────────────────────────────────────────
    print "  ${BOLD}Model Download${RESET}  ${DIM}(optional)${RESET}"
    print "  ${DIM}Paste a model repo ID now and the installer will download it after setup.${RESET}"
    print "  ${DIM}When the download finishes, the installer will start that model automatically.${RESET}"
    print "  ${YELLOW}Only quantized MLX models are supported.${RESET}"
    print "  ${DIM}Not supported: GGUF models or non-quantized MLX repos.${RESET}"
    print "  ${DIM}Example: mlx-community/Qwen3.5-9B-MLX-4bit${RESET}"
    print ""

    local model_menu_items=(
        "I want to download a model now  —  paste the model name"
        "Show me how to pick a model  —  then paste it"
        "Skip for now"
    )
    menu_select "Model download" "${model_menu_items[@]}"
    case "$MENU_RESULT" in
        0)
            print ""
            prompt_for_model_download
            ;;
        1)
            print ""
            show_model_download_guide
            print ""
            prompt_for_model_download
            ;;
        *)
            print_info "Skipped — you can download a model later with: daydream pull <model>"
            print_info "Use a quantized MLX repo such as: mlx-community/Qwen3.5-9B-MLX-4bit"
            ;;
    esac
    print ""

    # ── Shell Integration ─────────────────────────────────────────
    print "  ${BOLD}Shell Integration${RESET}"
    print "  ${DIM}Add 'daydream' command to your PATH so you can use it from any terminal.${RESET}"
    print ""

    if confirm "Add daydream to PATH?" "y"; then
        SHELL_INTEGRATE=true
        print_step "Will update shell profile"
    else
        SHELL_INTEGRATE=false
        print_info "Skipped — activate manually with: source <install_dir>/.venv/bin/activate"
    fi
    print ""
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 3: Confirmation
# ══════════════════════════════════════════════════════════════════════
show_summary() {
    print "  ${DIM}────────────────────────────────────────────────${RESET}"
    print ""
    print "  ${BOLD}Installation Summary${RESET}"
    print ""
    print "  ${CYAN}Location${RESET}       ${INSTALL_DIR}"
    print "  ${CYAN}Python${RESET}         ${PYTHON_CMD}"
    if [[ "$HF_TOKEN_CONFIGURED" == true ]]; then
        if [[ -n "$HF_TOKEN" ]]; then
            print "  ${CYAN}HF Token${RESET}       hf_****${HF_TOKEN: -4}"
        else
            print "  ${CYAN}HF Token${RESET}       ${DIM}already set in environment${RESET}"
        fi
    else
        print "  ${CYAN}HF Token${RESET}       ${DIM}not set${RESET}"
    fi
    if [[ "$MODEL_DOWNLOAD_REQUESTED" == true ]] && [[ -n "$MODEL_TO_PULL" ]]; then
        print "  ${CYAN}Model${RESET}          ${MODEL_TO_PULL}"
    else
        print "  ${CYAN}Model${RESET}          ${DIM}not set${RESET}"
    fi
    if [[ "$SHELL_INTEGRATE" == true ]]; then
        print "  ${CYAN}Shell PATH${RESET}     yes"
    else
        print "  ${CYAN}Shell PATH${RESET}     ${DIM}no${RESET}"
    fi
    print "  ${CYAN}Repository${RESET}     ${REPO_URL}"
    print ""

    if ! confirm "Start installation?" "y"; then
        print ""
        print "  ${DIM}Installation cancelled.${RESET}"
        print ""
        exit 0
    fi
    print ""
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 4: Install
# ══════════════════════════════════════════════════════════════════════
run_install() {
    print "  ${BOLD}Installing Daydream CLI${RESET}"
    print ""

    # ── 1. Clone or update repo ───────────────────────────────────
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        print "  ${DIM}Repository already exists, updating...${RESET}"
        ( cd "$INSTALL_DIR" && git pull --ff-only ) > /tmp/daydream-install.log 2>&1 &
        if spinner $! "Updating repository"; then
            print_step "Repository updated"
        else
            print_warn "Update failed, continuing with existing code"
        fi
    elif [[ -d "$INSTALL_DIR" ]]; then
        print "  ${RED}Error: ${INSTALL_DIR} already exists and is not a git repo.${RESET}"
        print "  ${DIM}Please remove it or choose a different location.${RESET}"
        exit 1
    else
        git clone "$REPO_URL" "$INSTALL_DIR" > /tmp/daydream-install.log 2>&1 &
        if spinner $! "Cloning repository"; then
            print_step "Repository cloned"
        else
            print_fail "Failed to clone repository"
            print "  ${DIM}Check your network connection and try again.${RESET}"
            tail -5 /tmp/daydream-install.log 2>/dev/null || true
            exit 1
        fi
    fi

    # ── 2. Create virtual environment ─────────────────────────────
    if [[ -d "$INSTALL_DIR/.venv" ]]; then
        print_step "Virtual environment already exists"
    else
        "$PYTHON_CMD" -m venv "$INSTALL_DIR/.venv" > /tmp/daydream-install.log 2>&1 &
        if spinner $! "Creating virtual environment"; then
            print_step "Virtual environment created"
        else
            print_fail "Failed to create virtual environment"
            tail -5 /tmp/daydream-install.log 2>/dev/null || true
            exit 1
        fi
    fi

    local PIP="$INSTALL_DIR/.venv/bin/pip"

    # ── 3. Upgrade pip, setuptools, wheel ─────────────────────────
    "$PIP" install -U pip setuptools wheel > /tmp/daydream-install.log 2>&1 &
    if spinner $! "Upgrading pip, setuptools, wheel"; then
        print_step "Build tools upgraded"
    else
        print_warn "Failed to upgrade build tools, continuing..."
    fi

    # ── 4. Install Daydream CLI ───────────────────────────────────
    "$PIP" install -e "$INSTALL_DIR" > /tmp/daydream-install.log 2>&1 &
    if spinner $! "Installing Daydream CLI and dependencies (this may take a while)"; then
        print_step "Daydream CLI installed"
    else
        print_fail "Failed to install Daydream CLI"
        print ""
        print "  ${DIM}Error log:${RESET}"
        tail -20 /tmp/daydream-install.log 2>/dev/null || true
        exit 1
    fi

    # ── 5. Shell profile integration ──────────────────────────────
    local SHELL_PROFILE=""
    local CURRENT_SHELL
    CURRENT_SHELL=$(basename "$SHELL")

    case "$CURRENT_SHELL" in
        zsh)  SHELL_PROFILE="$HOME/.zshrc" ;;
        bash) SHELL_PROFILE="$HOME/.bashrc" ;;
        *)    SHELL_PROFILE="$HOME/.profile" ;;
    esac

    local DAYDREAM_BIN="$INSTALL_DIR/.venv/bin"
    local WROTE_PROFILE=false

    if [[ "$SHELL_INTEGRATE" == true ]] || [[ -n "$HF_TOKEN" ]]; then
        local MARKER="# >>> daydream >>>"
        local MARKER_END="# <<< daydream <<<"

        # Remove old daydream block if exists
        if [[ -f "$SHELL_PROFILE" ]] && grep -q "$MARKER" "$SHELL_PROFILE"; then
            local tmp_profile
            tmp_profile=$(mktemp)
            awk "/$MARKER/{skip=1; next} /$MARKER_END/{skip=0; next} !skip" "$SHELL_PROFILE" > "$tmp_profile"
            mv "$tmp_profile" "$SHELL_PROFILE"
        fi

        # Write new block
        {
            print ""
            print "$MARKER"
            if [[ "$SHELL_INTEGRATE" == true ]]; then
                print "export PATH=\"${DAYDREAM_BIN}:\$PATH\""
            fi
            if [[ -n "$HF_TOKEN" ]]; then
                print "export HF_TOKEN=\"${HF_TOKEN}\""
            fi
            print "$MARKER_END"
        } >> "$SHELL_PROFILE"

        WROTE_PROFILE=true
        print_step "Updated ${SHELL_PROFILE}"
    fi

    if [[ -n "$HF_TOKEN" ]]; then
        export HF_TOKEN
    fi

    # ── 6. Verify ─────────────────────────────────────────────────
    print ""
    local DAYDREAM_CMD="$INSTALL_DIR/.venv/bin/daydream"
    if [[ -x "$DAYDREAM_CMD" ]]; then
        local dd_version
        dd_version=$("$DAYDREAM_CMD" --version 2>&1 || echo "unknown")
        print_step "Verified: ${dd_version}"
    else
        print_warn "daydream binary not found at expected path"
    fi

    # ── 7. Optional model download ────────────────────────────────
    if [[ "$MODEL_DOWNLOAD_REQUESTED" == true ]] && [[ -n "$MODEL_TO_PULL" ]] && [[ -x "$DAYDREAM_CMD" ]]; then
        print ""
        print "  ${BOLD}Optional Model Download${RESET}"
        print "  ${DIM}Only quantized MLX models are supported.${RESET}"
        "$DAYDREAM_CMD" pull "$MODEL_TO_PULL" > /tmp/daydream-install.log 2>&1 &
        if spinner $! "Downloading ${MODEL_TO_PULL}"; then
            MODEL_DOWNLOAD_COMPLETED=true
            print_step "Model downloaded: ${MODEL_TO_PULL}"
        else
            MODEL_DOWNLOAD_FAILED=true
            print_warn "Model download failed: ${MODEL_TO_PULL}"
            print_info "Daydream only supports quantized MLX models"
            tail -20 /tmp/daydream-install.log 2>/dev/null || true
        fi
    fi

    print ""

    # ══════════════════════════════════════════════════════════════
    #  DONE
    # ══════════════════════════════════════════════════════════════
    print "  ${DIM}────────────────────────────────────────────────${RESET}"
    print ""
    print "  ${GREEN}${BOLD}Installation complete!${RESET}"
    print ""
    print "  ${CYAN}Install location${RESET}  ${INSTALL_DIR}"
    print "  ${CYAN}CLI command${RESET}       ${DAYDREAM_CMD}"
    if [[ "$HF_TOKEN_CONFIGURED" == true ]]; then
        if [[ -n "$HF_TOKEN" ]]; then
            print "  ${CYAN}HF Token${RESET}          ${DIM}saved to your shell profile${RESET}"
        else
            print "  ${CYAN}HF Token${RESET}          ${DIM}already available in your environment${RESET}"
        fi
    else
        print "  ${CYAN}HF Token${RESET}          ${DIM}not set${RESET}"
    fi
    if [[ "$MODEL_DOWNLOAD_COMPLETED" == true ]]; then
        print "  ${CYAN}Model${RESET}             ${DIM}downloaded: ${MODEL_TO_PULL}${RESET}"
    elif [[ "$MODEL_DOWNLOAD_FAILED" == true ]]; then
        print "  ${CYAN}Model${RESET}             ${DIM}download failed: ${MODEL_TO_PULL}${RESET}"
    elif [[ "$MODEL_DOWNLOAD_REQUESTED" == true ]]; then
        print "  ${CYAN}Model${RESET}             ${DIM}queued: ${MODEL_TO_PULL}${RESET}"
    else
        print "  ${CYAN}Model${RESET}             ${DIM}not downloaded${RESET}"
    fi
    print ""

    if [[ "$WROTE_PROFILE" == true ]]; then
        print "  ${YELLOW}→${RESET} Restart your terminal or run:"
        print ""
        print "    ${BOLD}source ${SHELL_PROFILE}${RESET}"
        print ""
    fi

    print "  ${BOLD}Quick Start${RESET}"
    print ""

    if [[ "$SHELL_INTEGRATE" == true ]]; then
        print "  ${CYAN}# Run a model${RESET}"
        print "  ${WHITE}daydream run qwen3${RESET}"
        print ""
        print "  ${CYAN}# Pull a model first${RESET}"
        print "  ${WHITE}daydream pull qwen3:8b${RESET}"
        print ""
        print "  ${CYAN}# Start API server${RESET}"
        print "  ${WHITE}daydream serve qwen3 8b${RESET}"
        print ""
        print "  ${CYAN}# List available models${RESET}"
        print "  ${WHITE}daydream models${RESET}"
    else
        print "  ${CYAN}# Activate the environment${RESET}"
        print "  ${WHITE}source ${INSTALL_DIR}/.venv/bin/activate${RESET}"
        print ""
        print "  ${CYAN}# Then run a model${RESET}"
        print "  ${WHITE}daydream run qwen3${RESET}"
        print ""
        print "  ${CYAN}# Or use the full path${RESET}"
        print "  ${WHITE}${DAYDREAM_BIN}/daydream run qwen3${RESET}"
    fi

    if [[ "$MODEL_DOWNLOAD_COMPLETED" == true ]]; then
        print ""
        print "  ${CYAN}# Starting the downloaded model now${RESET}"
        print "  ${WHITE}${DAYDREAM_CMD} run ${MODEL_TO_PULL}${RESET}"
    fi

    if [[ "$HF_TOKEN_CONFIGURED" == false ]]; then
        print ""
        print "  ${YELLOW}Optional: add a Hugging Face token later${RESET}"
        print "  ${DIM}${HF_TOKEN_URL}${RESET}"
        print "  ${DIM}Choose ${HF_TOKEN_RECOMMENDED_ROLE} for local downloads and inference.${RESET}"
    fi

    if [[ "$MODEL_DOWNLOAD_REQUESTED" == false ]]; then
        print ""
        print "  ${YELLOW}Optional: download a model later${RESET}"
        print "  ${DIM}Paste a quantized MLX repo ID like:${RESET}"
        print "  ${DIM}mlx-community/Qwen3.5-9B-MLX-4bit${RESET}"
        print "  ${DIM}Then run: daydream pull <model>${RESET}"
    fi

    print ""
    print "  ${DIM}Documentation: https://github.com/Starry331/Daydream-cli${RESET}"
    print "  ${DIM}Need help?     daydream --help${RESET}"
    print ""

    if [[ "$MODEL_DOWNLOAD_COMPLETED" == true ]]; then
        print "  ${BOLD}Launching your model${RESET}"
        print "  ${DIM}Type /quit when you want to leave the chat and return to the shell.${RESET}"
        print ""
        if "$DAYDREAM_CMD" run "$MODEL_TO_PULL"; then
            print ""
            print_step "Model session ended"
        else
            print ""
            print_warn "Could not start ${MODEL_TO_PULL} automatically"
            print_info "Try again with: ${DAYDREAM_CMD} run ${MODEL_TO_PULL}"
        fi
        print ""
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
main() {
    ensure_interactive_terminal
    clear
    show_banner
    check_system
    configure
    show_summary
    run_install
}

main "$@"
