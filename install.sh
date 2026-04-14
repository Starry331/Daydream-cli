#!/bin/zsh
# ──────────────────────────────────────────────────────────────────────
# Daydream CLI — Interactive Installer
# One-click setup for Apple Silicon local model inference
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

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
SHELL_INTEGRATE=true
REPO_URL="https://github.com/Starry331/Daydream-cli.git"
PYTHON_CANDIDATES=()
PYTHON_VERSIONS=()

# ── Cleanup trap ──────────────────────────────────────────────────────
cleanup() {
    print -n "${SHOW_CURSOR}"
    stty sane 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Utility functions ─────────────────────────────────────────────────
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
        print -n "\033[A\033[K"
    done
}

# ── Arrow-key interactive menu ───────────────────────────────────────
# Sets global MENU_RESULT to the selected index
menu_select() {
    local title=$1
    shift
    local options=("$@")
    local selected=1  # zsh 1-indexed
    local total=${#options[@]}

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
    local draw_lines=$(( total + 4 ))

    while true; do
        local key
        read -rsk1 key
        case "$key" in
            $'\e')
                local seq
                read -rsk2 seq
                case "$seq" in
                    '[A') selected=$(( (selected - 2 + total) % total + 1 )) ;;
                    '[B') selected=$(( selected % total + 1 )) ;;
                esac
                ;;
            $'\n'|$'\r')
                clear_lines $draw_lines
                print "  ${GREEN}✓${RESET} ${BOLD}${title}${RESET}  ${DIM}${options[$selected]}${RESET}"
                print -n "${SHOW_CURSOR}"
                MENU_RESULT=$(( selected - 1 ))  # convert to 0-indexed for array access later
                return 0
                ;;
            $'\x03')
                print -n "${SHOW_CURSOR}"
                print ""
                exit 1
                ;;
        esac
        clear_lines $draw_lines
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
        print -n "  ${BOLD}${prompt}${RESET} ${DIM}(${default})${RESET}: "
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
    print "  ${DIM}Get a free token: https://huggingface.co/settings/tokens${RESET}"
    print ""

    if [[ -n "${HF_TOKEN:-}" ]] || [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        print_step "HF_TOKEN already set in environment"
        HF_TOKEN=""  # don't overwrite
    else
        if confirm "Set up HF_TOKEN?" "n"; then
            secret_input "Paste your token (hidden)"
            HF_TOKEN="$SECRET_RESULT"
            if [[ -n "$HF_TOKEN" ]]; then
                print_step "Token saved (will be added to shell profile)"
            else
                print_info "Skipped — no token entered"
            fi
        else
            print_info "Skipped — you can set it later with: export HF_TOKEN=your_token"
        fi
    fi
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
    if [[ -n "$HF_TOKEN" ]]; then
        print "  ${CYAN}HF Token${RESET}       hf_****${HF_TOKEN: -4}"
    else
        print "  ${CYAN}HF Token${RESET}       ${DIM}not set${RESET}"
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

    print ""

    # ══════════════════════════════════════════════════════════════
    #  DONE
    # ══════════════════════════════════════════════════════════════
    print "  ${DIM}────────────────────────────────────────────────${RESET}"
    print ""
    print "  ${GREEN}${BOLD}Installation complete!${RESET}"
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

    print ""
    print "  ${DIM}Documentation: https://github.com/Starry331/Daydream-cli${RESET}"
    print "  ${DIM}Need help?     daydream --help${RESET}"
    print ""
}

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
main() {
    clear
    show_banner
    check_system
    configure
    show_summary
    run_install
}

main "$@"
