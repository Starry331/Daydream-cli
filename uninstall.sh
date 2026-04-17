#!/bin/zsh
# ──────────────────────────────────────────────────────────────────────
# Daydream CLI — Interactive Uninstaller
# Cleanly removes Daydream CLI, config, models, and shell integration
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

# ── Detected state ───────────────────────────────────────────────────
INSTALL_DIR=""
DAYDREAM_HOME="$HOME/.daydream"
HF_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
SHELL_PROFILE=""

# What to remove (user choices)
RM_INSTALL=false
RM_CONFIG=false
RM_SHELL=false
RM_LAUNCHERS=false
MODELS_TO_REMOVE=()     # indices into FOUND_MODELS
RM_ALL_MODELS=false

# Discovered items
FOUND_INSTALL_DIR=""
FOUND_CONFIG=false
FOUND_SHELL_BLOCK=false
FOUND_LAUNCHERS=()
FOUND_MODELS=()         # display names
FOUND_MODEL_DIRS=()     # actual paths

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

format_size() {
    local bytes=$1
    if (( bytes >= 1073741824 )); then
        printf "%.1f GB" "$(( bytes / 1073741824.0 ))"
    elif (( bytes >= 1048576 )); then
        printf "%.1f MB" "$(( bytes / 1048576.0 ))"
    elif (( bytes >= 1024 )); then
        printf "%.0f KB" "$(( bytes / 1024.0 ))"
    else
        printf "%d B" "$bytes"
    fi
}

dir_size_bytes() {
    local dir=$1
    if [[ -d "$dir" ]]; then
        du -sk "$dir" 2>/dev/null | awk '{print $1 * 1024}'
    else
        echo 0
    fi
}

# Single-line spinner (same as installer — no flooding)
run_spinner_step() {
    local msg=$1
    shift

    "$@" > /tmp/daydream-uninstall.log 2>&1 &
    local pid=$!
    local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    local nframes=${#frames[@]}
    local i=1

    print -n "${HIDE_CURSOR}"
    while kill -0 "$pid" 2>/dev/null; do
        print -n "\r  ${CYAN}${frames[$i]}${RESET} ${DIM}${msg}${RESET}\033[K"
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

# ── Multi-select checkbox menu ───────────────────────────────────────
# Arrow keys to move, Space to toggle, Enter to confirm, A to select all
# Sets MULTISELECT_RESULT as an array of selected indices (0-based)
MULTISELECT_RESULT=()

multiselect() {
    local title=$1
    shift
    local options=("$@")
    local total=${#options[@]}
    local selected=1  # zsh 1-indexed cursor position
    local -a checked=()
    local i

    # Initialize all unchecked
    for (( i = 1; i <= total; i++ )); do
        checked[$i]=0
    done

    local enter_alt leave_alt move_home clear_to_end hide_cursor show_cursor
    enter_alt=$(tput smcup 2>/dev/null || printf '\033[?1049h')
    leave_alt=$(tput rmcup 2>/dev/null || printf '\033[?1049l')
    move_home=$(tput home 2>/dev/null || printf '\033[H')
    clear_to_end=$(tput ed 2>/dev/null || printf '\033[J')
    hide_cursor=$(tput civis 2>/dev/null || printf '%s' "${HIDE_CURSOR}")
    show_cursor=$(tput cnorm 2>/dev/null || printf '%s' "${SHOW_CURSOR}")

    _ms_draw() {
        printf '%s%s' "$move_home" "$clear_to_end"
        print "  ${BOLD}${title}${RESET}"
        print ""
        for (( i = 1; i <= total; i++ )); do
            local box
            if [[ ${checked[$i]} -eq 1 ]]; then
                box="${GREEN}[x]${RESET}"
            else
                box="${DIM}[ ]${RESET}"
            fi
            if [[ $i -eq $selected ]]; then
                print "  ${BOLD}${CYAN}> ${box} ${options[$i]}${RESET}"
            else
                print "    ${box} ${options[$i]}"
            fi
        done
        print ""
        local count=0
        for (( i = 1; i <= total; i++ )); do
            [[ ${checked[$i]} -eq 1 ]] && (( count++ ))
        done
        print "  ${DIM}↑/↓ move  Space toggle  A select all  Enter confirm (${count} selected)${RESET}"
    }

    printf '%s%s' "$hide_cursor" "$enter_alt"
    _ms_draw

    # Suppress tracing (same fix as installer)
    {
        setopt localoptions noxtrace noverbose nolog
    } 2>/dev/null

    while true; do
        local key=""
        read -rsk1 key 2>/dev/null
        if [[ "$key" == $'\e' ]]; then
            local seq=""
            read -rsk2 seq 2>/dev/null
            case "$seq" in
                '[A') selected=$(( (selected - 2 + total) % total + 1 )) ;;
                '[B') selected=$(( selected % total + 1 )) ;;
            esac
        elif [[ "$key" == " " ]]; then
            # Toggle current item
            if [[ ${checked[$selected]} -eq 1 ]]; then
                checked[$selected]=0
            else
                checked[$selected]=1
            fi
        elif [[ "$key" == "a" || "$key" == "A" ]]; then
            # Toggle all
            local any_unchecked=0
            for (( i = 1; i <= total; i++ )); do
                [[ ${checked[$i]} -eq 0 ]] && any_unchecked=1 && break
            done
            local new_val=$any_unchecked
            for (( i = 1; i <= total; i++ )); do
                checked[$i]=$new_val
            done
        elif [[ "$key" == $'\n' || "$key" == $'\r' || -z "$key" ]]; then
            printf '%s%s' "$leave_alt" "$show_cursor"
            # Build result (0-based indices)
            MULTISELECT_RESULT=()
            local count=0
            for (( i = 1; i <= total; i++ )); do
                if [[ ${checked[$i]} -eq 1 ]]; then
                    MULTISELECT_RESULT+=$(( i - 1 ))
                    (( count++ ))
                fi
            done
            print "  ${GREEN}✓${RESET} ${BOLD}${title}${RESET}  ${DIM}${count} selected${RESET}"
            return 0
        elif [[ "$key" == $'\x03' ]]; then
            printf '%s%s' "$leave_alt" "$show_cursor"
            print ""
            exit 1
        fi
        _ms_draw
    done
}

# ══════════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════════
show_banner() {
    print ""
    print -n "${RED}"
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
    print "  ${BOLD}Uninstaller${RESET}"
    print "  ${DIM}This will remove Daydream CLI from your system.${RESET}"
    print "  ${DIM}Your Python, Homebrew, and other tools will NOT be touched.${RESET}"
    print ""
    print "  ${DIM}────────────────────────────────────────────────${RESET}"
    print ""
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 1: Detect installation
# ══════════════════════════════════════════════════════════════════════
detect() {
    print "  ${BOLD}Scanning${RESET}"
    print ""

    local found_anything=false

    # ── Install directory ─────────────────────────────────────────
    # Check common locations
    local candidate
    for candidate in "$HOME/Daydream-cli" "$HOME/daydream-cli" "$HOME/Documents/Daydreamcli" "$HOME/Desktop/Daydreamcli"; do
        if [[ -d "$candidate" ]] && [[ -d "$candidate/.venv/bin" ]] && [[ -f "$candidate/pyproject.toml" ]]; then
            FOUND_INSTALL_DIR="$candidate"
            break
        fi
    done

    # Also try to find via the daydream command itself
    if [[ -z "$FOUND_INSTALL_DIR" ]]; then
        local dd_path
        dd_path=$(command -v daydream 2>/dev/null || true)
        if [[ -n "$dd_path" ]]; then
            # Resolve symlinks/launchers
            dd_path=$(readlink "$dd_path" 2>/dev/null || echo "$dd_path")
            # If it's a launcher script, extract the real path
            if [[ -f "$dd_path" ]] && grep -q "daydream installer launcher" "$dd_path" 2>/dev/null; then
                dd_path=$(grep 'exec ' "$dd_path" 2>/dev/null | sed 's/exec "//;s/" .*//' || echo "")
            fi
            if [[ -n "$dd_path" ]]; then
                # Go from .venv/bin/daydream → install root
                local maybe_root
                maybe_root=$(dirname "$(dirname "$(dirname "$dd_path")")" 2>/dev/null || true)
                if [[ -n "$maybe_root" ]] && [[ -f "$maybe_root/pyproject.toml" ]]; then
                    FOUND_INSTALL_DIR="$maybe_root"
                fi
            fi
        fi
    fi

    if [[ -n "$FOUND_INSTALL_DIR" ]]; then
        local size
        size=$(format_size $(dir_size_bytes "$FOUND_INSTALL_DIR"))
        print_step "Install directory: ${FOUND_INSTALL_DIR} (${size})"
        found_anything=true
    else
        print_info "No install directory found"
    fi

    # ── Config directory (~/.daydream) ────────────────────────────
    if [[ -d "$DAYDREAM_HOME" ]]; then
        local size
        size=$(format_size $(dir_size_bytes "$DAYDREAM_HOME"))
        print_step "Config directory: ${DAYDREAM_HOME} (${size})"
        FOUND_CONFIG=true
        found_anything=true
    else
        print_info "No config directory found"
    fi

    # ── Shell profile block ───────────────────────────────────────
    local CURRENT_SHELL
    CURRENT_SHELL=$(basename "$SHELL")
    case "$CURRENT_SHELL" in
        zsh)  SHELL_PROFILE="$HOME/.zshrc" ;;
        bash) SHELL_PROFILE="$HOME/.bashrc" ;;
        *)    SHELL_PROFILE="$HOME/.profile" ;;
    esac

    if [[ -f "$SHELL_PROFILE" ]] && grep -q '# >>> daydream >>>' "$SHELL_PROFILE" 2>/dev/null; then
        print_step "Shell profile entry in ${SHELL_PROFILE}"
        FOUND_SHELL_BLOCK=true
        found_anything=true
    else
        print_info "No shell profile entry found"
    fi

    # ── Launchers / symlinks ──────────────────────────────────────
    local launcher
    for launcher in \
        /usr/local/bin/daydream \
        "$HOME/.local/bin/daydream" \
        "$HOME/.npm-global/bin/daydream" \
        "$HOME/bin/daydream"; do
        if [[ -e "$launcher" ]] || [[ -L "$launcher" ]]; then
            # Verify it's a daydream launcher (not something else named daydream)
            if [[ -L "$launcher" ]]; then
                local target
                target=$(readlink "$launcher" 2>/dev/null || true)
                if [[ "$target" == *daydream* ]] || [[ "$target" == *Daydream* ]]; then
                    FOUND_LAUNCHERS+=("$launcher")
                    print_step "Launcher: ${launcher}"
                    found_anything=true
                fi
            elif grep -q "daydream" "$launcher" 2>/dev/null; then
                FOUND_LAUNCHERS+=("$launcher")
                print_step "Launcher: ${launcher}"
                found_anything=true
            fi
        fi
    done

    if [[ ${#FOUND_LAUNCHERS[@]} -eq 0 ]]; then
        print_info "No launchers or symlinks found"
    fi

    # ── Downloaded models ─────────────────────────────────────────
    if [[ -d "$HF_CACHE_DIR" ]]; then
        local model_dir
        for model_dir in "$HF_CACHE_DIR"/models--*; do
            [[ -d "$model_dir" ]] || continue
            # Extract owner/name from dir name: models--owner--name
            local dirname
            dirname=$(basename "$model_dir")
            local repo_id
            repo_id=$(echo "$dirname" | sed 's/^models--//;s/--/\//g')

            local size
            size=$(format_size $(dir_size_bytes "$model_dir"))
            FOUND_MODELS+=("${repo_id}  ${DIM}(${size})${RESET}")
            FOUND_MODEL_DIRS+=("$model_dir")
        done
    fi

    if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
        print_step "Downloaded models: ${#FOUND_MODELS[@]} found"
        found_anything=true
    else
        print_info "No downloaded models found"
    fi

    print ""

    if [[ "$found_anything" == false ]]; then
        print "  ${DIM}Nothing to uninstall. Daydream does not appear to be installed.${RESET}"
        print ""
        exit 0
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 2: Choose what to remove
# ══════════════════════════════════════════════════════════════════════
choose() {
    print "  ${BOLD}What to remove${RESET}"
    print "  ${DIM}Each item is optional. Your system Python and tools will not be touched.${RESET}"
    print ""

    # ── Install directory ─────────────────────────────────────────
    if [[ -n "$FOUND_INSTALL_DIR" ]]; then
        local size
        size=$(format_size $(dir_size_bytes "$FOUND_INSTALL_DIR"))
        if confirm "Remove install directory? ${DIM}${FOUND_INSTALL_DIR} (${size})${RESET}" "y"; then
            RM_INSTALL=true
        fi
        print ""
    fi

    # ── Config ────────────────────────────────────────────────────
    if [[ "$FOUND_CONFIG" == true ]]; then
        local size
        size=$(format_size $(dir_size_bytes "$DAYDREAM_HOME"))
        print "  ${DIM}Contains: config, registry, chat history, memories${RESET}"
        if confirm "Remove config directory? ${DIM}${DAYDREAM_HOME} (${size})${RESET}" "y"; then
            RM_CONFIG=true
        fi
        print ""
    fi

    # ── Shell profile ─────────────────────────────────────────────
    if [[ "$FOUND_SHELL_BLOCK" == true ]]; then
        print "  ${DIM}This removes the PATH and HF_TOKEN entries added by the installer.${RESET}"
        if confirm "Remove shell profile entries? ${DIM}${SHELL_PROFILE}${RESET}" "y"; then
            RM_SHELL=true
        fi
        print ""
    fi

    # ── Launchers ─────────────────────────────────────────────────
    if [[ ${#FOUND_LAUNCHERS[@]} -gt 0 ]]; then
        if confirm "Remove launcher scripts/symlinks?" "y"; then
            RM_LAUNCHERS=true
        fi
        print ""
    fi

    # ── Models (multi-select) ─────────────────────────────────────
    if [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
        print "  ${BOLD}Downloaded Models${RESET}"
        print "  ${DIM}These are cached in your Hugging Face hub directory.${RESET}"
        print "  ${DIM}Other (non-Daydream) Hugging Face downloads will not be shown.${RESET}"
        print ""

        if confirm "Remove any downloaded models?" "n"; then
            print ""
            multiselect "Select models to remove (Space to toggle, A for all)" "${FOUND_MODELS[@]}"
            MODELS_TO_REMOVE=("${MULTISELECT_RESULT[@]}")
        fi
        print ""
    fi

    # ── Check if anything was selected ────────────────────────────
    local removing_something=false
    [[ "$RM_INSTALL" == true ]] && removing_something=true
    [[ "$RM_CONFIG" == true ]] && removing_something=true
    [[ "$RM_SHELL" == true ]] && removing_something=true
    [[ "$RM_LAUNCHERS" == true ]] && removing_something=true
    [[ ${#MODELS_TO_REMOVE[@]} -gt 0 ]] && removing_something=true

    if [[ "$removing_something" == false ]]; then
        print "  ${DIM}Nothing selected for removal.${RESET}"
        print ""
        exit 0
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 3: Confirm
# ══════════════════════════════════════════════════════════════════════
show_summary() {
    print "  ${DIM}────────────────────────────────────────────────${RESET}"
    print ""
    print "  ${BOLD}Removal Summary${RESET}"
    print ""

    if [[ "$RM_INSTALL" == true ]]; then
        print "  ${RED}✗${RESET} ${FOUND_INSTALL_DIR}"
    fi
    if [[ "$RM_CONFIG" == true ]]; then
        print "  ${RED}✗${RESET} ${DAYDREAM_HOME}"
    fi
    if [[ "$RM_SHELL" == true ]]; then
        print "  ${RED}✗${RESET} Shell entries in ${SHELL_PROFILE}"
    fi
    if [[ "$RM_LAUNCHERS" == true ]]; then
        local l
        for l in "${FOUND_LAUNCHERS[@]}"; do
            print "  ${RED}✗${RESET} ${l}"
        done
    fi
    if [[ ${#MODELS_TO_REMOVE[@]} -gt 0 ]]; then
        local idx
        for idx in "${MODELS_TO_REMOVE[@]}"; do
            local zi=$(( idx + 1 ))  # 0-based → 1-based
            print "  ${RED}✗${RESET} Model: ${FOUND_MODELS[$zi]}"
        done
    fi

    print ""
    print "  ${YELLOW}${BOLD}This cannot be undone.${RESET}"
    print ""

    if ! confirm "Proceed with removal?" "n"; then
        print ""
        print "  ${DIM}Uninstall cancelled.${RESET}"
        print ""
        exit 0
    fi
    print ""
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 4: Execute removal
# ══════════════════════════════════════════════════════════════════════
run_uninstall() {
    print "  ${BOLD}Removing${RESET}"
    print ""

    local freed_bytes=0

    # ── 1. Stop running server ────────────────────────────────────
    if [[ -n "$FOUND_INSTALL_DIR" ]] && [[ -x "$FOUND_INSTALL_DIR/.venv/bin/daydream" ]]; then
        local server_state="$DAYDREAM_HOME/server.json"
        if [[ -f "$server_state" ]]; then
            local pid
            pid=$(python3 -c "import json; d=json.load(open('$server_state')); print(d.get('pid',''))" 2>/dev/null || true)
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                sleep 0.5
                kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
                print_step "Stopped running Daydream server (pid ${pid})"
            fi
        fi
    fi

    # ── 2. Remove launchers/symlinks ──────────────────────────────
    if [[ "$RM_LAUNCHERS" == true ]]; then
        local l
        for l in "${FOUND_LAUNCHERS[@]}"; do
            if rm -f "$l" 2>/dev/null; then
                print_step "Removed ${l}"
            else
                print_warn "Could not remove ${l} (permission denied?)"
            fi
        done
    fi

    # ── 3. Remove shell profile entries ───────────────────────────
    if [[ "$RM_SHELL" == true ]] && [[ -f "$SHELL_PROFILE" ]]; then
        local MARKER="# >>> daydream >>>"
        local MARKER_END="# <<< daydream <<<"
        local tmp_profile
        tmp_profile=$(mktemp)
        awk "/$MARKER/{skip=1; next} /$MARKER_END/{skip=0; next} !skip" "$SHELL_PROFILE" > "$tmp_profile"
        mv "$tmp_profile" "$SHELL_PROFILE"
        print_step "Cleaned ${SHELL_PROFILE}"
    fi

    # ── 4. Remove models ──────────────────────────────────────────
    if [[ ${#MODELS_TO_REMOVE[@]} -gt 0 ]]; then
        local idx
        for idx in "${MODELS_TO_REMOVE[@]}"; do
            local zi=$(( idx + 1 ))
            local model_dir="${FOUND_MODEL_DIRS[$zi]}"
            local model_name="${FOUND_MODELS[$zi]}"
            if [[ -d "$model_dir" ]]; then
                local mbytes
                mbytes=$(dir_size_bytes "$model_dir")
                if run_spinner_step "Removing model: ${model_name}" rm -rf "$model_dir"; then
                    freed_bytes=$(( freed_bytes + mbytes ))
                    print_step "Removed model: ${model_name}"
                else
                    print_warn "Failed to remove: ${model_dir}"
                fi
            fi
        done
    fi

    # ── 5. Remove config directory ────────────────────────────────
    if [[ "$RM_CONFIG" == true ]] && [[ -d "$DAYDREAM_HOME" ]]; then
        local cbytes
        cbytes=$(dir_size_bytes "$DAYDREAM_HOME")
        if run_spinner_step "Removing config directory" rm -rf "$DAYDREAM_HOME"; then
            freed_bytes=$(( freed_bytes + cbytes ))
            print_step "Removed ${DAYDREAM_HOME}"
        else
            print_warn "Failed to remove ${DAYDREAM_HOME}"
        fi
    fi

    # ── 6. Remove install directory ───────────────────────────────
    if [[ "$RM_INSTALL" == true ]] && [[ -n "$FOUND_INSTALL_DIR" ]] && [[ -d "$FOUND_INSTALL_DIR" ]]; then
        local ibytes
        ibytes=$(dir_size_bytes "$FOUND_INSTALL_DIR")
        if run_spinner_step "Removing install directory" rm -rf "$FOUND_INSTALL_DIR"; then
            freed_bytes=$(( freed_bytes + ibytes ))
            print_step "Removed ${FOUND_INSTALL_DIR}"
        else
            print_warn "Failed to remove ${FOUND_INSTALL_DIR}"
        fi
    fi

    # ── Done ──────────────────────────────────────────────────────
    print ""
    print "  ${DIM}────────────────────────────────────────────────${RESET}"
    print ""

    local freed
    freed=$(format_size $freed_bytes)
    print "  ${GREEN}${BOLD}Uninstall complete.${RESET}  ${DIM}Freed ~${freed}${RESET}"
    print ""

    if [[ "$RM_SHELL" == true ]]; then
        print "  ${YELLOW}→${RESET} Restart your terminal for PATH changes to take effect."
        print ""
    fi

    print "  ${DIM}The following were NOT touched:${RESET}"
    print "  ${DIM}  - Python, Homebrew, pip, and other system tools${RESET}"
    print "  ${DIM}  - Your Hugging Face account and token (unless you removed shell entries)${RESET}"
    if [[ ${#MODELS_TO_REMOVE[@]} -eq 0 ]] && [[ ${#FOUND_MODELS[@]} -gt 0 ]]; then
        print "  ${DIM}  - All downloaded models in ${HF_CACHE_DIR}${RESET}"
    fi
    print ""
    print "  ${DIM}To reinstall: ./install.sh${RESET}"
    print ""
}

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
main() {
    if [[ ! -t 0 || ! -t 1 ]]; then
        print ""
        print "  ${RED}${BOLD}This uninstaller needs an interactive terminal.${RESET}"
        print ""
        exit 1
    fi

    clear
    show_banner
    detect
    choose
    show_summary
    run_uninstall
}

main "$@"
