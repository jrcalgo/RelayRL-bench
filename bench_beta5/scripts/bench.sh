#!/usr/bin/env bash
# RelayRL Benchmark Launcher
# Interactive CLI menu for bench_beta4 and bench_beta5 benchmark binaries.
# Usage: bash bench_beta5/scripts/bench.sh
set -euo pipefail

trap 'echo ""; echo "  Interrupted."; exit 130' INT

# ── Path bootstrap ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BETA5_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BETA4_DIR="$(cd "${SCRIPT_DIR}/../../bench_beta4" && pwd)"

# ── Cargo package names per workspace ─────────────────────────────────────────
declare -A WS_PACKAGE
WS_PACKAGE[beta4]="bench-beta4"
WS_PACKAGE[beta5]="bench-beta5"

# ── Dependency flags ───────────────────────────────────────────────────────────
declare -A NEEDS_ORT NEEDS_LIBTORCH NEEDS_GYMNASIUM NEEDS_ENVPOOL

for b in bench_lunar_ppo_scalar1 bench_grid_ppo_scalar1 bench_lunar_ppo_1env \
          bench_lunar_ppo_64env bench_lunar_ppo_tch bench_lunar_ppo_py \
          bench_start_latency bench_lunar_eval_py bench_lunar_sfppo_py \
          bench_lunar_eval_envpool; do
  NEEDS_ORT[$b]=1
done

for b in bench_lunar_ppo_tch bench_lunar_ppo_py bench_lunar_sfppo_py \
          bench_lunar_eval_envpool_tch; do
  NEEDS_LIBTORCH[$b]=1
done

for b in bench_lunar_ppo_py bench_lunar_eval_py bench_lunar_sfppo_py; do
  NEEDS_GYMNASIUM[$b]=1
done

for b in bench_lunar_eval_envpool bench_lunar_eval_envpool_tch; do
  NEEDS_ENVPOOL[$b]=1
done

# Binaries that call std::env::set_var("ORT_DYLIB_PATH", ...) internally,
# overriding whatever the script exports. Only bench_lunar_direct_scalar1 and
# bench_lunar_set_env_scalar1 respect the script's ORT_DYLIB_PATH export.
declare -A OVERRIDES_ORT_INTERNALLY
for b in bench_lunar_ppo_scalar1 bench_grid_ppo_scalar1 bench_lunar_ppo_1env \
          bench_lunar_ppo_64env bench_lunar_ppo_tch bench_lunar_ppo_py \
          bench_start_latency bench_lunar_eval_py bench_lunar_sfppo_py \
          bench_lunar_eval_envpool; do
  OVERRIDES_ORT_INTERNALLY[$b]=1
done

# ── Compile-time constants (display-only; require recompile to change) ─────────
declare -A BENCH_CONSTANTS
BENCH_CONSTANTS[bench_lunar_direct_scalar1]="ENV_COUNT=1  MAX_STEPS=500  TARGET_ITERS=100000  WARMUP_ITERS=10000"
BENCH_CONSTANTS[bench_lunar_set_env_scalar1]="ENV_COUNT=1  MAX_STEPS=500  TARGET_STEPS=100000  WARMUP_ITERS=10000"
BENCH_CONSTANTS[bench_lunar_ppo_scalar1]="ENV_COUNT=64  TOTAL_STEPS=23438  GAMMA=0.999  LAM=0.98  CLIP=0.2  PI_LR=2.5e-4  VF_LR=2.5e-4  MINI_BATCH=64"
BENCH_CONSTANTS[bench_grid_ppo_scalar1]="ENV_COUNT=64  TOTAL_STEPS=7813  GAMMA=0.99  LAM=0.95  CLIP=0.2  PI_LR=3e-4  VF_LR=3e-4"
BENCH_CONSTANTS[bench_lunar_ppo_1env]="ENV_COUNT=1  TOTAL_STEPS=100000  GAMMA=0.999  LAM=0.98  CLIP=0.2  PI_LR=2.5e-4  VF_LR=2.5e-4  MINI_BATCH=64"
BENCH_CONSTANTS[bench_lunar_ppo_64env]="ENV_COUNT=64  TOTAL_STEPS=1563  GAMMA=0.999  LAM=0.98  CLIP=0.2  PI_LR=2.5e-4  VF_LR=2.5e-4  MINI_BATCH=64"
BENCH_CONSTANTS[bench_lunar_ppo_tch]="ENV_COUNT=64  TOTAL_STEPS=600000  GAMMA=0.999  LAM=0.98  CLIP=0.2  PI_LR=2.5e-4  MINI_BATCH=5760  backend=LibTorch"
BENCH_CONSTANTS[bench_lunar_ppo_py]="ENV_COUNT=64  TOTAL_STEPS=600000  GAMMA=0.999  LAM=0.98  CLIP=0.2  PI_LR=2.5e-4  MINI_BATCH=5760  backend=LibTorch+Py"
BENCH_CONSTANTS[bench_start_latency]="one-shot: agent build / start / shutdown latency — no loop"
BENCH_CONSTANTS[bench_lunar_eval_py]="ENV_COUNT=1024  WARMUP=500  TIMED=5000  model=lunarlander_policy.onnx  backend=gymnasium"
BENCH_CONSTANTS[bench_lunar_sfppo_py]="ENV_COUNT=64  TOTAL_STEPS=600000  ROLLOUT=32  MINI_BATCH=2048  CLIP=0.1  backend=LibTorch+Py"
BENCH_CONSTANTS[bench_lunar_eval_envpool]="ENV_COUNT=runtime(--envs N)  WARMUP=500  TIMED=5000  model=lunarlander_policy.onnx"
BENCH_CONSTANTS[bench_lunar_eval_envpool_tch]="ENV_COUNT=1024  WARMUP=500  TIMED=5000  model=lunarlander_policy.pt  backend=LibTorch+EnvPool"

# ── Category → benchmark mapping ──────────────────────────────────────────────
CATEGORIES=(
  "LunarLander/Latency"
  "LunarLander/PPO"
  "LunarLander/PPO-tch"
  "LunarLander/PPO-py"
  "LunarLander/Eval"
  "LunarLander/Eval-tch"
  "GridWorld/PPO"
  "Latency"
)

declare -A CATEGORY_BINARIES
CATEGORY_BINARIES["LunarLander/Latency"]="bench_lunar_direct_scalar1 bench_lunar_set_env_scalar1"
CATEGORY_BINARIES["LunarLander/PPO"]="bench_lunar_ppo_scalar1 bench_lunar_ppo_1env bench_lunar_ppo_64env"
CATEGORY_BINARIES["LunarLander/PPO-tch"]="bench_lunar_ppo_tch"
CATEGORY_BINARIES["LunarLander/PPO-py"]="bench_lunar_ppo_py bench_lunar_sfppo_py"
CATEGORY_BINARIES["LunarLander/Eval"]="bench_lunar_eval_py bench_lunar_eval_envpool"
CATEGORY_BINARIES["LunarLander/Eval-tch"]="bench_lunar_eval_envpool_tch"
CATEGORY_BINARIES["GridWorld/PPO"]="bench_grid_ppo_scalar1"
CATEGORY_BINARIES["Latency"]="bench_start_latency"

# ── Globals set by menu / config functions ────────────────────────────────────
SELECTED_WORKSPACE=""
SELECTED_CATEGORY=""
SELECTED_BINARY=""
CONF_ORT_PATH=""       # ORT_DYLIB_PATH; defaults to $ORT_DYLIB_PATH from env, else empty
CONF_LIBTORCH_DIR=""   # prepended to LD_LIBRARY_PATH; defaults to $LIBTORCH_DIR from env, else empty
CONF_RAYON_THREADS=""
CONF_ENVS=""
CONF_PROFILE=""
CONF_LOG=""

# ═══════════════════════════════════════════════════════════════════════════════
# Box-drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════
BOX_W=60  # inner width (between ║ chars)

hrtop() { printf "  ╔"; printf '═%.0s' $(seq 1 $BOX_W); printf "╗\n"; }
hrbot() { printf "  ╚"; printf '═%.0s' $(seq 1 $BOX_W); printf "╝\n"; }
hrsep() { printf "  ╠"; printf '═%.0s' $(seq 1 $BOX_W); printf "╣\n"; }
row()   { printf "  ║  %-$((BOX_W-2))s║\n" "$1"; }

# Word-wrap text into box rows, each at most (BOX_W-2) chars wide.
wrap_row() {
  local text="$1"
  local max=$(( BOX_W - 2 ))
  local -a words
  read -ra words <<< "$text"
  local line=""
  for word in "${words[@]}"; do
    local candidate="${line:+$line }$word"
    if [[ ${#candidate} -gt $max && -n "$line" ]]; then
      row "$line"
      line="$word"
    else
      line="$candidate"
    fi
  done
  [[ -n "$line" ]] && row "$line"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════════════
print_banner() {
  clear 2>/dev/null || true
  echo ""
  hrtop
  row ""
  row "   RelayRL Benchmark Launcher"
  row "   bench_beta4  (RelayRL 0.5.0-beta.4)"
  row "   bench_beta5  (RelayRL 0.5.0-beta.5)"
  row ""
  hrbot
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Prerequisite checks  (one-line errors, exit 1)
# ═══════════════════════════════════════════════════════════════════════════════
check_prereqs() {
  local binary="$1"

  if [[ -n "${NEEDS_ORT[$binary]+_}" ]] && [[ -z "$CONF_ORT_PATH" || ! -f "$CONF_ORT_PATH" ]]; then
    echo "Error: ORT_DYLIB_PATH not set or file not found — set it in the config prompt or export ORT_DYLIB_PATH before running"; exit 1
  fi

  if [[ -n "${NEEDS_LIBTORCH[$binary]+_}" ]] && [[ -n "$CONF_LIBTORCH_DIR" ]] && [[ ! -d "$CONF_LIBTORCH_DIR" ]]; then
    echo "Error: LibTorch dir '$CONF_LIBTORCH_DIR' does not exist — set LIBTORCH_DIR or export LD_LIBRARY_PATH before running"; exit 1
  fi

  if [[ -n "${NEEDS_GYMNASIUM[$binary]+_}" ]]; then
    python3 -c "import gymnasium" 2>/dev/null || \
      { echo "Error: Python package 'gymnasium' not found — run: pip install 'gymnasium[box2d]'"; exit 1; }
  fi

  if [[ -n "${NEEDS_ENVPOOL[$binary]+_}" ]]; then
    python3 -c "import envpool" 2>/dev/null || \
      { echo "Error: Python package 'envpool' not found — run: pip install envpool"; exit 1; }
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# ORT override notice
# Most PPO/eval binaries call std::env::set_var("ORT_DYLIB_PATH", ...) at startup,
# baking in their compile-time ORT path and overriding the script's export.
# Only bench_lunar_direct_scalar1 and bench_lunar_set_env_scalar1 respect the
# exported value. Show a one-time notice so the user knows what to expect.
# ═══════════════════════════════════════════════════════════════════════════════
warn_ort_internal_override() {
  local binary="$1"
  if [[ -n "${OVERRIDES_ORT_INTERNALLY[$binary]+_}" ]]; then
    echo ""
    hrtop
    row "  ℹ  ORT path note"
    hrsep
    row "This binary sets ORT_DYLIB_PATH internally at startup"
    row "using the path it was compiled against, overriding any"
    row "value exported by this script."
    row ""
    row "If it was compiled against a different ORT version than"
    row "what is installed, it will fail at runtime. To fix:"
    row "  · symlink the missing libonnxruntime.so to the one"
    row "    that is installed, or recompile the binary."
    hrbot
    echo ""
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Auto-build if binary is missing
# ═══════════════════════════════════════════════════════════════════════════════
ensure_binary() {
  local ws="$1" binary="$2"
  local ws_dir bin_path
  if [[ "$ws" == "beta4" ]]; then ws_dir="$BETA4_DIR"; else ws_dir="$BETA5_DIR"; fi
  bin_path="${ws_dir}/target/release/${binary}"

  if [[ ! -x "$bin_path" ]]; then
    echo ""
    echo "  Binary not found: $bin_path"
    echo "  Package         : ${WS_PACKAGE[$ws]}"
    local ans
    read -rp "  Auto-build with cargo build --release? [Y/n]: " ans || true
    if [[ "${ans:-Y}" =~ ^[Yy]$ ]]; then
      echo ""
      echo "  Building ${binary}…"
      (
        cd "$ws_dir"
        LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
          cargo build --release -p "${WS_PACKAGE[$ws]}" --bin "$binary"
      )
      echo ""
      echo "  Build complete."
    else
      echo "  Aborted."; exit 0
    fi
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Menus — all functions set SELECTED_* globals (never use $() for menus)
# ═══════════════════════════════════════════════════════════════════════════════

select_workspace() {
  echo "  Select workspace:"
  echo ""
  local -a items=(
    "bench_beta5   RelayRL 0.5.0-beta.5  (current)"
    "bench_beta4   RelayRL 0.5.0-beta.4  (reference)"
    "Quit"
  )
  PS3=$'\n  Workspace: '
  select item in "${items[@]}"; do
    case "$item" in
      *beta5*) SELECTED_WORKSPACE="beta5"; break ;;
      *beta4*) SELECTED_WORKSPACE="beta4"; break ;;
      "Quit")  exit 0 ;;
      "")      echo "  Invalid — try again." ;;
    esac
  done
  echo ""
}

# Returns 0 on selection; loops on invalid input; exits on Quit.
select_category() {
  echo "  Select category:"
  echo ""
  local -a items=("${CATEGORIES[@]}" "← Change workspace" "Quit")
  PS3=$'\n  Category: '
  select item in "${items[@]}"; do
    case "$item" in
      "← Change workspace") echo ""; select_workspace ;;
      "Quit")               exit 0 ;;
      "")                   echo "  Invalid — try again." ;;
      *)                    SELECTED_CATEGORY="$item"; break ;;
    esac
  done
  echo ""
}

# Returns 0 on selection, 1 on Back; exits on Quit.
select_binary() {
  local category="$1"
  local -a bins
  read -ra bins <<< "${CATEGORY_BINARIES[$category]}"

  echo "  [$category] — select benchmark:"
  echo ""
  local -a items=("${bins[@]}" "← Back" "Quit")
  PS3=$'\n  Benchmark: '
  select item in "${items[@]}"; do
    case "$item" in
      "← Back") return 1 ;;
      "Quit")   exit 0 ;;
      "")       echo "  Invalid — try again." ;;
      *)        SELECTED_BINARY="$item"; return 0 ;;
    esac
  done
}

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark info panel
# ═══════════════════════════════════════════════════════════════════════════════
show_bench_info() {
  local ws="$1" binary="$2"
  local ws_dir bin_path
  if [[ "$ws" == "beta4" ]]; then ws_dir="$BETA4_DIR"; else ws_dir="$BETA5_DIR"; fi
  bin_path="${ws_dir}/target/release/${binary}"

  echo ""
  hrtop
  row "  $binary"
  row "  workspace: $ws"
  hrsep
  row "  Compile-time constants  (READ-ONLY — recompile to change)"
  hrsep
  wrap_row "${BENCH_CONSTANTS[$binary]}"
  hrsep

  # Dependencies
  local deps="none"
  [[ -n "${NEEDS_ORT[$binary]+_}" ]]       && deps="ORT"
  [[ -n "${NEEDS_LIBTORCH[$binary]+_}" ]]  && { [[ "$deps" == "none" ]] && deps="LibTorch"   || deps="${deps} + LibTorch"; }
  [[ -n "${NEEDS_GYMNASIUM[$binary]+_}" ]] && { [[ "$deps" == "none" ]] && deps="gymnasium"  || deps="${deps} + gymnasium"; }
  [[ -n "${NEEDS_ENVPOOL[$binary]+_}" ]]   && { [[ "$deps" == "none" ]] && deps="envpool"    || deps="${deps} + envpool"; }
  row "  Deps     : $deps"

  # Show current env var values (or unset notice) so the user knows what they're starting with
  if [[ -n "${NEEDS_ORT[$binary]+_}" ]]; then
    local cur_ort="${ORT_DYLIB_PATH:-}"
    if [[ -n "$cur_ort" ]]; then
      row "  ORT_DYLIB_PATH  : $cur_ort"
    else
      row "  ORT_DYLIB_PATH  : (not set — will prompt)"
    fi
  fi

  if [[ -n "${NEEDS_LIBTORCH[$binary]+_}" ]]; then
    local cur_lt="${LIBTORCH_DIR:-}"
    if [[ -n "$cur_lt" ]]; then
      row "  LIBTORCH_DIR    : $cur_lt"
    else
      row "  LIBTORCH_DIR    : (not set — will prompt)"
    fi
  fi

  if [[ -x "$bin_path" ]]; then
    row "  Binary   : found"
  else
    row "  Binary   : NOT BUILT  (will auto-build on run)"
  fi

  hrbot
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime configuration prompts
# ═══════════════════════════════════════════════════════════════════════════════
collect_config() {
  local binary="$1"
  local default_threads
  default_threads="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  local default_log="/tmp/bench_${binary}_$(date +%Y%m%dT%H%M%S).txt"

  hrtop
  row "  Runtime Configuration"
  hrbot
  echo ""

  # ORT_DYLIB_PATH — shown only for ORT-dependent binaries
  CONF_ORT_PATH=""
  if [[ -n "${NEEDS_ORT[$binary]+_}" ]]; then
    local cur_ort="${ORT_DYLIB_PATH:-}"
    local ort_hint="${cur_ort:-not set}"
    read -rp "  ORT_DYLIB_PATH  [${ort_hint}]: " CONF_ORT_PATH || true
    CONF_ORT_PATH="${CONF_ORT_PATH:-$cur_ort}"
  fi

  # LIBTORCH_DIR — prepended to LD_LIBRARY_PATH for LibTorch-dependent binaries
  CONF_LIBTORCH_DIR=""
  if [[ -n "${NEEDS_LIBTORCH[$binary]+_}" ]]; then
    local cur_lt="${LIBTORCH_DIR:-}"
    local lt_hint="${cur_lt:-not set}"
    read -rp "  LIBTORCH_DIR (prepended to LD_LIBRARY_PATH)  [${lt_hint}]: " CONF_LIBTORCH_DIR || true
    CONF_LIBTORCH_DIR="${CONF_LIBTORCH_DIR:-$cur_lt}"
  fi

  # RAYON_NUM_THREADS — all benchmarks except bench_start_latency
  CONF_RAYON_THREADS=""
  if [[ "$binary" != "bench_start_latency" ]]; then
    read -rp "  RAYON_NUM_THREADS  [${default_threads}]: " CONF_RAYON_THREADS || true
    CONF_RAYON_THREADS="${CONF_RAYON_THREADS:-$default_threads}"
    if ! [[ "$CONF_RAYON_THREADS" =~ ^[1-9][0-9]*$ ]]; then
      echo "  Error: RAYON_NUM_THREADS must be a positive integer, got: $CONF_RAYON_THREADS"; exit 1
    fi
  fi

  # --envs N — bench_lunar_eval_envpool only
  CONF_ENVS=""
  if [[ "$binary" == "bench_lunar_eval_envpool" ]]; then
    read -rp "  --envs (parallel environments)  [1024]: " CONF_ENVS || true
    CONF_ENVS="${CONF_ENVS:-1024}"
    if ! [[ "$CONF_ENVS" =~ ^[1-9][0-9]*$ ]]; then
      echo "  Error: --envs must be a positive integer, got: $CONF_ENVS"; exit 1
    fi
  fi

  # Profiling with /usr/bin/time -v
  read -rp "  Profiling (/usr/bin/time -v)?  [y/N]: " CONF_PROFILE || true
  CONF_PROFILE="${CONF_PROFILE:-N}"

  # Output log path (Enter = default, '-' or 'none' = disable)
  echo ""
  echo "  Output log: press Enter for default path, or type '-' to disable."
  read -rp "  Log path  [${default_log}]: " CONF_LOG || true
  if [[ -z "$CONF_LOG" ]]; then
    CONF_LOG="$default_log"
  elif [[ "$CONF_LOG" == "-" || "$CONF_LOG" == "none" ]]; then
    CONF_LOG=""
  fi

  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Run summary
# ═══════════════════════════════════════════════════════════════════════════════
show_run_summary() {
  local ws="$1" binary="$2"
  local ws_dir bin_path
  if [[ "$ws" == "beta4" ]]; then ws_dir="$BETA4_DIR"; else ws_dir="$BETA5_DIR"; fi
  bin_path="${ws_dir}/target/release/${binary}"

  # Build env display string from configured values only (omit unset entries)
  local env_str=""
  [[ -n "$CONF_ORT_PATH" ]]       && env_str="${env_str:+$env_str  }ORT_DYLIB_PATH=${CONF_ORT_PATH}"
  [[ -n "$CONF_LIBTORCH_DIR" ]]   && env_str="${env_str:+$env_str  }LD_LIBRARY_PATH=${CONF_LIBTORCH_DIR}:..."
  [[ -n "${NEEDS_LIBTORCH[$binary]+_}" ]] && \
    env_str="${env_str:+$env_str  }LIBTORCH_USE_PYTORCH=1  LIBTORCH_BYPASS_VERSION_CHECK=1"
  [[ -n "$CONF_RAYON_THREADS" ]]  && env_str="${env_str:+$env_str  }RAYON_NUM_THREADS=${CONF_RAYON_THREADS}"
  [[ -z "$env_str" ]] && env_str="(none)"

  # Build command display string
  local cmd_str="$bin_path"
  [[ "$binary" == "bench_lunar_eval_envpool" ]] && cmd_str="${cmd_str} --envs ${CONF_ENVS}"
  [[ "$CONF_PROFILE" =~ ^[Yy] ]]               && cmd_str="/usr/bin/time -v  ${cmd_str}"
  [[ -n "$CONF_LOG" ]]                          && cmd_str="${cmd_str}  2>&1 | tee ${CONF_LOG}"

  echo ""
  hrtop
  row "  Run Summary"
  hrsep
  row "  workspace  : $ws"
  row "  binary     : $binary"
  hrsep
  row "  ENV:"
  wrap_row "    $env_str"
  hrsep
  row "  CMD:"
  wrap_row "    $cmd_str"
  if [[ -n "$CONF_LOG" ]]; then
    hrsep
    row "  LOG  →  $CONF_LOG"
  fi
  hrbot
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Confirm and execute
# ═══════════════════════════════════════════════════════════════════════════════
confirm_and_run() {
  local ws="$1" binary="$2"
  local ws_dir bin_path
  if [[ "$ws" == "beta4" ]]; then ws_dir="$BETA4_DIR"; else ws_dir="$BETA5_DIR"; fi
  bin_path="${ws_dir}/target/release/${binary}"

  local ans
  read -rp "  Run now? [Y/n]: " ans || true
  [[ "${ans:-Y}" =~ ^[Yy]$ ]] || { echo "  Aborted."; return; }

  echo ""
  echo "  ── Started: $(date '+%Y-%m-%d %H:%M:%S')  ──  Ctrl-C to abort ──"
  echo ""

  # ── Export environment (only set vars that were actually configured) ──────
  [[ -n "$CONF_ORT_PATH" ]]      && export ORT_DYLIB_PATH="$CONF_ORT_PATH"

  if [[ -n "${NEEDS_LIBTORCH[$binary]+_}" ]]; then
    [[ -n "$CONF_LIBTORCH_DIR" ]] && \
      export LD_LIBRARY_PATH="${CONF_LIBTORCH_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBTORCH_USE_PYTORCH=1
    export LIBTORCH_BYPASS_VERSION_CHECK=1
  fi

  [[ -n "$CONF_RAYON_THREADS" ]] && export RAYON_NUM_THREADS="$CONF_RAYON_THREADS"

  # ── Build command array ──────────────────────────────────────────────────
  local -a cmd=()
  [[ "$CONF_PROFILE" =~ ^[Yy] ]] && cmd+=("/usr/bin/time" "-v")
  cmd+=("$bin_path")
  [[ "$binary" == "bench_lunar_eval_envpool" ]] && cmd+=("--envs" "$CONF_ENVS")

  # ── Execute (capture exit code without triggering set -e) ───────────────
  local exit_code=0
  if [[ -n "$CONF_LOG" ]]; then
    echo "  Capturing output → $CONF_LOG"
    echo ""
    "${cmd[@]}" 2>&1 | tee "$CONF_LOG" || exit_code=${PIPESTATUS[0]}
  else
    "${cmd[@]}" || exit_code=$?
  fi

  echo ""
  echo "  ── Finished: $(date '+%Y-%m-%d %H:%M:%S') ──"
  [[ $exit_code -ne 0 ]] && echo "  (exited with code $exit_code)"
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════
main() {
  print_banner
  select_workspace

  while true; do
    select_category

    while true; do
      if select_binary "$SELECTED_CATEGORY"; then
        show_bench_info            "$SELECTED_WORKSPACE" "$SELECTED_BINARY"
        warn_ort_internal_override "$SELECTED_BINARY"
        collect_config             "$SELECTED_BINARY"
        check_prereqs              "$SELECTED_BINARY"
        ensure_binary              "$SELECTED_WORKSPACE" "$SELECTED_BINARY"
        show_run_summary  "$SELECTED_WORKSPACE" "$SELECTED_BINARY"
        confirm_and_run   "$SELECTED_WORKSPACE" "$SELECTED_BINARY"

        local again
        read -rp "  Run another benchmark? [Y/n]: " again || true
        [[ "${again:-Y}" =~ ^[Yy]$ ]] || exit 0
        echo ""
        break  # back to category select
      else
        # User chose Back — re-enter category selection
        echo ""
        select_category
      fi
    done
  done
}

main "$@"
