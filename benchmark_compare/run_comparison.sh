#!/usr/bin/env bash
# Comparative benchmark:
#   A) patched relayrl_framework 0.5.0-beta (workspace)
#   B) relayrl_framework 0.5.0-beta.2 with inference-engine + trajectory-guard patches applied
# 1 actor · 1 / 32 / 128 parallel GridWorld sub-envs · 20 000 calls each
#
# Usage (from repo root):
#   bash benchmark_compare/run_comparison.sh
#
# Outputs one log file per run in benchmark_compare/results/.

set -euo pipefail

export ORT_DYLIB_PATH="/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.25.0"

COMPARE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="/home/user/RelayRL-end2end"
RESULTS_DIR="$COMPARE_DIR/results"
mkdir -p "$RESULTS_DIR"

PERF="/usr/lib/linux-tools-6.8.0-31/perf"
PERF_EVENTS="task-clock,context-switches,cpu-migrations,page-faults"
PERF_FLAGS="-e $PERF_EVENTS --"

CALLS=20000
GRID=10

# ── Binary paths ─────────────────────────────────────────────────────────────

PATCHED_BIN="$REPO_ROOT/target/release/bench_gridworld_vecenv"
BETA2_PATCHED_BIN="$COMPARE_DIR/target/release/bench_gridworld_vecenv_beta2_patched"

# ── Build ─────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════════════"
echo "  Building patched 0.5.0-beta (workspace)…"
echo "════════════════════════════════════════════════════════════════════════"
cargo build --release -p relayrl-e2e --bin bench_gridworld_vecenv \
    --manifest-path "$REPO_ROOT/Cargo.toml" 2>&1

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Building beta.2 + patches (standalone)…"
echo "════════════════════════════════════════════════════════════════════════"
cargo build --release \
    --manifest-path "$COMPARE_DIR/Cargo.toml" 2>&1

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Both binaries built.  Starting benchmark runs…"
echo "  Config: --target-calls $CALLS  --grid-size $GRID"
echo "  Env counts: 1  32  128"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ── Run helper ────────────────────────────────────────────────────────────────

run_bench() {
    local label="$1"   # e.g. "patched_1env"
    local bin="$2"
    local num_envs="$3"
    local baseline_sps="$4"   # optional: 1-env sps for S(n); empty = skip
    local log="$RESULTS_DIR/${label}.log"

    echo "────────────────────────────────────────────────────────────────────────"
    echo "  RUN: $label  (num_envs=$num_envs)"
    echo "  bin: $bin"
    echo "────────────────────────────────────────────────────────────────────────"

    local baseline_flag=()
    if [[ -n "$baseline_sps" ]]; then
        baseline_flag=(--baseline-sps "$baseline_sps")
    fi

    {
        echo "# label=$label  num_envs=$num_envs  calls=$CALLS  grid=$GRID"
        echo "# perf events: $PERF_EVENTS"
        echo ""
        "$PERF" stat $PERF_FLAGS \
            "$bin" \
                --num-envs   "$num_envs" \
                --target-calls "$CALLS" \
                --grid-size  "$GRID" \
                "${baseline_flag[@]}" \
            2>&1
    } | tee "$log"

    echo ""
    echo "  → saved to $log"
    echo ""
}

# Extract 1-env sps from a completed log file
extract_1env_sps() {
    local log="$1"
    grep "env-steps/sec (global)" "$log" | grep -oP '[\d.]+' | head -1
}

# ── Patched 0.5.0-beta (workspace) ───────────────────────────────────────────

run_bench "patched_beta_1env"   "$PATCHED_BIN"  1  ""
PATCHED_BASE=$(extract_1env_sps "$RESULTS_DIR/patched_beta_1env.log")
echo "  → patched 1-env baseline: $PATCHED_BASE sps"
run_bench "patched_beta_32env"  "$PATCHED_BIN"  32   "$PATCHED_BASE"
run_bench "patched_beta_128env" "$PATCHED_BIN"  128  "$PATCHED_BASE"

# ── beta.2 + inference-engine patch ──────────────────────────────────────────

run_bench "beta2_patched_1env"   "$BETA2_PATCHED_BIN"  1  ""
BETA2_BASE=$(extract_1env_sps "$RESULTS_DIR/beta2_patched_1env.log")
echo "  → beta2_patched 1-env baseline: $BETA2_BASE sps"
run_bench "beta2_patched_32env"  "$BETA2_PATCHED_BIN"  32   "$BETA2_BASE"
run_bench "beta2_patched_128env" "$BETA2_PATCHED_BIN"  128  "$BETA2_BASE"

# ── Comparison summary ────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  COMPARISON SUMMARY"
echo "  (env-steps/sec  |  call mean µs  |  ctx-sw/sec  |  RSS peak MB)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

extract() {
    local log="$1"
    local envs="$2"
    local sps call_mean ctx_sw rss_peak
    sps=$(grep "env-steps/sec (global)" "$log" | grep -oP '[\d.]+' | head -1)
    call_mean=$(grep "call mean" "$log" | grep -oP '[\d.]+' | head -1)
    ctx_sw=$(grep "context switches/sec" "$log" | grep -oP '[\d.]+' | head -1)
    rss_peak=$(grep "RSS peak" "$log" | grep -oP '[\d.]+' | head -1)
    printf "  %-30s | %12s sps | %10s µs | %10s ctx/s | %8s MB RSS-peak\n" \
        "$envs" "$sps" "$call_mean" "$ctx_sw" "$rss_peak"
}

echo "  --- patched 0.5.0-beta (workspace) ---"
extract "$RESULTS_DIR/patched_beta_1env.log"   "1 env"
extract "$RESULTS_DIR/patched_beta_32env.log"  "32 envs"
extract "$RESULTS_DIR/patched_beta_128env.log" "128 envs"
echo ""
echo "  --- 0.5.0-beta.2 + inference-engine patch ---"
extract "$RESULTS_DIR/beta2_patched_1env.log"   "1 env"
extract "$RESULTS_DIR/beta2_patched_32env.log"  "32 envs"
extract "$RESULTS_DIR/beta2_patched_128env.log" "128 envs"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Full logs: $RESULTS_DIR/"
echo "════════════════════════════════════════════════════════════════════════"
