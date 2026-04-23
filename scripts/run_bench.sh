#!/usr/bin/env bash
# run_bench.sh — benchmark both gridworld variants with perf instrumentation
#
# Runs bench_gridworld (multi-actor) and bench_gridworld_vecenv (32-env VecEnv)
# against relayrl_framework 0.5.0-beta.2, collecting:
#   • the benchmark's own 40+ metrics (throughput, latency percentiles, memory, CPU)
#   • perf stat software events (task-clock, ctx-switches, migrations, page-faults)
#   • perf c2c record/report for cache-to-cache coherence traffic (if PMU available)
#
# Usage:
#   ./scripts/run_bench.sh [results-dir]
#
# Results written to:
#   <results-dir>/bench_gridworld_Nactor_{perf_stat,bench,c2c}.txt
#   <results-dir>/bench_gridworld_vecenv32_{perf_stat,bench,c2c}.txt
#   <results-dir>/summary.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINDIR="$REPO_ROOT/target/release"
RESULTS_DIR="${1:-$REPO_ROOT/bench_results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

# Locate ONNX Runtime shared library (needed by the ort crate for inference)
if [[ -z "${ORT_DYLIB_PATH:-}" ]]; then
    ORT_DYLIB_PATH=$(find /usr -name "libonnxruntime.so*" 2>/dev/null | head -1 || true)
    if [[ -z "$ORT_DYLIB_PATH" ]]; then
        echo "ERROR: libonnxruntime.so not found. Set ORT_DYLIB_PATH or install onnxruntime:" >&2
        echo "  pip install onnxruntime" >&2
        exit 1
    fi
fi
export ORT_DYLIB_PATH

BENCH_GW="$BINDIR/bench_gridworld"
BENCH_VEC="$BINDIR/bench_gridworld_vecenv"

# ── Validate binaries ─────────────────────────────────────────────────────────
for bin in "$BENCH_GW" "$BENCH_VEC"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin not found. Build first with:" >&2
        echo "  cargo build --release -p relayrl-e2e --bin bench_gridworld --bin bench_gridworld_vecenv" >&2
        exit 1
    fi
done

PERF=$(command -v perf 2>/dev/null || true)
if [[ -z "$PERF" ]]; then
    echo "WARNING: perf not found — skipping perf instrumentation" >&2
fi

# ── perf event set ────────────────────────────────────────────────────────────
# Hardware PMU events (cycles, instructions, cache-misses) are not available
# inside containers / VMs without PMU pass-through.  We fall back to the
# software events that the kernel always exposes.
PERF_SW_EVENTS="task-clock,context-switches,cpu-migrations,page-faults,minor-faults"

# Seconds perf stat may wait after the traced process exits before we kill it.
# bench_gridworld's Tokio runtime leaves a background monitor thread alive after
# main() returns; perf stat waits for all threads to exit.  A 30s ceiling lets
# the summary write on clean hosts (< 1 ms) while unblocking CI on containerised
# ones where the thread never exits.
PERF_STAT_TIMEOUT="${PERF_STAT_TIMEOUT:-30}"

# ── Helper functions ──────────────────────────────────────────────────────────

run_perf_stat() {
    local label="$1"; shift         # e.g. "gridworld_1actor"
    local out_combined="$RESULTS_DIR/${label}_bench.txt"

    echo "=== perf stat: $label ===" | tee -a "$RESULTS_DIR/summary.txt"

    if [[ -n "$PERF" ]]; then
        # perf stat writes its summary to stderr; redirect both streams so the
        # bench output AND the counter block end up in one file.
        # `timeout --foreground` sends SIGTERM to the entire process group so
        # perf stat can't block indefinitely waiting for Tokio monitor threads.
        timeout --foreground "$PERF_STAT_TIMEOUT" \
            "$PERF" stat -e "$PERF_SW_EVENTS" -- "$@" \
            >"$out_combined" 2>&1 || true
    else
        "$@" >"$out_combined" 2>&1
    fi
    echo "  bench+perf → $out_combined"

    # Print throughput summary line from bench output
    grep -E "env-steps/sec|calls/sec|Throughput" "$out_combined" | tail -5 \
        | sed "s/^/  /" | tee -a "$RESULTS_DIR/summary.txt" || true
    # Print perf stat counters if present
    if grep -q "Performance counter stats" "$out_combined" 2>/dev/null; then
        grep -A 8 "Performance counter stats" "$out_combined" | tail -9 \
            | sed "s/^/  [perf] /" | tee -a "$RESULTS_DIR/summary.txt" || true
    fi
    echo "" | tee -a "$RESULTS_DIR/summary.txt"
}

run_perf_c2c() {
    local label="$1"; shift         # e.g. "gridworld_1actor"
    local out_c2c="$RESULTS_DIR/${label}_c2c.txt"
    local data_file="$RESULTS_DIR/${label}_c2c.data"

    if [[ -z "$PERF" ]]; then
        return
    fi

    echo "=== perf c2c: $label ===" | tee -a "$RESULTS_DIR/summary.txt"

    # Record phase — fails gracefully if PMU memory events are unavailable
    if "$PERF" c2c record \
            --output "$data_file" \
            -- "$@" >/dev/null 2>"$out_c2c.record_err"; then
        # Report phase
        "$PERF" c2c report \
            --stdio \
            --input "$data_file" \
            >"$out_c2c" 2>&1 && \
            echo "  c2c report → $out_c2c" | tee -a "$RESULTS_DIR/summary.txt" || true
    else
        local err
        err=$(cat "$out_c2c.record_err")
        printf "  c2c: not available on this host (%s)\n" \
            "$(echo "$err" | head -1)" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  (hardware memory-access PMU events require bare-metal or PMU pass-through)"
        # Still emit a placeholder so callers know we tried
        printf "perf c2c record failed:\n%s\n" "$err" >"$out_c2c"
    fi
    echo "" | tee -a "$RESULTS_DIR/summary.txt"
}

# ── Banner ────────────────────────────────────────────────────────────────────
{
echo "RelayRL framework benchmark — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Framework version : 0.5.0-beta.2"
echo "Host kernel       : $(uname -r)"
echo "CPU cores         : $(nproc)"
echo "Results dir       : $RESULTS_DIR"
echo "perf version      : $(perf --version 2>/dev/null || echo 'not available')"
echo ""
} | tee "$RESULTS_DIR/summary.txt"

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — multi-actor GridWorld (bench_gridworld)
#
# Tests the standard multi-actor integration: N RelayRL actors each step a
# shared GridWorldEnv independently.  request_action(all_ids) fans out to all
# N actors in parallel.
# ─────────────────────────────────────────────────────────────────────────────
echo "### Part 1 — bench_gridworld (multi-actor) ###" | tee -a "$RESULTS_DIR/summary.txt"

for ACTORS in 1 2 4 8; do
    LABEL="gridworld_${ACTORS}actor"
    echo "--- $LABEL ---" | tee -a "$RESULTS_DIR/summary.txt"

    run_perf_stat "$LABEL" \
        "$BENCH_GW" \
        --actor-count "$ACTORS" \
        --target-calls 200000

    run_perf_c2c "$LABEL" \
        "$BENCH_GW" \
        --actor-count "$ACTORS" \
        --target-calls 50000
done

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — vectorized GridWorld (bench_gridworld_vecenv, 32 sub-envs)
#
# Standard integration: 1 RelayRL actor batches 32 sub-environments in a
# single request_action call.  One ONNX forward pass over [32, obs_dim].
# ─────────────────────────────────────────────────────────────────────────────
echo "### Part 2 — bench_gridworld_vecenv (32 envs, standard integration) ###" \
    | tee -a "$RESULTS_DIR/summary.txt"

LABEL="gridworld_vecenv32"
run_perf_stat "$LABEL" \
    "$BENCH_VEC" \
    --num-envs 32 \
    --target-calls 50000

run_perf_c2c "$LABEL" \
    "$BENCH_VEC" \
    --num-envs 32 \
    --target-calls 20000

# ── Final summary ─────────────────────────────────────────────────────────────
echo "=== All benchmarks complete ===" | tee -a "$RESULTS_DIR/summary.txt"
echo "Results in: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
