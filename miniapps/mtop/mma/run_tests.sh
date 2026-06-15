#!/usr/bin/env bash
# =============================================================================
# run_tests.sh  —  Human-readable test runner for the MMA/GCMMA/SQ test suite
#
# Usage:
#   bash run_tests.sh -b BUILD_DIR [-n NRANKS] [-m MODE]
#
# Options:
#   -b BUILD_DIR   Path to the cmake build directory (required).
#   -n NRANKS      Number of MPI ranks for parallel tests (default: 4).
#   -m MODE        What to run: serial | parallel | all  (default: all).
#
# Adding a new test
# -----------------
# 1. Write the test binary (cmake target) — see CMakeLists.txt.
#
# 2. Add a call to one of the helper functions below in the appropriate section:
#
#    run_serial   LABEL  BINARY [ARGS...]
#      Runs ./BINARY [ARGS...] as a single process.
#      LABEL is the display name shown in the output.
#
#    run_parallel LABEL  BINARY [ARGS...]
#      Runs mpirun -np $NRANKS ./BINARY [ARGS...].
#      Skipped silently when mpirun is not available.
#
#    run_serial_and_parallel LABEL_SER LABEL_PAR BINARY [ARGS...]
#      Runs both a serial (1-rank) and a parallel (NRANKS-rank) entry.
#
# 3. Place the call in the relevant section:
#    ── Core: serial          → MMA/GCMMA serial tests
#    ── Core: parallel        → MMA/GCMMA parallel tests
#    ── Device (CPU)          → device-backend tests
#    ── SQ: serial            → SQ optimizer serial tests
#    ── SQ: parallel          → SQ optimizer parallel tests
#    ── Relaxed equalities    → PackFivalRelaxed / WithRelaxedEqualities tests
#    ── [add your section]    → new sections can be inserted anywhere
#
# The BINARY name must match the cmake target name exactly (without a path).
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_DIR=""
NRANKS=4
MODE="all"          # serial | parallel | all

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b) BUILD_DIR="$2"; shift 2 ;;
        -n) NRANKS="$2";    shift 2 ;;
        -m) MODE="$2";      shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$BUILD_DIR" ]]; then
    echo "Error: -b BUILD_DIR is required." >&2
    echo "Usage: bash run_tests.sh -b BUILD_DIR [-n NRANKS] [-m MODE]" >&2
    exit 1
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: build directory not found: $BUILD_DIR" >&2
    exit 1
fi

# ── Detect mpirun ─────────────────────────────────────────────────────────────
MPIRUN=""
for cmd in mpirun mpiexec; do
    if command -v "$cmd" &>/dev/null; then
        MPIRUN="$cmd"; break
    fi
done

# ── State ─────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
declare -a FAILED_TESTS=()

# ── Helpers ───────────────────────────────────────────────────────────────────
COL=45   # column at which PASS/FAIL is printed

print_result() {
    local label="$1" status="$2" detail="${3:-}"
    local pad=$(( COL - ${#label} ))
    [[ $pad -lt 1 ]] && pad=1
    printf "  %-${COL}s %s\n" "$label" "$status"
    [[ -n "$detail" ]] && printf "    %s\n" "$detail"
}

# Run a single-process test.
# Usage: run_serial LABEL BINARY [ARGS...]
run_serial() {
    local label="$1"; shift
    local binary="$1"; shift
    local bin_path="$BUILD_DIR/$binary"

    if [[ ! -x "$bin_path" ]]; then
        print_result "$label" "SKIP  (binary not found)"
        (( SKIP++ )) || true
        return
    fi

    local output exit_code
    output=$("$bin_path" "$@" 2>&1) && exit_code=0 || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        print_result "$label" "PASS"
        (( PASS++ )) || true
    else
        local first_fail
        first_fail=$(echo "$output" | grep '\[FAIL\]' | head -1)
        print_result "$label" "ERROR (exit $exit_code)" "$first_fail"
        (( FAIL++ )) || true
        FAILED_TESTS+=("$label")
    fi
}

# Run a parallel (multi-rank) test.
# Usage: run_parallel LABEL BINARY [ARGS...]
run_parallel() {
    local label="$1"; shift
    local binary="$1"; shift
    local bin_path="$BUILD_DIR/$binary"

    if [[ -z "$MPIRUN" ]]; then
        print_result "$label" "SKIP  (no mpirun)"
        (( SKIP++ )) || true
        return
    fi

    if [[ ! -x "$bin_path" ]]; then
        print_result "$label" "SKIP  (binary not found)"
        (( SKIP++ )) || true
        return
    fi

    local output exit_code
    output=$("$MPIRUN" -np "$NRANKS" "$bin_path" "$@" 2>&1) && exit_code=0 || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        print_result "$label" "PASS"
        (( PASS++ )) || true
    else
        local first_fail
        first_fail=$(echo "$output" | grep '\[FAIL\]' | head -1)
        print_result "$label" "ERROR (exit $exit_code)" "$first_fail"
        (( FAIL++ )) || true
        FAILED_TESTS+=("$label")
    fi
}

# Run a binary as both 1-rank serial and NRANKS-rank parallel.
# Usage: run_serial_and_parallel LABEL_SER LABEL_PAR BINARY [ARGS...]
run_serial_and_parallel() {
    local lser="$1" lpar="$2"; shift 2
    run_serial   "$lser" "$@"
    run_parallel "$lpar" "$@"
}

# Print a section header.
section() { printf "\n── %s\n" "$1"; }

# ── Banner ────────────────────────────────────────────────────────────────────
printf "╔══════════════════════════════════════════════════════════╗\n"
printf "║  MMA/GCMMA/SQ test suite                                ║\n"
printf "╠══════════════════════════════════════════════════════════╣\n"
printf "║  Build dir : %-43s║\n" "$BUILD_DIR"
printf "║  MPI ranks : %-43s║\n" "$NRANKS"
printf "║  Mode      : %-43s║\n" "$MODE"
printf "╚══════════════════════════════════════════════════════════╝\n"

# =============================================================================
# ── Core: serial ─────────────────────────────────────────────────────────────
# =============================================================================
if [[ "$MODE" == "serial" || "$MODE" == "all" ]]; then

section "Core: serial"
run_serial  mma_serial              test_mma_serial
run_serial  mma_unconstrained       test_mma_unconstrained
run_serial  gcmma_serial            test_gcmma
run_serial  gcmma_callback_serial   test_gcmma_callback
run_serial  zero_ranks_1            test_zero_ranks
run_serial  overconstrained_serial  test_overconstrained
run_serial  equalities_serial       test_equalities

# ── Device (CPU) ─────────────────────────────────────────────────────────────

section "Device (CPU)"
run_serial  mma_device_cpu          test_mma_device  --device cpu
run_serial  sq_device_cpu           test_sq_device   --device cpu

# ── SQ approximation: serial ─────────────────────────────────────────────────

section "SQ approximation: serial"
run_serial  sq_combined             test_sq
run_serial  sq_serial               test_sq_serial
run_serial  sq_unconstrained        test_sq_unconstrained
run_serial  sq_gcmma_serial         test_sq_gcmma
run_serial  sq_gcmma_cb_serial      test_sq_gcmma_callback
run_serial  sq_overconstrained      test_sq_overconstrained
run_serial  sq_equalities_serial    test_sq_equalities
run_serial  sq_nonconvex_serial     test_sq_nonconvex

# ── Relaxed equalities: serial ───────────────────────────────────────────────
# Tests for PackFivalRelaxed() and WithRelaxedEqualities() — serial classes only.
# Binary: test_relaxed_equalities_serial (no MPI_Init required).

section "Relaxed equalities: serial"
run_serial  relaxed_eq_serial       test_relaxed_equalities_serial

fi  # MODE serial | all

# =============================================================================
# ── Core: parallel ───────────────────────────────────────────────────────────
# =============================================================================
if [[ "$MODE" == "parallel" || "$MODE" == "all" ]]; then

section "Core: parallel ($NRANKS ranks)"
run_parallel  mma_parallel          test_mma_parallel
run_parallel  mma_unconstrained_par test_mma_unconstrained
run_parallel  equalities_par        test_equalities

# ── SQ approximation: parallel ───────────────────────────────────────────────

section "SQ approximation: parallel ($NRANKS ranks)"
run_parallel  sq_combined_par       test_sq
run_parallel  sq_par_serial         test_sq_parallel
run_parallel  sq_unconstrained_par  test_sq_unconstrained

# ── Relaxed equalities: parallel ─────────────────────────────────────────────
# Tests for MMAOptimizerParallel and SQOptimizerParallel with relaxed bands.
# Binary: test_relaxed_equalities_parallel (requires MPI).
#   Serial entry  — runs the parallel binary with 1 rank (tests the serial fallback).
#   Parallel entry — runs with NRANKS ranks.

section "Relaxed equalities: parallel ($NRANKS ranks)"
run_serial_and_parallel \
    relaxed_eq_par_serial  \
    relaxed_eq_par_${NRANKS}r \
    test_relaxed_equalities_parallel

fi  # MODE parallel | all

# =============================================================================
# ── Summary ───────────────────────────────────────────────────────────────────
# =============================================================================
TOTAL=$(( PASS + FAIL + SKIP ))
printf "\n╔══════════════════════════════════════════════════════════╗\n"
printf "║  Results: %d passed  %d failed  %d skipped  (%d total)%*s║\n" \
    $PASS $FAIL $SKIP $TOTAL \
    $(( 20 - ${#PASS} - ${#FAIL} - ${#SKIP} - ${#TOTAL} )) ""

if [[ $FAIL -eq 0 ]]; then
    printf "║  All tests PASSED.%-39s║\n" ""
else
    printf "║  FAILED tests:%-43s║\n" ""
    for t in "${FAILED_TESTS[@]}"; do
        printf "║    %-54s║\n" "$t"
    done
fi
printf "╚══════════════════════════════════════════════════════════╝\n"

[[ $FAIL -eq 0 ]]
