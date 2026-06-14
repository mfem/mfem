#!/usr/bin/env bash
# =============================================================================
# run_tests.sh  —  MMA/GCMMA/SQ MFEM test runner
#
# Usage:
#   ./run_tests.sh [OPTIONS]
#
# Options:
#   -b DIR      Build directory (default: ./build)
#   -j N        MPI ranks for parallel tests (default: 4)
#   --serial    Run only serial (1-rank) tests
#   --gpu       Also run GPU device test (requires CUDA/HIP MFEM build)
#   --large     Pass --large to test_nonconvex / test_sq_nonconvex (very slow, 100k–1M DOF)
#   --units     Include unit tests (test_solvedense, test_packfival)
#   --legacy    Include legacy test_mma_mfem
#   --debug     Include test_eq_debug
#   -h          Show this help
#
# Exit code: 0 if all selected tests pass, 1 if any fail.
#
# Examples:
#   ./run_tests.sh                          # full suite, 4 MPI ranks
#   ./run_tests.sh --serial                 # no mpirun required
#   ./run_tests.sh -b ./build -j 8         # custom build dir and rank count
#   ./run_tests.sh --gpu --large            # include GPU and large tests
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BUILD_DIR="./build"
NP=4
SERIAL_ONLY=0
GPU=0
LARGE=0
UNITS=0
LEGACY=0
DEBUG=0
MPIRUN="${MPIRUN:-mpirun}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b)         BUILD_DIR="$2"; shift 2 ;;
        -j)         NP="$2";        shift 2 ;;
        --serial)   SERIAL_ONLY=1;  shift   ;;
        --gpu)      GPU=1;          shift   ;;
        --large)    LARGE=1;        shift   ;;
        --units)    UNITS=1;        shift   ;;
        --legacy)   LEGACY=1;       shift   ;;
        --debug)    DEBUG=1;        shift   ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# ====/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BUILD_DIR="${BUILD_DIR%/}"   # strip trailing slash

# ---------------------------------------------------------------------------
# Terminal colours and counters
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

PASS=0; FAIL=0; SKIP=0
FAILED_TESTS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# run LABEL CMD [ARGS…]
#   Runs a test, prints PASS/FAIL/ERROR, records the result.
run() {
    local label="$1"; shift
    printf "  %-45s " "$label"
    local out
    if out=$("$@" 2>&1); then
        if echo "$out" | grep -q '^\s*\[FAIL\]'; then
            echo -e "${RED}FAIL${NC}"
            echo "$out" | grep '^\s*\[FAIL\]' | sed 's/^/    /'
            (( FAIL++ )) || true
            FAILED_TESTS+=("$label")
        else
            echo -e "${GREEN}PASS${NC}"
            (( PASS++ )) || true
        fi
    else
        local code=$?
        echo -e "${RED}ERROR (exit ${code})${NC}"
        if echo "$out" | grep -q '^\s*\[FAIL\]'; then
            echo "$out" | grep '^\s*\[FAIL\]' | sed 's/^/    /'
        else
            echo "$out" | tail -8 | sed 's/^/    /'
        fi
        (( FAIL++ )) || true
        FAILED_TESTS+=("$label")
    fi
}

# skip LABEL  — record a deliberate skip
skip() {
    printf "  %-45s " "$1"
    echo -e "${YELLOW}SKIP${NC}"
    (( SKIP++ )) || true
}

# section TITLE — print a section header
section() { echo; echo -e "${CYAN}── $1${NC}"; }

# require BIN — abort if the binary doesn't exist
require() {
    if [[ ! -x "$BUILD_DIR/$1" ]]; then
        echo -e "${RED}Error:${NC} $BUILD_DIR/$1 not found. Build the project first."
        exit 1
    fi
}

# serial  BIN [ARGS…] — run 1-rank test
serial()   { "$BUILD_DIR/$1" "${@:2}"; }

# par BIN [ARGS…] — run NP-rank test
par()      { "$MPIRUN" -np "$NP" "$BUILD_DIR/$1" "${@:2}"; }

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  MMA/GCMMA MFEM test suite                              ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════════╣${NC}"
printf  "${BOLD}║  Build dir : %-43s║${NC}\n" "$BUILD_DIR"
printf  "${BOLD}║  MPI ranks : %-43s║${NC}\n" "$NP"
printf  "${BOLD}║  Mode      : %-43s║${NC}\n" \
    "$([ $SERIAL_ONLY -eq 1 ] && echo "serial only" || echo "serial+parallel")"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"

[[ -d "$BUILD_DIR" ]] || { echo "Error: build directory '$BUILD_DIR' not found."; exit 1; }

# ---------------------------------------------------------------------------
# Core MMA / GCMMA  —  serial
# ---------------------------------------------------------------------------
section "Core: serial"
require test_mma_serial
run "mma_serial"              serial test_mma_serial
run "mma_unconstrained"       serial test_mma_unconstrained
run "gcmma_serial"            serial test_gcmma
run "gcmma_callback_serial"   serial test_gcmma_callback
run "zero_ranks_1"            serial test_zero_ranks
run "overconstrained_serial"  serial test_overconstrained
run "equalities_serial"       serial test_equalities

# ---------------------------------------------------------------------------
# Device test (CPU)
# ---------------------------------------------------------------------------
section "Device (CPU)"
require test_mma_device
run "mma_device_cpu"  serial test_mma_device --device cpu

# ---------------------------------------------------------------------------
# Device test (GPU) — opt-in with --gpu
# ---------------------------------------------------------------------------
if [[ $GPU -eq 1 ]]; then
    section "Device (GPU)"
    if command -v nvidia-smi &>/dev/null; then
        run "mma_device_cuda"  serial test_mma_device --device cuda
    elif command -v rocm-smi &>/dev/null; then
        run "mma_device_hip"   serial test_mma_device --device hip
    else
        skip "mma_device_gpu (no GPU detected)"
    fi
fi

# ---------------------------------------------------------------------------
# SQ approximation  —  serial
# ---------------------------------------------------------------------------
section "SQ approximation: serial"
require test_sq_serial
run "sq_combined"             serial test_sq
run "sq_serial"               serial test_sq_serial
run "sq_unconstrained"        serial test_sq_unconstrained
run "sq_gcmma_serial"         serial test_sq_gcmma
run "sq_gcmma_cb_serial"      serial test_sq_gcmma_callback
run "sq_device_cpu"           serial test_sq_device --device cpu
run "sq_overconstrained"      serial test_sq_overconstrained
run "sq_equalities_serial"    serial test_sq_equalities
run "sq_nonconvex_serial"     serial test_sq_nonconvex

# ---------------------------------------------------------------------------
# SQ approximation  —  parallel
# ---------------------------------------------------------------------------
if [[ $SERIAL_ONLY -eq 0 ]]; then
    section "SQ approximation: parallel (${NP} ranks)"
    run "sq_combined_par_${NP}r"        par test_sq
    run "sq_par_serial_${NP}r"          par test_sq_serial
    run "sq_unconstrained_par_${NP}r"   par test_sq_unconstrained
    run "sq_gcmma_par_${NP}r"           par test_sq_gcmma
    run "sq_gcmma_cb_par_${NP}r"        par test_sq_gcmma_callback
    run "sq_zero_ranks_${NP}r"          par test_sq_zero_ranks
    run "sq_overconstrained_par_${NP}r" par test_sq_overconstrained
    run "sq_equalities_par_${NP}r"      par test_sq_equalities
    run "sq_nonconvex_par_${NP}r"       par test_sq_nonconvex
    run "sq_parallel_${NP}r"            par test_sq_parallel
fi

# ---------------------------------------------------------------------------
# Nonconvex SIMP
# ---------------------------------------------------------------------------
section "Nonconvex SIMP"
require test_nonconvex
run "nonconvex_serial"  serial test_nonconvex
if [[ $LARGE -eq 1 ]]; then
    run "nonconvex_serial_large"    serial test_nonconvex    --large
    run "sq_nonconvex_serial_large" serial test_sq_nonconvex --large
fi

# ---------------------------------------------------------------------------
# Core MMA / GCMMA  —  parallel
# ---------------------------------------------------------------------------
if [[ $SERIAL_ONLY -eq 0 ]]; then
    section "Parallel (${NP} ranks)"
    require test_mma_parallel
    run "mma_parallel_${NP}r"           par test_mma_parallel
    run "mma_serial_par_${NP}r"         par test_mma_serial
    run "gcmma_parallel_${NP}r"         par test_gcmma
    run "gcmma_callback_par_${NP}r"     par test_gcmma_callback
    run "zero_ranks_${NP}r"             par test_zero_ranks
    run "overconstrained_par_${NP}r"    par test_overconstrained
    run "equalities_parallel_${NP}r"    par test_equalities
    run "nonconvex_parallel_${NP}r"     par test_nonconvex
    if [[ $LARGE -eq 1 ]]; then
        run "nonconvex_par_large_${NP}r"     par test_nonconvex    --large
        run "sq_nonconvex_par_large_${NP}r" par test_sq_nonconvex --large
    fi
fi

# ---------------------------------------------------------------------------
# Unit tests  —  opt-in with --units
# ---------------------------------------------------------------------------
if [[ $UNITS -eq 1 ]]; then
    section "Unit tests"
    [[ -x "$BUILD_DIR/test_solvedense" ]] && \
        run "solvedense"  serial test_solvedense || \
        skip "test_solvedense (not built)"
    [[ -x "$BUILD_DIR/test_packfival" ]] && \
        run "packfival"   serial test_packfival  || \
        skip "test_packfival (not built)"
fi

# ---------------------------------------------------------------------------
# Debug  —  opt-in with --debug
# ---------------------------------------------------------------------------
if [[ $DEBUG -eq 1 ]] && [[ -x "$BUILD_DIR/test_eq_debug" ]]; then
    section "Debug"
    run "eq_debug_serial"  serial test_eq_debug
fi

# ---------------------------------------------------------------------------
# Legacy  —  opt-in with --legacy
# ---------------------------------------------------------------------------
if [[ $LEGACY -eq 1 ]] && [[ -x "$BUILD_DIR/test_mma_mfem" ]]; then
    section "Legacy"
    run "mma_mfem_legacy"  serial test_mma_mfem
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
TOTAL=$(( PASS + FAIL ))
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
if [[ $FAIL -eq 0 ]]; then
    printf "${BOLD}║  ${GREEN}All %d test(s) PASSED${NC}${BOLD}%-36s║${NC}\n" "$TOTAL" ""
else
    printf "${BOLD}║  ${RED}%d of %d test(s) FAILED${NC}${BOLD}%-34s║${NC}\n" "$FAIL" "$TOTAL" ""
fi
[[ $SKIP -gt 0 ]] && printf "${BOLD}║  %d test(s) skipped%-38s║${NC}\n" "$SKIP" ""
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"

if [[ $FAIL -gt 0 ]]; then
    echo
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do echo "  • $t"; done
    exit 1
fi
exit 0
