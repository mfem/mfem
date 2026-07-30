#!/bin/bash

# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Run the MFEM benchmarks that are tracked by CodSpeed, see
# .github/workflows/codspeed.yml.
#
# The benchmarks are measured with CodSpeed's CPU simulation instrument, which
# runs every benchmark once inside a CPU simulator: this gives low-variance
# results, but it is also about 20 times slower than a native run. Only the
# smallest configuration of each benchmark suite is therefore selected here,
# which keeps the total run time of the order of ten minutes while still
# covering all the kernels.
#
# The Google Benchmark filters below are matched against the CodSpeed benchmark
# names, i.e. '<source file>::<benchmark name>[<arguments>]', and the square
# brackets are escaped as '[[]' and '[]]' since the filters are POSIX extended
# regular expressions.
#
# Usage:
#   tests/benchmarks/codspeed.sh [<directory with the bench_* executables>]
#
# To measure and upload the results, run the script through the CodSpeed CLI:
#   codspeed run --mode simulation -- tests/benchmarks/codspeed.sh build/tests/benchmarks

set -e

BENCH_DIR="${1:-.}"

run()
{
   local bench="$1"
   shift
   echo "--- ${bench} $*"
   "${BENCH_DIR}/${bench}" "$@"
}

# Vector kernels and virtual function overhead: all the sizes, up to 1 KB.
run bench_vector
run bench_virtuals

# Bake-off problems (PCG solves), their setup and the bake-off kernels, for the
# partial, element and full assembly levels: 1024 dofs, orders 1 to 3.
run bench_assembly_levels --benchmark_filter='[[]1024/[123][]]$'

# DG convection on non-conforming meshes: 1024 dofs, orders 1 to 3.
run bench_dg_amr --benchmark_filter='[[]1024/[123]/'

# Elasticity gradient and its action: 10x10x10 elements, orders 1 to 3.
run bench_elasticity --benchmark_filter='[[][123]/10/[01][]]$'

# TMOP kernels: orders 1 and 2, with and without the p == q optimization.
run bench_tmop --benchmark_filter='[[][01]/[12][]]$'

# CEED bake-off kernels, on hexahedra and tetrahedra: smallest mesh (16 cells
# per direction), orders 1 to 3. The bake-off problems of this suite are not
# included: their PCG solves are too slow to run inside the CPU simulator.
run bench_ceed --benchmark_filter='BK[0-9][[]BK.*[[][123]/16[]]$'
