#!/usr/bin/env bash

solver_types="FULL_CG LOCAL_CG_LOBATTO LOCAL_CG_LEGENDRE DIRECT DIRECT_CUSOLVER DIRECT_CUBLAS"
op_type="Setup Solve SetupAndSolve"

GREEN='\033[0;32m'
NC='\033[0m'

for stype in $solver_types
do
   for op in $op_type
   do
      bench_name="${stype}_${op}"
      bench_filter="${bench_name}_0_5/.*/.*"
      output_name="${bench_name,,}.csv"
      cmd="./bench_dgmassinv --benchmark_filter=${bench_filter} --benchmark_context=device=cuda --benchmark_out_format=csv --benchmark_out=${output_name}"
      echo -e "${GREEN}${cmd}${NC}"
      $cmd
   done
done
