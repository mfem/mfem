#!/bin/bash

# Launch with: flux batch --exclusive -q pbatch -N 1024 plor-4k.flux

#flux: --job-name=plor-4k       # Job name
#flux: --output='{{id}}.out'    # Name of stdout output file
#flux: --error='{{id}}.err'     # Name of stderr error file
#flux: -t 10                    # Run time (minutes)

machine=`hostname -s`
today=`date +"%H:%M.%m-%d-%Y.%a"`
commit=`git log --pretty=format:'%h' -n 1`
output=${commit}-${today}.out

export EKS=1
export MFEM_DEBUG=1
export MFEM_DEBUG_MPI=0
export MPICH_GPU_SUPPORT_ENABLED=1

echo "Start:     `date`" > ${output}
echo "Machine:   ${machine} (`flux resource info`)" >> ${output}
echo "Path:      `pwd`" >> ${output}
echo "===============================================================" >> ${output}
flux run --exclusive -N 1024 --tasks-per-node=4 ./bench_plor --benchmark_context=device=hip,ndev=1 --benchmark_out_format=csv --benchmark_out=plor-p6-x4096-{{id}}.org --benchmark_filter=pLOR
echo "===============================================================" >> ${output}
echo "End:        `date`" >> ${output}

