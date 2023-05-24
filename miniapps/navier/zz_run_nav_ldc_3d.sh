#!/bin/bash
#BSUB -G asccasc
#BSUB -W 4:00
#BSUB -nnodes 1
#BSub -q pbatch
#BSUB -B
#BSUB -N
#BSUB -u gillette7@llnl.gov
#BSUB -o yy_jobinfo_%J.out

# on lassen: max 4 cores per node i.e. n <= 4*N

cd /usr/WS1/gillette/nfp4va/mfemstuff/mfem/miniapps/navier
srun -n32 navier_ldc_3d --device cuda
# jsrun --nrs 32 --rs_per_host 4 --np 32 navier_ldc_3d --device cuda
# srun -n32 -ppdebug navier_ldc_3d --device cuda
echo 'Done'