#!/bin/bash
#BSUB -G asccasc
#BSUB -t 4:00:00
#BSUB -nnodes 8
#BSUB --mail-type=ALL

# on lassen: max 4 cores per node i.e. n <= 4*N

cd /usr/WS1/gillette/nfp4va/mfemstuff/mfem/miniapps/navier
jsrun --nrs 32 --rs_per_host 4 --np 32 navier_ldc_3d --device cuda
# srun -n32 -ppdebug navier_ldc_3d --device cuda
echo 'Done'