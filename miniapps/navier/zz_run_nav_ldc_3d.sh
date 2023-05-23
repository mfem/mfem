#!/bin/bash
#SBATCH -A asccasc
#SBATCH -t 4:00:00
#SBATCH -N 8
#SBATCH -o yy_%j.out
#SBATCH -J slurm1
#SBATCH --mail-type=ALL
#SBATCH -p pbatch

# on lasseen: max 4 cores per node i.e. n <= 4*N

cd /usr/WS1/gillette/nfp4va/mfemstuff/mfem/miniapps/navier
srun -n32 -ppdebug navier_ldc_3d --device cuda
echo 'Done'