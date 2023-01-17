#!/bin/bash
#SBATCH -A m2698
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=60
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64

echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
srun -n 128 --cpu-bind=cores -c 4 ../imAMRMHDp -m ../Meshes/xperiodic_plasmoid_short.mesh -rs 5 -rp 1 -o 3 -i 7 -no-vis -tf 700 -dt .5 -vs 2 -usepetsc --petscopts ../petscrc/rc_debug2 -s 3 -shell -amrl 4 -ltol 4e-3 -derefine -resi 3.3e-3 -visc 3.3e-3 -refs 8 -im_supg 1 -supg -i_supgpre 3 -no-fd -yrange -ytop 20.0 -L0 10 -err-ratio 0.1 -derefine-ratio 0.0001 -err-fraction 0.3 -derefine-fraction 0.0001  -init-refine -beta 0.0001 -x-factor 5
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"
