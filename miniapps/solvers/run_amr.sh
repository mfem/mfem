#!/bin/bash

ORDERS="1 2 3 4 5 6 7"

for p in $ORDERS
do
   MFEM_EKS=1 MFEM_NVTX=1 lrun -n 12 nsys profile -t nvtx,cuda -s none -o amr/prof/o${p}_%q{OMPI_COMM_WORLD_RANK} -f true ./plor_solvers -m ./amr_mesh.mesh -o $p -rs 0 -rp 0 -d cuda | tee amr/out/o$p.txt
done

# After creating the NVTX files, convert to CSV using the following:

# for p in 1 2 3 4 5 6 7
# do
#    nsys stats -r nvtxsum -f csv o${p}_0.nsys-rep > o${p}.csv
# done
