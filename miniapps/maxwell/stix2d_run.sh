#!/bin/bash

# echo
# if [ $formulation == ultraweak ]; then
#     exec=DPG_uweak_acoustics
#     echo "Running ultraweak formulation..."
#     echo "Problem          = " $prob_verb
#     echo "geometry file    = " $geometry
#     echo "Order            = " $p
#     echo "Enriched order   = " $dp
#     echo "Test norm        = " $testnorm_verb
#     echo "bc               = " $bc_verb
#     echo "Exact solution   = " $exact_verb
#     echo "# of wavelengths = " $rnum
#     echo "solver           = " $solver_verb
#     echo "pml              = " $ipml
#     echo "OMP threads      = " $nthreads
# fi
# echo
# read -p "Press [Enter] key to continue..."

# Executable
exec=stix2d
# mpi processes
nproc=6


# Problem specifics
order=1
numref=2
mgref=1
comp_conv=herm
dirichlet='3 5'
frequency=8e5
magnetic_flux='0 0 5.4'
wave_type=J
slab_params='0 1 0 0.06 0.02'
num_densities='2e20 2e20'
# mesh_dim = 0.24

# mpirun -np 6 ./stix1d -o 1 -nex 480 -ney 3 -nez 3 -dbcs '3 5' -f 8e6 -B '0 0 5.4' -w J -slab '0 1 0 0.06 0.02' -num '2e20 2e20' -herm
# -B '0 0 5.4' -slab '0 1 0 0.06 0.02' -num '2e20 2e20' -herm

mpirun -np 6 ./stix2d -rod '0 0 1 0 0 0.1' -dbcs '1' -w Z -o 2 -f 1e6

# # gdb --args \
# mpirun -np $nproc ./$exec \
#                            -o           $order      \
#                            -initref     $numref     \
#                            -maxref      $mgref     \
#                            -$comp_conv              \
#                            -nex         $num_elem_x \
#                            -ney         $num_elem_y \
#                            -nez         $num_elem_z \
#                            -f           $frequency  \
#                            -w           $wave_type  \
#                            -dbcs        "${dirichlet}" \
#                            -B           "${magnetic_flux}"   \
#                            -slab        "${slab_params}"  \
#                            -num         "${num_densities}"
