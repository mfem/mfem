# -tf is the final time
#
#     Note, for testing I'm taking a single time step and the 
#     displacement is small enough that the problem is linear and should converge 
#     in a single Newton iteration, which it does for the NeoHookean material
#
# -dt is the time step. 
#
# -attrid specifies the boundary attribute ids specified in the mesh file for which there are both 
# homogeneous and inhomogeneous Dirichlet boundary conditions. 
#
# -disp specifies the three displacement components to be enforced for each boundary attribute id specified in -attrid
#
# -bcid specifies the Dirichlet boundary condition id dictating which component combination is to be enforced.
# These are as follows: x,y,z (-1), x (1), y (2), z (3), xy (4), yz (5), xz (6), free (7).
#
#export ASAN_SYMBOLIZER_PATH=/usr/tce/packages/clang/clang-6.0.0/bin/llvm-symbolizer
export ASAN_OPTIONS=halt_on_error=0
export ASAN_OPTIONS=fast_unwind_on_malloc=0
#srun -ppdebug -n8 ./mechanics_driver -hexmesh -mx 1.0 -nx 10 -tf 1.0 -dt 0.1 -vardt -nsteps 2 -cust-dt 'custom_dt.txt' -attrid '1 2 3 4' -disp '0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.001' -bcid '3 1 2 3' -rs 0 --visit -nsvars 43 -svars 'state_cp_becker.txt' -nprops 88 -props 'props_cp_becker.txt' -umat -cp -ng 500 -gmap './CA_results/grainIDs_u.in' -g './CA_results/fe.in' -gc -gcstride 9 -rel 5e-6 -abs 1e-8 -no-pcg -gmres -m ../../data/cube-hex-ro.mesh
srun -ppdebug -n1 ./mechanics_driver -hexmesh -mx 1.0 -nx 1 -tf 1.0 -dt 1.0 -attrid '1 2 3 4' -disp '0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.01' -bcid '3 1 2 3' -rs 0 -visit -nsvars 1 -svars 'state_lin.txt' -nprops 2 -props 'props_lin.txt' -umat -rel 1e-6 -abs 1e-3 #-no-pcg -gmres
