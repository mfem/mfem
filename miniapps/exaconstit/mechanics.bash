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
srun -n1 ./mechanics_driver -m ../../data/cube-hex-ro.mesh -tf 1.0 -dt 1.0 -attrid '1 2 3 4' -disp '0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. -0.002' -bcid '3 1 2 3' -rs 0

