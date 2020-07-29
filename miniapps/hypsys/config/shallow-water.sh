EXEC="mpirun -np 7 phypsys"

## 1D Dam break
SCHEME=1
MESH=data/wall-bdr-4segment.mesh
CONFIG="-p 4 -c 1 -vf 1000 -tf 0.02 -s 3 -m $MESH -e $SCHEME"

# h-refinement
# $EXEC $CONFIG -o 1 -r 5 -dt 0.0001
# $EXEC $CONFIG -o 1 -r 6 -dt 0.00005
# $EXEC $CONFIG -o 1 -r 7 -dt 0.000025
# $EXEC $CONFIG -o 1 -r 8 -dt 0.0000125

# # p-refinement & h-coarsening
$EXEC $CONFIG -o 1 -r 5 -dt 0.000025
$EXEC $CONFIG -o 3 -r 4 -dt 0.000025
$EXEC $CONFIG -o 7 -r 3 -dt 0.000025

# ## Radial dambreak
# MESH=data/outflow-bdr-4quad.mesh
# CONFIG="-p 4 -c 2 -vf 50 -tf 0.01 -s 3 -m $MESH -e 1"

# # p-refinement & h-coarsening
# $EXEC $CONFIG -o 1 -r 5 -dt 0.0001
# $EXEC $CONFIG -o 3 -r 4 -dt 0.0001
# $EXEC $CONFIG -o 7 -r 3 -dt 0.0001

# ## Constricted channel
# MESH=data/constr-channel.mesh
# CONFIG="-p 4 -c 3 -vf 100 -tf 1000 -s 1 -m $MESH -e $SCHEME"
# $EXEC $CONFIG -r 1 -o 1 -dt 0.025
