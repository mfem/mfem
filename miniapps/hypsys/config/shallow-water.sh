EXEC="mpirun -np 3 phypsys"

SCHEME=1
MESH=data/wall-bdr-4segment.mesh
CONFIG="-p 3 -c 1 -vs 1000 -tf 0.02 -s 3 -m $MESH -e $SCHEME"

## 1D Dam break

# h-refinement
# $EXEC $CONFIG -o 1 -r 5 -dt 0.0001
# $EXEC $CONFIG -o 1 -r 6 -dt 0.00005
# $EXEC $CONFIG -o 1 -r 7 -dt 0.000025
# $EXEC $CONFIG -o 1 -r 8 -dt 0.0000125

# # p-refinement & h-coarsening
# $EXEC $CONFIG -o 1 -r 5 -dt 0.000025
# $EXEC $CONFIG -o 3 -r 4 -dt 0.000025
# $EXEC $CONFIG -o 7 -r 3 -dt 0.000025

## Radial dambreak
MESH=data/outflow-bdr-4quad.mesh
CONFIG="-p 3 -c 2 -vs 50 -tf 0.6 -s 3 -m $MESH -e $SCHEME"

# p-refinement & h-coarsening
$EXEC $CONFIG -o 1 -r 5 -dt 0.0001
$EXEC $CONFIG -o 3 -r 4 -dt 0.0001
$EXEC $CONFIG -o 7 -r 3 -dt 0.0001

## Constricted channel
# MESH=data/constr-channel.mesh
# CONFIG="-p 3 -c 3 -vs 100 -tf 1000 -s 1 -m $MESH -e $SCHEME"
# $EXEC $CONFIG -r 2 -o 2 -dt 0.002