EXEC="mpirun -np 7 phypsys"

## 1D Dam break
SCHEME=1
MESH=data/wall-bdr-4segment.mesh
CONFIG="-p 4 -c 1 -vf 1000 -tf 0.02 -s 3 -m $MESH -es $SCHEME"

# # h-refinement
# $EXEC $CONFIG -o 1 -r 5 -dt 0.0001
# $EXEC $CONFIG -o 1 -r 6 -dt 0.00005
# $EXEC $CONFIG -o 1 -r 7 -dt 0.000025
# $EXEC $CONFIG -o 1 -r 8 -dt 0.0000125

# # p-refinement & h-coarsening
# $EXEC $CONFIG -o 1 -r 5 -dt 0.000025
# $EXEC $CONFIG -o 3 -r 4 -dt 0.000025
# $EXEC $CONFIG -o 7 -r 3 -dt 0.000025

# ## Radial dambreak
# MESH=data/outflow-bdr-4quad.mesh
# CONFIG="-p 4 -c 2 -vf 50 -tf 0.06 -s 3 -m $MESH -es 1"

# # p-refinement & h-coarsening
# $EXEC $CONFIG -o 1 -r 5 -dt 0.0001
# $EXEC $CONFIG -o 3 -r 4 -dt 0.0001
# $EXEC $CONFIG -o 7 -r 3 -dt 0.0001

# ## Constricted channel
# MESH=data/constr-channel.mesh
# CONFIG="-p 4 -c 3 -vf 100 -tf 1000 -s 1 -m $MESH -es $SCHEME"
# $EXEC $CONFIG -r 1 -o 1 -dt 0.025

## Vortex advection
SCHEME=0
MESH3=data/periodic-3quad.mesh
MESH4=data/periodic-4quad.mesh

ORDER=1
ODESOLVER=2
DT=0.00064
# ORDER=2
# ODESOLVER=3
# DT=0.0004
# ORDER=3
# ODESOLVER=3
# DT=0.00025
CONFIG="-p 4 -c 0 -vf 100 -tf 1 -s $ODESOLVER -dt $DT -o $ORDER -es $SCHEME"

rm errors.txt
$EXEC $CONFIG -m $MESH3 -r 3
$EXEC $CONFIG -m $MESH4 -r 3
$EXEC $CONFIG -m $MESH3 -r 4
$EXEC $CONFIG -m $MESH4 -r 4
$EXEC $CONFIG -m $MESH3 -r 5
