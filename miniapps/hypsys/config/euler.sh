EXEC="mpirun -np 7 phypsys"
EXEC="./hypsys"

SCHEME=1
MESH=data/wall-bdr-4segment.mesh

## SOD Shock tube
CONFIG="-p 4 -c 1 -vs 1000 -tf 0.231 -s 3 -m $MESH -e $SCHEME"

# h-refinement
$EXEC $CONFIG -o 1 -r 5 -dt 0.00064
$EXEC $CONFIG -o 1 -r 6 -dt 0.00032
$EXEC $CONFIG -o 1 -r 7 -dt 0.00016
# $EXEC $CONFIG -o 1 -r 8 -dt 0.00008

# # p-refinement & h-coarsening
$EXEC $CONFIG -r 5 -o 1 -dt 0.0005
$EXEC $CONFIG -r 4 -o 3 -dt 0.0005
$EXEC $CONFIG -r 3 -o 7 -dt 0.0005


# ## Woodward Colella
# MESH=data/wall-bdr-100segment.mesh
# CONFIG="-p 4 -c 2 -vs 1000 -tf 0.038 -s 3 -m $MESH -e $SCHEME"
# $EXEC $CONFIG -o 1 -r 2 -dt 1e-6


# ## Double Mach reflection
# MESH=data/double-mach-quad.mesh
# CONFIG="-p 4 -c 3 -vs 1000 -tf 0.2 -s 3 -m $MESH -e $SCHEME"
# $EXEC $CONFIG -o 1 -r 3 -dt 5e-5


# ## Smooth vortex
# SCHEME=1
# MESH=data/periodic-square.mesh
# CONFIG="-p 4 -c 0 -vs 100 -tf 2 -s 3 -dt 0.0008 -m $MESH -e $SCHEME"

# rm errors.txt

# $EXEC $CONFIG -o 1 -r 2
# $EXEC $CONFIG -o 1 -r 3
# $EXEC $CONFIG -o 1 -r 4
# $EXEC $CONFIG -o 1 -r 5