EXEC="mpirun -np 7 phypsys"

SCHEME=1
MESH=data/wall-bdr-4segment.mesh

## SOD Shock tube
CONFIG="-p 5 -c 1 -vf 1000 -tf 0.231 -s 3 -m $MESH -es $SCHEME"

# # h-refinement
# $EXEC $CONFIG -o 1 -r 5 -dt 0.00064
# $EXEC $CONFIG -o 1 -r 6 -dt 0.00032
# $EXEC $CONFIG -o 1 -r 7 -dt 0.00016

# p-refinement & h-coarsening
$EXEC $CONFIG -r 5 -o 1 -dt 0.0004
$EXEC $CONFIG -r 4 -o 3 -dt 0.0004
$EXEC $CONFIG -r 3 -o 7 -dt 0.0004
$EXEC $CONFIG -r 2 -o 15 -dt 0.0004
$EXEC $CONFIG -r 1 -o 31 -dt 0.0004


# ## Woodward Colella
# MESH=data/wall-bdr-100segment.mesh
# CONFIG="-p 5 -c 2 -vf 1000 -tf 0.038 -s 3 -m $MESH -es $SCHEME"
# $EXEC $CONFIG -o 1 -r 2 -dt 1e-6


# ## Double Mach reflection
# MESH=data/double-mach-quad.mesh
# CONFIG="-p 5 -c 3 -vf 1000 -tf 0.2 -s 3 -m $MESH -es $SCHEME"
# $EXEC $CONFIG -o 1 -r 3 -dt 5e-5


# ## Vortex advection
# SCHEME=0
# MESH3=data/periodic-3quad.mesh
# MESH4=data/periodic-4quad.mesh
# ORDER=1
# ODESOLVER=2
# DT=0.000625
# CONFIG="-p 5 -c 0 -vf 100 -tf 1 -s $ODESOLVER -dt $DT -o $ORDER -es $SCHEME"

# rm errors.txt
# $EXEC $CONFIG -m $MESH3 -r 3
# $EXEC $CONFIG -m $MESH4 -r 3
# $EXEC $CONFIG -m $MESH3 -r 4
# $EXEC $CONFIG -m $MESH4 -r 4
# $EXEC $CONFIG -m $MESH3 -r 5
