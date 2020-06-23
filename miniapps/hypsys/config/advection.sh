EXEC="mpirun -np 7 phypsys"

SCHEME=0
MESH=data/periodic-segment.mesh
CONFIG="-p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -m $MESH"

# $EXEC $CONFIG -e $SCHEME -o 1 -r 5
# $EXEC $CONFIG -e $SCHEME -o 3 -r 4
# $EXEC $CONFIG -e $SCHEME -o 7 -r 3

SCHEME=1

$EXEC $CONFIG -e $SCHEME -o 1 -r 5
$EXEC $CONFIG -e $SCHEME -o 3 -r 4
$EXEC $CONFIG -e $SCHEME -o 7 -r 3

# Grid convergence test
rm errors.txt

SCHEME=1
# MESH0=data/periodic-3segment.mesh
# ORDER=1 # -dt 0.0004
# # ORDER=2 # -dt 0.00025
# # ORDER=3 # -dt 0.0001
# # ORDER=4 # -dt 0.00005
# CONFIG0="-p 0 -c 3 -vs 1000 -tf 1 -s 3 -dt 0.0001 -m $MESH0 -o $ORDER -e $SCHEME"
# CONFIG1="-p 0 -c 3 -vs 1000 -tf 1 -s 3 -dt 0.0001 -m $MESH -o $ORDER -e $SCHEME"

# $EXEC $CONFIG0 -r 4
# $EXEC $CONFIG1 -r 4
# $EXEC $CONFIG0 -r 5
# $EXEC $CONFIG1 -r 5
# $EXEC $CONFIG0 -r 6
# $EXEC $CONFIG1 -r 6
# $EXEC $CONFIG0 -r 7
