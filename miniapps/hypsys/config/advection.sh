# EXEC="mpirun -np 7 phypsys"
EXEC="./hypsys"

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
ORDER=1
CONFIG="-p 0 -c 3 -vs 1000 -tf 1 -s 3 -dt 0.0004 -m $MESH -o $ORDER -e $SCHEME"

# $EXEC $CONFIG -r 2
# $EXEC $CONFIG -r 3
# $EXEC $CONFIG -r 4
# $EXEC $CONFIG -r 5
# $EXEC $CONFIG -r 6
# $EXEC $CONFIG -r 7