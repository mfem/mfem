EXEC="mpirun -np 7 phypsys"

SCHEME=0
MESH=data/periodic-segment.mesh
CONFIG="-p 0 -c 2 -vf 100 -tf 1 -s 3 -dt 0.001 -m $MESH"

$EXEC $CONFIG -e $SCHEME -o 1 -r 5
$EXEC $CONFIG -e $SCHEME -o 3 -r 4
$EXEC $CONFIG -e $SCHEME -o 7 -r 3

SCHEME=1

$EXEC $CONFIG -e $SCHEME -o 1 -r 5
$EXEC $CONFIG -e $SCHEME -o 3 -r 4
$EXEC $CONFIG -e $SCHEME -o 7 -r 3
