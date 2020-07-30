EXEC="mpirun -np 7 phypsys"

SCHEME=0
MESH=data/inline-4quad.mesh
CONFIG="-p 1 -c 1 -vf 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH"

$EXEC $CONFIG -es $SCHEME -o 0 -r 5
$EXEC $CONFIG -es $SCHEME -o 1 -r 4

SCHEME=1

$EXEC $CONFIG -es $SCHEME -o  1 -r 4
$EXEC $CONFIG -es $SCHEME -o  3 -r 3
$EXEC $CONFIG -es $SCHEME -o  7 -r 2
$EXEC $CONFIG -es $SCHEME -o 15 -r 1
