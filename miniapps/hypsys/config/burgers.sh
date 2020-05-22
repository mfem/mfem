EXEC="mpirun -np 7 phypsys"

SCHEME=0
MESH=data/inline-4quad.mesh
CONFIG="-p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH"

$EXEC $CONFIG -e $SCHEME -o 1 -r 4

SCHEME=1

$EXEC $CONFIG -e $SCHEME -o 1 -r 4
$EXEC $CONFIG -e $SCHEME -o 3 -r 3
$EXEC $CONFIG -e $SCHEME -o 7 -r 2
$EXEC $CONFIG -e $SCHEME -o 15 -r 1

# Grid convergence test
rm errors.txt

SCHEME=1
ORDER=1
MESH=data/periodic-segment.mesh
CONFIG="-p 1 -c 0 -vs 1000 -tf 0.1 -s 3 -dt 0.00025 -m $MESH -o $ORDER -e $SCHEME"

$EXEC $CONFIG -r 2
$EXEC $CONFIG -r 3
$EXEC $CONFIG -r 4
$EXEC $CONFIG -r 5
$EXEC $CONFIG -r 6
$EXEC $CONFIG -r 7