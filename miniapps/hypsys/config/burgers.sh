EXEC="mpirun -np 7 phypsys"

SCHEME=0
MESH=data/inline-4quad.mesh
CONFIG="-p 1 -c 1 -vf 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH"

# $EXEC $CONFIG -es $SCHEME -o 0 -r 5
# $EXEC $CONFIG -es $SCHEME -o 1 -r 4

SCHEME=1

$EXEC $CONFIG -es $SCHEME -o  1 -r 4
$EXEC $CONFIG -es $SCHEME -o  3 -r 3
$EXEC $CONFIG -es $SCHEME -o  7 -r 2
$EXEC $CONFIG -es $SCHEME -o 15 -r 1


# GRID CONVERGENCE TEST
SCHEME=0
ORDER=1
DT=0.0004
ORDER=2
DT=0.00016
ORDER=3
DT=-dt 0.0001
ORDER=4
DT=0.00005
MESH="data/periodic-4segment.mesh"
MESH0="data/periodic-3segment.mesh"
CONFIG0="-p 1 -c 0 -vf 1000 -tf 0.1 -s 3 -dt $DT -m $MESH0 -o $ORDER"
CONFIG1="-p 1 -c 0 -vf 1000 -tf 0.1 -s 3 -dt $DT -m $MESH -o $ORDER"

rm errors.txt
$EXEC $CONFIG0 -r 4 -es $SCHEME
$EXEC $CONFIG1 -r 4 -es $SCHEME
$EXEC $CONFIG0 -r 5 -es $SCHEME
$EXEC $CONFIG1 -r 5 -es $SCHEME
$EXEC $CONFIG0 -r 6 -es $SCHEME
$EXEC $CONFIG1 -r 6 -es $SCHEME
$EXEC $CONFIG0 -r 7 -es $SCHEME
