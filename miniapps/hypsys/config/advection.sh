EXEC="mpirun -np 7 phypsys"

SCHEME=1
MESH="data/periodic-4segment.mesh"
MESH0="data/periodic-3segment.mesh"
CONFIG="-p 0 -c 4 -vf 100 -tf 1 -s 3 -dt 0.001 -m $MESH"

$EXEC $CONFIG -es $SCHEME -o 1 -r 5
$EXEC $CONFIG -es $SCHEME -o 3 -r 4
$EXEC $CONFIG -es $SCHEME -o 7 -r 3


# GRID CONVERGENCE TEST
ORDER=1
DT=0.0004
# ORDER=2
# DT=0.00025
# ORDER=3
# DT=0.0001
# ORDER=4
# DT=0.000025
CONFIG0="-p 0 -c 3 -vf 1000 -tf 1 -s 3 -dt $DT -m $MESH0 -o $ORDER"
CONFIG1="-p 0 -c 3 -vf 1000 -tf 1 -s 3 -dt $DT -m $MESH -o $ORDER"

# rm errors.txt
# $EXEC $CONFIG0 -r 4 -es $SCHEME
# $EXEC $CONFIG1 -r 4 -es $SCHEME
# $EXEC $CONFIG0 -r 5 -es $SCHEME
# $EXEC $CONFIG1 -r 5 -es $SCHEME
# $EXEC $CONFIG0 -r 6 -es $SCHEME
# $EXEC $CONFIG1 -r 6 -es $SCHEME
# $EXEC $CONFIG0 -r 7 -es $SCHEME
