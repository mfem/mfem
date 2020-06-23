EXEC="mpirun -np 7 phypsys"

SCHEME=0
MESH=data/inline-4quad.mesh
CONFIG="-p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH"

$EXEC $CONFIG -e $SCHEME -o 0 -r 5
$EXEC $CONFIG -e $SCHEME -o 1 -r 4

SCHEME=1

$EXEC $CONFIG -e $SCHEME -o 1 -r 4
$EXEC $CONFIG -e $SCHEME -o 3 -r 3
$EXEC $CONFIG -e $SCHEME -o 7 -r 2
$EXEC $CONFIG -e $SCHEME -o 15 -r 1

# Grid convergence test
rm errors.txt

SCHEME=1
MESH0=data/periodic-3segment.mesh
MESH=data/periodic-segment.mesh
ORDER=1 # -dt 0.0004
# ORDER=2 # -dt 0.00025
# ORDER=3 # -dt 0.0001
# ORDER=4 # -dt 0.00005
CONFIG0="-p 1 -c 0 -vs 1000 -tf 0.1 -s 3 -dt 0.0004 -m $MESH0 -o $ORDER -e $SCHEME"
CONFIG1="-p 1 -c 0 -vs 1000 -tf 0.1 -s 3 -dt 0.0004 -m $MESH -o $ORDER -e $SCHEME"

# $EXEC $CONFIG0 -r 4
# $EXEC $CONFIG1 -r 4
# $EXEC $CONFIG0 -r 5
# $EXEC $CONFIG1 -r 5
# $EXEC $CONFIG0 -r 6
# $EXEC $CONFIG1 -r 6
# $EXEC $CONFIG0 -r 7

# TF=0.1 # smooth solution
TF=0.2 # solution with shock
CONFIG="-p 1 -c 0 -vs 1000 -tf $TF -s 3 -dt 0.00025 -m $MESH -e $SCHEME"

# $EXEC $CONFIG -o 0 -r 5
# scripts/gridfunc-scatter $MESH ultimate.gf
# mv scatter.dat scripts/scatter1.dat
# $EXEC $CONFIG -o 1 -r 4
# scripts/gridfunc-scatter $MESH ultimate.gf
# mv scatter.dat scripts/scatter2.dat
# $EXEC $CONFIG -o 3 -r 3
# scripts/gridfunc-scatter $MESH ultimate.gf
# mv scatter.dat scripts/scatter3.dat
# $EXEC $CONFIG -o 7 -r 2
# scripts/gridfunc-scatter $MESH ultimate.gf
# mv scatter.dat scripts/scatter4.dat
# $EXEC $CONFIG -o 15 -r 1
# scripts/gridfunc-scatter $MESH ultimate.gf
# mv scatter.dat scripts/scatter5.dat
# $EXEC $CONFIG -o 31 -r 0
# scripts/gridfunc-scatter $MESH ultimate.gf
# mv scatter.dat scripts/scatter6.dat
