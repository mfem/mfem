SCHEME=0
MESH=data/periodic-segment.mesh
CONFIG="-p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -m $MESH"

./hypsys $CONFIG -e $SCHEME -o 1 -r 5
./hypsys $CONFIG -e $SCHEME -o 3 -r 4
./hypsys $CONFIG -e $SCHEME -o 7 -r 3

SCHEME=1

./hypsys $CONFIG -e $SCHEME -o 1 -r 5
./hypsys $CONFIG -e $SCHEME -o 3 -r 4
./hypsys $CONFIG -e $SCHEME -o 7 -r 3

# Grid convergence test
rm errors.txt

SCHEME=1
ORDER=1
CONFIG="-p 0 -c 3 -vs 1000 -tf 1 -s 3 -dt 0.0004 -m $MESH -o $ORDER -e $SCHEME"

mpirun -np 7 phypsys $CONFIG -r 2
mpirun -np 7 phypsys $CONFIG -r 3
mpirun -np 7 phypsys $CONFIG -r 4
mpirun -np 7 phypsys $CONFIG -r 5
mpirun -np 7 phypsys $CONFIG -r 6
mpirun -np 7 phypsys $CONFIG -r 7