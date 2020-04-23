MESH=data/wall-bdr-4tri.mesh
SCHEME=0
CONFIG="-p 3 -c 1 -vs 50 -tf 0.7 -s 3 -dt 0.002 -m $MESH"

mpirun -np 4 phypsys -r 4 -o 1 $CONFIG -e $SCHEME
# mpirun -np 4 phypsys -r 3 -o 3 $CONFIG -e $SCHEME
# mpirun -np 4 phypsys -r 2 -o 7 $CONFIG -e $SCHEME

SCHEME=1

mpirun -np 4 phypsys -r 4 -o 1 $CONFIG -e $SCHEME
# mpirun -np 4 phypsys -r 3 -o 3 $CONFIG -e $SCHEME
# mpirun -np 4 phypsys -r 2 -o 7 $CONFIG -e $SCHEME
