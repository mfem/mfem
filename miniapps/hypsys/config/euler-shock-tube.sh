MESH=data/wall-bdr-4segment.mesh
SCHEME=1

mpirun -np 2 phypsys -p 4 -c 1 -tf 0.231 -dt 0.00032 -s 3 -vs 100 -r 5 -o 2 -m $MESH -e $SCHEME
mpirun -np 2 phypsys -p 4 -c 1 -tf 0.231 -dt 0.00032 -s 3 -vs 100 -r 4 -o 5 -m $MESH -e $SCHEME
mpirun -np 2 phypsys -p 4 -c 1 -tf 0.231 -dt 0.00032 -s 3 -vs 100 -r 3 -o 11 -m $MESH -e $SCHEME