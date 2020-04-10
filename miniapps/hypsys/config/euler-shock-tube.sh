MESH=data/wall-bdr-4segment.mesh
SCHEME=1

mpirun -np 4 phypsys -p 4 -c 1 -tf 0.231 -dt 0.001 -s 3 -vs 100 -r 5 -o 1 -m $MESH -e $SCHEME
mpirun -np 4 phypsys -p 4 -c 1 -tf 0.231 -dt 0.001 -s 3 -vs 100 -r 4 -o 3 -m $MESH -e $SCHEME
mpirun -np 4 phypsys -p 4 -c 1 -tf 0.231 -dt 0.001 -s 3 -vs 100 -r 3 -o 7 -m $MESH -e $SCHEME