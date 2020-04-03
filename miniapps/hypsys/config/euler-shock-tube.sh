MESH=data/wall-bdr-4segment.mesh
SCHEME=1

mpirun -np 4 phypsys -p 4 -c 1 -m $MESH -tf 0.231 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 5 -o 1
mpirun -np 4 phypsys -p 4 -c 1 -m $MESH -tf 0.231 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 4 -o 3
mpirun -np 4 phypsys -p 4 -c 1 -m $MESH -tf 0.231 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 3 -o 7