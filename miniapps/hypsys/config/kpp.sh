MESH=data/inline-4tri.mesh
SCHEME=1

mpirun -np 4 phypsys -p 2 -c 1 -m $MESH -tf 0.25 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 4 -o 2
mpirun -np 4 phypsys -p 2 -c 1 -m $MESH -tf 0.25 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 3 -o 5
mpirun -np 4 phypsys -p 2 -c 1 -m $MESH -tf 0.25 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 2 -o 11