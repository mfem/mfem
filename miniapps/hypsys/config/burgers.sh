MESH=data/inline-4quad.mesh
SCHEME=0

mpirun -np 4 phypsys -p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH -e $SCHEME -o 1 -r 4

SCHEME=1

mpirun -np 4 phypsys -p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH -e $SCHEME -o 1 -r 4
mpirun -np 4 phypsys -p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH -e $SCHEME -o 3 -r 3
mpirun -np 4 phypsys -p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m $MESH -e $SCHEME -o 7 -r 2