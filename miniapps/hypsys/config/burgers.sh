MESH=data/inline-4quad.mesh
SCHEME=1

mpirun -np 4 phypsys -p 1 -c 1 -m $MESH -tf 0.5 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 4 -o 1
mpirun -np 4 phypsys -p 1 -c 1 -m $MESH -tf 0.5 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 3 -o 3
mpirun -np 4 phypsys -p 1 -c 1 -m $MESH -tf 0.5 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 2 -o 7