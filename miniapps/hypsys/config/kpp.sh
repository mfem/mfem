MESH=data/inline-4quad.mesh
SCHEME=1

mpirun -np 4 phypsys -p 2 -c 1 -vs 100 -tf 0.25 -s 3 -dt 0.0005 -r 4 -o 2 -m $MESH -e $SCHEME
mpirun -np 4 phypsys -p 2 -c 1 -vs 100 -tf 0.25 -s 3 -dt 0.0005 -r 3 -o 5 -m $MESH -e $SCHEME
mpirun -np 4 phypsys -p 2 -c 1 -vs 100 -tf 0.25 -s 3 -dt 0.0005 -r 2 -o 11 -m $MESH -e $SCHEME