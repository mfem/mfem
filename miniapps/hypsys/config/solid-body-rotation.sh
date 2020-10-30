EXEC="mpirun -np 7 phypsys"

MESH=data/periodic-4quad.mesh
SCHEME=1

$EXEC -p 0 -c 1 -vf 200 -tf 1 -s 3 -dt 0.00032 -m $MESH -r 4 -o 2 -es $SCHEME
