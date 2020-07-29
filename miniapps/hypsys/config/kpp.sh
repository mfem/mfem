EXEC="mpirun -np 7 phypsys"
MESH=data/inline-4quad.mesh

# KPP
$EXEC -p 2 -c 1 -vf 100 -tf 0.25 -s 3 -dt 0.001 -r 5 -o 0 -m $MESH -e 0
$EXEC -p 2 -c 1 -vf 100 -tf 0.25 -s 3 -dt 0.001 -r 4 -o 1 -m $MESH -e 1

# Buckley-Leverett
$EXEC -p 3 -c 2 -vf 100 -tf 0.1666667 -s 3 -dt 0.0005 -r 5 -o 0 -m $MESH -e 0
$EXEC -p 3 -c 2 -vf 100 -tf 0.1666667 -s 3 -dt 0.0005 -r 4 -o 1 -m $MESH -e 1