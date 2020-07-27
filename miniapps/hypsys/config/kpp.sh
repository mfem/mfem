MESH=data/inline-4quad.mesh

./hypsys -p 2 -c 1 -vs 100 -tf 0.25 -s 3 -dt 0.001 -r 5 -o 0 -m $MESH -e 0
./hypsys -p 2 -c 1 -vs 100 -tf 0.25 -s 3 -dt 0.001 -r 4 -o 1 -m $MESH -e 1
