MESH=data/periodic-segment.mesh
SCHEME=1

./hypsys -p 0 -c 2 -m $MESH -tf 1 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 5 -o 1
./hypsys -p 0 -c 2 -m $MESH -tf 1 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 4 -o 3
./hypsys -p 0 -c 2 -m $MESH -tf 1 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 3 -o 7
./hypsys -p 0 -c 2 -m $MESH -tf 1 -dt 0.001 -s 3 -vs 100 -e $SCHEME -r 2 -o 15