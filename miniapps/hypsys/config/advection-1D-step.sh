MESH=data/periodic-segment.mesh
SCHEME=0

./hypsys -p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -o 1 -r 5 -m $MESH -e $SCHEME
./hypsys -p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -o 3 -r 4 -m $MESH -e $SCHEME
./hypsys -p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -o 7 -r 3 -m $MESH -e $SCHEME

SCHEME=1

./hypsys -p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -o 1 -r 5 -m $MESH -e $SCHEME
./hypsys -p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -o 3 -r 4 -m $MESH -e $SCHEME
./hypsys -p 0 -c 2 -vs 100 -tf 1 -s 3 -dt 0.001 -o 7 -r 3 -m $MESH -e $SCHEME