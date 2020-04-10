MESH=data/periodic-segment.mesh
# SCHEME=0

# ./hypsys -p 0 -c 2 -tf 1 -dt 0.001 -s 3 -vs 100 -r 5 -o 1 -m $MESH -e $SCHEME
# ./hypsys -p 0 -c 2 -tf 1 -dt 0.001 -s 3 -vs 100 -r 4 -o 3 -m $MESH -e $SCHEME
# ./hypsys -p 0 -c 2 -tf 1 -dt 0.001 -s 3 -vs 100 -r 3 -o 7 -m $MESH -e $SCHEME

SCHEME=1

./hypsys -p 0 -c 2 -tf 1 -dt 0.001 -s 3 -vs 100 -r 5 -o 1 -m $MESH -e $SCHEME
./hypsys -p 0 -c 2 -tf 1 -dt 0.001 -s 3 -vs 100 -r 4 -o 3 -m $MESH -e $SCHEME
./hypsys -p 0 -c 2 -tf 1 -dt 0.001 -s 3 -vs 100 -r 3 -o 7 -m $MESH -e $SCHEME
# ./hypsys -p 0 -c 2 -tf 1 -dt 0.001 -s 3 -vs 100 -r 2 -o 15 -m $MESH -e $SCHEME