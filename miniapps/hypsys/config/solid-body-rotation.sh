MESH=data/periodic-tri.mesh
SCHEME=0

./hypsys -p 0 -c 1 -m $MESH -tf 1 -dt 0.0005 -s 3 -vs 200 -e $SCHEME -r 4 -o 1
