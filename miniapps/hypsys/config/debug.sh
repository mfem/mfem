MESH=data/wall-bdr-100segment.mesh
CONFIG="-p 3 -c 1 -vs 1000 -tf 0.02 -s 3 -m $MESH -e 1"

./hypsys $CONFIG -r 0 -o 1 -dt 0.0001
# mv ultimate.gf cmp.gf
mpirun -np 3 phypsys $CONFIG -r 0 -o 1 -dt 0.0001
# meld ultimate.gf cmp.gf