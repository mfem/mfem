MESH=data/wall-bdr-100segment.mesh
CONFIG="-p 3 -c 1 -vs 1000 -tf 0.02 -s 1 -m $MESH -r 0 -e 1"

./hypsys $CONFIG -o 3 -dt 0.00005
mv ultimate.gf cmp.gf
mpirun -np 3 phypsys $CONFIG -o 3 -dt 0.00005
meld ultimate.gf cmp.gf