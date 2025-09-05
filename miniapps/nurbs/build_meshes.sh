# quick and dirty script to build meshes for testing LOR preconditioning
rm -rf ho_meshes
mkdir ho_meshes

# fixed: 3d, 1 patch
for a in 1 2; do
    for nel in 8 11 16 23 32 45 64 92 128; do
        for o in 2 3 4; do
            #for m in 1 $o; do
            for m in 1; do
                ./nurbs_lor_cartesian -d 3 -n 1 -a $a -nel $nel -o $o -m $m
                mv ho_mesh.mesh ho_meshes/d3_n1_a${a}_nel${nel}_o${o}_m${m}.mesh
                rm lo_mesh.mesh
            done
        done
    done
done
