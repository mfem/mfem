#!/bin/sh

#
# Uses MFEM's mesh-explorer mini app to
#     1. read beam-2quad.mesh (-m beam-2quad.mesh)
#     2. refine it uniformly by a factor of 16 (r u 16),
#     3. set curvature to $c (c $c)
#     4. set jitter to $j (j $j)
#     5. Save it with ZFP compression to HDF5 (V h 1 $c $c ...)
#            Note that two saves are performed here, one to HDF5
#            and another to MFME-native ascii format by reading
#            back the compressed HDF5 data. So, this second file
#            provides a way to read data that has been altered
#            by compression back into MFEM and assess the impact.
#     6. Save it a second time to VisIt ascii (second V a)
#            This second save allows us to compare the original
#            data with that altered by compression and saved in
#            step 5.
#     7. Quit the app
#
#     All the inputs to mesh-explorer are encoded in this shell
#     script as a "here" doc between EOF monikers.
#
# To run this loop of tests...
#
# env HDF5_PLUGIN_PATH=<path-to-h5z-zfp-plugin> sh mfem_zfp_loop.sh

# Jitter loop
for j in 0 0.025 0.1;
do

    # Curvature loop
    for c in 5 6 9 11 17;
    do
        c1=$(expr $c - 1)

        # ZFP param loop
        zfpmode=3 # 1=rate, 2=prec, 3=acc
        for z in 0.00001 0.0001 0.001 0.01 0.1;
        do

{
cat <<EOF
r
u
16
c
$c
j
$j
V
h
1
$c1 $c1 1
$zfpmode
$z
1
$c1
$c1
1
0
V
a
q
EOF
} | ./mesh-explorer -m beam-2x2quad.mesh

            # Rename result files
            mv zfpmesh zfpmesh_${j}_${c}_${z}
            mv mesh-explorer_000000/mfem h5zzfpmesh_${j}_${c}_${z}
            mv mesh-explorer_000000/mesh.000000 origmesh_${j}_${c}_${z}

        done # zfp param loop

    done # curvature loop

done # jitter loop
