#!/bin/bash
#
# 2D weak-scaling sweep for electrostatic-pic.
#
# The domain is always a square of side L = 2*pi/k, so the mesh is refined
# ISOTROPICALLY and the rank count grows by 4x per step (2x per direction):
#
#   ranks:            1        4       16       64
#   mesh:          32^2     64^2    128^2    256^2
#   elems/rank:    1024     1024     1024     1024
#   parts/rank:  409600   409600   409600   409600   (400 particles/cell,
#                                                      as in the reference
#                                                      2D Landau run)
#
# IMPORTANT: q and m are NOT free constants in this test case. The Landau
# setup assumes unit charge density (unit plasma frequency), i.e.
#     q = m = L^2 / npt
# (check: the reference run has q*npt = 0.001181640625 * 409600 = L^2).
# Since npt changes with scale, q and m are recomputed for every run below.

k=0.2855993321
alpha=0.05
dt=0.1

ranks=(1 4 16 64)
cells=(32 64 128 256)   # n^2 mesh

ppc=400                 # particles per cell (matches the reference run)
nt=10                   # long enough that one-time setup doesn't dominate

# Set to your node's core count so all runs pack nodes identically.
PPN=""

for i in "${!ranks[@]}"
do
    t=${ranks[$i]}
    n=${cells[$i]}
    npt=$(($n * $n * $ppc))

    # q = m = L^2 / npt, with L = 2*pi/k  (unit charge density)
    q=$(awk -v k=$k -v npt=$npt \
        'BEGIN { L = 2*atan2(0,-1)/k; printf "%.17g", L*L/npt }')

    echo "#############################"
    echo "ranks=$t  mesh=${n}^2  npt=$npt  q=m=$q"
    echo "#############################"

    srun -n $t ${PPN:+--ntasks-per-node=$PPN} --cpu-bind=cores \
        ./electrostatic-pic \
        -no-vis \
        -rdi 1 \
        -dim 2 \
        -npt $npt \
        -k $k -a $alpha \
        -nt $nt \
        -nx $n -ny $n \
        -O 1 \
        -q $q \
        -m $q \
        -oci -1 \
        -dt $dt \
        -fa -d cpu


done