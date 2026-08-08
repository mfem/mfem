#!/bin/bash
#
# 3D weak-scaling sweep for electrostatic-pic.
#
# The domain is always a cube of side L = 2*pi/k, so the mesh is refined
# ISOTROPICALLY and the rank count grows by 8x per step (2x per direction):
#
#   ranks:            1        8       64
#   mesh:          16^3     32^3     64^3
#   elems/rank:    4096     4096     4096
#   parts/rank:  409600   409600   409600      (100 particles per cell)
#
# IMPORTANT: q and m are NOT free constants in this test case. The Landau
# setup assumes unit charge density (unit plasma frequency), i.e.
#     q = m = L^3 / npt
# (check: the reference 3D run has q*npt = 0.00004844730731 * 40960000 = L^3).
# Since npt changes with scale, q and m are recomputed for every run below.

k=0.5
alpha=0.01
dt=0.02

ranks=(1 8 64)
cells=(16 32 64)     # n^3 mesh; shift to (32 64 128) for finer resolution --
                     # per-rank load then grows 8x but stays constant across
                     # the sweep, which is all weak scaling requires.

ppc=100              # particles per cell
nt=10                # long enough that one-time setup doesn't dominate

# Set to your node's core count so all runs pack nodes identically.
PPN=""

for i in "${!ranks[@]}"
do
    t=${ranks[$i]}
    n=${cells[$i]}
    npt=$(($n * $n * $n * $ppc))

    # q = m = L^3 / npt, with L = 2*pi/k  (unit charge density)
    q=$(awk -v k=$k -v npt=$npt \
        'BEGIN { L = 2*atan2(0,-1)/k; printf "%.17g", L*L*L/npt }')

    echo "#############################"
    echo "ranks=$t  mesh=${n}^3  npt=$npt  q=m=$q"
    echo "#############################"

    # NOTE: -n sets the task count; -p selects a PARTITION.
    srun -n $t ${PPN:+--ntasks-per-node=$PPN} --cpu-bind=cores \
        ./electrostatic-pic \
        -no-vis \
        -rdi 1 \
        -dim 3 \
        -npt $npt \
        -k $k -a $alpha \
        -nt $nt \
        -nx $n -ny $n -nz $n \
        -O 1 \
        -q $q \
        -m $q \
        -oci -1 \
        -dt $dt \
        -fa -d cuda


done