SCHEME=1

# 1D Burgers
./hypsys -p 1 -c 0 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m data/periodic-segment.mesh -o 1 -r 4 -e $SCHEME
# 2D Burgers
mpirun -np 4 phypsys -p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.001 -m data/inline-4quad.mesh -o 1 -r 4 -e $SCHEME

# 1D SWE Dam Break
./hypsys -p 3 -c 1 -vs 100 -tf 0.02 -s 3 -dt 0.0005 -m data/wall-bdr-4segment.mesh -r 5 -o 0 -e $SCHEME