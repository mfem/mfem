SCHEME=1

# 1D Advection
./hypsys -p 0 -c 2 -vf 100 -tf 1 -s 3 -dt 0.001 -o 7 -r 3 -m data/periodic-segment.mesh -es $SCHEME

# 1D Burgers
./hypsys -p 1 -c 0 -vf 100 -tf 0.5 -s 3 -dt 0.001 -m data/inline-3segment.mesh -o 31 -r 0 -es $SCHEME
# 2D Burgers
mpirun -np 4 phypsys -p 1 -c 1 -vf 100 -tf 0.5 -s 3 -dt 0.001 -m data/inline-3tri.mesh -o 3 -r 3 -es $SCHEME

# 1D SWE Dam Break
./hypsys -p 3 -c 1 -vf 100 -tf 0.02 -s 3 -dt 0.00002 -m data/wall-bdr-4segment.mesh -r 8 -o 1 -es $SCHEME
