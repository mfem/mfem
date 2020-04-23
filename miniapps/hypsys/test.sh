# make clean && make -j;

SCHEME=1

## Advection

# Solid Body Roatation
# ./hypsys -r 0 -e $SCHEME
# ./phypsys -r 0 -e $SCHEME
# mpirun -np 4 ./phypsys -r 0 -e $SCHEME

# Steady Circular Convection
# ./hypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0 -e $SCHEME
# ./phypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0 -e $SCHEME
# mpirun -np 4 ./phypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0 -e $SCHEME

# Translation
CONFIG="-p 0 -c 2 -vs 100 -tf 0.4 -s 3 -dt 0.004 -m data/periodic-tri.mesh -o 2 -r 3 -e $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 4 phypsys $CONFIG

## Burgers
CONFIG="-p 1 -c 1 -vs 100 -tf 0.5 -s 3 -dt 0.005 -m data/inline-4quad.mesh -o 1 -r 3 -e $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 3 phypsys $CONFIG

## KPP
CONFIG="-p 2 -c 1 -vs 50 -tf 0.25 -s 3 -dt 0.005 -m data/inline-4tri.mesh -o 0 -r 4 -e $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 1 phypsys $CONFIG

## Shallow-Water

# Dam break
CONFIG="-p 3 -c 1 -vs 14 -tf 0.7 -s 3 -dt 0.005 -m data/wall-bdr-4tri.mesh -o 1 -r 3 -e $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 2 phypsys $CONFIG

## Euler

# Smooth vortex
CONFIG="-p 4 -c 0 -vs 100 -tf 2 -s 3 -dt 0.004 -m data/periodic-square.mesh -r 2 -e $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 4 phypsys $CONFIG
