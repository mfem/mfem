SCHEME=0

## Advection

# Solid Body Roatation
# ./hypsys -r 0 -es $SCHEME
# ./phypsys -r 0 -es $SCHEME
# mpirun -np 4 ./phypsys -r 0 -es $SCHEME

# Steady Circular Convection
# ./hypsys -vf 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0 -es $SCHEME
# ./phypsys -vf 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0 -es $SCHEME
# mpirun -np 4 ./phypsys -vf 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0 -es $SCHEME

# Translation
CONFIG="-p 0 -c 2 -vf 100 -tf 0.4 -s 3 -dt 0.002 -m data/periodic-3tri.mesh -o 2 -r 3 -es $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 4 phypsys $CONFIG

## Burgers
CONFIG="-p 1 -c 1 -vf 100 -tf 0.5 -s 3 -dt 0.004 -m data/inline-3quad.mesh -o 1 -r 3 -es $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 7 phypsys $CONFIG

## KPP
CONFIG="-p 2 -c 1 -vf 50 -tf 0.25 -s 3 -dt 0.005 -m data/inline-4tri.mesh -o 0 -r 4 -es 0"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 1 phypsys $CONFIG

## Shallow-Water

# Dam break
CONFIG="-p 4 -c 2 -vf 20 -tf 0.1 -s 3 -dt 0.0005 -m data/wall-bdr-4tri.mesh -o 1 -r 3 -es $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 2 phypsys $CONFIG

## Euler

# Smooth vortex
CONFIG="-p 5 -c 0 -vf 100 -tf 1 -s 3 -dt 0.00125 -m data/periodic-3quad.mesh -r 2 -es $SCHEME"
./hypsys $CONFIG
./phypsys $CONFIG
mpirun -np 4 phypsys $CONFIG
