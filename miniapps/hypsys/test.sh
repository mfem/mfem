make clean && make -j;

## Advection

# Solid Body Roatation
./hypsys -r 0
./phypsys -r 0
mpirun -np 4 ./phypsys -r 0

# Steady Circular Convection
./hypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0
./phypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0
mpirun -np 4 ./phypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0

## Burgers
./hypsys -p 1 -o 0 -tf 0.5 -r 4 -m data/inline-4quad.mesh
./phypsys -p 1 -o 0 -tf 0.5 -r 4 -m data/inline-4quad.mesh
mpirun -np 4 ./phypsys -p 1 -o 0 -tf 0.5 -r 4 -m data/inline-4quad.mesh

## KPP
./hypsys -m data/inline-4quad.mesh -r 5 -o 0 -p 2 -tf 0.25 -dt 0.004
./phypsys -m data/inline-4quad.mesh -r 5 -o 0 -p 2 -tf 0.25 -dt 0.004
mpirun -np 4 ./phypsys -m data/inline-4quad.mesh -r 5 -o 0 -p 2 -tf 0.25 -dt 0.004

## Shallow-Water

# Dam break
./hypsys -p 3 -m data/wall-bdr-4tri.mesh -r 3 -o 1 -tf 0.25
./phypsys -p 3 -m data/wall-bdr-4tri.mesh -r 3 -o 1 -tf 0.25
mpirun -np 4 ./phypsys -p 3 -m data/wall-bdr-4tri.mesh -r 3 -o 1 -tf 0.25

## Euler

# Smooth vortex
./hypsys -p 4 -c 0 -tf 2 -m data/periodic-square.mesh -r 2
./phypsys -p 4 -c 0 -tf 2 -m data/periodic-square.mesh -r 2
mpirun -np 4 ./phypsys -p 4 -c 0 -tf 2 -m data/periodic-square.mesh -r 2
