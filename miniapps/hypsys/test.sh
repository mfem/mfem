make clean && make -j;

## Advection

# Solid Body Roatation
./hypsys
./phypsys
mpirun -np 4 ./phypsys
# Steady Circular Convection
./hypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0
./phypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0
mpirun -np 4 ./phypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0

## Burgers
./hypsys -p 1 -o 0 -tf 0.5 -r 4 -m data/inline-4quad.mesh
./phypsys -p 1 -o 0 -tf 0.5 -r 4 -m data/inline-4quad.mesh
mpirun -np 4 ./phypsys -p 1 -o 0 -tf 0.5 -r 4 -m data/inline-4quad.mesh

## Shallow-Water

# Vorticity advection
./hypsys -p 3 -c 0 -tf 2.828427124746190 -m data/periodic-tri.mesh -r 3
./phypsys -p 3 -c 0 -tf 2.828427124746190 -m data/periodic-tri.mesh -r 3
mpirun -np 4 ./phypsys -p 3 -c 0 -tf 2.828427124746190 -m data/periodic-tri.mesh -r 3
