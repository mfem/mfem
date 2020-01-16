make clean && make library -j && make exec -j
hypsys
hypsys -vs 1000 -m data/inline-4quad.mesh -dt 0.0001 -o 2 -s 1 -r 3 -c 0
