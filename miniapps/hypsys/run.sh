make clean && make library -j && make exec -j
valgrind hypsys -tf 0.1
# hypsys -c 0 -r 0 -o 1 -tf 0.002 -vs 100
# hypsys -c 1 -r 4 -o 1 -dt 0.001
# hypsys -c 1 -m data/inline-quad.mesh -r 4 -o 1 -dt 0.00016
# hypsys -c 1 -m data/inline-quad.mesh -r 5 -o 0 -dt 0.00016