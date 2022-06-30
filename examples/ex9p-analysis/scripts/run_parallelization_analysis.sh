#! /bin/bash

# This is a script to run convergence analysis on ex9p-continuous
mfem_dir="/Users/sheridan7/Workspace/mfem/examples"
create_parallelization="/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/scripts/create_parallelization_table.py"
temp_output="/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/temp_output/"
analysis_dir="/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/"
cd $analysis_dir
rm *.out
rm *.txt

cd $temp_output
rm *.out
rm *.txt

cd $mfem_dir
# mpirun -np 1 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 5 -rp 0 -p 4 -tf 2
# mpirun -np 2 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 5 -rp 0 -p 4 -tf 2
# mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 5 -rp 0 -p 4 -tf 2

mpirun -np 1 ex9p-continuous -m ../data/periodic-square.mesh -dt 0.005 -rs 5 -rp 0 -p 4 -tf 2.84
mpirun -np 2 ex9p-continuous -m ../data/periodic-square.mesh -dt 0.005 -rs 5 -rp 0 -p 4 -tf 2.84
mpirun -np 4 --oversubscribe ex9p-continuous -m ../data/periodic-square.mesh -dt 0.005 -rs 5 -rp 0 -p 4 -tf 2.84

# mpirun -np 1 ex9p-discontinuous -m ../data/periodic-square.mesh -ct -rs 5 -rp 0 -p 4 -ots
# mpirun -np 2 ex9p-discontinuous -m ../data/periodic-square.mesh -ct -rs 5 -rp 0 -p 4 -ots
# mpirun -np 4 ex9p-discontinuous -m ../data/periodic-square.mesh -ct -rs 5 -rp 0 -p 4 -ots

# mpirun -np 1 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 6 -rp 0 -p 4 -ots
# mpirun -np 2 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 6 -rp 0 -p 4 -ots
# mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 6 -rp 0 -p 4 -ots

# mpirun -np 1 ex9p-continuous -m ../data/periodic-cube.mesh -dt 0.01 -rs 2 -rp 0 -p 4 -tf 1
# mpirun -np 2 ex9p-continuous -m ../data/periodic-cube.mesh -dt 0.01 -rs 2 -rp 0 -p 4 -tf 1
# mpirun -np 4 ex9p-continuous -m ../data/periodic-cube.mesh -dt 0.01 -rs 2 -rp 0 -p 4 -tf 1

# mpirun -np 1 ex9p-discontinuous -m ../data/periodic-cube.mesh -dt 0.01 -rs 2 -rp 0 -p 4 -tf 1
# mpirun -np 2 ex9p-discontinuous -m ../data/periodic-cube.mesh -dt 0.01 -rs 2 -rp 0 -p 4 -tf 1
# mpirun -np 4 ex9p-discontinuous -m ../data/periodic-cube.mesh -dt 0.01 -rs 2 -rp 0 -p 4 -tf 1

cd $analysis_dir
mv *.out temp_output/.
cd scripts

# python stuff for iterating over files to build convergence table
python $create_parallelization $temp_output
