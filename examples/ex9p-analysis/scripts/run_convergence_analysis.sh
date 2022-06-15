#! /bin/bash

# This is a script to run convergence analysis on ex9p-continuous
mfem_dir="/Users/sheridan7/Workspace/mfem/examples"
create_convergence="/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/scripts/create_convergence_table.py"
temp_output="/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/temp_output/"
analysis_dir="/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/"

cd $analysis_dir
rm *.out
rm *.txt

cd $temp_output
rm *.out
rm *.txt

cd $mfem_dir
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 1 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 2 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 3 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 4 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 5 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 6 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 7 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 8 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 9 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 10 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 11 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 12 -rp 0 -p 0 -tf 2
mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -ct -rs 13 -rp 0 -p 0 -tf 2
# mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 1 -rp 0 -p 3 -ots
# mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 2 -rp 0 -p 3 -ots
# mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 3 -rp 0 -p 3 -ots
# mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 4 -rp 0 -p 3 -ots
# mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -ct -rs 5 -rp 0 -p 3 -ots

cd ex9p-analysis
mv *.out temp_output/.
cd scripts

# python stuff for iterating over files to build convergence table
python $create_convergence $temp_output
