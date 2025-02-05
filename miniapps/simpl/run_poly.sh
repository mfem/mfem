trap "exit" INT
rm simpl
make simpl -j
BACKTRACK="-bb"
# Run MBB
for back in $BACKTRACK
do
  for p in 221
  do
    mpirun -np 8 ./simpl -rs 9 -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 1 -oc 0 -os 1 -of 1
    mpirun -np 8 ./simpl -rs 8 -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 1 -oc 1 -os 1 -of 1
    # mpirun -np 8 ./simpl -rs 7 -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 1 -oc 2 -os 2 -of 2
    # for i in 1 2 
    # do
    #   mpirun -np 8 ./simpl -rs 7 -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 1 -oc $i -os $i -of $i 
    # done
  done
done
