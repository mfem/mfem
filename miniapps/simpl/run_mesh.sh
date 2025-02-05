trap "exit" INT
rm *.o
make simpl oc mma -j
BACKTRACK="-bb -ab"
# Run MBB
for back in $BACKTRACK
do
  for p in 22
  do
    mpirun -np 8 ./simpl -rs 6 -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -oc 0 -os 1 -of 1
    for i in 1 2 3 4 
    do
      mpirun -np 8 ./simpl -rs 6 -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 1 -oc $i -os $i -of $i 
    done
  done
done
