rm *.o
make simpl oc mma -j
BACKTRACK="-bb -ab"
for back in $BACKTRACK
do
  for p in 22
  do
    for i in 6 7 8
    do
      mpirun -np 8 ./simpl -rs $i -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-04 -rtol 1e-04
    done
  done
done

mpirun -np 8 ./simpl -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-05 -rtol 1e-05 -bb -L2
mpirun -np 8 ./simpl -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-05 -rtol 1e-05 -ab -L2
mpirun -np 8 ./oc -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-05 -rtol 1e-05
mpirun -np 8 ./mma -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-05 -rtol 1e-05
#
# mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-06 -rtol 1e-04 -bb
# mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-06 -rtol 1e-04 -ab
# mpirun -np 8 ./simpl -rs 4 -rp 4 -p 24 -vs 1 -bb
# mpirun -np 8 ./simpl -rs 4 -rp 4 -p 24 -vs 1 -ab
#
