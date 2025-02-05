trap "exit" INT
rm oc_org mma
make -j8 oc_org mma
for p in 22
do
  for i in 7
  do
    for ch in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4
    do
      mpirun -np 8 ./oc_org -rs $i -rp 0 -p $p  -no-vis -atol 1e-05 -rtol 1e-05 -ch $ch --max-it 200
      mpirun -np 8 ./mma -rs $i -rp 0 -p $p  -no-vis -atol 1e-05 -rtol 1e-05 -ch $ch --max-it 200
    done
  done
done
