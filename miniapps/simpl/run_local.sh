BACKTRACK="-ab -bb"
for back in $BACKTRACK
do
  for p in 21 22
  do
    for i in 6 7 8
    do
      mpirun -np 8 ./simpl -rs $i -rp 0 -p $p $back
    done
  done
done
