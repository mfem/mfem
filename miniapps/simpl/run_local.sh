trap "exit" INT
rm *.o
make simpl oc mma -j
BACKTRACK="-bb -ab"
# Run Cantilever2 and MBB with KKT condition
for back in $BACKTRACK
do
  for p in 21
  do
    for i in 1 2 3 4 
    do
      mpirun -np 8 ./simpl -rs 6 -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 2 -oc $i -os $i -of $i 
    done
  done
done
for back in $BACKTRACK
do
  for p in 21
  do
    for i in 6 7 8 9
    do
      mpirun -np 8 ./simpl -rs $i -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 2
    done
  done
done
for back in $BACKTRACK
do
  for p in 22
  do
    for i in 6 7 8
    do
      mpirun -np 8 ./simpl -rs $i -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05 -kkt-type 1
    done
  done
done

# Run MBB with L2 condition
mpirun -np 8 ./simpl -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05 -bb -L2 -kkt-type 2
mpirun -np 8 ./simpl -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05 -ab -L2 -kkt-type 2
mpirun -np 8 ./oc -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05
mpirun -np 8 ./mma -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05
mpirun -np 8 ./simpl -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-04 -rtol 1e-04 -bb -L2 -kkt-type 1
mpirun -np 8 ./simpl -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-04 -rtol 1e-04 -ab -L2 -kkt-type 1 
mpirun -np 8 ./oc -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-04 -rtol 1e-04
mpirun -np 8 ./mma -rs 8 -rp 0 -p 22 -vs 10 -atol 1e-04 -rtol 1e-04

# Bridge
mpirun -np 8 ./simpl -rs 4 -rp 4 -p 24 -vs 1 -bb -kkt-type 1
mpirun -np 8 ./simpl -rs 4 -rp 4 -p 24 -vs 1 -ab -kkt-type 1

# Compliant Mechanism with no initial design
sed -i '' '251s/.*/      \/\/ ForceInverterInitialDesign\(control_gf, \&entropy\);/' simpl.cpp
make simpl -j
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -bb -kkt-type 1 -mini 20
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -ab -kkt-type 1 -mini 20
mv ParaView/SiMPL-A-ForceInverter2-8-0 ParaView/SiMPL-A-ForceInverter2-8-0-const
mv SiMPL-A-ForceInverter2-8-0.csv SiMPL-A-ForceInverter2-8-0-const.csv
mv ParaView/SiMPL-B-ForceInverter2-8-0 ParaView/SiMPL-B-ForceInverter2-8-0-const
mv SiMPL-B-ForceInverter2-8-0.csv SiMPL-B-ForceInverter2-8-0-const.csv

# Compliant Mechanism with Initial Design connecting center
sed -i '' '251s/.*/      ForceInverterInitialDesign\(control_gf, \&entropy\);/' simpl.cpp
sed -i '' '65s/.*/   Vector domain_center\(\{1.0,0.5\}\);/' topopt_problems.cpp
make simpl -j
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -bb -kkt-type 1 -mini 20
mv ParaView/SiMPL-B-ForceInverter2-8-0 ParaView/SiMPL-B-ForceInverter2-8-0-center
mv SiMPL-B-ForceInverter2-8-0.csv SiMPL-B-ForceInverter2-8-0-center.csv

# Compliant Mechanism with Initial Design connecting bottom
sed -i '' '251s/.*/      ForceInverterInitialDesign\(control_gf, \&entropy\);/' simpl.cpp
sed -i '' '65s/.*/   Vector domain_center\(\{1.0,0.0\}\);/' topopt_problems.cpp
make simpl -j
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -bb -kkt-type 1 -mini 20
mv ParaView/SiMPL-B-ForceInverter2-8-0 ParaView/SiMPL-B-ForceInverter2-8-0-bottom
mv SiMPL-B-ForceInverter2-8-0.csv SiMPL-B-ForceInverter2-8-0-bottom.csv
