rm *.o
make simpl oc mma -j
BACKTRACK="-bb -ab"
# Run Cantilever2 and MBB with KKT condition
for back in $BACKTRACK
do
  for p in 21 22
  do
    for i in 6 7 8
    do
      mpirun -np 8 ./simpl -rs $i -rp 0 -p $p $back -no-vis -vs 10 -atol 1e-05 -rtol 1e-05
    done
  done
done

# Run MBB with L2 condition
mpirun -np 8 ./simpl -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05 -bb -L2
mpirun -np 8 ./simpl -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05 -ab -L2
mpirun -np 8 ./oc -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05
mpirun -np 8 ./mma -rs 8 -rp 0 -p 21 -vs 10 -atol 1e-05 -rtol 1e-05

# Bridge
mpirun -np 8 ./simpl -rs 4 -rp 4 -p 24 -vs 1 -bb
mpirun -np 8 ./simpl -rs 4 -rp 4 -p 24 -vs 1 -ab

# Compliant Mechanism with no initial design
sed -i '' '251s/.*/      \/\/ ForceInverterInitialDesign\(control_gf, \&entropy\);/' simpl.cpp
make simpl -j
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -bb
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -ab
mv ParaView/SiMPL-A-ForceInverter2-8 ParaView/SiMPL-A-ForceInverter2-8-const
mv SiMPL-A-ForceInverter2-8.csv SiMPL-A-ForceInverter2-8-const.csv
mv ParaView/SiMPL-B-ForceInverter2-8 ParaView/SiMPL-B-ForceInverter2-8-const
mv SiMPL-B-ForceInverter2-8.csv SiMPL-B-ForceInverter2-8-const.csv

# Compliant Mechanism with Initial Design connecting center
sed -i '' '251s/.*/      ForceInverterInitialDesign\(control_gf, \&entropy\);/' simpl.cpp
sed -i '' '65s/.*/   Vector domain_center\(\{1.0,0.5\}\);/' topopt_problems.cpp
make simpl -j
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -bb
mv ParaView/SiMPL-B-ForceInverter2-8 ParaView/SiMPL-B-ForceInverter2-8-center
mv SiMPL-B-ForceInverter2-8.csv SiMPL-B-ForceInverter2-8-center.csv

# Compliant Mechanism with Initial Design connecting bottom
sed -i '' '65s/.*/   Vector domain_center\(\{1.0,0.0\}\);/' topopt_problems.cpp
make simpl -j
mpirun -np 8 ./simpl -rs 4 -rp 4 -p -21 -vs 1 -atol 1e-08 -rtol 1e-08 -bb
mv ParaView/SiMPL-B-ForceInverter2-8 ParaView/SiMPL-B-ForceInverter2-8-bottom
mv SiMPL-B-ForceInverter2-8.csv SiMPL-B-ForceInverter2-8-bottom.csv
