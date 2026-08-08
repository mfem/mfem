nx=32
ny=32
 
for t in 1 2 4 8 16 24 32 64
do

# echo $t

nz=$(($t * 4))
npt=$(($nx * $ny * $nz * 100))

echo "#############################"
echo $t
echo "#############################"

srun -n $t ./electrostatic-pic \
    -no-vis \
    -rdi 1 \
    -dim 3 \
    -npt $npt \
    -k 0.5 -a 0.01 \
    -nt 10 \
    -nx $nx -ny $ny -nz $nz \
    -O 1 \
    -q 0.00004844730731 \
    -m 0.00004844730731 \
    -oci -1 \
    -dt 0.02 \
    -fa -d cpu

done