#!/bin.sh


alpha=0.9
beta=1.5
gamma=0.9
lambda=1806600
# lambda=100.0
R0=2.4
rho_gamma=12
mu=1.0
mesh_file="meshes/iter_gen.msh"
data_file="separated_file.data"
refinement_factor=0
do_test=0
do_manufactured_solution=0
max_krylov_iter=1000
max_newton_iter=20
krylov_tol=1e-12
newton_tol=1e-12
c1=0
c2=-300
c3=100
c4=-300
c5=-300
c6=-300
c7=-300


./main.o \
    -m $mesh_file \
    -o 1 \
    -d $data_file \
    -g $refinement_factor \
    -t $do_test \
    --alpha $alpha \
    --beta $beta \
    --lambda $lambda \
    --gamma $gamma \
    --mu $mu \
    --r_zero $R0 \
    --rho_gamma $rho_gamma \
    --do_manufactured_solution $do_manufactured_solution \
    --max_krylov_iter $max_krylov_iter \
    --max_newton_iter $max_newton_iter \
    --krylov_tol $krylov_tol \
    --newton_tol $newton_tol \
    --c1 $c1 \
    --c2 $c2 \
    --c3 $c3 \
    --c4 $c4 \
    --c5 $c5 \
    --c6 $c6 \
    --c7 $c7


