#!/bin.sh

# more coils
# iter psi relations
#

# todo!
# take a look at the tolerances, should probably be squared!
# minres - useful for symmetric systems
# block diag PC
# more newton steps

alpha=0.0
beta=1.0
gamma=1.0
lambda=-10000
# lambda=100.0
R0=2.4
rho_gamma=16
# mu=1e-6
mu=12.5663706144e-7
mesh_file="meshes/iter_gen.msh"
data_file="separated_file.data"
refinement_factor=2
do_test=0
do_manufactured_solution=0
max_krylov_iter=10000
max_newton_iter=30
krylov_tol=1e-9 # check this...
newton_tol=1e-12
# center solenoids
c6=4.552585e+06
c7=-3.180596e+06
c8=-5.678096e+06
c9=-3.825538e+06
c10=-1.066498e+07
c11=2.094771e+07
# poloidal flux coils
c1=1.143284e+03
c2=-2.478694e+04
c3=-3.022037e+04
c4=-2.205664e+04
c5=-2.848113e+03

ur_coeff=1.0

do_control=1
weight=1e-18

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
    --c7 $c7 \
    --c8 $c8 \
    --c9 $c9 \
    --c10 $c10 \
    --c11 $c11 \
    --ur_coeff $ur_coeff \
    --do_control $do_control \
    --weight $weight



