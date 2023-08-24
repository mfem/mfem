# more coils
# iter psi relations
#

# coefficient of ff' term
alpha=1.0
# coefficient of p' term
beta=1.0
# unused?
gamma=0.0

# model
# 1: ff' defined from fpol data
# 2: Taylor equilibrium
# 3: ff' defined from ff' data
model=1

# plasma current
Ip=1.5e+7

R0=2.4
rho_gamma=16
mu=12.5663706144e-7
# mu=1.0
mesh_file="meshes/iter_gen.msh"
data_file="separated_file.data"
refinement_factor=3

do_test=0
do_manufactured_solution=0
do_initial=1

# linear solver parameters
max_krylov_iter=10000
max_newton_iter=5
krylov_tol=1e-9 # check this...
newton_tol=1e-12

# poloidal flux coils
# c6=4.552585e+06
# c7=-3.180596e+06
# c8=-5.678096e+06
# c9=-3.825538e+06
# c10=-1.066498e+07
# c11=2.094771e+07

c6=-4.552585e+06
c7=3.180596e+06
c8=5.678096e+06
c9=3.825538e+06
c10=1.066498e+07
c11=-2.094771e+07

# center solenoids
# c1=1.143284e+03
# c2=-2.478694e+04
# c3=-3.022037e+04
# c4=-2.205664e+04
# c5=-2.848113e+03

# # poloidal flux coils
# c6=0.0
# c7=0.0
# c8=0.0
# c9=0.0
# c10=0.0
# c11=0.0

# # center solenoids
c1=-1.143284e+06
c2=2.478694e+07
c3=3.022037e+07
c4=2.205664e+07
c5=2.848113e+06

ur_coeff=1.0

# number of control points on plasma
N_control=40

do_control=0
weight_coils=1e-16
weight_solenoids=1e-16

# lldb -- main.o \
./main.o \
    -m $mesh_file \
    -o 1 \
    -d $data_file \
    -g $refinement_factor \
    -t $do_test \
    --model $model \
    --alpha $alpha \
    --beta $beta \
    --gamma $gamma \
    --plasma_current $Ip \
    --N_control $N_control \
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
    --weight_coils $weight_coils \
    --weight_solenoids $weight_solenoids \
    --initial $do_initial




