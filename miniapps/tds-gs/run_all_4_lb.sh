# coefficient of ff' term
alpha=-4.19
# coefficient of p' term
beta=0.0
# unused?
gamma=0.0

# model
# 1: ff' defined from fpol data
# 2: Taylor equilibrium
# 3: ff' defined from ff' data
# 4: LB
model=4

# plasma current
Ip=1.5e+7
# Ip=1.2e+7

R0=2.4
rho_gamma=16
mu=12.5663706144e-7
# mu=1.0

mesh_file="meshes/iter_gen.msh"
# mesh_file="meshes/RegGSTriMeshVeryCoarse0beta.msh"
data_file="data/separated_file.data"
initial_gf="initial/initial.gf"

refinement_factor=1
amr_frac_in=0.08
# amr_frac_in=0.3
amr_frac_out=0.3
max_levels=1
max_dofs=100000

do_test=0
do_manufactured_solution=0
do_initial=0

# linear solver parameters
light_tol=1e-8
max_krylov_iter=300
krylov_tol=1e-6
max_newton_iter=20
newton_tol=1e-6

# number of control points on plasma
N_control=100
do_control=1
weight_coils=1e-12
weight_solenoids=1e-12
weight_obj=1.0
optimize_alpha=1

# objective function
# 0: sum_k (psi_k - psi_0) ^ 2
# 1: sum_k (psi_N_k - 1) ^ 2
# 2: sum_k (psi_k - psi_x) ^ 2
obj_option=1

alpha_in=1.61803398875
gamma_in=1.0

# 0: block AMG
# 1: block AMG schur comp
# 2: AMG on full
# 3: AMG on partial full block
# 4: schur complement
# 5: gauss seidel
pc_option=5
amg_cycle_type=0
amg_num_sweeps_a=1
amg_num_sweeps_b=1
amg_max_iter=1

# poloidal flux coils
c6=-4.552585e+06
c7=3.180596e+06
c8=5.678096e+06
c9=3.825538e+06
c10=1.066498e+07
c11=-2.094771e+07

# center solenoids
c1=1.143284e+03
c2=-2.478694e+04
c3=-3.022037e+04
c4=-2.205664e+04
c5=-2.848113e+03

ur_coeff=1.0

for pc_option in 0 5 6
do
    for amg_cycle_type in 0 1
    do
        for amg_max_iter in 1 3 5 10
        do
            mpirun -np 1 main.o \
                   -m $mesh_file \
                   --initial_gf $initial_gf \
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
                   --initial $do_initial \
                   --weight_obj $weight_obj \
                   --obj_option $obj_option \
                   --optimize_alpha $optimize_alpha \
                   --pc_option $pc_option \
                   --max_levels $max_levels \
                   --max_dofs $max_dofs \
                   --light_tol $light_tol \
                   --alpha_in $alpha_in \
                   --gamma_in $gamma_in \
                   --amg_cycle_type $amg_cycle_type \
                   --amg_num_sweeps_a $amg_num_sweeps_a \
                   --amg_num_sweeps_b $amg_num_sweeps_b \
                   --amg_max_iter $amg_max_iter \
                   --amr_frac_in $amr_frac_in \
                   --amr_frac_out $amr_frac_out
        done
    done
done
    




# ./../gslib/field-interp -m1 mesh.mesh -m2 meshes/geqdsk.msh -s1 final.gf -no-vis
