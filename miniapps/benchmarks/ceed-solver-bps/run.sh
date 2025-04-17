bsep="============================================================"
ssep="----------------------------------------"
# Enable GPU-aware MPI:
# gpu_aware_mpi_env_cmd="env MPICH_GPU_SUPPORT_ENABLED=1"
# gpu_aware_mpi="-g"
# number of nodes, number of MPI ranks:
nnodes=1
np=1
# dev="-d hip ${gpu_aware_mpi}"
eps="0.3"
# mpirun_np="mpirun -np"
mpirun_np="env MFEM_REPORT_KERNELS=1 mpirun -np"
# mpirun_np="${gpu_aware_mpi_env_cmd} flux run --exclusive -N ${nnodes} -n"
# dry run:
# mpirun_np="echo ${mpirun_np}"
#    p-MG/LOR + FA-hypre, or diagonal (Jacobi smoother)
#    prec_type: "p-mg", "lor", or "diag"
prec_type="lor"
p_mg_opts="-cb 1"
# p_mg_opts="-cb 5 -sli -sli-it 6"
# lor_opts="-cls -cb 5 -sli -sli-it 6"
# lor_opts="-cls -cb 2 -sli -sli-it 2"
lor_opts="-cb 2 -sli -sli-it 2"
mg_set=("1" "1 2" "1 3" "1 2 4" "1 3 5" "1 3 6")
# mg_set=("1 2")
#    p=7 and p=8 fail at the moment: "1 3 5 7" "1 3 5 8"
# per-rank limits on the number of LOR elements for different p, in 2^20 units:
# (bigger sizes run out of GPU memory, at least with LOR prec.)
lor_ne_max_all=(4 4 4 4 4 4 4 4)
# lor_ne_max_all=(18 22 24 24 27 24 8 8) # MI250X
((lor_ne_min = 40*2**10))
((np_ = np))
((mm = 1))
while ((np_ > 8)); do
   ((mm++))
   ((np_ = (np_-1)/8+1))
done
((mf = 2**mm))
((mff = 3*mf))
echo " *** mf = ${mf}, mff = ${mff}"
for mg in "${mg_set[@]}"; do
   echo "${bsep}"
   p=(${mg})
   # p=${p[-1]}
   p="${p[$((${#p[@]}-1))]}"
   lor_ne_max="${lor_ne_max_all[$((p-1))]}"
   ((lor_ne_max *= 2**20))
   # n_max = floor(lor_ne_max^(1/3))
   n_max=$(echo "a=e((1/3)*l(${np}*${lor_ne_max}));scale=0;a/1" | bc -l)
   # for np*lor_ne_max=256^3, the above gives 255, so we adjust the result:
   while (( (n_max+1)**3 <= np*lor_ne_max )); do
      ((n_max++))
   done
   echo " *** p = ${p}, n_max = ${n_max}"
   if (( n_max**3 > np*lor_ne_max )); then
      echo "error: n_max^3 > np*lor_ne_max"
      exit 1
   fi
   echo "${bsep}"
   nx_set=()
   for ((nx = (n_max/p/mff)*mff, last_nx = 2*nx; nx >= 6; nx -= mff)); do
      ((last_ne = last_nx**3))
      ((ne = nx**3))
      ((lor_ne = (p*nx)**3))
      if ((np*lor_ne_min > lor_ne)); then break; fi
      if ((last_ne < ne*4/3)); then continue; fi
      nx_set=("${nx}" "${nx_set[@]}")
      ((ndofs = (p*nx+1)**3))
      ((rhs_n=0))
      while ((2*3**(rhs_n+1) <= p*nx)); do
         ((rhs_n++))
      done
      # 2*3**rhs_n <= p*nx < 2*3**(rhs_n+1)
      printf "np = ${np}, p = ${p}, nx = ${nx}, ndofs = ${ndofs}"
      # rhs_n for eps = 1:
      # printf ", rhs_n = ${rhs_n}"
      printf "\n"
      ((last_nx = nx))
   done
   for nx in "${nx_set[@]}"; do
      # break;
      if ((nx % mf != 0)); then
         echo " *** internal error!"
         exit 1
      fi
      ((rp = mm))
      ((nx /= mf))
      if false; then
         # 0, 1, or 2 additional parallel refinements for 1, 8, or 64 ranks
         ((np_=np))
         while ((np_%8 == 0)); do
            ((np_=np_/8))
            ((rp++))
         done
      fi
      ((ndofs = (p*nx*2**rp+1)**3))
      echo "${bsep}"
      echo "np = ${np}, p = ${p}, ndofs = ${ndofs}"
      if [[ "$prec_type" == "p-mg" ]]; then
         # p-MG
         printf "$mpirun_np ${np} ./solver-bp ${dev}"
         printf " -ey ${eps} -mg \"${mg}\" -cs 1 ${p_mg_opts}"
         printf " -nx ${nx} -rp ${rp}\n"
         echo "${ssep}"
         $mpirun_np "${np}" ./solver-bp ${dev} \
            -ey ${eps} -mg "${mg}" -cs 1 ${p_mg_opts} -nx "${nx}" -rp "${rp}"
      elif [[ "$prec_type" == "lor" ]]; then
         # LOR
         printf "$mpirun_np ${np} ./solver-bp ${dev}"
         printf " -ey ${eps} -mg \"${p}\" -cs 2 ${lor_opts}"
         printf " -nx ${nx} -rp ${rp}\n"
         echo "${ssep}"
         $mpirun_np "${np}" ./solver-bp ${dev} \
            -ey ${eps} -mg "${p}" -cs 2 ${lor_opts} -nx "${nx}" -rp "${rp}"
      elif [[ "$prec_type" == "diag" ]]; then
         # Diag
         printf "$mpirun_np ${np} ./solver-bp ${dev}"
         printf " -ey ${eps} -mg \"${p}\" -cs 0 -nx ${nx} -rp ${rp}\n"
         echo "${ssep}"
         $mpirun_np "${np}" ./solver-bp ${dev} \
            -ey ${eps} -mg "${p}" -cs 0 -nx "${nx}" -rp "${rp}"
      fi
   done
done
