#!/usr/bin/env bash
# Dane entry point for the three-source, square-pyramid inverse problem.
#
# Submit from RUN_ROOT, not from the Git checkout, so Slurm's .out/.err files
# and all solver output remain outside the repository:
#
#   mkdir -p /p/lustre1/$USER/topopt_runs/pyramid_inverse
#   cd /p/lustre1/$USER/topopt_runs/pyramid_inverse
#   MODE=reference sbatch /path/to/mfem/miniapps/mtop/topopt/topopt_transient/run_pyramid_inverse_dane.sh "$PWD"
#   MODE=optimize MAX_ITER=10 sbatch /path/to/mfem/miniapps/mtop/topopt/topopt_transient/run_pyramid_inverse_dane.sh "$PWD"
#
# The first command creates the reusable three-shot Q3 trace cache. The second
# loads it and carries out the optimization. Omitting MODE=reference is valid:
# MODE=optimize generates a missing cache before its first MMA step, but that
# folds the expensive reference generation into the optimization allocation.
#
# Typical parameters to change at submission time (all are environment
# variables):
#   MAX_ITER=10        absolute number of MMA updates (RESTART=1 resumes)
#   REF_LEVELS=2       uniform mesh refinements; 2 is the 100M-DoF case
#   STATE_ORDER=2      Lagrange order of displacement/velocity
#   DESIGN_ORDER=1     required by the direct DG(Q0) inverse mode
#   T_FINAL=0.5 DT=0.000125 PULSE_DURATION=0.25 FREQUENCY=4
#   REFERENCE_ORDER=3 REFERENCE_DT=0.00003125 NCHK=512 MMA_MOVE=0.1
#
# A restart must retain exactly the original MPI decomposition because its
# rank-local checkpoint is not repartitioned. Request a different node/rank
# count with sbatch options only for a fresh run.

#SBATCH --job-name=pyramid_inverse
#SBATCH --partition=pbatch
#SBATCH --nodes=32
#SBATCH --ntasks=3584
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=pyramid_inverse_%j.out
#SBATCH --error=pyramid_inverse_%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
   echo "usage: MODE={reference|optimize} $0 RUN_ROOT" >&2
   exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd -P)"
run_root="$(realpath -m -- "$1")"
mode="${MODE:-optimize}"
max_iter="${MAX_ITER:-10}"
ref_levels="${REF_LEVELS:-2}"
state_order="${STATE_ORDER:-2}"
design_order="${DESIGN_ORDER:-1}"
t_final="${T_FINAL:-0.5}"
dt="${DT:-0.000125}"
frequency="${FREQUENCY:-4}"
pulse_duration="${PULSE_DURATION:-0.25}"
reference_order="${REFERENCE_ORDER:-3}"
reference_dt="${REFERENCE_DT:-0.00003125}"
checkpoints="${NCHK:-512}"
mma_move="${MMA_MOVE:-0.1}"
restart="${RESTART:-0}"
output_name="${OUTPUT_NAME:-optimization}"
reference_cache="${run_root}/reference_boundary_cache"

case "${run_root}" in
   "${repo_root}"|"${repo_root}"/*)
      echo "ERROR: RUN_ROOT must lie outside the Git checkout: ${repo_root}" >&2
      exit 2
      ;;
esac
if [[ "${mode}" != "reference" && "${mode}" != "optimize" ]]; then
   echo "ERROR: MODE must be reference or optimize." >&2
   exit 2
fi
if ! [[ "${max_iter}" =~ ^[1-9][0-9]*$ &&
         "${ref_levels}" =~ ^[0-9]+$ &&
         "${state_order}" =~ ^[1-9][0-9]*$ &&
         "${design_order}" == "1" &&
         "${reference_order}" =~ ^[1-9][0-9]*$ &&
         "${checkpoints}" =~ ^[1-9][0-9]*$ ]]; then
   echo "ERROR: invalid integer parameter; DESIGN_ORDER must be 1 for DG(Q0)." >&2
   exit 2
fi
if [[ "${restart}" != "0" && "${restart}" != "1" ]]; then
   echo "ERROR: RESTART must be 0 or 1." >&2
   exit 2
fi

solver="${script_dir}/TopOptTransient"
if [[ ! -x "${solver}" ]]; then
   cat >&2 <<EOF
ERROR: missing executable: ${solver}
Build it from the checkout first, using the Dane compiler/MPI modules:
  module --force purge
  module load StdEnv gcc/13.3.1 mvapich2/2.3.7
  make -C ${script_dir} TopOptTransient
EOF
   exit 1
fi

mkdir -p "${run_root}"
if [[ "${mode}" == "reference" ]]; then
   output_dir="${run_root}/reference_generation"
   if [[ -e "${reference_cache}" || -e "${output_dir}" ]]; then
      echo "ERROR: reference cache or output already exists under ${run_root}." >&2
      echo "       Use MODE=optimize to reuse a completed cache, or choose a new RUN_ROOT." >&2
      exit 1
   fi
   mode_args=(
      -reference-only -no-reference-audit
      -no-ckpt -no-pv
   )
else
   output_dir="${run_root}/${output_name}"
   if [[ -e "${output_dir}" && "${restart}" != "1" ]]; then
      echo "ERROR: output already exists: ${output_dir}" >&2
      echo "       Choose OUTPUT_NAME=... for a fresh run, or use RESTART=1." >&2
      exit 1
   fi
   if [[ "${restart}" == "1" &&
         ! -f "${output_dir}/optimization_checkpoint/metadata.txt" ]]; then
      echo "ERROR: RESTART=1 needs ${output_dir}/optimization_checkpoint/metadata.txt" >&2
      exit 1
   fi
   mode_args=(
      -mi "${max_iter}" -mv "${mma_move}" -tol 1e-4
      -mma-objective-scale 1e10 -ckpt -no-ckpt-history -no-pv
   )
   if [[ "${restart}" == "1" ]]; then
      mode_args+=( -restart )
   fi
fi

module --force purge
module load StdEnv gcc/13.3.1 mvapich2/2.3.7

echo "Started $(date -Is)"
echo "mode=${mode}; nodes=${SLURM_JOB_NUM_NODES}; ranks=${SLURM_NTASKS}"
echo "run_root=${run_root}"
echo "reference_cache=${reference_cache}"
echo "mesh_refinements=${ref_levels}; state=H1(Q${state_order}); density=DG(Q0)"
echo "truth=square pyramid; linear SIMP; no volume constraint; no Helmholtz filter"
if [[ "${ref_levels}" == "2" && "${state_order}" == "2" &&
      "${SLURM_NTASKS}" != "3584" ]]; then
   echo "WARNING: the 100M-DoF preset was measured on 3,584 ranks; this is a fresh, different decomposition." >&2
fi

exec srun --mpi=pmi2 --kill-on-bad-exit=1 -n "${SLURM_NTASKS}" \
   env OMP_NUM_THREADS=1 \
   "${solver}" \
   -problem elastic-inclusion-identification-3d-multisource \
   -inclusion-truth pyramid \
   -matrix-free-symplectic-euler -inverse-identity-density-transfer \
   -lumped-mass -damp \
   -r "${ref_levels}" -o "${state_order}" -do "${design_order}" \
   -tf "${t_final}" -dt "${dt}" -freq "${frequency}" -dur "${pulse_duration}" \
   -no-filter -no-volume-constraint -simp-p 1 \
   -reference-order "${reference_order}" -reference-dt "${reference_dt}" \
   -reference-cache "${reference_cache}" -no-reference-audit \
   -objective-quadrature legacy -trajectory-storage revolve -nchk "${checkpoints}" \
   -init uniform -adjoint-mode discrete \
   "${mode_args[@]}" -out "${output_dir}"
