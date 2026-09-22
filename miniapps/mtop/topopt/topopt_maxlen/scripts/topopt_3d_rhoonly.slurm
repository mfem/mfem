#!/bin/bash
# Submit with:
#   sbatch /p/lustre2/yu52/codes/run_topopt_3d/topopt_3d_rhoonly.slurm
#
# Optional output-directory override:
#   sbatch /p/lustre2/yu52/codes/run_topopt_3d/topopt_3d_rhoonly.slurm /path/to/run_directory

#SBATCH --job-name=elast3d_rhoonly
#SBATCH --partition=pbatch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu52
#SBATCH --output=/p/lustre2/yu52/codes/run_topopt_3d/run_outputs/elast3d_rhoonly_%j.out
#SBATCH --error=/p/lustre2/yu52/codes/run_topopt_3d/run_errors/elast3d_rhoonly_%j.err
#SBATCH -A gradient

set -euo pipefail

# Same toolchain MFEM/hypre were built with (Intel classic 2021.6.0 + MVAPICH2 2.3.7).
module load intel-classic/2021.6.0-magic
module load mvapich2/2.3.7

ulimit -c 0

solver=/p/lustre2/yu52/codes/mfem/miniapps/mtop/topopt/topopt_maxlen/ElastTopOpt_3d_rhoonly
mesh=/p/lustre2/yu52/codes/run_topopt_3d/circular_plate_hex_sleeves_embedded_cylinder.msh
run_dir=${1:-/p/lustre2/yu52/codes/run_topopt_3d/elast3d_rhoonly_${SLURM_JOB_ID}}

if [[ ! -x "${solver}" ]]; then
   echo "ERROR: executable is missing or not executable: ${solver}" >&2
   exit 1
fi
if [[ ! -r "${mesh}" ]]; then
   echo "ERROR: mesh is missing: ${mesh}" >&2
   exit 1
fi
if [[ -e "${run_dir}/run.log" ]]; then
   echo "ERROR: refusing to overwrite existing ${run_dir}/run.log" >&2
   exit 1
fi

mkdir -p "${run_dir}"
cd "${run_dir}"

solver_args=(
   -m "${mesh}"       # Input mesh
   -r 2               # Parallel uniform refinements (~8x elements per level in 3D)
   -o 1               # H1 elasticity/filter polynomial order
   -vf 0.3            # Target volume fraction
   -rf 1.5            # Helmholtz-filter width
   -amin 1e-6         # Lower thickness bound (only with -thickness)
   -amax 25           # Upper thickness bound: the value rho_a is penalized above
   -b 4               # Initial Heaviside beta
   -bs 100            # Iterations between beta increases
   -bm 4              # Maximum beta
   -eta 0.2           # Heaviside threshold
   -e 1e-2            # Initial thickness-constraint tolerance (unnormalized; see header)
   -d 0.5             # Thickness-tolerance decay factor (only with -thickness)
   -di 50             # Iterations between tolerance reductions (only with -thickness)
   -ii 5              # Initial iterations before decay/convergence checks
   -mi 300            # Maximum topology iterations
   -tol 1e-15         # Stopping tolerance on max design change
   -mv 0.02           # MMA move limit (density only in this driver)
   -pc 4              # Elasticity PC: 0 Jacobi, 1 LOR diag AMG, 2 LOR mono AMG, 3 full diag AMG, 4 full mono AMG
   -elast-fa          # Fully assembled elasticity operator (alt: -elast-pa)
   -vdim              # byVDIM ordering + rigid-body modes for monolithic AMG (alt: -nodes)
   -ert 1e-5          # Elasticity relative tolerance
   -erc 10            # Recompute true residual every N CG iterations (0 = native CG stopping)
   -cp 1              # Checkpointing: 0 off, 1 density, 2 density + thickness/MMA state
   -restart 0         # Restart: 0 fresh, 1 density, 2 full state
   -pv                # ParaView output (alt: -no-pv)
   -pvi 50            # Intermediate ParaView output every N iterations (0 = off)
   -no-vis            # No GLVis
   -opt               # Run optimization (alt: -no-opt)
   -thickness         # Thickness constraint on (alt: -no-thickness)
   -rt 4              # Rays: 1 radial outward, 2 vertical +z, 3 xy plane, 4 = 2 and 3
   -spl 1             # Solver print level: 0 quiet, 1 iterations, 2 verbose
   -dgo 1             # DG order for thickness advection (ignored with -no-thickness)
   -adv-gmres         # GMRES + BlockILU advection (needs -adv-fa; unusable with -rt 1)
   -adv-fa            # Full-assembly advection (alt: -adv-pa)
   -cfl 0.5           # Pseudo-time CFL (ignored with -adv-gmres)
   -atf 50            # Pseudo-time terminal cap (ignored with -adv-gmres)
   -atol 1e-6         # Steady-state rate tolerance (ignored with -adv-gmres)
   -minv-fa           # Assembled DG mass inverse (alt: -minv-mf)
   -mit 1e-8          # Matrix-free mass-inverse tolerance (only with -thickness -minv-mf)
)

echo "Started: $(date -Is)"
echo "Nodes: ${SLURM_JOB_NUM_NODES}; MPI ranks: ${SLURM_NTASKS}"
echo "Run directory: ${run_dir}"
echo "Executable: ${solver}"

export OMP_NUM_THREADS=1
exec srun --kill-on-bad-exit=1 -n "${SLURM_NTASKS}" "${solver}" "${solver_args[@]}" \
   > run.log 2>&1
