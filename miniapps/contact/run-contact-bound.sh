#!/usr/bin/env bash
set -euo pipefail

# Trike uses OpenMPI with CPU/CUDA. LC retains its CPU/Flux and HIP/srun paths;
# HIP ranks bind GPU 0..3 to their local NUMA domains 0..3.
contact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$contact_dir/../../.." && pwd)
host=$(hostname -s)
trike=0
default_ranks=4
case $host in
  trike|trike.*) trike=1; default_ranks=1 ;;
esac
cores=${CORES_PER_RANK:-}
ranks=${CONTACT_RANKS:-$default_ranks}
driver=${CONTACT_DRIVER:-contact}
worker=0
if [[ ${1:-} == --rank ]]; then
  worker=1
  shift
else
  while (( $# > 0 )); do
    case $1 in
      --ranks|-n)
        if (( $# < 2 )); then
          echo "$1 requires a count" >&2
          exit 2
        fi
        ranks=$2
        shift 2
        ;;
      --cores)
        if (( $# < 2 )); then
          echo "--cores requires a count" >&2
          exit 2
        fi
        cores=$2
        shift 2
        ;;
      --driver)
        if (( $# < 2 )); then
          echo "--driver requires contact or contact_examples" >&2
          exit 2
        fi
        driver=$2
        shift 2
        ;;
      --help)
        cat <<'EOF'
Usage: ./run-contact-bound.sh [--driver contact|contact_examples] [--ranks N] [--cores N] [--] [CONTACT_OPTIONS...]

On trike: OpenMPI with CPU/CUDA, default 1 rank and 1 core per rank.
CUDA ranks share device 0 in CUDA_VISIBLE_DEVICES (default physical GPU 0).
On LC: one allocated node, default 4 ranks; CPU uses Flux and HIP uses srun.
LC HIP runs use 1 to 4 ranks, one GPU per rank, default 4 cores per rank.
--cores sets CPU cores per rank (default 1 for CPU).
All contact options are passed through verbatim.
contact accepts problems 0..2; contact_examples accepts 3..8.
With no CONTACT_OPTIONS, the saved default case is used.
On LC, the HYPRE GPU-aware MPI option automatically enables MPICH GPU support.

Examples:
  ./run-contact-bound.sh
  # Trike (CUDA):
  ./run-contact-bound.sh --ranks 1 -- -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4 -amgf -amgf-fsolver mumps -fa -d cuda -no-vis -no-paraview
  ./run-contact-bound.sh --driver contact_examples --ranks 1 -- -prob 6 -sr 0 -nsteps 32 -msteps 32 -amgf -amgf-fsolver mumps -d cuda -no-vis -paraview
  # LC (HIP):
  ./run-contact-bound.sh --ranks 4 -- -prob 2 -sr 1 -pr 0 -tr 2 -nsteps 4 -msteps 0 -amgf -amgf-fsolver strumpack -fa -d hip
  ./run-contact-bound.sh --ranks 2 --cores 8 -- -prob 2 -sr 1 -d hip
  ./run-contact-bound.sh --ranks 16 -- -prob 0 -sr 0 -pr 3 -amgf -amgf-fsolver mumps -d cpu
  ./run-contact-bound.sh -- -h
EOF
        exit 0
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done
fi
case $driver in
  contact|contact_examples) ;;
  *) echo "--driver must be contact or contact_examples" >&2; exit 2 ;;
esac
if (( $# == 0 )); then
  if [[ $driver == contact_examples ]]; then
    set -- -prob 8 -sr 0 -pr 0 -strip-nx 4 -twist-angle 360 \
      -nsteps 36 -msteps 0 -amgf -amgf-fsolver auto -d cpu -no-vis -paraview
  elif (( trike )); then
    set -- -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4 -msteps 0 \
      -amgf -amgf-fsolver mumps -fa -d cuda
  else
    set -- -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4 -msteps 0 \
      -amgf -amgf-fsolver strumpack -fa -d hip
  fi
fi

if (( trike )); then
  export PATH="/usr/local/cuda-13.4/bin:/usr/lib64/openmpi/bin:/opt/rh/gcc-toolset-15/root/usr/bin:$PATH"
  export LD_LIBRARY_PATH="$root/local/slate/lib:$root/local/cudss/lib:/usr/local/cuda-13.4/lib64:/usr/lib64/openmpi/lib:/opt/rh/gcc-toolset-15/root/usr/lib64:/opt/rh/gcc-toolset-15/root/usr/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$root/local/slate-hip/lib:/opt/rocm-7.2.1/lib:/opt/rocm-7.2.1/llvm/lib:/opt/cray/pe/mpich/9.1.0/ofi/amd/7.0/lib:${LD_LIBRARY_PATH:-}"
fi
# Determine the actual backend before choosing resources and affinity.
# MFEM_DEVICE, when set, overrides the miniapp's -d option in MFEM itself.
device=cpu
hypre_gpu_mpi=0
app_args=("$@")
for (( i=0; i<${#app_args[@]}; i++ )); do
  case ${app_args[i]} in
    -d|--device)
      if (( i + 1 >= ${#app_args[@]} )); then
        echo "${app_args[i]} requires a device configuration" >&2
        exit 2
      fi
      i=$((i + 1))
      device=${app_args[i]}
      ;;
    -hypre-gpu-aware-mpi|--hypre-gpu-aware-mpi) hypre_gpu_mpi=1 ;;
    -no-hypre-gpu-aware-mpi|--no-hypre-gpu-aware-mpi) hypre_gpu_mpi=0 ;;
  esac
done
device=${MFEM_DEVICE:-$device}
if (( trike )); then
  case $device in
    *hip*)
      echo "Trike supports CPU and CUDA; use -d cuda for its GPU" >&2
      exit 2
      ;;
    *cuda*|*gpu*) backend=cuda ;;
    *) backend=cpu ;;
  esac
else
  case $device in
    *cuda*)
      echo "This Tuolumne launcher supports host and HIP backends, not CUDA" >&2
      exit 2
      ;;
    *hip*) backend=hip ;;
    *) backend=cpu ;;
  esac
fi
if [[ -z $cores ]]; then
  if [[ $backend == hip ]]; then cores=4; else cores=1; fi
fi
if ! [[ $cores =~ ^[1-9][0-9]*$ && $ranks =~ ^[1-9][0-9]*$ ]]; then
  echo "Rank count and CPU cores per rank must be positive integers" >&2
  exit 2
fi
if [[ $backend == hip ]] && (( cores > 24 || ranks > 4 )); then
  echo "HIP runs support 1 to 4 ranks and 1 to 24 CPU cores per rank" >&2
  exit 2
fi
if [[ $backend == cpu ]] && (( hypre_gpu_mpi )); then
  echo "HYPRE GPU-aware MPI requires a GPU backend; omit that option for -d cpu" >&2
  exit 2
fi
export CORES_PER_RANK=$cores
export CONTACT_RANKS=$ranks
export CONTACT_DRIVER=$driver

# Enabling HYPRE device-buffer communication also requires MPI GPU support.
if (( ! trike )); then
  if [[ $backend == cpu ]]; then
    export MPICH_GPU_SUPPORT_ENABLED=0
  elif (( hypre_gpu_mpi )); then
    export MPICH_GPU_SUPPORT_ENABLED=1
  else
    export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}
  fi
  export SPINDLE_FLUXOPT=disable
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd "$contact_dir"

if (( worker )); then
  if (( trike )); then
    rank=${OMPI_COMM_WORLD_LOCAL_RANK:-}
  else
    rank=${FLUX_TASK_RANK:-${SLURM_LOCALID:-}}
  fi
  if ! [[ $rank =~ ^[0-9]+$ ]] || (( rank >= ranks )); then
    echo "Expected local rank 0..$((ranks - 1)), got '${rank:-unset}'" >&2
    exit 2
  fi

  if (( trike )); then
    gpu=none
    if [[ $backend == cuda ]]; then
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
      gpu=${CUDA_VISIBLE_DEVICES%%,*}
    fi
    cpu_list=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
    printf 'host=%s rank=%s gpu=%s cpus=%s mpi=openmpi hypre_gpu=%s\n' \
      "$(hostname)" "$rank" "$gpu" "$cpu_list" "$hypre_gpu_mpi" >&2
    exec "./$driver" "$@"
  fi

  if [[ $backend == cpu ]]; then
    cpu_list=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
    printf 'host=%s rank=%s gpu=none numa=local cpus=%s mpich_gpu=0 hypre_gpu=0\n' \
      "$(hostname)" "$rank" "$cpu_list" >&2
    exec numactl --localalloc "./$driver" "$@"
  fi

  # One physical-core range per GPU/NUMA node, with no overlap between ranks.
  first=$((24 * rank))
  last=$((first + cores - 1))
  cpu_range="$first-$last"
  actual_node=$(lscpu -p=CPU,NODE | awk -F, -v cpu="$first" '$1 == cpu { print $2; exit }')
  if [[ $actual_node != "$rank" ]]; then
    echo "CPU $first belongs to NUMA node '$actual_node', expected $rank; topology changed" >&2
    exit 2
  fi

  export ROCR_VISIBLE_DEVICES=$rank
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  taskset -pc "$cpu_range" "$$" >/dev/null
  cpu_list=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
  printf 'host=%s rank=%s gpu=%s numa=%s cpus=%s mpich_gpu=%s hypre_gpu=%s\n' \
    "$(hostname)" "$rank" "$ROCR_VISIBLE_DEVICES" "$rank" "$cpu_list" \
    "${MPICH_GPU_SUPPORT_ENABLED}" "$hypre_gpu_mpi" >&2
  exec numactl --membind="$rank" "./$driver" "$@"
fi

if (( trike )); then
  exec /usr/bin/time -f 'TOTAL_WALL_SECONDS=%e exit_status=%x' \
    mpirun -np "$ranks" --map-by "slot:PE=$cores" --bind-to core \
    "$contact_dir/run-contact-bound.sh" --rank "$@"
fi

if [[ $backend == cpu ]]; then
  # Use Flux directly: the site's srun wrapper ignores --cpu-bind=cores.
  # Per-task binding uses the allocated CPU sets, including ranks above 3.
  exec /usr/bin/time -f 'TOTAL_WALL_SECONDS=%e exit_status=%x' \
    flux run -N 1 -n "$ranks" -c "$cores" \
    -o cpu-affinity=per-task -o gpu-affinity=off -o mpibind=off \
    "$contact_dir/run-contact-bound.sh" --rank "$@"
fi

# The site's srun is a Flux wrapper. Turn off its affinity plugins so numactl
# controls both CPU and memory locality. -N 1 keeps local rank == GPU number.
exec /usr/bin/time -f 'TOTAL_WALL_SECONDS=%e exit_status=%x' \
  srun -N 1 -n "$ranks" -c "$cores" --gpus-per-task=1 \
  --gpu-bind=none --cpu-bind=none --mpibind=off \
  "$contact_dir/run-contact-bound.sh" --rank "$@"
