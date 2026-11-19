#!/usr/bin/env bash
set -euo pipefail

# Single-node CPU or Tuolumne HIP runs. CPU ranks use scheduler affinity;
# HIP ranks bind GPU 0..3 to their local NUMA domains 0..3.
contact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$contact_dir/../../.." && pwd)
cores=${CORES_PER_RANK:-}
ranks=${CONTACT_RANKS:-4}
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
      --help)
        cat <<'EOF'
Usage: ./run-contact-bound.sh [--ranks N] [--cores N] [--] [CONTACT_OPTIONS...]

Run on one allocated node (default 4 MPI ranks).
With -d cpu: any positive rank count that fits the allocation; no GPUs requested.
With -d hip: 1 to 4 MPI ranks, one GPU per rank.
--cores sets CPU cores per rank (default 1 on CPU, 4 on HIP).
All contact options are passed through verbatim.
With no CONTACT_OPTIONS, the saved default case is used.
The HYPRE GPU-aware MPI option automatically enables MPICH GPU support.

Examples:
  ./run-contact-bound.sh
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
if (( $# == 0 )); then
  set -- -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4 -msteps 0 \
    -amgf -amgf-fsolver strumpack -fa -d hip
fi

export LD_LIBRARY_PATH="$root/local/slate-hip/lib:/opt/rocm-7.2.1/lib:/opt/rocm-7.2.1/llvm/lib:/opt/cray/pe/mpich/9.1.0/ofi/amd/7.0/lib:${LD_LIBRARY_PATH:-}"
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
case $device in
  *cuda*)
    echo "This Tuolumne launcher supports host and HIP backends, not CUDA" >&2
    exit 2
    ;;
  *hip*) backend=hip ;;
  *) backend=cpu ;;
esac
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

# Enabling HYPRE device-buffer communication also requires MPI GPU support.
if [[ $backend == cpu ]]; then
  export MPICH_GPU_SUPPORT_ENABLED=0
elif (( hypre_gpu_mpi )); then
  export MPICH_GPU_SUPPORT_ENABLED=1
else
  export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SPINDLE_FLUXOPT=disable

cd "$contact_dir"

if (( worker )); then
  rank=${FLUX_TASK_RANK:-${SLURM_LOCALID:-}}
  if ! [[ $rank =~ ^[0-9]+$ ]] || (( rank >= ranks )); then
    echo "Expected local rank 0..$((ranks - 1)), got '${rank:-unset}'" >&2
    exit 2
  fi

  if [[ $backend == cpu ]]; then
    cpu_list=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
    printf 'host=%s rank=%s gpu=none numa=local cpus=%s mpich_gpu=0 hypre_gpu=0\n' \
      "$(hostname)" "$rank" "$cpu_list" >&2
    exec numactl --localalloc ./contact "$@"
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
  exec numactl --membind="$rank" ./contact "$@"
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
