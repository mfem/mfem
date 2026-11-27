#!/usr/bin/env bash
set -euo pipefail

# Laptop is the default CPU/OpenMPI launcher. Trike uses CPU/CUDA;
# LC uses CPU/Flux or HIP/srun with GPU/NUMA affinity.
contact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$contact_dir/../../.." && pwd)
site=${CONTACT_SITE:-laptop}
cores=${CORES_PER_RANK:-}
ranks=${CONTACT_RANKS:-}
driver=${CONTACT_DRIVER:-contact}
worker=0
if [[ ${1:-} == --rank ]]; then
  worker=1
  shift
else
  while (( $# > 0 )); do
    case $1 in
      --site)
        if (( $# < 2 )); then
          echo "--site requires laptop, trike, or lc" >&2
          exit 2
        fi
        site=$2
        shift 2
        ;;
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
Usage: ./run-contact.sh [--site laptop|trike|lc] [--driver contact|contact_examples] [--ranks N] [--cores N] [--] [CONTACT_OPTIONS...]

Default site: laptop (or CONTACT_SITE environment variable).
Laptop: CPU/OpenMPI, default 1 rank; no Linux affinity tools required.
Trike: OpenMPI with CPU/CUDA, default 1 rank and 1 core per rank.
CUDA ranks share device 0 in CUDA_VISIBLE_DEVICES (default physical GPU 0).
LC: one allocated node, default 4 ranks; CPU uses Flux and HIP uses srun.
LC HIP runs use 1 to 4 ranks, one GPU per rank, default 4 cores per rank.
--cores sets cores per rank for Trike/LC; laptop uses MPI's default mapping.
--ranks N selects the number of MPI processes for every site.
Driver defaults: zero refinements, CG+AMGF, automatic filtered backend, CPU,
GLVis and ParaView enabled, print level 2.
Nonlinear geometry/material updates are enabled automatically, default limit 1.
-load-steps sets the loading ramp length. contact also accepts -extra-load-steps
for sliding (0/1) or continued pressure loading (2). The examples are loading-only.
All contact options are passed through verbatim.
contact accepts problems 0..2; contact_examples accepts 3..8.
With no CONTACT_OPTIONS, the saved default case for the selected site is used.
On LC, the HYPRE GPU-aware MPI option automatically enables MPICH GPU support.

Sample runs (laptop defaults; change --ranks to set MPI processes):
  # 0 - Two blocks: compression
  ./run-contact.sh --ranks 1 --driver contact -- -prob 0 -load-steps 4 -extra-load-steps 0

  # 1 - Ironing
  ./run-contact.sh --ranks 1 --driver contact -- -prob 1 -load-steps 4 -extra-load-steps 6

  # 2 - Beam/sphere: nonlinear elasticity
  ./run-contact.sh --ranks 1 --driver contact -- -prob 2 -nonlin -load-steps 20 -contact-max-updates 1

  # 3 - Nested spherical shells
  ./run-contact.sh --ranks 1 --driver contact_examples -- -prob 3 -load-steps 4

  # 4 - Folded strip: pressure loading
  ./run-contact.sh --ranks 1 --driver contact_examples -- -prob 4 -load-steps 20 -contact-max-updates 5

  # 5 - Compressed ring: pressure loading
  ./run-contact.sh --ranks 1 --driver contact_examples -- -prob 5 -load-steps 20 -contact-max-updates 5

  # 6 - C-clip/elastic pin: insertion and elastic opening
  ./run-contact.sh --ranks 1 --driver contact_examples -- -prob 6 -load-steps 32 -contact-max-updates 5

  # 7 - Two initially touching elastic strips
  ./run-contact.sh --ranks 1 --driver contact_examples -- -prob 7 -hex -strip-nx 24 -twist-angle 360 -load-steps 72 -contact-max-updates 5

  # 8 - Rope tightening and squeezing an elastic cylinder
  ./run-contact.sh --ranks 1 --driver contact_examples -- -prob 8 -tet -load-steps 48 -contact-max-updates 5

Other sites:
  # Trike (CUDA)
  ./run-contact.sh --site trike --ranks 1 -- -prob 0 -load-steps 4 -fa -d cuda
  # LC (HIP)
  ./run-contact.sh --site lc --ranks 4 -- -prob 2 -load-steps 4 -amgf-fsolver strumpack -fa -d hip
  # LC (CPU)
  ./run-contact.sh --site lc --ranks 16 -- -prob 0 -pr 3 -d cpu

Solver option help:
  ./run-contact.sh -- -h
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
trike=0
laptop=0
case $site in
  laptop) laptop=1; default_ranks=1 ;;
  trike) trike=1; default_ranks=1 ;;
  lc) default_ranks=4 ;;
  *) echo "--site must be laptop, trike, or lc" >&2; exit 2 ;;
esac
ranks=${ranks:-$default_ranks}
case $driver in
  contact|contact_examples) ;;
  *) echo "--driver must be contact or contact_examples" >&2; exit 2 ;;
esac
if (( $# == 0 )); then
  # Common solver/output defaults are defined in the drivers.
  if [[ $driver == contact_examples ]]; then
    set -- -prob 7 -strip-nx 4 -twist-angle 360 -load-steps 36 -contact-max-updates 5
  elif (( laptop )); then
    set -- -prob 0 -load-steps 4 -extra-load-steps 0
  elif (( trike )); then
    set -- -prob 0 -tr 2 -load-steps 4 -extra-load-steps 0 \
      -amgf-fsolver mumps -fa -d cuda
  else
    set -- -prob 0 -tr 2 -load-steps 4 -extra-load-steps 0 \
      -amgf-fsolver strumpack -fa -d hip
  fi
fi

if (( trike )); then
  export PATH="/usr/local/cuda-13.4/bin:/usr/lib64/openmpi/bin:/opt/rh/gcc-toolset-15/root/usr/bin:$PATH"
  export LD_LIBRARY_PATH="$root/local/slate/lib:$root/local/cudss/lib:/usr/local/cuda-13.4/lib64:/usr/lib64/openmpi/lib:/opt/rh/gcc-toolset-15/root/usr/lib64:/opt/rh/gcc-toolset-15/root/usr/lib:${LD_LIBRARY_PATH:-}"
elif (( ! laptop )); then
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
elif (( laptop )); then
  case $device in
    *cuda*|*hip*|*gpu*)
      echo "The laptop launcher uses CPU; use -d cpu" >&2
      exit 2
      ;;
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
export CONTACT_SITE=$site
export CORES_PER_RANK=$cores
export CONTACT_RANKS=$ranks
export CONTACT_DRIVER=$driver

# Enabling HYPRE device-buffer communication also requires MPI GPU support.
if (( ! trike && ! laptop )); then
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

if (( laptop )); then
  if (( worker )); then
    exec "./$driver" "$@"
  fi
  # Bash timing works on macOS and Linux without GNU /usr/bin/time flags.
  TIMEFORMAT='TOTAL_WALL_SECONDS=%3R'
  time mpirun -np "$ranks" "$contact_dir/run-contact.sh" --rank "$@"
  exit 0
fi

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
    "$contact_dir/run-contact.sh" --rank "$@"
fi

if [[ $backend == cpu ]]; then
  # Use Flux directly: the site's srun wrapper ignores --cpu-bind=cores.
  # Per-task binding uses the allocated CPU sets, including ranks above 3.
  exec /usr/bin/time -f 'TOTAL_WALL_SECONDS=%e exit_status=%x' \
    flux run -N 1 -n "$ranks" -c "$cores" \
    -o cpu-affinity=per-task -o gpu-affinity=off -o mpibind=off \
    "$contact_dir/run-contact.sh" --rank "$@"
fi

# The site's srun is a Flux wrapper. Turn off its affinity plugins so numactl
# controls both CPU and memory locality. -N 1 keeps local rank == GPU number.
exec /usr/bin/time -f 'TOTAL_WALL_SECONDS=%e exit_status=%x' \
  srun -N 1 -n "$ranks" -c "$cores" --gpus-per-task=1 \
  --gpu-bind=none --cpu-bind=none --mpibind=off \
  "$contact_dir/run-contact.sh" --rank "$@"
