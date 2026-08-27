#!/usr/bin/env bash
# Representative Slurm runs for PRESB, block diagonal, and direct references.
# Run this script from an existing allocation after building the miniapps.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXE_DIR=${EXE_DIR:-"${script_dir}/../../../../build/miniapps/mtop/frq"}
NTASKS=${NTASKS:-4}
DEVICE=${DEVICE:-cpu}
RESULT_PREFIX=${RESULT_PREFIX:-frequency_domain}
RUN_MUMPS=${RUN_MUMPS:-0}
MMS_VIS=${MMS_VIS:-0}
DIMENSION=${DIMENSION:-2}
X_ELEMENTS=${X_ELEMENTS:-24}
Y_ELEMENTS=${Y_ELEMENTS:-6}
Z_ELEMENTS=${Z_ELEMENTS:-6}
SERIAL_REFINEMENTS=${SERIAL_REFINEMENTS:-0}
PARALLEL_REFINEMENTS=${PARALLEL_REFINEMENTS:-0}
MMS_REFINEMENT_LEVELS=${MMS_REFINEMENT_LEVELS:-3}
H_INVERSES=${H_INVERSES:-"lor-amg lor-cg-amg"}

usage()
{
   cat <<EOF
Usage: $(basename "$0") [options]

  --dimension N              Spatial dimension: 2 or 3
  --x-elements N             Elements in the x direction
  --y-elements N             Elements in the y direction
  --z-elements N             Elements in the z direction (3D)
  --serial-refinements N     Serial mesh refinements
  --parallel-refinements N   Parallel mesh refinements
  --mms-refinement-levels N  Number of MMS refinement levels
  --h-inverses "LIST"        Space/comma-separated H inverses chosen from
                              lor-amg, lor-cg-amg, and mumps
  --help                     Show this message

The same settings can be supplied through the uppercase environment variables.
MUMPS cases require RUN_MUMPS=1.
EOF
}

require_value()
{
   if [[ $# -lt 2 ]]; then
      echo "Missing value for $1" >&2
      usage >&2
      exit 2
   fi
}

while [[ $# -gt 0 ]]; do
   case "$1" in
      --dimension)
         require_value "$@"; DIMENSION=$2; shift 2 ;;
      --x-elements)
         require_value "$@"; X_ELEMENTS=$2; shift 2 ;;
      --y-elements)
         require_value "$@"; Y_ELEMENTS=$2; shift 2 ;;
      --z-elements)
         require_value "$@"; Z_ELEMENTS=$2; shift 2 ;;
      --serial-refinements)
         require_value "$@"; SERIAL_REFINEMENTS=$2; shift 2 ;;
      --parallel-refinements)
         require_value "$@"; PARALLEL_REFINEMENTS=$2; shift 2 ;;
      --mms-refinement-levels)
         require_value "$@"; MMS_REFINEMENT_LEVELS=$2; shift 2 ;;
      --h-inverses)
         require_value "$@"; H_INVERSES=${2//,/ }; shift 2 ;;
      --help|-h)
         usage; exit 0 ;;
      *)
         echo "Unknown option: $1" >&2
         usage >&2
         exit 2 ;;
   esac
done

if [[ "${DIMENSION}" != 2 && "${DIMENSION}" != 3 ]]; then
   echo "DIMENSION must be 2 or 3" >&2
   exit 2
fi
for value in X_ELEMENTS Y_ELEMENTS Z_ELEMENTS; do
   if ! [[ ${!value} =~ ^[1-9][0-9]*$ ]]; then
      echo "${value} must be a positive integer" >&2
      exit 2
   fi
done
for value in SERIAL_REFINEMENTS PARALLEL_REFINEMENTS; do
   if ! [[ ${!value} =~ ^[0-9]+$ ]]; then
      echo "${value} must be a nonnegative integer" >&2
      exit 2
   fi
done
if ! [[ ${MMS_REFINEMENT_LEVELS} =~ ^[0-9]+$ ]] ||
   (( 10#${MMS_REFINEMENT_LEVELS} < 2 )); then
   echo "MMS_REFINEMENT_LEVELS must be an integer of at least 2" >&2
   exit 2
fi

H_INVERSES=${H_INVERSES//,/ }
if [[ "${RUN_MUMPS}" == 1 && " ${H_INVERSES} " != *" mumps "* ]]; then
   H_INVERSES+=" mumps"
fi
read -r -a h_inverse_options <<< "${H_INVERSES}"
if [[ ${#h_inverse_options[@]} -eq 0 ]]; then
   echo "At least one H inverse must be selected" >&2
   exit 2
fi
for h_inverse in "${h_inverse_options[@]}"; do
   case "${h_inverse}" in
      lor-amg|lor-cg-amg) ;;
      mumps)
         if [[ "${RUN_MUMPS}" != 1 ]]; then
            echo "The mumps H inverse requires RUN_MUMPS=1" >&2
            exit 2
         fi
         ;;
      *)
         echo "Unknown H inverse: ${h_inverse}" >&2
         exit 2
         ;;
   esac
done

read -r -a srun_extra <<< "${SRUN_ARGS:-}"

geometry_options=(
   --dimension "${DIMENSION}"
   --x-elements "${X_ELEMENTS}"
   --y-elements "${Y_ELEMENTS}"
   --z-elements "${Z_ELEMENTS}"
   --serial-refinements "${SERIAL_REFINEMENTS}"
   --parallel-refinements "${PARALLEL_REFINEMENTS}"
)

mms_visualization=(--no-visualization)
if [[ "${MMS_VIS}" == 1 ]]; then
   mms_visualization=(--visualization --visualization-levels final \
      --output-prefix "${RESULT_PREFIX}_paraview")
fi

# Exercise every requested H inverse with both block preconditioners. Automatic
# selects GMRES/MINRES for fixed inverses and FGMRES for nested CG/AMG.
for h_inverse in "${h_inverse_options[@]}"; do
   for preconditioner in presb block-diagonal; do
      case_name=${preconditioner//-/_}_${h_inverse//-/_}
      srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
         "${EXE_DIR}/frequency_domain_cantilever" \
         --device "${DEVICE}" "${geometry_options[@]}" \
         --frequency-factor 0.5 --linear-solver automatic \
         --preconditioner "${preconditioner}" --h-inverse "${h_inverse}" \
         --csv "${RESULT_PREFIX}_cantilever_${case_name}.csv" \
         --no-visualization
   done
done

# MMS convergence with PRESB and Rayleigh damping.
srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
   "${EXE_DIR}/frequency_domain_cantilever_mms_regression" \
   --device "${DEVICE}" --dimension "${DIMENSION}" --boundary-case all \
   --refinement-levels "${MMS_REFINEMENT_LEVELS}" \
   --frequency-factor 0.5 --linear-solver automatic \
   --preconditioner presb --h-inverse lor-amg --damping-model rayleigh \
   --damping-alpha 0.08 --damping-beta 0.015 \
   --csv "${RESULT_PREFIX}_mms_presb.csv" "${mms_visualization[@]}"

# MMS convergence with block diagonal and independent damping.
srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
   "${EXE_DIR}/frequency_domain_cantilever_mms_regression" \
   --device "${DEVICE}" --dimension "${DIMENSION}" --boundary-case all \
   --refinement-levels "${MMS_REFINEMENT_LEVELS}" \
   --frequency-factor 0.5 --linear-solver automatic \
   --preconditioner block-diagonal --h-inverse lor-amg \
   --damping-model independent --mass-damping 0.072 \
   --damping-lambda 0.0345 --damping-mu 0.0255 \
   --csv "${RESULT_PREFIX}_mms_block_diagonal.csv" \
   "${mms_visualization[@]}"

# The assembled direct reference is opt-in because it requires MFEM_USE_MUMPS.
if [[ "${RUN_MUMPS}" == 1 ]]; then
   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever" \
      --device "${DEVICE}" "${geometry_options[@]}" --frequency-factor 0.5 \
      --linear-solver mumps \
      --csv "${RESULT_PREFIX}_cantilever_mumps_low.csv" --no-visualization

   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever" \
      --device "${DEVICE}" "${geometry_options[@]}" --frequency-factor 1.0 \
      --linear-solver mumps \
      --csv "${RESULT_PREFIX}_cantilever_mumps_resonance.csv" \
      --no-visualization

   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever" \
      --device "${DEVICE}" "${geometry_options[@]}" --frequency-factor 1.25 \
      --linear-solver mumps \
      --csv "${RESULT_PREFIX}_cantilever_mumps_high.csv" --no-visualization

   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever_mms_regression" \
      --device "${DEVICE}" --dimension "${DIMENSION}" \
      --boundary-case clamped \
      --refinement-levels "${MMS_REFINEMENT_LEVELS}" \
      --frequency-factor 1.25 --linear-solver mumps \
      --csv "${RESULT_PREFIX}_mms_mumps_high.csv" --no-visualization
fi
