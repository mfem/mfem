#!/usr/bin/env bash
# Slurm study for frequency-domain solver and preconditioner combinations.
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
STUDY_REFINEMENT_LEVELS=${STUDY_REFINEMENT_LEVELS:-3}
H_INVERSES=${H_INVERSES:-"lor-amg lor-cg-amg"}
LINEAR_SOLVERS=${LINEAR_SOLVERS:-"automatic gmres"}
PRECONDITIONERS=${PRECONDITIONERS:-"presb block-diagonal"}
DAMPING_COEFFICIENT=${DAMPING_COEFFICIENT:-0.02}
DAMPING_BETA=${DAMPING_BETA:-0.0}
DAMPING_VALUES=${DAMPING_VALUES:-"0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.125 0.15"}
FREQUENCY_FACTORS=${FREQUENCY_FACTORS:-"0.25 0.5 0.75 0.9"}

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
  --study-refinement-levels N
                             Number of cantilever study mesh levels
  --h-inverses "LIST"        Space/comma-separated H inverses chosen from
                              lor-amg, lor-cg-amg, and mumps
  --linear-solvers "LIST"    Space/comma-separated solvers chosen from
                              automatic, gmres, fgmres, and minres
  --preconditioners "LIST"   Space/comma-separated preconditioners chosen
                              from presb and block-diagonal
  --damping-coefficient C    Rayleigh mass coefficient alpha (default: 0.02)
  --damping-beta C           Fixed Rayleigh stiffness coefficient
  --damping-values "LIST"    Rayleigh alpha values for the cantilever study
  --frequency-factors "LIST" Excitation/first-eigenfrequency ratios in (0,1)
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
      --study-refinement-levels)
         require_value "$@"; STUDY_REFINEMENT_LEVELS=$2; shift 2 ;;
      --h-inverses)
         require_value "$@"; H_INVERSES=${2//,/ }; shift 2 ;;
      --linear-solvers)
         require_value "$@"; LINEAR_SOLVERS=${2//,/ }; shift 2 ;;
      --preconditioners)
         require_value "$@"; PRECONDITIONERS=${2//,/ }; shift 2 ;;
      --damping-coefficient)
         require_value "$@"; DAMPING_COEFFICIENT=$2; shift 2 ;;
      --damping-beta)
         require_value "$@"; DAMPING_BETA=$2; shift 2 ;;
      --damping-values)
         require_value "$@"; DAMPING_VALUES=${2//,/ }; shift 2 ;;
      --frequency-factors)
         require_value "$@"; FREQUENCY_FACTORS=${2//,/ }; shift 2 ;;
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
if ! [[ ${STUDY_REFINEMENT_LEVELS} =~ ^[0-9]+$ ]] ||
   (( 10#${STUDY_REFINEMENT_LEVELS} < 1 )); then
   echo "STUDY_REFINEMENT_LEVELS must be a positive integer" >&2
   exit 2
fi

nonnegative_number='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
for value in DAMPING_COEFFICIENT DAMPING_BETA; do
   if ! [[ ${!value} =~ ${nonnegative_number} ]]; then
      echo "${value} must be a nonnegative number" >&2
      exit 2
   fi
done

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

LINEAR_SOLVERS=${LINEAR_SOLVERS//,/ }
read -r -a linear_solver_options <<< "${LINEAR_SOLVERS}"
if [[ ${#linear_solver_options[@]} -eq 0 ]]; then
   echo "At least one linear solver must be selected" >&2
   exit 2
fi
for linear_solver in "${linear_solver_options[@]}"; do
   case "${linear_solver}" in
      automatic|gmres|fgmres|minres) ;;
      *)
         echo "Unknown linear solver: ${linear_solver}" >&2
         exit 2
         ;;
   esac
done

PRECONDITIONERS=${PRECONDITIONERS//,/ }
read -r -a preconditioner_options <<< "${PRECONDITIONERS}"
if [[ ${#preconditioner_options[@]} -eq 0 ]]; then
   echo "At least one preconditioner must be selected" >&2
   exit 2
fi
for preconditioner in "${preconditioner_options[@]}"; do
   case "${preconditioner}" in
      presb|block-diagonal) ;;
      *)
         echo "Unknown preconditioner: ${preconditioner}" >&2
         exit 2
         ;;
   esac
done

DAMPING_VALUES=${DAMPING_VALUES//,/ }
read -r -a damping_values <<< "${DAMPING_VALUES}"
if [[ ${#damping_values[@]} -eq 0 ]]; then
   echo "At least one damping value must be selected" >&2
   exit 2
fi
for damping in "${damping_values[@]}"; do
   if ! [[ ${damping} =~ ${nonnegative_number} ]]; then
      echo "Invalid nonnegative damping value: ${damping}" >&2
      exit 2
   fi
done

FREQUENCY_FACTORS=${FREQUENCY_FACTORS//,/ }
read -r -a frequency_factors <<< "${FREQUENCY_FACTORS}"
if [[ ${#frequency_factors[@]} -eq 0 ]]; then
   echo "At least one frequency factor must be selected" >&2
   exit 2
fi
for frequency_factor in "${frequency_factors[@]}"; do
   if ! [[ ${frequency_factor} =~ ${nonnegative_number} ]] ||
      ! awk -v value="${frequency_factor}" \
         'BEGIN { exit !(value > 0.0 && value < 1.0) }'; then
      echo "Frequency factors must be strictly between zero and one: " \
           "${frequency_factor}" >&2
      exit 2
   fi
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

damping_options=(
   --damping-model rayleigh
   --damping-alpha "${DAMPING_COEFFICIENT}"
   --damping-beta "${DAMPING_BETA}"
)

mms_visualization=(--no-visualization)
if [[ "${MMS_VIS}" == 1 ]]; then
   mms_visualization=(--visualization --visualization-levels final \
      --output-prefix "${RESULT_PREFIX}_paraview")
fi

# An automatic solve chooses GMRES/MINRES for fixed inverses and FGMRES for
# nested CG/AMG. Explicitly requested incompatible combinations are skipped.
valid_solver_combination()
{
   local linear_solver=$1
   local preconditioner=$2
   local h_inverse=$3
   [[ ${linear_solver} != gmres || ${h_inverse} != lor-cg-amg ]] &&
      [[ ${linear_solver} != minres ||
         (${preconditioner} == block-diagonal &&
          ${h_inverse} != lor-cg-amg) ]]
}

study_csv="${RESULT_PREFIX}_study.csv"
study_table="${RESULT_PREFIX}_study.md"
study_case_dir="${RESULT_PREFIX}_study_cases"
mkdir -p "${study_case_dir}"
study_header_written=0
study_cases=0
study_failures=0

for damping in "${damping_values[@]}"; do
   damping_tag=${damping//./p}
   study_damping_options=(
      --damping-model rayleigh
      --damping-alpha "${damping}"
      --damping-beta "${DAMPING_BETA}"
   )
   for frequency_factor in "${frequency_factors[@]}"; do
      frequency_tag=${frequency_factor//./p}
      for linear_solver in "${linear_solver_options[@]}"; do
         for preconditioner in "${preconditioner_options[@]}"; do
            for h_inverse in "${h_inverse_options[@]}"; do
               if ! valid_solver_combination "${linear_solver}" \
                  "${preconditioner}" "${h_inverse}"; then
                  continue
               fi
               for ((mesh_level = 0;
                     mesh_level < STUDY_REFINEMENT_LEVELS;
                     ++mesh_level)); do
                  study_parallel_refinements=$((
                     PARALLEL_REFINEMENTS + mesh_level))
                  total_refinements=$((
                     SERIAL_REFINEMENTS + study_parallel_refinements))
                  refinement_scale=$((2 ** total_refinements))
                  effective_x=$((X_ELEMENTS * refinement_scale))
                  effective_y=$((Y_ELEMENTS * refinement_scale))
                  effective_z=1
                  if [[ ${DIMENSION} == 3 ]]; then
                     effective_z=$((Z_ELEMENTS * refinement_scale))
                  fi

                  solver_tag=${linear_solver//-/_}
                  preconditioner_tag=${preconditioner//-/_}
                  h_inverse_tag=${h_inverse//-/_}
                  case_name="d${damping_tag}_f${frequency_tag}_l${mesh_level}"
                  case_name+="_${solver_tag}_${preconditioner_tag}"
                  case_name+="_${h_inverse_tag}"
                  case_csv="${study_case_dir}/${case_name}.csv"

                  # Avoid mistaking a stale result for output from a failed
                  # launch. A normal non-converged solve still writes one row.
                  : > "${case_csv}"
                  case_exit=0
                  srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
                     "${EXE_DIR}/frequency_domain_cantilever" \
                     --device "${DEVICE}" --dimension "${DIMENSION}" \
                     --x-elements "${X_ELEMENTS}" \
                     --y-elements "${Y_ELEMENTS}" \
                     --z-elements "${Z_ELEMENTS}" \
                     --serial-refinements "${SERIAL_REFINEMENTS}" \
                     --parallel-refinements \
                     "${study_parallel_refinements}" \
                     --frequency-factor "${frequency_factor}" \
                     --linear-solver "${linear_solver}" \
                     --preconditioner "${preconditioner}" \
                     --h-inverse "${h_inverse}" \
                     "${study_damping_options[@]}" --csv "${case_csv}" \
                     --no-visualization || case_exit=$?

                  case_converged=$(awk -F, '
                     NR == 1 {
                        for (column = 1; column <= NF; ++column)
                        {
                           if ($column == "converged") { converged = column }
                        }
                        next
                     }
                     NR == 2 && converged { print $converged; found = 1; exit }
                     END { if (!found) { exit 1 } }
                  ' "${case_csv}") || {
                     echo "Study case did not produce a valid CSV row: " \
                          "${case_name}" >&2
                     if (( case_exit == 0 )); then
                        case_exit=1
                     fi
                     exit "${case_exit}"
                  }
                  if [[ ${case_converged} != 1 ]]; then
                     ((++study_failures))
                     echo "NON-CONVERGED study case: ${case_name}" >&2
                  elif (( case_exit != 0 )); then
                     echo "Study case reported convergence but srun exited " \
                          "with status ${case_exit}: ${case_name}" >&2
                     exit "${case_exit}"
                  fi

                  if (( study_header_written == 0 )); then
                     {
                        printf '%s' \
                           'mesh_level,x_elements,y_elements,z_elements,'
                        printf '%s' \
                           'requested_damping,requested_frequency_factor,'
                        sed -n '1p' "${case_csv}"
                     } > "${study_csv}"
                     study_header_written=1
                  fi
                  while IFS= read -r result_row; do
                     printf '%s,%s,%s,%s,%s,%s,%s\n' \
                        "${mesh_level}" "${effective_x}" "${effective_y}" \
                        "${effective_z}" "${damping}" \
                        "${frequency_factor}" "${result_row}" \
                        >> "${study_csv}"
                  done < <(sed -n '2,$p' "${case_csv}")
                  ((++study_cases))
               done
            done
         done
      done
   done
done

if (( study_cases == 0 )); then
   echo "No valid solver/preconditioner/H-inverse combinations" >&2
   exit 2
fi

awk -F, '
   NR == 1 {
      for (column = 1; column <= NF; ++column) { field[$column] = column }
      header = "| Level | Dim | nx | ny | nz | Elements | Total DOFs | "
      header = header "Damping | Frequency / omega1 | Requested solver | "
      header = header "Active solver | Preconditioner | H inverse | "
      header = header "Converged | Outer iterations | H applications | "
      print header "H iterations |"
      separator = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
      separator = separator "---|---|---|---|---|---:|---:|---:|"
      print separator
      next
   }
   {
      printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
             $(field["mesh_level"]), $(field["dimension"]),
             $(field["x_elements"]), $(field["y_elements"]),
             $(field["z_elements"]), $(field["elements"]),
             $(field["total_dofs"]), $(field["requested_damping"]),
             $(field["frequency_ratio"]), $(field["requested_solver"]),
             $(field["active_solver"])
      printf " %s | %s | %s | %s | %s | %s |\n",
             $(field["preconditioner"]), $(field["h_inverse"]),
             $(field["converged"]), $(field["outer_iterations"]),
             $(field["h_inverse_applications"]),
             $(field["h_inverse_iterations"])
   }
' "${study_csv}" > "${study_table}"

echo "Wrote ${study_cases} study rows to ${study_csv}"
echo "Wrote the iteration table to ${study_table}"
if (( study_failures > 0 )); then
   echo "Recorded ${study_failures} non-converged study cases; continuing " \
        "with the remaining regression runs." >&2
fi

# MMS convergence uses the same Rayleigh damping for both preconditioners.
srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
   "${EXE_DIR}/frequency_domain_cantilever_mms_regression" \
   --device "${DEVICE}" --dimension "${DIMENSION}" --boundary-case all \
   --refinement-levels "${MMS_REFINEMENT_LEVELS}" \
   --frequency-factor 0.5 --linear-solver automatic \
   --preconditioner presb --h-inverse lor-amg "${damping_options[@]}" \
   --csv "${RESULT_PREFIX}_mms_presb.csv" "${mms_visualization[@]}"

# MMS convergence with block diagonal and the same damping configuration.
srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
   "${EXE_DIR}/frequency_domain_cantilever_mms_regression" \
   --device "${DEVICE}" --dimension "${DIMENSION}" --boundary-case all \
   --refinement-levels "${MMS_REFINEMENT_LEVELS}" \
   --frequency-factor 0.5 --linear-solver automatic \
   --preconditioner block-diagonal --h-inverse lor-amg \
   "${damping_options[@]}" \
   --csv "${RESULT_PREFIX}_mms_block_diagonal.csv" \
   "${mms_visualization[@]}"

# The assembled direct reference is opt-in because it requires MFEM_USE_MUMPS.
if [[ "${RUN_MUMPS}" == 1 ]]; then
   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever" \
      --device "${DEVICE}" "${geometry_options[@]}" --frequency-factor 0.5 \
      --linear-solver mumps "${damping_options[@]}" \
      --csv "${RESULT_PREFIX}_cantilever_mumps_low.csv" --no-visualization

   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever" \
      --device "${DEVICE}" "${geometry_options[@]}" --frequency-factor 1.0 \
      --linear-solver mumps "${damping_options[@]}" \
      --csv "${RESULT_PREFIX}_cantilever_mumps_resonance.csv" \
      --no-visualization

   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever" \
      --device "${DEVICE}" "${geometry_options[@]}" --frequency-factor 1.25 \
      --linear-solver mumps "${damping_options[@]}" \
      --csv "${RESULT_PREFIX}_cantilever_mumps_high.csv" --no-visualization

   srun --ntasks="${NTASKS}" "${srun_extra[@]}" \
      "${EXE_DIR}/frequency_domain_cantilever_mms_regression" \
      --device "${DEVICE}" --dimension "${DIMENSION}" \
      --boundary-case clamped \
      --refinement-levels "${MMS_REFINEMENT_LEVELS}" \
      --frequency-factor 1.25 --linear-solver mumps \
      "${damping_options[@]}" \
      --csv "${RESULT_PREFIX}_mms_mumps_high.csv" --no-visualization
fi

if (( study_failures > 0 )); then
   echo "Study completed with ${study_failures} non-converged cases." >&2
   exit 1
fi
