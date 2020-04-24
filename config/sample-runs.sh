#!/bin/bash

# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

make="${MAKE:-make}"
mpiexec="${MPIEXEC:-mpirun}"
mpiexec_np="${MPIEXEC_NP:--np}"
run_prefix=""
run_vg="valgrind --leak-check=full --show-reachable=yes --track-origins=yes"
run_suffix="-no-vis"
skip_gen_meshes="yes"
# filter-out device runs ("no") or non-device runs ("yes"):
device_runs="no"
cur_dir="${PWD}"
mfem_dir="$(cd "$(dirname "$0")"/.. && pwd)"
mfem_build_dir=""
build_log=""
output_dir=""
output_sfx=".out"
# The group format is: '"group-name" "group-summary-title" "group-directory"
# "group-source-patterns"'
groups_serial=(
'"examples"
   "Examples:"
   "examples"
   "ex{,1,2}[0-9].cpp"'
#   "ex1.cpp"'
'"sundials"
   "SUNDIALS examples:"
   "examples/sundials"
   "ex{9,10,16}.cpp"'
'"performance"
   "Performance miniapps:"
   "miniapps/performance"
   "ex1.cpp"'
#   ""'
'"meshing"
   "Meshing miniapps:"
   "miniapps/meshing"
   "mobius-strip.cpp klein-bottle.cpp extruder.cpp toroid.cpp
    mesh-optimizer.cpp minimal-surface.cpp"'
)
# Parallel groups
groups_parallel=(
'"examples"
   "Examples:"
   "examples"
   "ex{,1,2}[0-9]p.cpp"'
#   "ex1p.cpp"'
'"sundials"
   "SUNDIALS examples:"
   "examples/sundials"
   "ex{9,10,16}p.cpp"'
'"petsc"
   "PETSc examples:"
   "examples/petsc"
   "ex{,1}[0-9]p.cpp"'
'"performance"
   "Performance miniapps:"
   "miniapps/performance"
   "ex1p.cpp"'
#   ""'
'"meshing"
   "Meshing miniapps:"
   "miniapps/meshing"
   "pmesh-optimizer.cpp pminimal-surface.cpp"'
'"electromagnetics"
   "Electromagnetics miniapps:"
   "miniapps/electromagnetics"
   "joule.cpp"'
#   "{volta,tesla,joule}.cpp"' # todo: multiline sample runs
)
# All groups serial + parallel runs mixed in the same group:
groups_all=(
'"examples"
   "Examples:"
   "examples"
   "ex\"{,1,2}[0-9]\"{,p}.cpp"'
'"sundials"
   "SUNDIALS examples:"
   "examples/sundials"
   "ex\"{9,10,16}\"{,p}.cpp"'
'"petsc"
   "PETSc examples:"
   "examples/petsc"
   "ex{,1}[0-9]p.cpp"'
'"performance"
   "Performance miniapps:"
   "miniapps/performance"
   "ex1{,p}.cpp"'
'"meshing"
   "Meshing miniapps:"
   "miniapps/meshing"
   "mobius-strip.cpp klein-bottle.cpp extruder.cpp toroid.cpp
    {,p}mesh-optimizer.cpp {,p}minimal-surface.cpp"'
'"electromagnetics"
   "Electromagnetics miniapps:"
   "miniapps/electromagnetics"
   "joule.cpp"'
#   "{volta,tesla,joule}.cpp"' # todo: multiline sample runs
)
make_all="all"
base_timeformat=$'real: %3Rs  user: %3Us  sys: %3Ss  %%cpu: %P'
# separator
sep='----------------------------------------------------------------'

# Command line parameters:
opt_help="no"
opt_show="no"
mfem_config="MFEM_USE_MPI=NO MFEM_DEBUG=NO"
groups=()
group_name=""
group_title=""
valgrind="no"
make_j="-j $(getconf _NPROCESSORS_ONLN)"
color="no"
built="no"
timing="no"

# Read the sample runs from the source "$1" and put them in the array variable
# "runs".
function extract_sample_runs()
{
   local old_IFS="${IFS}" sruns="" pruns=""
   local src="$1"
   if [ "${src}" == "" ]; then runs=(); return 1; fi
   local app=${src%.cpp}
   local vg_app="${app}"
   if [ "${valgrind}" == "yes" ]; then vg_app="${run_vg} ${app}"; fi
   # parallel sample runs are lines matching "^//.*  mpirun .* ${app}" with
   # everything in front of "mpirun" removed:
   pruns=`grep "^//.*  mpirun .* ${app}" "${src}" |
          sed -e "s/.*  mpirun \(.*\) ${app}/${mpiexec} \1 ${app}/g" \
              -e "s/ -np\(.*\) ${app}/ ${mpiexec_np}\1 ${vg_app}/g"`
   # serial sample runs are lines that are not parallel sample runs and matching
   # "^//.*  ${app}" with everything in front of "${app}" removed:
   sruns=`grep -v "^//.*  mpirun .* ${app}" "${src}" |
          grep "^//.*  ${app}" |
          sed -e "s/.*  ${app}/${vg_app}/g"`
   runs="${sruns}${pruns}"
   if [ "$skip_gen_meshes" == "yes" ]; then
      runs=`printf "%s" "$runs" | grep -v ".* -m .*\.gen"`
   fi
   if [ "$device_runs" == "yes" ]; then
      runs=`printf "%s" "$runs" | grep ".* -d .*"`
      if [ "$have_occa" == "no" ]; then
         runs=`printf "%s" "$runs" | grep -v ".* -d occa-.*"`
      fi
      if [ "$have_raja" == "no" ]; then
         runs=`printf "%s" "$runs" | grep -v ".* -d raja-.*"`
      fi
      if [ "$have_ceed" == "no" ]; then
         runs=`printf "%s" "$runs" | grep -v ".* -d ceed-.*"`
      fi
   else
      runs=`printf "%s" "$runs" | grep -v ".* -d .*"`
   fi
   IFS=$'\n'
   runs=(${runs})
   IFS="${old_IFS}"
}

# Echo usage information
function help_message()
{
   cat <<EOF

   $0 [options]

   Options:
      -h|-help    Print this usage information and exit
      -p|-par     Build the parallel MFEM library + examples + miniapps.
                  The default is to build the serial MFEM library + examples +
                  miniapps. The build can be customized by setting the variable
                  'mfem_config' or by building separately and using '-b'
      -g <dir> <pattern>
                  Specify explicitly a group (dir + file pattern) to run; This
                  option can be used multiple times to define multiple groups
      -dev        configure only sample runs using devices.
                  To test with a parallel build, the parallel (-p|-par) option
                  should be set first on the command line.
      -v          Enable valgrind
      -o <dir>    [${output_dir:-"<empty>: output goes to stdout"}]
                  If not empty, save output to files inside <dir>
      -d <dir>    [${mfem_build_dir}]
                  If <dir> is different from <mfem_dir> then use an
                  out-of-source build in <dir>
      -j <np>     [${make_j}] Specify the number of jobs to use for building
      -c|-color   Always use colors for the status messages: OK, FAILED, etc
      -b|-built   Do NOT rebuild the library and the executables
      -t|-time    Measure and print execution time for each sample run
      -s|-show    Show all configured sample runs and exit
      -n          Dry run: replace "\$sample_run" with "echo \$sample_run"
      <var>=<value>
                  Set a shell script variable; see below for valid variables
       *          Any other parameter is treated as <mfem_dir>
      <mfem_dir>  [${mfem_dir}] is the MFEM source directory

   This script tests all the sample runs listed in the beginning comments of
   MFEM's serial or parallel example and miniapp codes. The list of sample runs
   is auto-generated and can be viewed with the -s|-show option.

   The following shell script variables can be set with <var>=<value>:
      output_dir [${output_dir}]
         Same as '-o': if not empty, save output to files in that directory
      output_sfx [${output_sfx}]
         Suffix to append to the output files
      mfem_config [${mfem_config}]
         Set MFEM configuration options
      make [${make}], mpiexec [${mpiexec}], mpiexec_np [${mpiexec_np}]
         Their values can also set using the respective uppercase environment
         variable
      mfem_build_dir [${mfem_build_dir}]
         Same as '-d': set this variable to something different from <mfem_dir>
         to use an out-of-source build

   For other valid variables, see the script source.

   The following environment variables, if non-empty, are used:
      MAKE, MPIEXEC, MPIEXEC_NP

   Example usage:
       $0 -s                [ Show all configured sample runs ]
       $0 -o baseline       [ Serial build; run and save all built sample runs ]
       $0 -p -o baseline    [ Parallel build; run and save all sample runs ]
       $0 -b -g examples ex8.cpp [ Use the existing build; run ex8 sample runs ]

EOF
}

function show_runs()
{
   echo "${sep}"
   for group_params in "${groups[@]}"; do
      eval params=(${group_params})
      name="${params[0]}"
      title="${params[1]}"
      group_dir="${mfem_dir}/${params[2]}"
      pattern="${params[3]}"
      printf "group name:    [%s]\n" "${name}"
      printf "summary title: [%s]\n" "${title}"
      printf "directory:     [%s]\n" "${group_dir}"
      printf "pattern:       [%s]\n" "${pattern}"
      cd "${cur_dir}"; cd "${group_dir}" || exit 1
      eval sources=(${pattern})
      eval sources=("${sources[@]}")
      printf "sources:       (%s)\n" "${sources[*]}"
      printf "sample runs:\n"
      for src in "${sources[@]}"; do
         extract_sample_runs "${src}"
         for run in "${runs[@]}"; do
            printf "   %s\n" "${run}"
         done
      done
      echo "${sep}"
   done
}

# Process command line parameters
while [ $# -gt 0 ]; do

case "$1" in
   -h|-help)
      opt_help="yes"
      ;;
   -p|-par)
      mfem_config="MFEM_USE_MPI=YES MFEM_DEBUG=NO"
      ;;
   -g)
      gbasename="$(basename "$2")"
      gname="${group_name:-${gbasename}}"
      gtitle="${group_title:-"Group <${gbasename}>:"}"
      test_group="\"${gname}\" \"${gtitle}\" \"$2\" \"$3\""
      groups=("${groups[@]}" "${test_group}")
      shift 2
      ;;
   -dev)
       device_runs="yes"
       mfem_config+=" MFEM_USE_CUDA=YES MFEM_USE_OPENMP=YES"
       # OCCA, RAJA, libCEED are enabled below, if available
      ;;
   -v)
      valgrind="yes"
      ;;
   -o)
      shift
      output_dir="$1"
      ;;
   -d)
      shift
      mfem_build_dir="$1"
      ;;
   -j)
      shift
      make_j="-j $1"
      ;;
   -c|-color)
      color="yes"
      ;;
   -b|-built)
      built="yes"
      ;;
   -t|-time)
      timing="yes"
      ;;
   -s|-show)
      opt_show="yes"
      ;;
   -n)
      run_prefix="echo"
      ;;
   -*)
      echo "unknown option: '$1'"
      exit 1
      ;;
   *=*)
      eval $1
      ;;
   *)
      mfem_dir="$1"
      ;;
esac

shift
done # while ...

mfem_build_dir="${mfem_build_dir:-${mfem_dir}}"

build_log="${build_log:-${mfem_build_dir}/config/sample-runs-build.log}"

if [ 0 -eq ${#groups[*]} ]; then
   groups=("${groups_all[@]}")
   # These can be used as command line arguments:
   #   'groups=("${groups_serial[@]}")'
   #   'groups=("${groups_parallel[@]}")'
fi

if [ "${opt_help}" == "yes" ]; then
   help_message
   exit
fi

if [ "${opt_show}" == "yes" ]; then
   show_runs
   exit
fi

# Setup colors
if [ -t 1 ] && [ -z "${output_dir}" ] || [ "${color}" == "yes" ]; then
   red='\033[0;31m'
   green='\033[0;32m'
   yellow='\033[0;33m'
   magenta='\033[0;35m'
   cyan='\033[0;36m'
   none='\033[0m'
else
   red=
   green=
   yellow=
   magenta=
   cyan=
   none=
fi

# Run the given command, saving the rune time in the variable "timer".
function timed_run()
{
   timer="$({ time "$@" 1>&3 2>&4; } 2>&1)"
} 3>&1 4>&2

# This function is used to execute the sample runs
function go()
{
   local cmd=("$@")
   local res=""
   echo $sep
   echo "<${group}>" "${cmd[@]}"
   echo $sep
   if [ "${timing}" == "yes" ]; then
      timed_run "${cmd[@]}"
   else
      "${cmd[@]}"
   fi
   if [ "$?" -eq 0 ]; then
      res="${green}  OK  ${none}"
   else
      res="${red}FAILED${none}"
   fi
   printf "[${res}] <${group}> ${cmd[*]}\n"
   if [ "${timing}" == "yes" ]; then
      printf "Run time: %s\n" "${timer}"
      timer=(${timer})
      timer="${timer[1]}"
      printf -v line "[$res](%8s) ${cmd[*]}" "$timer"
      summary=("${summary[@]}" "$line")
   else
      summary=("${summary[@]}" "[${res}] ${cmd[*]}")
   fi
   echo $sep
}

# This function is used to run a group of sample runs (in the same directory)
function go_group()
{
   local res=""
   if [ $# -eq 0 ]; then return 0; fi
   local group_output_dir="" output_file="" output=""
   if [ ! -z "$output_dir" ]; then
      group_output_dir="${output_dir}/${group_dir}"
      mkdir -p "${group_output_dir}" || exit 1
   fi
   for src in "$@"; do
      cd "${mfem_dir}/${group_dir}" || exit 1
      extract_sample_runs "${src}" || continue
      [ "${#runs[@]}" -eq 0 ] && continue
      cd "${mfem_build_dir}/${group_dir}" || exit 1
      if [ ! -x "${src%.cpp}" ]; then
         res="${magenta} SKIP ${none}"
         echo $sep
         printf "[${res}] <${group}> <${src}>\n"
         echo $sep
         summary=("${summary[@]}" "[${res}] <${src}>")
         continue
      fi
      if [ ! -z "$output_dir" ]; then
         output_file="${group_output_dir}/${src}${output_sfx}"
         : > "${output_file}"
         output=">> \"${output_file}\" 2>&1"
      fi
      for run in "${runs[@]}"; do
         if [ "${run}" == "" ]; then continue; fi
         eval go \${run_prefix} \${run} \${run_suffix} $output
      done
   done
   ${make} clean-exec
}

# Make sure $mfem_dir exists and we can cd into it
cd "$mfem_dir" || exit 1
# Make sure $mfem_dir is an absolute path
mfem_dir="$PWD"
cd "${cur_dir}"
if [ "${built}" == "no" ]; then
   mkdir -p "${mfem_build_dir}" || exit 1
fi
# Make sure $mfem_build_dir exists and we can cd into it
cd "${mfem_build_dir}" || exit 1
# Make sure $mfem_build_dir is an absolute path
mfem_build_dir="$PWD"
# Setup 'output_dir'
if [ ! -z "$output_dir" ]; then
   cd "${cur_dir}"
   mkdir -p "${output_dir}" && cd "${output_dir}" || exit 1
   output_dir="$PWD"
   echo "Sending output to files in: [${output_dir}]"
   echo "Using suffix: [${output_sfx}]"
fi

TIMEFORMAT="${base_timeformat}"

# Setup optional libraries when not using externally built MFEM:
if [ "${built}" == "no" ]; then
   have_occa="no"
   have_raja="no"
   have_ceed="no"
   if [ "${device_runs}" == "yes" ]; then
      if [ -n "${CUDA_ARCH}" ]; then
         mfem_config+=" CUDA_ARCH=${CUDA_ARCH}"
      fi
      if [ -d "${mfem_dir}/../occa" ]; then
         mfem_config+=" MFEM_USE_OCCA=YES"
         have_occa="yes"
      fi
      if [ -d "${mfem_dir}/../raja" ]; then
         mfem_config+=" MFEM_USE_RAJA=YES"
         have_raja="yes"
      fi
      if [ -d "${mfem_dir}/../libCEED" ]; then
         mfem_config+=" MFEM_USE_CEED=YES"
         have_ceed="yes"
      fi
   fi
fi

function set_echo_log()
{
   local dirname=`dirname "$1"`
   cd "${cur_dir}"
   mkdir -p "${dirname}" && cd "${dirname}" || exit 1
   echo_log="$PWD"/`basename "$1"`
}

# Echo the given command line; then run it sending all output to $echo_log
function echo_run()
{
   echo "   $@"
   { echo "   $@"; echo "$sep";
     "$@"
     echo "$sep"; } >> "$echo_log" 2>&1
}

# Function that builds the mfem library, examples and miniapps
function build_all()
{
   printf "Building MFEM with all examples and miniapps:\n"
   set_echo_log "${build_log}"
   echo "   ### build log: [$echo_log]"
   { echo "$sep"; echo "   MFEM build log"; echo "$sep"; } > "$echo_log"
   echo_run cd "${mfem_build_dir}"
   if [ "${mfem_dir}" != "${mfem_build_dir}" ]; then
      echo_run ${make} -f "${mfem_dir}"/makefile config
   fi
   # Don't use 'make distclean' as it will delete the default $build_log
   echo_run ${make} clean || exit 1
   echo_run ${make} config ${mfem_config} || exit 1
   echo_run ${make} ${make_j} || exit 1
   echo_run ${make} ${make_all} ${make_j} || exit 1
}

# Function that runs all sample runs, given by the array variable "groups".
function all_go()
{
   for group_params in "${groups[@]}"; do
      eval params=(${group_params})
      group="${params[0]}"
      group_dir="${params[2]}"
      cd "${mfem_dir}/${group_dir}" || exit 1
      eval sources=(${params[3]})
      eval sources=("${sources[@]}")
      summary=("${summary[@]}" "${params[1]}")
      go_group "${sources[@]}"
   done

   printf "Summary:\n--------\n"
   for line in "${summary[@]}"; do
      printf "${line}\n"
   done
}

function main()
{
   # Build all mfem examples and miniapps
   if [ "${built}" == "no" ]; then
      if [ "${timing}" == "yes" ]; then
         timed_run build_all
         printf "Build time: %s\n" "${timer}"
      else
         build_all
      fi
   fi

   summary=()
   PATH=.:$PATH

   # Print the MFEM configuration info
   cd "${mfem_build_dir}"
   echo "$sep"
   echo "MFEM configuration"
   echo "$sep"
   ${make} info
   echo "$sep"

   # Run all sample runs.
   if [ "${timing}" == "yes" ]; then
      timed_run all_go
      printf "Total run time: %s\n" "${timer}"
   else
      all_go
   fi
   echo
}

output=""
if [ ! -z "$output_dir" ]; then
   output=">> \"${output_dir}/main${output_sfx}\" 2>&1"
fi
eval main $output
