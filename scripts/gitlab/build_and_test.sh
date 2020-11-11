#!/usr/bin/env bash

set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
project_dir="$(pwd)"

build_root=${BUILD_ROOT:-""}
hostconfig=${HOST_CONFIG:-""}
spec=${SPEC:-""}

sys_type=${SYS_TYPE:-""}
py_env_path=${PYTHON_ENVIRONMENT_PATH:-""}


# Dependencies
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ -z ${spec} ]]
    then
        echo "SPEC is undefined, aborting..."
        exit 1
    fi

    upstream_opt=""
    if [[ ${CI_COMMIT_BRANCH:-"master"} != "master" && ${sys_type} != "" ]]
    then
        upstream_opt="--upstream=/usr/workspace/mfem/mfem-spack-upstream/uberenv_libs"
    fi

    prefix_opt=""

    if [[ -d "/dev/shm" ]]
    then
        prefix="/dev/shm/${hostname}/${spec// /_}"
        mkdir -p ${prefix}
        prefix_opt="--prefix=${prefix}"
    fi

    python scripts/uberenv/uberenv.py --spec="${spec}" "${upstream_opt}"

fi

# Host config file
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"hc-*.mk ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
        echo "Found host config file: ${hostconfig_path}"
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "No result for: ${project_dir}/hc-*.mk"
        echo "Spack generated host-config not found."
        exit 1
    else
        echo "More than one result for: ${project_dir}/hc-*.mk"
        echo "${hostconfigs[@]}"
        echo "Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
fi

# Build Directory
if [[ -z ${build_root} ]]
then
    build_root=$(pwd)
fi

build_dir="${build_root}/build_${hostconfig//.mk/}"

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~ Host-config: ${hostconfig_path}"
    echo "~ Build Dir:   ${build_dir}"
    echo "~ Project Dir: ${project_dir}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building MFEM"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    make config \
      USER_CONFIG=${hostconfig_path}
    make all -j
fi

# Test
if [[ "${option}" != "--build-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing MFEM"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    return_code=0
    make test 2>&1 | tee tests_output.txt || return_code=${?}

    if grep -q "All tests passed." ./tests_output.txt
    then
        echo "SUCCESS: No error during tests" && exit 0
    else
        echo "ERROR: failure(s) while running MFEM tests" && exit 1
    fi
fi
