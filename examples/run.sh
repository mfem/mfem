#!/usr/bin/env bash

diffColorIndented() {
    echo -e "$(diff --color=always -us ${1} ${2} | sed 's/^/        /')"
}

printHelp () {
    echo -e "To be able to execute this script one needs to:
    - clone MFEM (https://github.com/mfem/mfem.git)
    - OPTIONALLY clone (mfem) data (https://github.com/mfem/data)
    - clone gmsh (https://gitlab.onelab.info/gmsh/gmsh.git)
    - have gmsh's install bin dir in \$PATH to be able to execute the gmsh command.
    Note that sparse checkout will suffice for (mfem) data and gmsh because we
    will only need the content of the \"gmsh\" and \"tutorials\" folders, respectively"
	echo "Usage: bash $0.sh [options]"
	echo "Executes some mfem examples comparing mfem msh reader for format versions 2.2 and 4.1 ."
	echo "      --gmsh-home <path>: Path to gmsh"
	echo "      --mfem-home <path>: Path to mfem"
	echo "--mfem-build-home <path>: Path to mfem's build (default is build in mfem's home)"
	echo " --mfem-data-gmsh <path>: (OPTIONAL) Path to mfem's data \"gmsh\" subfolder"
	echo "                      -h: Print this help"
}

#GL: is this conditionnal required ?
#if [[ $# -eq 0 ]] ; then
#    printHelp
#    exit 0
#fi

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
        printHelp
        exit 0
        ;;
    --gmsh-home)
        GMSH_DIR="$2"
        shift
        shift
        ;;
    --mfem-home)
        MFEM_DIR="$2"
        shift
        shift
        ;;
    --mfem-build-home)
        MFEM_BUILD_DIR="$2"
        shift
        ;;
    --mfem-data-gmsh)
        MFEM_DATA_GMSH_DIR="$2"
        shift
        ;;
    -*|--*)
        echo "Unknown option $1"
        exit 1
        ;;
    *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"
if [[ -z "${MFEM_DIR}" || -z "${GMSH_DIR}" ]]; then
    echo "Missing mfem's and/or gmsh's home path!" && exit 1
    echo "MFEM_DIR=$MFEM_DIR"
    echo "GMSH_DIR=$GMSH_DIR"
fi

if [[ -z ${MFEM_DATA_GMSH_DIR} ]]; then
    echo -e "\033[1m!!! Missing mfem's data gmsh folder. The corresponding tests will not be run!!!\n\033[0m"
fi

if [ -z ${MFEM_BUILD_DIR} ]; then
     MFEM_BUILD_DIR="${MFEM_DIR}/build"
fi

if ! command -v gmsh >/dev/null 2>&1; then
    echo "gmsh command not found. Please adjust your PATH accordingly."
    exit 1
fi

MFEM_DATA="${MFEM_DIR}/data"
MFEM_BUILD_DATA="${MFEM_BUILD_DIR}/data"

echo -e "\033[1mGenerating meshes\033[0m in ${MFEM_BUILD_DATA}..."
cd ${MFEM_DATA}
MFEM_GEO_FILES=($(ls *.geo))
MFEM_MESH_NAMES=()
cd $MFEM_BUILD_DATA;
for i in ${MFEM_GEO_FILES[@]}; do
    echo -e "\t${i}"
    MFEM_MESH_NAMES+=(${i%.geo})
    gmsh -v 2 ${MFEM_DATA}/${i} -3 -format msh41 -o ${MFEM_MESH_NAMES[-1]}_41.msh &
    gmsh -v 2 ${MFEM_DATA}/${i} -3 -format msh22 -o ${MFEM_MESH_NAMES[-1]}_22.msh &
    wait
done


GMSH_TUTO_DIR=${GMSH_DIR}/tutorials
GMSH_TUT_MESH_LIST="1 3 4 5 6 7 8 10 11 13 14 15 16 17 18 19 20"

for i in $GMSH_TUT_MESH_LIST; do
    echo -e "\tt${i}.geo"
    gmsh -v 2 ${GMSH_TUTO_DIR}/t${i}.geo -3 -format msh41 -o t${i}_41.msh &
    gmsh -v 2 ${GMSH_TUTO_DIR}/t${i}.geo -3 -format msh22 -o t${i}_22.msh &
    wait
done

echo -e "\033[1m\nBuilding examples...\033[0m"
EXAMPLES_LIST="exPrintMesh ex0 ex1 ex4 ex39"
(cd $MFEM_BUILD_DIR; make --quiet -j 12 ${EXAMPLES_LIST})


# Multipe checks
cd $MFEM_BUILD_DIR/examples
echo -e "\n\033[1mRunning tests on gmsh tutorials meshes...\033[0m"
for i in $GMSH_TUT_MESH_LIST; do
    for ex in ${EXAMPLES_LIST}; do
        echo -e "- ${ex}, t${i}"
        (./${ex} -m "${GMSH_TUTO_DIR}/t${i}_22.msh" | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > res22 2>err22
        (./${ex} -m "${GMSH_TUTO_DIR}/t${i}_41.msh" | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > res41 2>err41
        [[ -s res22 || -s res41 ]] && diffColorIndented res22 res41
        if grep -q "No convergence" res*; then
            echo -e "\033[1m\tConvergence was not reached!\033[0m"
        fi
        [[ -s err22 || -s err41 ]] && echo -e "\033[1m\tErrors have occurred!\033[0m" && diffColorIndented err22 err41
        rm -f res22 res41 err22 err41
    done
done

echo -e "\n\033[1mRunning examples on mfem meshes...\033[0m"
for i in ${MFEM_MESH_NAMES[@]}; do
    for ex in ${EXAMPLES_LIST}; do
        echo -e "- ${ex}, ${i}"
        (./${ex} -m ${MFEM_BUILD_DATA}/${i}_22.msh | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > res22 2>err22
        (./${ex} -m ${MFEM_BUILD_DATA}/${i}_41.msh | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > res41 2>err41
        [[ -s res22 || -s res41 ]] && diffColorIndented res22 res41
        if grep -q "No convergence" res*; then
            echo -e "\033[1m\tConvergence was not reached!\033[0m"
        fi
        [[ -s err22 || -s err41 ]] && echo -e "\033[1m\tErrors have occurred!\033[0m" && diffColorIndented err22 err41
        rm -f res22 res41 err22 err41
    done
done

if [[ ! -z ${MFEM_DATA_GMSH_DIR} ]]; then
    GMSH_MESHES_IN_MFEM_LARGE_DATA="homology piece surfaces_in_3d"
    echo -e "\n\033[1mRunning examples on mfem data gmsh meshes...\033[0m"
    for i in ${GMSH_MESHES_IN_MFEM_LARGE_DATA}; do
        for ex in ${EXAMPLES_LIST}; do
            echo -e "- ${ex}, ${i}"
            (./${ex} -m ${MFEM_DATA_GMSH_DIR}/v22/${i}.asc.v22.msh | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > res22 2>err22
            (./${ex} -m ${MFEM_DATA_GMSH_DIR}/v41/${i}.asc.v41.msh | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > res41 2>err41
            [[ -s res22 || -s res41 ]] && diffColorIndented res22 res41
            if grep -q "No convergence" res*; then
                echo -e "\033[1m\tConvergence was not reached!\033[0m"
            fi
            [[ -s err22 || -s err41 ]] && echo -e "\033[1m\tErrors have occurred!\033[0m" && diffColorIndented err22 err41
            rm -f res22 res41 err22 err41
        done
    done
fi

echo -e "\n\033[1mRunning examples on periodic-square.mesh and the equivalent 3x3-periodic-square_41.msh...\033[0m"
for ex in exPrintMesh ex4; do
    echo -e "- ${ex}, periodic-square.mesh and 3x3-periodic-square_41.msh"
    (./${ex} -m ${MFEM_BUILD_DATA}/periodic-square.mesh | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > resMesh 2>errMesh
    (./${ex} -m ${MFEM_BUILD_DATA}/3x3-periodic-square_41.msh | grep -v -e "mesh" -e "Iteration" -e "Average reduction") > res41 2>err41
    [[ -s res22 || -s res41 ]] && diffColorIndented resMesh res41
    if grep -q "No convergence" res*; then
        echo -e "\033[1m\tConvergence was not reached!\033[0m"
    fi
    [[ -s err22 || -s err41 ]] && echo -e "\033[1m\tErrors have occurred!\033[0m" && diffColorIndented errMesh err41
    rm -f resMesh res41 errMesh err41
done

