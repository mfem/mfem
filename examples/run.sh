#!/usr/bin/env bash

# One needs to clone GMSH and MFEM open source repositories before executing this script

# git clone https://gitlab.onelab.info/gmsh/gmsh.git
GMSH_TUTO_DIR=${HOME}/Work/gmsh/tutorials
MFEM_DIR=${HOME}/Work/mfem_reader
MFEM_BUILD_DIR=${MFEM_DIR}/build
CMDEX1=${MFEM_BUILD_DIR}/examples/ex1

MESH_LIST="1 3 4 7 8 11 13 14 16 19 20" 

(cd $MFEM_BUILD_DIR; make -j 12 ex1 ex4 ex39)
(cd $MFEM_DIR/data;
 gmsh compass.geo -3 -format msh4 -o compass41.msh
 gmsh compass.geo -3 -format msh2 -o compass22.msh
 gmsh cube-periodic.geo -3 -format msh4 -o cubeper41.msh
 gmsh cube-periodic.geo -3 -format msh2 -o cubeper22.msh
)

cd $GMSH_TUTO_DIR
for i in $MESH_LIST; do
    echo "t${i}.geo"
    gmsh t${i}.geo -3 -format msh41 -o t${i}_41.msh &
    gmsh t${i}.geo -3 -format msh22 -o t${i}_22.msh &
    wait
done

# Multipe checks
cd $GMSH_TUTO_DIR
for i in $MESH_LIST; do
    echo "t${i}.geo"
    $CMDEX1 -m t${i}_22.msh > pok22 || exit 1
    $CMDEX1 -m t${i}_41.msh > pok41 || exit 1
    (tail -n 10 pok22) > res22
    (tail -n 10 pok41) > res41
    diff -s res22 res41
done

cd $MFEM_BUILD_DIR/examples
echo compass.msh
(./ex39 -m ../../data/compass22.msh | tail -n 10) > res22
(./ex39 -m ../../data/compass41.msh | tail -n 10) > res41
diff -s res22 res41
cd $MFEM_BUILD_DIR/examples
echo cubeper.msh
./ex4 -m ../../data/cubeper22.msh
./ex4 -m ../../data/cubeper41.msh
#diff -s res22 res41


