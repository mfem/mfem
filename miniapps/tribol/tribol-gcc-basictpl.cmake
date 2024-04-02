#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(COMPILER_HOME "/usr" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_HOME}/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_HOME}/bin/g++" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran" CACHE PATH "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME             "/usr" CACHE PATH "")
set(MPI_C_COMPILER       "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER     "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER "${MPI_HOME}/bin/mpif90" CACHE PATH "")

set(MPIEXEC              "/usr/bin/mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-np" CACHE PATH "")

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------

set(BUILD_REDECOMP ON CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "${CMAKE_SOURCE_DIR}/.." CACHE PATH "")
set(AXOM_DIR "${TPL_ROOT}/axom" CACHE PATH "")
set(MFEM_DIR "${TPL_ROOT}/mfem/mfem" CACHE PATH "")


