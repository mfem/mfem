#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "gfortran" CACHE PATH "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_C_COMPILER       "mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER     "mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER "mpif90" CACHE PATH "")

set(MPIEXEC              "mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-np" CACHE PATH "")

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------

set(ENABLE_FORTRAN OFF CACHE BOOL "")

set(TRIBOL_ENABLE_TESTS OFF CACHE BOOL "")
set(TRIBOL_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(TRIBOL_ENABLE_DOCS OFF CACHE BOOL "")

set(BUILD_REDECOMP ON CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "${CMAKE_SOURCE_DIR}/.." CACHE PATH "")
set(AXOM_DIR "${TPL_ROOT}/axom" CACHE PATH "")
set(MFEM_DIR "${TPL_ROOT}/mfem/mfem" CACHE PATH "")


