#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "gfortran" CACHE PATH "")

#set(BLT_EXE_LINKER_FLAGS " -Wl,-rpath,${COMPILER_HOME}/lib" CACHE STRING "Adds a missing libstdc++ rpath")

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

# Sidre requires conduit and hdf5, so disable it in this host-config
# Inlet and Klee both depend on Sidre, so disable them too
set(AXOM_ENABLE_SIDRE OFF CACHE BOOL "")
set(AXOM_ENABLE_INLET OFF CACHE BOOL "")
set(AXOM_ENABLE_KLEE OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# Other
#------------------------------------------------------------------------------
#set(CLANGFORMAT_EXECUTABLE "clang-format" CACHE PATH "")

