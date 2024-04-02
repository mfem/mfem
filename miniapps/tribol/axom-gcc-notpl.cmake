#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(ENABLE_FORTRAN ON CACHE BOOL "")

set(COMPILER_HOME "/usr" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_HOME}/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_HOME}/bin/g++" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran" CACHE PATH "")

# Sidre requires conduit and hdf5, so disable it in this host-config
# Inlet and Klee both depend on Sidre, so disable them too
set(AXOM_ENABLE_SIDRE OFF CACHE BOOL "")
set(AXOM_ENABLE_INLET OFF CACHE BOOL "")
set(AXOM_ENABLE_KLEE OFF CACHE BOOL "")

set(BLT_EXE_LINKER_FLAGS " -Wl,-rpath,${COMPILER_HOME}/lib" CACHE STRING "Adds a missing libstdc++ rpath")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME             "/usr" CACHE PATH "")
set(MPI_C_COMPILER       "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER     "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER "${MPI_HOME}/bin/mpif90" CACHE PATH "")

set(MPIEXEC              "/usr/bin/mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n " CACHE PATH "")

#------------------------------------------------------------------------------
# Other
#------------------------------------------------------------------------------
set(CLANGFORMAT_EXECUTABLE "/usr/bin/clang-format" CACHE PATH "")

