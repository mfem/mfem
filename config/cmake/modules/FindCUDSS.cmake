if (NOT cudss_DIR AND CUDSS_DIR)
  set(cudss_DIR ${CUDSS_DIR}/lib/cmake/cudss)
endif()
message(STATUS "Looking for CUDSS ...")
message(STATUS "   in CUDSS_DIR = ${CUDSS_DIR}")
message(STATUS "      cudss_DIR = ${cudss_DIR}")
find_package(cudss)
set(CUDSS_FOUND ${cudss_FOUND})
set(CUDSS_LIBRARIES "cudss")
if (CUDSS_FOUND)
  message(STATUS
    "Found CUDSS target: ${CUDSS_LIBRARIES} (version: ${cudss_VERSION})")
else()
  set(msg STATUS)
  if (CUDSS_FIND_REQUIRED)
    set(msg FATAL_ERROR)
  endif()
  message(${msg}
    "CUDSS not found. Please set CUDSS_DIR to the install prefix.")
endif()

get_target_property(CUDSS_LIBRARY_LOCATION cudss LOCATION)
get_filename_component(CUDSS_LIBRARY_DIR ${CUDSS_LIBRARY_LOCATION} DIRECTORY)

# Set the full name of the cuDSS threading library if OpenMP is enabled.
# The threading layer library (libcudss_mtlayer_gomp.so) is located under the
# cuDSS library directory by default.
if (MFEM_USE_OPENMP)
  find_file(
    CUDSS_THREADING_LIB
    NAMES libcudss_mtlayer_gomp.so
    PATHS ${CUDSS_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  if (NOT DEFINED MFEM_CUDSS_THREADING_LIB AND CUDSS_THREADING_LIB)
    set(MFEM_CUDSS_THREADING_LIB "${CUDSS_THREADING_LIB}")
  endif()
  message(STATUS "CUDSS threading layer library: ${MFEM_CUDSS_THREADING_LIB}")
endif()

# Set the full name of the cuDSS communication library if MFEM use OpenMPI.
# The communication layer library (libcudss_commlayer_mpi.so) is located under the
# cuDSS library directory by default.
# The communication layer library is used pre-built communication layers for OpenMPI 
# by default.
if (MFEM_USE_MPI)
  find_file(
    CUDSS_COMM_LIB
    NAMES libcudss_commlayer_openmpi.so
    PATHS ${CUDSS_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  if (NOT DEFINED MFEM_CUDSS_COMM_LIB AND CUDSS_COMM_LIB)
    set(MFEM_CUDSS_COMM_LIB "${CUDSS_COMM_LIB}")
  endif()
  message(STATUS "CUDSS communication layer library: ${MFEM_CUDSS_COMM_LIB}")
endif()
