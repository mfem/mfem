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

if(CUDSS_FOUND AND TARGET cudss)
  get_target_property(CUDSS_LIBRARY_LOCATION cudss IMPORTED_LOCATION)
  if(NOT CUDSS_LIBRARY_LOCATION)
    get_target_property(CUDSS_LIBRARY_LOCATION cudss IMPORTED_LOCATION_RELEASE)
  endif()
  if(CUDSS_LIBRARY_LOCATION)
    get_filename_component(CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY_LOCATION}" DIRECTORY)
  else()
    message(WARNING "Could not determine the location of the cuDSS library.")
  endif()
else()
    message(WARNING "cuDSS target not available; cannot determine library directory.")
endif()

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

# Set the full name of the cuDSS communication layer library when MFEM is built
# with MPI. cuDSS requires a communication layer that matches the MPI
# implementation MFEM is linked against, so we must not bake in a layer for the
# wrong implementation. NVIDIA ships a prebuilt layer for Open MPI
# (libcudss_commlayer_openmpi.so), which we locate automatically. For other
# implementations (e.g. MPICH) no prebuilt layer is shipped: a matching layer
# must be built (see cuDSS's cudss_build_commlayer.sh) and its path supplied via
# the CUDSS_COMM_LIB environment variable at runtime. These layers live under
# the cuDSS library directory by default.
if (MFEM_USE_MPI)
  # cuDSS requires a communication layer matching the MPI implementation. Use
  # the vendor detected during MPI setup (MFEM_MPI_IS_MPICH). We only deviate
  # from the prebuilt Open MPI layer when the MPI is positively identified as
  # MPICH-family, so the previous behavior is preserved when the vendor cannot
  # be determined.
  if (MFEM_MPI_IS_MPICH)
    # MPICH-family MPI: NVIDIA does not ship a prebuilt layer. Use a user-built
    # layer if one has been installed alongside cuDSS; otherwise leave
    # MFEM_CUDSS_COMM_LIB unset and rely on the CUDSS_COMM_LIB environment
    # variable at runtime.
    find_file(
      CUDSS_COMM_LIB
      NAMES libcudss_commlayer_mpich.so libcudss_commlayer_mpi.so
      PATHS ${CUDSS_LIBRARY_DIR}
      NO_DEFAULT_PATH
    )
  else()
    # Open MPI (or an undetermined implementation): use the prebuilt Open MPI
    # communication layer shipped with cuDSS.
    find_file(
      CUDSS_COMM_LIB
      NAMES libcudss_commlayer_openmpi.so
      PATHS ${CUDSS_LIBRARY_DIR}
      NO_DEFAULT_PATH
    )
  endif()

  if (NOT DEFINED MFEM_CUDSS_COMM_LIB AND CUDSS_COMM_LIB)
    set(MFEM_CUDSS_COMM_LIB "${CUDSS_COMM_LIB}")
  endif()

  if (MFEM_CUDSS_COMM_LIB)
    message(STATUS "CUDSS communication layer library: ${MFEM_CUDSS_COMM_LIB}")
  else()
    message(STATUS
      "CUDSS communication layer library not found for the detected MPI "
      "implementation; set the CUDSS_COMM_LIB environment variable at runtime to "
      "a layer matching your MPI (see cuDSS's cudss_build_commlayer.sh).")
  endif()
endif()
