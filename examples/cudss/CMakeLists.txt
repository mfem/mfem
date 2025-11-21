# A test CMakeLists for cuDSS solver class

set(CUDSSSOLVER_EXAMPLES_SRCS)

if (MFEM_USE_CUDA)
    set(MFEM_TEST_DEVICE "cuda")
endif()

if (MFEM_USE_MPI)
    list(APPEND CUDSSSOLVER_EXAMPLES_SRCS
        ex1p.cpp
        )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_cudsssolver" target, see below.
add_custom_target(test_cudss
  ${CMAKE_CTEST_COMMAND} -R cudss USES_TERMINAL)

# Add one executable per cpp file, adding "cudss_" as prefix. Sets
# "test_cudss" as a target that depends on the given examples.
set(PFX cudss_)
add_mfem_examples(CUDSSSOLVER_EXAMPLES_SRCS ${PFX} "" test_cudss)

# Testing.
# The cuDSS solver tests can be run separately using the target "test_cudss"
# which builds the examples and runs:
#   ctest -R cudsssolver
if (MFEM_ENABLE_TESTING)
  # Command line options for the tests.
  # Example 0: Test cuDSSSover on the simple Poisson problem
#   set(EX0P_COMMON_OPTS -d cuda -cudss)

  # Add the tests: one test per source file.
  foreach(SRC_FILE ${CUDSSSOLVER_EXAMPLES_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
    string(REPLACE ".cpp" "" TEST_NAME ${SRC_FILENAME})
    string(TOUPPER ${TEST_NAME} UP_TEST_NAME)
    set(TEST_NAME ${PFX}${TEST_NAME})

    set(THIS_TEST_OPTIONS "-d" "cuda" "-cudss")
    list(APPEND THIS_TEST_OPTIONS ${${UP_TEST_NAME}_TEST_OPTS})
    # message(STATUS "Test ${TEST_NAME} options: ${THIS_TEST_OPTIONS}")

    if ((${TEST_NAME} MATCHES ".*p$"))
      add_test(NAME ${TEST_NAME}_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
        ${MPIEXEC_POSTFLAGS})
    endif()
  endforeach()
endif()