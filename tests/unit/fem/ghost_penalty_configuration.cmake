foreach(lapack_enabled OFF ON OFF)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${BINARY_DIR}"
      "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
      -DMFEM_USE_MPI=OFF -DMFEM_USE_METIS=OFF
      -DMFEM_ENABLE_TESTING=ON -DMFEM_ENABLE_MINIAPPS=ON
      "-DMFEM_USE_LAPACK=${lapack_enabled}"
      "-DBLAS_LIBRARIES=${BLAS_LIBRARIES}"
      "-DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
    TIMEOUT 60)
  if (NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR
      "LAPACK=${lapack_enabled} configuration failed: ${result}\n${output}\n${error}")
  endif()

  file(STRINGS "${BINARY_DIR}/config/_config.hpp" lapack_definition
    REGEX "^#define MFEM_USE_LAPACK")
  file(READ "${BINARY_DIR}/tests/unit/CTestTestfile.cmake" unit_tests)
  file(READ "${BINARY_DIR}/miniapps/mtop/CTestTestfile.cmake" demo_tests)
  if (lapack_enabled)
    if (NOT lapack_definition OR NOT unit_tests MATCHES "ghost_penalty_serial"
        OR NOT demo_tests MATCHES "ghost_penalty_demo_serial")
      message(FATAL_ERROR "LAPACK did not automatically enable ghost penalties")
    endif()
  else()
    if (lapack_definition OR unit_tests MATCHES "ghost_penalty_"
        OR demo_tests MATCHES "ghost_penalty_")
      message(FATAL_ERROR "Ghost penalties remain enabled without LAPACK")
    endif()
  endif()
endforeach()
