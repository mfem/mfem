if (CMAKE_HOST_UNIX)
  find_program(ghost_make NAMES gmake make)
  if (ghost_make)
    execute_process(COMMAND "${ghost_make}" --version
      OUTPUT_VARIABLE make_version ERROR_VARIABLE make_error
      RESULT_VARIABLE make_result)
    if (NOT make_result EQUAL 0 OR NOT make_version MATCHES "GNU Make")
      set(ghost_make "")
    endif()
  endif()
endif()

if (ghost_make)
  set(make_directory "${BINARY_DIR}/ghost-make-regression")
  file(MAKE_DIRECTORY "${make_directory}")
  file(WRITE "${make_directory}/makefile"
    "include ${SOURCE_DIR}/tests/unit/makefile\n"
    "ifneq ($(filter %/ghost_penalty_failure.cpp,$(SOURCE_FILES)),)\n"
    "$(error Standalone failure harness is in shared test sources)\nendif\n"
    "ifeq ($(filter %/test_ghost_penalty.cpp,$(SOURCE_FILES)),)\n"
    "$(error Guarded numerical tests are missing from shared sources)\nendif\n"
    "ifneq ($(filter ghost_penalty_failure,$(UNIT_TESTS)),)\n"
    "$(error Failure harness cannot run as an ordinary unit test)\nendif\n"
    "ifneq ($(.DEFAULT_GOAL),all)\n"
    "$(error The default unit-test make goal changed)\nendif\n"
    "ifeq ($(MFEM_USE_LAPACK),YES)\n"
    "ifneq ($(GHOST_FAILURE_TEST),ghost_penalty_failure)\n"
    "$(error Missing separate failure-harness target)\nendif\nendif\n"
    ".PHONY: ghost-penalty-check\n"
    "ghost-penalty-check: fem/test_ghost_penalty.o $(GHOST_FAILURE_OBJ)\n"
    "ifeq ($(MFEM_USE_LAPACK),NO)\n"
    "\t$(CCC) $(MFEM_FLAGS) $(INCLUDES) -c "
    "$(SRC)fem/ghost_penalty_failure.cpp -o ghost_penalty_failure-check.o\n"
    "endif\n")
  set(mtop_directory "${BINARY_DIR}/ghost-mtop-regression")
  file(MAKE_DIRECTORY "${mtop_directory}")
  file(WRITE "${mtop_directory}/makefile"
    "include ${SOURCE_DIR}/miniapps/mtop/makefile\n"
    "ifneq ($(.DEFAULT_GOAL),all)\n"
    "$(error The default mtop make goal changed)\nendif\n"
    "ifeq ($(MFEM_USE_LAPACK),YES)\n"
    "ifneq ($(filter projection_ghost_penalty_serial,$(SEQ_MINIAPPS)),projection_ghost_penalty_serial)\n"
    "$(error Missing sequential ghost demo target)\nendif\n"
    "ifneq ($(filter projection_ghost_penalty_serial,$(MINIAPPS)),projection_ghost_penalty_serial)\n"
    "$(error Ghost demo is not built by default)\nendif\n"
    "ifeq ($(MFEM_USE_MPI),YES)\n"
    "ifneq ($(filter projection_ghost_penalty_parallel,$(PAR_MINIAPPS)),projection_ghost_penalty_parallel)\n"
    "$(error Missing parallel ghost demo target)\nendif\n"
    "ifneq ($(filter projection_ghost_penalty_parallel,$(MINIAPPS)),projection_ghost_penalty_parallel)\n"
    "$(error Parallel ghost demo is not built by default)\nendif\n"
    "else\n"
    "ifneq ($(filter projection_ghost_penalty_parallel,$(PAR_MINIAPPS) $(MINIAPPS)),)\n"
    "$(error Parallel ghost demo remains enabled without MPI)\nendif\nendif\n"
    "else\n"
    "ifneq ($(filter projection_ghost_penalty_serial projection_ghost_penalty_parallel,$(SEQ_MINIAPPS) $(PAR_MINIAPPS) $(MINIAPPS)),)\n"
    "$(error Ghost demo remains enabled without LAPACK)\nendif\nendif\n"
    ".PHONY: ghost-penalty-check\n"
    "ghost-penalty-check: ;\n")
else()
  message(STATUS "GNU Make unavailable; skipping legacy Make compile checks")
endif()

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
  file(READ "${BINARY_DIR}/CMakeFiles/TargetDirectories.txt" targets)
  if (targets MATCHES "/projection_ghost_penalty_parallel.dir")
    message(FATAL_ERROR "Parallel ghost demo remains enabled without MPI")
  endif()
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
  if (ghost_make)
    execute_process(
      COMMAND "${ghost_make}" -B -j2 -C "${make_directory}"
        "MFEM_DIR=${SOURCE_DIR}" "MFEM_BUILD_DIR=${BINARY_DIR}"
        "CCOPTS=" "MFEM_CXXFLAGS=-std=c++17 -O0" ghost-penalty-check
      RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
      TIMEOUT 90)
    if (NOT "${result}" STREQUAL "0")
      message(FATAL_ERROR
        "LAPACK=${lapack_enabled} Make compile failed: ${result}\n${output}\n${error}")
    endif()
    foreach(mpi_enabled NO YES)
      execute_process(
        COMMAND "${ghost_make}" -C "${mtop_directory}"
          "MFEM_DIR=${SOURCE_DIR}" "MFEM_BUILD_DIR=${BINARY_DIR}"
          "MFEM_USE_MPI=${mpi_enabled}" ghost-penalty-check
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
        TIMEOUT 30)
      if (NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
          "LAPACK=${lapack_enabled} MPI=${mpi_enabled} Make demo registration failed: ${result}\n${output}\n${error}")
      endif()
    endforeach()
  endif()
endforeach()
