cmake_minimum_required(VERSION 3.16)

# =============================================================================
# mtop opt-interface  --  OptimProblem abstraction + MMA OptimizationSolver
#
# Header-only library (opt_prob.hpp, MMAOptSolver.hpp) plus its unit tests.
# This mirrors the miniapps/mtop/mma test harness but lives entirely in this
# directory; the mma directory is not modified.  The MMA optimiser is consumed
# through the sibling `mma_mfem_lib` target, which publicly exposes both MFEM
# and the MMA headers (MMA_MFEM.hpp).
#
# Two invocation modes (same as mma/CMakeLists.txt)
# -------------------------------------------------
# 1. Sub-directory of the MFEM source tree (miniapp convention):
#      # In miniapps/CMakeLists.txt, AFTER mtop/mma:
#      add_subdirectory(mtop/opt-interface)
#    `mma_mfem_lib` (and MFEM) are already defined by the parent.
#
# 2. Standalone:
#      cmake -B build -DMFEM_BUILD_DIR=/path/to/mfem/build
#    The sibling mma/CMakeLists.txt is pulled in to provide `mma_mfem_lib`
#    (it performs MFEM discovery from MFEM_BUILD_DIR).
# =============================================================================

# Only call project() in standalone mode (parent already called it otherwise).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(mtop_opt_interface CXX)
endif()

# ---------------------------------------------------------------------------
# MMA library dependency
# ---------------------------------------------------------------------------
# In subdirectory mode the parent already added mtop/mma, so `mma_mfem_lib`
# exists.  In standalone mode, pull in the sibling directory to build it (that
# CMakeLists.txt also handles MFEM discovery via MFEM_BUILD_DIR).
if(NOT TARGET mma_mfem_lib)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../mma"
                   "${CMAKE_CURRENT_BINARY_DIR}/mma")
endif()

# ---------------------------------------------------------------------------
# Helper: register one test executable  (mirror of mma_add_test)
# ---------------------------------------------------------------------------
# Usage:  opt_add_test(NAME <name> [PARALLEL] [ARGS ...] [LABELS ...])
# <name> must match the .cpp filename (without extension).  Executables link
# mma_mfem_lib (MFEM + MMA headers) and see this directory for opt_prob.hpp /
# MMAOptSolver.hpp.  Targets whose source is absent are skipped silently.
function(opt_add_test)
  cmake_parse_arguments(T "PARALLEL" "NAME" "ARGS;LABELS" ${ARGN})

  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${T_NAME}.cpp")
    message(STATUS "opt-interface: skipping ${T_NAME} (source not found)")
    return()
  endif()

  add_executable(${T_NAME} "${T_NAME}.cpp")
  target_link_libraries(${T_NAME} PRIVATE mma_mfem_lib)
  target_include_directories(${T_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

  add_test(NAME "${T_NAME}_serial" COMMAND ${T_NAME} ${T_ARGS})
  if(T_LABELS)
    set_tests_properties("${T_NAME}_serial" PROPERTIES LABELS "${T_LABELS}")
  endif()

  if(T_PARALLEL AND MPIEXEC_EXECUTABLE)
    add_test(
      NAME "${T_NAME}_parallel_4"
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
              ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${T_NAME}>
              ${T_ARGS} ${MPIEXEC_POSTFLAGS})
    if(T_LABELS)
      set_tests_properties("${T_NAME}_parallel_4" PROPERTIES LABELS "${T_LABELS}")
    endif()
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

opt_add_test(NAME test_opt_prob)                # OptimProblem abstraction (no MMA)
opt_add_test(NAME test_opt_mma)                 # MMA/GCMMA -> OptimizationSolver bridge
opt_add_test(NAME test_opt_mma_parallel PARALLEL)  # parallel MMA bridge (MPI)
opt_add_test(NAME test_opt_ccsa)                # CCSA/Bregman -> OptimizationSolver bridge
opt_add_test(NAME test_opt_ccsa_parallel PARALLEL) # parallel CCSA bridge (MPI)
opt_add_test(NAME test_opt_sq)                  # SQ -> OptimizationSolver bridge
opt_add_test(NAME test_opt_sq_parallel PARALLEL)   # parallel SQ bridge (MPI)

# Cross-method comparison harness (built, run manually — not a ctest).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/compare_opt.cpp")
  add_executable(compare_opt compare_opt.cpp)
  target_link_libraries(compare_opt PRIVATE mma_mfem_lib)
  target_include_directories(compare_opt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# Installation (header-only)
# ---------------------------------------------------------------------------
install(FILES opt_prob.hpp MMAOptSolver.hpp
  DESTINATION include/mtop_opt_interface)
