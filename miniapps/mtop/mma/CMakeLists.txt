cmake_minimum_required(VERSION 3.16)

# =============================================================================
# mma_mfem  --  MMA/GCMMA/SQ optimisers for MFEM
#
# Two invocation modes
# --------------------
#
# 1. Standalone (the common case):
#
#      cmake -B build -DMFEM_BUILD_DIR=/path/to/mfem/build
#      cmake --build build -j$(nproc)
#      bash run_tests.sh -b build
#
#    MFEM_BUILD_DIR must point to a directory that contains mfem.hpp and
#    libmfem.{a,so} (GNU-make MFEM build) or MFEMConfig.cmake (cmake build).
#
# 2. Sub-directory of the MFEM source tree (miniapp convention):
#
#      # In the MFEM top-level CMakeLists.txt:
#      add_subdirectory(miniapps/mtop/mma)
#
#    In this mode MFEM's own cmake targets (MFEM::MFEM or mfem) are already
#    defined by the parent project.  MFEM_BUILD_DIR is not needed and is
#    ignored.  The project() call is also skipped so the parent's compiler
#    settings and policies are inherited unchanged.
#
# Optional variables (standalone mode only)
# -----------------------------------------
#   CMAKE_BUILD_TYPE      Debug | Release | RelWithDebInfo  (default: Release)
#   MPI_CXX_COMPILER      Override the MPI C++ wrapper (e.g. mpicxx)
# =============================================================================

# Only call project() in standalone mode.
# When used as add_subdirectory() the parent already called project().
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(mma_mfem CXX)
endif()

# ---------------------------------------------------------------------------
# Build type default (standalone only; parent controls it in subdirectory mode)
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# MFEM discovery
# ---------------------------------------------------------------------------
# Check whether MFEM is already available from a parent project.
# This is the case when the CMakeLists.txt is included via add_subdirectory()
# inside the MFEM source tree.

if(TARGET MFEM::MFEM OR TARGET mfem)
  # ---- Sub-directory / parent-project mode ---------------------------------
  # MFEM targets are already defined.  Populate the plain variables so the
  # rest of this file can use them uniformly regardless of mode.
  set(_mma_standalone FALSE)
  if(TARGET MFEM::MFEM)
    get_target_property(MFEM_INCLUDE_DIRS MFEM::MFEM INTERFACE_INCLUDE_DIRECTORIES)
    set(MFEM_LIBRARIES MFEM::MFEM)
  else()
    get_target_property(MFEM_INCLUDE_DIRS mfem INTERFACE_INCLUDE_DIRECTORIES)
    set(MFEM_LIBRARIES mfem)
  endif()
  message(STATUS "mma_mfem: using MFEM from parent project (${MFEM_LIBRARIES})")
  message(STATUS "mma_mfem: MFEM_INCLUDE_DIRS = ${MFEM_INCLUDE_DIRS}")

else()
  # ---- Standalone mode: locate MFEM ourselves ------------------------------
  set(_mma_standalone TRUE)

  if(NOT DEFINED MFEM_BUILD_DIR)
    message(FATAL_ERROR
      "MFEM_BUILD_DIR is not set.\n"
      "Please configure with:\n"
      "  cmake -B build -DMFEM_BUILD_DIR=/path/to/mfem/build\n"
      "The directory must contain mfem.hpp and libmfem.{a,so}.\n"
      "If you used the Makefile, it passes this automatically.\n"
      "\n"
      "If you are including this CMakeLists.txt via add_subdirectory() from\n"
      "within the MFEM source tree, make sure the MFEM targets (MFEM::MFEM\n"
      "or mfem) are defined before this subdirectory is added.")
  endif()

  get_filename_component(_mfem_abs "${MFEM_BUILD_DIR}" ABSOLUTE
    BASE_DIR "${CMAKE_SOURCE_DIR}")
  message(STATUS "MFEM_BUILD_DIR resolved to: ${_mfem_abs}")

  set(MFEM_DIR "${_mfem_abs}" CACHE PATH "MFEM build or install directory" FORCE)

  # Strategy A: cmake-built MFEM (provides MFEMConfig.cmake)
  find_package(MFEM CONFIG QUIET
    HINTS "${_mfem_abs}"
          "${_mfem_abs}/lib/cmake/mfem"
          "${_mfem_abs}/share/mfem")

  if(NOT MFEM_FOUND)
    foreach(_cfg IN ITEMS
        "${_mfem_abs}/MFEMConfig.cmake"
        "${_mfem_abs}/mfem.cmake"
        "${_mfem_abs}/lib/cmake/mfem/MFEMConfig.cmake"
        "${_mfem_abs}/share/mfem/MFEMConfig.cmake")
      if(EXISTS "${_cfg}")
        include("${_cfg}")
        set(MFEM_FOUND TRUE)
        break()
      endif()
    endforeach()
  endif()

  # Strategy B: GNU-make-built MFEM (provides mfem.hpp + libmfem.*)
  if(NOT MFEM_FOUND)
    set(_mfem_inc "")
    foreach(_d IN ITEMS
        "${_mfem_abs}"
        "${_mfem_abs}/include"
        "${_mfem_abs}/InstallHeaders")
      if(EXISTS "${_d}/mfem.hpp")
        set(_mfem_inc "${_d}")
        break()
      endif()
    endforeach()

    set(_mfem_lib "")
    foreach(_d IN ITEMS "${_mfem_abs}" "${_mfem_abs}/lib")
      foreach(_name IN ITEMS "libmfem.a" "libmfem.so" "libmfem.dylib")
        if(EXISTS "${_d}/${_name}")
          set(_mfem_lib "${_d}/${_name}")
          break()
        endif()
      endforeach()
      if(_mfem_lib)
        break()
      endif()
    endforeach()

    if(_mfem_inc AND _mfem_lib)
      set(MFEM_FOUND TRUE)
      set(MFEM_INCLUDE_DIRS "${_mfem_inc}")
      if(_mfem_lib MATCHES "\\.a$")
        add_library(mfem_imported STATIC IMPORTED)
      else()
        add_library(mfem_imported SHARED IMPORTED)
      endif()
      set_target_properties(mfem_imported PROPERTIES
        IMPORTED_LOCATION             "${_mfem_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${_mfem_inc}")
      add_library(mfem ALIAS mfem_imported)
      set(MFEM_LIBRARIES mfem_imported)
      message(STATUS "MFEM found (GNU-make build):")
      message(STATUS "  include : ${_mfem_inc}")
      message(STATUS "  library : ${_mfem_lib}")
    endif()
  endif()

  if(NOT MFEM_FOUND)
    message(FATAL_ERROR
      "Could not find MFEM in '${_mfem_abs}'.\n"
      "Diagnose with:\n"
      "  find ${_mfem_abs} -name 'mfem.hpp'\n"
      "  find ${_mfem_abs} -name 'libmfem*'\n"
      "  find ${_mfem_abs} -name '*.cmake' | grep -iv CMakeFiles")
  endif()

  # Normalise: cmake-config builds export MFEM::MFEM but may not set plain vars.
  if(TARGET MFEM::MFEM AND NOT MFEM_INCLUDE_DIRS)
    get_target_property(MFEM_INCLUDE_DIRS MFEM::MFEM INTERFACE_INCLUDE_DIRECTORIES)
    set(MFEM_LIBRARIES MFEM::MFEM)
  endif()

  message(STATUS "MFEM_INCLUDE_DIRS : ${MFEM_INCLUDE_DIRS}")
  message(STATUS "MFEM_LIBRARIES    : ${MFEM_LIBRARIES}")

endif()  # MFEM target detection

# ---------------------------------------------------------------------------
# Dependencies: LAPACK, BLAS  (MPI is inherited from MFEM, not found here)
# ---------------------------------------------------------------------------
find_package(LAPACK REQUIRED)
find_package(BLAS   REQUIRED)

# ---------------------------------------------------------------------------
# Core library  (mma_mfem_lib)
# ---------------------------------------------------------------------------
# Everything links against this target.  It exposes MMA_MFEM.hpp publicly
# so that test executables only need to link mma_mfem_lib without repeating
# the include/link flags.

add_library(mma_mfem_lib STATIC MMA_MFEM.cpp)
target_sources(mma_mfem_lib PRIVATE LatentMirrorOptimizer.cpp)


# ── Adding extra library source files ────────────────────────────────────────
# Add the .cpp file with target_sources(); the matching .hpp is found
# automatically because CMAKE_CURRENT_SOURCE_DIR is already on the include path.
#
# Same directory:
#   target_sources(mma_mfem_lib PRIVATE MyOptimizer.cpp)
#
# Subdirectory (also expose the header path to consumers):
#   target_include_directories(mma_mfem_lib PUBLIC
#     "${CMAKE_CURRENT_SOURCE_DIR}/mysubdir")
#   target_sources(mma_mfem_lib PRIVATE mysubdir/MyOptimizer.cpp)
#
# ─────────────────────────────────────────────────────────────────────────────

target_include_directories(mma_mfem_lib PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${MFEM_INCLUDE_DIRS}")

target_link_libraries(mma_mfem_lib PUBLIC
  ${MFEM_LIBRARIES}
  LAPACK::LAPACK
  BLAS::BLAS)

target_compile_features(mma_mfem_lib PUBLIC cxx_std_14)

set_target_properties(mma_mfem_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# Forward all compile flags and definitions that MFEM requires, including
# -DMFEM_USE_MPI when MFEM was built with MPI.  This is the authoritative
# source: the parallel optimizer classes (#ifdef MFEM_USE_MPI) compile iff
# MFEM itself was built with MPI — no separate find_package(MPI) needed.
if(MFEM_COMPILE_FLAGS)
  target_compile_options(mma_mfem_lib PUBLIC ${MFEM_COMPILE_FLAGS})
endif()
if(MFEM_DEFINITIONS)
  target_compile_definitions(mma_mfem_lib PUBLIC ${MFEM_DEFINITIONS})
endif()

# Report MPI status based on MFEM_DEFINITIONS so the message is reliable.
if("MFEM_USE_MPI" IN_LIST MFEM_DEFINITIONS)
  message(STATUS "mma_mfem: MFEM_USE_MPI set — parallel optimizers enabled")
else()
  message(STATUS "mma_mfem: MFEM_USE_MPI not set — serial optimizers only")
endif()

# ---------------------------------------------------------------------------
# Helper: register one test executable
# ---------------------------------------------------------------------------
#
# Usage:
#   mma_add_test(NAME <name> [PARALLEL] [ARGS arg1 arg2 ...] [LABELS lbl ...])
#
# <name> must match the .cpp filename exactly (without the extension).
# The macro creates the executable, links it against mma_mfem_lib, and
# registers CTest entries:
#   <name>_serial       — runs the binary directly (always created)
#   <name>_parallel_4   — runs under mpirun -np 4 (only when PARALLEL is
#                         specified AND MPIEXEC_EXECUTABLE is available)
#
# Keyword arguments:
#   PARALLEL            — also register the 4-rank CTest entry
#   ARGS  arg …         — extra arguments appended to both CTest commands
#                         (e.g. ARGS --device cpu)
#   LABELS lbl …        — CTest labels, e.g. LABELS large for slow tests
#                         selectable with: ctest -L large
#
# ── Adding a new test ─────────────────────────────────────────────────────────
#
# 1. Write test_my_feature.cpp in this directory.  main() must return 0 on
#    success and non-zero on failure.
#
# 2. Add one line in the "Test targets" section below:
#
#      mma_add_test(NAME test_my_feature)              # serial-only test
#      mma_add_test(NAME test_my_feature PARALLEL)     # serial + 4-rank MPI
#
#    To pass a fixed argument to the test binary (e.g. a flag):
#      mma_add_test(NAME test_my_feature ARGS --verbose)
#
#    To mark a test as slow so it can be skipped with  ctest -LE large:
#      mma_add_test(NAME test_my_feature PARALLEL LABELS large)
#
# 3. Re-run cmake (or just make; CMake re-checks automatically).
#
# The macro silently skips entries whose .cpp file is absent, so it is safe
# to commit a CMakeLists.txt entry before the source file exists.
#
# ---------------------------------------------------------------------------

function(mma_add_test)
  cmake_parse_arguments(T "PARALLEL" "NAME" "ARGS;LABELS" ${ARGN})

  # Silently skip targets whose source file is absent.
  # This makes the CMakeLists.txt safe when used as an MFEM subdirectory
  # where only the core files (MMA_MFEM.cpp/.hpp) are distributed.
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${T_NAME}.cpp")
    message(STATUS "mma_mfem: skipping ${T_NAME} (source not found)")
    return()
  endif()

  add_executable(${T_NAME} "${T_NAME}.cpp")
  target_link_libraries(${T_NAME} PRIVATE mma_mfem_lib)

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
# Test targets
# ---------------------------------------------------------------------------
# Add new tests here with mma_add_test() — see the helper documentation above.
# ---------------------------------------------------------------------------
enable_testing()

# ── Core MMA / GCMMA ---------------------------------------------------------

mma_add_test(NAME test_mma_serial        PARALLEL)
mma_add_test(NAME test_mma_parallel      PARALLEL)
mma_add_test(NAME test_mma_unconstrained PARALLEL)
mma_add_test(NAME test_gcmma             PARALLEL)
mma_add_test(NAME test_gcmma_callback    PARALLEL)
mma_add_test(NAME test_zero_ranks        PARALLEL)
mma_add_test(NAME test_overconstrained   PARALLEL)
mma_add_test(NAME test_nonconvex         PARALLEL)
mma_add_test(NAME test_equalities                    PARALLEL)
mma_add_test(NAME test_relaxed_equalities_serial)
mma_add_test(NAME test_relaxed_equalities_parallel   PARALLEL)

# Large nonconvex run: opt-in via ctest -L large  (slow, 100k-1M DOF)
if(MPIEXEC_EXECUTABLE)
  add_test(
    NAME test_nonconvex_large_4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_nonconvex>
            --large ${MPIEXEC_POSTFLAGS})
  set_tests_properties(test_nonconvex_large_4 PROPERTIES
    TIMEOUT 600 LABELS "large")
endif()

# Large SQ nonconvex run: same problem suite with SQOptimizerParallel
if(MPIEXEC_EXECUTABLE)
  add_test(
    NAME test_sq_nonconvex_large_4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_sq_nonconvex>
            --large ${MPIEXEC_POSTFLAGS})
  set_tests_properties(test_sq_nonconvex_large_4 PROPERTIES
    TIMEOUT 600 LABELS "large")
endif()

# ---- Device test (CPU always; GPU optional) ---------------------------------

mma_add_test(NAME test_mma_device ARGS --device cpu)

find_program(NVIDIA_SMI nvidia-smi)
find_program(ROCM_INFO  rocm_info)
if(NVIDIA_SMI OR ROCM_INFO OR DEFINED ENV{CUDA_VISIBLE_DEVICES})
  set(_gpu "cuda")
  if(ROCM_INFO)
    set(_gpu "hip")
  endif()
  add_test(NAME test_mma_device_gpu COMMAND test_mma_device --device ${_gpu})
  if(MPIEXEC_EXECUTABLE)
    add_test(
      NAME test_mma_device_gpu_parallel_4
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
              ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mma_device>
              --device ${_gpu} ${MPIEXEC_POSTFLAGS})
  endif()
endif()

# ---- SQ approximation -------------------------------------------------------

mma_add_test(NAME test_sq                PARALLEL)
mma_add_test(NAME test_sq_serial         PARALLEL)
mma_add_test(NAME test_sq_unconstrained  PARALLEL)
mma_add_test(NAME test_sq_gcmma          PARALLEL)
mma_add_test(NAME test_sq_gcmma_callback PARALLEL)
mma_add_test(NAME test_sq_overconstrained PARALLEL)
mma_add_test(NAME test_sq_zero_ranks     PARALLEL)
mma_add_test(NAME test_sq_nonconvex      PARALLEL)
mma_add_test(NAME test_sq_equalities     PARALLEL)
mma_add_test(NAME test_sq_parallel       PARALLEL)

mma_add_test(NAME test_sq_device ARGS --device cpu)

# ---- LMG optimizer ---------------------------------------------------------

mma_add_test(NAME test_lmg_serial) #serial test
mma_add_test(NAME test_lmg_parallel  PARALLEL) #parallel test



# ---- Unit tests (internal helpers, no MPI parallel runs needed) -------------
# These files are part of the standalone development tree but are not
# distributed with the MFEM source.  Guard every target with EXISTS so the
# CMakeLists.txt is safe when used as an MFEM subdirectory.

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_solvedense.cpp")
  mma_add_test(NAME test_solvedense)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_packfival.cpp")
  mma_add_test(NAME test_packfival)
endif()

# ---- Debug / development aid (excluded from default ctest run) --------------

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_eq_debug.cpp")
  add_executable(test_eq_debug test_eq_debug.cpp)
  target_link_libraries(test_eq_debug PRIVATE mma_mfem_lib)
  add_test(NAME test_eq_debug_serial COMMAND test_eq_debug)
  set_tests_properties(test_eq_debug_serial PROPERTIES LABELS "debug")
endif()

# ---- Legacy combined test (kept for backward compatibility) -----------------

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_mma_mfem.cpp")
  add_executable(test_mma_mfem test_mma_mfem.cpp)
  target_link_libraries(test_mma_mfem PRIVATE mma_mfem_lib)
  add_test(NAME test_mma_mfem_legacy COMMAND test_mma_mfem)
  set_tests_properties(test_mma_mfem_legacy PROPERTIES LABELS "legacy")
endif()

# ---- Legacy MMA.cpp (original non-MFEM implementation) ---------------------
# Built only to verify it compiles; not linked into any test.

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MMA.cpp")
  add_library(mma_legacy STATIC MMA.cpp)
  target_include_directories(mma_legacy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(mma_legacy PUBLIC cxx_std_14)
endif()

# ---------------------------------------------------------------------------
# Installation (miniapp convention)
# ---------------------------------------------------------------------------
install(TARGETS mma_mfem_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES MMA_MFEM.hpp MMA_MFEM.cpp
  DESTINATION include/mma_mfem)

# ---------------------------------------------------------------------------
# Summary (standalone mode only; suppress noise in parent builds)
# ---------------------------------------------------------------------------
if(_mma_standalone)
  message(STATUS "")
  message(STATUS "=== mma_mfem configuration ===")
  message(STATUS "  Source dir  : ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "  MFEM        : ${_mfem_abs}")
  message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
  message(STATUS "  MPI         : ${MPI_CXX_COMPILER}")
  message(STATUS "")
  message(STATUS "Build commands:")
  message(STATUS "  cmake --build build -j\$(nproc)")
  message(STATUS "  cmake --build build --target test   # CTest")
  message(STATUS "  bash run_tests.sh -b build          # human-readable runner")
endif()
