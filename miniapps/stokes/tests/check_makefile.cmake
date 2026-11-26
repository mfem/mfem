cmake_minimum_required(VERSION 3.12)
if(NOT MAKE_EXECUTABLE)
  message(FATAL_ERROR "A GNU make executable is required")
endif()
get_filename_component(package "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(mfem "${package}/../.." ABSOLUTE)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef identifier)
set(workspace "${CMAKE_CURRENT_BINARY_DIR}/stokes-makefile-${identifier}")
file(MAKE_DIRECTORY "${workspace}/mfem/config" "${workspace}/objects" "${workspace}/install/share/mfem")
file(WRITE "${workspace}/mfem/libmfem.a" "")
set(configuration
"MFEM_USE_MPI = YES
MFEM_USE_DOUBLE = YES
MFEM_USE_MUMPS = YES
MFEM_CXX = c++
MFEM_FLAGS = -std=c++17
MFEM_LIB_FILE = ${workspace}/mfem/libmfem.a
MFEM_LIBS = -lmfem
MFEM_TEST_MK = ${mfem}/config/test.mk
MFEM_MPIEXEC = mpiexec
MFEM_MPIEXEC_NP = -n
MFEM_MPI_NP = 2
")
file(WRITE "${workspace}/mfem/config/config.mk" "${configuration}")
file(WRITE "${workspace}/install/share/mfem/config.mk" "${configuration}")

function(run_make expected)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env "MAKEFLAGS=" "${MAKE_EXECUTABLE}"
    --no-print-directory -f "${package}/makefile"
    "MFEM_DIR=${mfem}" "MFEM_BUILD_DIR=${workspace}/mfem"
    "MFEM_INSTALL_DIR=${workspace}/absent" ${ARGN}
    WORKING_DIRECTORY "${workspace}/objects"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 30)
  set(text "${output}${errors}")
  if(expected STREQUAL "success")
    if(NOT "${result}" STREQUAL "0")
      message(FATAL_ERROR "make ${ARGN} failed: ${result}\n${text}")
    endif()
  elseif("${result}" STREQUAL "0" OR NOT "${result}" MATCHES "^[0-9]+$" OR NOT text MATCHES "${expected}")
    message(FATAL_ERROR "make ${ARGN} did not reject the configuration: ${result}\n${text}")
  endif()
  set(make_output "${text}" PARENT_SCOPE)
endfunction()

run_make(success -n all)
foreach(application stokes_pspg flow_example channel_flow refine_channel_serial stokes_mms
    stokes_adjoint_example steady_navier_stokes_setup steady_navier_stokes_workflow
    steady_navier_stokes_gmg_benchmark steady_navier_stokes stokes_brinkman stokes_brinkman_linear_example)
  if(NOT make_output MATCHES " -o ${application} ")
    message(FATAL_ERROR "Missing make application: ${application}")
  endif()
endforeach()
string(REGEX MATCHALL " -c [^\n]+/stokes\\.cpp " core_compilations "${make_output}")
list(LENGTH core_compilations count)
if(NOT count EQUAL 1)
  message(FATAL_ERROR "make must compile the solver only once, found ${count} compilations")
endif()
if(NOT make_output MATCHES "libstokes_example_support\\.a libstokes_core\\.a -o stokes_mms ")
  message(FATAL_ERROR "Manufactured-solution archive order is incorrect")
endif()
run_make(success -n all MFEM_USE_MUMPS=NO)
if(make_output MATCHES " -o (steady_navier_stokes|steady_navier_stokes_gmg_benchmark|stokes_brinkman|stokes_brinkman_linear_example) ")
  message(FATAL_ERROR "MUMPS-only applications were enabled without MUMPS")
endif()
run_make("requires MFEM_USE_MUMPS=YES" -n steady_navier_stokes MFEM_USE_MUMPS=NO)
foreach(disabled MFEM_USE_MPI MFEM_USE_DOUBLE)
  run_make(success -n all "${disabled}=NO")
  if(NOT make_output MATCHES "Skipping stokes" OR make_output MATCHES " -c ")
    message(FATAL_ERROR "Unsupported configuration did not skip the package")
  endif()
  run_make("requires MFEM_USE_MPI=YES and MFEM_USE_DOUBLE=YES" -n flow_example "${disabled}=NO")
endforeach()
run_make("MFEM is not configured" -n all "MFEM_BUILD_DIR=${workspace}/absent")
run_make(success -n flow_example "MFEM_BUILD_DIR=${workspace}/absent" "MFEM_INSTALL_DIR=${workspace}/install")
run_make(success -n test-par)
if(NOT make_output MATCHES "--convection --drag 2" OR NOT make_output MATCHES "--solver fgmres --no-paraview")
  message(FATAL_ERROR "Missing make smoke checks")
endif()
file(WRITE "${workspace}/objects/stokes.o" "generated object")
file(WRITE "${workspace}/objects/libstokes_core.a" "generated archive")
file(WRITE "${workspace}/objects/flow_example" "generated application")
file(WRITE "${workspace}/objects/preserve.txt" "user output")
run_make(success clean "MFEM_BUILD_DIR=${workspace}/absent")
foreach(product stokes.o libstokes_core.a flow_example)
  if(EXISTS "${workspace}/objects/${product}")
    message(FATAL_ERROR "make clean left ${product}")
  endif()
endforeach()
if(NOT EXISTS "${workspace}/objects/preserve.txt")
  message(FATAL_ERROR "make clean removed unrelated output")
endif()
file(REMOVE_RECURSE "${workspace}")
message(STATUS "GNU make package build rules passed")
