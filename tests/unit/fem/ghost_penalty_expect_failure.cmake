if (MPIEXEC)
  execute_process(COMMAND ${MPIEXEC} ${NUMPROC_FLAG} 2 ${MPI_PREFLAGS}
    ${EXECUTABLE} ${SCENARIO} ${MPI_POSTFLAGS}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 45)
else()
  execute_process(COMMAND ${EXECUTABLE} ${SCENARIO}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 45)
endif()
if ("${result}" STREQUAL "0")
  message(FATAL_ERROR "Expected failure for ${SCENARIO}, but command succeeded")
endif()
string(FIND "${output}${error}" "${EXPECTED}" match)
if (match LESS 0)
  message(FATAL_ERROR "Missing diagnostic '${EXPECTED}'; result=${result}\n${output}\n${error}")
endif()
if ("${result}" MATCHES "[Tt]imeout|[Tt]ime.*out")
  message(FATAL_ERROR "Failure test hung: ${SCENARIO}")
endif()
