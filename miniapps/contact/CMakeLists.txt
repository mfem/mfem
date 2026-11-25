# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if (MFEM_USE_TRIBOL AND MFEM_USE_MPI)
  list(APPEND CONTACT_SOURCES
  coulombsolver.cpp
  ip.cpp
  optcontactproblem.cpp
  elastoperator.cpp
  solver_utils.cpp)
  list(APPEND CONTACT_HEADERS
  coulomblaw.hpp
  contactsolver.hpp
  coulombsolver.hpp
  ip.hpp
  optcontactproblem.hpp
  elastoperator.hpp
  solver_utils.hpp)

  convert_filenames_to_full_paths(CONTACT_SOURCES)
  convert_filenames_to_full_paths(CONTACT_HEADERS)

  set(CONTACT_COMMON_FILES
      EXTRA_SOURCES ${CONTACT_SOURCES}
      EXTRA_HEADERS ${CONTACT_HEADERS})

  add_mfem_miniapp(contact
    MAIN contact.cpp
    ${CONTACT_COMMON_FILES}
    LIBRARIES mfem)

# The way the libraries are included, by default, in the command line is based
# on the idea that MFEM depends on the other libraries. Here, the setting is
# such that Tribol, in fact, depends on MFEM, while MFEM does not depend on
# Tribol. This addresses that issue.
  target_link_libraries(contact PUBLIC ${TPL_LIBRARIES} mfem)

  if (MFEM_ENABLE_TESTING)
    add_executable(contact_coulomb_law EXCLUDE_FROM_ALL test_coulomblaw.cpp)
    target_link_libraries(contact_coulomb_law PRIVATE mfem)
    add_dependencies(contact contact_coulomb_law)
    add_test(NAME contact_coulomb_law COMMAND $<TARGET_FILE:contact_coulomb_law>)

    # Share MPI launch options and constrain threaded TPLs in the contact tests.
    function(add_contact_test name ranks expected_state)
      set(_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:contact> ${MPIEXEC_POSTFLAGS}
        -sr 0 -pr 0 -no-vis -no-paraview ${ARGN})
      add_test(NAME ${name} COMMAND ${CMAKE_COMMAND}
        "-DCONTACT_COMMAND=${_command}" "-DEXPECT_STATE=${expected_state}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_contact.cmake)
      set_tests_properties(${name} PROPERTIES TIMEOUT 180 PROCESSORS ${ranks})
    endfunction()

    add_contact_test(contact_ironing_frictionless 2 none
      -prob 1 -tr 2 -mu 0.0 -nsteps 1 -msteps 1 -indent 0.02 -travel 0.04)

    add_contact_test(contact_coulomb_np=1 1 slip
      -prob 1 -tr 2 -mu 0.1 -nsteps 1 -msteps 1
      -indent 0.02 -travel 0.04 -fstep 0.02)
    add_contact_test(contact_coulomb_np=2 2 slip
      -prob 1 -tr 2 -mu 0.1 -nsteps 1 -msteps 1
      -indent 0.02 -travel 0.04 -fstep 0.02)
    add_contact_test(contact_coulomb_stick 1 stick
      -prob 1 -tr 2 -mu 1.0 -nsteps 1 -msteps 1
      -indent 0.02 -travel 0.002 -fstep 0.02)
    add_contact_test(contact_coulomb_zero_friction 2 zero
      -prob 1 -tr 2 -coulomb -mu 0.0 -nsteps 1 -msteps 1
      -indent 0.02 -travel 0.04 -fstep 0.02)

  endif()

endif ()

