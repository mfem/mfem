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
  ip.cpp
  optcontactproblem.cpp
  elastoperator.cpp
  solver_utils.cpp
  neo_hookean.cpp
  contact_driver.cpp
  mesh_generator.cpp
  driver_output.cpp)
  list(APPEND CONTACT_HEADERS
  contact_output.hpp
  ip.hpp
  optcontactproblem.hpp
  elastoperator.hpp
  solver_utils.hpp
  neo_hookean.hpp
  contact_driver.hpp
  mesh_generator.hpp
  driver_output.hpp)

  convert_filenames_to_full_paths(CONTACT_SOURCES)
  convert_filenames_to_full_paths(CONTACT_HEADERS)

  add_mfem_miniapp(contact
    MAIN contact.cpp
    EXTRA_SOURCES ${CONTACT_SOURCES}
    EXTRA_HEADERS ${CONTACT_HEADERS}
    LIBRARIES mfem)

  add_mfem_miniapp(contact_examples
    MAIN contact_examples.cpp
    EXTRA_SOURCES ${CONTACT_SOURCES}
    EXTRA_HEADERS ${CONTACT_HEADERS}
    LIBRARIES mfem)

  foreach(driver contact contact_examples)
    if (MFEM_USE_CUDA OR MFEM_USE_HIP)
      # Device-enabled Tribol dependencies require C++20; MFEM may use C++17.
      target_compile_features(${driver} PRIVATE cxx_std_20)
      if (MFEM_USE_CUDA)
        set_property(TARGET ${driver} PROPERTY CUDA_STANDARD 20)
        set_property(TARGET ${driver} PROPERTY CUDA_STANDARD_REQUIRED ON)
      endif()
    endif()
    # Tribol depends on MFEM; place MFEM after its third-party libraries too.
    target_link_libraries(${driver} PUBLIC ${TPL_LIBRARIES} mfem)
  endforeach()

endif ()
