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

execute_process(COMMAND "${CMAKE_COMMAND}" --build "${MFEM_BINARY_DIR}"
  --target copy_data RESULT_VARIABLE result)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "Building copy_data failed")
endif()
string(RANDOM LENGTH 12 suffix)
set(staging "${MFEM_BINARY_DIR}/install-prefix-${suffix}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "DESTDIR=${staging}"
  "${CMAKE_COMMAND}" --install "${MFEM_BINARY_DIR}" --prefix /mfem-relocated
  RESULT_VARIABLE result OUTPUT_QUIET)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "Staged installation failed")
endif()
if (NOT EXISTS "${staging}/mfem-relocated/include/mfem.hpp")
  message(FATAL_ERROR "Headers did not follow the install prefix")
endif()
if (NOT EXISTS "${staging}/mfem-relocated/data/star.mesh")
  message(FATAL_ERROR "Mesh data did not follow the install prefix")
endif()
file(REMOVE_RECURSE "${staging}")
