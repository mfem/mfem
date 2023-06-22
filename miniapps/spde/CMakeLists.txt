# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if (MFEM_USE_MPI)
  add_mfem_miniapp(generate_random_field
    MAIN generate_random_field.cpp
    EXTRA_SOURCES material_metrics.cpp spde_solver.cpp util.cpp visualizer.cpp
                  transformation.cpp
    LIBRARIES mfem)

  # Copy files to the build folder.
  file(COPY ${PROJECT_SOURCE_DIR}/miniapps/spde/README.md
       DESTINATION ${PROJECT_BINARY_DIR}/miniapps/spde)
else ()
  message("This miniapp requires MPI to be enabled."
          "Please compile with MFEM_USE_MPI.")
endif()
