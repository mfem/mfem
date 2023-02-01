# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
  list(APPEND SDF_COMMON_SOURCES
      SDF_Generator.cpp
      SDF_Object.cpp
      SDF_Mesh.cpp)
  list(APPEND SDF_COMMON_HEADERS
      SDF_Generator.hpp
      SDF_Object.hpp
      SDF_Mesh.hpp
      SDF_Tools.hpp)

  convert_filenames_to_full_paths(SDF_COMMON_SOURCES)
  convert_filenames_to_full_paths(SDF_COMMON_HEADERS)

  set(SDF_COMMON_FILES
      EXTRA_SOURCES ${SDF_COMMON_SOURCES}
      EXTRA_HEADERS ${SDF_COMMON_HEADERS})

  add_mfem_miniapp(SDF
    MAIN SDF.cpp
    ${SDF_COMMON_FILES}
    LIBRARIES mfem)

endif ()
