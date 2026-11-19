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

  add_mfem_miniapp(ContactPatchTest
    MAIN contact-patch-test.cpp
    LIBRARIES mfem)

  if (MFEM_USE_CUDA OR MFEM_USE_HIP)
    target_compile_features(ContactPatchTest PRIVATE cxx_std_20)
    if (MFEM_USE_CUDA)
      set_property(TARGET ContactPatchTest PROPERTY CUDA_STANDARD 20)
      set_property(TARGET ContactPatchTest PROPERTY CUDA_STANDARD_REQUIRED ON)
    endif()
  endif()

# The way the libraries are included, by default, in the command line is based
# on the idea that MFEM depends on the other libraries. Here, the setting is
# such that Tribol, in fact, depends on MFEM, while MFEM does not depend on
# Tribol. This addresses that issue.
  target_link_libraries(ContactPatchTest PUBLIC ${TPL_LIBRARIES} mfem)

endif()
