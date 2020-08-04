# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

add_mfem_miniapp(cvsRoberts_ASAi_dns
  MAIN cvsRoberts_ASAi_dns.cpp
  LIBRARIES mfem mfem-common)

if (MFEM_USE_MPI)
  add_mfem_miniapp(adjoint_advection_diffusion
    MAIN adjoint_advection_diffusion.cpp
    LIBRARIES mfem mfem-common)
endif()
