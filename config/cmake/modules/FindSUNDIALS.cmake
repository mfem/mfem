# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Defines the following variables:
#   - SUNDIALS_FOUND
#   - SUNDIALS_LIBRARIES
#   - SUNDIALS_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(SUNDIALS SUNDIALS SUNDIALS_DIR
  "include" sundials/sundials_config.h "lib" ""
  "Paths to headers required by SUNDIALS." "Libraries required by SUNDIALS."
  ADD_COMPONENT NVector_Serial
    "include" nvector/nvector_serial.h "lib" sundials_nvecserial
  ADD_COMPONENT NVector_Cuda
    "include" nvector/nvector_cuda.h "lib" sundials_nveccuda
  ADD_COMPONENT NVector_Hip
    "include" nvector/nvector_hip.h "lib" sundials_nvechip
  ADD_COMPONENT NVector_Parallel
    "include" nvector/nvector_parallel.h "lib" sundials_nvecparallel
  ADD_COMPONENT NVector_MPIPlusX
    "include" nvector/nvector_mpiplusx.h "lib" sundials_nvecmpiplusx
  ADD_COMPONENT CVODE "include" cvode/cvode.h "lib" sundials_cvode
  ADD_COMPONENT CVODES "include" cvodes/cvodes.h "lib" sundials_cvodes
  ADD_COMPONENT ARKODE "include" arkode/arkode.h "lib" sundials_arkode
  ADD_COMPONENT KINSOL "include" kinsol/kinsol.h "lib" sundials_kinsol
  ADD_COMPONENT Core "include" sundials/sundials_core.h "lib" sundials_core)
