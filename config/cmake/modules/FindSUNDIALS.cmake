# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

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
  ADD_COMPONENT NVector_Parallel
    "include" nvector/nvector_parallel.h "lib" sundials_nvecparallel
  ADD_COMPONENT NVector_ParHyp
    "include" nvector/nvector_parhyp.h "lib" sundials_nvecparhyp
  ADD_COMPONENT CVODE "include" cvode/cvode.h "lib" sundials_cvode
  ADD_COMPONENT ARKODE "include" arkode/arkode.h "lib" sundials_arkode
  ADD_COMPONENT KINSOL "include" kinsol/kinsol.h "lib" sundials_kinsol)
