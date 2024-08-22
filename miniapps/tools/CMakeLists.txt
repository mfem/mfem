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

add_mfem_miniapp(display-basis
  MAIN display-basis.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common)

add_mfem_miniapp(get-values
  MAIN get-values.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common)

add_mfem_miniapp(load-dc
  MAIN load-dc.cpp LIBRARIES mfem)

add_mfem_miniapp(convert-dc
  MAIN convert-dc.cpp LIBRARIES mfem)

add_mfem_miniapp(lor-transfer
  MAIN lor-transfer.cpp LIBRARIES mfem)

if (MFEM_USE_MPI)
  add_mfem_miniapp(plor-transfer
    MAIN plor-transfer.cpp LIBRARIES mfem)
endif()

add_mfem_miniapp(tmop-check-metric
  MAIN tmop-check-metric.cpp LIBRARIES mfem)

add_mfem_miniapp(tmop-metric-magnitude
  MAIN tmop-metric-magnitude.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common)

if (MFEM_USE_MPI)
   add_mfem_miniapp(nodal-transfer
     MAIN nodal-transfer.cpp LIBRARIES mfem)
endif()
