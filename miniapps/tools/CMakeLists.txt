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

add_mfem_miniapp(display-basis
  MAIN display-basis.cpp
  EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem mfem_miniapps_common)

add_mfem_miniapp(load-dc
  MAIN load-dc.cpp LIBRARIES mfem)

add_mfem_miniapp(convert-dc
  MAIN convert-dc.cpp LIBRARIES mfem)

add_mfem_miniapp(lor-transfer
  MAIN lor-transfer.cpp LIBRARIES mfem)
