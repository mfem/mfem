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

set(KERS
  kGeometry.kpp
  kGlobalToLocal.kpp
  kIntDiffusionAssemble.kpp
  kIntDiffusionMultAdd.kpp
  kLocalToGlobal.kpp
)

set(HDRS
  kGeometry.hpp
  kGlobalToLocal.hpp
  kIntDiffusion.hpp
  kLocalToGlobal.hpp
)

convert_filenames_to_full_paths(KERS)
convert_filenames_to_full_paths(HDRS)

set(KERNELS ${KERNELS} ${KERS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
