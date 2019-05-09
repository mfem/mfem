// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BILININTEG_EXT_DIFFUSION
#define MFEM_BILININTEG_EXT_DIFFUSION

namespace mfem
{

bool SmemPADiffusionApply(const int dim,
                          const int D1D,
                          const int Q1D,
                          const int NE,
                          const double* B,
                          const double* G,
                          const double* Bt,
                          const double* Gt,
                          const double* op,
                          const double* x,
                          double* y);

} // namespace mfem

#endif
