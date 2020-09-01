// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CEEDSOLVERS_INTERPOLATION_H
#define MFEM_CEEDSOLVERS_INTERPOLATION_H

#ifdef MFEM_USE_CEED
#include <ceed.h>

typedef struct CeedInterpolation_private *CeedInterpolation;

struct CeedInterpolation_private {
  Ceed ceed;
  CeedQFunction qf_restrict, qf_prolong;
  CeedOperator op_interp, op_restrict;
  CeedVector fine_multiplicity_r;
  CeedVector fine_work;
};

int CeedInterpolationCreate(Ceed ceed, CeedBasis basisctof,
                            CeedElemRestriction erestrictu_coarse,
                            CeedElemRestriction erestrictu_fine,
                            CeedInterpolation *interp);
int CeedInterpolationDestroy(CeedInterpolation *interp);
int CeedInterpolationInterpolate(CeedInterpolation interp,
                                 CeedVector in, CeedVector out);
int CeedInterpolationRestrict(CeedInterpolation interp,
                              CeedVector in, CeedVector out);

#endif // MFEM_USE_CEED

#endif // include guard
