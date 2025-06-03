// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_GSLIB_PA_HPP
#define MFEM_GSLIB_PA_HPP

#include "../../config/config.hpp"
#include "../../linalg/dtensor.hpp"

#include "../kernels.hpp"

#include <unordered_map>

namespace mfem
{

#define sDIM 3
#define rDIM 1

// 12*8 + 4 = 100 bytes
struct findptsElementPoint_t
{
   double x[sDIM], r, oldr, dist2, dist2p, tr;
   int flags;
};

// 6*8 = 48 bytes
struct findptsElementGFace_t
{
   double *x[sDIM], *dxdn[sDIM];
};
// 15*8 = 120 bytes
struct findptsElementGEdge_t
{
   double *x[sDIM], *dxdn[sDIM], *d2xdn[sDIM];
};
// 30*8 = 240 bytes
struct findptsElementGPT_t
{
   double x[sDIM], jac[sDIM],
          hes[sDIM*(1+1)];  // r2,s2,rs for sDIM coordinates
};
// 2*8 = 16 bytes
struct dbl_range_t
{
   double min, max;
};
// 12*8 + 3*16 = 96+48 = 144 bytes
struct obbox_t
{
   double c0[sDIM], A[sDIM*sDIM];
   dbl_range_t x[sDIM];
};

// 4 + 3*8 + 3*16 + 4 + xyz = 80 bytes + xyz
struct findptsLocalHashData_t
{
   int hash_n;
   dbl_range_t bnd[sDIM];
   double fac[sDIM];
   unsigned int *offset;
   // int max;
};
#undef sDIM
#undef rDIM

} // namespace mfem

#endif // MFEM_GSLIB_PA_HPP
