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

#define sDIM 2
#define rDIM 1

struct findptsElementPoint_t
{
   double x[sDIM], r, oldr, dist2, dist2p, tr;
   int flags;
};

struct findptsElementGEdge_t
{
   double *x[sDIM];
};

/* jac: jacobian matrix Jij = delx_i/delr_j; i=x,y; j=r
 * hes: hessian matrix  Hij = del^2 x_i/delr_j^2; i=x,y; j=r
*/
struct findptsElementGPT_t
{
   double x[sDIM], jac[sDIM*rDIM], hes[sDIM*rDIM];
};

struct dbl_range_t
{
   double min, max;
};
struct obbox_t
{
   double c0[sDIM], A[sDIM*sDIM];
   dbl_range_t x[sDIM];
};

struct findptsLocalHashData_t
{
   int hash_n;
   dbl_range_t bnd[sDIM];
   double fac[sDIM];
   unsigned int *offset;
   int max;
};

#undef sDIM
#undef rDIM

} // namespace mfem

#endif // MFEM_GSLIB_PA_HPP
