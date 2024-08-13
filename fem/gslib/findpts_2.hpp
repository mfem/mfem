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

#define sDim 2
struct findptsElementPoint_t
{
   double x[sDim], r[sDim], oldr[sDim], dist2, dist2p, tr;
   int flags;
};

struct findptsElementGEdge_t
{
   double *x[sDim], *dxdn[2];
};

struct findptsElementGPT_t
{
   double x[sDim], jac[sDim * sDim], hes[4];
};

struct dbl_range_t
{
   double min, max;
};
struct obbox_t
{
   double c0[sDim], A[sDim * sDim];
   dbl_range_t x[sDim];
};

struct findptsLocalHashData_t
{
   int hash_n;
   dbl_range_t bnd[sDim];
   double fac[sDim];
   unsigned int *offset;
   int max;
};
#undef sDim

} // namespace mfem

#endif // MFEM_GSLIB_PA_HPP
