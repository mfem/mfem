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

#define pDIM 2
#define dlong int
#define dfloat double
struct findptsElementPoint_t
{
   dfloat x[pDIM], r[pDIM], oldr[pDIM], dist2, dist2p, tr;
   dlong flags;
};

struct findptsElementGEdge_t
{
   dfloat *x[pDIM], *dxdn[2];
};

struct findptsElementGPT_t
{
   dfloat x[pDIM], jac[pDIM * pDIM], hes[4];
};

struct dbl_range_t
{
   dfloat min, max;
};
struct obbox_t
{
   dfloat c0[pDIM], A[pDIM * pDIM];
   dbl_range_t x[pDIM];
};

struct findptsLocalHashData_t
{
   dlong hash_n;
   dbl_range_t bnd[pDIM];
   dfloat fac[pDIM];
   dlong *offset;
   dlong max;
};
#undef dlong
#undef dfloat
#undef pdim

} // namespace mfem

#endif // MFEM_GSLIB_PA_HPP
