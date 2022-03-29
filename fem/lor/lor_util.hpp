// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_UTIL
#define MFEM_LOR_UTIL

#include "../../config/config.hpp"
#include "../../general/backends.hpp"
#include "../../general/globals.hpp"

namespace mfem
{

template <int ORDER>
MFEM_HOST_DEVICE inline void LORVertexCoordinates2D(
   const double *X, int iel_ho, int kx, int ky, double vx[4], double vy[4])
{
   const int dim = 2;
   const int nd1d = ORDER + 1;
   const int nvert_per_el = nd1d*nd1d;

   const int v0 = kx + nd1d*ky;
   const int v1 = kx + 1 + nd1d*ky;
   const int v2 = kx + 1 + nd1d*(ky + 1);
   const int v3 = kx + nd1d*(ky + 1);

   const int e0 = dim*(v0 + nvert_per_el*iel_ho);
   const int e1 = dim*(v1 + nvert_per_el*iel_ho);
   const int e2 = dim*(v2 + nvert_per_el*iel_ho);
   const int e3 = dim*(v3 + nvert_per_el*iel_ho);

   // Vertex coordinates
   vx[0] = X[e0 + 0];
   vy[0] = X[e0 + 1];

   vx[1] = X[e1 + 0];
   vy[1] = X[e1 + 1];

   vx[2] = X[e2 + 0];
   vy[2] = X[e2 + 1];

   vx[3] = X[e3 + 0];
   vy[3] = X[e3 + 1];
}

}

#endif
