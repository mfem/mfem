// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../fem.hpp"
#include "../../general/forall.hpp"

#define MFEM_DEBUG_COLOR 227
#include "../../general/debug.hpp"

#define MFEM_NVTX_COLOR MediumVioletRed
#include "../../general/nvtx.hpp"

namespace mfem
{

#define M1D 8

template<int D1D, int Q1D>
void NodalInterpolation3D(const int NE,
                          const Vector& localL, Vector& localH,
                          const Array<double> &B)
{
   MFEM_NVTX;
   dbg("D1D:%d Q1D:%d", D1D, Q1D);

   static constexpr int VDIM = 3;

   const auto x_ = Reshape(localL.Read(), D1D, D1D, D1D, VDIM, NE);
   const auto B_ = Reshape(B.Read(), Q1D, D1D);

   auto y_ = Reshape(localH.Write(), VDIM, Q1D, Q1D, Q1D, NE);

   {
      NVTX("localH = 0.0");
      localH = 0.0;
   }

   {
      NVTX("NodalInterpolation3D Kernel");
      MFEM_FORALL(e, NE,
      {
         for (int vd = 0; vd < VDIM; ++vd)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               double sol_xy[M1D][M1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_xy[qy][qx] = 0.0;
                  }
               }
               for (int dy = 0; dy < D1D; ++dy)
               {
                  double sol_x[M1D];
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_x[qx] = 0;
                  }
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double s = x_(dx, dy, dz, vd, e);
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        sol_x[qx] += B_(qx, dx) * s;
                     }
                  }
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = B_(qy, dy);
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        sol_xy[qy][qx] += wy * sol_x[qx];
                     }
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const double wz = B_(qz, dz);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        y_(vd, qx, qy, qz, e) += wz * sol_xy[qy][qx];
                     }
                  }
               }
            }
         }
      });
   }
}

#define NODAL_INTERP_3D_INSTANCE(D1D,Q1D) \
template void NodalInterpolation3D<D1D,Q1D>\
    (const int, const Vector&, Vector&,const Array<double>&)

NODAL_INTERP_3D_INSTANCE(2,2);
NODAL_INTERP_3D_INSTANCE(2,3);
NODAL_INTERP_3D_INSTANCE(2,4);
NODAL_INTERP_3D_INSTANCE(2,5);
NODAL_INTERP_3D_INSTANCE(2,6);
NODAL_INTERP_3D_INSTANCE(2,7);

NODAL_INTERP_3D_INSTANCE(4,2);
NODAL_INTERP_3D_INSTANCE(4,3);
NODAL_INTERP_3D_INSTANCE(4,4);
NODAL_INTERP_3D_INSTANCE(4,5);
NODAL_INTERP_3D_INSTANCE(4,6);
NODAL_INTERP_3D_INSTANCE(4,7);

NODAL_INTERP_3D_INSTANCE(6,2);
NODAL_INTERP_3D_INSTANCE(6,3);
NODAL_INTERP_3D_INSTANCE(6,4);
NODAL_INTERP_3D_INSTANCE(6,5);
NODAL_INTERP_3D_INSTANCE(6,6);
NODAL_INTERP_3D_INSTANCE(6,7);

} // namespace mfem
