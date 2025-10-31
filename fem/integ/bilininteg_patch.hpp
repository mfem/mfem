// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG_PATCH
#define MFEM_BILININTEG_PATCH

#include "../integrator.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/tensor.hpp"

namespace mfem
{

/**
 * Compute gradient at quadrature points for a NURBS patch
 */
template <int vdim>
void PatchG3D(const PatchBasisInfo &pb,
              const Vector &Uv,
              Vector &sumXYv,
              Vector &sumXv,
              DeviceTensor<5, real_t> &gradu)
{
   // Unpack
   static constexpr int dim = 3;
   MFEM_VERIFY(pb.dim == 3, "");
   const Array<int>& Q1D = pb.Q1D;
   const Array<int>& D1D = pb.D1D;
   const std::vector<Array2D<real_t>>& B = pb.B;
   const std::vector<Array2D<real_t>>& G = pb.G;
   // minD/maxD are maps from 1D dof index -> 1D quadrature index
   // For component c and dof index d, [minD[c][d], maxD[c][d]] are
   // the min/max quadrature indices supported by the shape function
   // B_{cd}. Because shape functions on patches don't necessarily
   // support the whole domain, these maps are used to eliminate
   // unnecessary interpolations.
   const std::vector<std::vector<int>> minD = pb.minD;
   const std::vector<std::vector<int>> maxD = pb.maxD;

   // Shape as tensors
   const auto U = Reshape(Uv.HostRead(), D1D[0], D1D[1], D1D[2], vdim);
   auto sumXY = Reshape(sumXYv.HostReadWrite(), vdim, dim, pb.MAX1D[0],
                        pb.MAX1D[1]);
   auto sumX = Reshape(sumXv.HostReadWrite(), vdim, dim, pb.MAX1D[0]);
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      sumXYv = 0.0;
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         sumXv = 0.0;
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int c = 0; c < vdim; ++c)
            {
               const real_t u = U(dx,dy,dz,c);
               for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
               {
                  sumX(c,0,qx) += u * B[0](qx,dx);
                  sumX(c,1,qx) += u * G[0](qx,dx);
               }
            }
         } // dx
         for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
         {
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            for (int c = 0; c < vdim; ++c)
            {
               // This full range of qx values is generally necessary.
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  const real_t wx  = sumX(c,0,qx);
                  const real_t wDx = sumX(c,1,qx);
                  sumXY(c,0,qx,qy) += wDx * wy;
                  sumXY(c,1,qx,qy) += wx  * wDy;
                  sumXY(c,2,qx,qy) += wx  * wy;
               } // qx
            } // c
         } // qy
      } // dy

      for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
      {
         const real_t wz  = B[2](qz,dz);
         const real_t wDz = G[2](qz,dz);
         for (int c = 0; c < vdim; ++c)
         {
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  gradu(c,0,qx,qy,qz) += sumXY(c,0,qx,qy) * wz;
                  gradu(c,1,qx,qy,qz) += sumXY(c,1,qx,qy) * wz;
                  gradu(c,2,qx,qy,qz) += sumXY(c,2,qx,qy) * wDz;
               }
            } // qy
         } // c
      } // qz
   } // dz
}

/**
 * Contraction with grad_v^T
 */
template<int vdim>
void PatchGT3D(const PatchBasisInfo &pb,
               const DeviceTensor<5, real_t> &S,
               Vector &sumXYv,
               Vector &sumXv,
               Vector &y)
{
   // Unpack patch basis info
   static constexpr int dim = 3;
   MFEM_VERIFY(pb.dim == 3, "");
   const Array<int>& Q1D = pb.Q1D;
   const Array<int>& D1D = pb.D1D;
   const std::vector<Array2D<real_t>>& B = pb.B;
   const std::vector<Array2D<real_t>>& G = pb.G;
   const std::vector<std::vector<int>> minQ = pb.minQ;
   const std::vector<std::vector<int>> maxQ = pb.maxQ;

   // Shape as tensors
   auto sumXY = Reshape(sumXYv.HostReadWrite(), vdim, dim, pb.MAX1D[0],
                        pb.MAX1D[1]);
   auto sumX = Reshape(sumXv.HostReadWrite(), vdim, dim, pb.MAX1D[0]);

   auto Y = Reshape(y.HostReadWrite(), D1D[0], D1D[1], D1D[2], vdim);

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      sumXYv = 0.0;
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         sumXv = 0.0;
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const auto s = mfem::future::make_tensor<vdim, dim>(
            [&](int i, int j) { return S(i,j,qx,qy,qz); });
            for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
            {
               const real_t wx  = B[0](qx,dx);
               const real_t wDx = G[0](qx,dx);

               for (int c = 0; c < vdim; ++c)
               {
                  sumX(c,0,dx) += s(c,0) * wDx;
                  sumX(c,1,dx) += s(c,1) * wx;
                  sumX(c,2,dx) += s(c,2) * wx;
               }
            }
         }
         for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
         {
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int c = 0; c < vdim; ++c)
               {
                  sumXY(c,0,dx,dy) += sumX(c,0,dx) * wy + sumX(c,1,dx) * wDy;
                  sumXY(c,1,dx,dy) += sumX(c,2,dx) * wy;
               }
            }
         }
      }
      for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
      {
         const real_t wz  = B[2](qz,dz);
         const real_t wDz = G[2](qz,dz);
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int c = 0; c < vdim; ++c)
               {
                  Y(dx,dy,dz,c) +=
                     (sumXY(c,0,dx,dy) * wz +
                      sumXY(c,1,dx,dy) * wDz);
               }
            }
         }
      } // dz
   } // qz

}

} // namespace mfem

#endif