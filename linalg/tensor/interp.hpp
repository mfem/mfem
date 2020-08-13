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

#ifndef MFEM_TENSOR_INTERP
#define MFEM_TENSOR_INTERP

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include <utility>

namespace mfem
{

// Functions to interpolate from degrees of freedom to quadrature points
// Non-tensor case
template<int D, int Q> MFEM_HOST_DEVICE inline
dTensor<Q>&& Interpolate(const dTensor<Q,D> &B,
                         const dTensor<D> &u)
{
   dTensor<Q> u_q;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         v += b * u(d);
      }
      u_q(q) = v;
   }
   MFEM_SYNC_THREAD;
   return std::move(u_q);
}

// Non-tensor and 1D cases with VDim components
template<int Q, int D, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,Q>&& Interpolate(const dTensor<Q,D> &B,
                                      const Tensor<dTensor<VDim>,D> &u)
{
   Tensor<dTensor<VDim>,Q> u_q;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v[VDim];
      for (int c = 0; c < VDim; c++)
      {
         v[c] = 0.0;
      }
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         for (int c = 0; c < VDim; c++)
         {
            const double x = u(d)(c);
            v[c] += b * x;
         }
      }
      for (int c = 0; c < VDim; c++)
      {
         u_q(q)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u_q);
}

// 3D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d,Q1d>&& Interpolate(const dTensor<Q1d,D1d> &B,
                                   const dTensor<D1d,D1d,D1d> &u)
{
   dTensor<Q1d,D1d,D1d> Bu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1d)
         {
            double val = 0.0;
            for (int dx = 0; dx < D1d; ++dx)
            {
               const double b = B(qx,dx);
               const double x = u(dx,dy,dz);
               val += b * x;
            }
            Bu(qx,dy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<Q1d,Q1d,D1d> BBu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1d)
         {
            double val = 0.0;
            for (int dy = 0; dy < D1d; ++dy)
            {
               const double b = B(qy,dy);
               const double x = Bu(qx,dy,dz);
               val += b * x;
            }
            BBu(qx,qy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<Q1d,Q1d,Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(qz,z,D1d)
         {
            double val = 0.0;
            for (int dz = 0; dz < D1d; ++dz)
            {
               const double b = B(qz,dz);
               const double x = Bu(qx,qy,dz);
               val += b * x;
            }
            u_q(qx,qy,qz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u_q);
}

// 3D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,Q1d,Q1d,Q1d>&& Interpolate(const dTensor<Q1d,D1d> &B,
                                                const Tensor<dTensor<VDim>,D1d,D1d,D1d> &u)
{
   Tensor<dTensor<VDim>,Q1d,D1d,D1d> Bu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1d)
         {
            double val[VDim];
            for (int c = 0; c < VDim; c++)
            {
               val[c] = 0.0;
            }
            for (int dx = 0; dx < D1d; ++dx)
            {
               const double b = B(qx,dx);
               for (int c = 0; c < VDim; c++)
               {
                  const double x = u(dx,dy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               Bu(qx,dy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,Q1d,Q1d,D1d> BBu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1d)
         {
            double val[VDim];
            for (int c = 0; c < VDim; c++)
            {
               val[c] = 0.0;
            }
            for (int dy = 0; dy < D1d; ++dy)
            {
               const double b = B(qy,dy);
               for (int c = 0; c < VDim; c++)
               {
                  const double x = Bu(qx,dy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               BBu(qx,qy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,Q1d,Q1d,Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(qz,z,D1d)
         {
            double val[VDim];
            for (int c = 0; c < VDim; c++)
            {
               val[c] = 0.0;
            }
            for (int dz = 0; dz < D1d; ++dz)
            {
               const double b = B(qz,dz);
               for (int c = 0; c < VDim; c++)
               {
                  const double x = Bu(qx,qy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               u_q(qx,qy,qz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u_q);
}

// 2D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d>&& Interpolate(const dTensor<Q1d,D1d> &B,
                               const dTensor<D1d,D1d> &u)
{
   dTensor<Q1d,D1d> Bu;
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         double val = 0.0;
         for (int dx = 0; dx < D1d; ++dx)
         {
            const double b = B(qx,dx);
            const double x = u(dx,dy);
            val += b * x;
         }
         Bu(qx,dy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<Q1d,Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         double val = 0.0;
         for (int dy = 0; dy < D1d; ++dy)
         {
            const double b = B(qy,dy);
            const double x = Bu(qx,dy);
            val += b * x;
         }
         u_q(qx,qy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u_q);
}

// 2D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,Q1d,Q1d>&& Interpolate(const dTensor<Q1d,D1d> &B,
                                            const Tensor<dTensor<VDim>,D1d,D1d> &u)
{
   Tensor<dTensor<VDim>,Q1d,D1d> Bu;
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         double val[VDim];
         for (int c = 0; c < VDim; c++)
         {
            val[c] = 0.0;
         }
         for (int dx = 0; dx < D1d; ++dx)
         {
            const double b = B(qx,dx);
            for (int c = 0; c < VDim; c++)
            {
               const double x = u(dx,dy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            Bu(qx,dy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,Q1d,Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         double val[VDim];
         for (int c = 0; c < VDim; c++)
         {
            val[c] = 0.0;
         }
         for (int dy = 0; dy < D1d; ++dy)
         {
            const double b = B(qy,dy);
            for (int c = 0; c < VDim; c++)
            {
               const double x = Bu(qx,dy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            u_q(qx,qy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u_q);
}

// 1D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d>&& Interpolate(const dTensor<Q1d,D1d> &B,
                           const dTensor<D1d> &u)
{
   dTensor<Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      double val = 0.0;
      for (int dx = 0; dx < D1d; ++dx)
      {
         const double b = B(qx,dx);
         const double x = u(dx);
         val += b * x;
      }
      u_q(qx) = val;
   }
   MFEM_SYNC_THREAD;
   return std::move(u_q);
}

// Functions to interpolate from degrees of freedom to quadrature points
// Non-tensor and 1D cases
template<int P, int Q> MFEM_HOST_DEVICE inline
dTensor<P>&& InterpolateT(const dTensor<Q,P> &B,
                          const dTensor<Q> &u_q)
{
   dTensor<Q> u;
   MFEM_FOREACH_THREAD(d,x,P)
   {
      double v = 0.0;
      for (int q = 0; q < Q; ++q)
      {
         const double b = B(q,d);
         const double x = u_q(q);
         v += b * x;
      }
      u(d) = v;
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// Non-tensor and 1D cases with VDim components
template<int Q, int P, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,P>&& InterpolateT(const dTensor<Q,P> &B,
                                       const Tensor<dTensor<VDim>,Q> &u_q)
{
   Tensor<dTensor<VDim>,P> u;
   MFEM_FOREACH_THREAD(d,x,P)
   {
      double v[VDim];
      for (int c = 0; c < VDim; c++)
      {
         v[c] = 0.0;
      }
      for (int q = 0; q < Q; ++q)
      {
         const double b = B(q,d);
         for (int c = 0; c < VDim; c++)
         {
            const double x = u_q(q)(c);
            v[c] += b * x;
         }
      }
      for (int c = 0; c < VDim; c++)
      {
         u(d)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 3D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d,D1d>&& InterpolateT(const dTensor<Q1d,D1d> &B,
                                   const dTensor<Q1d,Q1d,Q1d> &u_q)
{
   dTensor<D1d,Q1d,Q1d> Bu;
   MFEM_FOREACH_THREAD(qz,z,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            double val = 0.0;
            for (int qx = 0; qx < Q1d; ++qx)
            {
               const double b = B(qx,dx);
               const double x = u_q(qx,qy,qz);
               val += b * x;
            }
            Bu(dx,qy,qz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<D1d,D1d,Q1d> BBu;
   MFEM_FOREACH_THREAD(qz,z,Q1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            double val = 0.0;
            for (int qy = 0; qy < Q1d; ++qy)
            {
               const double b = B(qy,dy);
               const double x = Bu(dx,qy,qz);
               val += b * x;
            }
            BBu(dx,dy,qz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<D1d,D1d,D1d> u;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dz,z,Q1d)
         {
            double val = 0.0;
            for (int qz = 0; qz < Q1d; ++qz)
            {
               const double b = B(qz,dz);
               const double x = Bu(dx,dy,qz);
               val += b * x;
            }
            u(dx,dy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 3D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,D1d,D1d,D1d>&& InterpolateT(const dTensor<Q1d,D1d> &B,
                                                 const Tensor<dTensor<VDim>,Q1d,Q1d,Q1d> &u_q)
{
   Tensor<dTensor<VDim>,D1d,Q1d,Q1d> Bu;
   MFEM_FOREACH_THREAD(qz,z,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            double val[VDim];
            for (int c = 0; c < VDim; c++)
            {
               val[c] = 0.0;
            }
            for (int qx = 0; qx < Q1d; ++qx)
            {
               const double b = B(qx,dx);
               for (int c = 0; c < VDim; c++)
               {
                  const double x = u_q(qx,qy,qz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               Bu(dx,qy,qz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,D1d,D1d,Q1d> BBu;
   MFEM_FOREACH_THREAD(qz,z,Q1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            double val[VDim];
            for (int c = 0; c < VDim; c++)
            {
               val[c] = 0.0;
            }
            for (int qy = 0; qy < Q1d; ++qy)
            {
               const double b = B(qy,dy);
               for (int c = 0; c < VDim; c++)
               {
                  const double x = Bu(dx,qy,qz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               BBu(dx,dy,qz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,D1d,D1d,D1d> u;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dz,z,D1d)
         {
            double val[VDim];
            for (int c = 0; c < VDim; c++)
            {
               val[c] = 0.0;
            }
            for (int qz = 0; qz < Q1d; ++qz)
            {
               const double b = B(qz,dz);
               for (int c = 0; c < VDim; c++)
               {
                  const double x = Bu(dx,dy,qz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               u(dx,dy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 2D Tensor case
template<int D1d, int Q1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d>&& InterpolateT(const dTensor<Q1d,D1d> &B,
                                const dTensor<Q1d,Q1d> &u_q)
{
   dTensor<D1d,Q1d> Bu;
   MFEM_FOREACH_THREAD(qy,y,Q1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         double val = 0.0;
         for (int qx = 0; qx < Q1d; ++qx)
         {
            const double b = B(qx,dx);
            const double x = u_q(qx,qy);
            val += b * x;
         }
         Bu(dx,qy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<D1d,D1d> u;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         double val = 0.0;
         for (int qy = 0; qy < Q1d; ++qy)
         {
            const double b = B(qy,dy);
            const double x = Bu(dx,qy);
            val += b * x;
         }
         u(dx,dy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 2D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,D1d,D1d>&& InterpolateT(const dTensor<Q1d,D1d> &B,
                                             const Tensor<dTensor<VDim>,Q1d,Q1d> &u_q)
{
   Tensor<dTensor<VDim>,D1d,Q1d> Bu;
   MFEM_FOREACH_THREAD(qy,y,Q1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         double val[VDim];
         for (int c = 0; c < VDim; c++)
         {
            val[c] = 0.0;
         }
         for (int qx = 0; qx < Q1d; ++qx)
         {
            const double b = B(qx,dx);
            for (int c = 0; c < VDim; c++)
            {
               const double x = u_q(qx,qy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            Bu(dx,qy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,D1d,D1d> u;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         double val[VDim];
         for (int c = 0; c < VDim; c++)
         {
            val[c] = 0.0;
         }
         for (int qy = 0; qy < Q1d; ++qy)
         {
            const double b = B(qy,dy);
            for (int c = 0; c < VDim; c++)
            {
               const double x = Bu(dx,qy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            u(dx,dy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP
