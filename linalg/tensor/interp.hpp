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
#include "../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Functions to interpolate from degrees of freedom to quadrature points
// Non-tensor case
template<int P, int Q> MFEM_HOST_DEVICE inline
dTensor<Q>&& Interp(const dTensor<Q,P> &B,
                    const dTensor<P> &u)
{
   dTensor<Q> u_q;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < P; ++d)
      {
         const double b = B(q,d);
         v += b * u(d);
      }
      u_q(q) = v;
   }
   MFEM_SYNC_THREAD;
   return u_q;
}

// Non-tensor case with VDIM components
template<int Q, int P, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,Q>&& Interp(const dTensor<Q,P> &B,
                                 const Tensor<dTensor<VDIM>,P> &u)
{
   Tensor<dTensor<VDIM>,Q> u_q;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v[VDIM];
      for (int c = 0; c < VDIM; c++)
      {
         v[c] = 0.0;
      }
      for (int d = 0; d < P; ++d)
      {
         const double b = B(q,d);
         for (int c = 0; c < VDIM; c++)
         {
            v[c] += b * u(d)(c);
         }
      }
      for (int c = 0; c < VDIM; c++)
      {
         u_q(q)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return u_q;
}

// 3D Tensor case
template<int Q1d, int P1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d,Q1d>&& Interp(const dTensor<Q1d,P1d> &B,
                              const dTensor<P1d,P1d,P1d> &u)
{
   dTensor<Q1d,P1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double val = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
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
   dTensor<Q1d,Q1d,P1d> BBu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            double val = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
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
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qz,z,D1D)
         {
            double val = 0.0;
            for (int dz = 0; dz < D1D; ++dz)
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
   return u_q;
}

// 3D Tensor case with VDIM components
template<int Q1d, int P1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,Q1d,Q1d,Q1d>&& Interp(const dTensor<Q1d,P1d> &B,
                                           const Tensor<dTensor<VDIM>,P1d,P1d,P1d> &u)
{
   Tensor<dTensor<VDIM>,Q1d,P1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double val[VDIM];
            for (int c = 0; c < VDIM; c++)
            {
               val[c] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double b = B(qx,dx);
               for (int c = 0; c < count; c++)
               {
                  const double x = u(dx,dy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               Bu(qx,dy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDIM>,Q1d,Q1d,P1d> BBu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            double val[VDIM];
            for (int c = 0; c < VDIM; c++)
            {
               val[c] = 0.0;
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double b = B(qy,dy);
               for (int c = 0; c < count; c++)
               {
                  const double x = Bu(qx,dy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               BBu(qx,qy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDIM>,Q1d,Q1d,Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qz,z,D1D)
         {
            double val[VDIM];
            for (int c = 0; c < VDIM; c++)
            {
               val[c] = 0.0;
            }
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double b = B(qz,dz);
               for (int c = 0; c < count; c++)
               {
                  const double x = Bu(qx,qy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               u_q(qx,qy,qz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u_q;
}

// 2D Tensor case
template<int Q1d, int P1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d>&& Interp(const dTensor<Q1d,P1d> &B,
                          const dTensor<P1d,P1d> &u)
{
   dTensor<Q1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double val = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
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
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         double val = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double b = B(qy,dy);
            const double x = Bu(qx,dy);
            val += b * x;
         }
         u_q(qx,qy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   return u_q;
}

// 2D Tensor case with VDIM components
template<int Q1d, int P1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,Q1d,Q1d>&& Interp(const dTensor<Q1d,P1d> &B,
                                       const Tensor<dTensor<VDIM>,P1d,P1d> &u)
{
   Tensor<dTensor<VDIM>,Q1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double val[VDIM];
         for (int c = 0; c < VDIM; c++)
         {
            val[c] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double b = B(qx,dx);
            for (int c = 0; c < VDIM; c++)
            {
               const double x = u(dx,dy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDIM; c++)
         {
            Bu(qx,dy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDIM>,Q1d,Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         double val[VDIM];
         for (int c = 0; c < VDIM; c++)
         {
            val[c] = 0.0;
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double b = B(qy,dy);
            for (int c = 0; c < VDIM; c++)
            {
               const double x = Bu(qx,dy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDIM; c++)
         {
            u_q(qx,qy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u_q;
}

// 1D Tensor case
template<int Q1d, int P1d> MFEM_HOST_DEVICE inline
dTensor<Q1d>&& Interp(const dTensor<Q1d,P1d> &B,
                      const dTensor<P1d> &u)
{
   dTensor<Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      double val = 0.0;
      for (int dx = 0; dx < D1D; ++dx)
      {
         const double b = B(qx,dx);
         const double x = u(dx);
         val += b * x;
      }
      u_q(qx) = val;
   }
   MFEM_SYNC_THREAD;
   return u_q;
}

// 1D Tensor case with VDIM components
template<int Q1d, int P1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,Q1d>&& Interp(const dTensor<Q1d,P1d> &B,
                                   const Tensor<dTensor<VDIM>,P1d> &u)
{
   Tensor<dTensor<VDIM>,Q1d> u_q;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      double val[VDIM];
      for (int c = 0; c < VDIM; c++)
      {
         val[c] = 0.0;
      }
      for (int dx = 0; dx < D1D; ++dx)
      {
         const double b = B(qx,dx);
         for (int c = 0; c < VDIM; c++)
         {
            const double x = u(dx)(c);
            val[c] += b * x;
         }
      }
      for (int c = 0; c < VDIM; c++)
      {
         u_q(qx)(c) = val[c];
      }
   }
   MFEM_SYNC_THREAD;
   return u_q;
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP
