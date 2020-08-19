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

#ifndef MFEM_TENSOR_GRAD
#define MFEM_TENSOR_GRAD

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Functions to interpolate the gradient from degrees of freedom to derivatives
// at quadrature points.
// Non-tensor case
template<int D, int Q, int Dim> MFEM_HOST_DEVICE inline
Tensor<dTensor<Dim>,Q>&& Gradient(const dTensor<Q,D> &B,
                                  const Tensor<dTensor<Dim>,Q,D> &G,
                                  const dTensor<D> &u)
{
   Tensor<dTensor<Dim>,Q> gu_q;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v[Dim];
      for (int c = 0; c < Dim; c++)
      {
         v[c] = 0.0;
      }      
      for (int d = 0; d < D; ++d)
      {
         const double x = u(d);
         for (int c = 0; c < Dim; c++)
         {
            const double g = G(q,d)(c);
            v[c] += g * x;
         }
      }
      for (int c = 0; c < Dim; c++)
      {
         gu_q(q)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// Non-tensor case with VDim components
template<int Q, int D, int Dim, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<Dim,VDim>,Q>&& Gradient(const dTensor<Q,D> &B,
                                       const Tensor<dTensor<Dim>,Q,D> &G,
                                       const Tensor<dTensor<VDim>,D> &u)
{
   Tensor<dTensor<Dim,VDim>,Q> gu_q;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v[Dim][VDim];
      for (int s = 0; s < Dim; s++)
      {
         for (int c = 0; c < VDim; c++)
         {
            v[s][c] = 0.0;
         }
      }
      for (int d = 0; d < D; ++d)
      {
         double b[Dim];
         double x[VDim];
         for (int c = 0; c < VDim; c++)
         {
            x[c] = u(d)(c);
         }
         for (int s = 0; s < Dim; s++)
         {
            b[s] = G(q,d)(s);
         }
         for (int s = 0; s < Dim; s++)
         {
            for (int c = 0; c < VDim; c++)
            {
               v[s][c] += b[s] * x[c];
            }
         }
      }
      for (int s = 0; s < Dim; s++)
      {
         for (int c = 0; c < VDim; c++)
         {
            gu_q(q)(s,c) = v[s][c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// 3D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
Tensor<dTensor<3>,Q1d,Q1d,Q1d>&& Gradient(const dTensor<Q1d,D1d> &B,
                                          const dTensor<Q1d,D1d> &G,
                                          const dTensor<D1d,D1d,D1d> &u)
{
   dTensor<Q1d,D1d,D1d> Bu;
   dTensor<Q1d,D1d,D1d> Gu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1d)
         {
            double bu = 0.0;
            double gu = 0.0;
            for (int dx = 0; dx < D1d; ++dx)
            {
               const double x = u(dx,dy,dz);
               const double b = B(qx,dx);
               const double g = G(qx,dx);
               bu += b * x;
               gu += g * x;
            }
            Bu(qx,dy,dz) = bu;
            Gu(qx,dy,dz) = gu;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<Q1d,Q1d,D1d> BBu;
   dTensor<Q1d,Q1d,D1d> GBu;
   dTensor<Q1d,Q1d,D1d> BGu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1d)
         {
            double bbu = 0.0;
            double gbu = 0.0;
            double bgu = 0.0;
            for (int dy = 0; dy < D1d; ++dy)
            {
               const double bu = Bu(qx,dy,dz);
               const double gu = Gu(qx,dy,dz);
               const double b = B(qy,dy);
               const double g = G(qy,dy);
               bbu += b * bu;
               gbu += g * bu;
               bgu += b * gu;  
            }
            BBu(qx,qy,dz) = bbu;
            GBu(qx,qy,dz) = gbu;
            BGu(qx,qy,dz) = bgu;
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<3>,Q1d,Q1d,Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(qz,z,D1d)
         {
            double gbbu = 0.0;
            double bgbu = 0.0;
            double bbgu = 0.0;
            for (int dz = 0; dz < D1d; ++dz)
            {
               const double b = B(qz,dz);
               const double g = G(qz,dz);
               const double bbu = BBu(qx,qy,dz);
               const double gbu = GBu(qx,qy,dz);
               const double bgu = BGu(qx,qy,dz);
               gbbu += g * bbu;
               bgbu += b * gbu;
               bbgu += b * bgu;
            }
            gu_q(qx,qy,qz)(0) = bbgu;
            gu_q(qx,qy,qz)(1) = bgbu;
            gu_q(qx,qy,qz)(2) = gbbu;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// 3D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim,3>,Q1d,Q1d,Q1d>&& Gradient(const dTensor<Q1d,D1d> &B,
                                               const dTensor<Q1d,D1d> &G,
                                               const dTensor<D1d,D1d,D1d> &u)
{
   Tensor<dTensor<VDim>,Q1d,D1d,D1d> Bu;
   Tensor<dTensor<VDim>,Q1d,D1d,D1d> Gu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1d)
         {
            double bu[VDim];
            double gu[VDim];
            for (int c = 0; c < VDim; c++)
            {
               bu[c] = 0.0;
               gu[c] = 0.0;
            }
            for (int dx = 0; dx < D1d; ++dx)
            {
               const double b = B(qx,dx);
               const double g = G(qx,dx);
               for (int c = 0; c < VDim; c++)
               {
                  const double x = u(dx,dy,dz);
                  bu[c] += b * x;
                  gu[c] += g * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               Bu(qx,dy,dz)(c) = bu[c];
               Gu(qx,dy,dz)(c) = gu[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,Q1d,Q1d,D1d> BBu;
   Tensor<dTensor<VDim>,Q1d,Q1d,D1d> GBu;
   Tensor<dTensor<VDim>,Q1d,Q1d,D1d> BGu;
   MFEM_FOREACH_THREAD(dz,z,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1d)
         {
            double bbu[VDim];
            double gbu[VDim];
            double bgu[VDim];
            for (int c = 0; c < VDim; c++)
            {
               bbu[c] = 0.0;
               gbu[c] = 0.0;
               bgu[c] = 0.0;
            }
            for (int dy = 0; dy < D1d; ++dy)
            {
               const double b = B(qy,dy);
               const double g = G(qy,dy);
               for (int c = 0; c < VDim; c++)
               {
                  const double bu = Bu(qx,dy,dz)(c);
                  const double gu = Gu(qx,dy,dz)(c);
                  bbu[c] += b * bu;
                  gbu[c] += g * bu;
                  bgu[c] += b * gu;
               }               
            }
            for (int c = 0; c < VDim; c++)
            {
               BBu(qx,qy,dz)(c) = bbu[c];
               GBu(qx,qy,dz)(c) = gbu[c];
               BGu(qx,qy,dz)(c) = bgu[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim,3>,Q1d,Q1d,Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(qz,z,D1d)
         {
            double gbbu[VDim];
            double bgbu[VDim];
            double bbgu[VDim];
            for (int c = 0; c < VDim; c++)
            {
               gbbu[c] = 0.0;
               bgbu[c] = 0.0;
               bbgu[c] = 0.0;
            }
            for (int dz = 0; dz < D1d; ++dz)
            {
               const double b = B(qz,dz);
               const double g = G(qz,dz);
               for (int c = 0; c < VDim; c++)
               {
                  const double bbu = BBu(qx,qy,dz)(c);
                  const double gbu = GBu(qx,qy,dz)(c);
                  const double bgu = BGu(qx,qy,dz)(c);
                  gbbu[c] += g * bbu;
                  bgbu[c] += b * gbu;
                  bbgu[c] += b * bgu;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               gu_q(qx,qy,qz)(c,0) = bbgu[c];
               gu_q(qx,qy,qz)(c,1) = bgbu[c];
               gu_q(qx,qy,qz)(c,2) = gbbu[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// 2D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
Tensor<dTensor<2>,Q1d,Q1d>&& Gradient(const dTensor<Q1d,D1d> &B,
                                      const dTensor<Q1d,D1d> &G,
                                      const dTensor<D1d,D1d> &u)
{
   dTensor<Q1d,D1d> Bu;
   dTensor<Q1d,D1d> Gu;
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         double bu = 0.0;
         double gu = 0.0;
         for (int dx = 0; dx < D1d; ++dx)
         {
            const double b = B(qx,dx);
            const double g = G(qx,dx);
            const double x = u(dx,dy);
            bu += b * x;
            gu += g * x;
         }
         Bu(qx,dy) = bu;
         Gu(qx,dy) = gu;
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<2>,Q1d,Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         double bgu = 0.0;
         double gbu = 0.0;
         for (int dy = 0; dy < D1d; ++dy)
         {
            const double b = B(qy,dy);
            const double g = G(qy,dy);
            const double bu = Bu(qx,dy);
            const double gu = Gu(qx,dy);
            gbu += g * bu;
            bgu += b * gu;
         }
         gu_q(qx,qy)(0) = bgu;
         gu_q(qx,qy)(1) = gbu;
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// 2D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim,2>,Q1d,Q1d>&& Gradient(const dTensor<Q1d,D1d> &B,
                                           const dTensor<Q1d,D1d> &G,
                                           const Tensor<dTensor<VDim>,D1d,D1d> &u)
{
   Tensor<dTensor<VDim>,Q1d,D1d> Bu;
   Tensor<dTensor<VDim>,Q1d,D1d> Gu;
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         double bu[VDim];
         double gu[VDim];
         for (int c = 0; c < VDim; c++)
         {
            bu[c] = 0.0;
            gu[c] = 0.0;
         }
         for (int dx = 0; dx < D1d; ++dx)
         {
            const double b = B(qx,dx);
            const double g = G(qx,dx);
            for (int c = 0; c < VDim; c++)
            {
               const double x = u(dx,dy)(c);
               bu[c] += b * x;
               gu[c] += g * x;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            Bu(qx,dy)(c) = bu[c];
            Gu(qx,dy)(c) = gu[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim,2>,Q1d,Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         double bgu[VDim];
         double gbu[VDim];
         for (int c = 0; c < VDim; c++)
         {
            bgu[c] = 0.0;
            gbu[c] = 0.0;
         }
         for (int dy = 0; dy < D1d; ++dy)
         {
            const double b = B(qy,dy);
            const double g = G(qy,dy);
            for (int c = 0; c < VDim; c++)
            {
               const double bu = Bu(qx,dy)(c);
               const double gu = Gu(qx,dy)(c);
               gbu[c] += g * bu;
               bgu[c] += b * gu;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            gu_q(qx,qy)(c,0) = bgu[c];
            gu_q(qx,qy)(c,1) = gbu[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// 1D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d>&& Gradient(const dTensor<Q1d,D1d> &B,
                        const dTensor<Q1d,D1d> &G,
                        const dTensor<D1d> &u)
{
   dTensor<Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      double gu = 0.0;
      for (int dx = 0; dx < D1d; ++dx)
      {
         const double g = G(qx,dx);
         const double x = u(dx);
         gu += g * x;
      }
      gu_q(qx) = gu;
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// 1D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,Q1d>&& Gradient(const dTensor<Q1d,D1d> &B,
                                     const dTensor<Q1d,D1d> &G,
                                     const Tensor<dTensor<VDim>,D1d> &u)
{
   Tensor<dTensor<VDim>,Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      double gu[VDim];
      for (int c = 0; c < VDim; c++)
      {
         gu[c] = 0.0;
      }
      for (int dx = 0; dx < D1d; ++dx)
      {
         const double g = G(qx,dx);
         for (int c = 0; c < VDim; c++)
         {
            const double x = u(dx)(c);
            gu[c] += g * x;
         }
      }
      for (int c = 0; c < VDim; c++)
      {
         gu_q(qx)(c) = gu[c];
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu_q);
}

// Functions to interpolate the gradient from degrees of freedom to derivatives
// at quadrature points.
// Non-tensor case
template<int Q, int D, int Dim> MFEM_HOST_DEVICE inline
dTensor<D>&& GradientT(const dTensor<Q,D> &B,
                       const Tensor<dTensor<Dim>,Q,D> &G,
                       const Tensor<dTensor<Dim>,Q> &u_q)
{
   dTensor<D> gu;
   MFEM_FOREACH_THREAD(d,x,D)
   {
      double val = 0.0;
      for (int q = 0; q < Q; ++q)
      {
         for (int s = 0; s < Dim; s++)
         {
            const double x = u_q(q)(s);
            const double g = G(q,d)(s);
            val += g * x;
         }
      }
      gu(d) = val;
   }
   MFEM_SYNC_THREAD;
   return std::move(gu);
}

// Non-tensor case with VDim components
template<int D, int Q, int Dim, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,D>&& GradientT(const dTensor<Q,D> &B,
                                    const Tensor<dTensor<Dim>,Q,D> &G,
                                    const Tensor<dTensor<VDim,Dim>,Q> &u_q)
{
   Tensor<dTensor<VDim>,D> gu;
   MFEM_FOREACH_THREAD(d,x,D)
   {
      double v[VDim];
      double b[Dim];
      for (int c = 0; c < VDim; c++)
      {
         v[c] = 0.0;
      }
      for (int q = 0; q < Q; ++q)
      {
         for (int s = 0; s < Dim; s++)
         {
            b[s] = G(q,d)(s);
         }
         for (int s = 0; s < Dim; s++)
         {
            for (int c = 0; c < VDim; c++)
            {
               const double x = u_q(q)(s,c);
               v[c] += b[s] * x;
            }
         }
      }
      for (int c = 0; c < VDim; c++)
      {
         gu(d)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu);
}

// 3D Tensor case
template<int D1d, int Q1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d,D1d>&& GradientT(const dTensor<Q1d,D1d> &B,
                                 const dTensor<Q1d,D1d> &G,
                                 const Tensor<dTensor<3>,Q1d,Q1d,Q1d> &u_q)
{
   dTensor<D1d,Q1d,Q1d> Gux;
   dTensor<D1d,Q1d,Q1d> Buy;
   dTensor<D1d,Q1d,Q1d> Buz;
   for (int qz = 0; qz < Q1d; qz++)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            double gux = 0.0;
            double buy = 0.0;
            double buz = 0.0;
            for (int qx = 0; qx < Q1d; ++qx)
            {
               const double b = B(qx,dx);
               const double g = G(qx,dx);
               gux += g * u_q(qx,qy,qz)(0);
               buy += b * u_q(qx,qy,qz)(1);
               buz += b * u_q(qx,qy,qz)(2);
            }
            Gux(dx,qy,qz) = gux;
            Buy(dx,qy,qz) = buy;
            Buz(dx,qy,qz) = buz;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<D1d,D1d,Q1d> BGux;
   dTensor<D1d,D1d,Q1d> GBuy;
   dTensor<D1d,D1d,Q1d> BBuz;
   for (int qz = 0; qz < Q1d; qz++)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            double bgux = 0.0;
            double gbuy = 0.0;
            double bbuz = 0.0;
            for (int qy = 0; qy < Q1d; ++qy)
            {
               const double b = B(qy,dy);
               const double g = G(qy,dy);
               bgux += b * Gux(dx,qy,qz);
               gbuy += g * Buy(dx,qy,qz);
               bbuz += b * Buz(dx,qy,qz);
            }
            BGux(dx,dy,qz) = bgux;
            GBuy(dx,dy,qz) = gbuy;
            BBuz(dx,dy,qz) = bbuz;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<D1d,D1d,D1d> gu;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         for (int dz = 0; dz < D1d; dz++)
         {
            double val = 0.0;
            for (int qz = 0; qz < Q1d; ++qz)
            {
               const double b = B(qz,dz);
               const double g = G(qz,dz);
               const double bgux = BGux(dx,dy,qz);
               const double gbuy = GBuy(dx,dy,qz);
               const double bbuz = BBuz(dx,dy,qz);
               val += b * bgux + b * gbuy + g * bbuz;
            }
            gu(dx,dy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu);
}

// 3D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,D1d,D1d,D1d>&& GradientT(const dTensor<Q1d,D1d> &B,
                                              const dTensor<Q1d,D1d> &G,
                                              const Tensor<dTensor<VDim,3>,Q1d,Q1d,Q1d> &u_q)
{
   Tensor<dTensor<VDim>,D1d,Q1d,Q1d> Gux;
   Tensor<dTensor<VDim>,D1d,Q1d,Q1d> Buy;
   Tensor<dTensor<VDim>,D1d,Q1d,Q1d> Buz;
   for (int qz = 0; qz < Q1d; qz++)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            double gux[VDim];
            double buy[VDim];
            double buz[VDim];
            for (int c = 0; c < VDim; c++)
            {
               gux[c] = 0.0;
               buy[c] = 0.0;
               buz[c] = 0.0;
            }            
            for (int qx = 0; qx < Q1d; ++qx)
            {
               const double b = B(qx,dx);
               const double g = G(qx,dx);
               for (int c = 0; c < VDim; c++)
               {
                  gux[c] += g * u_q(qx,qy,qz)(c,0);
                  buy[c] += b * u_q(qx,qy,qz)(c,1);
                  buz[c] += b * u_q(qx,qy,qz)(c,2);
               }               
            }
            for (int c = 0; c < VDim; c++)
            {
               Gux(dx,qy,qz)(c) = gux[c];
               Buy(dx,qy,qz)(c) = buy[c];
               Buz(dx,qy,qz)(c) = buz[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,D1d,D1d,Q1d> BGux;
   Tensor<dTensor<VDim>,D1d,D1d,Q1d> GBuy;
   Tensor<dTensor<VDim>,D1d,D1d,Q1d> BBuz;
   for (int qz = 0; qz < Q1d; qz++)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            double bgux[VDim];
            double gbuy[VDim];
            double bbuz[VDim];
            for (int c = 0; c < VDim; c++)
            {
               bgux[c] = 0.0;
               gbuy[c] = 0.0;
               bbuz[c] = 0.0;
            }        
            for (int qy = 0; qy < Q1d; ++qy)
            {
               const double b = B(qy,dy);
               const double g = G(qy,dy);
               for (int c = 0; c < VDim; c++)
               {
                  bgux[c] += b * Gux(dx,qy,qz)(c);
                  gbuy[c] += g * Buy(dx,qy,qz)(c);
                  bbuz[c] += b * Buz(dx,qy,qz)(c);
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               BGux(dx,dy,qz)(c) = bgux[c];
               GBuy(dx,dy,qz)(c) = gbuy[c];
               BBuz(dx,dy,qz)(c) = bbuz[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,D1d,D1d,D1d> gu;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         for (int dz = 0; dz < D1d; dz++)
         {
            double val[VDim];
            for (int c = 0; c < VDim; c++)
            {
               val[c] = 0.0;
            }
            for (int qz = 0; qz < Q1d; ++qz)
            {
               const double b = B(qz,dz);
               const double g = G(qz,dz);
               for (int c = 0; c < VDim; c++)
               {
                  const double bgux[c] = BGux(dx,dy,qz)(c);
                  const double gbuy[c] = GBuy(dx,dy,qz)(c);
                  const double bbuz[c] = BBuz(dx,dy,qz)(c);
                  val[c] += b * bgux[c] + b * gbuy[c] + g * bbuz[c];
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               gu(dx,dy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu);
}

// 2D Tensor case
template<int D1d, int Q1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d>&& GradientT(const dTensor<Q1d,D1d> &B,
                             const dTensor<Q1d,D1d> &G,
                             const Tensor<dTensor<2>,Q1d,Q1d> &u_q)
{
   dTensor<D1d,Q1d> Gux;
   dTensor<D1d,Q1d> Buy;
   MFEM_FOREACH_THREAD(qy,y,Q1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         double gux = 0.0;
         double buy = 0.0;
         for (int qx = 0; qx < Q1d; ++qx)
         {
            const double b = B(qx,dx);
            const double g = G(qx,dx);
            gux += g * u_q(qx,qy)(0);
            buy += b * u_q(qx,qy)(1);
         }
         Gux(dx,qy) = gux;
         Buy(dx,qy) = buy;
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<D1d,D1d> gu;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         double bgux = 0.0;
         double gbuy = 0.0;
         double bbuz = 0.0;
         for (int qy = 0; qy < Q1d; ++qy)
         {
            const double b = B(qy,dy);
            const double g = G(qy,dy);
            bgux += b * Gux(dx,qy);
            gbuy += g * Buy(dx,qy);
         }
         gu(dx,dy) = bgux + gbuy;
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu);
}

// 2D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDim>,D1d,D1d>&& GradientT(const dTensor<Q1d,D1d> &B,
                                              const dTensor<Q1d,D1d> &G,
                                              const Tensor<dTensor<VDim,2>,Q1d,Q1d> &u_q)
{
   Tensor<dTensor<VDim>,D1d,Q1d> Gux;
   Tensor<dTensor<VDim>,D1d,Q1d> Buy;
   MFEM_FOREACH_THREAD(qy,y,Q1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         double gux[VDim];
         double buy[VDim];
         for (int c = 0; c < VDim; c++)
         {
            gux[c] = 0.0;
            buy[c] = 0.0;
         }            
         for (int qx = 0; qx < Q1d; ++qx)
         {
            const double b = B(qx,dx);
            const double g = G(qx,dx);
            for (int c = 0; c < VDim; c++)
            {
               gux[c] += g * u_q(qx,qy)(c,0);
               buy[c] += b * u_q(qx,qy)(c,1);
            }               
         }
         for (int c = 0; c < VDim; c++)
         {
            Gux(dx,qy)(c) = gux[c];
            Buy(dx,qy)(c) = buy[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDim>,D1d,D1d> gu;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         double bgux[VDim];
         double gbuy[VDim];
         for (int c = 0; c < VDim; c++)
         {
            bgux[c] = 0.0;
            gbuy[c] = 0.0;
         }        
         for (int qy = 0; qy < Q1d; ++qy)
         {
            const double b = B(qy,dy);
            const double g = G(qy,dy);
            for (int c = 0; c < VDim; c++)
            {
               bgux[c] += b * Gux(dx,qy)(c);
               gbuy[c] += g * Buy(dx,qy)(c);
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            gu(dx,dy)(c) = bgux[c] + gbuy[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(gu);
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD
