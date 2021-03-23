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
#include "basis.hpp"
#include "contraction.hpp"
#include "concatenate.hpp"

namespace mfem
{


// Non-tensor
template <int Dim, int D, int Q, typename Dofs>
auto operator*(const BasisGradient<Dim,false,D,Q> &basis, const Dofs &u)
{
   auto G = basis.GetG();
   return G * u;
}

// 1D Tensor
template <int D, int Q, typename Dofs>
auto operator*(const BasisGradient<1,true,D,Q> &basis, const Dofs &u)
{
   auto G = basis.GetG();
   return ContractX(G,u);
}

// 2D Tensor
template <int D, int Q, typename Dofs>
auto operator*(const BasisGradient<2,true,D,Q> &basis, const Dofs &u)
{
   auto B = basis.GetB();
   auto G = basis.GetG();
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto GBu = ContractY(G,Bu);
   auto BGu = ContractY(B,Gu);
   return Concatenate(BGu,GBu);
}

// 3D Tensor
template <int D, int Q, typename Dofs>
auto operator*(const BasisGradient<3,true,D,Q> &basis, const Dofs &u)
{
   auto B = basis.GetB();
   auto G = basis.GetG();
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto BBu = ContractY(B,Bu);
   auto BGu = ContractY(B,Gu);
   auto GBu = ContractY(G,Bu);
   auto BBGu = ContractZ(B,BGu);
   auto BGBu = ContractZ(B,GBu);
   auto GBBu = ContractZ(G,BBu);
   return Concatenate(BBGu,BGBu,GBBu);
}

// Non-tensor
template <int Dim, int D, int Q, typename Dofs>
auto operator*(const BasisGradientTranspose<Dim,false,D,Q> &basis, const Dofs &u)
{
   auto Gt = basis.GetGt();
   return Gt * u;
}

// 1D Tensor
template <int D, int Q, typename Dofs>
auto operator*(const BasisGradientTranspose<1,true,D,Q> &basis, const Dofs &u)
{
   auto Gt = basis.GetGt();
   return ContractX(Gt,u);
}

// 2D Tensor
template <int D, int Q, typename Dofs>
auto operator*(const BasisGradientTranspose<2,true,D,Q> &basis, const Dofs &u)
{
   // TODO this is completely wrong
   auto Bt = basis.GetBt();
   auto Gt = basis.GetGt();
   auto Bu = ContractX(Bt,u);
   auto Gu = ContractX(Gt,u);
   auto v = ContractY(Gt,Bu);
   v += ContractY(Bt,Gu);
   return v;
}

// 3D Tensor
template <int D, int Q, typename Dofs>
auto operator*(const BasisGradientTranspose<3,true,D,Q> &basis, const Dofs &u)
{
   // TODO this is completely wrong
   auto Bt = basis.GetBt();
   auto Gt = basis.GetGt();
   auto Bu = ContractX(Bt,u);
   auto Gu = ContractX(Gt,u);
   auto BBu = ContractY(Bt,Bu);
   auto GBu = ContractY(Gt,Bu);
   auto BGu = ContractY(Bt,Gu);
   auto v = ContractZ(Gt,BBu);
   v += ContractZ(Bt,GBu);
   v += ContractZ(Bt,BGu);
   return v;
}


/////////////////////
// Old implementation
/*
// Functions to interpolate the gradient from degrees of freedom to derivatives
// at quadrature points.
// Non-tensor case
template<int D, int Q, int Dim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<Dim>,Q> Gradient(const dTensor<Q,D> &B,
                                const StaticTensor<dTensor<Dim>,Q,D> &G,
                                const dTensor<D> &u)
{
   return Contract(G, u);
}

// Non-tensor case with VDim components
template<int Q, int D, int Dim, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<Dim,VDim>,Q> Gradient(const dTensor<Q,D> &B,
                                     const StaticTensor<dTensor<Dim>,Q,D> &G,
                                     const StaticTensor<dTensor<VDim>,D> &u)
{
   return Contract(G, u);
}

// 3D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<3>,Q1d,Q1d,Q1d> Gradient(const dTensor<Q1d,D1d> &B,
                                        const dTensor<Q1d,D1d> &G,
                                        const dTensor<D1d,D1d,D1d> &u)
{
   auto Bu   = ContractX3D(B, u);
   auto Gu   = ContractX3D(G, u);
   auto BBu  = ContractY3D(B, Bu);
   auto GBu  = ContractY3D(G, Bu);
   auto BGu  = ContractY3D(B, Gu);
   auto GBBu = ContractZ3D(G, BBu);
   auto BGBu = ContractZ3D(B, GBu);
   auto BBGu = ContractZ3D(B, BGu);
   StaticTensor<dTensor<3>,Q1d,Q1d,Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         for (int qz = 0; qz < Q1d; qz++)
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
   return gu_q;
}

// 3D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim,3>,Q1d,Q1d,Q1d> Gradient(const dTensor<Q1d,D1d> &B,
                                             const dTensor<Q1d,D1d> &G,
                                             const dTensor<D1d,D1d,D1d> &u)
{
   auto Bu = ContractX3D(B, u);
   auto Gu = ContractX3D(G, u);
   auto BBu = ContractY3D(B, Bu);
   auto GBu = ContractY3D(G, Bu);
   auto BGu = ContractY3D(B, Gu);
   StaticTensor<dTensor<VDim,3>,Q1d,Q1d,Q1d> gu_q;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         for (int qz = 0; qz < Q1d; qz++)
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
   return gu_q;
}

// 2D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<2>,Q1d,Q1d> Gradient(const dTensor<Q1d,D1d> &B,
                                    const dTensor<Q1d,D1d> &G,
                                    const dTensor<D1d,D1d> &u)
{
   auto Bu = ContractX2D(B, u);
   auto Gu = ContractX2D(G, u);
   StaticTensor<dTensor<2>,Q1d,Q1d> gu_q;
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
   return gu_q;
}

// 2D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim,2>,Q1d,Q1d> Gradient(const dTensor<Q1d,D1d> &B,
                                         const dTensor<Q1d,D1d> &G,
                                         const StaticTensor<dTensor<VDim>,D1d,D1d> &u)
{
   auto Bu = ContractX2D(B, u);
   auto Gu = ContractX2D(G, u);
   StaticTensor<dTensor<VDim,2>,Q1d,Q1d> gu_q;
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
   return gu_q;
}

// 1D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d> Gradient(const dTensor<Q1d,D1d> &B,
                      const dTensor<Q1d,D1d> &G,
                      const dTensor<D1d> &u)
{
   return ContractX1D(G, u);
}

// 1D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d> Gradient(const dTensor<Q1d,D1d> &B,
                                   const dTensor<Q1d,D1d> &G,
                                   const StaticTensor<dTensor<VDim>,D1d> &u)
{
   return ContractX1D(G, u);
}

// Functions to interpolate the gradient from degrees of freedom to derivatives
// at quadrature points.
// Non-tensor case
template<int Q, int D, int Dim> MFEM_HOST_DEVICE inline
dTensor<D> GradientT(const dTensor<Q,D> &B,
                     const StaticTensor<dTensor<Dim>,Q,D> &G,
                     const StaticTensor<dTensor<Dim>,Q> &u)
{
   return ContractT(G, u);
}

// Non-tensor case with VDim components
template<int D, int Q, int Dim, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D> GradientT(const dTensor<Q,D> &B,
                                  const StaticTensor<dTensor<Dim>,Q,D> &G,
                                  const StaticTensor<dTensor<VDim,Dim>,Q> &u)
{
   return ContractT(G, u);
}

// 3D Tensor case
template<int D1d, int Q1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d,D1d> GradientT(const dTensor<Q1d,D1d> &B,
                               const dTensor<Q1d,D1d> &G,
                               const StaticTensor<dTensor<3>,Q1d,Q1d,Q1d> &u_q)
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
   return gu;
}

// 3D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,D1d,D1d> GradientT(const dTensor<Q1d,D1d> &B,
                                            const dTensor<Q1d,D1d> &G,
                                            const StaticTensor<dTensor<VDim,3>,Q1d,Q1d,Q1d> &u_q)
{
   StaticTensor<dTensor<VDim>,D1d,Q1d,Q1d> Gux;
   StaticTensor<dTensor<VDim>,D1d,Q1d,Q1d> Buy;
   StaticTensor<dTensor<VDim>,D1d,Q1d,Q1d> Buz;
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
   StaticTensor<dTensor<VDim>,D1d,D1d,Q1d> BGux;
   StaticTensor<dTensor<VDim>,D1d,D1d,Q1d> GBuy;
   StaticTensor<dTensor<VDim>,D1d,D1d,Q1d> BBuz;
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
   StaticTensor<dTensor<VDim>,D1d,D1d,D1d> gu;
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
   return gu;
}

// 2D Tensor case
template<int D1d, int Q1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d> GradientT(const dTensor<Q1d,D1d> &B,
                           const dTensor<Q1d,D1d> &G,
                           const StaticTensor<dTensor<2>,Q1d,Q1d> &u_q)
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
   return gu;
}

// 2D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,D1d> GradientT(const dTensor<Q1d,D1d> &B,
                                        const dTensor<Q1d,D1d> &G,
                                        const StaticTensor<dTensor<VDim,2>,Q1d,Q1d> &u_q)
{
   StaticTensor<dTensor<VDim>,D1d,Q1d> Gux;
   StaticTensor<dTensor<VDim>,D1d,Q1d> Buy;
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
   StaticTensor<dTensor<VDim>,D1d,D1d> gu;
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
   return gu;
}
*/
} // namespace mfem

#endif // MFEM_TENSOR_GRAD
