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

#ifndef MFEM_CONTRACTION
#define MFEM_CONTRACTION

#include "tensor.hpp"

namespace mfem
{
auto ContractX1D(const DynamicSharedDTensor<2> &B,
                 const DynamicDTensor<1> &u)
{
   const int Q = B.template Size<0>();
   const int D = B.template Size<1>();
   DynamicDTensor<1> Bu(Q);
   // TODO Abstract for_each(int q )
   for(int q = 0; q < Q; ++q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = u(d);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template <int D, int Q>
auto ContractX1D(const StaticSharedDTensor<Q,D> &B,
                 const StaticDTensor<D> &u)
{
   StaticDTensor<Q> Bu;
   for(int q = 0; q < Q; ++q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = u(d);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template <int D, int Q>
auto ContractX1D(const StaticSharedDTensor<Q,D> &B,
                 const StaticBlockDTensor<D> &u)
{
   StaticBlockDTensor<Q> Bu;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = u(d);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template <int BatchSize>
auto ContractX1D(const DynamicSharedDTensor<2> &B,
                 const DynamicBlockDTensor<1,BatchSize> &u)
{
   const int Q = B.template Size<0>();
   const int D = B.template Size<1>();
   DynamicBlockDTensor<1,BatchSize> Bu(Q);
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = u(d);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

// template <typename Basis, >
// auto ContractX1D(const Basis &B,
//                  const StaticBlockDTensor<D> &u)
// {
//    constexpr Q = Basis.Q;
//    StaticBlockDTensor<Q> Bu;
//    MFEM_FOREACH_THREAD(q,x,Q)
//    {
//       double v = 0.0;
//       for (int d = 0; d < D; ++d)
//       {
//          const double b = B(q,d);
//          const double x = u(d);
//          v += b * x;
//       }
//       Bu(q) = v;
//    }
//    MFEM_SYNC_THREAD;
//    return Bu;
// }

////////
// 1D //
template<int D, int Q> MFEM_HOST_DEVICE inline
dTensor<Q> ContractX1D(const dTensor<Q,D> &B,
                       const dTensor<D> &u)
{
   dTensor<Q> Bu;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = u(d);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int P, int Q> MFEM_HOST_DEVICE inline
dTensor<P> ContractTX1D(const dTensor<Q,P> &B,
                        const dTensor<Q> &u)
{
   dTensor<Q> Bu;
   MFEM_FOREACH_THREAD(d,x,P)
   {
      double v = 0.0;
      for (int q = 0; q < Q; ++q)
      {
         const double b = B(q,d);
         const double x = u(q);
         v += b * x;
      }
      Bu(d) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q, int D, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q> ContractX1D(const dTensor<Q,D> &B,
                                          const StaticTensor<dTensor<VDim>,D> &u)
{
   StaticTensor<dTensor<VDim>,Q> Bu(Q);
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
         Bu(q)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q, int P, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,P> ContractTX1D(const dTensor<Q,P> &B,
                                           const StaticTensor<dTensor<VDim>,Q> &u)
{
   StaticTensor<dTensor<VDim>,P> Bu;
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
            const double x = u(q)(c);
            v[c] += b * x;
         }
      }
      for (int c = 0; c < VDim; c++)
      {
         Bu(d)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

////////
// 2D //
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,D1d> ContractX2D(const dTensor<Q1d,D1d> &B,
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
   return Bu;
}

template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,Q1d> ContractTX2D(const dTensor<Q1d,D1d> &B,
                              const dTensor<Q1d,Q1d> &u)
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
            const double x = u(qx,qy);
            val += b * x;
         }
         Bu(dx,qy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d> ContractY2D(const dTensor<Q1d,D1d> &B,
                             const dTensor<Q1d,D1d> &u)
{
   dTensor<Q1d,Q1d> Bu;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         double val = 0.0;
         for (int dy = 0; dy < D1d; ++dy)
         {
            const double b = B(qy,dy);
            const double x = u(qx,dy);
            val += b * x;
         }
         Bu(qx,qy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d> ContractTY2D(const dTensor<Q1d,D1d> &B,
                              const dTensor<D1d,Q1d> &u)
{
   dTensor<D1d,D1d> Bu;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         double val = 0.0;
         for (int qy = 0; qy < Q1d; ++qy)
         {
            const double b = B(qy,dy);
            const double x = u(dx,qy);
            val += b * x;
         }
         Bu(dx,dy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d,D1d> ContractX2D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,D1d,D1d> &u)
{
   StaticTensor<dTensor<VDim>,Q1d,D1d> Bu;
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
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,Q1d> ContractTX2D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,Q1d> &u)
{
   StaticTensor<dTensor<VDim>,D1d,Q1d> Bu;
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
               const double x = u(qx,qy)(c);
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
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d,Q1d> ContractY2D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,D1d> &u)
{
   StaticTensor<dTensor<VDim>,Q1d,Q1d> Bu;
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
               const double x = u(qx,dy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            Bu(qx,qy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,Q1d> ContractTY2D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,Q1d> &u)
{
   StaticTensor<dTensor<VDim>,D1d,D1d> Bu;
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
               const double x = u(dx,qy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDim; c++)
         {
            Bu(dx,dy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

////////
// 3D //
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,D1d,D1d> ContractX3D(const dTensor<Q1d,D1d> &B,
                                 const dTensor<D1d,D1d,D1d> &u)
{
   dTensor<Q1d,D1d,D1d> Bu;
   for (int dz = 0; dz < D1d; dz++)
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
   return Bu;
}


template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,Q1d,Q1d> ContractTX3D(const dTensor<Q1d,D1d> &B,
                                  const dTensor<Q1d,Q1d,Q1d> &u)
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
               const double x = u(qx,qy,qz);
               val += b * x;
            }
            Bu(dx,qy,qz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d,D1d> ContractY3D(const dTensor<Q1d,D1d> &B,
                                 const dTensor<Q1d,D1d,D1d> &u)
{
   dTensor<Q1d,Q1d,D1d> Bu;
   for (int dz = 0; dz < D1d; dz++)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1d)
         {
            double val = 0.0;
            for (int dy = 0; dy < D1d; ++dy)
            {
               const double b = B(qy,dy);
               const double x = u(qx,dy,dz);
               val += b * x;
            }
            Bu(qx,qy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d,Q1d> ContractTY3D(const dTensor<Q1d,D1d> &B,
                                  const dTensor<D1d,Q1d,Q1d> &u)
{
   dTensor<D1d,D1d,Q1d> Bu;
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
               const double x = u(dx,qy,qz);
               val += b * x;
            }
            Bu(dx,dy,qz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d,Q1d> ContractZ3D(const dTensor<Q1d,D1d> &B,
                                 const dTensor<Q1d,Q1d,D1d> &u)
{
   dTensor<Q1d,Q1d,Q1d> Bu;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         for (int qz = 0; qz < Q1d; qz++)
         {
            double val = 0.0;
            for (int dz = 0; dz < D1d; ++dz)
            {
               const double b = B(qz,dz);
               const double x = u(qx,qy,dz);
               val += b * x;
            }
            Bu(qx,qy,qz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d,D1d> ContractTZ3D(const dTensor<Q1d,D1d> &B,
                                  const dTensor<D1d,D1d,Q1d> &u)
{
   dTensor<D1d,D1d,D1d> Bu;
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
               const double x = u(dx,dy,qz);
               val += b * x;
            }
            Bu(dx,dy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d,D1d,D1d> ContractX3D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,D1d,D1d,D1d> &u)
{
   StaticTensor<dTensor<VDim>,Q1d,D1d,D1d> Bu;
   for (int dz = 0; dz < D1d; dz++)
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
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,Q1d,Q1d> ContractTX3D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,Q1d,Q1d> &u)
{
   StaticTensor<dTensor<VDim>,D1d,Q1d,Q1d> Bu;
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
                  const double x = u(qx,qy,qz)(c);
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
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d,Q1d,D1d> ContractY3D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,D1d,D1d> &u)
{
   StaticTensor<dTensor<VDim>,Q1d,Q1d,D1d> Bu;
   for (int dz = 0; dz < D1d; dz++)
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
                  const double x = u(qx,dy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               Bu(qx,qy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,D1d,Q1d> ContractTY3D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,D1d,Q1d,Q1d> &u)
{
   StaticTensor<dTensor<VDim>,D1d,D1d,Q1d> Bu;
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
                  const double x = u(dx,qy,qz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               Bu(dx,dy,qz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d,Q1d,Q1d> ContractZ3D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,Q1d,D1d> &u)
{
   StaticTensor<dTensor<VDim>,Q1d,Q1d,Q1d> Bu;
   MFEM_FOREACH_THREAD(qx,x,Q1d)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         for (int qz = 0; qz < Q1d; qz++)
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
                  const double x = u(qx,qy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               Bu(qx,qy,qz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}


template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,D1d,Q1d> ContractTZ3D(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,D1d,Q1d,Q1d> &u)
{
   StaticTensor<dTensor<VDim>,D1d,D1d,D1d> Bu;
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
                  const double x = u(dx,dy,qz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDim; c++)
            {
               Bu(dx,dy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

////////////////
// Non-tensor //
template<int D, int Q, int Dim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<Dim>,Q> Contract(const StaticTensor<dTensor<Dim>,Q,D> &G,
                                      const dTensor<D> &u)
{
   StaticTensor<dTensor<Dim>,Q> Gu_q;
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
         Gu_q(q)(c) = v[c];
      }
   }
   MFEM_SYNC_THREAD;
   return Gu_q;
}

template<int D, int Q, int Dim> MFEM_HOST_DEVICE inline
dTensor<D> ContractT(const StaticTensor<dTensor<Dim>,Q,D> &G,
                     const StaticTensor<dTensor<Dim>,Q> &u)
{
   dTensor<D> gu;
   MFEM_FOREACH_THREAD(d,x,D)
   {
      double val = 0.0;
      for (int q = 0; q < Q; ++q)
      {
         for (int s = 0; s < Dim; s++)
         {
            const double x = u(q)(s);
            const double g = G(q,d)(s);
            val += g * x;
         }
      }
      gu(d) = val;
   }
   MFEM_SYNC_THREAD;
   return gu;
}

template<int Q, int D, int Dim, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<Dim,VDim>,Q> Contract(
   const StaticTensor<dTensor<Dim>,Q,D> &G,
   const StaticTensor<dTensor<VDim>,D> &u)
{
   StaticTensor<dTensor<Dim,VDim>,Q> Gu_q;
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
            Gu_q(q)(s,c) = v[s][c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Gu_q;
}

template<int Q, int D, int Dim, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<Dim,VDim>,D> ContractT(
   const StaticTensor<dTensor<Dim>,Q,D> &G,
   const StaticTensor<dTensor<VDim>,Q> &u)
{
   StaticTensor<dTensor<VDim>,D> gu;
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
               const double x = u(q)(s,c);
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
   return gu;
}

} // namespace mfem

#endif // MFEM_CONTRACTION
