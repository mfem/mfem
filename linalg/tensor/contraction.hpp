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
#include "basis.hpp"

namespace mfem
{

// 1D
/// Contraction on X dimension
MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<1> &B, const DynamicDTensor<1> &u)
{
   const int Q = B.Size<0>();
   const int D = B.Size<1>();
   DynamicDTensor<1> Bu(Q);
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
   return Bu;
}

/// Contraction on X dimension
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<1> &B,
               const DynamicBlockDTensor<1,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int D = B.Size<1>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<1,BatchSize> Bu(Q);
   MFEM_SHARED DynamicDTensor<2> slice(D,BatchSize);
   MFEM_FOREACH_THREAD(d,x,D)
   {
      slice(d,tid) = u(d);
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = slice(d,tid);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on X dimension
template <int D, int Q> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<1,Q,D> &B, const StaticDTensor<D> &u)
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
   return Bu;
}

/// Contraction on X dimension
template <int BatchSize, int D, int Q> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<1,Q,D> &B,
               const StaticBlockDTensor<BatchSize,D> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q> Bu;
   MFEM_SHARED StaticDTensor<D,BatchSize> slice;
   MFEM_FOREACH_THREAD(d,x,D)
   {
      slice(d,tid) = u(d);
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         const double x = slice(d,tid);
         v += b * x;
      }
      Bu(q) = v;
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/////////////
/// With Vdim

/// Contraction on X dimension with VDim
MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<1> &B, const DynamicDTensor<2> &u)
{
   const int Q = B.Size<0>();
   const int D = B.Size<1>();
   const int VDim = u.Size<1>();
   DynamicDTensor<2> Bu(Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = B(q,d);
            const double x = u(d,c);
            v += b * x;
         }
         Bu(q,c) = v;
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<1> &B,
               const DynamicBlockDTensor<2,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int D = B.Size<1>();
   const int VDim = u.template Size<1>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<2,BatchSize> Bu(Q,VDim); // TODO might be a problem
   MFEM_SHARED DynamicDTensor<3> slice(D,VDim,BatchSize);
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(d,x,D)
      {
         slice(d,c,tid) = u(d,c);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = B(q,d);
            const double x = slice(d,c,tid);
            v += b * x;
         }
         Bu(q,c) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on X dimension with VDim
template <int D, int Q, int VDim> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<1,Q,D> &B,
               const StaticDTensor<D,VDim> &u)
{
   StaticDTensor<Q,VDim> Bu;
   for(int q = 0; q < Q; ++q)
   {
      StaticDTensor<VDim> v = 0.0;
      for (int d = 0; d < D; ++d)
      {
         const double b = B(q,d);
         for(int c = 0; c< VDim; ++c)
         {
            const double x = u(d,c);
            v(c) += b * x;
         }
      }
      for(int c = 0; c< VDim; ++c)
      {
         Bu(q,c) = v(c);
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int D, int Q, int VDim, int BatchSize>
auto ContractX(const StaticBasisTensor<1,Q,D> &B,
               const StaticBlockDTensor<BatchSize,D,VDim> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,VDim> Bu;
   MFEM_SHARED StaticDTensor<D,VDim,BatchSize> slice;
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(d,x,D)
      {
         slice(d,c,tid) = u(d,c);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(c,y,VDim)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = B(q,d);
            const double x = slice(d,c,tid);
            v += b * x;
         }
         Bu(q,c) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

// 2D

/// Contraction on X dimension
MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<2> &B,
               const DynamicDTensor<2> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.Size<1>();
   DynamicDTensor<2> Bu(Q,Dy);
   for (int dy = 0; dy < Dy; dy++)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            const double x = u(dx,dy);
            v += b * x;
         }
         Bu(q,dy) = v;
      }
   }
   return Bu;
}

/// Contraction on X dimension
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<2> &B,
               const DynamicBlockDTensor<2,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.template Size<1>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<2,BatchSize> Bu(Q,Dy);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,tid) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            const double x = slice(dx,dy,tid);
            v += b * x;
         }
         Bu(q,dy) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on X dimension
template <int Dx, int Dy, int Q> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<2,Q,Dx> &B,
               const StaticDTensor<Dx,Dy> &u)
{
   StaticDTensor<Q,Dy> Bu;
   for (int dy = 0; dy < Dy; dy++)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            const double x = u(dx,dy);
            v += b * x;
         }
         Bu(q,dy) = v;
      }
   }
   return Bu;
}

/// Contraction on X dimension
template <int Dx, int Dy, int Q, int BatchSize> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<2,Q,Dx> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,tid) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            const double x = slice(dx,dy,tid);
            v += b * x;
         }
         Bu(q,dy) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on Y dimension
MFEM_HOST_DEVICE inline
auto ContractY(const DynamicBasisTensor<2> &B, const DynamicDTensor<2> &u)
{
   const int Q = B.Size<0>();
   const int Dy = B.Size<1>();
   const int Dx = u.Size<0>();
   DynamicDTensor<2> Bu(Dx,Q);
   for (int dx = 0; dx < Dx; dx++)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            const double x = u(dx,dy);
            v += b * x;
         }
         Bu(dx,q) = v;
      }
   }
   return Bu;
}

/// Contraction on Y dimension
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractY(const DynamicBasisTensor<2> &B,
               const DynamicBlockDTensor<2,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dy = B.Size<1>();
   const int Dx = u.template Size<0>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<2,BatchSize> Bu(Dx,Q);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,tid) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,Dx)
   {
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         double v = 0.0;
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            const double x = slice(dx,dy,tid);
            v += b * x;
         }
         Bu(dx,q) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on Y dimension
template <int Dx, int Dy, int Q> MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<2,Q,Dy> &B,
               const StaticDTensor<Dx,Dy> &u)
{
   StaticDTensor<Dx,Q> Bu;
   for (int dx = 0; dx < Dx; dx++)
   {
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            const double x = u(dx,dy);
            v += b * x;
         }
         Bu(dx,q) = v;
      }
   }
   return Bu;
}

/// Contraction on Y dimension
template <int Dx, int Dy, int Q, int BatchSize> MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<2,Q,Dy> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         slice(dx,dy,tid) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,Dx)
   {
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         double v = 0.0;
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            const double x = slice(dx,dy,tid);
            v += b * x;
         }
         Bu(dx,q) = v;
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/////////////
/// With Vdim

/// Contraction on X dimension with VDim
MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<2> &B, const DynamicDTensor<3> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.Size<1>();
   const int VDim = u.Size<2>();
   DynamicDTensor<3> Bu(Q,Dy,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = u(dx,dy,c);
               v += b * x;
            }
            Bu(q,dy,c) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<2> &B,
               const DynamicBlockDTensor<3,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.template Size<1>();
   const int VDim = u.template Size<2>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Q,Dy,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,tid) = u(dx,dy,c);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = slice(dx,dy,tid);
               v += b * x;
            }
            Bu(q,dy,c) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int Dx, int Dy, int Q, int VDim> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<2,Q,Dx> &B,
               const StaticDTensor<Dx,Dy,VDim> &u)
{
   StaticDTensor<Q,Dy,VDim> Bu;
   for(int c = 0; c < VDim; ++c)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = u(dx,dy,c);
               v += b * x;
            }
            Bu(q,dy,c) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int Dx, int Dy, int Q, int VDim, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<2,Q,Dx> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,VDim> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,VDim,BatchSize> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         for(int c = 0; c < VDim; ++c)
         {
            slice(dx,dy,c,tid) = u(dx,dy,c);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         StaticDTensor<VDim> v;
         v = 0.0;
         for (int dx = 0; dx < Dx; ++dx)
         {
            const double b = B(q,dx);
            for(int c = 0; c < VDim; ++c)
            {
               const double x = slice(dx,dy,c,tid);
               v(c) += b * x;
            }
         }
         for(int c = 0; c < VDim; ++c)
         {
            Bu(q,dy,c) = v(c);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

/// Contraction on Y dimension with VDim
MFEM_HOST_DEVICE inline
auto ContractY(const DynamicBasisTensor<2> &B, const DynamicDTensor<3> &u)
{
   const int Q = B.Size<0>();
   const int Dy = B.Size<1>();
   const int Dx = u.Size<0>();
   const int VDim = u.Size<2>();
   DynamicDTensor<3> Bu(Dx,Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = u(dx,dy,c);
               v += b * x;
            }
            Bu(dx,q,c) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractY(const DynamicBasisTensor<2> &B,
               const DynamicBlockDTensor<3,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dy = B.Size<1>();
   const int Dx = u.template Size<0>();
   const int VDim = u.template Size<2>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Dx,Q,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,tid) = u(dx,dy,c);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = slice(dx,dy,tid);
               v += b * x;
            }
            Bu(dx,q,c) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <int Dx, int Dy, int Q, int VDim> MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<2,Q,Dy> &B,
               const StaticDTensor<Dx,Dy,VDim> &u)
{
   StaticDTensor<Dx,Q,VDim> Bu;
   for(int c = 0; c < VDim; ++c)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = u(dx,dy,c);
               v += b * x;
            }
            Bu(dx,q,c) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <int Dx, int Dy, int Q, int VDim, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<2,Q,Dy> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,VDim> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,VDim,BatchSize> slice;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         for(int c = 0; c < VDim; ++c)
         {
            slice(dx,dy,c,tid) = u(dx,dy,c);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,Dx)
   {
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         StaticDTensor<VDim> v;
         v = 0.0;
         for (int dy = 0; dy < Dy; ++dy)
         {
            const double b = B(q,dy);
            for(int c = 0; c < VDim; ++c)
            {
               const double x = slice(dx,dy,c,tid);
               v(c) += b * x;
            }
         }
         for(int c = 0; c < VDim; ++c)
         {
            Bu(dx,q,c) = v(c);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return Bu;
}

// 3D 

/// Contraction on X dimension
MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<3> &B, const DynamicDTensor<3> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.Size<1>();
   const int Dz = u.Size<2>();
   DynamicDTensor<3> Bu(Q,Dy,Dz);
   for (int dz = 0; dz < Dz; dz++)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on X dimension
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<3> &B,
               const DynamicBlockDTensor<3,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Q,Dy,Dz);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,tid) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = slice(dx,dy,tid);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/// Contraction on X dimension
template <int Dx, int Dy, int Dz, int Q> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<3,Q,Dx> &B,
               const StaticDTensor<Dx,Dy,Dz> &u)
{
   StaticDTensor<Q,Dy,Dz> Bu;
   for (int dz = 0; dz < Dz; dz++)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on X dimension
template <int Dx, int Dy, int Dz, int Q, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<3,Q,Dx> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,Dz> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy,Dz> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,tid) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               const double x = slice(dx,dy,tid);
               v += b * x;
            }
            Bu(q,dy,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/// Contraction on Y dimension
MFEM_HOST_DEVICE inline
auto ContractY(const DynamicBasisTensor<3> &B,
               const DynamicDTensor<3> &u)
{
   const int Q = B.Size<0>();
   const int Dx = u.Size<0>();
   const int Dy = B.Size<1>();
   const int Dz = u.Size<2>();
   DynamicDTensor<3> Bu(Dx,Q,Dz);
   for (int dz = 0; dz < Dz; dz++)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractY(const DynamicBasisTensor<3> &B,
               const DynamicBlockDTensor<3,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = B.Size<1>();
   const int Dz = u.template Size<2>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<3,BatchSize> Bu(Dx,Q,Dz);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,tid) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = slice(dx,dy,tid);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/// Contraction on Y dimension
template <int Dx, int Dy, int Dz, int Q> MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<3,Q,Dy> &B,
               const StaticDTensor<Dx,Dy,Dz> &u)
{
   StaticDTensor<Dx,Q,Dz> Bu;
   for (int dz = 0; dz < Dz; dz++)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension
template <int Dx, int Dy, int Dz, int Q, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<3,Q,Dy> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,Dz> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q,Dz> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,BatchSize> slice;
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            slice(dx,dy,tid) = u(dx,dy,dz);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               const double x = slice(dx,dy,tid);
               v += b * x;
            }
            Bu(dx,q,dz) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/// Contraction on Z dimension
MFEM_HOST_DEVICE inline
auto ContractZ(const DynamicBasisTensor<3> &B, const DynamicDTensor<3> &u)
{
   const int Q = B.Size<0>();
   const int Dx = u.Size<0>();
   const int Dy = u.Size<1>();
   const int Dz = B.Size<1>();
   DynamicDTensor<3> Bu(Dx,Dy,Q);
   for (int dy = 0; dy < Dy; dy++)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dz = 0; dz < Dz; ++dz)
            {
               const double b = B(q,dz);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,dy,q) = v;
         }
      }
   }
   return Bu;
}

template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractZ(const DynamicBasisTensor<3> &B,
               const DynamicBlockDTensor<3,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = B.Size<1>();
   DynamicBlockDTensor<3,BatchSize> Bu(Dx,Dy,Q);
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         for (int q = 0; q < Q; q++)
         {
            double v = 0.0;
            for (int dz = 0; dz < Dz; ++dz)
            {
               const double b = B(q,dz);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,dy,q) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Z dimension
template <int Dx, int Dy, int Dz, int Q> MFEM_HOST_DEVICE inline
auto ContractZ(const StaticBasisTensor<3,Q,Dz> &B,
               const StaticDTensor<Dx,Dy,Dz> &u)
{
   StaticDTensor<Dx,Dy,Q> Bu;
   for (int dy = 0; dy < Dy; ++dy)
   {
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dz = 0; dz < Dz; dz++)
            {
               const double b = B(q,dz);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,dy,q) = v;
         }
      }
   }
   return Bu;
}

/// Contraction on Z dimension
template <int Dx, int Dy, int Dz, int Q, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractZ(const StaticBasisTensor<3,Q,Dz> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,Dz> &u)
{
   StaticBlockDTensor<BatchSize,Dx,Dy,Q> Bu;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dz = 0; dz < Dz; dz++)
            {
               const double b = B(q,dz);
               const double x = u(dx,dy,dz);
               v += b * x;
            }
            Bu(dx,dy,q) = v;
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/////////////
/// With Vdim

/// Contraction on X dimension with VDim
MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<3> &B, const DynamicDTensor<4> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.Size<1>();
   const int Dz = u.Size<2>();
   const int VDim = u.Size<3>();
   DynamicDTensor<4> Bu(Q,Dy,Dz,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         for (int dy = 0; dy < Dy; dy++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dx = 0; dx < Dx; ++dx)
               {
                  const double b = B(q,dx);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(q,dy,dz,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractX(const DynamicBasisTensor<3> &B,
               const DynamicBlockDTensor<4,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = B.Size<1>();
   const int Dy = u.template Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = u.template Size<3>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<4,BatchSize> Bu(Q,Dy,Dz,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(dx,x,Dx)
            {
               slice(dx,dy,tid) = u(dx,dy,dz,c);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(q,x,Q)
            {
               double v = 0.0;
               for (int dx = 0; dx < Dx; ++dx)
               {
                  const double b = B(q,dx);
                  const double x = slice(dx,dy,tid);
                  v += b * x;
               }
               Bu(q,dy,dz,c) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int Dx, int Dy, int Dz, int Q, int VDim> MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<3,Q,Dx> &B,
               const StaticDTensor<Dx,Dy,Dz,VDim> &u)
{
   StaticDTensor<Q,Dy,Dz,VDim> Bu;
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         for (int dy = 0; dy < Dy; dy++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dx = 0; dx < Dx; ++dx)
               {
                  const double b = B(q,dx);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(q,dy,dz,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on X dimension with VDim
template <int Dx, int Dy, int Dz, int Q, int VDim, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractX(const StaticBasisTensor<3,Q,Dx> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,Dz,VDim> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Q,Dy,Dz,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,VDim,BatchSize> slice; // TODO invert VDIM and BatchSize?
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            for(int c = 0; c < VDim; ++c)
            {
               slice(dx,dy,c,tid) = u(dx,dy,dz,c);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            StaticDTensor<VDim> v;
            v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = B(q,dx);
               for(int c = 0; c < VDim; ++c)
               {
                  const double x = slice(dx,dy,c,tid);
                  v(c) += b * x;
               }
            }
            for(int c = 0; c < VDim; ++c)
            {
               Bu(q,dy,dz,c) = v(c);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
   }

   /// Contraction on Y dimension with VDim
   MFEM_HOST_DEVICE inline
   auto ContractY(const DynamicBasisTensor<3> &B, const DynamicDTensor<4> &u)
   {
   const int Q = B.Size<0>();
   const int Dx = u.Size<0>();
   const int Dy = B.Size<1>();
   const int Dz = u.Size<2>();
   const int VDim = u.Size<3>();
   DynamicDTensor<4> Bu(Dx,Q,Dz,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         for (int dx = 0; dx < Dx; dx++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dy = 0; dy < Dy; ++dy)
               {
                  const double b = B(q,dy);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(dx,q,dz,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractY(const DynamicBasisTensor<3> &B,
               const DynamicBlockDTensor<4,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = B.Size<1>();
   const int Dz = u.template Size<2>();
   const int VDim = u.template Size<3>();
   const int tid = MFEM_THREAD_ID(z);
   DynamicBlockDTensor<4,BatchSize> Bu(Dx,Q,Dz,VDim);
   MFEM_SHARED DynamicDTensor<3> slice(Dx,Dy,BatchSize);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(dx,x,Dx)
            {
               slice(dx,dy,tid) = u(dx,dy,dz,c);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            MFEM_FOREACH_THREAD(q,y,Q)
            {
               double v = 0.0;
               for (int dy = 0; dy < Dy; ++dy)
               {
                  const double b = B(q,dy);
                  const double x = slice(dx,dy,tid);
                  v += b * x;
               }
               Bu(dx,q,dz,c) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <int Dx, int Dy, int Dz, int Q, int VDim> MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<3,Q,Dy> &B,
               const StaticDTensor<Dx,Dy,Dz,VDim> &u)
{
   StaticDTensor<Dx,Q,Dz,VDim> Bu;
   for(int c = 0; c < VDim; ++c)
   {
      for (int dz = 0; dz < Dz; dz++)
      {
         for (int dx = 0; dx < Dx; dx++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dy = 0; dy < Dy; ++dy)
               {
                  const double b = B(q,dy);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(dx,q,dz,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on Y dimension with VDim
template <int Dx, int Dy, int Dz, int Q, int VDim, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractY(const StaticBasisTensor<3,Q,Dy> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,Dz,VDim> &u)
{
   const int tid = MFEM_THREAD_ID(z);
   StaticBlockDTensor<BatchSize,Dx,Q,Dz,VDim> Bu;
   MFEM_SHARED StaticDTensor<Dx,Dy,VDim,BatchSize> slice;
   for (int dz = 0; dz < Dz; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            for(int c = 0; c < VDim; ++c)
            {
               slice(dx,dy,c,tid) = u(dx,dy,dz,c);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            StaticDTensor<VDim> v;
            v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = B(q,dy);
               for(int c = 0; c < VDim; ++c)
               {
                  const double x = slice(dx,dy,c,tid);
                  v(c) += b * x;
               }
            }
            for(int c = 0; c < VDim; ++c)
            {
               Bu(dx,q,dz,c) = v(c);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}

/// Contraction on Z dimension with VDim
MFEM_HOST_DEVICE inline
auto ContractZ(const DynamicBasisTensor<3> &B, const DynamicDTensor<4> &u)
{
   const int Q = B.Size<0>();
   const int Dx = u.Size<0>();
   const int Dy = u.Size<1>();
   const int Dz = B.Size<1>();
   const int VDim = u.Size<3>();
   DynamicDTensor<4> Bu(Dx,Dy,Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      for (int dy = 0; dy < Dy; dy++)
      {
         for (int dx = 0; dx < Dx; dx++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dz = 0; dz < Dz; ++dz)
               {
                  const double b = B(q,dz);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(dx,dy,q,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on Z dimension with VDim
template <int BatchSize> MFEM_HOST_DEVICE inline
auto ContractZ(const DynamicBasisTensor<3> &B,
               const DynamicBlockDTensor<4,BatchSize> &u)
{
   const int Q = B.Size<0>();
   const int Dx = u.template Size<0>();
   const int Dy = u.template Size<1>();
   const int Dz = B.Size<1>();
   const int VDim = u.template Size<3>();
   DynamicBlockDTensor<4,BatchSize> Bu(Dx,Dy,Q,VDim);
   for(int c = 0; c < VDim; ++c)
   {
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            for (int q = 0; q < Q; q++)
            {
               double v = 0.0;
               for (int dz = 0; dz < Dz; ++dz)
               {
                  const double b = B(q,dz);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(dx,dy,q,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on Z dimension with VDim
template <int Dx, int Dy, int Dz, int Q, int VDim> MFEM_HOST_DEVICE inline
auto ContractZ(const StaticBasisTensor<3,Q,Dz> &B,
               const StaticDTensor<Dx,Dy,Dz> &u)
{
   StaticDTensor<Dx,Dy,Q,VDim> Bu;
   for(int c = 0; c < VDim; ++c)
   {
      for (int dy = 0; dy < Dy; ++dy)
      {
         for (int dx = 0; dx < Dx; dx++)
         {
            for(int q = 0; q < Q; ++q)
            {
               double v = 0.0;
               for (int dz = 0; dz < Dz; dz++)
               {
                  const double b = B(q,dz);
                  const double x = u(dx,dy,dz,c);
                  v += b * x;
               }
               Bu(dx,dy,q,c) = v;
            }
         }
      }
   }
   return Bu;
}

/// Contraction on Z dimension with VDim
template <int Dx, int Dy, int Dz, int Q, int VDim, int BatchSize>
MFEM_HOST_DEVICE inline
auto ContractZ(const StaticBasisTensor<3,Q,Dz> &B,
               const StaticBlockDTensor<BatchSize,Dx,Dy,Dz,VDim> &u)
{
   StaticBlockDTensor<BatchSize,Dx,Dy,Q,VDim> Bu;
   MFEM_FOREACH_THREAD(dy,y,Dy)
   {
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         for(int q = 0; q < Q; ++q)
         {
            StaticDTensor<VDim> v;
            v = 0.0;
            for (int dz = 0; dz < Dz; dz++)
            {
               const double b = B(q,dz);
               for(int c = 0; c < VDim; ++c)
               {
                  const double x = u(dx,dy,dz,c);
                  v(c) += b * x;
               }
            }
            for(int c = 0; c < VDim; ++c)
            {
               Bu(dx,dy,q,c) = v(c);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
   return Bu;
}


//////////////////////
/// Old implementation
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
