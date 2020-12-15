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

#ifndef MFEM_BASIS
#define MFEM_BASIS

#include "util.hpp"
#include "tensor.hpp"
#include "config.hpp"

namespace mfem
{

template <int Dim, typename Basis>
class BasisTensor;

template <int Dim>
using DynamicBasisTensor = BasisTensor<Dim,SharedDTensor<2>>;

template <int Dim, int Q, int D>
using StaticBasisTensor = BasisTensor<Dim,StaticSharedDTensor<Q,D>>;

// 1D Dynamic Basis
template <typename Basis>
class BasisTensor<1,Basis> : public Basis
{
public:
   /// Contraction on X dimension
   MFEM_HOST_DEVICE inline
   auto ContractX(const DynamicDTensor<1> &u)
   {
      const int Q = this->template Size<0>();
      const int D = this->template Size<1>();
      DynamicDTensor<1> Bu(Q);
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = this->operator()(q,d);
            const double x = u(d);
            v += b * x;
         }
         Bu(q) = v;
      }
      return Bu;
   }

   /// Contraction on X dimension
   MFEM_HOST_DEVICE inline
   auto ContractX(const DynamicBlockDTensor<1> &u)
   {
      const int Q = this->template Size<0>();
      const int D = this->template Size<1>();
      DynamicBlockDTensor<1> Bu(Q);
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = this->operator()(q,d);
            const double x = u(d);
            v += b * x;
         }
         Bu(q) = v;
      }
      MFEM_SYNC_THREAD;
      return Bu;
   }

   /// Contraction on X dimension
   template <int D>
   auto ContractX(const StaticDTensor<D> &u)
   {
      constexpr int Q = this->template Size<0>();
      StaticDTensor<Q> Bu;
      for(int q = 0; q < Q; ++q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = this->operator()(q,d);
            const double x = u(d);
            v += b * x;
         }
         Bu(q) = v;
      }
      return Bu;
   }

   /// Contraction on X dimension
   template <int D>
   auto ContractX(const BlockDTensor<D> &u)
   {
      constexpr int Q = this->template Size<0>();
      BlockDTensor<Q> Bu;
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         double v = 0.0;
         for (int d = 0; d < D; ++d)
         {
            const double b = this->operator()(q,d);
            const double x = u(d);
            v += b * x;
         }
         Bu(q) = v;
      }
      MFEM_SYNC_THREAD;
      return Bu;
   }
};

// 2D Dynamic Basis
template <typename Basis>
class BasisTensor<2,Basis> : public Basis
{
public:
   /// Contraction on X dimension
   MFEM_HOST_DEVICE inline
   auto ContractX(const DynamicDTensor<2> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = this->template Size<1>();
      const int Dy = u.Size<1>();
      DynamicDTensor<2> Bu(Q,Dy);
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = this->operator()(q,dx);
               const double x = u(dx,dy);
               v += b * x;
            }
            Bu(q,dy) = v;
         }
      }
      return Bu;
   }

   /// Contraction on X dimension
   MFEM_HOST_DEVICE inline
   auto ContractX(const DynamicBlockDTensor<2> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = this->template Size<1>();
      const int Dy = u.Size<1>();
      DynamicBlockDTensor<2> Bu(Q,Dy);
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = this->operator()(q,dx);
               const double x = u(dx,dy);
               v += b * x;
            }
            Bu(q,dy) = v;
         }
      }
      MFEM_SYNC_THREAD;
      return Bu;
   }

   /// Contraction on X dimension
   template <int Dx, int Dy> MFEM_HOST_DEVICE inline
   auto ContractX(const StaticDTensor<Dx,Dy> &u)
   {
      constexpr int Q = this->template Size<0>();
      StaticDTensor<Q,Dy> Bu;
      for (int dy = 0; dy < Dy; dy++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = this->operator()(q,dx);
               const double x = u(dx,dy);
               v += b * x;
            }
            Bu(q,dy) = v;
         }
      }
      return Bu;
   }

   /// Contraction on X dimension
   template <int Dx, int Dy> MFEM_HOST_DEVICE inline
   auto ContractX(const BlockDTensor<Dx,Dy> &u)
   {
      constexpr int Q = this->template Size<0>();
      BlockDTensor<Q,Dy> Bu;
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            double v = 0.0;
            for (int dx = 0; dx < Dx; ++dx)
            {
               const double b = this->operator()(q,dx);
               const double x = u(dx,dy);
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
   auto ContractY(const DynamicDTensor<2> &u)
   {
      const int Q = this->template Size<0>();
      const int Dy = this->template Size<1>();
      const int Dx = u.Size<0>();
      DynamicDTensor<2> Bu(Dx,Q);
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = this->operator()(q,dy);
               const double x = u(dx,dy);
               v += b * x;
            }
            Bu(dx,q) = v;
         }
      }
      return Bu;
   }

   /// Contraction on Y dimension
   MFEM_HOST_DEVICE inline
   auto ContractY(const DynamicBlockDTensor<2> &u)
   {
      const int Q = this->template Size<0>();
      const int Dy = this->template Size<1>();
      const int Dx = u.Size<0>();
      DynamicBlockDTensor<2> Bu(Dx,Q);
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = this->operator()(q,dy);
               const double x = u(dx,dy);
               v += b * x;
            }
            Bu(dx,q) = v;
         }
      }
      MFEM_SYNC_THREAD;
      return Bu;
   }

   /// Contraction on Y dimension
   template <int Dx, int Dy> MFEM_HOST_DEVICE inline
   auto ContractY(const StaticDTensor<Dx,Dy> &u)
   {
      const int Q = this->template Size<0>();
      StaticDTensor<Dx,Q> Bu;
      for (int dx = 0; dx < Dx; dx++)
      {
         for(int q = 0; q < Q; ++q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = this->operator()(q,dy);
               const double x = u(dx,dy);
               v += b * x;
            }
            Bu(dx,q) = v;
         }
      }
      return Bu;
   }

   /// Contraction on Y dimension
   template <int Dx, int Dy> MFEM_HOST_DEVICE inline
   auto ContractY(const BlockDTensor<Dx,Dy> &u)
   {
      const int Q = this->template Size<0>();
      BlockDTensor<Dx,Q> Bu;
      StaticSharedDTensor<Dx,Dy> slice; // TODO Change BlockTensor in 2D and 1D
      MFEM_FOREACH_THREAD(dx,x,Dx)
      {
         MFEM_FOREACH_THREAD(q,y,Q)
         {
            double v = 0.0;
            for (int dy = 0; dy < Dy; ++dy)
            {
               const double b = this->operator()(q,dy);
               const double x = u(dx,dy);
               v += b * x;
            }
            Bu(dx,q) = v;
         }
      }
      MFEM_SYNC_THREAD;
      return Bu;
   }
};

// 3D Dynamic Basis
template <typename Basis>
class BasisTensor<3,Basis> : public Basis
{
   /// Contraction on X dimension
   MFEM_HOST_DEVICE inline
   auto ContractX(const DynamicDTensor<3> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = this->template Size<1>();
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
                  const double b = this->operator()(q,dx);
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
   MFEM_HOST_DEVICE inline
   auto ContractX(const DynamicBlockDTensor<3> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = this->template Size<1>();
      const int Dy = u.Size<1>();
      const int Dz = u.Size<2>();
      DynamicBlockDTensor<3> Bu(Q,Dy,Dz);
      SharedDTensor<2> slice(Dx,Dy); // TODO Add more than one element (with rank=3)
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(dx,x,Dx)
            {
               slice(dx,dy) = u(dx,dy,dz);
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
                  const double b = this->operator()(q,dx);
                  const double x = slice(dx,dy);
                  v += b * x;
               }
               Bu(q,dy) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
      return Bu;
   }

   /// Contraction on Y dimension
   MFEM_HOST_DEVICE inline
   auto ContractY(const DynamicDTensor<3> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = u.Size<0>();
      const int Dy = this->template Size<1>();
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
                  const double b = this->operator()(q,dy);
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
   MFEM_HOST_DEVICE inline
   auto ContractY(const DynamicBlockDTensor<3> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = u.Size<0>();
      const int Dy = this->template Size<1>();
      const int Dz = u.Size<2>();
      DynamicBlockDTensor<3> Bu(Dx,Q,Dz);
      SharedDTensor<2> slice(Dx,Dy);
      for (int dz = 0; dz < Dz; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,Dy)
         {
            MFEM_FOREACH_THREAD(dx,x,Dx)
            {
               slice(dx,dy) = u(dx,dy,dz);
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
                  const double b = this->operator()(q,dy);
                  const double x = u(dx,dy);
                  v += b * x;
               }
               Bu(dx,q) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
      return Bu;
   }

   /// Contraction on Z dimension
   MFEM_HOST_DEVICE inline
   auto ContractZ(const DynamicDTensor<3> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = u.Size<0>();
      const int Dy = u.Size<1>();
      const int Dz = this->template Size<2>();
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
                  const double b = this->operator()(q,dz);
                  const double x = u(dx,dy,dz);
                  v += b * x;
               }
               Bu(dx,dy,q) = v;
            }
         }
      }
      return Bu;
   }

   MFEM_HOST_DEVICE inline
   auto ContractZ(const DynamicBlockDTensor<3> &u)
   {
      const int Q = this->template Size<0>();
      const int Dx = u.Size<0>();
      const int Dy = u.Size<1>();
      const int Dz = this->template Size<2>();
      DynamicBlockDTensor<3> Bu(Dx,Dy,Q);
      MFEM_FOREACH_THREAD(dy,y,Dy)
      {
         MFEM_FOREACH_THREAD(dx,x,Dx)
         {
            for (int q = 0; q < Q; q++)
            {
               double v = 0.0;
               for (int dz = 0; dz < Dz; ++dz)
               {
                  const double b = this->operator()(q,dz);
                  const double x = u(dx,dy,dz);
                  v += b * x;
               }
               Bu(dx,dy,q) = v;
            }
         }
      }
      return Bu;
   }
};

class BasisMatrix: public SharedDTensor<2,1024> // TODO pick a better value than 1024
{

};

template <int Q, int D>
class StaticBasisMatrix: public StaticSharedDTensor<Q,D>
{

};

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct Basis;

template <int Dim>
struct Basis<Dim,true,0,0>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = true;
   static constexpr int MaxSize = pow(16,2);
   const int dofs1D;
   const int quads1D;
   const int dofs;
   const int quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      DynamicBasisTensor<Dim> s_B(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      DynamicBasisTensor<Dim> s_Bt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      DynamicBasisTensor<Dim> s_G(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      DynamicBasisTensor<Dim> s_Gt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Gt(d,q) = Gt[d+dofs1D*q];
         }
      }
      return s_Gt;
   }
};

template <int Dim, int Dofs1D, int Quads1D>
struct Basis<Dim,true,Dofs1D,Quads1D>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = true;
   static constexpr int dofs1D = Dofs1D;
   static constexpr int quads1D = Quads1D;
   static constexpr int dofs = pow(Dofs1D,Dim);
   static constexpr int quads = pow(Quads1D,Dim);
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      StaticBasisTensor<dim,quads1D,dofs1D> s_B(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      StaticBasisTensor<dim,dofs1D,quads1D> s_Bt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      StaticBasisTensor<dim,quads1D,dofs1D> s_G(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      StaticBasisTensor<dim,dofs1D,quads1D> s_Gt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Gt(d,q) = Gt[d+dofs1D*q];
         }
      }
      return s_Gt;
   }
};

template <int Dim>
struct Basis<Dim,false,0,0>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = false;
   static constexpr int MaxSize = pow(16,3);
   const int dofs;
   const int quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      SharedTensor<2,double,MaxSize> s_B(quads,dofs);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      SharedTensor<2,double,MaxSize> s_Bt(dofs,quads);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      SharedTensor<3,double,MaxSize> s_G(quads,dofs,dim);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_G(q,d,i) = G[q+quads*d+dofs*quads*i];
            }
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      SharedTensor<3,double,MaxSize> s_Gt(dofs,quads,dim);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_Gt(d,q,i) = Gt[d+dofs*q+dofs*quads*i];
            }
         }
      }
      return s_Gt;
   }
};

template <int Dim, int Dofs, int Quads>
struct Basis<Dim,false,Dofs,Quads>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = false;
   static constexpr int dofs = Dofs;
   static constexpr int quads = Quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      StaticSharedTensor<double,quads,dofs> s_B(quads,dofs);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      StaticSharedTensor<double,dofs,quads> s_Bt(dofs,quads);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      StaticSharedTensor<double,quads,dofs,dim> s_G(quads,dofs,dim);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_G(q,d,i) = G[q+quads*d+quads*dofs*i];
            }
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      StaticSharedTensor<double,dofs,quads,dim> s_Gt(dofs,quads,dim);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_Gt(d,q,i) = Gt[d+dofs*q+quads*dofs*i];
            }
         }
      }
      return s_Gt;
   }
};

/// Functor for building a statically sized tensor Basis
template <int Dim, int Dofs, int Quads, int BatchSize>
auto MakeBasis(KernelConfig<Dim,true,Dofs,Quads,BatchSize> &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   return Basis<Dim,true,Dofs,Quads>{b,bt,g,gt};
}

/// Functor for building a dynamically sized tensor Basis
template <int Dim, int BatchSize>
auto MakeBasis(KernelConfig<Dim,true,0,0,BatchSize> &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   const int dofs1d = config.dofs;
   const int dofs = pow(dofs1d,config.Dim);
   const int quads1d = config.quads;
   const int quads = pow(quads1d,config.Dim);
   return Basis<Dim,true,0,0>{dofs1d,quads1d,dofs,quads,b,bt,g,gt};
}

/// Functor for building a statically sized non-tensor Basis
template <int Dim, int Dofs, int Quads, int BatchSize>
auto MakeBasis(KernelConfig<Dim,false,Dofs,Quads,BatchSize> &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   return Basis<Dim,false,Dofs,Quads>{b,bt,g,gt};
}

/// Functor for building a dynamically sized non-tensor Basis
template <int Dim, int BatchSize>
auto MakeBasis(KernelConfig<Dim,false,0,0,BatchSize> &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   return Basis<Dim,false,0,0>{config.dofs,config.quads,b,bt,g,gt};
}

/// A structure to represent a transposed basis
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisTranspose
{
   Basis<Dim,IsTensor,Dofs,Quads> &basis;
};

/// A structure to represent a basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradient
{
   Basis<Dim,IsTensor,Dofs,Quads> &basis;
};

/// A structure to represent a transposed basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradientTranspose
{
   Basis<Dim,IsTensor,Dofs,Quads> &basis;
};

/// Functor to transpose a Basis
template <int Dim, bool IsTensor, int Dofs, int Quads>
auto transpose(Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return BasisTranspose<Dim,IsTensor,Dofs,Quads>{basis};
}

/// Functor to transpose a Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
auto transpose(BasisGradient<Dim,IsTensor,Dofs,Quads> &G)
{
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>{G.basis};
}

/// Functor to represent a Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
auto grad(Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return BasisGradient<Dim,IsTensor,Dofs,Quads>{basis};
}

/// Functor to represent a transposed Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
auto grad(BasisTranspose<Dim,IsTensor,Dofs,Quads> &Bt)
{
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>{Bt.basis};
}

} // mfem namespace

#endif // MFEM_BASIS
