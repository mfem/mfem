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
using DynamicBasisTensor = BasisTensor<Dim,DynamicSharedDTensor<2>>;

template <int Dim, int Q, int D>
using StaticBasisTensor = BasisTensor<Dim,StaticSharedDTensor<Q,D>>;

class BasisMatrix: public DynamicSharedDTensor<2,1024> // TODO pick a better value than 1024
{

};

template <int Q, int D>
class StaticBasisMatrix: public StaticSharedDTensor<Q,D>
{

};

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct Basis;

template <int Dim>
struct Basis<Dim,true,Dynamic,Dynamic>
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

   auto GetB() const
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

   auto GetBt() const
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

   auto GetG() const
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

   auto GetGt() const
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

   auto GetB() const
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

   auto GetBt() const
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

   auto GetG() const
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

   auto GetGt() const
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
struct Basis<Dim,false,Dynamic,Dynamic>
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

   auto GetB() const
   {
      DynamicSharedDTensor<2,MaxSize> s_B(quads,dofs);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   auto GetBt() const
   {
      DynamicSharedDTensor<2,MaxSize> s_Bt(dofs,quads);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   auto GetG() const
   {
      DynamicSharedDTensor<3,MaxSize> s_G(quads,dofs,dim);
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

   auto GetGt() const
   {
      DynamicSharedDTensor<3,MaxSize> s_Gt(dofs,quads,dim);
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

   auto GetB() const
   {
      StaticSharedDTensor<quads,dofs> s_B(quads,dofs);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   auto GetBt() const
   {
      StaticSharedDTensor<dofs,quads> s_Bt(dofs,quads);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   auto GetG() const
   {
      StaticSharedDTensor<quads,dofs,dim> s_G(quads,dofs,dim);
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

   auto GetGt() const
   {
      StaticSharedDTensor<dofs,quads,dim> s_Gt(dofs,quads,dim);
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
auto MakeBasis(KernelConfig<Dim,true,Dynamic,Dynamic,BatchSize> &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   // TODO check that dofs and quads are not 0.
   const int dofs1d = config.dofs;
   const int dofs = pow(dofs1d,Dim);
   const int quads1d = config.quads;
   const int quads = pow(quads1d,Dim);
   return Basis<Dim,true,Dynamic,Dynamic>{dofs1d,quads1d,dofs,quads,b,bt,g,gt};
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
auto MakeBasis(KernelConfig<Dim,false,Dynamic,Dynamic,BatchSize> &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   // TODO check that dofs and quads are not 0.
   return Basis<Dim,false,Dynamic,Dynamic>{config.dofs,config.quads,b,bt,g,gt};
}

/// A structure to represent a transposed basis
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisTranspose : public Basis<Dim,IsTensor,Dofs,Quads> { };

/// A structure to represent a basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradient : public Basis<Dim,IsTensor,Dofs,Quads> { };

/// A structure to represent a transposed basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradientTranspose : public Basis<Dim,IsTensor,Dofs,Quads> { };

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
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>{G};
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
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>{Bt};
}

} // mfem namespace

#endif // MFEM_BASIS
