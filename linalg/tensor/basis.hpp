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
// ALL THIS SHOULD BE REWRITTEN...
// TODO Maybe remove this class?
// TODO maybe D before Q?
template <int Dim, bool IsTensor, typename TensorType>
class BasisTensor : public TensorType
{
public:
   MFEM_HOST_DEVICE
   BasisTensor(int quads, int dofs): TensorType(quads,dofs) { }
};

/// Represent the rank 2 tensor containing B1d or G1d with dynamic sizes
template <int Dim>
using DynamicBasisTensor = BasisTensor<Dim,true,DynamicDTensor<2>>;

/// Represent the rank 2 tensor containing B1d or G1d with static sizes
template <int Dim, int Q, int D>
using StaticBasisTensor = BasisTensor<Dim,true,StaticDTensor<Q,D>>;

/// Represent the rank 2 tensor containing B or G with dynamic sizes
template <int Dim>
using DynamicBasisNonTensor = BasisTensor<Dim,
                                          false,
                                          DynamicDTensor<2,pow(16,2*Dim)>>;

/// Represent the rank 2 tensor containing B or G with static sizes
template <int Dim, int Q, int D>
using StaticBasisNonTensor = BasisTensor<Dim,false,StaticDTensor<Q,D>>;

/// A structure to access the rank 2 tensor B, G, and B1d, G1d in the tensor case.
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

   MFEM_HOST_DEVICE inline
   auto GetB() const
   {
      DynamicBasisTensor<Dim> s_B(quads1D,dofs1D);
      for (int d = 0; d < dofs; d++)
      // MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         for (int q = 0; q < quads; q++)
         // MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt() const
   {
      DynamicBasisTensor<Dim> s_Bt(dofs1D,quads1D);
      for (int q = 0; q < quads; q++)
      // MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         for (int d = 0; d < dofs; d++)
         // MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG() const
   {
      DynamicBasisTensor<Dim> s_G(quads1D,dofs1D);
      for (int d = 0; d < dofs; d++)
      // MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         for (int q = 0; q < quads; q++)
         // MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetGt() const
   {
      DynamicBasisTensor<Dim> s_Gt(dofs1D,quads1D);
      for (int q = 0; q < quads; q++)
      // MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         for (int d = 0; d < dofs; d++)
         // MFEM_FOREACH_THREAD(d,x,dofs1D)
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

   MFEM_HOST_DEVICE inline
   auto GetB() const
   {
      StaticBasisTensor<dim,quads1D,dofs1D> s_B(quads1D,dofs1D);
      for (int d = 0; d < dofs; d++)
      // MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         for (int q = 0; q < quads; q++)
         // MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt() const
   {
      StaticBasisTensor<dim,dofs1D,quads1D> s_Bt(dofs1D,quads1D);
      for (int q = 0; q < quads; q++)
      // MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         for (int d = 0; d < dofs; d++)
         // MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG() const
   {
      StaticBasisTensor<dim,quads1D,dofs1D> s_G(quads1D,dofs1D);
      for (int d = 0; d < dofs; d++)
      // MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         for (int q = 0; q < quads; q++)
         // MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetGt() const
   {
      StaticBasisTensor<dim,dofs1D,quads1D> s_Gt(dofs1D,quads1D);
      for (int q = 0; q < quads; q++)
      // MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         for (int d = 0; d < dofs; d++)
         // MFEM_FOREACH_THREAD(d,x,dofs1D)
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

   MFEM_HOST_DEVICE inline
   auto GetB() const
   {
      DynamicBasisTensor<Dim> s_B(quads,dofs);
      for (int d = 0; d < dofs; d++)
      // MFEM_FOREACH_THREAD(d,y,dofs)
      {
         for (int q = 0; q < quads; q++)
         // MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt() const
   {
      DynamicBasisTensor<Dim> s_Bt(dofs,quads);
      for (int q = 0; q < quads; q++)
      // MFEM_FOREACH_THREAD(q,y,quads)
      {
         for (int d = 0; d < dofs; d++)
         // MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG() const
   {
      DynamicBasisTensor<Dim> s_G(quads,dofs,dim);
      for (int d = 0; d < dofs; d++)
      // MFEM_FOREACH_THREAD(d,y,dofs)
      {
         for (int q = 0; q < quads; q++)
         // MFEM_FOREACH_THREAD(q,x,quads)
         {
            for (int i = 0; i < dim; i++)
            {
               s_G(q,d,i) = G[q+quads*d+dofs*quads*i];
            }
         }
      }
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetGt() const
   {
      DynamicBasisTensor<Dim> s_Gt(dofs,quads,dim);
      for (int q = 0; q < quads; q++)
      // MFEM_FOREACH_THREAD(q,y,quads)
      {
         for (int d = 0; d < dofs; d++)
         // MFEM_FOREACH_THREAD(d,x,dofs)
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

   MFEM_HOST_DEVICE inline
   auto GetB() const
   {
      StaticBasisTensor<dim,quads,dofs> s_B(quads,dofs);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt() const
   {
      StaticBasisTensor<dim,quads,dofs> s_Bt(dofs,quads);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG() const
   {
      // TODO change type
      StaticDTensor<quads,dofs,dim> s_G(quads,dofs,dim);
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

   MFEM_HOST_DEVICE inline
   auto GetGt() const
   {
      // TODO change type
      StaticDTensor<dofs,quads,dim> s_Gt(dofs,quads,dim);
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
struct BasisTranspose : public Basis<Dim,IsTensor,Dofs,Quads>
{
   MFEM_HOST_DEVICE
   BasisTranspose(Basis<Dim,IsTensor,Dofs,Quads> &basis)
   : Basis<Dim,IsTensor,Dofs,Quads>(basis)
   { }
};

/// A structure to represent a basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradient : public Basis<Dim,IsTensor,Dofs,Quads>
{
   MFEM_HOST_DEVICE
   BasisGradient(Basis<Dim,IsTensor,Dofs,Quads> &basis)
   : Basis<Dim,IsTensor,Dofs,Quads>(basis)
   { }
};

/// A structure to represent a transposed basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradientTranspose : public Basis<Dim,IsTensor,Dofs,Quads>
{
   MFEM_HOST_DEVICE
   BasisGradientTranspose(Basis<Dim,IsTensor,Dofs,Quads> &basis)
   : Basis<Dim,IsTensor,Dofs,Quads>(basis)
   { }

   MFEM_HOST_DEVICE
   BasisGradientTranspose(BasisTranspose<Dim,IsTensor,Dofs,Quads> &basis)
   : Basis<Dim,IsTensor,Dofs,Quads>(basis)
   { }

   MFEM_HOST_DEVICE
   BasisGradientTranspose(BasisGradient<Dim,IsTensor,Dofs,Quads> &basis)
   : Basis<Dim,IsTensor,Dofs,Quads>(basis)
   { }
};

/// Functor to transpose a Basis
template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto transpose(Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return BasisTranspose<Dim,IsTensor,Dofs,Quads>(basis);
}

/// Functor to transpose a Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto transpose(BasisGradient<Dim,IsTensor,Dofs,Quads> G)
{
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>(G);
}

/// Functor to represent a Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto grad(Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return BasisGradient<Dim,IsTensor,Dofs,Quads>(basis);
}

/// Functor to represent a transposed Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto grad(BasisTranspose<Dim,IsTensor,Dofs,Quads> &Bt)
{
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>(Bt);
}

////////////////
// Basis Traits

// get_basis_dim
template <typename Basis>
struct get_basis_dim_v;

template <int Dim, bool IsTensor, typename TensorType>
struct get_basis_dim_v<BasisTensor<Dim,IsTensor,TensorType>>
{
   static constexpr int value = Dim;
};

template <typename Basis>
constexpr int get_basis_dim = get_basis_dim_v<Basis>::value;

// is_tensor_basis
template <typename Basis>
struct is_tensor_basis;

template <int Dim, bool IsTensor, typename TensorType>
struct is_tensor_basis<BasisTensor<Dim,IsTensor,TensorType>>
{
   static constexpr bool value = IsTensor;
};

// get_basis_quads
template <typename Basis>
struct get_basis_quads_v;

template <int Dim, bool IsTensor, typename TensorType>
struct get_basis_quads_v<BasisTensor<Dim, IsTensor, TensorType>>
{
   static constexpr int value = get_tensor_size<0,TensorType>;
};

template <typename Basis>
constexpr int get_basis_quads = get_basis_quads_v<Basis>::value;

// get_basis_dofs
template <typename Basis>
struct get_basis_dofs_v;

template <int Dim, bool IsTensor, typename TensorType>
struct get_basis_dofs_v<BasisTensor<Dim, IsTensor, TensorType>>
{
   static constexpr int value = get_tensor_size<1,TensorType>;
};

template <typename Basis>
constexpr int get_basis_dofs = get_basis_dofs_v<Basis>::value;

} // mfem namespace

#endif // MFEM_BASIS
