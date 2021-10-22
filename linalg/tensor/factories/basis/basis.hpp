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

#include "../../utilities/utilities.hpp"
#include "../../tensor.hpp"

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

   MFEM_HOST_DEVICE
   BasisTensor(double *shared_mem, int quads, int dofs)
      : TensorType(shared_mem,quads,dofs) { }
};

/// Represent the rank 2 tensor containing B1d or G1d with dynamic sizes
template <int Dim>
using DynamicBasisTensor = BasisTensor<Dim,true,DynamicDTensor<2>>;
template <int Dim>
using DynamicSharedBasisTensor = BasisTensor<Dim,true,DeviceDTensor<2>>;

/// Represent the rank 2 tensor containing B1d or G1d with static sizes
template <int Dim, int Q, int D>
using StaticBasisTensor = BasisTensor<Dim,true,StaticDTensor<Q,D>>;
template <int Dim, int Q, int D>
using StaticSharedBasisTensor = BasisTensor<Dim,true,StaticPointerDTensor<Q,D>>;

/// Represent the rank 2 tensor containing B or G with dynamic sizes
template <int Dim>
using DynamicBasisNonTensor = BasisTensor<
   Dim, false, DynamicDTensor<2,pow(DynamicMaxSize,2*Dim)>>;

/// Represent the rank 2 tensor containing B or G with static sizes
template <int Dim, int Q, int D>
using StaticBasisNonTensor = BasisTensor<Dim,false,StaticDTensor<Q,D>>;

template <typename KernelConfig>
struct ConfigBasis
{
   using Config = KernelConfig;

   Config config;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   ConfigBasis(const Config &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
   : config(config), B(b), Bt(bt), G(g), Gt(gt)
   { }

   MFEM_HOST_DEVICE
   int GetQuads() const
   {
      return config.quads;
   }

   MFEM_HOST_DEVICE
   int GetDofs() const
   {
      return config.dofs;
   }

   MFEM_HOST_DEVICE inline
   auto GetB(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = get_config_dofs<Config>;
      StaticSharedBasisTensor<Dim,Q,D> s_B(shared_mem,Q,D);
      MFEM_FOREACH_THREAD(d,y,D)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            s_B(q,d) = B[q+Q*d];
         }
      }
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = get_config_dofs<Config>;
      StaticSharedBasisTensor<Dim,D,Q> s_Bt(shared_mem,D,Q);
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         MFEM_FOREACH_THREAD(d,x,D)
         {
            s_Bt(d,q) = Bt[d+D*q];
         }
      }
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = get_config_dofs<Config>;
      StaticSharedBasisTensor<Dim,Q,D> s_G(shared_mem,Q,D);
      MFEM_FOREACH_THREAD(d,y,D)
      {
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            s_G(q,d) = G[q+Q*d];
         }
      }
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetGt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = get_config_dofs<Config>;
      StaticSharedBasisTensor<Dim,D,Q> s_Gt(shared_mem,D,Q);
      MFEM_FOREACH_THREAD(q,y,Q)
      {
         MFEM_FOREACH_THREAD(d,x,D)
         {
            s_Gt(d,q) = Gt[d+D*q];
         }
      }
      return s_Gt;
   }

   // MFEM_HOST_DEVICE inline
   // auto GetB(double* shared_mem) const
   // {
   //    constexpr int Dim = get_config_dim<Config>;
   //    const int quads1D = config.quads;
   //    const int dofs1D = config.dofs;
   //    DynamicSharedBasisTensor<Dim> s_B(shared_mem,quads1D,dofs1D);
   //    MFEM_FOREACH_THREAD(d,y,dofs1D)
   //    {
   //       MFEM_FOREACH_THREAD(q,x,quads1D)
   //       {
   //          s_B(q,d) = B[q+quads1D*d];
   //       }
   //    }
   //    return s_B;
   // }

   // MFEM_HOST_DEVICE inline
   // auto GetBt(double* shared_mem) const
   // {
   //    constexpr int Dim = get_config_dim<Config>;
   //    const int quads1D = config.quads;
   //    const int dofs1D = config.dofs;
   //    DynamicSharedBasisTensor<Dim> s_Bt(shared_mem,dofs1D,quads1D);
   //    MFEM_FOREACH_THREAD(q,y,quads1D)
   //    {
   //       MFEM_FOREACH_THREAD(d,x,dofs1D)
   //       {
   //          s_Bt(d,q) = Bt[d+dofs1D*q];
   //       }
   //    }
   //    return s_Bt;
   // }

   // MFEM_HOST_DEVICE inline
   // auto GetG(double* shared_mem) const
   // {
   //    constexpr int Dim = get_config_dim<Config>;
   //    const int quads1D = config.quads;
   //    const int dofs1D = config.dofs;
   //    DynamicSharedBasisTensor<Dim> s_G(shared_mem,quads1D,dofs1D);
   //    MFEM_FOREACH_THREAD(d,y,dofs1D)
   //    {
   //       MFEM_FOREACH_THREAD(q,x,quads1D)
   //       {
   //          s_G(q,d) = G[q+quads1D*d];
   //       }
   //    }
   //    return s_G;
   // }

   // MFEM_HOST_DEVICE inline
   // auto GetGt(double* shared_mem) const
   // {
   //    constexpr int Dim = get_config_dim<Config>;
   //    const int quads1D = config.quads;
   //    const int dofs1D = config.dofs;
   //    DynamicSharedBasisTensor<Dim> s_Gt(shared_mem,dofs1D,quads1D);
   //    MFEM_FOREACH_THREAD(q,y,quads1D)
   //    {
   //       MFEM_FOREACH_THREAD(d,x,dofs1D)
   //       {
   //          s_Gt(d,q) = Gt[d+dofs1D*q];
   //       }
   //    }
   //    return s_Gt;
   // }
};


/// A structure to access the rank 2 tensor B, G, and B1d, G1d in the tensor case.
template <int Dim, bool IsTensor, int Dofs, int Quads>
struct Basis;

template <int Dim>
struct Basis<Dim,true,Dynamic,Dynamic>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = true;
   static constexpr int MaxSize = pow(DynamicMaxSize,2);
   const int dofs1D;
   const int quads1D;
   const int dofs;
   const int quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   template <typename Config>
   Basis(Config &config,
         const double *b,
         const double *bt,
         const double *g = nullptr,
         const double *gt = nullptr)
   : dofs1D(config.dofs), quads1D(config.quads),
     dofs(pow(dofs1D,dim)), quads(pow(quads1D,dim)),
     B(b), Bt(bt), G(g), Gt(gt)
   { }

   MFEM_HOST_DEVICE inline
   auto GetB(double* shared_mem) const
   {
      DynamicSharedBasisTensor<Dim> s_B(shared_mem,quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt(double* shared_mem) const
   {
      DynamicSharedBasisTensor<Dim> s_Bt(shared_mem,dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG(double* shared_mem) const
   {
      DynamicSharedBasisTensor<Dim> s_G(shared_mem,quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetGt(double* shared_mem) const
   {
      DynamicSharedBasisTensor<Dim> s_Gt(shared_mem,dofs1D,quads1D);
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

   template <typename Config>
   Basis(Config &config,
         const double *b,
         const double *bt,
         const double *g = nullptr,
         const double *gt = nullptr)
   : B(b), Bt(bt), G(g), Gt(gt)
   { }

   MFEM_HOST_DEVICE inline
   auto GetB(double* shared_mem) const
   {
      StaticSharedBasisTensor<dim,quads1D,dofs1D> s_B(shared_mem,quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt(double* shared_mem) const
   {
      StaticSharedBasisTensor<dim,dofs1D,quads1D> s_Bt(shared_mem,dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG(double* shared_mem) const
   {
      StaticSharedBasisTensor<dim,quads1D,dofs1D> s_G(shared_mem,quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetGt(double* shared_mem) const
   {
      StaticSharedBasisTensor<dim,dofs1D,quads1D> s_Gt(shared_mem,dofs1D,quads1D);
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
   static constexpr int MaxSize = pow(DynamicMaxSize,3);
   const int dofs;
   const int quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   template <typename Config>
   Basis(Config &config,
         const double *b,
         const double *bt,
         const double *g = nullptr,
         const double *gt = nullptr)
   : dofs(config.dofs), quads(config.quads), B(b), Bt(bt), G(g), Gt(gt)
   { }
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

   template <typename Config>
   Basis(Config &config,
         const double *b,
         const double *bt,
         const double *g = nullptr,
         const double *gt = nullptr)
   : B(b), Bt(bt), G(g), Gt(gt)
   { }
};

template <typename Config>
auto MakeBasis(Config &config,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   // constexpr int Dim = get_config_dim<Config>;
   // constexpr bool IsTensor = is_tensor_config<Config>;
   // constexpr int Dofs = get_config_dofs<Config>;
   // constexpr int Quads = get_config_quads<Config>;
   // return Basis<Dim,IsTensor,Dofs,Quads>(config,b,bt,g,gt);
   return ConfigBasis<Config>(config,b,bt,g,gt);
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

template <typename Basis>
struct Trans
{
   const Basis &basis;

   MFEM_HOST_DEVICE
   Trans(const Basis &basis): basis(basis) { }

   MFEM_HOST_DEVICE
   int GetQuads() const
   {
      return basis.GetQuads();
   }

   MFEM_HOST_DEVICE
   int GetDofs() const
   {
      return basis.GetDofs();
   }

   MFEM_HOST_DEVICE inline
   auto GetB(double* shared_mem) const
   {
      return basis.GetB(shared_mem);
   }

   MFEM_HOST_DEVICE inline
   auto GetBt(double* shared_mem) const
   {
      return basis.GetBt(shared_mem);
   }

   MFEM_HOST_DEVICE inline
   auto GetG(double* shared_mem) const
   {
      return basis.GetG(shared_mem);
   }

   MFEM_HOST_DEVICE inline
   auto GetGt(double* shared_mem) const
   {
      return basis.GetGt(shared_mem);
   }
};

template <typename Basis>
struct Grad
{
   const Basis &basis;

   MFEM_HOST_DEVICE
   Grad(const Basis &basis): basis(basis) { }

   MFEM_HOST_DEVICE
   int GetQuads() const
   {
      return basis.GetQuads();
   }

   MFEM_HOST_DEVICE
   int GetDofs() const
   {
      return basis.GetDofs();
   }

   MFEM_HOST_DEVICE inline
   auto GetB(double* shared_mem) const
   {
      return basis.GetB(shared_mem);
   }

   MFEM_HOST_DEVICE inline
   auto GetBt(double* shared_mem) const
   {
      return basis.GetBt(shared_mem);
   }

   MFEM_HOST_DEVICE inline
   auto GetG(double* shared_mem) const
   {
      return basis.GetG(shared_mem);
   }

   MFEM_HOST_DEVICE inline
   auto GetGt(double* shared_mem) const
   {
      return basis.GetGt(shared_mem);
   }
};

template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto transpose(const Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return Trans<Basis<Dim,IsTensor,Dofs,Quads>>(basis);
}

template <typename Config> MFEM_HOST_DEVICE inline
auto transpose(const ConfigBasis<Config> &basis)
{
   return Trans<ConfigBasis<Config>>(basis);
}

/// Functor to transpose a Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto transpose(const Grad<Basis<Dim,IsTensor,Dofs,Quads>> &G)
{
   return Trans<Grad<Basis<Dim,IsTensor,Dofs,Quads>>>(G);
}

template <typename Config> MFEM_HOST_DEVICE inline
auto transpose(const Grad<ConfigBasis<Config>> &G)
{
   return Trans<Grad<ConfigBasis<Config>>>(G);
}

/// Functor to represent a Basis gradient
template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto grad(const Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return Grad<Basis<Dim,IsTensor,Dofs,Quads>>(basis);
}

template <typename Config> MFEM_HOST_DEVICE inline
auto grad(const ConfigBasis<Config> &basis)
{
   return Grad<ConfigBasis<Config>>(basis);
}

template <int Dim, bool IsTensor, int Dofs, int Quads> MFEM_HOST_DEVICE inline
auto grad(const Trans<Basis<Dim,IsTensor,Dofs,Quads>> &Bt)
{
   return Trans<Grad<Basis<Dim,IsTensor,Dofs,Quads>>>(grad(Bt.basis));
}

template <typename Config> MFEM_HOST_DEVICE inline
auto grad(const Trans<ConfigBasis<Config>> &Bt)
{
   return Trans<Grad<ConfigBasis<Config>>>(grad(Bt.basis));
}

////////////////
// Basis Traits

// is_basis
template <typename Basis>
struct is_basis_v
{
   static constexpr bool value = false;
};

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct is_basis_v<Basis<Dim,IsTensor,Dofs,Quads>>
{
   static constexpr bool value = true;
};

template <typename Config>
struct is_basis_v<ConfigBasis<Config>>
{
   static constexpr bool value = true;
};

template <typename Basis>
constexpr bool is_basis = is_basis_v<Basis>::value;

// get_basis_dim
template <typename Basis>
struct get_basis_dim_v
{
   static constexpr int value = -1;
};

template <int Dim, bool IsTensor, int D, int Q>
struct get_basis_dim_v<Basis<Dim,IsTensor,D,Q>>
{
   static constexpr int value = Dim;
};

template <typename Config>
struct get_basis_dim_v<ConfigBasis<Config>>
{
   static constexpr int value = get_config_dim<Config>;
};

template <int Dim, bool IsTensor, typename TensorType>
struct get_basis_dim_v<BasisTensor<Dim,IsTensor,TensorType>>
{
   static constexpr int value = Dim;
};

template <typename Basis>
struct get_basis_dim_v<Trans<Basis>>
{
   static constexpr int value = get_basis_dim_v<Basis>::value;
};

template <typename Basis>
constexpr int get_basis_dim = get_basis_dim_v<Basis>::value;

// is_tensor_basis
template <typename Basis>
struct is_tensor_basis_v
{
   static constexpr bool value = false;
};

template <int Dim, bool IsTensor, int D, int Q>
struct is_tensor_basis_v<Basis<Dim,IsTensor,D,Q>>
{
   static constexpr bool value = IsTensor;
};

template <typename Config>
struct is_tensor_basis_v<ConfigBasis<Config>>
{
   static constexpr bool value = is_tensor_config<Config>;
};

template <int Dim, bool IsTensor, typename TensorType>
struct is_tensor_basis_v<BasisTensor<Dim,IsTensor,TensorType>>
{
   static constexpr bool value = IsTensor;
};

template <typename Basis>
struct is_tensor_basis_v<Trans<Basis>>
{
   static constexpr bool value = is_tensor_basis_v<Basis>::value;
};

template <typename Basis>
constexpr bool is_tensor_basis = is_tensor_basis_v<Basis>::value;

// is_non_tensor_basis
template <typename Basis>
struct is_non_tensor_basis_v
{
   static constexpr bool value = false;
};

template <int Dim, bool IsTensor, int D, int Q>
struct is_non_tensor_basis_v<Basis<Dim,IsTensor,D,Q>>
{
   static constexpr bool value = !IsTensor;
};

template <typename Config>
struct is_non_tensor_basis_v<ConfigBasis<Config>>
{
   static constexpr bool value = !is_tensor_config<Config>;
};

template <int Dim, bool IsTensor, typename TensorType>
struct is_non_tensor_basis_v<BasisTensor<Dim,IsTensor,TensorType>>
{
   static constexpr bool value = !IsTensor;
};

template <typename Basis>
struct is_non_tensor_basis_v<Trans<Basis>>
{
   static constexpr bool value = is_non_tensor_basis_v<Basis>::value;
};

template <typename Basis>
constexpr bool is_non_tensor_basis = is_non_tensor_basis_v<Basis>::value;

// get_basis_quads
template <typename Basis>
struct get_basis_quads_v;

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct get_basis_quads_v<Basis<Dim,IsTensor,Dofs,Quads>>
{
   static constexpr int value = Quads;
};

template <typename Config>
struct get_basis_quads_v<ConfigBasis<Config>>
{
   static constexpr int value = get_config_quads<Config>;
};

// get_basis_dofs
template <typename Basis>
struct get_basis_dofs_v;

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct get_basis_dofs_v<Basis<Dim,IsTensor,Dofs,Quads>>
{
   static constexpr int value = Dofs;
};

template <typename Config>
struct get_basis_dofs_v<ConfigBasis<Config>>
{
   static constexpr int value = get_config_dofs<Config>;
};

template <int Dim, bool IsTensor, typename TensorType>
struct get_basis_quads_v<BasisTensor<Dim, IsTensor, TensorType>>
{
   static constexpr int value = get_tensor_size<0,TensorType>;
};

template <typename Basis>
struct get_basis_quads_v<Trans<Basis>>
{
   static constexpr int value = get_basis_dofs_v<Basis>::value;
};

template <typename Basis>
struct get_basis_quads_v<Grad<Basis>>
{
   static constexpr int value = get_basis_quads_v<Basis>::value;
};

template <typename Basis>
constexpr int get_basis_quads = get_basis_quads_v<Basis>::value;

template <int Dim, bool IsTensor, typename TensorType>
struct get_basis_dofs_v<BasisTensor<Dim, IsTensor, TensorType>>
{
   static constexpr int value = get_tensor_size<1,TensorType>;
};

template <typename Basis>
struct get_basis_dofs_v<Trans<Basis>>
{
   static constexpr int value = get_basis_quads_v<Basis>::value;
};

template <typename Basis>
struct get_basis_dofs_v<Grad<Basis>>
{
   static constexpr int value = get_basis_dofs_v<Basis>::value;
};

template <typename Basis>
constexpr int get_basis_dofs = get_basis_dofs_v<Basis>::value;

// get_basis_size
template <int N, typename Basis>
struct get_basis_size_v;

template <int N, int Dim, bool IsTensor, typename TensorType>
struct get_basis_size_v<N, BasisTensor<Dim, IsTensor, TensorType>>
{
   static constexpr int value = get_tensor_size<N,TensorType>;
};

template <int N, typename Basis>
constexpr int get_basis_size = get_basis_size_v<N,Basis>::value;

// get_basis_capacity
template <typename Basis, typename Enable = void> //std::enable_if_t<is_basis<Basis>> >
struct get_basis_capacity_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize; // TODO
};

template <int Dim>
struct get_basis_capacity_v<Basis<Dim,true,Dynamic,Dynamic>,void>
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <int Dim>
struct get_basis_capacity_v<Basis<Dim,false,Dynamic,Dynamic>,void>
{
   static constexpr int value = 64*64;
};

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct get_basis_capacity_v<Basis<Dim,IsTensor,Dofs,Quads>,void>
{
   static constexpr int value = Dofs*Quads;
};

template <typename Config>
struct get_basis_capacity_v<ConfigBasis<Config>,
std::enable_if_t<
   get_config_dofs<Config> != Dynamic &&
   get_config_quads<Config> != Dynamic
> >
{
   static constexpr int D = get_config_dofs<Config>;
   static constexpr int Q = get_config_quads<Config>;
   static constexpr int value = D*Q;
};

template <typename Config>
struct get_basis_capacity_v<ConfigBasis<Config>,
std::enable_if_t<
   get_config_dofs<Config> == Dynamic &&
   get_config_quads<Config> == Dynamic &&
   is_tensor_config<Config>
> >
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <typename Config>
struct get_basis_capacity_v<ConfigBasis<Config>,
std::enable_if_t<
   get_config_dofs<Config> == Dynamic &&
   get_config_quads<Config> == Dynamic &&
   !is_tensor_config<Config>
> >
{
   static constexpr int value = 64*64;
};

template <typename Basis>
struct get_basis_capacity_v<Grad<Basis>, std::enable_if_t<is_basis<Basis>> >
{
   static constexpr bool IsTensor = is_tensor_basis<Basis>;
   static constexpr int Dim = get_basis_dim<Basis>;
   static constexpr int value = (IsTensor ? 1 : Dim) *
                                 get_basis_capacity_v<Basis>::value;
};

template <typename Basis>
struct get_basis_capacity_v<Trans<Basis>, std::enable_if_t<is_basis<Basis>> >
{
   static constexpr int value = get_basis_capacity_v<Basis>::value;
};

template <typename Basis>
struct get_basis_capacity_v<Trans<Grad<Basis>>, std::enable_if_t<is_basis<Basis>> >
{
   static constexpr int value = get_basis_capacity_v<Basis>::value;
};

// template <int Dim, bool IsTensor, typename TensorType>
// struct get_basis_capacity_v<BasisTensor<Dim, IsTensor, TensorType>>
// {
//    static constexpr int value = DynamicMaxSize*DynamicMaxSize; // TODO
// };

template <typename Basis>
constexpr int get_basis_capacity = get_basis_capacity_v<Basis>::value;

// ResultTensor
template <typename Basis, typename Enable = void> //std::enable_if_t<is_basis<Basis>> >
struct basis_result_tensor;

template <typename Config>
struct basis_result_tensor<ConfigBasis<Config>,void>
{
   template <int... Sizes>
   using type = typename config_result_tensor<Config>
                   ::template type<Sizes...>;
};

template <typename Basis>
struct basis_result_tensor<Trans<Basis>,std::enable_if_t<is_basis<Basis>>>
{
   template <int... Sizes>
   using type = typename basis_result_tensor<Basis>
                   ::template type<Sizes...>;
};

template <typename Basis>
struct basis_result_tensor<Grad<Basis>,std::enable_if_t<is_basis<Basis>>>
{
   template <int... Sizes>
   using type = typename basis_result_tensor<Basis>
                   ::template type<Sizes...>;
};

template <typename Basis>
struct basis_result_tensor<Trans<Grad<Basis>>,std::enable_if_t<is_basis<Basis>>>
{
   template <int... Sizes>
   using type = typename basis_result_tensor<Basis>
                   ::template type<Sizes...>;
};

template <typename Basis, int... Sizes>
using ResultTensor = typename basis_result_tensor<Basis>
                        ::template type<Sizes...>;

} // mfem namespace

#endif // MFEM_BASIS
