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

#ifndef MFEM_STANDARD_BASIS
#define MFEM_STANDARD_BASIS

#include "basis_impl.hpp"
#include "../../utilities/load.hpp"
#include "../../utilities/config.hpp"

namespace mfem
{

/// A simple basis structure meant to load basis operators in shared memory.
template <int Dofs, typename KernelConfig>
struct SharedBasis
{
   using Config = KernelConfig;

   Config config;
   const int dofs;
   const int quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   SharedBasis(const Config &config,
               const int dofs,
               const int quads,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
   : config(config), dofs(dofs), quads(quads), B(b), Bt(bt), G(g), Gt(gt)
   { }

   MFEM_HOST_DEVICE
   int GetQuads() const
   {
      return quads;
   }

   MFEM_HOST_DEVICE
   int GetDofs() const
   {
      return dofs;
   }

   MFEM_HOST_DEVICE inline
   auto GetB(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = Dofs;
      StaticSharedBasisTensor<Dim,Q,D> s_B(shared_mem,quads,dofs);
      load_with_2dthreads(B,quads,dofs,s_B);
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetBt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = Dofs;
      StaticSharedBasisTensor<Dim,D,Q> s_Bt(shared_mem,dofs,quads);
      load_with_2dthreads(Bt,dofs,quads,s_Bt);
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetG(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = Dofs;
      StaticSharedBasisTensor<Dim,Q,D> s_G(shared_mem,quads,dofs);
      load_with_2dthreads(G,quads,dofs,s_G);
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetGt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      constexpr int D = Dofs;
      StaticSharedBasisTensor<Dim,D,Q> s_Gt(shared_mem,dofs,quads);
      load_with_2dthreads(Gt,dofs,quads,s_Gt);
      return s_Gt;
   }
};

template <int Dofs, typename Config>
auto MakeBasis(Config &config,
               const int dofs,
               const int quads,
               const double *b,
               const double *bt,
               const double *g = nullptr,
               const double *gt = nullptr)
{
   return SharedBasis<Dofs,Config>(config,dofs,quads,b,bt,g,gt);
}

/// A structure to represent a transposed basis
template <typename Basis>
struct Trans: public Basis
{
   MFEM_HOST_DEVICE
   Trans(const Basis &basis): Basis(basis) { }
};

/// A structure to represent a gradient basis
template <typename Basis>
struct Grad: public Basis
{
   MFEM_HOST_DEVICE
   Grad(const Basis &basis): Basis(basis) { }
};

/// A structure to represent the divergence operator of a basis
template <typename Basis>
struct Div: public Basis
{
   MFEM_HOST_DEVICE
   Div(const Basis &basis): Basis(basis) { }
};

/// Factory to transpose a Basis
template <typename Basis> MFEM_HOST_DEVICE inline
auto transpose(const Basis &basis)
{
   return Trans<Basis>(basis);
}

/// Factory to transpose a Basis gradient
template <typename Basis> MFEM_HOST_DEVICE inline
auto transpose(const Grad<Basis> &G)
{
   return Trans<Grad<Basis>>(G);
}

/// Factory to transpose a Basis divergence
template <typename Basis> MFEM_HOST_DEVICE inline
auto transpose(const Div<Basis> &div)
{
   return Trans<Div<Basis>>(div);
}

/// Factory to represent a Basis gradient
template <typename Basis> MFEM_HOST_DEVICE inline
auto grad(const Basis &basis)
{
   return Grad<Basis>(basis);
}

template <typename Basis> MFEM_HOST_DEVICE inline
auto grad(const Trans<Basis> &Bt)
{
   return Trans<Grad<Basis>>(grad(Bt));
}

/// Factory to represent a Basis divergence operator
template <typename Basis> MFEM_HOST_DEVICE inline
auto div(const Basis &basis)
{
   return Div<Basis>(basis);
}

template <typename Basis> MFEM_HOST_DEVICE inline
auto div(const Trans<Basis> &Bt)
{
   return Trans<Div<Basis>>(Bt);
}

////////////////
// Basis Traits

// is_basis
template <typename Basis>
struct is_basis_v
{
   static constexpr bool value = false;
};

template <int Dofs, typename Config>
struct is_basis_v<SharedBasis<Dofs, Config>>
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

template <int Dofs, typename Config>
struct get_basis_dim_v<SharedBasis<Dofs, Config>>
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

template <int Dofs, typename Config>
struct is_tensor_basis_v<SharedBasis<Dofs,Config>>
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

template <int Dofs, typename Config>
struct is_non_tensor_basis_v<SharedBasis<Dofs,Config>>
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

template <int Dofs, typename Config>
struct get_basis_quads_v<SharedBasis<Dofs,Config>>
{
   static constexpr int value = get_config_quads<Config>;
};

// get_basis_dofs
template <typename Basis>
struct get_basis_dofs_v;

template <int Dofs, typename Config>
struct get_basis_dofs_v<SharedBasis<Dofs,Config>>
{
   static constexpr int value = Dofs;
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

template <int Dofs, typename Config>
struct get_basis_capacity_v<SharedBasis<Dofs,Config>,
std::enable_if_t<
   Dofs != Dynamic &&
   get_config_quads<Config> != Dynamic
> >
{
   static constexpr int Q = get_config_quads<Config>;
   static constexpr int value = Dofs*Q;
};

template <int Dofs, typename Config>
struct get_basis_capacity_v<SharedBasis<Dofs,Config>,
std::enable_if_t<
   Dofs == Dynamic &&
   get_config_quads<Config> == Dynamic &&
   is_tensor_config<Config>
> >
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <int Dofs, typename Config>
struct get_basis_capacity_v<SharedBasis<Dofs,Config>,
std::enable_if_t<
   Dofs == Dynamic &&
   get_config_quads<Config> == Dynamic &&
   !is_tensor_config<Config>
> >
{
   static constexpr int value = 64*64; // FIXME magic number
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
template <typename Basis>
struct basis_result_tensor;

template <int Dofs, typename Config>
struct basis_result_tensor<SharedBasis<Dofs,Config>>
{
   template <int... Sizes>
   using type = typename config_result_tensor<Config>
                   ::template type<Sizes...>;
};

template <typename Basis>
struct basis_result_tensor<Trans<Basis>>
{
   template <int... Sizes>
   using type = typename basis_result_tensor<Basis>
                   ::template type<Sizes...>;
};

template <typename Basis>
struct basis_result_tensor<Grad<Basis>>
{
   template <int... Sizes>
   using type = typename basis_result_tensor<Basis>
                   ::template type<Sizes...>;
};

template <typename Basis>
struct basis_result_tensor<Trans<Grad<Basis>>>
{
   template <int... Sizes>
   using type = typename basis_result_tensor<Basis>
                   ::template type<Sizes...>;
};

template <typename Basis, int... Sizes>
using BasisResultTensor = typename basis_result_tensor<Basis>
                             ::template type<Sizes...>;

} // mfem namespace

#endif // MFEM_STANDARD_BASIS
