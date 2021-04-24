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

#ifndef MFEM_TENSOR_DOF
#define MFEM_TENSOR_DOF

#include "util.hpp"
#include "tensor.hpp"
#include "config.hpp"

namespace mfem
{

/// A class to encapsulate degrees of freedom in a Tensor.
template <typename DofTensor>
class DegreesOfFreedom
: public DofTensor
{
public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   DegreesOfFreedom(double *x, Sizes... sizes)
   : DofTensor(x, sizes...)
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      constexpr int Rank = get_tensor_rank<DofTensor>;
      return this->template Get<Rank-1>(e);
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      constexpr int Rank = get_tensor_rank<DofTensor>;
      return this->template Get<Rank-1>(e);
   }
};

/// A structure to call the constructor of T with the right sizes...
template <typename T, bool IsTensor, int Dim, int VDim, int N=0>
struct InitDof;

// Tensor
template <typename T, int Dim, int VDim, int N>
struct InitDof<T,true,Dim,VDim,N>
{
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   static T make(double *x, int dofs, int ne, Sizes... sizes)
   {
      return InitDof<T,true,Dim,VDim,N+1>::make(x, dofs, ne, dofs, sizes...);
   }
};

template <typename T, int Dim, int VDim>
struct InitDof<T,true,Dim,VDim,Dim>
{
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   static T make(double *x, int dofs, int ne, Sizes... sizes)
   {
      return T(x, sizes..., VDim, ne);
   }
};

template <typename T, int Dim>
struct InitDof<T,true,Dim,0,Dim>
{
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   static T make(double *x, int dofs, int ne, Sizes... sizes)
   {
      return T(x, sizes..., ne);
   }
};

// Non-Tensor
template <typename T, int Dim>
struct InitDof<T,false,Dim,0,1>
{
   MFEM_HOST_DEVICE inline
   static T make(double *x, int dofs, int ne)
   {
      return T(x, dofs, ne);
   }
};

template <typename T, int Dim, int VDim>
struct InitDof<T,false,Dim,VDim,1>
{
   MFEM_HOST_DEVICE inline
   static T make(double *x, int dofs, int ne)
   {
      return T(x, dofs, VDim, ne);
   }
};

/// get_dof_layout
template <bool IsTensor, int Dofs, int Dim, int VDim>
struct get_dof_layout_t;
// {
//    static constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using type = DynamicLayout<Rank>;
// };

// Tensor Dynamic no-VDim
template <int Dim>
struct get_dof_layout_t<true, Dynamic, Dim, 0>
{
   static constexpr int Rank = Dim+1;
   using type = DynamicLayout<Rank>;
};

// Tensor Dynamic VDim
template <int Dim, int VDim>
struct get_dof_layout_t<true, Dynamic, Dim, VDim>
{
   static constexpr int Rank = Dim+1+1;
   using type = DynamicLayout<Rank>;
};

// Non-Tensor Dynamic no-VDim
template <int Dim>
struct get_dof_layout_t<false, Dynamic, Dim, 0>
{
   static constexpr int Rank = 1+1;
   using type = DynamicLayout<Rank>;
};

// Non-Tensor Dynamic VDim
template <int Dim, int VDim>
struct get_dof_layout_t<false, Dynamic, Dim, VDim>
{
   static constexpr int Rank = 1+1+1;
   using type = DynamicLayout<Rank>;
};

// Tensor Static no-VDim
template <int Dofs, int Dim>
struct get_dof_layout_t<true, Dofs, Dim, 0>
{
   using sizes = int_repeat<Dofs,Dim>;
   using type = instantiate< StaticELayout, sizes >;
};

// Tensor Static VDim
template <int Dofs, int Dim, int VDim>
struct get_dof_layout_t<true, Dofs, Dim, VDim>
{
   using sizes = append< int_repeat<Dofs,Dim>, int_list<VDim> >;
   using type = instantiate< StaticELayout, sizes >;
};

// Non-Tensor Static no-VDim
template <int Dofs, int Dim>
struct get_dof_layout_t<false, Dofs, Dim, 0>
{
   using type = StaticELayout<Dofs>;
};

// Non-Tensor Static VDim
template <int Dofs, int Dim, int VDim>
struct get_dof_layout_t<false, Dofs, Dim, VDim>
{
   using type = StaticELayout<Dofs,VDim>;
};

template <bool IsTensor, int Dofs, int Dim, int VDim>
using get_dof_layout = typename get_dof_layout_t<IsTensor,Dofs,Dim,VDim>::type;

/// Functor to represent degrees of freedom
template <int VDim, typename Config>
auto MakeDoFs(Config &config, double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Dofs = get_config_dofs<Config>;
   using Layout = get_dof_layout<IsTensor,Dofs,Dim,VDim>;
   using DofTensor = Tensor<DeviceContainer<double>,Layout>;
   using Dof = DegreesOfFreedom<DofTensor>;
   return InitDof<Dof,IsTensor,Dim,VDim>::make(x,config.dofs,ne);
}

template <int VDim, typename Config>
auto MakeDoFs(Config &config, const double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Dofs = get_config_dofs<Config>;
   using Layout = get_dof_layout<IsTensor,Dofs,Dim,VDim>;
   using DofTensor = Tensor<ReadContainer<double>,Layout>;
   using Dof = DegreesOfFreedom<DofTensor>;
   return InitDof<Dof,IsTensor,Dim,VDim>::
      make(const_cast<double*>(x),config.dofs,ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_DOF
