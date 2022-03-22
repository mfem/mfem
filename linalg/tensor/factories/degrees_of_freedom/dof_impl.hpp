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

#ifndef MFEM_TENSOR_DOF_IMPL
#define MFEM_TENSOR_DOF_IMPL

#include "../../utilities/utilities.hpp"
#include "../../tensor.hpp"

namespace mfem
{

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

} // mfem namespace

#endif // MFEM_TENSOR_DOF_IMPL
