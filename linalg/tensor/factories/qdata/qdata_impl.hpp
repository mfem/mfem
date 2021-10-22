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

#ifndef MFEM_TENSOR_QDATA_IMPL
#define MFEM_TENSOR_QDATA_IMPL

#include "../../utilities/utilities.hpp"
#include "../../tensor.hpp"
#include "../diagonal_tensor.hpp"

namespace mfem
{

/// A structure to call the constructor of T with the right sizes...
// accumulate `dim`
template <typename T, bool IsTensor, int Dim, int DimComp, int NDim=0, int NComp=0>
struct InitQData
{
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   static T make(double *x, int quads, int dim, int ne, Sizes... sizes)
   {
      return InitQData<T,IsTensor,Dim,DimComp,NDim,NComp+1>::
         make(x, quads, dim, ne, dim, sizes...);
   }
};

// accumulate `quads`
template <typename T, int Dim, int DimComp, int NDim>
struct InitQData<T,true,Dim,DimComp,NDim,DimComp>
{
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   static T make(double *x, int quads, int dim, int ne, Sizes... sizes)
   {
      return InitQData<T,true,Dim,DimComp,NDim+1,DimComp>::
         make(x, quads, dim, ne, quads, sizes...);
   }
};

// terminal case for Tensor
template <typename T, int Dim, int DimComp>
struct InitQData<T,true,Dim,DimComp,Dim,DimComp>
{
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   static T make(double *x, int quads, int dim, int ne, Sizes... sizes)
   {
      return T(x, sizes..., ne);
   }
};

// terminal case for Non-Tensor
template <typename T, int Dim, int DimComp>
struct InitQData<T,false,Dim,DimComp,0,DimComp>
{
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   static T make(double *x, int quads, int dim, int ne, Sizes... sizes)
   {
      return T(x, quads, sizes..., ne);
   }
};

/// get_qdata_layout
template <bool IsTensor, int Quads, int Dim, int DimComp, int CompSize = Dim>
struct get_qdata_layout_t;
// {
//    static constexpr int Rank = (IsTensor?Dim:1)+DimComp+1;
//    using type = DynamicLayout<Rank>;
// };

// Tensor Dynamic
template <int Dim, int DimComp, int CompSize>
struct get_qdata_layout_t<true, Dynamic, Dim, DimComp, CompSize>
{
   static constexpr int Rank = Dim+DimComp+1;
   using type = DynamicLayout<Rank>;
};

// Non-Tensor Dynamic
template <int Dim, int DimComp, int CompSize>
struct get_qdata_layout_t<false, Dynamic, Dim, DimComp, CompSize>
{
   static constexpr int Rank = 1+DimComp+1;
   using type = DynamicLayout<Rank>;
};

// Tensor Static
template <int Quads, int Dim, int DimComp, int CompSize>
struct get_qdata_layout_t<true, Quads, Dim, DimComp, CompSize>
{
   using sizes = append< int_repeat<Quads,Dim>, int_repeat<CompSize,DimComp> >;
   using type = instantiate< StaticELayout, sizes >;
};

// Non-Tensor Static
template <int Quads, int Dim, int DimComp, int CompSize>
struct get_qdata_layout_t<false, Quads, Dim, DimComp, CompSize>
{
   using sizes = append< int_list<Quads>, int_repeat<CompSize,DimComp> >;
   using type = instantiate< StaticELayout, sizes >;
};

template <bool IsTensor, int Quads, int Dim, int DimComp, int CompSize = Dim>
using get_qdata_layout = typename get_qdata_layout_t<IsTensor,Quads,Dim,DimComp,CompSize>::type;

} // mfem namespace

#endif // MFEM_TENSOR_QDATA_IMPL
