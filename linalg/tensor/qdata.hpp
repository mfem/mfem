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

#ifndef MFEM_TENSOR_QDATA
#define MFEM_TENSOR_QDATA

#include "util.hpp"
#include "tensor.hpp"
#include "diagonal_tensor.hpp"
#include "config.hpp"

namespace mfem
{

/// A class to encapsulate quadrature data in a Diagonal Tensor.
template <int DiagDim, typename QuadTensor>
class QData
: public QuadTensor
{
public:
   template <typename... Sizes>
   QData(double *x, Sizes... sizes)
   : QuadTensor(x, sizes...)
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the QData of element e
   auto operator()(int e) const
   {
      constexpr int Rank = get_tensor_rank<QuadTensor>;
      return makeDiagonalTensor<DiagDim>(this->template Get<Rank-1>(e));
   }

   auto operator()(int e)
   {
      constexpr int Rank = get_tensor_rank<QuadTensor>;
      return makeDiagonalTensor<DiagDim>(this->template Get<Rank-1>(e));
   }
};

/// A class to encapsulate quadrature data in a Diagonal Symmetric Tensor.
template <int DiagDim, typename QuadTensor>
class SymmQData
: public QuadTensor
{
public:
   template <typename... Sizes>
   SymmQData(double *x, Sizes... sizes)
   : QuadTensor(x, sizes...)
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      constexpr int Rank = get_tensor_rank<QuadTensor>;
      return makeDiagonalSymmetricTensor<DiagDim>(this->template Get<Rank-1>(e));
   }

   auto operator()(int e)
   {
      constexpr int Rank = get_tensor_rank<QuadTensor>;
      return makeDiagonalSymmetricTensor<DiagDim>(this->template Get<Rank-1>(e));
   }
};

/// A structure to call the constructor of T with the right sizes...
// accumulate `dim`
template <typename T, bool IsTensor, int Dim, int DimComp, int NDim=0, int NComp=0>
struct InitQData
{
   template <typename... Sizes>
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
   template <typename... Sizes>
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
   template <typename... Sizes>
   static T make(double *x, int quads, int dim, int ne, Sizes... sizes)
   {
      return T(x, sizes..., ne);
   }
};

// terminal case for Non-Tensor
template <typename T, int Dim, int DimComp>
struct InitQData<T,false,Dim,DimComp,0,DimComp>
{
   template <typename... Sizes>
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

/// Functor to represent QData
template <int DimComp, typename Config>
auto MakeQData(Config &config, double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Quads = get_config_quads<Config>;
   constexpr int DiagDim = IsTensor ? Dim : 1;
   constexpr int Rank = DiagDim+DimComp+1;
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp>;
   using QDataTensor = Tensor<Rank,
                              double,
                              DeviceContainer<double>,
                              Layout >;
   using QD = QData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,DimComp>::make(x,config.quads,Dim,ne);
}

template <int DimComp, typename Config>
auto MakeQData(Config &config, const double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Quads = get_config_quads<Config>;
   constexpr int DiagDim = IsTensor ? Dim : 1;
   constexpr int Rank = DiagDim+DimComp+1;
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp>;
   using QDataTensor = Tensor<Rank,
                              double,
                              ReadContainer<double>,
                              Layout >;
   using QD = QData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,DimComp>::
      make(const_cast<double*>(x),config.quads,Dim,ne);
}

/// Functor to represent symmetric data at quadrature points
template <int DimComp, typename Config>
auto MakeSymmQData(Config &config, double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Quads = get_config_quads<Config>;
   constexpr int DiagDim = IsTensor ? Dim : 1;
   constexpr int Rank = DiagDim+DimComp+1;
   constexpr int SymmSize = Dim*(Dim+1)/2;
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp,SymmSize>;
   using QDataTensor = Tensor<Rank,
                              double,
                              DeviceContainer<double>,
                              Layout >;
   using QD = SymmQData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,DimComp>::
      make(x,config.quads,SymmSize,ne);
}

template <int DimComp, typename Config>
auto MakeSymmQData(Config &config, const double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Quads = get_config_quads<Config>;
   constexpr int DiagDim = IsTensor? Dim : 1;
   constexpr int Rank = DiagDim+DimComp+1;
   constexpr int SymmSize = Dim*(Dim+1)/2;
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp,SymmSize>;
   using QDataTensor = Tensor<Rank,
                              double,
                              ReadContainer<double>,
                              Layout >;
   using QD = SymmQData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,DimComp>::
      make(const_cast<double*>(x),config.quads,SymmSize,ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_QDATA
