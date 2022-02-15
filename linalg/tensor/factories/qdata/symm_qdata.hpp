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

#ifndef MFEM_TENSOR_SYMM_QDATA
#define MFEM_TENSOR_SYMM_QDATA

#include "qdata_traits.hpp"
#include "qdata_impl.hpp"
#include "../diagonal_symm_tensor.hpp"

namespace mfem
{

/// A class to encapsulate quadrature data in a Diagonal Symmetric Tensor.
template <int DiagDim, typename QuadTensor>
class SymmQData
: public QuadTensor
{
public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   SymmQData(double *x, Sizes... sizes)
   : QuadTensor(x, sizes...)
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      constexpr int Element = get_tensor_rank<QuadTensor> - 1;
      return makeDiagonalSymmetricTensor<DiagDim>(Get<Element>(e, *this));
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      constexpr int Element = get_tensor_rank<QuadTensor> - 1;
      return makeDiagonalSymmetricTensor<DiagDim>(Get<Element>(e, *this));
   }
};

/// Functor to represent symmetric data at quadrature points
template <int DimComp, typename Config>
auto MakeSymmQData(Config &config, double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Quads = get_config_quads<Config>;
   constexpr int DiagDim = IsTensor ? Dim : 1;
   constexpr int SymmSize = Dim*(Dim+1)/2;
   constexpr int SymmDims = DimComp/2;
   static_assert(
      DimComp%2 == 0,
      "The number of symmetric dimensions must be even.");
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,SymmDims,SymmSize>;
   using QDataTensor = Tensor<DeviceContainer<double>,Layout>;
   using QD = SymmQData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,SymmDims>::
      make(x,config.quads,SymmSize,ne);
}

template <int DimComp, typename Config>
auto MakeSymmQData(Config &config, const double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Quads = get_config_quads<Config>;
   constexpr int DiagDim = IsTensor? Dim : 1;
   constexpr int SymmSize = Dim*(Dim+1)/2;
   constexpr int SymmDims = DimComp/2;
   static_assert(
      DimComp%2 == 0,
      "The number of symmetric dimensions must be even.");
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,SymmDims,SymmSize>;
   using QDataTensor = Tensor<ReadContainer<double>,Layout>;
   using QD = SymmQData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,SymmDims>::
      make(const_cast<double*>(x),config.quads,SymmSize,ne);
}

/// is_qdata
template <int DiagDim, typename QuadTensor>
struct is_qdata_v<SymmQData<DiagDim,QuadTensor>>
{
   static constexpr bool value = true;
};

template <int DRank,
          int SRank,
          typename Container,
          typename Layout>
struct is_qdata_v<DiagonalSymmetricTensor<DRank,SRank,Container,Layout>>
{
   static constexpr bool value = true;
};

} // mfem namespace

#endif // MFEM_TENSOR_SYMM_QDATA
