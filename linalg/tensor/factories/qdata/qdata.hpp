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

#include "qdata_traits.hpp"
#include "qdata_impl.hpp"
#include "../diagonal_tensor.hpp"

namespace mfem
{

/// A class to encapsulate quadrature data in a Diagonal Tensor.
template <int DiagDim, typename QuadTensor>
class QData
: public QuadTensor
{
public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   QData(double *x, Sizes... sizes)
   : QuadTensor(x, sizes...)
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the QData of element e
   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      constexpr int Element = get_tensor_rank<QuadTensor> - 1;
      return makeDiagonalTensor<DiagDim>(Get<Element>(e, *this));
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      constexpr int Element = get_tensor_rank<QuadTensor> - 1;
      return makeDiagonalTensor<DiagDim>(Get<Element>(e, *this));
   }
};

/// Functor to represent QData
template <int DimComp, typename Config>
auto MakeQData(Config &config, double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Quads = get_config_quads<Config>;
   constexpr int DiagDim = IsTensor ? Dim : 1;
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp>;
   using QDataTensor = Tensor<DeviceContainer<double>,Layout>;
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
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp>;
   using QDataTensor = Tensor<ReadContainer<double>,Layout>;
   using QD = QData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,DimComp>::
      make(const_cast<double*>(x),config.quads,Dim,ne);
}

/// is_qdata
template <int DiagDim, typename QuadTensor>
struct is_qdata_v<QData<DiagDim,QuadTensor>>
{
   static constexpr bool value = true;
};

template <int DRank,
          int Rank,
          typename Container,
          typename Layout>
struct is_qdata_v<DiagonalTensor<DRank,Rank,Container,Layout>>
{
   static constexpr bool value = true;
};

} // mfem namespace

#endif // MFEM_TENSOR_QDATA
