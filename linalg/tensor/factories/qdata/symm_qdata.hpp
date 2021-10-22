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
      constexpr int Rank = get_tensor_rank<QuadTensor>;
      return makeDiagonalSymmetricTensor<DiagDim>(this->template Get<Rank-1>(e));
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      constexpr int Rank = get_tensor_rank<QuadTensor>;
      return makeDiagonalSymmetricTensor<DiagDim>(this->template Get<Rank-1>(e));
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
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp,SymmSize>;
   using QDataTensor = Tensor<DeviceContainer<double>,Layout>;
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
   constexpr int SymmSize = Dim*(Dim+1)/2;
   using Layout = get_qdata_layout<IsTensor,Quads,Dim,DimComp,SymmSize>;
   using QDataTensor = Tensor<ReadContainer<double>,Layout>;
   using QD = SymmQData<DiagDim,QDataTensor>;
   return InitQData<QD,IsTensor,Dim,DimComp>::
      make(const_cast<double*>(x),config.quads,SymmSize,ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_SYMM_QDATA
