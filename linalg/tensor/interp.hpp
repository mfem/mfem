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

#ifndef MFEM_TENSOR_INTERP
#define MFEM_TENSOR_INTERP

#include "tensor.hpp"
#include "contraction.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include <utility>

namespace mfem
{

// Functions to interpolate from degrees of freedom to quadrature points
// Non-tensor and 1D cases
template<int D, int Q> MFEM_HOST_DEVICE inline
dTensor<Q> Interpolate(const dTensor<Q,D> &B,
                       const dTensor<D> &u)
{
   return ContractX1D(B,u);
}

// Non-tensor and 1D cases with VDim components
template<int Q, int D, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q> Interpolate(const dTensor<Q,D> &B,
                                          const StaticTensor<dTensor<VDim>,D> &u)
{
   return ContractX1D(B,u);
}

// 3D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d,Q1d> Interpolate(const dTensor<Q1d,D1d> &B,
                                 const dTensor<D1d,D1d,D1d> &u)
{
   auto Bu = ContractX3D(B,u);
   auto BBu = ContractY3D(B,Bu);
   return ContractZ3D(B,BBu);
}

// 3D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d,Q1d,Q1d> Interpolate(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,D1d,D1d,D1d> &u)
{
   auto Bu = ContractX3D(B,u);
   auto BBu = ContractY3D(B,Bu);   
   return ContractZ3D(B,BBu);
}

// 2D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d> Interpolate(const dTensor<Q1d,D1d> &B,
                             const dTensor<D1d,D1d> &u)
{
   auto Bu = ContractX2D(B,u);
   return ContractY2D(B,Bu);
}

// 2D Tensor case with VDim components
template<int Q1d, int D1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,Q1d,Q1d> Interpolate(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,D1d,D1d> &u)
{
   ContractX2D(B,u);
   return ContractY2D(B,u);
}

// Functions to interpolate from degrees of freedom to quadrature points
// Non-tensor and 1D cases
template<int P, int Q> MFEM_HOST_DEVICE inline
dTensor<P> InterpolateT(const dTensor<Q,P> &B,
                        const dTensor<Q> &u)
{
   return ContractTX1D(B,u);
}

// Non-tensor and 1D cases with VDim components
template<int Q, int P, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,P> InterpolateT(const dTensor<Q,P> &B,
                                           const StaticTensor<dTensor<VDim>,Q> &u)
{
   return ContractTX1D(B,u);
}

// 3D Tensor case
template<int Q1d, int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d,D1d> InterpolateT(const dTensor<Q1d,D1d> &B,
                                  const dTensor<Q1d,Q1d,Q1d> &u)
{
   auto Bu = ContractTX3D(B,u);
   auto BBu = ContractTY3D(B,Bu);
   return ContractTZ3D(B,BBu);
}

// 3D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,D1d,D1d> InterpolateT(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,Q1d,Q1d> &u)
{
   auto Bu = ContractTX3D(B,u);
   auto BBu = ContractTY3D(B,Bu);
   return ContractTZ3D(B,BBu);
}

// 2D Tensor case
template<int D1d, int Q1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d> InterpolateT(const dTensor<Q1d,D1d> &B,
                              const dTensor<Q1d,Q1d> &u)
{
   auto Bu = ContractTX2D(B,u);
   return ContractTY2D(B,Bu);
}

// 2D Tensor case with VDim components
template<int D1d, int Q1d, int VDim> MFEM_HOST_DEVICE inline
StaticTensor<dTensor<VDim>,D1d,D1d> InterpolateT(
   const dTensor<Q1d,D1d> &B,
   const StaticTensor<dTensor<VDim>,Q1d,Q1d> &u)
{
   auto Bu = ContractTX2D(B,u);
   return ContractTY2D(B,Bu);
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP
