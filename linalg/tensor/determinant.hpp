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

#ifndef MFEM_TENSOR_DET
#define MFEM_TENSOR_DET

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Determinant
template <typename T> MFEM_HOST_DEVICE inline
T&& Determinant(const Tensor<T,3,3> &J)
{
   return J(0,0)*J(1,1)*J(2,2)-J(0,2)*J(1,1)*J(2,0)
         +J(0,1)*J(1,2)*J(2,0)-J(0,1)*J(1,0)*J(2,2)
         +J(0,2)*J(1,0)*J(2,1)-J(0,0)*J(1,2)*J(2,1);
}

template <typename T> MFEM_HOST_DEVICE inline
T&& Determinant(const Tensor<T,2,2> &J)
{
   return J(0,0)*J(1,1)-J(0,1)*J(1,0);
}

template <typename T> MFEM_HOST_DEVICE inline
T&& Determinant(const Tensor<T,1,1> &J)
{
   return J(0,0);
}

// Computes determinant for all quadrature points
template<int Q,int Dim> MFEM_HOST_DEVICE inline
dTensor<Q>&& Determinant(const Tensor<dTensor<Dim,Dim>,Q> &J)
{
   dTensor<Q> det;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      det(q) = Determinant(J(q));
   }
   return std::move(det);
}

template<int Q1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d,Q1d>&& Determinant(const Tensor<dTensor<3,3>,Q1d,Q1d,Q1d> &J)
{
   dTensor<Q1d,Q1d,Q1d> det;
   for (int qz = 0; qz < Q1d; qz++)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1d)
         {
            det(qx,qy,qz) = Determinant(J(qx,qy,qz));
         }
      }
   }
   return std::move(det);
}

template<int Q1d> MFEM_HOST_DEVICE inline
dTensor<Q1d,Q1d>&& Determinant(const Tensor<dTensor<2,2>,Q1d,Q1d> &J)
{
   dTensor<Q1d,Q1d> det;
   MFEM_FOREACH_THREAD(qy,y,Q1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         det(qx,qy) = Determinant(J(qx,qy));
      }
   }
   return std::move(det);
}

} // namespace mfem

#endif // MFEM_TENSOR_DET
