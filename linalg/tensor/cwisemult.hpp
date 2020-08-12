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

#ifndef MFEM_TENSOR_CWISEMULT
#define MFEM_TENSOR_CWISEMULT

#include "tensor.hpp"
#include "../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Non-tensor and 1D tensor coefficient-wise multiplication
template <typename T1, typename T2, int Q> MFEM_HOST_DEVICE inline
auto CWiseMult(const Tensor<T1,Q> &D, const Tensor<T2,Q> &u)
-> Tensor<decltype(D(0)*u(0)),Q>&&
{
   Tensor<decltype(D(0)*u(0)),Q> Du;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      Du(q) = D(q) * u(q);
   }
   return Du;
}

// 3D tensor coefficient-wise multiplication
template <typename T1, typename T2, int Q1d> MFEM_HOST_DEVICE inline
auto CWiseMult(const Tensor<T1,Q1d,Q1d,Q1d> &D, const Tensor<T2,Q1d,Q1d,Q1d> &u)
-> Tensor<decltype(D(0)*u(0)),Q1d,Q1d,Q1d>&&
{
   Tensor<decltype(D(0)*u(0)),Q1d,Q1d,Q1d> Du;
   for (int qz = 0; qz < Q1d; qz++)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1d)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1d)
         {
            Du(qx,qy,qz) = D(qx,qy,qz) * u(qx,qy,qz);
         }
      }
   }
   return Du;
}

// 2D tensor coefficient-wise multiplication
template <typename T1, typename T2, int Q1d> MFEM_HOST_DEVICE inline
auto CWiseMult(const Tensor<T1,Q1d,Q1d> &D, const Tensor<T2,Q1d,Q1d> &u)
-> Tensor<decltype(D(0)*u(0)),Q1d,Q1d>&&
{
   Tensor<decltype(D(0)*u(0)),Q1d,Q1d> Du;
   MFEM_FOREACH_THREAD(qy,y,Q1d)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1d)
      {
         Du(qx,qy) = D(qx,qy) * u(qx,qy);
      }
   }
   return Du;
}

} // namespace mfem

#endif // MFEM_TENSOR_CWISEMULT