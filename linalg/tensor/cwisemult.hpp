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
#include "diagonal_tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include "product.hpp"
#include <utility>

namespace mfem
{

// 1D
template <typename Container, typename Layout>
auto operator*(const DiagonalTensor<1,0,double,Container,Layout> &D,
               const DynamicDTensor<1> &u)
{
   const int Q = u.template Size<0>();
   DynamicDTensor<1> Du(Q);
   for(int q = 0; q < Q; ++q)
   {
      Du(q) = D(q)*u(q);
   }
   return Du;
}

template <typename Container, typename Layout, int Q>
auto operator*(const DiagonalTensor<1,0,double,Container,Layout> &D,
               const StaticDTensor<Q> &u)
{
   StaticDTensor<Q> Du;
   for(int q = 0; q < Q; ++q)
   {
      Du(q) = D(q)*u(q);
   }
   return Du;
}

// 2D
template <typename Container, typename Layout>
auto operator*(const DiagonalTensor<2,0,double,Container,Layout> &D,
               const DynamicDTensor<2> &u)
{
   const int Q = u.template Size<0>();
   DynamicDTensor<2> Du(Q,Q);
   for(int q2 = 0; q2 < Q; ++q2)
   {
      for(int q1 = 0; q1 < Q; ++q1)
      {
         Du(q1,q2) = D(q1,q2)*u(q1,q2);
      }
   }
   return Du;
}

template <typename Container, typename Layout, int Q>
auto operator*(const DiagonalTensor<2,0,double,Container,Layout> &D,
               const StaticDTensor<Q,Q> &u)
{
   StaticDTensor<Q,Q> Du;
   for(int q2 = 0; q2 < Q; ++q2)
   {
      for(int q1 = 0; q1 < Q; ++q1)
      {
         Du(q1,q2) = D(q1,q2)*u(q1,q2);
      }
   }
   return Du;
}

// 3D
template <typename Container, typename Layout>
auto operator*(const DiagonalTensor<3,0,double,Container,Layout> &D,
               const DynamicDTensor<3> &u)
{
   const int Q = u.template Size<0>();
   DynamicDTensor<3> Du(Q,Q,Q);
   for(int q3 = 0; q3 < Q; ++q3)
   {
      for(int q2 = 0; q2 < Q; ++q2)
      {
         for(int q1 = 0; q1 < Q; ++q1)
         {
            Du(q1,q2,q3) = D(q1,q2,q3)*u(q1,q2,q3);
         }
      }
   }
   return Du;
}

template <typename Container, typename Layout, int Q>
auto operator*(const DiagonalTensor<3,0,double,Container,Layout> &D,
               const StaticDTensor<Q,Q,Q> &u)
{
   StaticDTensor<Q,Q,Q> Du;
   for(int q3 = 0; q3 < Q; ++q3)
   {
      for(int q2 = 0; q2 < Q; ++q2)
      {
         for(int q1 = 0; q1 < Q; ++q1)
         {
            Du(q1,q2,q3) = D(q1,q2,q3)*u(q1,q2,q3);
         }
      }
   }
   return Du;
}

/// OLD CODE

// Non-tensor and 1D tensor coefficient-wise multiplication
template <typename T1, typename T2, int Q> MFEM_HOST_DEVICE inline
auto CWiseMult(const StaticTensor<T1,Q> &D, const StaticTensor<T2,Q> &u)
-> StaticTensor<decltype(D(0)*u(0)),Q>
{
   StaticTensor<decltype(D(0)*u(0)),Q> Du;
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      Du(q) = D(q) * u(q);
   }
   return Du;
}

// 3D tensor coefficient-wise multiplication
template <typename T1, typename T2, int Q1d> MFEM_HOST_DEVICE inline
auto CWiseMult(const StaticTensor<T1,Q1d,Q1d,Q1d> &D, const StaticTensor<T2,Q1d,Q1d,Q1d> &u)
-> StaticTensor<decltype(D(0,0,0)*u(0,0,0)),Q1d,Q1d,Q1d>
{
   StaticTensor<decltype(D(0,0,0)*u(0,0,0)),Q1d,Q1d,Q1d> Du;
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
auto CWiseMult(const StaticTensor<T1,Q1d,Q1d> &D, const StaticTensor<T2,Q1d,Q1d> &u)
-> StaticTensor<decltype(D(0,0)*u(0,0)),Q1d,Q1d>
{
   StaticTensor<decltype(D(0,0)*u(0,0)),Q1d,Q1d> Du;
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