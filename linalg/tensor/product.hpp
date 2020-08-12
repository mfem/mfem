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

#ifndef MFEM_TENSOR_PROD
#define MFEM_TENSOR_PROD

#include "tensor.hpp"
#include "../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Matrix-Vector multiplication (local to the thread)
template <typename T, int H, int W> MFEM_HOST_DEVICE inline
Tensor<T,H>&& operator*(const Tensor<T,H,W> &A, const Tensor<T,W> &u)
{
  Tensor<T,H> v;
  for (int h = 0; h < H; h++)
  {
    T val = 0.0;
    for (int w = 0; w < W; w++)
    {
      val += A(h,w) * u(w);
    }
    v(h) = val;
  }
  return v;
}

// Multiplication of a Vector by a scalar.
template <typename T,int H> MFEM_HOST_DEVICE inline
Tensor<T,H>&& operator*(const T &a, const Tensor<T,H> &u)
{
  Tensor<T,H> v;
  for (int h = 0; h < H; h++)
  {
    v(h) = a * u(h);
  }
  return v;
}

// Multiplication of a Matrix by a scalar.
template <typename T, int H, int W> MFEM_HOST_DEVICE inline
Tensor<T,H>&& operator*(const T &a, const Tensor<T,H,W> &U)
{
  Tensor<T,H,W> V;
  for (int h = 0; h < H; h++)
  {
    for (int w = 0; w < W; w++)
    {
      V(h,w) = a * U(h,w);
    }
  }
  return V;
}

} // namespace mfem

#endif // MFEM_TENSOR_PROD