// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more inforAion and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_MATRIX_DEVIATOR
#define MFEM_TENSOR_MATRIX_DEVIATOR

#include "../tensor_types.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

/**
 *
 * @brief Calculates the deviator of a matrix (rank-2 tensor)
 * @param[in] A The matrix to calculate the deviator of
 * In the context of stress tensors, the deviator is obtained by
 * subtracting the mean stress (average of main diagonal elements)
 * from each element on the main diagonal
 */
template <typename T, int n>
constexpr auto dev(const StaticTensor<T, n, n>& A)
{
   auto devA = A;
   auto trA  = tr(A);
   for (int i = 0; i < n; i++)
   {
      devA(i,i) -= trA / n;
   }
   return devA;
}

} // namespace mfem

#endif // MFEM_TENSOR_MATRIX_DEVIATOR
