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

#ifndef MFEM_TENSOR_MATRIX_TRACE
#define MFEM_TENSOR_MATRIX_TRACE

#include "../tensor_types.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

/**
 * @brief Returns the trace of a square matrix
 * @param[in] A The matrix to compute the trace of
 * @return The sum of the elements on the main diagonal
 */
template <typename T, int n>
constexpr auto tr(const StaticTensor<T, n, n>& A)
{
   T trA{};
   for (int i = 0; i < n; i++)
   {
      trA = trA + A(i,i);
   }
   return trA;
}

} // namespace mfem

#endif // MFEM_TENSOR_MATRIX_TRACE
