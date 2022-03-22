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

#ifndef MFEM_TENSOR_MATRIX_INVERSE
#define MFEM_TENSOR_MATRIX_INVERSE

#include "../tensor_types.hpp"
#include "../utilities/foreach.hpp"
#include "determinant.hpp"

namespace mfem
{

/**
 * @brief Inverts a matrix
 * @param[in] A The matrix to invert
 * @note Uses a shortcut for inverting a 2-by-2 matrix
 */
StaticTensor<double, 2, 2> inv(const StaticTensor<double, 2, 2>& A)
{
   double inv_detA(1.0 / Determinant(A));

   StaticTensor<double, 2, 2> invA{};

   invA(0,0) =  A(1,1) * inv_detA;
   invA(0,1) = -A(0,1) * inv_detA;
   invA(1,0) = -A(1,0) * inv_detA;
   invA(1,1) =  A(0,0) * inv_detA;

   return invA;
}

/**
 * @overload
 * @note Uses a shortcut for inverting a 3-by-3 matrix
 */
StaticTensor<double, 3, 3> inv(const StaticTensor<double, 3, 3>& A)
{
   double inv_detA(1.0 / Determinant(A));

   StaticTensor<double, 3, 3> invA{};

   invA(0,0) = (A(1,1) * A(2,2) - A(1,2) * A(2,1)) * inv_detA;
   invA(0,1) = (A(0,2) * A(2,1) - A(0,1) * A(2,2)) * inv_detA;
   invA(0,2) = (A(0,1) * A(1,2) - A(0,2) * A(1,1)) * inv_detA;
   invA(1,0) = (A(1,2) * A(2,0) - A(1,0) * A(2,2)) * inv_detA;
   invA(1,1) = (A(0,0) * A(2,2) - A(0,2) * A(2,0)) * inv_detA;
   invA(1,2) = (A(0,2) * A(1,0) - A(0,0) * A(1,2)) * inv_detA;
   invA(2,0) = (A(1,0) * A(2,1) - A(1,1) * A(2,0)) * inv_detA;
   invA(2,1) = (A(0,1) * A(2,0) - A(0,0) * A(2,1)) * inv_detA;
   invA(2,2) = (A(0,0) * A(1,1) - A(0,1) * A(1,0)) * inv_detA;

   return invA;
}

} // namespace mfem

#endif // MFEM_TENSOR_MATRIX_INVERSE
