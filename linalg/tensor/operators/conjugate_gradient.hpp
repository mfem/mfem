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

#ifndef MFEM_TENSOR_CG
#define MFEM_TENSOR_CG

#include "../../../general/forall.hpp"
#include "matrix_multiplication.hpp"

namespace mfem
{

template<typename Matrix, typename Rhs, typename Preconditioner>
MFEM_HOST_DEVICE
auto conjugate_gradient(const Matrix& A, const Rhs& rhs,
                        const Preconditioner& P, int& iters,
                        double& tol_error)
{
   using Scalar = get_matrix_type<Rhs>;
   using RealScalar = Scalar;
   using Index = int;
   
   RealScalar tol = tol_error;
   Index maxIters = iters;

   auto layout = GetLayout(rhs);
   StaticResultTensor<Rhs,decltype(layout)> x(layout);
   StaticResultTensor<Rhs,decltype(layout)> residual(rhs);
   residual -= A * x; //initial residual
  
   RealScalar rhsNorm2 = SquaredNorm(rhs);
   if(rhsNorm2 == 0) 
   {
      x = 0;
      iters = 0;
      tol_error = 0;
      return x;
   }
   const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
   RealScalar threshold = max(RealScalar(tol*tol*rhsNorm2),considerAsZero);
   RealScalar residualNorm2 = SquaredNorm(residual);
   if (residualNorm2 < threshold)
   {
      iters = 0;
      tol_error = sqrt(residualNorm2 / rhsNorm2);
      return x;
   }
  
   StaticResultTensor<Rhs,NRows> p(n_rows);
   p = P * residual;      // initial search direction
  
   StaticResultTensor<Rhs,NCols> z(n_cols), tmp(n_cols);
   RealScalar absNew = Dot(residual,p);  // the square of the absolute value of r scaled by invM
   Index i = 0;
   while(i < maxIters)
   {
      tmp = A * p;                    // the bottleneck of the algorithm

      Scalar alpha = absNew / Dot(p,tmp);         // the amount we travel on dir
      x += alpha * p;                             // update solution
      residual -= alpha * tmp;                    // update residual
      
      residualNorm2 = SquaredNorm(residual);
      if(residualNorm2 < threshold) return x;
      
      z = P * residual;                // approximately solve for "A z = residual"

      RealScalar absOld = absNew;
      absNew = Dot(residual,z);     // update the absolute value of r
      RealScalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                           // update search direction
      i++;
   }
   tol_error = sqrt(residualNorm2 / rhsNorm2);
   iters = i;
   return x;
}

} // namespace mfem

#endif // MFEM_TENSOR_CG
