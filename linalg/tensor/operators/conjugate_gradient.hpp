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
#include "../tensor_traits.hpp"
#include "scalar_multiplication.hpp"
#include "addition.hpp"
#include "element_operator.hpp"

namespace mfem
{

// get_cg_result_type
template <typename Matrix, typename T>
struct get_cg_result_type_t;

template <typename Matrix, typename T>
using get_cg_result_type = typename get_cg_result_type_t<Matrix,T>::type;

template <typename QData, typename TrialBasis, typename TestBasis,
          typename C, typename L>
struct get_cg_result_type_t<ElementOperator<QData,TrialBasis,TestBasis>,Tensor<C,L>>
{
   using sizes = get_tensor_sizes<Tensor<C,L>>;
   using type = instantiate< basis_result_tensor<TrialBasis>::template type, sizes>;
};

// get_cg_value_type
template <typename T>
struct get_cg_value_type_t;

template <typename T>
using get_cg_value_type = typename get_cg_value_type_t<T>::type;

template <typename C, typename L>
struct get_cg_value_type_t<Tensor<C,L>>
{
   using type = get_tensor_value_type<Tensor<C,L>>;
};

template<typename Matrix, typename Rhs, typename Preconditioner>
MFEM_HOST_DEVICE
auto conjugate_gradient(const Matrix& A, const Rhs& rhs,
                        const Preconditioner& P, int& iters,
                        double& tol_error)
{
   using Scalar = get_cg_value_type<Rhs>;
   using Index = int;
   using Vector = get_cg_result_type<Matrix,Rhs>;

   Scalar tol = tol_error;
   Index maxIters = iters;

   Vector x(rhs);
   // one_print("\n  ||x|| %d of element %d: %e\n", 0, MFEM_BLOCK_ID(x), SquaredNorm(x));
   Vector residual(rhs);
   // one_print("  ||residual|| %d of element %d: %e\n", 0, MFEM_BLOCK_ID(x), SquaredNorm(residual));
   residual -= A * x; //initial residual
   // one_print("||Ax|| %d of element %d: %e\n", 0, MFEM_BLOCK_ID(x), SquaredNorm(residual));

   Scalar rhsNorm2 = SquaredNorm(rhs);
   if(rhsNorm2 == 0)
   {
      x = 0;
      iters = 0;
      // one_print("==> Number of iterations for element %d: %d\n\n",MFEM_BLOCK_ID(x),0);
      tol_error = 0;
      return x;
   }
   Scalar threshold = tol*tol*rhsNorm2;
   Scalar residualNorm2 = SquaredNorm(residual);
   // one_print("  residualNorm2 %d of element %d: %e\n", 0, MFEM_BLOCK_ID(x), residualNorm2);
   if (residualNorm2 < threshold)
   {
      iters = 0;
      // one_print("==> Number of iterations for element %d: %d\n\n",MFEM_BLOCK_ID(x),0);
      tol_error = sqrt(residualNorm2 / rhsNorm2);
      return x;
   }

   Vector p = residual; //P * residual;      // initial search direction
   // one_print("  ||p|| %d of element %d: %e\n", 0, MFEM_BLOCK_ID(x), SquaredNorm(p));

   Scalar absNew = Dot(residual,p);  // the square of the absolute value of r scaled by invM
   Index i = 0;
   // one_print("  Residual norm %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), absNew);
   while(i < maxIters)
   {
      Vector tmp = A * p;                    // the bottleneck of the algorithm
      // one_print("\n||Ap|| %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), SquaredNorm(tmp));

      Scalar alpha = absNew / Dot(p,tmp);         // the amount we travel on dir
      // one_print("  alpha %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), alpha);
      x += alpha * p;                             // update solution
      // one_print("  ||p|| %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), SquaredNorm(p));
      // one_print("  ||x|| %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), SquaredNorm(x));
      residual -= alpha * tmp;                    // update residual

      residualNorm2 = SquaredNorm(residual);
      // one_print("  residualNorm2 %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), residualNorm2);
      if(residualNorm2 < threshold)
      {
         // one_print("==> Number of iterations for element %d: %d\n\n",MFEM_BLOCK_ID(x),i);
         return x;
      }

      Vector z = residual; // P * residual;                // approximately solve for "A z = residual"
      // one_print("  ||z|| %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), SquaredNorm(z));

      Scalar absOld = absNew;
      absNew = Dot(residual,z);     // update the absolute value of r
      Scalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                           // update search direction
      i++;
      // one_print("  Residual norm %d of element %d: %e\n", i, MFEM_BLOCK_ID(x), absNew);
   }
   tol_error = sqrt(residualNorm2 / rhsNorm2);
   iters = i;
   // one_print("==> Number of iterations for element %d: %d\n\n",MFEM_BLOCK_ID(x),i);
   return x;
}

} // namespace mfem

#endif // MFEM_TENSOR_CG
