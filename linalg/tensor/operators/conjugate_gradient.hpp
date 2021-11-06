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
   using Result = get_cg_result_type<Matrix,Rhs>;
   
   Scalar tol = tol_error;
   Index maxIters = iters;

   Result x(rhs);
   Result residual(rhs);
   residual -= A * x; //initial residual

   Scalar rhsNorm2 = SquaredNorm(rhs);
   if(rhsNorm2 == 0) 
   {
      x = 0;
      iters = 0;
      tol_error = 0;
      return x;
   }
   Scalar threshold = tol*tol*rhsNorm2;
   Scalar residualNorm2 = SquaredNorm(residual);
   if (residualNorm2 < threshold)
   {
      iters = 0;
      tol_error = sqrt(residualNorm2 / rhsNorm2);
      return x;
   }

   auto p = P * residual;      // initial search direction

   Scalar absNew = Dot(residual,p);  // the square of the absolute value of r scaled by invM
   Index i = 0;
   while(i < maxIters)
   {
      auto tmp = A * p;                    // the bottleneck of the algorithm

      Scalar alpha = absNew / Dot(p,tmp);         // the amount we travel on dir
      x += alpha * p;                             // update solution
      residual -= alpha * tmp;                    // update residual
      
      residualNorm2 = SquaredNorm(residual);
      if(residualNorm2 < threshold) return x;
      
      auto z = P * residual;                // approximately solve for "A z = residual"

      Scalar absOld = absNew;
      absNew = Dot(residual,z);     // update the absolute value of r
      Scalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                           // update search direction
      i++;
   }
   tol_error = sqrt(residualNorm2 / rhsNorm2);
   iters = i;
   return x;
}

} // namespace mfem

#endif // MFEM_TENSOR_CG
