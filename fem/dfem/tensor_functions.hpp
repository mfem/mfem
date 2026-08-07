// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

/**
 * @file tensor_functions.hpp
 *
 * @brief Differentiable functions of tensors
 */

#pragma once

#include <cmath>

#include "../../linalg/dual.hpp"
#include "../../linalg/tensor.hpp"
#include "tuple.hpp"
#include "util.hpp"

// Force-inline every tensor operation under clang
#if defined(__clang__)
#pragma clang attribute push (__attribute__((always_inline)), apply_to = function)
#endif

namespace mfem
{
namespace future
{

/**
 * @brief Differentiable approximation of maximum eigenvale of a symmetric tensor
 *
 * Estimates the maximum eigenvalue using
 * \f[
 *     smooth_max_eigenvalue(A) = \log\Big( \mathrm{tr}\big(\exp(\beta A) \big) \Big)
 * \f]
 * which is equivalent to using the log-sum-exp function on the eigenvalues of A.
 *
 * @param A The input tensor
 * @param beta Sharpness parameter. Must be > 0. Larger values makes the approximation sharper.
 * @return Approximate maximum eigenvalue of A
 */
 template <int n> MFEM_HOST_DEVICE
 real_t smooth_max_eigenvalue_symm(const tensor<real_t, n, n>& A, real_t beta)
{
  auto [lambda, V] = eig_symm(get_value(A));
  real_t lambda_max = lambda[n - 1];
  real_t sum = 0;
  for (int i = 0; i < n - 1; i++) {
    sum += std::exp(beta*(lambda[i] - lambda_max));
  }
  return lambda_max + std::log1p(sum)/beta;
}

#ifdef MFEM_USE_ENZYME
// Custom forward-mode derivative rule for Enzyme

namespace detail
{

template<int n> MFEM_HOST_DEVICE
dual<real_t, real_t> smooth_max_eigenvalue_symm_fwddiff(const tensor<real_t, n, n>& A, const tensor<real_t, n, n>& A_dot, real_t beta, real_t beta_dot)
{
  WARN("CALLING CUSTOM DERIVATIVE (REF VER)");
  auto [lambda, V] = eig_symm(A);
  real_t lambda_max = lambda[n - 1];
  real_t sum = 0;
  tensor<real_t, n> eg;
  for (int i = 0; i < n; i++) {
    eg[i] = std::exp(beta*(lambda[i] - lambda_max));
    if (i != n - 1) sum += eg[i];
  }
  real_t value = lambda_max + std::log1p(sum)/beta;

  real_t derivative{};
  for (int mu = 0; mu < n; mu++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        derivative += V[i][mu]*eg[mu]*V[j][mu]*A_dot[i][j]/(sum + 1.0);
      }
    }
  }
  return {value, derivative};
}

} // namespace detail

// Register custom derivatives with Enzyme
__attribute__((used))
void* __enzyme_register_derivative_smooth_max_eigenvalue_symm_2d[] = {
    reinterpret_cast<void*>(smooth_max_eigenvalue_symm<2>),
    reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_fwddiff<2>)
  };

__attribute__((used))
void* __enzyme_register_derivative_smooth_max_eigenvalue_symm_3d[] = {
    reinterpret_cast<void*>(smooth_max_eigenvalue_symm<3>),
    reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_fwddiff<3>)
  };

#endif // MFEM_USE_ENZYME

} // namespace future
} // namespace mfem

#if defined(__clang__)
#pragma clang attribute pop
#endif
