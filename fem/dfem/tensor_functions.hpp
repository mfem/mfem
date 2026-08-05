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
// #if defined(__clang__)
// #pragma clang attribute push (__attribute__((always_inline)), apply_to = function)
// #endif

namespace mfem
{
namespace future
{

/**
 * @brief Constructs a tensor of dual numbers from a tensor of values
 * @param[in] A The tensor of values
 * @note a d-order tensor's gradient will be initialized to the (2*d)-order identity tensor
 */
template <int... n> MFEM_HOST_DEVICE constexpr
auto make_dual(const tensor<real_t, n...>& A)
{
  tensor<dual<real_t, tensor<double, n...>>, n...> A_dual{};
  for_constexpr<n...>([&](auto... i) {
    A_dual(i...).value = A(i...);
    A_dual(i...).gradient(i...) = 1.0;
  });
  return A_dual;
}

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
 __attribute__((noinline))
 double smooth_max_eigenvalue_symm(const tensor<real_t, n, n>& A, double beta)
{
  auto [lambda, V] = eig_symm(get_value(A));
  double lambda_max = lambda[n - 1];
  double sum = 0;
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
dual<real_t, real_t> smooth_max_eigenvalue_symm_fwddiff(const tensor<real_t, n, n>& A, const tensor<real_t, n, n>& A_dot, double beta, double beta_dot)
{
  auto [lambda, V] = eig_symm(A);
  real_t lambda_max = lambda[n - 1];
  real_t sum = 0;
  tensor<double, n> eg;
  for (int i = 0; i < n; i++) {
    eg[i] = std::exp(beta*(lambda[i] - lambda_max));
    if (i != n - 1) sum += eg[i];
  }
  real_t value = lambda_max + std::log1p(sum)/beta;

  real_t derivative{};
  for (int mu = 0; mu < 3; mu++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        derivative += V[i][mu]*eg[mu]*V[j][mu]*A_dot[i][j]/(sum + 1.0);
      }
    }
  }
  WARN("CALLING CUSTOM DERIVATIVE (REF VER)");
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

// NOTE: I can't implmement this yet, the get_value() function for tensors is not implemented in MFEM.
// Consider upstreaming it from Smith.
// /**
//  * @overload
//  *
//  * Custom derivative rule. This variant accepts tensors with dual number entries.
//  */
// template <typename gradient_type, int n> MFEM_HOST_DEVICE
// auto smooth_max_eigenvalue_symm(const tensor<dual<real_t, gradient_type>, n, n>& A, double beta)
// {
//   using std::exp, std::log1p;
//   auto [lambda, V] = eig_symm(get_value(A));
//   double lambda_max = lambda[n - 1];
//   double sum = 0;
//   tensor<double, n - 1> eg;
//   for (int i = 0; i < n; i++) {
//     eg[i] = exp(beta*(lambda[i] - lambda_max));
//     if (i != (n - 1)) sum += eg[i];
//   }
//   double value = lambda_max + std::log1p(sum)/beta;

//   gradient_type gradient{};
//   for (int mu = 0; mu < n; mu++) {
//     for (int i = 0; i < n; i++) {
//       for (int j = 0; j < n; j++) {
//         gradient += V[i][mu]*eg[mu]*V[j][mu]*A[i][j].gradient/(sum + 1.0);
//       }
//     }
//   }
//   return dual<real_t, gradient_type>{value, gradient};
// }


} // namespace future
} // namespace mfem

// #if defined(__clang__)
// #pragma clang attribute pop
// #endif
