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
 * $$
 *     smooth_max_eigenvalue(A) = \frac{1}{\beta} \log\Big( \mathrm{tr}\big(\exp(\beta A) \big) \Big)
 * $$
 * which is equivalent to using the log-sum-exp function on the eigenvalues of A.
 *
 * @param A The input tensor
 * @param beta Sharpness parameter. Must be > 0. Larger values makes the approximation sharper.
 * @return Approximate maximum eigenvalue of A
 */
template <int n> MFEM_HOST_DEVICE
real_t smooth_max_eigenvalue_symm(const tensor<real_t, n, n>& A, real_t beta)
{
   auto [lambda, V] = eig_symm(A);
   real_t lambda_max = lambda[n - 1];
   real_t sum = 0;
   for (int i = 0; i < n - 1; i++)
   {
      sum += std::exp(beta*(lambda[i] - lambda_max));
   }
   return lambda_max + std::log1p(sum)/beta;
}

/**
 * @brief Differentiable approximation of minimum eigenvale of a symmetric tensor
 *
 * Estimates the minimum eigenvalue using
 * $$
 *     smooth_min_eigenvalue(A) = -\frac{1}{\beta} \log\Big( \mathrm{tr}\big(\exp(-\beta A) \big) \Big)
 * $$
 * which is equivalent to using the negated log-sum-exp function on the eigenvalues of -A.
 *
 * @param A The input tensor
 * @param beta Sharpness parameter. Must be > 0. Larger values makes the approximation sharper.
 * @return Approximate minimum eigenvalue of A
 */
template <int n> MFEM_HOST_DEVICE
real_t smooth_min_eigenvalue_symm(const tensor<real_t, n, n>& A, real_t beta)
{
   return -smooth_max_eigenvalue_symm<n>(-A, beta);
}

#ifdef MFEM_USE_ENZYME

namespace detail
{

// Custom forward-mode derivative rule for Enzyme
template<int n> MFEM_HOST_DEVICE
dual<real_t, real_t> smooth_max_eigenvalue_symm_fwddiff(
   const tensor<real_t, n, n>& A, const tensor<real_t, n, n>& A_dot, real_t beta,
   real_t beta_dot)
{
   auto [lambda, V] = eig_symm(A);
   real_t lambda_max = lambda[n - 1];
   real_t sum = 0;
   tensor<real_t, n> eg;
   tensor<real_t, n> lambda_shifted;
   for (int i = 0; i < n; i++)
   {
      lambda_shifted[i] = lambda[i] - lambda_max;
      eg[i] = std::exp(beta*lambda_shifted[i]);
      if (i != n - 1) { sum += eg[i]; }
   }
   real_t value = lambda_max + std::log1p(sum)/beta;

   real_t Z = sum + 1.0;
   real_t derivative{};
   for (int mu = 0; mu < n; mu++)
   {
      real_t w_mu = eg[mu]/Z;
      for (int i = 0; i < n; i++)
      {
         for (int j = 0; j < n; j++)
         {
            derivative += w_mu*V[i][mu]*V[j][mu]*A_dot[i][j];
         }
      }
   }
   derivative += (lambda_max - value + dot(eg, lambda_shifted)/Z)/beta * beta_dot;
   return {value, derivative};
}

// Types and functions for Enzyme custom reverse mode derivative
template <int n>
struct SmoothMaxEigenvalueSymmTape
{
   tensor<real_t, n> lambda;
   tensor<real_t, n, n> V;
   tensor<real_t, n> eg;
   real_t sum;
   real_t logZ;
};

template <int n>
struct SmoothMaxEigenvalueSymmAugmentedReturn
{
   void* tape;
   real_t value;
};

template <int n> MFEM_HOST_DEVICE
SmoothMaxEigenvalueSymmAugmentedReturn<n>
smooth_max_eigenvalue_symm_aug(const tensor<real_t, n, n>* A,
                               tensor<real_t, n, n>* A_bar,
                               real_t beta)
{
   (void)A_bar; // accumulated in reverse pass

   auto [lambda, V] = eig_symm(*A);
   const real_t lambda_max = lambda[n - 1];

   tensor<real_t, n> eg;
   real_t sum = 0;
   for (int i = 0; i < n; i++)
   {
      eg[i] = std::exp(beta*(lambda[i] - lambda_max));
      if (i != n - 1) { sum += eg[i]; }
   }

   const real_t logZ = std::log1p(sum);
   const real_t value = lambda_max + logZ/beta;

   auto* tape = static_cast<SmoothMaxEigenvalueSymmTape<n>*>(
                   std::malloc(sizeof(SmoothMaxEigenvalueSymmTape<n>)));
   if (tape)
   {
      tape->lambda = lambda;
      tape->V = V;
      tape->eg = eg;
      tape->sum = sum;
      tape->logZ = logZ;
   }

   return {static_cast<void*>(tape), value};
}

template <int n> MFEM_HOST_DEVICE
real_t smooth_max_eigenvalue_symm_rev(const tensor<real_t, n, n>* A,
                                      tensor<real_t, n, n>* A_bar,
                                      real_t beta,
                                      real_t out_bar,
                                      void* tape_ptr)
{
   (void)A; // all needed info is on the tape

   const auto* tape = static_cast<const SmoothMaxEigenvalueSymmTape<n>*>(tape_ptr);
   if (!tape)
   {
      return 0.0;
   }

   const real_t Z = tape->sum + 1.0;

   // d/dA = Σ_mu w_mu v_mu v_mu^T, where w_mu = eg[mu]/Z
   for (int mu = 0; mu < n; mu++)
   {
      const real_t w_mu = tape->eg[mu] / Z;
      for (int i = 0; i < n; i++)
      {
         for (int j = 0; j < n; j++)
         {
            (*A_bar)[i][j] += out_bar * w_mu * tape->V[i][mu] * tape->V[j][mu];
         }
      }
   }

   // d/dβ = -(log Z)/β^2 + (1/(β Z)) Σ_{i<n-1} exp(β(λ_i-λ_max)) (λ_i-λ_max)
   real_t dZ_dBeta = 0.0;
   const real_t& lambda_max = tape->lambda[n - 1];
   for (int i = 0; i < n - 1; i++)
   {
      dZ_dBeta += tape->eg[i] * (tape->lambda[i] - lambda_max);
   }

   const real_t beta2 = beta * beta;
   const real_t d_value_dBeta = -(tape->logZ)/beta2 + dZ_dBeta/(beta * Z);

   std::free(const_cast<SmoothMaxEigenvalueSymmTape<n>*>(tape));

   return out_bar * d_value_dBeta;
}

} // namespace detail

// Register custom derivatives (forward mode) with Enzyme
__attribute__((used))
void* __enzyme_register_derivative_smooth_max_eigenvalue_symm_2d[] =
{
   reinterpret_cast<void*>(smooth_max_eigenvalue_symm<2>),
   reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_fwddiff<2>)
};

__attribute__((used))
void* __enzyme_register_derivative_smooth_max_eigenvalue_symm_3d[] =
{
   reinterpret_cast<void*>(smooth_max_eigenvalue_symm<3>),
   reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_fwddiff<3>)
};

// Register custom gradients (combined reverse mode) with Enzyme
__attribute__((used))
void* __enzyme_register_gradient_smooth_max_eigenvalue_symm_2d[] =
{
   reinterpret_cast<void*>(smooth_max_eigenvalue_symm<2>),
   reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_aug<2>),
   reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_rev<2>)
};

__attribute__((used))
void* __enzyme_register_gradient_smooth_max_eigenvalue_symm_3d[] =
{
   reinterpret_cast<void*>(smooth_max_eigenvalue_symm<3>),
   reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_aug<3>),
   reinterpret_cast<void*>(detail::smooth_max_eigenvalue_symm_rev<3>)
};

#endif // MFEM_USE_ENZYME

} // namespace future
} // namespace mfem

#if defined(__clang__)
#pragma clang attribute pop
#endif
