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
 * @file univarsolvers.hpp
 *
 * @brief Solvers of functions of a single variable suitable for use in ∂FEM q-functions.
 */

#ifndef MFEM_UNIVARSOLVERS
#define MFEM_UNIVARSOLVERS

#include "../config/config.hpp"
#include "../general/enzyme.hpp"

#ifdef MFEM_USE_ENZYME

namespace mfem {
namespace future {

/// Representation of bound constraints
struct Bounds {
  real_t lower, upper;
};

/// Settings for univariate solver
struct SolverSettings {
  real_t residual_abs_tol = 1e-10; ///< Tolerance for convergence check on absolute value of residual
  real_t residual_rel_tol = 0.0;   ///< Tolerance for convergence check on absolute value of current residual relative to absolute value of residual at initial guess
  Bounds bounds{.lower = -std::numeric_limits<real_t>::infinity(), .upper = std::numeric_limits<real_t>::infinity()}; ///< Bounds on root
};

template <auto f, typename T>
__attribute__((noinline))
MFEM_HOST_DEVICE void NewtonBisection_impl(const real_t* x0_ptr, const T* p_ptr, const SolverSettings* settings_ptr, real_t* x_ptr)
{
  constexpr int max_iters = 50;

  const real_t& x0 = *x0_ptr;
  const T& p = *p_ptr;
  const SolverSettings& settings = *settings_ptr;
  const real_t& left_bracket = settings.bounds.lower;
  const real_t& right_bracket = settings.bounds.upper;
  real_t& x = *x_ptr;
  using std::abs;

  auto fprime = [&p](real_t x) {
    real_t x_dot = 1.0;
    return __enzyme_fwddiff<real_t>((void*)+f, enzyme_dup, x, x_dot, enzyme_const, p);
  };

  real_t fl = f(left_bracket, p);
  real_t fh = f(right_bracket, p);

   // handle corner cases where one of the brackets is the root
  if (abs(fl) < settings.residual_abs_tol) {
    x = left_bracket;
    return;
  } else if (abs(fh) < settings.residual_abs_tol) {
    x = right_bracket;
    return;
  }

  MFEM_ASSERT(fl * fh < 0.0, "Root is not bracketed, solve cannot continue.");
  MFEM_ASSERT(x0 >= left_bracket && x0 <= right_bracket, "Initial guess must be within bounds.");

  // Orient search so that f(xl) < 0
  real_t xl = left_bracket;
  real_t xh = right_bracket;
  if (fl > 0.0) {
    xl = right_bracket;
    xh = left_bracket;
    real_t tmp = fl;
    fl = fh;
    fh = tmp;
  }

  real_t dx_old = abs(right_bracket - left_bracket);
  real_t dx = dx_old;
  x = x0;
  real_t r = f(x, p);
  real_t dr_dx = fprime(x);
  real_t r_old = r;
  for (int i = 0; i < max_iters; i++) {
    if ((((x - xh) * dr_dx - r)*((x - xl)*dr_dx - r) >= 0.0) || // Newton out of range
        (std::abs(2.0*r) > std::abs(dx_old*dr_dx))) {           // Newton decreasing bracket slower than bisection
      // Take bisection step
      dx_old = dx;
      dx = 0.5*(xh - xl);
      real_t x_old = x;
      x = xl + dx;
      if (x == x_old) return;
    } else {
      // Take Newton step
      dx_old = dx;
      dx = -r/dr_dx;
      real_t x_old = x;
      x += dx;
      if (x == x_old) return;
    }

    // update residual and jacobian
    r = f(x, p);
    dr_dx = fprime(x);

    // Check convergence
    if (abs(r) < settings.residual_rel_tol*r_old || abs(r) < settings.residual_abs_tol) {
      return;
    }

    // Update bracket
    if (r < 0.0) {
      xl = x;
      fl = r;
    } else {
      xh = x;
      fh = r;
    }
  }
  mfem_error("Newton solve did note converge.");
}

template <auto f, typename T>
void NewtonBisection_impl_fwddiff(const real_t* x0, const real_t* /* unused shadow */,
                                   const T* p, const T* dp,
                                   const SolverSettings* settings, const SolverSettings* /* unused shadow */,
                                   real_t* x, real_t* dx)
{
  NewtonBisection_impl<f>(x0, p, settings, x);
  real_t dfdx = __enzyme_fwddiff<real_t>((void*)+f, enzyme_dup, *x, 1.0, enzyme_const, *p);
  real_t dfdp = __enzyme_fwddiff<real_t>((void*)+f, enzyme_const, *x, enzyme_dup, *p, *dp);
  *dx = -dfdp/dfdx;
}

/**
 * @brief Find the root of a univariate funtion
 */
template<auto f, typename T>
MFEM_HOST_DEVICE real_t SolveNewtonBisection(real_t x0, T p, SolverSettings settings) {
  real_t x;
  NewtonBisection_impl<f>(&x0, &p, &settings, &x);
  return x;
}


} // namespace future
} // namespace mfem

#endif // MFEM_USE_ENZYME
#endif // MFEM_UNIVARSOLVERS
