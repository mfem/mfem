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

#include <cmath>
#include <limits>

#include "../config/config.hpp"
#include "general/enzyme.hpp"
#include "general/error.hpp"

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
  Bounds bounds; ///< Bounds on root
};
} // namespace future

namespace internal {
/// @cond DO_NOT_DOCUMENT

using future::SolverSettings;

template <auto f, typename T>
__attribute__((noinline))
MFEM_HOST_DEVICE void SolveNewtonBisection_impl(const real_t* x0_ptr, const T* p_ptr, const SolverSettings* settings_ptr, real_t* x_ptr)
{
    // It would be better to have the max iterations in the settings instead of
    // hard-coded.
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
        if ((((x - xh) * dr_dx - r)*((x - xl)*dr_dx - r) >= 0.0) ||  // Newton out of range
            (std::abs(2.0*r) > std::abs(dx_old*dr_dx))) {            // Newton decreasing bracket slower than bisection
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
    mfem_error("Univariate solve did not converge.");
}

template <auto f, typename T>
void SolveNewtonBisection_impl_fwddiff(const real_t* x0, const real_t* /* unused shadow */,
                                       const T* p, const T* dp,
                                       const SolverSettings* settings, const SolverSettings* /* unused shadow */,
                                       real_t* x, real_t* dx)
{
  SolveNewtonBisection_impl<f>(x0, p, settings, x);
  real_t dfdx = __enzyme_fwddiff<real_t>((void*)+f, enzyme_dup, *x, 1.0, enzyme_const, *p);
  real_t dfdp = __enzyme_fwddiff<real_t>((void*)+f, enzyme_const, *x, enzyme_dup, *p, *dp);
  *dx = -dfdp/dfdx;
}


template<auto f, typename T>
void SolveNewtonBisection_impl_aug(const real_t* x0, real_t* x0_bar,
                                   const T* p, T* p_bar,
                                   const SolverSettings* settings, SolverSettings* settings_bar,
                                   real_t* x, real_t* x_bar)
{
    SolveNewtonBisection_impl<f>(x0, p, settings, x);
    std::cout << "augmented forward, x = " << *x << std::endl;
}

// Change the residual function to return-by-reference so that there is a
// slot to provide the downstream cotangent (ie the shadow for y)
// in the reverse mode call.
template<auto f, typename T>
void wrapper(real_t x, T& p, real_t& y)
{
    y = f(x, p);
}

template<auto f, typename T>
void SolveNewtonBisection_impl_rev(const real_t* x0, real_t* x0_bar,
                                   const T* p, T* p_bar,
                                   const SolverSettings* settings, SolverSettings* settings_bar,
                                   real_t* x, real_t* x_bar)
{
    real_t drdx = __enzyme_fwddiff<real_t>((void*)+f, enzyme_dup, *x, 1.0, enzyme_const, *p);
    std::cout << "reverse pass" << std::endl;
    std::cout << "x = " << *x << std::endl;
    std::cout << "x from tape = " << *x << std::endl;
    std::cout << "drdx = " << drdx << std::endl;
    real_t lambda = -(*x_bar / drdx);
    std::cout << "lambda = " << lambda << std::endl;
    real_t r;
    __enzyme_autodiff<void>((void*)wrapper<f, T>, enzyme_const, *x, enzyme_dup, p, p_bar, enzyme_dupnoneed, &r, &lambda);

    std::cout << "p_bar = " << *p_bar << std::endl;

    // TODO: Make enzyme treat these as enzyme_const
    // The solution has no sensitivity to these parameters.
    *x0_bar = 0.0;
    *settings_bar = SolverSettings{};
    std::cout << "settings_bar.bounds.upper = " << settings_bar->bounds.upper << std::endl;
}

/// @endcond
} // namespace internal


namespace future {

/**
 * @brief Find the root of a univariate funtion
 */
template<auto f, typename T>
MFEM_HOST_DEVICE __attribute__((always_inline)) real_t SolveNewtonBisection(real_t x0, T p, SolverSettings settings) {
  real_t x;
  internal::SolveNewtonBisection_impl<f>(&x0, &p, &settings, &x);
  return x;
}


} // namespace future
} // namespace mfem

#endif // MFEM_USE_ENZYME
#endif // MFEM_UNIVARSOLVERS
