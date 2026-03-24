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

#include <cmath>

#include "mfem.hpp"
#include "unit_tests.hpp"

extern int enzyme_strong_zero;

using mfem::real_t;
using namespace mfem::future;

MFEM_HOST_DEVICE inline real_t FlowResistance(real_t eqps, real_t sigma_y, real_t n, real_t ep_0)
{
    return sigma_y*(1.0 + std::pow((eqps + 1e-8)/ep_0, n));
}

using J2PlasticityParameters = tuple<real_t, real_t, real_t, real_t, real_t, real_t>;

real_t J2PlasticityResidual(real_t delta_eqps, J2PlasticityParameters p)
{
    auto [eqps, q, G, sigma_y, n, ep_0] = p;
    return q - 3.0*G*delta_eqps - FlowResistance(eqps + delta_eqps, sigma_y, n, ep_0);
}

struct J2Plasticity {
  static constexpr int dim = 3;  ///< spatial dimension
  static constexpr int N_INTERNAL_STATES = 10;
  static constexpr real_t tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  real_t E;        ///< Young's modulus
  real_t nu;       ///< Poisson's ratio
  real_t sigma_y;  ///< Yield strength
  real_t n;        ///< Hardening index
  real_t ep_0;     ///< Reference plastic strain
  
  /// @brief variables required to characterize the hysteresis response
  struct InternalState {
    tensor<real_t, dim, dim> plastic_strain;
    real_t accumulated_plastic_strain;
  };

  using PackedInternalState = mfem::future::tensor<real_t, N_INTERNAL_STATES>;

  MFEM_HOST_DEVICE static inline InternalState unpack_internal_state(
      const mfem::future::tensor<real_t, N_INTERNAL_STATES>& packed_state)
  {
    auto plastic_strain =
        mfem::future::make_tensor<dim, dim>([&packed_state](int i, int j) { return packed_state[dim * i + j]; });
    real_t accumulated_plastic_strain = packed_state[N_INTERNAL_STATES - 1];
    return {plastic_strain, accumulated_plastic_strain};
  }

  MFEM_HOST_DEVICE static inline PackedInternalState pack_internal_state(
      const mfem::future::tensor<real_t, dim, dim>& plastic_strain, real_t accumulated_plastic_strain)
  {
    PackedInternalState packed_state{};
    for (int i = 0, ij = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++, ij++) {
        packed_state[ij] = plastic_strain[i][j];
      }
    }
    packed_state[N_INTERNAL_STATES - 1] = accumulated_plastic_strain;
    return packed_state;
  }

  MFEM_HOST_DEVICE inline tuple<tensor<real_t, dim, dim>, PackedInternalState>
  update(tensor<real_t, dim, dim> dudxi,
         PackedInternalState internal_state,
         tensor<real_t, dim, dim> J,
         real_t w) const
  {
    auto invJ = inv(J);
    const auto dudX = dudxi * invJ;
    auto I = IdentityMatrix<dim>();
    const real_t K = E / (3.0 * (1.0 - 2.0 * nu));
    const real_t G = 0.5 * E / (1.0 + nu);

    auto [plastic_strain, accumulated_plastic_strain] = unpack_internal_state(internal_state);
    
    auto el_strain = sym(dudX) - plastic_strain;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * dev(el_strain);
    auto q = std::sqrt(1.5) * norm(s);
    real_t denom = q > 0.0? q : 1.0;
    auto Np = 1.5 * s / denom;

    if (q > FlowResistance(accumulated_plastic_strain, sigma_y, n, ep_0)) {
        real_t lb = 0.0;
        // FIXME: Reverse mode derivative produces nans if ub depends on state
        // real_t ub = (q - FlowResistance(accumulated_plastic_strain, sigma_y, n, ep_0))/(3*G);
        real_t ub = 2.0;
        SolverSettings settings{.residual_abs_tol = 1e-10*sigma_y, .residual_rel_tol = 1e-10, 
                                .bounds{.lower = lb, .upper = ub}};
        real_t delta_eqps = SolveNewtonBisection<J2PlasticityResidual>(
            0.5*(lb + ub), make_tuple(accumulated_plastic_strain, q, G, sigma_y, n, ep_0), settings);
        accumulated_plastic_strain += delta_eqps;
        plastic_strain += delta_eqps * Np;
        s -= 2.0 * G * delta_eqps * Np;
    }
    auto Q_new = pack_internal_state(plastic_strain, accumulated_plastic_strain);
    auto stress = s + p * I;
    const real_t dV = det(J)*w;
    // Question: if I use make_tuple as in this comment, I get a segfault in
    // derivatives of this function. Is this expected?
    // return make_tuple(stress*transpose(invJ)*dV, Q_new);
    return {stress*transpose(invJ)*dV, Q_new};
  }
  
  MFEM_HOST_DEVICE inline tensor<real_t, dim, dim>
  stress(tensor<real_t, dim, dim> dudxi,
         PackedInternalState internal_state,
         tensor<real_t, dim, dim> J,
         real_t w) const
  {
    auto [stress, internal_state_new] = update(dudxi, internal_state, J, w);
    return stress;
  }

  MFEM_HOST_DEVICE inline PackedInternalState
  internal_state_new(tensor<real_t, dim, dim> dudxi,
                     PackedInternalState internal_state,
                     tensor<real_t, dim, dim> J,
                     real_t w) const
  {
    auto [stress, internal_state_new] = update(dudxi, internal_state, J, w);
    return internal_state_new;
  }
};


// Register the custom derivative for the solver.
// This needs to be done for every residual function that the solver is applied on,
// since the SolveNewtonBisection_impl is a function template, and we need a real
// function with an address to specify the custom derivative.
__attribute__((used))
void *  __enzyme_register_derivative_newton_bisection_on_j2[2] = {
  (void*) mfem::internal::SolveNewtonBisection_impl<J2PlasticityResidual, J2PlasticityParameters>,
  (void*) mfem::internal::SolveNewtonBisection_impl_fwddiff<J2PlasticityResidual, J2PlasticityParameters>
};

tensor<real_t, 3, 3> ComputeStress(
    J2Plasticity* material, tensor<real_t, 3, 3> dudxi, 
    J2Plasticity::PackedInternalState Q, tensor<real_t, 3, 3> J, real_t w)
{
    return material->stress(dudxi, Q, J, w);
}

template <int dim>
real_t norm_inf(tensor<real_t, dim, dim> A) {
  real_t maxval = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      maxval = std::max(std::abs(A[i][j]), maxval);
    }
  }
  return maxval;
}

void ComputeStressRef(const J2Plasticity* material, const tensor<real_t, 3, 3>& dudxi,
                      const J2Plasticity::PackedInternalState& Q,
                      const tensor<real_t, 3, 3>& J, real_t w,
                      tensor<real_t, 3, 3>& sigma)
{
  sigma = material->stress(dudxi, Q, J, w);
}


__attribute__((used))
void* __enzyme_register_gradient_SolveNewtonBisectionJ2[3] = {
  (void*)mfem::internal::SolveNewtonBisection_impl<J2PlasticityResidual, J2PlasticityParameters>,
  (void*)mfem::internal::SolveNewtonBisection_impl_aug<J2PlasticityResidual, J2PlasticityParameters>,
  (void*)mfem::internal::SolveNewtonBisection_impl_rev<J2PlasticityResidual, J2PlasticityParameters>
};

TEST_CASE("Univariate function solver on qfunction", "[univar]")
{
    J2Plasticity material{.E = 70.0e3, .nu = 0.34, .sigma_y = 240.0, .n = 0.15, .ep_0 = 1e-3};
    tensor<real_t, 3, 3> H{{{0.947667  , 0.9785799 , 0.33229148},
                            {0.46866846, 0.5698887 , 0.16550303},
                            {0.3101946 , 0.68948054, 0.74676657}}};
    J2Plasticity::PackedInternalState Q{};
    const tensor<real_t, 3, 3> J = IdentityMatrix<3>();
    const real_t w = 1.0;

    SECTION("Correctness")
    {
        auto [stress, Q_new] = material.update(H, Q, IdentityMatrix<3>(), 1.0);
        INFO("stress = " << stress << "\ninternal vars =" << Q_new << "\n");
        real_t mises = std::sqrt(1.5)*norm(dev(stress));
        real_t eqps = Q_new[9];
        real_t Y = FlowResistance(eqps, material.sigma_y, material.n, material.ep_0);
        REQUIRE(mises == MFEM_Approx(Y, 0.0, 1e-8));
    }

    SECTION("JVP")
    {
        tensor<real_t, 3, 3> H_dot{{{1.0, 0.0 , 0.0},
                                    {0.0, 0.0 , 0.0},
                                    {0.0, 0.0 , 0.0}}};

        // Enzyme directional derivative (uses custom derivative of solver)
        auto sigma_dot = __enzyme_fwddiff<tensor<real_t, 3, 3>>((void*)ComputeStress, 
                                                                enzyme_const, &material,
                                                                enzyme_dup, H, H_dot,
                                                                enzyme_const, Q,
                                                                enzyme_const, J,
                                                                enzyme_const, w);
        INFO("sigma dot = " << sigma_dot << "\n");
        REQUIRE(sigma_dot[0][0] > 0.0);

        // Finite difference derivative
        constexpr int dim = 3;
        real_t eps = 1e-5;
        tensor<real_t, 3, 3> sigma = ComputeStress(&material, H, Q, J, w);
        tensor<real_t, 3, 3> sigma_p = ComputeStress(&material, H + eps*H_dot, Q, J, w);
        tensor<real_t, 3, 3> sigma_dot_h = (1.0/eps)*(sigma_p - sigma);
        INFO("sigma dot_h = " << sigma_dot_h << "\n");

        tensor<real_t, 3, 3> rel_error = sigma_dot - sigma_dot_h;
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            real_t denom = sigma[i][j] != 0? sigma[i][j] : 1.0;
            rel_error[i][j] /= denom;
          }
        }
        INFO("real errors = " << rel_error);
        REQUIRE(norm_inf(rel_error) < 1e-5);
    }

    SECTION("VJP")
    {
      tensor<real_t, 3, 3> sigma;
      ComputeStressRef(&material, H, Q, J, w, sigma);
      double epsilon = 1e-5;
      tensor<real_t, 3, 3> dH{{{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
      auto H_p = H + epsilon*dH;
      tensor<real_t, 3, 3> sigma_p;
      ComputeStressRef(&material, H_p, Q, J, w, sigma_p);
      auto sigma_dot = (sigma_p - sigma)/epsilon;
      INFO("sigma_dot = " << sigma_dot << "\n");


      tensor<real_t, 3, 3> sigma_bar{{{1.0, 0.0, 0.0},
                                      {0.0, 0.0, 0.0},
                                      {0.0, 0.0, 0.0}}};
      INFO("H = " << H);
      INFO("Q = " << Q);

      J2Plasticity material_bar;
      tensor<real_t, 3, 3> H_bar{};
      J2Plasticity::PackedInternalState Q_bar{};
      tensor<real_t, 3, 3> J_bar{};
      __enzyme_autodiff<void>(
        (void*)ComputeStressRef,  enzyme_strong_zero, enzyme_const, &material, enzyme_dup, &H, &H_bar,
        enzyme_dup, &Q, &Q_bar, enzyme_dup, &J, &J_bar, enzyme_const, w,
        enzyme_dup, &sigma, &sigma_bar);
      std::cout << "Check that console output works" << std::endl;
      INFO("H_bar = " << H_bar);
      INFO("Q_bar = " << Q_bar);
      INFO("J_bar = " << J_bar);
      REQUIRE(norm(H_bar) == 0);
    }
}


real_t nthroot_res(real_t x, tuple<real_t, real_t> p)
{
    auto [index, radicand] = p;
    return std::pow(x, index) - radicand;
}

real_t sqrt_res(real_t x, real_t p)
{
  return x*x - p;
}

__attribute__((used))
void* __enzyme_register_gradient_solver[3] = {
  (void*)mfem::internal::SolveNewtonBisection_impl<nthroot_res, tuple<real_t, real_t>>,
  (void*)mfem::internal::SolveNewtonBisection_impl_aug<nthroot_res, tuple<real_t, real_t>>,
  (void*)mfem::internal::SolveNewtonBisection_impl_rev<nthroot_res, tuple<real_t, real_t>>
};

real_t mysqrt(real_t x)
{
  real_t x0 = x;
  real_t index = 2.0;
  SolverSettings settings{.bounds = {.lower = 0, .upper = 10.0}};
  return SolveNewtonBisection<nthroot_res>(x0, make_tuple(index, x), settings);
}

TEST_CASE("Univariate solver reverse mode", "[univar]")
{
  real_t x = 2.0;
  real_t y = mysqrt(x);
  std::cout << "x = " << x << " sqrt(x) = " << y << std::endl;
  REQUIRE(y == MFEM_Approx(M_SQRT2, 0.0, 1e-8));

  std::cout << "Computing derivative" << std::endl;
  real_t dydx = __enzyme_autodiff<real_t>((void*)mysqrt, enzyme_out, x);
  REQUIRE(dydx == MFEM_Approx(0.5/std::sqrt(2.0)));
}

TEST_CASE("Univariate function solver robustness", "[univar]")
{
    SECTION("Simple case")
    {
        auto Nthroot = [](real_t x, real_t n) {
            real_t x0 = std::max(x, 1.0);
            SolverSettings settings{.bounds{0.0, x0}};
            return SolveNewtonBisection<nthroot_res>(x0, make_tuple(n, x), settings);
        };
        real_t x = 8.0;
        real_t y = Nthroot(x, 3.0);
        REQUIRE(y == MFEM_Approx(2.0));
    }

    SECTION("Stiff problem")
    {
        auto f = [](real_t x, real_t p) { return std::pow(x, p) - 1.0; };
        real_t x0 = 0.1;
        real_t p = 50;
        SolverSettings settings{.bounds{.lower = 0.0, .upper = 5.1}};
        real_t x = SolveNewtonBisection<+f>(x0, p, settings);
        REQUIRE(x == MFEM_Approx(1.0));
    }

    SECTION("Escapes local minimum")
    {
        auto f = [](real_t x, real_t m) { return std::cos(2*M_PI*x) - m*x + 2.5; };
        real_t x0 = 0.1;
        real_t m = 2.0;
        SolverSettings settings{.bounds{.lower = 0.0, .upper = 2.0}};
        real_t x = SolveNewtonBisection<+f>(x0, m, settings);
        real_t y = f(x, m);
        mfem::out << "f(x) = " << f(x, m) << "\n";
        INFO("f(x) = " << f(x, m));
        REQUIRE(x == MFEM_Approx(1.25));
    }
}

