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

using mfem::real_t;
using namespace mfem::future;

MFEM_HOST_DEVICE inline real_t FlowResistance(real_t eqps, real_t sigma_y, real_t n, real_t ep_0)
{
    return sigma_y*(1.0 + std::pow(eqps/ep_0, n));
}

using J2PlasticityParameters = mfem::future::tuple<real_t, real_t, real_t, real_t, real_t, real_t>;

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
  operator()(tensor<real_t, dim, dim> dudxi,
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
        real_t ub = (q - FlowResistance(accumulated_plastic_strain, sigma_y, n, ep_0))/(3*G);
        SolverSettings settings{.residual_abs_tol = 1e-10*sigma_y, .residual_rel_tol = 1e-10, 
                                .bounds{.lower = lb, .upper = ub}};
        real_t delta_eqps = SolveNewtonBisection<J2PlasticityResidual>(
            0.5*(lb + ub), make_tuple(accumulated_plastic_strain, q, G, sigma_y, n, ep_0), settings);
        accumulated_plastic_strain += delta_eqps;
        plastic_strain += delta_eqps * Np;
        s -= 2.0 * G * delta_eqps * Np;
    }
    internal_state = pack_internal_state(plastic_strain, accumulated_plastic_strain);
    auto stress = s + p * I;
    const real_t dV = det(J)*w;
    return make_tuple(stress*transpose(invJ)*dV, internal_state);
  }
};

__attribute__((used))
void *  __enzyme_register_derivative_newton_bisection_on_j2[2] = {
  (void*) SolveNewtonBisection_impl<J2PlasticityResidual, J2PlasticityParameters>,
  (void*) SolveNewtonBisection_impl_fwddiff<J2PlasticityResidual, J2PlasticityParameters>
};

TEST_CASE("Univariate function solver on qfunction", "[univar]")
{
    tensor<real_t, 3, 3> H{{{0.947667  , 0.9785799 , 0.33229148},
                            {0.46866846, 0.5698887 , 0.16550303},
                            {0.3101946 , 0.68948054, 0.74676657}}};
    J2Plasticity::PackedInternalState Q{};
    J2Plasticity material{.E = 70.0e3, .nu = 0.34, .sigma_y = 240.0, .n = 0.15, .ep_0 = 1e-3};
    auto [stress, Q_new] = material(H, Q, IdentityMatrix<3>(), 1.0);
    INFO("stress = " << stress << "\ninternal vars =" << Q_new << "\n");
    real_t mises = std::sqrt(1.5)*norm(dev(stress));
    real_t eqps = Q_new[9];
    real_t Y = FlowResistance(eqps, material.sigma_y, material.n, material.ep_0);
    REQUIRE(mises == MFEM_Approx(Y, 0.0, 1e-8));
}


real_t nthroot_res(real_t x, tuple<real_t, real_t> p)
{
    auto [index, radicand] = p;
    return std::pow(x, index) - radicand;
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



