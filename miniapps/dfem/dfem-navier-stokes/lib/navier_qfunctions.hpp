// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_DFEM_NAVIER_QFUNCTIONS_HPP
#define MFEM_DFEM_NAVIER_QFUNCTIONS_HPP

#include "mfem.hpp"
#include "../../../../fem/dfem/backends/local_qf/prelude.hpp"

namespace mfem
{
namespace dfem_navier
{

using namespace mfem::future;

// Add to fix problem with overload resolution
// (mainly sqrt)
using std::pow;
using std::sqrt;
using std::expm1;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

constexpr int U = 0;
constexpr int P = 1;
constexpr int Coords = 2;

enum class RheologyType
{
   Newtonian,
   PowerLaw,
   Bingham
};

/* NOTE: The specialized Rheology q-function is meant to provide the
viscous stress tensor tau. We could in principle just return the viscosity since all the models
assume tau = 2 mu D. However this abstraction might help if designing models that
don't follow this assumption, e.g. viscoelastic models.

Inside the Rheology q-function, the split into viscosity and stress is just a convenience
to allow the NavierStokesSolver to retrieve the effective viscosity for post-processing.
*/

// Spatial residual shared by incompressible Navier-Stokes rheology models.
// The derived model supplies the viscous stress tau(grad(u)); this adapter adds
// pressure, convection, continuity, and the physical-to-reference mapping.
template <typename Rheology, int DIM>
struct NavierStokesQFunction
{
   MFEM_HOST_DEVICE inline void operator()(
      const tensor<dscalar_t, DIM> &u,
      const tensor<dscalar_t, DIM, DIM> &dudxi,
      const dscalar_t &p,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &weight,
      tensor<dscalar_t, DIM, DIM> &momentum_gradient,
      tensor<dscalar_t, DIM> &momentum_value,
      dscalar_t &continuity_value) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const real_t dxw = det(J) * weight;
      const auto total_stress =
         rheology().stress(dudx) - p * IdentityMatrix<DIM>();

      // Tested against Gradient<U>: (tau - p I) : grad(v)
      momentum_gradient = total_stress * transpose(invJ) * dxw;

      // Tested against Value<U>: (u . grad)u = grad(u) * u
      momentum_value = (dudx * u) * dxw;

      // Tested against Value<P>: q div(u)
      continuity_value = tr(dudx) * dxw;
   }

private:
   MFEM_HOST_DEVICE inline const Rheology &rheology() const
   {
      return static_cast<const Rheology &>(*this);
   }
};

/// Newtonian viscosity in vector-Laplacian form, tau = nu grad(u).
template <int DIM>
struct NewtonianNavierStokesQFunction :
   NavierStokesQFunction<NewtonianNavierStokesQFunction<DIM>, DIM>
{
   real_t viscosity = 1.0;

   template <typename scalar_t>
   MFEM_HOST_DEVICE inline scalar_t effective_viscosity(
      const tensor<scalar_t, DIM, DIM> &) const
   {
      return viscosity;
   }

   MFEM_HOST_DEVICE inline auto stress(
      const tensor<dscalar_t, DIM, DIM> &velocity_gradient) const
   {
      return viscosity * velocity_gradient;
   }
};

/// Regularized power-law rheology,
/// tau = 2 mu D, mu = K (gamma_eps)^(n - 1), gamma_eps = sqrt( 2 D:D + epsilon^2 )
/// See: Barrett, John W., and W. B. Liu. "Finite element error analysis of a quasi-Newtonian flow obeying the Carreau or power law." Numerische Mathematik 64.1 (1993): 433-453.
template <int DIM>
struct RegularizedPowerLawNavierStokesQFunction :
   NavierStokesQFunction<RegularizedPowerLawNavierStokesQFunction<DIM>, DIM>
{
   real_t consistency = 1.0;
   real_t power_index = 0.5;
   real_t regularization = 1.0e-3;

   template <typename scalar_t>
   MFEM_HOST_DEVICE inline scalar_t effective_viscosity(
      const tensor<scalar_t, DIM, DIM> &D) const
   {
      const auto gamma_eps =
         sqrt(2.0_r * ddot(D, D) + regularization * regularization);
      return consistency * pow(gamma_eps, power_index - 1.0_r);
   }

   MFEM_HOST_DEVICE inline auto stress(
      const tensor<dscalar_t, DIM, DIM> &velocity_gradient) const
   {
      const auto D = sym(velocity_gradient);
      return 2.0_r * effective_viscosity(D) * D;
   }
};


/// Regularized Bingham rheology,
/// tau = 2 mu D, mu = mu_p + tau_y (1 - exp(-m gamma_eps)) / gamma_eps, gamma_eps = sqrt( 2 D:D + epsilon^2 )
/// See: Papanastasiou, Tasos C. "Flows of materials with yield." Journal of rheology 31.5 (1987): 385-404.
///
/// Classic Bingham fluids are not differentiable at zero strain rate, as they follow:
/// sigma = 2μ_p D(u) − pI + √2 τ_y  D(u)/|D(u)|,   if  τ ≥ τ_y   (yielded)
/// D(u) = 0,                                       if  τ < τ_y   (unyielded/rigid)
template <int DIM>
struct RegularizedBinghamNavierStokesQFunction :
   NavierStokesQFunction<RegularizedBinghamNavierStokesQFunction<DIM>, DIM>
{
   real_t yield_stress = 1.0;
   real_t mu_p = 1.0;
   real_t tau_regularization = 1e2;
   real_t regularization = 1.0e-3;

   template <typename scalar_t>
   MFEM_HOST_DEVICE inline scalar_t effective_viscosity(
      const tensor<scalar_t, DIM, DIM> &D) const
   {
      const auto gamma_eps =
         sqrt(2.0_r * ddot(D, D) + regularization * regularization);
      return mu_p + yield_stress *
             (-expm1(-tau_regularization * gamma_eps)) / gamma_eps;
   }

   MFEM_HOST_DEVICE inline auto stress(
      const tensor<dscalar_t, DIM, DIM> &velocity_gradient) const
   {
      const auto D = sym(velocity_gradient);
      return 2.0_r * effective_viscosity(D) * D;
   }
};


} // namespace dfem_navier
} // namespace mfem

#endif // MFEM_DFEM_NAVIER_QFUNCTIONS_HPP