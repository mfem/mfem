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
   PowerLaw
};

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
/// For divergence-free flow its divergence equals that of 2 nu sym(grad(u)),
/// while preserving the boundary traction used by the existing examples.
template <int DIM>
struct NewtonianNavierStokesQFunction :
   NavierStokesQFunction<NewtonianNavierStokesQFunction<DIM>, DIM>
{
   real_t viscosity = 1.0;

   MFEM_HOST_DEVICE inline auto stress(
      const tensor<dscalar_t, DIM, DIM> &velocity_gradient) const
   {
      return viscosity * velocity_gradient;
   }
};

/// Regularized power-law rheology,
/// tau = 2 mu D, mu = K (2 D:D + epsilon^2)^((n - 1)/2).
template <int DIM>
struct RegularizedPowerLawNavierStokesQFunction :
   NavierStokesQFunction<RegularizedPowerLawNavierStokesQFunction<DIM>, DIM>
{
   real_t consistency = 1.0;
   real_t power_index = 0.5;
   real_t regularization = 1.0e-3;

   MFEM_HOST_DEVICE inline auto stress(
      const tensor<dscalar_t, DIM, DIM> &velocity_gradient) const
   {
      const auto D = sym(velocity_gradient);
      const auto gamma_eps =
         sqrt(2.0_r * ddot(D, D) + regularization * regularization);
      const auto mu = consistency * pow(gamma_eps, power_index - 1.0_r);
      return 2.0_r * mu * D;
   }
};


} // namespace dfem_navier
} // namespace mfem

#endif // MFEM_DFEM_NAVIER_QFUNCTIONS_HPP