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

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

constexpr int dim = 2;
constexpr int U = 0;
constexpr int P = 1;
constexpr int Coords = 2;

// ----------------------------------------------------------------------------
// Pointwise physics (q-function) for incompressible Navier-Stokes
// ----------------------------------------------------------------------------
//
// Ideally that's where a user would implement their own extension of the Navier-Stokes equations,
// e.g. non-Newtonian rheology, stabilization, etc...
// And then plug it in the wrapper NavierStokesOperator, where the actual
// DifferentiableOperator is constructed.
// (TODO: Maybe when we have more qfunction versions we could make it a parameter passed to the constructor of NavierStokesOperator,
// instead of hardcoding it to NavierStokesQFunction, but for now this is fine.)

// Spatial (steady) residual for incompressible Navier-Stokes equations.
template <int DIM>
struct NavierStokesQFunction
{
   real_t viscosity = 1.0;

   MFEM_HOST_DEVICE inline void operator()(const tensor<dscalar_t, DIM> &u,
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

      //--------------------------
      //   Momentum equation
      //--------------------------
      // Tested against Gradient<U>: (nu grad(u) - p I) : grad(v)
      momentum_gradient =
         (viscosity * dudx - p * IdentityMatrix<DIM>()) * transpose(invJ) * dxw;

      // Tested against Value<U>: (u . grad)u = grad(u) * u
      momentum_value = (dudx * u) * dxw;

      //--------------------------
      //   Continuity equation
      //--------------------------
      // Tested against Value<P>: q div(u)
      continuity_value = tr(dudx) * dxw;
   }
};

} // namespace dfem_navier
} // namespace mfem

#endif // MFEM_DFEM_NAVIER_QFUNCTIONS_HPP