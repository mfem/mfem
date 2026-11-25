// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CONTACT_COULOMB_LAW
#define MFEM_CONTACT_COULOMB_LAW

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfem
{
namespace internal
{

/// Owned calculation of one nodal Coulomb projection equation.
struct CoulombProjectionResult
{
   Array<real_t> equation, slip; ///< Cartesian constraint and weighted slip.
   DenseMatrix dk, dl;           ///< Derivatives with fixed unit normal.
   real_t gap = 0.0, pressure = 0.0; ///< Opening-positive gap and compression.
   int state = 0;                    ///< Open=0, stick=1, slip=2.

   CoulombProjectionResult() : equation(3), slip(3), dk(3), dl(3)
   {
      equation = 0.0;
      slip = 0.0;
      dk = 0.0;
      dl = 0.0;
   }
};

/** @brief Calculate a 3D Coulomb equation for frozen mortar geometry.
    @param normal Unit outward normal on the non-mortar surface.
    @param kinematics Weighted normal gap plus tangential slip increment.
    @param multiplier Cartesian multiplier in Tribol's residual convention.
    @param coefficient Finite, nonnegative friction coefficient.
    @param scale Positive pressure per weighted-gap/slip scale.
    @param supported Whether the node has mortar integration support.
    @return Valid equation and selected-branch derivatives. Nonzero equations
    are valid calculations, not converged contact. Unsupported nodes enforce
    zero multiplier. Pressure is -normal.multiplier; opening gap is positive.
    @throws std::invalid_argument Invalid sizes, nonfinite inputs, invalid
    coefficients, or a supported normal that is not unit length. */
inline CoulombProjectionResult
EvaluateCoulombProjection(const Array<real_t> &normal,
                          const Array<real_t> &kinematics,
                          const Array<real_t> &multiplier, real_t coefficient,
                          real_t scale, bool supported = true)
{
   if (normal.Size() != 3 || kinematics.Size() != 3 || multiplier.Size() != 3 ||
       !std::isfinite(coefficient) || coefficient < 0.0 ||
       !std::isfinite(scale) || scale <= 0.0)
   {
      throw std::invalid_argument("Invalid Coulomb projection input.");
   }
   real_t normal_squared = 0.0;
   for (int d = 0; d < 3; ++d)
   {
      if (!std::isfinite(normal[d]) || !std::isfinite(kinematics[d]) ||
          !std::isfinite(multiplier[d]))
      {
         throw std::invalid_argument("Nonfinite Coulomb projection input.");
      }
      normal_squared += normal[d] * normal[d];
   }
   if (supported && std::abs(normal_squared - 1.0) > 1.0e-10)
   {
      throw std::invalid_argument("Coulomb projection requires a unit normal.");
   }
   CoulombProjectionResult result;
   if (!supported)
   {
      result.equation = multiplier;
      for (int d = 0; d < 3; ++d)
      {
         result.dl(d, d) = 1.0;
      }
      return result;
   }
   for (int d = 0; d < 3; ++d)
   {
      result.gap += normal[d] * kinematics[d];
      result.pressure -= normal[d] * multiplier[d];
   }
   const bool active = result.pressure - scale * result.gap > 0.0;
   const real_t rn =
       result.pressure - std::max(0.0, result.pressure - scale * result.gap);
   Array<real_t> tangent(3), trial(3);
   real_t trial_squared = 0.0;
   for (int d = 0; d < 3; ++d)
   {
      tangent[d] = multiplier[d] + result.pressure * normal[d];
      result.slip[d] = kinematics[d] - result.gap * normal[d];
      trial[d] = tangent[d] + scale * result.slip[d];
      trial_squared += trial[d] * trial[d];
   }
   const real_t radius = coefficient * std::max(0.0, result.pressure);
   const real_t trial_norm = std::sqrt(trial_squared);
   const bool stick = radius > 0.0 && trial_norm <= radius;
   const real_t projection =
       stick ? 1.0 : (radius > 0.0 ? radius / trial_norm : 0.0);
   result.state = active ? (stick ? 1 : 2) : 0;
   for (int d = 0; d < 3; ++d)
   {
      result.equation[d] = -normal[d] * rn + tangent[d] - projection * trial[d];
      for (int e = 0; e < 3; ++e)
      {
         const real_t nn = normal[d] * normal[e];
         const real_t projector = (d == e ? 1.0 : 0.0) - nn;
         result.dk(d, e) = active ? -scale * nn : 0.0;
         result.dl(d, e) = (active ? 0.0 : nn) + projector;
         if (stick)
         {
            result.dk(d, e) -= scale * projector;
            result.dl(d, e) -= projector;
         }
         else if (radius > 0.0)
         {
            const real_t direction_d = trial[d] / trial_norm;
            const real_t direction_e = trial[e] / trial_norm;
            const real_t disk_derivative =
                projection * (projector - direction_d * direction_e);
            result.dk(d, e) -= scale * disk_derivative;
            result.dl(d, e) -= disk_derivative;
            result.dl(d, e) += coefficient * direction_d * normal[e];
         }
      }
   }
   return result;
}

} // namespace internal
} // namespace mfem
#endif
