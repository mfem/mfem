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

#include "coulomblaw.hpp"
#include <iostream>
#include <limits>

using namespace mfem;
using mfem::internal::EvaluateCoulombProjection;

namespace
{

/// Fail a test with a description that is useful without a debugger.
void Require(bool condition, const char *message)
{
   if (!condition)
   {
      throw std::runtime_error(message);
   }
}

/** @brief Compare both law derivatives against centered differences.
    @param normal Fixed unit normal.
    @param kinematics Inputs strictly within one contact-law branch.
    @param multiplier Cartesian multiplier away from zero-pressure switching.
    @param coefficient Friction coefficient under test.
    @param supported Whether mortar integration supports this node. */
void CheckDerivatives(const Array<real_t> &normal,
                      const Array<real_t> &kinematics,
                      const Array<real_t> &multiplier, real_t coefficient,
                      bool supported = true)
{
   constexpr real_t scale = 2.0;
   constexpr real_t step = 1.0e-6;
   const auto result = EvaluateCoulombProjection(normal, kinematics, multiplier,
                                                 coefficient, scale, supported);
   for (int field = 0; field < 2; ++field)
   {
      for (int col = 0; col < 3; ++col)
      {
         Array<real_t> kplus(kinematics), kminus(kinematics);
         Array<real_t> lplus(multiplier), lminus(multiplier);
         (field == 0 ? kplus : lplus)[col] += step;
         (field == 0 ? kminus : lminus)[col] -= step;
         const auto plus = EvaluateCoulombProjection(
             normal, kplus, lplus, coefficient, scale, supported);
         const auto minus = EvaluateCoulombProjection(
             normal, kminus, lminus, coefficient, scale, supported);
         for (int row = 0; row < 3; ++row)
         {
            const real_t difference =
                (plus.equation[row] - minus.equation[row]) / (2.0 * step);
            const real_t derivative =
                field == 0 ? result.dk(row, col) : result.dl(row, col);
            Require(std::abs(difference - derivative) < 1.0e-8,
                    "Coulomb law derivative differs from finite differences.");
         }
      }
   }
}

/// Check that a calculated law value is a root, with the expected branch.
void CheckRoot(const Array<real_t> &gap, const Array<real_t> &multiplier,
               real_t coefficient, int state)
{
   const Array<real_t> normal{0.0, 0.0, 1.0};
   const auto result =
       EvaluateCoulombProjection(normal, gap, multiplier, coefficient, 2.0);
   Require(result.state == state, "Unexpected Coulomb contact state.");
   for (int d = 0; d < 3; ++d)
   {
      Require(std::abs(result.equation[d]) < 1.0e-12,
              "Admissible contact data must solve the Coulomb equation.");
   }
}

} // namespace

int main()
{
   try
   {
      const Array<real_t> normal{0.0, 0.0, 1.0};
      CheckRoot(Array<real_t>{0.1, 0.0, 0.2}, Array<real_t>{0.0, 0.0, 0.0}, 0.5,
                0);
      CheckRoot(Array<real_t>{0.0, 0.0, 0.0}, Array<real_t>{0.1, 0.0, -1.0},
                0.5, 1);
      CheckRoot(Array<real_t>{0.2, 0.15, 0.0}, Array<real_t>{0.16, 0.12, -1.0},
                0.2, 2);
      CheckRoot(Array<real_t>{0.2, -0.15, 0.0}, Array<real_t>{0.0, 0.0, -1.0},
                0.0, 2);
      CheckDerivatives(normal, Array<real_t>{0.01, 0.02, 0.3},
                       Array<real_t>{0.03, 0.04, 0.1}, 0.5);
      CheckDerivatives(normal, Array<real_t>{0.01, 0.02, -0.03},
                       Array<real_t>{0.05, -0.04, -1.0}, 0.5);
      CheckDerivatives(normal, Array<real_t>{0.2, 0.15, -0.03},
                       Array<real_t>{0.4, 0.3, -1.0}, 0.2);
      CheckDerivatives(normal, Array<real_t>{0.2, 0.15, -0.03},
                       Array<real_t>{0.0, 0.0, -1.0}, 0.0);
      CheckDerivatives(Array<real_t>{0.6, 0.0, 0.8},
                       Array<real_t>{0.2, 0.15, -0.03},
                       Array<real_t>{-0.3, 0.3, -1.0}, 0.2);
      CheckDerivatives(Array<real_t>{0.0, 0.0, 0.0},
                       Array<real_t>{0.1, 0.2, 0.3},
                       Array<real_t>{0.4, 0.5, 0.6}, 0.5, false);
      bool rejected = false;
      try
      {
         EvaluateCoulombProjection(normal, Array<real_t>{0.0, 0.0, 0.0},
                                   Array<real_t>{0.0, 0.0, -1.0}, -0.1, 2.0);
      }
      catch (const std::invalid_argument &)
      {
         rejected = true;
      }
      Require(rejected, "A negative friction coefficient must be rejected.");
      rejected = false;
      try
      {
         EvaluateCoulombProjection(
             normal,
             Array<real_t>{0.0, std::numeric_limits<real_t>::infinity(), 0.0},
             Array<real_t>{0.0, 0.0, -1.0}, 0.1, 2.0);
      }
      catch (const std::invalid_argument &)
      {
         rejected = true;
      }
      Require(rejected, "Nonfinite Coulomb inputs must be rejected.");
   }
   catch (const std::exception &error)
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
   std::cout << "Coulomb roots and selected-branch derivatives passed.\n";
   return 0;
}
