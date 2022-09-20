#pragma once

#include <mfem.hpp>
#include "navierstokes_operator.hpp"

namespace mfem
{

class RKIMEX_BEFE
{
public:
   RKIMEX_BEFE(NavierStokesOperator &op) :
      op(op),
      b(op.Height()),
      f1(op.Height())
   {}

   void Step(Vector &X, double &t, double &dt)
   {
      const double aI_22 = 1.0;
      const double aE_21 = 1.0;
      const double c2 = 1.0;

      // Stage 1
      op.SetTime(t);
      op.SetEvalMode(TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
      op.Mult(X, f1);

      // Stage 2
      op.SetTime(t + c2 * dt);
      op.Setup(aI_22 * dt);

      op.MassMult(X, b);

      for (int i = 0; i < b.Size(); i++)
      {
         b[i] += dt * aE_21 * f1[i];
      }

      // Initialize reasonable initial guess
      X = f1;

      op.Solve(b, X);

      t += dt;
   }

   NavierStokesOperator &op;
   Vector b, f1;
};

} // namespace mfem
