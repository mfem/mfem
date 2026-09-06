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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <cmath>

using namespace mfem;

#ifdef MFEM_USE_SUNDIALS

namespace
{

/** @brief Robertson's chemical kinetics problem, the canonical index-1 DAE
    and SUNDIALS' own idaRoberts_dns example:

        r0 = y0' + 0.04 y0 - 1e4 y1 y2
        r1 = y1' - 0.04 y0 + 1e4 y1 y2 + 3e7 y1^2
        r2 = y0 + y1 + y2 - 1                          <- algebraic

    Declared HOMOGENEOUS, so G = 0 and the residual R(t,y,y') is exactly
    F(y,y',t) = ImplicitMult(). The third row carries no time derivative,
    which makes dR/dy' = diag(1,1,0) singular: this is a DAE and not an ODE
    with a mass matrix.

    Both linear system routes are implemented, on the same operator, so that
    the identity J = cj A(1/cj) of IDASolver::UseMFEMLinearSolverFromODEForm()
    can be checked entrywise rather than only through a trajectory. */
class RobertsonDAE : public TimeDependentOperator
{
private:
   /// The matrix formed by the last setup call, kept for the identity check.
   DenseMatrix Jac;
   /// Its factorization, used by both solve methods.
   DenseMatrixInverse Jinv;

public:
   RobertsonDAE()
      : TimeDependentOperator(3, (real_t) 0.0, HOMOGENEOUS), Jac(3) { }

   /// The residual R(t, @a y, @a yp), since G = 0.
   void ImplicitMult(const Vector &y, const Vector &yp,
                     Vector &r) const override
   {
      r[0] = yp[0] + 0.04*y[0] - 1.0e4*y[1]*y[2];
      r[1] = yp[1] - 0.04*y[0] + 1.0e4*y[1]*y[2] + 3.0e7*y[1]*y[1];
      r[2] = y[0] + y[1] + y[2] - 1.0;
   }

   /// dR/dy at @a y. The dR/dy' half is diag(1,1,0) and is added by hand.
   static void ResidualGradient(const Vector &y, DenseMatrix &K)
   {
      K.SetSize(3);
      K(0,0) =  0.04;
      K(0,1) = -1.0e4*y[2];
      K(0,2) = -1.0e4*y[1];
      K(1,0) = -0.04;
      K(1,1) =  1.0e4*y[2] + 6.0e7*y[1];
      K(1,2) =  1.0e4*y[1];
      K(2,0) =  1.0;
      K(2,1) =  1.0;
      K(2,2) =  1.0;
   }

   /// J = dR/dy + cj dR/dy', the matrix IDA asks for directly.
   int SUNImplicitSetupDAE(const Vector &y, const Vector &yp,
                           const Vector &res, real_t cj) override
   {
      ResidualGradient(y, Jac);
      Jac(0,0) += cj;
      Jac(1,1) += cj;
      Jinv.Factor(Jac);
      return 0;
   }

   int SUNImplicitSolveDAE(const Vector &b, Vector &x, real_t tol) override
   {
      Jinv.Mult(b, x);
      return 0;
   }

   /** @brief A(gamma) = dF/dk + gamma (dF/du - dG/du), the matrix
       SUNImplicitSetup() documents. Here F = R and G = 0, so
       dF/dk = diag(1,1,0) and dF/du = dR/dy. */
   int SUNImplicitSetup(const Vector &y, const Vector &v, int jok, int *jcur,
                        real_t gamma) override
   {
      ResidualGradient(y, Jac);
      Jac *= gamma;
      Jac(0,0) += 1.0;
      Jac(1,1) += 1.0;
      *jcur = 1;
      Jinv.Factor(Jac);
      return 0;
   }

   int SUNImplicitSolve(const Vector &r, Vector &dk, real_t tol) override
   {
      Jinv.Mult(r, dk);
      return 0;
   }

   /// The matrix the last setup call formed.
   const DenseMatrix &GetMatrix() const { return Jac; }
};

/** @brief Robertson again, split the other way: only the mass term in F and
    every reaction term in G. That is form 2 of TimeDependentOperator's own
    list -- F(u,k,t) = M k with M = diag(1,1,0), and G(u,t) = g(u,t).
    Declared IMPLICIT, so IDASolver has to assemble R = F - G itself.

    G is far from zero at every state the residual case probes, so the sign
    of that subtraction is under test: this operator must produce the *same*
    residual as RobertsonDAE, which folds G into F and is HOMOGENEOUS. */
class RobertsonIMPLICIT : public TimeDependentOperator
{
public:
   RobertsonIMPLICIT()
      : TimeDependentOperator(3, (real_t) 0.0, IMPLICIT) { }

   /// F(y, y', t) = M y' with M = diag(1,1,0): the constraint row carries
   /// no time derivative.
   void ImplicitMult(const Vector &y, const Vector &yp,
                     Vector &v) const override
   {
      v[0] = yp[0];
      v[1] = yp[1];
      v[2] = 0.0;
   }

   /// G(y, t): the reactions and the constraint, signed so that F - G is
   /// Robertson's residual.
   void ExplicitMult(const Vector &y, Vector &v) const override
   {
      const real_t rate1 = 0.04*y[0];
      const real_t rate2 = 1.0e4*y[1]*y[2];
      const real_t rate3 = 3.0e7*y[1]*y[1];
      v[0] = -rate1 + rate2;
      v[1] = rate1 - rate2 - rate3;
      v[2] = 1.0 - y[0] - y[1] - y[2];
   }
};

/** @brief A linear ODE written the way an operator for CVODESolver is
    written: type EXPLICIT, so F(u,k,t) = k and G(u,t) = A u, with only
    Mult() and the ODE-form SUNImplicitSetup()/SUNImplicitSolve() pair.

        y' = A y,   A = [ -1  2 ; -2  -1 ],

    whose exact solution from y(0) = (1,0) is e^{-t} (cos 2t, -sin 2t).
    Nothing here knows about IDA, which is the point of the case that uses
    it: the EXPLICIT row of IDASolver's residual dispatch means such an
    operator runs under IDA unchanged, as a DAE with an identity mass
    matrix. */
class SpiralODE : public TimeDependentOperator
{
private:
   DenseMatrix A;      ///< dg/dy, constant.
   DenseMatrix Amat;   ///< A(gamma) = I - gamma dg/dy.
   DenseMatrixInverse Ainv;

public:
   SpiralODE()
      : TimeDependentOperator(2, (real_t) 0.0, EXPLICIT), A(2), Amat(2)
   {
      A(0,0) = -1.0;
      A(0,1) =  2.0;
      A(1,0) = -2.0;
      A(1,1) = -1.0;
   }

   void Mult(const Vector &y, Vector &dydt) const override
   {
      A.Mult(y, dydt);
   }

   int SUNImplicitSetup(const Vector &y, const Vector &v, int jok, int *jcur,
                        real_t gamma) override
   {
      Amat = A;
      Amat *= -gamma;
      Amat(0,0) += 1.0;
      Amat(1,1) += 1.0;
      *jcur = 1;
      Ainv.Factor(Amat);
      return 0;
   }

   int SUNImplicitSolve(const Vector &r, Vector &dk, real_t tol) override
   {
      Ainv.Mult(r, dk);
      return 0;
   }

   static void Exact(real_t t, Vector &y)
   {
      const real_t e = std::exp(-t);
      y[0] =  e*std::cos(2.0*t);
      y[1] = -e*std::sin(2.0*t);
   }
};

/// Number of states the residual dispatch is probed at.
const int robertson_probes = 3;

/** @brief Probe states for the residual dispatch. No entry of y, of y' or
    of the residual they produce is zero, so a dropped term or a flipped
    sign cannot pass. The third entry of y' is non-zero too, and the
    residual must ignore it: the constraint row has no time derivative. */
const real_t robertson_probe_y[robertson_probes][3] =
{
   {0.7, 3.0e-4, 0.5},
   {0.25, 1.5e-3, 0.9},
   {0.5, -1.0e-3, 0.8}
};
const real_t robertson_probe_yp[robertson_probes][3] =
{
   {0.35, -0.21, 0.13},
   {-0.6, 0.45, -0.2},
   {0.9, -1.1, 0.7}
};

/** @brief Robertson's residual written out by hand in the chemistry form,
    independently of either fixture: this is what the Type dispatch has to
    reproduce, whichever way the equations are split between F and G. */
void RobertsonResidualByHand(const Vector &y, const Vector &yp, Vector &r)
{
   const real_t rate1 = 0.04*y[0];
   const real_t rate2 = 1.0e4*y[1]*y[2];
   const real_t rate3 = 3.0e7*y[1]*y[1];
   r.SetSize(3);
   r[0] = yp[0] + rate1 - rate2;
   r[1] = yp[1] - rate1 + rate2 + rate3;
   r[2] = y[0] + y[1] + y[2] - 1.0;
}

/// Copy row @a s of a three-column probe table into @a v.
void GetProbe(const real_t table[][3], int s, Vector &v)
{
   v.SetSize(3);
   for (int j = 0; j < 3; j++) { v[j] = table[s][j]; }
}

/// Number of decade outputs taken by RunRobertson().
const int robertson_outputs = 12;

/** @brief Integrate Robertson from the standard consistent initial
    condition y = (1,0,0), y' = (-0.04, 0.04, 0) to t = 4e10 in the decade
    outputs idaRoberts_dns uses, through the DAE linear system route
    (@a ode_form false) or the ODE-form one (@a ode_form true).

    Column i of @a traj holds the state at output i. */
void RunRobertson(bool ode_form, DenseMatrix &traj)
{
   RobertsonDAE oper;
   IDASolver ida;
   ida.Init(oper);

   if (ode_form) { ida.UseMFEMLinearSolverFromODEForm(); }
   else { ida.UseMFEMLinearSolver(); }

   Vector atol({1.0e-8, 1.0e-14, 1.0e-6});
   ida.SetSVtolerances(1.0e-8, atol);

   // y2 is the algebraic unknown: it carries no derivative, so it is kept
   // out of the local error test as well.
   Array<int> is_differential({1, 1, 0});
   ida.SetDifferentialComponents(is_differential);
   ida.SetSuppressAlgebraic(true);

   Vector y({1.0, 0.0, 0.0});
   Vector yp({-0.04, 0.04, 0.0});
   ida.SetInitialDerivative(yp);

   traj.SetSize(3, robertson_outputs);
   real_t t = 0.0;
   real_t tout = 0.4;
   for (int i = 0; i < robertson_outputs; i++)
   {
      real_t dt = tout - t;
      ida.Step(y, t, dt);
      for (int j = 0; j < 3; j++) { traj(j, i) = y[j]; }
      tout *= 10.0;
   }
}

} // anonymous namespace


TEST_CASE("IDA integrates an index-1 DAE", "[SUNDIALS][IDA]")
{
   DenseMatrix traj;
   RunRobertson(false, traj);

   SECTION("The algebraic constraint is enforced at every output")
   {
      // The constraint row is exactly linear, so an *enforced* constraint
      // sits at round-off relative to |y| = 1 at every output -- four
      // orders below the 1e-8 relative integration tolerance, which is
      // what a merely integrated constraint would drift by. The bound is
      // 1e-12 rather than round-off itself because IDA scales the Newton
      // correction by 2/(1 + cj/cj_old) when the factorization is stale,
      // which leaves this row at (1 - scale) times the predictor's
      // residual rather than at exactly zero.
      const real_t constraint_tol = 1.0e-12;
      real_t tout = 0.4;
      for (int i = 0; i < traj.Width(); i++)
      {
         const real_t c = traj(0,i) + traj(1,i) + traj(2,i) - 1.0;
         CAPTURE(i, tout, c);
         REQUIRE(std::abs(c) <= constraint_tol);
         tout *= 10.0;
      }
   }

   SECTION("The answer at t = 4e10 is SUNDIALS' published one")
   {
      const int last = robertson_outputs - 1;
      const real_t y0 = traj(0, last);
      const real_t y1 = traj(1, last);
      const real_t y2 = traj(2, last);
      CAPTURE(y0, y1, y2);

      // idaRoberts_dns, whose reference comes from a dense direct solver
      // rather than from this one. y0 and y1 have decayed below their own
      // absolute tolerances (1e-8 and 1e-14), so neither is controlled to
      // better than a few percent by then; y2 is O(1) and is.
      const real_t rel_small = 5.0e-2;
      const real_t rel_one   = 1.0e-8;
      REQUIRE(y0 == MFEM_Approx(5.2083495e-08, 0.0, rel_small));
      REQUIRE(y1 == MFEM_Approx(2.0833937e-13, 0.0, rel_small));
      REQUIRE(y2 == MFEM_Approx(9.9999995e-01, 0.0, rel_one));
   }
}


TEST_CASE("IDACalcIC recovers a consistent initial condition",
          "[SUNDIALS][IDA]")
{
   RobertsonDAE oper;
   IDASolver ida;
   ida.Init(oper);

   Vector atol({1.0e-8, 1.0e-14, 1.0e-6});
   ida.SetSVtolerances(1.0e-8, atol);

   Array<int> is_differential({1, 1, 0});
   ida.SetDifferentialComponents(is_differential);
   ida.SetSuppressAlgebraic(true);

   // Deliberately inconsistent: y2 = 0.5 breaks the constraint row outright,
   // and y' = 0 solves neither differential row.
   Vector y({1.0, 0.0, 0.5});
   Vector yp(3);
   yp = 0.0;
   ida.SetInitialDerivative(yp);

   Vector res(3);
   oper.ImplicitMult(y, yp, res);   // G = 0, so this is the residual itself
   const real_t res_before = res.Norml2();
   CAPTURE(res_before);
   REQUIRE(res_before > 5.0e-1);
   REQUIRE(res_before < 5.1e-1);

   ida.ComputeConsistentIC(y, 0.4);

   SECTION("The algebraic component of y is recovered")
   {
      // IDA_YA_YDP_INIT holds the differential components of y fixed and
      // solves the constraint row for the algebraic one, so the corrected
      // state is exactly the standard initial condition (1, 0, 0).
      CAPTURE(y[0], y[1], y[2]);
      REQUIRE(y[0] == MFEM_Approx(1.0, 1.0e-14, 1.0e-14));
      REQUIRE(y[1] == MFEM_Approx(0.0, 1.0e-14, 1.0e-14));
      REQUIRE(std::abs(y[2]) <= 1.0e-12);
   }

   SECTION("The whole derivative is recovered")
   {
      const Vector &ypc = ida.GetDerivative();
      REQUIRE(ypc.Size() == 3);
      CAPTURE(ypc[0], ypc[1], ypc[2]);
      // Both differential rows are linear in y' at y = (1,0,0), so the
      // consistent derivative is exact: y0' = -0.04, y1' = +0.04. The
      // algebraic component's derivative is not an unknown of
      // IDA_YA_YDP_INIT and keeps the zero it was seeded with.
      REQUIRE(ypc[0] == MFEM_Approx(-0.04, 1.0e-12, 1.0e-12));
      REQUIRE(ypc[1] == MFEM_Approx( 0.04, 1.0e-12, 1.0e-12));
      REQUIRE(std::abs(ypc[2]) <= 1.0e-12);
   }

   SECTION("The residual at the corrected pair is at round-off")
   {
      oper.ImplicitMult(y, ida.GetDerivative(), res);
      const real_t res_after = res.Norml2();
      CAPTURE(res_before, res_after);
      REQUIRE(res_after <= 1.0e-12);
   }
}


TEST_CASE("The ODE-form linear system route agrees with the DAE one",
          "[SUNDIALS][IDA]")
{
   SECTION("J and cj A(1/cj) agree entrywise")
   {
      // This is the sharp test of UseMFEMLinearSolverFromODEForm(), and the
      // reason it is written this way rather than as a comparison of the
      // two trajectories is that the two runs take different adaptive step
      // sequences: comparing where they land measures the step controller,
      // not the identity. Compare the matrices instead.
      //
      //   J        = dR/dy + cj dR/dy'              SUNImplicitSetupDAE()
      //   A(gamma) = dF/dk + gamma (dF/du - dG/du)   SUNImplicitSetup()
      //
      // With F = R and G = 0 these give J = cj A(1/cj) identically. The
      // only arithmetic between the two forms is the round trip through
      // 1/cj, so they must agree to a couple of units in the last place.
      RobertsonDAE oper;

      // States spanning the trajectory, from the initial condition through
      // the transient to the t = 4e10 end state.
      const real_t states[5][3] =
      {
         {1.0,        0.0,        0.0      },
         {0.98517,    3.3864e-05, 0.0147966},
         {0.5,        1.0e-05,    0.49999  },
         {0.3,        2.0e-03,    0.698    },
         {5.2083e-08, 2.0834e-13, 1.0      }
      };
      // cj is positive with units of one over time; for a BDF1 step it is
      // 1/dt, and IDA's steps here span far more than these twelve decades.
      const real_t cjs[7] =
      {
         1.0e-6, 1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4, 1.0e6
      };
      const real_t tol = 1.0e-13;

      Vector yp(3), res(3), v(3);
      yp = 0.0;
      res = 0.0;
      v = 0.0;

      for (int s = 0; s < 5; s++)
      {
         Vector y(3);
         for (int j = 0; j < 3; j++) { y[j] = states[s][j]; }

         for (int c = 0; c < 7; c++)
         {
            const real_t cj = cjs[c];

            oper.SUNImplicitSetupDAE(y, yp, res, cj);
            DenseMatrix J(oper.GetMatrix());

            int jcur = 0;
            oper.SUNImplicitSetup(y, v, 0, &jcur, 1.0/cj);
            DenseMatrix A(oper.GetMatrix());
            A *= cj;

            const real_t scale = J.MaxMaxNorm();
            A -= J;
            const real_t diff = A.MaxMaxNorm()/scale;
            CAPTURE(s, cj, scale, diff);
            REQUIRE(diff <= tol);
         }
      }
   }

   SECTION("The two routes integrate to the same answer")
   {
      // The weak test, kept as a secondary check and given a tolerance that
      // says so: the two runs take different step sequences, so what they
      // can be held to is the integration tolerance, not round-off. The
      // absolute margin is the largest absolute tolerance handed to IDA.
      DenseMatrix dae, ode;
      RunRobertson(false, dae);
      RunRobertson(true, ode);

      REQUIRE(ode.Width() == dae.Width());

      const real_t abs_tol = 1.0e-6;
      const real_t rel_tol = 1.0e-4;
      real_t tout = 0.4;
      for (int i = 0; i < dae.Width(); i++)
      {
         for (int j = 0; j < 3; j++)
         {
            CAPTURE(i, j, tout, dae(j,i), ode(j,i));
            REQUIRE(ode(j,i) == MFEM_Approx(dae(j,i), abs_tol, rel_tol));
         }
         tout *= 10.0;
      }
   }
}


TEST_CASE("IDA integrates an ODE operator written for CVODE",
          "[SUNDIALS][IDA]")
{
   // SpiralODE implements Mult() and the ODE-form linear system pair and
   // nothing else -- exactly what an operator driven by CVODESolver has.
   // Driving it with IDASolver exercises the EXPLICIT row of the residual
   // dispatch, r = y' - f(y,t), and the ODE-form linear system route.
   SpiralODE oper;
   IDASolver ida;
   ida.Init(oper);
   ida.UseMFEMLinearSolverFromODEForm();
   ida.SetSStolerances(1.0e-10, 1.0e-12);

   Vector y({1.0, 0.0});
   Vector yp(2);
   oper.Mult(y, yp);   // y'(0) = A y(0), consistent by definition
   ida.SetInitialDerivative(yp);

   Vector exact(2);
   const real_t tol = 1.0e-7;
   real_t t = 0.0;
   for (int i = 0; i < 8; i++)
   {
      real_t dt = 0.25;
      ida.Step(y, t, dt);

      SpiralODE::Exact(t, exact);
      CAPTURE(i, t, y[0], y[1], exact[0], exact[1]);
      REQUIRE(y[0] == MFEM_Approx(exact[0], tol, tol));
      REQUIRE(y[1] == MFEM_Approx(exact[1], tol, tol));
   }
   REQUIRE(t == MFEM_Approx(2.0));
}


TEST_CASE("IDA builds the residual from the operator's expression form",
          "[SUNDIALS][IDA]")
{
   // GetResidual() is the only way to see the Type dispatch documented on
   // IDASolver directly, rather than through whether an integration
   // converges -- and a converging integration is a weak witness, since the
   // residual is whatever it drove to zero either way.
   const real_t abs_tol = 1.0e-12;
   const real_t rel_tol = 1.0e-12;

   SECTION("EXPLICIT: R = y' - f(y,t)")
   {
      SpiralODE oper;
      IDASolver ida;
      ida.Init(oper);

      const real_t ys[2][2] = {{0.6, -0.35}, {-0.25, 0.8}};
      const real_t yps[2][2] = {{0.2, 0.9}, {-0.7, 0.45}};

      for (int s = 0; s < 2; s++)
      {
         Vector y(2), yp(2), r(2), f(2), expect(2);
         for (int j = 0; j < 2; j++)
         {
            y[j] = ys[s][j];
            yp[j] = yps[s][j];
         }

         // f is the term a dropped dispatch would lose, so check the probe
         // is not one where it happens to vanish.
         oper.Mult(y, f);
         CAPTURE(s, f[0], f[1]);
         REQUIRE(f.Norml2() > 1.0);

         ida.GetResidual(y, yp, r);

         // By hand, with A = [ -1 2 ; -2 -1 ].
         expect[0] = yp[0] - (-1.0*y[0] + 2.0*y[1]);
         expect[1] = yp[1] - (-2.0*y[0] - 1.0*y[1]);

         CAPTURE(r[0], r[1], expect[0], expect[1]);
         REQUIRE(r[0] == MFEM_Approx(expect[0], abs_tol, rel_tol));
         REQUIRE(r[1] == MFEM_Approx(expect[1], abs_tol, rel_tol));
      }
   }

   SECTION("HOMOGENEOUS: R = F(y,y',t)")
   {
      RobertsonDAE oper;
      IDASolver ida;
      ida.Init(oper);

      for (int s = 0; s < robertson_probes; s++)
      {
         Vector y, yp, expect, r(3);
         GetProbe(robertson_probe_y, s, y);
         GetProbe(robertson_probe_yp, s, yp);
         RobertsonResidualByHand(y, yp, expect);

         ida.GetResidual(y, yp, r);

         for (int j = 0; j < 3; j++)
         {
            CAPTURE(s, j, r[j], expect[j]);
            REQUIRE(r[j] == MFEM_Approx(expect[j], abs_tol, rel_tol));
         }
         // Not vacuous: no component of this residual is near zero, and
         // the third is the constraint row, whose non-zero y' entry the
         // dispatch has to ignore.
         REQUIRE(r.Norml2() > 1.0);
      }
   }

   SECTION("IMPLICIT: R = F(y,y',t) - G(y,t)")
   {
      // Two fixtures, the same equations split two different ways.
      // RobertsonDAE folds G into F and is HOMOGENEOUS; RobertsonIMPLICIT
      // keeps a non-zero G and leaves the subtraction to the dispatch. The
      // sign of that subtraction is the thing a dispatch bug gets wrong,
      // and it cannot survive both of the assertions below.
      RobertsonDAE hom_oper;
      RobertsonIMPLICIT imp_oper;
      IDASolver hom_ida, imp_ida;
      hom_ida.Init(hom_oper);
      imp_ida.Init(imp_oper);

      for (int s = 0; s < robertson_probes; s++)
      {
         Vector y, yp, expect, g(3), rh(3), ri(3);
         GetProbe(robertson_probe_y, s, y);
         GetProbe(robertson_probe_yp, s, yp);
         RobertsonResidualByHand(y, yp, expect);

         // G is what the subtraction acts on. If it vanished here, R = F+G
         // would pass and the sign would not be under test at all.
         imp_oper.ExplicitMult(y, g);
         CAPTURE(s, g[0], g[1], g[2]);
         REQUIRE(g.Norml2() > 1.0);

         hom_ida.GetResidual(y, yp, rh);
         imp_ida.GetResidual(y, yp, ri);

         for (int j = 0; j < 3; j++)
         {
            CAPTURE(j, rh[j], ri[j], expect[j]);
            REQUIRE(ri[j] == MFEM_Approx(expect[j], abs_tol, rel_tol));
            REQUIRE(ri[j] == MFEM_Approx(rh[j], abs_tol, rel_tol));
         }
      }
   }
}

#endif // MFEM_USE_SUNDIALS
