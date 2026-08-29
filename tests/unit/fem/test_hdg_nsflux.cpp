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

#include "mfem.hpp"
#include "unit_tests.hpp"

// The class under test lives in the miniapp, not the library. The unit test
// makefile adds -I<mfem root> and the CMake build does not, so the include is
// written relative to this file, which both builds resolve identically. It
// also has to live in tests/unit/fem/ rather than tests/unit/miniapps/: the
// makefile's MINI_SOURCE_FILES are explicitly filtered *out* of the sources
// that make up libtests.o, so a test dropped in tests/unit/miniapps/ is never
// linked into unit_tests at all.
#include "../../../miniapps/hdg/nsflux.hpp"

#include <cmath>

using namespace mfem;
using namespace mfem::hdg;

namespace hdg_nsflux
{

// Why this file exists at all. ArtificialCompressibilityFlux supplies four
// analytic derivatives -- ComputeFluxJacobian, ComputeFluxJacobianDotN,
// MaxCharSpeedDotNGrad and, through HDGLaxFriedrichsFlux, AverageGrad. Under
// hybridization none of them is ever assembled into a globally checked
// operator: a wrong entry does not produce a wrong answer, only a slower
// Newton, and no convergence table or regression would report it. Differencing
// each one against the function it claims to differentiate is the only thing
// that would catch it.

// Central differences are second order. At a relative step of 1e-6 the
// truncation error is O(h^2) ~ 1e-12 and the cancellation error O(eps/h) ~
// 1e-10, so the difference carries about ten significant digits. A tolerance
// of 1e-6 sits four orders above that noise floor and is still far tighter
// than every failure mode that matters here -- a wrong sign, a transposed
// index pair, a dropped term -- each of which is an O(1) relative error.
const real_t fd_step = 1e-6;
const real_t fd_tol  = 1e-6;

// Two analytic expressions of the same quantity (the hand-rolled
// ComputeFluxJacobianDotN against the contraction of ComputeFluxJacobian, or
// a stabilization that must cancel identically) may only differ by round-off.
const real_t exact_tol = 1e-13;

/// A relative finite-difference step, floored so that a zero component still
/// gets perturbed.
inline real_t Step(real_t x)
{
   return fd_step * std::max((real_t) 1.0, std::abs(x));
}

/// A single element with an interior face, giving both an
/// ElementTransformation and a FaceElementTransformations with an integration
/// point set -- what the flux functions need in order to be called. Neither
/// class actually reads the transformation, but the interface demands one.
struct Geom
{
   Mesh mesh;
   ElementTransformation *Tr;
   FaceElementTransformations *FTr;

   Geom(int dim)
      : mesh((dim == 2)
             ? Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0)
             : Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                                     2.0, 1.0, 1.0))
   {
      Tr = mesh.GetElementTransformation(0);
      IntegrationPoint ip;
      ip.Set3(0.3, 0.4, 0.2);
      Tr->SetIntPoint(&ip);

      int f = -1;
      for (int i = 0; i < mesh.GetNumFaces(); i++)
      {
         if (mesh.FaceIsInterior(i)) { f = i; break; }
      }
      FTr = mesh.GetFaceElementTransformations(f);
      IntegrationPoint fip;
      fip.Set2(0.37, 0.0);
      FTr->SetAllIntPoints(&fip);
   }
};

/// The face normal. Every component is a dyadic rational, so that the state
/// built to satisfy `v.n == 0` below satisfies it to the last bit rather than
/// to round-off -- the whole point of that state is to sit exactly on the kink
/// of `|v.n|`, and a normal of 0.6 would put it 1e-17 to one side.
void MakeNormal(int dim, Vector &nor)
{
   nor.SetSize(dim);
   nor(0) = 0.5;
   nor(1) = 0.25;
   if (dim == 3) { nor(2) = -0.125; }
}

enum StateId
{
   ZERO_VELOCITY = 0,  ///< v = 0: on the kink of |v.n|, and every quadratic
   ///  term in the flux Jacobian vanishes.
   ORTHOGONAL,         ///< v.n == 0 with v != 0: on the kink, but with the
   ///  quadratic terms alive.
   GENERIC_POS,        ///< v.n > 0: sgn(v.n) = +1.
   GENERIC_NEG,        ///< v.n < 0: sgn(v.n) = -1.
   NUM_STATES
};

const char *StateName(int id)
{
   switch (id)
   {
      case ZERO_VELOCITY: return "zero velocity";
      case ORTHOGONAL:    return "v.n == 0 exactly";
      case GENERIC_POS:   return "generic, v.n > 0";
      default:            return "generic, v.n < 0";
   }
}

/// The trace state `uhat = (p, v)`, `num_equations = dim + 1`.
void MakeState(int dim, int id, Vector &u)
{
   u.SetSize(dim + 1);
   u = 0.0;
   switch (id)
   {
      case ZERO_VELOCITY:
         u(0) = 0.7;
         break;
      case ORTHOGONAL:
         // Dyadic components orthogonal to MakeNormal()'s normal, so that the
         // running sum in MaxCharSpeedDotN() is exactly zero.
         u(0) = -1.3;
         if (dim == 2) { u(1) = -0.5; u(2) = 1.0; }
         else          { u(1) = 0.25; u(2) = 0.5; u(3) = 2.0; }
         break;
      case GENERIC_POS:
         u(0) = 0.7;
         for (int d = 0; d < dim; d++) { u(1 + d) = 0.4 * (d + 1) - 0.15; }
         break;
      default:
         u(0) = -0.4;
         for (int d = 0; d < dim; d++) { u(1 + d) = 0.15 - 0.4 * (d + 1); }
         break;
   }
}

/// The element state, offset from the trace state so that `u - uhat` is
/// nonzero in every component -- the factor the stabilization Jacobian's
/// extra term multiplies.
void MakeElementState(const Vector &uhat, Vector &u)
{
   static const real_t off[4] = { 0.23, -0.41, 0.17, 0.09 };
   u.SetSize(uhat.Size());
   for (int i = 0; i < uhat.Size(); i++) { u(i) = uhat(i) + off[i]; }
}

/// `v.n` as the class computes it, so that the "is this state on the kink"
/// test in the checks below asks exactly the question the code answers.
real_t DotN(const Vector &state, const Vector &nor)
{
   real_t vn = 0.0;
   for (int d = 0; d < nor.Size(); d++) { vn += state(1 + d) * nor(d); }
   return vn;
}

real_t NormalSq(const Vector &nor)
{
   real_t nn = 0.0;
   for (int d = 0; d < nor.Size(); d++) { nn += nor(d) * nor(d); }
   return nn;
}

/// `lambda_max(u, n)` as the doc comment states it. Dropping `v (x) v` drops
/// the `(v.n) I + v n^T` block of `A_n`, so the Stokes eigenvalues collapse to
/// `+- sqrt(beta)|n|` and lambda stops depending on the state -- which is also
/// why the kink of `|v.n|` is absent there and KinkBias() must not be applied.
real_t Lambda(bool stokes, real_t vn, real_t nn, real_t beta)
{
   if (stokes) { return std::sqrt(beta * nn); }
   return std::abs(vn) + std::sqrt(vn * vn + beta * nn);
}

/// Central difference of NumericalFlux::Average() with respect to the first
/// (@a side == 1, the trace) or second (@a side == 2, the element) state.
void FDAverage(const NumericalFlux &nf, int side, const Vector &s1,
               const Vector &s2, const Vector &nor,
               FaceElementTransformations &Tr, DenseMatrix &fd)
{
   const int neq = s1.Size();
   fd.SetSize(neq);
   for (int j = 0; j < neq; j++)
   {
      Vector p1(s1), p2(s2), m1(s1), m2(s2);
      const real_t h = Step((side == 1) ? s1(j) : s2(j));
      if (side == 1) { p1(j) += h; m1(j) -= h; }
      else           { p2(j) += h; m2(j) -= h; }

      Vector fp(neq), fm(neq);
      nf.Average(p1, p2, nor, Tr, fp);
      nf.Average(m1, m2, nor, Tr, fm);
      for (int i = 0; i < neq; i++)
      {
         fd(i, j) = (fp(i) - fm(i)) / (2.0 * h);
      }
   }
}

/** @brief The exact bias the central difference of AverageGrad(1, ...) carries
    at a state with `v.n == 0`, in the velocity column @a j.

    `S(uhat) = |v.n| + sqrt((v.n)^2 + beta|n|^2)` has a kink at `v.n = 0`, and
    the difference does not average it away. Perturbing `uhat_{1+jj}` by `+-h`
    from `v.n = 0` gives the *same* enlarged stabilization
    `Sh = a + sqrt(a^2 + beta|n|^2)`, `a = h|n_jj|`, on both sides, while
    `(u - uhat)_i` changes by `-+h delta_ij`, so the difference of the
    stabilization term is `-Sh delta_ij` where the derivative is `-S delta_ij`
    (the `(u-uhat)(dS/duhat)^T` term is analytically zero there, since both
    `sgn(0) := 0` and `v.n / r = 0`). The flux part `F(uhat).n` is quadratic
    and its central difference is exact. So the whole discrepancy is
    `-(Sh - S)` on the diagonal and nothing anywhere else.

    This is a property of the function, not of the code, and it is first order
    in the step, and it was measured rather than assumed: at `h = 1e-6` the
    difference of the exact code less its analytic Jacobian is 5.000e-07,
    2.500e-07 and 1.250e-07 in the three velocity diagonal entries, against
    `|n| = 0.5, 0.25, 0.125` -- `h|n_jj|` to four digits -- and 3e-11, the
    round-off floor, in every off-diagonal entry. At `beta = 1` that is a
    relative error of 8.9e-07 against a 1e-6 tolerance, a margin of 12%.
    Correcting the difference by the derived bias, rather than widening the
    tolerance to hide it or leaving the check one state change away from
    flaking, keeps `v.n == 0` exactly as tight as everywhere else: corrected,
    the same entries agree to 3e-11. */
real_t KinkBias(const Vector &uhat, const Vector &nor, real_t beta, int j)
{
   if (j == 0) { return 0.0; }   // S does not depend on the pressure
   const real_t nn = NormalSq(nor);
   const real_t a  = Step(uhat(j)) * std::abs(nor(j - 1));
   return a + std::sqrt(a * a + beta * nn) - std::sqrt(beta * nn);
}

} // namespace hdg_nsflux

TEST_CASE("nsflux.hpp: the artificial-compressibility flux and its Jacobians",
          "[HyperbolicFlux][NSFlux]")
{
   using namespace hdg_nsflux;

   // dim is 2 or 3 only: the flux writes dim entries and the face
   // transformation must match, as in test_hyperbolic_hdg.cpp.
   const int dim = GENERATE(2, 3);
   // beta = 1 puts the pseudo-acoustic speed at the same order as the
   // convective one; beta = 100 puts the stabilization two orders above it,
   // which is the regime section 5 of the roadmap is about and the one in
   // which a relative check is easiest to pass by accident.
   const real_t beta = GENERATE((real_t) 1.0, (real_t) 100.0);
   const int sid = GENERATE(0, 1, 2, 3);
   // The Stokes branch drops v (x) v, which makes the flux linear in the
   // state and lambda constant. Every analytic derivative therefore has a
   // second, structurally different form, and a term guarded by the wrong
   // side of an `if (stokes)` is exactly the kind of edit no solve would
   // report.
   const bool stokes = GENERATE(false, true);
   CAPTURE(dim, beta, StateName(sid), stokes);

   Geom g(dim);
   ArtificialCompressibilityFlux flux(dim, beta, stokes);
   const int neq = dim + 1;
   REQUIRE(flux.num_equations == neq);
   REQUIRE(flux.dim == dim);
   REQUIRE(flux.GetBeta() == beta);
   REQUIRE(flux.IsStokes() == stokes);

   Vector nor;
   MakeNormal(dim, nor);
   Vector U;
   MakeState(dim, sid, U);

   const real_t vn = DotN(U, nor);
   const real_t nn = NormalSq(nor);
   const bool vn_zero = (sid == ZERO_VELOCITY || sid == ORTHOGONAL);
   // The state generator promises what it says on the tin; if it ever stops
   // doing so the kink cases below are silently testing nothing.
   if (vn_zero) { REQUIRE(vn == 0.0); }
   else         { REQUIRE(std::abs(vn) > 0.1); }
   // Only the Navier-Stokes lambda has a kink: the Stokes one is constant.
   const bool on_kink = vn_zero && !stokes;

   SECTION("ComputeFluxJacobian differences ComputeFlux")
   {
      DenseTensor J(neq, neq, dim);
      flux.ComputeFluxJacobian(U, *g.Tr, J);

      for (int j = 0; j < neq; j++)
      {
         const real_t h = Step(U(j));
         Vector Up(U), Um(U);
         Up(j) += h;
         Um(j) -= h;

         DenseMatrix Fp(neq, dim), Fm(neq, dim);
         flux.ComputeFlux(Up, *g.Tr, Fp);
         flux.ComputeFlux(Um, *g.Tr, Fm);

         for (int i = 0; i < neq; i++)
         {
            for (int d = 0; d < dim; d++)
            {
               const real_t diff = (Fp(i, d) - Fm(i, d)) / (2.0 * h);
               INFO("dF(" << i << "," << d << ")/du(" << j << ") : analytic "
                    << J(i, j, d) << " vs difference " << diff);
               REQUIRE(J(i, j, d) == MFEM_Approx(diff, fd_tol, fd_tol));
            }
         }
      }
   }

   SECTION("ComputeFluxDotN is ComputeFlux contracted with the normal")
   {
      DenseMatrix F(neq, dim);
      flux.ComputeFlux(U, *g.Tr, F);

      Vector FdotN(neq);
      const real_t speed = flux.ComputeFluxDotN(U, nor, *g.FTr, FdotN);

      Vector expect(neq);
      F.Mult(nor, expect);
      for (int i = 0; i < neq; i++)
      {
         INFO("component " << i);
         REQUIRE(FdotN(i) == MFEM_Approx(expect(i), exact_tol, exact_tol));
      }

      // The characteristic speed is the documented eigenvalue bound.
      const real_t lambda = Lambda(stokes, vn, nn, beta);
      REQUIRE(speed == MFEM_Approx(lambda, exact_tol, exact_tol));
      REQUIRE(flux.MaxCharSpeedDotN(U, nor) ==
              MFEM_Approx(lambda, exact_tol, exact_tol));

      // ComputeFlux() reports the same bound for a unit normal, |n| = 1.
      DenseMatrix Fdummy(neq, dim);
      real_t vv = 0.0;
      for (int d = 0; d < dim; d++) { vv += U(1 + d) * U(1 + d); }
      const real_t unit = stokes ? std::sqrt(beta)
                          : std::sqrt(vv) + std::sqrt(vv + beta);
      REQUIRE(flux.ComputeFlux(U, *g.Tr, Fdummy) ==
              MFEM_Approx(unit, exact_tol, exact_tol));
   }

   SECTION("ComputeFluxJacobianDotN differences ComputeFluxDotN")
   {
      DenseMatrix JDotN(neq);
      flux.ComputeFluxJacobianDotN(U, nor, *g.FTr, JDotN);

      for (int j = 0; j < neq; j++)
      {
         const real_t h = Step(U(j));
         Vector Up(U), Um(U);
         Up(j) += h;
         Um(j) -= h;

         Vector Fp(neq), Fm(neq);
         flux.ComputeFluxDotN(Up, nor, *g.FTr, Fp);
         flux.ComputeFluxDotN(Um, nor, *g.FTr, Fm);

         for (int i = 0; i < neq; i++)
         {
            const real_t diff = (Fp(i) - Fm(i)) / (2.0 * h);
            INFO("d(F.n)(" << i << ")/du(" << j << ") : analytic "
                 << JDotN(i, j) << " vs difference " << diff);
            REQUIRE(JDotN(i, j) == MFEM_Approx(diff, fd_tol, fd_tol));
         }
      }
   }

   SECTION("ComputeFluxJacobianDotN is ComputeFluxJacobian contracted with n")
   {
      // The override exists only to skip allocating the (m,m,dim) tensor, so
      // it must reproduce the base class's contraction to round-off, not to a
      // difference tolerance.
      DenseMatrix JDotN(neq);
      flux.ComputeFluxJacobianDotN(U, nor, *g.FTr, JDotN);

      DenseTensor J(neq, neq, dim);
      flux.ComputeFluxJacobian(U, *g.Tr, J);

      DenseMatrix expect(neq);
      expect = 0.0;
      for (int d = 0; d < dim; d++) { expect.Add(nor(d), J(d)); }

      for (int i = 0; i < neq; i++)
      {
         for (int j = 0; j < neq; j++)
         {
            INFO("(" << i << "," << j << ") : override " << JDotN(i, j)
                 << " vs contraction " << expect(i, j));
            REQUIRE(JDotN(i, j) == MFEM_Approx(expect(i, j),
                                               exact_tol, exact_tol));
         }
      }
   }

   SECTION("MaxCharSpeedDotNGrad differences MaxCharSpeedDotN")
   {
      Vector dlambda;
      flux.MaxCharSpeedDotNGrad(U, nor, dlambda);
      REQUIRE(dlambda.Size() == neq);

      // lambda does not depend on the pressure, in the code or in the formula.
      REQUIRE(dlambda(0) == 0.0);

      for (int j = 0; j < neq; j++)
      {
         const real_t h = Step(U(j));
         Vector Up(U), Um(U);
         Up(j) += h;
         Um(j) -= h;

         const real_t diff = (flux.MaxCharSpeedDotN(Up, nor) -
                              flux.MaxCharSpeedDotN(Um, nor)) / (2.0 * h);
         INFO("dlambda/du(" << j << ") : analytic " << dlambda(j)
              << " vs difference " << diff);
         REQUIRE(dlambda(j) == MFEM_Approx(diff, fd_tol, fd_tol));
      }

      if (stokes)
      {
         // Stokes: lambda is the constant sqrt(beta)|n|, so the gradient is
         // zero for a reason that has nothing to do with the kink, and the
         // difference above confirms it at every state including v.n != 0.
         for (int j = 0; j < neq; j++)
         {
            INFO("component " << j);
            REQUIRE(dlambda(j) == 0.0);
         }
      }
      else if (vn_zero)
      {
         // At v.n == 0 the |v.n| term is not differentiable, and the loop
         // above does NOT establish that it is. lambda is an even function of
         // the perturbation there -- both the +h and the -h state give
         // |h n_j| + sqrt(h^2 n_j^2 + beta|n|^2) -- so the central difference
         // returns zero identically, for every implementation of |.|, and it
         // agrees with the analytic zero for a reason that has nothing to do
         // with the analytic zero being right. What the check above pins at
         // this state is therefore only the convention, sgn(0) := 0, which is
         // the symmetric subgradient and the one the difference happens to
         // report. The two statements worth making explicitly are that the
         // gradient is exactly zero here (so no one-sided value leaked in)
         // and that the smooth half of lambda, sqrt((v.n)^2 + beta|n|^2), has
         // zero derivative here too, which is what makes 0 a legitimate
         // choice rather than a convenient one.
         for (int j = 0; j < neq; j++)
         {
            INFO("component " << j);
            REQUIRE(dlambda(j) == 0.0);
         }
         const real_t r = std::sqrt(vn * vn + beta * nn);
         REQUIRE(r == MFEM_Approx(std::sqrt(beta * nn), exact_tol, exact_tol));
         REQUIRE(vn / r == 0.0);
      }
      else
      {
         // Away from the kink, check the closed form the doc comment states
         // as well as the difference: n_j (sgn(v.n) + (v.n)/r).
         const real_t r = std::sqrt(vn * vn + beta * nn);
         const real_t s = ((vn > 0.0) ? 1.0 : -1.0) + vn / r;
         for (int d = 0; d < dim; d++)
         {
            INFO("velocity component " << d);
            REQUIRE(dlambda(1 + d) ==
                    MFEM_Approx(s * nor(d), exact_tol, exact_tol));
         }
      }
   }
}

TEST_CASE("nsflux.hpp: the HDG Lax-Friedrichs numerical flux and its Jacobians",
          "[HyperbolicFlux][NSFlux]")
{
   using namespace hdg_nsflux;

   const int dim = GENERATE(2, 3);
   const real_t beta = GENERATE((real_t) 1.0, (real_t) 100.0);
   const int sid = GENERATE(0, 1, 2, 3);
   const bool stokes = GENERATE(false, true);
   CAPTURE(dim, beta, StateName(sid), stokes);

   Geom g(dim);
   ArtificialCompressibilityFlux flux(dim, beta, stokes);
   HDGLaxFriedrichsFlux nflux(flux);
   const int neq = dim + 1;

   Vector nor;
   MakeNormal(dim, nor);

   // state1 == uhat, the trace; state2 == u, the element state. That order is
   // what fem/hyperbolic.cpp's HDG path passes and what the class documents.
   Vector uhat, u;
   MakeState(dim, sid, uhat);
   MakeElementState(uhat, u);

   const real_t vn = DotN(uhat, nor);
   const real_t nn = NormalSq(nor);
   const bool vn_zero = (sid == ZERO_VELOCITY || sid == ORTHOGONAL);
   const bool on_kink = vn_zero && !stokes;
   // dS/duhat is identically zero both in Stokes (lambda is constant) and at
   // v.n == 0 (sgn(0) := 0 and v.n/r = 0), and those are exactly the states at
   // which the frozen flag has nothing to drop.
   const bool no_dS = stokes || vn_zero;
   const real_t S = Lambda(stokes, vn, nn, beta);

   SECTION("Average is F(uhat).n + S(uhat)(u - uhat)")
   {
      Vector avg(neq);
      const real_t speed = nflux.Average(uhat, u, nor, *g.FTr, avg);

      Vector FN(neq);
      flux.ComputeFluxDotN(uhat, nor, *g.FTr, FN);

      for (int i = 0; i < neq; i++)
      {
         const real_t expect = FN(i) + S * (u(i) - uhat(i));
         INFO("component " << i);
         REQUIRE(avg(i) == MFEM_Approx(expect, exact_tol, exact_tol));
      }
      // Both the flux's reported speed and S are lambda_max here, so the
      // max() of them is lambda_max.
      REQUIRE(speed == MFEM_Approx(S, exact_tol, exact_tol));
   }

   SECTION("Average(u, u) is the flux at u, the stabilization cancelling")
   {
      // The consistency the whole HDG discretization rests on: when the trace
      // and the element state agree, the numerical flux is the physical one.
      Vector avg(neq), FN(neq);
      nflux.Average(uhat, uhat, nor, *g.FTr, avg);
      flux.ComputeFluxDotN(uhat, nor, *g.FTr, FN);
      for (int i = 0; i < neq; i++)
      {
         INFO("component " << i);
         REQUIRE(avg(i) == FN(i));   // S * 0.0 is exactly 0.0
      }
   }

   SECTION("AverageGrad(2, ...) differences Average in the element state")
   {
      DenseMatrix grad(neq);
      nflux.AverageGrad(2, uhat, u, nor, *g.FTr, grad);

      DenseMatrix fd;
      FDAverage(nflux, 2, uhat, u, nor, *g.FTr, fd);

      for (int i = 0; i < neq; i++)
      {
         for (int j = 0; j < neq; j++)
         {
            INFO("d(Fhat_" << i << ")/du_" << j << " : analytic " << grad(i, j)
                 << " vs difference " << fd(i, j));
            REQUIRE(grad(i, j) == MFEM_Approx(fd(i, j), fd_tol, fd_tol));
            // and it is S I exactly: the element state enters linearly.
            REQUIRE(grad(i, j) == MFEM_Approx((i == j) ? S : 0.0,
                                              exact_tol, exact_tol));
         }
      }
   }

   SECTION("AverageGrad(1, ...) differences Average in the trace state")
   {
      DenseMatrix grad(neq);
      nflux.AverageGrad(1, uhat, u, nor, *g.FTr, grad);

      DenseMatrix fd;
      FDAverage(nflux, 1, uhat, u, nor, *g.FTr, fd);

      for (int j = 0; j < neq; j++)
      {
         // See KinkBias(): zero unless this state sits on |v.n|'s kink.
         const real_t bias = on_kink ? KinkBias(uhat, nor, beta, j) : 0.0;
         for (int i = 0; i < neq; i++)
         {
            const real_t ref = fd(i, j) + ((i == j) ? bias : 0.0);
            INFO("d(Fhat_" << i << ")/duhat_" << j << " : analytic "
                 << grad(i, j) << " vs difference " << fd(i, j)
                 << " (kink bias " << ((i == j) ? bias : 0.0) << ")");
            REQUIRE(grad(i, j) == MFEM_Approx(ref, fd_tol, fd_tol));
         }
      }
   }

   SECTION("SetFrozenStabilizationJacobian drops exactly (u-uhat)(dS/duhat)^T")
   {
      HDGLaxFriedrichsFlux frozen(flux);
      frozen.SetFrozenStabilizationJacobian(true);

      DenseMatrix gx(neq), gf(neq);
      nflux.AverageGrad(1, uhat, u, nor, *g.FTr, gx);
      frozen.AverageGrad(1, uhat, u, nor, *g.FTr, gf);

      Vector dS;
      flux.MaxCharSpeedDotNGrad(uhat, nor, dS);

      // The two Jacobians differ by the outer product, exactly.
      for (int i = 0; i < neq; i++)
      {
         for (int j = 0; j < neq; j++)
         {
            const real_t drop = (u(i) - uhat(i)) * dS(j);
            INFO("(" << i << "," << j << ") exact " << gx(i, j) << " frozen "
                 << gf(i, j) << " dropped term " << drop);
            REQUIRE(gx(i, j) - gf(i, j) ==
                    MFEM_Approx(drop, exact_tol, exact_tol));
         }
      }

      DenseMatrix fd;
      FDAverage(nflux, 1, uhat, u, nor, *g.FTr, fd);

      if (no_dS)
      {
         // dS is identically zero here -- in Stokes because lambda does not
         // depend on the state at all, and at v.n == 0 because sgn(0) := 0 and
         // v.n/r = 0 -- so there is nothing for the flag to drop and the two
         // Jacobians are the same matrix. The flag's meaning cannot be pinned
         // at these states; it is pinned in the branch below.
         REQUIRE(dS.Norml2() == 0.0);
         for (int i = 0; i < neq; i++)
            for (int j = 0; j < neq; j++)
            {
               REQUIRE(gf(i, j) == gx(i, j));
            }
      }
      else
      {
         // The dropped term is not a rounding-order quantity: if it were, the
         // disagreement asserted next would be meaningless.
         real_t drop_max = 0.0;
         for (int i = 0; i < neq; i++)
            for (int j = 0; j < neq; j++)
            {
               drop_max = std::max(drop_max,
                                   std::abs((u(i) - uhat(i)) * dS(j)));
            }
         CAPTURE(drop_max);
         REQUIRE(drop_max > 1e-2);

         // With u != uhat the frozen Jacobian is NOT the derivative: it misses
         // the term by exactly that outer product.
         real_t worst = 0.0;
         for (int i = 0; i < neq; i++)
            for (int j = 0; j < neq; j++)
            {
               worst = std::max(worst, std::abs(gf(i, j) - fd(i, j)));
            }
         CAPTURE(worst);
         REQUIRE(worst > 1e-3);
         REQUIRE(worst == MFEM_Approx(drop_max, fd_tol, fd_tol));

         // With u == uhat the dropped term is multiplied by zero, so the
         // frozen Jacobian is the derivative after all. That is the whole
         // content of the flag: it is exact at a converged trace and wrong
         // only off it.
         DenseMatrix gf0(neq), fd0;
         frozen.AverageGrad(1, uhat, uhat, nor, *g.FTr, gf0);
         FDAverage(nflux, 1, uhat, uhat, nor, *g.FTr, fd0);
         for (int j = 0; j < neq; j++)
         {
            for (int i = 0; i < neq; i++)
            {
               INFO("u == uhat, (" << i << "," << j << ") : frozen "
                    << gf0(i, j) << " vs difference " << fd0(i, j));
               REQUIRE(gf0(i, j) == MFEM_Approx(fd0(i, j), fd_tol, fd_tol));
            }
         }
      }
   }
}
