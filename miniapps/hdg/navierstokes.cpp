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
//
//                            Navier-Stokes miniapp
//
// Compile with: make navierstokes
//
// Sample runs:  navierstokes -p 1 -nx 8 -ny 8 -o 2 -rtol 1e-12
//               navierstokes -p 1 -nx 8 -ny 8 -o 2 -stokes
//               navierstokes -p 3 -nx 6 -ny 4 -o 2 -rtol 1e-12
//               navierstokes -p 4 -nx 6 -ny 4 -o 2 -rtol 1e-12
//               navierstokes -p 2 -nx 24 -ny 16 -o 2 -re 40 -cont
//               navierstokes -p 2 -nx 24 -ny 16 -o 2 -re 40 -cont -tau 1
//               navierstokes -p 1 -nx 16 -ny 4 -sx 4 -o 2 -re 200 -cont
//               navierstokes -p 1 -nx 8 -ny 8 -o 2 -bcphys -cont -rtol 1e-12 -atol 1e-12
//               navierstokes -p 1 -nx 8 -ny 8 -o 2 -bcphys -bcnat 3 -cont -atol 1e-12
//
// Description:  This miniapp solves the steady incompressible Navier-Stokes
//               equations by the HDG method of Peraire, Nguyen and Cockburn,
//               *A hybridizable discontinuous Galerkin method for the
//               compressible Euler and Navier-Stokes equations*, AIAA
//               2010-363, on the mixed (flux/potential/trace) interface that
//               DarcyForm and DarcyHybridization provide.
//
//               The equations are put in the first-order form that paper's
//               Eq. (13) uses,
//
//                   q + nu grad u          = 0,
//                   div (F(u) + G(u,q))    = s,
//
//               with the state written in Chorin's artificial-compressibility
//               variables
//
//                   u = (p, v_1, ..., v_d),        neq = dim + 1,
//
//               so that the inviscid flux F and the viscous flux G are
//
//                   F_{0,d}   = beta v_d,          G_{0,d}   = 0,
//                   F_{1+i,d} = v_i v_d + p d_id,  G_{1+i,d} = +q_{1+i,d}.
//
//               **Both of those signs are corrections, and the ones they
//               replace were wrong.** This comment used to write the first
//               equation as `q - nu grad u = 0` and hence `G = -q`. That pair
//               is internally consistent but is not what the code does:
//               ExactFlux() below, and its doxygen, define `q = -nu grad u`,
//               and `-bcfull` reproduces it to 4.4e-15 -- a relative flux
//               error that would be 2, not 4e-15, if the sign were the other
//               way. The pair above is the code's. Nothing in the
//               discretisation changed when this was corrected; what changed
//               is that the boundary datum of step 11, which was written from
//               this comment, had the viscous half backwards until a sweep of
//               its scaling put the zero at -1 instead of +1.
//
//               The continuity row is then `beta div v = s_0`, so the
//               incompressibility is *imposed*, not penalised: at a root of
//               the steady residual `div v = s_0/beta` exactly, whatever beta
//               is. What beta buys is a hyperbolic character for the pressure,
//               and with it a well-defined stabilization -- see nsflux.hpp.
//
//               The numerical flux is Eq. (3) of the reference,
//
//                   (F + G)^ . n = (F(u^) + G(u^,q)) . n + S (u - u^),
//
//               with S the local Lax-Friedrichs matrix of its Eq. (6),
//               `S = lambda_max(u^, n) I`. The diffusive half of the
//               stabilization is carried separately by HDGDiffusionIntegrator,
//               which is the `s = s_diff + s_conv` splitting of NPC-1 section
//               3.6 rather than one lumped constant.
//
//               **Why this problem, and what it is for.** The open question
//               it was built to ask is whether one tau can serve convection in
//               one coordinate direction and diffusion in another at the same
//               time. Plane Poiseuille flow is the smallest problem that
//               has that character: it is diffusive across the channel and
//               convective along it. On a face whose normal lies along the
//               flow `lambda_max = |v| + sqrt(v^2 + beta)`; on one across the
//               flow it is `sqrt(beta)`. A single constant tau cannot be both,
//               and `-tau <c>` against the default is how this miniapp asks
//               what that costs.
//
//               **What that sweep measured, and it is not what was expected.**
//               937 runs over both problems, orders 1-3, Re 10 to 1000, cell
//               aspect ratios 1/4 to 4, tau 0.125 to 10, and beta over a
//               factor of 16:
//
//                 - `S = lambda_max(u^,n) I` is 2.0-3.6x *worse* than the best
//                   constant tau in the flux and in the pressure, and
//                   indistinguishable from it in the potential. Over-
//                   stabilization also costs the flux its rate: the last
//                   measured rate at k = 2 falls from 2.60 to 2.18 as tau goes
//                   0.5 to 5.
//                 - The mechanism, established by sweeping beta rather than by
//                   inspection. beta cannot change the steady answer -- the
//                   continuity row is `beta div v = s_0` -- but it sets
//                   `lambda_max = sqrt(beta)` on every face where v.n = 0.
//                   lambda_max's error tracks the constant sqrt(beta) to
//                   within 5-16% and *moves with beta*, while a fixed tau is
//                   beta-independent to 0.02%. So lambda_max is a constant
//                   sqrt(beta) in disguise on the faces that do the work: on
//                   the along-flow faces, where it is larger, both exact
//                   solutions are already representable and the stabilization
//                   there does nothing. Its level is set by an arbitrary
//                   parameter of the formulation rather than by the physics,
//                   and the extra weight it carries is a penalty.
//                 - It wins the other half. lambda_max converged on every one
//                   of ~300 Kovasznay cases; every constant tau <= 1 diverges
//                   somewhere, always on *coarse* meshes at high Re and
//                   recovering under refinement. Cold on plane Poiseuille at
//                   Re = 100 it converges in 9 iterations where tau in
//                   [0.25, 2] all fail and only tau >= 5 recovers -- and since
//                   lambda_max never exceeds 2.4 there, it is not the amount
//                   of stabilization but where it is put. The accuracy optimum
//                   sits exactly at that robustness boundary: at Re = 400 the
//                   best converging tau is 0.375 at 48x32 and 0.5 at 24x16.
//                 - The mesh aspect ratio changes none of it; lambda_max's
//                   penalty is if anything largest on cells stretched along
//                   the flow.
//
//               **What that does not settle, and the reason is about these two
//               problems rather than about tau.** Both put their sharp
//               structure across the flow and little or none along it, so the
//               along-flow faces -- the only ones where lambda_max differs
//               from sqrt(beta) -- are exactly the faces where the solution is
//               easiest to represent. Kovasznay cannot repair that on its own
//               window, because its decay rate
//
//                   lambda = Re/2 - sqrt(Re^2/4 + 4 pi^2)  ->  -4 pi^2 / Re,
//
//               so the parameter that makes it convective is the parameter
//               that flattens its along-flow structure: `e^(lambda x)` varies
//               by 94x across the standard window at Re = 10 and by **1.16x at
//               Re = 400**, with the measured consequence that the errors are
//               identical to four digits over a 16x range in nx. Pass
//               `-sx 4 -re 40` when along-flow structure is wanted at a
//               convective Reynolds number. A genuinely two-directional exact
//               solution is what would settle the general question, and none
//               of the four problems here is one.
//
//               **The solve is NPC**, Newton on the full (q, u, u_hat)
//               system with the Jacobian solved by hybridized elimination --
//               DarcyNPCOperator and DarcyNPCSolver, see
//               DarcyHybridization::NPCResidual(). It used to go through
//               DarcyOperator, whose unknown is the TRACE alone and which
//               rebuilds the fields from it afterwards; NPC keeps them as
//               Newton state, so there is nothing to recover and the
//               pseudo-time step a steady problem never wanted is gone.
//               GetNumLocalNLIterations() prints 0, which is what says the
//               method really changed.
//
//               **One consequence a caller has to know: the convergence test
//               is now on the FULL residual, not the trace alone.** The local
//               rows dominate its norm, so a given -rtol stops earlier in
//               trace terms than it used to. Order 3 Poiseuille reads 8.9e-13
//               at -rtol 1e-10 and 8.7e-15 at 1e-14; the answer was never
//               wrong, the test was measuring half the system. An rtol that
//               was adequate before may not be now.
//
//               **What is verified.** Both routes reach the same discrete
//               solution, each run at a tolerance that stops it improving --
//               relative error in q, old trace-only against NPC:
//               8.09e-15 / 4.42e-15 on Poiseuille at order 2, 1.01e-14 /
//               8.72e-15 at order 3, 4.26e-16 / 3.88e-16 on uniform flow,
//               1.05e-14 / 9.68e-15 on Couette, and Kovasznay identical to
//               every digit at 0.00853108, which is the case where the
//               discretisation rather than the solver sets the error. NPC is
//               not more accurate and should not be: it is the same discrete
//               problem by a different method.
//
//               Plane Poiseuille at order >= 2 comes back at round-off and is
//               correctly inexact at order 1. That holds for BOTH boundary
//               sets: -bcfull gives 4.6e-15 / 2.2e-15 / 3.8e-16 in q / p / v at
//               order 2, and -bcphys -cont gives 1.1e-14 / 3.1e-14 / 2.4e-15,
//               with -bcnat 3 -- which puts a whole wall on the traction
//               condition, the only case here whose datum has a nonzero
//               VISCOUS part -- at 2.1e-13 / 1.1e-13 / 1.2e-14. Kovasznay at Re = 40 gives the
//               optimal k+1 in the potential -- v rate 2.11 at k = 1, 3.09 at
//               k = 2, 4.11 at k = 3 over 1/h = 4, 8, 16, 32 -- while the flux
//               rates lag and are still climbing there, which is the roadmap's
//               warning that rates must be taken asymptotically. Any rate
//               study needs `-gm 0` and a tight `-rtol`: the default
//               GS-preconditioned trace solve leaves the pressure error 13%
//               off the converged value at 48x32 (2.569e-5 against 2.274e-5)
//               with Newton's *relative* test satisfied both times, so the
//               comparison would be measuring the preconditioner rather than
//               the discretisation.
//
//               Problem 1, plane Poiseuille, is an exact solution of the
//               equations with **zero source term**, and it is a polynomial:
//               pressure linear in x, velocity quadratic in y. At order k >= 2
//               it therefore lies in the discrete space and the HDG solution
//               must reproduce it to round-off. That is the sharpest
//               correctness control the problem offers, and it is checked
//               before anything about rates is believed.
//
//               Problem 2, Kovasznay flow, is an exact solution with genuine
//               convection (v.grad v is not zero), also with zero source, and
//               is what a convergence table is taken on.
//
//               The miniapp is written so that the compressible equations drop
//               in later: the state layout puts the continuity variable first
//               exactly as EulerFlux's (rho, rho v, rho E) does, and the only
//               things that know which equations they are are the flux
//               function and the per-equation viscosity table. See
//               "Extending to the compressible equations" at the foot of this
//               file.
//
//               We recommend viewing the convdiff miniapp before this one.

#include "mfem.hpp"
#include "darcyop.hpp"
#include "nsflux.hpp"

#ifndef MFEM_USE_SUNDIALS
#error This miniapp requires that MFEM is built with MFEM_USE_SUNDIALS=YES
#endif

#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;
using namespace mfem::hdg;

enum Problem
{
   PlanePoiseuille = 1,
   Kovasznay,
   /** @name Bisection problems
       Two more exact zero-source solutions, ordered by which terms of the
       operator they can see. They exist because the residual at the exact
       solution is the only measurement that says whether the assembled
       operator is the intended one, and a single problem that fails the test
       does not say which term failed.

       - UniformFlow: `v` constant, `p` constant, so `q = 0` and `div q = 0`.
         Everything in the flux/divergence/constraint chain is multiplied by
         zero, and only the inviscid flux and the trace treatment can produce
         a residual.
       - Couette: `v_x` linear in `y`, so `q` is a nonzero constant and
         `div q = 0`. This adds the flux recovery -- `q = -nu grad u` -- and
         the constraint that carries it, but still no second derivative.

       Plane Poiseuille is then the first problem with `div q` nonzero. */
   ///@{
   UniformFlow,
   Couette,
   ///@}
};

/// Everything the exact solution and the boundary conditions need.
struct ProblemParams
{
   Problem prob{Problem::PlanePoiseuille};
   int dim{2};
   real_t nu{1.};        ///< kinematic viscosity
   real_t beta{1.};      ///< artificial compressibility
   real_t U{1.};         ///< reference velocity (centreline / freestream)
   real_t x0{0.}, y0{0.};///< mesh origin
   real_t sx{1.}, sy{1.};///< mesh extent
   real_t Re{40.};       ///< Reynolds number, Kovasznay only
};

/// The Kovasznay decay rate, lambda = Re/2 - sqrt(Re^2/4 + 4 pi^2).
static inline real_t KovasznayLambda(real_t Re)
{
   return Re / 2. - sqrt(Re * Re / 4. + 4. * M_PI * M_PI);
}

/** @brief The exact state `u = (p, v)` at @a x.

    Both problems are exact solutions of the steady incompressible equations
    with **zero source term**, which is why no forcing appears anywhere below.
    That is not an accident of the choice: for plane Poiseuille the convective
    term vanishes identically on the exact profile and the viscous term
    balances the constant pressure gradient, and Kovasznay is constructed to
    balance. It means every term of the residual is exercised and none of them
    is being fed a manufactured right-hand side that could hide a sign. */
static void ExactState(const ProblemParams &pars, const Vector &x, Vector &u)
{
   const int dim = pars.dim;
   u.SetSize(dim + 1);
   u = 0.;

   switch (pars.prob)
   {
      case Problem::PlanePoiseuille:
      {
         // Channel (x0, x0+sx) x (y0, y0+sy); half height h, centreline yc.
         const real_t h = 0.5 * pars.sy;
         const real_t yc = pars.y0 + h;
         const real_t yr = (x(1) - yc) / h;
         // v_x = U (1 - yr^2), driven by dp/dx = -G with G = 2 nu U / h^2.
         const real_t G = 2. * pars.nu * pars.U / (h * h);
         u(0) = -G * (x(0) - pars.x0);        // p, gauge: p = 0 at the inlet
         u(1) = pars.U * (1. - yr * yr);      // v_x
         u(2) = 0.;                           // v_y
         if (dim == 3) { u(3) = 0.; }
         break;
      }
      case Problem::Kovasznay:
      {
         const real_t lam = KovasznayLambda(pars.Re);
         const real_t ex = exp(lam * x(0));
         u(0) = 0.5 * (1. - exp(2. * lam * x(0)));                   // p
         u(1) = 1. - ex * cos(2. * M_PI * x(1));                     // v_x
         u(2) = lam / (2. * M_PI) * ex * sin(2. * M_PI * x(1));      // v_y
         if (dim == 3) { u(3) = 0.; }
         break;
      }
      case Problem::UniformFlow:
         u(0) = 0.;
         u(1) = pars.U;
         break;
      case Problem::Couette:
         u(0) = 0.;
         u(1) = pars.U * (x(1) - pars.y0) / pars.sy;
         break;
   }
}

/// The exact velocity alone, as the error norms and the BCs want it.
static void ExactVelocity(const ProblemParams &pars, const Vector &x, Vector &v)
{
   Vector u;
   ExactState(pars, x, u);
   v.SetSize(pars.dim);
   for (int d = 0; d < pars.dim; d++) { v(d) = u(1 + d); }
}

static real_t ExactPressure(const ProblemParams &pars, const Vector &x)
{
   Vector u;
   ExactState(pars, x, u);
   return u(0);
}

/** @brief The exact flux `q = -nu grad u`, laid out as the flux space expects:
    component `e*dim + d` is direction @a d of equation @a e.

    Equation 0 (the pressure row) is identically zero, and that is not a
    simplification of the exact solution -- it is what the discretisation
    imposes. See "the pressure has no gradient variable" in main(). */
static void ExactFlux(const ProblemParams &pars, const Vector &x, Vector &q)
{
   const int dim = pars.dim;
   q.SetSize((dim + 1) * dim);
   q = 0.;

   switch (pars.prob)
   {
      case Problem::PlanePoiseuille:
      {
         const real_t h = 0.5 * pars.sy;
         const real_t yc = pars.y0 + h;
         const real_t yr = (x(1) - yc) / h;
         // d(v_x)/dy = -2 U yr / h; every other derivative vanishes.
         q(1 * dim + 1) = -pars.nu * (-2. * pars.U * yr / h);
         break;
      }
      case Problem::Kovasznay:
      {
         const real_t lam = KovasznayLambda(pars.Re);
         const real_t ex = exp(lam * x(0));
         const real_t c = cos(2. * M_PI * x(1)), s = sin(2. * M_PI * x(1));
         // v_x = 1 - e^{lam x} cos(2 pi y)
         q(1 * dim + 0) = -pars.nu * (-lam * ex * c);
         q(1 * dim + 1) = -pars.nu * (2. * M_PI * ex * s);
         // v_y = lam/(2 pi) e^{lam x} sin(2 pi y)
         q(2 * dim + 0) = -pars.nu * (lam * lam / (2. * M_PI) * ex * s);
         q(2 * dim + 1) = -pars.nu * (lam * ex * c);
         break;
      }
      case Problem::UniformFlow:
         break;                                    // q = 0
      case Problem::Couette:
         q(1 * dim + 1) = -pars.nu * pars.U / pars.sy;
         break;
   }
}

/** @brief `F(x) - b`, so that a nonlinear solver which ignores its right-hand
    side still solves the right problem.

    **`KINSolver::Mult(const Vector &b, Vector &x)` discards `b`.** The
    parameter is unnamed in `linalg/sundials.cpp` and nothing reads it, so the
    KINSOL route solves `F(x) = 0` where `NewtonSolver::Mult` solves
    `F(x) = b`. Here `b` is the prescribed boundary flux of step 11, and the
    consequence was that `-ls` and `-nls 4` silently dropped the whole datum
    and solved the zero-numerical-flux problem instead -- the state
    `-no-bcflux` restores, and wrong by more than 100%.

    That diagnosis was measured before it was acted on, by the prediction it
    makes and nothing else could: `-ls` must then reproduce `-no-bcflux`
    exactly. It does, to every printed digit, at three orders --
    1.96933 / 2.14019 / 1.01017 in q/p/v at order 1, 2.83502 / 2.20303 /
    1.00533 at order 2, 3.68725 / 2.48441 / 1.00333 at order 3 -- and `-ls`
    agrees with plain Newton to every digit wherever the datum is zero, which
    is every `-bcfull` run, because `SetSubVector(ess_tdofs, 0.)` leaves `b`
    identically zero when every trace component is essential. That is why the
    fault survived: the boundary set the miniapp verifies itself on is exactly
    the one that cannot see it.

    Folding the load into the operator makes both routes solve the same problem
    and costs one vector subtraction per residual evaluation. The solvers are
    then called with no right-hand side at all, so there is nothing left for
    one of them to ignore. */
class LoadedOperator : public Operator
{
   Operator &op;
   const Vector &load;   ///< by reference: -cont re-assembles it between solves

public:
   LoadedOperator(Operator &op_, const Vector &load_)
      : Operator(op_.Height(), op_.Width()), op(op_), load(load_) { }

   void Mult(const Vector &x, Vector &y) const override
   {
      op.Mult(x, y);
      y -= load;
   }

   Operator &GetGradient(const Vector &x) const override
   {
      return op.GetGradient(x);   // the load is constant, so it drops out
   }
};

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.

   int iproblem = Problem::PlanePoiseuille;
   int nx = 8, ny = 8;
   real_t sx = 1., sy = 1.;
   int order = 2;
   int ref_levels = 0;
   real_t nu = -1.;
   real_t beta = -1.;
   real_t Uref = 1.;
   real_t Re = 40.;
   real_t td = 0.5;
   real_t tau_const = -1.;
   bool stokes = false;
   bool frozen_stab = false;
   bool hybridization = true;
   int solver_type = (int) DarcyOperator::SolverType::Newton;
   real_t newton_rtol = -1.;
   real_t newton_atol = 0.;
   int newton_iters = 1000;
   real_t check_tol = -1.;
   bool line_search = false;
   int gradient_mode = -1;
   bool threaded_assembly = false;
   bool batched_factor = false;
   real_t hsign = -1.;
   bool bc_full = true;
   bool bc_flux = true;
   int bc_nat = 0;
   real_t bc_sf = 1., bc_sg = 1.;
   int bc_io = 0;
   bool continuation = false;
   bool pgrad = false;
   bool bface = false;
   bool exact_init = false;
   bool bc_lin = false;
   real_t u_init = 0.;
   bool visualization = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=plane Poiseuille (exact, zero source, polynomial)\n\t\t"
                  "2=Kovasznay flow (exact, zero source, convective)\n\t\t"
                  "3=uniform flow (bisection: q = 0)\n\t\t"
                  "4=Couette flow (bisection: q constant, div q = 0)\n\t\t");
   args.AddOption(&nx, "-nx", "--ncells-x", "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y", "Number of cells in y.");
   args.AddOption(&sx, "-sx", "--size-x", "Size along x axis.");
   args.AddOption(&sy, "-sy", "--size-y", "Size along y axis.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of uniform refinement levels.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree). Plane Poiseuille "
                  "is a polynomial of degree 2, so order >= 2 must reproduce "
                  "it to round-off.");
   args.AddOption(&nu, "-nu", "--viscosity",
                  "Kinematic viscosity. Negative derives it from -re.");
   args.AddOption(&Re, "-re", "--reynolds",
                  "Reynolds number. Sets nu when -nu is not given, and is the "
                  "parameter of the Kovasznay solution.");
   args.AddOption(&Uref, "-u", "--velocity", "Reference velocity.");
   args.AddOption(&beta, "-b", "--beta",
                  "Artificial compressibility. It does not change the steady "
                  "answer, only the pressure's characteristic speed and hence "
                  "the stabilization floor. Negative picks U^2. That "
                  "documented no-op makes it the cheapest controlled "
                  "experiment on the stabilization available here: at fixed "
                  "-tau the answer is beta-independent to 0.02%, while under "
                  "lambda_max it moves with beta and tracks the constant "
                  "sqrt(beta) to 5-16%. See the header comment.");
   args.AddOption(&td, "-td", "--stab-diff",
                  "Diffusive stabilization factor, giving tau_d = td*nu/h.");
   args.AddOption(&tau_const, "-tau", "--stab-conv-const",
                  "Use the library's constant-Ctau HDGFlux with this value "
                  "instead of the state-dependent Lax-Friedrichs S = "
                  "lambda_max(u^,n) I. Negative (default) uses "
                  "lambda_max. It replaces a stabilization that knows the "
                  "face normal with one that does not, and the sweep in the "
                  "header comment is what it was for: a constant near 0.5 is "
                  "2-3.6x more accurate in the flux and the pressure, and "
                  "diverges on coarse meshes at high Re where lambda_max does "
                  "not.");
   args.AddOption(&stokes, "-stokes", "--stokes", "-ns", "--navier-stokes",
                  "Drop the v (x) v term, leaving the Stokes problem. The "
                  "system is then linear and Newton converges in one step; it "
                  "is the control that separates a convection bug from a "
                  "diffusion or pressure-coupling bug.");
   args.AddOption(&frozen_stab, "-fstab", "--frozen-stab-jacobian",
                  "-no-fstab", "--no-frozen-stab-jacobian",
                  "Drop the (u - u^) dS/du^ term from the trace Jacobian. "
                  "Cannot change the answer, only the Newton history.");
   args.AddOption(&hybridization, "-hb", "--hybridization",
                  "-no-hb", "--no-hybridization", "Enable hybridization.");
   args.AddOption(&solver_type, "-nls", "--nonlinear-solver",
                  "Nonlinear solver type (1=LBFGS, 2=LBB, 3=Newton, "
                  "4=KINSol). Newton is the default and the only one that "
                  "asks for a gradient.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rtol",
                  "Relative tolerance of the outer nonlinear solver. Negative "
                  "keeps the default of 1e-6, which is far too loose to see "
                  "the Poiseuille exactness check.");
   args.AddOption(&newton_atol, "-atol", "--newton-atol",
                  "Absolute tolerance of the outer nonlinear solver. Zero by "
                  "default, which is Newton's purely relative test, and that "
                  "is what an exactness check needs -- any floor stops the "
                  "iteration above round-off (atol 1e-10 leaves plane "
                  "Poiseuille at 2e-9 instead of 2.6e-15). It is here for the "
                  "opposite case: with -cont on a problem where the "
                  "continuation already lands on the answer, the second solve "
                  "starts at the *linear* solver's noise floor, no relative "
                  "reduction is achievable, and Newton spins to max_iters -- "
                  "measured r0 = 3.7e-10 then 4.7e-10, 8.5e-10, 8.9e-10. A "
                  "floor is the fix there.");
   args.AddOption(&newton_iters, "-nit", "--newton-iterations",
                  "Outer Newton iteration cap.");
   args.AddOption(&check_tol, "-chk", "--check-tolerance",
                  "Grade the run: exit non-zero if any relative error exceeds "
                  "this. Negative (the default) prints and always exits 0. It "
                  "exists so `make test` can check a NUMBER rather than a "
                  "return code -- mfem-test grades on the exit status alone "
                  "and deletes the output, so without this the miniapp's "
                  "sharpest property is unverified by any suite. Use an "
                  "ABSOLUTE threshold: plane Poiseuille at order >= 2 is exact "
                  "in the discrete space and lands at round-off, but WHERE in "
                  "the 1e-15s moves with the BLAS (2.6e-15, 7.7e-15 and "
                  "4.4e-15 have all been measured here), so a relative "
                  "comparison against a stored number can never pass. Needs "
                  "-rtol 1e-12: the default 1e-6 stops the Newton far above "
                  "the discretisation and the check fails for that reason "
                  "alone.");
   args.AddOption(&line_search, "-ls", "--line-search",
                  "-no-ls", "--no-line-search",
                  "Globalise with KINSOL's KIN_LINESEARCH, which implies "
                  "-nls 4. It backtracks on the FULL residual, well defined "
                  "here only because the flux, potential and trace are one "
                  "Newton vector so a step scales all three together. Off by "
                  "default: it changes no converged answer, only whether a "
                  "cold start reaches one, and -cont covers the same ground "
                  "for the cases here. On the multi-rooted -bcphys outflow it "
                  "stops the divergence without choosing the root -- see the "
                  "boundary-condition step. Note it is an l2 merit over all three "
                  "blocks -- where the nonlinearity sits in the potential "
                  "block and the flux and trace rows are linear, a full step "
                  "is exactly optimal for two of the three and damping can be "
                  "worse than none; see doc/HDG-ORDERING-API.md section 6.");
   args.AddOption(&threaded_assembly, "-thr", "--threaded-assembly",
                  "-no-thr", "--no-threaded-assembly",
                  "Run the element-local half of the hybridized assembly on "
                  "several threads: DarcyHybridization::AssemblyMode::"
                  "Threaded. The scatter into the trace matrix stays serial "
                  "and ordered, so the answer is identical to the serial run "
                  "whatever OMP_NUM_THREADS says. Needs an MFEM_USE_OPENMP "
                  "and MFEM_THREAD_SAFE build and aborts without one. This "
                  "miniapp is the only driver that puts a SYSTEM and a "
                  "HyperbolicFormIntegrator on that loop, so it is where "
                  "their thread safety is exercised end to end.");
   args.AddOption(&batched_factor, "-batch", "--batched-factor",
                  "-no-batch", "--no-batched-factor",
                  "Factor the element-local blocks through BatchedLinAlg in "
                  "one call: DarcyHybridization::LocalFactorMode::Batched. "
                  "Requires every element's block to be the same size, which "
                  "a structured mesh at uniform order satisfies; the class "
                  "falls back to the serial loop when it is not. The NATIVE "
                  "backend is bit-for-bit the serial path, so this must not "
                  "move any number here.");
   args.AddOption(&gradient_mode, "-gm", "--gradient-mode",
                  "How much of the hybridized trace system to build: "
                  "0=assemble and precondition directly, 1=assemble and "
                  "precondition with GS, 2=matrix free. Negative leaves it.");
   args.AddOption(&pgrad, "-pgrad", "--pressure-gradient",
                  "-no-pgrad", "--no-pressure-gradient",
                  "Give the pressure row a real gradient variable instead of "
                  "zeroing its mass coupling, divergence and constraint. It "
                  "adds -nu lap p to the continuity equation, so it is not a "
                  "consistent discretisation in general -- but for a problem "
                  "whose exact pressure is harmonic it changes nothing, which "
                  "makes it a clean control on whether the zeroing is wired "
                  "correctly.");
   args.AddOption(&continuation, "-cont", "--stokes-continuation",
                  "-no-cont", "--no-stokes-continuation",
                  "Solve the Stokes problem first and continue onto "
                  "Navier-Stokes from its answer. Needed on Kovasznay at "
                  "Re = 40 on the coarser meshes, where a cold Newton from "
                  "rest diverges -- the local element solve runs away before "
                  "the trace system has any information in it. Off by "
                  "default, and deliberately: where the continuation already "
                  "lands on the answer (plane Poiseuille, whose exact profile "
                  "has v.grad v = 0, so the Stokes and Navier-Stokes "
                  "solutions coincide) the second solve starts at the linear "
                  "solver's noise floor and stops there, costing five orders "
                  "of accuracy -- 5.0e-10 against 2.6e-15 at order 2. Ignored "
                  "with -stokes.");
   args.AddOption(&bc_full, "-bcfull", "--bc-full-state",
                  "-bcphys", "--bc-physical",
                  "Boundary treatment. -bcfull (default) makes every component "
                  "of the trace essential on every boundary, carrying the "
                  "exact solution: over-specified as a PDE, but it is the "
                  "standard verification condition. -bcphys uses the physical "
                  "set -- no-slip walls, velocity in at the inlet, pressure at "
                  "the outlet -- and takes the PRESCRIBED NUMERICAL FLUX on "
                  "every trace component it leaves free; see the note at the "
                  "boundary conditions in main(). Plane Poiseuille then comes "
                  "back at round-off at order >= 2 like -bcfull, but it needs "
                  "-cont to get there: the outflow datum is quadratic in each "
                  "free trace velocity dof, so the discrete problem has a "
                  "combinatorial FAMILY of roots -- four were reached at "
                  "order 2 on 8x8 from fifteen starts, and a uniform stream at "
                  "the channel's mean velocity is not one of the starts that "
                  "finds the true one. See the boundary-condition step.");
   args.AddOption(&bc_flux, "-bcflux", "--bc-prescribed-flux",
                  "-no-bcflux", "--no-bc-prescribed-flux",
                  "Supply the prescribed numerical flux <(F+G).n, mu> on the "
                  "boundary trace components that are not essential. On by "
                  "default and a no-op under -bcfull, where there are none. "
                  "Turning it off restores the state -bcphys was in before it "
                  "existed -- a converged solve, at 3e-13, of a problem with "
                  "zero numerical flux at the inlet and the outlet, whose "
                  "answer is wrong by more than 100%. It is kept as the "
                  "control that says the datum is what repairs that.");
   args.AddOption(&bc_nat, "-bcnat", "--bc-natural-attribute",
                  "Make this boundary attribute wholly natural: no essential "
                  "trace component at all, every equation carrying the "
                  "prescribed numerical flux instead. Zero (default) leaves "
                  "the problem's own set alone. It exists because none of the "
                  "four problems here has a nonzero VISCOUS datum on any row "
                  "its physical set leaves free -- they are all unidirectional "
                  "flows whose outlet normal lies along the flow, so q.n "
                  "vanishes there -- and without it half of what -bcflux "
                  "assembles is never exercised. `-p 1 -bcnat 3` puts the top "
                  "wall of the Poiseuille channel on the traction condition, "
                  "where G.n = nu dv_x/dy is the largest term in the datum, "
                  "and the solution is still a degree-2 polynomial so "
                  "round-off is still the test. Use it on a WALL, not on the "
                  "outlet: `-p 1 -bcnat 2` removes the only essential pressure "
                  "dof in the problem and the system goes SINGULAR -- measured, "
                  "the linear Stokes solve then converges to a wrong answer "
                  "with the exact one still a root to 5.1e-16, and the "
                  "nonlinear solve diverges from every start at every order.");
   args.AddOption(&bc_sf, "-bcsf", "--bc-flux-scale-inviscid",
                  "Scaling of the inviscid half of the prescribed boundary "
                  "flux. A diagnostic: 1 is right, and it was measured, not "
                  "reasoned. Sweeping it and -bcsg separately is what "
                  "distinguishes a mis-signed term from an absent one -- a "
                  "sweep with a minimum that is not round-off means a term is "
                  "missing.");
   args.AddOption(&bc_sg, "-bcsg", "--bc-flux-scale-viscous",
                  "Scaling of the viscous half, G.n = -q.n, of the prescribed "
                  "boundary flux. See -bcsf.");
   args.AddOption(&bc_io, "-bcio", "--bc-flux-int-order-offset",
                  "Integration order offset of the prescribed boundary flux. "
                  "A diagnostic, and the one that mattered.");
   args.AddOption(&hsign, "-hsign", "--hyperbolic-sign",
                  "Sign passed to HyperbolicFormIntegrator. A diagnostic: the "
                  "Darcy residual convention is not derivable from the "
                  "documentation and was fixed by measuring the residual at "
                  "the exact solution.");
   args.AddOption(&bface, "-bface", "--b-face-integrator",
                  "-no-bface", "--no-b-face-integrator",
                  "Add the DGNormalTraceIntegrator face term to the flux "
                  "divergence form. Expected to be a no-op under "
                  "hybridization, where B's face integrators are never "
                  "evaluated; kept as the control that says so.");
   args.AddOption(&bc_lin, "-bclin", "--bc-linearised-flux",
                  "-no-bclin", "--no-bc-linearised-flux",
                  "Linearise the boundary numerical flux about the prescribed "
                  "exterior state on attributes carrying a free velocity "
                  "trace, making that row linear in u_hat. Consistent -- the "
                  "exact solution stays a root to 4.3e-16 -- and a no-op under "
                  "-bcfull, where no trace component is free. **OFF by "
                  "default, because it does NOT do what it was written for**: "
                  "it was meant to remove the outflow's family of roots and "
                  "only moves them, 3 spurious to 2 on the sweep in the "
                  "boundary-condition step. What it bought instead is the "
                  "attribution, which was wrong before it existed. See "
                  "HDGLinearisedBdrFlux in nsflux.hpp.");
   args.AddOption(&u_init, "-uinit", "--uniform-init",
                  "Initialise the velocity to a uniform stream of this "
                  "magnitude along x, and the trace with it. Zero leaves the "
                  "state at zero. This exists to make the root sweep "
                  "reproducible: sweeping it is how the outflow's root family "
                  "was counted, and how -bclin was shown to collapse it.");
   args.AddOption(&exact_init, "-xinit", "--exact-init",
                  "-no-xinit", "--no-exact-init",
                  "Start from the exact solution in every field. For a problem "
                  "whose exact solution lies in the discrete space -- plane "
                  "Poiseuille at order >= 2 -- the initial Newton residual is "
                  "then a direct measurement of whether the assembled operator "
                  "is the right one, with the solver taken out of the "
                  "question.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization", "Enable GLVis output.");
   args.AddOption(&device_config, "-d", "--device", "Device configuration.");
   args.ParseCheck();

   const Problem problem = (Problem) iproblem;
   if (iproblem < 1 || iproblem > 4)
   {
      cerr << "Unknown problem " << iproblem << endl;
      return 1;
   }
   if (!hybridization)
   {
      // The non-hybridized LDG face terms of MixedConductionNLFIntegrator
      // abort for more than one equation (nonlininteg_mixed.cpp:299), and the
      // hyperbolic term would then need the two-state Eval path. Refusing is
      // better than aborting deep in an integrator.
      cerr << "This miniapp is hybridized only; -no-hb is not implemented."
           << endl;
      return 1;
   }

   Device device(device_config);
   device.Print();

   // 2. The mesh, and the problem's own geometry.

   ProblemParams pars;
   pars.prob = problem;
   pars.U = Uref;
   pars.Re = Re;

   if (problem == Problem::Kovasznay)
   {
      // The standard Kovasznay window, (-0.5, 1) x (-0.5, 0.5).
      pars.x0 = -0.5; pars.y0 = -0.5;
      if (sx == 1. && sy == 1.) { sx = 1.5; sy = 1.; }
      pars.nu = 1. / Re;
   }
   else
   {
      pars.x0 = 0.; pars.y0 = 0.;
      pars.nu = (nu > 0.) ? nu : (Uref * sy / Re);
   }
   if (nu > 0.) { pars.nu = nu; }
   pars.sx = sx; pars.sy = sy;
   pars.beta = (beta > 0.) ? beta : (Uref * Uref);

   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                     sx, sy);
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   // Translate to the problem's window. MakeCartesian2D always starts at the
   // origin, and the exact solutions are written in absolute coordinates.
   if (pars.x0 != 0. || pars.y0 != 0.)
   {
      mesh.EnsureNodes();
      GridFunction *nodes = mesh.GetNodes();
      const int nn = nodes->Size() / 2;
      for (int i = 0; i < nn; i++)
      {
         (*nodes)(i)      += pars.x0;
         (*nodes)(nn + i) += pars.y0;
      }
   }

   const int dim = mesh.Dimension();
   pars.dim = dim;
   const int neq = dim + 1;

   // Boundary attributes of MakeCartesian2D: 1 = y_min, 2 = x_max,
   // 3 = y_max, 4 = x_min.
   enum { BDR_BOTTOM = 1, BDR_RIGHT = 2, BDR_TOP = 3, BDR_LEFT = 4 };

   // 3. Finite element spaces.
   //
   //    All three are byNODES, and that is a requirement, not a preference:
   //    HyperbolicFormIntegrator reinterprets the element dof vector as a
   //    DenseMatrix(data, ndof, neq), which is column-major, so it reads the
   //    equation index as the *outer* one. MixedConductionNLFIntegrator and
   //    DarcyHybridization::ProjectSolution assume the same.

   L2_FECollection q_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   DG_Interface_FECollection t_coll(order, dim);

   FiniteElementSpace fes_q(&mesh, &q_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_u(&mesh, &u_coll, neq,       Ordering::byNODES);
   FiniteElementSpace fes_t(&mesh, &t_coll, neq,       Ordering::byNODES);

   // Scalar and vector views used only for error norms and visualisation.
   FiniteElementSpace fes_s(&mesh, &u_coll, 1);
   FiniteElementSpace fes_v(&mesh, &u_coll, dim, Ordering::byNODES);

   DarcyForm darcy(&fes_q, &fes_u);

   cout << "dim(q) = " << fes_q.GetVSize()
        << ", dim(u) = " << fes_u.GetVSize()
        << ", dim(t) = " << fes_t.GetVSize()
        << ", neq = " << neq << endl;
   cout << "nu = " << pars.nu << ", beta = " << pars.beta
        << ", U = " << pars.U << endl;

   // 4. Coefficients.

   ConstantCoefficient zero_coeff(0.);
   ConstantCoefficient one_coeff(1.);
   ConstantCoefficient inu_coeff(1. / pars.nu);
   ConstantCoefficient nu_coeff(pars.nu);

   auto state_fun = [&pars](const Vector & x, Vector & u) { ExactState(pars, x, u); };
   auto vel_fun   = [&pars](const Vector & x, Vector & v) { ExactVelocity(pars, x, v); };
   auto flux_fun  = [&pars](const Vector & x, Vector & q) { ExactFlux(pars, x, q); };
   auto pres_fun  = [&pars](const Vector & x) { return ExactPressure(pars, x); };

   VectorFunctionCoefficient state_coeff(neq, state_fun);
   VectorFunctionCoefficient vel_coeff(dim, vel_fun);
   VectorFunctionCoefficient flux_coeff(neq * dim, flux_fun);
   FunctionCoefficient pres_coeff(pres_fun);

   // 5. The flux mass, `(nu^-1 q, v)`.
   //
   //    Equation 0 is the pressure row. The pressure has no gradient variable
   //    in this formulation -- the HDG-INS literature carries only grad v --
   //    so its row is given an identity mass and a *zero* divergence and
   //    constraint block below, which makes q_0 identically zero and costs
   //    only the dofs it occupies. Writing it this way rather than shrinking
   //    the flux space keeps every block square and lets
   //    VectorBlockDiagonalIntegrator do the replication; a null entry there
   //    would shrink the element matrix instead of zeroing it
   //    (bilininteg.hpp, AssembleMat), which the hybridization's size
   //    assertions would then reject.

   {
      std::vector<BilinearFormIntegrator *> mass(neq);
      mass[0] = new VectorMassIntegrator(pgrad ? inu_coeff : one_coeff);
      for (int e = 1; e < neq; e++)
      {
         mass[e] = new VectorMassIntegrator(inu_coeff);
      }
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(mass));
   }

   // 6. The flux divergence, `(div q, w)`, zero on the pressure row.

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   {
      std::vector<BilinearFormIntegrator *> div(neq);
      div[0] = pgrad ? new VectorDivergenceIntegrator()
               : new VectorDivergenceIntegrator(zero_coeff);
      for (int e = 1; e < neq; e++)
      {
         div[e] = new VectorDivergenceIntegrator();
      }
      B->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(div));

      // The hybridized path never *evaluates* B's face integrators -- only
      // AssembleDivLDGFaces(), which the reduction branch calls, ever does.
      // What a boundary face integrator on B does supply is a **marker**:
      // DarcyForm::Assemble() reads B->GetBFBFI_Marker() and installs the
      // hybridization's constraint integrator on exactly those attributes.
      //
      // Without it the constraint <lambda, v.n> is assembled on interior
      // faces only, and then row 1 of every element touching the boundary
      // fails the identity -(u, div v)_K + <lambda, v.n>_dK = 0 that a
      // constant state has to satisfy. Measured on the uniform-flow problem:
      // the potential comes back exact to 1e-16 and the flux does not, with
      // an error no scaling of the interior constraint can remove (0.40, 0.34,
      // 0.297, 0.30, 0.60 as that scaling runs 0.25 to 2). With the marker in
      // place it is 1e-16 like the rest.
      //
      // The integrator handed over is never called, so its coefficients are
      // immaterial; it has to exist and to carry the right attributes.
      {
         Array<int> bdr_all(mesh.bdr_attributes.Max());
         bdr_all = 1;
         B->AddBdrFaceIntegrator(
            new VectorBlockDiagonalIntegrator(
               neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))),
            bdr_all);
      }
   }

   // 6b. Which boundary attributes prescribe what.
   //
   //     This lives here, ahead of the forms, for one mechanical reason:
   //     DarcyForm::EnableHybridization() reads the potential mass form's
   //     face integrators AT CALL TIME and hands them to the hybridization,
   //     so a boundary face integrator added after it is never seen. The
   //     boundary numerical flux below is chosen per attribute from these
   //     markers, so the markers have to exist first. Registering it after
   //     EnableHybridization() instead cost a debugging cycle: the residual
   //     loses its whole boundary inviscid term and every problem diverges to
   //     inf, including -bcfull, where the change should have been a no-op.
   //
   //     The essential trace dof list built from the same markers stays in
   //     step 9, where fes_t exists.

   Array<int> bdr_vel(mesh.bdr_attributes.Max());   // velocity prescribed
   Array<int> bdr_pres(mesh.bdr_attributes.Max());  // pressure prescribed
   bdr_vel = 0; bdr_pres = 0;

   switch (problem)
   {
      case Problem::PlanePoiseuille:
         // No-slip walls top and bottom, velocity profile in at the left,
         // pressure held at the right. The velocity is left free at the
         // outlet and the pressure free everywhere else; that is what makes
         // the outlet an outflow rather than a second inlet.
         bdr_vel[BDR_BOTTOM - 1] = 1;
         bdr_vel[BDR_TOP    - 1] = 1;
         bdr_vel[BDR_LEFT   - 1] = 1;
         bdr_pres[BDR_RIGHT - 1] = 1;
         break;
      case Problem::Kovasznay:
      case Problem::UniformFlow:
      case Problem::Couette:
         // The velocity is given on the whole boundary; the pressure then
         // needs one datum to fix its gauge, and the outflow edge carries it.
         bdr_vel = 1;
         bdr_pres[BDR_RIGHT - 1] = 1;
         break;
   }

   if (bc_full) { bdr_vel = 1; bdr_pres = 1; }

   //    A wholly natural attribute, for the measurement -bcnat documents.
   if (bc_nat > 0)
   {
      MFEM_VERIFY(bc_nat <= mesh.bdr_attributes.Max(), "-bcnat out of range");
      bdr_vel[bc_nat - 1] = 0;
      bdr_pres[bc_nat - 1] = 0;
   }

   // 7. The potential mass, carrying both halves of the stabilization and the
   //    whole of the inviscid flux. Everything goes on the *nonlinear* form:
   //    DarcyForm keeps one potential mass form, and the convective term is
   //    nonlinear, so the diffusive stabilization has to join it there.
   //    HDGDiffusionIntegrator is a BilinearFormIntegrator and therefore also
   //    a NonlinearFormIntegrator, so that is legal and is what convdiff does
   //    in its own nonlinear branch.

   NonlinearForm *Mtnl = darcy.GetPotentialMassNonlinearForm();

   //    (a) the diffusive half, s_diff = td*nu/h, per NPC-1 section 3.6.
   //        Zero on the pressure row: there is no diffusion in the continuity
   //        equation, and its face stabilization comes entirely from S below.
   {
      std::vector<BilinearFormIntegrator *> stab_i(neq), stab_b(neq);
      stab_i[0] = new HDGDiffusionIntegrator(zero_coeff, td);
      stab_b[0] = new HDGDiffusionIntegrator(zero_coeff, td);
      for (int e = 1; e < neq; e++)
      {
         stab_i[e] = new HDGDiffusionIntegrator(nu_coeff, td);
         stab_b[e] = new HDGDiffusionIntegrator(nu_coeff, td);
      }
      Mtnl->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(stab_i));
      Mtnl->AddBdrFaceIntegrator(new VectorBlockDiagonalIntegrator(stab_b));
   }

   //    (b) the inviscid flux and its Lax-Friedrichs stabilization.
   ArtificialCompressibilityFlux ac_flux(dim, pars.beta, stokes);
   unique_ptr<NumericalFlux> num_flux;
   if (tau_const >= 0.)
   {
      num_flux = make_unique<HDGFlux>(ac_flux, HDGFlux::HDGScheme::HDG_1,
                                      tau_const);
   }
   else
   {
      auto lf = make_unique<HDGLaxFriedrichsFlux>(ac_flux);
      lf->SetFrozenStabilizationJacobian(frozen_stab);
      num_flux = std::move(lf);
   }

   //    The sign is convdiff's: the Darcy residual convention wants -1 here,
   //    which HyperbolicFormIntegrator multiplies through every contribution.
   Mtnl->AddDomainIntegrator(new HyperbolicFormIntegrator(*num_flux, 0, hsign));
   Mtnl->AddInteriorFaceIntegrator(new HyperbolicFormIntegrator(*num_flux, 0,
                                                                hsign));
   //    The BOUNDARY one is split in two, and this is the outflow repair.
   //
   //    An attribute whose VELOCITY trace is free is where the row quadratic
   //    in u_hat lives, and it is the only place the linearisation is wanted:
   //    the pressure row is beta v_hat.n, already linear, so an attribute that
   //    frees only the pressure needs nothing. Under -bcfull nothing is free,
   //    bdr_lin is empty, and this reduces to the single unmarked registration
   //    it replaced -- a no-op by construction rather than by a branch, the
   //    same way the prescribed-flux datum is.
   Array<int> bdr_lin(mesh.bdr_attributes.Max());
   Array<int> bdr_ord(mesh.bdr_attributes.Max());
   int n_lin = 0;
   for (int a = 0; a < bdr_lin.Size(); a++)
   {
      bdr_lin[a] = (bc_lin && !bdr_vel[a]) ? 1 : 0;
      bdr_ord[a] = 1 - bdr_lin[a];
      n_lin += bdr_lin[a];
   }
   HDGLinearisedBdrFlux lin_flux(ac_flux, state_coeff, tau_const);
   if (n_lin == 0)
   {
      Mtnl->AddBdrFaceIntegrator(new HyperbolicFormIntegrator(*num_flux, 0,
                                                              hsign));
   }
   else
   {
      Mtnl->AddBdrFaceIntegrator(
         new HyperbolicFormIntegrator(*num_flux, 0, hsign), bdr_ord);
      Mtnl->AddBdrFaceIntegrator(
         new HyperbolicFormIntegrator(lin_flux, 0, hsign), bdr_lin);
   }
   cout << "linearised boundary flux on " << n_lin << " attribute(s)" << endl;

   // 8. Hybridization. The constraint is `<[q.n], mu>`, again zero on the
   //    pressure row so that q_0 stays uncoupled; NormalTraceJumpIntegrator's
   //    sign argument is the cheapest way to get a correctly shaped zero.

   Array<int> ess_flux_tdofs;   // empty: the DG flux has no essential dofs
   {
      std::vector<BilinearFormIntegrator *> constr(neq);
      constr[0] = new NormalTraceJumpIntegrator(pgrad ? 1. : 0.);
      for (int e = 1; e < neq; e++)
      {
         constr[e] = new NormalTraceJumpIntegrator();
      }
      darcy.EnableHybridization(&fes_t,
                                new VectorBlockDiagonalIntegrator(constr),
                                ess_flux_tdofs);
   }

   DarcyHybridization *hyb = darcy.GetHybridization();
   if (gradient_mode >= 0)
   {
      hyb->SetGradientMode((gradient_mode == 2)
                           ? DarcyHybridization::GradientMode::MatrixFree
                           : DarcyHybridization::GradientMode::Assembled);
   }
   if (threaded_assembly)
   {
      hyb->SetAssemblyMode(DarcyHybridization::AssemblyMode::Threaded);
   }
   if (batched_factor)
   {
      hyb->SetLocalFactorMode(DarcyHybridization::LocalFactorMode::Batched);
   }

   // 9. Boundary conditions, imposed essentially on the trace, per component.
   //
   //    The weak route the reference uses -- its boundary flux vector B^ of
   //    Eq. (8) -- is not available here. It would go through
   //    BdrHyperbolicDirichletIntegrator, and that integrator reads its
   //    prescribed state only when bit 0 of `type` is set, which
   //    DarcyHybridization never sets on a boundary face: every `type |= 1`
   //    site sits inside an interior-face branch. Registered on the hybridized
   //    form it degrades to an ordinary HyperbolicFormIntegrator with no
   //    warning and no abort -- the interior state is used and the boundary
   //    datum is silently dropped. So the datum is put on the trace instead.
   //
   //    DarcyHybridization::SetEssentialBC marks *every* component of a marked
   //    attribute, which is wrong for flow: a wall fixes the velocity and says
   //    nothing about the pressure. The per-component lists are built here and
   //    handed to SetEssentialTrueDofs, which is purely index-based and so is
   //    component-blind in the way that is wanted.
   //
   //    **A trace component left non-essential is not a free boundary.** It
   //    keeps the constraint row <(F^ + q^).n, mu> = 0, and on a boundary face
   //    that row has only one side, so nothing cancels it: it imposes ZERO
   //    numerical flux. -bcphys used to stop there, and the result was a solve
   //    that converged happily to 3e-13 and was wrong by more than 100% at
   //    every order, on all four problems -- 2.84 / 2.20 / 1.01 relative in
   //    q / p / v on plane Poiseuille, 0.28 / 2.47 / 1.07 on uniform flow,
   //    6.31 / 1.25 / 0.98 on Couette, 4.34 / 10.7 / 0.96 on Kovasznay --
   //    while -bcfull on the same problems was at round-off. `-no-bcflux`
   //    restores exactly that state and is kept as the control.
   //
   //    What it needs is the other half of the mixed condition: the PRESCRIBED
   //    NUMERICAL FLUX on the rows the trace does not fix. That is step 11's
   //    HDGPrescribedFluxLFIntegrator, a linear form on the trace carrying
   //    <(F + G).n, mu> from the exact solution -- the same shape as the
   //    Neumann datum convdiff puts on its trace through
   //    BoundaryNormalLFIntegrator. Per boundary face and per equation the
   //    hybridized system offers exactly one of the two, so the physical set
   //    is well posed: dim+1 conditions per face either way.
   //
   //    **Why not the reference's characteristic condition**, B^ = A+_n(u-u^)
   //    - A-_n(u_inf-u^), which was the other candidate. It is not a datum on
   //    this row; it REPLACES the row, and the row is a sum of contributions
   //    from two different forms -- the inviscid part from the potential mass
   //    nonlinear form and the viscous q^.n from the constraint block C, which
   //    DarcyForm installs from B's boundary marker. A boundary integrator on
   //    the potential mass form can add to that row but cannot cancel C's
   //    contribution to it, because C is in q and that form is not. So the
   //    characteristic condition is not expressible here without new machinery
   //    in DarcyHybridization, while the datum is a linear form the interface
   //    already has. That, and not its accuracy, is the reason for the choice.
   //
   //    **What the datum costs, and it is a FAMILY of roots rather than a
   //    second one.** This entry used to say "the discrete problem has a
   //    second root"; a sweep of the initial state says otherwise. On the
   //    outlet the pressure is essential and the momentum rows carry the
   //    datum, so per quadrature point the row reads
   //
   //        v^_x^2 + p^ + q.n + S(u - u^) = b,      p^ prescribed,
   //
   //    which fixes v^.n only up to SIGN. Every free outlet velocity dof
   //    therefore enters its own row quadratically and the count of roots is
   //    combinatorial, not two. Measured at order 2 on 8x8, from fifteen
   //    uniform-stream initial states: FOUR distinct roots, each converged
   //    quadratically to ||r||/||r_0|| of 1e-16 to 1e-15, at 1.31, 3.30 and
   //    5.63 relative in q besides the true one, and six divergences. The
   //    cold start (uniform-stream 0) finds the 5.63 one.
   //
   //    **The attribution, and it was WRONG in the interesting half until
   //    -bclin was built to test it.** v (x) v is quadratic in the volume as
   //    well, so the family could have been the equations' rather than the
   //    boundary condition's. This entry used to call two controls decisive:
   //    -bcfull, every trace component essential and hence no free row,
   //    converges to the SAME root from all fifteen starts; and -stokes,
   //    linear in the state, converges in one step from any of them. Both are
   //    true. **Neither separates "the row is quadratic in u_hat" from "the
   //    row is free at all"** -- -bcfull removes the free dofs and -stokes
   //    removes the interior nonlinearity, and the entry read the pair as
   //    convicting the row's DEGREE.
   //
   //    -bclin is the experiment that separates them: it makes the outflow row
   //    exactly linear in u_hat -- F(w).n + A_n(w)(u_hat - w) + S(w)(u - u_hat)
   //    about the prescribed exterior state w -- and changes nothing else. It
   //    is consistent, the exact solution staying a root to 4.3e-16. If the
   //    row's degree were the mechanism the family would collapse. Measured at
   //    order 2 on 8x8, sweeping -uinit over [-1.5, 3]:
   //
   //        -no-bclin   spurious roots 5.63, 1.31, 3.30    8 divergences
   //        -bclin      spurious roots 1.18, 2.77         10 divergences
   //
   //    **The family moves and does not go.** So the row's degree is not the
   //    mechanism. A Reynolds sweep says what is: at Re = 1 and Re = 5 BOTH
   //    modes reach the true root from every start (1e-6 to 1e-12), and the
   //    spurious roots appear only at Re = 40, in both. **It is the interior
   //    convective nonlinearity, admitted by a free outlet trace; the boundary
   //    row's own quadratic term is incidental to it.**
   //
   //    That leaves the cure where the measurements already put it: -cont,
   //    which reaches the physical root at 1.6e-14. And it retires the claim
   //    below that the reference's characteristic condition is the only real
   //    one -- that is a linearisation too, and -bclin is the same idea inside
   //    what the interface allows.
   //    (The beta sweep this entry used to rest on is weak evidence by
   //    comparison: 5.627, 5.612, 5.357 over a sixteenfold range.)
   //
   //    **They are not second solutions of the continuous problem.** Refining
   //    does not converge them: the outlet trace velocity peaks at 2.62, 2.60,
   //    3.11 and 3.88 as nx runs 4, 8, 16, 32 -- growing without bound, on a
   //    profile whose true peak is 1 -- while the relative error goes 4.33,
   //    5.63, 6.38, 6.60 and the boundary jump |u-u^|^2 falls only like h.
   //    They are artefacts of a condition that fixes a sign nowhere.
   //
   //    **Physically sensible initial data does not protect.** A uniform
   //    stream at the channel's mean velocity 2/3 lands on the 1.31 root at
   //    order 2. Only -xinit and -cont reach the true one, which is a root to
   //    5.1e-16 -- the datum is right, the condition is non-unique.
   //
   //    **Dropping the essential outlet pressure is NOT the cure**, though it
   //    looks like it should be: with the whole outlet natural the continuity
   //    row's datum beta v^.n = beta v_ex.n would fix v^.n linearly and the
   //    momentum row would then fix p^. Measured, -bcnat 2 diverges from every
   //    start at orders 1, 2 and 3 and on every mesh, and the giveaway is that
   //    the LINEAR Stokes problem then converges to a wrong answer -- v off by
   //    1.9e-3 where the exact solution is in the space -- with the exact
   //    solution still a root to 5.1e-16. A linear system with two solutions
   //    is a singular one: the outlet's essential pressure is what makes this
   //    problem nonsingular, and it cannot simply be given up.
   //    (The characteristic condition is no longer "the only real cure": see
   //    the attribution above, which -bclin corrected.)
   //
   //    So -bcphys wants -cont on a convective problem, exactly as -bcfull
   //    does on Kovasznay.
   //
   //    **And the order-1 entry that used to sit in the roadmap was wrong in
   //    both halves.** It said "-bcphys at order 1 diverges, cold and under
   //    -cont alike", because the exact solution is not in the space there so
   //    the continuation has nothing to hand on. Cold at order 1 does not
   //    diverge -- it converges, to the 3.89 member of the family above. And
   //    the -cont divergence is not about order 1 and not about the exact
   //    solution: at order 1 with -cont, nx = 4, 6, 10, 12, 16 and 32 all
   //    reach the physical root, with q and v errors matching -bcfull's to
   //    within 10-35% and converging at the expected O(h^2); only nx = 8
   //    fails. The exact solution is equally unrepresentable at nx = 16, and
   //    that case takes six Newton steps. At nx = 8 every Reynolds number in
   //    20, 25, 28, 30, 32, 34, 36, 38 and 45 converges and only 40 does not.
   //
   //    The failure is CHAOTIC, and the cheapest proof is a parameter that
   //    cannot change any root: beta. At (order 1, nx 8, Re 40), beta = 0.99,
   //    0.999, 1.01 and 2 converge while 0.9, 1, 1.001 and 1.1 diverge. So do
   //    ny = 7 and 9 against ny = 8, and Re = 39.9 against 40.1. That is a
   //    Newton path wandering in a landscape with many roots, not a property
   //    of the order-1 space -- the same finding as the family above, seen
   //    from the solver's side. KINSOL's line search stops the divergence
   //    without selecting the physical root: -ls -cont on that case converges
   //    to a root at 0.175 / 0.0528 / 0.0165, stable over rtol 1e-10 to
   //    1e-14 and about three times -bcfull's error there.

   //    The markers themselves are computed BEFORE the forms, because the
   //    boundary numerical flux registration needs them; see step 6b. What
   //    stays here is the per-component essential list, which needs fes_t.

   Array<int> ess_tdofs;
   {
      Array<int> list;
      for (int c = 1; c < neq; c++)   // velocity components
      {
         fes_t.GetEssentialTrueDofs(bdr_vel, list, c);
         ess_tdofs.Append(list);
      }
      fes_t.GetEssentialTrueDofs(bdr_pres, list, 0);   // pressure component
      ess_tdofs.Append(list);
      ess_tdofs.Sort();
      ess_tdofs.Unique();
   }
   hyb->SetEssentialTrueDofs(ess_tdofs);
   cout << "essential trace tdofs: " << ess_tdofs.Size()
        << " of " << fes_t.GetTrueVSize() << endl;

   darcy.Assemble();

   // 10. State vectors. The trace block carries the boundary data.

   Array<int> offsets = DarcyOperator::ConstructOffsets(darcy);
   BlockVector x(offsets), rhs(offsets);
   x = 0.;
   rhs = 0.;

   GridFunction q_h, u_h, tr_h;
   q_h.MakeRef(&fes_q, x.GetBlock(0), 0);
   u_h.MakeRef(&fes_u, x.GetBlock(1), 0);
   tr_h.MakeRef(&fes_t, x.GetBlock(2), 0);

   //     Project the exact state onto the whole boundary trace. Only the
   //     components in ess_tdofs are enforced; the rest are an initial guess
   //     the solver overwrites. Both problems here have a known exact
   //     solution, so this is the honest thing to do -- a problem without one
   //     would need the per-component projection instead.
   Array<int> all_bdr(mesh.bdr_attributes.Max());
   all_bdr = 1;
   tr_h.ProjectBdrCoefficient(state_coeff, all_bdr);

   if (u_init != 0.)
   {
      // A uniform stream, pressure zero. Set on the fields and on the whole
      // trace, so the start is a genuine state and not a half-set one; the
      // essential components are then overwritten below by the projection
      // that follows, which is what makes only the FREE ones vary.
      VectorFunctionCoefficient stream(neq, [&](const Vector &, Vector &v)
      {
         v = 0.;
         v(1) = u_init;
      });
      // The FIELD only. The trace space is face-based, so
      // GridFunction::ProjectCoefficient() is not available on it -- it
      // segfaults rather than refusing -- and the boundary trace is already
      // the exact datum from the projection above. So what this sweeps is the
      // interior state the Newton path starts from, which is exactly what the
      // root count is a function of.
      u_h.ProjectCoefficient(stream);
   }

   if (exact_init)
   {
      q_h.ProjectCoefficient(flux_coeff);
      u_h.ProjectCoefficient(state_coeff);
      // The trace space is face-based, so an element projection is not
      // available; DarcyHybridization averages the two element traces, which
      // for a solution continuous across the face is that solution's trace.
      BlockVector sol2(x.GetData(), darcy.GetOffsets());
      hyb->ProjectSolution(sol2, x.GetBlock(2));
   }

   // 11. Right-hand sides. Both problems have zero source, so all three forms
   //     are empty -- but they still have to exist and, crucially, they have
   //     to be Update()d onto the blocks of `rhs` rather than allocated on
   //     their own. DarcyOperator does `rhs.Update(g->GetData(), offsets)`,
   //     which reads the *first* form's data pointer as the base of the whole
   //     block vector; a form owning its own storage segfaults there.
   //
   //     The coefficient list is empty for the same kind of reason: the
   //     operator walks it calling SetTime() without a null check.

   LinearForm gform, fform, hform;
   gform.Update(&fes_q, rhs.GetBlock(0), 0);
   fform.Update(&fes_u, rhs.GetBlock(1), 0);
   hform.Update(&fes_t, rhs.GetBlock(2), 0);

   //     The trace form is the one that is not empty: it carries the Neumann
   //     half of the boundary conditions, the prescribed numerical flux
   //     <(F + G).n, mu> on the components the trace does not fix. See the
   //     boundary-condition step above for what it repairs and how the sign
   //     was measured. The datum is assembled on the WHOLE boundary and then
   //     zeroed on the essential trace dofs, which is why -bcfull -- where
   //     every component of every attribute is essential -- is untouched by
   //     construction rather than by a branch.
   if (bc_flux)
   {
      hform.AddBoundaryIntegrator(
         new HDGPrescribedFluxLFIntegrator(ac_flux, state_coeff, &flux_coeff,
                                           bc_sf, bc_sg, bc_io));
   }

   // 12. Solve, by NPC: Newton on the FULL (q, u, u_hat) system with the
   //     Jacobian solved by hybridized elimination. See
   //     DarcyHybridization::NPCResidual().
   //
   //     This miniapp used to go through DarcyOperator and a single
   //     backward-Euler step, which drives an unknown that is the TRACE alone
   //     and then rebuilds the fields from it with RecoverFEMSolution(). NPC
   //     does not want that back-substitution: q and u are Newton state, so
   //     they are already in `x` when the solve returns, and the whole
   //     pseudo-time apparatus for a steady problem goes with it.
   //
   //     What that buys, beyond being the method NPC actually is: the
   //     convergence test is on the full residual rather than on the trace
   //     alone, and a line search scales the fields and the trace together
   //     because they are one vector. Every local operation is one linear
   //     solve against one factorisation, and GetNumLocalNLIterations() stays
   //     at zero, which is the acceptance signal that it really is NPC.

   darcy.Finalize();

   //     The load is the (flux, potential) pair; the trace form rides in the
   //     right-hand side of Newton's own Mult(), because the trace row of the
   //     residual carries no load of its own.
   BlockVector load(rhs, darcy.GetOffsets());
   DarcyNPCOperator npc(*hyb, offsets, load);

   BlockVector b(offsets);
   b = 0.;

   //     The trace load is re-assembled before every solve rather than once,
   //     because it reads the FLUX FUNCTION, and -cont flips that function
   //     between the two solves. Assembling it once gave the Stokes stage the
   //     full Navier-Stokes datum -- v (x) v included, on a solve that does
   //     not have that term -- so the continuation landed on the wrong answer
   //     and handed it to the second solve. Measured: -cont came back at 5.63
   //     relative in the flux, the same wrong root a cold start finds, while
   //     -stokes alone (whose datum happens to match) was exact.
   auto assemble_trace_load = [&]()
   {
      hform.Assemble();
      rhs.GetBlock(2).SetSubVector(ess_tdofs, 0.);
      b.GetBlock(2) = rhs.GetBlock(2);
   };
   assemble_trace_load();

   //     The trace load goes into the OPERATOR rather than into the solver's
   //     right-hand side, because KINSolver::Mult() ignores the latter -- see
   //     LoadedOperator above, and the measurement that established it.
   LoadedOperator npc_b(npc, b);
   const Vector no_rhs;   // size 0, so NewtonSolver's have_b is false

   //     The residual at the exact state, block by block. One norm over all
   //     three cannot say which row is wrong, and the boundary datum lives in
   //     the trace row alone -- so this is the instrument the -bcphys repair
   //     was measured with, and the -bcsf/-bcsg sweep reads out of it.
   if (exact_init)
   {
      Vector r0(x.Size());
      npc_b.Mult(x, r0);
      const BlockVector r0b(r0, offsets);
      cout << "residual at the exact state: ||r_q|| = "
           << r0b.GetBlock(0).Norml2() << ", ||r_u|| = "
           << r0b.GetBlock(1).Norml2() << ", ||r_tr|| = "
           << r0b.GetBlock(2).Norml2() << endl;
   }

   //     The reduced trace system, by -gm. Matrix-free has no matrix to
   //     precondition with, so it gets a plain Krylov method.
   unique_ptr<Solver> trace_solver;
   unique_ptr<Solver> trace_prec;
   if (gradient_mode == 2)
   {
      auto gmres = make_unique<GMRESSolver>();
      gmres->SetKDim(200);
      gmres->SetMaxIter(2000);
      gmres->SetRelTol(1e-12);
      gmres->SetAbsTol(0.);
      gmres->SetPrintLevel(-1);
      trace_solver = std::move(gmres);
   }
   else
   {
#ifdef MFEM_USE_SUITESPARSE
      if (gradient_mode <= 0)
      {
         trace_solver = make_unique<UMFPackSolver>();
      }
      else
#endif
      {
         trace_prec = make_unique<GSSmoother>();
         auto gmres = make_unique<GMRESSolver>();
         gmres->SetKDim(200);
         gmres->SetMaxIter(2000);
         gmres->SetRelTol(1e-12);
         gmres->SetAbsTol(0.);
         gmres->SetPrintLevel(-1);
         gmres->SetPreconditioner(*trace_prec);
         trace_solver = std::move(gmres);
      }
   }
   DarcyNPCSolver lin(*trace_solver);

   //     NPC needs a Jacobian solve, so a gradient-free outer solver has
   //     nothing to do with the elimination. LBFGS and LBB never call
   //     GetGradient(), so they are accepted by the operator and then diverge
   //     -- measured, to NaN, on a case Newton solves in four steps -- and a
   //     matrix-free Krylov method over the Jacobian is refused outright,
   //     because the handle is solve-only. Refuse both here rather than let a
   //     flag combination fail obscurely.
   MFEM_VERIFY(solver_type == (int) DarcyOperator::SolverType::Newton ||
               solver_type == (int) DarcyOperator::SolverType::KINSol,
               "This miniapp is driven by NPC, which needs a Jacobian solve. "
               "Use -nls 3 (Newton) or -nls 4 (KINSol); LBFGS and LBB cannot "
               "drive it.");

   // Globalisation comes from KINSOL, not from anything written here. This
   // miniapp used to carry a hand-rolled backtracking Newton, and it was a
   // worse implementation of exactly what KIN_LINESEARCH already does:
   // Dennis & Schnabel with the sufficient-decrease AND curvature conditions,
   // a minimum-step test that reports non-convergence, and a maximum-step
   // constraint. Ours accepted on `Norm(rt) < n0` with no sufficient-decrease
   // constant, and since the Newton direction is always a descent direction
   // for an l2 merit, ANY small enough step passed -- measured, alpha = 1.2e-4
   // still "succeeding" for a 1e-4 relative improvement. So where a full step
   // was rejected it did not fail, it crept. meq copied it and lost an
   // afternoon to that. Section 6 of doc/HDG-ORDERING-API.md has the whole
   // account, including why neither a correct line search nor a block-weighted
   // merit rescues the case that motivated it.
   unique_ptr<NewtonSolver> newton;
   KINSolver *kin = nullptr;
   if (line_search || solver_type == (int) DarcyOperator::SolverType::KINSol)
   {
      kin = new KINSolver(line_search ? KIN_LINESEARCH : KIN_NONE);
      newton.reset(kin);
   }
   else
   {
      newton.reset(new NewtonSolver());
   }
   newton->SetOperator(npc_b);
   newton->SetSolver(lin);
   newton->SetRelTol((newton_rtol > 0.) ? newton_rtol : 1e-6);
   newton->SetAbsTol(newton_atol);
   newton->SetMaxIter(newton_iters);
   newton->SetPrintLevel(1);

   // A true Newton rather than a lagged-Jacobian one; without it KINSOL reuses
   // a setup and the comparison against -nls 3 is not like for like.
   //
   // **It has to come after SetOperator(), and it used to come before.**
   // KINSolver::SetMaxSetupCalls() writes straight into `sundials_mem`, which
   // SetOperator() is what creates, so called first it is a no-op guarded only
   // by an MFEM_ASSERT -- compiled out of this build. The whole visible symptom
   // was one line of SUNDIALS' own stderr, `kinsol_mem = NULL illegal`, and
   // -ls then ran with KINSOL's default of a Jacobian every ten steps.
   // examples/sundials/ex10.cpp has the order right; both this miniapp and
   // pnavierstokes.cpp had it wrong. Note the asymmetry in the library:
   // EnableAndersonAcc() checks for the null and defers, SetMaxSetupCalls()
   // does not.
   if (kin) { kin->SetMaxSetupCalls(1); }

   chrono.Clear();
   chrono.Start();
   {
      // Two solves for Stokes continuation, the second starting from the
      // first's answer -- which under NPC is simply `x`, all three blocks of
      // it, with nothing to recover in between.
      if (continuation && !stokes)
      {
         ac_flux.SetStokes(true);
         assemble_trace_load();
         newton->Mult(no_rhs, x);
         ac_flux.SetStokes(false);
         assemble_trace_load();
         cout << "--- Stokes continuation done; continuing onto "
              "Navier-Stokes ---" << endl;
      }
      newton->Mult(no_rhs, x);
   }
   chrono.Stop();

   cout << "local nonlinear iterations: " << hyb->GetNumLocalNLIterations()
        << " (NPC runs none; anything else means the ordering did not change)"
        << endl;

   // 13. Errors.

   const int order_quad = max(2, 2 * order + 3);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   //     byNODES means component c of u_h occupies [c*nd, (c+1)*nd), so the
   //     pressure is a scalar field at offset 0 and the velocity a dim-wide
   //     one at offset nd. No copy is needed, and none is made.
   const int nd = fes_s.GetNDofs();
   GridFunction p_h(&fes_s, u_h.GetData());
   GridFunction v_h(&fes_v, u_h.GetData() + nd);

   const real_t err_p  = p_h.ComputeL2Error(pres_coeff, irs);
   const real_t norm_p = ComputeLpNorm(2., pres_coeff, mesh, irs);
   const real_t err_v  = v_h.ComputeL2Error(vel_coeff, irs);
   const real_t norm_v = ComputeLpNorm(2., vel_coeff, mesh, irs);
   const real_t err_q  = q_h.ComputeL2Error(flux_coeff, irs);
   const real_t norm_q = ComputeLpNorm(2., flux_coeff, mesh, irs);

   //     Per-equation flux errors. The flux is the block the local solve gets
   //     wrong when row 1 is miswired, and a single lumped norm cannot say
   //     which equation is at fault -- the pressure row, whose coupling is
   //     deliberately zeroed, or the momentum rows.
   {
      const int ndq = fes_q.GetNDofs();
      FiniteElementSpace fes_qe(&mesh, &q_coll, dim, Ordering::byNODES);
      for (int e = 0; e < neq; e++)
      {
         GridFunction qe(&fes_qe, q_h.GetData() + e * dim * ndq);
         auto qe_fun = [&pars, e, dim](const Vector & x, Vector & v)
         {
            Vector all;
            ExactFlux(pars, x, all);
            v.SetSize(dim);
            for (int d = 0; d < dim; d++) { v(d) = all(e * dim + d); }
         };
         VectorFunctionCoefficient qec(dim, qe_fun);
         cout << "   flux eq " << e << ": || q_h - q_ex || = "
              << qe.ComputeL2Error(qec, irs)
              << ",  || q_ex || = " << ComputeLpNorm(2., qec, mesh, irs) << "\n";
      }
   }

   cout << "|| q_h - q_ex || / || q_ex || = "
        << ((norm_q > 0.) ? (err_q / norm_q) : err_q) << "\n";
   cout << "|| p_h - p_ex || / || p_ex || = "
        << ((norm_p > 0.) ? (err_p / norm_p) : err_p) << "\n";
   cout << "|| v_h - v_ex || / || v_ex || = "
        << ((norm_v > 0.) ? (err_v / norm_v) : err_v) << "\n";

   // 13b. Grade the run, if asked. Returning non-zero is what lets the
   // existing navierstokes-test-seq target check anything at all: mfem-test
   // (config/test.mk) reads the exit status and then DELETES the output, so a
   // number printed above reaches no suite. Deliberately a plain `return 1`
   // rather than MFEM_ABORT -- without MFEM_USE_EXCEPTIONS that is a SIGABRT,
   // and a clean exit with our own message is easier to read in a test log.
   int failures = 0;
   if (check_tol > 0.)
   {
      const real_t rel_q = (norm_q > 0.) ? (err_q / norm_q) : err_q;
      const real_t rel_p = (norm_p > 0.) ? (err_p / norm_p) : err_p;
      const real_t rel_v = (norm_v > 0.) ? (err_v / norm_v) : err_v;
      const char *names[3] = { "q", "p", "v" };
      const real_t vals[3] = { rel_q, rel_p, rel_v };
      for (int i = 0; i < 3; i++)
      {
         if (!(vals[i] <= check_tol))
         {
            cout << "CHECK FAILED: " << names[i] << " relative error "
                 << vals[i] << " exceeds " << check_tol << "\n";
            failures++;
         }
      }
      cout << (failures ? "CHECK FAILED" : "CHECK PASSED") << " at tolerance "
           << check_tol << "\n";
   }

   // 14. Visualisation.

   if (visualization)
   {
      char vishost[] = "localhost";
      const int visport = 19916;
      socketstream p_sock(vishost, visport), v_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << mesh << p_h
             << "window_title 'Pressure'" << endl;
      v_sock.precision(8);
      v_sock << "solution\n" << mesh << v_h
             << "window_title 'Velocity'\nkeys vvv" << endl;
   }

   return failures ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Extending to the compressible equations
// ---------------------------------------------------------------------------
//
// The structure above is the reference's, not an incompressible special case,
// and three edits move it to the compressible equations:
//
//  1. Swap ArtificialCompressibilityFlux for EulerFlux(dim, gamma). Both are
//     FluxFunction subclasses with analytic Jacobians; the state layout
//     already matches, the continuity variable first and the momentum ones
//     next, so nothing that indexes the state has to change. neq goes from
//     dim+1 to dim+2. IsothermalFlux(dim, c_s) is the intermediate step and is
//     the closer analogue of what is here: it is also neq = dim+1, with
//     p = c_s^2 rho playing the part beta v plays above.
//
//  2. Replace the per-equation constant viscosity of step 5/6 with a genuine
//     G(u,q). The viscous flux of the compressible equations is not diagonal
//     in the equations -- the stress couples the momentum rows through
//     div v, and the energy row sees both the stress and the heat flux -- so
//     the constant VectorBlockDiagonalIntegrator pair has to become a
//     MixedFluxFunction driven by MixedConductionNLFIntegrator on
//     DarcyForm::GetBlockNonlinearForm(). That integrator's element terms are
//     already generic in the number of equations; its *LDG* face terms are
//     not (nonlininteg_mixed.cpp:299 refuses more than one equation), but the
//     hybridized path never calls those.
//
//  3. Give the pressure row back its gradient. Steps 5, 6 and 8 zero equation
//     0's mass coupling, divergence and constraint because the incompressible
//     formulation has no grad p variable. The compressible equations do carry
//     grad rho, so those three zeros simply become the same integrators the
//     other equations get.
//
// What does *not* change: the trace space, the hybridization, the
// stabilization splitting, the boundary treatment, and the observation that
// the boundary datum has to reach the trace rather than the numerical flux.
//
// One thing that will need attention at step 2: MixedConductionNLFIntegrator's
// HDG face stabilization for more than one equation is `face_w * TauVar(e)`, a
// single constant per equation set once through SetVariableStabilization().
// That cannot express a stabilization that depends on the state or on the
// face normal -- which is the whole subject of the tau question above. The
// route taken here sidesteps it by carrying the convective stabilization on
// the NumericalFlux instead; a *viscous* stabilization that varied with
// direction would run straight into it.
