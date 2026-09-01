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
//                    Navier-Stokes miniapp - Parallel version
//
// Compile with: make pnavierstokes
//
// Sample runs:  mpirun -np 4 pnavierstokes -p 1 -nx 8 -ny 8 -o 2 -rtol 1e-12
//               mpirun -np 4 pnavierstokes -p 1 -nx 8 -ny 8 -o 2 -stokes
//               mpirun -np 4 pnavierstokes -p 3 -nx 6 -ny 4 -o 2 -rtol 1e-12
//               mpirun -np 4 pnavierstokes -p 4 -nx 6 -ny 4 -o 2 -rtol 1e-12
//               mpirun -np 4 pnavierstokes -p 2 -nx 24 -ny 16 -o 2 -re 40 -cont
//               mpirun -np 4 pnavierstokes -p 2 -nx 24 -ny 16 -o 2 -re 40 -cont -tau 1
//               mpirun -np 4 pnavierstokes -p 1 -nx 16 -ny 4 -sx 4 -o 2 -re 200 -cont
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
//                   q - nu grad u          = 0,
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
//                   F_{1+i,d} = v_i v_d + p d_id,  G_{1+i,d} = -q_{1+i,d}.
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
//               correctly inexact at order 1. Kovasznay at Re = 40 gives the
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
//               **What is parallel here, and what is not.** Almost nothing,
//               and that is a property of the method rather than a shortcut.
//               The flux and the potential are L2 spaces, so their L-dofs are
//               their true dofs and every element-local operation -- the
//               residual, the local Jacobian blocks, their factorisation, the
//               reduction and the recovery -- is rank-local and identical to
//               the serial one. **Only the trace is shared**, prolonged in
//               before the element loop and assembled out after it, which
//               DarcyHybridization::NPCResidual() does internally. So the
//               unknown vector this miniapp hands NewtonSolver has its flux
//               and potential blocks in L-dofs and its trace block in TRUE
//               dofs; that is the one place the two representations meet, and
//               it is why the offsets below are built here rather than taken
//               from DarcyOperator::ConstructOffsets(), which is all-L-dof.
//
//               The consequence worth stating: the number of ranks changes no
//               element loop, so it must change no answer, and that is the
//               sharp check on this file. **The sharp form of it needs a case
//               whose error is NOT at round-off**, because two numbers that
//               are both 1e-15 agree for a reason that has nothing to do with
//               the partition. Kovasznay at Re = 40, 12x8, order 2, with
//               -cont, is that case, and it is identical to every printed
//               digit:
//
//                                   q            p            v
//                   serial     0.00851914   0.00109304   0.000965313
//                   np = 1     0.00851914   0.00109304   0.000965313
//                   np = 2     0.00851914   0.00109304   0.000965313
//                   np = 3     0.00851914   0.00109304   0.000965313
//                   np = 4     0.00851914   0.00109304   0.000965313
//
//               So is plane Poiseuille at order 1, where the discretisation
//               error dominates (0.0595198 / 0.00482045 / 0.00678842 at every
//               rank count and in serial), and so is the same Kovasznay case
//               under -tau 0.5 and under -gm 2.
//
//               Where the answer IS at round-off, the exactness check is the
//               statement instead. Plane Poiseuille at order 2, 8x8, relative
//               q / p / v:
//
//                   serial     4.42e-15   2.06e-15   3.25e-16
//                   np = 1     4.63e-15   2.03e-15   3.39e-16
//                   np = 2     4.48e-15   1.58e-15   3.36e-16
//                   np = 3     4.63e-15   2.16e-15   3.24e-16
//                   np = 4     4.65e-15   2.72e-15   3.57e-16
//
//               and at order 3, 8.913e-13 / 1.535e-12 / 1.601e-13 to four
//               digits at every rank count. Those last digits move because the
//               trace solve is iterative and its convergence path follows the
//               AMG hierarchy, which follows the partition; the discrete
//               solution does not.
//
//               The trace solve is the one thing that is genuinely different.
//               Serial reaches for UMFPACK; here the reduced trace system is a
//               HypreParMatrix and the solve is GMRES preconditioned by
//               BoomerAMG with SetAdvectiveOptions(), which is what
//               darcyop.cpp does on the same system. It is iterative, so its
//               error enters the Newton correction in a way the direct solve's
//               did not -- see -ltol, whose docstring carries the measurement
//               that says tightening it is not monotonically better.
//
//               The miniapp is written so that the compressible equations drop
//               in later: the state layout puts the continuity variable first
//               exactly as EulerFlux's (rho, rho v, rho E) does, and the only
//               things that know which equations they are are the flux
//               function and the per-equation viscosity table. See
//               "Extending to the compressible equations" at the foot of this
//               file.
//
//               We recommend viewing the navierstokes and pconvdiff miniapps
//               before this one.

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

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialize MPI and HYPRE.

   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();
   const bool root = (myid == 0);

   // 2. Parse command-line options.

   int iproblem = Problem::PlanePoiseuille;
   int nx = 8, ny = 8;
   real_t sx = 1., sy = 1.;
   int order = 2;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
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
   real_t trace_rtol = 1e-12;
   real_t hsign = -1.;
   bool bc_full = true;
   bool continuation = false;
   bool pgrad = false;
   bool bface = false;
   bool exact_init = false;
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
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of uniform refinement levels before partitioning.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of uniform refinement levels after partitioning.");
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
                  "this. Negative (the default) prints and always exits 0. "
                  "Same flag and same reasoning as the serial miniapp -- "
                  "mfem-test grades on the exit status alone and deletes the "
                  "output, so without this pnavierstokes-test-par checks "
                  "nothing but that the binary ran. Use an ABSOLUTE threshold: "
                  "plane Poiseuille at order >= 2 lies in the discrete space "
                  "and lands at round-off, but WHERE in the 1e-15s moves with "
                  "the BLAS and the partition, so a relative comparison "
                  "against a stored number cannot pass. Needs -rtol 1e-12. "
                  "Every rank computes this identically -- ComputeL2Error and "
                  "ComputeGlobalLpNorm both return the GLOBAL value on all "
                  "ranks -- so the exit code is the same everywhere and needs "
                  "no broadcast.");
   args.AddOption(&line_search, "-ls", "--line-search",
                  "-no-ls", "--no-line-search",
                  "Globalise with KINSOL's KIN_LINESEARCH, which implies "
                  "-nls 4. It backtracks on the FULL residual, well defined "
                  "here only because the flux, potential and trace are one "
                  "Newton vector so a step scales all three together. Off by "
                  "default: it changes no converged answer, only whether a "
                  "cold start reaches one, and -cont covers the same ground "
                  "for the cases here. Note it is an l2 merit over all three "
                  "blocks -- where the nonlinearity sits in the potential "
                  "block and the flux and trace rows are linear, a full step "
                  "is exactly optimal for two of the three and damping can be "
                  "worse than none; see doc/HDG-ORDERING-API.md section 6.");
   args.AddOption(&gradient_mode, "-gm", "--gradient-mode",
                  "How much of the hybridized trace system to build: "
                  "0 or 1=assemble, and precondition GMRES with BoomerAMG, "
                  "2=matrix free, which leaves GMRES nothing to precondition "
                  "with. Negative leaves it. The serial miniapp separates 0 "
                  "and 1 because 0 reaches for UMFPACK; there is no direct "
                  "solver on a HypreParMatrix in this build, so both are the "
                  "same preconditioned Krylov solve here. -gm 2 is the one "
                  "option that found a library defect: DarcyHybridization::"
                  "ParGradient::Mult() passed the unsized darcy_rhs to "
                  "ParMultNL(), which restricts the load before it dispatches, "
                  "so any caller driving NPC without FormLinearSystem() "
                  "segfaulted on the first matrix-free gradient. Fixed there; "
                  "this flag is what pins it.");
   args.AddOption(&trace_rtol, "-ltol", "--trace-rtol",
                  "Relative tolerance of the reduced trace solve. It is a "
                  "parallel-only option because serial has a direct solver "
                  "and this does not: BoomerAMG-preconditioned GMRES has an "
                  "error of its own, and it enters the Newton correction. On "
                  "a NONLINEAR problem it does not reach the answer -- Newton "
                  "keeps correcting until the full residual is down, so plane "
                  "Poiseuille comes back at 4.5e-15 whatever this is set to. "
                  "On a LINEAR one it is the answer, and then tightening it "
                  "is not monotonically better: -stokes at 8x8 order 2 gives "
                  "4.4e-15 at 1e-8, 4.5e-15 at 1e-10, 1.5e-12 at 1e-12 and "
                  "1.7e-14 at 1e-14. The 1e-12 row is the bad one because "
                  "Newton's test is RELATIVE -- a loose trace solve leaves a "
                  "residual above rtol*r0, so Newton takes a second step and "
                  "that step is iterative refinement, while a solve just tight "
                  "enough to pass on the first step leaves its own error in "
                  "place. Serial's direct solve has no such error, which is "
                  "why -stokes reads 1.2e-14 there and this option does not "
                  "exist.");
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
                  "standard verification condition and it is the one that "
                  "works. -bcphys uses the physical set (no-slip walls, "
                  "velocity in at the inlet, pressure at the outlet) and is "
                  "NOT yet correct -- see the note at the boundary conditions "
                  "in main().");
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
      if (root) { cerr << "Unknown problem " << iproblem << endl; }
      return 1;
   }
   if (!hybridization)
   {
      // The non-hybridized LDG face terms of MixedConductionNLFIntegrator
      // abort for more than one equation (nonlininteg_mixed.cpp:299), and the
      // hyperbolic term would then need the two-state Eval path. Refusing is
      // better than aborting deep in an integrator.
      if (root)
      {
         cerr << "This miniapp is hybridized only; -no-hb is not implemented."
              << endl;
      }
      return 1;
   }

   Device device(device_config);
   if (root) { device.Print(); }

   // 3. The mesh, and the problem's own geometry.

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
   for (int l = 0; l < serial_ref_levels; l++) { mesh.UniformRefinement(); }

   // Translate to the problem's window. MakeCartesian2D always starts at the
   // origin, and the exact solutions are written in absolute coordinates.
   // Done on the serial mesh, before partitioning, so every rank agrees by
   // construction rather than by each repeating the same arithmetic.
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

   // 4. Partition, then refine further in parallel. The serial mesh goes.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int l = 0; l < parallel_ref_levels; l++) { pmesh.UniformRefinement(); }

   const int dim = pmesh.Dimension();
   pars.dim = dim;
   const int neq = dim + 1;

   // Boundary attributes of MakeCartesian2D: 1 = y_min, 2 = x_max,
   // 3 = y_max, 4 = x_min.
   enum { BDR_BOTTOM = 1, BDR_RIGHT = 2, BDR_TOP = 3, BDR_LEFT = 4 };

   // 5. Finite element spaces.
   //
   //    All three are byNODES, and that is a requirement, not a preference:
   //    HyperbolicFormIntegrator reinterprets the element dof vector as a
   //    DenseMatrix(data, ndof, neq), which is column-major, so it reads the
   //    equation index as the *outer* one. MixedConductionNLFIntegrator and
   //    DarcyHybridization::ProjectSolution assume the same.

   L2_FECollection q_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   DG_Interface_FECollection t_coll(order, dim);

   ParFiniteElementSpace fes_q(&pmesh, &q_coll, neq * dim, Ordering::byNODES);
   ParFiniteElementSpace fes_u(&pmesh, &u_coll, neq,       Ordering::byNODES);
   ParFiniteElementSpace fes_t(&pmesh, &t_coll, neq,       Ordering::byNODES);

   // Scalar and vector views used only for error norms and visualisation.
   ParFiniteElementSpace fes_s(&pmesh, &u_coll, 1);
   ParFiniteElementSpace fes_v(&pmesh, &u_coll, dim, Ordering::byNODES);

   ParDarcyForm darcy(&fes_q, &fes_u);

   //    GlobalTrueVSize() is COLLECTIVE -- it builds the dof-true-dof
   //    HypreParMatrix, whose constructor calls MPI_Allreduce -- so the three
   //    calls happen on every rank and only the printing is root-only. Putting
   //    them inside the `if (root)` deadlocks: rank 0 waits in that Allreduce
   //    while every other rank walks on into EnableHybridization and blocks in
   //    ParMesh::ExchangeFaceNbrData. Measured here rather than reasoned: the
   //    first parallel run of this file hung with exactly those two stacks.
   const HYPRE_BigInt gq = fes_q.GlobalTrueVSize();
   const HYPRE_BigInt gu = fes_u.GlobalTrueVSize();
   const HYPRE_BigInt gt = fes_t.GlobalTrueVSize();

   if (root)
   {
      cout << "dim(q) = " << gq
           << ", dim(u) = " << gu
           << ", dim(t) = " << gt
           << ", neq = " << neq << endl;
      cout << "nu = " << pars.nu << ", beta = " << pars.beta
           << ", U = " << pars.U << endl;
   }

   // 6. Coefficients.

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

   // 7. The flux mass, `(nu^-1 q, v)`.
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
      darcy.GetParFluxMassForm()->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(mass));
   }

   // 8. The flux divergence, `(div q, w)`, zero on the pressure row.

   ParMixedBilinearForm *B = darcy.GetParFluxDivForm();
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
         Array<int> bdr_all(pmesh.bdr_attributes.Max());
         bdr_all = 1;
         B->AddBdrFaceIntegrator(
            new VectorBlockDiagonalIntegrator(
               neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))),
            bdr_all);
      }
   }

   // 9. The potential mass, carrying both halves of the stabilization and the
   //    whole of the inviscid flux. Everything goes on the *nonlinear* form:
   //    DarcyForm keeps one potential mass form, and the convective term is
   //    nonlinear, so the diffusive stabilization has to join it there.
   //    HDGDiffusionIntegrator is a BilinearFormIntegrator and therefore also
   //    a NonlinearFormIntegrator, so that is legal and is what convdiff does
   //    in its own nonlinear branch.

   ParNonlinearForm *Mtnl = darcy.GetParPotentialMassNonlinearForm();

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
   Mtnl->AddBdrFaceIntegrator(new HyperbolicFormIntegrator(*num_flux, 0, hsign));

   // 10. Hybridization. The constraint is `<[q.n], mu>`, again zero on the
   //     pressure row so that q_0 stays uncoupled; NormalTraceJumpIntegrator's
   //     sign argument is the cheapest way to get a correctly shaped zero. The
   //     trace space is the only one of the three that is shared between
   //     ranks, so it is the only one whose prolongation is ever formed.

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

   // 11. Boundary conditions, imposed essentially on the trace, per component.
   //
   //     The weak route the reference uses -- its boundary flux vector B^ of
   //     Eq. (8) -- is not available here. It would go through
   //     BdrHyperbolicDirichletIntegrator, and that integrator reads its
   //     prescribed state only when bit 0 of `type` is set, which
   //     DarcyHybridization never sets on a boundary face: every `type |= 1`
   //     site sits inside an interior-face branch. Registered on the hybridized
   //     form it degrades to an ordinary HyperbolicFormIntegrator with no
   //     warning and no abort -- the interior state is used and the boundary
   //     datum is silently dropped. So the datum is put on the trace instead.
   //
   //     DarcyHybridization::SetEssentialBC marks *every* component of a marked
   //     attribute, which is wrong for flow: a wall fixes the velocity and says
   //     nothing about the pressure. The per-component lists are built here and
   //     handed to SetEssentialTrueDofs, which is purely index-based and so is
   //     component-blind in the way that is wanted.
   //
   //     **What -bcphys does not yet do, and it matters.** A boundary trace
   //     component that is *not* essential keeps the constraint row
   //     <(F^ + q^).n, mu> = 0, and on a boundary face that row has only one
   //     side, so nothing cancels it: it imposes zero numerical flux, which is
   //     not the intended condition. Measured on plane Poiseuille with the
   //     physical set -- no-slip walls, profile in, pressure at the outlet --
   //     the solve converges to 3e-13 and the answer is wrong by more than
   //     100%, at every order. The interior discretisation is not at fault:
   //     with -bcfull the same problem is exact to 2.5e-15. Making -bcphys
   //     right needs the prescribed numerical flux on those faces, either as a
   //     linear form on the trace (the Neumann datum, which convdiff supplies
   //     through BoundaryNormalLFIntegrator) or as the reference's
   //     characteristic B^ = A+_n (u - u^) - A-_n (u_inf - u^), which for the
   //     artificial-compressibility system needs the eigen-decomposition of
   //     A_n. That is the next piece of work, not a defect in what is here.

   Array<int> bdr_vel(pmesh.bdr_attributes.Max());   // velocity prescribed
   Array<int> bdr_pres(pmesh.bdr_attributes.Max());  // pressure prescribed
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
   //     ParFiniteElementSpace::GetEssentialTrueDofs() already returns TRUE
   //     dofs, which is what SetEssentialTrueDofs() wants and what the trace
   //     block of the NPC vector is indexed in. Nothing here is L-dof.
   hyb->SetEssentialTrueDofs(ess_tdofs);
   {
      long long loc[2] = { ess_tdofs.Size(), fes_t.GetTrueVSize() }, glob[2];
      MPI_Reduce(loc, glob, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if (root)
      {
         cout << "essential trace tdofs: " << glob[0]
              << " of " << glob[1] << endl;
      }
   }

   darcy.Assemble();

   // 12. State vectors, and the one place the two dof representations meet.
   //
   //     The NPC unknown is (q, u, lambda). The flux and the potential are L2,
   //     so their L-dofs ARE their true dofs and the blocks below are both at
   //     once; the trace is shared, and its block is in TRUE dofs, because
   //     that is what DarcyHybridization::NPCResidual() prolongs from and
   //     assembles back into. So the offsets are built here rather than taken
   //     from DarcyOperator::ConstructOffsets(), which is all-L-dof and would
   //     size the trace block wrongly on more than one rank -- and by exactly
   //     the count of shared face dofs, so it would be right on one rank and
   //     silently wrong on two.

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = fes_q.GetVSize();
   offsets[2] = fes_u.GetVSize();
   offsets[3] = fes_t.GetTrueVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.;
   rhs = 0.;

   ParGridFunction q_h, u_h;
   q_h.MakeRef(&fes_q, x.GetBlock(0), 0);
   u_h.MakeRef(&fes_u, x.GetBlock(1), 0);

   //     The trace grid function owns its own L-dof storage: it cannot alias
   //     x.GetBlock(2), which is true dofs. It is the L-dof face of the same
   //     data, written by projection and read back by ParallelProject().
   ParGridFunction tr_h(&fes_t);
   tr_h = 0.;

   //     Project the exact state onto the whole boundary trace. Only the
   //     components in ess_tdofs are enforced; the rest are an initial guess
   //     the solver overwrites. Both problems here have a known exact
   //     solution, so this is the honest thing to do -- a problem without one
   //     would need the per-component projection instead.
   Array<int> all_bdr(pmesh.bdr_attributes.Max());
   all_bdr = 1;
   tr_h.ProjectBdrCoefficient(state_coeff, all_bdr);

   if (exact_init)
   {
      q_h.ProjectCoefficient(flux_coeff);
      u_h.ProjectCoefficient(state_coeff);
      // The trace space is face-based, so an element projection is not
      // available; DarcyHybridization averages the two element traces, which
      // for a solution continuous across the face is that solution's trace.
      //
      // In parallel that average is one-sided on a shared face, because the
      // local face list gives Elem2No < 0 there. It does not matter for the
      // four problems here -- all four have a continuous exact solution, so
      // the two sides agree and the average is the value -- and it is an
      // initial guess in any case. A discontinuous datum would need the
      // neighbour's element values communicated first.
      BlockVector sol2(x.GetData(), darcy.GetOffsets());
      hyb->ProjectSolution(sol2, tr_h);
   }

   //     Down to true dofs, which is the representation the solve works in.
   //     ParallelProject() takes the owning rank's value; on a boundary face,
   //     which is never shared, that is the value just projected.
   tr_h.ParallelProject(x.GetBlock(2));

   // 13. Right-hand sides. Both problems have zero source, so all three forms
   //     are empty -- but they still have to exist, and the flux and potential
   //     ones are Update()d onto the blocks of `rhs` so that the (flux,
   //     potential) load handed to the NPC operator is one contiguous piece.
   //
   //     The trace form cannot be: `rhs.GetBlock(2)` is true dofs and a
   //     ParLinearForm assembles in L-dofs, so it owns its storage and is
   //     reduced with ParallelAssemble(). Assembling a trace load onto the
   //     true-dof block directly would be a size mismatch on one rank and a
   //     wrong sum on several.

   ParLinearForm gform, fform;
   gform.Update(&fes_q, rhs.GetBlock(0), 0);
   fform.Update(&fes_u, rhs.GetBlock(1), 0);

   ParLinearForm hform(&fes_t);
   hform = 0.;
   hform.Assemble();
   hform.ParallelAssemble(rhs.GetBlock(2));

   // 14. Solve, by NPC: Newton on the FULL (q, u, u_hat) system with the
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
   //     residual carries no load of its own. Both blocks of the load are
   //     rank-local, which is why `darcy.GetOffsets()` -- an L-dof pair -- is
   //     still the right thing to wrap them with.
   BlockVector load(rhs, darcy.GetOffsets());
   DarcyNPCOperator npc(*hyb, offsets, load);

   BlockVector b(offsets);
   b = 0.;
   b.GetBlock(2) = rhs.GetBlock(2);

   //     The reduced trace system, by -gm. Under GradientMode::Assembled it
   //     is a HypreParMatrix, so BoomerAMG can precondition it; under
   //     MatrixFree it is an operator that only applies S, and there is
   //     nothing for AMG to coarsen. The serial miniapp reaches for UMFPACK
   //     at -gm 0 and this cannot: neither MUMPS nor SuperLU is a required
   //     dependency, so the trace solve is iterative in every mode here and
   //     -ltol is what says how accurate it is.
   unique_ptr<Solver> trace_solver;
   unique_ptr<Solver> trace_prec;
   {
      auto gmres = make_unique<GMRESSolver>(MPI_COMM_WORLD);
      gmres->SetKDim(200);
      gmres->SetMaxIter(2000);
      gmres->SetRelTol(trace_rtol);
      gmres->SetAbsTol(0.);
      gmres->SetPrintLevel(-1);
      if (gradient_mode != 2)
      {
         // SetAdvectiveOptions() is darcyop.cpp's choice on the same matrix:
         // the trace system of a convection-dominated HDG problem is not
         // symmetric and the default BoomerAMG options are tuned for one
         // that is.
         auto *amg = new HypreBoomerAMG();
         amg->SetAdvectiveOptions();
         amg->SetPrintLevel(0);
         trace_prec.reset(amg);
         gmres->SetPreconditioner(*trace_prec);
      }
      trace_solver = std::move(gmres);
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
   if (line_search || solver_type == (int) DarcyOperator::SolverType::KINSol)
   {
      auto *kin = new KINSolver(MPI_COMM_WORLD,
                                line_search ? KIN_LINESEARCH : KIN_NONE);
      // A true Newton rather than a lagged-Jacobian one; without it KINSOL
      // reuses a setup and the comparison against -nls 3 is not like for like.
      kin->SetMaxSetupCalls(1);
      newton.reset(kin);
   }
   else
   {
      newton.reset(new NewtonSolver(MPI_COMM_WORLD));
   }
   newton->SetOperator(npc);
   newton->SetSolver(lin);
   newton->SetRelTol((newton_rtol > 0.) ? newton_rtol : 1e-6);
   newton->SetAbsTol(newton_atol);
   newton->SetMaxIter(newton_iters);
   newton->SetPrintLevel(root ? 1 : -1);

   chrono.Clear();
   chrono.Start();
   {
      // Two solves for Stokes continuation, the second starting from the
      // first's answer -- which under NPC is simply `x`, all three blocks of
      // it, with nothing to recover in between.
      if (continuation && !stokes)
      {
         ac_flux.SetStokes(true);
         newton->Mult(b, x);
         ac_flux.SetStokes(false);
         if (root)
         {
            cout << "--- Stokes continuation done; continuing onto "
                 "Navier-Stokes ---" << endl;
         }
      }
      newton->Mult(b, x);
   }
   chrono.Stop();

   //     Summed over the ranks, because it is a count of element-local work
   //     and the elements are divided among them. NPC does none of it, so the
   //     sum is zero on any rank count; a nonzero one on some ranks and not
   //     others would be invisible in a rank-0 print.
   {
      long long loc = hyb->GetNumLocalNLIterations(), glob = 0;
      MPI_Reduce(&loc, &glob, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if (root)
      {
         cout << "local nonlinear iterations: " << glob
              << " (NPC runs none; anything else means the ordering did not "
              "change)" << endl;
      }
   }

   // 15. Errors.

   const int order_quad = max(2, 2 * order + 3);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   //     byNODES means component c of u_h occupies [c*nd, (c+1)*nd), so the
   //     pressure is a scalar field at offset 0 and the velocity a dim-wide
   //     one at offset nd. No copy is needed, and none is made.
   //
   //     ParGridFunction::ComputeL2Error() reduces over the communicator, and
   //     ComputeGlobalLpNorm() is the matching norm of the coefficient. Mixing
   //     one with the serial ComputeLpNorm() would divide a global error by a
   //     rank-local norm and the ratio would change with the rank count for
   //     no reason at all.
   const int nd = fes_s.GetNDofs();
   ParGridFunction p_h(&fes_s, u_h.GetData());
   ParGridFunction v_h(&fes_v, u_h.GetData() + nd);

   const real_t err_p  = p_h.ComputeL2Error(pres_coeff, irs);
   const real_t norm_p = ComputeGlobalLpNorm(2., pres_coeff, pmesh, irs);
   const real_t err_v  = v_h.ComputeL2Error(vel_coeff, irs);
   const real_t norm_v = ComputeGlobalLpNorm(2., vel_coeff, pmesh, irs);
   const real_t err_q  = q_h.ComputeL2Error(flux_coeff, irs);
   const real_t norm_q = ComputeGlobalLpNorm(2., flux_coeff, pmesh, irs);

   //     Per-equation flux errors. The flux is the block the local solve gets
   //     wrong when row 1 is miswired, and a single lumped norm cannot say
   //     which equation is at fault -- the pressure row, whose coupling is
   //     deliberately zeroed, or the momentum rows.
   {
      const int ndq = fes_q.GetNDofs();
      ParFiniteElementSpace fes_qe(&pmesh, &q_coll, dim, Ordering::byNODES);
      for (int e = 0; e < neq; e++)
      {
         ParGridFunction qe(&fes_qe, q_h.GetData() + e * dim * ndq);
         auto qe_fun = [&pars, e, dim](const Vector & x, Vector & v)
         {
            Vector all;
            ExactFlux(pars, x, all);
            v.SetSize(dim);
            for (int d = 0; d < dim; d++) { v(d) = all(e * dim + d); }
         };
         VectorFunctionCoefficient qec(dim, qe_fun);
         const real_t err_qe = qe.ComputeL2Error(qec, irs);
         const real_t nrm_qe = ComputeGlobalLpNorm(2., qec, pmesh, irs);
         if (root)
         {
            cout << "   flux eq " << e << ": || q_h - q_ex || = " << err_qe
                 << ",  || q_ex || = " << nrm_qe << "\n";
         }
      }
   }

   if (root)
   {
      cout << "|| q_h - q_ex || / || q_ex || = "
           << ((norm_q > 0.) ? (err_q / norm_q) : err_q) << "\n";
      cout << "|| p_h - p_ex || / || p_ex || = "
           << ((norm_p > 0.) ? (err_p / norm_p) : err_p) << "\n";
      cout << "|| v_h - v_ex || / || v_ex || = "
           << ((norm_v > 0.) ? (err_v / norm_v) : err_v) << "\n";
   }

   //     Grade the run, if asked; see the -chk help. Computed on EVERY rank,
   //     not just root, so that all of them return the same status -- the
   //     norms above are global on every rank, so this needs no communication.
   int failures = 0;
   if (check_tol > 0.)
   {
      const real_t rels[3] = { (norm_q > 0.) ? (err_q / norm_q) : err_q,
                               (norm_p > 0.) ? (err_p / norm_p) : err_p,
                               (norm_v > 0.) ? (err_v / norm_v) : err_v
                             };
      const char *names[3] = { "q", "p", "v" };
      for (int i = 0; i < 3; i++)
      {
         if (!(rels[i] <= check_tol))
         {
            failures++;
            if (root)
            {
               cout << "CHECK FAILED: " << names[i] << " relative error "
                    << rels[i] << " exceeds " << check_tol << "\n";
            }
         }
      }
      if (root)
      {
         cout << (failures ? "CHECK FAILED" : "CHECK PASSED")
              << " at tolerance " << check_tol << "\n";
      }
   }

   // 16. Visualisation.

   if (visualization)
   {
      const int num_procs = Mpi::WorldSize();
      char vishost[] = "localhost";
      const int visport = 19916;
      socketstream p_sock(vishost, visport), v_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "parallel " << num_procs << " " << myid << "\n"
             << "solution\n" << pmesh << p_h
             << "window_title 'Pressure'" << endl;
      v_sock.precision(8);
      v_sock << "parallel " << num_procs << " " << myid << "\n"
             << "solution\n" << pmesh << v_h
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
