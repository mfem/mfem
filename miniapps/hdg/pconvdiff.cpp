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
//                   Convection-diffusion miniapp - Parallel version
//
// Compile with: make pconvdiff
//
// Sample runs:  mpirun -np 4 pconvdiff -nx 20 -p 2 -o 2 -hb -dg
//               mpirun -np 4 pconvdiff -nx 20 -p 2 -o 2 -hb -up -bcn
//               mpirun -np 4 pconvdiff -m ../../data/amr-quad.mesh -rs 1 -p 2 -o 2 -hb -brt -up -bcn
//               mpirun -np 4 pconvdiff -nx 20 -p 2 -o 2 -hb -dg -up -bcn -trbc -trh1
//               mpirun -np 4 pconvdiff -m ../../data/amr-quad.mesh -rs 1 -p 2 -o 2 -rd -brt -up -bcn
//               mpirun -np 4 pconvdiff -nx 20 -p 2 -o 0 -hb -brt -up -rec
//               mpirun -np 4 pconvdiff -nx 20 -p 2 -o 1
//               mpirun -np 4 pconvdiff -nx 40 -p 3 -o 2 -k 1e-4 -hb -dg -up
//               mpirun -np 4 pconvdiff -nx 40 -p 4 -k 1e-4 -tf 0.5 -nt 20 -ode 3 -o 2 -hb -dg
//               mpirun -np 2 pconvdiff -m ../../data/inline-tri.mesh -rs 2 -p 10 -o 1 -dg -hb -nls 3 -rtol 1e-11 -tau0 1 -pp
//             * mpirun -np 4 pconvdiff -nx 120 -ny 30 -sx 10 -sy 2.5 -p 5 -k 1e-2 -tf 10 -nt 100 -ode 3 -o 2 -dg -hb
//
// Device sample runs:
//
// Description:  This miniapp solves a convection-diffusion problem in the mixed
//               formulation corresponding to the system
//
//                                 1/k q + ∇ T           =  g
//                                   ∇⋅q + ∇⋅(Tc) + dT/dt = -f
//
//               with natural Dirichlet boundary condition T = <given
//               temperature> and/or Neumann boundary condition (q + Tc)⋅n =
//               = <given total flux>, where n is the outer normal. The scalar k
//               is the heat conductivity and c the given velocity field.
//               Multiple problems are offered based on the paper: N.C. Nguyen
//               et al., Journal of Computational Physics 228 (2009) 3232–3254.
//               In particular, they are (corresponding to the subsections of
//               section 5):
//               1) steady diffusion - sine profile steady diffusion with zero
//                                     Dirichlet temperature BCs
//               2) steady advection-diffusion - sine profile steady advection-
//                                               -diffusion with zero Dirichlet
//                                               temperature BCs (and Neumann
//                                               outflow BCs optionally)
//               3) steady advection - step function circular advection with
//                                     Dirichlet temperature inflow and Neumann
//                                     total flux outflow BC (instead of paper's
//                                     embedded BC for a full circle advection)
//               4) non-steady advection(-diffusion) - blob circular advection
//                                                     with diffusion with
//                                                     Dirichlet temperature BCs
//               5) Kovasznay flow - advection-diffusion of periodically
//                                   injected contaminant concentration (repr.
//                                   by T) with Dirichlet temperature inflow BC
//                                   and Neumann total flux outflow BCs
//               6) steady Burgers flow - with zero Dirichlet temperature BCs
//               7) non-steady Burgers flow - with zero Dirichlet temperature
//                                            BCs
//              10) steady reaction-diffusion - CCSZ Example 4.1: -Delta u +
//                                              F(u) = f with F(u) = u^3 - u
//                                              and u = prod sin(pi x_i), zero
//                                              Dirichlet on the unit box.
//                                              Needs -hb and -nls 3.
//
//               PROBLEM 10 IS THE INTERPOLATORY-HDG PROBLEM, and it is the
//               only one in this miniapp whose nonlinearity is a reaction
//               rather than a flux law. Chen, Cockburn, Singler & Zhang,
//               "Superconvergent Interpolatory HDG Methods for Reaction
//               Diffusion Equations I: An HDG_k Method", J. Sci. Comput. 81
//               (2019) 2188-2212. Their claim is that the LOCALLY
//               POSTPROCESSED potential u* converges at k+2 where u and q
//               converge at k+1, for k >= 1, and does NOT superconverge at
//               k = 0 -- the rate is min{k,1} better. Three flags reach it:
//
//                 -rx 1  interpolate F at the nodes of the enriched space,
//                        which is the method (default)
//                 -rx 2  integrate F(u*) against the potential basis under
//                        quadrature, which is the control it is measured
//                        against and is NOT the paper's method
//                 -pp    compute u* and report its error
//                 -tau0  a CONSTANT stabilization
//
//               -tau0 IS NOT OPTIONAL FOR THE THEOREM and it is not a tuning
//               knob. The built-in HDG stabilization is kappa/h, the LDG-style
//               scaling; CCSZ's result is stated for an O(1) elementwise
//               constant tau. Measured here on uniform triangulations of the
//               unit square, -rx 1, rates of u / q / u*:
//
//                    k    -tau0 1                 default kappa/h
//                    0    1.00 1.00  1.00         -0.10  0.07  -0.25
//                    1    2.02 2.01  3.00          2.03  1.14   1.96
//                    2    3.04 3.05  4.00          3.50  2.02   3.20
//                    3    4.03 4.01  5.01
//
//               The k = 0 row is the sharp half of the check and it holds
//               both ways: at tau = O(1) u* does not beat u, which is CCSZ's
//               own Table 1 (0.97); at kappa/h the postprocessed error stops
//               decreasing and starts GROWING. A ladder run at the default
//               tau is not a test of this theorem and would report the method
//               as failing.
//
//               INTERPOLATING COSTS NOTHING MEASURABLE. Against -rx 2 on the
//               same meshes at -tau0 1, u* agrees to the FIFTH digit at k = 1
//               (8.6227e-04 against 8.6247e-04) and the SIXTH at k = 2, with
//               the same rates and the same constants. That is the paper's
//               Remark 2.3 borne out here, and it is why the -rx 2 references
//               in regress_test/ are a CONTROL rather than a discriminator:
//               the two arms are supposed to agree, so a reference pinning
//               them apart would be pinning noise.
//
//               A tight -rtol matters: the reaction puts the local element
//               solve on an iterative Newton whose tolerance is derived from
//               the outer one, and at the default the outer solver stops one
//               step in and reports the local solver's noise floor as the
//               discretisation error.
//               We discretize with (broken) Raviart-Thomas finite elements
//               (heat flux q) and piecewise discontinuous polynomials
//               (temperature T). Alternatively, the piecewise discontinuous
//               polynomials are used for both quantities with stabilization,
//               yielding the Local Discontinuous Galerkin method. Optionally,
//               the mixed system is algebraically reduced or hybridized with
//               DG interface elements or H1 trace elements. The schemes can be
//               also upwinded along the velocity field in both, diffusion and
//               convection parts, or centered (default).
//
//               The miniapp demonstrates the use of the DarcyForm class and
//               time evolution of the system operator provided by DarcyOperator
//               with different boundary conditions and discretizations.
//
//               We recommend viewing examples 1-5, 9 and 41 before viewing this
//               miniapp.

#include "mfem.hpp"
#include "darcyop.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <functional>

using namespace std;
using namespace mfem;
using namespace mfem::hdg;

// Define the analytical solution and forcing terms / boundary conditions
typedef function<real_t(const Vector &, real_t)> TFunc;
typedef function<void(const Vector &, Vector &)> VecFunc;
typedef function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef function<real_t(real_t f, const Vector &x)> KFunc;

enum Problem
{
   SteadyDiffusion = 1,
   SteadyAdvectionDiffusion,
   SteadyAdvection,
   NonsteadyAdvectionDiffusion,
   KovasznayFlow,
   SteadyBurgers,
   NonsteadyBurgers,
   SteadyLinearKappa,
   NonsteadyLinearKappa,
   SteadyReactionDiffusion,
};

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

/** @brief `F(u) = u^3 - u`, Example 4.1 of Chen, Cockburn, Singler & Zhang,
    J. Sci. Comput. 81 (2019) 2188-2212 -- the reaction of problem 10.

    A NodalReactionFunction rather than a Coefficient because the interpolatory
    method evaluates it AT the nodes of the enriched space and never under a
    quadrature rule; the quadrature control reached by `-rx 2` uses the same
    object, which is what makes the two arms differ in one thing only. */
struct CubicReaction : public NodalReactionFunction
{
   int NumEquations() const override { return 1; }
   void Eval(const Vector &, const Vector &u, Vector &F) const override
   {
      F.SetSize(1);
      F(0) = u(0) * u(0) * u(0) - u(0);
   }
   void EvalJacobian(const Vector &, const Vector &u,
                     DenseMatrix &J) const override
   {
      J.SetSize(1);
      J(0, 0) = 3.0 * u(0) * u(0) - 1.0;
   }
};

/** @brief A stabilization that is the constant @a tau0, ignoring the `1/h` the
    integrator forms for itself.

    **This is not a tuning knob, it is a different method.** The built-in
    stabilization is `kappa/h`, the LDG-style scaling; CCSZ's theorem is stated
    for an O(1) elementwise-constant `tau`, and the superconvergence of `u*` is
    lost without it -- at order 0 the postprocessed error stops decreasing and
    starts growing. See HDGPotentialPostprocessor's doxygen, which carries the
    measured rates both ways. */
struct ConstStabilization : public HDGStabilization
{
   real_t tau0;
   ConstStabilization(real_t t) : tau0(t) { }
   real_t Eval(real_t, real_t, real_t, real_t,
               ElementTransformation &) const override { return tau0; }
};

struct ProblemParams
{
   Problem prob;
   real_t k;
   real_t t_0;
   real_t c;
};

TFunc GetTFun(const ProblemParams &params);
VecTFunc GetQFun(const ProblemParams &params);
VecFunc GetCFun(const ProblemParams &params);
TFunc GetFFun(const ProblemParams &params);
unique_ptr<FluxFunction> GetFluxFun(const ProblemParams &params,
                                    VectorCoefficient &ccoeff);
unique_ptr<MixedFluxFunction> GetHeatFluxFun(const ProblemParams &params,
                                             int dim);

// Visualize the grid function in GLVis
bool VisualizeField(socketstream &sout, const ParGridFunction &gf,
                    const char *name, int iter = 0, bool verbose = true);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   //const int num_procs = Mpi::WorldSize();
   const int myid = Mpi::WorldRank();
   Hypre::Init();
   const bool root = (myid == 0);

   // 2. Parse command-line options.
   string mesh_file = "";
   int nx = 0;
   int ny = 0;
   int serial_ref_levels = -1;
   int parallel_ref_levels = 0;
   real_t sx = 1.;
   real_t sy = 1.;
   int order = 1;
   bool dg = false;
   bool brt = false;
   bool upwinded = false;
   int iproblem = Problem::SteadyDiffusion;
   int reaction = 1;
   real_t tau0 = 0.;
   bool postprocess = false;
   ProblemParams pars;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   pars.k = 1.;
   pars.c = 1.;
   real_t td = 0.5;
   bool bc_neumann = false;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   bool trace_ess_bc = false;
   bool nonlinear = false;
   bool nonlinear_flux = false;
   bool nonlinear_pot = false;
   bool nonlinear_conv = false;
   bool use_npc = false;
   int gradient_mode = -1;
   real_t newton_rtol = -1.;
   bool nonlinear_diff = false;
   int hdg_scheme = 1;
   int solver_type = (int)DarcyOperator::SolverType::Default;
   int prec_type = (int)DarcyOperator::PrecType::Default;
   bool pa = false;
   const char *device_config = "cpu";
   bool reconstruct = false;
   bool mfem = false;
   bool visit = false;
   bool paraview = false;
   bool visualization = true;
   bool analytic = false;
   bool par_format = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&sx, "-sx", "--size-x",
                  "Size along x axis.");
   args.AddOption(&sy, "-sy", "--size-y",
                  "Size along y axis.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels (automatic to 10000 elements by default)");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels (default 0)");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&upwinded, "-up", "--upwinded", "-ce", "--centered",
                  "Switches between upwinded (1) and centered (0=default) stabilization.");
   args.AddOption(&reaction, "-rx", "--reaction",
                  "How problem 10's reaction term is discretised: "
                  "1=interpolatory (CCSZ, the default), 2=quadrature. "
                  "The two solve the SAME continuous problem and the "
                  "quadrature arm is the control the interpolatory one is "
                  "measured against; ignored by every other problem.");
   args.AddOption(&tau0, "-tau0", "--stab-const",
                  "Replace the HDG diffusion stabilization by this CONSTANT. "
                  "Zero (the default) keeps the built-in kappa/h. It is not a "
                  "tuning knob: CCSZ's k+2 superconvergence of the "
                  "postprocessed potential is stated for an O(1) tau and is "
                  "lost at kappa/h -- measured rates both ways are on "
                  "HDGPotentialPostprocessor.");
   args.AddOption(&postprocess, "-pp", "--postprocess", "-no-pp",
                  "--no-postprocess",
                  "Compute the local postprocessing u* of the potential into "
                  "the enriched space of degree order+1 and report its error. "
                  "This is the quantity that superconverges at k+2, so it is "
                  "the one a convergence ladder for CCSZ wants.");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=steady diff\n\t\t"
                  "2=steady adv-diff\n\t\t"
                  "3=steady adv\n\t\t"
                  "4=nonsteady adv-diff\n\t\t"
                  "5=Kovasznay flow\n\t\t"
                  "6=steady Burgers\n\t\t"
                  "7=nonsteady Burgers\n\t\t"
                  "8=steady linear kappa\n\t\t"
                  "9=nonsteady linear kappa\n\t\t");
   args.AddOption(&tf, "-tf", "--time-final",
                  "Final time.");
   args.AddOption(&nt, "-nt", "--ntimesteps",
                  "Number of time steps.");
   args.AddOption(&ode, "-ode", "--ode-solver",
                  "ODE time solver (1=Bacward Euler, 2=RK23L, 3=RK23A, 4=RK34).");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&pars.c, "-c", "--velocity",
                  "Convection velocity");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&bc_neumann, "-bcn", "--bc-neumann", "-no-bcn",
                  "--no-bc-neumann", "Enable Neumann outflow boundary condition.");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&trace_ess_bc, "-trbc", "--trace-ess-bc", "-no-trbc",
                  "--no-trace-ess-bc", "Switch between essential and weak trace BC.");
   args.AddOption(&nonlinear, "-nl", "--nonlinear", "-no-nl",
                  "--no-nonlinear", "Enable non-linear regime.");
   args.AddOption(&nonlinear_flux, "-nlu", "--nonlinear-flux", "-no-nlu",
                  "--no-nonlinear-flux", "Enable non-linear regime of flux.");
   args.AddOption(&nonlinear_pot, "-nlp", "--nonlinear-pot", "-no-nlp",
                  "--no-nonlinear-pot", "Enable non-linear regime of potential.");
   args.AddOption(&nonlinear_conv, "-nlc", "--nonlinear-convection", "-no-nlc",
                  "--no-nonlinear-convection", "Enable non-linear convection regime.");
   args.AddOption(&nonlinear_diff, "-nld", "--nonlinear-diffusion", "-no-nld",
                  "--no-nonlinear-diffusion", "Enable non-linear diffusion regime.");
   args.AddOption(&hdg_scheme, "-hdg", "--hdg_scheme",
                  "HDG scheme (1=HDG-I, 2=HDG-II, 3=Rusanov, 4=Godunov).");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rtol",
                  "Relative tolerance of the outer nonlinear solver. "
                  "Negative keeps the default, which is 1e-6 and is loose "
                  "enough that a mildly nonlinear problem stops after one "
                  "step.");
   args.AddOption(&use_npc, "-npc", "--npc", "-no-npc", "--no-npc",
                  "Solve by NPC -- Newton on the full (q, u, u_hat) system "
                  "with the Jacobian solved by hybridized elimination -- "
                  "instead of on the trace alone. Every local operation is "
                  "then one linear solve against one factorisation and "
                  "GetNumLocalNLIterations() stays at zero, and the "
                  "convergence test is on the FULL residual, so an -nrtol "
                  "that was adequate before may not be. In parallel only the "
                  "trace is communicated: the L2 flux and potential are "
                  "rank-local. Needs -hb, a nonlinear problem, a Newton-type "
                  "-nls, and a DISCONTINUOUS flux -- so not the H(div) "
                  "default, but -brt IS allowed and works: broken RT reports "
                  "DISCONTINUOUS and has no inter-element continuity, so it "
                  "is not the conforming scatter the guard is about. Measured "
                  "on -p 2 -o 1 -brt -hb -nl: identical to the reduced route "
                  "to every printed digit, 0 local nonlinear iterations "
                  "against 256. This text used to claim -brt was refused; it "
                  "never was.");
   args.AddOption(&gradient_mode, "-gm", "--gradient-mode",
                  "How much of the hybridized trace system to build: "
                  "0=assemble and precondition directly, 1=assemble and "
                  "precondition with GS, 2=do not assemble, apply it and "
                  "solve unpreconditioned. Negative leaves it alone. "
                  "Only meaningful with -hb, and only where the solver asks "
                  "for a gradient at all -- LBFGS does not.");
   args.AddOption(&solver_type, "-nls", "--nonlinear-solver",
                  "Nonlinear solver type (1=LBFGS, 2=LBB, 3=Newton, 4=KINSol).");
   args.AddOption(&prec_type, "-prec", "--preconditioner",
                  "Serial preconditioner where the code offers a choice "
                  "(0=the build's own, 1=iterative/GS, 2=direct/UMFPack). The "
                  "choice used to be compile-time only, so a build could not "
                  "reproduce a reference recording the other one and the "
                  "regression suite SKIPPED the case -- 49 of 157 serial "
                  "references, all through the Schur preconditioner. "
                  "Default 0 reproduces the build's behaviour exactly; "
                  "2 aborts without SuiteSparse. Ignored in parallel, where "
                  "HypreBoomerAMG offers no choice, and under -pa, which has "
                  "no matrix to factor.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&reconstruct, "-rec", "--reconstruct", "-no-rec",
                  "--no-reconstruct",
                  "Enable or disable quantities reconstruction.");
   args.AddOption(&mfem, "-mfem", "--mfem", "-no-mfem",
                  "--no-mfem",
                  "Enable or disable MFEM output.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable Visit output.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&analytic, "-anal", "--analytic", "-no-anal",
                  "--no-analytic",
                  "Enable or disable analytic solution.");
   args.AddOption(&par_format, "-pf", "--parallel-format", "-sf",
                  "--serial-format",
                  "Format to use when saving the results for VisIt.");

   args.ParseCheck();

   if (prec_type < 0 || prec_type > 2)
   {
      cerr << "-prec must be 0 (build default), 1 (iterative) or 2 (direct)"
           << endl;
      return 1;
   }

   // 3. Set the problem options
   pars.prob = (Problem)iproblem;
   const Problem &problem = pars.prob;
   bool bconv = false, bnlconv = false, bnldiff = nonlinear_diff, btime = false;
   bool breaction = false;
   switch (problem)
   {
      case Problem::SteadyDiffusion:
         break;
      case Problem::SteadyReactionDiffusion:
         breaction = true;
         break;
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
         btime = true;
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyAdvection:
         bconv = !nonlinear_conv;
         bnlconv = nonlinear_conv;
         break;
      case Problem::NonsteadyBurgers:
         btime = true;
      case Problem::SteadyBurgers:
         bnlconv = true;
         break;
      case Problem::NonsteadyLinearKappa:
         btime = true;
      case Problem::SteadyLinearKappa:
         bnldiff = true;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   if (nonlinear)
   {
      nonlinear_flux = nonlinear_pot = true;
   }

   if (trace_ess_bc && !dg && !brt)
   {
      cerr << "Essential trace BC does not work with continuous elements" << endl;
      return 1;
   }

   if (trace_ess_bc && (nonlinear_flux || nonlinear_pot))
   {
      cerr << "Essential trace BC is not implemented for non-linear forms" << endl;
      return 1;
   }

   if (bnldiff && reduction)
   {
      cerr << "Reduction is not possible with non-linear diffusion" << endl;
      return 1;
   }

   if (!bconv && !bnlconv && upwinded)
   {
      cerr << "Upwinded scheme cannot work without advection" << endl;
      return 1;
   }

   if (bnlconv && !nonlinear_pot)
   {
      cerr << "Nonlinear convection can only work in the nonlinear regime" << endl;
      return 1;
   }

   if (breaction && !hybridization)
   {
      cerr << "Problem 10's reaction term is a block nonlinear integrator, "
           "which only the hybridized route assembles; add -hb" << endl;
      return 1;
   }

   if (breaction && reaction != 1 && reaction != 2)
   {
      cerr << "-rx must be 1 (interpolatory) or 2 (quadrature)" << endl;
      return 1;
   }

   // LBFGS never calls GetGradient(), so it cannot see the (1,0) block this
   // term contributes and converges linearly at best. Refused rather than
   // left to look slow, which is the same treatment -npc already gets.
   if (breaction && solver_type == 1)
   {
      cerr << "Problem 10 needs a gradient-using solver; LBFGS (-nls 1) never "
           "calls GetGradient(). Use -nls 3" << endl;
      return 1;
   }

   if (!breaction && !postprocess && tau0 > 0.)
   {
      // Not an error -- a constant tau is meaningful for any HDG problem --
      // but say it, because it silently changes the method.
      cout << "NOTE: -tau0 replaces the built-in kappa/h stabilization\n";
   }

   if (btime && nt <= 0)
   {
      cerr << "You must specify the number of time steps for time evolving problems"
           << endl;
      return 1;
   }

   // 4. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (root) { device.Print(); }

   // 5. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   if (ny <= 0)
   {
      ny = nx;
   }

   Mesh mesh;
   if (!mesh_file.empty())
   {
      mesh = Mesh(mesh_file, 1, 1);
   }
   else
   {
      mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                   sx, sy);
   }

   int dim = mesh.Dimension();

   // 6. Mark boundary conditions based on the selected problem
   const int bdr_attrs = mesh.bdr_attributes.Size() > 0 ?
                         mesh.bdr_attributes.Max() : 1;
   Array<int> bdr_is_dirichlet(bdr_attrs);
   Array<int> bdr_is_neumann(bdr_attrs);
   bdr_is_dirichlet = 0;
   bdr_is_neumann = 0;

   switch (problem)
   {
      case Problem::SteadyDiffusion:
      case Problem::SteadyReactionDiffusion:
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         // Free BC (zero Dirichlet)
         if (bc_neumann)
         {
            bdr_is_neumann[1] = -1; // Outflow
            bdr_is_neumann[2] = -1; // Outflow
         }
         break;
      case Problem::SteadyAdvection:
         bdr_is_dirichlet[3] = -1; // Inflow
         bdr_is_neumann[0] = -1; // Outflow
         break;
      case Problem::NonsteadyAdvectionDiffusion:
         bdr_is_dirichlet = -1;
         //bdr_is_neumann = -1;
         break;
      case Problem::KovasznayFlow:
         bdr_is_neumann = -1; // Outflow
         bdr_is_neumann[3] = 0; // Inflow
         break;
   }

   Array<int> bdr_is_free(bdr_attrs);
   for (int a = 0; a < bdr_attrs; a++)
   {
      bdr_is_free[a] = !bdr_is_dirichlet[a] && !bdr_is_neumann[a];
   }

   // 7. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (!mesh_file.empty())
   {
      int ref_levels = (serial_ref_levels >= 0)?(serial_ref_levels):
                       (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 8. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < parallel_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 9. Define a finite element space on the mesh. Here we use the
   //    (broken) Raviart-Thomas finite elements of the specified order for the
   //    heat flux or discontinuous Galerkin alternatively. The temperature is
   //    always discretized by discontinuous Galerkin elements.
   unique_ptr<FiniteElementCollection> V_coll; // Heat flux FE collection
   unique_ptr<FiniteElementCollection> V_coll_dg; // DG heat flux FE colection
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      V_coll = make_unique<L2_FECollection>(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      V_coll = make_unique<BrokenRT_FECollection>(order, dim);
      // For broken Raviart-Thomas elements, we define an auxiliary DG space
      // for visualization with an older version of GLVIs without support of
      // this element family.
      V_coll_dg = make_unique<L2_FECollection>(order+1, dim);
   }
   else
   {
      V_coll = make_unique<RT_FECollection>(order, dim);
   }

   // Temperature FE collection
   auto W_coll = make_unique<L2_FECollection>(order, dim, BasisType::GaussLobatto);

   // Heat flux FE space
   auto V_space = make_unique<ParFiniteElementSpace>(&pmesh, V_coll.get(),
                                                     (dg)?(dim):(1));
   auto V_space_dg = (V_coll_dg)?(make_unique<ParFiniteElementSpace>(
                                     &pmesh, V_coll_dg.get(), dim)):(nullptr);
   // Temperature FE space
   auto W_space = make_unique<ParFiniteElementSpace>(&pmesh, W_coll.get());

   // Darcy form
   auto darcy = make_unique<ParDarcyForm>(V_space.get(), W_space.get());

   // 10. Define the coefficients, analytical solution, and rhs of the PDE.
   pars.t_0 = 1.; // Base temperature

   ConstantCoefficient kcoeff(pars.k); // Conductivity
   ConstantCoefficient ikcoeff(1./pars.k); // Inverse conductivity

   auto cFun = GetCFun(pars);
   VectorFunctionCoefficient ccoeff(dim, cFun); // Velocity
   NormalizedVectorCoefficient nccoeff(ccoeff); // Normalized velocity

   auto tFun = GetTFun(pars);
   FunctionCoefficient tcoeff(tFun); // Analytic temperature
   ProductCoefficient gcoeff(-1., tcoeff); // Boundary heat flux r.h.s.

   auto fFun = GetFFun(pars);
   FunctionCoefficient fcoeff(fFun); // Temperature r.h.s.

   auto qFun = GetQFun(pars);
   VectorFunctionCoefficient qcoeff(dim, qFun); // Analytic heat flux
   ConstantCoefficient one;
   VectorSumCoefficient qtcoeff_(ccoeff, qcoeff, tcoeff,
                                 one); // Analytic total flux
   VectorCoefficient &qtcoeff = (bconv)?((VectorCoefficient&)qtcoeff_)
                                :((VectorCoefficient&)qcoeff); //<-- velocity is undefined

   // 11. Assemble the finite element matrices for the Darcy operator
   //
   //                     ┌        ┐
   //                     | Mq -Bᵀ |
   //                     | B  D   |
   //                     └        ┘
   //     where:
   //     RTDG:
   //     Mq = (kˉ¹ q, v)                            q, v ∈ V
   //     B = (∇⋅q, w)                               q ∈ V, w ∈ W
   //     D = (1/dt T, w) - (c T, ∇w) - <c⋅n{T}, [w]>  T, w ∈ W
   //     LBRT:
   //     Mq = (kˉ¹ q, v)                            q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>                 q ∈ V, w ∈ W
   //     D = (1/dt T, w) - (c T, ∇w) - <c⋅n{T}, [w]>  T, w ∈ W
   //     LDG:
   //     Mq = (kˉ¹ q, v)                            q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>                 q ∈ V, w ∈ W
   //     D = (1/dt T, w) + <td k hˉ¹[T], [w]> -
   //         - (c T, ∇w) - <c⋅n{T}, [w]>             T, w ∈ W

   // Diffusion

   unique_ptr<MixedFluxFunction> HeatFluxFun;
   if (!bnldiff)
   {
      // Linear diffusion
      if (!nonlinear_flux)
      {
         ParBilinearForm *Mq = darcy->GetParFluxMassForm();
         if (dg)
         {
            Mq->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
         }
         else
         {
            Mq->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
         }
      }
      else
      {
         ParNonlinearForm *Mqnl = darcy->GetParFluxMassNonlinearForm();
         if (dg)
         {
            Mqnl->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
         }
         else
         {
            Mqnl->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
         }
      }
   }
   else
   {
      // Nonlinear diffusion
      ParBlockNonlinearForm *Mnl = darcy->GetParBlockNonlinearForm();
      HeatFluxFun = GetHeatFluxFun(pars, dim);
      if (dg)
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
         if (upwinded && td > 0. && !hybridization)
         {
            Mnl->AddInteriorFaceIntegrator(new MixedConductionNLFIntegrator(
                                              *HeatFluxFun, ccoeff, td));
            Mnl->AddBdrFaceIntegrator(new MixedConductionNLFIntegrator(
                                         *HeatFluxFun, ccoeff, td), bdr_is_neumann);
         }
         else if (!upwinded && td > 0.)
         {
            Mnl->AddInteriorFaceIntegrator(new MixedConductionNLFIntegrator(
                                              *HeatFluxFun, td));
            Mnl->AddBdrFaceIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun, td),
                                      bdr_is_neumann);
         }
      }
      else
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
         if (brt)
         {
            MFEM_ABORT("Not implemented");
         }
      }
   }

   // Diffusion stabilization

   // One hook object for every face integrator below. It outlives them: the
   // forms hold raw pointers to it and are destroyed with `darcy`, which is
   // declared above this.
   unique_ptr<ConstStabilization> const_stab;
   if (tau0 > 0.) { const_stab = make_unique<ConstStabilization>(tau0); }
   // Every HDGDiffusionIntegrator in this section goes through here, so a
   // constant tau cannot reach some faces and not others -- which is the
   // shape of bug a per-site edit would have introduced across ten call
   // sites, and it would have looked like a stabilization that half works.
   auto with_stab = [&](HDGDiffusionIntegrator *i)
   {
      if (const_stab) { i->SetStabilization(*const_stab); }
      return i;
   };

   if (dg && (!bnldiff || hybridization) && td > 0.)
   {
      if (!nonlinear_pot)
      {
         ParBilinearForm *Mt = darcy->GetParPotentialMassForm();
         if (upwinded && hybridization)
         {
            Mt->AddInteriorFaceIntegrator(with_stab(new HDGDiffusionIntegrator(ccoeff,
                                                                               kcoeff, td)));
            Mt->AddBdrFaceIntegrator(with_stab(new HDGDiffusionIntegrator(ccoeff, kcoeff,
                                                                          td)),
                                     bdr_is_neumann);
            if (trace_ess_bc)
            {
               Mt->AddBdrFaceIntegrator(with_stab(new HDGDiffusionIntegrator(ccoeff, kcoeff,
                                                                             td)),
                                        bdr_is_dirichlet);
            }
         }
         else if (!upwinded)
         {
            Mt->AddInteriorFaceIntegrator(with_stab(new HDGDiffusionIntegrator(kcoeff,
                                                                               td)));
            Mt->AddBdrFaceIntegrator(with_stab(new HDGDiffusionIntegrator(kcoeff, td)),
                                     bdr_is_neumann);
            if (trace_ess_bc)
            {
               Mt->AddBdrFaceIntegrator(with_stab(new HDGDiffusionIntegrator(kcoeff, td)),
                                        bdr_is_dirichlet);
            }
         }
      }
      else
      {
         ParNonlinearForm *Mtnl = darcy->GetParPotentialMassNonlinearForm();
         if (upwinded && hybridization)
         {
            Mtnl->AddInteriorFaceIntegrator(with_stab(new HDGDiffusionIntegrator(ccoeff,
                                                                                 kcoeff, td)));
            Mtnl->AddBdrFaceIntegrator(with_stab(new HDGDiffusionIntegrator(ccoeff, kcoeff,
                                                                            td)),
                                       bdr_is_neumann);
         }
         else if (!upwinded)
         {
            Mtnl->AddInteriorFaceIntegrator(with_stab(new HDGDiffusionIntegrator(kcoeff,
                                                                                 td)));
            Mtnl->AddBdrFaceIntegrator(with_stab(new HDGDiffusionIntegrator(kcoeff, td)),
                                       bdr_is_neumann);
         }
      }
   }

   // Divergence/weak gradient

   ParMixedBilinearForm *B = darcy->GetParFluxDivForm();
   if (dg)
   {
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   }
   else
   {
      B->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   }

   if (dg || brt)
   {
      if (upwinded)
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(ccoeff, -1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            ccoeff, -1.)), bdr_is_neumann);
         if (hybridization && trace_ess_bc)
         {
            B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                               ccoeff, -1.)), bdr_is_dirichlet);
         }
      }
      else
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(-1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            -2.)), bdr_is_neumann);
         if (hybridization && trace_ess_bc)
         {
            B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                               -2.)), bdr_is_dirichlet);
         }
      }
   }

   // Linear convection

   if (bconv)
   {
      if (!nonlinear_pot)
      {
         ParBilinearForm *Mt = darcy->GetParPotentialMassForm();
         Mt->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
         if (upwinded)
         {
            Mt->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
            if (hybridization && trace_ess_bc)
            {
               Mt->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff),
                                        bdr_is_neumann);
            }
            else
            {
               Mt->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
            }
         }
         else
         {
            Mt->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
            if (hybridization)
            {
               // centered scheme does not work with Dirichlet when hybridized,
               // giving an diverging system, we use the full BC flux here
               Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                        bdr_is_neumann);
            }
            else
            {
               // we use averaged interior + BC flux for Dirichlet for stability
               // reasons and full BC flux for Neumann
               Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                        bdr_is_dirichlet);
               Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                        bdr_is_free);
            }
         }
      }
      else
      {
         ParNonlinearForm *Mtnl = darcy->GetParPotentialMassNonlinearForm();
         Mtnl->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
         if (upwinded)
         {
            Mtnl->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
            Mtnl->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
         }
         else
         {
            Mtnl->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
            if (hybridization)
            {
               //centered scheme does not work with Dirichlet when hybridized,
               //giving an diverging system, we use the full BC flux here
               Mtnl->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                          bdr_is_neumann);
            }
            else
            {
               Mtnl->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                          bdr_is_dirichlet);
               Mtnl->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                          bdr_is_free);
            }
         }
      }
   }

   // Nonlinear convection in the nonlinear regime

   unique_ptr<FluxFunction> FluxFun;
   unique_ptr<NumericalFlux> FluxSolver;
   if (bnlconv && nonlinear_pot)
   {
      FluxFun = GetFluxFun(pars, ccoeff);
      switch (hdg_scheme)
      {
         case 1: FluxSolver = make_unique<HDGFlux>(*FluxFun, HDGFlux::HDGScheme::HDG_1);
            break;
         case 2: FluxSolver = make_unique<HDGFlux>(*FluxFun, HDGFlux::HDGScheme::HDG_2);
            break;
         case 3: FluxSolver = make_unique<RusanovFlux>(*FluxFun); break;
         case 4: FluxSolver = make_unique<ComponentwiseUpwindFlux>(*FluxFun); break;
         default:
            cerr << "Unknown HDG scheme" << endl;
            exit(1);
      }
      ParNonlinearForm *Mtnl = darcy->GetParPotentialMassNonlinearForm();
      Mtnl->AddDomainIntegrator(new HyperbolicFormIntegrator(*FluxSolver, 0, -1.));
      Mtnl->AddInteriorFaceIntegrator(new HyperbolicFormIntegrator(
                                         *FluxSolver, 0, -1.));
      Mtnl->AddBdrFaceIntegrator(new HyperbolicFormIntegrator(
                                    *FluxSolver, 0, -1.));
   }

   // Inertial term
   // Note the mass integrator is added in DarcyOperator, but we must construct
   // the corresponding bilinear form here
   if (btime)
   {
      if (!nonlinear_pot) { darcy->GetParPotentialMassForm(); }
      else { darcy->GetParPotentialMassNonlinearForm(); }
   }

   // Reaction term (problem 10)
   //
   // **Everything here has to be in place BEFORE EnableHybridization().**
   // DarcyForm reads the forms at that call: which blocks exist, which face
   // integrators are linear, and -- through GetBlockRowMask() -- whether the
   // flux mass can stay factored once. An integrator added afterwards is
   // silently never seen. Same trap as the boundary flux term in
   // navierstokes.cpp.
   unique_ptr<L2_FECollection> S_coll;
   unique_ptr<ParFiniteElementSpace> S_space;
   unique_ptr<HDGPostprocessBlocks> pp_blocks;
   CubicReaction Freact;

   // The enriched space is needed by the reaction term AND by -pp, and they
   // must be the SAME one: the interpolatory method's nodes are that space's
   // nodes, so a postprocessing run on a different enrichment would not be
   // reporting the u* the residual was built from.
   if (breaction || postprocess)
   {
      // The DEFAULT L2 basis, which is open (Gauss-Legendre), rather than the
      // closed one W_coll uses. CCSZ's interpolation nodes are an open set
      // (section 6.5 of doc/HDG-INTERPOLATORY-CCSZ.md), and a closed basis
      // puts nodes on the element boundary where a reaction with a singular
      // factor -- meq's F/r on the symmetry axis -- is not evaluable. The
      // node set is a free choice with an unmeasured effect on the answer,
      // which is why it is stated here rather than inherited.
      S_coll = make_unique<L2_FECollection>(order + 1, dim);
      S_space = make_unique<ParFiniteElementSpace>(&pmesh, S_coll.get());

      pp_blocks = make_unique<HDGPostprocessBlocks>(*V_space, *W_space,
                                                    *S_space);
      pp_blocks->SetDiffusionInverse(ikcoeff);
      pp_blocks->Assemble();
   }

   if (breaction)
   {
      // The form OWNS its domain integrators and deletes them.
      HDGReactionIntegratorBase *react =
         (reaction == 1)
         ? (HDGReactionIntegratorBase*)
         new HDGInterpolatoryReactionIntegrator(Freact, *pp_blocks)
         : (HDGReactionIntegratorBase*)
         new HDGQuadratureReactionIntegrator(Freact, *pp_blocks);
      react->Assemble();
      darcy->GetBlockNonlinearForm()->AddDomainIntegrator(react);
   }

   // Set hybridization / reduction / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg && !brt)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   unique_ptr<FiniteElementCollection> trace_coll;
   unique_ptr<ParFiniteElementSpace> trace_space;

   if (hybridization)
   {
      // Hybridization
      chrono.Clear();
      chrono.Start();

      if (trace_h1)
      {
         trace_coll = make_unique<H1_Trace_FECollection>(max(order, 1), dim);
      }
      else
      {
         trace_coll = make_unique<DG_Interface_FECollection>(order, dim);
      }
      trace_space = make_unique<ParFiniteElementSpace>(&pmesh, trace_coll.get());
      darcy->EnableHybridization(trace_space.get(),
                                 new NormalTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);
      // Set essential BC
      if (trace_ess_bc)
      {
         darcy->GetHybridization()->SetEssentialBC(bdr_is_dirichlet);
      }
      if (gradient_mode >= 0)
      {
         darcy->GetHybridization()->SetGradientMode(
            (gradient_mode == 2)
            ? DarcyHybridization::GradientMode::MatrixFree
            : DarcyHybridization::GradientMode::Assembled);
      }
      chrono.Stop();
      if (root) { cout << "Hybridization init took " << chrono.RealTime() << "s.\n"; }
   }
   else if (reduction)
   {
      // Reduction
      chrono.Clear();
      chrono.Start();

      if (dg || brt)
      {
         darcy->EnableFluxReduction();
      }
      // UNREACHABLE, exactly as in convdiff.cpp -- see the note there. No
      // problem in this miniapp is transient without also being convective
      // or nonlinearly diffusive, so -rd never reaches potential reduction.
      else if (!bconv && !bnlconv && btime)
      {
         darcy->EnablePotentialReduction(ess_flux_tdofs_list);
      }
      else
      {
         cerr << "No possible reduction!" << endl;
         return 1;
      }

      chrono.Stop();
      if (root) { cout << "Reduction init took " << chrono.RealTime() << "s.\n"; }
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // 12. Define the block structure of the problem, i.e. define the array of
   //     offsets for each variable. The last component of the Array is the sum
   //     of the dimensions of each block.
   const Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

   if (root)
   {
      cout << "***********************************************************\n";
      cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
      cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
      if (hybridization)
      {
         cout << "dim(M) = " << block_offsets[3] - block_offsets[2] << "\n";
         cout << "dim(V+W+M) = " << block_offsets.Last() << "\n";
      }
      else
      {
         cout << "dim(V+W) = " << block_offsets.Last() << "\n";
      }
      cout << "***********************************************************\n";
   }

   // 13. Allocate memory (x, rhs) for the analytical solution and the right
   //     hand side. Define the GridFunction q_h, t_h for the finite element
   //     solution and linear forms fform and gform for the right hand side.
   //     The data allocated by x and rhs are passed as a reference to the grid
   //     functions (q,t) and the linear forms (fform, gform). With
   //     hybridization, linear form hform for the constraint is constructed
   //     as well together with the trace grid function tr_h.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   x = 0.;
   ParGridFunction q_h, t_h, tr_h, qt_h, q_hs, t_hs, tr_hs;
   q_h.MakeRef(V_space.get(), x.GetBlock(0), 0);
   t_h.MakeRef(W_space.get(), x.GetBlock(1), 0);
   if (hybridization)
   {
      tr_h.MakeRef(trace_space.get(), x.GetBlock(2), 0);
   }

   if (btime)
   {
      t_h.ProjectCoefficient(tcoeff); // Initial condition
   }

   if (!dg && !brt)
   {
      q_h.ProjectBdrCoefficientNormal(qcoeff,
                                      bdr_is_neumann);   // Essential Neumann BC
   }

   if (hybridization && trace_ess_bc)
   {
      tr_h.ProjectBdrCoefficient(tcoeff, bdr_is_dirichlet); // Essential Dirichlet BC
   }

   // Flux r.h.s.
   unique_ptr<ParLinearForm> gform(new ParLinearForm);
   gform->Update(V_space.get(), rhs.GetBlock(0), 0);

   if (!hybridization || !trace_ess_bc)
   {
      // Dirichlet BC
      if (dg)
      {
         gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(gcoeff),
                                     bdr_is_dirichlet);
      }
      else if (brt)
      {
         gform->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                     bdr_is_dirichlet);
      }
      else
      {
         gform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                      bdr_is_dirichlet);
      }
   }

   // Potential r.h.s.
   unique_ptr<ParLinearForm> fform(new ParLinearForm);
   fform->Update(W_space.get(), rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   if (!hybridization)
   {
      // Neumann BC (non-hybridized)
      if (dg || brt)
      {
         if (upwinded)
         {
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(qcoeff, nccoeff, +1.,
                                                                   -0.5), bdr_is_neumann);
         }
         else
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qcoeff, +2., 0.),
                                        bdr_is_neumann);
      }
      if (bconv)
      {
         if (upwinded)
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1.),
                                        bdr_is_neumann);
         else
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +2., 0.),
                                        bdr_is_neumann);
      }
   }

   // Dirichlet BC (convective)
   if (bconv)
   {
      if (upwinded)
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1.),
                                     bdr_is_dirichlet);
      else
      {
         if (hybridization)
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +2., 0.),
                                        bdr_is_dirichlet);//<-- full BC flux, see above
         else
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1., 0.),
                                        bdr_is_dirichlet);//<-- half BC flux, see above
      }
   }

   // Constraint r.h.s.
   unique_ptr<ParLinearForm> hform;

   if (hybridization)
   {
      // Neumann BC for the hybridized system
      hform = make_unique<ParLinearForm>();
      hform->Update(trace_space.get(), rhs.GetBlock(2), 0);
      //note that Neumann BC must be applied only for the heat flux
      //and not the total flux for stability reasons
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(qcoeff, 2),
                                   bdr_is_neumann);
   }

   // 14. Construct the spatial operator

   DarcyOperator op(ess_flux_tdofs_list, darcy.get(),
   {gform.get(), fform.get(), hform.get()},
   {&gcoeff, &fcoeff, &qtcoeff},
   (DarcyOperator::SolverType) solver_type, false, btime);
   op.SetTraceSolveLevel(gradient_mode);
   op.SetPrecType((DarcyOperator::PrecType) prec_type);
   if (use_npc) { op.SetNPC(); }
   if (newton_rtol > 0.) { op.SetTolerance(newton_rtol); }

   // 15. Construct the time ODE solver
   unique_ptr<ODESolver> ode_solver;

   switch (ode)
   {
      case 1: ode_solver = make_unique<BackwardEulerSolver>(); break;
      case 2: ode_solver = make_unique<SDIRK23Solver>(2); break;
      case 3: ode_solver = make_unique<SDIRK23Solver>(); break;
      case 4: ode_solver = make_unique<SDIRK34Solver>(); break;
      default:
         MFEM_ABORT("Unknown solver");
         return 1;
   }

   ode_solver->Init(op);

   // 16. The main time loop

   if (!btime) { nt = 1; }

   const real_t dt = tf / nt; // Time step

   int i_Kovasznay = 0; // Injection iteration - Kovasznay flow
   constexpr real_t dt_Kovasznay = 2.; // Injection period - Kovasznay flow

   for (int ti = 0; ti < nt; ti++)
   {
      // Set the current time

      real_t t = tf * ti / nt;

      // Perform injection - Kovasznay flow
      if (problem == Problem::KovasznayFlow &&
          t >= ((i_Kovasznay+1) * dt_Kovasznay) * (1. - 100*epsilon))
      {
         i_Kovasznay++;
         GridFunction t_Kovasznay(W_space.get());
         t_Kovasznay.ProjectCoefficient(tcoeff);
         t_h += t_Kovasznay;
      }

      // 17. Perform time step

      real_t dt_ = dt; //<--- ignore time step changes
      ode_solver->Step(x, t, dt_);

      // 18. Compute the L2 error norms.

      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      real_t err_q  = q_h.ComputeL2Error(qcoeff, irs);
      real_t norm_q = ComputeGlobalLpNorm(2., qcoeff, pmesh, irs);
      real_t err_t  = t_h.ComputeL2Error(tcoeff, irs);
      real_t norm_t = ComputeGlobalLpNorm(2., tcoeff, pmesh, irs);

      // The local postprocessing, BEFORE the two lines below. The regression
      // script reads the flux and potential errors off the LAST two lines of
      // output and the solver line at [-4], so anything printed after them
      // breaks every existing reference; this line is found by its own prefix
      // instead. See regression_test.py's get_postproc().
      if (postprocess)
      {
         // A space of its own is not built here: pp_blocks was assembled on
         // S_space above, and under -rx the reaction's nodes ARE that space's
         // nodes. Handing Compute() a different enrichment would report a u*
         // the solve never saw.
         ParGridFunction t_hpp(S_space.get());
         HDGPotentialPostprocessor pp(q_h, t_h);
         pp.SetDiffusionInverse(ikcoeff);
         pp.Compute(t_hpp);
         const real_t err_tpp = t_hpp.ComputeL2Error(tcoeff, irs);
         if (root)
         {
            cout << "|| t_h* - t_ex || / || t_ex || = "
                 << err_tpp / norm_t << "\n";
         }
      }

      if (root)
      {
         if (btime)
         {
            cout << "iter:\t" << ti
                 << "\ttime:\t" << t
                 << "\tq_err:\t" << err_q / norm_q
                 << "\tt_err:\t" << err_t / norm_t
                 << endl;
         }
         else
         {
            cout << "|| q_h - q_ex || / || q_ex || = " << err_q / norm_q << "\n";
            cout << "|| t_h - t_ex || / || t_ex || = " << err_t / norm_t << "\n";
         }
      }

      if (reconstruct)
      {
         darcy->Reconstruct(x, x.GetBlock(2), qt_h, q_hs, t_hs, tr_hs);
         real_t err_qt = qt_h.ComputeL2Error(qtcoeff, irs);
         real_t norm_qt = ComputeGlobalLpNorm(2., qtcoeff, pmesh, irs);
         real_t err_qs = q_hs.ComputeL2Error(qcoeff, irs);
         real_t err_ts = t_hs.ComputeL2Error(tcoeff, irs);
         if (root)
         {
            cout << "|| qt_h - qt_ex || / || qt_ex || = " << err_qt / norm_qt << "\n";
            cout << "|| q_hs - q_ex || / || q_ex || = " << err_qs / norm_q << "\n";
            cout << "|| t_hs - t_ex || / || t_ex || = " << err_ts / norm_t << "\n";
         }
      }

      // 19. Project the fluxes

      ParGridFunction q_vh;

      if (V_space_dg)
      {
         VectorGridFunctionCoefficient coeff(&q_h);
         q_vh.SetSpace(V_space_dg.get());
         q_vh.ProjectCoefficient(coeff);
      }
      else
      {
         q_vh.MakeRef(V_space.get(), q_h, 0);
      }

      // 20. Project the analytic solution

      static ParGridFunction q_a, qt_a, t_a, c_gf;

      q_a.SetSpace((V_space_dg)?(V_space_dg.get()):(V_space.get()));
      q_a.ProjectCoefficient(qcoeff);

      qt_a.SetSpace((V_space_dg)?(V_space_dg.get()):(V_space.get()));
      qt_a.ProjectCoefficient(qtcoeff);

      t_a.SetSpace(W_space.get());
      t_a.ProjectCoefficient(tcoeff);

      if (bconv)
      {
         c_gf.SetSpace((V_space_dg)?(V_space_dg.get()):(V_space.get()));
         c_gf.ProjectCoefficient(ccoeff);
      }

      // 21. Save the mesh and the solution. This output can be viewed later
      //     using GLVis: "glvis -m convdiff.mesh -g sol_q.gf" or "glvis -m
      //     convdiff.mesh -g sol_t.gf".
      if (mfem)
      {
         stringstream ss;
         ss.str("");
         ss << "pconvdiff." << setfill('0') << setw(6) << myid;
         if (btime) { ss << "_" << ti; }
         ss << ".mesh";
         ofstream mesh_ofs(ss.str());
         mesh_ofs.precision(8);
         pmesh.Print(mesh_ofs);

         ss.str("");
         ss << "sol_q." << setfill('0') << setw(6) << myid;
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream q_ofs(ss.str());
         q_ofs.precision(8);
         q_vh.Save(q_ofs);

         ss.str("");
         ss << "sol_t." << setfill('0') << setw(6) << myid;
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream t_ofs(ss.str());
         t_ofs.precision(8);
         t_h.Save(t_ofs);
      }

      // 22. Save data in the VisIt format
      if (visit)
      {
         static VisItDataCollection visit_dc("Pconvdiff", &pmesh);
         if (ti == 0)
         {
            visit_dc.RegisterField("heat flux", &q_vh);
            visit_dc.RegisterField("temperature", &t_h);
            if (analytic)
            {
               visit_dc.RegisterField("heat flux analytic", &q_a);
               visit_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         visit_dc.SetFormat(!par_format ?
                            DataCollection::SERIAL_FORMAT :
                            DataCollection::PARALLEL_FORMAT);
         visit_dc.SetCycle(ti);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }

      // 23. Save data in the ParaView format
      if (paraview)
      {
         static ParaViewDataCollection paraview_dc("PConvdiff", &pmesh);
         if (ti == 0)
         {
            paraview_dc.SetPrefixPath("ParaView");
            paraview_dc.SetLevelsOfDetail(order);
            paraview_dc.SetDataFormat(VTKFormat::BINARY);
            paraview_dc.SetHighOrderOutput(true);
            paraview_dc.RegisterField("heat flux",&q_vh);
            paraview_dc.RegisterField("temperature",&t_h);
            if (analytic)
            {
               paraview_dc.RegisterField("heat flux analytic", &q_a);
               paraview_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         paraview_dc.SetCycle(ti);
         paraview_dc.SetTime(t);
         paraview_dc.Save();
      }

      // 24. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         static socketstream q_sock, t_sock;
         VisualizeField(q_sock, q_vh, "Heat flux", ti, root);
         VisualizeField(t_sock, t_h, "Temperature", ti, root);
         if (reconstruct)
         {
            static socketstream qt_sock, qs_sock, ts_sock;
            VisualizeField(qt_sock, qt_h, "Total flux", ti, root);
            VisualizeField(qs_sock, q_hs, "Recon. flux", ti, root);
            VisualizeField(ts_sock, t_hs, "Recon. temperature", ti, root);
         }
         if (analytic)
         {
            static socketstream qa_sock, qta_sock, ta_sock, c_sock;
            VisualizeField(qa_sock, q_a, "Heat flux analytic", ti, root);
            if (bconv || bnlconv)
            {
               VisualizeField(qa_sock, qt_a, "Total heat flux analytic", ti, root);
            }
            VisualizeField(ta_sock, t_a, "Temperature analytic", ti, root);
            if (bconv)
            {
               VisualizeField(c_sock, c_gf, "Velocity", ti, root);
            }
         }
      }
   }

   return 0;
}

TFunc GetTFun(const ProblemParams &params)
{
   const Problem &prob = params.prob;
   const real_t &k = params.k;
   const real_t &t_0 = params.t_0;
   const real_t &c = params.c;

   switch (prob)
   {
      case Problem::SteadyReactionDiffusion:
         // CCSZ Example 4.1's solution, which vanishes on the boundary of the
         // unit box -- so the "free BC (zero Dirichlet)" marking in section 5
         // is the exact boundary condition and not an approximation of one.
         return [=](const Vector &x, real_t) -> real_t
         {
            real_t t0 = t_0;
            for (int i = 0; i < x.Size(); i++) { t0 *= sin(M_PI * x(i)); }
            return t0;
         };
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * exp(x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            return t0;
         };
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            constexpr real_t x0 = 1.;
            constexpr real_t y0 = 1.;
            real_t denom = ((1. - exp(-c)) * (1. - exp(-c)));
            real_t t0 = (t_0 * x(0) * x(1) * (1. - exp(c*(x(0)-x0)) ) * (1. - exp(c*(x(1)-y0))
                                                                        )) / denom;
            return t0;
         };
      case Problem::SteadyAdvection:
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector xc(x);
            //xc -= .5;
            real_t t0 = 1. - tanh(10. * (-1. + 4.*xc.Norml2()));
            return t0;
         };
      case Problem::NonsteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const int vdim = x.Size();
            Vector xc(x);
            xc -= .5;
            Vector dx(vdim);
            const real_t ct = 4.*c*t * M_PI/4.;
            constexpr real_t dx_x = 0.2;
            constexpr real_t dx_y = 0.0;
            dx(0) = +xc(0) * cos(ct) + xc(1) * sin(ct) + dx_x;
            dx(1) = -xc(0) * sin(ct) + xc(1) * cos(ct) + dx_y;

            constexpr real_t sigma = 0.1;
            constexpr real_t sigma2 = 2*sigma*sigma;
            // 2 sigma^2 + 4 k t, NOT 4 k t pi/4. A Gaussian under
            // du/dt = k lap u spreads as D' = 4k -- substitute
            // u = (A/D) exp(-r^2/D) and both sides come out as
            // u (r^2/D - 1) times D'/D and 4k/D respectively -- so the pi/4
            // that the rotation legitimately carries does not belong here.
            // With it the exact solution is not a solution: the PDE residual
            // is O(k) and the miniapp converged, in dt AND in h, to a limit
            // 0.0121998 away from this function. See the note at the head of
            // this file.
            const real_t denom = sigma2 + 4.*k*t;
            return sigma2 / denom * exp(- (dx*dx) / denom);
         };
      case Problem::KovasznayFlow:
         return [=](const Vector &x, real_t t) -> real_t
         {
            Vector xc(x);
            xc(1) -= 1.25;
            constexpr real_t cx[] = {1., 1., 1.};
            constexpr real_t cy[] = {0., +.5, -.5};
            constexpr real_t sigma = .5;
            real_t w0 = 0.;
            for (int i = 0; i < 3; i++)
            {
               real_t dx = xc(0) - cx[i];
               real_t dy = xc(1) - cy[i];
               w0 += exp(-(dx*dx + dy*dy)/(sigma*sigma));
            }
            return w0;
         };
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t ux = x(0) * tanh((1.-x(0))/k);
            const real_t uy = x(1) * tanh((1.-x(1))/k);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            return u;
         };
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t ux = x(0) * tanh((1.-x(0))/k);
            const real_t uy = x(1) * tanh((1.-x(1))/k);
            const real_t ut = (prob == Problem::SteadyLinearKappa)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            return u;
         };
   }
   return TFunc();
}

VecTFunc GetQFun(const ProblemParams &params)
{
   const Problem &prob = params.prob;
   const real_t &k = params.k;
   const real_t &t_0 = params.t_0;
   const real_t &c = params.c;

   switch (prob)
   {
      case Problem::SteadyReactionDiffusion:
         // q = -k grad T, this miniapp's sign for the flux throughout.
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            for (int d = 0; d < vdim; d++)
            {
               real_t r = t_0 * M_PI * cos(M_PI * x(d));
               for (int j = 0; j < vdim; j++)
               {
                  if (j != d) { r *= sin(M_PI * x(j)); }
               }
               v(d) = -k * r;
            }
         };
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            v = 0.;
            v(0) = t_0 * (sin(M_PI*x(0)) + M_PI * cos(M_PI*x(0))) * exp(
                      x.Sum()) * sin(M_PI*x(1));
            v(1) = t_0 * (sin(M_PI*x(1)) + M_PI * cos(M_PI*x(1))) * exp(
                      x.Sum()) * sin(M_PI*x(0));
            if (vdim > 2)
            {
               v(0) *= sin(M_PI*x(2));
               v(1) *= sin(M_PI*x(2));
               v(2) = t_0 * (sin(M_PI*x(2)) + M_PI * cos(M_PI*x(2))) * exp(
                         x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            }

            v *= -k;
         };
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t, Vector &v)
         {
            constexpr real_t x0 = 1.;
            constexpr real_t y0 = 1.;
            real_t cdx = exp((x(0)-x0)*c);
            real_t cdy = exp((x(1)-y0)*c);
            real_t coef = (1. - cdx) * (1. - cdy);
            real_t denom = ((1. - exp(-c)) * (1. - exp(-c)));

            v(0) = (x(1)*coef - x(0)*x(1)*c*cdx*(1.-cdy))/denom;
            v(1) = (x(0)*coef - x(0)*x(1)*c*cdy*(1.-cdx))/denom;
            v *= -k*t_0;
         };
      case Problem::SteadyAdvection:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
            Vector xc(x);
            //xc -= .5;

            real_t r = xc.Norml2();
            if (r <= 0.) { return; }
            real_t csh = cosh(10. * (-1. + 4. * r));
            real_t q0 = k * 10. * 4. / (csh*csh * r);
            v.Set(q0, xc);
         };
      case Problem::NonsteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            const int vdim = x.Size();
            Vector xc(x);
            xc -= .5;
            Vector dx(vdim);
            const real_t ct = 4.*c*t * M_PI/4.;
            constexpr real_t dx_x = 0.2;
            constexpr real_t dx_y = 0.0;
            dx(0) = +xc(0) * cos(ct) + xc(1) * sin(ct) + dx_x;
            dx(1) = -xc(0) * sin(ct) + xc(1) * cos(ct) + dx_y;

            v.SetSize(vdim);
            constexpr real_t sigma = 0.1;
            constexpr real_t sigma2 = 2*sigma*sigma;
            // 2 sigma^2 + 4 k t, NOT 4 k t pi/4. A Gaussian under
            // du/dt = k lap u spreads as D' = 4k -- substitute
            // u = (A/D) exp(-r^2/D) and both sides come out as
            // u (r^2/D - 1) times D'/D and 4k/D respectively -- so the pi/4
            // that the rotation legitimately carries does not belong here.
            // With it the exact solution is not a solution: the PDE residual
            // is O(k) and the miniapp converged, in dt AND in h, to a limit
            // 0.0121998 away from this function. See the note at the head of
            // this file.
            const real_t denom = sigma2 + 4.*k*t;
            const real_t u = sigma2 / denom * exp(- (dx*dx) / denom);
            const real_t v0 = 2. * k * u / denom;
            v(0) = xc(0) + cos(ct) * dx_x - sin(ct) * dx_y;
            v(1) = xc(1) + sin(ct) * dx_x + cos(ct) * dx_y;
            v *= v0;
         };
      case Problem::KovasznayFlow:
         return [](const Vector &x, real_t t, Vector &v)
         {
            v.SetSize(x.Size());
            v = 0.;
         };
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            v.SetSize(x.Size());
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t u_x = (x(0) == 0.)?(0.):
                               (u / x(0) - ut * uy * x(0) / (k * chx*chx));
            const real_t u_y = (x(1) == 0.)?(0.):
                               (u / x(1) - ut * ux * x(1) / (k * chy*chy));
            v(0) = -k * u_x;
            v(1) = -k * u_y;
         };
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            v.SetSize(x.Size());
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t u_x = (x(0) == 0.)?(0.):
                               (u / x(0) - ut * uy * x(0) / (k * chx*chx));
            const real_t u_y = (x(1) == 0.)?(0.):
                               (u / x(1) - ut * ux * x(1) / (k * chy*chy));
            v(0) = -(k + u) * u_x;
            v(1) = -(k + u) * u_y;
         };
   }
   return VecTFunc();
}

VecFunc GetCFun(const ProblemParams &params)
{
   const Problem &prob = params.prob;
   const real_t &c = params.c;

   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::SteadyReactionDiffusion:
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         // null
         break;
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;

            v(0) = c;
            v(1) = c;
            if (ndim > 2)
            {
               v(2) = c;
            }
         };
      case Problem::SteadyAdvection:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;
            Vector xc(x);
            //xc -= .5;

            v(0) = +xc(1) * c;
            v(1) = -xc(0) * c;
         };
      case Problem::NonsteadyAdvectionDiffusion:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;
            Vector xc(x);
            xc -= .5;

            v(0) = -4. * xc(1) * c * M_PI/4.;
            v(1) = +4. * xc(0) * c * M_PI/4.;
         };
      case Problem::KovasznayFlow:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;
            Vector xc(x);
            xc(1) -= 1.25;

            //Kovasznay flow
            constexpr real_t Re = 100.;
            const real_t gamma = Re/2. - sqrt(Re*Re/4. + 4.*M_PI*M_PI);
            v(0) = 1. - exp(gamma * xc(0)) * cos(2.*M_PI * xc(1));
            v(1) = gamma / (2.*M_PI) * exp(gamma * xc(0)) * sin(2.*M_PI * xc(1));
         };
   }
   return VecFunc();
}

TFunc GetFFun(const ProblemParams &params)
{
   const Problem &prob = params.prob;
   const real_t &k = params.k;
   const real_t &t_0 = params.t_0;
   const real_t &c = params.c;

   switch (prob)
   {
      case Problem::SteadyReactionDiffusion:
         // f = k Delta T - F(T): the sign SteadyDiffusion below uses for its
         // own diffusive half (it returns -diff for diff = -k Delta T), with
         // the reaction subtracted. The sign of the reaction half was
         // established by RUNNING both -- the other one collapses u* to rate
         // zero while leaving the potential's own rate looking plausible.
         return [=](const Vector &x, real_t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0;
            for (int i = 0; i < ndim; i++) { t0 *= sin(M_PI * x(i)); }
            const real_t lap = -ndim * M_PI * M_PI * t0;
            return k * lap - (t0 * t0 * t0 - t0);
         };
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            const int ndim = x.Size();

            real_t t0   = t_0 * exp(x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            real_t diff = -k * t_0 * exp(x.Sum()) * (sin(M_PI*x(1)) * ((1.-M_PI*M_PI) * sin(M_PI*x(0))
                                                                       + 2. * M_PI * cos(M_PI*x(0)))
                                                     + sin(M_PI*x(0)) * ((1.-M_PI*M_PI)
                                                                         * sin(M_PI*x(1)) + 2. * M_PI
                                                                         * cos(M_PI*x(1))));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));

               diff *= sin(M_PI*x(2));
               diff -= k*M_PI*M_PI*t0;
            }

            return -diff;
         };
      case Problem::SteadyAdvectionDiffusion:
         return [=](const Vector &x, real_t) -> real_t
         {
            // div c*u: (assuming cx and cy are constant)
            constexpr real_t x0 = 1.;
            constexpr real_t y0 = 1.;
            real_t cdx = exp((x(0)-x0)*c);
            real_t cdy = exp((x(1)-y0)*c);
            real_t coef = (1. - cdx) * (1. - cdy);
            real_t denom = ((1. - exp(-c)) * (1. - exp(-c)));

            real_t conv = (c*(x(1)*coef - x(0)*x(1)*c*cdx*(1.-cdy)))/denom +
            (c*(x(0)*coef - x(0)*x(1)*c*cdy*(1.-cdx)))/denom;

            // div q:
            real_t diff = -k*((-2.*c*cdy*x(0)*(1-cdx) - 2.*c*cdx*x(1)*(1-cdy)
                               -c*c*cdy*(1-cdx)*x(0)*x(1) - c*c*cdx*(1-cdy)*x(0)*x(1)
                              ))/denom;

            return -(conv+diff)*t_0;
         };
      case Problem::SteadyAdvection:
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
         return [](const Vector &x, real_t) -> real_t { return 0.; };
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t ut = (prob == Problem::SteadyBurgers)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t u_x = (x(0) != 0.)?(u / x(0) - ut * uy * x(0) / (k * chx*chx)):(0.);
            const real_t u_y = (x(1) != 0.)?(u / x(1) - ut * ux * x(1) / (k * chy*chy)):(0.);
            const real_t u_xx = -2. * (u + k * ut * uy) / (k*k * chx*chx);
            const real_t u_yy = -2. * (u + k * ut * ux) / (k*k * chy*chy);
            const real_t divq = -k * (u_xx + u_yy);
            const real_t divF = u * (u_x + u_y);
            const real_t ft = ((prob == Problem::SteadyBurgers)?(0.):(exp(t)  * ux * uy));
            const real_t f = divq + divF + ft;
            return -f;
         };
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t argx = (1. - x(0)) / k;
            const real_t argy = (1. - x(1)) / k;
            const real_t ux = x(0) * tanh(argx);
            const real_t uy = x(1) * tanh(argy);
            const real_t chx = cosh(argx);
            const real_t chy = cosh(argy);
            const real_t ut = (prob == Problem::SteadyLinearKappa)?(1.):(exp(t) - 1.);
            const real_t u = ut * ux * uy;
            const real_t u_x = (x(0) != 0.)?(u / x(0) - ut * uy * x(0) / (k * chx*chx)):(0.);
            const real_t u_y = (x(1) != 0.)?(u / x(1) - ut * ux * x(1) / (k * chy*chy)):(0.);
            const real_t u_xx = -2. * (u + k * ut * uy) / (k*k * chx*chx);
            const real_t u_yy = -2. * (u + k * ut * ux) / (k*k * chy*chy);
            const real_t divq = -(u_x*u_x + u_y*u_y + (k + u)*u_xx + (k + u)*u_yy);
            const real_t ft = ((prob == Problem::SteadyLinearKappa)?(0.):(exp(t)  * ux * uy));
            const real_t f = divq + ft;
            return -f;
         };
   }
   return TFunc();
}

unique_ptr<FluxFunction> GetFluxFun(const ProblemParams &params,
                                    VectorCoefficient &ccoef)
{
   const Problem &prob = params.prob;

   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
         //null
         break;
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyAdvection:
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
         return make_unique<AdvectionFlux>(ccoef);
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         return make_unique<BurgersFlux>(ccoef.GetVDim());
   }

   return nullptr;
}

unique_ptr<MixedFluxFunction> GetHeatFluxFun(const ProblemParams &params,
                                             int dim)
{
   const Problem &prob = params.prob;
   const real_t &k = params.k;

   switch (prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::SteadyAdvectionDiffusion:
      case Problem::SteadyAdvection:
      case Problem::NonsteadyAdvectionDiffusion:
      case Problem::KovasznayFlow:
      case Problem::SteadyBurgers:
      case Problem::NonsteadyBurgers:
         static FunctionCoefficient ikappa([=](const Vector &x) -> real_t { return 1./k; });
         return make_unique<LinearDiffusionFlux>(dim, ikappa);
      case Problem::SteadyLinearKappa:
      case Problem::NonsteadyLinearKappa:
      {
         auto ikappa = [=](const Vector &x, real_t T) -> real_t
         {
            return 1./(k+T);
         };
         auto dikappa = [=](const Vector &x, real_t T) -> real_t
         {
            return -1./((k+T)*(k+T));
         };
         return make_unique<FunctionDiffusionFlux>(dim, ikappa, dikappa);
      }
   }

   return nullptr;
}

bool VisualizeField(socketstream &sout, const ParGridFunction &gf,
                    const char *name, int iter, bool verbose)
{
   const char vishost[] = "localhost";
   const int visport = 19916;
   if (!sout.is_open())
   {
      sout.open(vishost, visport);
   }
   if (!sout)
   {
      if (verbose)
      {
         cout << "Unable to connect to GLVis server at " << vishost << ':'
              << visport << endl;
         cout << "GLVis visualization disabled.\n";
      }
      return false;
   }
   else
   {
      // Make sure all ranks have sent the previous solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(gf.ParFESpace()->GetComm());
      const int num_procs = gf.ParFESpace()->GetNRanks();
      const int myid = gf.ParFESpace()->GetMyRank();
      sout << "parallel " << num_procs << " " << myid << "\n";
      constexpr int precision = 8;
      sout.precision(precision);
      sout << "solution\n" << *gf.FESpace()->GetMesh() << gf;
      if (iter == 0)
      {
         sout << "window_title '" << name << "'\n";
         if (gf.VectorDim() > 1)
         {
            sout << "keys Rljvvvvvmmc" << endl;
         }
         else
         {
            sout << "keys Rljmmc" << endl;
         }
      }
      sout << flush;
   }
   return true;
}
