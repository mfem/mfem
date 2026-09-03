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
//                        Anisotropic diffusion miniapp
//
// Compile with: make anisodiff
//
// Sample runs:  anisodiff -nx 20 -p 1 -o 2 -hb -dg
//               anisodiff -nx 20 -p 1 -o 2 -rd -brt -a 1e2 -bcn
//               anisodiff -nx 20 -p 1 -o 2 -rd -dg
//               anisodiff -nx 20 -p 1 -o 2 -hb -dg -trh1 -a 1e2 -bcn
//               anisodiff -nx 20 -p 1 -o 0 -hb -brt -rec -trbc
//               anisodiff -nx 20 -p 1 -o 1 -pa
//               anisodiff -nx 40 -p 2 -a 1 -ks 1e-4 -o 2 -hb -dg
//               anisodiff -m ../../data/square-disc.mesh -p 2 -a 1 -k 1e-2 -ks 1e-4 -o 2 -rd
//               anisodiff -nx 20 -p 5 -ks 1e+1 -o 2 -hb -brt -amr 5
//               anisodiff -nx 20 -p 9 -ks 1e-4 -o 2 -hb -dg -amr 5
//               anisodiff -nx 8 -p 5 -ks 1e2 -o 2 -hb -dg -amr 16 -dorf
//               anisodiff -nx 8 -p 5 -ks 1e2 -o 2 -hb -dg -amr 16 -dorf -hp
//                         -pmax 5 -ppest
//
// Device sample runs:
//
// Description:  This miniapp solves asymptotic heat diffusion problem with
//               anisotropic conductivity in the mixed formulation corresponding
//               to the system
//
//                                 kˉ¹⋅q + ∇ T =  g
//                                   ∇⋅q + a T = -f
//
//               with essential (RT) / natural (DG) Neumann boundary condition
//               q⋅n = 0, where n is the outer normal, or Dirichlet b.c. T =
//               = <given temperature>. The tensor k represents the heat
//               conductivity, where its symmetric and antisymmetric parts can
//               be adjusted. The scalar a is the heat capacity, which can be
//               zero, changing the problem to steady-state, indefinite,
//               saddle-point. The initial condition is f = 0 and g = -a *
//               <initial temperature> for the definite problem and g =
//               - <initial temperature> for the indefinite one. These problems
//               are offered:
//               1) sine diffusion - Sine profile diffusion with the asymptotic
//                                   (a -> infinity) reference solution with
//                                   the first order correction
//               2) diffusion ring - arc segment IC diffused along circle
//               3) diff. ring (Gauss) - Gaussian blobs IC diffused along circle
//               4) diff. ring (sine) - sine profile in radial and angular
//                                      direction is diffused along circle,
//                                      analytic solution for asymptotic
//                                      diffusion with zero radial diffusion
//               5) boundary layer - exponentially decaying boundary layer
//                                   problem
//               6) steady peak - a peak profile with a constant conductivity
//                                and a manufactured steady-state solution
//               7) steady varying angle - a concave radial profile diffused
//                                         along the circle with a manufactured
//                                         steady-state solution
//               8) Sovinec problem - a sine profile with diffusion
//                                    perpendicular to gradient of potential
//                                    with a manufactured steady-state solution
//               9) single-null diverted tokamak - Two-wire model of tokamak
//                                                 with a single X-point
//               10) double-null diverted tokamak - Two-wire model of tokamak
//                                                  with two X-points
//               We discretize with (broken) Raviart-Thomas finite elements
//               (heat flux q) and piecewise discontinuous polynomials
//               (temperature T). Alternatively, the piecewise discontinuous
//               polynomials are used for both quantities with stabilization,
//               yielding the Local Discontinuous Galerkin method. Optionally,
//               the mixed system is algebraically reduced or hybridized with
//               DG interface elements or H1 trace elements.
//
//               The miniapp demonstrates the use of the DarcyForm class and
//               the wrapping system operator DarcyOperator in an AMR loop with
//               the HDG error estimator.
//
//               We recommend viewing examples 1-6 before viewing this miniapp.
//
// ADAPTIVITY, AND WHAT hp BUYS
//
//    The loop is estimate, mark, refine. --amr-ref-levels sets the number of
//    cycles; --doerfler-marking picks the bulk criterion over the maximum one;
//    --postprocessed-estimate builds the estimate on the postprocessed
//    potential; --hp-adaptivity spends a marked element's refinement on its
//    DEGREE where Persson & Peraire's smoothness sensor says it is resolved
//    and on h where it is not.
//
//    Problem 5 with a strong anisotropy is what the whole thing is for: the
//    solution is analytic but carries a boundary layer of thickness
//    1/(pi sqrt(ks)) at y = 0 and y = 1, so it is smooth wherever it is
//    resolved and hopeless where it is not, which is exactly the distinction
//    the sensor is meant to make. On `-nx 8 -p 5 -ks 1e2 -o 2 -hb -dg`,
//    relative L2 error in the potential against the globally coupled unknowns
//    -- the size of the trace solve, which is what a hybridized method costs:
//
//      || t - t_ex ||          uniform M    h-adapt M       hp M
//      ---------------------------------------------------------------
//      1.0e-3                     24960         1146         727
//      1.0e-4                     99072         3501         967
//      3.0e-5                         -         6258        1057
//      1.0e-7                         -            -        2016
//      9.9e-10                        -            -        3365
//
//    Read across a row: at 1e-4 the same error costs 102 times fewer globally
//    coupled unknowns than uniform refinement and 3.6 times fewer than
//    h-adaptivity, and hp keeps going four decades past where the others were
//    stopped -- uniform would need nx beyond 1500 and h-adaptivity does not
//    get there at all, dying on direct-solver memory at M around 1.4 million.
//
//    The hp column moved when the trace surplus stopped being retired and
//    started being constrained, and the uniform and h columns did not -- they
//    are bit-identical, which is what says the change is in the hp path and
//    nowhere else. A hanging-node family used to be forced to the CEILING
//    degree, because the retired route could not coarsen through a conforming
//    prolongation that interpolates in the ceiling basis; it now takes the
//    same rule as any other face. That enrichment was worth something -- 717
//    against 727 at 1e-3, 952 against 967 at 1e-4, 1847 against 2016 at 1e-7,
//    so 1 to 9 per cent -- and the last row is the clearest: hp reached
//    4.5e-10 at M = 3264 with families enriched and reaches 9.9e-10 at 3365
//    without. What it buys is that the configuration is no longer refused,
//    which is what makes the boundary datum and the parallel runs work.
//    The uniform column is nx = 64 and 128; both adaptive columns are
//    --doerfler-marking --postprocessed-estimate at the defaults, the hp one
//    adding --hp-adaptivity. Each row is the nearest cycle rather than an
//    interpolation, so the errors within a row are close but not equal --
//    1.0e-3 / 1.2e-3 / 8.7e-4, then 8.3e-5 / 8.8e-5 / 8.8e-5, then 3.4e-5 /
//    2.6e-5, then 1.1e-7.
//
//    IN SECONDS RATHER THAN DOFS the ranking is not the same, and that is
//    worth knowing before quoting the table. At a relative error near 1e-4
//    h-adaptivity is the fastest of the three -- 0.38 s against uniform's
//    2.12 s and hp's 0.67 s -- because an adaptive loop pays for every
//    intermediate solve and hp takes more cycles to get there. hp overtakes
//    below about 1e-5 and then wins outright: 4.8 times faster than uniform at
//    7e-6 and 11 times at 2e-6, where h-adaptivity can no longer reach at all.
//    Two of the three defaults below were chosen on that curve.
//
//    Those three seconds figures were 0.51 / 2.22 / 0.81 when first measured,
//    on the same machine and with two of the three paths unchanged since. So
//    the drift is the machine and not the method, and only the RANKING should
//    be quoted from here; where the ceiling's cost is wanted as a number, the
//    controlled sweep in DarcyHybridization::SetTraceOrders() varies nothing
//    but the ceiling and is the one to use.
//
//    THREE THINGS ABOUT THE ESTIMATE THAT ARE NOT OPTIONAL, each measured
//    rather than reasoned about, and each of which silently stopped the loop
//    converging until it was found:
//
//    1. The Dirichlet datum is imposed WEAKLY here, and on such a face lambda
//       is not approximating the potential's trace, so |p^ - lambda| there is
//       not an error and does not vanish with h. Measured, it is one fixed
//       amount per face, so eta grows like 1/h -- 2.00, 2.83, 4.00 over
//       nx = 8, 16, 32 while the true error fell by 265x -- and every marked
//       element sits on the boundary. HDGErrorEstimator::SetExcludedBoundary()
//       leaves those attributes out and the total then converges at h^2, which
//       is optimal at k = 2.
//
//    2. THE SPLIT'S TWO JOBS COME FROM DIFFERENT FIELDS. Refining only the
//       direction that carries the error is a large win -- this problem's
//       layer is in y, and refining only in y reaches 1.0e-3 on 640 elements
//       where isotropic refinement needs 4096 -- but the DIRECTION has to be
//       taken from the computed potential and the MAGNITUDE from the
//       postprocessed one. Taking both from the postprocessed field flags x,
//       six of six, on a problem whose layer is in y, and the loop then sits
//       at 0.283893 through twelve cycles and 5352 dofs having started there.
//       Taking direction from the computed potential instead reaches 2.5e-4 at
//       M = 2217 on the same problem. --anisotropic-estimate 1 is the old
//       behaviour and is kept so that stays measured.
//
//       AND UNDER hp IT NEEDS ONE MORE THING, which is
//       --skip-enriched-direction and is on by default. A hanging-node family
//       has to run at the ceiling degree (see
//       DarcyHybridization::SetTraceOrders()), so its master trace fits the
//       several fine elements better than the one coarse element and the
//       coarse element's |p^ - lambda| genuinely grows. As a MAGNITUDE that is
//       right -- it is the mismatched element. As a DIRECTION it is exactly
//       wrong: refining in y puts hanging nodes on VERTICAL faces, whose
//       energy the geometric split attributes to x, so the neighbour is split
//       in x when another y is what would match it, and the loop alternates
//       forever. Measured on one identical hanging-node mesh, moving only the
//       ceiling from 2 to 3, the twelve elements next to a hanging node carry
//       a d0 sum of 1.11e-4 against 5.45e-2 -- a factor of 490, entirely in
//       d0, with four of them flipping y to x at 17 times their estimate --
//       while the other 58 elements are unchanged. Keeping the magnitude and
//       dropping the direction is what makes the last column of the table
//       exist at all.
//
//    3. --postprocessed-estimate IS WORTH ABOUT 1.4x IN DOFS, once (2) is in
//       place, and that is a controlled comparison: both configurations take
//       the direction from the computed potential, so the only difference is
//       the magnitude field. h-adaptive reaches 8.8e-5 at M = 3501 against
//       1.1e-4 at M = 4812, and 3.4e-5 at M = 6258 against 3.1e-5 at M = 9330.
//
//       Without (2) it looked worthless. Taking both jobs from the same field,
//       the two estimates trace the same error-against-M curve to within a
//       factor of two and neither dominates -- at M ~ 2800 the computed
//       potential gives 2.2e-5 and the postprocessed one 3.8e-5 -- and all it
//       bought was cycles, 26 against 34 for the same answer. Separating the
//       jobs is what turned it into dofs.
//
//       WHY, as far as it has been measured, is in
//       HDGErrorEstimator::SetAnisotropic(). Three explanations are dead: not
//       the degree gap against the trace (--postprocessed-projected-down
//       removes it entirely and every flag stays put), not the anisotropy (the
//       same stall at -ks 1 and on problem 6, both isotropic), and not the
//       marking (the two estimates select the same elements for two cycles,
//       to every printed digit, under isotropic refinement). What is left is
//       that the two measure different things: on the computed potential
//       p^ - lambda is the scheme's own stabilization term, on a
//       superconverged one it is essentially lambda's error, and attributing
//       that to the direction NORMAL to the face is not the direction that
//       would reduce it.
//
//    4. A FACE RICHER THAN ITS ELEMENT NEEDS BOTH HALVES HANDLING, and with
//       them the `max` face rule is usable and is the default under hp. The
//       DIRECTION half is (2). The MAGNITUDE half is
//       --cap-trace-at-element: compare such an element against lambda
//       projected down to its own degree, so the modes it cannot represent are
//       not charged to it.
//
//       Without the cap the loop runs away, and the numbers are worth keeping
//       because nothing about them is subtle. At the plateau it marked a
//       cluster of degree-2 elements sitting next to degree-5 ones, at
//       x ~ 0.63 in the middle of the domain, with eta = 1.2e-5 against a TRUE
//       element error of 3.8e-9 -- a ratio of 3000, and rising 443, 1700, 3000
//       over three cycles -- while the elements actually carrying the error,
//       five times more of it, went unmarked. Splitting them in x makes them
//       narrower, tau ~ 1/h on their vertical faces grows, and eta grows with
//       it: the refinement the estimate triggers is what makes the estimate
//       bigger.
//
//       With both halves, `max` is worth about 10 per cent of the dofs at
//       every matched error and reaches an order deeper in the same cycle
//       budget -- 4.5e-10 at M = 3264 against `min`'s 3.0e-9 at M = 3022 --
//       which is consistent with what a prescribed interface says about the
//       rule on its own: `min` gets WORSE as the degree jump grows, the flux
//       error going 1.066e-2, 1.620e-2, 1.519e-2 on `convdiff -o 1 -nx 8` as
//       the refined half is raised by one, two and three degrees, where `max`
//       holds 1.051e-2 throughout.
//       Comparing lambda against the projection of the postprocessed trace
//       into the trace space rather than against the trace itself -- which is
//       what the published estimator asks for, and is
//       HDGErrorEstimator::SetTraceComparison() -- does not change that: the
//       flags are the same six, and eta moves by 5%. So the floor the
//       projection removes is not what misdirects the split, and what does is
//       still open.

#include "mfem.hpp"
#include "darcyop.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <functional>
#include <algorithm>

using namespace std;
using namespace mfem;
using namespace mfem::hdg;

// Define the analytical solution and forcing terms / boundary conditions
typedef function<real_t(const Vector &, real_t)> TFunc;
typedef function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef function<void(const Vector &, DenseMatrix &)> MatFunc;

enum Problem
{
   SineDiffusion = 1,
   DiffusionRing,
   DiffusionRingGauss,
   DiffusionRingSine,
   BoundaryLayer,
   SteadyPeak,
   SteadyVaryingAngle,
   Sovinec,
   SingleNull,
   DoubleNull,
};

struct ProblemParams
{
   Problem prob;
   real_t x0, y0, sx, sy;
   real_t k, ks, ka;
   real_t t_0;
   real_t a;
};

MatFunc GetKFun(const ProblemParams &params);
TFunc GetTFun(const ProblemParams &params);
VecTFunc GetQFun(const ProblemParams &params);
TFunc GetFFun(const ProblemParams &params);
unique_ptr<MixedFluxFunction> GetHeatFluxFun(const ProblemParams &params,
                                             int dim);

// Visualize the grid function in GLVis
bool VisualizeField(socketstream &sout, const GridFunction &gf,
                    const char *name, int iter = 0);

/** @brief Doerfler (bulk) marking: the smallest set carrying @a gamma of the
    total squared estimate.

    The set is built by sorting the elements by their estimate and taking the
    largest until the bulk is reached, which is minimal because the
    accumulation stops at the first index that reaches it rather than after
    it. This is the criterion the HDG convergence analysis assumes -- Cockburn,
    Nochetto & Zhang -- and it is the one that drove this problem's estimate
    down; the maximum criterion below did not, and the two are kept side by
    side so that difference stays measured rather than asserted.

    @a local_err is eta_K itself, not the squares: this squares them, so that
    both criteria take the same argument and neither can be fed the wrong one.
    gamma near zero marks few elements and refines locally, gamma near one
    approaches uniform refinement -- the opposite direction to MarkMaximum(). */
static void MarkDoerfler(const Vector &local_err, real_t gamma,
                         Array<int> &marked)
{
   MFEM_VERIFY(gamma > 0. && gamma <= 1.,
               "the marking parameter must lie in (0, 1]");

   const int ne = local_err.Size();
   Array<int> order(ne);
   for (int e = 0; e < ne; e++) { order[e] = e; }
   std::sort(order.begin(), order.end(), [&local_err](int a, int b)
   {
      return local_err(a) > local_err(b);
   });

   real_t total = 0.;
   for (int e = 0; e < ne; e++) { total += local_err(e) * local_err(e); }

   marked.SetSize(0);
   real_t accumulated = 0.;
   const real_t wanted = gamma * total;
   for (int i = 0; i < ne && accumulated < wanted; i++)
   {
      marked.Append(order[i]);
      accumulated += local_err(order[i]) * local_err(order[i]);
   }
}

/** @brief Maximum marking: every element whose estimate is at least @a gamma
    times the largest.

    This is what ThresholdRefiner does with SetTotalErrorFraction(gamma) and an
    infinite total norm, which is its default, and this miniapp's adaptive loop
    used it through that class until the hp work replaced the refiner with an
    explicit mark-then-refine so that the marked set could be split between h
    and p. Large gamma refines locally, small gamma approaches uniform. */
static void MarkMaximum(const Vector &local_err, real_t gamma,
                        Array<int> &marked)
{
   MFEM_VERIFY(gamma >= 0. && gamma <= 1.,
               "the marking parameter must lie in [0, 1]");

   marked.SetSize(0);
   const real_t largest = local_err.Max();
   if (largest <= 0.) { return; }

   for (int e = 0; e < local_err.Size(); e++)
   {
      if (local_err(e) > gamma * largest) { marked.Append(e); }
   }
}

/** @brief Element-local L2 projection of @a src onto @a dst's space.

    Exists for one measurement: the postprocessed potential differs from the
    computed one in two ways at once -- a higher degree than the trace, and
    different field content -- and the anisotropic split of the error estimate
    goes wrong when it is used. Projecting it back down to the potential's own
    degree keeps the field and removes the gap, so estimating on the result
    separates the two. Both spaces must be scalar and discontinuous; the
    projection is elementwise and means nothing across a continuous space. */
static void ProjectDown(const GridFunction &src, GridFunction &dst)
{
   const FiniteElementSpace *fes_s = src.FESpace();
   FiniteElementSpace *fes_d = dst.FESpace();
   Mesh *mesh = fes_d->GetMesh();
   MFEM_VERIFY(fes_s->GetVDim() == 1 && fes_d->GetVDim() == 1,
               "scalar spaces only");
   MFEM_VERIFY(fes_s->GetMesh() == mesh, "both spaces must be on one mesh");

   Array<int> vd_s, vd_d;
   Vector loc_s, shape_s, shape_d, b, c;
   DenseMatrix M;
   DenseMatrixInverse Mi;

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const FiniteElement *fe_s = fes_s->GetFE(e);
      const FiniteElement *fe_d = fes_d->GetFE(e);
      ElementTransformation *T = mesh->GetElementTransformation(e);

      const int nd_s = fe_s->GetDof();
      const int nd_d = fe_d->GetDof();
      fes_s->GetElementVDofs(e, vd_s);
      src.GetSubVector(vd_s, loc_s);

      M.SetSize(nd_d);
      M = 0.;
      b.SetSize(nd_d);
      b = 0.;
      shape_s.SetSize(nd_s);
      shape_d.SetSize(nd_d);

      const int order = fe_s->GetOrder() + fe_d->GetOrder() + T->OrderW();
      const IntegrationRule &ir = IntRules.Get(fe_d->GetGeomType(), order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         fe_s->CalcShape(ip, shape_s);
         fe_d->CalcShape(ip, shape_d);

         const real_t w = ip.weight * T->Weight();
         AddMult_a_VVt(w, shape_d, M);
         b.Add(w * (shape_s * loc_s), shape_d);
      }

      Mi.Factor(M);
      c.SetSize(nd_d);
      Mi.Mult(b, c);

      fes_d->GetElementVDofs(e, vd_d);
      dst.SetSubVector(vd_d, c);
   }
}

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   string mesh_file = "";
   int nx = 0;
   int ny = 0;
   int ref_levels = -1;
   real_t dr = 0.;
   int order = 1;
   bool dg = false;
   bool brt = false;
   int iproblem = Problem::SineDiffusion;
   ProblemParams pars;
   pars.x0 = 0.;
   pars.y0 = 0.;
   pars.sx = 1.;
   pars.sy = 1.;
   pars.k = 1.;
   pars.ks = 1.;
   pars.ka = 0.;
   pars.a = 0.;
   real_t td = 0.5;
   bool bc_neumann = false;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   bool trace_ess_bc = false;
   bool nonlinear = false;
   bool nonlinear_diff = false;
   int solver_type = (int)DarcyOperator::SolverType::LBFGS;
   int isol_ctrl = (int)DarcyOperator::SolutionController::Type::Native;
   int amr_nrefs = 0;
   real_t theta = 0.7;
   bool doerfler = false;
   int aniso = -1;
   bool est_pp = false;
   bool tproj = true;
   bool skip_edir = true;
   bool cap_tr = true;
   bool pp_down = false;
   bool hp = false;
   int p_max = -1;
   int pface_max = -1;
   real_t hp_shift = 0.;
   bool pa = false;
   const char *device_config = "cpu";
   bool reconstruct = false;
   bool mfem = false;
   bool visit = false;
   bool paraview = false;
   bool visualization = true;
   int vis_iters = -1;
   bool analytic = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of refinement levels (automatic to 10000 elements by default)");
   args.AddOption(&dr, "-dr", "--delta-random",
                  "Relative random displacement of the mesh nodes.");
   args.AddOption(&nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&pars.sx, "-sx", "--size-x",
                  "Size along x axis.");
   args.AddOption(&pars.sy, "-sy", "--size-y",
                  "Size along y axis.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=sine diffusion\n\t\t"
                  "2=diffusion ring\n\t\t"
                  "3=diffusion ring - Gauss source\n\t\t"
                  "4=diffusion ring - sine source\n\t\t"
                  "5=boundary layer\n\t\t"
                  "6=steady peak\n\t\t"
                  "7=steady varying angle\n\t\t"
                  "8=Sovinec\n\t\t"
                  "9=Single null\n\t\t"
                  "10=Double null\n\t\t");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&pars.ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.ka, "-ka", "--kappa_anti",
                  "Antisymmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.a, "-a", "--heat_capacity",
                  "Heat capacity coefficient (0=indefinite problem)");
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
   args.AddOption(&nonlinear_diff, "-nld", "--nonlinear-diffusion", "-no-nld",
                  "--no-nonlinear-diffusion", "Enable non-linear diffusion regime.");
   args.AddOption(&solver_type, "-nls", "--nonlinear-solver",
                  "Nonlinear solver type (1=LBFGS, 2=LBB, 3=Newton).");
   args.AddOption(&isol_ctrl, "-sn", "--solution-norm",
                  "Solution norm (0=native, 1=flux, 2=potential).");
   args.AddOption(&amr_nrefs, "-amr", "--amr-ref-levels",
                  "AMR refinement levels");
   args.AddOption(&theta, "-theta", "--marking-fraction",
                  "Marking parameter. Under the maximum criterion it refines\n\t\t"
                  "every element whose estimate is at least this fraction of\n\t\t"
                  "the largest, so LARGE refines locally; under Doerfler it is\n\t\t"
                  "the bulk fraction of the total the marked set must carry, so\n\t\t"
                  "large refines widely. The two run in opposite directions.");
   args.AddOption(&aniso, "-aniso", "--anisotropic-estimate",
                  "Split each element's estimate along the reference directions\n\t\t"
                  "and refine only the ones that carry it. 0=off; 1=on, taking\n\t\t"
                  "the direction from the same field as the magnitude, which is\n\t\t"
                  "wrong whenever that field is the postprocessed potential;\n\t\t"
                  "2=on, taking the direction from the computed potential and\n\t\t"
                  "the magnitude from whichever field was asked for; -1 picks\n\t\t"
                  "2. Refining only the direction that carries the error reaches\n\t\t"
                  "1.0e-3 on 640 elements where isotropic refinement needs 4096.");
   args.AddOption(&doerfler, "-dorf", "--doerfler-marking",
                  "-maxm", "--maximum-marking",
                  "Doerfler (bulk) marking rather than the maximum criterion.");
   args.AddOption(&est_pp, "-ppest", "--postprocessed-estimate",
                  "-no-ppest", "--no-postprocessed-estimate",
                  "Build the error estimate on the postprocessed potential\n\t\t"
                  "rather than on the computed one.");
   args.AddOption(&cap_tr, "-captr", "--cap-trace-at-element",
                  "-no-captr", "--no-cap-trace-at-element",
                  "Where a face's trace degree exceeds the element's, compare\n\t\t"
                  "that element against the trace projected down to its own\n\t\t"
                  "degree.");
   args.AddOption(&skip_edir, "-sed", "--skip-enriched-direction",
                  "-no-sed", "--no-skip-enriched-direction",
                  "Where a face's trace degree exceeds the element's, keep its\n\t\t"
                  "contribution to that element's error but not to its\n\t\t"
                  "direction. Inert unless a face outruns its element, which\n\t\t"
                  "under hp is every hanging node -- and there it is what lets\n\t\t"
                  "anisotropic refinement work at all.");
   args.AddOption(&pp_down, "-ppdown", "--postprocessed-projected-down",
                  "-no-ppdown", "--no-postprocessed-projected-down",
                  "Project the postprocessed potential back onto the\n\t\t"
                  "potential's own degree before estimating on it. Keeps its\n\t\t"
                  "field content and removes its degree gap against the trace,\n\t\t"
                  "which is what separates the two as explanations for the\n\t\t"
                  "anisotropic split going wrong under --postprocessed-estimate.");
   args.AddOption(&tproj, "-tproj", "--projected-trace-comparison",
                  "-no-tproj", "--literal-trace-comparison",
                  "Compare the trace unknown against the projection of the\n\t\t"
                  "potential's trace into the trace space rather than against\n\t\t"
                  "the trace itself. A no-op unless the two carry different\n\t\t"
                  "degrees, which is what --postprocessed-estimate makes them.");
   args.AddOption(&hp, "-hp", "--hp-adaptivity", "-no-hp", "--no-hp-adaptivity",
                  "Spend a marked element's refinement on its degree where the\n\t\t"
                  "smoothness sensor says it is resolved, and on h where it is\n\t\t"
                  "not. Off, every marked element is refined in h.");
   args.AddOption(&p_max, "-pmax", "--max-order",
                  "Highest degree --hp-adaptivity may reach (default order+5).\n\t\t"
                  "It is also the degree the trace space is BUILT at, which is a\n\t\t"
                  "ceiling rather than a starting point, so raising it costs\n\t\t"
                  "trace storage on every face whether or not any face uses it.");
   args.AddOption(&pface_max, "-pfmax", "--p-face-rule",
                  "Give a face the higher of its two elements' degrees (1), the\n\t\t"
                  "lower (0), or the higher whenever --hp-adaptivity is on (-1,\n\t\t"
                  "the default). Measured 25% cheaper in dofs at fixed error in\n\t\t"
                  "the hp loop, where the p-interface sits on the feature; a\n\t\t"
                  "wash at a prescribed one-degree interface. See\n\t\t"
                  "DarcyHybridization::TraceOrderRule.");
   args.AddOption(&hp_shift, "-hps", "--hp-sensor-shift",
                  "Shift of the smoothness threshold s_0 = -4 log10(p). Positive\n\t\t"
                  "calls more elements smooth and so spends more on p.");
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
   args.AddOption(&vis_iters, "-vis-its", "--visualization-iters",
                  "Set step for GLVis visualization of the solver iterations (<0=off).");
   args.AddOption(&analytic, "-anal", "--analytic", "-no-anal",
                  "--no-analytic",
                  "Enable or disable analytic solution.");

   args.ParseCheck();

   // 2. Set the problem options
   pars.prob = (Problem)iproblem;
   const Problem &problem = pars.prob;
   bool bnldiff = nonlinear_diff;

   if (trace_ess_bc && !dg && !brt)
   {
      cerr << "Essential trace BC does not work with continuous elements" << endl;
      return 1;
   }

   if (trace_ess_bc && nonlinear)
   {
      cerr << "Essential trace BC is not implemented for non-linear forms" << endl;
      return 1;
   }

   if (bnldiff && reduction)
   {
      cerr << "Reduction is not possible with non-linear diffusion" << endl;
      return 1;
   }

   if (nonlinear && !hybridization)
   {
      cerr << "Warning: A linear solver is used" << endl;
   }

   /* Both jobs of the split, everywhere, now that each is taken from the
      field that answers it and an enriched face contributes no direction. */
   if (aniso < 0) { aniso = 2; }

   /* `max` under hp, which it can be now that the magnitude at a degree
      mismatch is handled -- see HDGErrorEstimator::SetCapTraceAtElement(). It
      is the better rule wherever it is usable: on a prescribed interface `min`
      gets WORSE as the degree jump grows, the flux error on
      `convdiff -o 1 -nx 8` going 1.066e-2, 1.620e-2, 1.519e-2 as the refined
      half is raised by one, two and three degrees where `max` holds 1.051e-2
      throughout, and in this loop it is worth about 10% of the dofs at every
      matched error and reaches an order deeper in the same cycle budget --
      4.5e-10 at M = 3264 against 3.0e-9 at M = 3022.

      Without the cap it is unusable, because `max` is exactly what CREATES a
      face richer than one of its elements: the loop then plateaus at 8.8e-7.
      Off hp the two rules coincide, no element having a neighbour of a
      different degree. */
   if (pface_max < 0) { pface_max = hp ? 1 : 0; }

   if (hp)
   {
      // Each of these is a property the per-face trace order relies on rather
      // than a convenience.
      if (amr_nrefs <= 0)
      {
         cerr << "--hp-adaptivity needs an adaptive loop: pass --amr-ref-levels"
              << endl;
         return 1;
      }
      if (!hybridization)
      {
         cerr << "--hp-adaptivity is a per-face TRACE degree, so it needs "
              "--hybridization" << endl;
         return 1;
      }
      if (!dg)
      {
         cerr << "--hp-adaptivity needs the discontinuous flux space (--dg). "
              "A variable order on an L2 space needs nothing but "
              "SetElementOrder(); on RT it is a different question and this "
              "branch does not touch the RT pathways." << endl;
         return 1;
      }
      if (trace_h1)
      {
         cerr << "--hp-adaptivity needs the DG trace space (--trace-DG); an H1 "
              "trace is continuous across faces and has no per-face degree"
              << endl;
         return 1;
      }
      if (reconstruct)
      {
         cerr << "--reconstruct builds a local problem whose shapes assume "
              "one trace degree per element and aborts in "
              "DenseMatrixInverse::Factor when they differ -- tried, not "
              "assumed. The postprocessed potential the estimate uses is the "
              "reconstruction that a per-face degree does not disturb."
              << endl;
         return 1;
      }
      /* order+5, and the +5 is measured on three axes rather than chosen.
         Raising the ceiling from order+3 to order+5 on this problem takes the
         dofs at a relative error of 1e-7 from 13191 to 4189 -- 3.3x -- and the
         wall clock for the same 26 cycles from 10.8 s to 6.4 s, at a BETTER
         error. It is nearly free because the retired unit rows cost almost
         nothing: holding the mesh and every face degree fixed and moving only
         the ceiling from 2 to 7, a 2.67x storage ratio, assembly comes out
         0.98-1.07x, the preconditioner 0.94-1.23x, the trace solve 1.03-1.19x
         and peak RSS at most 1.15x. Only the hybridization's own setup scales,
         1.46-1.64x, and it is 5% of the run.

         The reason the ceiling matters so much is that it, not the smoothness
         sensor, is making the hp decision: spend_on_p requires p < p_max, so
         an element at the ceiling goes to h whatever the sensor says. Raising
         it moves the h-refinement count by 45 to 100x. */
      if (p_max < 0) { p_max = order + 5; }
      if (p_max < order)
      {
         cerr << "--max-order is below --order, so there is nowhere to refine to"
              << endl;
         return 1;
      }
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 4. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   if (ny <= 0)
   {
      ny = nx;
   }

   Mesh mesh;
   if (!mesh_file.empty())
   {
      mesh = Mesh(mesh_file, 1, 1);

      Vector x_min(2), x_max(2);
      mesh.GetBoundingBox(x_min, x_max);
      pars.x0 = x_min(0);
      pars.y0 = x_min(1);
      pars.sx = x_max(0) - x_min(0);
      pars.sy = x_max(1) - x_min(1);
   }
   else
   {
      mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                   pars.sx, pars.sy);
   }

   int dim = mesh.Dimension();

   // 5. Mark boundary conditions based on the selected problem
   const int bdr_attrs = mesh.bdr_attributes.Size() > 0 ?
                         mesh.bdr_attributes.Max() : 1;
   Array<int> bdr_is_dirichlet(bdr_attrs);
   Array<int> bdr_is_neumann(bdr_attrs);
   bdr_is_dirichlet = 0;
   bdr_is_neumann = 0;

   switch (problem)
   {
      case Problem::SineDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyPeak:
      case Problem::Sovinec:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         // Free BC (zero Dirichlet)
         if (bc_neumann)
         {
            bdr_is_neumann[1] = -1; // Outflow
            bdr_is_neumann[2] = -1; // Outflow
         }
         break;
      case Problem::BoundaryLayer:
         /* The layer is at y = 0 and y = 1 and the solution is zero at
            x = 0 and x = 1, so the datum belongs on every face except the two
            normal to x. Mesh::Make2D numbers them 1=y0, 2=x1, 3=y1, 4=x0 and
            Mesh::Make3D numbers them 1=z0, 2=y0, 3=x1, 4=y1, 5=x0, 6=z1, so
            the two orderings share no index -- the 2D pair {1,3} lands on
            z = 0 and x = 1 in 3D, leaving the layer faces with no condition
            at all. That was not an abort: it ran to completion and returned a
            relative error of 0.986 as though nothing were wrong. */
         if (dim > 2)
         {
            bdr_is_dirichlet = -1;
            bdr_is_dirichlet[2] = 0;    // x = 1
            bdr_is_dirichlet[4] = 0;    // x = 0
         }
         else
         {
            bdr_is_dirichlet[0] = -1;   // y = 0
            bdr_is_dirichlet[2] = -1;   // y = 1
         }
         break;
      case Problem::SteadyVaryingAngle:
         bdr_is_dirichlet = -1;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   // 6. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (!mesh_file.empty())
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   if (dr > 0.) { RandomizeMesh(mesh, dr); }

   // A variable-order space is refused on a conforming mesh, even an L2 one
   // that needs no prolongation at all, so hp asks for the nonconforming
   // representation up front. On an unrefined mesh that is bit-for-bit a
   // no-op on the answer -- it carries no hanging nodes and its conforming
   // prolongation is the identity, which DarcyOperator reads as a null
   // pointer.
   // The `true` is for simplices: EnsureNCMesh() defaults to leaving them
   // conforming, and FiniteElementSpace::Construct() then refuses the variable
   // order with "Variable-order space requires a nonconforming mesh". That
   // default made hp unreachable on every triangle and tetrahedron mesh, in
   // 2D as well as 3D.
   if (hp) { mesh.EnsureNCMesh(true); }

   // 7. Define a finite element space on the mesh. Here we use the
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
   auto V_space = make_unique<FiniteElementSpace>(&mesh, V_coll.get(),
                                                  (dg)?(dim):(1));
   auto V_space_dg = (V_coll_dg)?(make_unique<FiniteElementSpace>(
                                     &mesh, V_coll_dg.get(), dim)):(nullptr);
   // Temperature FE space
   auto W_space = make_unique<FiniteElementSpace>(&mesh, W_coll.get());

   // Darcy form
   auto darcy = make_unique<DarcyForm>(V_space.get(), W_space.get());

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   pars.t_0 = 1.; // Base temperature

   ConstantCoefficient acoeff(pars.a); // Heat capacity

   auto kFun = GetKFun(pars);
   MatrixFunctionCoefficient kcoeff(dim, kFun); // Tensor conductivity
   InverseMatrixCoefficient ikcoeff(kcoeff); // Inverse tensor conductivity

   auto tFun = GetTFun(pars);
   FunctionCoefficient tcoeff(tFun); // Analytic temperature
   ProductCoefficient gcoeff(-1., tcoeff); // Boundary heat flux r.h.s.

   auto fFun = GetFFun(pars);
   FunctionCoefficient fcoeff(fFun); // Temperature r.h.s.

   auto qFun = GetQFun(pars);
   VectorFunctionCoefficient qcoeff(dim, qFun); // Analytic heat flux
   ConstantCoefficient one;

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                     ┌        ┐
   //                     | Mq -Bᵀ |
   //                     | B  Mt  |
   //                     └        ┘
   //     where:
   //     RTDG:
   //     Mq = (kˉ¹ q, v)                    q, v ∈ V
   //     B = (∇⋅q, w)                       q ∈ V, w ∈ W
   //     Mt = (a T, w)                      T, w ∈ W
   //     LBRT:
   //     Mq = (kˉ¹ q, v)                    q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>         q ∈ V, w ∈ W
   //     Mt = (a T, w)                      T, w ∈ W
   //     LDG:
   //     Mq = (kˉ¹ q, v)                    q, v ∈ V
   //     B = (∇⋅q, w) + <[q⋅n], {w}>         q ∈ V, w ∈ W
   //     Mt = (a T, w) + <td k hˉ¹[T], [w]> T, w ∈ W

   // Diffusion

   unique_ptr<MixedFluxFunction> HeatFluxFun;
   if (!bnldiff)
   {
      // Linear diffusion
      if (!nonlinear)
      {
         BilinearForm *Mq = darcy->GetFluxMassForm();
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
         NonlinearForm *Mqnl = darcy->GetFluxMassNonlinearForm();
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
      BlockNonlinearForm *Mnl = darcy->GetBlockNonlinearForm();
      HeatFluxFun = GetHeatFluxFun(pars, dim);
      if (dg)
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
         Mnl->AddInteriorFaceIntegrator(new MixedConductionNLFIntegrator(
                                           *HeatFluxFun, td));
         Mnl->AddBdrFaceIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun, td),
                                   bdr_is_neumann);
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
   if (dg)
   {
      if (bnldiff)
      {
         cerr << "Warning: Using linear stabilization for non-linear diffusion" << endl;
      }

      if (td > 0.)
      {
         if (!nonlinear)
         {
            BilinearForm *Mt = darcy->GetPotentialMassForm();
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
            Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                     bdr_is_neumann);
            if (trace_ess_bc)
            {
               Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                        bdr_is_dirichlet);
            }
         }
         else
         {
            NonlinearForm *Mtnl = darcy->GetPotentialMassNonlinearForm();
            Mtnl->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
            Mtnl->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                       bdr_is_neumann);
         }
      }
   }

   // Divergence/weak gradient

   MixedBilinearForm *B = darcy->GetFluxDivForm();
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

   // Inertial term

   if (pars.a > 0.)
   {
      if (!nonlinear)
      {
         BilinearForm *Mt = darcy->GetPotentialMassForm();
         Mt->AddDomainIntegrator(new MassIntegrator(acoeff));
      }
      else
      {
         NonlinearForm *Mtnl = darcy->GetPotentialMassNonlinearForm();
         Mtnl->AddDomainIntegrator(new MassIntegrator(acoeff));
      }
   }

   // Set hybridization / reduction / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg && !brt)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   unique_ptr<FiniteElementCollection> trace_coll;
   unique_ptr<FiniteElementSpace> trace_space;

   /** @brief Give the hybridization one trace degree per face, read off the
       element degrees the spaces are actually carrying.

       **Straight after EnableHybridization() and before Assemble().** That
       call has already built C, E, G and H at the ceiling; stating the degrees
       rebuilds them and resets the assembly. Getting this wrong does not
       fail -- the DOF count comes out exactly right and the answer is wrong.

       The rule is Min, the lower of a face's two elements. A trace above both
       its neighbours is exactly redundant rather than unstable, so nothing is
       lost there; whether Max pays at a genuine p-interface is open and needs
       a convergence study rather than a preference. */
   auto set_trace_orders = [&]()
   {
      if (!hp) { return; }

      Array<int> elem_order(mesh.GetNE());
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         elem_order[i] = W_space->GetElementOrder(i);
      }

      Array<int> face_order;
      DarcyHybridization::FaceOrdersFromElementOrders(
         mesh, elem_order,
         (pface_max != 0) ? DarcyHybridization::TraceOrderRule::Max
         : DarcyHybridization::TraceOrderRule::Min,
         p_max, face_order);
      darcy->GetHybridization()->SetTraceOrders(face_order);
   };

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
         // Under hp the constraint space's degree is a CEILING, not a
         // starting point: SetTraceOrders() reuses its storage, so a face can
         // be set below the degree it was built at and never above.
         trace_coll = make_unique<DG_Interface_FECollection>(hp ? p_max : order,
                                                             dim);
      }
      trace_space = make_unique<FiniteElementSpace>(&mesh, trace_coll.get());
      darcy->EnableHybridization(trace_space.get(),
                                 new NormalTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);
      set_trace_orders();
      // Set essential BC
      if (trace_ess_bc)
      {
         darcy->GetHybridization()->SetEssentialBC(bdr_is_dirichlet);
      }
      chrono.Stop();
      cout << "Hybridization init took " << chrono.RealTime() << "s.\n";
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
      else if (pars.a > 0.)
      {
         darcy->EnablePotentialReduction(ess_flux_tdofs_list);
      }
      else
      {
         cerr << "No possible reduction!" << endl;
         return 1;
      }

      chrono.Stop();
      cout << "Reduction init took " << chrono.RealTime() << "s.\n";
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // 10. Define the block structure of the problem, i.e. define the array of
   //     offsets for each variable. The last component of the Array is the sum
   //     of the dimensions of each block.
   Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

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

   // 11. Allocate memory (x, rhs) for the analytical solution and the right
   //     hand side. Define the GridFunction q_h, t_h for the finite element
   //     solution and linear forms fform and gform for the right hand side.
   //     The data allocated by x and rhs are passed as a reference to the grid
   //     functions (q,t) and the linear forms (fform, gform). With
   //     hybridization, linear form hform for the constraint is constructed
   //     as well together with the trace grid function tr_h.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   x = 0.;
   GridFunction q_h, t_h, tr_h, qt_h, q_hs, t_hs, tr_hs;
   q_h.MakeRef(V_space.get(), x.GetBlock(0), 0);
   t_h.MakeRef(W_space.get(), x.GetBlock(1), 0);
   if (hybridization)
   {
      tr_h.MakeRef(trace_space.get(), x.GetBlock(2), 0);
   }

   // Project essential b.c.
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
   unique_ptr<LinearForm> gform(new LinearForm);
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
   unique_ptr<LinearForm> fform(new LinearForm);
   fform->Update(W_space.get(), rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   if (!hybridization)
   {
      // Neumann BC (non-hybridized)
      fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qcoeff, +2., 0.),
                                  bdr_is_neumann);
   }

   // Constraint r.h.s.
   unique_ptr<LinearForm> hform;

   if (hybridization)
   {
      // Neumann BC for the hybridized system
      hform = make_unique<LinearForm>();
      hform->Update(trace_space.get(), rhs.GetBlock(2), 0);
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(qcoeff, 2),
                                   bdr_is_neumann);
   }

   // 12. Construct the spatial operator

   DarcyOperator op(ess_flux_tdofs_list, darcy.get(),
   {gform.get(), fform.get(), hform.get()},
   {&gcoeff, &fcoeff, &qcoeff},
   (DarcyOperator::SolverType) solver_type);

   op.SetTolerance(1e-8);

   op.EnableSolutionController(
      (DarcyOperator::SolutionController::Type) isol_ctrl);

   if (vis_iters >= 0)
   {
      op.EnableIterationsVisualization(vis_iters);
   }

   // 13. Set up an error estimator. Here we use the HDG estimator which
   //     evaluates the difference between the face values of the potential and
   //     the trace variable in an energy norm with respect to a given operator,
   //     which is represented by the provided integrator implementing
   //     ComputeHDGFaceEnergy() method.

   unique_ptr<BilinearFormIntegrator> amr_bfi;

   if (amr_nrefs > 0 && hybridization)
   {
      amr_bfi.reset(new HDGDiffusionIntegrator(kcoeff, td));
   }
   else
   {
      amr_nrefs = 0;
   }

   // 14. The marking strategy. This used to be a ThresholdRefiner, which marks
   //     and refines in one call; hp needs the marked set in hand so that it
   //     can be split between the elements that get another degree and the
   //     ones that get another element, so the two steps are explicit here.
   //     MarkMaximum() with --marking-fraction is what the refiner was doing.

   // 15. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int amr_it = 0; amr_it <= amr_nrefs; amr_it++)
   {

      // 16. Solve the steady/asymptotic problem

      Vector dx(x.Size()); dx = 0.;
      op.SetTime(1.);
      op.ImplicitSolve(1., x, dx);
      x += dx;

      // 17. Compute the L2 error norms.

      /* The rule follows the highest degree actually in the mesh, not the
         degree the run started at, or a p-refined element's error is measured
         with a rule that cannot see it. The +1 is the postprocessed
         potential's enrichment, which is measured here too.

         This is generous by two degrees for a uniform-order run, and
         deliberately so: on this miniapp's sharper problems the rule is
         resolving the EXACT solution rather than the discrete one. Measured on
         `-p 5 -ks 1e2 -o 2 -hb -dg -nx 8`, the reported relative error is
         0.2505 at 2*order+1 and 0.2828 at 2*(order+1)+1, and the second is the
         right number -- a boundary layer of thickness 1/31 across a cell of
         width 1/8 is not integrated by five points. */
      int max_order = order;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         max_order = max(max_order, W_space->GetElementOrder(e));
      }
      int order_quad = max(2, 2*(max_order + 1) + 1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      real_t err_q  = q_h.ComputeL2Error(qcoeff, irs);
      real_t norm_q = ComputeLpNorm(2., qcoeff, mesh, irs);
      real_t err_t  = t_h.ComputeL2Error(tcoeff, irs);
      real_t norm_t = ComputeLpNorm(2., tcoeff, mesh, irs);

      /* The postprocessed potential, wanted here twice over: it is what the
         error estimate below is built on, and it is the quantity a p-adaptive
         run is really buying, converging one order above t_h where the theory
         offers it. Its enriched space follows the element degrees element by
         element without being told about them. Computed once per cycle rather
         than again inside the refinement block, which is where it was needed
         first. */
      GridFunction t_pp, t_pd;
      real_t err_tpp = -1., err_tpd = -1.;
      if (est_pp)
      {
         HDGPotentialPostprocessor pp(q_h, t_h);
         pp.SetDiffusionInverse(ikcoeff);
         pp.Compute(t_pp);
         err_tpp = t_pp.ComputeL2Error(tcoeff, irs);

         if (pp_down)
         {
            t_pd.SetSpace(W_space.get());
            ProjectDown(t_pp, t_pd);
            err_tpd = t_pd.ComputeL2Error(tcoeff, irs);
         }
      }

      if (amr_nrefs > 0)
      {
         int p_lo = order, p_hi = order;
         if (hp)
         {
            p_lo = p_hi = W_space->GetElementOrder(0);
            for (int e = 1; e < mesh.GetNE(); e++)
            {
               const int p = W_space->GetElementOrder(e);
               p_lo = min(p_lo, p);
               p_hi = max(p_hi, p);
            }
         }

         /* The globally coupled unknowns, which is the cost an adaptive
            method has to be judged against. dim(M) is the trace space's own
            size and does not move when a face is coarsened -- the storage
            stays at the ceiling -- so it is the wrong number here.
            GetTraceTrueVSize() is what the trace solve actually carries: one
            unknown per degree of freedom a face has, the ceiling's surplus
            constrained away rather than retired into a unit row. */
         int ndof_m = 0;
         if (hybridization)
         {
            ndof_m = darcy->GetHybridization()->GetTraceTrueVSize()
                     - darcy->GetHybridization()->GetEssentialTrueDofs().Size();
         }

         cout << "iter:\t" << amr_it
              << "\tne:\t" << mesh.GetNE()
              << "\tM:\t" << ndof_m
              << "\tp:\t" << p_lo << "-" << p_hi
              << "\tq_err:\t" << err_q / norm_q
              << "\tt_err:\t" << err_t / norm_t;
         if (err_tpp >= 0.) { cout << "\tpp_err:\t" << err_tpp / norm_t; }
         if (err_tpd >= 0.) { cout << "\tpd_err:\t" << err_tpd / norm_t; }
         cout << endl;
      }
      else
      {
         cout << "|| q_h - q_ex || / || q_ex || = " << err_q / norm_q << "\n";
         cout << "|| t_h - t_ex || / || t_ex || = " << err_t / norm_t << "\n";
      }

      if (reconstruct)
      {
         darcy->Reconstruct(x, x.GetBlock(2), qt_h, q_hs, t_hs, tr_hs);
         real_t err_qt = qt_h.ComputeL2Error(qcoeff, irs);
         real_t norm_qt = ComputeLpNorm(2., qcoeff, mesh, irs);
         cout << "|| qt_h - qt_ex || / || qt_ex || = " << err_qt / norm_qt << "\n";
         real_t err_qs = q_hs.ComputeL2Error(qcoeff, irs);
         cout << "|| q_hs - q_ex || / || q_ex || = " << err_qs / norm_q << "\n";
         real_t err_ts = t_hs.ComputeL2Error(tcoeff, irs);
         cout << "|| t_hs - t_ex || / || t_ex || = " << err_ts / norm_t << "\n";
      }

      // 18. Project the fluxes

      GridFunction q_vh;

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

      // 19. Project the analytic solution

      static GridFunction q_a, t_a;

      q_a.SetSpace((V_space_dg)?(V_space_dg.get()):(V_space.get()));
      q_a.ProjectCoefficient(qcoeff);

      t_a.SetSpace(W_space.get());
      t_a.ProjectCoefficient(tcoeff);

      // 20. Save the mesh and the solution. This output can be viewed later
      //     using GLVis: "glvis -m anisodiff.mesh -g sol_q.gf" or "glvis -m
      //     anisodiff.mesh -g sol_t.gf".
      if (mfem)
      {
         stringstream ss;
         ss.str("");
         ss << "anisodiff";
         if (amr_nrefs > 0) { ss << "_" << amr_it; }
         ss << ".mesh";
         ofstream mesh_ofs(ss.str());
         mesh_ofs.precision(8);
         mesh.Print(mesh_ofs);

         ss.str("");
         ss << "sol_q";
         if (amr_nrefs > 0) { ss << "_" << amr_it; }
         ss << ".gf";
         ofstream q_ofs(ss.str());
         q_ofs.precision(8);
         q_vh.Save(q_ofs);

         ss.str("");
         ss << "sol_t";
         if (amr_nrefs > 0) { ss << "_" << amr_it; }
         ss << ".gf";
         ofstream t_ofs(ss.str());
         t_ofs.precision(8);
         t_h.Save(t_ofs);
      }

      // 21. Save data in the VisIt format
      if (visit)
      {
         static VisItDataCollection visit_dc("Anisodiff", &mesh);
         if (amr_it == 0)
         {
            visit_dc.RegisterField("heat flux", &q_vh);
            visit_dc.RegisterField("temperature", &t_h);
            if (analytic)
            {
               visit_dc.RegisterField("heat flux analytic", &q_a);
               visit_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         visit_dc.SetCycle(amr_it);
         visit_dc.Save();
      }

      // 22. Save data in the ParaView format
      if (paraview)
      {
         static ParaViewDataCollection paraview_dc("Anisodiff", &mesh);
         if (amr_it == 0)
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
         paraview_dc.SetCycle(amr_it);
         paraview_dc.Save();
      }

      // 23. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         static socketstream q_sock, t_sock;
         VisualizeField(q_sock, q_vh, "Heat flux", amr_it);
         VisualizeField(t_sock, t_h, "Temperature", amr_it);
         if (reconstruct)
         {
            static socketstream qt_sock, qs_sock, ts_sock;
            VisualizeField(qt_sock, qt_h, "Total flux", amr_it);
            VisualizeField(qs_sock, q_hs, "Recon. flux", amr_it);
            VisualizeField(ts_sock, t_hs, "Recon. temperature", amr_it);
         }
         if (analytic)
         {
            static socketstream qa_sock, ta_sock;
            VisualizeField(qa_sock, q_a, "Heat flux analytic", amr_it);
            VisualizeField(ta_sock, t_a, "Temperature analytic", amr_it);
         }
      }

      // 24. Estimate, mark, and refine

      if (amr_it < amr_nrefs)
      {
         /* THE ESTIMATE IS BUILT ON THE POSTPROCESSED POTENTIAL, when
            --postprocessed-estimate asks for it.

            HDGErrorEstimator compares the element's trace of the potential
            against the trace unknown on each face. Built on t_h that
            difference is the raw solution's, which is one order down; built on
            the postprocessed potential -- which converges one order better
            where the theory offers it -- it is the published estimator's
            eta_5. It costs one element-local solve per element and reads
            neither the trace space nor a neighbour, so a per-face trace degree
            cannot reach it.

            The switch is here rather than hard-wired so the difference stays a
            measurement: --no-postprocessed-estimate recovers the estimate this
            loop used before. */
         /* Rebuilt every cycle on purpose. The estimator caches on the mesh
            sequence, which does not move when only the degrees change, and it
            holds a reference to the potential -- whose space is a new object
            each cycle when the estimate is the postprocessed one. */
         HDGErrorEstimator amr_err(*amr_bfi, tr_h,
                                   (est_pp && pp_down) ? t_pd :
                                   est_pp ? t_pp : t_h);
         /* The magnitude estimator carries the split only when it is also the
            right field to take the direction from -- which it is whenever the
            two fields are the same one, so `2` without a postprocessed
            estimate is `1`. Getting this wrong made --anisotropic-estimate 2
            mean ISOTROPIC whenever --postprocessed-estimate was off, silently.
            */
         const bool split_here = (aniso != 0) && !(aniso == 2 && est_pp);
         amr_err.SetAnisotropic(split_here);

         /* The Dirichlet datum is imposed WEAKLY here -- it enters the flux
            equation as <T_D, v.n> and the constraint is not assembled on
            those faces at all -- so on them lambda is not approximating the
            potential's trace and |p^-lambda| is not an error. Left in, it is
            one fixed amount per face and so grows like 1/h: measured on this
            problem eta went 2.00, 2.83, 4.00 over nx = 8, 16, 32 while the
            error fell by 265x, and every marked element was on the boundary.
            With --trace-ess-bc the trace IS the datum and the term is real,
            which is why the exclusion follows the flag rather than being
            unconditional. */
         if (!trace_ess_bc) { amr_err.SetExcludedBoundary(bdr_is_dirichlet); }

         /* The per-face trace degrees live in the hybridization, so the
            estimator has to be told where to find them; the constraint space
            it would otherwise read is uniform at the ceiling. */
         if (hp) { amr_err.SetHybridization(*darcy->GetHybridization()); }
         if (!skip_edir) { amr_err.SetSkipEnrichedDirection(false); }
         if (!cap_tr) { amr_err.SetCapTraceAtElement(false); }

         if (tproj)
         {
            amr_err.SetTraceComparison(
               HDGErrorEstimator::TraceComparison::Projected);
         }

         const Vector &local_err = amr_err.GetLocalErrors();

         /* DIRECTION FROM THE COMPUTED POTENTIAL, MAGNITUDE FROM WHICHEVER
            FIELD WAS ASKED FOR.

            The two answer different questions. |p^ - lambda| on the computed
            potential is the scheme's own stabilization term, and its
            directional split is right -- it flags y on a problem whose layer
            is in y, and more sharply as the layer sharpens. On the
            postprocessed potential the same difference is essentially
            lambda's own error, which is a real quantity but is not the
            element's, and attributing it to the direction NORMAL to the face
            is not the direction that would reduce it; measured, it flags x at
            every anisotropy over four decades and the loop then refines
            forever without touching the layer. The magnitude is the other way
            round: the postprocessed estimate is the one worth reading as an
            error, converging an order faster.

            So take each from the field that answers it. It costs one more pass
            over the faces, which is nothing beside a solve, and it is only
            built when the two fields differ. --anisotropic-estimate 1 keeps
            both from the magnitude field, which is what this loop did before
            and is kept so the difference stays measured. */
         unique_ptr<HDGErrorEstimator> amr_dir;
         if (aniso == 2 && est_pp)
         {
            amr_dir.reset(new HDGErrorEstimator(*amr_bfi, tr_h, t_h));
            amr_dir->SetAnisotropic();
            if (!trace_ess_bc)
            { amr_dir->SetExcludedBoundary(bdr_is_dirichlet); }
            if (hp) { amr_dir->SetHybridization(*darcy->GetHybridization()); }
            if (!skip_edir) { amr_dir->SetSkipEnrichedDirection(false); }
            if (!cap_tr) { amr_dir->SetCapTraceAtElement(false); }
         }

         const Array<int> &aniso_flags = amr_dir
                                         ? amr_dir->GetAnisotropicFlags()
                                         : amr_err.GetAnisotropicFlags();

         Array<int> marked;
         if (doerfler) { MarkDoerfler(local_err, theta, marked); }
         else { MarkMaximum(local_err, theta, marked); }

         if (marked.Size() == 0) { break; }

         /* THE h-OR-p DECISION. The estimate says WHERE to spend; it cannot
            say on what, because a badly under-resolved smooth region and a
            well-resolved layer look alike to it. Persson & Peraire's sensor
            answers the other half: it measures how much of an element's energy
            sits in its top degree, so an element whose expansion is already
            decaying gets another degree -- where convergence in p is
            exponential -- and one whose is not gets another element. */
         Array<Refinement> h_refs;
         Array<int> p_refs;
         Vector s_e;
         if (hp)
         {
            PerssonPeraireSmoothness sensor(t_h);
            sensor.GetLogSensor(s_e);
         }

         for (int i = 0; i < marked.Size(); i++)
         {
            const int e = marked[i];
            const int p = hp ? W_space->GetElementOrder(e) : order;
            const bool spend_on_p =
               hp && p < p_max &&
               s_e(e) < PerssonPeraireSmoothness::Threshold(p) + hp_shift;

            if (spend_on_p) { p_refs.Append(e); continue; }

            Refinement ref(e);
            if (aniso_flags.Size() > 0) { ref.SetType(aniso_flags[e]); }
            h_refs.Append(ref);
         }

         // The split types are reported because on this problem they are what
         // decides whether the loop converges at all: the layer is in y, and
         // an estimate that flags x refines forever without touching it.
         // One counter per axis plus one for everything mixed, because a
         // three-dimensional run has a Z type and four mixed ones and the
         // tally used to drop all five into the isotropic bucket.
         int n_ax[3] = {0, 0, 0}, niso_ref = 0;
         for (int i = 0; i < h_refs.Size(); i++)
         {
            switch (h_refs[i].GetType())
            {
               case Refinement::X: n_ax[0]++; break;
               case Refinement::Y: n_ax[1]++; break;
               case Refinement::Z: n_ax[2]++; break;
               default: niso_ref++; break;
            }
         }

         cout << "mark:\t" << marked.Size() << " / " << mesh.GetNE()
              << "\tp:\t" << p_refs.Size()
              << "\th:\t" << h_refs.Size()
              << " (x" << n_ax[0] << " y" << n_ax[1]
              << ((dim > 2) ? " z" : "") << ((dim > 2) ? std::to_string(
                                                n_ax[2]) : string())
              << " *" << niso_ref << ")"
              << "\teta:\t" << amr_err.GetTotalError();
         if (hp)
         {
            // The threshold is per element, because it depends on that
            // element's own degree, so the range is what can be printed. It
            // used to print Threshold(order) -- the BASE order -- which is
            // -1.204 where an element at degree 5 is actually judged against
            // -2.796, so the printed pair understated the strictness applied
            // and stayed constant while p: climbed. Reading the decision off
            // it was not possible.
            real_t s_lo = s_e(marked[0]), s_hi = s_e(marked[0]);
            real_t t_lo = infinity(), t_hi = -infinity();
            for (int i = 0; i < marked.Size(); i++)
            {
               s_lo = min(s_lo, s_e(marked[i]));
               s_hi = max(s_hi, s_e(marked[i]));
               const real_t t = PerssonPeraireSmoothness::Threshold(
                                   W_space->GetElementOrder(marked[i])) + hp_shift;
               t_lo = min(t_lo, t);
               t_hi = max(t_hi, t);
            }
            cout << "\ts:\t" << s_lo << " .. " << s_hi
                 << "\ts0:\t" << t_lo << " .. " << t_hi;
         }
         cout << endl;

         /* TWO UPDATES, NOT ONE, AND THE DEGREES BEFORE THE MESH.
            SetElementOrder() insists the space is already in sync with the
            mesh, so the degrees have to be stated while it is; and Update()
            refuses to absorb both changes at once -- "Updating space after
            both mesh change and element order change is not supported".
            Refining afterwards is safe because UpdateElementOrders() carries a
            parent's degree onto its children. */
         if (p_refs.Size() > 0)
         {
            for (int i = 0; i < p_refs.Size(); i++)
            {
               const int e = p_refs[i];
               const int p = W_space->GetElementOrder(e) + 1;
               V_space->SetElementOrder(e, p);
               W_space->SetElementOrder(e, p);
            }
            V_space->Update(false);
            W_space->Update(false);
         }

         if (h_refs.Size() > 0) { mesh.GeneralRefinement(h_refs, -1, 0); }

         // Update FE spaces
         V_space->Update();
         if (V_space_dg) { V_space_dg->Update(); }
         W_space->Update();
         if (hybridization) { trace_space->Update(); }
         if (reconstruct)
         {
            qt_h.FESpace()->Update();
            t_hs.FESpace()->Update();
            q_hs.FESpace()->Update();
            tr_hs.FESpace()->Update();
         }

         // Update grid functions and linear forms
         block_offsets = DarcyOperator::ConstructOffsets(*darcy);
         x.Update(block_offsets, mt);
         rhs.Update(block_offsets, mt);

         x = 0.;
         q_h.MakeRef(V_space.get(), x.GetBlock(0), 0);
         t_h.MakeRef(W_space.get(), x.GetBlock(1), 0);
         if (reconstruct)
         {
            qt_h.SetSpace(qt_h.FESpace());
            t_hs.SetSpace(t_hs.FESpace());
            q_hs.SetSpace(q_hs.FESpace());
            tr_hs.SetSpace(tr_hs.FESpace());
         }

         gform->Update(V_space.get(), rhs.GetBlock(0), 0);
         fform->Update(W_space.get(), rhs.GetBlock(1), 0);

         if (hybridization)
         {
            tr_h.MakeRef(trace_space.get(), x.GetBlock(2), 0);
            hform->Update(trace_space.get(), rhs.GetBlock(2), 0);
         }

         // Project essential b.c.
         if (!dg && !brt)
         {
            V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
            q_h.ProjectBdrCoefficientNormal(qcoeff,
                                            bdr_is_neumann);   //essential Neumann BC
         }

         if (hybridization && trace_ess_bc)
         {
            tr_h.ProjectBdrCoefficient(tcoeff, bdr_is_dirichlet); // essential Dirichlet BC
         }

         // Update Darcy form, where hybridization must be reinitialized to
         // reintegrate the constraint and eliminate the essential b.c.
         darcy->Update();
         if (hybridization)
         {
            darcy->EnableHybridization(trace_space.get(),
                                       new NormalTraceJumpIntegrator(),
                                       ess_flux_tdofs_list);
            set_trace_orders();
            // Set essential b.c.
            if (trace_ess_bc)
            {
               darcy->GetHybridization()->SetEssentialBC(bdr_is_dirichlet);
            }
         }

         // Update Darcy operator
         op.Update();
      }
   }

   return 0;
}

MatFunc GetKFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   const real_t &ka = params.ka;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   switch (params.prob)
   {
      case Problem::SineDiffusion:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
         // Axial conductivity
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            kappa.Diag(k, ndim);
            kappa(0,0) *= ks;
            kappa(0,1) = +ka * k;
            kappa(1,0) = -ka * k;
            if (ndim > 2)
            {
               kappa(0,2) = +ka * k;
               kappa(2,0) = -ka * k;
            }
         };
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyVaryingAngle:
         // Radial vs. tangential conductivity
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            b(0) = (r>0.)?(-dx(1) / r):(1.);
            b(1) = (r>0.)?(+dx(0) / r):(0.);

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            //const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_x = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_y = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
            const real_t psi_norm = hypot(psi_x, psi_y);
            if (psi_norm > 0.)
            {
               b(0) = -psi_y / psi_norm;
               b(1) = +psi_x / psi_norm;
            }
            else
            {
               b = 0.;
            }

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::SingleNull:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);

            constexpr real_t x1 = 0.5;
            constexpr real_t y1 = -0.25;
            constexpr real_t x2 = 0.5;
            constexpr real_t y2 = 0.75;
            const real_t dx1 = x(0) - x1;
            const real_t dy1 = x(1) - y1;
            const real_t dx2 = x(0) - x2;
            const real_t dy2 = x(1) - y2;
            const real_t rr1 = dx1*dx1 + dy1*dy1;
            const real_t rr2 = dx2*dx2 + dy2*dy2;
            constexpr real_t Bt = 1.;
            // Bp = curl log(sqrt(rr1) * sqrt(rr2) * z)
            const real_t Bp_x = + ((rr1 > 0.)?(dy1 / rr1):(0.))
                                + ((rr2 > 0.)?(dy2 / rr2):(0.));
            const real_t Bp_y = - ((rr1 > 0.)?(dx1 / rr1):(0.))
                                - ((rr2 > 0.)?(dx2 / rr2):(0.));

            const real_t B = sqrt(Bp_x*Bp_x + Bp_y*Bp_y + Bt*Bt);
            b(0) = +Bp_x / B;
            b(1) = +Bp_y / B;

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::DoubleNull:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);

            constexpr real_t xc = 0.5;
            constexpr real_t yc = 0.5;
            const real_t dx = x(0) - xc;
            const real_t dy = x(1) - yc;
            constexpr real_t Bt = 1.;
            // Bp = curl ((1/2*(x-xc)**2 + 1/2*(1/4*sin(2pi*(y-yc)))**2) * z)
            const real_t Bp_x = +1./16.*M_PI * sin(4.*M_PI * dy);
            const real_t Bp_y = -dx;

            const real_t B = sqrt(Bp_x*Bp_x + Bp_y*Bp_y + Bt*Bt);
            b(0) = +Bp_x / B;
            b(1) = +Bp_y / B;

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
   }
   return MatFunc();
}

TFunc GetTFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SineDiffusion:
         // Sine profile diffusion with asymptotic (a -> infinity)
         // solution and the first order correction
         return [=](const Vector &x, real_t t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            if (a <= 0.) { return t0; }

            Vector ddT((ndim<=2)?(2):(4));
            ddT(0) = -t_0 * M_PI*M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1));//xx,yy
            ddT(1) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * cos(M_PI*x(1));//xy
            if (ndim > 2)
            {
               ddT(0) *= sin(M_PI*x(2));//xx,yy,zz
               ddT(1) *= sin(M_PI*x(2));//xy
               //xz
               ddT(2) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
               //yz
               ddT(3) = +t_0 * M_PI*M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1)) * cos(M_PI*x(2));

            }

            DenseMatrix kappa;
            kFun(x, kappa);

            real_t div = -(kappa(0,0) + kappa(1,1)) * ddT(0) - (kappa(0,1) + kappa(1,0)) * ddT(1);
            if (ndim > 2)
            {
               div += -kappa(2,2) * ddT(0) - (kappa(0,2) + kappa(2,0)) * ddT(2)
               - (kappa(1,2) + kappa(2,1)) * ddT(3);
            }
            return t0 - div / a * t;
         };
      case Problem::DiffusionRing:
         // Arc segment IC for diffusion along circle
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.25;
            constexpr real_t r1 = 0.35;
            constexpr real_t dr01 = 0.025;
            constexpr real_t theta0 = 11./12. * M_PI;
            constexpr real_t dtheta0 = 1./48. * M_PI;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t theta = fabs(atan2(dx(1), dx(0)));

            if (r < r0 - dr01 || r > r1 + dr01 || theta < theta0 - dtheta0)
            {
               return 0.;
            }

            const real_t dr = min(r - r0 + dr01, r1 + dr01 - r) / dr01;
            const real_t dth = (theta - theta0 + dtheta0) / dtheta0;
            return min(1_r, dr) * min(1_r, dth) * t_0;
         };
      case Problem::DiffusionRingGauss:
         // Gaussian blobs IC for diffusion along circle
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.025;
            constexpr real_t x_c = 0.15;

            const real_t dx_l = x(0) - (x0       + x_c  * sx);
            const real_t dx_r = x(0) - (x0 + (1. - x_c) * sx);
            const real_t dy = x(1) - (y0 + 0.5*sy);
            const real_t r_l = hypot(dx_l, dy);
            const real_t r_r = hypot(dx_r, dy);

            return - exp(- r_l*r_l/(r0*r0)) + exp(- r_r*r_r/(r0*r0));
         };
      case Problem::DiffusionRingSine:
         // Sine profile in radial and angular direction is diffused along
         // circle, where analytic solution for asymptotic diffusion with
         // zero radial diffusion is provided (ks -> 0)
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.05;
            constexpr real_t w0 = 16.;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            if (r <= 0.) { return 0.; }
            const real_t th = atan2(dx(1), dx(0));

            const real_t C = w0 / r;
            return 1. / (1. + t * k * C*C / a) * cos(w0*th) * sin(M_PI * r/r0);
         };
      case Problem::BoundaryLayer:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t k_para = M_PI*M_PI * k * ks;
            const real_t k_perp = k;
            const real_t k_frac = sqrt(k_para/k_perp);
            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            return - (e_down + e_up) / denom * sin(M_PI * x(0));
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            return x(0)*x(1) * pow(arg, s);
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            return 1. - r*r*r;
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t &kappa_perp = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return psi / kappa_perp;
         };
      case Problem::SingleNull:
      case Problem::DoubleNull:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t xc = 0.5;
            constexpr real_t yc = 0.5;
            constexpr real_t wc = 1./8.;
            const real_t dx = (x(0) - xc) / wc;
            const real_t dy = (x(1) - yc) / wc;
            return t_0 * exp(-0.5 * (dx*dx + dy*dy));
         };
   }
   return TFunc();
}

VecTFunc GetQFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SineDiffusion:
         // Sine profile diffusion with asymptotic (a -> infinity)
         // solution and the first order correction
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector gT(vdim);
            gT = 0.;
            gT(0) = t_0 * M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1));
            gT(1) = t_0 * M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1));
            if (vdim > 2)
            {
               gT(0) *= sin(M_PI*x(2));
               gT(1) *= sin(M_PI*x(2));
               gT(2) = t_0 * M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
            }

            DenseMatrix kappa;
            kFun(x, kappa);

            if (vdim <= 2)
            {
               v(0) = -kappa(0,0) * gT(0) -kappa(0,1) * gT(1);
               v(1) = -kappa(1,0) * gT(0) -kappa(1,1) * gT(1);
            }
            else
            {
               kappa.Mult(gT, v);
               v.Neg();
            }
         };
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
         };
      case Problem::DiffusionRingSine:
         // Sine profile in radial and angular direction is diffused along
         // circle, where analytic solution for asymptotic diffusion with
         // zero radial diffusion is provided (ks -> 0)
         return [=](const Vector &x, real_t t, Vector &v)
         {
            constexpr real_t r0 = 0.05;
            constexpr real_t w0 = 16.;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            if (r <= 0.) { v = 0.; return;  }
            const real_t th = atan2(dx(1), dx(0));

            const real_t C = w0 / r;
            const real_t T_r = -C / (1. + t * k * C*C / a) * sin(w0*th)
                               * sin(M_PI * r/r0);
            v(0) = + k * T_r * sin(th);
            v(1) = - k * T_r * cos(th);
         };
      case Problem::BoundaryLayer:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            const real_t k_para = M_PI*M_PI * kappa(0,0);
            const real_t k_perp = kappa(1,1);
            const real_t k_frac = sqrt(k_para/k_perp);

            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            const real_t T_x = - (e_down + e_up) / denom * M_PI * cos(M_PI * x(0));
            const real_t T_y = k_frac * (e_down - e_up) / denom * sin(M_PI * x(0));
            v(0) = -kappa(0,0) * T_x;
            v(1) = -kappa(1,1) * T_y;
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_x = x(1) * pow(arg, s-1) * (arg + x(0) * s * arg_x);
            const real_t T_y = x(0) * pow(arg, s-1) * (arg + x(1) * s * arg_y);
            v(0) = -kappa(0,0) * T_x;
            v(1) = -kappa(1,1) * T_y;
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            const real_t kappa_r = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_r = - 3. * r;
            v(0) = -kappa_r * T_r * dx(0);
            v(1) = -kappa_r * T_r * dx(1);
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            v(0) = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            v(1) = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
         };
   }
   return VecTFunc();
}

TFunc GetFFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto TFun = GetTFun(params);
   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SineDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t T = TFun(x, 0);
            return -((a > 0.)?(a):(1.)) * T;
         };
      case Problem::BoundaryLayer:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t) -> real_t
         {
            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_xx = x(1) * pow(arg, s-2) * (2.*s * arg_x * arg + x(0) * s * ((s-1) * arg_x*arg_x - M_PI*M_PI * arg*arg));
            const real_t T_yy = x(0) * pow(arg, s-2) * (2.*s * arg_y * arg + x(1) * s * ((s-1) * arg_y*arg_y - M_PI*M_PI * arg*arg));
            return kappa(0,0) * T_xx + kappa(1,1) * T_yy;
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATION METHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t kappa_r = ks * k;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_rr = - 9. * r;
            return kappa_r * T_rr;
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return -2.*M_PI*M_PI * psi;
         };
   }
   return TFunc();
}

unique_ptr<MixedFluxFunction> GetHeatFluxFun(const ProblemParams &params,
                                             int dim)
{
   auto KFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SineDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
      case Problem::SteadyVaryingAngle:
      case Problem::Sovinec:
      case Problem::SingleNull:
      case Problem::DoubleNull:
         static MatrixFunctionCoefficient kappa(dim, KFun);
         static InverseMatrixCoefficient ikappa(kappa);
         return make_unique<LinearDiffusionFlux>(ikappa);
   }

   return nullptr;
}

bool VisualizeField(socketstream &sout, const GridFunction &gf,
                    const char *name, int iter)
{
   const char vishost[] = "localhost";
   const int visport = 19916;
   if (!sout.is_open())
   {
      sout.open(vishost, visport);
   }
   if (!sout)
   {
      cout << "Unable to connect to GLVis server at " << vishost << ':'
           << visport << endl;
      cout << "GLVis visualization disabled.\n";
      return false;
   }
   else
   {
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
