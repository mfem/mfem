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
//            ---------------------------------------------------
//            HDG host performance harness: one hard problem, every
//            host performance axis, and a ledger of what ran where
//            ---------------------------------------------------
//
// Compile with: make hdgperf
//
// Sample runs:
//    hdgperf                             the baseline: every option off
//    hdgperf -thr -nt 8                  threaded element loops, 8 threads
//    hdgperf -thr -nt 8 -lfac 1 -tasm 1  and the batched local/trace routes
//    hdgperf -n 96 -o 2                  a smaller case, for a quick check
//
// WHAT THIS IS, and what it deliberately is not.
//
// It is `convdiff -p 6 -o 3 -nx 128 -ny 128 -dg -hb -nl -npc -nls 3` and
// nothing else: steady Burgers, hybridized, discontinuous flux, solved by NPC
// -- Newton on the full (q, u, uhat) system with the Jacobian eliminated
// hybridically. That command takes about 80 s single-threaded here and runs 8
// Newton steps, so the per-linearisation work is paid eight times and is what
// the host performance options are actually aimed at. The problem, its exact
// solution and its source are lifted VERBATIM from convdiff.cpp so the two
// binaries solve the same discrete system; the check is that the two print
// the same error norms, and `-cmp` prints them for exactly that comparison.
//
// It is NOT a device harness. hdgdevice.cpp is that, and it answers a
// different question -- which stages have somewhere to run on a GPU and where
// the data comes back. Everything here runs on the host, and the axes are the
// ones a host build can actually turn: OpenMP threading of the element loops,
// the batched local factorisation and trace assembly, the preconditioner, and
// -- outside this file, in the build -- LAPACK and thread safety.
//
// WHY A SEPARATE BINARY AT ALL. Three reasons, and the third is the one that
// earns it. convdiff carries fifty options and nine problems, so a timing
// taken from it is a timing of a configuration nobody can restate in a
// sentence. TraceAssemblyMode is a performance axis convdiff does not expose
// at all, so it could not be measured from there without adding a flag to a
// miniapp that has no other use for it. And a performance option that changes
// the ANSWER is a defect and not a speedup -- so this prints the error norms
// and the Newton count on every run, next to the time, because a
// configuration that converges in six steps where the baseline took eight is
// faster for a reason that has nothing to do with the option under test.
//
// THE RESULT THIS HARNESS WAS BUILT TO FIND, recorded here because it is
// about the LIBRARY and not about this file.
//
// **The single biggest host win on this problem is -gm 0, and it is worth
// 2.9x.** n = 128, order 3, k = 0.1: 49.6 s at the default against 17.1 s
// with it. The default is not merely slower, it is LESS ACCURATE -- err_t
// 2.09e-07 against 3.86e-08 -- and the two facts have one cause.
//
// DarcyOperator::ImplicitSolve builds a preconditioner for the hybridized
// nonlinear trace solve, names it in the reported solver string, and then
// never attaches it: darcyop.cpp says so in a comment, and says the
// correction is left alone because it would move the recorded iteration
// counts of the -nlc -nls regressions. SetTraceSolveLevel() is the only
// thing that attaches one, and no default path calls it. So the string
// reads "Newton+GMRES+GS" while an UNPRECONDITIONED GMRES runs, hits its
// 1000-iteration cap on every Newton step, and returns an inexact step --
// which is why the answer is worse as well as dearer. The profile of the
// default run is 37% SparseMatrix::AddMult and 30% Vector::operator*, which
// is a Krylov method orthogonalising against a subspace it never gets to
// discard.
//
// The ladder, same case: -gm 0 (UMFPack preconditioner) 17.1 s, -gm 1 (GS,
// attached) 54.6 s, the default 49.6 s, -gm 2 (matrix-free, nothing to
// precondition) 239.9 s at n = 96 where -gm 0 is 11.3 s. -gm 0 and -gm 1
// agree on the answer to six digits, which is the check that they solve the
// same problem and the default does not.
//
// THREE THINGS THAT DO NOT WORK HERE, each checked rather than assumed,
// because each is the first thing a reader will reach for.
//
// **Partial and element assembly are not available to a hybridized HDG
// solve at all.** DarcyForm::EnableHybridization() refuses any assembly
// level but LEGACY (darcyform.cpp, "Hybridization not supported for this
// assembly level"), and EnableReduction() refuses likewise. -pa on convdiff
// therefore silently turns the hybridization OFF rather than accelerating
// it. There is nothing to measure.
//
// **MFEM's template library (mfem.org/tperformance) cannot reach this
// method.** grep Face over all eight fem/t*.hpp returns ZERO, and
// TBilinearForm has no AddInteriorFaceIntegrator: the templated assembly
// covers volumetric mass and diffusion kernels on tensor-product elements
// and has no face terms of any kind. An HDG solve is face-coupled local
// blocks plus a trace system, so the templated path has no counterpart for
// the part that costs -- this is a structural absence, not a gap somebody
// could fill with a new kernel.
//
// **LAPACK makes this SLOWER, at MKL_NUM_THREADS=1.** MFEM_USE_LAPACK puts
// LUFactors::Factor/Solve on dgetrf_/dgetrs_, which sounds like exactly what
// an element-local HDG solve wants. Measured on the case above: 19.7 and
// 20.7 s against the same build without it at 17.1 and 17.4. The blocks are
// about 32x32 at order 3 in 2-D, which is small enough that MKL's call
// overhead beats an inlined loop.
//
// **AND MKL'S OWN THREADING IS A CLIFF, IN EVERY BUILD HERE INCLUDING THE
// ONE WITHOUT OPENMP.** n = 32, order 2: 0.89 s at MKL_NUM_THREADS=1, 17.0 s
// at 2, 54.1 s at 4 -- a per-element factorisation is far too small to
// thread, and the runtime spins. This tree links -lgomp through
// ONEAPI_MKL_LIB whatever MFEM_USE_OPENMP says, so the serial build is NOT
// immune: a profile of the -gm 0 run taken without pinning spent 63% of its
// samples inside libgomp. **Set MKL_NUM_THREADS=1 for every run of this
// harness that is not deliberately measuring MKL.** The same cliff is
// recorded independently in meq's MEASUREMENTS.md as M-58 and M-59;
// DarcyHybridization::AssemblyMode::Threaded is immune to it, because MKL
// suppresses itself inside an active OpenMP region.
//
// **AND -thr IS REFUSED ON THIS PROBLEM, WHICH IS WHY -lin EXISTS.** It used
// to return NaN deterministically at two threads and above -- nine runs out
// of nine, and from convdiff as well as from here, so it was never this
// file's doing; DarcyHybridization::MultNL() now ABORTS instead of
// corrupting, naming SetAssemblyMode(). So no threaded figure can be taken in
// the default arm at all. `-lin` drops the Burgers term from the operator AND
// the source, which restores the linear-gradient cache and makes the element
// loops safe; every threaded number below was taken there.
//
// **THE LEG TABLE, which is what this harness is for.** `-n 128 -o 3 -lin
// -gm 0`, MKL_NUM_THREADS=1, seconds, and the point is which columns move:
//
//     threads        1       2       4       8     speedup
//     setup       0.465   0.426   0.423   0.416     1.12x   <-- flat
//     computeH    0.880   0.609   0.471   0.418     2.11x
//     npctrav     0.195   0.106   0.057   0.050     3.90x
//     remainder   2.918   2.880   2.834   2.831     1.03x   <-- flat
//     total       4.458   4.021   3.785   3.715     1.20x
//
// `npctrav` is NPCReduce + NPCRecover, threaded this session and previously
// the flattest column on the table. What is left flat is the two ends:
// **setup**, which is Assemble/FormLinearSystem and an upstream
// `BilinearForm` change (see doc/HDG-ELEMENT-LOCAL-PARALLELISM.md), and the
// **remainder**, three quarters of which is the direct trace solve -- 15.4%
// of samples in libumfpack and 18.7% in MKL, and UMFPACK does not thread.
//
// **TWO THINGS OUTSIDE THE CODE THAT ARE WORTH MORE THAN THEY LOOK.**
// `OMP_WAIT_POLICY=passive` is worth **1.12x on the whole run** here (4.209 ->
// 3.766 s at eight threads): with only ~1 s of 3.7 s inside a parallel
// region, idle threads spin at barriers, and a profile of the default run
// puts **24.7% of its samples in libgomp** -- the largest single DSO, ahead
// of UMFPACK. And `-lfac 1` on the HOST is a pessimisation, not an
// optimisation: it takes computeH from 0.418 s to 1.075 s and stops it
// scaling, because the batched route replaces an OpenMP loop with an
// `mfem::forall` that is serial on `-d cpu`.
// Problems 1 and 2 thread correctly, so it is specific to a nonlinear face
// constraint that is a HyperbolicFormIntegrator. Bisected with an
// environment-variable gate on the two OpenMP regions: with MultNL()'s
// region serial and ComputeH()'s still threaded the answer is correct, and
// with the gates the other way round it is not, so the corruption is in
// MultNL()'s element loop and ComputeH() is merely where the damaged blocks
// are first USED -- the abort's backtrace names LUFactors::Solve under
// ComputeElementH, which is downstream of the fault. BuildElementColouring()
// colours on the element-to-element table, i.e. on FACE adjacency, so two
// elements sharing a face are never in one pass and ordinary face conflict
// is excluded.
//
// WHAT THE LEDGER IS FOR. Every batched route in DarcyHybridization falls
// back SILENTLY when its preconditions are not met, and a route that fell
// back and a route that ran are indistinguishable from the wall clock. So the
// ledger asks the library -- CanBatchLocalFactor(), CanBatchTraceAssembly()
// and the rest -- rather than reporting what was requested. hdgdevice.cpp's
// own ledger went stale four times by reporting the request; this one does
// not repeat that.

#include "mfem.hpp"
#include "darcyop.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

// For -fpe. GNU-specific, and the point of it is that a NaN aborts where it is
// CREATED rather than where a Krylov method later notices it -- the abort this
// harness kept producing named IsFinite(beta) in GMRES, which is downstream of
// everything worth looking at.
#ifdef __GNUC__
#include <fenv.h>
#endif

using namespace std;
using namespace mfem;
using namespace mfem::hdg;

namespace
{

// ---------------------------------------------------------------------------
// The problem. convdiff.cpp's Problem::SteadyBurgers, verbatim:
//
//    u = x tanh((1-x)/k) * y tanh((1-y)/k)      on the unit square
//    q = -k grad u
//    1/k q + grad u = 0,   div q + div(u^2/2 (1,1)) = -f
//
// with the free (zero Dirichlet) boundary condition that problem takes, which
// is no boundary marker at all -- every attribute is "free", so the boundary
// face integrators convdiff installs against bdr_is_neumann and
// bdr_is_dirichlet install nothing. That is why this extraction has no
// boundary integrators on g, on h, or on the diffusion stabilization: they
// are inactive in the case being measured, not omitted.
// ---------------------------------------------------------------------------

/// Conductivity. 1.0 is convdiff's own default, so `hdgperf` and
/// `convdiff -p 6 -o 3 -nx 128 -ny 128 -dg -hb -nl -npc -nls 3` are the same
/// run with no flags on either side. It is also the harder case: about 80 s
/// and eight Newton steps here against 50 s and five at k = 0.1, and it is
/// the per-linearisation work that the host options are aimed at.
real_t kappa = 1.0;   ///< set from -k before any of the three are called

/** @brief -lin: drop the Burgers term, from the OPERATOR and the SOURCE alike.

    **The harness needs this to be able to measure threading at all**, and
    that is the whole reason it exists rather than being a convenience. The
    Burgers face constraint makes DarcyHybridization::CopyLinearGradBlocks()
    decline, and without that cache ConstructGrad() evaluates the shared
    integrators once per element -- which MultNL() now refuses to do on
    several threads, because MFEM's stock integrators are not uniformly
    thread safe. So `-thr` on the default problem ABORTS, and every threaded
    figure this harness could otherwise report would be unobtainable.

    It drops the term from fExact() too, so the manufactured solution is
    unchanged and err_q / err_t stay meaningful -- this is the same exact
    solution for a linear diffusion problem, not the Burgers problem with a
    mismatched source. That matters because this harness's own rule is that
    an option which changes the answer is a defect: -lin changes the
    PROBLEM, deliberately and in both places, and the errors it reports are
    its own problem's. */
bool linear_problem = false;

/// The exact potential.
real_t uExact(const Vector &x)
{
   const real_t ux = x(0) * tanh((1. - x(0)) / kappa);
   const real_t uy = x(1) * tanh((1. - x(1)) / kappa);
   return ux * uy;
}

/// The exact flux, q = -k grad u.
void qExact(const Vector &x, Vector &v)
{
   v.SetSize(x.Size());
   const real_t argx = (1. - x(0)) / kappa;
   const real_t argy = (1. - x(1)) / kappa;
   const real_t ux = x(0) * tanh(argx);
   const real_t uy = x(1) * tanh(argy);
   const real_t u = ux * uy;
   const real_t chx = cosh(argx);
   const real_t chy = cosh(argy);
   const real_t u_x = (x(0) == 0.) ? (0.)
                      : (u / x(0) - uy * x(0) / (kappa * chx * chx));
   const real_t u_y = (x(1) == 0.) ? (0.)
                      : (u / x(1) - ux * x(1) / (kappa * chy * chy));
   v(0) = -kappa * u_x;
   v(1) = -kappa * u_y;
}

/// The source, returned NEGATED -- convdiff's convention for every problem.
real_t fExact(const Vector &x)
{
   const real_t argx = (1. - x(0)) / kappa;
   const real_t argy = (1. - x(1)) / kappa;
   const real_t ux = x(0) * tanh(argx);
   const real_t uy = x(1) * tanh(argy);
   const real_t chx = cosh(argx);
   const real_t chy = cosh(argy);
   const real_t u = ux * uy;
   const real_t u_x = (x(0) != 0.) ? (u / x(0) - uy * x(0) / (kappa * chx * chx))
                      : (0.);
   const real_t u_y = (x(1) != 0.) ? (u / x(1) - ux * x(1) / (kappa * chy * chy))
                      : (0.);
   const real_t u_xx = -2. * (u + kappa * uy) / (kappa * kappa * chx * chx);
   const real_t u_yy = -2. * (u + kappa * ux) / (kappa * kappa * chy * chy);
   const real_t divq = -kappa * (u_xx + u_yy);
   // -lin removes the convective term from the source as well as from the
   // operator; see linear_problem.
   const real_t divF = linear_problem ? 0. : (u * (u_x + u_y));
   return -(divq + divF);
}

/// The velocity convdiff builds for this problem. BurgersFlux reads only its
/// VDim, so the values never enter the discretisation -- it is constructed
/// because the flux function's constructor asks a VectorCoefficient for its
/// dimension, and for no other reason.
void cZero(const Vector &x, Vector &v) { v.SetSize(x.Size()); v = 0.; }

/// One line of the "what actually ran" ledger.
///
/// Two questions, not one, and conflating them is how a ledger lies. @a asked
/// is the mode the caller set; @a able is what the library's own predicate
/// says about this problem. A route RUNS only when both hold -- every batched
/// mode in DarcyHybridization falls back silently -- and a predicate that
/// returns true while the mode is left Serial means "this would work", not
/// "this happened". The first draft of this printed @a able alone and
/// reported the local factorisation as batched on a run that never asked for
/// it.
void Say(const char *what, bool asked, bool able,
         const char *ran, const char *fell_back, const char *off)
{
   cout << "  " << left << setw(30) << what;
   if (!asked) { cout << off << (able ? "  (available)" : ""); }
   else if (able) { cout << ran; }
   else { cout << fell_back << "  <-- ASKED FOR AND NOT TAKEN"; }
   cout << "\n";
}

} // namespace

int main(int argc, char *argv[])
{
   StopWatch total_sw, setup_sw, solve_sw;
   total_sw.Start();

   // ---- options -----------------------------------------------------------
   int n = 128;
   int order = 3;
   real_t k = 1.0;
   real_t td = 0.5;
   int nthreads = 0;
   bool threaded = false;
   int local_factor_mode = -1;
   int trace_asm_mode = -1;
   int gradient_mode = -1;
   int prec_type = (int)DarcyOperator::PrecType::Default;
   real_t rtol = -1.;
   int reps = 1;
   bool compare = false;
   bool use_npc = true;
   bool trap_fpe = false;
   bool linear = false;
   int ncols = 0;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--ncells",
                  "Cells per side of the unit square. The default 128 with "
                  "--order 3 is the case this harness exists for: about 80 s "
                  "single-threaded, eight Newton steps.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&k, "-k", "--kappa",
                  "Conductivity. It sets the width of the boundary layer in "
                  "the exact solution, so it changes how hard the problem is "
                  "as well as what it is -- do not vary it between two runs "
                  "being compared.");
   args.AddOption(&td, "-td", "--stab-diff", "Diffusion stabilization.");
   args.AddOption(&rtol, "-rtol", "--newton-rtol",
                  "Relative tolerance of the outer Newton. Negative keeps "
                  "DarcyOperator's default.");
   args.AddOption(&reps, "-rep", "--repeats",
                  "Solve this many times and report the best and the mean. "
                  "One run is a noisy measurement; this is how to see how "
                  "noisy without leaving the process.");
   args.AddOption(&threaded, "-thr", "--threaded", "-no-thr", "--no-threaded",
                  "DarcyHybridization::AssemblyMode::Threaded -- run the "
                  "element-local loops (assembly, residual, Jacobian) on "
                  "several OpenMP threads. The scatter into the trace matrix "
                  "stays serial and ordered. Needs an MFEM_USE_OPENMP and "
                  "MFEM_THREAD_SAFE build and aborts without one.");
   args.AddOption(&nthreads, "-nt", "--num-threads",
                  "omp_set_num_threads(). Zero leaves the environment's "
                  "OMP_NUM_THREADS alone. Note this retunes MKL too in a "
                  "LAPACK build, so it is not only MFEM's thread count.");
   args.AddOption(&local_factor_mode, "-lfac", "--local-factor-mode",
                  "0 = one LUFactors per element, 1 = the whole array through "
                  "BatchedLinAlg. Negative leaves it alone. Needs uniform "
                  "block sizes; the ledger says whether it was taken.");
   args.AddOption(&trace_asm_mode, "-tasm", "--trace-assembly-mode",
                  "0 = scatter per element into an unfinalized SparseMatrix, "
                  "1 = build the CSR from the mesh connectivity once and "
                  "refill it per linearisation. Negative leaves it alone. "
                  "This axis is not exposed by convdiff at all, and on a "
                  "NONLINEAR problem it is the one that should pay: the "
                  "serial route throws the matrix away and rebuilds a linked "
                  "list of one RowNode per nonzero on every Newton step.");
   args.AddOption(&gradient_mode, "-gm", "--gradient-mode",
                  "0 = assemble the trace system and precondition directly, "
                  "1 = assemble and precondition with Gauss-Seidel, 2 = do "
                  "not assemble it, apply it unpreconditioned. Negative "
                  "leaves it alone.");
   args.AddOption(&prec_type, "-prec", "--preconditioner",
                  "0 = the build's own choice, 1 = Gauss-Seidel, 2 = UMFPACK.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string. 'cpu' is the host baseline; "
                  "'omp' puts the mfem::forall kernels -- which is what the "
                  "batched routes are written in -- on OpenMP threads, and is "
                  "a DIFFERENT axis from -thr.");
   args.AddOption(&trap_fpe, "-fpe", "--trap-fpe", "-no-fpe", "--no-trap-fpe",
                  "Raise SIGFPE on the first invalid operation, so a NaN "
                  "aborts where it is produced. Run under gdb with it.");
   args.AddOption(&ncols, "-cols", "--border-columns",
                  "After the solve, time NPCReduce/NPCRecover applied to this "
                  "many right-hand sides against the one factorisation, which "
                  "is what a BORDERED Newton does (meq's free-boundary "
                  "Grad-Shafranov queues N+4 columns through "
                  "DarcyNPCSolver::ArrayMult). 0 disables. The single-vector "
                  "legs are O(elements) per step but these are O(elements x "
                  "columns), so this is the axis on which the traversal "
                  "becomes the largest leg of a step.");
   args.AddOption(&linear, "-lin", "--linear", "-no-lin", "--no-linear",
                  "Drop the Burgers term, from the operator AND the source, "
                  "leaving linear diffusion on the same exact solution. This "
                  "is what makes -thr usable: the Burgers face constraint "
                  "defeats the linear-gradient cache, and without that cache "
                  "the threaded element loop is refused because MFEM's stock "
                  "integrators are not uniformly thread safe. Threaded "
                  "figures can only be taken in this arm.");
   args.AddOption(&use_npc, "-npc", "--npc", "-no-npc", "--no-npc",
                  "Solve by NPC -- Newton on the full (q, u, uhat) system "
                  "with the Jacobian eliminated hybridically -- instead of "
                  "Newton on the trace alone with a local nonlinear solve "
                  "per element. On by default because that is the "
                  "configuration this harness was extracted from, and "
                  "because it is the one whose per-linearisation cost the "
                  "host options are aimed at.");
   args.AddOption(&compare, "-cmp", "--compare", "-no-cmp", "--no-compare",
                  "Print the error norms in convdiff's exact format, so a run "
                  "of this harness can be diffed against the miniapp command "
                  "it was extracted from.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   kappa = k;
   linear_problem = linear;

   if (trap_fpe)
   {
#ifdef __GNUC__
      feenableexcept(FE_INVALID | FE_DIVBYZERO);
#else
      cout << "-fpe needs glibc; ignored.\n";
#endif
   }

   if (nthreads > 0)
   {
#ifdef MFEM_USE_OPENMP
      omp_set_num_threads(nthreads);
#else
      cout << "\n-nt asks for " << nthreads << " threads and this build has no "
           "OpenMP; ignored.\n";
#endif
   }

   Device device(device_config);
   device.Print();

   real_t best = 0., sum = 0.;
   real_t err_q = 0., err_t = 0.;
   int newton_its = 0;
   real_t t_assembly = 0., t_computeH = 0., t_npctrav = 0., t_bordered = 0.;
   long computeH_calls = 0, npctrav_calls = 0;

   for (int rep = 0; rep < reps; rep++)
   {
      setup_sw.Clear();
      setup_sw.Start();

      // ---- mesh and spaces ------------------------------------------------
      Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                        false, 1., 1.);
      const int dim = mesh.Dimension();

      // Exactly convdiff's -dg spaces: an L2 vector flux, an L2 potential and
      // a DG_Interface trace.
      L2_FECollection V_coll(order, dim, BasisType::GaussLobatto);
      // GaussLobatto for BOTH, which is convdiff's choice and not the
      // collection's default. Getting this wrong leaves the SPACE right and
      // the answer right to every printed digit while the residual norms
      // differ by about sqrt(2) -- the load is a different vector in a
      // different basis. That is what it looked like here before it was
      // fixed, and it is why the check on this extraction is the error
      // norms AND the residual history, not the error norms alone.
      L2_FECollection W_coll(order, dim, BasisType::GaussLobatto);
      DG_Interface_FECollection trace_coll(order, dim);

      FiniteElementSpace V_space(&mesh, &V_coll, dim);
      FiniteElementSpace W_space(&mesh, &W_coll);
      FiniteElementSpace trace_space(&mesh, &trace_coll);

      // ---- coefficients ---------------------------------------------------
      ConstantCoefficient kcoeff(k);
      ConstantCoefficient ikcoeff(1. / k);
      VectorFunctionCoefficient ccoeff(dim, cZero);
      FunctionCoefficient tcoeff(uExact);
      ProductCoefficient gcoeff(-1., tcoeff);
      FunctionCoefficient fcoeff(fExact);
      VectorFunctionCoefficient qcoeff(dim, qExact);

      // ---- the form -------------------------------------------------------
      // The third argument is the SIGN CONVENTION and it defaults to true --
      // the symmetric system with -Bt in the flux equation. convdiff takes
      // the default; passing false here flipped it, and the symptom was not a
      // wrong answer but an inner GMRES that would not converge at all while
      // the assembly timing stayed identical to the miniapp's.
      DarcyForm darcy(&V_space, &W_space);

      // convdiff's boundary markers for this problem. Every attribute is
      // FREE -- problem 6 takes the zero-Dirichlet "free BC" -- so both
      // arrays are all-zero and every integrator registered against them is
      // inactive. They are registered anyway, because the registration is not
      // inert: DarcyForm::Assemble() walks B->GetBFBFI_Marker() and installs
      // one boundary flux constraint integrator PER ENTRY, so a B with no
      // boundary face integrator at all is a different code path from a B
      // with one carrying an empty marker.
      const int bdr_attrs = mesh.bdr_attributes.Size() > 0
                            ? mesh.bdr_attributes.Max() : 1;
      Array<int> bdr_is_dirichlet(bdr_attrs), bdr_is_neumann(bdr_attrs);
      bdr_is_dirichlet = 0;
      bdr_is_neumann = 0;

      // -nl puts the flux mass on the NONLINEAR slot; that is what makes the
      // local operator nonlinear and is not incidental to the measurement.
      NonlinearForm *Mqnl = darcy.GetFluxMassNonlinearForm();
      Mqnl->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));

      NonlinearForm *Mtnl = darcy.GetPotentialMassNonlinearForm();
      // Diffusion stabilization, centered (convdiff's default is -ce).
      // convdiff guards this whole block on td > 0, so -td 0 drops it -- and
      // that makes -td 0 the control that isolates the OTHER face
      // integrator, because it takes the face slot from two integrators to
      // one and so out of SumNLFIntegrator entirely.
      if (td > 0.)
      {
         Mtnl->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
         Mtnl->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                    bdr_is_neumann);
      }

      // The Burgers nonlinearity, through the HDG numerical flux. -lin drops
      // it, which is what makes the problem threadable at all; see
      // linear_problem. The flux objects are still CONSTRUCTED either way so
      // that the two arms differ in the integrators alone.
      unique_ptr<FluxFunction> FluxFun(new BurgersFlux(dim));
      unique_ptr<NumericalFlux> FluxSolver(
         new HDGFlux(*FluxFun, HDGFlux::HDGScheme::HDG_1));
      if (!linear_problem)
      {
         Mtnl->AddDomainIntegrator(
            new HyperbolicFormIntegrator(*FluxSolver, 0, -1.));
         Mtnl->AddInteriorFaceIntegrator(
            new HyperbolicFormIntegrator(*FluxSolver, 0, -1.));
         Mtnl->AddBdrFaceIntegrator(
            new HyperbolicFormIntegrator(*FluxSolver, 0, -1.));
      }

      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      B->AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.)));
      B->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-2.)),
         bdr_is_neumann);

      // ---- hybridization --------------------------------------------------
      Array<int> ess_flux_tdofs_list;   // empty: no essential flux dofs here
      darcy.EnableHybridization(&trace_space,
                                new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs_list);

      DarcyHybridization *dh = darcy.GetHybridization();

      // Every mode is set here, BEFORE Assemble(), which is the contract for
      // all of them.
      if (threaded)
      {
         dh->SetAssemblyMode(DarcyHybridization::AssemblyMode::Threaded);
      }
      if (local_factor_mode >= 0)
      {
         dh->SetLocalFactorMode(local_factor_mode == 1
                                ? DarcyHybridization::LocalFactorMode::Batched
                                : DarcyHybridization::LocalFactorMode::Serial);
      }
      if (trace_asm_mode >= 0)
      {
         dh->SetTraceAssemblyMode(trace_asm_mode == 1
                                  ? DarcyHybridization::TraceAssemblyMode::Batched
                                  : DarcyHybridization::TraceAssemblyMode::Serial);
      }
      if (gradient_mode >= 0)
      {
         dh->SetGradientMode(gradient_mode == 2
                             ? DarcyHybridization::GradientMode::MatrixFree
                             : DarcyHybridization::GradientMode::Assembled);
      }

      // ---- state and right-hand sides -------------------------------------
      const Array<int> block_offsets(DarcyOperator::ConstructOffsets(darcy));
      BlockVector x(block_offsets), rhs(block_offsets);
      x = 0.;
      rhs = 0.;

      GridFunction q_h, t_h, tr_h;
      q_h.MakeRef(&V_space, x.GetBlock(0), 0);
      t_h.MakeRef(&W_space, x.GetBlock(1), 0);
      tr_h.MakeRef(&trace_space, x.GetBlock(2), 0);
      q_h = 0.;
      t_h = 0.;
      tr_h = 0.;

      // All three forms must exist and must be Update()d onto the rhs blocks:
      // DarcyOperator dereferences its coeffs without a null check and does
      // rhs.Update(g->GetData(), offsets), so a LinearForm owning its own
      // storage segfaults. g and h carry no integrator for this problem --
      // every boundary attribute is free -- but they are still built.
      unique_ptr<LinearForm> gform(new LinearForm);
      gform->Update(&V_space, rhs.GetBlock(0), 0);
      gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(gcoeff),
                                  bdr_is_dirichlet);

      unique_ptr<LinearForm> fform(new LinearForm);
      fform->Update(&W_space, rhs.GetBlock(1), 0);
      fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

      unique_ptr<LinearForm> hform(new LinearForm);
      hform->Update(&trace_space, rhs.GetBlock(2), 0);
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(qcoeff, 2),
                                   bdr_is_neumann);

      setup_sw.Stop();

      // ---- solve ----------------------------------------------------------
      DarcyHybridization::ResetComputeHTime();
      DarcyHybridization::ResetNPCTraversalTime();

      DarcyOperator op(ess_flux_tdofs_list, &darcy,
      {gform.get(), fform.get(), hform.get()},
      {&gcoeff, &fcoeff, (VectorCoefficient*)&qcoeff},
      DarcyOperator::SolverType::Newton, false, false);
      op.SetTraceSolveLevel(gradient_mode);
      op.SetPrecType((DarcyOperator::PrecType) prec_type);
      if (use_npc) { op.SetNPC(); }
      if (rtol > 0.) { op.SetTolerance(rtol); }

      BackwardEulerSolver ode_solver;
      ode_solver.Init(op);

      solve_sw.Clear();
      solve_sw.Start();
      real_t t = 0., dt = 1.;
      ode_solver.Step(x, t, dt);
      solve_sw.Stop();

      const real_t secs = setup_sw.RealTime() + solve_sw.RealTime();
      if (rep == 0 || secs < best) { best = secs; }
      sum += secs;

      t_computeH = DarcyHybridization::GetComputeHTime();
      computeH_calls = DarcyHybridization::GetComputeHCalls();
      t_npctrav = DarcyHybridization::GetNPCTraversalTime();
      npctrav_calls = DarcyHybridization::GetNPCTraversalCalls();
      t_assembly = setup_sw.RealTime();

      // ---- the answer -----------------------------------------------------
      int order_quad = max(2, 2 * order + 1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }
      const real_t q_norm = ComputeLpNorm(2., qcoeff, mesh, irs);
      const real_t t_norm = ComputeLpNorm(2., tcoeff, mesh, irs);
      err_q = q_h.ComputeL2Error(qcoeff, irs) / q_norm;
      err_t = t_h.ComputeL2Error(tcoeff, irs) / t_norm;

      // ---- the BORDERED traversal, which is the axis that matters to a
      // free-boundary solver and which the ordinary step does not exercise.
      // NPCReduce/NPCRecover are O(elements x columns) against the
      // integrator-bound legs' O(elements), so their share of a step is set
      // by the column count -- and a single-column run reports the one point
      // on that line where they look negligible.
      if (ncols > 0)
      {
         BlockVector bb(rhs, darcy.GetOffsets()), xf(x, darcy.GetOffsets());
         Vector &x_tr = x.GetBlock(2);

         // A real residual and a factored Jacobian for the legs to read.
         BlockVector r0(darcy.GetOffsets());
         Vector r_tr0;
         dh->NPCResidual(bb, xf, x_tr, r0, r_tr0);
         dh->NPCGradient(xf, x_tr);

         std::vector<BlockVector> rc(ncols), dxc(ncols);
         std::vector<Vector> rtc(ncols), btc(ncols), dtc(ncols);
         Array<const BlockVector *> r_ptr(ncols);
         Array<const Vector *> rt_ptr(ncols), dt_ptr(ncols);
         Array<Vector *> bt_ptr(ncols);
         Array<BlockVector *> dx_ptr(ncols);
         for (int j = 0; j < ncols; j++)
         {
            rc[j].Update(darcy.GetOffsets());
            for (int blk = 0; blk < 2; blk++)
            {
               Vector &rb = rc[j].GetBlock(blk);
               const Vector &r0b = r0.GetBlock(blk);
               for (int i = 0; i < rb.Size(); i++)
               {
                  rb(i) = r0b(i) * (1. + 0.3 * j) + sin(0.41 * i + 1.3 * j);
               }
            }
            rc[j].SyncFromBlocks();
            rtc[j] = r_tr0;
            dxc[j].Update(darcy.GetOffsets());
            r_ptr[j] = &rc[j];
            rt_ptr[j] = &rtc[j];
            bt_ptr[j] = &btc[j];
            dx_ptr[j] = &dxc[j];
         }

         // Timed through the library's own accumulator, so this is the same
         // quantity the npctrav column reports and the two are comparable.
         DarcyHybridization::ResetNPCTraversalTime();
         dh->NPCReduce(r_ptr, rt_ptr, bt_ptr);
         for (int j = 0; j < ncols; j++)
         {
            dtc[j].SetSize(btc[j].Size());
            for (int i = 0; i < dtc[j].Size(); i++)
            {
               dtc[j](i) = cos(0.23 * i + 0.9 * j);
            }
            dt_ptr[j] = &dtc[j];
         }
         dh->NPCRecover(r_ptr, dt_ptr, dx_ptr);
         t_bordered = DarcyHybridization::GetNPCTraversalTime();
      }

      // ---- the ledger, asked of the library rather than of the request ----
      if (rep == 0)
      {
         cout << "\n  what actually ran\n"
              << "  ------------------------------------------------------\n";
         const bool lfac_on = (local_factor_mode == 1);
         const bool tasm_on = (trace_asm_mode == 1);
         Say("element loops", threaded, true,
             "OpenMP threads", "", "one thread");
         Say("local factorisation", lfac_on, dh->CanBatchLocalFactor(),
             "batched", "per element", "per element");
         Say("local solves", lfac_on, dh->CanBatchLocalSolve(),
             "batched", "per element", "per element");
         Say("trace assembly", tasm_on, dh->CanBatchTraceAssembly(),
             "refilled CSR", "rebuilt per step", "rebuilt per step");
         // These three are the AssemblyMode::Batched routes. They are device
         // modes and -bam is not offered here at all, so they are reported
         // only as availability -- nothing in a host run asks for them.
         Say("interior face constraint", false, dh->CanBatchPotFaceAssembly(),
             "", "", "per face");
         Say("NL face residual", false, dh->CanBatchNLFaceResidual(),
             "", "", "per face");
         Say("NL face Jacobian", false, dh->CanBatchNLFaceGrad(),
             "", "", "per face");
         cout << "  ------------------------------------------------------\n";
#ifdef MFEM_USE_OPENMP
         cout << "  OpenMP: yes, max threads " << omp_get_max_threads() << "\n";
#else
         cout << "  OpenMP: NO -- this build cannot thread anything\n";
#endif
#ifdef MFEM_USE_LAPACK
         cout << "  LAPACK: yes -- LUFactors on dgetrf_/dgetrs_\n";
#else
         cout << "  LAPACK: NO -- LUFactors on MFEM's own scalar loops\n";
#endif
#ifdef MFEM_THREAD_SAFE
         cout << "  MFEM_THREAD_SAFE: yes\n";
#else
         cout << "  MFEM_THREAD_SAFE: NO\n";
#endif
      }
   }

   total_sw.Stop();

   if (compare)
   {
      cout << "\n|| q_h - q_ex || / || q_ex || = " << err_q << "\n"
           << "|| t_h - t_ex || / || t_ex || = " << err_t << "\n";
   }

   // One machine-readable line, so a driver script never has to parse prose.
   cout << "\nPERF"
        << " n=" << n
        << " order=" << order
        << " thr=" << (threaded ? 1 : 0)
        << " nt=" << nthreads
        << " lfac=" << local_factor_mode
        << " tasm=" << trace_asm_mode
        << " gm=" << gradient_mode
        << " prec=" << prec_type
        << " npc=" << (use_npc ? 1 : 0)
        << " dev=" << device_config
        << " best=" << fixed << setprecision(3) << best
        << " mean=" << (sum / reps)
        << " setup=" << t_assembly
        << " computeH=" << t_computeH
        << " computeH_calls=" << computeH_calls
        << " npctrav=" << t_npctrav
        << " npctrav_calls=" << npctrav_calls
        << " cols=" << ncols
        << " bordered=" << t_bordered
        << " err_q=" << scientific << setprecision(6) << err_q
        << " err_t=" << err_t
        << "\n";

   return 0;
}
