// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/lor/lor.hpp"
#include "fem/lor/lor_batched.hpp"

#define MFEM_DEBUG_COLOR 206
#include "general/debug.hpp"

#include <functional>
#include <memory>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
static std::string config_device = "hip";
static int config_ndev = 4; // default 4 GPU per node

static bool config_debug = false;
static bool config_save = false;

static int config_cg_max_iter = 32;

////////////////////////////////////////////////////////////////////////////////
namespace analytics
{

static constexpr double pi = M_PI, pi2 = M_PI*M_PI;

// Exact solution for definite Helmholtz problem with RHS corresponding to f
// defined below.
double u(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];
   if (dim == 2) { return sin(x)*sin(y); }
   else { double z = pi*xvec[2]; return sin(x)*sin(y)*sin(z); }
}

double f(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];

   if (dim == 2)
   {
      return sin(x)*sin(y) + 2*pi2*sin(x)*sin(y);
   }
   else // dim == 3
   {
      double z = pi*xvec[2];
      return sin(x)*sin(y)*sin(z) + 3*pi2*sin(x)*sin(y)*sin(z);
   }
}

} // namespace analytics

////////////////////////////////////////////////////////////////////////////////

Mesh MakeCartesianMesh(int p, int requested_ndof, int dim)
{
   const int ne = std::max(1, (int)std::ceil(requested_ndof / pow(p, dim)));
   const int nx = static_cast<int>(cbrt(ne));
   const int ny = static_cast<int>(sqrt(ne / nx));
   const int nz = ne / nx / ny;
   if (Mpi::Root()) { dbg("\033[33mnx:%d ny:%d nz:%d", nx, ny, nz); }
   return Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON);
}

ParMesh MakeParCartesianMesh(int p, int requested_ndof, int dim)
{
   Mesh mesh = MakeCartesianMesh(p, requested_ndof, dim);
   return ParMesh(MPI_COMM_WORLD, mesh);
}

////////////////////////////////////////////////////////////////////////////////
struct PLOR_Solvers_Bench
{
   static constexpr int dim = 3;
   const int num_procs, myid;
   const int p;

   // Init cuSparse before timings, used in Dof_TrueDof_Matrix
   // Nvtx nvtx_cusph = {"cusparseHandle"};
   // std::function<cusparseHandle_t()> InitCuSparse = [&]()
   // {
   //    dbg("InitCuSparse");
   //    MFEM_DEVICE_SYNC;
   //    NVTX("InitCuSparse");
   //    cusparseCreate(&cusph);
   //    MFEM_DEVICE_SYNC;
   //    return cusph;
   // };
   // cusparseHandle_t cusph;

   // Init cuRandGenerator before timings, used in vector Randomize
   // Nvtx nvtx_curng = {"curandGenerator"};
   // std::function<curandGenerator_t()> InitCuRandNumberGenerator = [&]()
   // {
   //    dbg("InitCuRandNumberGenerator");
   //    MFEM_DEVICE_SYNC;
   //    NVTX("InitCuRNG");
   //    curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
   //    curandSetPseudoRandomGeneratorSeed(curng, 0);
   //    MFEM_DEVICE_SYNC;
   //    return curng;
   // };
   // curandGenerator_t curng;

   ParMesh pmesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   ConstantCoefficient one;
   FunctionCoefficient f_coeff, u_coeff;
   Array<int> ess_dofs;
   ParGridFunction x;
   ParBilinearForm a;
   ParLinearForm b;
   const HYPRE_Int ndofs, global_ne;
   Vector X, B;
   OperatorHandle A;
   LORSolver<HypreBoomerAMG> *solv_lor = nullptr;
   CGSolver cg;
   int cg_niter;

   //        sw_setup = sw_setup_PA + sw_setup_LOR + sw_setup_AMG;
   StopWatch sw_setup,  sw_setup_PA,  sw_setup_LOR,  sw_setup_AMG;
   StopWatch sw_solve;

   ~PLOR_Solvers_Bench() { delete solv_lor; }

   PLOR_Solvers_Bench(int order, int requested_ndof) :
      num_procs(Mpi::WorldSize()),
      myid(Mpi::WorldRank()),
      p(order),
      // cusph(InitCuSparse()),
      // curng(InitCuRandNumberGenerator()),
      pmesh(MakeParCartesianMesh(p, requested_ndof, dim)),
      fec(p, dim),
      fes(&pmesh, &fec),
      one(1.0),
      f_coeff(analytics::f), u_coeff(analytics::u),
      x(&fes),
      a(&fes),
      b(&fes),
      ndofs(fes.GlobalTrueVSize()), // builds Dof_TrueDof_Matrix
      global_ne(pmesh.GetGlobalNE()),
      cg(MPI_COMM_WORLD),
      cg_niter(-2) { dbg(); }

   void Setup()
   {
      dbg();
      x.Randomize(); // force random init

      x.ProjectCoefficient(u_coeff);
      fes.GetBoundaryTrueDofs(ess_dofs);

      dbg("Assembling b");
      b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
      b.UseFastAssembly(true);
      b.Assemble();

      dbg("Setting a");
      a.AddDomainIntegrator(new MassIntegrator);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

      dbg("Overall SETUP");
      sw_setup.Clear();
      sw_setup_PA.Clear();
      sw_setup_LOR.Clear();
      sw_setup_AMG.Clear();

      sw_setup.Start();

      dbg("PA SETUP");
      MFEM_DEVICE_SYNC;
      sw_setup_PA.Start();
      a.Assemble();
      a.FormLinearSystem(ess_dofs, x, b, A, X, B);
      MFEM_DEVICE_SYNC;
      sw_setup_PA.Stop();

      dbg("LOR SETUP");
      MFEM_DEVICE_SYNC;
      sw_setup_LOR.Start();
      solv_lor = new LORSolver<HypreBoomerAMG>(a, ess_dofs);
      MFEM_DEVICE_SYNC;
      sw_setup_LOR.Stop();

      dbg("AMG SETUP");
      solv_lor->GetSolver().SetPrintLevel(0);
      MFEM_DEVICE_SYNC;
      sw_setup_AMG.Start();
      solv_lor->GetSolver().Setup(B, X);
      MFEM_DEVICE_SYNC;
      sw_setup_AMG.Stop();

      sw_setup.Stop();

      cg.SetRelTol(0.0);
      cg.SetAbsTol(0.0);
      cg.SetOperator(*A);
      cg.iterative_mode = false;
      cg.SetPreconditioner(*solv_lor);
      cg.SetMaxIter(config_cg_max_iter);
      cg.SetPrintLevel(config_debug ? 3: -1);

      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      cg_niter = cg.GetNumIterations();

      dbg("Clear all previous calls to Mult (FormLinearSystem, Mult)");
      sw_solve.Clear();
      cg.SwAxpy().Clear();
      cg.SwOper().Clear();
      cg.SwPrec().Clear();
      cg.SwPdot().Clear();
      solv_lor->GetSolver().sw_apply.Clear();
      a.SwApplyPA().Clear();
   }

   void Run()
   {
      MFEM_DEVICE_SYNC;
      sw_solve.Start();
      cg.Mult(B,X);
      MFEM_DEVICE_SYNC;
      sw_solve.Stop();
      assert(cg.GetNumIterations() == cg_niter);
   }

   BatchedLORAssembly *GetBatchedLOR() const
   {
      return solv_lor->GetLOR().GetBatchedLOR();
   }

   /// Setup
   double T_OUTER_ALL_Setup() { return sw_setup.RealTime(); }
   double T_OUTER_0_PA_Setup() { return sw_setup_PA.RealTime(); }
   double T_OUTER_1_LOR_Setup() { return sw_setup_LOR.RealTime(); }
   double T_OUTER_2_AMG_Setup() { return sw_setup_AMG.RealTime(); }

   double T_INNER_LOR_Setup() { return GetBatchedLOR()->sw_LOR.RealTime(); }
   double T_INNER_RAP_Setup() { return GetBatchedLOR()->sw_RAP.RealTime(); }
   double T_INNER_BC_Setup() { return GetBatchedLOR()->sw_BC.RealTime(); }

   double T_INNER_AMG_Setup() { return solv_lor->GetSolver().sw_setup.RealTime(); }

   /// Solve/Apply
   double T_OUTER_ALL_Solve() { return sw_solve.RealTime(); }

   double T_INNER_CG_Axpy() { return cg.SwAxpy().RealTime(); }
   double T_INNER_CG_Oper() { return cg.SwOper().RealTime(); }
   double T_INNER_CG_Prec() { return cg.SwPrec().RealTime(); }
   double T_INNER_CG_pDot() { return cg.SwPdot().RealTime(); }

   double T_INNER_AMG_Apply() { return solv_lor->GetSolver().sw_apply.RealTime(); }
   double T_INNER_PA_Apply() { return a.SwApplyPA().RealTime(); }
};

// [0] Requested log_ndof
// 30 max: 1076.88M NDOFs @ 1024 GPU on Lassen
#define LOG_NDOFS bm::CreateDenseRange(31,33,1)

// Maximum number of dofs per rank
// #define MAX_NDOFS 7*1024*1024

// [1] The different orders the tests can run
#define P_ORDERS {6}
// #define P_ORDERS bm::CreateDenseRange(1,2,1)

static void pLOR(bm::State &state)
{
   const int order = state.range(1);
   const int log_ndof = state.range(0);
   const int requested_ndof = static_cast<int>(pow(2, log_ndof));

   PLOR_Solvers_Bench plor(order, requested_ndof);

   const int ndofs = plor.ndofs;
   const int nranks = Mpi::WorldSize();

   dbg("log_ndof:%d order:%d ndofs:%d nranks:%d", log_ndof, order, ndofs, nranks);
   // dbg("%d >? %d", ndofs, nranks*MAX_NDOFS);
   // const bool skip = ndofs > (nranks*MAX_NDOFS);
   // if (skip) { state.SkipWithError("MAX_NDOFS"); return;}

   plor.Setup();

   while (state.KeepRunning()) { plor.Run(); }

   //state.counters["CG"] = bm::Counter(plor.cg_niter);
   state.counters["MPI"] = bm::Counter(nranks);
   state.counters["NDOFs"] = bm::Counter(ndofs);
   //state.counters["NELMs"] = bm::Counter(plor.global_ne);
   state.counters["ORDER"] = bm::Counter(order);

   /// OUTER SETUP = OUTER(PA + LOR + AMG)
   const double setup = plor.T_OUTER_ALL_Setup();
   const double setup_pa = plor.T_OUTER_0_PA_Setup();
   const double setup_lor = plor.T_OUTER_1_LOR_Setup();
   const double setup_amg = plor.T_OUTER_2_AMG_Setup();
   const double setup_pa_lor_amg = setup_pa + setup_lor + setup_amg;
   const double setup_delta = fabs(setup - setup_pa_lor_amg);
   dbg("[setup:outer] %f = pa:%f + lor:%f + amg:%f = %f",
       setup, setup_pa, setup_lor, setup_amg, setup_pa_lor_amg);
   dbg("\033[%dm[setup:outer] delta %f", setup_delta < 1e-3 ? 32:31, setup_delta);
   state.counters["Setup"] = bm::Counter(setup);
   state.counters["Setup_AMG"] = bm::Counter(setup_amg);
   //state.counters["Setup_LOR"] = bm::Counter(setup_lor); // use the other one below
   state.counters["Setup_HO"] = bm::Counter(setup_pa);

   /// OUTER LOR = INNER(LOR + RAP + BC) + eps
   const double s_i_rap = plor.T_INNER_RAP_Setup();
   const double s_i_lor = plor.T_INNER_LOR_Setup();
   const double s_i_bc = plor.T_INNER_BC_Setup();
   const double s_i_lor_rap_bc = s_i_lor + s_i_rap +  s_i_bc;
   const double s_i_delta = fabs(setup_lor - s_i_lor_rap_bc);
   dbg("[setup:inner] %f = lor:%f + rap:%f + bc:%f = %f + eps",
       setup_lor, s_i_lor, s_i_rap, s_i_bc, s_i_lor_rap_bc);
   dbg("\033[%dm[setup:inner] s_i_delta %f",
       s_i_delta < 1e-2 ? 32:31, s_i_delta);
   state.counters["Setup_LOR"] = bm::Counter(s_i_lor);
   state.counters["Setup_RAP"] = bm::Counter(s_i_rap);
   state.counters["Setup_BC"] = bm::Counter(s_i_bc);

   /// OUTER SOLVE = INNER( AXPY + PA(Oper) + AMG(Prec) + pDot)
   /// R*PA*P == Oper, AMG == Prec
   const double solve = plor.T_OUTER_ALL_Solve();

   const double solve_axpy = plor.cg.SwAxpy().RealTime();
   const double solve_oper = plor.cg.SwOper().RealTime();
   const double solve_prec = plor.cg.SwPrec().RealTime();
   const double solve_pdot = plor.cg.SwPdot().RealTime();
   const double solve_sum = solve_axpy + solve_oper + solve_prec + solve_pdot;
   const double solve_delta = fabs(solve - solve_sum);
   dbg("\033[33m[apply] %f = axpy:%f + oper:%f + prec:%f + dot:%f = %f + eps",
       solve, solve_axpy, solve_oper, solve_prec, solve_pdot, solve_sum);
   dbg("\033[%dm[apply] a_cg_delta %f",
       solve_delta < 1e-2 ? 32:31, solve_delta);

   // we don't measure the R.PA.P, just PA

   const double a_i_amg = plor.T_INNER_AMG_Apply();
   dbg("[apply] AMG == Prec: %f = %f", a_i_amg, solve_prec);
   const double a_amg_prec_delta = fabs(a_i_amg - solve_prec);
   dbg("\033[%dm[apply] a_amg_prec_delta %f",
       a_amg_prec_delta < 1e-1 ? 32:31, a_amg_prec_delta);

   bm::Counter::Flags kAvg = bm::Counter::kAvgIterations;
   const double tm_solve_amg = plor.T_INNER_AMG_Apply();
   const double tm_solve_ho = solve - tm_solve_amg;
   state.counters["Solve"] = bm::Counter(solve, kAvg);
   state.counters["Solve_AMG"] = bm::Counter(tm_solve_amg, kAvg);
   state.counters["Solve_HO"] = bm::Counter(tm_solve_ho, kAvg);
   dbg("done");
}

BENCHMARK(pLOR)->Unit(bm::kMillisecond)\
->ArgsProduct( {LOG_NDOFS, P_ORDERS} )->Iterations(10);

// BENCHMARK(pLOR)->Unit(bm::kSecond)->ArgsProduct( {LOG_NDOFS, P_ORDERS} );

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   bm::Initialize(&argc, argv);
   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("device", config_device); // device=cuda
      bmi::FindInContext("debug", config_debug); // debug=true
      bmi::FindInContext("save", config_save);
      bmi::FindInContext("ndev", config_ndev); // has to be set to 1 for bsub
      bmi::FindInContext("cgmi", config_cg_max_iter);
   }
   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

   const int mpi_rank = Mpi::WorldRank();
   const int mpi_size = Mpi::WorldSize();
   const int dev = mpi_rank % config_ndev;

   Device device(config_device.c_str(), dev);
   device.SetGPUAwareMPI();
   if (Mpi::Root()) { device.Print(); }

   dbg("[MPI] %d/%d @ device #%d", 1+mpi_rank, mpi_size, dev);

   bm::ConsoleReporter CR;
   if (Mpi::Root()) { bm::RunSpecifiedBenchmarks(&CR); }
   else
   {
      // No display_reporter and file_reporter
      std::unique_ptr<bm::BenchmarkReporter> file_reporter, display_reporter;
      file_reporter.reset(new NoReporter());
      display_reporter.reset(new NoReporter());
      bm::RunSpecifiedBenchmarks(display_reporter.get(), file_reporter.get());
   }

   MPI_Barrier(MPI_COMM_WORLD);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
