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

#define MFEM_NVTX_COLOR Lime
#include "general/nvtx.hpp"

#define MFEM_DEBUG_COLOR 206
#include "general/debug.hpp"

#include <functional>
#include <memory>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
#ifdef MFEM_USE_CUDA
#include <curand.h>
#else
using cusparseHandle_t = void*;
using curandGenerator_t = void*;
#define cusparseCreate(...)
#define curandCreateGenerator(...)
#define curandSetPseudoRandomGeneratorSeed(...)
#endif // MFEM_USE_CUDA

////////////////////////////////////////////////////////////////////////////////
static int config_ndev = 4; // default 4 GPU per node
static bool config_nxyz = false; // cartesian partitioning in x

static bool config_debug = false;
static bool config_save = false;

static int config_cg_max_iter = 32;

////////////////////////////////////////////////////////////////////////////////
namespace analytics
{
static constexpr double pi = M_PI, pi2 = M_PI*M_PI;

// Exact solution for definite Helmholtz problem with RHS corresponding to f
// defined below.
static double u(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];
   if (dim == 2) { return sin(x)*sin(y); }
   else { double z = pi*xvec[2]; return sin(x)*sin(y)*sin(z); }
}

static double f(const Vector &xvec)
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
namespace kershaw
{

// 1D transformation at the right boundary.
double right(const double eps, const double x)
{
   return (x <= 0.5) ? (2.-eps) * x : 1. + eps*(x-1.);
}

// 1D transformation at the left boundary
double left(const double eps, const double x) { return 1.-right(eps,1.-x); }

// Transition from a value of "a" for x=0, to a value of "b" for x=1.
// Smoothness is controlled by the parameter "s", taking values 0, 1, or 2.
double step(const double a, const double b, const double x, const int s)
{
   if (x <= 0.) { return a; }
   if (x >= 1.) { return b; }
   switch (s)
   {
      case 0: return a + (b-a) * (x);
      case 1: return a + (b-a) * (x*x*(3.-2.*x));
      case 2: return a + (b-a) * (x*x*x*(x*(6.*x-15.)+10.));
      default: MFEM_ABORT("Smoothness values: 0, 1, or 2.");
   }
   return 0.0;
}

// 3D version of a generalized Kershaw mesh transformation, see D. Kershaw,
// "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
// JCP, 39:375–395, 1981.
//
// The input mesh should be Cartesian nx x ny x nz with nx divisible by 6 and
// ny, nz divisible by 2.
//
// The eps parameters are in (0, 1]. Uniform mesh is recovered for epsy=epsz=1.
void kershaw(const double epsy, const double epsz, const int smoothness,
             const double x, const double y, const double z,
             double &X, double &Y, double &Z)
{
   X = x;

   const int layer = 6.0*x;
   const double lambda = (x-layer/6.0)*6;

   // The x-range is split in 6 layers going from left-to-left, left-to-right,
   // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
   switch (layer)
   {
      case 0:
         Y = left(epsy, y);
         Z = left(epsz, z);
         break;
      case 1:
      case 4:
         Y = step(left(epsy, y), right(epsy, y), lambda, smoothness);
         Z = step(left(epsz, z), right(epsz, z), lambda, smoothness);
         break;
      case 2:
         Y = step(right(epsy, y), left(epsy, y), lambda/2.0, smoothness);
         Z = step(right(epsz, z), left(epsz, z), lambda/2.0, smoothness);
         break;
      case 3:
         Y = step(right(epsy, y), left(epsy, y), (1.0+lambda)/2.0, smoothness);
         Z = step(right(epsz, z), left(epsz, z), (1.0+lambda)/2.0, smoothness);
         break;
      default:
         Y = right(epsy, y);
         Z = right(epsz, z);
         break;
   }
}

struct Transformation : VectorCoefficient
{
   double epsy, epsz;
   int dim, s;
   Transformation(int dim, double epsy, double epsz, int s = 0):
      VectorCoefficient(dim),
      epsy(epsy),
      epsz(epsz),
      dim(dim),
      s(s) { }

   using VectorCoefficient::Eval;

   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (dim == 1)
      {
         V[0] = xyz[0]; // no transformation in 1D
      }
      else if (dim == 2)
      {
         double z = 0, zt;
         kershaw(epsy, epsz, s, xyz[0], xyz[1], z, V[0], V[1], zt);
      }
      else // dim == 3
      {
         kershaw(epsy, epsz, s, xyz[0], xyz[1], xyz[2], V[0], V[1], V[2]);
      }
   }
};

} // namespace kershaw

////////////////////////////////////////////////////////////////////////////////
enum FE {H1 = 0, ND, RT, L2};

////////////////////////////////////////////////////////////////////////////////
struct PLOR_Solvers_Bench
{
   static constexpr int dim = 3;
   const int num_procs, myid;
   const int p, c;
   const int n, nx,ny,nz;
   const bool check_x, check_y, check_z, checked;
   const double epsilon, kappa;

   // Init cuSparse before timings, used in Dof_TrueDof_Matrix
   Nvtx nvtx_cusph = {"cusparseHandle"};
   std::function<cusparseHandle_t()> InitCuSparse = [&]()
   {
      MFEM_DEVICE_SYNC;
      NVTX("InitCuSparse");
      cusparseCreate(&cusph);
      MFEM_DEVICE_SYNC;
      return cusph;
   };
   cusparseHandle_t cusph;

   // Init cuRandGenerator before timings, used in vector Randomize
   Nvtx nvtx_curng = {"curandGenerator"};
   std::function<curandGenerator_t()> InitCuRandNumberGenerator = [&]()
   {
      MFEM_DEVICE_SYNC;
      NVTX("InitCuRNG");
      curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(curng, 0);
      MFEM_DEVICE_SYNC;
      return curng;
   };
   curandGenerator_t curng;

   std::function<ParMesh()> GetCoarseKershawMesh = [&]()
   {
      dbg("nx:%d ny:%d nz:%d", nx, ny, nz);
      Mesh smesh =
         Mesh::MakeCartesian3D((config_nxyz?num_procs:1)*nx, ny, nz,
                               Element::HEXAHEDRON);

      // Set the curvature of the initial mesh
      /*if (smoothness > 0)
      {
         dbg("smoothness");
         smesh.SetCurvature(2*smoothness+1);
      }*/

      // Kershaw transformation
      dbg("Kershaw");
      kershaw::Transformation kt(dim, epsilon, epsilon);
      smesh.Transform(kt);

      int *partitioning = nullptr;
      auto GetPartitioning = [&]()
      {
         int nxyz[3] = {num_procs,1,1};
         return config_nxyz ? smesh.CartesianPartitioning(nxyz) : nullptr;
      };
      dbg("ParMesh");
      ParMesh mesh(MPI_COMM_WORLD, smesh, partitioning=GetPartitioning());
      smesh.Clear();
      delete partitioning;

      dbg("GetCoarseKershawMesh done");
      return mesh;
   };
   ParMesh pmesh;

   const bool H1, ND, RT, L2;
   const HYPRE_Int global_ne;

   FunctionCoefficient f_coeff, u_coeff;

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   std::function<FiniteElementCollection*()> SetFECollection = [&]()
   {
      if (H1) { fec = new H1_FECollection(p, dim, b1); }
      if (ND) { fec = new ND_FECollection(p, dim, b1, b2); }
      if (RT) { fec = new RT_FECollection(p-1, dim, b1, b2); }
      if (L2) { fec = new L2_FECollection(p, dim, b1); }
      MFEM_VERIFY(fec, "Bad FE type");
      return fec;
   };
   FiniteElementCollection *fec;
   ParFiniteElementSpace fes;
   ParBilinearForm a;
   ParLinearForm b;
   ParGridFunction x;

   const int ndofs;
   Array<int> ess_dofs;
   double mdof;

   Vector X, B;
   OperatorHandle A;

   std::unique_ptr<Solver> solv_lor;
   HypreBoomerAMG *amg = nullptr;

   CGSolver cg;
   int cg_niter;

   // "HO Assemble", in bilinearform_ext.cpp
   // "HO Apply", in bilinearform_ext.cpp
   // "AMG Setup" in hypre.cpp
   // "AMG V-cycle", in hypre.cpp
   // "LOR Assemble" lor_batched.cpp:494
   // "RAP", in pbilinearform.cpp:130

   //        sw_setup = sw_setup_PA + sw_setup_LOR + sw_setup_AMG;
   StopWatch sw_setup,  sw_setup_PA,  sw_setup_LOR,  sw_setup_AMG;

   StopWatch sw_apply;

   ~PLOR_Solvers_Bench() {delete fec;}

   PLOR_Solvers_Bench(int fe, int order, int side, double epsilon) :
      num_procs(Mpi::WorldSize()),
      myid(Mpi::WorldRank()),
      p(order),
      c(side),
      n((assert(c>=p), c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      epsilon(epsilon),

      cusph(InitCuSparse()),
      curng(InitCuRandNumberGenerator()),

      kappa((p+1)*(p+1)), // Penalty used for DG discretizations
      pmesh(GetCoarseKershawMesh()),
      H1(fe == FE::H1),
      ND(fe == FE::ND),
      RT(fe == FE::RT),
      L2(fe == FE::L2),
      global_ne(pmesh.GetGlobalNE()),
      f_coeff(analytics::f), u_coeff(analytics::u),
      fec(SetFECollection()),
      fes(&pmesh, fec),
      a(&fes),
      b(&fes),
      x(&fes),
      ndofs(fes.GlobalTrueVSize()),
      mdof(0.0),
      cg_niter(-2)

   {
      assert(H1);
      fes.GetBoundaryTrueDofs(ess_dofs);

      b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
      b.Assemble();

      a.AddDomainIntegrator(new MassIntegrator);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

      /// Overall SETUP
      MFEM_DEVICE_SYNC;
      sw_setup.Clear();
      sw_setup_PA.Clear();
      sw_setup_LOR.Clear();
      sw_setup_AMG.Clear();

      sw_setup.Start();

      /// PA SETUP
      sw_setup_PA.Start();
      a.Assemble();
      a.FormLinearSystem(ess_dofs, x, b, A, X, B);
      MFEM_DEVICE_SYNC;
      sw_setup_PA.Stop();

      /// LOR SETUP
      sw_setup_LOR.Start();
      if (H1 || L2)
      {
         auto *lor_solver = new LORSolver<HypreBoomerAMG>(a, ess_dofs);
         solv_lor.reset(lor_solver);
         MFEM_DEVICE_SYNC;
         sw_setup_LOR.Stop();

         /// AMG SETUP
         sw_setup_AMG.Start();
         amg = &((LORSolver<HypreBoomerAMG>&)*solv_lor).GetSolver();
         amg->SetPrintLevel(0);
         amg->Setup(B, X);
      }
      else if (RT && dim == 3)
      {
         solv_lor.reset(new LORSolver<HypreADS>(a, ess_dofs));
         MFEM_DEVICE_SYNC;
         sw_setup_LOR.Stop();

         /// AMG SETUP
         sw_setup_AMG.Start();
         ((LORSolver<HypreADS>&)*solv_lor).GetSolver().Setup(B, X);
      }
      else
      {
         solv_lor.reset(new LORSolver<HypreAMS>(a, ess_dofs));
         MFEM_DEVICE_SYNC;
         sw_setup_LOR.Stop();

         /// AMG SETUP
         sw_setup_AMG.Start();
         ((LORSolver<HypreAMS>&)*solv_lor).GetSolver().Setup(B, X);
      }
      MFEM_DEVICE_SYNC;
      sw_setup_AMG.Stop();
      sw_setup.Stop();

      x.ProjectCoefficient(u_coeff);

      cg.SetRelTol(0.0);
      cg.SetOperator(*A);
      cg.iterative_mode = false;
      cg.SetAbsTol(sqrt(1e-16));
      cg.SetPreconditioner(*solv_lor);
      cg.SetMaxIter(config_cg_max_iter);
      cg.SetPrintLevel(config_debug ? 3: -1);

      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;

      cg_niter = cg.GetConverged() ? cg.GetNumIterations() : -1;

      // Clear all previous calls to Mult (FormLinearSystem, Mult)
      sw_apply.Clear();
      cg.SwAxpy().Clear();
      cg.SwOper().Clear();
      cg.SwPrec().Clear();
      cg.SwPdot().Clear();
      amg->sw_apply.Clear();
      a.SwApplyPA().Clear();
   }

   void Run()
   {
      sw_apply.Start();
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      sw_apply.Stop();
      // make sure we have the same number of iterations
      if (!cg.GetConverged()) { sw_apply.Clear(); }
      else { assert(std::abs(cg.GetNumIterations() - cg_niter) < 4); }
   }

   const LORBase &GetLOR() const
   {
      assert(H1);
      return ((LORSolver<HypreBoomerAMG>&)*solv_lor).GetLOR();
   }

   BatchedLORAssembly *GetBatchedLOR() const { return GetLOR().GetBatchedLOR(); }

   /// Setup
   double T_OUTER_ALL_Setup() { return sw_setup.RealTime(); }
   double T_OUTER_0_PA_Setup() { return sw_setup_PA.RealTime(); }
   double T_OUTER_1_LOR_Setup() { return sw_setup_LOR.RealTime(); }
   double T_OUTER_2_AMG_Setup() { return sw_setup_AMG.RealTime(); }

   double T_INNER_LOR_Setup() { return GetBatchedLOR()->sw_LOR.RealTime(); }
   double T_INNER_RAP_Setup() { return GetBatchedLOR()->sw_RAP.RealTime(); }
   double T_INNER_BC_Setup() { return GetBatchedLOR()->sw_BC.RealTime(); }

   double T_INNER_AMG_Setup() { return amg->sw_setup.RealTime(); }

   /// Apply
   double T_OUTER_ALL_Apply() { return sw_apply.RealTime(); }

   double T_INNER_CG_Axpy() { return cg.SwAxpy().RealTime(); }
   double T_INNER_CG_Oper() { return cg.SwOper().RealTime(); }
   double T_INNER_CG_Prec() { return cg.SwPrec().RealTime(); }
   double T_INNER_CG_pDot() { return cg.SwPdot().RealTime(); }

   double T_INNER_AMG_Apply() { return amg->sw_apply.RealTime(); }
   double T_INNER_PA_Apply() { return a.SwApplyPA().RealTime(); }

};


// [0] The different side sizes
// When generating tex data, at order 6:
//    - 144 max for one rank with LORBatch, 3.04862M dofs
//    - 264 max for one rank with MG*, 18.6096M dofs
//    - 540 max for 160M dofs, 128 ranks
#define P_SIDES bm::CreateDenseRange(12,144,2)

// Maximum number of dofs
#define MAX_NDOFS 1024*1024*1024
//if (plor.dofs > (nranks*MAX_NDOFS)) {state.SkipWithError("MAX_NDOFS");}

// [1] The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)

// [2] The different epsilons dividers
#define P_EPSILONS {1,2,3}

void pLOR(bm::State &state)
{
   const int side = state.range(0);
   const int order = state.range(1);
   const int epsilon = state.range(2);

   PLOR_Solvers_Bench plor(FE::H1, order, side, epsilon);

   const int ndofs = plor.ndofs;
   const int nranks = Mpi::WorldSize();
   if (ndofs > (nranks*MAX_NDOFS)) { state.SkipWithError("MAX_NDOFS"); }

   while (state.KeepRunning()) { plor.Run(); }

   state.counters["NRanks"] = bm::Counter(nranks);
   state.counters["NELMs"] = bm::Counter(plor.global_ne);
   state.counters["NDOFs"] = bm::Counter(ndofs);
   state.counters["P"] = bm::Counter(order);
   state.counters["CGiters"] = bm::Counter(plor.cg_niter);
   state.counters["Epsilon"] = bm::Counter(epsilon);

   /// OUTER SETUP = OUTER(PA + LOR + AMG)
   const double s_all = plor.T_OUTER_ALL_Setup();
   const double s_pa = plor.T_OUTER_0_PA_Setup();
   const double s_lor = plor.T_OUTER_1_LOR_Setup();
   const double s_amg = plor.T_OUTER_2_AMG_Setup();
   const double s_pa_lor_amg = s_pa + s_lor + s_amg;
   const double s_delta = fabs(s_all - s_pa_lor_amg);
   dbg("[setup:outer] %f = pa:%f + lor:%f + amg:%f = %f",
       s_all, s_pa, s_lor, s_amg, s_pa_lor_amg);
   dbg("\033[%dm[setup:outer] delta %f", s_delta < 1e-3 ? 32:31, s_delta);

   state.counters["T0_0_ALL"] = bm::Counter(plor.T_OUTER_ALL_Setup());
   state.counters["T0_1_PA"] = bm::Counter(plor.T_OUTER_0_PA_Setup());
   state.counters["T0_2_LOR"] = bm::Counter(plor.T_OUTER_1_LOR_Setup());
   state.counters["T0_3_AMG"] = bm::Counter(plor.T_OUTER_2_AMG_Setup());

   /// OUTER LOR = INNER(LOR + RAP + BC) + eps
   const double s_i_rap = plor.T_INNER_RAP_Setup();
   const double s_i_lor = plor.T_INNER_LOR_Setup();
   const double s_i_bc = plor.T_INNER_BC_Setup();
   const double s_i_lor_rap_bc = s_i_lor + s_i_rap +  s_i_bc;
   const double s_i_delta = fabs(s_lor - s_i_lor_rap_bc);
   dbg("[setup:inner] %f = lor:%f + rap:%f + bc:%f = %f + eps",
       s_lor, s_i_lor, s_i_rap, s_i_bc, s_i_lor_rap_bc);
   dbg("\033[%dm[setup:inner] s_i_delta %f",
       s_i_delta < 1e-2 ? 32:31, s_i_delta);
   state.counters["T0_I_1_LOR"] = bm::Counter(plor.T_INNER_LOR_Setup());
   state.counters["T0_I_2_RAP"] = bm::Counter(plor.T_INNER_RAP_Setup());
   state.counters["T0_I_3_BC"] = bm::Counter(plor.T_INNER_BC_Setup());

   /// OUTER APPLY = INNER( AXPY + PA(Oper) + AMG(Prec) + pDot)
   /// R*PA*P == Oper, AMG == Prec
   const double a_all = plor.T_OUTER_ALL_Apply();

   const double a_cg_axpy = plor.cg.SwAxpy().RealTime();
   const double a_cg_oper = plor.cg.SwOper().RealTime();
   const double a_cg_prec = plor.cg.SwPrec().RealTime();
   const double a_cg_pdot = plor.cg.SwPdot().RealTime();
   const double a_cg_all = a_cg_axpy + a_cg_oper + a_cg_prec + a_cg_pdot;
   const double a_cg_delta = fabs(a_all - a_cg_all);
   dbg("\033[33m[apply] %f = axpy:%f + oper:%f + prec:%f + dot:%f = %f + eps",
       a_all, a_cg_axpy, a_cg_oper, a_cg_prec, a_cg_pdot, a_cg_all);
   dbg("\033[%dm[apply] a_cg_delta %f",
       a_cg_delta < 1e-2 ? 32:31, a_cg_delta);

   // we don't measure thee R.PA.P, just PA
   /*const double a_i_pa = plor.T_INNER_PA_Apply();
   dbg("[apply] PA == Oper: %f = %f", a_i_pa, a_cg_oper);
   const double a_pa_oper_delta = fabs(a_i_pa - a_cg_oper);
   dbg("\033[%dm[apply] a_pa_oper_delta %f",
       a_pa_oper_delta < 1e-1 ? 32:31, a_pa_oper_delta);*/

   const double a_i_amg = plor.T_INNER_AMG_Apply();
   dbg("[apply] AMG == Prec: %f = %f", a_i_amg, a_cg_prec);
   const double a_amg_prec_delta = fabs(a_i_amg - a_cg_prec);
   dbg("\033[%dm[apply] a_amg_prec_delta %f",
       a_amg_prec_delta < 1e-1 ? 32:31, a_amg_prec_delta);

   bm::Counter::Flags kAvg = bm::Counter::kAvgIterations;
   state.counters["A_ALL"] = bm::Counter(plor.T_OUTER_ALL_Apply(), kAvg);
   state.counters["A_AMG"] = bm::Counter(plor.T_INNER_AMG_Apply(), kAvg);
   state.counters["A_PA"] = bm::Counter(plor.T_INNER_PA_Apply(), kAvg);
}

BENCHMARK(pLOR)->Unit(bm::kMillisecond)\
->ArgsProduct( {P_SIDES, P_ORDERS, P_EPSILONS})->Iterations(10);

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string config_device = "cpu";

   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("device", config_device); // device=cuda
      bmi::FindInContext("debug", config_debug); // debug=true
      bmi::FindInContext("save", config_save);
      bmi::FindInContext("nxyz", config_nxyz); // nxyz=true
      bmi::FindInContext("ndev", config_ndev); // ndev=4
      bmi::FindInContext("cgmi", config_cg_max_iter);
   }

   const int mpi_rank = Mpi::WorldRank();
   const int mpi_size = Mpi::WorldSize();
   const int dev = config_ndev > 0 ? mpi_rank % config_ndev : 0;
   dbg("[MPI] rank: %d/%d, using device #%d", 1+mpi_rank, mpi_size, dev);

   Device device(config_device.c_str(), dev);
   if (Mpi::Root()) { device.Print(); }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

   if (Mpi::Root()) { bm::RunSpecifiedBenchmarks(&CR); }
   else
   {
      // No display_reporter and file_reporter
      // bm::RunSpecifiedBenchmarks(NoReporter());
      bm::BenchmarkReporter *file_reporter = new NoReporter();
      bm::BenchmarkReporter *display_reporter = new NoReporter();
      bm::RunSpecifiedBenchmarks(display_reporter, file_reporter);
   }
   return 0;
}

#endif // MFEM_USE_BENCHMARK
