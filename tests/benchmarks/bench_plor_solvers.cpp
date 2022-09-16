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

#include <cmath>
#include <memory>
#include <functional>

#include "bench.hpp"

//#include "mfem.hpp"

#include "miniapps/solvers/lor_mms.hpp"
bool grad_div_problem = false;

#include "fem/lor/lor_batched.hpp"
#include "fem/lor/lor_ads.hpp"
#include "fem/lor/lor_ams.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/lor/lor.hpp"

#define MFEM_NVTX_COLOR Lime
#include "general/nvtx.hpp"

#define MFEM_DEBUG_COLOR 206
#include "general/debug.hpp"

#ifndef MFEM_USE_CUDA
using cusparseHandle_t = void*;
#define cusparseCreate(...)
#endif // MFEM_USE_CUDA


enum FE {H1 = 0, ND, RT, L2};
bool operator ==(int a, FE fe) { return a == static_cast<int>(fe); }

////////////////////////////////////////////////////////////////////////////////
Mesh MakeCartesianMesh(int p, int requested_ndof, int dim)
{
   const int ne = std::max(1, (int)std::ceil(requested_ndof / pow(p, dim)));
   if (dim == 2)
   {
      const int nx = sqrt(ne);
      const int ny = ne / nx;
      return Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL);
   }
   else
   {
      const int nx = cbrt(ne);
      const int ny = sqrt(ne / nx);
      const int nz = ne / nx / ny;
      const bool sfc_ordering = p > 1;
      return Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                   1.0,1.0,1.0, sfc_ordering);
   }
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
   const int p;
   const double kappa;
   ParMesh mesh;

   const bool H1, ND, RT, L2;
   const HYPRE_Int global_ne;

   FunctionCoefficient f_coeff, u_coeff;
   VectorFunctionCoefficient f_vec_coeff, u_vec_coeff;

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

   // Init cuSparse before timings,
   std::function<cusparseHandle_t()> InitCuSparse = [&]()
   {
      NVTX("InitCuSparse");
      cusparseCreate(&cusph);
      return cusph;
   };
   cusparseHandle_t cusph;

   // "HO Assemble", in bilinearform_ext.cpp
   // "HO Apply", in bilinearform_ext.cpp
   // "AMG Setup" in hypre.cpp
   // "AMG V-cycle", in hypre.cpp
   // "LOR Assemble" lor_batched.cpp:494
   // "RAP", in pbilinearform.cpp:130

   //        sw_setup = sw_setup_PA + sw_setup_LOR + sw_setup_AMG;
   StopWatch sw_setup,  sw_setup_PA,  sw_setup_LOR,  sw_setup_AMG;

   StopWatch sw_apply;

   PLOR_Solvers_Bench(int p, int requested_ndof, int fe) :
      p(p),
      kappa((p+1)*(p+1)), // Penalty used for DG discretizations
      mesh(MakeParCartesianMesh(p, requested_ndof, dim)),
      H1(fe == FE::H1),
      ND(fe == FE::ND),
      RT(fe == FE::RT),
      L2(fe == FE::L2),
      global_ne(mesh.GetGlobalNE()),
      f_coeff(f), u_coeff(u),
      f_vec_coeff(dim, f_vec), u_vec_coeff(dim, u_vec),
      fec(SetFECollection()),
      fes(&mesh, fec),
      a(&fes),
      b(&fes),
      x(&fes),
      ndofs(fes.GlobalTrueVSize()),
      mdof(0.0),
      cg(MPI_COMM_WORLD),
      cusph(InitCuSparse())

   {
      if (RT) { grad_div_problem = true; }
      MFEM_VERIFY(H1||ND||RT||L2, "Bad FE type. Must be 'h', 'n', 'r', or 'l'.");
      MFEM_VERIFY(dim == 3, "Spatial dimension must be 3.");
      if (Mpi::Root()) { mfem::out << "Number of ELMs: " << global_ne << std::endl; }

      if (mesh.ncmesh && (RT || ND))
      { MFEM_ABORT("LOR AMS and ADS solvers are not supported with AMR meshes."); }
      if (Mpi::Root()) { mfem::out << "Number of DOFs: " << ndofs << std::endl; }


      if (H1 || L2) { b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff)); }
      else { b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_vec_coeff)); }
      if (L2)
      {
         // DG boundary conditions are enforced weakly with this integrator.
         b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(u_coeff, -1.0, kappa));
      }
      b.Assemble();

      // In DG, boundary conditions are enforced weakly, so no essential DOFs.
      if (!L2) { fes.GetBoundaryTrueDofs(ess_dofs); }

      if (H1 || L2)
      {
         a.AddDomainIntegrator(new MassIntegrator);
         a.AddDomainIntegrator(new DiffusionIntegrator);
      }
      else { a.AddDomainIntegrator(new VectorFEMassIntegrator); }

      if (ND) { a.AddDomainIntegrator(new CurlCurlIntegrator); }
      else if (RT) { a.AddDomainIntegrator(new DivDivIntegrator); }
      else if (L2)
      {
         a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
         a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
      }
      // TODO: L2 diffusion not implemented with partial assembly
      if (!L2) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

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
         /*HYPRE_BigInt nnz =
            lor_solver->GetLOR().GetAssembledSystem().As<HypreParMatrix>()->NNZ();*/
         //if (Mpi::Root()) { mfem::out << "Number of NNZ:  " << nnz << std::endl; }
         solv_lor.reset(lor_solver);
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
         sw_setup_LOR.Stop();
         /// AMG SETUP
         sw_setup_AMG.Start();
         ((LORSolver<HypreADS>&)*solv_lor).GetSolver().Setup(B, X);
      }
      else
      {
         solv_lor.reset(new LORSolver<HypreAMS>(a, ess_dofs));
         sw_setup_LOR.Stop();
         /// AMG SETUP
         sw_setup_AMG.Start();
         ((LORSolver<HypreAMS>&)*solv_lor).GetSolver().Setup(B, X);
      }
      MFEM_DEVICE_SYNC;
      sw_setup_AMG.Stop();
      sw_setup.Stop();

      if (H1 || L2) { x.ProjectCoefficient(u_coeff);}
      else { x.ProjectCoefficient(u_vec_coeff); }

      cg.SetAbsTol(0.0);
      cg.SetRelTol(0.0); // 1e-12);
      cg.SetMaxIter(40);
      cg.SetPrintLevel(-1); // 3
      cg.SetOperator(*A);
      cg.SetPreconditioner(*solv_lor);
      cg.iterative_mode = false;
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;

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

#define MAX_NDOFS 32*1024*1024

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)

// The different sides of the mesh
#define LOG_NDOFS bm::CreateDenseRange(7,32,1)

template<int FE>
static void pLOR(bm::State &state)
{
   const int fe = FE;
   const int log_ndof = state.range(0);
   const int p = state.range(1);

   const int requested_ndof = pow(2, log_ndof);
   if (p == 1 && log_ndof >= 21) { state.SkipWithError("Problem size"); return; }
   if (p == 2 && log_ndof >= 22) { state.SkipWithError("Problem size"); return; }
   if (p == 3 && log_ndof >= 22) { state.SkipWithError("Problem size"); return; }

   PLOR_Solvers_Bench plor(p, requested_ndof, fe);
   const int ndofs = plor.ndofs;
   if (ndofs/Mpi::WorldSize() > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }

   while (state.KeepRunning()) { plor.Run(); }

   state.counters["MPI"] = bm::Counter(Mpi::WorldSize());
   state.counters["ndofs"] = bm::Counter(ndofs);
   state.counters["p"] = bm::Counter(p);

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

#define PLOR_BENCHMARK(FE)\
BENCHMARK_WITH_UNIT(pLOR<FE>, bm::kMillisecond)\
   ->ArgsProduct({LOG_NDOFS, P_ORDERS})->Iterations(10)

PLOR_BENCHMARK(H1);

//PLOR_BENCHMARK(ND);

//PLOR_BENCHMARK(RT);

//PLOR_BENCHMARK(L2);

int main(int argc, char *argv[])
{
   Mpi::Init();

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string config_device = "cpu";
   int config_dev_size = 4; // default 4 GPU per node

   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("device", config_device); // device=cuda
      bmi::FindInContext("ndev", config_dev_size); // ndev=4
   }

   const int mpi_rank = Mpi::WorldRank();
   const int mpi_size = Mpi::WorldSize();
   const int dev = config_dev_size > 0 ? mpi_rank % config_dev_size : 0;
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
