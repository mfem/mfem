// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

#include "fem/tmop.hpp"
#include <cassert>
//#include <memory>
//#include <string>
//#include <cmath>

#define MFEM_DEBUG_COLOR 226
#include "general/debug.hpp"

////////////////////////////////////////////////////////////////////////////////
static MPI_Session *mpi = nullptr;
static int config_dev_size = 4; // default 4 GPU per node

static bool config_d1d_eq_q1d = false;
static int config_max_number_of_dofs_per_ranks = 2*1024*1024;

////////////////////////////////////////////////////////////////////////////////
struct TMOP
{
   const int p, q, n, mpi_world_size, nx, ny, nz, dim = 3;
   const bool pa, check_x, check_y, check_z, checked, barriered;
   Mesh smesh;
   const int ne;
   //int nxyz[3] = {mpi_world_size,1,1};
   //int *partitioning;
   ParMesh pmesh;
   TMOP_Metric_302 metric;
   TargetConstructor::TargetType target_t;
   TargetConstructor target_c;
   TMOP_Integrator *he_nlf_integ;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const Operator *R;
   const IntegrationRule &ir;
   const int dofs;
   ParGridFunction x;
   Vector b,xl,xe,ye,de;
   ParNonlinearForm a;
   const AssemblyLevel assembly_level;
   TMOPNewtonSolver solver;
   CGSolver *cg;
   double mdof;

   TMOP(int p, int c, bool d1d_eq_q1d, bool pa = true):
      p(p),
      q(2*p + (d1d_eq_q1d ? 0 : 2)),
      n((assert(c>=p), c/p)),
      mpi_world_size(mpi->WorldSize()),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      pa(pa),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      barriered((MPI_Barrier(MPI_COMM_WORLD), true)),
      smesh(Mesh::MakeCartesian3D(mpi_world_size*nx,ny,nz,Element::HEXAHEDRON)),
      ne((assert(smesh.GetNE() == mpi_world_size*nx*ny*nz), smesh.GetNE())),
      //nxyz(mpi_world_size,1,1),
      //partitioning(smesh.CartesianPartitioning(nxyz)),
      pmesh(MPI_COMM_WORLD, smesh/*, partitioning*/),
      target_t(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE),
      target_c(target_t, MPI_COMM_WORLD),
      he_nlf_integ(new TMOP_Integrator(&metric, &target_c)),
      fec(p, dim),
      pfes(&pmesh, &fec, dim),
      R(pfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      ir(IntRules.Get(pfes.GetFE(0)->GetGeomType(), q)),
      dofs(pfes.GlobalTrueVSize()),
      x(&pfes),
      b(0),
      xl(pfes.GetVSize()),
      xe(R->Height(), Device::GetMemoryType()),
      ye(R->Height(), Device::GetMemoryType()),
      de(R->Height(), Device::GetMemoryType()),
      a(&pfes),
      assembly_level(pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY),
      solver(pfes.GetComm(), ir),
      cg(new CGSolver(MPI_COMM_WORLD)),
      mdof(0.0)
   {
      dbg("mpi_world_size:%d",mpi_world_size);
      pmesh.SetNodalFESpace(&pfes);
      pmesh.SetNodalGridFunction(&x);
      x.SetTrueVector();
      x.SetFromTrueVector();

      target_c.SetNodes(x);
      he_nlf_integ->SetExactActionFlag(false);
      he_nlf_integ->SetIntegrationRule(ir);

      a.SetAssemblyLevel(assembly_level);
      a.AddDomainIntegrator(he_nlf_integ);

      if (pa) { a.Setup(); }

      solver.SetInitialScale(1.0);
      solver.SetOperator(a);
      {
         const int max_lin_iter    = 100;
         const double linsol_rtol  = 1e-12;
         const int verbosity_level = 0;
         cg->SetMaxIter(max_lin_iter);
         cg->SetRelTol(linsol_rtol);
         cg->SetAbsTol(0.0);
         cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      }
      solver.SetPreconditioner(*cg);

      pfes.GetProlongationMatrix()->Mult(x.GetTrueVector(), xl);
      const double tauval = solver.MinDetJpr_3D(&pfes,xl);
      dbg("tauval: %.15e",tauval);

      R->Mult(xl, xe);
      ye = 0.0;

      he_nlf_integ->SetIntegrationRule(ir);
      he_nlf_integ->AssemblePA(pfes);
      he_nlf_integ->AssembleGradPA(xe, pfes);
   }

   ~TMOP() { dbg("~"); }

   void SolverMult()
   {
      solver.Mult(b, x.GetTrueVector());
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void AddMultPA()
   {
      he_nlf_integ->AddMultPA(xe,ye);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void AddMultGradPA()
   {
      he_nlf_integ->AddMultGradPA(xe,ye);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void GetLocalStateEnergyPA()
   {
      const double energy = he_nlf_integ->GetLocalStateEnergyPA(xe);
      MFEM_DEVICE_SYNC;
      MFEM_CONTRACT_VAR(energy);
      mdof += 1e-6 * dofs;
   }

   void AssembleGradDiagonalPA()
   {
      he_nlf_integ->AssembleGradDiagonalPA(de);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   double Mdof() const { return mdof; }
};

/*static void OrderSideArgs(bmi::Benchmark *b)
{
   const auto est = [](int c, int p)
   {
      const int n = c*p;
      return (n+1)*(n+1)*(n+1);
   };
   for (int p = 1; p <= 4; ++p)
   {
      for (int c = 10; est(c,p) <= 1*1024*1024; c += 1)
      {
         b->Args({p,c});
      }
   }
}*/

#define P_ORDERS bm::CreateDenseRange(1,4,1)
#define P_SIDES bm::CreateDenseRange(10,64,1)

/**
  Kernels definitions and registrations
*/
#define BENCHMARK_TMOP(Bench)\
static void Bench(bm::State &state){\
   const int p = state.range(0);\
   const int c = state.range(1);\
   TMOP ker(p, c, config_d1d_eq_q1d);\
   while (state.KeepRunning()) { ker.Bench(); }\
   const bm::Counter::Flags isRate = bm::Counter::kIsRate;\
   state.counters["MDofs"] = bm::Counter(ker.Mdof(), isRate);\
   state.counters["Dofs"] = bm::Counter(ker.dofs);\
   state.counters["P"] = bm::Counter(p);\
   state.counters["Ranks"] = bm::Counter(mpi->WorldSize());\
}\
BENCHMARK(Bench) \
    -> ArgsProduct({P_ORDERS,P_SIDES})\
    -> Unit(bm::kMillisecond);
//-> Apply(OrderSideArgs)

/// creating/registering
//BENCHMARK_TMOP(AddMultPA)
//BENCHMARK_TMOP(SolverMult)
BENCHMARK_TMOP(AddMultGradPA)
//BENCHMARK_TMOP(GetLocalStateEnergyPA)
//BENCHMARK_TMOP(AssembleGradDiagonalPA)

/**
 * @brief main entry point
 * --benchmark_filter=AddMultPA/4
 * --benchmark_context=dev=cuda
 */
int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   mfem::MPI_Session main_mpi(argc, argv);
   mpi = &main_mpi;
#endif

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string config_device = "cpu";

   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("dev", config_device); // dev=cuda
      bmi::FindInContext("ndev", config_dev_size); // ndev=4
      bmi::FindInContext("nmax", config_max_number_of_dofs_per_ranks);
      bmi::FindInContext("peqq", config_d1d_eq_q1d);
   }

   const int mpi_rank = mpi->WorldRank();
   const int mpi_size = mpi->WorldSize();
   const int dev = config_dev_size > 0 ? mpi_rank % config_dev_size : 0;
   dbg("[MPI] rank: %d/%d, using device #%d", 1+mpi_rank, mpi_size, dev);

   Device device(config_device.c_str(), dev);
   if (mpi->Root()) { device.Print(); }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

#ifndef MFEM_USE_MPI
   bm::RunSpecifiedBenchmarks(&CR);
#else
   if (mpi->Root()) { bm::RunSpecifiedBenchmarks(&CR); }
   else
   {
      // No display_reporter and file_reporter
      //bm::BenchmarkReporter *file_reporter = NoReporter();
      bm::BenchmarkReporter *display_reporter = NoReporter();
      bm::RunSpecifiedBenchmarks(display_reporter);
      //bm::RunSpecifiedBenchmarks(display_reporter, file_reporter);
   }
#endif

   return 0;
}

#endif // MFEM_USE_BENCHMARK
