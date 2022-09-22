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

#include "fem/tmop.hpp"
#include <cassert>
#include <memory>
#include <string>
#include <cmath>

#define MFEM_DEBUG_COLOR 226
#include "general/debug.hpp"

////////////////////////////////////////////////////////////////////////////////
static MPI_Session *mpi = nullptr;
static int config_dev_size = 4; // default 4 GPU per node
static bool config_d1d_eq_q1d = false;

////////////////////////////////////////////////////////////////////////////////
struct TMOP
{
   const int mpi_world_size, p, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   const bool device_is_enabled;
   Mesh smesh;
   int nxyz[3];
   bool set_nxyz;
   int *partitioning;
   ParMesh pmesh;
   TMOP_Metric_302 metric;
   TargetConstructor::TargetType target_t;
   TargetConstructor target_c;
   TMOP_Integrator nlfi;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const Operator *R;
   const IntegrationRule &ir;
   const int dofs;
   ParGridFunction x;
   Vector xl,xe,ye,de;
   double mdof;

   TMOP(int p, int c, bool d1d_eq_q1d):
      mpi_world_size(mpi->WorldSize()),
      p(p),
      q(2*p + (d1d_eq_q1d ? 0 : 2)),
      n((assert(c>=p), c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      device_is_enabled(Device::IsEnabled()),
      smesh(Mesh::MakeCartesian3D(mpi_world_size*nx,ny,nz,Element::HEXAHEDRON)),
      set_nxyz((nxyz[0]=mpi_world_size, nxyz[1]=1, nxyz[2]=1, true)),
      partitioning(smesh.CartesianPartitioning(nxyz)),
      pmesh(MPI_COMM_WORLD, smesh, partitioning),
      target_t(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE),
      target_c(target_t, MPI_COMM_WORLD),
      nlfi(&metric, &target_c),
      fec(p, dim),
      pfes(&pmesh, &fec, dim),
      R(pfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      ir(IntRules.Get(pfes.GetFE(0)->GetGeomType(), q)),
      dofs(pfes.GlobalTrueVSize()),
      x(&pfes),
      xl(pfes.GetVSize()),
      xe(R->Height(), Device::GetMemoryType()),
      ye(R->Height(), Device::GetMemoryType()),
      de(R->Height(), Device::GetMemoryType()),
      mdof(0.0)
   {
      pmesh.SetNodalFESpace(&pfes);
      pmesh.SetNodalGridFunction(&x);
      x.SetTrueVector();
      x.SetFromTrueVector();

      target_c.SetNodes(x);

      pfes.GetProlongationMatrix()->Mult(x.GetTrueVector(), xl);
      R->Mult(xl, xe);
      ye = 0.0;

      nlfi.SetIntegrationRule(ir);
      nlfi.AssemblePA(pfes);
      nlfi.AssembleGradPA(xe, pfes);
   }

   void AddMultPA()
   {
      nlfi.AddMultPA(xe,ye);
      if (device_is_enabled) { MFEM_DEVICE_SYNC; }
      mdof += 1e-6 * dofs;
   }

   void AddMultGradPA()
   {
      nlfi.AddMultGradPA(xe,ye);
      if (device_is_enabled) { MFEM_DEVICE_SYNC; }
      mdof += 1e-6 * dofs;
   }

   void GetLocalStateEnergyPA()
   {
      nlfi.GetLocalStateEnergyPA(xe);
      if (device_is_enabled) { MFEM_DEVICE_SYNC; }
      mdof += 1e-6 * dofs;
   }

   void AssembleGradDiagonalPA()
   {
      nlfi.AssembleGradDiagonalPA(de);
      if (device_is_enabled) { MFEM_DEVICE_SYNC; }
      mdof += 1e-6 * dofs;
   }

   double Mdof() const { return mdof; }
};

#define MAX_NDOFS 8*1024*1024
#define P_ORDERS bm::CreateDenseRange(1,4,1)
#define P_SIDES bm::CreateDenseRange(10,88,1)

/**
  Kernels definitions and registrations
*/
#define BENCHMARK_TMOP(Bench)\
static void Bench(bm::State &state){\
   const int p = state.range(0);\
   const int c = state.range(1);\
   TMOP ker(p, c, config_d1d_eq_q1d);\
   if (mpi->WorldSize() > (c*c*c)/(p*p*p)) { state.SkipWithError("MIN_NDOFS"); }\
   if (ker.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { ker.Bench(); }\
   const bm::Counter::Flags isRate = bm::Counter::kIsRate; \
   state.counters["MDofs"] = bm::Counter(ker.Mdof(), isRate); \
   state.counters["Dofs"] = bm::Counter(ker.dofs); \
   state.counters["P"] = bm::Counter(p); \
   state.counters["Ranks"] = bm::Counter(mpi->WorldSize()); \
}\
BENCHMARK(Bench)\
-> ArgsProduct( {P_ORDERS,P_SIDES})\
-> Unit(bm::kMillisecond)\
-> Iterations(100);

/// creating/registering
//BENCHMARK_TMOP(AddMultPA)
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
      bm::BenchmarkReporter *file_reporter = NoReporter();
      bm::BenchmarkReporter *display_reporter = NoReporter();
      bm::RunSpecifiedBenchmarks(display_reporter, file_reporter);
   }
#endif
   return 0;
}

#endif // MFEM_USE_BENCHMARK
