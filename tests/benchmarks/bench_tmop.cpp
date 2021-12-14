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
#include <memory>
#include <string>
#include <cmath>

#define MFEM_DEBUG_COLOR 226
#include "general/debug.hpp"

////////////////////////////////////////////////////////////////////////////////
static MPI_Session *mpi = nullptr;
static int config_dev_size = 4; // default 4 GPU per node
static int config_serial_refinements = 0;
static int config_parallel_refinements = 0;
static int config_max_number_of_dofs_per_ranks = 2*1024*1024;

////////////////////////////////////////////////////////////////////////////////
struct TMOP
{
   const int p, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked, barriered;
   std::function<ParMesh*(int)> MakeParCartesian3D =
      [&](int mpi_world_size)
   {
      Mesh serial_mesh =
         Mesh::MakeCartesian3D(mpi_world_size*nx,
                               ny,
                               nz,
                               Element::HEXAHEDRON);

      for (int i=0; i<config_serial_refinements; i++)
      { serial_mesh.UniformRefinement(); }

      assert(serial_mesh.GetNE() == mpi_world_size*nx*ny*nz);
      const bool enough_elements = mpi_world_size < serial_mesh.GetNE();
      bool global_ok;
      MPI_Allreduce(&enough_elements, &global_ok,
                    1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
      assert(global_ok);

      int nxyz[3] = {mpi_world_size,1,1};
      int *partitioning = serial_mesh.CartesianPartitioning(nxyz);
      ParMesh *coarse_pmesh =
         new ParMesh(MPI_COMM_WORLD, serial_mesh, partitioning);
      delete [] partitioning;

      for (int i=0; i<config_parallel_refinements; i++)
      { coarse_pmesh->UniformRefinement(); }

      return coarse_pmesh;
   };
   ParMesh *pmesh;
   TMOP_Metric_302 metric;
   TargetConstructor::TargetType target_t;
   TargetConstructor target_c;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const Operator *R;
   const IntegrationRule *ir;
   TMOP_Integrator nlfi;
   const int dofs;
   ParGridFunction x;
   Vector xl,xe,ye,de;
   double mdof;

   TMOP(int p, int c, bool p_eq_q = false):
      p(p),
      q(2*p + (p_eq_q?-1:3)),
      n((assert(c>=p), c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      barriered((MPI_Barrier(MPI_COMM_WORLD), true)),
      pmesh(MakeParCartesian3D(mpi->WorldSize())),
      target_t(TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE),
      target_c(target_t),
      fec(p, dim),
      pfes(pmesh, &fec, dim),
      R(pfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      ir(&IntRules.Get(pfes.GetFE(0)->GetGeomType(), q)),
      nlfi(&metric, &target_c),
      dofs(pfes.GlobalTrueVSize()),
      x(&pfes),
      xl(pfes.GetVSize()),
      xe(R->Height(), Device::GetMemoryType()),
      ye(R->Height(), Device::GetMemoryType()),
      de(R->Height(), Device::GetMemoryType()),
      mdof(0.0)
   {
      pmesh->SetNodalGridFunction(&x);
      target_c.SetNodes(x);
      x.SetTrueVector();
      x.SetFromTrueVector();

      pfes.GetProlongationMatrix()->Mult(x.GetTrueVector(), xl);
      R->Mult(xl, xe);
      ye = 0.0;

      nlfi.SetIntegrationRule(*ir);
      nlfi.AssemblePA(pfes);
      nlfi.AssembleGradPA(xe, pfes);
   }

   ~TMOP()
   {
      delete pmesh;
   }

   void AddMultPA()
   {
      nlfi.AddMultPA(xe,ye);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void AddMultGradPA()
   {
      nlfi.AddMultGradPA(xe,ye);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void GetLocalStateEnergyPA()
   {
      const double energy = nlfi.GetLocalStateEnergyPA(xe);
      MFEM_DEVICE_SYNC;
      MFEM_CONTRACT_VAR(energy);
      mdof += 1e-6 * dofs;
   }

   void AssembleGradDiagonalPA()
   {
      nlfi.AssembleGradDiagonalPA(de);
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   double Mdof() const { return mdof; }
};

static void OrderSideArgs(bmi::Benchmark *b)
{
   const auto est = [](int c) { return (c+1)*(c+1)*(c+1); };
   for (int p = 1; p <= 4; ++p)
   {
      for (int c = 10; est(c) <= 2*1024*1024; c += 1)
      {
         b->Args({p,c});
      }
   }
}

/**
  Kernels definitions and registrations
*/
#define BENCHMARK_TMOP(Bench)\
static void Bench(bm::State &state){\
   const int p = state.range(0);\
   const int c = state.range(1);\
   const bool p_eq_q = false;\
   TMOP ker(p, c, p_eq_q);\
   while (state.KeepRunning()) { ker.Bench(); }\
   const bm::Counter::Flags isRate = bm::Counter::kIsRate;\
   state.counters["MDofs"] = bm::Counter(ker.Mdof(), isRate);\
   state.counters["Dofs"] = bm::Counter(ker.dofs);\
   state.counters["P"] = bm::Counter(p);\
   state.counters["Ranks"] = bm::Counter(mpi->WorldSize());\
}\
BENCHMARK(Bench) \
    -> Apply(OrderSideArgs) \
    -> Unit(bm::kMillisecond);

/// creating/registering
BENCHMARK_TMOP(AddMultPA)
BENCHMARK_TMOP(AddMultGradPA)
//BENCHMARK_TMOP(GetLocalStateEnergyPA)
//BENCHMARK_TMOP(AssembleGradDiagonalPA)

/**
 * @brief main entry point
 * --benchmark_filter=AddMultPA/4
 * --benchmark_context=device=cuda,pref=0
 */
int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   mfem::MPI_Session main_mpi(argc, argv);
   mpi = &main_mpi;
#endif
   MPI_Barrier(MPI_COMM_WORLD);

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string config_device = "cpu";

   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("dev", config_device); // dev=cuda
      bmi::FindInContext("ndev", config_dev_size); // ndev=4
      bmi::FindInContext("sref", config_serial_refinements); // sref=1
      bmi::FindInContext("pref", config_parallel_refinements); // pref=1
      bmi::FindInContext("mdofs", config_max_number_of_dofs_per_ranks);
   }

   //const int mpi_rank = mpi->WorldRank();
   //const int mpi_size = mpi->WorldSize();
   //const int dev = config_dev_size > 0 ? mpi_rank % config_dev_size : 0;
   //dbg("[MPI] rank: %d/%d, using device #%d", 1+mpi_rank, mpi_size, dev);

   Device device(config_device.c_str());//, dev);
   if (mpi->Root()) { device.Print(); }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

#ifndef MFEM_USE_MPI
   bm::RunSpecifiedBenchmarks(&CR);
#else
   if (mpi->Root()) { bm::RunSpecifiedBenchmarks(&CR); }
   else { bm::RunSpecifiedBenchmarks(NoReporter()); }
#endif

   return 0;
}

#endif // MFEM_USE_BENCHMARK
