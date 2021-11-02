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

#include "fem/lor.hpp"
#include "fem/lor_assembly.hpp"

#define MFEM_DEBUG_COLOR 206
#include "general/debug.hpp"

#include <cassert>
#include <cmath>

struct LORBench
{
   const int p, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes_ho;
   Array<int> ess_dofs;
   LORDiscretization lor_disc;
   IntegrationRules irs;
   const IntegrationRule &ir_el;
   FiniteElementSpace &fes_lo;
   BilinearForm a_ho, a_lo;
   const int dofs;
   double mdof;

   LORBench(int p, int c):
      p(p),
      q(2*p + 2),
      n((assert(c>=p),c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      mesh(Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON)),
      fec(p, dim, BasisType::GaussLobatto),
      fes_ho(&mesh, &fec),
      lor_disc(fes_ho, BasisType::GaussLobatto),
      irs(0, Quadrature1D::GaussLobatto),
      ir_el(irs.Get(Geometry::Type::CUBE, 1)),
      fes_lo(lor_disc.GetFESpace()),
      a_ho(&fes_ho),
      a_lo(&fes_lo),
      dofs(fes_ho.GetVSize()),
      mdof(0.0)
   {
      a_lo.AddDomainIntegrator(new DiffusionIntegrator(&ir_el));
      a_ho.AddDomainIntegrator(new DiffusionIntegrator);
      a_ho.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      MFEM_VERIFY(SanityCheck(), "Sanity check failed!");
      tic_toc.Clear();
   }

   bool SanityCheck()
   {
      if (dofs > 4000) { return true; }

      OperatorHandle A_lo;
      tic();
      a_lo.Assemble();
      dbg("Standard LOR time = %f",toc());
      a_lo.FormSystemMatrix(ess_dofs, A_lo);

      tic();
      OperatorHandle A_batched;
      AssembleBatchedLOR(a_lo, fes_ho, ess_dofs, A_batched);
      dbg(" Batched LOR time = %f",toc());

      A_batched.As<SparseMatrix>()->Add(-1.0, *A_lo.As<SparseMatrix>());
      const double max_norm = A_batched.As<SparseMatrix>()->MaxNorm();

      return max_norm < 1.e-15;
   }

   void Standard()
   {
      tic_toc.Start();
      a_lo.Assemble();
      tic_toc.Stop();
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   void Batched()
   {
      tic_toc.Start();
      OperatorHandle A_batched;
      AssembleBatchedLOR(a_lo, fes_ho, ess_dofs, A_batched);
      tic_toc.Stop();
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,4,1)

// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(4,32,4)
#define MAX_NDOFS 2*1024*1024

/// Kernels definitions and registrations
#define Bench_LOR(Type)\
static void LOR_##Type(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   LORBench lor(p, side);\
   if (lor.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { lor.Type(); }\
   bm::Counter::Flags flags = bm::Counter::kIsIterationInvariantRate;\
   state.counters["Dofs/s"] = bm::Counter(lor.dofs, flags);\
   state.counters["MDof/s"] = bm::Counter(lor.Mdofs());\
   state.counters["dofs"] = bm::Counter(lor.dofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(LOR_##Type)\
            -> ArgsProduct( {P_ORDERS,N_SIDES})\
            -> Unit(bm::kMillisecond);

Bench_LOR(Standard)
Bench_LOR(Batched)

/**
 * @brief main entry point
 * --benchmark_filter=LorAssembly/4/16
 * --benchmark_context=device=cuda
 */
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_config = "cpu";
   if (bmi::global_context != nullptr)
   {
      const auto device = bmi::global_context->find("device");
      if (device != bmi::global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
