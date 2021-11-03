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

#define MFEM_DEBUG_COLOR 119
#include "general/debug.hpp"

#include <cassert>
#include <cmath>

constexpr int SEED = 0x100001b3;

struct LORBench
{
   const int p, c, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace mfes, fes_ho;
   Array<int> ess_dofs;
   LORDiscretization lor_disc;
   IntegrationRules irs;
   const IntegrationRule &ir_el;
   FiniteElementSpace &fes_lo;
   BilinearForm a_ho, a_lo;
   GridFunction x;
   const int dofs;
   double mdof;

   LORBench(int p, int side):
      p(p),
      c(side),
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
      mfes(&mesh, &fec, dim),
      fes_ho(&mesh, &fec),
      lor_disc(fes_ho, BasisType::GaussLobatto),
      irs(0, Quadrature1D::GaussLobatto),
      ir_el(irs.Get(Geometry::Type::CUBE, 1)),
      fes_lo(lor_disc.GetFESpace()),
      a_ho(&fes_ho),
      a_lo(&fes_lo),
      x(&mfes),
      dofs(fes_ho.GetVSize()),
      mdof(0.0)
   {
      //dbg("MakeCartesian3D(%d,%d,%d)",nx,ny,nz);
      a_lo.AddDomainIntegrator(new DiffusionIntegrator(&ir_el));
      a_ho.AddDomainIntegrator(new DiffusionIntegrator);
      a_ho.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      SetupRandomMesh();
      tic_toc.Clear();
   }

   bool SanityChecks()
   {
      //if (dofs > 100*1024) { return true; }

      constexpr double EPS = 1e-15;

      OperatorHandle A_lo, A_batched, A_deviced;
      tic();
      a_lo = 0.0; // have to flush these results
      a_lo.Assemble();
      dbg("Standard LOR time = %f",toc());
      a_lo.FormSystemMatrix(ess_dofs, A_lo);
      A_lo.As<SparseMatrix>()->HostReadWriteI();
      A_lo.As<SparseMatrix>()->HostReadWriteJ();
      A_lo.As<SparseMatrix>()->HostReadWriteData();

      Vector x(A_lo->Width()), y(A_lo->Height());
      x.Randomize(SEED);
      y.Randomize(SEED);
      a_lo.Finalize();
      const double dot_lo = A_lo.As<SparseMatrix>()->InnerProduct(x,y);

      tic();
      AssembleBatchedLOR(a_lo, fes_ho, ess_dofs, A_batched);
      A_batched.As<SparseMatrix>()->HostReadWriteI();
      A_batched.As<SparseMatrix>()->HostReadWriteJ();
      A_batched.As<SparseMatrix>()->HostReadWriteData();
      dbg(" Batched LOR time = %f",toc());
      const double dot_batch = A_batched.As<SparseMatrix>()->InnerProduct(x,y);
      assert(almost_equal(dot_lo,dot_batch));
      A_batched.As<SparseMatrix>()->Add(-1.0, *A_lo.As<SparseMatrix>());
      const double max_norm = A_batched.As<SparseMatrix>()->MaxNorm();

      if (max_norm > EPS) { return false; }

      tic();
      AssembleBatchedLOR_GPU(a_lo, fes_ho, ess_dofs, A_deviced);
      dbg(" Batched GPU time = %f",toc());
      A_deviced.As<SparseMatrix>()->HostReadWriteI();
      A_deviced.As<SparseMatrix>()->HostReadWriteJ();
      A_deviced.As<SparseMatrix>()->HostReadWriteData();
      const double dot_device = A_deviced.As<SparseMatrix>()->InnerProduct(x,y);
      A_deviced.As<SparseMatrix>()->Add(-1.0, *A_lo.As<SparseMatrix>());
      const double max_norm_GPU = A_deviced.As<SparseMatrix>()->MaxNorm();
      assert(almost_equal(dot_lo,dot_device));

      return max_norm_GPU < EPS;
   }

   void SetupRandomMesh() noexcept
   {
      mesh.SetNodalFESpace(&mfes);
      mesh.SetNodalGridFunction(&x);
      const double jitter = 1./(M_PI*M_PI);
      const double h0 = mesh.GetElementSize(0);
      GridFunction rdm(&mfes);
      rdm.Randomize(SEED);
      rdm -= 0.5; // Shift to random values in [-0.5,0.5]
      rdm *= jitter * h0; // Scale the random values to be of same order
      x -= rdm;
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

   void Deviced()
   {
      tic_toc.Start();
      OperatorHandle A_batched;
      AssembleBatchedLOR_GPU(a_lo, fes_ho, ess_dofs, A_batched);
      tic_toc.Stop();
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * dofs;
   }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,4,1)

// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(4,64,4)
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

Bench_LOR(SanityChecks)
Bench_LOR(Standard)
Bench_LOR(Batched)
Bench_LOR(Deviced)

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
