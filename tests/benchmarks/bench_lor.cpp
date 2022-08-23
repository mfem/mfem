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
#include <cassert>

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/lor/lor.hpp"

#define MFEM_DEBUG_COLOR 119
#include "general/debug.hpp"

#define MFEM_NVTX_COLOR Lime
#include "general/nvtx.hpp"

struct LORBench
{
   const Nvtx LORBench_tx = {"LORBench"};
   const double EPS = 1e-14;
   const int RANDOM_SEED = 0x100001b3;
   GeometricFactors::FactorFlags DETERMINANTS = GeometricFactors::DETERMINANTS;
   const int p, c, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   IntegrationRules irs;
   const IntegrationRule *ir;
   const Nvtx mesh_tx = {"mesh"};
   Mesh mesh;
   H1_FECollection fec;
   const Nvtx fes_mesh_tx = {"fes_mesh"};
   FiniteElementSpace fes_mesh;
   const Nvtx mesh_coords_tx = {"mesh_coords"};
   GridFunction mesh_coords;
   const bool postfix_mesh_coords;
   FiniteElementSpace fes_ho;
   Array<int> ess_tdofs, ess_tdofs_empty;
   const bool prefix_lor_setup;
   const Nvtx a_tx = {"a"};
   BilinearForm a;
   ConstantCoefficient diff_coeff, mass_coeff;
   const bool a_ho_setup;
   const Nvtx lor_tx = {"lor"};
   LORDiscretization lor;
   const Nvtx fes_lor_tx = {"fes_lor"};
   FiniteElementSpace &fes_lor;
   const Nvtx GeometricFactors_tx = {"GeometricFactors"};
   const GeometricFactors *gf_ho, *gf_lor;
   const Nvtx a_lor_legacy_full_tx = {"a_lor_legacy_full"};
   BilinearForm a_lor_legacy, a_lor_full;
   OperatorHandle A_lor_legacy, A_lor_full;
   const int nvdofs;
   double mdof;
   Vector x, y;
   const Nvtx Ready_tx = {"Ready"};

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
      irs(0, Quadrature1D::GaussLobatto),
      ir(&irs.Get(Geometry::CUBE, 1)),
      mesh(Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON)),
      fec(p, dim),
      fes_mesh(&mesh, &fec, dim),
      mesh_coords(&fes_mesh),
      postfix_mesh_coords((SetupRandomMesh(), true)),
      fes_ho(&mesh, &fec),
      prefix_lor_setup((fes_ho.GetBoundaryTrueDofs(ess_tdofs), true)),
      a(&fes_ho),
      diff_coeff(M_PI),
      mass_coeff(1.0/M_PI),
      a_ho_setup((a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff)),
                  a.AddDomainIntegrator(new MassIntegrator(mass_coeff)),
                  true)),
      lor(fes_ho),
      fes_lor(lor.GetFESpace()),
      gf_ho(fes_mesh.GetMesh()->GetGeometricFactors(*ir, DETERMINANTS)),
      gf_lor(fes_lor.GetMesh()->GetGeometricFactors(*ir, DETERMINANTS)),
      a_lor_legacy(&fes_lor),
      a_lor_full(&fes_lor),
      A_lor_legacy(),
      A_lor_full(),
      nvdofs(fes_ho.GetVSize()),
      mdof(0.0),
      x(nvdofs),
      y(nvdofs)
   {
      a_lor_legacy.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff,ir));
      a_lor_legacy.AddDomainIntegrator(new MassIntegrator(mass_coeff,ir));
      a_lor_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);

      a_lor_full.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff,ir));
      a_lor_full.AddDomainIntegrator(new MassIntegrator(mass_coeff,ir));
      a_lor_full.SetAssemblyLevel(AssemblyLevel::FULL);

      MFEM_VERIFY(gf_ho->detJ.Min() > 0.0, "Invalid HO mesh!");
      MFEM_VERIFY(gf_lor->detJ.Min() > 0.0, "Invalid LOR mesh!");

      {
         NVTX("Randomize");
         x.Randomize(RANDOM_SEED);
         y.Randomize(RANDOM_SEED);
      }
   }

   void SetupRandomMesh() noexcept
   {
      NVTX("SetupRandomMesh");
      mesh.SetNodalFESpace(&fes_mesh);
      mesh.SetNodalGridFunction(&mesh_coords);
      const double jitter = 1e-4/(M_PI*M_PI);
      const double h0 = mesh.GetElementSize(0);
      GridFunction rdm(&fes_mesh);
      rdm.Randomize(RANDOM_SEED);
      rdm -= 0.5; // Shift to random values in [-0.5,0.5]
      rdm *= jitter * h0; // Scale the random values to be of same order
      mesh_coords -= rdm;
      assert(mesh.GetGeometricFactors(*ir, DETERMINANTS)->detJ.Min() > 0.0);
   }

   // void SanityChecks()
   // {
   //    NVTX("SanityChecks");
   //    a_lor_legacy = 0.0;
   //    a_lor_legacy.Assemble();
   //    a_lor_legacy.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
   //    SparseMatrix &A_lor_legacy_sp = a_lor_legacy.SpMat();
   //    const double dot_legacy = A_lor_legacy_sp.InnerProduct(x,y);

   //    lor.LegacyAssembleSystem(a, ess_tdofs);
   //    SparseMatrix A_lor_assembled_sp = lor.GetAssembledMatrix();
   //    const double dot_lor_legacy = A_lor_assembled_sp.InnerProduct(x,y);
   //    MFEM_VERIFY(almost_equal(dot_legacy, dot_lor_legacy), "dot_lor_legacy error!");

   //    // full does +=, a_lor_full = 0.0;
   //    a_lor_full.Assemble();
   //    SparseMatrix &A_lor_full_sp = a_lor_full.SpMat();
   //    A_lor_full_sp.HostReadI();
   //    A_lor_full_sp.HostReadJ();
   //    A_lor_full_sp.HostReadData();
   //    a_lor_full.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
   //    const double dot_full = A_lor_full_sp.InnerProduct(x,y);
   //    MFEM_VERIFY(almost_equal(dot_legacy, dot_full), "dot_full error!");
   //    A_lor_full_sp.Add(-1.0, A_lor_legacy_sp);
   //    const double max_norm_full = a_lor_full.SpMat().MaxNorm();
   //    MFEM_VERIFY(max_norm_full < EPS, "max_norm_full error!");

   //    lor.AssembleSystem(a, ess_tdofs);
   //    SparseMatrix &A_lor_batched_sp = lor.GetAssembledMatrix();
   //    A_lor_batched_sp.HostReadI();
   //    A_lor_batched_sp.HostReadJ();
   //    A_lor_batched_sp.HostReadData();
   //    const double dot_batched = A_lor_batched_sp.InnerProduct(x,y);
   //    MFEM_VERIFY(almost_equal(dot_legacy, dot_batched),
   //                "dot_batched error: " << dot_legacy << ", " << dot_batched);
   //    A_lor_batched_sp.Add(-1.0, A_lor_legacy_sp);
   //    const double max_norm_batched = A_lor_batched_sp.MaxNorm();
   //    MFEM_VERIFY(max_norm_batched < EPS, "max_norm_batched");
   // }

   void KerLegacy()
   {
      NVTX("KerLegacy");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * nvdofs;
      a_lor_legacy.Assemble();
      // ess_tdofs_empty or:
      // a_lor_legacy.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
   }

   void KerFull()
   {
      NVTX("KerFull");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * nvdofs;
      a_lor_full.Assemble();
      // ess_tdofs_empty or:
      // a_lor_full.EliminateVDofs(ess_tdofs, Operator::DIAG_KEEP);
   }

   void KerBatched()
   {
      NVTX("KerBatched");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * nvdofs;
      lor.AssembleSystem(a, ess_tdofs_empty);
   }

   void AllLegacy()
   {
      NVTX("AllLegacy");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * nvdofs;
      LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
      FiniteElementSpace &fes_lo = lor_disc.GetFESpace();
      BilinearForm bf_legacy(&fes_lo);
      bf_legacy.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      bf_legacy.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      bf_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);
      bf_legacy.Assemble();
      // no EliminateVDofs
   }

   void AllFull()
   {
      NVTX("AllFull");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * nvdofs;
      LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
      FiniteElementSpace &fes_lo = lor_disc.GetFESpace();
      BilinearForm bf_full(&fes_lo);
      bf_full.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      bf_full.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      bf_full.SetAssemblyLevel(AssemblyLevel::FULL);
      bf_full.Assemble();
      // no EliminateVDofs
   }

   void AllBatched()
   {
      NVTX("AllBatched");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * nvdofs;
      LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
      BilinearForm bf_ho(&fes_ho);
      bf_ho.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      bf_ho.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      lor_disc.AssembleSystem(bf_ho, ess_tdofs_empty);
      // working with no BC
   }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)

// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(10,120,10)
#define MAX_NDOFS 2*1024*1024

/// Kernels definitions and registrations
#define Benchmark(Name)\
static void Name(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   LORBench lor(p, side);\
   if (lor.nvdofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { lor.Name(); }\
   bm::Counter::Flags flags = bm::Counter::kIsIterationInvariantRate;\
   state.counters["MDof/s"] = bm::Counter(1e-6*lor.nvdofs, flags);\
   state.counters["dofs"] = bm::Counter(lor.nvdofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name)\
            -> ArgsProduct({P_ORDERS, N_SIDES})\
            -> Unit(bm::kMillisecond)\
            -> Iterations(10);

// Benchmark(SanityChecks)

Benchmark(KerLegacy)
Benchmark(KerFull)
Benchmark(KerBatched)

//Benchmark(AllLegacy)
Benchmark(AllFull)
Benchmark(AllBatched)

/**
 * @brief main entry point
 * --benchmark_filter=SanityChecks/3/16
 * --benchmark_filter=\(Batched\|Deviced\|Full\)/4/16
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
