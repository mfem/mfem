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


struct LORBench
{
   static constexpr double EPS = 1e-14;
   static constexpr int RANDOM_SEED = 0x100001b3;
   GeometricFactors::FactorFlags DETERMINANTS = GeometricFactors::DETERMINANTS;
   const int p, c, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes_mesh, fes_ho;
   Array<int> ess_tdofs_ho;
   LORDiscretization lor;
   IntegrationRules irs;
   const IntegrationRule *ir;
   FiniteElementSpace &fes_lor;
   const GeometricFactors *gf_ho, *gf_lor;
   ConstantCoefficient diff_coeff, mass_coeff;
   BilinearForm a_ho, a_lor_legacy, a_lor_full;
   OperatorHandle A_lor_legacy, A_lor_full;
   GridFunction mesh_coords;
   const int nvdofs;
   double mdof;
   Vector x, y;

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
      fes_mesh(&mesh, &fec, dim),
      fes_ho(&mesh, &fec),
      lor(fes_ho, BasisType::GaussLobatto),
      irs(0, Quadrature1D::GaussLobatto),
      ir(&irs.Get(mesh.GetElementGeometry(0), 1)),
      fes_lor(lor.GetFESpace()),
      gf_ho(mesh.GetGeometricFactors(*ir, DETERMINANTS)),
      gf_lor(lor.GetFESpace().GetMesh()->GetGeometricFactors(*ir, DETERMINANTS)),
      diff_coeff(M_PI),
      mass_coeff(1.0/M_PI),
      a_ho(&fes_ho),
      a_lor_legacy(&fes_lor),
      a_lor_full(&fes_lor),
      A_lor_legacy(),
      A_lor_full(),
      mesh_coords(&fes_mesh),
      nvdofs(fes_ho.GetVSize()),
      mdof(0.0),
      x(nvdofs),
      y(nvdofs)
   {
      dbg("p:%d side:%d nvdofs:%d/%d", p, side, nvdofs, fes_lor.GetVSize());
      a_ho.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      a_ho.AddDomainIntegrator(new MassIntegrator(mass_coeff));
      fes_ho.GetBoundaryTrueDofs(ess_tdofs_ho);

      a_lor_legacy.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      a_lor_legacy.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      a_lor_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);

      a_lor_full.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      a_lor_full.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      a_lor_full.SetAssemblyLevel(AssemblyLevel::FULL);

      SetupRandomMesh();

      MFEM_VERIFY(gf_ho->detJ.Min() > 0.0, "Invalid HO mesh!");
      MFEM_VERIFY(gf_lor->detJ.Min() > 0.0, "Invalid LOR mesh!");

      x.Randomize(RANDOM_SEED);
      y.Randomize(RANDOM_SEED);

      tic_toc.Clear();
   }

   void SetupRandomMesh() noexcept
   {
      mesh.SetNodalFESpace(&fes_mesh);
      mesh.SetNodalGridFunction(&mesh_coords);
      const double jitter = 0.000001;//1./(M_PI*M_PI);
      const double h0 = mesh.GetElementSize(0);
      GridFunction rdm(&fes_mesh);
      rdm.Randomize(RANDOM_SEED);
      rdm -= 0.5; // Shift to random values in [-0.5,0.5]
      rdm *= jitter * h0; // Scale the random values to be of same order
      mesh_coords -= rdm;
   }

   void SanityChecks()
   {
      dbg();
      a_lor_legacy = 0.0;
      MFEM_DEVICE_SYNC;
      tic();
      a_lor_legacy.Assemble();
      a_lor_legacy.EliminateVDofs(ess_tdofs_ho, Operator::DIAG_KEEP);
      MFEM_DEVICE_SYNC;
      dbg(" Legacy time = %f", toc());
      SparseMatrix &A_lor_legacy_sp = a_lor_legacy.SpMat();
      A_lor_legacy_sp.HostReadWriteI();
      A_lor_legacy_sp.HostReadWriteJ();
      A_lor_legacy_sp.HostReadWriteData();
      const double dot_legacy = A_lor_legacy_sp.InnerProduct(x,y);
      dbg("dot_legacy:%.15e", dot_legacy);

      MFEM_DEVICE_SYNC;
      tic();
      a_lor_full.Assemble();
      a_lor_full.EliminateVDofs(ess_tdofs_ho, Operator::DIAG_KEEP);
      MFEM_DEVICE_SYNC;
      dbg("   Full time = %f", toc());
      SparseMatrix &A_lor_full_sp = a_lor_full.SpMat();
      A_lor_full_sp.HostReadWriteI();
      A_lor_full_sp.HostReadWriteJ();
      A_lor_full_sp.HostReadWriteData();
      const double dot_full = A_lor_full_sp.InnerProduct(x,y);
      dbg("dot_full:%.15e",dot_full);
      MFEM_VERIFY(almost_equal(dot_legacy, dot_full), "dot_full error!");
      A_lor_full_sp.Add(-1.0, A_lor_legacy_sp);
      const double max_norm_full = a_lor_full.SpMat().MaxNorm();
      MFEM_VERIFY(max_norm_full < EPS, "max_norm_full error!");

      MFEM_DEVICE_SYNC;
      tic();
      lor.AssembleSystem(a_ho, ess_tdofs_ho);
      MFEM_DEVICE_SYNC;
      dbg("Batched time = %f", toc());
      SparseMatrix &A_lor_batched_sp = lor.GetAssembledMatrix();
      A_lor_batched_sp.HostReadWriteI();
      A_lor_batched_sp.HostReadWriteJ();
      A_lor_batched_sp.HostReadWriteData();
      const double dot_batched = A_lor_batched_sp.InnerProduct(x,y);
      dbg("dot_batched:%.15e",dot_batched);
      MFEM_VERIFY(almost_equal(dot_legacy, dot_batched), "dot_batched error!");
      A_lor_batched_sp.Add(-1.0, A_lor_legacy_sp);
      const double max_norm_batched = A_lor_batched_sp.MaxNorm();
      dbg("max_norm_batched:%.15e",max_norm_batched);
      MFEM_VERIFY(max_norm_batched < EPS, "max_norm_batched");
   }

   void GLVis(Mesh &gl_mesh, GridFunction *gl_x = nullptr)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      if (!gl_x) { sol_sock << "mesh\n" << gl_mesh; }
      else { sol_sock << "solution\n" << gl_mesh << *gl_x ;}
      sol_sock << std::flush;
   }

   void Test()
   {
      MFEM_DEVICE_SYNC;
      tic();
      lor.AssembleSystem(a_ho, ess_tdofs_ho);
      MFEM_DEVICE_SYNC;
      dbg(" Deviced time = %f",toc());
      dbg("Exiting!");
      std::exit(0);
   }

   void Dump()
   {
      a_lor_legacy.Assemble();
      a_lor_legacy.FormSystemMatrix(ess_tdofs_ho, A_lor_legacy);
      a_lor_legacy.Finalize();
      SparseMatrix &A_lor_legacy_sp = *A_lor_legacy.As<SparseMatrix>();
      A_lor_legacy_sp.HostReadWriteI();
      A_lor_legacy_sp.HostReadWriteJ();
      A_lor_legacy_sp.HostReadWriteData();

      dbg("Saving 'A.mtx' file");
      {
         std::ofstream mtx_file("A.mtx");
         A_lor_legacy_sp.PrintMM(mtx_file);
      }

      dbg("fes_lo.GetVSize: %d", fes_lor.GetVSize());
      Mesh *mesh_lor =fes_lor.GetMesh();
      GridFunction ids_lor(&fes_lor);
      Array<int> vdofs_lor(fes_lor.GetVSize());
      fes_lor.GetVDofs(0, vdofs_lor);
      assert(fes_lor.GetVDim()==1);
      for (int i=0; i<fes_lor.GetVSize(); i++) { ids_lor(i) = vdofs_lor[i]; }
      GLVis(*mesh_lor, &ids_lor);
      dbg("LO GLVis done!");

      dbg("fes_ho.GetVSize: %d", fes_ho.GetVSize());
      GridFunction ids(&fes_ho);
      Array<int> vdofs_ho(fes_ho.GetVSize());
      fes_ho.GetVDofs(0, vdofs_ho);
      assert(fes_ho.GetVDim()==1);
      for (int i=0; i<fes_ho.GetVSize(); i++) { ids(i) = vdofs_ho[i]; }
      GLVis(mesh, &ids);
      dbg("HO GLVis done!");

      dbg("Exiting!");
      std::exit(0);
   }

   void KerLegacy()
   {
      dbg();
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      a_lor_legacy.Assemble();
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   void KerFull()
   {
      dbg();
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      a_lor_full.Assemble();
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   void KerBatched()
   {
      dbg();
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      lor.AssembleSystem(a_ho, ess_tdofs_ho);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   void AllFull()
   {
      dbg();
      BilinearForm bf_full(&fes_lor);
      bf_full.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
      bf_full.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
      bf_full.SetAssemblyLevel(AssemblyLevel::FULL);
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      bf_full.Assemble();
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   void AllBatched()
   {
      dbg();
      BilinearForm bf_batched(&fes_ho);
      bf_batched.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      bf_batched.AddDomainIntegrator(new MassIntegrator(mass_coeff));
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      lor.AssembleSystem(bf_batched, ess_tdofs_ho);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * nvdofs;
   }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)

// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(4,20,1)
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
   state.counters["Ker_(Dofs/s)"] = bm::Counter(lor.nvdofs, flags);\
   state.counters["All_(MDof/s)"] = bm::Counter(lor.Mdofs());\
   state.counters["dofs"] = bm::Counter(lor.nvdofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name)\
            -> ArgsProduct({P_ORDERS,N_SIDES})\
            -> Unit(bm::kMillisecond)\
            ->Iterations(4);

Benchmark(SanityChecks)

Benchmark(KerLegacy)
Benchmark(KerFull)
Benchmark(KerBatched)

Benchmark(AllFull)
Benchmark(AllBatched)

Benchmark(Dump)
Benchmark(Test)

/**
 * @brief main entry point
 * --benchmark_filter=SanityChecks/4/16
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
