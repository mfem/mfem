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
#include "kershaw.hpp"

#ifdef MFEM_USE_BENCHMARK

Mesh CreateKershawMesh(int N, double eps)
{
   Mesh mesh = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON);
   KershawTransformation kt(mesh.Dimension(), eps, eps);
   mesh.Transform(kt);
   return mesh;
}

struct DGMassBenchmark
{
   const int p;
   const int N;
   const int dim = 3;
   Mesh mesh;
   L2_FECollection fec;
   FiniteElementSpace fes;
   const int n;

   BilinearForm m;
   DGMassInverse massinv_lobatto;
   DGMassInverse massinv_legendre;

   Vector B, X;

   OperatorJacobiSmoother jacobi;
   CGSolver cg;

   const int dofs;
   double mdofs;

   DGMassBenchmark(int p_, int N_, double eps_):
      p(p_),
      N(N_),
      // mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      mesh(CreateKershawMesh(N,eps_)),
      fec(p, dim, BasisType::GaussLobatto),
      fes(&mesh, &fec),
      n(fes.GetTrueVSize()),
      m(&fes),
      massinv_lobatto(fes, BasisType::GaussLobatto),
      massinv_legendre(fes, BasisType::GaussLegendre),
      B(n),
      X(n),
      dofs(n),
      mdofs(0.0)
   {
      m.AddDomainIntegrator(new MassIntegrator);
      m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      m.Assemble();

      jacobi.SetOperator(m);

      B.Randomize(1);

      const double tol = 1e-10;

      cg.SetAbsTol(tol);
      cg.SetRelTol(0.0);
      cg.SetMaxIter(100);
      cg.SetOperator(m);
      cg.SetPreconditioner(jacobi);

      massinv_lobatto.SetAbsTol(tol);
      massinv_lobatto.SetRelTol(0.0);

      massinv_legendre.SetAbsTol(tol);
      massinv_legendre.SetRelTol(0.0);

      tic_toc.Clear();
   }

   void FullCG()
   {
      X = 0.0;
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   void LocalCGLobatto()
   {
      X = 0.0;
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      massinv_lobatto.Mult(B, X);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   void LocalCGLegendre()
   {
      X = 0.0;
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      massinv_legendre.Mult(B, X);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   void MassApply()
   {
      X = 0.0;
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      m.Mult(B, X);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   double Mdofs() const { return mdofs / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,5,1)
// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(2,20,1)

#define MAX_NDOFS 2*1024*1024

/// Kernels definitions and registrations
#define Benchmark(Name, prefix, eps)\
static void Name##_##prefix(bm::State &state){\
   const int side = state.range(0);\
   const int p = state.range(1);\
   DGMassBenchmark mb(p, side, eps);\
   if (mb.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { mb.Name(); }\
   state.counters["MDof/s"] = bm::Counter(mb.Mdofs());\
   state.counters["dofs"] = bm::Counter(mb.dofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name##_##prefix)\
            -> ArgsProduct({N_SIDES,P_ORDERS})\
            -> Unit(bm::kMillisecond);

#define MassBenchmark(prefix, eps) \
   Benchmark(FullCG, prefix, eps) \
   Benchmark(LocalCGLobatto, prefix, eps) \
   Benchmark(LocalCGLegendre, prefix, eps) \
   Benchmark(MassApply, prefix, eps)

MassBenchmark(1_0, 1.0)
MassBenchmark(0_5, 0.5)
MassBenchmark(0_3, 0.3)

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
