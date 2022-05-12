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
#include <memory>

#ifdef MFEM_USE_BENCHMARK

static constexpr double tol = 1e-10;

enum class MassSolverType
{
   FULL_CG,
   LOCAL_CG_LOBATTO,
   LOCAL_CG_LEGENDRE,
   DIRECT,
   DIRECT_CUSOLVER,
   DIRECT_CUBLAS
};

Mesh CreateKershawMesh(int N, double eps)
{
   Mesh mesh = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON);
   KershawTransformation kt(mesh.Dimension(), eps, eps);
   mesh.Transform(kt);
   return mesh;
}

struct DGMassInverse_FullCG : Solver
{
   BilinearForm m;
   OperatorJacobiSmoother jacobi;
   CGSolver cg;

   DGMassInverse_FullCG(FiniteElementSpace &fes) : m(&fes)
   {
      m.AddDomainIntegrator(new MassIntegrator);
      m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      m.Assemble();

      jacobi.SetOperator(m);

      cg.SetAbsTol(tol);
      cg.SetRelTol(0.0);
      cg.SetMaxIter(100);
      cg.SetOperator(m);
      cg.SetPreconditioner(jacobi);
   }

   void Mult(const Vector &b, Vector &x) const
   {
      cg.Mult(b, x);
   }

   void Setup()
   {
      m.Update();
      m.Assemble();

      jacobi.SetOperator(m);
   }

   void SetOperator(const Operator &op) { }
};

struct DGMassBenchmark
{
   MassSolverType solver_type;
   const int p;
   const int N;
   const int dim = 3;
   Mesh mesh;
   L2_FECollection fec;
   FiniteElementSpace fes;
   const int n;

   std::unique_ptr<Solver> massinv;

   Vector B, X;

   const int dofs;
   double mdofs;

   DGMassBenchmark(MassSolverType type_, int p_, int N_, double eps_) :
      solver_type(type_),
      p(p_),
      N(N_),
      mesh(CreateKershawMesh(N,eps_)),
      fec(p, dim, BasisType::Positive),
      fes(&mesh, &fec),
      n(fes.GetTrueVSize()),
      B(n),
      X(n),
      dofs(n),
      mdofs(0.0)
   {
      B.Randomize(1);
      tic_toc.Clear();
   }

   void NewFullCG()
   {
      if (massinv)
      {
         static_cast<DGMassInverse_FullCG*>(massinv.get())->Setup();
      }
      else
      {
         massinv.reset(new DGMassInverse_FullCG(fes));
      }
   }

   void NewLocalCG(int btype)
   {
      if (massinv)
      {
         static_cast<DGMassInverse*>(massinv.get())->Setup();
      }
      else
      {
         DGMassInverse *massinv_ = new DGMassInverse(fes, btype);
         massinv_->SetAbsTol(tol);
         massinv_->SetRelTol(0.0);
         massinv.reset(massinv_);
      }
   }

   void NewDirect(BatchSolverMode mode)
   {
      DGMassInverse_Direct *massinv_ = new DGMassInverse_Direct(fes, mode);
      massinv.reset(massinv_);
   }

   void NewSolver()
   {
      switch (solver_type)
      {
         case MassSolverType::FULL_CG:
            NewFullCG();
            break;
         case MassSolverType::LOCAL_CG_LOBATTO:
            NewLocalCG(BasisType::GaussLobatto);
            break;
         case MassSolverType::LOCAL_CG_LEGENDRE:
            NewLocalCG(BasisType::GaussLegendre);
            break;
         case MassSolverType::DIRECT:
            NewDirect(BatchSolverMode::NATIVE);
            break;
         case MassSolverType::DIRECT_CUBLAS:
            NewDirect(BatchSolverMode::CUBLAS);
            break;
         case MassSolverType::DIRECT_CUSOLVER:
            NewDirect(BatchSolverMode::CUSOLVER);
            break;
      }
   }

   void Setup()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      NewSolver();
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   void Solve()
   {
      if (!massinv) { NewSolver(); }
      X = 0.0;
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      massinv->Mult(B, X);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   void SetupAndSolve()
   {
      X = 0.0;
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      NewSolver();
      massinv->Mult(B, X);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   double Mdofs() const { return mdofs / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,5,1)
// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(2,30,2)

#define MAX_NDOFS 2*1024*1024

/// Kernels definitions and registrations
#define Benchmark(solver_type, op_name, suffix, eps)\
static void solver_type##_##op_name##_##suffix(bm::State &state){\
   const int side = state.range(0);\
   const int p = state.range(1);\
   DGMassBenchmark mb(MassSolverType:: solver_type, p, side, eps);\
   if (mb.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { mb.op_name(); }\
   state.counters["MDof/s"] = bm::Counter(mb.Mdofs());\
   state.counters["dofs"] = bm::Counter(mb.dofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(solver_type##_##op_name##_##suffix)\
            -> ArgsProduct({N_SIDES,P_ORDERS})\
            -> Unit(bm::kMillisecond);

#define MassBenchmarks(solver_type, suffix, eps) \
   Benchmark(solver_type, Setup, suffix, eps) \
   Benchmark(solver_type, Solve, suffix, eps) \
   Benchmark(solver_type, SetupAndSolve, suffix, eps)

#define AllMassBenchmarks(suffix, eps) \
   MassBenchmarks(FULL_CG, suffix, eps) \
   MassBenchmarks(LOCAL_CG_LOBATTO, suffix, eps) \
   MassBenchmarks(LOCAL_CG_LEGENDRE, suffix, eps) \
   MassBenchmarks(DIRECT, suffix, eps) \
   MassBenchmarks(DIRECT_CUBLAS, suffix, eps) \
   MassBenchmarks(DIRECT_CUSOLVER, suffix, eps)

AllMassBenchmarks(1_0, 1.0)
AllMassBenchmarks(0_5, 0.5)
AllMassBenchmarks(0_3, 0.3)

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
