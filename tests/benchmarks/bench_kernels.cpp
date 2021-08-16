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

#include <cassert>
#include <memory>
#include <iomanip>

#ifdef MFEM_USE_BENCHMARK

struct PA_3D_Kernels
{
   const int N, problem, order;
   const int dim = 3;
   const double rtol = 1e-12;
   const int max_it = 32;
   const int print_lvl = -1;

   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const int dofs;
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   ConstantCoefficient one;
   LinearForm b;
   GridFunction x;
   BilinearForm a;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;
   double mdof;

   PA_3D_Kernels(int problem, int order):
      N(Device::IsEnabled()?16:8),
      problem(problem),
      order(order),
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      fec(order, dim),
      fes(&mesh, &fec),
      dofs(fes.GetVSize()),
      ess_bdr(mesh.bdr_attributes.Max()),
      one((ess_bdr=1,fes.GetEssentialTrueDofs(ess_bdr,ess_tdof_list), 1.0)),
      b(&fes),
      x(&fes),
      a(&fes),
      mdof(0.0)
   {
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      if (problem == 0) { a.AddDomainIntegrator(new MassIntegrator(one)); }
      if (problem == 1) { a.AddDomainIntegrator(new DiffusionIntegrator(one)); }
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetRelTol(rtol);
      cg.SetOperator(*A);
      {
         Vector Y(X);
         cg.SetMaxIter(2);
         cg.SetPrintLevel(-1);
         cg.Mult(B,Y);
         MFEM_DEVICE_SYNC;
      }
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      tic_toc.Clear();
   }

   // benchmark this problem
   void benchmark()
   {
      X = 0.0;
      tic_toc.Start();
      cg.Mult(B,X);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += (1e-6 * dofs) * cg.GetNumIterations();
   }

   double Mdof() const { return mdof; }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

/**
 * @brief The Kernel::benchmark::Fixture struct
 */
struct Kernel: public ::benchmark::Fixture
{
   std::unique_ptr<PA_3D_Kernels> ker;
   ~Kernel() { assert(ker == nullptr); }

   using ::benchmark::Fixture::SetUp;
   void SetUp(const ::benchmark::State& state) BENCHMARK_OVERRIDE
   {
      if (state.thread_index == 0)
      {
         assert(!ker.get());
         const int problem = state.range(0);
         const int order = state.range(1);
         ker.reset(new PA_3D_Kernels(problem, order));
      }
      tic_toc.Clear();
   }

   using ::benchmark::Fixture::TearDown;
   void TearDown(const ::benchmark::State &state) BENCHMARK_OVERRIDE
   {
      if (state.thread_index == 0)
      {
         assert(ker.get());
         ker.reset();
      }
   }
};

/**
  Fixture kernels
*/
#define ORDERS {1,2,3,4,5,6}

#define BENCHMARK_KERNEL_F(Name,Problem)\
BENCHMARK_DEFINE_F(Kernel, Name)(benchmark::State &state){\
   assert(ker.get());\
   while (state.KeepRunning()) { ker->benchmark(); }\
   state.counters["MDof"] = bm::Counter(ker->Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker->Mdofs());}\
BENCHMARK_REGISTER_F(Kernel, Name)->ArgsProduct({{Problem},ORDERS});


/**
  Kernels w/o fixture
*/
#define BENCHMARK_KERNEL(Name,Problem)\
static void Name(benchmark::State &state){\
   const int problem = Problem;\
   const int order = state.range(0);\
   PA_3D_Kernels ker(problem, order);\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker.Mdofs());}\
BENCHMARK(Name)->DenseRange(1,6);

/**
  Launch all benchmarks w/ and w/o fixtures: mass & diffusion
  */
BENCHMARK_KERNEL_F(Mass,0)
BENCHMARK_KERNEL(Kernel_Mass,0)

BENCHMARK_KERNEL_F(Diffusion,1)
BENCHMARK_KERNEL(Kernel_Diffusion,1)


/**
 * @brief main entry point
 * --benchmark_filter=Kernel/Mass/0/6
 * --benchmark_filter=Kernel_Mass/6
 * --benchmark_context=device=cpu
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
