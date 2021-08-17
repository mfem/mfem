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

#include <memory>
#include <cassert>
#include <functional>

using Kernel = std::function<BilinearFormIntegrator*(Coefficient&)>;

struct PA_3D_Kernels
{
   const int N, order;
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

   PA_3D_Kernels(int order, Kernel kernel):
      N(Device::IsEnabled()?16:8),
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
      a.AddDomainIntegrator(kernel(one));
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetRelTol(rtol);
      cg.SetOperator(*A);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      cg.iterative_mode = false;

      tic_toc.Clear();
   }

   void benchmark()
   {
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
  Kernels
*/
#define BENCHMARK_KERNEL(K)\
BilinearFormIntegrator *_##K##_(Coefficient &c) { return new K##Integrator(c); }\
static void K(bm::State &state){\
   const int order = state.range(0);\
   PA_3D_Kernels ker(order, _##K##_);\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker.Mdofs());}\
BENCHMARK(K)->DenseRange(1,6);

/**
  Launch all benchmarks: Mass & Diffusion
  */
BENCHMARK_KERNEL(Mass)
BENCHMARK_KERNEL(Diffusion)

/**
 * @brief main entry point
 * --benchmark_filter=Mass/6
 * --benchmark_filter=Diffusion/6
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
