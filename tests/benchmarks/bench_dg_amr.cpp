// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

/*
  This benchmark is inspired by the CEED's bake-off problems to benchmark the
  performance of the action of the partial assembly (PA) of DG Convection on
  nonconforming meshes.

   * --benchmark_filter=BK_DG/[1024-max_dofs]/[1-max_order]/[prob]
   * --benchmark_context=device=[cpu/cuda/hip]
*/

// The maximum polynomial order used for benchmarking
const int max_order = 6;
// The maximum number of dofs for benchmarking
const int max_dofs = 1e7;

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   switch (dim)
   {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = x(1); v(1) = -x(0); break;
      case 3: v(0) = x(1); v(1) = -x(0); v(2) = x(0); break;
   }
}

/// A kernel testing DG Convection
struct KernelMesh
{
   const int N;
   Mesh mesh;

   KernelMesh(int N, double prob)
      : N(N), mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON))
   {
      if (prob >= 0.0)
      {
         mesh.EnsureNCMesh();
         if (prob > 0.0)
         {
            mesh.RandomRefinement(prob);
         }
      }
   }
};

struct Kernel: public KernelMesh
{
   const int p, q, dim = 3;
   DG_FECollection fec;
   FiniteElementSpace fes;
   const int dofs;
   GridFunction x,y;
   BilinearForm a;
   double mdofs;
   VectorFunctionCoefficient velocity;

   Kernel(int order, int N, double prob = -1, bool GLL = false)
      :
      KernelMesh(N, prob),
      p(order),
      q(2*p + (GLL?-1:3)),
      fec(p, dim, BasisType::GaussLobatto),
      fes(&mesh, &fec),
      dofs(fes.GetTrueVSize()),
      x(&fes),
      y(&fes),
      a(&fes),
      mdofs(0.0),
      velocity(dim, velocity_function)
   {
      if (is_runnable())
      {
         x.Randomize(1);
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
         a.AddInteriorFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
         a.AddBdrFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
         a.Assemble();
         a.Mult(x, y);
         MFEM_DEVICE_SYNC;
      }
   }

   /// @brief Heuristic to evaluate if the case will run out of memory
   bool is_runnable() const
   {
      const long long int gB = 1073741824/8;
      const int mem_size = Device::IsEnabled()?16:256;
      const long long int max_mem = mem_size * gB;
      const int num_elems = fes.GetNE();
      long long int mem = num_elems * pow(p+1, dim+1) * 8;
      // std::cout << "mem = " << mem << " , max_mem = " << max_mem << std::endl;
      return mem < max_mem;
   }

   void setup()
   {
      a.Assemble();
   }

   void benchmark_setup()
   {
      setup();
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }

   void benchmark_action()
   {
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// BK_DG inspired benchmark for the action
static void BK_DG(bm::State &state)
{
   const int dim = 3;
   const int p = state.range(1);
   const int target_dofs = state.range(0);
   const int elem_dofs = pow(p+1, dim);
   const int N = pow(target_dofs / elem_dofs, 1.0/dim) + 1;
   const double prob = ((double)state.range(2))/100;
   Kernel ker(p, N, prob);
   if ( !ker.is_runnable() ) { state.SkipWithError("MAX_MEM"); }
   while (state.KeepRunning()) { ker.benchmark_action(); }
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);
   state.counters["Dofs"] = bm::Counter(ker.dofs, bm::Counter::kDefaults);
   state.counters["Order"] = bm::Counter(ker.p);
   state.counters["Prob"] = bm::Counter(state.range(2));
}

BENCHMARK(BK_DG)->ArgsProduct(
{
   benchmark::CreateRange(1024, max_dofs, /*step=*/2),
   benchmark::CreateDenseRange(1, max_order, /*step=*/1),
   {-1, 0, 1, 10, 30}
})->Unit(bm::kMillisecond);

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
