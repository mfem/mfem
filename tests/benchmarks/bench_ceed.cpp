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

/*
 This benchmark contains the implementation of the CEED's bake-off problems:
 high-order kernels/benchmarks designed to test and compare the performance
 of high-order codes.
 See: ceed.exascaleproject.org/bps and github.com/CEED/benchmarks
*/

/// Bake-off Problems (BPs)
template<typename BFI, int VDIM = 1, bool P_EQ_Q = false>
struct BakeOffProblem
{
   const int N, p, q, dim = 3;
   const double rtol = 1e-12;
   const int max_it = 32;
   const int print_lvl = -1;

   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const IntegrationRule *ir;
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
   double mdofs;

   BakeOffProblem(int order):
      N(Device::IsEnabled()?32:8),
      p(order), q(2*p + (P_EQ_Q ? 0 : 2)),
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      fec(order, dim),
      fes(&mesh, &fec, VDIM),
      ir(&IntRules.Get(fes.GetFE(0)->GetGeomType(), q)),
      dofs(fes.GetTrueVSize()),
      ess_bdr(mesh.bdr_attributes.Max()),
      one((ess_bdr=1,fes.GetEssentialTrueDofs(ess_bdr,ess_tdof_list), 1.0)),
      b(&fes),
      x(&fes),
      a(&fes),
      mdofs(0.0)
   {
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, ir));
      a.Assemble();

      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetRelTol(rtol);
      cg.SetOperator(*A);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      cg.iterative_mode = false;
      MFEM_DEVICE_SYNC;
   }

   void benchmark()
   {
      cg.Mult(B,X);
      MFEM_DEVICE_SYNC;
      mdofs += MDofs() * cg.GetNumIterations();
   }

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Generic CEED BPi
#define BakeOff_Problem(i,Kernel,VDIM,p_eq_q)\
static void BP##i(bm::State &state){\
   BakeOffProblem<Kernel##Integrator,VDIM,p_eq_q> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BP##i)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// BP1: scalar PCG with mass matrix, q=p+2
BakeOff_Problem(1,Mass,1,false)

/// BP2: vector PCG with mass matrix, q=p+2
BakeOff_Problem(2,VectorMass,3,false)

/// BP3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Problem(3,Diffusion,1,false)

/// BP4: vector PCG with stiffness matrix, q=p+2
BakeOff_Problem(4,VectorDiffusion,3,false)

/// BP5: scalar PCG with stiffness matrix, q=p+1
BakeOff_Problem(5,Diffusion,1,true)

/// BP6: vector PCG with stiffness matrix, q=p+1
BakeOff_Problem(6,VectorDiffusion,3,true)


/// Bake-off Kernels (BKs)
template <typename BFI, int VDIM = 1, bool P_EQ_Q = false>
struct BakeOffKernel
{
   const int N, p, q, dim = 3;

   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   const int dofs;
   GridFunction x, y;
   BilinearForm a;
   double mdofs;

   BakeOffKernel(int order):
      N(Device::IsEnabled()?32:8),
      p(order), q(2*p + (P_EQ_Q ? 0 : 2)),
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      fec(order, dim),
      fes(&mesh, &fec, VDIM),
      ir(&IntRules.Get(fes.GetFE(0)->GetGeomType(), q)),
      one(1.0),
      dofs(fes.GetTrueVSize()),
      x(&fes),
      y(&fes),
      a(&fes),
      mdofs(0.0)
   {
      x.Randomize(1);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, ir));
      a.Assemble();
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
   }

   void benchmark()
   {
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Generic CEED BKi
#define BakeOff_Kernel(i,Kernel,VDIM,p_eq_q)\
static void BK##i(bm::State &state){\
   BakeOffKernel<Kernel##Integrator,VDIM,p_eq_q> ker(state.range(0));\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BK##i)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// BK1: scalar E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(1,Mass,1,false)

/// BK2: vector E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(2,VectorMass,3,false)

/// BK3: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(3,Diffusion,1,false)

/// BK4: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(4,VectorDiffusion,3,false)

/// BK5: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(5,Diffusion,1,true)

/// BK6: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(6,VectorDiffusion,3,true)

/**
 * @brief main entry point
 * --benchmark_filter=BK1/6
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
