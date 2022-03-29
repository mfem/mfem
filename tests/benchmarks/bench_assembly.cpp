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

#ifdef MFEM_USE_BENCHMARK

/*
  This benchmark contains the implementation of the CEED's bake-off problems:
  high-order kernels/benchmarks designed to test and compare the performance
  of high-order codes.

  See: ceed.exascaleproject.org/bps and github.com/CEED/benchmarks
*/

struct BakeOff
{
   const AssemblyLevel assembly;
   const int N, p, q, dim = 3;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   IntegrationRules IntRulesGLL;
   const IntegrationRule *irGLL;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   const int dofs;
   GridFunction x,y;
   BilinearForm a;
   double mdofs;

   BakeOff(AssemblyLevel assembly, int p, int N, int vdim, bool GLL):
      assembly(assembly),
      N(N),
      p(p),
      q(2*p + (GLL?-1:3)),
      mesh(Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON)),
      fec(p, dim, BasisType::GaussLobatto),
      fes(&mesh, &fec, vdim),
      geom_type(fes.GetFE(0)->GetGeomType()),
      IntRulesGLL(0, Quadrature1D::GaussLobatto),
      irGLL(&IntRulesGLL.Get(geom_type, q)),
      ir(&IntRules.Get(geom_type, q)),
      one(1.0),
      dofs(fes.GetTrueVSize()),
      x(&fes),
      y(&fes),
      a(&fes),
      mdofs(0.0) {}

   bool is_runnable() const
   {
      const long long int gB = 1073741824/8;
      const int mem_size = Device::IsEnabled()?16:256;
      const long long int max_mem = mem_size * gB;
      const int num_elems = fes.GetNE();
      long long int mem = num_elems * pow(p+1, dim+1) * 8;
      if (assembly == AssemblyLevel::ELEMENT)
      {
         mem += num_elems * pow(p+1, 2*dim) * 8;
      }
      if (assembly == AssemblyLevel::FULL)
      {
         mem += 3 * num_elems * pow(p+1, 2*dim) * 8;
      }
      // std::cout << "mem = " << mem << " , max_mem = " << max_mem << std::endl;
      return mem < max_mem;
   }

   virtual void setup() = 0;

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Bake-off Problems (BPs)
template<typename BFI, int VDIM = 1, bool GLL = false>
struct Problem: public BakeOff
{
   const double rtol = 1e-12;
   const int max_it = 32;
   const int print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   LinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;

   Problem(AssemblyLevel assembly, int order, int N):
      BakeOff(assembly,order,N,VDIM,GLL),
      ess_bdr(mesh.bdr_attributes.Max()),
      b(&fes)
   {
      if (is_runnable())
      {
         ess_bdr = 1;
         fes.GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
         b.AddDomainIntegrator(new DomainLFIntegrator(one));
         b.Assemble();

         a.SetAssemblyLevel(assembly);
         a.AddDomainIntegrator(new BFI(one, GLL?irGLL:ir));
         a.Assemble();
         a.Mult(x, y);

         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

         cg.SetRelTol(rtol);
         cg.SetOperator(*A);
         cg.SetMaxIter(max_it);
         cg.SetPrintLevel(print_lvl);
         cg.iterative_mode = false;
         MFEM_DEVICE_SYNC;
      }
   }

   void setup() override
   {
      a.Assemble();
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }

   void benchmark() override
   {
      cg.Mult(B,X);
      MFEM_DEVICE_SYNC;
      mdofs += MDofs() * cg.GetNumIterations();
   }
};

/// Bake-off Problems (BPs)
#define BakeOff_Problem(assembly,i,Kernel,VDIM,MaxDofs,p_eq_q)\
static void SetupBP##i##assembly(bm::State &state){\
   const int dim = 3;\
   const int p = state.range(0);\
   const int target_dofs = state.range(1);\
   const int elem_dofs = pow(p+1, dim);\
   const int N = pow(target_dofs / elem_dofs, 1.0/dim) + 1;\
   Problem<Kernel##Integrator,VDIM,p_eq_q> ker(AssemblyLevel::assembly, p, N);\
   if ( !ker.is_runnable() ) { state.SkipWithError("MAX_MEM"); }\
   while (state.KeepRunning()) { ker.setup(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);\
   state.counters["Dofs"] = bm::Counter(ker.dofs, bm::Counter::kDefaults);\
   state.counters["Order"] = bm::Counter(ker.p);}\
BENCHMARK(SetupBP##i##assembly)->ArgsProduct({\
      benchmark::CreateDenseRange(1, 6, /*step=*/1),\
      benchmark::CreateRange(1024, MaxDofs, /*step=*/2)\
    })->Unit(bm::kMillisecond);\
static void BP##i##assembly(bm::State &state){\
   const int dim = 3;\
   const int p = state.range(0);\
   const int target_dofs = state.range(1);\
   const int elem_dofs = pow(p+1, dim);\
   const int N = pow(target_dofs / elem_dofs, 1.0/dim) + 1;\
   Problem<Kernel##Integrator,VDIM,p_eq_q> ker(AssemblyLevel::assembly, p, N);\
   if ( !ker.is_runnable() ) { state.SkipWithError("MAX_MEM"); }\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);\
   state.counters["Dofs"] = bm::Counter(ker.dofs, bm::Counter::kDefaults);\
   state.counters["Order"] = bm::Counter(ker.p);}\
BENCHMARK(BP##i##assembly)->ArgsProduct({\
      benchmark::CreateDenseRange(1, 6, /*step=*/1),\
      benchmark::CreateRange(1024, MaxDofs, /*step=*/2)\
    })->Unit(bm::kMillisecond);

const int maxDofs = 1e7;
// Device::IsEnabled()?32:8
// const int maxDofs = 1e6;
// auto pa_N = [](int p, int MaxSize)
// {
//    int dofs_per_elem = (p+1)*(p+1)*(p+1);
//    return MaxSize/dofs_per_elem;
// };

// PARTIAL:
/// BP1: scalar PCG with mass matrix, q=p+2
BakeOff_Problem(PARTIAL,1,Mass,1,maxDofs,false)

/// BP2: vector PCG with mass matrix, q=p+2
BakeOff_Problem(PARTIAL,2,VectorMass,3,maxDofs,false)

/// BP3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Problem(PARTIAL,3,Diffusion,1,maxDofs,false)

/// BP4: vector PCG with stiffness matrix, q=p+2
BakeOff_Problem(PARTIAL,4,VectorDiffusion,3,maxDofs,false)

/// BP5: scalar PCG with stiffness matrix, q=p+1
BakeOff_Problem(PARTIAL,5,Diffusion,1,maxDofs,true)

/// BP6: vector PCG with stiffness matrix, q=p+1
BakeOff_Problem(PARTIAL,6,VectorDiffusion,3,maxDofs,true)

// ELEMENT:
/// BP1: scalar PCG with mass matrix, q=p+2
BakeOff_Problem(ELEMENT,1,Mass,1,maxDofs,false)

/// BP2: vector PCG with mass matrix, q=p+2
// BakeOff_Problem(ELEMENT,2,VectorMass,3,maxDofs,false)

/// BP3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Problem(ELEMENT,3,Diffusion,1,maxDofs,false)

/// BP4: vector PCG with stiffness matrix, q=p+2
// BakeOff_Problem(ELEMENT,4,VectorDiffusion,3,maxDofs,false)

/// BP5: scalar PCG with stiffness matrix, q=p+1
BakeOff_Problem(ELEMENT,5,Diffusion,1,maxDofs,true)

/// BP6: vector PCG with stiffness matrix, q=p+1
// BakeOff_Problem(ELEMENT,6,VectorDiffusion,3,maxDofs,true)

// FULL:
/// BP1: scalar PCG with mass matrix, q=p+2
BakeOff_Problem(FULL,1,Mass,1,maxDofs,false)

/// BP2: vector PCG with mass matrix, q=p+2
// BakeOff_Problem(FULL,2,VectorMass,3,maxDofs,false)

/// BP3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Problem(FULL,3,Diffusion,1,maxDofs,false)

/// BP4: vector PCG with stiffness matrix, q=p+2
// BakeOff_Problem(FULL,4,VectorDiffusion,3,maxDofs,false)

/// BP5: scalar PCG with stiffness matrix, q=p+1
BakeOff_Problem(FULL,5,Diffusion,1,maxDofs,true)

/// BP6: vector PCG with stiffness matrix, q=p+1
// BakeOff_Problem(FULL,6,VectorDiffusion,3,maxDofs,true)

/// Bake-off Kernels (BKs)
template <typename BFI, int VDIM = 1, bool GLL = false>
struct Kernel: public BakeOff
{
   GridFunction y;

   Kernel(AssemblyLevel assembly, int order, int N)
   : BakeOff(assembly,order,N,VDIM,GLL), y(&fes)
   {
      if (is_runnable())
      {
         x.Randomize(1);
         a.SetAssemblyLevel(assembly);
         a.AddDomainIntegrator(new BFI(one, GLL?irGLL:ir));
         a.Assemble();
         a.Mult(x, y);
         MFEM_DEVICE_SYNC;
      }
   }

   void setup() override
   {
      a.Assemble();
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }

   void benchmark() override
   {
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Generic CEED BKi
#define BakeOff_Kernel(assembly,i,KER,VDIM,MaxDofs,GLL)\
static void BK##i##assembly(bm::State &state){\
   const int dim = 3;\
   const int p = state.range(0);\
   const int target_dofs = state.range(1);\
   const int elem_dofs = pow(p+1, dim);\
   const int N = pow(target_dofs / elem_dofs, 1.0/dim) + 1;\
   Kernel<KER##Integrator,VDIM,GLL> ker(AssemblyLevel::assembly, p, N);\
   if ( !ker.is_runnable() ) { state.SkipWithError("MAX_MEM"); }\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);\
   state.counters["Dofs"] = bm::Counter(ker.dofs, bm::Counter::kDefaults);\
   state.counters["Order"] = bm::Counter(ker.p);}\
BENCHMARK(BK##i##assembly)->ArgsProduct({\
      benchmark::CreateDenseRange(1, 6, /*step=*/1),\
      benchmark::CreateRange(1024, MaxDofs, /*step=*/2)\
    })->Unit(bm::kMillisecond);\

// PARTIAL:
/// BK1PARTIAL: scalar E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(PARTIAL,1,Mass,1,maxDofs,false)

/// BK2PARTIAL: vector E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(PARTIAL,2,VectorMass,3,maxDofs,false)

/// BK3PARTIAL: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(PARTIAL,3,Diffusion,1,maxDofs,false)

/// BK4PARTIAL: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(PARTIAL,4,VectorDiffusion,3,maxDofs,false)

/// BK5PARTIAL: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(PARTIAL,5,Diffusion,1,maxDofs,true)

/// BK6PARTIAL: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(PARTIAL,6,VectorDiffusion,3,maxDofs,true)

// ELEMENT
/// BK1ELEMENT: scalar E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(ELEMENT,1,Mass,1,maxDofs,false)

/// BK2ELEMENT: vector E-vector-to-E-vector evaluation of mass matrix, q=p+2
// BakeOff_Kernel(ELEMENT,2,VectorMass,3,maxDofs,false)

/// BK3ELEMENT: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(ELEMENT,3,Diffusion,1,maxDofs,false)

/// BK4ELEMENT: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
// BakeOff_Kernel(ELEMENT,4,VectorDiffusion,3,maxDofs,false)

/// BK5ELEMENT: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(ELEMENT,5,Diffusion,1,maxDofs,true)

/// BK6ELEMENT: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
// BakeOff_Kernel(ELEMENT,6,VectorDiffusion,3,maxDofs,true)

// FULL
/// BK1FULL: scalar E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(FULL,1,Mass,1,maxDofs,false)

/// BK2FULL: vector E-vector-to-E-vector evaluation of mass matrix, q=p+2
// BakeOff_Kernel(FULL,2,VectorMass,3,maxDofs,false)

/// BK3FULL: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(FULL,3,Diffusion,1,maxDofs,false)

/// BK4FULL: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
// BakeOff_Kernel(FULL,4,VectorDiffusion,3,maxDofs,false)

/// BK5FULL: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(FULL,5,Diffusion,1,maxDofs,true)

/// BK6FULL: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
// BakeOff_Kernel(FULL,6,VectorDiffusion,3,maxDofs,true)
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
