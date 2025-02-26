// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
template <int VDIM, bool GLL>
struct BakeOff
{
   static constexpr int DIM = 3;
   const int N, p, q;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   GridFunction x, y;
   BilinearForm a;
   double mdofs{};

   BakeOff(int p):
      N(Device::IsEnabled() ? 32 : 4),
      p(p),
      q(2 * p + (GLL ? -1 : 3)),
      mesh(Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON)),
      fec(p, DIM, BasisType::GaussLobatto),
      fes(&mesh, &fec, VDIM, VDIM == 3 ? Ordering::byVDIM : Ordering::byNODES),
      geom_type(mesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)),
      one(1.0),
      uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(fes.GetTrueVSize()),
      x(&fes),
      y(&fes),
      a(&fes)
   {
      x = 0.0;
   }

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Bake-off Problems (BPs)
template <typename BFI, int VDIM, bool GLL>
struct Problem : public BakeOff<VDIM, GLL>
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

   using BakeOff<VDIM, GLL>::a;
   using BakeOff<VDIM, GLL>::ir;
   using BakeOff<VDIM, GLL>::one;
   using BakeOff<VDIM, GLL>::mesh;
   using BakeOff<VDIM, GLL>::fes;
   using BakeOff<VDIM, GLL>::x;
   using BakeOff<VDIM, GLL>::y;
   using BakeOff<VDIM, GLL>::mdofs;

   Problem(int order):
      BakeOff<VDIM, GLL>(order),
      ess_bdr(mesh.bdr_attributes.Max()),
      b(&fes)
   {
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if (VDIM == 1)
      {
         b.AddDomainIntegrator(new DomainLFIntegrator(this->one));
      }
      else
      {
         b.AddDomainIntegrator(new VectorDomainLFIntegrator(this->unit_vec));
      }
      b.UseFastAssembly(true);
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

   void benchmark() override
   {
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs() * cg.GetNumIterations();
   }
};

/// Bake-off Problems (BPs)
#define BakeOff_Problem(i, Kernel, VDIM, p_eq_q)                     \
   static void BP##i(bm::State &state)                               \
   {                                                                 \
      Problem<Kernel##Integrator, VDIM, p_eq_q> ker(state.range(0)); \
      while (state.KeepRunning()) { ker.benchmark(); }               \
      state.counters["MDof/s"] =                                     \
         bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);          \
   }                                                                 \
   BENCHMARK(BP##i)->DenseRange(1, 6)->Unit(bm::kMillisecond);

/// BP1: scalar PCG with mass matrix, q=p+2
BakeOff_Problem(1, Mass, 1, false)

/// BP2: vector PCG with mass matrix, q=p+2
BakeOff_Problem(2, VectorMass, 3, false)

/// BP3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Problem(3, Diffusion, 1, false)

/// BP4: vector PCG with stiffness matrix, q=p+2
BakeOff_Problem(4, VectorDiffusion, 3, false)

/// BP5: scalar PCG with stiffness matrix, q=p+1
BakeOff_Problem(5, Diffusion, 1, true)

/// BP6: vector PCG with stiffness matrix, q=p+1
BakeOff_Problem(6, VectorDiffusion, 3, true)

/// Bake-off Kernels (BKs)
template <typename BFI, int VDIM, bool GLL>
struct Kernel : public BakeOff<VDIM, GLL>
{
   using BakeOff<VDIM, GLL>::a;
   using BakeOff<VDIM, GLL>::ir;
   using BakeOff<VDIM, GLL>::one;
   using BakeOff<VDIM, GLL>::fes;
   using BakeOff<VDIM, GLL>::x;
   using BakeOff<VDIM, GLL>::y;
   using BakeOff<VDIM, GLL>::mdofs;

   Kernel(int order): BakeOff<VDIM, GLL>(order)
   {
      x.Randomize(1);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, ir));
      a.Assemble();
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }
};

/// Generic CEED BKi
#define BakeOff_Kernel(i, KER, VDIM, GLL)                      \
   static void BK##i(bm::State &state)                         \
   {                                                           \
      Kernel<KER##Integrator, VDIM, GLL> ker(state.range(0));  \
      while (state.KeepRunning()) { ker.benchmark(); }         \
      state.counters["MDof/s"] =                               \
         bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);    \
   }                                                           \
   BENCHMARK(BK##i)->DenseRange(1, 6)->Unit(bm::kMillisecond);

/// BK1: scalar E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(1, Mass, 1, false)

/// BK2: vector E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(2, VectorMass, 3, false)

/// BK3: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(3, Diffusion, 1, false)

/// BK4: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(4, VectorDiffusion, 3, false)

/// BK5: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(5, Diffusion, 1, true)

/// BK6: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(6, VectorDiffusion, 3, true)

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
   auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
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
