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
//
//
//  This benchmark contains the implementation of the CEED's bake-off problems:
//  high-order kernels/benchmarks designed to test and compare the performance
//  of high-order codes.
//
//  See: ceed.exascaleproject.org/bps and github.com/CEED/benchmarks

#include "bench.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_BENCHMARK

#include <cassert>

#include "general/backends.hpp" // IWYU pragma: keep
#include "fem/qinterp/det.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep

constexpr int MAX_ORDER = 6, INC_NDOFS = 25;
constexpr int MAX_NDOFS = 16 * 1024 * (mfem_use_gpu ? 1024 : 16);

static void CustomArguments(bmi::Benchmark *b)
{
   constexpr auto estimate_dofs = [](int cells) constexpr -> int
   {
      return (cells + 1) * (cells + 1) * (cells + 1);
   };

   for (int order = MAX_ORDER; order >= 1; --order)
   {
      for (int cells = INC_NDOFS; estimate_dofs(cells) <= MAX_NDOFS;
           cells += INC_NDOFS)
      {
         b->Args({order, cells});
      }
   }
}

static void AddKernelSpecializations()
{
   using DET = QuadratureInterpolator::DetKernels;
   DET::Specialization<3, 3, 2, 2>::Add();
   DET::Specialization<3, 3, 2, 3>::Add();
   DET::Specialization<3, 3, 2, 5>::Add();
   DET::Specialization<3, 3, 2, 6>::Add();
   DET::Specialization<3,3, 2,7>::Add();
   // DET::Specialization<3,3, 2,8>::Add(); // exceeds memory limits

   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();

   using LIN = DomainLFIntegrator::AssembleKernels;
   LIN::Specialization<3, 7, 7>::Add();
}

/// Bake-off base class
template <int VDIM, bool GLL>
struct BakeOff
{
   static constexpr int DIM = 3;
   const int p, c, q, n, nx, ny, nz;
   const bool check_x, check_y, check_z, checked;
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

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)),
      nz(n),
      check_x(p * nx * p * ny * p * nz <= c * c * c),
      check_y(p * (nx + 1) * p * (ny + 1) * p * nz > c * c * c),
      check_z(p * (nx + 1) * p * (ny + 1) * p * (nz + 1) > c * c * c),
      checked((assert(check_x &&check_y &&check_z), true)),
      mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      fec(p, DIM, BasisType::GaussLobatto),
      fes(&mesh, &fec, VDIM, VDIM == 3 ? Ordering::byVDIM : Ordering::byNODES),
      geom_type(mesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)), one(1.0), uvec(DIM),
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
   const int max_it = 32, print_lvl = -1;

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
   using BakeOff<VDIM, GLL>::unit_vec;

   Problem(int order, int side):
      BakeOff<VDIM, GLL>(order, side),
      ess_bdr(mesh.bdr_attributes.Max()),
      b(&fes)
   {
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if constexpr (VDIM == 1)
      {
         b.AddDomainIntegrator(new DomainLFIntegrator(one));
      }
      else
      {
         b.AddDomainIntegrator(new VectorDomainLFIntegrator(unit_vec));
      }
      b.UseFastAssembly(true);
      b.Assemble();

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, ir));

      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetOperator(*A);
      cg.SetAbsTol(0.0);
      cg.iterative_mode = false;
      {
         cg.SetPrintLevel(-1);
         cg.SetMaxIter(2000);
         cg.SetRelTol(1e-8);
         cg.Mult(B, X);
         MFEM_VERIFY(cg.GetConverged(), "CG solver did not converge!");
      }
      cg.SetRelTol(0.0);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);

      benchmark();
      mdofs = 0.0;
   }

   void benchmark() override
   {
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs() * cg.GetNumIterations();
   }
};

/// Bake-off Problems benchmarks
#define BakeOff_Problem(i, Kernel, VDIM, p_eq_q)                     \
   static void BP##i(bm::State &state)                               \
   {                                                                 \
      const int p = state.range(0);                                  \
      const int side = state.range(1);                               \
      Problem<Kernel##Integrator, VDIM, p_eq_q> ker(p, side);        \
      while (state.KeepRunning()) { ker.benchmark(); }               \
      bm::Counter::Flags flags = bm::Counter::kIsRate;               \
      state.counters["Dofs"] = bm::Counter(ker.dofs);                \
      state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags); \
      state.counters["Order"] = bm::Counter(p);                      \
   }                                                                 \
   BENCHMARK(BP##i)                                                  \
      ->Apply(CustomArguments)                                       \
      ->Unit(bm::kMillisecond);

/// BP1: scalar PCG with mass matrix, q=p+2
BakeOff_Problem(1, Mass, 1, false);

/// BP2: vector PCG with mass matrix, q=p+2
BakeOff_Problem(2, VectorMass, 3, false);

/// BP3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Problem(3, Diffusion, 1, false);

/// BP4: vector PCG with stiffness matrix, q=p+2
BakeOff_Problem(4, VectorDiffusion, 3, false);

/// BP5: scalar PCG with stiffness matrix, q=p+1
BakeOff_Problem(5, Diffusion, 1, true);

/// BP6: vector PCG with stiffness matrix, q=p+1
BakeOff_Problem(6, VectorDiffusion, 3, true);

/// Bake-off Kernels (BKs)
template <typename BFI, int VDIM, bool GLL>
struct Kernel : public BakeOff<VDIM, GLL>
{
   BilinearFormIntegrator *bfi;
   Vector xe, ye;

   using base = BakeOff<VDIM, GLL>;
   using base::ir;
   using base::one;
   using base::fes;
   using base::mdofs;

   Kernel(int order, int side): BakeOff<VDIM, GLL>(order, side)
   {
      bfi = new BFI(one, ir);
      bfi->AssemblePA(fes);

      const Table &el2dof = fes.GetElementToDofTable();
      const int e_size = el2dof.Size_of_connections()*fes.GetVDim();
      const auto R = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      MFEM_VERIFY(e_size == R->Height(), "Input/Output E-vector size mismatch!");

      xe.SetSize(R->Height());
      ye.SetSize(R->Height());
      xe.UseDevice(true);
      ye.UseDevice(true);

      xe.Randomize(1);
      xe.Read();

      benchmark();
      mdofs = 0.0;
   }

   void benchmark() override
   {
      ye = 0.0;
      bfi->AddMultPA(xe, ye);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }
};

/// Bake-off Kernels benchmarks
#define BakeOff_Kernel(i, KER, VDIM, GLL)                            \
   static void BK##i(bm::State &state)                               \
   {                                                                 \
      const int p = state.range(0);                                  \
      const int side = state.range(1);                               \
      Kernel<KER##Integrator, VDIM, GLL> ker(p, side);               \
      while (state.KeepRunning()) { ker.benchmark(); }               \
      bm::Counter::Flags flags = bm::Counter::kIsRate;               \
      state.counters["Dofs"] = bm::Counter(ker.dofs);                \
      state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags); \
      state.counters["Order"] = bm::Counter(p);                      \
   }                                                                 \
   BENCHMARK(BK##i)                                                  \
      ->Apply(CustomArguments)                                       \
      ->Unit(bm::kMillisecond);

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
 * Command line examples:
 *    --benchmark_context=device=gpu
 *    --benchmark_filter=BP1/6
 */
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   AddKernelSpecializations();

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
   bm::Shutdown();
   return 0;
}

#endif // MFEM_USE_BENCHMARK
