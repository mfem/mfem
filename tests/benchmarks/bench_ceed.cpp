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
//  See: https://ceed.exascaleproject.org/bps

#include "bench.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_BENCHMARK

#include <cassert>
#include <string>

#include "fem/qinterp/det.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep
#include "fem/integ/bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep

// Custom benchmark arguments generator
static void CustomArguments(bmi::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 16 * 1024 * (mfem_use_gpu ? 1024 : 8);

   const auto orders = { 7, 6, 5, 4, 3, 2, 1 };

   constexpr auto ndofs = [](int n) constexpr noexcept -> int
   {
      return (n + 1) * (n + 1) * (n + 1);
   };

   constexpr auto inc = [](int n) constexpr noexcept -> int
   {
      return n < 160 ?  4 : n < 240 ?  8 : n < 320 ? 16 : 32;
   };

   for (auto p : orders)
   {
      for (int n = 16; ndofs(n) <= MAX_NDOFS; n += inc(n))
      {
         b->Args({p, n});
      }
   }
}

// Register kernel specializations used in the benchmarks
static void AddKernelSpecializations()
{
   using DET = QuadratureInterpolator::DetKernels;
   DET::Specialization<3, 3, 2, 2>::Add();
   DET::Specialization<3, 3, 2, 3>::Add();
   DET::Specialization<3, 3, 2, 5>::Add();
   DET::Specialization<3, 3, 2, 6>::Add();
   DET::Specialization<3, 3, 5, 5>::Add();
   // Others might exceed memory limits

   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 2>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 9>::Add();

   using LIN = DomainLFIntegrator::AssembleKernels;
   LIN::Specialization<3, 7, 7>::Add();
   LIN::Specialization<3, 6, 6>::Add();
   LIN::Specialization<3, 8, 8>::Add();

   using VDIFF = VectorDiffusionIntegrator::ApplyPAKernels;
   VDIFF::Specialization<3, 3, 3, 3>::Add();
   VDIFF::Specialization<3, 3, 4, 4>::Add();
   VDIFF::Specialization<3, 3, 5, 5>::Add();
   VDIFF::Specialization<3, 3, 6, 6>::Add();
   VDIFF::Specialization<3, 3, 7, 7>::Add();
   VDIFF::Specialization<3, 3, 8, 8>::Add();
}

// Bake-off base class
template <int BFI, int VDIM, bool GLL>
struct BakeOff
{
   inline static constexpr int DIM = 3;
   const int p, c, q, n, nx, ny, nz;
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
   BilinearFormIntegrator *bfi;

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)),
      n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)),
      nz(n),
      mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
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
      if constexpr (BFI == 1)
      {
         bfi = new MassIntegrator(one, ir);
      }
      else if constexpr (BFI == 2)
      {
         bfi = new VectorMassIntegrator(one, ir);
      }
      else if constexpr (BFI == 3 || BFI == 5)
      {
         bfi = new DiffusionIntegrator(one, ir);
      }
      else if constexpr (BFI == 4 || BFI == 6)
      {
         bfi = new VectorDiffusionIntegrator(one, ir);
      }
      else
      {
         static_assert(BFI >= 1 && BFI <= 6, "Invalid BilinearFormIntegrator");
      }
      a.AddDomainIntegrator(bfi);
   }

   virtual void benchmark() = 0;

   [[nodiscard]] double SumMdofs() const noexcept { return mdofs; }

   [[nodiscard]] double MDofs() const noexcept { return 1e-6 * dofs; }
};

// Bake-off Problems (BPs)
template <int BFI, int VDIM, bool GLL>
struct BP : public BakeOff<BFI, VDIM, GLL>
{
   const int max_it = 32, print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   LinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;

   using base = BakeOff<BFI, VDIM, GLL>;
   using base::a;
   using base::ir;
   using base::one;
   using base::mesh;
   using base::fes;
   using base::x;
   using base::y;
   using base::mdofs;
   using base::unit_vec;
   using base::bfi;

   BP(int p, int side) noexcept: base(p, side),
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
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetOperator(*A);
      cg.SetAbsTol(0.0);
      cg.iterative_mode = false;
      {
         cg.SetPrintLevel(-1);
         cg.SetMaxIter(1000);
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

// Bake-off Kernels (BKs)
template <int BFI, int VDIM, bool GLL>
struct BK : public BakeOff<BFI, VDIM, GLL>
{
   Vector xe, ye;

   using base = BakeOff<BFI, VDIM, GLL>;
   using base::ir;
   using base::one;
   using base::bfi;
   using base::fes;
   using base::mdofs;

   BK(int order, int side) noexcept: base(order, side)
   {
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
      ye = 0.0;

      benchmark();
      mdofs = 0.0;
   }

   void benchmark() override
   {
      bfi->AddMultPA(xe, ye);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }
};

// Benchmarks
template <typename T>
static void Benchmark(bm::State& state) noexcept
{
   T run(state.range(0), state.range(1));
   while (state.KeepRunning()) { run.benchmark(); }
   state.counters["Dofs"] = bm::Counter(run.dofs);
   state.counters["MDof/s"] = bm::Counter(run.SumMdofs(), bm::Counter::kIsRate);
   state.counters["Order"] = bm::Counter(state.range(0));
}

#define REGISTER(PK, BFI, VDIM, GLL) \
   BENCHMARK_TEMPLATE(Benchmark, PK<BFI, VDIM, GLL>) \
   ->Name(#PK #BFI)->Apply(CustomArguments)->Unit(bm::kMillisecond)

// BP1: scalar PCG with mass matrix, q=p+2
REGISTER(BP, 1, 1, false);

// BP2: vector PCG with mass matrix, q=p+2
REGISTER(BP, 2, 3, false);

// BP3: scalar PCG with stiffness matrix, q=p+2
REGISTER(BP, 3, 1, false);

// BP4: vector PCG with stiffness matrix, q=p+2
REGISTER(BP, 4, 3, false);

// BP5: scalar PCG with stiffness matrix, q=p+1
REGISTER(BP, 5, 1, true);

// BP6: vector PCG with stiffness matrix, q=p+1
REGISTER(BP, 6, 3, true);

// BK1: scalar E-vector-to-E-vector evaluation of mass matrix, q=p+2
REGISTER(BK, 1, 1, false);

// BK2: vector E-vector-to-E-vector evaluation of mass matrix, q=p+2
REGISTER(BK, 2, 3, false);

// BK3: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
REGISTER(BK, 3, 1, false);

// BK4: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
REGISTER(BK, 4, 3, false);

// BK5: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
REGISTER(BK, 5, 1, true);

// BK6: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
REGISTER(BK, 6, 3, true);

/**
 * @brief CEED Bake-off Problems main entry point
 * Command line options:
 *    --benchmark_context=device=gpu
 *    --benchmark_filter=BP1
 *    --benchmark_out_format=csv
 *    --benchmark_out=bp1.csv
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

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks(&CR);
   bm::Shutdown();

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
