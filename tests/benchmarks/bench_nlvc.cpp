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

#include "bench.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_BENCHMARK

#include <cstdlib>
#include <functional>

#include <fem/qinterp/grad.hpp>

using namespace mfem;

// Custom benchmark arguments generator ///////////////////////////////////////
static void CustomArguments(bm::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 8 * 1024 * (mfem_use_gpu ? 1024 : 8);

   const auto orders = { 6, 5, 4, 3, 2, 1 };

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
      for (int n = (mfem_use_gpu ? 16 : 8); ndofs(n) <= MAX_NDOFS; n += inc(n))
      {
         b->Args({p, n});
      }
   }
}

/// Basic Kernels Specializations /////////////////////////////////////////////
static void AddBasicKernelSpecializations()
{
   using Grad = QuadratureInterpolator::GradKernels;
   // 2D
   Grad::Specialization<2, QVectorLayout::byNODES, false, 2,2,7>::Add();
   Grad::Specialization<2, QVectorLayout::byNODES, false, 2,2,8>::Add();
   Grad::Specialization<2, QVectorLayout::byNODES, false, 2,2,10>::Add();
   // 3D
   Grad::Specialization<3, QVectorLayout::byNODES, false, 3,2,7>::Add();
   Grad::Specialization<3, QVectorLayout::byNODES, false, 3,2,9>::Add();
   Grad::Specialization<3, QVectorLayout::byNODES, false, 3,2,10>::Add();
}

/// VectorConvectionNLFBenchmark //////////////////////////////////////////////
template <int DIM>
struct VectorConvectionNLFBenchmark
{
   const int p, c, q, n, nx, ny, nz;
   const std::function<Mesh()> MakeCartesianMesh = [&]()
   {
      if constexpr (DIM == 2)
      {
         return Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL);
      }
      else
      {
         return Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON);
      }
   };
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   ConstantCoefficient const_coeff { M_2_SQRTPI };
   NonlinearFormIntegrator *nlfi;
   NonlinearForm nlf;
   Operator *grad;
   GridFunction x, dx, y_pa;
   Vector xe, dxe, ye;
   const int dofs;
   const int q1d;
   double mdofs{};

   VectorConvectionNLFBenchmark(int p, int side):
      p(p), c(side), q(2 * p + 3), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
      mesh(MakeCartesianMesh()),
      fec(p, DIM),
      fes(&mesh, &fec, DIM),
      geom_type(mesh.GetTypicalElementGeometry()),
      irs(0, Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)),
      nlfi(new VectorConvectionNLFIntegrator(const_coeff)),
      nlf(&fes),
      x(&fes),
      dx(&fes),
      y_pa(&fes),
      dofs(fes.GetTrueVSize()),
      q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints())
   {
      MFEM_VERIFY(q1d*q1d*(DIM == 3 ? q1d : 1) == ir->GetNPoints(), "");

      nlf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      nlf.AddDomainIntegrator(nlfi);
      nlf.Setup();

      dx.Randomize(0x9e3779b9), x.Randomize(0x100001b3);

      grad = &nlf.GetGradient(x);

      const Table &el2dof = fes.GetElementToDofTable();
      const int e_size = el2dof.Size_of_connections()*fes.GetVDim();
      const auto R = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      MFEM_VERIFY(e_size == R->Height(), "Input/Output E-vector size mismatch!");
      xe.SetSize(R->Height()), dxe.SetSize(R->Height()), ye.SetSize(R->Height());
      xe.UseDevice(true), dxe.UseDevice(true), ye.UseDevice(true);
      xe.Randomize(0x100001b3), dxe.Randomize(0x9e3779b9), ye = 0.0;

      mdofs = 0.0;
   }

   void Setup()
   {
      nlfi->AssembleGradPA(xe, fes);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AddMult()
   {
      nlf.AddMult(x, y_pa);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AddMultPA()
   {
      nlfi->AddMultPA(xe, ye);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AddMultGrad()
   {
      grad->Mult(dx, y_pa);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AddMultGradPA()
   {
      nlfi->AddMultGradPA(dxe, ye);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AssembleGradDiagonal()
   {
      grad->AssembleDiagonal(ye);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   [[nodiscard]] double SumMdofs() const noexcept { return mdofs; }

   [[nodiscard]] double MDofs() const noexcept { return 1e-6 * dofs; }
};

///////////////////////////////////////////////////////////////////////////////
#define RegisterVectorConvectionNLFBenchmark(Benchmark, DIM)         \
   static void Benchmark##DIM##d(bm::State &state)                   \
   {                                                                 \
      const auto order = static_cast<int>(state.range(0));           \
      const auto side = static_cast<int>(state.range(1));            \
      VectorConvectionNLFBenchmark<DIM> ker(order, side);            \
      while (state.KeepRunning()) { ker.Benchmark(); }               \
      bm::Counter::Flags flags = bm::Counter::kIsRate;               \
      state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags); \
      state.counters["Dofs"] = bm::Counter(ker.dofs);                \
      state.counters["p"] = bm::Counter(order);                      \
   }                                                                 \
   BENCHMARK(Benchmark##DIM##d)                                      \
      ->Apply(CustomArguments)                                       \
      ->Unit(bm::kMillisecond)

RegisterVectorConvectionNLFBenchmark(Setup,3);
RegisterVectorConvectionNLFBenchmark(AddMult,3);
RegisterVectorConvectionNLFBenchmark(AddMultPA,3);
RegisterVectorConvectionNLFBenchmark(AddMultGrad,3);
RegisterVectorConvectionNLFBenchmark(AddMultGradPA,3);
RegisterVectorConvectionNLFBenchmark(AssembleGradDiagonal,3);

RegisterVectorConvectionNLFBenchmark(Setup,2);
RegisterVectorConvectionNLFBenchmark(AddMult,2);
RegisterVectorConvectionNLFBenchmark(AddMultPA,2);
RegisterVectorConvectionNLFBenchmark(AddMultGrad,2);
RegisterVectorConvectionNLFBenchmark(AddMultGradPA,2);
RegisterVectorConvectionNLFBenchmark(AssembleGradDiagonal,2);

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   AddBasicKernelSpecializations();

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_context = "cpu";
   const auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
      {
         mfem::out << device->first << " : "
                   << device->second << std::endl;
         device_context = device->second;
      }
   }
   Device device(device_context.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks(&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
