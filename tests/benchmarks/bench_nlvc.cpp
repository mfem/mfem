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

/// AlmostEqual ///////////////////////////////////////////////////////////////
// template <class T>
// [[nodiscard]]
// inline constexpr std::enable_if_t<std::is_floating_point_v<T>, bool>
// AlmostEqualEq(T a, T b, T eps_rel = 1e-10, T eps_abs = 1e-14) noexcept
// {
//    T diff = std::abs(a - b);
//    if (diff <= eps_abs) { return true; }
//    T scale = std::max(T(1), std::max(std::abs(a), std::abs(b)));
//    return diff <= eps_rel * scale;
// }

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
   const std::function<Mesh()> MakeMesh = [&]()
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
   FunctionCoefficient funct_coeff { [](const Vector &x) { return M_1_PI + x[0]*x[0];} };
   NonlinearFormIntegrator *nlfi_fa, *nlfi_pa;
   NonlinearForm nlf_fa, nlf_pa;
   GridFunction x, dx, y_fa, y_pa;
   Vector xe, dxe, ye;
   const int dofs;
   const int d1d, q1d;
   double mdofs{};

   VectorConvectionNLFBenchmark(int p, int side):
      p(p), c(side), q(2 * p + 3), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
      mesh(MakeMesh()),
      fec(p, DIM),
      fes(&mesh, &fec, DIM),
      geom_type(mesh.GetTypicalElementGeometry()),
      irs(0, Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)),
      // funct_coeff triggers projection, not needed for benchmark
      nlfi_fa(new VectorConvectionNLFIntegrator(const_coeff)),
      nlfi_pa(new VectorConvectionNLFIntegrator(const_coeff)),
      nlf_fa(&fes),
      nlf_pa(&fes),
      x(&fes),
      dx(&fes),
      y_fa(&fes),
      y_pa(&fes),
      dofs(fes.GetTrueVSize()),
      d1d(p + 1),
      q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints())
   {
      NVTX_MARK_FUNCTION;
      // db1("p:{} q:{} d1d:{} q1d:{} dofs:{}", p, q, d1d, q1d, dofs);
      MFEM_VERIFY(q1d*q1d*(DIM == 3 ? q1d : 1) == ir->GetNPoints(), "");

      NVTX_INI("NLF_PA Setup");
      nlf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      nlf_pa.AddDomainIntegrator(nlfi_pa);
      nlf_pa.Setup();
      NVTX_END("NLF_PA Setup");

      NVTX_INI("X Randomize");
      dx.Randomize(0x9e3779b9), x.Randomize(0x100001b3);
      NVTX_END("X Randomize");

      NVTX_INI("NLF_PA GetGradient");
      nlf_pa.GetGradient(x);
      NVTX_END("NLF_PA GetGradient");

      const Table &el2dof = fes.GetElementToDofTable();
      const int e_size = el2dof.Size_of_connections()*fes.GetVDim();
      const auto R = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      MFEM_VERIFY(e_size == R->Height(), "Input/Output E-vector size mismatch!");
      xe.SetSize(R->Height()), dxe.SetSize(R->Height()), ye.SetSize(R->Height());
      xe.UseDevice(true), dxe.UseDevice(true), ye.UseDevice(true);

      NVTX_INI("XE Randomize");
      xe.Randomize(0x100001b3), dxe.Randomize(0x9e3779b9), ye = 0.0;
      NVTX_END("XE Randomize");

      /*if (dofs < ((mfem_use_gpu ? 128 : 16) * 1024))
      {
         NVTX_INI("NLF_FA Setup");
         nlf_fa.AddDomainIntegrator(nlfi_fa);
         nlf_fa.Setup();
         NVTX_END("NLF_FA Setup");

         nlf_fa.Mult(x, y_fa);
         nlf_pa.Mult(x, y_pa);
         y_fa -= y_pa;
         MFEM_VERIFY(AlmostEqual(y_fa.Norml2(), 0.0),
                     "FA and PA Apply results differ: " << y_fa.Norml2());

         Operator &nlf_fa_grad = nlf_fa.GetGradient(x);
         Operator &nlf_pa_grad = nlf_pa.GetGradient(x);
         nlf_fa_grad.Mult(dx, y_fa);
         nlf_pa_grad.Mult(dx, y_pa);
         y_fa -= y_pa;
         MFEM_VERIFY(AlmostEqual(y_fa.Norml2(), 0.0),
                     "FA and PA Gradient results differ: " << y_fa.Norml2());

         Vector diag_fa(fes.GetVSize()), diag_pa(fes.GetVSize());
         dynamic_cast<SparseMatrix &>(nlf_fa.GetGradient(x)).GetDiag(diag_fa);
         nlf_pa.GetGradient(x).AssembleDiagonal(diag_pa);
         diag_fa -= diag_pa;
         MFEM_VERIFY(AlmostEqual(diag_fa.Norml2(), 0.0),
                     "FA and PA Diagonal results differ: " << diag_fa.Norml2());
         dbg("✅");
      }*/
      mdofs = 0.0;
   }

   void Setup()
   {
      NVTX_MARK_FUNCTION;
      nlfi_pa->AssembleGradPA(xe, fes);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AddMult()
   {
      NVTX_MARK_FUNCTION;
      nlf_pa.AddMult(x, y_pa);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AddMultPA()
   {
      NVTX_MARK_FUNCTION;
      nlfi_pa->AddMultPA(xe, ye);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AddMultGrad()
   {
      NVTX_MARK_FUNCTION;
      NVTX_INI("GetGradient");
      const auto &grad = nlf_pa.GetGradient(x);
      MFEM_DEVICE_SYNC;
      NVTX_END("GetGradient");

      NVTX_INI("Mult");
      grad.Mult(dx, y_pa);
      MFEM_DEVICE_SYNC;
      NVTX_END("Mult");
      mdofs += this->MDofs();
   }

   void AddMultGradPA()
   {
      NVTX_MARK_FUNCTION;
      nlfi_pa->AddMultGradPA(dxe, ye);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs();
   }

   void AssembleGradDiagonal()
   {
      NVTX_MARK_FUNCTION;
      NVTX_INI("GetGradient");
      const auto &grad = nlf_pa.GetGradient(x);
      MFEM_DEVICE_SYNC;
      NVTX_END("GetGradient");

      NVTX_INI("AssembleDiagonal");
      grad.AssembleDiagonal(ye);
      MFEM_DEVICE_SYNC;
      NVTX_END("AssembleDiagonal");
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

// RegisterVectorConvectionNLFBenchmark(Setup,2);
// RegisterVectorConvectionNLFBenchmark(AddMult,2);
// RegisterVectorConvectionNLFBenchmark(AddMultPA,2);
// RegisterVectorConvectionNLFBenchmark(AddMultGrad,2);
// RegisterVectorConvectionNLFBenchmark(AddMultGradPA,2);

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   dbg();
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
