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
#include <cmath>
#include <functional>
#include <string>

#include "fem/qinterp/det.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/integ/bilininteg_pa_mma.hpp" // IWYU pragma: keep
#include "fem/integ/bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_simplices_mma.hpp" // IWYU pragma: keep

static int SnapMmaElementsPerDir(int n) noexcept
{
   // if (n < 8) { n = 8; }
   if (n < 32)
   {
      n = (n + 3) / 4 * 4;
      if (n == 28) { n = 32; }
      return n;
   }
   return (n + 7) / 8 * 8;
}

// Note: use the public ::benchmark::Benchmark type (Homebrew google-benchmark).
// Some newer/custom installs nest Benchmark under ::benchmark::internal; if
// build fails with "no type named Benchmark", point BENCHMARK_DIR at Homebrew
// or change the parameter type to bm::internal::Benchmark*.
static void CustomArguments(bm::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 12 * 1024 * (mfem_use_gpu ? 1024 : 8);

   const auto orders = { 7, 6, 5, 4, 3, 2, 1 };

   constexpr auto ndofs_hex = [](int n) constexpr noexcept -> int
   {
      return (n + 1) * (n + 1) * (n + 1);
   };

   for (auto p : orders)
   {
      const int max_side = (p == 1) ? 128 : (p == 2) ? 160 : 1 << 30;
      int prev_side = -1;
      for (double ndofs_t = 1.0e4;
           ndofs_t <= static_cast<double>(MAX_NDOFS) * 1.01;
           ndofs_t *= 1.25)
      {
         const double p3 = static_cast<double>(p) * p * p;
         int N = static_cast<int>(std::lround(std::cbrt(ndofs_t / p3)));
         N = SnapMmaElementsPerDir(N);
         const int side = N * p; // MeshExtents → nx=ny=nz=N for hex
         if (side == prev_side) { continue; }
         if (side > max_side) { break; }
         if (ndofs_hex(side) > MAX_NDOFS) { break; }
         prev_side = side;
         b->Args({p, side});
      }
   }
}

struct MeshExtents { int n, nx, ny, nz; };

// Cubic (3D) / square (2D) meshes from the hex-ref DOF budget
template <int DIM>
static MeshExtents MeshExtentsFromHexRef(int p, int side) noexcept
{
   static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
   MFEM_ASSERT(p >= 1, "invalid order");
   MFEM_ASSERT(side >= p, "hex-reference side too small for order p");

   const int N_hex = std::max(1, side / p);
   const double ndofs_t =
      std::pow(static_cast<double>(N_hex) * p, 3.0); // == N_hex^3 * p^3

   if constexpr (DIM == 3)
   {
      const int n = SnapMmaElementsPerDir(N_hex);
      return {n, n, n, n};
   }
   else
   {
      // Match hex-ref DOF budget: N^2 * p^2 ≈ ndofs_t.
      const double p2 = static_cast<double>(p) * p;
      int n = static_cast<int>(std::lround(std::sqrt(ndofs_t / p2)));
      n = (n + 3) / 4 * 4;
      if (n < 4) { n = 4; }
      return {n, n, n, 1};
   }
}

// Register kernel specializations used in the benchmarks (both 2D and 3D)
static void AddKernelSpecializations()
{
   using DET = QuadratureInterpolator::DetKernels;
   DET::Specialization<3, 3, 2, 2>::Add();
   DET::Specialization<3, 3, 2, 3>::Add();
   DET::Specialization<3, 3, 2, 5>::Add();
   DET::Specialization<3, 3, 2, 6>::Add();
   DET::Specialization<3, 3, 5, 5>::Add();
   // Others might exceed memory limits
   DET::Specialization<2, 2, 2, 2>::Add();
   DET::Specialization<2, 2, 2, 3>::Add();
   DET::Specialization<2, 2, 2, 5>::Add();
   DET::Specialization<2, 2, 2, 6>::Add();
   DET::Specialization<2, 2, 5, 5>::Add();

   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 2>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 9>::Add();
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 2>::Add();
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 7>::Add();
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 8>::Add();
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 9>::Add();

   using LIN = DomainLFIntegrator::AssembleKernels;
   LIN::Specialization<3, 7, 7>::Add();
   LIN::Specialization<3, 6, 6>::Add();
   LIN::Specialization<3, 8, 8>::Add();
   LIN::Specialization<2, 7, 7>::Add();
   LIN::Specialization<2, 6, 6>::Add();
   LIN::Specialization<2, 8, 8>::Add();

   using VDIFF = VectorDiffusionIntegrator::ApplyPAKernels;
   VDIFF::Specialization<3, 3, 3, 3>::Add();
   VDIFF::Specialization<3, 3, 4, 4>::Add();
   VDIFF::Specialization<3, 3, 5, 5>::Add();
   VDIFF::Specialization<3, 3, 6, 6>::Add();
   VDIFF::Specialization<3, 3, 7, 7>::Add();
   VDIFF::Specialization<3, 3, 8, 8>::Add();
   VDIFF::Specialization<2, 2, 3, 3>::Add();
   VDIFF::Specialization<2, 2, 4, 4>::Add();
   VDIFF::Specialization<2, 2, 5, 5>::Add();
   VDIFF::Specialization<2, 2, 6, 6>::Add();
   VDIFF::Specialization<2, 2, 7, 7>::Add();
   VDIFF::Specialization<2, 2, 8, 8>::Add();
   // BP4quad: q = p+2 → (D1D, Q1D) = (p+1, p+2)
   VDIFF::Specialization<2, 2, 2, 3>::Add();
   VDIFF::Specialization<2, 2, 3, 4>::Add();
   VDIFF::Specialization<2, 2, 4, 5>::Add();
   VDIFF::Specialization<2, 2, 5, 6>::Add();
   VDIFF::Specialization<2, 2, 6, 7>::Add();
   VDIFF::Specialization<2, 2, 7, 8>::Add();
   VDIFF::Specialization<2, 2, 8, 9>::Add();
}

// Bake-off base class.
// POS / MMA:
//   GLL MMA:      SIMPLEX, !POS  (MMA default-on; force unused)
//   Positive SUM: SIMPLEX, POS, !MMA  (Stroud)
//   Positive MMA: SIMPLEX, POS, MMA   (ForceMMA)
//   Tensor SUM:   !SIMPLEX, !MMA
//   Tensor MMA:   !SIMPLEX, MMA       (ForceMMA, CUDA)
template <int BFI, int DIM, int VDIM, bool GLL,
          bool SIMPLEX, bool POS, bool MMA>
struct BakeOff
{
   static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
   static_assert(!MMA || !SIMPLEX || POS,
                 "On simplices, MMA marks Positive MMA only");
   static constexpr bool visualization = false;

   static constexpr bool Simplex = SIMPLEX;
   static constexpr bool simplex = SIMPLEX;
   static constexpr bool pos = POS;
   static constexpr bool mma = MMA;

   const int p, c, q, n, nx, ny, nz;

   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir, *ir_rhs;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   GridFunction x, y;
   BilinearForm a;
   double mdofs{};
   BilinearFormIntegrator *bfi;

   BakeOff(int p, int side)
      : BakeOff(p, side, MeshExtentsFromHexRef<DIM>(p, side)) {}

   BakeOff(int p, int side, MeshExtents e):
      p(p), c(side), // hex-reference 1D size; NDOf target ≈ (side+1)^3
      // with simplex, mass D1D = Q1D; diffusion D1D = Q1D + 1
      q(2 * p + (GLL ? -1 : (SIMPLEX ? (BFI==1 ? 0 : -1) : 3))),
      n(e.n), nx(e.nx), ny(e.ny), nz(e.nz),
      mesh([&]()
   {
      if constexpr (DIM == 2)
      {
         return Mesh::MakeCartesian2D(e.nx, e.ny,
                                      SIMPLEX ? Element::TRIANGLE
                                      : Element::QUADRILATERAL);
      }
      else
      {
         return Mesh::MakeCartesian3D(e.nx, e.ny, e.nz,
                                      SIMPLEX ? Element::TETRAHEDRON
                                      : Element::HEXAHEDRON);
      }
   }()),
   fec(p, DIM,
       SIMPLEX
       ? (POS ? BasisType::Positive : BasisType::GaussLobatto)
       : BasisType::GaussLobatto),
   fes(&mesh, &fec, VDIM, VDIM == DIM ? Ordering::byVDIM : Ordering::byNODES),
   geom_type(mesh.GetTypicalElementGeometry()),
   irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
   // pos_sum uses Stroud; gll_mma / pos_mma use standard simplex rules
   ir((SIMPLEX && POS && !MMA)
      ? &StroudIntRules.Get(geom_type, q)
      : &irs.Get(geom_type, q)),
   ir_rhs(&IntRules.Get(geom_type, 2*p)),
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
template
<int BFI, int DIM, int VDIM, bool GLL, bool SIMPLEX, bool POS, bool MMA>
struct BP : public BakeOff<BFI, DIM, VDIM, GLL, SIMPLEX, POS, MMA>
{
   const int max_it = 32, print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   LinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;

   using base = BakeOff<BFI, DIM, VDIM, GLL, SIMPLEX, POS, MMA>;
   using base::a;
   using base::ir_rhs;
   using base::one;
   using base::mesh;
   using base::fes;
   using base::x;
   using base::y;
   using base::dofs;
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
         b.AddDomainIntegrator(new DomainLFIntegrator(one, ir_rhs));
      }
      else
      {
         b.AddDomainIntegrator(new VectorDomainLFIntegrator(unit_vec, ir_rhs));
      }
      b.UseFastAssembly(true);
      b.Assemble();

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetOperator(*A);
      cg.iterative_mode = false;
      if (dofs < 64 * 1024)
      {
         cg.SetPrintLevel(-1);
         cg.SetMaxIter(1000);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(1e-8);
         cg.Mult(B, X);
         if (!cg.GetConverged())
         {
            cg.SetPrintLevel(3);
            cg.Mult(B, X);
         }
         MFEM_VERIFY(cg.GetConverged(), "CG solver did not converge!");
         if constexpr (base::visualization)
         {
            a.RecoverFEMSolution(X, b, x);
            socketstream glvis("localhost", 19916);
            glvis << "solution\n" << mesh << x << std::flush;
         }
      }
      cg.SetRelTol(0.0);
      cg.SetAbsTol(0.0);
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
template
<int BFI, int DIM, int VDIM, bool GLL, bool SIMPLEX, bool POS, bool MMA>
struct BK : public BakeOff<BFI, DIM, VDIM, GLL, SIMPLEX, POS, MMA>
{
   Vector xe, ye;

   using base = BakeOff<BFI, DIM, VDIM, GLL, SIMPLEX, POS, MMA>;
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
   ForceMMA(T::mma);
   if constexpr (!T::simplex && T::mma)
   {
      if (!Device::Allows(Backend::CUDA_MASK))
      {
         state.SkipWithError("Tensors MMA benchmarks require CUDA device enabled");
         return;
      }
   }

   T run(state.range(0), state.range(1));
   while (state.KeepRunning()) { run.benchmark(); }
   state.counters["Dofs"] = bm::Counter(run.dofs);
   state.counters["MDof/s"] = bm::Counter(run.SumMdofs(), bm::Counter::kIsRate);
   state.counters["Order"] = bm::Counter(state.range(0));
   state.counters["Simplex"] = bm::Counter(run.Simplex);
}

namespace ceed_bench
{

template <int Dim, bool Simplex, bool Pos, bool Mma, const char *Suffix>
struct Geom
{
   static constexpr int dim = Dim;
   static constexpr bool simplex = Simplex, pos = Pos, mma = Mma;
   static constexpr const char *suffix = Suffix;
};

inline constexpr char s_hex_sum[] = "hex_sum";
inline constexpr char s_hex_mma[] = "hex_mma";
inline constexpr char s_quad_sum[] = "quad_sum";
inline constexpr char s_quad_mma[] = "quad_mma";
inline constexpr char s_tet_gll_mma[] = "tet_gll_mma";
inline constexpr char s_tet_pos_sum[] = "tet_pos_sum";
inline constexpr char s_tet_pos_mma[] = "tet_pos_mma";
inline constexpr char s_tri_gll_mma[] = "tri_gll_mma";
inline constexpr char s_tri_pos_sum[] = "tri_pos_sum";
inline constexpr char s_tri_pos_mma[] = "tri_pos_mma";

using HexSum    = Geom<3, false, false, false, s_hex_sum>;
using HexMma    = Geom<3, false, false, true,  s_hex_mma>;
using QuadSum   = Geom<2, false, false, false, s_quad_sum>;
using QuadMma   = Geom<2, false, false, true,  s_quad_mma>;
using TetGllMma = Geom<3, true,  false, false, s_tet_gll_mma>;
using TetPosSum = Geom<3, true,  true,  false, s_tet_pos_sum>;
using TetPosMma = Geom<3, true,  true,  true,  s_tet_pos_mma>;
using TriGllMma = Geom<2, true,  false, false, s_tri_gll_mma>;
using TriPosSum = Geom<2, true,  true,  false, s_tri_pos_sum>;
using TriPosMma = Geom<2, true,  true,  true,  s_tri_pos_mma>;

} // namespace ceed_bench

using ceed_bench::HexSum;
using ceed_bench::HexMma;
using ceed_bench::QuadSum;
using ceed_bench::QuadMma;
using ceed_bench::TetGllMma;
using ceed_bench::TetPosSum;
using ceed_bench::TetPosMma;
using ceed_bench::TriGllMma;
using ceed_bench::TriPosSum;
using ceed_bench::TriPosMma;

#define REGISTER(PK, BFI, GEOM) \
   BENCHMARK_TEMPLATE(Benchmark, \
      PK<BFI, GEOM::dim, \
         ((BFI) % 2 ? 1 : GEOM::dim), \
         ((BFI) >= 5), \
         GEOM::simplex, GEOM::pos, GEOM::mma>) \
   ->Name(std::string(#PK #BFI) + GEOM::suffix) \
   ->Apply(CustomArguments)->Unit(bm::kMillisecond)

// BP1: scalar CG with mass matrix, q=p+2
REGISTER(BP, 1, HexSum);
REGISTER(BP, 1, HexMma);
REGISTER(BP, 1, QuadSum);
REGISTER(BP, 1, QuadMma);
REGISTER(BP, 1, TetGllMma);
REGISTER(BP, 1, TetPosSum);
REGISTER(BP, 1, TetPosMma);
REGISTER(BP, 1, TriGllMma);
REGISTER(BP, 1, TriPosSum);
REGISTER(BP, 1, TriPosMma);

// BP2: vector CG with mass matrix, q=p+2
REGISTER(BP, 2, HexSum);
REGISTER(BP, 2, QuadSum);

// BP3: scalar CG with stiffness matrix, q=p+2
REGISTER(BP, 3, HexSum);
REGISTER(BP, 3, HexMma);
REGISTER(BP, 3, QuadSum);
REGISTER(BP, 3, QuadMma);
REGISTER(BP, 3, TetGllMma);
REGISTER(BP, 3, TetPosSum);
REGISTER(BP, 3, TetPosMma);
REGISTER(BP, 3, TriGllMma);
REGISTER(BP, 3, TriPosSum);
REGISTER(BP, 3, TriPosMma);

// BP4: vector CG with stiffness matrix, q=p+2
REGISTER(BP, 4, HexSum);
REGISTER(BP, 4, QuadSum);

// BP5: scalar CG with stiffness matrix, q=p+1
REGISTER(BP, 5, HexSum);
REGISTER(BP, 5, QuadSum);
REGISTER(BP, 5, TetGllMma);
REGISTER(BP, 5, TetPosSum);
REGISTER(BP, 5, TetPosMma);
REGISTER(BP, 5, TriGllMma);
REGISTER(BP, 5, TriPosSum);
REGISTER(BP, 5, TriPosMma);

// BP6: vector CG with stiffness matrix, q=p+1
REGISTER(BP, 6, HexSum);
REGISTER(BP, 6, QuadSum);

// BK1: scalar E-to-E evaluation of mass matrix, q=p+2
REGISTER(BK, 1, HexSum);
REGISTER(BK, 1, QuadSum);

// BK2: vector E-to-E evaluation of mass matrix, q=p+2
REGISTER(BK, 2, HexSum);
REGISTER(BK, 2, QuadSum);

// BK3: scalar E-to-E evaluation of stiffness matrix, q=p+2
REGISTER(BK, 3, HexSum);
REGISTER(BK, 3, QuadSum);

// BK4: vector E-to-E evaluation of stiffness matrix, q=p+2
REGISTER(BK, 4, HexSum);
REGISTER(BK, 4, QuadSum);

// BK5: scalar E-to-E evaluation of stiffness matrix, q=p+1
REGISTER(BK, 5, HexSum);
REGISTER(BK, 5, QuadSum);

// BK6: vector E-to-E evaluation of stiffness matrix, q=p+1
REGISTER(BK, 6, HexSum);
REGISTER(BK, 6, QuadSum);

/**
 * @brief CEED Bake-off Problems main entry point
 * Command line options:
 *    --benchmark_context=device=gpu
 *    --benchmark_filter=BP1hex
 *    --benchmark_filter=BP3hex_mma
 *    --benchmark_filter=BP1tri_pos_sum
 *    --benchmark_filter=BP1tri_pos_mma
 *    --benchmark_out_format=csv
 *    --benchmark_out=bp1.csv
 *
 * Names encode geometry and PA backend:
 *    hex / quad           — tensor-product SUM (3D / 2D)
 *    hex_mma / quad_mma   — tensor-product sum-factored MMA (CUDA)
 *    tet_gll_mma / tri_gll_mma — simplex Gauss-Lobatto, dense MMA
 *    tet_pos_sum / tri_pos_sum — simplex Positive/Bernstein, Stroud sum-factorized
 *    tet_pos_mma / tri_pos_mma — simplex Positive/Bernstein, dense MMA
 *
 * Positive MMA (`*_pos_mma`) and tensors MMA (`hex_mma` / `quad_mma`)
 * runs on CPU (dense host path) as well as GPU, e.g.:
 *    --benchmark_context=device=cuda
 *    --benchmark_context=device=hip
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
