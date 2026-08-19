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
#include "fem/integ/mma/mma.hpp" // IWYU pragma: keep
#include "fem/integ/bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep
#include "fem/integ/mma/domain_lf.hpp" // IWYU pragma: keep

// CG verification for BP setups; enabled via --benchmark_context=cg=true
static bool cg_verify = false;

static int SnapMmaElementsPerDir(int n) noexcept
{
   if (n < 32)
   {
      n = (n + 3) / 4 * 4;
      return n;
   }
   return (n + 7) / 8 * 8;
}

static void CustomArguments(bm::Benchmark *benchmark) noexcept
{
#if defined(MFEM_USE_CUDA_OR_HIP_LANG)
   constexpr int MAX_NDOFS = 16 * 1024 * 1024;
#else
   constexpr int MAX_NDOFS = 16 * 1024 * 8;
#endif

   const auto orders = { 7, 6, 5, 4, 3, 2, 1 };

   constexpr auto ndofs_hex = [](int n) constexpr noexcept -> int
   {
      return (n + 1) * (n + 1) * (n + 1);
   };

   for (auto p : orders)
   {
      const int max_side = (p == 1) ? 128 : (p == 2) ? 160 : 1 << 30;
      int prev_side = -1;
      for (auto ndofs_t = 1.0e3;
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
         benchmark->Args({p, side});
      }
   }
}

struct MeshExtents { int n, nx, ny, nz; };

// Cubic (3D) / square (2D) meshes
template <int DIM>
static MeshExtents MeshExtentsFromHexRef(int p, int side) noexcept
{
   static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
   MFEM_ASSERT(p >= 1, "invalid order");
   MFEM_ASSERT(side >= p, "hex-reference side too small for order p");

   const int N_hex = std::max(1, side / p);
   const auto ndofs_t = std::pow(static_cast<double>(N_hex) * p, 3.0);

   if constexpr (DIM == 3)
   {
      const int n = SnapMmaElementsPerDir(N_hex);
      return {n, n, n, n};
   }
   else
   {
      // Match hex-ref DOF budget
      const auto p2 = static_cast<double>(p) * p;
      int n = static_cast<int>(std::lround(std::sqrt(ndofs_t / p2)));
      n = (n + 3) / 4 * 4;
      if (n < 4) { n = 4; }
      return {n, n, n, 1};
   }
}

// Register kernel specializations used in the benchmarks
static void AddKernelSpecializations()
{
   using DET = QuadratureInterpolator::DetKernels;
   DET::Specialization<2, 2, 2, 2>::Add();
   DET::Specialization<2, 2, 2, 3>::Add();
   DET::Specialization<2, 2, 2, 5>::Add();
   DET::Specialization<2, 2, 2, 6>::Add();
   DET::Specialization<2, 2, 5, 5>::Add();
   DET::Specialization<3, 3, 2, 2>::Add();
   DET::Specialization<3, 3, 2, 3>::Add();
   DET::Specialization<3, 3, 2, 5>::Add();
   DET::Specialization<3, 3, 2, 6>::Add();
   DET::Specialization<3, 3, 5, 5>::Add();

   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 2>::Add();
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 7>::Add();
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 8>::Add();
   GRAD::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 9>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 2>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 9>::Add();

   using LIN = DomainLFIntegrator::AssembleKernels;
   LIN::Specialization<2, 7, 7>::Add();
   LIN::Specialization<2, 6, 6>::Add();
   LIN::Specialization<2, 8, 8>::Add();
   LIN::Specialization<3, 7, 7>::Add();
   LIN::Specialization<3, 6, 6>::Add();
   LIN::Specialization<3, 8, 8>::Add();

   using VDIFF = VectorDiffusionIntegrator::ApplyPAKernels;
   VDIFF::Specialization<2, 2, 3, 3>::Add();
   VDIFF::Specialization<2, 2, 4, 4>::Add();
   VDIFF::Specialization<2, 2, 5, 5>::Add();
   VDIFF::Specialization<2, 2, 6, 6>::Add();
   VDIFF::Specialization<2, 2, 7, 7>::Add();
   VDIFF::Specialization<2, 2, 8, 8>::Add();
   VDIFF::Specialization<3, 3, 3, 3>::Add();
   VDIFF::Specialization<3, 3, 4, 4>::Add();
   VDIFF::Specialization<3, 3, 5, 5>::Add();
   VDIFF::Specialization<3, 3, 6, 6>::Add();
   VDIFF::Specialization<3, 3, 7, 7>::Add();
   VDIFF::Specialization<3, 3, 8, 8>::Add();
}

// Map CEED BP/BK index to the concrete bilinear-form integrator type
template <int BFI> struct BakeOffIntegrator;
template <> struct BakeOffIntegrator<1> { using type = MassIntegrator; };
template <> struct BakeOffIntegrator<2> { using type = VectorMassIntegrator; };
template <> struct BakeOffIntegrator<3> { using type = DiffusionIntegrator; };
template <> struct BakeOffIntegrator<4> { using type = VectorDiffusionIntegrator; };
template <> struct BakeOffIntegrator<5> { using type = DiffusionIntegrator; };
template <> struct BakeOffIntegrator<6> { using type = VectorDiffusionIntegrator; };

// Bake-off base class
// QGL: true → CEED GL quadrature q=p+2 (BP/BK 1–4); false → GLL q=p+1 (BP/BK 5–6)
template <int BFI, int DIM, int VDIM, bool QGL,
          bool SIMPLEX, bool POS, bool MMA>
struct BakeOff
{
   static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
   static_assert(BFI >= 1 && BFI <= 6, "Invalid BilinearFormIntegrator");
   static_assert(QGL ||
                 !SIMPLEX, "GLL quadrature (q=p+1) is only for tensor elements");

   using IntegratorType = typename BakeOffIntegrator<BFI>::type;

   static constexpr bool visualization = false;
   static constexpr bool simplex = SIMPLEX;
   static constexpr bool pos = POS;
   static constexpr bool mma = MMA;
   static constexpr bool mass = (BFI == 1);

   const int p, c, q, n, nx, ny, nz;

   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir, *ir_rhs;
   const int qnd;
   int q1d;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   GridFunction x, y;
   BilinearForm a;
   double mdofs{};
   BilinearFormIntegrator *bfi;
   int cg_final_iter{-1};
   real_t cg_final_norm{-1.0};

   BakeOff(int p, int side)
      : BakeOff(p, side, MeshExtentsFromHexRef<DIM>(p, side)) {}

   BakeOff(int p, int side, MeshExtents e): p(p), c(side),
      // q = integration-rule exactness (not CEED 1D point count)
      // Tensor QGL: q=2p+3 → Q1D=p+2; tensor !QGL (GLL): q=2p-1 → Q1D=p+1
      // Simplex Stroud: mass q=2p → D1D==Q1D; diffusion q=2p-1 → D1D==Q1D+1
      // Simplex MMA: mass q=2p; diffusion q=2p-2 (match FA GetRule on Pk)
      q(2 * p + (QGL ? (SIMPLEX ? (mass ? 0
                                   : ((POS && !MMA) ? -1 : -2))
                        : 3)
                 : -1)),
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
   irs(0, QGL ? Quadrature1D::GaussLegendre : Quadrature1D::GaussLobatto),
   // pos_sum uses Stroud; gll_mma / pos_mma use standard simplex rules
   ir((SIMPLEX && POS && !MMA)
      ? &StroudIntRules.Get(geom_type, q)
      : &irs.Get(geom_type, q)),
   ir_rhs(&IntRules.Get(geom_type, 2*p)),
   qnd(ir->GetNPoints()),
   q1d(0),
   one(1.0),
   uvec(DIM),
   unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
   dofs(fes.GetTrueVSize()),
   x(&fes),
   y(&fes),
   a(&fes)
   {
      constexpr auto ipow = [](int b, int e)
      {
         int r = 1;
         for (int i = 0; i < e; i++) { r *= b; }
         return r;
      };

      const int d1d = p + 1;

      if constexpr (!SIMPLEX) // tensor
      {
         q1d = QGL ? (q / 2 + 1) : (q / 2 + 2);
         MFEM_VERIFY(q == (QGL ? 2 * p + 3 : 2 * p - 1),
                     "tensor rule order");
         MFEM_VERIFY(q1d == (QGL ? p + 2 : p + 1),
                     "tensor Q1D must be p+2 (GL) or p+1 (GLL)");
         MFEM_VERIFY(qnd == ipow(q1d, DIM),
                     "tensor QND must be Q1D^dim");
      }
      else if constexpr (POS && !MMA) // Stroud sum-factorization
      {
         // mass: D1D == Q1D; diffusion: D1D == Q1D + 1
         q1d = mass ? d1d : d1d - 1;
         MFEM_VERIFY(q == (mass ? 2 * p : 2 * p - 1), "Stroud rule order");
         MFEM_VERIFY(qnd == ipow(q1d, DIM), "Stroud QND must be Q1D^dim");
      }
      else // simplex MMA (GLL nodal or Positive+ForceMMA): QND only, no 1D Q1D
      {
         MFEM_VERIFY(q == (mass ? 2 * p : 2 * p - 2), "simplex MMA rule order");
         MFEM_VERIFY(ir->GetOrder() >= q, "simplex MMA integration rule order");

         const FiniteElement &el = *fes.GetTypicalFE();
         ElementTransformation &T0 = *mesh.GetTypicalElementTransformation();
         const IntegrationRule &fa_ir = mass
                                        ? MassIntegrator::GetRule(el, el, T0, false)
                                        : DiffusionIntegrator::GetRule(el, el, false);
         const int fa_nq = fa_ir.GetNPoints();
         MFEM_VERIFY(qnd == fa_nq, "simplex MMA qnd must match FA GetRule nq");
      }

      x = 0.0;
      bfi = new IntegratorType(one, ir);
      a.AddDomainIntegrator(bfi);
   }

   void VerifyIntegratorNq() const
   {
      const int integ_nq = static_cast<const IntegratorType *>(bfi)->GetNq();
      MFEM_VERIFY(integ_nq == qnd, "integrator nq must match BakeOff qnd");
   }

   virtual void benchmark() = 0;

   [[nodiscard]] double SumMdofs() const noexcept { return mdofs; }

   [[nodiscard]] double MDofs() const noexcept { return 1e-6 * dofs; }
};

// Bake-off Problems (BPs)
template
<int BFI, int DIM, int VDIM, bool QGL, bool SIMPLEX, bool POS, bool MMA>
struct BP : public BakeOff<BFI, DIM, VDIM, QGL, SIMPLEX, POS, MMA>
{
   const int max_it = 32, print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   LinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;

   using base = BakeOff<BFI, DIM, VDIM, QGL, SIMPLEX, POS, MMA>;
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
      this->VerifyIntegratorNq();

      cg.SetOperator(*A);
      cg.iterative_mode = false;
      if (cg_verify)
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
         this->cg_final_iter = cg.GetNumIterations();
         this->cg_final_norm = cg.GetFinalNorm();
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
<int BFI, int DIM, int VDIM, bool QGL, bool SIMPLEX, bool POS, bool MMA>
struct BK : public BakeOff<BFI, DIM, VDIM, QGL, SIMPLEX, POS, MMA>
{
   Vector xe, ye;

   using base = BakeOff<BFI, DIM, VDIM, QGL, SIMPLEX, POS, MMA>;
   using base::ir;
   using base::one;
   using base::bfi;
   using base::fes;
   using base::mdofs;

   BK(int order, int side) noexcept: base(order, side)
   {
      bfi->AssemblePA(fes);
      this->VerifyIntegratorNq();

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
   MMAForce mma(T::mma);

   T run(state.range(0), state.range(1));
   while (state.KeepRunning()) { run.benchmark(); }
   BeginCounters();
   AddCounter(state, "Dofs", bm::Counter(run.dofs));
   AddCounter(state, "MDof/s",
              bm::Counter(run.SumMdofs(), bm::Counter::kIsRate));
   AddCounter(state, "Order", bm::Counter(state.range(0)),
              CounterColor::Default, true);
   AddCounter(state, "Q1D", bm::Counter(run.q1d),
              CounterColor::Default, true);
   AddCounter(state, "QND", bm::Counter(run.qnd),
              CounterColor::Default, true);
   if (run.cg_final_iter >= 0)
   {
      AddCounter(state, "CG_Iter", bm::Counter(run.cg_final_iter),
                 CounterColor::Magenta, true);
      AddCounter(state, "CG_Norm", bm::Counter(run.cg_final_norm),
                 CounterColor::Magenta);
   }
}

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

template <int DIM, bool SMPLX, bool POS, bool MMA, const char *SUFFIX>
struct Geom
{
   static constexpr int dim = DIM;
   static constexpr bool simplex = SMPLX, pos = POS, mma = MMA;
   static constexpr const char *suffix = SUFFIX;
};

//                   DIM, SMPLX, POS,   MMA,   SUFFIX
using HexSum    = Geom<3, false, false, false, s_hex_sum>;
using HexMma    = Geom<3, false, false, true,  s_hex_mma>;
using QuadSum   = Geom<2, false, false, false, s_quad_sum>;
using QuadMma   = Geom<2, false, false, true,  s_quad_mma>;
// GLL simplices already use dense MMA without ForceMMA; MMA=true matches *_mma
using TetGllMma = Geom<3, true,  false, true,  s_tet_gll_mma>;
using TetPosSum = Geom<3, true,  true,  false, s_tet_pos_sum>;
using TetPosMma = Geom<3, true,  true,  true,  s_tet_pos_mma>;
using TriGllMma = Geom<2, true,  false, true,  s_tri_gll_mma>;
using TriPosSum = Geom<2, true,  true,  false, s_tri_pos_sum>;
using TriPosMma = Geom<2, true,  true,  true,  s_tri_pos_mma>;

#define REGISTER(PK, BFI, GEOM) \
   BENCHMARK_TEMPLATE(Benchmark, \
      PK<BFI,\
        /* DIM*/ GEOM::dim, \
        /*VDIM*/ ((BFI) % 2 ? 1 : GEOM::dim), \
        /* QGL*/ ((BFI) <= 4), \
         GEOM::simplex, GEOM::pos, GEOM::mma>) \
   ->Name(std::string(#PK #BFI) + GEOM::suffix) \
   ->Apply(CustomArguments)->Unit(bm::kMillisecond)

// BP1: scalar CG with mass kernel, q=p+2
REGISTER(BP, 1, HexSum);
REGISTER(BP, 1, HexMma);
REGISTER(BP, 1, QuadSum);
REGISTER(BP, 1, QuadMma);
// REGISTER(BP, 1, TetGllMma);
REGISTER(BP, 1, TetPosSum);
REGISTER(BP, 1, TetPosMma);
// REGISTER(BP, 1, TriGllMma);
REGISTER(BP, 1, TriPosSum);
REGISTER(BP, 1, TriPosMma);

// BP2: vector CG with mass kernel, q=p+2
REGISTER(BP, 2, HexSum);
REGISTER(BP, 2, QuadSum);

// BP3: scalar CG with stiffness kernel, q=p+2
REGISTER(BP, 3, HexSum);
REGISTER(BP, 3, HexMma);
REGISTER(BP, 3, QuadSum);
REGISTER(BP, 3, QuadMma);
// REGISTER(BP, 3, TetGllMma);
REGISTER(BP, 3, TetPosSum);
REGISTER(BP, 3, TetPosMma);
// REGISTER(BP, 3, TriGllMma);
REGISTER(BP, 3, TriPosSum);
REGISTER(BP, 3, TriPosMma);

// BP4: vector CG with stiffness kernel, q=p+2
REGISTER(BP, 4, HexSum);
REGISTER(BP, 4, QuadSum);

// BP5: scalar CG with stiffness kernel, q=p+1
REGISTER(BP, 5, HexSum);
REGISTER(BP, 5, QuadSum);

// BP6: vector CG with stiffness kernel, q=p+1
REGISTER(BP, 6, HexSum);
REGISTER(BP, 6, QuadSum);

// BK1: scalar E-to-E mass kernel, q=p+2
REGISTER(BK, 1, HexSum);
REGISTER(BK, 1, QuadSum);

// BK2: vector E-to-E mass kernel, q=p+2
REGISTER(BK, 2, HexSum);
REGISTER(BK, 2, QuadSum);

// BK3: scalar E-to-E stiffness kernel, q=p+2
REGISTER(BK, 3, HexSum);
REGISTER(BK, 3, QuadSum);

// BK4: vector E-to-E stiffness kernel, q=p+2
REGISTER(BK, 4, HexSum);
REGISTER(BK, 4, QuadSum);

// BK5: scalar E-to-E stiffness kernel, q=p+1
REGISTER(BK, 5, HexSum);
REGISTER(BK, 5, QuadSum);

// BK6: vector E-to-E stiffness kernel, q=p+1
REGISTER(BK, 6, HexSum);
REGISTER(BK, 6, QuadSum);

/**
 * @brief CEED Bake-off Problems main entry point
 * Command line options:
 *    --benchmark_context=device=gpu
 *    --benchmark_context=cg=true|false  (default false)
 *    --benchmark_filter=BP1hex_sum
 *    --benchmark_filter=BP3hex_mma
 *    --benchmark_filter=BP1tet_gll_mma
 *    --benchmark_filter=BP1tri_pos_sum
 *    --benchmark_filter=BP1tri_pos_mma
 *    --benchmark_out_format=csv
 *    --benchmark_out=bp1.csv
 *
 * Names: {geom}[_{basis}]_{algo}
 *    hex_sum / quad_sum     — classic tensor sum-factorization
 *    hex_mma / quad_mma     — tensor sum-fact + small per-direction MMA
 *    tet_gll_mma / tri_gll_mma — simplex GLL nodal, dense MMA GEMM
 *    tet_pos_sum / tri_pos_sum — simplex Positive, Stroud sum-fact
 *    tet_pos_mma / tri_pos_mma — simplex Positive, dense MMA
 *
 * QGL (template): true for BP/BK 1–4 (GL q=p+2), false for 5–6 (GLL q=p+1).
 */
int main(int argc, char *argv[])
{
   ColorConsoleReporter CR;
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
      const auto ctx_cg = global_context->find("cg");
      if (ctx_cg != global_context->end())
      {
         mfem::out << ctx_cg->first << " : " << ctx_cg->second << std::endl;
         cg_verify = ParseBoolContext(ctx_cg->second);
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
