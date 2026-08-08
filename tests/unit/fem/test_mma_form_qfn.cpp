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

#include "unit_tests.hpp"
#include "mfem.hpp"
#include "fem/integ/mma/form/form.hpp"
#include "fem/integ/mma/mode/common.hpp"
#include "fem/integ/mma/mass.hpp"
#include "fem/integ/mma/diffusion.hpp"
#include "fem/integ/mma/domain_lf.hpp"

using namespace mfem;
using namespace mfem::internal::mma;
using namespace mfem::internal::mma::form;
using mfem::future::tensor;

TEST_CASE("MMA form QFn tensor algebra", "[MMA][Form]")
{
   SECTION("Mass y = d * u")
   {
      Mass q;
      eval_t u(2.5);
      eval_t y;
      q(u, y, real_t(3.0));
      REQUIRE(real_t(y) == MFEM_Approx(7.5));
   }

   SECTION("IdentityLoad y = d")
   {
      IdentityLoad q;
      eval_t y;
      q(y, real_t(4.0));
      REQUIRE(real_t(y) == MFEM_Approx(4.0));
   }

   SECTION("Diffusion y = A * u (2D)")
   {
      Diffusion<2, true> q;
      grad_t<2> u;
      u[0] = 1.0;
      u[1] = 2.0;
      tensor<real_t, 2, 2> A{};
      A(0, 0) = 2.0;
      A(0, 1) = 0.0;
      A(1, 0) = 0.0;
      A(1, 1) = 3.0;
      grad_t<2> y;
      q(u, y, A);
      REQUIRE(y[0] == MFEM_Approx(2.0));
      REQUIRE(y[1] == MFEM_Approx(6.0));
   }
}

TEST_CASE("MMA form qfn_traits presets", "[MMA][Form]")
{
   SECTION("Mass")
   {
      using Tr = qfn_traits<Mass>;
      static_assert(Tr::load_x);
      static_assert(!Tr::trial_is_grad);
      static_assert(std::is_same_v<Tr::trial_kind, eval_t>);
      static_assert(std::is_same_v<Tr::test_kind, eval_t>);
      REQUIRE(Tr::u_planes(2) == 1);
   }

   SECTION("IdentityLoad")
   {
      using Tr = qfn_traits<IdentityLoad>;
      static_assert(!Tr::load_x);
      static_assert(std::is_same_v<Tr::trial_kind, none_t>);
      static_assert(std::is_same_v<Tr::test_kind, eval_t>);
      REQUIRE(Tr::u_planes(3) == 1);
   }

   SECTION("Diffusion<3>")
   {
      using Tr = qfn_traits<Diffusion<3, true>>;
      static_assert(Tr::load_x);
      static_assert(Tr::trial_is_grad);
      static_assert(Tr::spatial_dim == 3);
      REQUIRE(Tr::u_planes(3) == 3);
   }
}

TEST_CASE("MMA form Eval plan goldens", "[MMA][Form][Plan]")
{
   auto check_mass_plan = [](auto dim_c, auto d1d_c, auto qnd_c)
   {
      constexpr int DIM = decltype(dim_c)::value;
      constexpr int D1D = decltype(d1d_c)::value;
      constexpr int QND = decltype(qnd_c)::value;

      CAPTURE(DIM, D1D, QND);

      const SmemPlan p = MakeEvalPlan<DIM, D1D, QND>(true);

      constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
      constexpr int BASIS = SimplexNdof<DIM, D1D>();
      constexpr int MQ = SimplexMaxNq<DIM, QND>();
      constexpr int NB = MassLikeNB<DIM, D1D, QND>();
      constexpr int X_LD = PadLdBank<MAP>(BASIS);
      constexpr int U_LD = PadLdBank<MAP>(MQ);

      REQUIRE(p.nb == NB);
      REQUIRE(p.x_ld == X_LD);
      REQUIRE(p.u_ld == U_LD);
      REQUIRE(p.n_u_planes == 1);
      REQUIRE(p.load_x == true);
      REQUIRE(p.use_q_tile == false);
      REQUIRE(p.tq == MQ);
      REQUIRE(p.smem_bytes == int(sizeof(real_t)) * (X_LD + U_LD) * NB);
      REQUIRE(p.nthreads == LaunchNthreads<QND>(MQ, BASIS));

      // From Mass traits
      const SmemPlan p2 = MakeDevicePlan<Mass, DIM, D1D, QND>();
      REQUIRE(p2.nb == p.nb);
      REQUIRE(p2.smem_bytes == p.smem_bytes);

      // LF: no X in smem budget of plan field (load_x false)
      const SmemPlan plf = MakeEvalPlan<DIM, D1D, QND>(false);
      REQUIRE(plf.nb == NB);
      REQUIRE(plf.load_x == false);
      REQUIRE(plf.smem_bytes == int(sizeof(real_t)) * U_LD * NB);
   };

   // Representative registered mass rows (tri + tet)
   check_mass_plan(std::integral_constant<int, 2> {},
                   std::integral_constant<int, 2> {},
                   std::integral_constant<int, 3> {});
   check_mass_plan(std::integral_constant<int, 2> {},
                   std::integral_constant<int, 4> {},
                   std::integral_constant<int, 16> {});
   check_mass_plan(std::integral_constant<int, 3> {},
                   std::integral_constant<int, 2> {},
                   std::integral_constant<int, 4> {});
   check_mass_plan(std::integral_constant<int, 3> {},
                   std::integral_constant<int, 4> {},
                   std::integral_constant<int, 20> {});
}

TEST_CASE("MMA form Eval plan runtime", "[MMA][Form][Plan]")
{
   const int ndof = 10;
   const int nq = 16;
   const SmemPlan p = MakeEvalPlanRuntime(ndof, nq, true);
   REQUIRE(p.nb == MassLikeNBRuntime(ndof, nq));
   REQUIRE(p.x_ld == PadLdBankRuntime(ndof));
   REQUIRE(p.u_ld == PadLdBankRuntime(nq));
   REQUIRE(p.nthreads == LaunchNthreads(nq, ndof));
}

TEST_CASE("MMA form host multi-RHS gate", "[MMA][Form]")
{
#ifdef MFEM_USE_LAPACK
   // Size-only gate lives in mma::lapack (no thin form wrapper).
   (void)lapack::PreferMultiRhs(16, 10, 100);
   (void)lapack::NB(16, 10);
   REQUIRE_FALSE(lapack::PreferMultiRhs(2, 2, 1));
#else
   SUCCEED("host multi-RHS gate requires MFEM_USE_LAPACK");
#endif
}


namespace
{

/** Independent serial reference: Y += P^T (D ⊙ (P X)). */
void MassRef(int NE, int nq, int ndof, const real_t *P, const real_t *D,
             const real_t *X, real_t *Y)
{
   for (int e = 0; e < NE; ++e)
   {
      for (int i = 0; i < ndof; ++i)
      {
         real_t yi = 0.0;
         for (int q = 0; q < nq; ++q)
         {
            real_t u = 0.0;
            for (int j = 0; j < ndof; ++j)
            {
               u += P[q + nq * j] * X[j + ndof * e];
            }
            yi += P[q + nq * i] * (D[q + nq * e] * u);
         }
         Y[i + ndof * e] += yi;
      }
   }
}

} // namespace

TEST_CASE("MMA form pipeline Eval Apply runtime vs ref",
          "[MMA][Form][Pipeline]")
{
   constexpr int nq = 3;
   constexpr int ndof = 3;
   constexpr int NE = 4;

   Array<real_t> P(nq * ndof);
   Vector D(nq * NE), X(ndof * NE), Y_ref(ndof * NE), Y_pipe(ndof * NE);
   for (int i = 0; i < ndof; ++i)
      for (int q = 0; q < nq; ++q)
      {
         P[q + nq * i] = real_t(1) + real_t(0.1) * q + real_t(0.01) * i;
      }
   for (int e = 0; e < NE; ++e)
   {
      for (int q = 0; q < nq; ++q)
      {
         D(q + nq * e) = real_t(0.5) + real_t(0.1) * e + real_t(0.01) * q;
      }
      for (int i = 0; i < ndof; ++i)
      {
         X(i + ndof * e) = real_t(1) + real_t(0.2) * i + real_t(0.05) * e;
      }
   }
   Y_ref = 0.0;
   Y_pipe = 0.0;

   MassRef(NE, nq, ndof, P.GetData(), D.GetData(), X.GetData(), Y_ref.GetData());
   form::ApplySimplex<Mass, 2>(NE, P, D, X, Y_pipe);

   for (int i = 0; i < ndof * NE; ++i)
   {
      REQUIRE(Y_pipe(i) == MFEM_Approx(Y_ref(i)));
   }
}

TEST_CASE("MMA form pipeline Eval Apply specialized vs ref",
          "[MMA][Form][Pipeline]")
{
   constexpr int DIM = 2, D1D = 2, QND = 3;
   constexpr int ndof = SimplexNdof<DIM, D1D>();
   constexpr int nq = SimplexMaxNq<DIM, QND>();
   constexpr int NE = 2;

   Array<real_t> P(nq * ndof);
   Vector D(nq * NE), X(ndof * NE), Y_ref(ndof * NE), Y_pipe(ndof * NE);
   for (int i = 0; i < nq * ndof; ++i) { P[i] = real_t(0.1) * (i % 7 + 1); }
   for (int i = 0; i < nq * NE; ++i) { D(i) = real_t(0.2) * (i % 5 + 1); }
   for (int i = 0; i < ndof * NE; ++i) { X(i) = real_t(0.3) * (i % 4 + 1); }
   Y_ref = 0.0;
   Y_pipe = 0.0;

   MassRef(NE, nq, ndof, P.GetData(), D.GetData(), X.GetData(), Y_ref.GetData());
   form::ApplySimplex<Mass, DIM, D1D, QND>(NE, P, D, X, Y_pipe);

   for (int i = 0; i < ndof * NE; ++i)
   {
      REQUIRE(Y_pipe(i) == MFEM_Approx(Y_ref(i)));
   }
}

TEST_CASE("MMA form ApplyLF IdentityLoad vs ref", "[MMA][Form][Pipeline]")
{
   constexpr int nq = 3;
   constexpr int ndof = 3;
   constexpr int NE = 4;
   constexpr int vdim = 1;
   constexpr int vc = 0;

   Array<real_t> P(nq * ndof);
   Vector D(nq * NE), Y_ref(ndof * NE), Y_pipe(ndof * NE);
   for (int i = 0; i < ndof; ++i)
      for (int q = 0; q < nq; ++q)
      {
         P[q + nq * i] = real_t(1) + real_t(0.1) * q + real_t(0.01) * i;
      }
   for (int e = 0; e < NE; ++e)
      for (int q = 0; q < nq; ++q)
      {
         D(q + nq * e) = real_t(0.5) + real_t(0.1) * e + real_t(0.01) * q;
      }
   Y_ref = 0.0;
   Y_pipe = 0.0;

   // Y += P^T D
   for (int e = 0; e < NE; ++e)
      for (int i = 0; i < ndof; ++i)
      {
         real_t yi = 0.0;
         for (int q = 0; q < nq; ++q)
         {
            yi += P[q + nq * i] * D(q + nq * e);
         }
         Y_ref(i + ndof * e) += yi;
      }

   form::ApplyLF<IdentityLoad, 2>(NE, P, D, Y_pipe.GetData(), vdim, vc);

   for (int i = 0; i < ndof * NE; ++i)
   {
      REQUIRE(Y_pipe(i) == MFEM_Approx(Y_ref(i)));
   }
}

TEST_CASE("MMA form Grad plan goldens", "[MMA][Form][Plan]")
{
   // 2D full-NQ (no Q-tile)
   {
      constexpr int DIM = 2, D1D = 2, QND = 3;
      const SmemPlan p = MakeGradPlan<DIM, D1D, QND>();
      constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
      constexpr int BASIS = SimplexNdof<DIM, D1D>();
      constexpr int MQ = SimplexMaxNq<DIM, QND>();
      REQUIRE(p.nb == BatchNB<DIM, D1D, QND>());
      REQUIRE(p.x_ld == PadLdBank<MAP>(BASIS));
      REQUIRE(p.u_ld == PadLdBank<MAP>(MQ));
      REQUIRE(p.n_u_planes == DIM);
      REQUIRE(p.use_q_tile == BatchUseQTile<DIM, D1D, QND>());
      REQUIRE(p.smem_bytes ==
              int(sizeof(real_t)) * (p.x_ld + DIM * p.u_ld) * p.nb);
   }
   // Traits-driven plan for Diffusion
   {
      constexpr int DIM = 2, D1D = 3, QND = 6;
      const SmemPlan p =
         MakeDevicePlan<Diffusion<DIM, true>, DIM, D1D, QND>();
      REQUIRE(p.nb == BatchNB<DIM, D1D, QND>());
      REQUIRE(p.n_u_planes == 2);
   }
}

TEST_CASE("MMA form Grad Apply dense vs ref", "[MMA][Form][Pipeline]")
{
   constexpr int DIM = 2;
   constexpr int nq = 3;
   constexpr int ndof = 3;
   constexpr int NE = 2;
   constexpr int PA = 3; // SYM 2D

   Array<real_t> G(nq * ndof * DIM);
   Vector D(nq * PA * NE), X(ndof * NE), Y_ref(ndof * NE), Y_pipe(ndof * NE);
   for (int i = 0; i < G.Size(); ++i) { G[i] = real_t(0.1) * (i % 5 + 1); }
   for (int i = 0; i < D.Size(); ++i) { D(i) = real_t(0.2) * (i % 3 + 1); }
   for (int i = 0; i < X.Size(); ++i) { X(i) = real_t(0.3) * (i % 4 + 1); }
   Y_ref = 0.0;
   Y_pipe = 0.0;

   // Independent ref: U = G X, V = A U (SYM), Y += G^T V
   for (int e = 0; e < NE; ++e)
   {
      real_t U[DIM * nq];
      for (int d = 0; d < DIM; ++d)
         for (int q = 0; q < nq; ++q)
         {
            real_t s = 0.0;
            for (int i = 0; i < ndof; ++i)
            {
               s += G[q + nq * (i + ndof * d)] * X(i + ndof * e);
            }
            U[d * nq + q] = s;
         }
      for (int q = 0; q < nq; ++q)
      {
         const real_t O11 = D(q + nq * (0 + PA * e));
         const real_t O21 = D(q + nq * (1 + PA * e));
         const real_t O22 = D(q + nq * (2 + PA * e));
         const real_t u1 = U[0 * nq + q], u2 = U[1 * nq + q];
         U[0 * nq + q] = O11 * u1 + O21 * u2;
         U[1 * nq + q] = O21 * u1 + O22 * u2;
      }
      for (int i = 0; i < ndof; ++i)
      {
         real_t s = 0.0;
         for (int d = 0; d < DIM; ++d)
            for (int q = 0; q < nq; ++q)
            {
               s += G[q + nq * (i + ndof * d)] * U[d * nq + q];
            }
         Y_ref(i + ndof * e) += s;
      }
   }

   form::ApplySimplex<Diffusion<2, true>, 2>(NE, G, D, X, Y_pipe);

   for (int i = 0; i < ndof * NE; ++i)
   {
      REQUIRE(Y_pipe(i) == MFEM_Approx(Y_ref(i)));
   }
}

TEST_CASE("MMA form dump disabled by default", "[MMA][Form][Dump]")
{
   // Without MFEM_MMA_FORM_DUMP, dump is off (or was cached from env).
   // Smoke: Apply still works; FormDumpEnabled is callable.
   (void)FormDumpEnabled();

   constexpr int nq = 3, ndof = 3, NE = 1;
   Array<real_t> P(nq * ndof);
   Vector D(nq * NE), X(ndof * NE), Y(ndof * NE);
   P = 1.0;
   D = 1.0;
   X = 1.0;
   Y = 0.0;
   form::ApplySimplex<Mass, 2>(NE, P, D, X, Y);
   // All-ones P,D,X: U_q = ndof, scaled by D → ndof; Y_i = nq*ndof
   REQUIRE(Y(0) == MFEM_Approx(real_t(nq * ndof)));
}

TEST_CASE("MMA form DumpFormApply no-op when disabled", "[MMA][Form][Dump]")
{
   // Explicit no-op path: helpers must not abort when dump is off.
   DumpFormApply<Mass, 2, 2, 3>("test", 1, 3, 3);
   DumpFormApplyRuntime<Mass, 2>("test", 1, 3, 3);
   DumpFormApply<IdentityLoad, 2, 2, 3>("test-lf", 1, 3, 3);
   DumpFormApply<Diffusion<2, true>, 2, 2, 3>("test-grad", 1, 3, 3);
   REQUIRE(true);
}

// ---------------------------------------------------------------------------
// PR8: custom QFn authoring example (see form/README.md)
// ---------------------------------------------------------------------------

namespace
{

/** Custom mass-like QFn: y = d*d * u  (not a built-in preset). */
struct DensitySquaredMass
{
   MFEM_HOST_DEVICE void operator()(const eval_t &u, eval_t &y, real_t d) const
   {
      y = (d * d) * u;
   }
};

} // namespace

namespace mfem::internal::mma::form
{
// Wire traits via helper — all a custom author needs beyond the QFn body.
template <>
struct qfn_traits<DensitySquaredMass> : EvalEvalQFnTraits {};
}

TEST_CASE("MMA form custom QFn DensitySquaredMass", "[MMA][Form][Author]")
{
   using form::ApplySimplex;
   using form::qfn_traits;

   SECTION("traits inherit EvalEval")
   {
      using Tr = qfn_traits<DensitySquaredMass>;
      static_assert(Tr::load_x);
      static_assert(!Tr::trial_is_grad);
      REQUIRE(Tr::u_planes(2) == 1);
   }

   SECTION("algebra y = d*d * u")
   {
      DensitySquaredMass q;
      eval_t u(2.0), y;
      q(u, y, real_t(3.0));
      REQUIRE(real_t(y) == MFEM_Approx(18.0));
   }

   SECTION("pipeline ApplySimplex vs independent ref")
   {
      constexpr int nq = 3;
      constexpr int ndof = 3;
      constexpr int NE = 2;

      Array<real_t> P(nq * ndof);
      Vector D(nq * NE), X(ndof * NE), Y_ref(ndof * NE), Y_pipe(ndof * NE);
      for (int i = 0; i < ndof; ++i)
         for (int q = 0; q < nq; ++q)
         {
            P[q + nq * i] = real_t(1) + real_t(0.1) * q + real_t(0.01) * i;
         }
      for (int e = 0; e < NE; ++e)
      {
         for (int q = 0; q < nq; ++q)
         {
            D(q + nq * e) = real_t(0.5) + real_t(0.1) * e + real_t(0.01) * q;
         }
         for (int i = 0; i < ndof; ++i)
         {
            X(i + ndof * e) = real_t(1) + real_t(0.2) * i + real_t(0.05) * e;
         }
      }
      Y_ref = 0.0;
      Y_pipe = 0.0;

      // Ref: Y += P^T ( D² ⊙ (P X) )
      for (int e = 0; e < NE; ++e)
      {
         for (int i = 0; i < ndof; ++i)
         {
            real_t yi = 0.0;
            for (int q = 0; q < nq; ++q)
            {
               real_t u = 0.0;
               for (int j = 0; j < ndof; ++j)
               {
                  u += P[q + nq * j] * X(j + ndof * e);
               }
               const real_t d = D(q + nq * e);
               yi += P[q + nq * i] * (d * d * u);
            }
            Y_ref(i + ndof * e) += yi;
         }
      }

      ApplySimplex<DensitySquaredMass, 2>(NE, P, D, X, Y_pipe);

      for (int i = 0; i < ndof * NE; ++i)
      {
         REQUIRE(Y_pipe(i) == MFEM_Approx(Y_ref(i)));
      }
   }
}
