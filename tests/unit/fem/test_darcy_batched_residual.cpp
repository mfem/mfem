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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <memory>

using namespace mfem;

namespace darcy_batched_residual
{

/// Which conductivity shape LinearDiffusionFlux is built over.
enum class Kappa { Scalar, Vector, Matrix };

/** @brief A LinearDiffusionFlux over each of the three coefficient shapes its
    ComputeDualFlux() branches on.

    Worth covering all three: the batched route reads the law through
    ComputeDualFluxJacobian(), and the equivalence of that with
    ComputeDualFlux() is a separate claim per branch -- Set() for a scalar,
    RightScaling() for a vector, MultABt() for a matrix. The matrix one is the
    only branch where the dual flux couples directions, so it is the only one
    that can catch a transposed contraction. */
struct Law
{
   std::unique_ptr<Coefficient> c;
   std::unique_ptr<VectorCoefficient> vc;
   std::unique_ptr<MatrixCoefficient> mc;
   std::unique_ptr<MixedFluxFunction> fun;

   Law(Kappa k, int dim)
   {
      switch (k)
      {
         case Kappa::Scalar:
            c.reset(new FunctionCoefficient([](const Vector &X)
            {
               return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
            }));
            fun.reset(new LinearDiffusionFlux(dim, *c));
            break;
         case Kappa::Vector:
            vc.reset(new VectorFunctionCoefficient(dim,
                                                   [](const Vector &X, Vector &v)
            {
               v(0) = 0.9 + 0.2 * X(1);
               v(1) = 1.7 - 0.3 * X(0);
            }));
            fun.reset(new LinearDiffusionFlux(*vc));
            break;
         case Kappa::Matrix:
            mc.reset(new MatrixFunctionCoefficient(dim,
                                                   [](const Vector &X, DenseMatrix &m)
            {
               // Deliberately NOT symmetric: a symmetric conductivity cannot
               // tell K from K^T, and the contraction order is exactly what
               // the batched route has to get right.
               m(0,0) = 1.1 + 0.2*X(0);
               m(0,1) = 0.6 + 0.1*X(1);
               m(1,0) = -0.35;
               m(1,1) = 1.4 - 0.25*X(1);
            }));
            fun.reset(new LinearDiffusionFlux(*mc));
            break;
      }
   }
};

/// A field with structure rather than a constant, reproducibly.
void FillWavy(Vector &v, real_t shift)
{
   for (int i = 0; i < v.Size(); i++)
   {
      v(i) = std::sin(0.71 * i + shift) + 0.5 * std::cos(0.113 * i);
   }
}

} // namespace darcy_batched_residual

/** @brief The batched local residual is the integrator's own element vector,
    entry for entry.

    This is the ISOLATED comparison and the one that discriminates: it calls
    HDGMixedConductionResidualBatched() for the whole mesh and
    MixedConductionNLFIntegrator::AssembleElementVector() one element at a
    time, on the same field, and demands the same bits. Nothing else in the
    problem varies, unlike the in-situ case below where AssemblyMode::Batched
    switches several kernels at once.

    BITWISE, and the kernel is written to earn that rather than hoping for it:
    the accumulation is grouped `w * shape(i) * mF` in the integrator's own
    order, and the dual flux is contracted in the k-order the law's MultABt()
    uses. Deliberately regrouping it as `(w * mF) * shape(i)` fails this case.

    The potential field handed to the integrator is NONZERO on purpose. A
    LinearDiffusionFlux ignores its state -- that is exactly the property
    HDGMixedConductionResidualCanBatch() admits it for -- so a law that did
    not would show up here as a disagreement rather than as a silently
    different answer in a solve. */
TEST_CASE("The batched local residual reproduces the element integrator",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_residual;

   const int order = GENERATE(0, 1, 2);
   const auto etype = GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   const auto kappa = GENERATE(Kappa::Scalar, Kappa::Vector, Kappa::Matrix);
   CAPTURE(order, (int)etype, (int)kappa);

   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(3, 4, etype, false, 0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll);

   Law law(kappa, dim);
   MixedConductionNLFIntegrator integ(*law.fun);

   REQUIRE(HDGMixedConductionResidualCanBatch(Vh, &integ));

   const int NE = mesh.GetNE();
   const int ND = Vh.GetFE(0)->GetDof();
   const int N = ND * dim;

   Vector u_all(N * NE), ru_all;
   FillWavy(u_all, 0.3);
   HDGMixedConductionResidualBatched(Vh, &integ, u_all, ru_all);
   ru_all.HostRead();

   const int NDP = Wh.GetFE(0)->GetDof();
   Vector p_e(NDP);
   FillWavy(p_e, 1.9);

   Vector u_e(N), out_u, out_p;
   int checked = 0;
   for (int e = 0; e < NE; e++)
   {
      for (int i = 0; i < N; i++) { u_e(i) = u_all(N * e + i); }
      Array<const FiniteElement*> fe_arr({Vh.GetFE(e), Wh.GetFE(e)});
      Array<const Vector*> x_arr({&u_e, &p_e});
      Array<Vector*> y_arr({&out_u, &out_p});
      integ.AssembleElementVector(fe_arr, *mesh.GetElementTransformation(e),
                                  x_arr, y_arr);
      REQUIRE(out_u.Size() == N);
      for (int i = 0; i < N; i++)
      {
         REQUIRE(ru_all(N * e + i) == out_u(i));
         checked++;
      }
   }
   // A comparison of nothing passes; say how much was compared.
   REQUIRE(checked == N * NE);
   REQUIRE(ru_all.Normlinf() > 1e-3);
}

/** @brief The gate refuses what it cannot take, and each refusal is reached.

    A guard written from reading the code is not a guard until something
    arrives at it; three of the four conditions here have a live caller in
    miniapps/hdg/regress_test/ and the fourth is the H(div) flux space. */
TEST_CASE("The batched local residual refuses what it cannot take",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_residual;

   const int dim = 2, order = 1;
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   RT_FECollection rt_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Rh(&mesh, &rt_coll);

   SECTION("A state-dependent conductivity")
   {
      // `-p 8`: ikappa = 1/(k+T). The conductivity is a host std::function of
      // the CURRENT potential, so there is no per-point weight to compute
      // ahead of the kernel and no device form of the callback. 5 of the 88
      // hybridized references are this.
      auto ik = [](const Vector &, real_t T) -> real_t { return 1.0/(0.3+T); };
      auto dik = [](const Vector &, real_t T) -> real_t
      { return -1.0/((0.3+T)*(0.3+T)); };
      FunctionDiffusionFlux fun(dim, ik, dik);
      MixedConductionNLFIntegrator integ(fun);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Vh, &integ));
   }

   SECTION("An H(div) flux space")
   {
      // A vector-valued basis: CalcVShape() is per element and the dofs are
      // laid out per equation, not per (equation, direction).
      Law law(Kappa::Scalar, dim);
      MixedConductionNLFIntegrator integ(*law.fun);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Rh, &integ));
   }

   SECTION("A flux space whose vdim is not the space dimension")
   {
      FiniteElementSpace Sh(&mesh, &u_coll);   // vdim 1
      Law law(Kappa::Scalar, dim);
      MixedConductionNLFIntegrator integ(*law.fun);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Sh, &integ));
   }

   SECTION("An integrator that is not a MixedConductionNLFIntegrator")
   {
      SumBlockNLFIntegrator sum(false);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Vh, &sum));
   }
}

namespace darcy_batched_residual
{

struct ResidualOutcome
{
   BlockVector r;
   Vector r_tr;
   Vector Sv;
   bool taken = false;
   int integ_calls = 0;
};

/** @brief One NPC residual evaluation on the `-nld` shape: the flux law is a
    MixedConductionNLFIntegrator on the BlockNonlinearForm and the HDG
    stabilization is a plain HDGDiffusionIntegrator on the linear potential
    mass form.

    That is `convdiff -nld -hb -npc` and it is the configuration 15 of the 88
    hybridized regression references have -- established by printing which
    integrator pointers DarcyHybridization ends up holding for each reference,
    not by reading the miniapp. */
void NPCResidualOnce(DarcyHybridization::AssemblyMode am, int order,
                     ResidualOutcome &out)
{
   const int dim = 2, n = 4;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   FunctionCoefficient ikappa([](const Vector &X)
   {
      return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
   });
   LinearDiffusionFlux law(dim, ikappa);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   // The flux mass comes from the LAW and from nowhere else, which is what
   // makes m_nlfi the only element integrator on the local residual.
   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(law));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();
   out.taken = dh->CanBatchLocalResidual();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   // A state with structure. At x = 0 the flux row is exactly zero for a law
   // linear in the flux, so a zero state would compare nothing.
   FillWavy(x, 0.5);
   Vector x_tr(Mh.GetVSize());
   FillWavy(x_tr, 2.1);

   out.r.Update(darcy.GetOffsets());
   dh->NPCResidual(b, x, x_tr, out.r, out.r_tr);
   out.r.HostRead();
   out.r_tr.HostRead();
}

} // namespace darcy_batched_residual

/** @brief In situ: the NPC residual is the same with the kernel and without.

    Reachability is the point of this case, and it is what a batched route
    most often lacks -- this branch lost AssemblyMode::Batched's face kernel
    to a dead `dynamic_cast` for three commits with nothing noticing. The two
    REQUIREs on @a taken are therefore load-bearing: two fallbacks agree
    perfectly and test nothing.

    A TOLERANCE and not the bits, unlike the isolated case above, and the
    reason is a confound rather than the kernel: AssemblyMode::Batched also
    switches the face, mass and divergence assembly kernels on, and those
    accumulate point by point where the per-element route adds one element
    matrix. The isolated case is what pins the kernel itself. */
TEST_CASE("The batched local residual reaches an NPC caller",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_residual;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   ResidualOutcome ref, bat;
   NPCResidualOnce(AM::Serial, order, ref);
   NPCResidualOnce(AM::Batched, order, bat);

   REQUIRE_FALSE(ref.taken);
   REQUIRE(bat.taken);

   REQUIRE(ref.r.GetBlock(0).Normlinf() > 1e-3);
   REQUIRE(ref.r.GetBlock(1).Normlinf() > 1e-3);
   REQUIRE(ref.r_tr.Normlinf() > 1e-3);

   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      // Floored: an equality test between two routes must not compare
      // round-off relatively, which is a standing note on this branch.
      REQUIRE(d.Normlinf() <= 1e-11 * std::max(a.Normlinf(), 1e-30));
   };
   close(ref.r.GetBlock(0), bat.r.GetBlock(0));
   close(ref.r.GetBlock(1), bat.r.GetBlock(1));
   close(ref.r_tr, bat.r_tr);
}

namespace darcy_batched_residual
{

/** @brief One NPC residual on a problem whose mass terms sit either on the
    BILINEAR forms or on the NONLINEAR ones, the two being the same discrete
    problem by construction.

    `class BilinearFormIntegrator : public NonlinearFormIntegrator`, so
    installing one on a nonlinear form compiles, and that is what
    `convdiff -nl` does: a VectorMassIntegrator on the flux mass nonlinear
    form. DarcyForm::EnableHybridization() then wraps it in a
    SumNLFIntegrator -- unconditionally, even for a single integrator -- and
    the element loop calls BilinearFormIntegrator::AssembleElementVector(),
    which re-assembles a constant element matrix and multiplies, on every
    residual evaluation.

    TWO integrators on the potential mass, and one of them NON-SYMMETRIC.
    Both are deliberate: a single-member sum cannot see a routine that stops
    after the first member, and a symmetric block cannot see a transposed
    write. */
void NPCResidualMassSlots(bool nonlinear_slots, int order,
                          ResidualOutcome &out)
{
   const int dim = 2, n = 4;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   FunctionCoefficient ikappa([](const Vector &X)
   {
      return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
   });
   Vector bvec(dim);
   bvec(0) = 0.9;
   bvec(1) = -0.6;
   VectorConstantCoefficient bcoeff(bvec);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   if (nonlinear_slots)
   {
      darcy.GetFluxMassNonlinearForm()->AddDomainIntegrator(
         new VectorMassIntegrator(ikappa));
      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new MassIntegrator(one));
      Mnl_p->AddDomainIntegrator(new ConservativeConvectionIntegrator(bcoeff));
   }
   else
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(ikappa));
      BilinearForm *M_p = darcy.GetPotentialMassForm();
      M_p->AddDomainIntegrator(new MassIntegrator(one));
      M_p->AddDomainIntegrator(new ConservativeConvectionIntegrator(bcoeff));
   }

   // The face constraint, on the LINEAR potential mass form either way --
   // CanBatchLinearResidual() requires c_bfi_p, and this is also what keeps
   // Df_lin_data non-empty when the domain terms have moved off it.
   BilinearForm *M_p_face = darcy.GetPotentialMassForm();
   M_p_face->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p_face->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   FillWavy(x, 0.5);
   Vector x_tr(Mh.GetVSize());
   FillWavy(x_tr, 2.1);

   out.r.Update(darcy.GetOffsets());
   dh->NPCResidual(b, x, x_tr, out.r, out.r_tr);
   // AFTER the residual: the route builds its cache on first use, and asking
   // before would report the question rather than the answer.
   out.taken = dh->CanBatchLinearResidual();
   out.r.HostRead();
   out.r_tr.HostRead();

   // **And the GRADIENT on the same state, which is the half this case was
   // missing.** It compared residuals only, and passed for a whole session
   // while the nonlinear-slot gradient was DOUBLE COUNTED: ConstructGrad()'s
   // `if (m_nlfi_u)` branch carried no !ad_done guard, so the element loop
   // added the element flux mass on top of CopyLinearGradBlocks()'s copy of
   // the same matrix. The residual was bit-identical throughout -- it is
   // computed by a different route -- so no residual comparison of any
   // tightness could have caught it, while 20 of the 152 serial references
   // did. Taken here, before dh goes out of scope, because the returned
   // Operator is owned by it.
   Operator &S = dh->NPCGradient(x, x_tr);
   Vector v(x_tr.Size());
   FillWavy(v, 3.7);
   out.Sv.SetSize(x_tr.Size());
   S.Mult(v, out.Sv);
   out.Sv.HostRead();
}

} // namespace darcy_batched_residual

/** @brief A mass term on the NONLINEAR form is the same problem as one on the
    bilinear form, and the batched residual takes both.

    The REQUIRE on the nonlinear arm's @a taken is the whole reachability
    half, and it is what fails if the route goes back to refusing a nonlinear
    mass slot: measured over the unit suite, no fixture reached this before
    the case existed, and every `convdiff -nl` run did.

    The equality half pins the assembled copy. Discriminated rather than
    assumed: truncating the sum to its first member, and transposing each
    element matrix, each fail this case on the first comparison. */
TEST_CASE("A bilinear integrator on a nonlinear mass form is assembled once",
          "[DarcyHybridization][BatchedLinAlg][NPC]")
{
   using namespace darcy_batched_residual;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   ResidualOutcome lin, nlin;
   NPCResidualMassSlots(false, order, lin);
   NPCResidualMassSlots(true, order, nlin);

   REQUIRE(lin.taken);
   REQUIRE(nlin.taken);

   REQUIRE(lin.r.GetBlock(0).Normlinf() > 1e-3);
   REQUIRE(lin.r.GetBlock(1).Normlinf() > 1e-3);
   REQUIRE(lin.r_tr.Normlinf() > 1e-3);

   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      REQUIRE(d.Normlinf() <= 1e-11 * std::max(a.Normlinf(), 1e-30));
   };
   close(lin.r.GetBlock(0), nlin.r.GetBlock(0));
   close(lin.r.GetBlock(1), nlin.r.GetBlock(1));
   close(lin.r_tr, nlin.r_tr);

   // The gradient has to agree too, and this is the assertion that fails on
   // the double count -- by a factor near two on |S v|, not by round-off.
   REQUIRE(lin.Sv.Normlinf() > 1e-3);
   close(lin.Sv, nlin.Sv);
}
