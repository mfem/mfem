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

#include <cmath>
#include <vector>

using namespace mfem;

namespace darcy_reaction
{

/// F(u) = a + c u, componentwise. Interpolation is EXACT on this one.
struct Affine : public NodalReactionFunction
{
   int neq;
   real_t a, c;
   Affine(int neq_, real_t a_, real_t c_) : neq(neq_), a(a_), c(c_) { }
   int NumEquations() const override { return neq; }
   void Eval(const Vector &, const Vector &u, Vector &F) const override
   {
      F.SetSize(neq);
      for (int e = 0; e < neq; e++) { F(e) = a + c * u(e); }
   }
   void EvalJacobian(const Vector &, const Vector &,
                     DenseMatrix &J) const override
   {
      J.SetSize(neq);
      J = 0.0;
      for (int e = 0; e < neq; e++) { J(e, e) = c; }
   }
};

/// CCSZ Example 4.1's u^3 - u, plus a term coupling the equations so that a
/// wrong off-diagonal Jacobian block cannot hide at neq == 1.
struct Cubic : public NodalReactionFunction
{
   int neq;
   Cubic(int neq_) : neq(neq_) { }
   int NumEquations() const override { return neq; }
   void Eval(const Vector &, const Vector &u, Vector &F) const override
   {
      F.SetSize(neq);
      for (int e = 0; e < neq; e++)
      {
         const int g = (e + 1) % neq;
         F(e) = u(e) * u(e) * u(e) - u(e) + 0.5 * u(g) * u(g);
      }
   }
   void EvalJacobian(const Vector &, const Vector &u,
                     DenseMatrix &J) const override
   {
      J.SetSize(neq);
      J = 0.0;
      for (int e = 0; e < neq; e++)
      {
         const int g = (e + 1) % neq;
         J(e, e) += 3.0 * u(e) * u(e) - 1.0;
         J(e, g) += u(g);
      }
   }
};

real_t Pex(const Vector &x)
{
   real_t r = 1.0;
   for (int i = 0; i < x.Size(); i++) { r *= sin(M_PI * x(i)); }
   return r;
}

void GradPex(const Vector &x, Vector &g)
{
   const int d = x.Size();
   g.SetSize(d);
   for (int i = 0; i < d; i++)
   {
      real_t r = M_PI * cos(M_PI * x(i));
      for (int j = 0; j < d; j++) { if (j != i) { r *= sin(M_PI * x(j)); } }
      g(i) = r;
   }
}



/// tau = tau0, ignoring the integrator's own 1/h: paper I's stabilization.
/// The default carries 1/h, which is a different method -- and measurably so.
struct ConstStab : public HDGStabilization
{
   real_t tau0;
   ConstStab(real_t t) : tau0(t) { }
   real_t Eval(real_t, real_t, real_t, real_t,
               ElementTransformation &) const override { return tau0; }
};

/** @brief The quadrature order the CONTROL is run at.

    Two constraints, and both were measured rather than guessed. It must be
    high enough that F(u*) phi -- degree 3(k+1)+k = 4k+3 for a cubic -- is
    integrated exactly, and that the rule's points are not the enriched
    space's nodes, which on a Gauss-Legendre box they otherwise are. And it
    must stay INSIDE MFEM's tabulated simplex rules: at degree 30 a triangle
    rule jumps from 126 points to 816 and silently loses accuracy, which
    showed up here as the control's own gradient disagreeing with a finite
    difference of its own residual by 1e-7 where every order from 8 to 25
    gives 4e-12. */
int kControlRule(int order) { return 4 * order + 6; }

/// Everything the two integrators need, and the comparison between them.
struct Fixture
{
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<L2_FECollection> q_coll, p_coll, s_coll;
   std::unique_ptr<FiniteElementSpace> fes_q, fes_p, fes_s;
   std::unique_ptr<GridFunction> q, p;
   std::unique_ptr<HDGPostprocessBlocks> blocks;
   ConstantCoefficient one{1.0};
   std::unique_ptr<RatioCoefficient> ik;
   int neq;

   Fixture(int n, int order, int dim, int neq_, bool simplex, int basis)
      : neq(neq_)
   {
      mesh.reset(new Mesh(
                    (dim == 3)
                    ? Mesh::MakeCartesian3D(n, n, n, simplex
                                            ? Element::TETRAHEDRON
                                            : Element::HEXAHEDRON)
                    : Mesh::MakeCartesian2D(n, n, simplex
                                            ? Element::TRIANGLE
                                            : Element::QUADRILATERAL)));
      q_coll.reset(new L2_FECollection(order, dim));
      p_coll.reset(new L2_FECollection(order, dim));
      s_coll.reset(new L2_FECollection(order + 1, dim, basis));
      fes_q.reset(new FiniteElementSpace(mesh.get(), q_coll.get(), neq * dim,
                                         Ordering::byNODES));
      fes_p.reset(new FiniteElementSpace(mesh.get(), p_coll.get(), neq,
                                         Ordering::byNODES));
      fes_s.reset(new FiniteElementSpace(mesh.get(), s_coll.get(), neq,
                                         Ordering::byNODES));

      // A smooth state. Nothing here has to solve anything: the reaction term
      // is a local map of whatever fields it is handed.
      VectorFunctionCoefficient qc(neq * dim, [neq_, dim](const Vector &x,
                                                          Vector &v)
      {
         Vector g;
         GradPex(x, g);
         for (int e = 0; e < neq_; e++)
            for (int d = 0; d < dim; d++)
            { v(e * dim + d) = -(1.0 + 0.2 * e) * g(d); }
      });
      q.reset(new GridFunction(fes_q.get()));
      q->ProjectCoefficient(qc);

      p.reset(new GridFunction(fes_p.get()));
      MassIntegrator mi;
      DenseMatrix M;
      DenseMatrixInverse Mi;
      Array<int> vd;
      for (int z = 0; z < mesh->GetNE(); z++)
      {
         const FiniteElement *fe = fes_p->GetFE(z);
         ElementTransformation *T = mesh->GetElementTransformation(z);
         mi.AssembleElementMatrix(*fe, *T, M);
         Mi.Factor(M);
         fes_p->GetElementVDofs(z, vd);
         const int nd = fe->GetDof();
         for (int e = 0; e < neq; e++)
         {
            FunctionCoefficient pc([e](const Vector &x)
            { return (1.0 + 0.2 * e) * Pex(x); });
            DomainLFIntegrator lf(pc);
            lf.SetIntRule(&IntRules.Get(fe->GetGeomType(),
                                        2 * fe->GetOrder() + 6));
            Vector rhs, sol(nd);
            lf.AssembleRHSElementVect(*fe, *T, rhs);
            Mi.Mult(rhs, sol);
            for (int i = 0; i < nd; i++) { (*p)(vd[e * nd + i]) = sol(i); }
         }
      }

      ik.reset(new RatioCoefficient(1.0, one));
      blocks.reset(new HDGPostprocessBlocks(*fes_q, *fes_p, *fes_s));
      blocks->SetDiffusionInverse(*ik);
      blocks->Assemble();
   }
};

struct Gaps
{
   real_t res = 0.0, res_norm = 0.0;
   real_t g10 = 0.0, g11 = 0.0, s10 = 0.0, s11 = 0.0;
   real_t fd_i = 0.0, fd_q = 0.0, fd_scale = 0.0;
};

/** @brief Run both integrators over every element and collect the
    disagreements. @a quad_order is the rule the QUADRATURE control uses;
    negative takes the default, which on a tensor-product element collocates
    -- see the test case that pins that. */
Gaps Compare(Fixture &X, const NodalReactionFunction &F, int quad_order,
             bool do_fd = false)
{
   HDGInterpolatoryReactionIntegrator I(F, *X.blocks);
   HDGQuadratureReactionIntegrator Q(F, *X.blocks);
   if (quad_order >= 0) { Q.SetIntegrationOrder(quad_order); }
   I.Assemble();
   Q.Assemble();

   Gaps G;
   IsoparametricTransformation Tr;
   Array<int> vq, vp;
   Vector lq, lp;
   MassIntegrator mi;
   DenseMatrix M;
   DenseMatrixInverse Mi;
   const int neq = X.neq;

   real_t gap2 = 0.0, nrm2 = 0.0;
   for (int z = 0; z < X.mesh->GetNE(); z++)
   {
      const FiniteElement *fe_q = X.fes_q->GetFE(z);
      const FiniteElement *fe_p = X.fes_p->GetFE(z);
      X.mesh->GetElementTransformation(z, &Tr);
      X.fes_q->GetElementVDofs(z, vq);
      X.q->GetSubVector(vq, lq);
      X.fes_p->GetElementVDofs(z, vp);
      X.p->GetSubVector(vp, lp);

      Array<const FiniteElement*> el(2);
      el[0] = fe_q;
      el[1] = fe_p;
      Array<const Vector*> elfun(2);
      elfun[0] = &lq;
      elfun[1] = &lp;
      Vector vI0, vI1, vQ0, vQ1;
      Array<Vector*> evI(2), evQ(2);
      evI[0] = &vI0;
      evI[1] = &vI1;
      evQ[0] = &vQ0;
      evQ[1] = &vQ1;
      I.AssembleElementVector(el, Tr, elfun, evI);
      Q.AssembleElementVector(el, Tr, elfun, evQ);

      // This term writes no flux row.
      REQUIRE(vI0.Size() == 0);
      REQUIRE(vQ0.Size() == 0);

      // r^T M^{-1} r is the squared L2 norm of the W_h projection of the
      // pointwise difference, so the comparison is mesh normalised.
      mi.AssembleElementMatrix(*fe_p, Tr, M);
      Mi.Factor(M);
      const int nd = fe_p->GetDof();
      for (int e = 0; e < neq; e++)
      {
         Vector d(nd), t(nd), rq(nd);
         for (int i = 0; i < nd; i++)
         {
            d(i) = vI1(e * nd + i) - vQ1(e * nd + i);
            rq(i) = vQ1(e * nd + i);
         }
         Mi.Mult(d, t);
         gap2 += d * t;
         Mi.Mult(rq, t);
         nrm2 += rq * t;
      }

      DenseMatrix mI00, mI01, mI10, mI11, mQ00, mQ01, mQ10, mQ11;
      Array2D<DenseMatrix*> emI(2, 2), emQ(2, 2);
      emI(0, 0) = &mI00;
      emI(0, 1) = &mI01;
      emI(1, 0) = &mI10;
      emI(1, 1) = &mI11;
      emQ(0, 0) = &mQ00;
      emQ(0, 1) = &mQ01;
      emQ(1, 0) = &mQ10;
      emQ(1, 1) = &mQ11;
      I.AssembleElementGrad(el, Tr, elfun, emI);
      Q.AssembleElementGrad(el, Tr, elfun, emQ);

      DenseMatrix D(mI10);
      D -= mQ10;
      G.g10 = std::max(G.g10, D.MaxMaxNorm());
      G.s10 = std::max(G.s10, mQ10.MaxMaxNorm());
      DenseMatrix E(mI11);
      E -= mQ11;
      G.g11 = std::max(G.g11, E.MaxMaxNorm());
      G.s11 = std::max(G.s11, mQ11.MaxMaxNorm());

      if (do_fd)
      {
         // Each gradient against a finite difference of its OWN residual, so
         // that "the two agree" cannot mean "both are wrong the same way".
         const real_t h = 1e-6;
         for (int j = 0; j < lp.Size(); j++)
         {
            Vector lpp(lp), lpm(lp);
            lpp(j) += h;
            lpm(j) -= h;
            Array<const Vector*> fp(2), fm(2);
            fp[0] = &lq;
            fp[1] = &lpp;
            fm[0] = &lq;
            fm[1] = &lpm;
            Vector a0, a1, b0, b1;
            Array<Vector*> ea(2), eb(2);
            ea[0] = &a0;
            ea[1] = &a1;
            eb[0] = &b0;
            eb[1] = &b1;
            I.AssembleElementVector(el, Tr, fp, ea);
            I.AssembleElementVector(el, Tr, fm, eb);
            for (int i = 0; i < a1.Size(); i++)
            {
               G.fd_i = std::max(G.fd_i,
                                 std::abs((a1(i) - b1(i)) / (2 * h)
                                          - mI11(i, j)));
            }
            Q.AssembleElementVector(el, Tr, fp, ea);
            Q.AssembleElementVector(el, Tr, fm, eb);
            for (int i = 0; i < a1.Size(); i++)
            {
               G.fd_q = std::max(G.fd_q,
                                 std::abs((a1(i) - b1(i)) / (2 * h)
                                          - mQ11(i, j)));
            }
         }
         G.fd_scale = std::max(G.fd_scale, mQ11.MaxMaxNorm());
      }
   }
   G.res = std::sqrt(gap2);
   G.res_norm = std::sqrt(nrm2);
   return G;
}

} // namespace darcy_reaction

TEST_CASE("An interpolatory reaction term is exact on an affine law",
          "[DarcyForm][Postprocess][Reaction]")
{
   // CCSZ-I replaces (F(u*), v) by (I_h F(u*), v). I_h reproduces Z_h, so the
   // two coincide exactly when F(u*) lies in Z_h -- which for a + c u with a
   // and c CONSTANT it does, and for nothing else does. That is the null test,
   // and it is taken on the gradient as well as the residual because a
   // residual comparison cannot catch a wrong Jacobian; this branch has paid
   // for that twice.
   //
   // The control's quadrature order is raised DELIBERATELY. At the default
   // rule the comparison is vacuous on a tensor-product element -- see
   // "collocates" below -- and a null test that cannot fail is worth nothing.
   using namespace darcy_reaction;

   const int order = GENERATE(1, 2);
   const int neq = GENERATE(1, 2);
   const bool simplex = GENERATE(false, true);
   CAPTURE(order, neq, simplex);

   Fixture X(4, order, 2, neq, simplex, BasisType::GaussLegendre);
   Affine F(neq, 0.7, -1.3);
   const Gaps G = Compare(X, F, kControlRule(order), true);

   CAPTURE(G.res, G.res_norm, G.g10, G.s10, G.g11, G.s11);
   REQUIRE(G.res_norm > 1e-3);        // there is something to disagree about
   REQUIRE(G.s10 > 1e-6);
   REQUIRE(G.s11 > 1e-6);
   REQUIRE(G.res / G.res_norm < 1e-12);
   REQUIRE(G.g10 / G.s10 < 1e-12);
   REQUIRE(G.g11 / G.s11 < 1e-12);

   // And neither gradient is merely consistent with the other.
   CAPTURE(G.fd_i, G.fd_q, G.fd_scale);
   REQUIRE(G.fd_i / G.fd_scale < 1e-7);
   REQUIRE(G.fd_q / G.fd_scale < 1e-7);
}

TEST_CASE("An interpolatory reaction term differs on a nonlinear law",
          "[DarcyForm][Postprocess][Reaction]")
{
   // The discriminating half, without which the null test above would pass
   // for an integrator that returned zero for every input. Interpolating u^3
   // is not integrating it, and the gap must be well clear of round-off.
   using namespace darcy_reaction;

   const int order = GENERATE(1, 2);
   const int neq = GENERATE(1, 2);
   const bool simplex = GENERATE(false, true);
   CAPTURE(order, neq, simplex);

   Fixture X(4, order, 2, neq, simplex, BasisType::GaussLegendre);
   Cubic F(neq);
   const Gaps G = Compare(X, F, kControlRule(order), true);

   CAPTURE(G.res, G.res_norm, G.g10, G.s10, G.g11, G.s11);
   REQUIRE(G.res / G.res_norm > 1e-6);
   REQUIRE(G.g10 / G.s10 > 1e-6);

   // The (1,1) block is deliberately NOT asserted to differ, and the reason
   // is a property of the method rather than slack in the test. B12 is rank
   // one and its column is the constant function -- a move of the potential
   // reaches u* only as a uniform shift -- so this block probes
   // <I_h J - J, phi> in the single constant direction. At k = 1 on a
   // tensor-product element the interpolation error of J is the degree-3
   // Legendre polynomial times a linear factor, which is orthogonal to phi in
   // Q1, and the gap is EXACTLY zero: measured 4.8e-16 against 2.4e-06 at
   // k = 2 and 8.9e-04 on a triangle at k = 1. Requiring it to differ would
   // fail on one configuration for a correct implementation.

   // Both gradients are still each other's independent check against a
   // finite difference of their own residual.
   CAPTURE(G.fd_i, G.fd_q, G.fd_scale);
   REQUIRE(G.fd_i / G.fd_scale < 1e-7);
   REQUIRE(G.fd_q / G.fd_scale < 1e-7);
}

TEST_CASE("Interpolating collocates with quadrature on a Gauss-Legendre box",
          "[DarcyForm][Postprocess][Reaction]")
{
   // A property of the discretisation that is easy to mistake for a passing
   // test, so it is pinned deliberately rather than left to be rediscovered.
   //
   // MFEM's L2_FECollection is nodal at the GAUSS-LEGENDRE points by default.
   // On a tensor-product element the enriched space's k+2 points per dimension
   // are therefore exactly the points of the rule the postprocessing already
   // uses, and A9 built with that rule gives (chi_j, phi_i) = w_j phi_i(x_j).
   // The interpolatory term is then the quadrature term IDENTICALLY, for any
   // F whatsoever -- interpolation has degenerated to collocation.
   //
   // So a comparison at the default rule tests nothing, and the two cases
   // above raise it. Neither a simplex (where the node and point counts do
   // not even match) nor a Gauss-Lobatto basis has this coincidence.
   using namespace darcy_reaction;

   const int order = 2, neq = 1;
   Cubic F(neq);

   SECTION("a box with the default basis and rule: identical")
   {
      Fixture X(4, order, 2, neq, false, BasisType::GaussLegendre);
      const int ns = X.fes_s->GetFE(0)->GetDof();
      const int np = IntRules.Get(X.fes_s->GetFE(0)->GetGeomType(),
                                  2 * X.fes_s->GetFE(0)->GetOrder()).GetNPoints();
      CAPTURE(ns, np);
      REQUIRE(ns == np);              // the coincidence, stated
      const Gaps G = Compare(X, F, -1);
      CAPTURE(G.res, G.res_norm);
      REQUIRE(G.res_norm > 1e-3);
      REQUIRE(G.res / G.res_norm < 1e-12);
   }

   SECTION("the same box with a Gauss-Lobatto enriched basis: not identical")
   {
      Fixture X(4, order, 2, neq, false, BasisType::GaussLobatto);
      const Gaps G = Compare(X, F, -1);
      CAPTURE(G.res, G.res_norm);
      REQUIRE(G.res / G.res_norm > 1e-6);
   }

   SECTION("a simplex at the default rule: not identical")
   {
      Fixture X(4, order, 2, neq, true, BasisType::GaussLegendre);
      const Gaps G = Compare(X, F, -1);
      CAPTURE(G.res, G.res_norm);
      REQUIRE(G.res / G.res_norm > 1e-6);
   }
}
namespace darcy_reaction
{

/** @brief Solve the affine-reaction problem of the two cases below and report
    the potential, the Newton count, and whether the batched local routes were
    actually taken.

    @a mode is the only thing that varies between the arms. The discrete
    problem does not depend on it, so the two arms must agree -- which is what
    makes this a control rather than two independent runs. */
struct AffineRun
{
   Vector p;                 ///< the potential dofs
   int iters{-1};
   bool can_factor{false};   ///< CanBatchLocalFactor(), asked after Assemble()
   bool can_solve{false};    ///< CanBatchLocalSolve()
   real_t err{-1.};
};

void RunAffine(DarcyHybridization::LocalFactorMode mode, AffineRun &out)
{
   const int order = 1, dim = 2, n = 4;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE);

   L2_FECollection u_coll(order, dim), p_coll(order, dim);
   L2_FECollection s_coll(order + 1, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, dim, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, 1);
   FiniteElementSpace fes_s(&mesh, &s_coll, 1);
   FiniteElementSpace fes_t(&mesh, &t_coll, 1);

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);
   const real_t a = 0.7, c = -1.3;
   Affine F(1, a, c);

   // g = Delta u - F(u). The sign is the form's, established by running both.
   FunctionCoefficient gco([&](const Vector &x)
   {
      const real_t u = Pex(x);
      return -2.0 * M_PI * M_PI * u - (a + c * u);
   });

   HDGPostprocessBlocks blocks(fes_u, fes_p, fes_s);
   blocks.SetDiffusionInverse(ik);
   blocks.Assemble();

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   // An O(1) tau: the default carries 1/h, which is a different method.
   HDGDiffusionIntegrator *hdg = new HDGDiffusionIntegrator(ik, 0.5);
   ConstStab stab(1.0);
   hdg->SetStabilization(stab);
   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(hdg);
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gco));

   // BlockNonlinearForm DELETES its domain integrators, so it owns this one.
   HDGInterpolatoryReactionIntegrator *react =
      new HDGInterpolatoryReactionIntegrator(F, blocks);
   react->Assemble();
   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(react);

   Array<int> ess;
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetLocalFactorMode(mode);
   // Anything on the block nonlinear form forces FullNL, whose local solve
   // defaults to rtol 1e-6 and would cap the outer Newton at ~1e-5.
   dh->SetLocalNLSolver(
      DarcyHybridization::LSsolveType::Newton, 1000, 1e-13, 1e-16);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   Vector X, RHS;
   OperatorPtr A;
   darcy.FormLinearSystem(ess, x, A, X, RHS, true);

   out.can_factor = dh->CanBatchLocalFactor()
                    && mode == DarcyHybridization::LocalFactorMode::Batched;
   out.can_solve = dh->CanBatchLocalSolve();

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf;
#else
   GMRESSolver umf;
   umf.SetRelTol(1e-14);
   umf.SetMaxIter(2000);
#endif
   NewtonSolver newton;
   newton.SetSolver(umf);
   newton.SetOperator(*A);
   newton.SetRelTol(1e-11);
   newton.SetAbsTol(1e-14);
   newton.SetMaxIter(20);
   newton.SetPrintLevel(-1);
   newton.Mult(RHS, X);

   REQUIRE(newton.GetConverged());
   out.iters = newton.GetNumIterations();

   darcy.RecoverFEMSolution(X, x);
   out.p = x.GetBlock(1);

   GridFunction p_h(&fes_p, x.GetBlock(1));
   FunctionCoefficient uc([](const Vector &y) { return Pex(y); });
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   { irs[i] = &IntRules.Get(i, 2 * (order + 2) + 6); }
   out.err = p_h.ComputeL2Error(uc, irs);
}

} // namespace darcy_reaction

TEST_CASE("A linear interpolatory reaction converges in one Newton step",
          "[DarcyForm][Postprocess][Reaction]")
{
   // The falsifier for the Jacobian's (1,0) block, and it is a THRESHOLD
   // rather than a trend. With F(u) = a + c u and a, c constant, the
   // interpolatory term is a LINEAR operator, so the whole discrete problem
   // is linear and a correct Jacobian must converge in exactly one step.
   //
   // A Jacobian missing the (1,0) block cannot: A10 = c A9 B11 is not zero,
   // so every step is wrong by a fixed non-vanishing amount. Measured before
   // the block existed: five steps at a CONSTANT factor of 4.2e-3, which is
   // linear convergence, on a problem that should take one.
   //
   // The residual is bit-identical either way -- the reaction's own residual
   // reaches the potential row through AddMultBlock() whatever the Jacobian
   // does -- so the discriminating quantity is the iteration COUNT and not
   // the answer. That is the same trap as ConstructGrad()'s double count.
   using namespace darcy_reaction;

   AffineRun run;
   RunAffine(DarcyHybridization::LocalFactorMode::Serial, run);

   REQUIRE(run.iters == 1);

   // And the answer is the right one, so "one step" is not one step to
   // nowhere.
   CAPTURE(run.err);
   REQUIRE(run.err < 0.2);
   REQUIRE(run.err > 1e-6);    // a discretisation error, not a zero solution
}

TEST_CASE("The batched local routes carry the (1,0) gradient block",
          "[DarcyForm][Postprocess][Reaction]")
{
   // LocalFactorMode::Batched is a STORAGE decision -- FactorElementsBatched(),
   // ComputeElementsHBatched() and MultInvBatched() are transcriptions of the
   // per-element routines -- so it cannot move the answer. That is what makes
   // this a control rather than two independent runs, and it is the only
   // check that reaches the batched arm of the (1,0) block at all.
   //
   // It could not pass before those three routines learned the distinction:
   // each built its DenseTensor from Bf_data, so the batched Jacobian used
   // the LINEAR (1,0) block while the dense one used Bg_data. The earlier
   // draft refused the combination outright with an MFEM_VERIFY rather than
   // returning a wrong gradient, which is why this case is new rather than
   // previously failing. MEASURED: gate GradBStore() to return the linear
   // store and the batched arm's Newton count goes 1 -> 6 while the dense
   // arm stays at 1, which is the same failure the (1,0) block was built to
   // remove, arriving one route over.
   using namespace darcy_reaction;

   AffineRun ser, bat;
   RunAffine(DarcyHybridization::LocalFactorMode::Serial, ser);
   RunAffine(DarcyHybridization::LocalFactorMode::Batched, bat);

   // The batched routes were actually TAKEN. Without this the case passes
   // when the mode is silently declined, which is this branch's own "a test
   // that cannot run cannot fail" arriving one level down.
   REQUIRE(bat.can_factor);
   REQUIRE(bat.can_solve);
   REQUIRE_FALSE(ser.can_solve);

   // One Newton step in BOTH arms: the linear problem's threshold, now asked
   // of the batched Jacobian.
   REQUIRE(ser.iters == 1);
   REQUIRE(bat.iters == 1);

   REQUIRE(bat.p.Size() == ser.p.Size());
   Vector d(bat.p);
   d -= ser.p;
   const real_t rel = d.Normlinf() / ser.p.Normlinf();
   CAPTURE(rel, ser.err, bat.err);
   REQUIRE(std::isfinite(rel));
   REQUIRE(rel < 1e-12);
}
