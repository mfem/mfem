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

// The nonlinear hybridized Darcy operator, and NPC.
//
// Two subjects, together because the second is the alternative to the first.
// CondenseThenLinearise reduces a nonlinear problem to an operator on the
// TRACE, whose gradient is the Schur complement and whose three gradient modes
// must agree; NPC (Nguyen, Peraire & Cockburn, JCP 228 (2009) 8841-8855)
// instead runs Newton on the FULL (q, u, lambda) system with the Jacobian
// solved by hybridized elimination.
//
// This file was test_darcy_linearise_first.cpp and tested a third thing, an
// NLOrdering::LineariseThenCondense that claimed to be NPC and was not: it was
// an operator on the trace alone, so its fields were a function of the trace
// where NPC's are Newton state. It is deleted, and the cases that were about
// it went with it.

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <cstring>
#include <vector>

using namespace mfem;

namespace darcy_npc
{

// A two-equation nonlinear system whose state dependence is scaled by eps, so
// the same problem can be turned linear without changing anything else. The
// coupling makes d(flux residual)/dp nonzero, which is the block the local
// elimination has to carry.
class ScaledCoupledFlux : public MixedFluxFunction
{
   real_t eps;

   void Entries(const Vector &u, real_t &a00, real_t &a11, real_t &a01) const
   {
      a00 = 1.0 + eps * 0.5 * u(0) * u(0);
      a11 = 2.0 + eps * 0.5 * u(1) * u(1);
      a01 = 0.25 + eps * 0.1 * u(0) * u(1);
   }

public:
   ScaledCoupledFlux(int dim_, real_t e)
      : MixedFluxFunction(2, dim_), eps(e) { }

   real_t ComputeDualFlux(const Vector &u, const DenseMatrix &flux,
                          ElementTransformation &,
                          DenseMatrix &df) const override
   {
      real_t a00, a11, a01;
      Entries(u, a00, a11, a01);
      df.SetSize(2, dim);
      for (int d = 0; d < dim; d++)
      {
         df(0, d) = a00 * flux(0, d) + a01 * flux(1, d);
         df(1, d) = a01 * flux(0, d) + a11 * flux(1, d);
      }
      return std::max(a00, a11);
   }

   real_t ComputeFlux(const Vector &, ElementTransformation &,
                      DenseMatrix &flux) const override
   { flux = 0.0; return 0.0; }

   void ComputeDualFluxJacobian(const Vector &u, const DenseMatrix &flux,
                                ElementTransformation &, DenseMatrix &J_u,
                                DenseMatrix &J_F) const override
   {
      real_t a00, a11, a01;
      Entries(u, a00, a11, a01);
      J_F.SetSize(2*dim, 2*dim);
      J_F = 0.0;
      J_u.SetSize(2*dim, 2);
      J_u = 0.0;
      for (int d = 0; d < dim; d++)
      {
         J_F(d, d)             = a00;
         J_F(d, dim + d)       = a01;
         J_F(dim + d, d)       = a01;
         J_F(dim + d, dim + d) = a11;

         J_u(d, 0)       = eps * (u(0)*flux(0,d) + 0.1*u(1)*flux(1,d));
         J_u(d, 1)       = eps * (0.1*u(0)*flux(1,d));
         J_u(dim + d, 0) = eps * (0.1*u(1)*flux(0,d));
         J_u(dim + d, 1) = eps * (u(1)*flux(1,d) + 0.1*u(0)*flux(0,d));
      }
   }
};

void SourceTerm(const Vector &x, Vector &g)
{
   g.SetSize(2);
   real_t s = 1.0;
   for (int d = 0; d < x.Size(); d++) { s *= std::sin(M_PI * x(d)); }
   g(0) = s;
   g(1) = -0.7 * s;
}

// A source (s(u), w) on the potential block with s(u) = c u^2. The point is
// that it drives GetPotentialMassNonlinearForm(), which is a different path
// from the block nonlinear form above, and c scales how stiff the local
// problem is.
class SquareSource : public NonlinearFormIntegrator
{
public:
   explicit SquareSource(real_t c_) : c(c_) { }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         elvect.Add(ip.weight * Tr.Weight() * c * u * u, shape);
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elmat.SetSize(dof);
      elmat = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         AddMult_a_VVt(ip.weight * Tr.Weight() * 2.0 * c * u, shape, elmat);
      }
   }

private:
   real_t c;
   Vector shape;
};

/// The same bits, which is what "a function of the trace" has to mean here.
bool BitwiseEqual(const Vector &a, const Vector &b)
{
   if (a.Size() != b.Size()) { return false; }
   return std::memcmp(a.GetData(), b.GetData(),
                      a.Size()*sizeof(real_t)) == 0;
}

/// Collects the outer residual norm of every Newton iteration.
class NormHistory : public IterativeSolverMonitor
{
public:
   std::vector<real_t> norms;
   void MonitorResidual(int, real_t norm, const Vector &, bool) override
   { norms.push_back(norm); }
};

struct Outcome
{
   std::vector<real_t> norms;   ///< outer residual, per Newton iteration
   Vector p;                    ///< the potential recovered at the end
   long local_nl_iters = 0;     ///< local nonlinear iterations, summed
   bool converged = false;
};

/// Solve the hybridized nonlinear problem.
Outcome Solve(Mesh &mesh, int order, real_t eps, int max_it = 20,
              DarcyHybridization::GradientMode gmode =
                 DarcyHybridization::GradientMode::Assembled)
{
   const int dim = mesh.Dimension();
   const int neq = 2;
   ScaledCoupledFlux flux(dim, eps);

   L2_FECollection u_coll(order, dim), p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_p(&mesh, &p_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_t(&mesh, &t_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&fes_u, &fes_p);

   BlockNonlinearForm *Mnl = darcy.GetBlockNonlinearForm();
   Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(flux));
   auto *face = new MixedConductionNLFIntegrator(flux);
   Vector taus(neq);
   taus = 1.0;
   face->SetVariableStabilization(taus);
   Mnl->AddInteriorFaceIntegrator(face);

   MixedBilinearForm *Bform = darcy.GetFluxDivForm();
   Bform->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
   Bform->AddInteriorFaceIntegrator(
      new VectorBlockDiagonalIntegrator(
         neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));

   VectorFunctionCoefficient gcoeff(neq, SourceTerm);
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new VectorDomainLFIntegrator(gcoeff));

   Array<int> ess;
   darcy.EnableHybridization(
      &fes_t,
      new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator),
      ess);

   darcy.Assemble();

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetLocalNLSolver(DarcyHybridization::LSsolveType::Newton, 100, 1e-13,
                        1e-15, -1);
   dh->SetGradientMode(gmode);

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr op;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, op, X, RHS, true);

   // GSSmoother needs a SparseMatrix, so the matrix-free mode goes without.
   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(4000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(0.0);
   if (gmode == DarcyHybridization::GradientMode::Assembled)
   {
      lin.SetPreconditioner(prec);
   }

   NormHistory history;
   NewtonSolver newton;
   newton.SetSolver(lin);
   newton.SetOperator(*op);
   newton.SetRelTol(1e-12);
   newton.SetAbsTol(1e-14);
   newton.SetMaxIter(max_it);
   newton.SetPrintLevel(-1);
   newton.SetMonitor(history);
   newton.Mult(RHS, X);

   Outcome out;
   out.norms = history.norms;
   out.converged = newton.GetConverged();
   out.local_nl_iters = dh->GetNumLocalNLIterations();
   darcy.RecoverFEMSolution(X, x);
   out.p = x.GetBlock(1);
   return out;
}

} // namespace darcy_npc

namespace darcy_npc
{

/// The semilinear problem of the two tests below: (c p^2, w) on the potential
/// mass form, Dirichlet trace all round. Returns the reduced operator.
struct SemilinearHDG
{
   Mesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace Vh, Wh, Mh;
   DarcyForm darcy;
   ConstantCoefficient one;
   Array<int> ess_flux;
   OperatorHandle R;
   Vector X, B;
   BlockVector sol;
   FunctionCoefficient src;

   /// @a src_scale drives a source on the potential, so that the problem has a
   /// solution other than zero. The gradient tests below leave it at zero and
   /// evaluate at a randomised trace instead; only a solve needs it.
   SemilinearHDG(int n, int order, real_t c,
                 DarcyHybridization::GradientMode gmode =
                    DarcyHybridization::GradientMode::Assembled,
                 real_t src_scale = 0.0)
      : mesh(Mesh::MakeCartesian2D(n, n, Element::TRIANGLE)),
        u_coll(order, 2, BasisType::GaussLobatto), p_coll(order, 2),
        t_coll(order, 2),
        Vh(&mesh, &u_coll, 2), Wh(&mesh, &p_coll), Mh(&mesh, &t_coll),
        darcy(&Vh, &Wh), one(1.0),
        src([src_scale](const Vector &X_)
   { return src_scale*std::sin(M_PI*X_(0))*std::sin(M_PI*X_(1)); })
   {
      if (src_scale != 0.0)
      {
         darcy.GetPotentialRHS()->AddDomainIntegrator(
            new DomainLFIntegrator(src));
      }
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new SquareSource(c));
      Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
      Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(),
                                ess_flux);
      darcy.GetHybridization()->SetGradientMode(gmode);
      // The control has to be a control. CondenseThenLinearise solves the
      // local problem to this tolerance, and an inexact local solve is itself
      // a residual error, so the default 1e-6 would put the reference at 1e-6
      // and hide anything smaller.
      darcy.GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton, 1000, 1e-14, 1e-30);
      darcy.GetHybridization()->SetEssentialBC(ess_bdr);
      darcy.Assemble();

      sol.Update(darcy.GetOffsets());
      sol = 0.0;
      darcy.FormLinearSystem(ess_flux, sol, R, X, B, true);
   }

   Operator &op() { return *R.Ptr(); }
   const Array<int> &ess() const
   { return darcy.GetHybridization()->GetEssentialTrueDofs(); }
};


/** @brief The semilinear problem again, with the boundary constraint carried
    by @a bdr_td.size() integrators instead of one, and the boundary trace left
    FREE.

    Both halves matter. The boundary face integrators of a NonlinearForm reach
    DarcyHybridization as a LIST and are applied one at a time, where the
    interior ones arrive already summed into a single c_nlfi_p -- so a boundary
    face is the only place where several constraint integrators write the same
    E and G block, and those were written rather than accumulated. And an
    essential trace dof gets a unit row and an eliminated column, so a wrong
    boundary E and G never reaches the reduced system: with SetEssentialBC()
    called, as everywhere else in this file, the defect is invisible.

    HDGDiffusionIntegrator's built-in stabilization is `wq*beta` with beta the
    constructor's argument and no state in it, so the integrators are exactly
    linear in that argument and a split list is exactly the single integrator
    whose argument is the sum. That is what makes the comparison below exact
    rather than approximate. */
struct SplitBdrStabHDG
{
   Mesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace Vh, Wh, Mh;
   DarcyForm darcy;
   ConstantCoefficient one;
   Array<int> ess_flux;
   OperatorHandle R;
   Vector X, B;
   BlockVector sol;

   SplitBdrStabHDG(int n, int order, real_t c,
                   const std::vector<real_t> &bdr_td,
                   DarcyHybridization::GradientMode gmode =
                      DarcyHybridization::GradientMode::Assembled)
      : mesh(Mesh::MakeCartesian2D(n, n, Element::TRIANGLE)),
        u_coll(order, 2, BasisType::GaussLobatto), p_coll(order, 2),
        t_coll(order, 2),
        Vh(&mesh, &u_coll, 2), Wh(&mesh, &p_coll), Mh(&mesh, &t_coll),
        darcy(&Vh, &Wh), one(1.0)
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new SquareSource(c));
      Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.5));
      for (real_t td : bdr_td)
      {
         Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, td));
      }

      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(),
                                ess_flux);
      darcy.GetHybridization()->SetGradientMode(gmode);
      darcy.GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton, 1000, 1e-14, 1e-30);
      // No SetEssentialBC: see above.
      darcy.Assemble();

      sol.Update(darcy.GetOffsets());
      sol = 0.0;
      darcy.FormLinearSystem(ess_flux, sol, R, X, B, true);
   }

   Operator &op() { return *R.Ptr(); }
};

} // namespace darcy_npc

TEST_CASE("The reduced gradient is the derivative of the reduced residual",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // GetGradient() against a central difference of Mult(). Under
   // LineariseThenCondense it was not the derivative: the retained local
   // residual was applied twice, once predicting and once correcting, so the
   // correction was evaluated a whole local Newton step away from the fields
   // the retained factors were built at, and the gradient was wrong by the
   // change in the local Jacobian over that step.
   //
   // The error was O(1) at a COLD linearisation -- the first one, which
   // retained the caller's initial guess -- and second-order small once the
   // retained fields had converged. That is why a mild problem was unaffected
   // and a stiff one lost the first Newton step, and why a line search, which
   // measures every trial against one linearisation, made it worse. This test
   // is deliberately cold: one Mult, one GetGradient, then the difference.
   //
   // With the defect present, at c = 100 this reported 3.2e-03 and was
   // independent of h across four decades -- which is what says a real
   // Jacobian error rather than a differencing artefact.
   const real_t c = GENERATE(1.0, 1.0e1, 1.0e2, 1.0e3);
   const real_t h = GENERATE(1.0e-4, 1.0e-5);
   // Both ways of producing the gradient have to be the derivative of the same
   // residual. The matrix-free one applies the Schur complement instead of
   // assembling it, and used to leave out d(flux residual)/dp and the diagonal
   // policy's regularisation, either of which makes it a different operator.
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(c, h,
           gmode == GM::MatrixFree);

   SemilinearHDG P(8, 1, c, gmode);
   Operator &op = P.op();
   const int m = op.Height();

   // The essential trace rows are masked: the residual is zeroed there and the
   // Jacobian carries a unit row, so comparing them is meaningless. Finding
   // none of them would mean the problem is not the Dirichlet problem it is
   // supposed to be, and the whole comparison would be measuring something
   // ill-posed -- so that is checked, not assumed.
   Array<int> ess_marker(m);
   ess_marker = 0;
   for (int i = 0; i < P.ess().Size(); i++) { ess_marker[P.ess()[i]] = 1; }
   CAPTURE(P.ess().Size(), m);
   REQUIRE(P.ess().Size() > 0);

   Vector x(m), v(m);
   x.Randomize(3);
   x *= 0.05;
   v.Randomize(7);
   for (int i = 0; i < m; i++)
   {
      if (ess_marker[i]) { x(i) = 0.0; v(i) = 0.0; }
   }
   v *= 1.0/v.Norml2();

   // Newton's own order, and the only order in which the question is well
   // posed: the residual, then the gradient at the same trace. The
   // linearisation then sits at x and Mult() never moves it, so both
   // difference evaluations see the linearisation the gradient belongs to.
   Vector r0(m);
   op.Mult(x, r0);
   op.GetGradient(x);

   Vector xp(x), xm(x), rp(m), rm(m), Jv(m);
   xp.Add(h, v);
   xm.Add(-h, v);
   op.Mult(xp, rp);
   op.Mult(xm, rm);
   Vector fd(rp);
   fd -= rm;
   fd *= 1.0/(2.0*h);

   op.GetGradient(x).Mult(v, Jv);   // idempotent at the retained trace

   real_t num = 0.0, den = 0.0;
   for (int i = 0; i < m; i++)
   {
      if (ess_marker[i]) { continue; }
      const real_t d = Jv(i) - fd(i);
      num += d*d;
      den += fd(i)*fd(i);
   }
   const real_t rel = std::sqrt(num)/std::max(real_t(1e-300), std::sqrt(den));

   CAPTURE(rel, std::sqrt(den));
   REQUIRE(std::sqrt(den) > 0.0);
   // A central difference of an exact Jacobian is limited by round-off, which
   // grows as 1/h -- about 1e-12 at h = 1e-4 and 1e-11 at h = 1e-5 here. The
   // bound is set well above that and far below the 3.2e-03 the defect gave.
   REQUIRE(rel < 1.0e-8);
}

TEST_CASE("Every constraint integrator on a boundary face reaches the gradient",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // D and H accumulate over the integrators that touch a face; E and G were
   // OVERWRITTEN, so a face reached by more than one kept only the last one's
   // blocks. Interior faces never see it -- they arrive summed into one
   // integrator -- and neither does an essential boundary trace, whose rows and
   // columns are eliminated. What is left is exactly a FREE boundary trace with
   // several boundary face integrators, which is what navierstokes -bcphys is
   // and what found this.
   //
   // The symptom is not a wrong answer. It is a gradient that is not the
   // derivative of its own residual, and under hybridization that gradient is
   // never assembled globally, so nothing complains: Newton simply stops being
   // Newton. On the miniapp's LINEAR Stokes problem, where one step is exact,
   // it took 35 at a fixed residual ratio of 0.517 -- a fixed-point iteration.
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   const real_t c = GENERATE(1.0, 1.0e2);
   CAPTURE(c, gmode == GM::MatrixFree);

   // 1.0 + 0.5 is exactly 1.5 for this integrator, so the two problems are the
   // same problem written two ways.
   SplitBdrStabHDG one_integ(6, 1, c, {1.5}, gmode);
   SplitBdrStabHDG two_integ(6, 1, c, {1.0, 0.5}, gmode);

   Operator &op1 = one_integ.op();
   Operator &op2 = two_integ.op();
   const int m = op1.Height();
   REQUIRE(op2.Height() == m);

   Vector x(m), v(m);
   x.Randomize(11);
   x *= 0.05;
   v.Randomize(13);
   v *= 1.0/v.Norml2();

   SECTION("the residual was never wrong")
   {
      // It adds every integrator, which is why the defect could not be seen in
      // an answer and had to be seen in a convergence history.
      Vector r1(m), r2(m);
      op1.Mult(x, r1);
      op2.Mult(x, r2);
      r2 -= r1;
      const real_t rel = r2.Norml2()/r1.Norml2();
      CAPTURE(rel, r1.Norml2());
      REQUIRE(rel < 1.0e-12);
   }

   SECTION("and the gradient now is not either")
   {
      // Split against unsplit. With the defect the split problem's gradient
      // was missing the first integrator's E and G entirely.
      Vector j1(m), j2(m);
      op1.GetGradient(x).Mult(v, j1);
      op2.GetGradient(x).Mult(v, j2);
      j2 -= j1;
      const real_t rel = j2.Norml2()/j1.Norml2();
      CAPTURE(rel, j1.Norml2());
      REQUIRE(j1.Norml2() > 0.0);
      REQUIRE(rel < 1.0e-12);
   }

   SECTION("and it is the derivative of the split residual")
   {
      // The absolute check, which does not lean on the unsplit problem being
      // right. Gradient first, then the difference: Mult() leaves the
      // linearisation at its own argument, so taking it afterwards measures a
      // different operator (see the case above).
      const real_t h = 1.0e-5;
      Vector r0(m);
      op2.Mult(x, r0);
      op2.GetGradient(x);

      Vector xp(x), xm(x), rp(m), rm(m), Jv(m);
      xp.Add(h, v);
      xm.Add(-h, v);
      op2.Mult(xp, rp);
      op2.Mult(xm, rm);
      Vector fd(rp);
      fd -= rm;
      fd *= 1.0/(2.0*h);

      op2.GetGradient(x).Mult(v, Jv);
      Vector d(Jv);
      d -= fd;
      const real_t rel = d.Norml2()/fd.Norml2();
      CAPTURE(rel, fd.Norml2());
      REQUIRE(fd.Norml2() > 0.0);
      REQUIRE(rel < 1.0e-8);
   }
}

TEST_CASE("The gradient matches a difference taken in the caller's order",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // The case above hoists GetGradient() above the difference, which is
   // Newton's own order. This one does what a caller writing a gradient check
   // naturally writes instead -- difference first, gradient afterwards -- and
   // is here because a caller reported the two disagreeing by three orders:
   // 3.968e-08 in this order against 4.023e-11 in the other, on a problem with
   // no stiffness anywhere.
   //
   // The cause was that Mult() did not move the linearisation, so the two
   // perturbed evaluations shared whichever linearisation happened to be
   // retained and the gradient was taken at a third trace: the difference
   // straddled three linearisation points. Mult() now linearises at its own
   // argument, so each evaluation is self-consistent and the order no longer
   // matters. That the two orders agree is the property; if they ever diverge
   // again, the reduced residual has stopped being a function of its argument.
   const real_t c = GENERATE(1.0, 1.0e1, 1.0e2);
   const real_t h = 1.0e-5;
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(c, h,
           gmode == GM::MatrixFree);

   SemilinearHDG P(8, 1, c, gmode);
   Operator &op = P.op();
   const int m = op.Height();

   Array<int> ess_marker(m);
   ess_marker = 0;
   for (int i = 0; i < P.ess().Size(); i++) { ess_marker[P.ess()[i]] = 1; }
   REQUIRE(P.ess().Size() > 0);

   Vector x(m), v(m);
   x.Randomize(3);
   x *= 0.05;
   v.Randomize(7);
   for (int i = 0; i < m; i++)
   {
      if (ess_marker[i]) { x(i) = 0.0; v(i) = 0.0; }
   }
   v *= 1.0/v.Norml2();

   // The caller's order: both difference evaluations first, and only then the
   // gradient. Nothing establishes a linearisation at x beforehand.
   Vector xp(x), xm(x), rp(m), rm(m), Jv(m);
   xp.Add(h, v);
   xm.Add(-h, v);
   op.Mult(xp, rp);
   op.Mult(xm, rm);
   Vector fd(rp);
   fd -= rm;
   fd *= 1.0/(2.0*h);

   op.GetGradient(x).Mult(v, Jv);

   real_t num = 0.0, den = 0.0;
   for (int i = 0; i < m; i++)
   {
      if (ess_marker[i]) { continue; }
      const real_t d = Jv(i) - fd(i);
      num += d*d;
      den += fd(i)*fd(i);
   }
   const real_t rel = std::sqrt(num)/std::max(real_t(1e-300), std::sqrt(den));

   CAPTURE(rel, std::sqrt(den));
   REQUIRE(std::sqrt(den) > 0.0);
   REQUIRE(rel < 1.0e-8);
}

TEST_CASE("The three trace solves reach the same solution",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // Hybridization leaves a choice of how much of the trace system to build,
   // and all of it has to be available and equivalent:
   //
   //   0  assemble the Schur complement and factor it       (direct)
   //   1  assemble it and solve with a Krylov method        (assembled)
   //   2  never assemble it, only apply it                  (matrix-free)
   //
   // Level 2 is the only one that is Jacobian-free in the sense a hybridized
   // formulation can be: the local blocks must still be factored per element
   // on every route -- that is what condensation is -- and what it avoids is
   // the global matrix, at one local back-substitution per element per
   // application instead of one per trace dof once.
   //
   // The three must agree. They did not: the matrix-free apply left out the
   // Jacobian's d(flux residual)/dp and the diagonal policy's regularisation
   // of rows nothing contributes to, so it was a different operator from the
   // matrix its own GetGradient() would have assembled.
   const real_t c = GENERATE(1.0, 5.0);
   const real_t src = 4.0;
   CAPTURE(c);

   // The reference: assembled and solved directly.
   Vector p_ref;
   int ref_its = -1;
   {
      SemilinearHDG P(6, 1, c, GM::Assembled, src);
      GSSmoother prec;
      GMRESSolver lin;
      lin.SetKDim(400);
      lin.SetMaxIter(2000);
      lin.SetRelTol(1e-14);
      lin.SetAbsTol(0.0);
      lin.SetPreconditioner(prec);
      NewtonSolver newton;
      newton.SetSolver(lin);
      newton.SetOperator(P.op());
      newton.SetRelTol(1e-12);
      newton.SetAbsTol(1e-14);
      newton.SetMaxIter(30);
      newton.SetPrintLevel(-1);
      newton.Mult(P.B, P.X);
      REQUIRE(newton.GetConverged());
      ref_its = newton.GetNumIterations();
      P.darcy.RecoverFEMSolution(P.X, P.sol);
      p_ref = P.sol.GetBlock(1);
   }
   REQUIRE(p_ref.Normlinf() > 0.0);

   const int level = GENERATE(0, 1, 2);
   CAPTURE(level);

   SemilinearHDG P(6, 1, c,
                   (level == 2) ? GM::MatrixFree : GM::Assembled, src);

   // Level 2 has no matrix, so it must be solved by something that needs only
   // the action. GSSmoother and the direct solvers all require a SparseMatrix.
   GSSmoother prec;
   GMRESSolver gmres;
   gmres.SetKDim(400);
   gmres.SetMaxIter(4000);
   gmres.SetRelTol(1e-13);
   gmres.SetAbsTol(0.0);
   gmres.SetPrintLevel(-1);
   if (level == 1) { gmres.SetPreconditioner(prec); }

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver direct;
#else
   GMRESSolver direct;
   direct.SetKDim(400);
   direct.SetMaxIter(4000);
   direct.SetRelTol(1e-14);
   direct.SetAbsTol(0.0);
   direct.SetPreconditioner(prec);
   direct.SetPrintLevel(-1);
#endif

   NewtonSolver newton;
   newton.SetSolver((level == 0) ? (Solver &)direct : (Solver &)gmres);
   newton.SetOperator(P.op());
   newton.SetRelTol(1e-12);
   newton.SetAbsTol(1e-14);
   newton.SetMaxIter(30);
   newton.SetPrintLevel(-1);
   newton.Mult(P.B, P.X);

   CAPTURE(newton.GetNumIterations(), ref_its);
   REQUIRE(newton.GetConverged());

   P.darcy.RecoverFEMSolution(P.X, P.sol);
   Vector d(P.sol.GetBlock(1));
   d -= p_ref;
   CAPTURE(d.Norml2(), p_ref.Norml2());
   REQUIRE(d.Norml2() < 1e-9 * p_ref.Norml2());
}

TEST_CASE("The trace solves agree where the matrix-free apply is hardest",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // "The three trace solves reach the same solution" drives a semilinear
   // potential mass, which exercises the matrix-free apply mechanically and
   // tests neither thing that was wrong with it: Bnl is empty there, and the
   // potential constraint has a boundary face term so no trace row needs the
   // diagonal policy's regularisation.
   //
   // This one drives the block nonlinear form, where both bite. The flux law
   // depends on the potential, so d(flux residual)/dp is non-empty and the
   // matrix-free Schur complement used to drop it; and the constraint has no
   // boundary face term, so 64 of the 160 trace rows are empty and carry a
   // unit diagonal in the assembled matrix that the apply has to reproduce.
   // Without either, this test fails.
   const real_t eps = GENERATE(0.0, 0.5, 5.0);
   CAPTURE(eps);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   Outcome assembled = Solve(mesh, 1, eps, 20, GM::Assembled);
   Outcome matfree = Solve(mesh, 1, eps, 20, GM::MatrixFree);

   CAPTURE(assembled.converged, matfree.converged,
           assembled.norms.size(), matfree.norms.size());
   REQUIRE(assembled.converged);
   REQUIRE(matfree.converged);
   REQUIRE(assembled.p.Normlinf() > 1e-4);

   Vector d(matfree.p);
   d -= assembled.p;
   CAPTURE(d.Norml2(), assembled.p.Norml2());
   REQUIRE(d.Norml2() < 1e-9 * assembled.p.Norml2());
}

namespace darcy_npc
{

/// The block-nonlinear-form problem of Solve(), stopped before the solve so a
/// gradient can be compared against a difference quotient at a chosen trace.
struct CoupledHDG
{
   Mesh mesh;
   ScaledCoupledFlux flux;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace fes_u, fes_p, fes_t;
   DarcyForm darcy;
   VectorFunctionCoefficient gcoeff;
   Array<int> ess;
   OperatorPtr op;
   Vector X, RHS;
   BlockVector x;

   CoupledHDG(int n, int order, real_t eps,
              DarcyHybridization::GradientMode gmode)
      : mesh(Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL)),
        flux(2, eps), u_coll(order, 2), p_coll(order, 2), t_coll(order, 2),
        fes_u(&mesh, &u_coll, 2*2, Ordering::byNODES),
        fes_p(&mesh, &p_coll, 2, Ordering::byNODES),
        fes_t(&mesh, &t_coll, 2, Ordering::byNODES),
        darcy(&fes_u, &fes_p), gcoeff(2, SourceTerm)
   {
      BlockNonlinearForm *Mnl = darcy.GetBlockNonlinearForm();
      Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(flux));
      auto *face = new MixedConductionNLFIntegrator(flux);
      Vector taus(2);
      taus = 1.0;
      face->SetVariableStabilization(taus);
      Mnl->AddInteriorFaceIntegrator(face);

      MixedBilinearForm *Bform = darcy.GetFluxDivForm();
      Bform->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(2, new VectorDivergenceIntegrator));
      Bform->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                          2, new TransposeIntegrator(
                                             new DGNormalTraceIntegrator(-1.))));

      darcy.GetPotentialRHS()->AddDomainIntegrator(
         new VectorDomainLFIntegrator(gcoeff));

      darcy.EnableHybridization(&fes_t, new VectorBlockDiagonalIntegrator(
                                   2, new NormalTraceJumpIntegrator), ess);
      darcy.Assemble();

      DarcyHybridization *dh = darcy.GetHybridization();
      dh->SetLocalNLSolver(DarcyHybridization::LSsolveType::Newton, 100, 1e-13,
                           1e-15, -1);
      dh->SetGradientMode(gmode);

      x.Update(darcy.GetOffsets());
      x = 0.0;
      darcy.FormLinearSystem(ess, x, op, X, RHS, true);
   }
};

} // namespace darcy_npc

TEST_CASE("Both reduced gradients are the derivative on a coupled flux law",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // This is the case that has teeth for the matrix-free apply, and a solve
   // does not: a wrong Jacobian changes the path Newton takes, not the root it
   // reaches, so "the three levels agree on the answer" passes with the
   // gradient broken. Only a difference quotient catches it.
   //
   // The flux law here depends on the potential, so the local Jacobian's (0,1)
   // block is -/+B^T PLUS d(flux residual)/dp. The matrix-free Schur
   // complement applied the linear part alone and was wrong by the rest.
   const real_t eps = GENERATE(0.5, 5.0);
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(eps,
           gmode == GM::MatrixFree);

   CoupledHDG P(4, 1, eps, gmode);
   Operator &op = *P.op;
   const int m = op.Height();

   Vector x(m), v(m), Jv(m), r0(m);
   x.Randomize(11);
   x *= 0.05;
   v.Randomize(13);
   v *= 1.0/v.Norml2();

   op.Mult(x, r0);
   op.GetGradient(x);

   const real_t h = 1e-6;
   Vector xp(x), xm(x), rp(m), rm(m);
   xp.Add(h, v);
   xm.Add(-h, v);
   op.Mult(xp, rp);
   op.Mult(xm, rm);
   Vector fd(rp);
   fd -= rm;
   fd *= 1.0/(2.0*h);

   op.GetGradient(x).Mult(v, Jv);

   // This problem's constraint has no boundary face term, so the trace rows on
   // the boundary get no contribution at all: the residual is identically zero
   // there and the diagonal policy gives them a unit row. A difference
   // quotient has nothing to say about such a row -- it moved by exactly zero
   // -- so they come out here, the same way the essential rows do elsewhere.
   int compared = 0;
   real_t num = 0.0, den = 0.0;
   for (int i = 0; i < m; i++)
   {
      if (rp(i) == rm(i)) { continue; }
      const real_t d = Jv(i) - fd(i);
      num += d*d;
      den += fd(i)*fd(i);
      compared++;
   }
   const real_t rel = std::sqrt(num)/std::max(real_t(1e-300), std::sqrt(den));

   CAPTURE(rel, compared, m);
   REQUIRE(compared > m/4);
   REQUIRE(den > 0.0);
   REQUIRE(rel < 1.0e-7);
}

TEST_CASE("Assembling the reduced gradient and applying it give one operator",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // The two gradient modes must be the same operator, not merely two things
   // a Krylov method can be driven to the same answer with. This compares them
   // row for row, which the difference-quotient test cannot: a difference
   // quotient has nothing to say about a row the residual never moves, and
   // those are exactly the rows the diagonal policy regularises.
   //
   // Both halves of that matter and neither is caught by a solve. Dropping
   // d(flux residual)/dp from the apply leaves the two disagreeing by 1e-3 on
   // the live rows; dropping the regularisation leaves them disagreeing by
   // 0.57 overall, on the 64 boundary trace rows of 160 that carry a unit
   // diagonal in the matrix and nothing at all in the apply.
   const real_t eps = GENERATE(0.0, 0.5, 5.0);
   CAPTURE(eps);

   CoupledHDG A(4, 1, eps, GM::Assembled);
   CoupledHDG M(4, 1, eps, GM::MatrixFree);
   const int m = A.op->Height();
   REQUIRE(M.op->Height() == m);

   Vector x(m), v(m), ya(m), ym(m), r(m);
   x.Randomize(11);
   x *= 0.05;
   v.Randomize(13);
   v *= 1.0/v.Norml2();

   // Both are put in the same state first: a residual, then a gradient at the
   // same trace, which is the order NewtonSolver uses and the only one in
   // which a retained linearisation is defined.
   A.op->Mult(x, r);
   M.op->Mult(x, r);
   A.op->GetGradient(x).Mult(v, ya);
   M.op->GetGradient(x).Mult(v, ym);

   Vector d(ym);
   d -= ya;
   const real_t rel = d.Norml2()/std::max(real_t(1e-300), ya.Norml2());
   CAPTURE(rel, ya.Norml2(), ym.Norml2());
   REQUIRE(ya.Norml2() > 0.0);
   REQUIRE(rel < 1.0e-12);
}

namespace darcy_npc
{

/// J*v by a difference quotient of the residual, as a Jacobian-free Krylov
/// method forms it: nothing but Mult(), no GetGradient().
class DQJacobian : public Operator
{
   Operator &R;
   const Vector &x, &r0;
   mutable Vector xt, rt;
public:
   DQJacobian(Operator &R_, const Vector &x_, const Vector &r0_)
      : Operator(R_.Height()), R(R_), x(x_), r0(r0_),
        xt(R_.Height()), rt(R_.Height()) { }

   void Mult(const Vector &v, Vector &Jv) const override
   {
      const real_t vn = v.Norml2();
      Jv.SetSize(height);
      if (vn == 0.0) { Jv = 0.0; return; }
      const real_t eps = std::sqrt(1e-16)*(1.0 + x.Norml2())/vn;
      add(x, eps, v, xt);
      R.Mult(xt, rt);
      rt -= r0;
      Jv.Set(1.0/eps, rt);
   }
};

/// A Jacobian-free Newton-Krylov solve of the reduced system: it differences
/// the residual and never asks for a gradient, which used to be the case this
/// ordering could not serve.
void SolveJFNK(SemilinearHDG &P, int max_it = 30)
{
   Operator &R = P.op();
   const int m = R.Height();
   Vector x(P.X), r(m), c(m);

   for (int k = 0; k < max_it; k++)
   {
      R.Mult(x, r);
      r -= P.B;
      if (r.Norml2() < 1e-11) { break; }

      // The whole of the contract, and the whole of what a matrix-based
      // NewtonSolver gets for free by asking for a gradient here.

      DQJacobian J(R, x, r);
      GMRESSolver gmres;
      gmres.SetOperator(J);
      gmres.SetKDim(200);
      gmres.SetMaxIter(400);
      gmres.SetRelTol(1e-10);
      gmres.SetAbsTol(0.0);
      gmres.SetPrintLevel(-1);
      c = 0.0;
      gmres.Mult(r, c);
      x -= c;
   }
   P.X = x;
}

} // namespace darcy_npc

namespace darcy_npc
{

/** @brief The pedestal source of Sanchez-Vizuet, Solano & Cerfon, CPC 255
    (2020) 107239 section 4.2, eqs (23)/(24), with c1 = 0.8 and c2 = 0.2:

        p(u) = ( c1 + c2 u^2 )( 1 - e ),   e = exp( -u^2/sigma ),   f = dp/du

    entered on the potential block as -(f(u), w). @a sigma is the pedestal
    width and is the only thing varied.

    A weaker source will not do, which is worth knowing before substituting
    one. The obvious simplification A u exp(-u^2/s)/s converges under both
    orderings at every configuration tried; the published expression carries
    c1/sigma against a much larger prefactor, and only it reaches the regime.
    Reported that way from a caller who lost most of a day to the simplified
    form showing nothing. */
class PedestalSource : public NonlinearFormIntegrator
{
public:
   PedestalSource(real_t amp_, real_t sigma_) : amp(amp_), sigma(sigma_) { }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 4);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcShape(ip, shape);
         elvect.Add(-ip.weight * Tr.Weight() * f(shape * elfun), shape);
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elmat.SetSize(dof);
      elmat = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 4);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcShape(ip, shape);
         AddMult_a_VVt(-ip.weight * Tr.Weight() * df(shape * elfun), shape,
                       elmat);
      }
   }

private:
   real_t f(real_t u) const
   {
      const real_t e = std::exp(-u*u/sigma);
      return amp * 2.0 * u * (0.2*(1.0 - e) + (0.8 + 0.2*u*u)*e/sigma);
   }
   /// Differenced rather than differentiated: this is a test source, and the
   /// question asked of it is about the hybridization's Jacobian, not its own.
   real_t df(real_t u) const
   {
      const real_t d = 1e-7;
      return (f(u + d) - f(u - d))/(2.0*d);
   }

   real_t amp, sigma;
   Vector shape;
};

/// tau = 1, constant, rather than the built-in {h^-1 Q}: the papers' choice,
/// and what the caller's reproducer uses.
class ConstantTau : public HDGStabilization
{
public:
   explicit ConstantTau(real_t t) : tau(t) { }
   bool IsConstant() const override { return true; }
   real_t Eval(real_t, real_t, real_t, real_t,
               ElementTransformation &) const override { return tau; }
private:
   real_t tau;
};

/** @brief The caller's reproducer, rebuilt: -div( grad u ) = f(u) on a
    0.8 x 1.2 rectangle of triangles, HDG on DarcyForm with a LINEAR flux mass
    and divergence and the whole nonlinearity on the potential mass nonlinear
    form -- which is the shape a semilinear problem forces -- with a linear
    ramp as both the Dirichlet datum and the initial guess.

    The setup matters as much as the source does. SemilinearHDG above
    evaluates at a randomised trace with no boundary data, and the pedestal
    shows nothing there: both orderings sit at round-off for every width
    tried, at every amplitude tried. It is the ramp datum driving the fields
    to O(1) that puts the local problem in the regime at all. */
struct PedestalHDG
{
   Mesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace Vh, Wh, Mh;
   ConstantCoefficient one;
   ConstantTau tau;
   DarcyForm darcy;
   Array<int> all, ess_flux, offs;
   BlockVector sol, rhs;
   Vector X, RHS;
   OperatorPtr R;

   /// @a amp = 0 makes the whole problem linear without changing anything
   /// else, which is what the NPC cases below need.
   PedestalHDG(int n, int order, real_t sigma, real_t amp = 1.0)
      : mesh(Mesh::MakeCartesian2D(n, n, Element::TRIANGLE, false, 0.8, 1.2)),
        u_coll(order, 2, BasisType::GaussLobatto),
        p_coll(order, 2, BasisType::GaussLobatto),
        t_coll(order, 2),
        Vh(&mesh, &u_coll, 2), Wh(&mesh, &p_coll), Mh(&mesh, &t_coll),
        one(1.0), tau(1.0), darcy(&Vh, &Wh), offs(4)
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));

      auto *interior = new HDGDiffusionIntegrator(one, 1.0);
      auto *boundary = new HDGDiffusionIntegrator(one, 1.0);
      interior->SetStabilization(tau);
      boundary->SetStabilization(tau);

      all.SetSize(mesh.bdr_attributes.Max());
      all = 1;

      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new PedestalSource(amp, sigma));
      Mnl_p->AddInteriorFaceIntegrator(interior);
      Mnl_p->AddBdrFaceIntegrator(boundary, all);

      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      B->AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
      B->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-2.0)), all);

      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(),
                                ess_flux);
      darcy.GetHybridization()->SetEssentialBC(all);
      // The reference has to be a reference: CondenseThenLinearise solves the
      // local problem to this tolerance, and so, now, does the linearisation
      // point of LineariseThenCondense. The default 1e-6 would put both at
      // 1e-6 and hide everything smaller.
      darcy.GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton, 100, 1e-12, 1e-16, -1);
      darcy.Assemble();

      offs[0] = 0;
      offs[1] = Vh.GetVSize();
      offs[2] = Wh.GetVSize();
      offs[3] = Mh.GetVSize();
      offs.PartialSum();
      sol.Update(offs);
      rhs.Update(offs);
      sol = 0.0;
      rhs = 0.0;

      FunctionCoefficient ramp([](const Vector &x)
      { return 0.5*(x(1) - 0.6); });
      GridFunction pgf, tgf;
      pgf.MakeRef(&Wh, sol.GetBlock(1), 0);
      pgf.ProjectCoefficient(ramp);
      tgf.MakeRef(&Mh, sol.GetBlock(2), 0);
      tgf.ProjectBdrCoefficient(ramp, all);

      X.MakeRef(sol, offs[2], Mh.GetVSize());
      RHS.MakeRef(rhs, offs[2], Mh.GetVSize());
      BlockVector dsol(sol, darcy.GetOffsets()),
                  drhs(rhs, darcy.GetOffsets());
      darcy.FormLinearSystem(ess_flux, dsol, drhs, R, X, RHS, true);
   }

   Operator &op() { return *R.Ptr(); }
   const Array<int> &ess() const
   { return darcy.GetHybridization()->GetEssentialTrueDofs(); }
   /// The load, blocks (flux, potential), as NPCResidual() wants it.
   BlockVector load() { return BlockVector(rhs, darcy.GetOffsets()); }
   /// The field state, same blocks. The trace state is @a X.
   BlockVector state() { return BlockVector(sol, darcy.GetOffsets()); }
};

struct NPCOutcome
{
   std::vector<real_t> norms;   ///< the FULL residual, per Newton step
   long local_nl_iters = 0;
   bool converged = false;
};

/** @brief One NPC Newton loop, driven the way NPCResidual()'s doxygen sets it
    out: residual, gradient, reduce, trace solve, recover,
    and advance all three blocks. Convergence is judged on the FULL residual,
    which is the half of NPC a reduced trace operator cannot express.

    @a line_search backtracks on that same full residual. It is well defined
    here precisely because the fields are Newton state, so the step scales the
    fields and the trace together. */
NPCOutcome RunNPC(PedestalHDG &P, int max_it, bool line_search,
                  DarcyHybridization::GradientMode gmode)
{
   DarcyHybridization &dh = *P.darcy.GetHybridization();
   dh.SetGradientMode(gmode);

   BlockVector b = P.load(), x = P.state();
   Vector &x_tr = P.X;
   BlockVector r(P.darcy.GetOffsets()), dx(P.darcy.GetOffsets());
   BlockVector xt(P.darcy.GetOffsets()), rt(P.darcy.GetOffsets());
   Vector r_tr, b_tr, dtr, xt_tr, rt_tr;

   NPCOutcome out;
   const long nl0 = dh.GetNumLocalNLIterations();

   for (int it = 0; it <= max_it; it++)
   {
      dh.NPCResidual(b, x, x_tr, r, r_tr);
      const real_t nrm = std::sqrt(r*r + r_tr*r_tr);
      out.norms.push_back(nrm);
      if (nrm < 1e-12) { out.converged = true; break; }
      if (it == max_it) { break; }

      Operator &S = dh.NPCGradient(x, x_tr);
      dh.NPCReduce(r, r_tr, b_tr);

      dtr.SetSize(b_tr.Size());
      dtr = 0.0;
      if (SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S))
      {
         UMFPackSolver lin(*Sm);
         lin.Mult(b_tr, dtr);
      }
      else
      {
         GMRESSolver lin;
         lin.SetOperator(S);
         lin.SetKDim(300);
         lin.SetMaxIter(3000);
         lin.SetRelTol(1e-14);
         lin.SetAbsTol(0.0);
         lin.SetPrintLevel(-1);
         lin.Mult(b_tr, dtr);
      }

      dh.NPCRecover(r, dtr, dx);

      real_t alpha = 1.0;
      if (line_search)
      {
         for (int k = 0; k < 20; k++)
         {
            xt = x;
            xt.Add(alpha, dx);
            xt_tr = x_tr;
            xt_tr.Add(alpha, dtr);
            dh.NPCResidual(b, xt, xt_tr, rt, rt_tr);
            if (std::sqrt(rt*rt + rt_tr*rt_tr) < nrm) { break; }
            alpha *= 0.5;
         }
      }
      x.Add(alpha, dx);
      x_tr.Add(alpha, dtr);
   }

   out.local_nl_iters = dh.GetNumLocalNLIterations() - nl0;
   return out;
}

} // namespace darcy_npc

TEST_CASE("A stiff source converges by condensation and by NPC alike",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // This case used to compare the two ORDERINGS, because a caller required
   // that no problem converging under CondenseThenLinearise may fail under
   // LineariseThenCondense. That mode is gone -- it was a condensation in
   // disguise, measurably slower than the one it was meant to beat and unable
   // to solve four configurations it solved -- so the comparison that is left
   // is the one that matters: CondenseThenLinearise, an operator on the trace
   // whose local problem is solved nonlinearly, against NPC, a Newton on the
   // whole (q, u, lambda) system whose local work is one linear solve.
   //
   // They are different methods reaching the same discrete solution, so what
   // is required of them is only that both get there. The second row is where
   // the deleted mode failed at sixty iterations.
   const int idx = GENERATE(0, 1);
   const int n = (idx == 0) ? 24 : 32;
   const real_t sigma = (idx == 0) ? 0.005 : 0.003;
   CAPTURE(n, sigma);

   SECTION("by condensation, on the trace alone")
   {
      PedestalHDG P(n, 1, sigma);
      UMFPackSolver lin;
      NewtonSolver newton;
      newton.SetOperator(P.op());
      newton.SetSolver(lin);
      newton.SetRelTol(1e-10);
      newton.SetAbsTol(1e-14);
      newton.SetMaxIter(30);
      newton.SetPrintLevel(-1);
      newton.iterative_mode = true;
      newton.Mult(P.RHS, P.X);

      CAPTURE(newton.GetNumIterations(), newton.GetFinalNorm());
      REQUIRE(newton.GetConverged());
   }

   SECTION("by NPC, on the full system")
   {
      // Backtracking on the full residual, which is the globalisation NPC
      // wants and which is well defined only because the fields are state:
      // the step scales them and the trace together.
      PedestalHDG P(n, 1, sigma);
      const NPCOutcome out = RunNPC(P, 40, true, GM::Assembled);
      CAPTURE(out.norms.size(), out.norms.back(), out.local_nl_iters);
      REQUIRE(out.converged);
      REQUIRE(out.local_nl_iters == 0);
   }
}

TEST_CASE("One NPC step is exact on a linear problem",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // The check that falsifies the whole construction if the elimination
   // algebra is wrong, and the reason to run it first. NPC solves the
   // JACOBIAN system exactly by hybridized elimination, so on a problem whose
   // full (q, u, lambda) system is linear -- amp = 0 leaves the HDG face
   // terms, which are linear in the potential -- one Newton step must land on
   // the solution from any starting point, and the second residual must be
   // round-off rather than merely small.
   //
   // It also pins the two blocks that are easy to swap silently: the trace row
   // of the Jacobian is [C' G | H] and the local rows take [C; E], so
   // NPCReduce() uses G and NPCRecover() uses E. Getting that wrong leaves a
   // consistent-looking iteration that converges to the wrong thing, or not at
   // all, and nothing else in the suite would notice.
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(gmode == GM::MatrixFree);

   PedestalHDG P(8, 1, 0.05, 0.0);
   const NPCOutcome out = RunNPC(P, 3, false, gmode);

   REQUIRE(out.norms.size() >= 2);
   CAPTURE(out.norms[0], out.norms[1]);
   REQUIRE(out.norms[0] > 1e-3);        // there was something to solve
   REQUIRE(out.norms[1] < 1e-12);       // and one step solved it
   // No element ran a nonlinear solve. This is the acceptance item that says
   // the method really is NPC and not a condensation in disguise.
   REQUIRE(out.local_nl_iters == 0);
}

TEST_CASE("NPC converges quadratically on the full residual",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // Quadratic convergence is what says the assembled Jacobian belongs to the
   // residual: a wrong Jacobian still converges, but linearly. Measured on the
   // pedestal source at a width both orderings handle:
   // 6.7e-01, 1.5e-02, 2.8e-04, 1.2e-07, 2.3e-14.
   //
   // Worth reading the split as well as the norm. After the first step the
   // TRACE residual sits at round-off and everything left is in the local
   // rows, every step, at every width. So an outer iteration judged on the
   // trace residual alone would report convergence at step one -- which is
   // what a caller meant by the reduced test being "judged on half of what it
   // is solving", and it is a property of the system rather than of any
   // implementation.
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(gmode == GM::MatrixFree);

   PedestalHDG P(12, 1, 0.05);
   const NPCOutcome out = RunNPC(P, 8, false, gmode);

   CAPTURE(out.norms.size(), out.local_nl_iters);
   REQUIRE(out.converged);
   REQUIRE(out.local_nl_iters == 0);
   REQUIRE(out.norms.size() <= 7);

   // r_{k+1} <= C r_k^2 with a generous C, checked only while the iterate is
   // far enough from round-off for the ratio to mean anything.
   for (std::size_t k = 0; k + 1 < out.norms.size(); k++)
   {
      if (out.norms[k] < 1e-5) { continue; }
      CAPTURE(k, out.norms[k], out.norms[k+1]);
      REQUIRE(out.norms[k+1] < 20.0 * out.norms[k] * out.norms[k]);
   }
}

TEST_CASE("NPC's two gradient modes are the same operator",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // GradientMode::MatrixFree never builds the global trace matrix -- it
   // applies S = H - C' M^-1 [C; E] one element at a time -- so a caller with
   // no room for the reduced matrix can still run NPC. Both modes must be the
   // same operator, or the choice is a change of method.
   PedestalHDG Pa(12, 1, 0.05);
   PedestalHDG Pf(12, 1, 0.05);
   const NPCOutcome a = RunNPC(Pa, 8, false, GM::Assembled);
   const NPCOutcome f = RunNPC(Pf, 8, false, GM::MatrixFree);

   REQUIRE(a.converged);
   REQUIRE(f.converged);
   REQUIRE(a.norms.size() == f.norms.size());
   for (std::size_t k = 0; k < a.norms.size(); k++)
   {
      // Only while the residual is above round-off. Past that both iterations
      // have converged and the difference between two round-off values is not
      // a property of anything: the last iterate here is 2.2510e-14 against
      // 2.2508e-14, which a relative test would call a four-order discrepancy.
      if (a.norms[k] < 1e-12) { continue; }
      CAPTURE(k, a.norms[k], f.norms[k]);
      // The matrix-free trace solve is a Krylov method to 1e-14 rather than a
      // direct one, so the iterates agree to that and not bitwise. In practice
      // every iterate above round-off agrees to all six printed digits.
      REQUIRE(std::abs(a.norms[k] - f.norms[k]) <= 1e-8 * a.norms[k]);
   }
}

TEST_CASE("NPC solves stiff problems LineariseThenCondense cannot",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // The payoff, and the reason the parity gap was mis-attributed. These are
   // configurations where CondenseThenLinearise converges and
   // LineariseThenCondense does not, and the doxygen used to say closing them
   // needed "the local step globalised". NPC has no local nonlinear iteration
   // to globalise; what it needs is a line search on the OUTER step, which is
   // well defined because the fields are Newton state and scale with it.
   //
   // Undamped, NPC wanders on these exactly as any cold Newton does. With
   // backtracking on the full residual, three of the four fall: k = 2 n = 8 in
   // 13 steps, k = 3 n = 12 in 10, k = 1 n = 32 in 17, all to below 1e-12 and
   // all with zero local nonlinear iterations. The fourth, k = 1 n = 24 at
   // 0.003, stalls at 2.9e-03 with the line search grinding -- ordinary Newton
   // stagnation, not an artefact of the ordering, and CondenseThenLinearise
   // needs 22 iterations there.
   const int idx = GENERATE(0, 1, 2);
   const int n     = (idx == 0) ? 8     : (idx == 1) ? 12    : 32;
   const int order = (idx == 0) ? 2     : (idx == 1) ? 3     : 1;
   const real_t sg = (idx == 0) ? 0.003 : (idx == 1) ? 0.002 : 0.003;
   CAPTURE(n, order, sg);

   PedestalHDG P(n, order, sg);
   const NPCOutcome out = RunNPC(P, 40, true, GM::Assembled);

   CAPTURE(out.norms.size(), out.norms.back(), out.local_nl_iters);
   REQUIRE(out.converged);
   REQUIRE(out.local_nl_iters == 0);
}

TEST_CASE("NewtonSolver drives NPC with no special support",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;

   // DarcyNPCOperator is an Operator over the FULL (q, u, lambda) vector, so
   // the fields are in x and an ordinary NewtonSolver carries them with no
   // special support: its convergence test is on the full residual because
   // that is what the operator returns, and its line search would scale the
   // fields and the trace together because they are one vector.
   //
   // What has nowhere to keep the fields is an operator on the TRACE alone.
   // That was a statement about the deleted trace-only mode and not about
   // NewtonSolver, and this file's own notes had it the wrong way round for a
   // while.
   //
   // The iterates must be the hand-written loop's, exactly: the wrapper is
   // bookkeeping, not a method.
   PedestalHDG Pn(12, 1, 0.05);
   PedestalHDG Pr(12, 1, 0.05);

   BlockVector load = Pn.load();
   DarcyNPCOperator npc(*Pn.darcy.GetHybridization(), Pn.offs, load);
   UMFPackSolver trace;
   DarcyNPCSolver lin(trace);

   NormHistory hist;
   NewtonSolver nw;
   nw.SetOperator(npc);
   nw.SetSolver(lin);
   nw.SetRelTol(0.0);
   nw.SetAbsTol(1e-12);
   nw.SetMaxIter(20);
   nw.SetPrintLevel(-1);
   nw.SetMonitor(hist);

   Vector zero(npc.Height());
   zero = 0.0;
   Vector x(Pn.sol.GetData(), npc.Height());
   nw.Mult(zero, x);

   CAPTURE(nw.GetNumIterations(), nw.GetFinalNorm());
   REQUIRE(nw.GetConverged());
   REQUIRE(Pn.darcy.GetHybridization()->GetNumLocalNLIterations() == 0);

   const NPCOutcome raw = RunNPC(Pr, 20, false,
                                 DarcyHybridization::GradientMode::Assembled);
   REQUIRE(raw.converged);
   // NormHistory records NewtonSolver's extra call with final = true, so it
   // holds one more entry than the loop's own list; the overlap is what is
   // being compared.
   REQUIRE(hist.norms.size() >= raw.norms.size());
   for (std::size_t k = 0; k < raw.norms.size(); k++)
   {
      if (raw.norms[k] < 1e-12) { continue; }
      CAPTURE(k, hist.norms[k], raw.norms[k]);
      REQUIRE(std::abs(hist.norms[k] - raw.norms[k]) <= 1e-10 * raw.norms[k]);
   }
}


TEST_CASE("The line search earns its place on the pedestal, and says which",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // Section 6 of doc/HDG-ORDERING-API.md recommends backtracking on the full
   // residual, and until this case existed the evidence for that lived in a
   // commit message and a scratch probe. It is here because meq measured the
   // OPPOSITE on their discretisation -- the same line search made every case
   // worse, including five that converge undamped -- so the recommendation is
   // problem-dependent and the branch needs its half of that on record.
   //
   // These two configurations are the ones where the line search decides the
   // outcome: undamped NPC wanders and backtracking reaches 1e-12. Note the
   // third of the stiff set, k = 3 n = 12, converges BOTH ways in 12 and 10
   // steps -- an earlier version of NPCResidual()'s doxygen claimed undamped
   // NPC wanders on all four, and sweeping them is what disproved it.
   //
   // If someone improves NPC so that the undamped run converges here, this
   // test fails, and that failure is the finding rather than a nuisance: it
   // would mean section 6's recommendation no longer rests on anything and
   // should be rewritten.
   const int idx = GENERATE(0, 1);
   const int n     = (idx == 0) ? 8     : 32;
   const int order = (idx == 0) ? 2     : 1;
   const real_t sg = 0.003;
   CAPTURE(n, order, sg);

   PedestalHDG Pd(n, order, sg);
   const NPCOutcome damped = RunNPC(Pd, 40, true, GM::Assembled);
   CAPTURE(damped.norms.size(), damped.norms.back());
   REQUIRE(damped.converged);
   REQUIRE(damped.local_nl_iters == 0);

   PedestalHDG Pu(n, order, sg);
   const NPCOutcome undamped = RunNPC(Pu, 40, false, GM::Assembled);
   CAPTURE(undamped.norms.size(), undamped.norms.back());
   REQUIRE_FALSE(undamped.converged);
}

TEST_CASE("ComputeSolution reproduces the fields NPC already holds",
          "[DarcyForm][NonlinearDarcy][HDG][NPC]")
{
   using namespace darcy_npc;
   using GM = DarcyHybridization::GradientMode;

   // ComputeSolution() reconstructs the fields from the trace, which is what
   // condensation wants and what NPC does not need -- under NPC the fields are
   // Newton state and the back-substitution is redundant. Redundant is not the
   // same as wrong, and until this case existed it was simply unchecked.
   //
   // At the NPC solution the two must agree, and for a reason worth stating:
   // NPC converges when the FULL residual vanishes, and the local rows of that
   // residual are exactly the local problem ComputeSolution() solves given the
   // trace. So agreement here is not a coincidence of this problem; a
   // disagreement would mean one of the two is solving something else.
   PedestalHDG P(12, 2, 0.02);
   DarcyHybridization &dh = *P.darcy.GetHybridization();

   const NPCOutcome out = RunNPC(P, 40, true, GM::Assembled);
   CAPTURE(out.norms.size(), out.norms.back());
   REQUIRE(out.converged);

   // RunNPC advances P.sol and P.X in place, so they now hold NPC's answer.
   BlockVector npc_fields(P.darcy.GetOffsets());
   npc_fields = P.state();

   BlockVector recovered(P.darcy.GetOffsets());
   recovered = 0.0;
   BlockVector load = P.load();
   dh.ComputeSolution(load, P.X, recovered);

   for (int b = 0; b < 2; b++)
   {
      Vector diff(recovered.GetBlock(b));
      diff -= npc_fields.GetBlock(b);
      const real_t scale = std::max(npc_fields.GetBlock(b).Norml2(), 1e-30);
      CAPTURE(b, diff.Norml2(), scale);
      REQUIRE(diff.Norml2() <= 1e-8 * scale);
   }
}


#ifdef MFEM_USE_MPI

namespace darcy_npc
{

/** @brief The pedestal problem on a ParMesh, for NPC on more than one rank.

    The flux and the potential are L2, so they are rank-local and their L-dofs
    are their true dofs; the trace lives on the skeleton and a face on the
    partition boundary is shared. **So the only thing NPC has to get right in
    parallel is the trace**, and that is what the case below is aimed at. */
struct ParPedestalHDG
{
   Mesh serial;
   ParMesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   ParFiniteElementSpace Vh, Wh, Mh;
   ConstantCoefficient one;
   ConstantTau tau;
   ParDarcyForm darcy;
   Array<int> all, ess_flux, offs;
   BlockVector sol, rhs;
   Vector X;

   ParPedestalHDG(int n, int order, real_t sigma, real_t amp)
      : serial(Mesh::MakeCartesian2D(n, n, Element::TRIANGLE, false, 0.8, 1.2)),
        mesh(MPI_COMM_WORLD, serial),
        u_coll(order, 2, BasisType::GaussLobatto),
        p_coll(order, 2, BasisType::GaussLobatto),
        t_coll(order, 2),
        Vh(&mesh, &u_coll, 2), Wh(&mesh, &p_coll), Mh(&mesh, &t_coll),
        one(1.0), tau(1.0), darcy(&Vh, &Wh), offs(4)
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));

      auto *interior = new HDGDiffusionIntegrator(one, 1.0);
      auto *boundary = new HDGDiffusionIntegrator(one, 1.0);
      interior->SetStabilization(tau);
      boundary->SetStabilization(tau);

      all.SetSize(mesh.bdr_attributes.Max());
      all = 1;

      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new PedestalSource(amp, sigma));
      Mnl_p->AddInteriorFaceIntegrator(interior);
      Mnl_p->AddBdrFaceIntegrator(boundary, all);

      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      B->AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
      B->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-2.0)), all);

      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(),
                                ess_flux);
      darcy.GetHybridization()->SetEssentialBC(all);
      darcy.GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton, 100, 1e-12, 1e-16, -1);
      darcy.Assemble();
      darcy.Finalize();

      // The trace block is sized on TRUE dofs, which is what NPC's interface
      // takes; the other two are L2 and are the same either way.
      offs[0] = 0;
      offs[1] = Vh.GetVSize();
      offs[2] = Wh.GetVSize();
      offs[3] = Mh.GetTrueVSize();
      offs.PartialSum();
      sol.Update(offs);
      rhs.Update(offs);
      sol = 0.0;
      rhs = 0.0;

      FunctionCoefficient ramp([](const Vector &x)
      { return 0.5*(x(1) - 0.6); });
      ParGridFunction pgf(&Wh), tgf(&Mh);
      pgf.ProjectCoefficient(ramp);
      sol.GetBlock(1) = pgf;
      tgf = 0.0;
      tgf.ProjectBdrCoefficient(ramp, all);
      tgf.ParallelProject(sol.GetBlock(2));

      X.MakeRef(sol, offs[2], offs[3] - offs[2]);
   }

   BlockVector load() { return BlockVector(rhs, darcy.GetOffsets()); }
   BlockVector state() { return BlockVector(sol, darcy.GetOffsets()); }
};

} // namespace darcy_npc

TEST_CASE("One NPC step is exact on a linear problem, in parallel",
          "[DarcyForm][NonlinearDarcy][HDG][NPC][Parallel]")
{
   using namespace darcy_npc;

   // The first [Parallel] Darcy case this branch has had, and it is aimed at
   // the only thing NPC does differently on more than one rank: the trace row
   // is shared, so it is prolonged on the way in and assembled on the way out,
   // while the L2 flux and potential need no mapping at all.
   //
   // A linear problem is the sharp instrument for that. One NPC step must land
   // on the solution exactly, so if any of the four prolongation/assembly
   // steps is wrong -- residual, gradient, reduction, recovery -- the second
   // residual is not round-off and this fails. A convergence-rate check could
   // not tell a mis-assembled trace from a merely slow one.
   CAPTURE(Mpi::WorldSize());

   ParPedestalHDG P(8, 1, 0.05, 0.0);
   DarcyHybridization &dh = *P.darcy.GetHybridization();

   BlockVector b = P.load(), x = P.state();
   Vector &x_tr = P.X;
   BlockVector r(P.darcy.GetOffsets()), dx(P.darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;

   // The flux and potential dofs are rank-local and disjoint, and the trace is
   // in true dofs, so a global sum of the local dot products is the norm.
   auto full_norm = [](const BlockVector &rl, const Vector &rt)
   {
      return std::sqrt(InnerProduct(MPI_COMM_WORLD, rl, rl)
                       + InnerProduct(MPI_COMM_WORLD, rt, rt));
   };

   dh.NPCResidual(b, x, x_tr, r, r_tr);
   const real_t n0 = full_norm(r, r_tr);

   Operator &S = dh.NPCGradient(x, x_tr);
   dh.NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   HypreParMatrix *Sp = dynamic_cast<HypreParMatrix*>(&S);
   REQUIRE(Sp != nullptr);
   {
      HypreBoomerAMG amg(*Sp);
      amg.SetPrintLevel(0);
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetOperator(*Sp);
      gmres.SetPreconditioner(amg);
      gmres.SetKDim(200);
      gmres.SetMaxIter(2000);
      gmres.SetRelTol(1e-14);
      gmres.SetAbsTol(0.0);
      gmres.SetPrintLevel(-1);
      gmres.Mult(b_tr, dtr);
   }

   dh.NPCRecover(r, dtr, dx);
   x += dx;
   x_tr += dtr;

   dh.NPCResidual(b, x, x_tr, r, r_tr);
   const real_t n1 = full_norm(r, r_tr);

   CAPTURE(n0, n1);
   REQUIRE(n0 > 1e-3);                              // something to solve
   // 1.0e-11 with headroom; it measures below 1e-13 here. A wrong
   // prolongation or a missing assembly gives O(1), not 1e-10, so nothing is
   // given up by not pinning the last two digits across hypre versions.
   REQUIRE(n1 < 1e-11);
   REQUIRE(dh.GetNumLocalNLIterations() == 0);
}

#endif // MFEM_USE_MPI
