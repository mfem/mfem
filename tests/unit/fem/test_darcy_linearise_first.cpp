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

#include <cstring>
#include <vector>

using namespace mfem;

namespace darcy_linearise_first
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

/// Solve the same hybridized nonlinear problem in one ordering or the other.
Outcome Solve(Mesh &mesh, int order, real_t eps,
              DarcyHybridization::NLOrdering ordering, int max_it = 20,
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
   dh->SetNonlinearOrdering(ordering);
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

} // namespace darcy_linearise_first

TEST_CASE("Linearise-then-condense reaches the same solution",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;

   const int order = GENERATE(0, 1);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   SECTION("a linear problem is solved in one step either way")
   {
      // With the state dependence switched off the local problems are linear,
      // and the first Newton step has to land on the answer whichever way the
      // two operations are ordered.
      Outcome old_way = Solve(mesh, order, 0.0, Ord::CondenseThenLinearise);
      Outcome new_way = Solve(mesh, order, 0.0, Ord::LineariseThenCondense);

      REQUIRE(old_way.norms.size() >= 2);
      REQUIRE(new_way.norms.size() >= 2);
      CAPTURE(old_way.norms[0], old_way.norms[1]);
      CAPTURE(new_way.norms[0], new_way.norms[1]);
      REQUIRE(new_way.norms[0] > 1e-3);
      REQUIRE(new_way.norms[1] < 1e-10 * new_way.norms[0]);
      REQUIRE(new_way.p.Normlinf() > 1e-4);

      Vector d(new_way.p);
      d -= old_way.p;
      REQUIRE(d.Norml2() < 1e-10 * old_way.p.Norml2());
   }

   SECTION("a nonlinear problem reaches the same discrete solution")
   {
      Outcome old_way = Solve(mesh, order, 0.5, Ord::CondenseThenLinearise);
      Outcome new_way = Solve(mesh, order, 0.5, Ord::LineariseThenCondense);

      REQUIRE(old_way.converged);
      REQUIRE(new_way.converged);
      REQUIRE(new_way.p.Normlinf() > 1e-4);

      // The two orderings are two ways of solving one discrete problem, so
      // where both converge they must agree on its solution.
      Vector d(new_way.p);
      d -= old_way.p;
      CAPTURE(d.Norml2(), old_way.p.Norml2());
      REQUIRE(d.Norml2() < 1e-9 * old_way.p.Norml2());
   }

   SECTION("no element runs a nonlinear solve")
   {
      // The count is what says the ordering really changed, rather than the
      // answer merely coming out the same.
      Outcome old_way = Solve(mesh, order, 0.5, Ord::CondenseThenLinearise);
      Outcome new_way = Solve(mesh, order, 0.5, Ord::LineariseThenCondense);

      CAPTURE(old_way.local_nl_iters, new_way.local_nl_iters);
      REQUIRE(old_way.local_nl_iters > 0);
      REQUIRE(new_way.local_nl_iters == 0);
   }

   SECTION("the outer iteration converges quadratically")
   {
      Outcome new_way = Solve(mesh, order, 0.5, Ord::LineariseThenCondense);
      REQUIRE(new_way.converged);

      // Once the iteration is in the asymptotic regime each residual is at
      // worst the square of the one before it, up to a constant. Only the
      // steps that are neither the first nor already at round-off say
      // anything, so the window is taken between those.
      // The lower cut used to be 1e-11 and excluded every step once Mult()
      // began linearising at its own argument: the history went from
      // 7.375e-02 2.241e-05 9.08e-09 to 7.375e-02 2.241e-05 4.003e-12, which
      // is CondenseThenLinearise's 4.00e-12 to three digits. A window that
      // empties because the thing it measures improved reports a failure, so
      // it is taken down to round-off instead.
      int checked = 0;
      for (size_t k = 1; k + 1 < new_way.norms.size(); k++)
      {
         const real_t r0 = new_way.norms[k-1], r1 = new_way.norms[k];
         if (r0 > 1e-3 || r1 < 1e-14) { continue; }
         CAPTURE(k, r0, r1);
         REQUIRE(r1 < 100.0 * r0 * r0);
         checked++;
      }
      std::string hist;
      for (real_t v : new_way.norms)
      {
         char b[32];
         std::snprintf(b, sizeof(b), " %.3e", v);
         hist += b;
      }
      CAPTURE(new_way.norms.size(), hist);
      REQUIRE(checked >= 1);
   }
}

TEST_CASE("The reduced operator is a function of the trace",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;

   // A residual asked for twice at the same trace has to answer the same
   // thing, whatever happened in between. It did not: every GetGradient()
   // advanced the linearisation point, which is a local Newton step at fixed
   // trace, so the operator was a function of its own history. On a stiff
   // local problem those ungloablised steps ran away -- the residual grew from
   // 1.9e+01 to 4.2e+03 between two calls at one trace, and again by 10^4 at
   // the next. The nonlinearity here is on the potential mass form, which is
   // the path that had no coverage.
   const real_t c = GENERATE(1.0, 1.0e2, 1.0e4, 1.0e5);
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   CAPTURE(c, ordering == Ord::LineariseThenCondense);

   const int order = 1;
   Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE);
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim);
   FiniteElementSpace Wh(&mesh, &p_coll);
   FiniteElementSpace Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new SquareSource(c));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_bdr(mesh.bdr_attributes.Max()), ess_flux;
   ess_bdr = 1;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   darcy.GetHybridization()->SetNonlinearOrdering(ordering);
   darcy.GetHybridization()->SetEssentialBC(ess_bdr);
   darcy.Assemble();

   BlockVector sol(darcy.GetOffsets()), rhs(darcy.GetOffsets());
   sol = 0.0;
   rhs = 0.0;

   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux, sol, rhs, R, X, B, true);

   Operator &op = *R.Ptr();
   Vector x(op.Height()), r1(op.Height()), r2(op.Height()), r3(op.Height());
   x.Randomize(1);
   x *= 0.1;

   op.Mult(x, r1);
   op.GetGradient(x);            // the same trace
   op.Mult(x, r2);
   op.GetGradient(x);            // and again
   op.Mult(x, r3);

   CAPTURE(r1.Norml2(), r2.Norml2(), r3.Norml2());
   REQUIRE(r1.Norml2() > 0.0);
   REQUIRE(BitwiseEqual(r2, r1));
   REQUIRE(BitwiseEqual(r3, r2));
}

namespace darcy_linearise_first
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
                 DarcyHybridization::NLOrdering ordering,
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
      darcy.GetHybridization()->SetNonlinearOrdering(ordering);
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

} // namespace darcy_linearise_first

TEST_CASE("The reduced gradient is the derivative of the reduced residual",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
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
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   const real_t h = GENERATE(1.0e-4, 1.0e-5);
   // Both ways of producing the gradient have to be the derivative of the same
   // residual. The matrix-free one applies the Schur complement instead of
   // assembling it, and used to leave out d(flux residual)/dp and the diagonal
   // policy's regularisation, either of which makes it a different operator.
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(c, h, ordering == Ord::LineariseThenCondense,
           gmode == GM::MatrixFree);

   SemilinearHDG P(8, 1, c, ordering, gmode);
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

TEST_CASE("The gradient matches a difference taken in the caller's order",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
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
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   const real_t h = 1.0e-5;
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(c, h, ordering == Ord::LineariseThenCondense,
           gmode == GM::MatrixFree);

   SemilinearHDG P(8, 1, c, ordering, gmode);
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

TEST_CASE("The reduced residual survives the linearisation advancing",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
   using GM = DarcyHybridization::GradientMode;

   // "The reduced operator is a function of the trace" asks for the residual
   // twice at a trace the linearisation is already at, where GetGradient() is
   // idempotent and the answers agree bit for bit. This asks the same thing at
   // a trace the linearisation ADVANCES to, which is what happens at every
   // Newton iteration after the first, and there the answers cannot agree
   // exactly: the retained fields move, and the residual is evaluated at
   // fields substituted from them.
   //
   // What can be required is that the move is second order -- that the
   // retained fields carry a local residual small enough for the substitution
   // to be insensitive to which of them it started from. That holds only
   // because the linearisation is formed with a local correction applied,
   // including the very first one; retaining the caller's raw initial guess
   // instead put this at 3.3e-05 rather than 5.0e-10.
   const real_t c = GENERATE(1.0, 1.0e1);
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(c, ordering == Ord::LineariseThenCondense,
           gmode == GM::MatrixFree);

   SemilinearHDG P(8, 1, c, ordering, gmode);
   Operator &op = P.op();
   const int m = op.Height();

   Vector x0(m), x1(m), r(m), ra(m), rb(m);
   x0.Randomize(1);
   x0 *= 0.1;
   x1.Randomize(5);
   x1 *= 0.1;

   op.Mult(x0, r);
   op.GetGradient(x0);      // the linearisation sits at x0
   op.Mult(x1, ra);         // a new trace, the old linearisation
   op.GetGradient(x1);      // the linearisation advances to x1
   op.Mult(x1, rb);         // the same trace as ra

   Vector d(rb);
   d -= ra;
   const real_t rel = d.Norml2()/std::max(real_t(1e-300), ra.Norml2());

   CAPTURE(rel, ra.Norml2(), rb.Norml2());
   REQUIRE(ra.Norml2() > 0.0);
   REQUIRE(rel < 1.0e-7);
}

TEST_CASE("The three trace solves reach the same solution",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
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
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   const real_t src = 4.0;
   CAPTURE(c, ordering == Ord::LineariseThenCondense);

   // The reference: assembled and solved directly.
   Vector p_ref;
   int ref_its = -1;
   {
      SemilinearHDG P(6, 1, c, ordering, GM::Assembled, src);
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

   SemilinearHDG P(6, 1, c, ordering,
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
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
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
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   CAPTURE(eps, ordering == Ord::LineariseThenCondense);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   Outcome assembled = Solve(mesh, 1, eps, ordering, 20, GM::Assembled);
   Outcome matfree = Solve(mesh, 1, eps, ordering, 20, GM::MatrixFree);

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

namespace darcy_linearise_first
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
              DarcyHybridization::NLOrdering ordering,
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
      dh->SetNonlinearOrdering(ordering);
      dh->SetGradientMode(gmode);

      x.Update(darcy.GetOffsets());
      x = 0.0;
      darcy.FormLinearSystem(ess, x, op, X, RHS, true);
   }
};

} // namespace darcy_linearise_first

TEST_CASE("Both reduced gradients are the derivative on a coupled flux law",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
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
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   const auto gmode = GENERATE(GM::Assembled, GM::MatrixFree);
   CAPTURE(eps, ordering == Ord::LineariseThenCondense,
           gmode == GM::MatrixFree);

   CoupledHDG P(4, 1, eps, ordering, gmode);
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
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
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
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   CAPTURE(eps, ordering == Ord::LineariseThenCondense);

   CoupledHDG A(4, 1, eps, ordering, GM::Assembled);
   CoupledHDG M(4, 1, eps, ordering, GM::MatrixFree);
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

namespace darcy_linearise_first
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

} // namespace darcy_linearise_first

TEST_CASE("A Jacobian-free solve gets the right answer without being told",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;
   using GM = DarcyHybridization::GradientMode;

   // This used to be the sharpest statement of a caller obligation, and it is
   // now the sharpest statement that there is no obligation.
   //
   // LineariseThenCondense expands about a retained linearisation. That point
   // used to advance only in GetGradient(), so a Jacobian-free Newton-Krylov
   // solve -- which differences the residual and asks for no gradient at all
   // -- converged onto the root of a *frozen* operator: the residual reached
   // round-off and reported success with an answer wrong in the fourth digit.
   // The section below measured exactly that, and required the error to be
   // large.
   //
   // Mult() now linearises at its own argument, so the frozen operator cannot
   // arise, and the same solve lands on the reference. What the section
   // requires is therefore inverted, and the error it now measures is 2.5e-15
   // where it used to be over 1e-7.
   const real_t c = 5.0, src = 4.0;

   Vector p_ref;
   {
      SemilinearHDG P(8, 1, c, Ord::CondenseThenLinearise, GM::Assembled, src);
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
      P.darcy.RecoverFEMSolution(P.X, P.sol);
      p_ref = P.sol.GetBlock(1);
   }
   REQUIRE(p_ref.Normlinf() > 1e-4);

   SECTION("no gradient is ever asked for, and the answer is right anyway")
   {
      SemilinearHDG P(8, 1, c, Ord::LineariseThenCondense, GM::Assembled, src);
      SolveJFNK(P);

      Vector r(P.op().Height());
      P.op().Mult(P.X, r);
      r -= P.B;
      P.darcy.RecoverFEMSolution(P.X, P.sol);
      Vector d(P.sol.GetBlock(1));
      d -= p_ref;
      const real_t err = d.Norml2()/p_ref.Norml2();

      // Converged by every measure the solver has, and now also correct.
      CAPTURE(r.Norml2(), err);
      REQUIRE(r.Norml2() < 1e-11);
      REQUIRE(err < 1e-10);
   }

   SECTION("the other ordering needs nothing, retaining nothing")
   {
      SemilinearHDG P(8, 1, c, Ord::CondenseThenLinearise, GM::Assembled, src);
      SolveJFNK(P);

      P.darcy.RecoverFEMSolution(P.X, P.sol);
      Vector d(P.sol.GetBlock(1));
      d -= p_ref;
      CAPTURE(d.Norml2()/p_ref.Norml2());
      REQUIRE(d.Norml2() < 1e-9 * p_ref.Norml2());
   }
}

namespace darcy_linearise_first
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

   PedestalHDG(int n, int order, real_t sigma,
               DarcyHybridization::NLOrdering ordering)
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
      Mnl_p->AddDomainIntegrator(new PedestalSource(1.0, sigma));
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
      darcy.GetHybridization()->SetNonlinearOrdering(ordering);
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
};

} // namespace darcy_linearise_first

TEST_CASE("The reduced gradient survives a stiff local problem",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;

   // The case above asks the same question of a c u^2 source and cannot reach
   // this. Swept to c = 1e6 on that configuration, LineariseThenCondense stays
   // at round-off until the local problem stops being solvable at all, so
   // there is no window in it where the defect below is visible.
   //
   // What the defect was: the linearisation point took a FIXED two
   // frozen-Jacobian local corrections, and GetGradient() is the Schur
   // complement of the Jacobian at the fields that leaves behind. It is the
   // derivative of Mult() only to the extent that those fields solve the local
   // problem -- what is left over enters as d(trace row)/d(fields) evaluated
   // at the wrong point -- so the gradient's accuracy was whatever two steps
   // happened to buy. The bound below is crossed by four orders on the
   // configuration this case runs: 1.086e-05 at sigma^2 = 0.05 and 9.299e-05
   // at 0.02, the same to seven digits at h = 1e-4 and h = 1e-5, which is
   // what says a Jacobian error rather than a differencing artefact.
   //
   // How it scales with the number of corrections, at n = 16:
   //
   //     steps   sigma^2 = 0.05    0.02        0.01
   //       1       1.98e-05      1.48e-04    1.13e-03
   //       2       2.99e-06      2.66e-05    3.06e-04     <- what shipped
   //       4       5.63e-08      1.29e-06    6.16e-05
   //       8       2.44e-11      6.25e-09    4.14e-06
   //      16       1.22e-11      1.30e-11    3.56e-08
   //
   // against 1e-10 or better for CondenseThenLinearise throughout. So on a
   // stiff source the outer Newton was solving with a Jacobian that did not
   // belong to its residual, and that is what a caller reported as this
   // ordering failing where the other one succeeds. The linearisation point
   // now iterates to the local solver's tolerance instead.
   const real_t sigma = GENERATE(0.05, 0.02);
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   const real_t h = GENERATE(1.0e-4, 1.0e-5);
   CAPTURE(sigma, h, ordering == Ord::LineariseThenCondense);

   PedestalHDG P(12, 1, sigma, ordering);
   Operator &op = P.op();
   const int m = op.Height();

   Array<int> ess_marker(m);
   ess_marker = 0;
   for (int i = 0; i < P.ess().Size(); i++) { ess_marker[P.ess()[i]] = 1; }
   REQUIRE(P.ess().Size() > 0);

   Vector v(m);
   v.Randomize(11);
   v -= 0.5;
   for (int i = 0; i < m; i++) { if (ess_marker[i]) { v(i) = 0.0; } }
   v *= 1.0/v.Norml2();

   // Newton's own order, and the only order in which the question is well
   // posed: the residual, then the gradient at that same trace, and the
   // gradient applied BEFORE the difference perturbs anything.
   //
   // Taking it again afterwards -- which is what the case above does, and
   // what a gradient check is naturally written as -- does not measure this.
   // Each perturbed Mult() leaves the linearisation at its own argument, so a
   // second GetGradient(x) drags it back to x and pays for more local
   // corrections on the way; with the fixed count that was four more than the
   // residual evaluations had, and it flattered the gradient by seven orders.
   // On exactly the configuration below, at sigma^2 = 0.05, the old code
   // reads 1.42e-12 with the gradient re-taken and 1.086e-05 without.
   // Newton never gets the second call, so neither does this test.
   Vector r0(m), Jv(m);
   op.Mult(P.X, r0);
   op.GetGradient(P.X).Mult(v, Jv);

   Vector xp(P.X), xm(P.X), rp(m), rm(m);
   xp.Add(h, v);
   xm.Add(-h, v);
   op.Mult(xp, rp);
   op.Mult(xm, rm);
   Vector fd(rp);
   fd -= rm;
   fd *= 1.0/(2.0*h);

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
   // Round-off on a central difference grows as 1/h, so this sits well above
   // the 1e-11 both orderings reach and far below the 3e-06 the fixed count
   // gave at the milder of the two widths.
   REQUIRE(rel < 1.0e-8);
}

TEST_CASE("A stiff source converges under both orderings",
          "[DarcyForm][NonlinearDarcy][HDG]")
{
   using namespace darcy_linearise_first;
   using Ord = DarcyHybridization::NLOrdering;

   // The acceptance a caller actually asked for, which is weaker than equal
   // iteration counts and stronger than anything the suite had: no problem
   // that converges under CondenseThenLinearise may fail under
   // LineariseThenCondense. The second row below is where that failed --
   // sixty iterations without converging against the other ordering's ten,
   // with the residual oscillating by an order of magnitude rather than
   // falling -- and it now takes eight. A direct trace solve, so the linear
   // solver is not in question.
   //
   // The requirement is NOT met everywhere and this test does not pretend
   // otherwise. Swept over 144 configurations (n = 8..24, k = 1..3, six
   // widths from 0.02 down to 0.001), the cases where CondenseThenLinearise
   // converges and this ordering does not went from six to three, with none
   // added; the three that remain are at widths where the local problem needs
   // more than a frozen Jacobian can give. Six further cases converged before
   // and no longer do, all of them ones where CondenseThenLinearise fails
   // too, so none is a parity failure -- but converging on a problem the
   // exact ordering cannot solve was never evidence of anything, and losing
   // it is not obviously a regression.
   const int idx = GENERATE(0, 1);
   const int n = (idx == 0) ? 24 : 32;
   const real_t sigma = (idx == 0) ? 0.005 : 0.003;
   const auto ordering = GENERATE(Ord::CondenseThenLinearise,
                                  Ord::LineariseThenCondense);
   CAPTURE(n, sigma, ordering == Ord::LineariseThenCondense);

   PedestalHDG P(n, 1, sigma, ordering);

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
