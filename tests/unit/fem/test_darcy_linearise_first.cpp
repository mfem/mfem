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
              DarcyHybridization::NLOrdering ordering, int max_it = 20)
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

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr op;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, op, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(2000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(0.0);
   lin.SetPreconditioner(prec);

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
      int checked = 0;
      for (size_t k = 1; k + 1 < new_way.norms.size(); k++)
      {
         const real_t r0 = new_way.norms[k-1], r1 = new_way.norms[k];
         if (r0 > 1e-3 || r1 < 1e-11) { continue; }
         CAPTURE(k, r0, r1);
         REQUIRE(r1 < 100.0 * r0 * r0);
         checked++;
      }
      CAPTURE(new_way.norms.size());
      REQUIRE(checked >= 1);
   }
}
