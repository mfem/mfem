// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//            ------------------------------------------------
//            Optimization miniapp: OptProblem / SubOptProblem
//            ------------------------------------------------
//
// Unit test of the optimization problem interface in this directory:
// registering fields and functionals, constraint ordering and bounds,
// derivatives and Jacobians, and subproblems over a subset of the fields
// (with and without essential dofs removed).
//
// Compile with: make optproblem_test
//
// Sample run:   optproblem_test

#include "mfem.hpp"
#include "optproblem.hpp"
#include <cmath>
#include <iostream>

using namespace mfem;

static bool Near(real_t a, real_t b) { return std::abs(a - b) < 1e-12; }

/// Derivative of G(u, v) = 1/2 u.u + 2 sum(v): dG = (u, 2).
class GDerivative : public Operator
{
public:
   GDerivative(int w) : Operator(w) {}
   void Mult(const Vector &, Vector &) const override
   { MFEM_ABORT("Use MultMV()."); }
   void MultMV(const MultiVector &X, MultiVector &Y) const override
   { Y[0] = X[0]; Y[1] = 2.0; }
};

/// G(u, v) = 1/2 u.u + 2 sum(v), a functional of two fields.
class GFunctional : public Functional
{
   std::shared_ptr<Operator> d;
public:
   GFunctional(int nu, int nv)
      : Functional(Array<int>({nu, nv})),
        d(std::make_shared<GDerivative>(nu + nv)) {}
   real_t Eval(const MultiVector &X) const override
   { return 0.5*(X[0]*X[0]) + 2.0*X[1].Sum(); }
   std::shared_ptr<Operator> Derivative() const override { return d; }
};

/// F(u) = 1/2 u^T M u with M the mass matrix, derivative u -> M u.
static FunctionalWrapper MassFunctional(SparseMatrix &M)
{
   return FunctionalWrapper(
             Array<int>({M.Height()}),
             [&M](const MultiVector &X) { return 0.5*M.InnerProduct(X[0], X[0]); },
             new LinearMVOperator(&M), true);
}

static void TestProblem(SparseMatrix &M)
{
   const int n = M.Height();
   FunctionalWrapper F = MassFunctional(M);

   OptProblem prob;
   const int u = prob.RegisterField(n);
   const int v = prob.RegisterField(3);
   Vector lb(n), ub(n);
   lb = -1.0; ub = 1.0;
   prob.SetFieldBounds(u, lb, ub);
   prob.UpdateFieldSize(v, 3, Array<int>({0, 2}));
   prob.RegisterFunctional(FunctionalRole::Objective(), Array<int>({u}), &F);
   prob.RegisterFunctional(FunctionalRole::Inequality(0.0, 1.0),
                           Array<int>({u}), &F);
   prob.RegisterFunctional(FunctionalRole::Equality(0.5), Array<int>({u}), &F);
   prob.RegisterFunctional(FunctionalRole::Inequality(-1.0, 2.0),
                           Array<int>({u}), &F);
   prob.Finalize();

   // equalities first, then inequalities in registration order
   MFEM_VERIFY(prob.GetNumEqualities() == 1 && prob.GetNumConstraints() == 3,
               "constraint counts");
   const Vector &lo = prob.GetConstraintsLowerBounds();
   MFEM_VERIFY(lo(0) == 0.5 && lo(1) == 0.0 && lo(2) == -1.0,
               "constraint ordering");

   Vector xu(n), xv(3);
   xu = 1.0; xv = 0.0;
   MultiVector X(xu, xv);
   // 1/2 * integral of 1 over [0, 1]
   MFEM_VERIFY(Near(prob.EvalObjective(X), 0.5), "objective value");

   MultiVector G;
   prob.EvalObjectiveDerivative(X, G);
   MFEM_VERIFY(G.NumBlocks() == 2 && Near(G[0].Sum(), 1.0) &&
               G[1].Norml2() == 0.0, "objective derivative");

   DenseMatrix JT;
   prob.EvalInequalityConstraintJacobianTranspose(X, JT);
   MFEM_VERIFY(JT.Height() == n + 3 && JT.Width() == 2, "inequality JT size");
   Vector col;
   JT.GetColumn(1, col);
   MFEM_VERIFY(Near(col.Sum(), 1.0), "inequality JT column");

   DenseMatrix JE;
   prob.EvalEqualityConstraintJacobianTranspose(X, JE);
   MFEM_VERIFY(JE.Width() == 1, "equality JT size");

   Vector c;
   prob.EvalConstraints(X, c);
   MFEM_VERIFY(c.Size() == 3 && Near(c(0), 0.5), "constraint values");
}

static void TestSubProblem(SparseMatrix &M)
{
   const int n = M.Height();
   FunctionalWrapper F = MassFunctional(M);
   GFunctional g(n, 3);

   OptProblem prob;
   const int u = prob.RegisterField(n, Array<int>({0, n-1}));
   const int v = prob.RegisterField(3);
   Vector lb(n), ub(n);
   for (int i = 0; i < n; i++) { lb(i) = -i; ub(i) = i; }
   prob.SetFieldBounds(u, lb, ub);
   const int f_obj = prob.RegisterFunctional(FunctionalRole::Objective(),
                                             Array<int>({u}), &F);
   const int f_g = prob.RegisterFunctional(FunctionalRole::Inequality(0.0, 1.0),
                                           Array<int>({u, v}), &g);
   prob.Finalize();

   Vector xu(n), xv(3);
   for (int i = 0; i < n; i++) { xu(i) = 1.0 + i; }
   xv = 0.5;
   MultiVector X(xu, xv);
   const real_t f_full = prob.EvalObjective(X);
   const real_t g_full = prob.EvalConstraint(0, X);
   MultiVector Gf;
   prob.EvalConstraintDerivative(0, X, Gf);

   // 1) keep essential dofs; the inactive field v comes from a copy of X
   {
      SubOptProblem s = prob.MakeSubProblem(Array<int>({u}),
                                            Array<int>({f_obj, f_g}), &X, true);
      xv = 100.0; // must not affect s (X was copied)
      Vector xu1(xu);
      MultiVector Xs(xu1);
      MFEM_VERIFY(Near(s.EvalObjective(Xs), f_full) &&
                  Near(s.EvalConstraint(0, Xs), g_full), "sub 1 values");
      MultiVector d;
      s.EvalConstraintDerivative(0, Xs, d);
      MFEM_VERIFY(d.NumBlocks() == 1 && d[0].Size() == n, "sub 1 derivative");
      for (int i = 0; i < n; i++)
      { MFEM_VERIFY(Near(d[0](i), Gf[0](i)), "sub 1 derivative entry"); }
      MFEM_VERIFY(s.GetEssDofs(0).Size() == 2, "sub 1 essential dofs");
      DenseMatrix JT;
      s.EvalInequalityConstraintJacobianTranspose(Xs, JT);
      MFEM_VERIFY(JT.Height() == n && JT.Width() == 1 &&
                  Near(JT(2, 0), Gf[0](2)), "sub 1 JT");
      xv = 0.5;
   }

   // 2) remove the essential dofs of u; their values come from X (shared)
   {
      SubOptProblem s = prob.MakeSubProblem(Array<int>({u}),
                                            Array<int>({f_obj, f_g}), &X,
                                            false, true);
      Vector xr(n-2);
      for (int i = 0; i < n-2; i++) { xr(i) = xu(i+1); }
      MultiVector Xs(xr);
      MFEM_VERIFY(Near(s.EvalObjective(Xs), f_full) &&
                  Near(s.EvalConstraint(0, Xs), g_full), "sub 2 values");
      MultiVector d;
      s.EvalConstraintDerivative(0, Xs, d);
      MFEM_VERIFY(d[0].Size() == n-2, "sub 2 derivative size");
      for (int i = 0; i < n-2; i++)
      { MFEM_VERIFY(Near(d[0](i), Gf[0](i+1)), "sub 2 derivative entry"); }
      MultiVector dO, dOf;
      s.EvalObjectiveDerivative(Xs, dO);
      prob.EvalObjectiveDerivative(X, dOf);
      for (int i = 0; i < n-2; i++)
      { MFEM_VERIFY(Near(dO[0](i), dOf[0](i+1)), "sub 2 objective derivative"); }
      MFEM_VERIFY(s.GetEssDofs(0).Size() == 0, "sub 2 essential dofs");
   }

   // 3) both fields active in a different order, constraint only
   {
      SubOptProblem s = prob.MakeSubProblem(Array<int>({v, u}),
                                            Array<int>({f_g}));
      MultiVector Xs(xv, xu);
      MFEM_VERIFY(!s.HasObjective() && Near(s.EvalConstraint(0, Xs), g_full),
                  "sub 3 values");
      MultiVector d;
      s.EvalConstraintDerivative(0, Xs, d);
      MFEM_VERIFY(d[0].Size() == 3 && Near(d[0](0), 2.0) &&
                  Near(d[1](3), Gf[0](3)), "sub 3 derivative");
      MFEM_VERIFY(s.ParentFieldId(0) == v, "sub 3 parent field id");
   }
}

int main(int argc, char *argv[])
{
   bool visualization = false; // accepted for the test harness; unused
   OptionsParser args(argc, argv);
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Unused; there is nothing to plot.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

   Mesh mesh = Mesh::MakeCartesian1D(4);
   H1_FECollection fec(1, 1);
   FiniteElementSpace fes(&mesh, &fec);
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new MassIntegrator);
   a.Assemble();
   a.Finalize();

   TestProblem(a.SpMat());
   TestSubProblem(a.SpMat());

   std::cout << "All tests passed.\n";
   return 0;
}
