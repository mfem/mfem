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
//
//      ----------------------------------------------------------------
//      Optimization Test: checks of the functionals and the problem API
//      ----------------------------------------------------------------
//
// Checks values, derivatives (against central differences) and Hessians of the
// functionals in linalg/functional.hpp and fem/fefunctional.hpp, the
// layout built by Problem::Finalize() in linalg/optimization.hpp, and
// SLBQPOptimizer. With MPI, the parallel checks use MPI_COMM_WORLD.
//
// Sample runs:  opt_test
//               opt_test -d debug
//               opt_test quadratic problem
//               mpirun -np 4 opt_test

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::optimization;

static int num_checks = 0, num_failed = 0;
static bool root = true;

static void Check(bool ok, const string &name)
{
   num_checks++;
   if (!ok)
   {
      num_failed++;
      cout << "FAILED: " << name << endl;
   }
}

static bool Near(real_t a, real_t b, real_t tol = 1e-10)
{
   return std::abs(a - b) <= tol * std::max<real_t>(1.0, std::abs(b));
}

static bool Near(const Vector &a, const Vector &b, real_t tol = 1e-10)
{
   if (a.Size() != b.Size()) { return false; }
   const real_t *ap = a.HostRead(), *bp = b.HostRead();
   for (int i=0; i<a.Size(); i++)
   {
      if (!Near(ap[i], bp[i], tol)) { return false; }
   }
   return true;
}

/// Symmetric positive definite n x n matrix M^T M + n I.
static void MakeSPD(int n, int seed, DenseMatrix &A)
{
   DenseMatrix M(n);
   for (int i=0; i<n; i++)
   {
      for (int j=0; j<n; j++) { M(i,j) = std::sin(seed + 3.0*i + 7.0*j); }
   }
   A.SetSize(n);
   MultAtB(M, M, A);
   for (int i=0; i<n; i++) { A(i,i) += n; }
}

/// Action of a DenseMatrix on vectors that may live on the device.
class DenseOperator : public Operator
{
   const DenseMatrix &A;
public:
   DenseOperator(const DenseMatrix &A_)
      : Operator(A_.Height(), A_.Width()), A(A_) {}
   void Mult(const Vector &x, Vector &y) const override
   {
      A.Mult(x.HostRead(), y.HostWrite());
   }
};

static Vector MakeVector(int n, int seed)
{
   Vector v(n);
   v.UseDevice(true);
   v.Randomize(seed);
   return v;
}

/// f_k(x) = x^T A_k x / 2 for k = 0, 1 on one field.
class TwoQuadratics : public GlobalFunctional
{
   const DenseOperator A0, A1;
   mutable Vector w;
public:
   TwoQuadratics(const DenseMatrix &A0_, const DenseMatrix &A1_)
      : GlobalFunctional(A0_.Width(), 2), A0(A0_), A1(A1_), w(A0_.Height()) {}

   void Eval(const MultiVector &x, Vector &fx) const override
   {
      fx.SetSize(2);
      A0.Mult(x[0], w); fx(0) = 0.5 * (x[0] * w);
      A1.Mult(x[0], w); fx(1) = 0.5 * (x[0] * w);
   }
   void ComponentDerivative(const MultiVector &x, int i,
                            MultiVector &df_i) const override
   {
      (i == 0 ? A0 : A1).Mult(x[0], df_i[0]);
   }
   const Operator *ComponentHessian(const MultiVector &x, int i) const override
   {
      return i == 0 ? &A0 : &A1;
   }
};

/// f(x0, x1) = z^T A z / 2 with z = (x0, x1).
class BlockQuadratic : public GlobalFunctional
{
   const DenseOperator A;
   mutable Vector z, w;
public:
   BlockQuadratic(int n0, int n1, const DenseMatrix &A_)
      : GlobalFunctional(Array<int>({n0, n1}), 1), A(A_), z(n0 + n1),
        w(n0 + n1) {}

   void Eval(const MultiVector &x, Vector &fx) const override
   {
      Concatenate(x);
      A.Mult(z, w);
      fx.SetSize(1);
      fx(0) = 0.5 * (z * w);
   }
   void ComponentDerivative(const MultiVector &x, int i,
                            MultiVector &df_i) const override
   {
      Concatenate(x);
      A.Mult(z, w);
      const int n0 = field_sizes[0], n1 = field_sizes[1];
      df_i[0] = Vector(w, 0, n0);
      df_i[1] = Vector(w, n0, n1);
   }
   const Operator *ComponentHessian(const MultiVector &x, int i) const override
   {
      return &A;
   }

private:
   void Concatenate(const MultiVector &x) const
   {
      z.SetVector(x[0], 0);
      z.SetVector(x[1], field_sizes[0]);
   }
};

/// A functional without outputs.
class NoOutput : public GlobalFunctional
{
public:
   NoOutput(int n) : GlobalFunctional(n, 0) {}
#ifdef MFEM_USE_MPI
   NoOutput(MPI_Comm comm, int n) : GlobalFunctional(comm, n, 0) {}
#endif
   void Eval(const MultiVector &x, Vector &fx) const override { fx.SetSize(0); }
   const Operator *ComponentHessian(const MultiVector &x, int i) const override
   {
      return nullptr;
   }
};

/// Concatenation of the blocks of @a x.
static Vector Concatenate(const MultiVector &x)
{
   Vector v(x.GetSizes().Sum());
   for (int b=0, offset=0; b<x.NumBlocks(); offset += x[b++].Size())
   {
      v.SetVector(x[b], offset);
   }
   return v;
}

/// max_k |(H d)_k - (g(x + h d) - g(x - h d))_k / (2h)| with
/// H = f.WeightedSumHessian(x, lambda) and g = sum_i lambda_i f_i'.
static real_t DiffHessianAgainstCentralDiff(const GlobalFunctional &f,
                                            const MultiVector &x,
                                            const Vector &lambda,
                                            const MultiVector &d, real_t h)
{
   const Array<int> sizes = x.GetSizes();
   MultiVector xp(sizes), xm(sizes), g(sizes);
   for (int b=0; b<sizes.Size(); b++)
   {
      xp[b] = x[b]; xp[b].Add(h, d[b]);
      xm[b] = x[b]; xm[b].Add(-h, d[b]);
   }
   Vector diff(sizes.Sum());
   diff = 0.0;
   for (int i=0; i<f.Height(); i++)
   {
      if (lambda(i) == 0.0) { continue; }
      f.ComponentDerivative(xp, i, g);
      diff.Add(lambda(i) / (2.0 * h), Concatenate(g));
      f.ComponentDerivative(xm, i, g);
      diff.Add(-lambda(i) / (2.0 * h), Concatenate(g));
   }
   Vector D = Concatenate(d), Hd(D.Size());
   const Operator *H = f.WeightedSumHessian(x, lambda);
   if (H) { H->Mult(D, Hd); }
   else { Hd = 0.0; }
   Hd -= diff;
   return Hd.Normlinf();
}

static void TestMultiVector()
{
   Vector a = MakeVector(3, 1), b = MakeVector(2, 2), c = MakeVector(4, 3);
   MultiVector mv(a, b, c);

   const Array<int> sizes = mv.GetSizes();
   Check(sizes.Size() == 3 && sizes[0] == 3 && sizes[1] == 2 && sizes[2] == 4,
         "MultiVector::GetSizes");

   MultiVector view = mv.GetBlockViews(Array<int>({2, 0}));
   Check(view.NumBlocks() == 2 && &view[0] == &c && &view[1] == &a,
         "MultiVector::GetBlockViews refers to the blocks");
   view[0](1) = 42.0;
   Check(c(1) == 42.0, "MultiVector::GetBlockViews writes through");

   const MultiVector &cmv = mv;
   const MultiVector cview = cmv.GetBlockViews(Array<int>({1}));
   Check(cview.NumBlocks() == 1 && &cview[0] == &b,
         "MultiVector::GetBlockViews const");
   Check(mv.GetBlockViews(Array<int>()).NumBlocks() == 0,
         "MultiVector::GetBlockViews empty");
}

static void TestWeightedSumOperator()
{
   DenseMatrix A0, A1;
   MakeSPD(3, 1, A0);
   MakeSPD(3, 2, A1);
   Array<const Operator *> ops(3);
   DenseOperator A0op(A0), A1op(A1);
   ops[0] = &A0op; ops[1] = nullptr; ops[2] = &A1op;
   Vector w(3);
   w(0) = 2.0; w(1) = 5.0; w(2) = -1.0;
   WeightedSumOperator op(ops, w);

   Vector x = MakeVector(3, 4), y(3), y0(3), y1(3);
   op.Mult(x, y);
   A0.Mult(x, y0);
   A1.Mult(x, y1);
   y0 *= 2.0;
   y0 -= y1;
   Check(op.Height() == 3 && op.Width() == 3 && Near(y, y0),
         "WeightedSumOperator::Mult");
}

static void TestLinearFunctional()
{
   Vector b = MakeVector(5, 1), xv = MakeVector(5, 2), dv(5);
   LinearFunctional f(&b);
   MultiVector x(xv), df(dv);

   Vector fx;
   f.Eval(x, fx);
   Check(fx.Size() == 1 && Near(fx(0), b * xv), "LinearFunctional::Eval");
   f.ComponentDerivative(x, 0, df);
   Check(Near(dv, b), "LinearFunctional::ComponentDerivative");
   Check(f.ComponentHessian(x, 0) == nullptr,
         "LinearFunctional::ComponentHessian");
   Vector lambda(1); lambda = 3.0;
   Check(f.WeightedSumHessian(x, lambda) == nullptr,
         "LinearFunctional::WeightedSumHessian");

   Vector d = MakeVector(5, 3);
   Check(DiffDerivativeAgainstCentralDiff(f, x, MultiVector(d), 1e-3) < 1e-10,
         "LinearFunctional central difference");
}

static void TestQuadraticFunctional()
{
   DenseMatrix A;
   MakeSPD(4, 1, A);
   Vector b = MakeVector(4, 2), xv = MakeVector(4, 3), dv(4), Ax(4);
   DenseOperator Aop(A);
   QuadraticFunctional f(&Aop, &b);
   MultiVector x(xv), df(dv);
   A.Mult(xv, Ax);

   Vector fx;
   f.Eval(x, fx);
   Check(Near(fx(0), 0.5 * (xv * Ax) - b * xv), "QuadraticFunctional::Eval");
   f.ComponentDerivative(x, 0, df);
   Vector g(Ax); g -= b;
   Check(Near(dv, g), "QuadraticFunctional::ComponentDerivative");
   Check(f.ComponentHessian(x, 0) == &Aop,
         "QuadraticFunctional::ComponentHessian");

   Vector lambda(1);
   lambda = 1.0;
   Check(f.WeightedSumHessian(x, lambda) == &Aop,
         "QuadraticFunctional::WeightedSumHessian, unit weight");
   lambda = 2.0;
   const Operator *H = f.WeightedSumHessian(x, lambda);
   Vector Hx(4);
   H->Mult(xv, Hx);
   Vector Ax2(Ax); Ax2 *= 2.0;
   Check(Near(Hx, Ax2), "QuadraticFunctional::WeightedSumHessian, weight 2");

   Vector d = MakeVector(4, 4);
   Check(DiffDerivativeAgainstCentralDiff(f, x, MultiVector(d), 1e-3) < 1e-8,
         "QuadraticFunctional central difference");
}

static void TestDefaultWeightedSumHessian()
{
   DenseMatrix A0, A1;
   MakeSPD(3, 1, A0);
   MakeSPD(3, 2, A1);
   TwoQuadratics f(A0, A1);
   Vector xv = MakeVector(3, 1);
   MultiVector x(xv);

   Vector lambda(2);
   lambda(0) = 2.0; lambda(1) = -1.0;
   const Operator *H = f.WeightedSumHessian(x, lambda);
   Vector y(3), y0(3), y1(3);
   H->Mult(xv, y);
   A0.Mult(xv, y0);
   A1.Mult(xv, y1);
   y0 *= 2.0; y0 -= y1;
   Check(Near(y, y0), "GlobalFunctional::WeightedSumHessian");

   Vector d = MakeVector(3, 2);
   Check(DiffDerivativeAgainstCentralDiff(f, x, MultiVector(d), 1e-3) < 1e-8,
         "TwoQuadratics central difference");
}

static void TestWeightedSumFunctional()
{
   DenseMatrix A0, A1;
   MakeSPD(2, 1, A0);
   MakeSPD(2, 2, A1);
   Vector b0 = MakeVector(3, 1);
   LinearFunctional lin(&b0);
   TwoQuadratics two(A0, A1);

   WeightedSumFunctional f(Array<int>({3, 2}));
   Vector w2(2), w1(1);
   w2(0) = 2.0; w2(1) = -1.0; w1(0) = 3.0;
   f.Add(0, Array<int>({0}), Vector(), &lin);
   f.Add(1, Array<int>({1}), Vector(), &two);
   f.Add(2, Array<int>({1}), w2, &two);
   f.Add(2, Array<int>({0}), w1, &lin);
   f.Finalize();
   Check(f.Height() == 4 && f.Width() == 5, "WeightedSumFunctional sizes");

   Vector x0 = MakeVector(3, 2), x1 = MakeVector(2, 3);
   MultiVector x(x0, x1);
   Vector fx, q;
   f.Eval(x, fx);
   two.Eval(MultiVector(x1), q);
   Vector expected(4);
   expected(0) = b0 * x0;
   expected(1) = q(0);
   expected(2) = q(1);
   expected(3) = 2.0 * q(0) - q(1) + 3.0 * (b0 * x0);
   Check(Near(fx, expected), "WeightedSumFunctional::Eval");

   Vector d0 = MakeVector(3, 4), d1 = MakeVector(2, 5);
   Check(DiffDerivativeAgainstCentralDiff(f, x, MultiVector(d0, d1), 1e-3)
         < 1e-8, "WeightedSumFunctional central difference");

   // Row 0 depends only on field 0.
   Vector g0(3), g1(2);
   MultiVector df(g0, g1);
   f.ComponentDerivative(x, 0, df);
   Check(Near(g0, b0) && g1.Normlinf() == 0.0,
         "WeightedSumFunctional::ComponentDerivative, row 0");
}

static void TestWeightedSumHessian()
{
   DenseMatrix A, B0, B1;
   MakeSPD(5, 1, A);
   MakeSPD(2, 2, B0);
   MakeSPD(2, 3, B1);
   Vector b0 = MakeVector(3, 1);
   LinearFunctional lin(&b0);
   TwoQuadratics two(B0, B1);
   BlockQuadratic bq(3, 2, A);

   WeightedSumFunctional f(Array<int>({3, 2}));
   Vector w2(2), w1(1);
   w2(0) = 2.0; w2(1) = -1.0; w1(0) = 3.0;
   f.Add(0, Array<int>({0}), Vector(), &lin);
   f.Add(1, Array<int>({1}), Vector(), &two);
   f.Add(2, Array<int>({1}), w2, &two);
   f.Add(2, Array<int>({0, 1}), w1, &bq);
   f.Add(2, Array<int>({0}), w1, &lin);
   f.Finalize();

   Vector x0 = MakeVector(3, 2), x1 = MakeVector(2, 3);
   Vector d0 = MakeVector(3, 4), d1 = MakeVector(2, 5);
   MultiVector x(x0, x1), d(d0, d1);
   Check(f.ComponentHessian(x, 0) == nullptr,
         "WeightedSumFunctional::ComponentHessian, linear row");

   bool ok = true;
   Vector D = Concatenate(d), y0(5), y1(5);
   for (int i=0; i<f.Height(); i++)
   {
      Vector e(f.Height());
      e = 0.0; e(i) = 1.0;
      ok = ok && DiffHessianAgainstCentralDiff(f, x, e, d, 1e-3) < 1e-8;
      const Operator *Hi = f.ComponentHessian(x, i);
      if (Hi == nullptr) { continue; }
      Hi->Mult(D, y0);
      f.WeightedSumHessian(x, e)->Mult(D, y1);
      ok = ok && Near(y0, y1);
   }
   Check(ok, "WeightedSumFunctional::ComponentHessian");

   Vector lambda = MakeVector(f.Height(), 6);
   Check(DiffHessianAgainstCentralDiff(f, x, lambda, d, 1e-3) < 1e-8,
         "WeightedSumFunctional::WeightedSumHessian");

   // A functional added twice with ownership is deleted once.
   WeightedSumFunctional owner(Array<int>({3}));
   const LinearFunctional *owned = new LinearFunctional(&b0);
   owner.Add(0, Array<int>({0}), Vector(), owned, true);
   owner.Add(1, Array<int>({0}), w1, owned, true);
   owner.Finalize();
   Check(owner.Height() == 2, "WeightedSumFunctional owns a functional once");
}

static void TestConstrainedFunctional()
{
   DenseMatrix A;
   MakeSPD(5, 1, A);
   BlockQuadratic bq(3, 2, A);
   Array<int> ess0({0, 2}), ess1({1});
   ConstrainedFunctional f({&ess0, &ess1}, &bq);

   Vector x0 = MakeVector(3, 1), x1 = MakeVector(2, 2);
   MultiVector x(x0, x1);
   Vector fx, fx_ref;
   f.Eval(x, fx);
   bq.Eval(x, fx_ref);
   Check(Near(fx, fx_ref), "ConstrainedFunctional::Eval");

   Vector g0(3), g1(2), r0(3), r1(2);
   MultiVector g(g0, g1), r(r0, r1);
   f.ComponentDerivative(x, 0, g);
   bq.ComponentDerivative(x, 0, r);
   r0(0) = 0.0; r0(2) = 0.0; r1(1) = 0.0;
   Check(Near(g0, r0) && Near(g1, r1),
         "ConstrainedFunctional::ComponentDerivative");

   // Hessian action: identity on the essential dofs (0, 2, 4).
   const Array<int> ess({0, 2, 4});
   Vector z = MakeVector(5, 3), zt(z), Hz(5), expected(5);
   for (int k : ess) { zt(k) = 0.0; }
   A.Mult(zt, expected);
   for (int k : ess) { expected(k) = z.HostRead()[k]; }
   f.ComponentHessian(x, 0)->Mult(z, Hz);
   Check(Near(Hz, expected), "ConstrainedFunctional::ComponentHessian");

   Vector lambda(1); lambda = 3.0;
   A.Mult(zt, expected);
   expected *= 3.0;
   for (int k : ess) { expected(k) = z.HostRead()[k]; }
   f.WeightedSumHessian(x, lambda)->Mult(z, Hz);
   Check(Near(Hz, expected), "ConstrainedFunctional::WeightedSumHessian");

   // Weights summing to zero still give a unit diagonal.
   DenseMatrix B0, B1;
   MakeSPD(3, 4, B0);
   MakeSPD(3, 5, B1);
   TwoQuadratics two(B0, B1);
   Array<int> ess_two({1});
   ConstrainedFunctional f2({&ess_two}, &two);
   Vector y = MakeVector(3, 6), Hy(3);
   MultiVector xy(y);
   lambda.SetSize(2); lambda(0) = 1.0; lambda(1) = -1.0;
   Vector e1(3); e1 = 0.0; e1(1) = 1.0;
   f2.WeightedSumHessian(xy, lambda)->Mult(e1, Hy);
   const real_t *hy = Hy.HostRead();
   Check(Near(hy[1], 1.0) && Near(hy[0], 0.0) && Near(hy[2], 0.0),
         "ConstrainedFunctional::WeightedSumHessian, zero weight sum");

   // The component Hessian stays valid over a weighted-sum call, and an
   // unchanged Hessian is not wrapped again.
   const Operator *H0 = f.ComponentHessian(x, 0);
   lambda.SetSize(1); lambda = 3.0;
   const Operator *HL = f.WeightedSumHessian(x, lambda);
   A.Mult(zt, expected);
   for (int k : ess) { expected(k) = z.HostRead()[k]; }
   H0->Mult(z, Hz);
   Check(HL != H0 && f.ComponentHessian(x, 0) == H0 && Near(Hz, expected),
         "ConstrainedFunctional Hessians coexist");

   Vector d0 = MakeVector(3, 7), d1 = MakeVector(2, 8);
   d0(0) = 0.0; d0(2) = 0.0; d1(1) = 0.0;
   Check(DiffDerivativeAgainstCentralDiff(f, x, MultiVector(d0, d1), 1e-3)
         < 1e-8, "ConstrainedFunctional central difference, free directions");
}

/// x^T A x / 2 - l(x) for true dofs x, from forms assembled on the local dofs.
/// In parallel, this is the contribution of the local elements.
static real_t ReferenceEnergy(const Operator *P, const SparseMatrix &A,
                              const Vector &l, const Vector &x)
{
   Vector xv(P ? P->Height() : x.Size()), Axv(xv.Size());
   if (P) { P->Mult(x, xv); }
   else { xv = x; }
   A.Mult(xv, Axv);
   return 0.5 * (xv * Axv) - l * xv;
}

static void TestBilinearFunctional(bool nonconforming)
{
   const string tag = nonconforming ? " (nonconforming)" : "";
   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   if (nonconforming)
   {
      mesh.EnsureNCMesh();
      Array<int> refs({0, 5, 10});
      mesh.GeneralRefinement(refs);
   }
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   ConstantCoefficient one(1.0);

   BilinearForm a(&fes), a_ref(&fes);
   for (BilinearForm *form : {&a, &a_ref})
   {
      form->AddDomainIntegrator(new DiffusionIntegrator);
      form->AddDomainIntegrator(new MassIntegrator);
   }
   LinearForm l(&fes);
   l.AddDomainIntegrator(new DomainLFIntegrator(one));
   a_ref.Assemble();
   a_ref.Finalize();

   BilinearFunctional f(&a, &l);
   f.Assemble();
   const int n = fes.GetTrueVSize();
   Check(f.GetFieldSizes()[0] == n &&
         (!nonconforming || n < fes.GetVSize()),
         "BilinearFunctional field size" + tag);

   Vector xv = MakeVector(n, 1);
   MultiVector x(xv);
   Vector fx;
   f.Eval(x, fx);
   const real_t ref = ReferenceEnergy(fes.GetProlongationMatrix(),
                                      a_ref.SpMat(), l, xv);
   Check(Near(fx(0), ref), "BilinearFunctional::Eval" + tag);

   Vector d = MakeVector(n, 2);
   Check(DiffDerivativeAgainstCentralDiff(f, x, MultiVector(d), 1e-3) < 1e-8,
         "BilinearFunctional central difference" + tag);

   f.Assemble();
   f.Eval(x, fx);
   Check(Near(fx(0), ref), "BilinearFunctional::Assemble twice" + tag);
}

static void TestProblem()
{
   DenseMatrix A, B0, B1;
   MakeSPD(5, 1, A);
   MakeSPD(2, 2, B0);
   MakeSPD(2, 3, B1);
   Vector b0 = MakeVector(3, 1), b1 = MakeVector(2, 2), c0 = MakeVector(3, 3);
   LinearFunctional lin0(&b0), lin1(&b1), lin0c(&c0);
   BlockQuadratic bq(3, 2, A);
   TwoQuadratics two(B0, B1);

   Problem prob;
   const int u = prob.RegisterField(3);
   const int v = prob.RegisterField(2);
   const int id_lin0 = prob.RegisterFunctional(Array<int>({u}), &lin0);
   const int id_bq = prob.RegisterFunctional(Array<int>({u, v}), &bq);
   const int id_two = prob.RegisterFunctional(Array<int>({v}), &two);
   const int id_lin1 = prob.RegisterFunctional(Array<int>({v}), &lin1);
   const int id_lin0c = prob.RegisterFunctional(Array<int>({u}), &lin0c);

   Vector lo(2), up(2), lo1(1);
   lo(0) = -1.0; lo(1) = -2.0; up(0) = 1.0; up(1) = 2.0; lo1(0) = 0.5;
   prob.SetRole(id_lin0, Objective(2.0));
   prob.SetRole(id_bq, Objective(1.0));
   prob.SetRole(id_two, Inequality(lo, up));
   prob.SetRole(id_lin1, Equality(5.0));
   prob.SetRole(id_lin0c, Inequality(lo1, Vector()));
   prob.Finalize();
   Check(prob.Finalized(), "Problem::Finalized");

   Vector xu = MakeVector(3, 4), xv = MakeVector(2, 5);
   MultiVector x(xu, xv);
   Vector du = MakeVector(3, 6), dv = MakeVector(2, 7);
   MultiVector d(du, dv);

   const GlobalFunctional &obj = prob.GetObjective();
   Vector fx, ref;
   obj.Eval(x, fx);
   bq.Eval(x, ref);
   Check(obj.Height() == 1 && Near(fx(0), 2.0 * (b0 * xu) + ref(0)),
         "Problem objective value");
   Check(DiffDerivativeAgainstCentralDiff(obj, x, d, 1e-3) < 1e-8,
         "Problem objective central difference");

   auto [eq, eq_target] = prob.GetEquality();
   eq.Eval(x, fx);
   Check(eq.Height() == 1 && eq_target.target.Size() == 1 &&
         Near(eq_target.target(0), 5.0) && Near(fx(0), b1 * xv),
         "Problem equality");

   auto [ineq, bounds] = prob.GetInequality();
   ineq.Eval(x, fx);
   two.Eval(MultiVector(xv), ref);
   Check(ineq.Height() == 3 && Near(fx(0), ref(0)) && Near(fx(1), ref(1)) &&
         Near(fx(2), c0 * xu), "Problem inequality value");
   Check(bounds.lower.Size() == 3 && bounds.upper.Size() == 3 &&
         bounds.lower(0) == -1.0 && bounds.lower(1) == -2.0 &&
         bounds.lower(2) == 0.5 && bounds.upper(0) == 1.0 &&
         bounds.upper(1) == 2.0 && bounds.upper(2) == mfem::infinity(),
         "Problem inequality bounds");
   Check(DiffDerivativeAgainstCentralDiff(ineq, x, d, 1e-3) < 1e-8,
         "Problem inequality central difference");
   Check(prob.GetEqualityIndex(id_lin1) == 0 &&
         prob.GetEqualityIndex(id_two) == -1 &&
         prob.GetInequalityIndex(id_two) == 0 &&
         prob.GetInequalityIndex(id_lin0c) == 2 &&
         prob.GetInequalityIndex(id_lin0) == -1,
         "Problem::GetEqualityIndex and GetInequalityIndex");

   // Essential dofs: zero objective derivative there.
   Array<int> ess_u({1});
   prob.SetFieldEssentialDofs(u, ess_u);
   prob.Finalize();
   ess_u[0] = 0; // the problem keeps its own copy
   Check(prob.GetFieldEssentialDofs(u)[0] == 1 &&
         prob.GetFieldEssentialDofs(v).Size() == 0,
         "Problem copies the essential dofs");
   Vector gu(3), gv(2), ru(3), rv(2);
   MultiVector g(gu, gv), r(ru, rv);
   prob.GetObjective().ComponentDerivative(x, 0, g);
   bq.ComponentDerivative(x, 0, r);
   ru.Add(2.0, b0);
   ru.HostReadWrite()[1] = 0.0;
   Check(Near(gu, ru) && Near(gv, rv), "Problem essential dofs");

   // Deactivated objective term.
   prob.DeactivateFunctional(id_lin0);
   prob.Finalize();
   prob.GetObjective().Eval(x, fx);
   bq.Eval(x, ref);
   Check(Near(fx(0), ref(0)), "Problem::DeactivateFunctional");

   // No constraints: zero rows.
   Problem unconstrained;
   unconstrained.RegisterField(3);
   unconstrained.RegisterField(2);
   const int id = unconstrained.RegisterFunctional(Array<int>({0, 1}), &bq);
   unconstrained.SetRole(id, Objective(1.0));
   unconstrained.Finalize();
   auto [eq0, target0] = unconstrained.GetEquality();
   auto [ineq0, bounds0] = unconstrained.GetInequality();
   eq0.Eval(x, fx);
   Check(eq0.Height() == 0 && target0.target.Size() == 0 && fx.Size() == 0,
         "Problem without equality");
   ineq0.Eval(x, fx);
   Check(ineq0.Height() == 0 && bounds0.lower.Size() == 0 &&
         bounds0.upper.Size() == 0 && fx.Size() == 0,
         "Problem without inequality");
}

static void TestObjectives()
{
   DenseMatrix A, B0, B1;
   MakeSPD(5, 1, A);
   MakeSPD(2, 2, B0);
   MakeSPD(2, 3, B1);
   Vector b0 = MakeVector(3, 1);
   LinearFunctional lin0(&b0);
   BlockQuadratic bq(3, 2, A);
   TwoQuadratics two(B0, B1);

   Problem prob;
   const int u = prob.RegisterField(3);
   const int v = prob.RegisterField(2);
   Vector w(2);
   w(0) = 0.5; w(1) = -1.5;
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &lin0),
                Objective(2.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u, v}), &bq),
                Objective(1.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({v}), &two),
                Objective(w));
   prob.SetRole(prob.RegisterFunctional(Array<int>({v}), &two),
                Objective(3.0, 2));
   prob.Finalize();

   Vector xu = MakeVector(3, 2), xv = MakeVector(2, 3);
   MultiVector x(xu, xv);
   Vector fx, q, r;
   prob.GetObjective().Eval(x, fx);
   two.Eval(MultiVector(xv), q);
   bq.Eval(x, r);
   const real_t expected = 2.0 * (b0 * xu) + r(0) + 0.5 * q(0) - 1.5 * q(1)
                           + 3.0 * (q(0) + q(1));
   Check(prob.GetObjective().Height() == 1 && Near(fx(0), expected),
         "Problem with several and vector-valued objectives");

   Vector du = MakeVector(3, 4), dv = MakeVector(2, 5);
   MultiVector d(du, dv);
   Check(DiffDerivativeAgainstCentralDiff(prob.GetObjective(), x, d, 1e-3)
         < 1e-8, "Problem with several objectives, central difference");
   Vector one(1); one = 1.0;
   Check(DiffHessianAgainstCentralDiff(prob.GetObjective(), x, one, d, 1e-3)
         < 1e-8, "Problem with several objectives, Hessian");

   // Essential dofs: the Hessian in the free directions.
   Array<int> ess_u({1});
   prob.SetFieldEssentialDofs(u, ess_u);
   prob.Finalize();
   du.HostReadWrite()[1] = 0.0;
   Check(DiffHessianAgainstCentralDiff(prob.GetObjective(), x, one, d, 1e-3)
         < 1e-8, "Problem with essential dofs, Hessian");
}

static void TestFixedField()
{
   DenseMatrix A;
   MakeSPD(5, 1, A);
   BlockQuadratic bq(3, 2, A);
   Problem prob;
   const int u = prob.RegisterField(3);
   const int v = prob.RegisterField(2);
   prob.SetRole(prob.RegisterFunctional(Array<int>({u, v}), &bq),
                Objective(1.0));
   prob.SetFieldFixed(v);
   prob.Finalize();
   const GlobalFunctional &obj = prob.GetObjective();

   Vector xu = MakeVector(3, 1), xv = MakeVector(2, 2);
   MultiVector x(xu, xv);
   Vector gu(3), gv(2), ru(3), rv(2);
   MultiVector g(gu, gv), r(ru, rv);
   obj.ComponentDerivative(x, 0, g);
   bq.ComponentDerivative(x, 0, r);
   Check(Near(gu, ru) && gv.Normlinf() == 0.0, "Fixed field, derivative");

   // Hessian: A_uu on the free field, the identity on the fixed field.
   Vector z = MakeVector(5, 3), zt(z), Hz(5), expected(5);
   const real_t *zp = z.HostRead();
   for (int k=3; k<5; k++) { zt.HostReadWrite()[k] = 0.0; }
   A.Mult(zt, expected);
   for (int k=3; k<5; k++) { expected(k) = zp[k]; }
   obj.ComponentHessian(x, 0)->Mult(z, Hz);
   Check(Near(Hz, expected), "Fixed field, Hessian");

   prob.SetFieldFixed(v, false);
   prob.Finalize();
   prob.GetObjective().ComponentDerivative(x, 0, g);
   Check(Near(gv, rv), "Field set free again");
}

static void TestInputOrder()
{
   DenseMatrix A;
   MakeSPD(5, 4, A);
   BlockQuadratic bq(2, 3, A); // inputs (v, u)

   Problem prob;
   const int u = prob.RegisterField(3);
   const int v = prob.RegisterField(2);
   prob.SetRole(prob.RegisterFunctional(Array<int>({v, u}), &bq),
                Objective(1.0));
   Array<int> ess_u({1});
   prob.SetFieldEssentialDofs(u, ess_u);
   prob.Finalize();

   Vector xu = MakeVector(3, 1), xv = MakeVector(2, 2);
   MultiVector x(xu, xv), x_swapped(xv, xu);
   Vector fx, ref;
   prob.GetObjective().Eval(x, fx);
   bq.Eval(x_swapped, ref);
   Check(Near(fx(0), ref(0)), "Problem with reordered inputs, value");

   Vector gu(3), gv(2), ru(3), rv(2);
   MultiVector g(gu, gv), r(rv, ru);
   prob.GetObjective().ComponentDerivative(x, 0, g);
   bq.ComponentDerivative(x_swapped, 0, r);
   ru(1) = 0.0;
   Check(Near(gu, ru) && Near(gv, rv),
         "Problem with reordered inputs, derivative");
}

static void TestZeroSized()
{
   Vector b0 = MakeVector(3, 1), empty;
   LinearFunctional lin0(&b0), lin_empty(&empty);
   NoOutput none(3);

   Problem prob;
   const int u = prob.RegisterField(3);
   const int e = prob.RegisterField(0);
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &lin0),
                Objective(1.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({e}), &lin_empty),
                Objective(1.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &none),
                Equality(Vector()));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &lin0),
                Equality(2.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &none),
                Inequality());
   Array<int> no_dofs;
   prob.SetFieldEssentialDofs(e, no_dofs);
   prob.Finalize();

   Vector xu = MakeVector(3, 2), xe;
   MultiVector x(xu, xe);
   Vector fx;
   prob.GetObjective().Eval(x, fx);
   Check(Near(fx(0), b0 * xu), "Zero-sized field, objective");

   auto [eq, target] = prob.GetEquality();
   eq.Eval(x, fx);
   Check(eq.Height() == 1 && target.target.Size() == 1 &&
         Near(target.target(0), 2.0) && Near(fx(0), b0 * xu),
         "Zero-output equality");
   auto [ineq, bounds] = prob.GetInequality();
   ineq.Eval(x, fx);
   Check(ineq.Height() == 0 && bounds.lower.Size() == 0 && fx.Size() == 0,
         "Zero-output inequality");

   Vector du = MakeVector(3, 3), de;
   MultiVector d(du, de);
   Check(DiffDerivativeAgainstCentralDiff(prob.GetObjective(), x, d, 1e-3)
         < 1e-10 &&
         DiffDerivativeAgainstCentralDiff(eq, x, d, 1e-3) < 1e-10,
         "Zero-sized field, central difference");
}

static void TestZeroOutputObjective()
{
   Vector b0 = MakeVector(3, 1);
   LinearFunctional lin0(&b0);
   NoOutput none(3);

   Problem prob;
   const int u = prob.RegisterField(3);
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &lin0),
                Objective(1.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &none),
                Objective(Vector()));
   prob.Finalize();

   Vector xu = MakeVector(3, 2), fx;
   MultiVector x(xu);
   prob.GetObjective().Eval(x, fx);
   Check(prob.GetObjective().Height() == 1 && Near(fx(0), b0 * xu),
         "Zero-output objective");
}

static void TestRoleChanges()
{
   Vector b0 = MakeVector(3, 1), b1 = MakeVector(3, 2);
   LinearFunctional lin0(&b0), lin1(&b1);

   Problem prob;
   const int u = prob.RegisterField(3);
   const int id0 = prob.RegisterFunctional(Array<int>({u}), &lin0);
   const int id1 = prob.RegisterFunctional(Array<int>({u}), &lin1);
   prob.SetRole(id0, Objective(1.0));
   prob.SetRole(id1, Equality(5.0));
   prob.Finalize();
   prob.Finalize();
   Check(std::get<0>(prob.GetEquality()).Height() == 1 &&
         std::get<0>(prob.GetInequality()).Height() == 0,
         "Finalize twice");

   Check(std::holds_alternative<Equality>(prob.GetRole(id1)) &&
         std::get<Equality>(prob.GetRole(id1)).target(0) == 5.0,
         "Problem::GetRole");
   prob.SetRole(id1, Inequality(0.0, 1.0));
   Check(!prob.Finalized(), "SetRole invalidates the layout");
   prob.Finalize();
   auto [ineq, bounds] = prob.GetInequality();
   Check(std::get<0>(prob.GetEquality()).Height() == 0 &&
         ineq.Height() == 1 && bounds.lower(0) == 0.0 &&
         bounds.upper(0) == 1.0, "Equality changed to inequality");

   prob.DeactivateFunctional(id1);
   prob.Finalize();
   Check(std::get<0>(prob.GetInequality()).Height() == 0,
         "Deactivated constraint");
   prob.DeactivateFunctional(id1, false);
   prob.Finalize();
   Check(std::get<0>(prob.GetInequality()).Height() == 1,
         "Reactivated constraint");

   prob.SetRole(id1, Objective(-1.0));
   prob.Finalize();
   Vector xu = MakeVector(3, 3), fx;
   MultiVector x(xu);
   prob.GetObjective().Eval(x, fx);
   Check(Near(fx(0), b0 * xu - b1 * xu) &&
         std::get<0>(prob.GetInequality()).Height() == 0,
         "Constraint changed to objective");
}

static void TestLocalConstraints()
{
   DenseMatrix F;
   MakeSPD(3, 1, F);
   Problem prob;
   const int u = prob.RegisterField(3);
   const int rho = prob.RegisterField(6);
   const int box = prob.RegisterLocalConstraint(u, new BoxConstraint(0.0, 1.0, 3),
                                                nullptr, true);
   const int filtered = prob.RegisterLocalConstraint(
                           u, new BoxConstraint(Vector(), MakeVector(3, 2)),
                           new DenseOperator(F), true, true);
   const SimplexConstraint simplex(3);
   const int mix = prob.RegisterLocalConstraint(rho, &simplex);
   Check(prob.NumLocalConstraints() == 3 && !prob.Finalized(),
         "Problem::RegisterLocalConstraint");
   // A rank without entries: both bounds are empty.
   const BoxConstraint empty_box{Vector(), Vector()};
   Check(empty_box.lower.Size() == 0 && empty_box.upper.Size() == 0,
         "BoxConstraint without entries");
   prob.Finalize();
   Check(prob.Finalized(), "Problem::Finalize with local constraints");

   auto [field_box, set_box, op_box] = prob.GetLocalConstraint(box);
   const auto *box_set = dynamic_cast<const BoxConstraint*>(&set_box);
   Check(field_box == u && op_box == nullptr && box_set != nullptr &&
         box_set->lower.Size() == 3 && box_set->upper.Max() == 1.0,
         "Dofwise box constraint");

   auto [field_f, set_f, op_f] = prob.GetLocalConstraint(filtered);
   const auto *filtered_set = dynamic_cast<const BoxConstraint*>(&set_f);
   Check(field_f == u && op_f != nullptr && op_f->Height() == 3 &&
         filtered_set != nullptr &&
         filtered_set->lower.Max() == -mfem::infinity(),
         "Box constraint on an operator");

   auto [field_mix, set_mix, op_mix] = prob.GetLocalConstraint(mix);
   const auto *simplex_set = dynamic_cast<const SimplexConstraint*>(&set_mix);
   Check(field_mix == rho && op_mix == nullptr && simplex_set != nullptr &&
         simplex_set->stride == 3 &&
         dynamic_cast<const BoxConstraint*>(&set_mix) == nullptr,
         "Simplex constraint");
}

static void TestMetric()
{
   DenseMatrix M, Minv;
   MakeSPD(3, 1, M);
   Minv = M;
   Minv.Invert();
   const DenseOperator Mop(M);

   Problem prob;
   const int u = prob.RegisterField(3);
   const int v = prob.RegisterField(2);
   // Owned operators, replaced once.
   prob.SetFieldMetric(u, new DenseOperator(M), new DenseOperator(M),
                       true, true);
   prob.SetFieldMetric(u, &Mop, new DenseOperator(Minv), false, true);
   // Replacing only the metric keeps the owned Riesz map.
   const DenseOperator *Rop = new DenseOperator(Minv);
   prob.SetFieldMetric(u, new DenseOperator(M), Rop, true, true);
   prob.SetFieldMetric(u, &Mop, Rop, false, true);

   Vector xu = MakeVector(3, 1), yu = MakeVector(3, 2), Myu(3);
   Vector xv = MakeVector(2, 3), yv = MakeVector(2, 4);
   M.Mult(yu, Myu);
   Check(Near(prob.InnerProduct(u, xu, yu), xu * Myu) &&
         Near(prob.InnerProduct(v, xv, yv), xv * yv),
         "Problem::InnerProduct per field");
   MultiVector x(xu, xv), y(yu, yv);
   Check(Near(prob.InnerProduct(x, y), xu * Myu + xv * yv),
         "Problem::InnerProduct");

   Vector pu(3), pv(2);
   MultiVector duals(Myu, yv), primals(pu, pv);
   prob.ApplyRiesz(duals, primals);
   Check(Near(pu, yu) && Near(pv, yv), "Problem::ApplyRiesz");
}

/// x = x0 + 1 on every field; checks the dispatch of OptimizationSolver.
class ShiftSolver : public OptimizationSolver
{
public:
   void MultMV(const MultiVector &x0, MultiVector &x) const override
   {
      MFEM_VERIFY(problem != nullptr, "No problem.");
      for (int b=0; b<x.NumBlocks(); b++) { x[b] = x0[b]; x[b] += 1.0; }
   }
   const Problem *GetProblem() const { return problem; }
};

static void TestOptimizationSolver()
{
   Problem prob;
   prob.RegisterField(3);
   prob.RegisterField(2);
   prob.Finalize();
   ShiftSolver solver;
   solver.SetOptimizationProblem(prob);
   Check(solver.GetProblem() == &prob,
         "OptimizationSolver::SetOptimizationProblem");

   // Mult splits a monolithic vector into the fields.
   Vector z0 = MakeVector(5, 3), z(5), z_expected(z0);
   z_expected += 1.0;
   solver.Mult(z0, z);
   Check(Near(z, z_expected), "OptimizationSolver::Mult");

   Vector x0 = MakeVector(3, 1), x(3), expected(x0);
   expected += 1.0;
   Vector y0 = MakeVector(2, 2), y(2);
   const MultiVector X0(x0, y0);
   MultiVector X(x, y);
   solver.MultMV(X0, X);
   Vector y_expected(y0);
   y_expected += 1.0;
   Check(Near(x, expected) && Near(y, y_expected),
         "OptimizationSolver::MultMV");
}

/// Checks the SLBQP solution x of min |x - xt| s.t. w.x = a, lo <= x <= hi:
/// feasibility and x = median(xt + l w, lo, hi) for some multiplier l.
static bool CheckSLBQP(const Vector &xt, const Vector &x, const Vector &w,
                       real_t a, real_t lo, real_t hi, real_t global_wx)
{
   const real_t *xp = x.HostRead(), *tp = xt.HostRead(), *wp = w.HostRead();
   bool ok = Near(global_wx, a, 1e-8);
   real_t l = 0.0;
   for (int i=0; i<x.Size(); i++)
   {
      ok = ok && xp[i] >= lo && xp[i] <= hi;
      if (xp[i] > lo && xp[i] < hi) { l = (xp[i] - tp[i]) / wp[i]; }
   }
   for (int i=0; i<x.Size(); i++)
   {
      const real_t y = std::min(hi, std::max(lo, tp[i] + l * wp[i]));
      ok = ok && Near(xp[i], y, 1e-8);
   }
   return ok;
}

static void TestSLBQP()
{
   const int n = 20;
   Vector xt = MakeVector(n, 1), w(n), x(n), x_prob(n);
   xt *= 2.0;
   w = 1.0;
   w.Add(0.5, MakeVector(n, 2));
   const real_t a = 0.4 * w.Sum();

   SLBQPOptimizer slbqp;
   Vector lo(n), hi(n);
   lo = 0.0; hi = 1.0;
   slbqp.SetMaxIter(100);
   slbqp.SetAbsTol(1e-14);
   slbqp.SetRelTol(1e-14);
   slbqp.SetPrintLevel(0);
   slbqp.SetBounds(lo, hi);
   slbqp.SetLinearConstraint(w, a);
   slbqp.Mult(xt, x);
   Check(slbqp.GetConverged() && CheckSLBQP(xt, x, w, a, 0.0, 1.0, w * x),
         "SLBQPOptimizer with SetBounds and SetLinearConstraint");

   // The same problem through Problem.
   LinearFunctional c(&w);
   Problem prob;
   const int u = prob.RegisterField(n);
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &c), Equality(a));
   prob.RegisterLocalConstraint(u, new BoxConstraint(0.0, 1.0, n), nullptr,
                                true);
   prob.Finalize();
   SLBQPOptimizer slbqp_prob;
   slbqp_prob.SetMaxIter(100);
   slbqp_prob.SetAbsTol(1e-14);
   slbqp_prob.SetRelTol(1e-14);
   slbqp_prob.SetPrintLevel(0);
   slbqp_prob.SetOptimizationProblem(prob);
   slbqp_prob.Mult(xt, x_prob);
   Check(slbqp_prob.GetConverged() && Near(x_prob, x, 1e-12),
         "SLBQPOptimizer with Problem");
}

/// GridFunction::ImposeBounds: bounds hold and the weighted sum is kept.
static void TestImposeBounds()
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   L2_FECollection fec(2, 2, BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction u(&fes);
   u.Randomize(1);
   u *= 2.0;
   u -= 0.5; // values in [-0.5, 1.5]

   Array<int> vdofs;
   fes.GetElementVDofs(0, vdofs);
   Vector weights(vdofs.Size()), before, after;
   // ImposeBounds works on the host.
   for (int k=0; k<weights.Size(); k++) { weights(k) = 1.0 + 0.1 * k; }
   u.GetSubVector(vdofs, before);
   u.ImposeBounds(0, weights, 0.0, 1.0);
   u.GetSubVector(vdofs, after);
   Check(after.Min() >= 0.0 && after.Max() <= 1.0 &&
         Near(weights * after, weights * before, 1e-10) &&
         !Near(after, before), "GridFunction::ImposeBounds");
}

#ifdef MFEM_USE_MPI
static void TestParallelSLBQP()
{
   MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank();
   // With several ranks, rank 0 owns no entries.
   const int n = (rank == 0 && Mpi::WorldSize() > 1) ? 0 : 10;

   Vector xt = MakeVector(n, 1 + rank), w(n), x(n);
   xt *= 2.0;
   w = 1.0;
   w.Add(0.5, MakeVector(n, 11 + rank));
   real_t a = 0.4 * w.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);

   LinearFunctional c(comm, &w);
   Problem prob(comm);
   const int u = prob.RegisterField(n);
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &c), Equality(a));
   prob.RegisterLocalConstraint(u, new BoxConstraint(0.0, 1.0, n), nullptr,
                                true);
   prob.Finalize();

   SLBQPOptimizer slbqp(comm);
   slbqp.SetMaxIter(100);
   slbqp.SetAbsTol(1e-14);
   slbqp.SetRelTol(1e-14);
   slbqp.SetPrintLevel(0);
   slbqp.SetOptimizationProblem(prob);
   slbqp.Mult(xt, x);

   real_t wx = w * x;
   MPI_Allreduce(MPI_IN_PLACE, &wx, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   Check(slbqp.GetConverged() && CheckSLBQP(xt, x, w, a, 0.0, 1.0, wx),
         "SLBQPOptimizer (parallel)");
}

static void CheckParallelBilinear(ParMesh &pmesh, const string &tag)
{
   MPI_Comm comm = pmesh.GetComm();
   const int rank = pmesh.GetMyRank();
   H1_FECollection fec(2, 2);
   ParFiniteElementSpace fes(&pmesh, &fec);
   ConstantCoefficient one(1.0);
   ParBilinearForm a(&fes), a_ref(&fes);
   for (ParBilinearForm *form : {&a, &a_ref})
   {
      form->AddDomainIntegrator(new DiffusionIntegrator);
      form->AddDomainIntegrator(new MassIntegrator);
   }
   ParLinearForm l(&fes);
   l.AddDomainIntegrator(new DomainLFIntegrator(one));
   a_ref.Assemble();
   a_ref.Finalize();

   BilinearFunctional f(&a, &l);
   f.Assemble();
   const int n = fes.GetTrueVSize();
   Check(f.IsParallel() && f.GetFieldSizes()[0] == n,
         "BilinearFunctional sizes" + tag);

   Vector uv = MakeVector(n, 21 + rank), fx;
   MultiVector u(uv);
   f.Eval(u, fx);
   real_t local = ReferenceEnergy(fes.GetProlongationMatrix(), a_ref.SpMat(),
                                  l, uv), global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   Check(Near(fx(0), global), "BilinearFunctional::Eval" + tag);

   Vector d = MakeVector(n, 31 + rank);
   Check(DiffDerivativeAgainstCentralDiff(f, u, MultiVector(d), 1e-3) < 1e-8,
         "BilinearFunctional central difference" + tag);

   Vector ones(n); ones = 1.0;
   LinearFunctional sum(comm, &ones);
   Problem prob(comm);
   const int field = prob.RegisterField(n);
   const int id_f = prob.RegisterFunctional(Array<int>({field}), &f);
   const int id_s = prob.RegisterFunctional(Array<int>({field}), &sum);
   prob.SetRole(id_f, Objective(1.0));
   prob.SetRole(id_s, Equality(0.0));
   prob.Finalize();
   prob.GetObjective().Eval(u, fx);
   Check(prob.GetObjective().IsParallel() && Near(fx(0), global),
         "Problem objective" + tag);
   auto [eq, target] = prob.GetEquality();
   eq.Eval(u, fx);
   local = uv.Sum();
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   Check(eq.IsParallel() && Near(fx(0), global), "Problem equality" + tag);
   local = uv * d;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   Check(Near(prob.InnerProduct(field, uv, d), global),
         "Problem::InnerProduct" + tag);
   Check(DiffDerivativeAgainstCentralDiff(prob.GetObjective(), u,
                                          MultiVector(d), 1e-3) < 1e-8,
         "Problem objective central difference" + tag);
}

static void TestParallel()
{
   MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank();

   // LinearFunctional reduces over the communicator.
   Vector b = MakeVector(4, 1 + rank), xv = MakeVector(4, 11 + rank);
   LinearFunctional lin(comm, &b);
   MultiVector x(xv);
   Vector fx;
   lin.Eval(x, fx);
   real_t local = b * xv, global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   Check(lin.IsParallel() && Near(fx(0), global),
         "LinearFunctional::Eval (parallel)");

   Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
   ParMesh pmesh(comm, mesh);
   mesh.Clear();
   CheckParallelBilinear(pmesh, " (parallel)");
}

/// Rank 0 owns no entries of the field.
static void TestParallelEmptyRank()
{
   MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank(), num_ranks = Mpi::WorldSize();
   const int n = rank == 0 ? 0 : 3;

   Vector b = MakeVector(n, 1 + rank), xv = MakeVector(n, 11 + rank), Ax(n);
   DenseMatrix A;
   MakeSPD(n, 2 + rank, A);
   LinearFunctional lin(comm, &b);
   DenseOperator Aop(A);
   QuadraticFunctional quad(comm, &Aop);
   NoOutput none(comm, n);

   Problem prob(comm);
   const int u = prob.RegisterField(n);
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &lin),
                Objective(1.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &quad),
                Objective(2.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &lin),
                Equality(1.0));
   prob.SetRole(prob.RegisterFunctional(Array<int>({u}), &none),
                Inequality());
   prob.Finalize();

   MultiVector x(xv);
   A.Mult(xv, Ax);
   real_t local = b * xv + xv * Ax, global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   Vector fx;
   prob.GetObjective().Eval(x, fx);
   Check(Near(fx(0), global), "Empty rank, objective");

   Vector d = MakeVector(n, 21 + rank);
   auto [eq, target] = prob.GetEquality();
   Check(DiffDerivativeAgainstCentralDiff(prob.GetObjective(), x,
                                          MultiVector(d), 1e-3) < 1e-8 &&
         DiffDerivativeAgainstCentralDiff(eq, x, MultiVector(d), 1e-3) < 1e-8,
         "Empty rank, central difference");
   Check(std::get<0>(prob.GetInequality()).Height() == 0,
         "Empty rank, zero-output inequality");

   if (num_ranks > 1)
   {
      Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
      Array<int> partitioning(mesh.GetNE());
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         partitioning[e] = 1 + e % (num_ranks - 1);
      }
      ParMesh pmesh(comm, mesh, partitioning.GetData());
      mesh.Clear();
      CheckParallelBilinear(pmesh, " (rank 0 without elements)");
   }
}
#endif

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   Mpi::Init(argc, argv);
   Hypre::Init();
   root = Mpi::Root();
#endif

   // Options: [-d device] [test names]; no names runs all tests.
   const char *device_config = "cpu";
   vector<string> names;
   for (int i = 1; i < argc; i++)
   {
      if (string(argv[i]) == "-d" && i + 1 < argc) { device_config = argv[++i]; }
      else { names.push_back(argv[i]); }
   }
   Device device(device_config);

   const vector<pair<string, void(*)()>> tests =
   {
      {"multivector", TestMultiVector},
      {"weighted_sum_operator", TestWeightedSumOperator},
      {"linear", TestLinearFunctional},
      {"quadratic", TestQuadraticFunctional},
      {"weighted_sum_hessian", TestDefaultWeightedSumHessian},
      {"weighted_sum", TestWeightedSumFunctional},
      {"weighted_sum_functional_hessian", TestWeightedSumHessian},
      {"constrained", TestConstrainedFunctional},
      {"bilinear", [] { TestBilinearFunctional(false); }},
      {"bilinear_nc", [] { TestBilinearFunctional(true); }},
      {"problem", TestProblem},
      {"objectives", TestObjectives},
      {"input_order", TestInputOrder},
      {"zero_sized", TestZeroSized},
      {"zero_output_objective", TestZeroOutputObjective},
      {"role_changes", TestRoleChanges},
      {"local_constraints", TestLocalConstraints},
      {"fixed_field", TestFixedField},
      {"metric", TestMetric},
      {"optimization_solver", TestOptimizationSolver},
      {"slbqp", TestSLBQP},
      {"impose_bounds", TestImposeBounds},
#ifdef MFEM_USE_MPI
      {"parallel", TestParallel},
      {"parallel_empty_rank", TestParallelEmptyRank},
      {"parallel_slbqp", TestParallelSLBQP},
#endif
   };
   for (const string &n : names)
   {
      bool known = false;
      for (const auto &[name, test] : tests) { known = known || name == n; }
      MFEM_VERIFY(known, "Unknown test: " << n);
   }
   for (const auto &[name, test] : tests)
   {
      bool selected = names.empty();
      for (const string &n : names) { selected = selected || name == n; }
      if (selected) { test(); }
   }

#ifdef MFEM_USE_MPI
   int failed = num_failed;
   MPI_Allreduce(&failed, &num_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
   if (root)
   {
      cout << num_checks << " checks, " << num_failed << " failed" << endl;
   }
   return num_failed == 0 ? 0 : 1;
}
