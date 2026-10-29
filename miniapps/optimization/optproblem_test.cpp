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
//            -----------------------------------------------------
//            OptProblem test: bookkeeping, roles, derivatives
//            -----------------------------------------------------
//
// Compile with: make optproblem_test
//
// Sample runs:  optproblem_test
//               mpirun -np 4 optproblem_test
//
// Description:  Unit test for OptProblem (optproblem.hpp) and the functionals
//               (functionals.hpp) on small analytic quantities: layout and
//               index mapping, values, derivatives and second derivatives of
//               the views (vs. central finite differences and adjoint
//               identities), objective/constraint changes, weighted, GE and
//               inactive constraints, empty objectives, ContractedOperator,
//               Update() notifications, dofwise bounds and the Riesz map.
//               With MPI, the functionals are also checked on a distributed
//               vector.

#include "mfem.hpp"
#include "optproblem.hpp"
#include "functionals.hpp"
#include <cmath>
#include <iostream>

using namespace std;
using namespace mfem;

using Type = OptProblem::QuantityType;

/// Derivative with a stacked second derivative; counts VJPs.
class TestDerivative : public Operator
{
public:
   DenseMatrix J;    ///< m x n
   DenseMatrix H;    ///< (m n) x n, block i = Hessian of output i
   mutable int n_vjp = 0;
   TestDerivative(int m, int n) : Operator(m, n), J(m, n), H(m*n, n)
   { H = 0.0; }
   void Mult(const Vector &x, Vector &y) const override { J.Mult(x, y); }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      n_vjp++;
      J.MultTranspose(x, y);
   }
   Operator &GetGradient(const Vector &) const override
   { return const_cast<DenseMatrix &>(H); }
};

/// q(x) = A x + c (zero second derivative); notified by Update().
class AffineQuantity : public UpdatableOperator
{
public:
   DenseMatrix A;
   Vector c;
   mutable int n_mult = 0;
   int n_update = 0;
   long last_seq = 0;
   mutable TestDerivative der;

   AffineQuantity(const DenseMatrix &A_, const Vector &c_)
      : UpdatableOperator(A_.Height(), A_.Width()), A(A_), c(c_),
        der(A_.Height(), A_.Width()) { der.J = A; }

   void Mult(const Vector &x, Vector &y) const override
   {
      n_mult++;
      A.Mult(x, y);
      y += c;
   }
   Operator &GetGradient(const Vector &) const override { return der; }
   void Update(const Vector &, long seq) override
   {
      n_update++;
      last_seq = seq;
   }
};

/// q_i(x) = sum_j C_ij x_j^2,  J_ij = 2 C_ij x_j,  H_i = diag(2 C_i.)
class QuadQuantity : public Operator
{
public:
   DenseMatrix C;
   mutable int n_mult = 0;
   mutable TestDerivative der;

   QuadQuantity(const DenseMatrix &C_)
      : Operator(C_.Height(), C_.Width()), C(C_),
        der(C_.Height(), C_.Width())
   {
      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < width; j++) { der.H(i*width + j, j) = 2.0*C(i,j); }
      }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      n_mult++;
      y.SetSize(height);
      for (int i = 0; i < height; i++)
      {
         y(i) = 0.0;
         for (int j = 0; j < width; j++) { y(i) += C(i,j)*x(j)*x(j); }
      }
   }
   Operator &GetGradient(const Vector &x) const override
   {
      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < width; j++) { der.J(i,j) = 2.0*C(i,j)*x(j); }
      }
      return der;
   }
};

static int n_fail = 0, n_check = 0;

static void Check(bool ok, const string &what)
{
   n_check++;
   if (!ok) { n_fail++; mfem::out << "  FAIL: " << what << '\n'; }
}

static void CheckNear(real_t a, real_t b, real_t tol, const string &what)
{
   const bool ok = std::abs(a - b) <= tol*(1.0 + std::abs(b));
   if (!ok) { mfem::out << "  (" << a << " vs " << b << ")\n"; }
   Check(ok, what);
}

static void CheckNear(const Vector &a, const Vector &b, real_t tol,
                      const string &what)
{
   bool ok = a.Size() == b.Size();
   for (int i = 0; ok && i < a.Size(); i++)
   {
      ok = std::abs(a(i) - b(i)) <= tol*(1.0 + std::abs(b(i)));
   }
   if (!ok) { a.Print(mfem::out); b.Print(mfem::out); }
   Check(ok, what);
}

static void Fill(Vector &v, real_t a, real_t b)
{
   for (int i = 0; i < v.Size(); i++) { v(i) = a + b*std::sin(1.0 + 1.7*i); }
}

/// y^T (A v) == (A^T y)^T v
static void CheckAdjoint(const Operator &A, const string &what)
{
   Vector v(A.Width()), y(A.Height()), Av(A.Height()), Aty(A.Width());
   Fill(v, 0.3, 1.0);
   Fill(y, -0.2, 0.8);
   A.Mult(v, Av);
   A.MultTranspose(y, Aty);
   CheckNear(y*Av, Aty*v, 1e-12, what + ": adjoint identity");
}

static const real_t eps = 1e-6;

/// Derivative JVP vs. central FD of the operator, plus adjoint identity.
static void CheckDerivative(const Operator &V, const Vector &x,
                            const Vector &v, const string &what)
{
   Vector jv;
   {
      Operator &D = V.GetGradient(x);
      D.Mult(v, jv);
      CheckAdjoint(D, what + " derivative");
   }
   Vector xp(x), xm(x), fp, fm;
   xp.Add(eps, v);
   xm.Add(-eps, v);
   V.Mult(xp, fp);
   V.Mult(xm, fm);
   fp -= fm;
   fp /= 2.0*eps;
   CheckNear(jv, fp, 1e-6, what + ": JVP vs FD");
}

/// Stacked second derivative vs. central FD of the row derivatives.
static void CheckSecondDerivative(const Operator &V, const Vector &x,
                                  const Vector &v, const string &what)
{
   const int n = x.Size(), R = V.Height();
   Vector Hv(R*n);   // outputs are sized by the caller, as for any Operator
   {
      Operator &D2 = V.GetGradient(x).GetGradient(x);
      Check(D2.Height() == R*n && D2.Width() == n, what + ": stacked shape");
      D2.Mult(v, Hv);
      CheckAdjoint(D2, what + " second derivative");
   }
   Vector e(R), gp, gm, fd(R*n), xp(x), xm(x);
   xp.Add(eps, v);
   xm.Add(-eps, v);
   for (int i = 0; i < R; i++)
   {
      e = 0.0;
      e(i) = 1.0;
      V.GetGradient(xp).MultTranspose(e, gp);
      V.GetGradient(xm).MultTranspose(e, gm);
      for (int j = 0; j < n; j++) { fd(i*n + j) = (gp(j) - gm(j))/(2.0*eps); }
   }
   CheckNear(Hv, fd, 1e-6, what + ": second derivative vs FD");
}

static void CheckAllDerivatives(const OptProblem &prob, const Vector &x,
                                const Vector &v, const string &what)
{
   const Operator *views[3] = { &prob.GetObjectiveOperator(),
                                &prob.GetEqualityConstraintOperator(),
                                &prob.GetInequalityConstraintOperator()
                              };
   const char *names[3] = {" f", " h", " g"};
   for (int k = 0; k < 3; k++)
   {
      if (views[k]->Height() == 0) { continue; }
      CheckDerivative(*views[k], x, v, what + names[k]);
      CheckSecondDerivative(*views[k], x, v, what + names[k]);
   }
}

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   Mpi::Init(argc, argv);
   if (!Mpi::Root()) { mfem::out.Disable(); }
#endif
   const int n = 4;

   // --- Quantities -----------------------------------------------------------
   // C: "multi-load compliance", height 3, quadratic
   DenseMatrix Cm(3, n);
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < n; j++) { Cm(i,j) = 1.0 + i + 0.5*j; }
   }
   QuadQuantity Cop(Cm);

   // V: "volume", height 1, mean of x
   DenseMatrix Vm(1, n);
   Vm = 1.0/n;
   Vector zero1(1); zero1 = 0.0;
   AffineQuantity Vop(Vm, zero1);

   // E: height 2, affine
   DenseMatrix Em(2, n);
   for (int j = 0; j < n; j++)
   {
      Em(0,j) = j + 1.0;
      Em(1,j) = (j % 2) ? -1.0 : 2.0;
   }
   Vector ce(2); ce(0) = 0.1; ce(1) = -0.3;
   AffineQuantity Eop(Em, ce);

   // P: scalar, affine
   DenseMatrix Pm(1, n);
   for (int j = 0; j < n; j++) { Pm(0,j) = 0.25*(j - 1.5); }
   AffineQuantity Pop(Pm, zero1);

   // S: height 2, quadratic, registered but unused at first
   DenseMatrix Sm(2, n);
   for (int j = 0; j < n; j++) { Sm(0,j) = 1.0; Sm(1,j) = j; }
   QuadQuantity Sop(Sm);

   Vector x(n), v(n);
   for (int j = 0; j < n; j++) { x(j) = 0.2 + 0.1*j; v(j) = 1.0 - 0.3*j; }

   Vector sE(2); sE(0) = 0.1; sE(1) = -0.2;
   Vector wC(3); wC(0) = 1.0; wC(1) = 2.0; wC(2) = 3.0;

   OptProblem prob(n);
   const int C = prob.AddQuantity(Cop, 0.5);
   const int V = prob.AddQuantity(Vop, 0.3);
   const int E = prob.AddQuantity(Eop, sE);
   const int P = prob.AddQuantity(Pop);
   const int S = prob.AddQuantity(Sop);

   sE = 100.0;   // shift is copied

   mfem::out << "Test 1: layout and index mapping\n";
   prob.SetObjective(C, wC);
   prob.SetConstraintType(V, Type::LE);
   prob.SetConstraintType(E, Type::EQ);
   prob.Deactivate(P);
   prob.Deactivate(S);   // INVALID but inactive: allowed

   const Operator &F = prob.GetObjectiveOperator();
   const Operator &Hop = prob.GetEqualityConstraintOperator();
   const Operator &Gop = prob.GetInequalityConstraintOperator();
   {
      Check(prob.GetType(C) == Type::OBJ, "C is OBJ");
      Check(prob.GetType(S) == Type::INVALID, "S stays INVALID");
      Check(prob.NumEq() == 2 && prob.NumIneq() == 1, "NumEq/NumIneq");
      Check(F.Height() == 1 && Hop.Height() == 2 && Gop.Height() == 1 &&
            F.Width() == n, "view shapes follow the layout");

      const Array<int> &eo = prob.GetEqOffsets();
      const Array<int> &lo = prob.GetIneqOffsets();
      Check(eo.Size() == 6 && lo.Size() == 6, "offset sizes");
      Check(eo[0] == 0 && eo[1] == 0 && eo[2] == 0 && eo[3] == 2 &&
            eo[4] == 2 && eo[5] == 2, "eq offsets");
      Check(lo[0] == 0 && lo[1] == 0 && lo[2] == 1 && lo[3] == 1 &&
            lo[4] == 1 && lo[5] == 1, "ineq offsets");
      auto [es, ee] = prob.GetEqRows(E);
      auto [vs, ve] = prob.GetIneqRows(V);
      Check(es == 0 && ee == 2 && vs == 0 && ve == 1, "row ranges");
      auto [cs, ce2] = prob.GetIneqRows(C);
      Check(cs == ce2, "objective owns no constraint rows");
   }

   mfem::out << "Test 2: view values\n";
   {
      Vector qC(3), qV(1), qE(2);
      Cop.Mult(x, qC); Vop.Mult(x, qV); Eop.Mult(x, qE);
      const int nC = Cop.n_mult, nV = Vop.n_mult, nE = Eop.n_mult;

      Vector f, h, g;
      F.Mult(x, f);
      Hop.Mult(x, h);
      Gop.Mult(x, g);
      Check(Cop.n_mult == nC + 1 && Vop.n_mult == nV + 1 &&
            Eop.n_mult == nE + 1, "one evaluation per view call");
      Check(Pop.n_mult == 0 && Sop.n_mult == 0,
            "inactive quantities not evaluated");

      real_t f_ex = 0.0;
      for (int i = 0; i < 3; i++) { f_ex += wC(i)*(qC(i) - 0.5); }
      CheckNear(f(0), f_ex, 1e-14, "objective value");
      Vector h_ex(2); h_ex(0) = qE(0) - 0.1; h_ex(1) = qE(1) + 0.2;
      CheckNear(h, h_ex, 1e-14, "h values (shift copied)");
      CheckNear(g(0), qV(0) - 0.3, 1e-14, "g values");

      BlockVector hb(h, prob.GetEqOffsets());
      CheckNear(hb.GetBlock(E)(1), h_ex(1), 1e-14, "BlockVector view of h");
   }

   mfem::out << "Test 3: derivatives and second derivatives\n";
   CheckAllDerivatives(prob, x, v, "base");

   mfem::out << "Test 4: single-row VJP\n";
   {
      prob.SetConstraintType(P, Type::EQ);   // also activates P
      Check(prob.IsActive(P), "SetConstraintType activates");
      Vector y(prob.NumEq()), dx;
      auto [ps, pe] = prob.GetEqRows(P);
      Check(pe - ps == 1, "P row");
      y = 0.0;
      y(ps) = 1.0;                 // only P's row
      Operator &D = Hop.GetGradient(x);
      const int e0 = Eop.der.n_vjp, p0 = Pop.der.n_vjp;
      D.MultTranspose(y, dx);
      Check(Eop.der.n_vjp == e0 + 1 && Pop.der.n_vjp == p0 + 1,
            "every included quantity applied (zero seeds included)");
      Vector dx_ex(n);
      for (int j = 0; j < n; j++) { dx_ex(j) = Pm(0,j); }
      CheckNear(dx, dx_ex, 1e-14, "single-row VJP");
      prob.Deactivate(P);
   }

   mfem::out << "Test 5: hot swap (min V s.t. w.C <= ...)\n";
   {
      const long seq0 = prob.GetLayoutSequence();
      prob.SetObjective(V);
      Check(prob.GetType(C) == Type::INVALID, "old objective -> INVALID");
      Check(prob.GetType(V) == Type::OBJ, "V -> OBJ");
      prob.SetConstraintType(C, Type::LE, true);   // one row w.C
      Check(prob.GetLayoutSequence() > seq0, "layout sequence bumped");
      Check(prob.NumIneq() == 1 && prob.NumEq() == 2, "weighted C: one row");
      Check(Gop.Height() == 1, "g view height updated");

      Vector qC(3), qV(1), f, g;
      Cop.Mult(x, qC); Vop.Mult(x, qV);
      real_t gC = 0.0;
      for (int i = 0; i < 3; i++) { gC += wC(i)*(qC(i) - 0.5); }
      Gop.Mult(x, g);
      F.Mult(x, f);
      CheckNear(g(0), gC, 1e-14, "weighted constraint value");
      CheckNear(f(0), qV(0) - 0.3, 1e-14, "new objective value");
      CheckAllDerivatives(prob, x, v, "swapped");

      prob.SetConstraintType(C, Type::LE, false);  // per-row
      Check(prob.NumIneq() == 3 && Gop.Height() == 3, "unweighted C: rows");
      CheckAllDerivatives(prob, x, v, "per-row");
   }

   mfem::out << "Test 6: deactivate / activate\n";
   {
      const long seq0 = prob.GetLayoutSequence();
      prob.Deactivate(E);
      Check(!prob.IsActive(E) && prob.GetType(E) == Type::EQ,
            "deactivate keeps the role");
      Check(prob.NumEq() == 0 && Hop.Height() == 0, "deactivated rows removed");
      Check(prob.GetLayoutSequence() > seq0, "layout sequence bumped");
      auto [es, ee] = prob.GetEqRows(E);
      Check(es == ee, "deactivated: empty range");
      const int nE = Eop.n_mult;
      Vector h;
      Hop.Mult(x, h);
      Check(h.Size() == 0 && Eop.n_mult == nE, "inactive: not evaluated");
      prob.SetConstraintType(E, Type::EQ);
      Check(prob.IsActive(E) && prob.NumEq() == 2,
            "SetConstraintType reactivates");
      const long seq1 = prob.GetLayoutSequence();
      prob.Activate(E);
      Check(prob.GetLayoutSequence() == seq1, "no-op activation: no rebuild");
   }

   mfem::out << "Test 7: Update notifications\n";
   {
      Check(prob.GetPointSequence() == 0, "no point before Update");
      const long lseq = prob.GetLayoutSequence();
      const int u0 = Pop.n_update;
      const long s1 = prob.Update(x);
      const long s2 = prob.Update(x);
      Check(s1 == 1 && s2 == 2 && prob.GetPointSequence() == 2,
            "monotone point sequence");
      Check(Vop.n_update == 2 && Eop.n_update == 2 && Vop.last_seq == 2,
            "updatable quantities notified with the sequence");
      Check(Pop.n_update == u0 + 2, "inactive quantities notified too");
      Check(prob.GetLayoutSequence() == lseq, "Update keeps the layout");
   }

   mfem::out << "Test 8: SetShift changes values only\n";
   {
      const long seq0 = prob.GetLayoutSequence();
      Vector f0, f1;
      F.Mult(x, f0);
      prob.SetShift(V, 0.5);
      F.Mult(x, f1);
      CheckNear(f1(0), f0(0) - 0.2, 1e-14, "shifted objective");
      Check(prob.GetLayoutSequence() == seq0, "no rebuild");
      prob.SetShift(V, 0.3);
   }

   mfem::out << "Test 9: multi-quantity objectives\n";
   {
      // Scalar quantities: f = 1.0 (V - 0.3) + 0.5 P
      Array<int> qs({V, P});
      Vector w(2); w(0) = 1.0; w(1) = 0.5;
      prob.SetObjective(qs, w);
      Check(prob.IsActive(P), "objective quantity made active");
      Vector qV(1), qP(1), f;
      Vop.Mult(x, qV); Pop.Mult(x, qP);
      F.Mult(x, f);
      CheckNear(f(0), (qV(0) - 0.3) + 0.5*qP(0), 1e-14, "scalar-sum objective");
      CheckAllDerivatives(prob, x, v, "scalar-sum");

      // Vector quantities: f = wC.(C - 0.5) + wE.(E - sE); V, P -> INVALID
      Array<int> qv({C, E});
      Vector wE(2); wE(0) = -1.0; wE(1) = 0.25;
      std::vector<Vector> ws = {wC, wE};
      prob.SetObjective(qv, ws);
      Check(prob.GetType(V) == Type::INVALID &&
            prob.GetType(P) == Type::INVALID,
            "replaced scalar objectives -> INVALID");
      prob.SetConstraintType(V, Type::LE);
      prob.Deactivate(P);
      prob.SetConstraintType(S, Type::LE);
      prob.Activate(S);
      Check(prob.NumEq() == 0 && prob.NumIneq() == 3, "layout after swap");
      CheckAllDerivatives(prob, x, v, "vector-sum");

      // Stored objective weights of E become constraint weights
      prob.SetObjective(C, wC);
      prob.SetConstraintType(E, Type::EQ, true);
      Check(prob.NumEq() == 1, "E weighted from stored weights");
      Vector qE(2), h;
      Eop.Mult(x, qE);
      Hop.Mult(x, h);
      CheckNear(h(0), wE(0)*(qE(0) - 0.1) + wE(1)*(qE(1) + 0.2), 1e-14,
                "weighted EQ value");
      CheckAllDerivatives(prob, x, v, "weighted EQ");

      // Multiplier-weighted Hessian by contraction: sum_i mu_i H_i v
      Vector mu(prob.NumIneq()), Y(prob.NumIneq()*n), Hmu, Hv;
      Fill(mu, 0.5, 0.4);
      for (int i = 0; i < mu.Size(); i++)
      {
         for (int j = 0; j < n; j++) { Y(i*n + j) = mu(i)*v(j); }
      }
      Operator &D2 = Gop.GetGradient(x).GetGradient(x);
      D2.MultTranspose(Y, Hmu);
      D2.Mult(v, Hv);
      Vector Hmu_ex(n);
      Hmu_ex = 0.0;
      for (int i = 0; i < mu.Size(); i++)
      {
         for (int j = 0; j < n; j++) { Hmu_ex(j) += mu(i)*Hv(i*n + j); }
      }
      CheckNear(Hmu, Hmu_ex, 1e-13, "multiplier contraction");
      ContractedOperator Hc(D2, mu);
      Vector Hcv;
      Hc.Mult(v, Hcv);
      CheckNear(Hcv, Hmu_ex, 1e-13, "ContractedOperator::Mult");
      CheckAdjoint(Hc, "ContractedOperator");
   }

   mfem::out << "Test 10: dofwise bounds\n";
   {
      Check(!prob.HasDofBounds(), "no bounds by default");
      const long seq0 = prob.GetLayoutSequence();
      Vector lo(n), hi(n);
      for (int j = 0; j < n; j++) { lo(j) = -j; hi(j) = j + 1.0; }
      hi(0) = infinity();
      prob.SetDofBounds(lo, hi);
      Check(prob.HasDofBounds(), "bounds set");
      lo = 5.0;                                   // bounds are copied
      Check(prob.GetDofLowerBound()(2) == -2.0, "lower bound copied");
      Check(prob.GetDofUpperBound()(0) == infinity(), "unbounded side");
      Check(prob.GetLayoutSequence() == seq0, "bounds do not touch the layout");
      prob.SetDofBounds(0.0, 1.0);
      CheckNear(prob.GetDofLowerBound().Min(), 0.0, 0.0, "uniform lower");
      CheckNear(prob.GetDofUpperBound().Max(), 1.0, 0.0, "uniform upper");
      Check(!prob.GetDofLowerBound().UseDevice(), "uniform: host by default");
      prob.SetDofBounds(-1.0, 2.0, true);
      Check(prob.GetDofLowerBound().UseDevice() &&
            prob.GetDofUpperBound().UseDevice(), "uniform: use_device");
      CheckNear(prob.GetDofUpperBound().Max(), 2.0, 0.0, "uniform upper 2");
   }

   mfem::out << "Test 11: functionals\n";
   {
      DenseMatrix Am(n);
      for (int i = 0; i < n; i++)
      {
         for (int j = 0; j < n; j++)
         {
            Am(i,j) = (i == j) ? 3.0 : 0.5/(1 + i + j);
         }
      }
      Vector bq(n), cl(n);
      for (int j = 0; j < n; j++) { bq(j) = 0.3*j - 0.2; cl(j) = 1.0 + j; }
      QuadraticFunctional fq(Am, bq, 0.7);
      LinearFunctional fl(cl, -0.4);

      Vector y(1), Ax(n);
      Am.Mult(x, Ax);
      fq.Mult(x, y);
      CheckNear(y(0), 0.5*(x*Ax) + bq*x + 0.7, 1e-14, "quadratic value");
      fl.Mult(x, y);
      CheckNear(y(0), cl*x - 0.4, 1e-14, "linear value");
      cl(0) += 1.0;                                  // c is referenced
      fl.Mult(x, y);
      CheckNear(y(0), cl*x - 0.4, 1e-14, "linear value references c");

      CheckDerivative(fq, x, v, "quadratic");
      CheckSecondDerivative(fq, x, v, "quadratic");
      CheckDerivative(fl, x, v, "linear");
      CheckSecondDerivative(fl, x, v, "linear");

      // Through the problem: min fq s.t. fl <= 0
      OptProblem p2(n);
      const int Q = p2.AddQuantity(fq);
      const int L = p2.AddQuantity(fl);
      p2.SetObjective(Q);
      p2.SetConstraintType(L, Type::LE);
      CheckAllDerivatives(p2, x, v, "functional problem");
   }

   mfem::out << "Test 12: GE constraints\n";
   {
      OptProblem p3(n);
      const int Vq = p3.AddQuantity(Vop, 0.3);
      const int Pq = p3.AddQuantity(Pop, 0.1);
      const int Eq = p3.AddQuantity(Eop, 0.2);
      p3.SetObjective(Vq);
      p3.SetConstraintType(Pq, Type::GE);
      p3.SetConstraintType(Eq, Type::LE);
      Check(p3.NumIneq() == 3 && p3.NumEq() == 0, "GE rows in g");
      Check(p3.GetType(Pq) == Type::GE, "GE type kept");
      Vector qP(1), qE(2), g;
      Pop.Mult(x, qP);
      Eop.Mult(x, qE);
      p3.GetInequalityConstraintOperator().Mult(x, g);
      auto [ps, pe] = p3.GetIneqRows(Pq);
      auto [es, ee] = p3.GetIneqRows(Eq);
      CheckNear(g(ps), -(qP(0) - 0.1), 1e-14, "GE exposed as -r <= 0");
      CheckNear(g(es + 1), qE(1) - 0.2, 1e-14, "LE unchanged");
      CheckAllDerivatives(p3, x, v, "GE");
      Vector w1(2); w1(0) = 0.5; w1(1) = -2.0;
      p3.SetConstraintType(Eq, Type::GE, w1);
      Check(p3.NumIneq() == 2, "weighted GE: one row");
      CheckAllDerivatives(p3, x, v, "weighted GE");
   }

   mfem::out << "Test 13: empty and zero objectives\n";
   {
      OptProblem p4(n);
      const int Eq = p4.AddQuantity(Eop);
      p4.SetConstraintType(Eq, Type::EQ);
      const Operator &F4 = p4.GetObjectiveOperator();
      Vector f, df, one(1);
      one = 1.0;
      F4.Mult(x, f);
      F4.GetGradient(x).MultTranspose(one, df);
      Check(f.Size() == 1 && f(0) == 0.0 && df.Normlinf() == 0.0,
            "no objective: f = 0");
      CheckAllDerivatives(p4, x, v, "no objective");

      Vector zero(n);
      zero = 0.0;
      LinearFunctional fz(zero);
      const int Z = p4.AddQuantity(fz);
      p4.SetObjective(Z);
      F4.Mult(x, f);
      Check(f(0) == 0.0, "zero functional");
      CheckAllDerivatives(p4, x, v, "zero functional");

      p4.SetObjective(Array<int>(), std::vector<Vector>());
      Check(p4.GetType(Z) == Type::INVALID, "cleared objective -> INVALID");
      p4.Deactivate(Z);
      F4.Mult(x, f);
      Check(f(0) == 0.0, "cleared objective: f = 0");
   }

   mfem::out << "Test 14: Riesz map\n";
   {
      OptProblem p5(n);
      Check(!p5.HasRieszMap(), "no Riesz map by default");
      IdentityOperator R(n);
      p5.SetRieszMap(R);
      Check(p5.HasRieszMap() && &p5.GetRieszMap() == &R, "Riesz map set");
   }

#ifdef MFEM_USE_MPI
   mfem::out << "Test 15: functionals on a distributed vector\n";
   {
      // Each rank owns n entries; A is block diagonal, so the local block is
      // the local part of the global product.
      const int rank = Mpi::WorldRank(), nranks = Mpi::WorldSize();
      Vector a(n), bq(n), cl(n), xl(n), vl(n);
      for (int i = 0; i < n; i++)
      {
         const int gi = rank*n + i;
         a(i) = 1.0 + gi;
         bq(i) = std::sin(1.0 + gi);
         cl(i) = 0.5 - 0.1*gi;
         xl(i) = std::cos(0.3*gi);
         vl(i) = 0.2 + 0.1*gi;
      }
      SparseMatrix A(a);
      QuadraticFunctional fq(MPI_COMM_WORLD, A, bq, 0.7);
      LinearFunctional fl(MPI_COMM_WORLD, cl, -0.4);
      auto sum = [](real_t s)
      {
         MPI_Allreduce(MPI_IN_PLACE, &s, 1, MFEM_MPI_REAL_T, MPI_SUM,
                       MPI_COMM_WORLD);
         return s;
      };
      Vector y(1), Ax(n), dx(n), one(1);
      one = 1.0;
      A.Mult(xl, Ax);
      fq.Mult(xl, y);
      CheckNear(y(0), sum(0.5*(xl*Ax) + bq*xl) + 0.7, 1e-12,
                "parallel quadratic value");
      fl.Mult(xl, y);
      CheckNear(y(0), sum(cl*xl) - 0.4, 1e-12, "parallel linear value");

      Operator &Dq = fq.GetGradient(xl);
      Dq.Mult(vl, y);                                // global JVP
      Ax += bq;
      CheckNear(y(0), sum(Ax*vl), 1e-12, "parallel quadratic JVP");
      Dq.MultTranspose(one, dx);                     // local covector
      CheckNear(dx, Ax, 1e-14, "parallel quadratic VJP");
      Dq.GetGradient(xl).Mult(vl, dx);               // local Hessian rows
      Vector Av(n);
      A.Mult(vl, Av);
      CheckNear(dx, Av, 1e-14, "parallel quadratic Hessian");
      Operator &Dl = fl.GetGradient(xl);
      Dl.Mult(vl, y);
      CheckNear(y(0), sum(cl*vl), 1e-12, "parallel linear JVP");
      Dl.MultTranspose(one, dx);
      CheckNear(dx, cl, 1e-14, "parallel linear VJP");

      // Through the views: the objective value is the global one and its
      // covector is local.
      OptProblem p6(n);
      p6.SetObjective(p6.AddQuantity(fq));
      p6.SetConstraintType(p6.AddQuantity(fl, 0.1), Type::LE);
      Vector f, g;
      p6.GetObjectiveOperator().Mult(xl, f);
      p6.GetInequalityConstraintOperator().Mult(xl, g);
      CheckNear(f(0), sum(0.5*(xl*(Ax -= bq)) + bq*xl) + 0.7, 1e-12,
                "parallel objective view");
      CheckNear(g(0), sum(cl*xl) - 0.4 - 0.1, 1e-12,
                "parallel constraint view");
      Check(nranks >= 1, "ranks");
   }
#endif

   mfem::out << n_check - n_fail << " / " << n_check << " checks passed\n";
   return n_fail == 0 ? 0 : 1;
}
