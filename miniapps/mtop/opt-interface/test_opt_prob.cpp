/**
 * test_opt_prob.cpp  —  Unit tests for the opt-interface OptimProblem
 *
 * Exercises miniapps/mtop/opt-interface/opt_prob.hpp in isolation (no MMA),
 * so the optimisation-problem abstraction is trusted before the MMA solvers
 * are wrapped on top of it.
 *
 * Coverage
 *   1. StackedOperator / OptimProblem assembly: heights, widths, block order,
 *      Mult(), Objective()/GetEnergy(), GetConstraintType(), ClassifyRows().
 *   2. Stacked derivative: JVP (J dx), VJP (Jᵀ dy), the adjoint identity
 *      <J dx, dy> = <dx, Jᵀ dy>, analytic Jacobian match, and x-dependence
 *      of a nonlinear (separable-quadratic) objective gradient.
 *   3. SetObjective / ReplaceObjective bookkeeping.
 *   4. Dof bounds accessors and Has* predicates.
 *   5. InnerProduct and Riesz-map plumbing.
 *   6. StackedOptimizationProblem adapter: CalcObjective/Grad, the EQ (C) and
 *      LE (D) operators and their dense Jacobians, and the constraint/bound
 *      vectors handed to a downstream OptimizationSolver.
 *
 * main() returns 0 on success, non-zero on any failure (CTest convention).
 */

#include "opt_prob.hpp"   // pulls in mfem.hpp
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

using namespace mfem;

// ── Tiny check harness (matches the other mma tests) ──────────────────────
static int g_nfail = 0;
static void Check(bool cond, const char *msg)
{
   if (cond) { printf("  [PASS] %s\n", msg); }
   else      { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}
static bool Close(real_t a, real_t b, real_t tol = 1e-10)
{
   return std::abs(double(a) - double(b)) <= tol * (1.0 + std::abs(double(b)));
}
static real_t MaxDiff(const Vector &a, const Vector &b)
{
   real_t m = 0.0;
   for (int j = 0; j < a.Size(); ++j)
   { m = std::max(m, real_t(std::abs(double(a(j) - b(j))))); }
   return m;
}

// ── Test operators with known Mult and Jacobian ───────────────────────────

/// Affine operator  y = A x + b.  Constant Jacobian A (h x n).
class AffineOp : public Operator
{
public:
   AffineOp(const DenseMatrix &A_, const Vector &b_)
      : Operator(A_.Height(), A_.Width()), A(A_), b(b_) {}

   void Mult(const Vector &x, Vector &y) const override
   {
      A.Mult(x, y);   // y = A x
      y += b;         // y = A x + b
   }
   /// Jacobian is the constant matrix A (itself an Operator).
   Operator &GetGradient(const Vector &) const override
   {
      return const_cast<DenseMatrix &>(A);
   }
private:
   DenseMatrix A;
   Vector b;
};

/// Height-1 separable quadratic  f(x) = c + 0.5 Σ_j w_j x_j².
/// Jacobian is the 1 x n row  g_j = w_j x_j  (depends on x).
class SepQuadObjective : public Operator
{
public:
   SepQuadObjective(const Vector &w_, real_t c_ = 0.0)
      : Operator(1, w_.Size()), w(w_), c(c_), g(w_.Size()), row(g) {}

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(1);
      real_t s = c;
      for (int j = 0; j < w.Size(); ++j) { s += 0.5 * w(j) * x(j) * x(j); }
      y(0) = s;
   }
   Operator &GetGradient(const Vector &x) const override
   {
      for (int j = 0; j < w.Size(); ++j) { g(j) = w(j) * x(j); }
      return row;
   }
private:
   /// 1 x n Jacobian view backed by the parent's gradient row g.
   class Row : public Operator
   {
   public:
      Row(const Vector &g_) : Operator(1, g_.Size()), g(g_) {}
      void Mult(const Vector &dx, Vector &y) const override
      {
         y.SetSize(1);
         y(0) = InnerProduct(g, dx);
      }
      void MultTranspose(const Vector &dy, Vector &dx) const override
      {
         dx.SetSize(g.Size());
         dx = g;
         dx *= dy(0);
      }
   private:
      const Vector &g;
   };
   Vector w;
   real_t c;
   mutable Vector g;
   mutable Row row;
};

/// Diagonal inner product  <x,y>_D = Σ_j d_j x_j y_j.
class DiagInnerProduct : public InnerProductOperator
{
public:
   DiagInnerProduct(const Vector &d_) : d(d_) {}
   real_t Eval(const Vector &x, const Vector &y) override
   {
      real_t s = 0.0;
      for (int j = 0; j < d.Size(); ++j) { s += d(j) * x(j) * y(j); }
      return s;
   }
private:
   Vector d;
};

/// Diagonal Riesz map (inverse metric)  y_j = x_j / d_j.
class DiagRieszMap : public Operator
{
public:
   DiagRieszMap(const Vector &d) : Operator(d.Size()), inv(d.Size())
   {
      for (int j = 0; j < d.Size(); ++j) { inv(j) = real_t(1.0) / d(j); }
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      for (int j = 0; j < x.Size(); ++j) { y(j) = inv(j) * x(j); }
   }
private:
   Vector inv;
};

// ── Fixtures ──────────────────────────────────────────────────────────────
static const int N = 4;   // number of design variables in the canonical problem

// Deterministic sample points / seeds (no RNG so runs are reproducible).
static Vector SampleX(real_t shift = 0.0)
{
   Vector x(N);
   for (int j = 0; j < N; ++j) { x(j) = real_t(0.3 + 0.15 * j + shift); }
   return x;
}
static Vector SampleDX()
{
   Vector v(N);
   for (int j = 0; j < N; ++j) { v(j) = real_t(std::sin(0.7 * (j + 1))); }
   return v;
}

// Objective weights and constraint matrices used across several tests.
static Vector Wobj()
{
   Vector w(N);
   w(0) = 1.0; w(1) = 2.0; w(2) = 3.0; w(3) = 4.0;
   return w;
}
static DenseMatrix A_le()   // 2 x N inequality Jacobian
{
   DenseMatrix A(2, N);
   A(0, 0) = 1.0; A(0, 1) = 0.5; A(0, 2) = -0.25; A(0, 3) = 2.0;
   A(1, 0) = 0.0; A(1, 1) = 1.0; A(1, 2) =  3.0;  A(1, 3) = -1.0;
   return A;
}
static Vector b_le() { Vector b(2); b(0) = -0.7; b(1) = 0.2; return b; }
static DenseMatrix A_eq()   // 1 x N equality Jacobian
{
   DenseMatrix A(1, N);
   A(0, 0) = 1.0; A(0, 1) = 1.0; A(0, 2) = 1.0; A(0, 3) = 1.0;
   return A;
}
static Vector b_eq() { Vector b(1); b(0) = -1.0; return b; }

/// Build the canonical problem: OBJ (row 0), 2 LE (rows 1,2), 1 EQ (row 3).
/// Ownership of the operators transfers to the returned OptimProblem.
static void BuildCanonical(OptimProblem &prob)
{
   prob.SetObjective(new SepQuadObjective(Wobj()));
   prob.AddConstraint(new AffineOp(A_le(), b_le()), OptimProblem::ConstType::LE);
   prob.AddConstraint(new AffineOp(A_eq(), b_eq()), OptimProblem::ConstType::EQ);
   prob.Finalize();
}

// Reference evaluations (independent recomputation of the same math).
static real_t RefObjective(const Vector &x)
{
   Vector w = Wobj();
   real_t s = 0.0;
   for (int j = 0; j < N; ++j) { s += 0.5 * w(j) * x(j) * x(j); }
   return s;
}
static Vector RefFull(const Vector &x)   // stacked [obj; le0; le1; eq]
{
   Vector y(4);
   y(0) = RefObjective(x);
   Vector le(2); A_le().Mult(x, le); le += b_le();
   y(1) = le(0); y(2) = le(1);
   Vector eq(1); A_eq().Mult(x, eq); eq += b_eq();
   y(3) = eq(0);
   return y;
}

// ── Test 1: assembly, Mult, Objective, ClassifyRows ───────────────────────
static void Test_Assembly()
{
   printf("\n--- Assembly / Mult / ClassifyRows ---\n");
   OptimProblem prob(N);
   BuildCanonical(prob);

   Check(prob.IsFinalized(),       "problem finalized");
   Check(prob.Width()  == N,       "Width == n");
   Check(prob.Height() == 4,       "Height == 1 + 2 + 1");

   Vector x = SampleX();
   Vector y(prob.Height());
   prob.Mult(x, y);
   Check(MaxDiff(y, RefFull(x)) < 1e-12, "Mult matches reference stacked output");

   Check(Close(prob.Objective(x), RefObjective(x)), "Objective(x) correct");
   Check(Close(prob.GetEnergy(x), RefObjective(x)), "GetEnergy == Objective");

   Array<int> eq_rows, le_rows; int obj_row = -99;
   prob.ClassifyRows(eq_rows, le_rows, obj_row);
   Check(obj_row == 0,                              "obj row == 0");
   Check(le_rows.Size() == 2 && le_rows[0] == 1 && le_rows[1] == 2,
         "LE rows == {1,2}");
   Check(eq_rows.Size() == 1 && eq_rows[0] == 3,    "EQ row == {3}");

   // Per-block constraint-type bookkeeping (blocks: obj, le(2), eq(1)).
   Check(prob.GetConstraintType(0)[0] == OptimProblem::ConstType::OBJ,
         "block 0 tagged OBJ");
   Check(prob.GetConstraintType(1).Size() == 2 &&
         prob.GetConstraintType(1)[0] == OptimProblem::ConstType::LE &&
         prob.GetConstraintType(1)[1] == OptimProblem::ConstType::LE,
         "block 1 tagged {LE,LE}");
   Check(prob.GetConstraintType(2)[0] == OptimProblem::ConstType::EQ,
         "block 2 tagged EQ");
}

// ── Test 2: stacked derivative (JVP / VJP / adjoint / analytic) ────────────
static void Test_Derivative()
{
   printf("\n--- Stacked derivative: JVP / VJP ---\n");
   OptimProblem prob(N);
   BuildCanonical(prob);

   Vector x = SampleX();
   Vector dx = SampleDX();
   Operator &J = prob.GetGradient(x);

   // Analytic JVP: row0 = g·dx (g = w.*x), rows1,2 = A_le dx, row3 = A_eq dx.
   Vector jvp(prob.Height());
   J.Mult(dx, jvp);

   Vector g(N); { Vector w = Wobj(); for (int j = 0; j < N; ++j) { g(j) = w(j) * x(j); } }
   Vector ref(4);
   ref(0) = InnerProduct(g, dx);
   Vector le(2); A_le().Mult(dx, le); ref(1) = le(0); ref(2) = le(1);
   Vector eq(1); A_eq().Mult(dx, eq); ref(3) = eq(0);
   Check(MaxDiff(jvp, ref) < 1e-12, "JVP matches analytic Jacobian action");

   // Analytic VJP: Jᵀ dy = g*dy0 + A_leᵀ dy[1:2] + A_eqᵀ dy3.
   Vector dy(4);
   for (int i = 0; i < 4; ++i) { dy(i) = real_t(std::cos(0.9 * (i + 1))); }
   Vector vjp(N);
   J.MultTranspose(dy, vjp);

   Vector refT(N); refT = 0.0;
   { for (int j = 0; j < N; ++j) { refT(j) += g(j) * dy(0); } }
   { Vector dle(2); dle(0) = dy(1); dle(1) = dy(2);
     Vector t(N); A_le().MultTranspose(dle, t); refT += t; }
   { Vector deq(1); deq(0) = dy(3);
     Vector t(N); A_eq().MultTranspose(deq, t); refT += t; }
   Check(MaxDiff(vjp, refT) < 1e-12, "VJP matches analytic transpose action");

   // Adjoint identity <J dx, dy> == <dx, Jᵀ dy> (self-consistency, no analytics).
   real_t lhs = InnerProduct(jvp, dy);
   real_t rhs = InnerProduct(dx, vjp);
   Check(Close(lhs, rhs, 1e-10), "adjoint identity <Jdx,dy>==<dx,J^T dy>");

   // Nonlinear objective gradient must depend on x: re-point and recheck row 0.
   Vector x2 = SampleX(0.4);
   Operator &J2 = prob.GetGradient(x2);
   Vector jvp2(prob.Height());
   J2.Mult(dx, jvp2);
   Vector g2(N); { Vector w = Wobj(); for (int j = 0; j < N; ++j) { g2(j) = w(j) * x2(j); } }
   Check(Close(jvp2(0), InnerProduct(g2, dx)), "obj-grad row tracks new x");
   Check(std::abs(double(jvp2(0) - jvp(0))) > 1e-6, "obj gradient changed with x");
}

// ── Test 3: SetObjective / ReplaceObjective ───────────────────────────────
static void Test_ReplaceObjective()
{
   printf("\n--- SetObjective / ReplaceObjective ---\n");
   OptimProblem prob(N);
   // Block 0: dedicated objective (OBJ at global row 0).
   prob.SetObjective(new SepQuadObjective(Wobj()));
   // Block 1: 2-output affine, both LE initially (global rows 1,2).
   prob.AddConstraint(new AffineOp(A_le(), b_le()), OptimProblem::ConstType::LE);
   prob.Finalize();

   Vector x = SampleX();
   Check(Close(prob.Objective(x), RefObjective(x)), "objective before replace");

   // Move OBJ to block 1, output 1 (global row 2); demote old objective to LE.
   prob.ReplaceObjective(/*org_blk*/0, /*org_loc*/0,
                         /*new_blk*/1, /*new_loc*/1,
                         OptimProblem::ConstType::LE);

   Vector le(2); A_le().Mult(x, le); le += b_le();
   Check(Close(prob.Objective(x), le(1)), "objective now block1 output1");

   Array<int> eq_rows, le_rows; int obj_row = -99;
   prob.ClassifyRows(eq_rows, le_rows, obj_row);
   Check(obj_row == 2, "OBJ row moved to 2");
   Check(le_rows.Size() == 2 && le_rows[0] == 0 && le_rows[1] == 1,
         "demoted obj + surviving LE -> rows {0,1}");
   Check(eq_rows.Size() == 0, "no EQ rows");
}

// ── Test 4: dof bounds ────────────────────────────────────────────────────
static void Test_Bounds()
{
   printf("\n--- Dof bounds ---\n");
   OptimProblem prob(N);
   BuildCanonical(prob);

   Check(!prob.HasDofBounds() && !prob.HasDofLowerBound() &&
         !prob.HasDofUpperBound(), "no bounds initially");

   Vector lb(N), ub(N);
   for (int j = 0; j < N; ++j) { lb(j) = real_t(-1.0 - j); ub(j) = real_t(2.0 + j); }
   prob.SetDofBounds(lb, ub);
   Check(prob.HasDofBounds(),                          "HasDofBounds after set");
   Check(MaxDiff(prob.GetDofLowerBound(), lb) == 0.0,  "lower bound stored");
   Check(MaxDiff(prob.GetDofUpperBound(), ub) == 0.0,  "upper bound stored");

   // One-sided set on a fresh problem.
   OptimProblem prob2(N);
   BuildCanonical(prob2);
   prob2.SetDofLowerBound(lb);
   Check(prob2.HasDofLowerBound() && !prob2.HasDofUpperBound(),
         "one-sided lower bound only");
}

// ── Test 5: inner product and Riesz map ───────────────────────────────────
static void Test_MetricRiesz()
{
   printf("\n--- InnerProduct / Riesz map ---\n");
   OptimProblem prob(N);
   BuildCanonical(prob);

   Check(!prob.HasInnerProduct(), "no inner product initially");
   Check(!prob.HasRieszMap(),     "no Riesz map initially");

   Vector d(N);
   for (int j = 0; j < N; ++j) { d(j) = real_t(1.0 + 0.5 * j); }
   prob.SetInnerProduct(new DiagInnerProduct(d));
   prob.SetRieszMap(new DiagRieszMap(d));
   Check(prob.HasInnerProduct(), "HasInnerProduct after set");
   Check(prob.HasRieszMap(),     "HasRieszMap after set");

   Vector a = SampleX(), b = SampleX(0.2);
   real_t ref = 0.0;
   for (int j = 0; j < N; ++j) { ref += d(j) * a(j) * b(j); }
   Check(Close(prob.GetInnerProduct().Eval(a, b), ref), "weighted dot correct");

   // Riesz map inverts the metric: <Riesz a, b>_D == <a,b>_{l2}.
   Vector ra(N); prob.GetRieszMap().Mult(a, ra);
   real_t lhs = 0.0, rhs = 0.0;
   for (int j = 0; j < N; ++j) { lhs += d(j) * ra(j) * b(j); rhs += a(j) * b(j); }
   Check(Close(lhs, rhs), "Riesz map inverts the metric");
}

// ── Test 6: StackedOptimizationProblem adapter ────────────────────────────
static void Test_Adapter()
{
   printf("\n--- StackedOptimizationProblem adapter ---\n");
   OptimProblem prob(N);
   BuildCanonical(prob);
   Vector lb(N), ub(N);
   for (int j = 0; j < N; ++j) { lb(j) = real_t(-1.0 - j); ub(j) = real_t(2.0 + j); }
   prob.SetDofBounds(lb, ub);

   StackedOptimizationProblem sopt(prob);
   Vector x = SampleX();

   Check(sopt.input_size == N,                          "input_size == n");
   Check(Close(sopt.CalcObjective(x), RefObjective(x)), "CalcObjective == Objective");

   // Objective gradient = w.*x (VJP with the objective-row unit seed).
   Vector grad(N); sopt.CalcObjectiveGrad(x, grad);
   Vector g(N); { Vector w = Wobj(); for (int j = 0; j < N; ++j) { g(j) = w(j) * x(j); } }
   Check(MaxDiff(grad, g) < 1e-12, "CalcObjectiveGrad == w.*x");

   Check(sopt.GetNumConstraints() == 3, "num constraints == 3 (1 EQ + 2 LE)");

   // Equality operator C(x): selects the EQ row (global row 3).
   const Operator *C = sopt.GetC();
   Check(C && C->Height() == 1 && C->Width() == N, "C is 1 x n");
   {
      Vector c(1); C->Mult(x, c);
      Vector eq(1); A_eq().Mult(x, eq); eq += b_eq();
      Check(Close(c(0), eq(0)), "C(x) == equality row value");
      // Dense Jacobian of C == A_eq.
      Operator &Jc = const_cast<Operator*>(C)->GetGradient(x);
      DenseMatrix *Dc = dynamic_cast<DenseMatrix*>(&Jc);
      Check(Dc != nullptr, "C gradient is a DenseMatrix");
      bool ok = (Dc && Dc->Height() == 1 && Dc->Width() == N);
      DenseMatrix Ae = A_eq();
      for (int j = 0; ok && j < N; ++j) { ok = Close((*Dc)(0, j), Ae(0, j)); }
      Check(ok, "C dense Jacobian == A_eq");
   }

   // Inequality operator D(x): selects the LE rows (global rows 1,2).
   const Operator *D = sopt.GetD();
   Check(D && D->Height() == 2 && D->Width() == N, "D is 2 x n");
   {
      Vector dvals(2); D->Mult(x, dvals);
      Vector le(2); A_le().Mult(x, le); le += b_le();
      Check(MaxDiff(dvals, le) < 1e-12, "D(x) == inequality row values");
      Operator &Jd = const_cast<Operator*>(D)->GetGradient(x);
      DenseMatrix *Dd = dynamic_cast<DenseMatrix*>(&Jd);
      Check(Dd != nullptr, "D gradient is a DenseMatrix");
      bool ok = (Dd && Dd->Height() == 2 && Dd->Width() == N);
      DenseMatrix Al = A_le();
      for (int i = 0; ok && i < 2; ++i)
         for (int j = 0; ok && j < N; ++j) { ok = Close((*Dd)(i, j), Al(i, j)); }
      Check(ok, "D dense Jacobian == A_le");
   }

   // Constraint / bound vectors handed to a downstream OptimizationSolver.
   Check(sopt.GetEqualityVec() && sopt.GetEqualityVec()->Size() == 1 &&
         Close((*sopt.GetEqualityVec())(0), 0.0), "equality RHS c_e == 0");
   const Vector *dlo = sopt.GetInequalityVec_Lo();
   const Vector *dhi = sopt.GetInequalityVec_Hi();
   Check(dlo && dhi && dlo->Size() == 2 && dhi->Size() == 2,
         "inequality bound vectors sized 2");
   Check(dlo && (*dlo)(0) == -infinity() && (*dlo)(1) == -infinity(),
         "inequality lower == -inf");
   Check(dhi && (*dhi)(0) == 0.0 && (*dhi)(1) == 0.0, "inequality upper == 0");
   Check(sopt.GetBoundsVec_Lo() && MaxDiff(*sopt.GetBoundsVec_Lo(), lb) == 0.0,
         "solution lower bound propagated");
   Check(sopt.GetBoundsVec_Hi() && MaxDiff(*sopt.GetBoundsVec_Hi(), ub) == 0.0,
         "solution upper bound propagated");
}

int main()
{
   printf("=== OptimProblem (opt-interface) unit tests ===\n");

   Test_Assembly();
   Test_Derivative();
   Test_ReplaceObjective();
   Test_Bounds();
   Test_MetricRiesz();
   Test_Adapter();

   printf("\n========================================\n");
   if (g_nfail == 0) { printf("All OptimProblem tests PASSED.\n"); }
   else              { printf("%d OptimProblem test(s) FAILED.\n", g_nfail); }
   printf("========================================\n");
   return g_nfail > 0 ? 1 : 0;
}
