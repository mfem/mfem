/**
 * test_opt_mma.cpp  —  End-to-end test of the MMA -> OptimizationSolver bridge.
 *
 * Drives a known MMA problem through the full stack:
 *     OptimProblem  ->  StackedOptimizationProblem  ->  MMAOptimizationSolver
 * and checks it reproduces the analytic optimum, using only the
 * IterativeSolver controls (SetAbsTol / SetMaxIter, GetConverged /
 * GetNumIterations / GetFinalNorm).
 *
 * Problems (min-compliance proxy, analytic optimum = uniform x = Vfrac):
 *   1. Inequality:  min Σ 1/x_j   s.t.  mean(x) <= Vfrac        (LE path)
 *   2. Equality:    min Σ 1/x_j   s.t.  mean(x)  = Vfrac        (±h EQ path)
 * with 0.001 <= x <= 1.  Also exercises SetRieszMap (identity = no-op, and a
 * nontrivial diagonal metric) and SetInitialGuess.
 *
 * A second group uses a hand-built mfem::OptimizationProblem (NOT via the
 * StackedOptimizationProblem adapter) to cover the wrapper paths the adapter
 * never emits: a nonzero equality RHS c_e, and a two-sided d_lo <= D <= d_hi
 * with an active upper / lower side.  Objective 0.5||x - a||².
 */

#include "MMAOptSolver.hpp"
#include "opt_prob.hpp"
#include <cmath>
#include <cstdio>
#include <memory>

using namespace mfem;
using namespace mfem_mma;

static int g_nfail = 0;
static void Check(bool cond, const char *msg)
{
   if (cond) { printf("  [PASS] %s\n", msg); }
   else      { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// Objective  f(x) = Σ_j 1/x_j.  Jacobian row  g_j = -1/x_j²  (1 x n).
class ComplianceObj : public Operator
{
public:
   ComplianceObj(int n) : Operator(1, n), g(n), row(g) {}
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(1);
      real_t s = 0.0;
      for (int j = 0; j < width; ++j) { s += real_t(1.0) / x(j); }
      y(0) = s;
   }
   Operator &GetGradient(const Vector &x) const override
   {
      for (int j = 0; j < width; ++j) { g(j) = real_t(-1.0) / (x(j) * x(j)); }
      return row;
   }
private:
   class Row : public Operator
   {
   public:
      Row(const Vector &g_) : Operator(1, g_.Size()), g(g_) {}
      void Mult(const Vector &dx, Vector &y) const override
      { y.SetSize(1); y(0) = InnerProduct(g, dx); }
      void MultTranspose(const Vector &dy, Vector &dx) const override
      { dx.SetSize(g.Size()); dx = g; dx *= dy(0); }
   private:
      const Vector &g;
   };
   mutable Vector g;
   mutable Row row;
};

// Mean operator  m(x) = (1/n) Σ_j x_j - Vfrac  (1 x n affine, constant grad).
class MeanConstraint : public Operator
{
public:
   MeanConstraint(int n, real_t Vfrac) : Operator(1, n), vf(Vfrac), A(1, n)
   {
      for (int j = 0; j < n; ++j) { A(0, j) = real_t(1.0) / n; }
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      A.Mult(x, y);
      y(0) -= vf;
   }
   Operator &GetGradient(const Vector &) const override
   {
      return const_cast<DenseMatrix &>(A);
   }
private:
   real_t vf;
   DenseMatrix A;
};

// Mean over a contiguous block  m(x) = (1/size) Σ_{block} x_j - target
// (1 x n affine, constant gradient).  Used for multi-constraint problems.
class BlockMeanConstraint : public Operator
{
public:
   BlockMeanConstraint(int n, int start, int size, real_t target)
      : Operator(1, n), tgt(target), A(1, n)
   {
      A = 0.0;
      for (int j = start; j < start + size; ++j) { A(0, j) = real_t(1.0) / size; }
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      A.Mult(x, y);
      y(0) -= tgt;
   }
   Operator &GetGradient(const Vector &) const override
   {
      return const_cast<DenseMatrix &>(A);
   }
private:
   real_t tgt;
   DenseMatrix A;
};

// Nonlinear constraint  g(x) = (1/n) Σ_j x_j² - s²  with x-dependent Jacobian
// row  2 x_j / n.  Exercises the wrapper's VJP on a non-constant Jacobian.
class SumSqConstraint : public Operator
{
public:
   SumSqConstraint(int n, real_t s2) : Operator(1, n), s2_(s2), g(n), row(g) {}
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(1);
      real_t s = 0.0;
      for (int j = 0; j < width; ++j) { s += x(j) * x(j); }
      y(0) = s / width - s2_;
   }
   Operator &GetGradient(const Vector &x) const override
   {
      for (int j = 0; j < width; ++j) { g(j) = real_t(2.0) * x(j) / width; }
      return row;
   }
private:
   class Row : public Operator
   {
   public:
      Row(const Vector &g_) : Operator(1, g_.Size()), g(g_) {}
      void Mult(const Vector &dx, Vector &y) const override
      { y.SetSize(1); y(0) = InnerProduct(g, dx); }
      void MultTranspose(const Vector &dy, Vector &dx) const override
      { dx.SetSize(g.Size()); dx = g; dx *= dy(0); }
   private:
      const Vector &g;
   };
   real_t s2_;
   mutable Vector g;
   mutable Row row;
};

// SPD diagonal operator, used as a Riesz map R = M⁻¹.
class DiagOperator : public Operator
{
public:
   DiagOperator(const Vector &diag) : Operator(diag.Size()), d(diag) {}
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      for (int j = 0; j < x.Size(); ++j) { y(j) = d(j) * x(j); }
   }
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
private:
   Vector d;
};

// Affine operator  y = A x, constant Jacobian A.  Used for linear C / D blocks.
class AffineOp : public Operator
{
public:
   AffineOp(const DenseMatrix &A_) : Operator(A_.Height(), A_.Width()), A(A_) {}
   void Mult(const Vector &x, Vector &y) const override { A.Mult(x, y); }
   Operator &GetGradient(const Vector &) const override
   { return const_cast<DenseMatrix &>(A); }
private:
   DenseMatrix A;
};

// General mfem::OptimizationProblem (bypasses StackedOptimizationProblem) to
// exercise the wrapper's c_e != 0 and two-sided d_lo <= D <= d_hi paths.
// Objective 0.5||x - a||².  C / D (if any) transfer ownership.
class QuadTargetProblem : public OptimizationProblem
{
public:
   QuadTargetProblem(const Vector &a, Operator *C_, Operator *D_,
                     const Vector *ce, const Vector *dlo, const Vector *dhi,
                     const Vector &xlo, const Vector &xhi)
      : OptimizationProblem(a.Size(), C_, D_), a_(a),
        C_own(C_), D_own(D_), xlo_(xlo), xhi_(xhi)
   {
      if (C_ && ce)         { ce_  = *ce;  SetEqualityConstraint(ce_); }
      if (D_ && dlo && dhi) { dlo_ = *dlo; dhi_ = *dhi;
                              SetInequalityConstraint(dlo_, dhi_); }
      SetSolutionBounds(xlo_, xhi_);
   }
   real_t CalcObjective(const Vector &x) const override
   {
      real_t s = 0.0;
      for (int j = 0; j < a_.Size(); ++j) { real_t d = x(j) - a_(j); s += 0.5 * d * d; }
      return s;
   }
   void CalcObjectiveGrad(const Vector &x, Vector &g) const override
   {
      g.SetSize(a_.Size());
      for (int j = 0; j < a_.Size(); ++j) { g(j) = x(j) - a_(j); }
   }
private:
   Vector a_;
   std::unique_ptr<Operator> C_own, D_own;
   Vector ce_, dlo_, dhi_, xlo_, xhi_;
};

// Compliance objective  min Σ 1/x_j  with a single linear equality C(x)=c_e
// (nonzero RHS carried in c_e, NOT folded into the operator).  Well scaled for
// MMA, unlike a quadratic objective whose gradient vanishes at the optimum.
class ComplianceEqProblem : public OptimizationProblem
{
public:
   ComplianceEqProblem(int n, Operator *C_, const Vector &ce,
                       const Vector &xlo, const Vector &xhi)
      : OptimizationProblem(n, C_, nullptr),
        C_own(C_), ce_(ce), xlo_(xlo), xhi_(xhi)
   {
      SetEqualityConstraint(ce_);
      SetSolutionBounds(xlo_, xhi_);
   }
   real_t CalcObjective(const Vector &x) const override
   {
      real_t s = 0.0;
      for (int j = 0; j < input_size; ++j) { s += real_t(1.0) / x(j); }
      return s;
   }
   void CalcObjectiveGrad(const Vector &x, Vector &g) const override
   {
      g.SetSize(input_size);
      for (int j = 0; j < input_size; ++j) { g(j) = real_t(-1.0) / (x(j) * x(j)); }
   }
private:
   std::unique_ptr<Operator> C_own;
   Vector ce_, xlo_, xhi_;
};

// riesz: optional metric (not owned).  via_setguess: exercise SetInitialGuess
// (with a decoy xt that must be ignored).  Returns the iteration count.
static int SolveAndCheck(bool equality, int n, real_t Vfrac,
                         const Operator *riesz, bool via_setguess,
                         const char *tag)
{
   printf("\n--- Compliance proxy (n=%d, Vfrac=%.2f, %s, %s) ---\n",
          n, Vfrac, equality ? "EQ" : "LE", tag);

   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.AddConstraint(new MeanConstraint(n, Vfrac),
                      equality ? OptimProblem::ConstType::EQ
                               : OptimProblem::ConstType::LE);
   prob.Finalize();

   Vector lb(n), ub(n);
   lb = real_t(0.001); ub = real_t(1.0);
   prob.SetDofBounds(lb, ub);

   StackedOptimizationProblem sopt(prob);

   MMAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   if (riesz) { solver.SetRieszMap(*riesz); }
   solver.SetRelTol(0.0);
   solver.SetAbsTol(1e-5);
   solver.SetMaxIter(300);

   Vector x(n);
   if (via_setguess)
   {
      Vector x0(n); x0 = real_t(0.5);
      solver.SetInitialGuess(x0);
      Vector decoy(n); decoy = real_t(0.9);   // must be ignored
      solver.Mult(decoy, x);
   }
   else
   {
      Vector x0(n); x0 = real_t(0.5);
      solver.Mult(x0, x);
   }

   real_t xmean = 0.0, maxerr = 0.0;
   for (int j = 0; j < n; ++j)
   {
      xmean += x(j);
      maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - Vfrac))));
   }
   xmean /= n;

   printf("  converged=%d iters=%d kkt=%.3e xmean=%.6f(%.2f) maxerr=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(),
          solver.GetFinalNorm(), xmean, Vfrac, maxerr);

   Check(solver.GetConverged(),                  "solver reports converged");
   Check(solver.GetFinalNorm() < 1e-4,           "final KKT < 1e-4");
   Check(std::abs(double(xmean - Vfrac)) < 0.01, "volume fraction met");
   Check(maxerr < 0.05,                          "uniform design");
   Check(solver.GetNumIterations() > 0,          "took > 0 iterations");
   return solver.GetNumIterations();
}

// ── General-path tests (custom OptimizationProblem, not via Stacked) ─────────

// Equality with nonzero RHS: min Σ1/x_j  s.t.  Σx = c_e (= n·Vfrac).  Exercises
// the wrapper's c_e != 0 path (the StackedOptimizationProblem adapter only ever
// emits c_e = 0).  Optimum is uniform x_j = Vfrac.
static void Test_Equality_NonzeroRHS()
{
   printf("\n--- General: equality c_e != 0 (Σx = n·Vfrac) ---\n");
   const int n = 100;
   const real_t Vfrac = real_t(0.4);

   DenseMatrix Ac(1, n); for (int j = 0; j < n; ++j) { Ac(0, j) = 1.0; }  // Σx
   Vector ce(1); ce(0) = real_t(n) * Vfrac;                                // = 40
   Vector xlo(n); xlo = real_t(0.001);
   Vector xhi(n); xhi = real_t(1.0);

   ComplianceEqProblem prob(n, new AffineOp(Ac), ce, xlo, xhi);
   MMAOptimizationSolver solver;
   solver.SetOptimizationProblem(prob);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);

   Vector x0(n); x0 = real_t(0.5); Vector x(n);
   solver.Mult(x0, x);

   real_t maxerr = 0.0, sum = 0.0;
   for (int j = 0; j < n; ++j)
   {
      maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - Vfrac))));
      sum += x(j);
   }
   printf("  converged=%d iters=%d kkt=%.3e sum=%.6f(%.2f) maxerr=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(),
          solver.GetFinalNorm(), double(sum), double(ce(0)), double(maxerr));
   Check(solver.GetConverged(),                        "converged");
   Check(std::abs(double(sum - ce(0))) < 1e-2,         "equality Σx = c_e satisfied");
   Check(maxerr < 0.02,                                "uniform x = Vfrac");
}

// Two-sided inequality with one active side: min 0.5(x-a)²  s.t.  1 <= x <= 3
// (encoded as an inequality operator D(x)=x, NOT as dof bounds).  a>3 pins the
// upper side (x*=3); a<1 pins the lower side (x*=1).
static void Test_TwoSided(real_t aval, real_t expected, const char *tag)
{
   printf("\n--- General: two-sided ineq, %s ---\n", tag);
   const int n = 1;
   Vector a(1); a(0) = aval;
   DenseMatrix Ad(1, 1); Ad(0, 0) = 1.0;              // D(x) = x
   Vector dlo(1); dlo(0) = real_t(1.0);
   Vector dhi(1); dhi(0) = real_t(3.0);
   Vector xlo(1); xlo(0) = real_t(-10.0);
   Vector xhi(1); xhi(0) = real_t(10.0);

   QuadTargetProblem prob(a, nullptr, new AffineOp(Ad),
                          nullptr, &dlo, &dhi, xlo, xhi);
   MMAOptimizationSolver solver;
   solver.SetOptimizationProblem(prob);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-7); solver.SetMaxIter(300);

   Vector x0(1); x0(0) = 0.0; Vector x(1);
   solver.Mult(x0, x);

   printf("  converged=%d iters=%d kkt=%.3e x=%.6f (expected %.2f)\n",
          solver.GetConverged(), solver.GetNumIterations(),
          solver.GetFinalNorm(), double(x(0)), double(expected));
   Check(solver.GetConverged(),                       "converged");
   Check(std::abs(double(x(0) - expected)) < 1e-3,    "x at active constraint");
}

// Matrix-free device-aware constraint Jacobian + UseDevice(true) design
// vectors: exercises the StackedOptimizationProblem(matrix_free_grad=true) path
// (MFGrad instead of DenseMatrix) and the wrapper's device buffers. On a CPU
// build the "device" is the host, but the Read/Write/GetSubVector code paths
// are the same ones a GPU build would take.
static void Test_MatrixFreeDevice()
{
   printf("\n--- Matrix-free device-aware Jacobian (UseDevice) ---\n");
   const int n = 100;
   const real_t Vfrac = real_t(0.4);

   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.AddConstraint(new MeanConstraint(n, Vfrac), OptimProblem::ConstType::LE);
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0);
   prob.SetDofBounds(lb, ub);

   StackedOptimizationProblem sopt(prob, /*matrix_free_grad=*/true);
   MMAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);

   Vector x0(n), x(n);
   x0.UseDevice(true); x.UseDevice(true);
   x0 = real_t(0.5);
   solver.Mult(x0, x);

   real_t xmean = 0.0, maxerr = 0.0;
   for (int j = 0; j < n; ++j)
   {
      xmean += x(j);
      maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - Vfrac))));
   }
   xmean /= n;
   printf("  converged=%d iters=%d kkt=%.3e xmean=%.6f maxerr=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(),
          solver.GetFinalNorm(), double(xmean), double(maxerr));
   Check(solver.GetConverged(),                   "converged (matrix-free/device)");
   Check(std::abs(double(xmean - Vfrac)) < 0.01,  "volume fraction met");
   Check(maxerr < 0.05,                           "uniform design");
}

// ── Mapping-coverage tests (via OptimProblem -> StackedOptimizationProblem) ──
// These exercise wrapper code paths a single-constraint test never hits:
// empty packing (m=0), multi-row packing offsets, mixed ±h + inequality slots,
// a non-constant Jacobian, and the dual SolveDense SVD fallback.

static real_t BlockMean(const Vector &x, int start, int size)
{
   real_t s = 0.0;
   for (int j = start; j < start + size; ++j) { s += x(j); }
   return s / size;
}

// Solve a finalized, bounded OptimProblem and return the design + stats.
static void RunSolve(OptimProblem &prob, Vector &x, int &iters, real_t &kkt,
                     bool &conv, real_t x0val = real_t(0.5))
{
   const int n = prob.Width();
   StackedOptimizationProblem sopt(prob);
   MMAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);
   Vector x0(n); x0 = x0val;
   x.SetSize(n);
   solver.Mult(x0, x);
   iters = solver.GetNumIterations();
   kkt   = solver.GetFinalNorm();
   conv  = solver.GetConverged();
}

static OptimProblem *NewCompliance(int n)   // objective-only, caller adds cons
{
   OptimProblem *p = new OptimProblem(n);
   p->SetObjective(new ComplianceObj(n));
   return p;
}
static void Bound01(OptimProblem &p, int n)
{
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0);
   p.SetDofBounds(lb, ub);
}

// 1. Unconstrained (m=0): min Σ1/x_j -> every x_j at the upper bound (=1).
static void Test_Unconstrained()
{
   printf("\n--- Unconstrained (m=0): min Σ1/x_j -> x=xmax ---\n");
   const int n = 50;
   OptimProblem *prob = NewCompliance(n);
   prob->Finalize();
   Bound01(*prob, n);
   Vector x; int it; real_t kkt; bool conv;
   RunSolve(*prob, x, it, kkt, conv);
   real_t maxerr = 0.0;
   for (int j = 0; j < n; ++j)
   { maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - 1.0)))); }
   printf("  converged=%d iters=%d kkt=%.3e maxerr(from 1)=%.2e\n",
          conv, it, kkt, double(maxerr));
   Check(conv,          "converged (m=0)");
   Check(maxerr < 1e-3, "all x at upper bound (=1)");
   delete prob;
}

// 2. Multiple inequalities: three block-volume caps -> each block at its cap.
static void Test_MultipleInequalities()
{
   printf("\n--- Multiple inequalities (3 block-volume LE) ---\n");
   const int n = 300, b = 100;
   const real_t V[3] = {real_t(0.3), real_t(0.5), real_t(0.4)};
   OptimProblem *prob = NewCompliance(n);
   for (int k = 0; k < 3; ++k)
   { prob->AddConstraint(new BlockMeanConstraint(n, k*b, b, V[k]),
                         OptimProblem::ConstType::LE); }
   prob->Finalize();
   Bound01(*prob, n);
   Vector x; int it; real_t kkt; bool conv;
   RunSolve(*prob, x, it, kkt, conv);
   real_t maxerr = 0.0;
   printf("  converged=%d iters=%d kkt=%.3e means=", conv, it, kkt);
   for (int k = 0; k < 3; ++k)
   {
      real_t m = BlockMean(x, k*b, b);
      printf("%.4f ", double(m));
      maxerr = std::max(maxerr, real_t(std::abs(double(m - V[k]))));
   }
   printf("maxerr=%.2e\n", double(maxerr));
   Check(conv,          "converged (m=3)");
   Check(maxerr < 0.01, "all 3 block volumes at target");
   delete prob;
}

// 3. Mixed EQ + LE: block1 mean fixed (EQ), block2 mean capped (LE).
static void Test_MixedEqLe()
{
   printf("\n--- Mixed EQ + LE (block1 mean=0.3 EQ, block2 mean<=0.5 LE) ---\n");
   const int n = 200, b = 100;
   OptimProblem *prob = NewCompliance(n);
   prob->AddConstraint(new BlockMeanConstraint(n, 0, b, 0.3),
                       OptimProblem::ConstType::EQ);
   prob->AddConstraint(new BlockMeanConstraint(n, b, b, 0.5),
                       OptimProblem::ConstType::LE);
   prob->Finalize();
   Bound01(*prob, n);
   Vector x; int it; real_t kkt; bool conv;
   RunSolve(*prob, x, it, kkt, conv);
   real_t m1 = BlockMean(x, 0, b), m2 = BlockMean(x, b, b);
   printf("  converged=%d iters=%d kkt=%.3e mean1=%.4f(0.30) mean2=%.4f(0.50)\n",
          conv, it, kkt, double(m1), double(m2));
   Check(conv,                             "converged (mixed EQ+LE)");
   Check(std::abs(double(m1 - 0.3)) < 0.01, "EQ block mean = 0.3");
   Check(std::abs(double(m2 - 0.5)) < 0.01, "LE block mean = 0.5");
   delete prob;
}

// 4. Multiple equalities: two block means fixed (EQ), EQ).
static void Test_MultipleEqualities()
{
   printf("\n--- Multiple equalities (block1 mean=0.3, block2 mean=0.6) ---\n");
   const int n = 200, b = 100;
   OptimProblem *prob = NewCompliance(n);
   prob->AddConstraint(new BlockMeanConstraint(n, 0, b, 0.3),
                       OptimProblem::ConstType::EQ);
   prob->AddConstraint(new BlockMeanConstraint(n, b, b, 0.6),
                       OptimProblem::ConstType::EQ);
   prob->Finalize();
   Bound01(*prob, n);
   Vector x; int it; real_t kkt; bool conv;
   RunSolve(*prob, x, it, kkt, conv);
   real_t m1 = BlockMean(x, 0, b), m2 = BlockMean(x, b, b);
   printf("  converged=%d iters=%d kkt=%.3e mean1=%.4f(0.30) mean2=%.4f(0.60)\n",
          conv, it, kkt, double(m1), double(m2));
   Check(conv,                             "converged (2 equalities)");
   Check(std::abs(double(m1 - 0.3)) < 0.01, "EQ block1 mean = 0.3");
   Check(std::abs(double(m2 - 0.6)) < 0.01, "EQ block2 mean = 0.6");
   delete prob;
}

// 5. Nonlinear constraint: (1/n)Σx² <= 0.25 -> uniform x = 0.5 (RMS active).
static void Test_NonlinearConstraint()
{
   printf("\n--- Nonlinear constraint ((1/n)Σx² <= 0.25 -> x=0.5) ---\n");
   const int n = 100;
   OptimProblem *prob = NewCompliance(n);
   prob->AddConstraint(new SumSqConstraint(n, 0.25), OptimProblem::ConstType::LE);
   prob->Finalize();
   Bound01(*prob, n);
   Vector x; int it; real_t kkt; bool conv;
   RunSolve(*prob, x, it, kkt, conv);
   real_t sumsq = 0.0, maxerr = 0.0;
   for (int j = 0; j < n; ++j)
   { sumsq += x(j) * x(j); maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - 0.5)))); }
   const real_t rms = real_t(std::sqrt(double(sumsq / n)));
   printf("  converged=%d iters=%d kkt=%.3e rms=%.4f(0.50) maxerr=%.2e\n",
          conv, it, kkt, double(rms), double(maxerr));
   Check(conv,                             "converged (nonlinear constraint)");
   Check(std::abs(double(rms - 0.5)) < 0.01, "RMS constraint active (=0.5)");
   Check(maxerr < 0.02,                    "uniform x = 0.5");
   delete prob;
}

// 6. Redundant / overconstrained: block1<=0.4, block2<=0.4, overall<=0.4 with
// overall = 0.5·(block1) + 0.5·(block2) — a linearly dependent active set that
// makes the dual Hessian singular and exercises the SolveDense SVD fallback.
static void Test_Redundant()
{
   printf("\n--- Redundant/overconstrained (3 dependent LE -> x=0.4) ---\n");
   const int n = 200, b = 100;
   OptimProblem *prob = NewCompliance(n);
   prob->AddConstraint(new BlockMeanConstraint(n, 0, b, 0.4),
                       OptimProblem::ConstType::LE);
   prob->AddConstraint(new BlockMeanConstraint(n, b, b, 0.4),
                       OptimProblem::ConstType::LE);
   prob->AddConstraint(new BlockMeanConstraint(n, 0, n, 0.4),   // overall mean
                       OptimProblem::ConstType::LE);
   prob->Finalize();
   Bound01(*prob, n);
   Vector x; int it; real_t kkt; bool conv;
   RunSolve(*prob, x, it, kkt, conv);
   real_t xmean = 0.0, maxerr = 0.0;
   for (int j = 0; j < n; ++j)
   { xmean += x(j); maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - 0.4)))); }
   xmean /= n;
   printf("  converged=%d iters=%d kkt=%.3e xmean=%.6f(0.40) maxerr=%.2e\n",
          conv, it, kkt, double(xmean), double(maxerr));
   Check(conv,                              "converged (redundant constraints)");
   Check(std::abs(double(xmean - 0.4)) < 0.01, "overall mean = 0.4");
   Check(maxerr < 0.02,                     "uniform x = 0.4");
   delete prob;
}

// GCMMA (globally-convergent MMA). Reaches the same optimum as plain MMA on the
// convex compliance proxy; exercises UpdateGCMMA (and, in conservative mode, the
// wrapper's true-model callback).
static void Test_GCMMA(bool conservative, const char *tag)
{
   printf("\n--- GCMMA (%s) on compliance LE ---\n", tag);
   const int n = 100;
   const real_t Vfrac = real_t(0.4);
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.AddConstraint(new MeanConstraint(n, Vfrac), OptimProblem::ConstType::LE);
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0);
   prob.SetDofBounds(lb, ub);

   StackedOptimizationProblem sopt(prob);
   MMAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetGCMMA(true, conservative);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);

   Vector x0(n), x(n); x0 = real_t(0.5);
   solver.Mult(x0, x);

   real_t xmean = 0.0, maxerr = 0.0;
   for (int j = 0; j < n; ++j)
   { xmean += x(j); maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - Vfrac)))); }
   xmean /= n;
   printf("  converged=%d iters=%d kkt=%.3e xmean=%.6f maxerr=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(),
          solver.GetFinalNorm(), double(xmean), double(maxerr));
   Check(solver.GetConverged(),                   "GCMMA converged");
   Check(std::abs(double(xmean - Vfrac)) < 0.01,  "volume fraction met");
   Check(maxerr < 0.05,                           "uniform design");
}

// Non-convex double-well objective  f = Σ (x_j²-1)².
class DoubleWellObj : public Operator
{
public:
   DoubleWellObj(int n) : Operator(1, n), g(n), row(g) {}
   void Mult(const Vector &x, Vector &y) const override
   { y.SetSize(1); real_t s = 0; for (int j = 0; j < width; ++j) { real_t t = x(j)*x(j)-1; s += t*t; } y(0) = s; }
   Operator &GetGradient(const Vector &x) const override
   { for (int j = 0; j < width; ++j) { g(j) = real_t(4)*x(j)*(x(j)*x(j)-1); } return row; }
private:
   class Row : public Operator {
   public:
      Row(const Vector &g_) : Operator(1, g_.Size()), g(g_) {}
      void Mult(const Vector &dx, Vector &y) const override { y.SetSize(1); y(0) = InnerProduct(g, dx); }
      void MultTranspose(const Vector &dy, Vector &dx) const override { dx.SetSize(g.Size()); dx = g; dx *= dy(0); }
   private: const Vector &g;
   };
   mutable Vector g; mutable Row row;
};

// Non-convex: min Σ(x²-1)² s.t. mean(x)<=0.5, x in [0.001,2]. No closed form —
// robustness check: MMA (and GCMMA) must converge to a FEASIBLE point.
static void Test_Nonconvex(bool gcmma)
{
   printf("\n--- Non-convex double-well (mean<=0.5, %s) ---\n",
          gcmma ? "GCMMA" : "MMA");
   const int n = 100;
   OptimProblem prob(n);
   prob.SetObjective(new DoubleWellObj(n));
   prob.AddConstraint(new MeanConstraint(n, 0.5), OptimProblem::ConstType::LE);
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(2.0); prob.SetDofBounds(lb, ub);
   StackedOptimizationProblem sopt(prob);
   MMAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   if (gcmma) { solver.SetGCMMA(true, true); }
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(500);
   Vector x0(n), x(n);
   for (int j = 0; j < n; ++j) { x0(j) = real_t(0.3 + 0.4 * ((j*7) % 11) / 11.0); }
   solver.Mult(x0, x);
   real_t mean = 0.0; for (int j = 0; j < n; ++j) { mean += x(j); } mean /= n;
   const real_t obj = prob.Objective(x);
   printf("  converged=%d iters=%d obj=%.4f mean=%.4f(<=0.50)\n",
          solver.GetConverged(), solver.GetNumIterations(), double(obj), double(mean));
   Check(solver.GetConverged(),       "converged (non-convex)");
   Check(double(mean) < 0.5 + 1e-3,   "feasible (mean <= 0.5)");
   Check(std::isfinite(double(obj)),  "objective finite");
}

int main()
{
   printf("=== MMAOptimizationSolver (MMA -> OptimizationSolver) tests ===\n");

   // Baselines: inequality and equality paths.
   const int it_base = SolveAndCheck(false, 100, 0.4, nullptr, false, "xt guess");
   SolveAndCheck(false,  50, 0.6, nullptr, false, "xt guess");
   SolveAndCheck(true,  100, 0.4, nullptr, false, "xt guess");

   // Riesz map: identity must be a no-op (identical iteration count).
   IdentityOperator I(100);
   const int it_id = SolveAndCheck(false, 100, 0.4, &I, false, "identity Riesz");
   Check(it_id == it_base, "identity Riesz matches no-Riesz iteration count");

   // Riesz map: nontrivial SPD diagonal metric -> same optimum.
   Vector diag(100);
   for (int j = 0; j < 100; ++j) { diag(j) = real_t(0.8 + 0.4 * ((j * 7) % 5) / 4.0); }
   DiagOperator R(diag);
   SolveAndCheck(false, 100, 0.4, &R, false, "diagonal Riesz");

   // Initial guess via SetInitialGuess (decoy xt ignored).
   SolveAndCheck(false, 100, 0.4, nullptr, true, "SetInitialGuess");

   // General OptimizationProblem paths not emitted by StackedOptimizationProblem.
   Test_Equality_NonzeroRHS();
   Test_TwoSided(real_t(5.0),  real_t(3.0), "upper side active -> x=d_hi");
   Test_TwoSided(real_t(-5.0), real_t(1.0), "lower side active -> x=d_lo");

   // Matrix-free device-aware Jacobian path.
   Test_MatrixFreeDevice();

   // Mapping-coverage: constraint-count / type combinations.
   Test_Unconstrained();
   Test_MultipleInequalities();
   Test_MixedEqLe();
   Test_MultipleEqualities();
   Test_NonlinearConstraint();
   Test_Redundant();

   // GCMMA (globally-convergent) path.
   Test_GCMMA(true,  "conservative");
   Test_GCMMA(false, "single-subproblem");

   // Non-convex robustness (registered; also compared across methods in compare_opt).
   Test_Nonconvex(false);
   Test_Nonconvex(true);

   printf("\n========================================\n");
   if (g_nfail == 0) { printf("All MMAOptimizationSolver tests PASSED.\n"); }
   else              { printf("%d MMAOptimizationSolver test(s) FAILED.\n", g_nfail); }
   printf("========================================\n");
   return g_nfail > 0 ? 1 : 0;
}
