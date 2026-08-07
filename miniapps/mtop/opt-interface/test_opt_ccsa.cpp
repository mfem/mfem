/**
 * test_opt_ccsa.cpp  —  End-to-end test of the CCSA (Bregman) -> OptimizationSolver
 * bridge (CCSAOptimizationSolver), mirroring the MMA compliance-proxy tests.
 *
 * Problems (analytic optimum = uniform x = Vfrac):
 *   1. Inequality:  min Σ1/x_j  s.t.  mean(x) <= Vfrac
 *   2. Equality:    min Σ1/x_j  s.t.  mean(x)  = Vfrac
 *   3. Multiple inequalities: three block-volume caps
 *   4. GCMMA variant of (1)
 * with 0.001 <= x <= 1.  Verifies the latent-variable outer loop, the ±h
 * equality path, multi-constraint packing, and GCMMA — all through the same
 * OptimizationSolver interface as the MMA wrapper.
 */

#include "CCSAOptSolver.hpp"
#include "opt_prob.hpp"
#include <cmath>
#include <cstdio>

using namespace mfem;
using namespace mfem_mma;

static int g_nfail = 0;
static void Check(bool cond, const char *msg)
{
   if (cond) { printf("  [PASS] %s\n", msg); }
   else      { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// Objective  f(x) = Σ_j 1/x_j.  Jacobian row  g_j = -1/x_j².
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

// Mean over a contiguous block  m(x) = (1/size) Σ_{block} x_j - target.
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
   { A.Mult(x, y); y(0) -= tgt; }
   Operator &GetGradient(const Vector &) const override
   { return const_cast<DenseMatrix &>(A); }
private:
   real_t tgt;
   DenseMatrix A;
};

static real_t BlockMean(const Vector &x, int start, int size)
{
   real_t s = 0.0;
   for (int j = start; j < start + size; ++j) { s += x(j); }
   return s / size;
}

// Compliance proxy with a single mean constraint (LE or EQ) -> uniform x=Vfrac.
static void Test_CCSA_Single(bool equality, int n, real_t Vfrac, bool gcmma)
{
   printf("\n--- CCSA compliance (n=%d, Vfrac=%.2f, %s%s) ---\n", n, Vfrac,
          equality ? "EQ" : "LE", gcmma ? ", GCMMA" : "");
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.AddConstraint(new BlockMeanConstraint(n, 0, n, Vfrac),
                      equality ? OptimProblem::ConstType::EQ
                               : OptimProblem::ConstType::LE);
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0);
   prob.SetDofBounds(lb, ub);

   StackedOptimizationProblem sopt(prob);
   CCSAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   if (gcmma) { solver.SetGCMMA(true, true); }
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-4); solver.SetMaxIter(500);

   Vector x0(n), x(n); x0 = real_t(0.5);
   solver.Mult(x0, x);

   real_t xmean = 0.0, maxerr = 0.0;
   for (int j = 0; j < n; ++j)
   { xmean += x(j); maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - Vfrac)))); }
   xmean /= n;
   printf("  converged=%d iters=%d kkt=%.3e xmean=%.6f(%.2f) maxerr=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(),
          solver.GetFinalNorm(), double(xmean), double(Vfrac), double(maxerr));
   Check(solver.GetConverged(),                   "converged");
   Check(std::abs(double(xmean - Vfrac)) < 0.01,  "volume fraction met");
   Check(maxerr < 0.05,                           "uniform design");
}

// Three block-volume inequalities -> each block at its cap.
static void Test_CCSA_MultiInequality()
{
   printf("\n--- CCSA multiple inequalities (3 block-volume LE) ---\n");
   const int n = 300, b = 100;
   const real_t V[3] = {real_t(0.3), real_t(0.5), real_t(0.4)};
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   for (int k = 0; k < 3; ++k)
   { prob.AddConstraint(new BlockMeanConstraint(n, k*b, b, V[k]),
                        OptimProblem::ConstType::LE); }
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0);
   prob.SetDofBounds(lb, ub);

   StackedOptimizationProblem sopt(prob);
   CCSAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-4); solver.SetMaxIter(500);

   Vector x0(n), x(n); x0 = real_t(0.5);
   solver.Mult(x0, x);

   real_t maxerr = 0.0;
   printf("  converged=%d iters=%d kkt=%.3e means=",
          solver.GetConverged(), solver.GetNumIterations(), solver.GetFinalNorm());
   for (int k = 0; k < 3; ++k)
   {
      real_t m = BlockMean(x, k*b, b);
      printf("%.4f ", double(m));
      maxerr = std::max(maxerr, real_t(std::abs(double(m - V[k]))));
   }
   printf("maxerr=%.2e\n", double(maxerr));
   Check(solver.GetConverged(),  "converged (m=3)");
   Check(maxerr < 0.01,          "all 3 block volumes at target");
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

// Interior-optimum quadratic  f = 0.5 Σ (x_j - t)².
class QuadTargetObj : public Operator
{
public:
   QuadTargetObj(int n, real_t t) : Operator(1, n), t_(t), g(n), row(g) {}
   void Mult(const Vector &x, Vector &y) const override
   { y.SetSize(1); real_t s = 0; for (int j = 0; j < width; ++j) { real_t d = x(j)-t_; s += 0.5*d*d; } y(0) = s; }
   Operator &GetGradient(const Vector &x) const override
   { for (int j = 0; j < width; ++j) { g(j) = x(j) - t_; } return row; }
private:
   class Row : public Operator {
   public:
      Row(const Vector &g_) : Operator(1, g_.Size()), g(g_) {}
      void Mult(const Vector &dx, Vector &y) const override { y.SetSize(1); y(0) = InnerProduct(g, dx); }
      void MultTranspose(const Vector &dy, Vector &dx) const override { dx.SetSize(g.Size()); dx = g; dx *= dy(0); }
   private: const Vector &g;
   };
   real_t t_; mutable Vector g; mutable Row row;
};

// Unconstrained (m=0): min 0.5 Σ (x_j-0.5)² -> INTERIOR optimum x=0.5. (A pure
// box-boundary optimum sits at latent infinity for CCSA's mirror map, so an
// interior target is the meaningful unconstrained check for CCSA.)
static void Test_CCSA_Unconstrained()
{
   printf("\n--- CCSA unconstrained (m=0) -> interior x=0.5 ---\n");
   const int n = 50;
   OptimProblem prob(n);
   prob.SetObjective(new QuadTargetObj(n, real_t(0.5)));
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0); prob.SetDofBounds(lb, ub);
   StackedOptimizationProblem sopt(prob);
   CCSAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-8); solver.SetMaxIter(1000);
   Vector x0(n), x(n); x0 = real_t(0.2);
   solver.Mult(x0, x);
   real_t maxerr = 0;
   for (int j = 0; j < n; ++j) { maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - 0.5)))); }
   printf("  converged=%d iters=%d maxerr(from 0.5)=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(), double(maxerr));
   Check(solver.GetConverged(), "converged (m=0)");
   Check(maxerr < 1e-3,         "x at interior optimum (=0.5)");
}

// Redundant / overconstrained: 3 linearly-dependent LE -> uniform x=0.4.
static void Test_CCSA_Overconstrained()
{
   printf("\n--- CCSA redundant/overconstrained (3 dependent LE -> x=0.4) ---\n");
   const int n = 200, b = 100;
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.AddConstraint(new BlockMeanConstraint(n, 0, b, 0.4), OptimProblem::ConstType::LE);
   prob.AddConstraint(new BlockMeanConstraint(n, b, b, 0.4), OptimProblem::ConstType::LE);
   prob.AddConstraint(new BlockMeanConstraint(n, 0, n, 0.4), OptimProblem::ConstType::LE);
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0); prob.SetDofBounds(lb, ub);
   StackedOptimizationProblem sopt(prob);
   CCSAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-4); solver.SetMaxIter(500);
   Vector x0(n), x(n); x0 = real_t(0.5);
   solver.Mult(x0, x);
   real_t xmean = 0, maxerr = 0;
   for (int j = 0; j < n; ++j) { xmean += x(j); maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - 0.4)))); }
   xmean /= n;
   printf("  converged=%d iters=%d xmean=%.6f(0.40) maxerr=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(), double(xmean), double(maxerr));
   Check(solver.GetConverged(),                "converged (redundant)");
   Check(std::abs(double(xmean - 0.4)) < 0.01, "overall mean = 0.4");
   Check(maxerr < 0.02,                        "uniform x = 0.4");
}

// Non-convex: min Σ(x²-1)² s.t. mean(x)<=0.5, x in [0.001,2]. Robustness check
// (no known closed form): must converge to a FEASIBLE point.
static void Test_CCSA_Nonconvex()
{
   printf("\n--- CCSA non-convex double-well (mean<=0.5) ---\n");
   const int n = 100;
   OptimProblem prob(n);
   prob.SetObjective(new DoubleWellObj(n));
   prob.AddConstraint(new BlockMeanConstraint(n, 0, n, 0.5), OptimProblem::ConstType::LE);
   prob.Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(2.0); prob.SetDofBounds(lb, ub);
   StackedOptimizationProblem sopt(prob);
   CCSAOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-4); solver.SetMaxIter(500);
   Vector x0(n), x(n);
   for (int j = 0; j < n; ++j) { x0(j) = real_t(0.3 + 0.4 * ((j*7) % 11) / 11.0); }
   solver.Mult(x0, x);
   real_t mean = BlockMean(x, 0, n), obj = prob.Objective(x);
   printf("  converged=%d iters=%d obj=%.4f mean=%.4f(<=0.50)\n",
          solver.GetConverged(), solver.GetNumIterations(), double(obj), double(mean));
   Check(solver.GetConverged(),          "converged (non-convex)");
   Check(double(mean) < 0.5 + 1e-3,      "feasible (mean <= 0.5)");
   Check(std::isfinite(double(obj)),     "objective finite");
}

int main()
{
   printf("=== CCSAOptimizationSolver (Bregman -> OptimizationSolver) tests ===\n");

   Test_CCSA_Single(/*equality=*/false, 100, real_t(0.4), /*gcmma=*/false);
   Test_CCSA_Single(/*equality=*/false,  50, real_t(0.6), /*gcmma=*/false);
   Test_CCSA_Single(/*equality=*/true,  100, real_t(0.4), /*gcmma=*/false);
   Test_CCSA_MultiInequality();
   Test_CCSA_Single(/*equality=*/false, 100, real_t(0.4), /*gcmma=*/true);
   Test_CCSA_Unconstrained();
   Test_CCSA_Overconstrained();
   Test_CCSA_Nonconvex();

   printf("\n========================================\n");
   if (g_nfail == 0) { printf("All CCSAOptimizationSolver tests PASSED.\n"); }
   else              { printf("%d CCSAOptimizationSolver test(s) FAILED.\n", g_nfail); }
   printf("========================================\n");
   return g_nfail > 0 ? 1 : 0;
}
