/**
 * test_opt_sq.cpp  —  End-to-end test of the SQ (separable-quadratic) ->
 * OptimizationSolver bridge (SQOptimizationSolver), mirroring the MMA/CCSA
 * compliance-proxy tests.
 *
 * Problems (analytic optimum = uniform x = Vfrac unless noted):
 *   1. Inequality LE       2. Equality EQ (±h)     3. Multiple inequalities
 *   4. Unconstrained (m=0) 5. GCMMA                6. SetSigmaScale (SQ knob)
 * with 0.001 <= x <= 1.
 */

#include "MMAOptSolver.hpp"   // SQOptimizationSolver
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

class ComplianceObj : public Operator            // f = Σ 1/x_j
{
public:
   ComplianceObj(int n) : Operator(1, n), g(n), row(g) {}
   void Mult(const Vector &x, Vector &y) const override
   { y.SetSize(1); real_t s = 0; for (int j = 0; j < width; ++j) { s += real_t(1) / x(j); } y(0) = s; }
   Operator &GetGradient(const Vector &x) const override
   { for (int j = 0; j < width; ++j) { g(j) = real_t(-1) / (x(j) * x(j)); } return row; }
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

class BlockMean : public Operator                // (1/size) Σ_block x_j - target
{
public:
   BlockMean(int n, int start, int size, real_t target)
      : Operator(1, n), tgt(target), A(1, n)
   { A = 0.0; for (int j = start; j < start+size; ++j) { A(0, j) = real_t(1)/size; } }
   void Mult(const Vector &x, Vector &y) const override { A.Mult(x, y); y(0) -= tgt; }
   Operator &GetGradient(const Vector &) const override { return const_cast<DenseMatrix&>(A); }
private: real_t tgt; DenseMatrix A;
};

static real_t BlockMeanOf(const Vector &x, int start, int size)
{ real_t s = 0; for (int j = start; j < start+size; ++j) { s += x(j); } return s/size; }

static void Bound01(OptimProblem &p, int n)
{ Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0); p.SetDofBounds(lb, ub); }

// Compliance proxy with a single mean constraint (LE or EQ), optional GCMMA.
static void Test_SQ_Single(bool equality, int n, real_t Vfrac, bool gcmma)
{
   printf("\n--- SQ compliance (n=%d, Vfrac=%.2f, %s%s) ---\n", n, Vfrac,
          equality ? "EQ" : "LE", gcmma ? ", GCMMA" : "");
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.AddConstraint(new BlockMean(n, 0, n, Vfrac),
                      equality ? OptimProblem::ConstType::EQ
                               : OptimProblem::ConstType::LE);
   prob.Finalize();
   Bound01(prob, n);

   StackedOptimizationProblem sopt(prob);
   SQOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   if (gcmma) { solver.SetGCMMA(true, true); }
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);

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

// Three block-volume inequalities -> each block at its cap (tests m>1 packing).
static void Test_SQ_MultiInequality()
{
   printf("\n--- SQ multiple inequalities (3 block-volume LE) ---\n");
   const int n = 300, b = 100;
   const real_t V[3] = {real_t(0.3), real_t(0.5), real_t(0.4)};
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   for (int k = 0; k < 3; ++k)
   { prob.AddConstraint(new BlockMean(n, k*b, b, V[k]), OptimProblem::ConstType::LE); }
   prob.Finalize();
   Bound01(prob, n);

   StackedOptimizationProblem sopt(prob);
   SQOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);

   Vector x0(n), x(n); x0 = real_t(0.5);
   solver.Mult(x0, x);

   real_t maxerr = 0.0;
   printf("  converged=%d iters=%d kkt=%.3e means=",
          solver.GetConverged(), solver.GetNumIterations(), solver.GetFinalNorm());
   for (int k = 0; k < 3; ++k)
   {
      real_t m = BlockMeanOf(x, k*b, b);
      printf("%.4f ", double(m));
      maxerr = std::max(maxerr, real_t(std::abs(double(m - V[k]))));
   }
   printf("maxerr=%.2e\n", double(maxerr));
   Check(solver.GetConverged(),  "converged (m=3)");
   Check(maxerr < 0.01,          "all 3 block volumes at target");
}

// Unconstrained: min Σ1/x_j -> every x_j at upper bound (=1).
static void Test_SQ_Unconstrained()
{
   printf("\n--- SQ unconstrained (m=0) -> x=xmax ---\n");
   const int n = 50;
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.Finalize();
   Bound01(prob, n);

   StackedOptimizationProblem sopt(prob);
   SQOptimizationSolver solver;
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);

   Vector x0(n), x(n); x0 = real_t(0.5);
   solver.Mult(x0, x);
   real_t maxerr = 0.0;
   for (int j = 0; j < n; ++j) { maxerr = std::max(maxerr, real_t(std::abs(double(x(j) - 1.0)))); }
   printf("  converged=%d iters=%d maxerr(from 1)=%.2e\n",
          solver.GetConverged(), solver.GetNumIterations(), double(maxerr));
   Check(solver.GetConverged(), "converged (m=0)");
   Check(maxerr < 1e-3,         "all x at upper bound (=1)");
}

// SQ-specific: a different sigma scale still converges to the same optimum.
static void Test_SQ_SigmaScale()
{
   printf("\n--- SQ SetSigmaScale(0.3) on compliance LE ---\n");
   const int n = 100; const real_t Vfrac = real_t(0.4);
   OptimProblem prob(n);
   prob.SetObjective(new ComplianceObj(n));
   prob.AddConstraint(new BlockMean(n, 0, n, Vfrac), OptimProblem::ConstType::LE);
   prob.Finalize();
   Bound01(prob, n);

   StackedOptimizationProblem sopt(prob);
   SQOptimizationSolver solver;
   solver.SetSigmaScale(real_t(0.3));
   solver.SetOptimizationProblem(sopt);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-5); solver.SetMaxIter(300);

   Vector x0(n), x(n); x0 = real_t(0.5);
   solver.Mult(x0, x);
   real_t xmean = 0.0; for (int j = 0; j < n; ++j) { xmean += x(j); } xmean /= n;
   printf("  converged=%d iters=%d xmean=%.6f(%.2f)\n",
          solver.GetConverged(), solver.GetNumIterations(), double(xmean), double(Vfrac));
   Check(solver.GetConverged(),                   "converged with sigma=0.3");
   Check(std::abs(double(xmean - Vfrac)) < 0.01,  "volume fraction met");
}

int main()
{
   printf("=== SQOptimizationSolver (SQ -> OptimizationSolver) tests ===\n");

   Test_SQ_Single(/*eq=*/false, 100, real_t(0.4), /*gcmma=*/false);
   Test_SQ_Single(/*eq=*/false,  50, real_t(0.6), /*gcmma=*/false);
   Test_SQ_Single(/*eq=*/true,  100, real_t(0.4), /*gcmma=*/false);
   Test_SQ_MultiInequality();
   Test_SQ_Unconstrained();
   Test_SQ_Single(/*eq=*/false, 100, real_t(0.4), /*gcmma=*/true);
   Test_SQ_SigmaScale();

   printf("\n========================================\n");
   if (g_nfail == 0) { printf("All SQOptimizationSolver tests PASSED.\n"); }
   else              { printf("%d SQOptimizationSolver test(s) FAILED.\n", g_nfail); }
   printf("========================================\n");
   return g_nfail > 0 ? 1 : 0;
}
