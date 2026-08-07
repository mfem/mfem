/**
 * compare_opt.cpp  —  Compare MMA, GCMMA and CCSA(Bregman) across problems.
 *
 * Runs the three solvers (all via the mfem::OptimizationSolver interface) on a
 * suite of convex problems (with a known optimum) plus one non-convex problem,
 * and reports:
 *   it*    iterations to reach design accuracy ||x-x*||_inf < 1e-3  — the FAIR,
 *          method-agnostic yardstick (the native KKT residuals are on different
 *          scales: MMA is a physical projected-gradient norm, CCSA a latent-
 *          space residual, so they are not directly comparable);
 *   obj    final objective F(x)  (lower is better; identical for convex);
 *   viol   max constraint violation (feasibility);
 *   conv   whether the method hit its native KKT tolerance, and in how many its.
 *
 * All methods use the same start x0, abs_tol and max_iter.
 */

#include "MMAOptSolver.hpp"
#include "CCSAOptSolver.hpp"
#include "opt_prob.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>

using namespace mfem;
using namespace mfem_mma;

// ── Test operators ─────────────────────────────────────────────────────────
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

class DoubleWellObj : public Operator            // f = Σ (x_j²-1)²  (non-convex)
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

class SumSq : public Operator                    // (1/n) Σ x_j² - s²  (nonlinear)
{
public:
   SumSq(int n, real_t s2) : Operator(1, n), s2_(s2), g(n), row(g) {}
   void Mult(const Vector &x, Vector &y) const override
   { y.SetSize(1); real_t s = 0; for (int j = 0; j < width; ++j) { s += x(j)*x(j); } y(0) = s/width - s2_; }
   Operator &GetGradient(const Vector &x) const override
   { for (int j = 0; j < width; ++j) { g(j) = real_t(2)*x(j)/width; } return row; }
private:
   class Row : public Operator {
   public:
      Row(const Vector &g_) : Operator(1, g_.Size()), g(g_) {}
      void Mult(const Vector &dx, Vector &y) const override { y.SetSize(1); y(0) = InnerProduct(g, dx); }
      void MultTranspose(const Vector &dy, Vector &dx) const override { dx.SetSize(g.Size()); dx = g; dx *= dy(0); }
   private: const Vector &g;
   };
   real_t s2_; mutable Vector g; mutable Row row;
};

// ── Common yardstick: iterations to reach ||x-x*||_inf < thr ────────────────
class DesignErrorRecorder : public IterativeSolverController
{
public:
   DesignErrorRecorder(const Vector &xstar, real_t thr) : xstar_(xstar), thr_(thr) {}
   bool RequiresUpdatedSolution() const override { return true; }
   void Reset() override { IterativeSolverController::Reset(); iters_ = -1; }
   void MonitorSolution(int it, real_t, const Vector &x, bool) override
   {
      if (iters_ >= 0 || x.Size() != xstar_.Size()) { return; }
      real_t e = 0;
      for (int j = 0; j < x.Size(); ++j)
      { e = std::max(e, real_t(std::abs(double(x(j) - xstar_(j))))); }
      if (e < thr_) { iters_ = it; }
   }
   int ItersToTarget() const { return iters_; }
private:
   Vector xstar_; real_t thr_; int iters_ = -1;
};

enum Method { M_MMA = 0, M_GCMMA = 1, M_CCSA = 2 };
static const char *MethodName(Method m)
{ return m == M_MMA ? "MMA" : m == M_GCMMA ? "GCMMA" : "CCSA"; }

struct Result { int it_target = -1; double obj = 0; double viol = 0; bool conv = false; int native_it = 0; };

static real_t MaxViolation(OptimProblem &prob, const Vector &x)
{
   Vector full(prob.Height()); prob.Mult(x, full);
   Array<int> eqr, ler; int objr; prob.ClassifyRows(eqr, ler, objr);
   real_t v = 0;
   for (int i = 0; i < ler.Size(); ++i) { v = std::max(v, std::max(real_t(0), full(ler[i]))); }
   for (int i = 0; i < eqr.Size(); ++i) { v = std::max(v, real_t(std::abs(double(full(eqr[i]))))); }
   return v;
}

static Result RunOne(OptimProblem &prob, Method m, const Vector &x0,
                     const Vector *xstar)
{
   StackedOptimizationProblem sopt(prob);
   Vector x(prob.Width());
   Result r;
   DesignErrorRecorder rec(xstar ? *xstar : x0, real_t(1e-3));
   const real_t atol = 1e-7;
   const int maxit = 200;
   if (m == M_CCSA)
   {
      CCSAOptimizationSolver s; s.SetOptimizationProblem(sopt);
      s.SetRelTol(0.0); s.SetAbsTol(atol); s.SetMaxIter(maxit);
      if (xstar) { s.SetController(rec); }
      s.Mult(x0, x);
      r.conv = s.GetConverged(); r.native_it = s.GetNumIterations();
   }
   else
   {
      MMAOptimizationSolver s; s.SetOptimizationProblem(sopt);
      if (m == M_GCMMA) { s.SetGCMMA(true, true); }
      s.SetRelTol(0.0); s.SetAbsTol(atol); s.SetMaxIter(maxit);
      if (xstar) { s.SetController(rec); }
      s.Mult(x0, x);
      r.conv = s.GetConverged(); r.native_it = s.GetNumIterations();
   }
   if (xstar) { r.it_target = rec.ItersToTarget(); }
   r.obj  = prob.Objective(x);
   r.viol = MaxViolation(prob, x);
   return r;
}

// A problem: builder (finalized + bounded), start x0, and known optimum (or none).
struct Problem
{
   std::string name;
   std::function<OptimProblem*()> build;
   Vector x0;
   Vector xstar;      // empty -> unknown (non-convex): it* not reported
   bool has_star;
};

static OptimProblem *Compliance(int n, real_t V, OptimProblem::ConstType type)
{
   OptimProblem *p = new OptimProblem(n);
   p->SetObjective(new ComplianceObj(n));
   p->AddConstraint(new BlockMean(n, 0, n, V), type);
   p->Finalize();
   Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0); p->SetDofBounds(lb, ub);
   return p;
}

int main()
{
   printf("=== MMA vs GCMMA vs CCSA — cross-method comparison ===\n");
   printf("  it*  = iterations to ||x-x*||_inf < 1e-3 (fair, method-agnostic)\n");
   printf("  conv = hit native KKT tol (abs_tol=1e-7) within 200 its\n\n");

   std::vector<Problem> problems;

   auto uniform = [](int n, real_t v){ Vector x(n); x = v; return x; };

   // 1. Compliance LE (n=100, V=0.40)
   { Problem p; p.name = "Compliance-LE  n=100 V=0.40";
     p.build = []{ return Compliance(100, 0.40, OptimProblem::ConstType::LE); };
     p.x0 = uniform(100, 0.5); p.xstar = uniform(100, 0.40); p.has_star = true;
     problems.push_back(p); }
   // 2. Compliance LE (n=500, V=0.40) — feasible: c=5000 > λ*≈n/V²=3125
   { Problem p; p.name = "Compliance-LE  n=500 V=0.40";
     p.build = []{ return Compliance(500, 0.40, OptimProblem::ConstType::LE); };
     p.x0 = uniform(500, 0.5); p.xstar = uniform(500, 0.40); p.has_star = true;
     problems.push_back(p); }
   // 3. Compliance EQ (n=100, V=0.40)
   { Problem p; p.name = "Compliance-EQ  n=100 V=0.40";
     p.build = []{ return Compliance(100, 0.40, OptimProblem::ConstType::EQ); };
     p.x0 = uniform(100, 0.5); p.xstar = uniform(100, 0.40); p.has_star = true;
     problems.push_back(p); }
   // 4. Multiple inequalities: 3 block volumes (0.3/0.5/0.4)
   { Problem p; p.name = "MultiIneq-3blk n=300";
     p.build = []{
        const int n = 300, b = 100; const real_t V[3] = {0.3, 0.5, 0.4};
        OptimProblem *q = new OptimProblem(n);
        q->SetObjective(new ComplianceObj(n));
        for (int k = 0; k < 3; ++k) { q->AddConstraint(new BlockMean(n, k*b, b, V[k]), OptimProblem::ConstType::LE); }
        q->Finalize();
        Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0); q->SetDofBounds(lb, ub);
        return q; };
     p.x0 = uniform(300, 0.5);
     { Vector xs(300); for (int j = 0; j < 300; ++j) { xs(j) = j < 100 ? real_t(0.3) : j < 200 ? real_t(0.5) : real_t(0.4); } p.xstar = xs; }
     p.has_star = true; problems.push_back(p); }
   // 5. Nonlinear constraint: (1/n)Σx² <= 0.25 -> x=0.5
   { Problem p; p.name = "Nonlinear-RMS  n=100";
     p.build = []{
        const int n = 100; OptimProblem *q = new OptimProblem(n);
        q->SetObjective(new ComplianceObj(n));
        q->AddConstraint(new SumSq(n, 0.25), OptimProblem::ConstType::LE);
        q->Finalize();
        Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0); q->SetDofBounds(lb, ub);
        return q; };
     p.x0 = uniform(100, 0.8); p.xstar = uniform(100, 0.5); p.has_star = true;
     problems.push_back(p); }
   // 6. Redundant / overconstrained: 3 dependent LE -> x=0.4
   { Problem p; p.name = "Redundant-3LE  n=200";
     p.build = []{
        const int n = 200, b = 100; OptimProblem *q = new OptimProblem(n);
        q->SetObjective(new ComplianceObj(n));
        q->AddConstraint(new BlockMean(n, 0, b, 0.4), OptimProblem::ConstType::LE);
        q->AddConstraint(new BlockMean(n, b, b, 0.4), OptimProblem::ConstType::LE);
        q->AddConstraint(new BlockMean(n, 0, n, 0.4), OptimProblem::ConstType::LE);
        q->Finalize();
        Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(1.0); q->SetDofBounds(lb, ub);
        return q; };
     p.x0 = uniform(200, 0.5); p.xstar = uniform(200, 0.4); p.has_star = true;
     problems.push_back(p); }
   // 7. TIGHT volume: c=max(1000,10n)=5000 < λ*≈n/V²=12500 — the elastic slack
   //    can't fully enforce the constraint, so ALL methods land slightly
   //    infeasible (shared wrapper penalty limit, not a method difference).
   { Problem p; p.name = "Compliance-LE  n=500 V=0.20 (tight,c<λ*)";
     p.build = []{ return Compliance(500, 0.20, OptimProblem::ConstType::LE); };
     p.x0 = uniform(500, 0.5); p.xstar = uniform(500, 0.20); p.has_star = true;
     problems.push_back(p); }
   // 8. NON-CONVEX double-well: min Σ(x²-1)² s.t. mean<=0.5, x in [0.001,2]
   { Problem p; p.name = "DoubleWell(ncvx) n=100 mean<=0.5";
     p.build = []{
        const int n = 100; OptimProblem *q = new OptimProblem(n);
        q->SetObjective(new DoubleWellObj(n));
        q->AddConstraint(new BlockMean(n, 0, n, 0.5), OptimProblem::ConstType::LE);
        q->Finalize();
        Vector lb(n), ub(n); lb = real_t(0.001); ub = real_t(2.0); q->SetDofBounds(lb, ub);
        return q; };
     { Vector x0(100); for (int j = 0; j < 100; ++j) { x0(j) = real_t(0.3 + 0.4 * ((j*7) % 11) / 11.0); } p.x0 = x0; }
     p.has_star = false; problems.push_back(p); }

   printf("%-32s %-6s  %4s  %12s  %9s  %-14s\n",
          "Problem", "Method", "it*", "objective", "viol", "native");
   printf("%s\n", std::string(86, '-').c_str());
   for (auto &pr : problems)
   {
      for (int mi = 0; mi < 3; ++mi)
      {
         Method m = (Method)mi;
         OptimProblem *prob = pr.build();
         Result r = RunOne(*prob, m, pr.x0, pr.has_star ? &pr.xstar : nullptr);
         char itbuf[16];
         if (!pr.has_star)         { snprintf(itbuf, sizeof itbuf, " -- "); }
         else if (r.it_target < 0) { snprintf(itbuf, sizeof itbuf, ">200"); }
         else                      { snprintf(itbuf, sizeof itbuf, "%4d", r.it_target); }
         char nbuf[24];
         snprintf(nbuf, sizeof nbuf, "%s(%d)", r.conv ? "conv" : "MAX", r.native_it);
         printf("%-32s %-6s  %4s  %12.6g  %9.2e  %-14s\n",
                mi == 0 ? pr.name.c_str() : "", MethodName(m), itbuf,
                r.obj, r.viol, nbuf);
         delete prob;
      }
      printf("%s\n", std::string(86, '-').c_str());
   }
   return 0;
}
