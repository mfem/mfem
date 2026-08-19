/**
 * test_relaxed_equalities_serial.cpp  —  Serial relaxed equality test suite
 *
 * Tests PackFivalRelaxed() and WithRelaxedEqualities() on MMAOptimizer and
 * SQOptimizer (single-process, plain mfem::Vector, no MPI).
 *
 * Relaxed equality convention
 * ───────────────────────────
 *   −leps_i < h_i(x) < ueps_i  is encoded as two standard inequality slots:
 *
 *     fi_pos_i = h_i(x) − ueps_i  ≤  0
 *     fi_neg_i = −h_i(x) − leps_i  ≤  0
 *
 *   PackFivalRelaxed(fi_ineq, h, leps, ueps)  — asymmetric tolerances
 *   PackFivalRelaxed(fi_ineq, h, eps)          — symmetric  leps = ueps = eps
 *   PackedDfidx for the gradients is unchanged from the strict-equality case.
 *
 * Problem design
 * ──────────────
 *   All problems use min (1/n)Σ(xj−aj)² (quadratic, analytic optimum known).
 *   SQ is exact for quadratic objectives; convergence expected in ≤ 5 steps.
 *   MMA converges reliably within 1 000 outer iterations on all cases.
 *
 *   Unconstrained minimiser mean(x)=mean(a) lies outside the band, so exactly
 *   one band boundary is active at the optimum:
 *     mean(a) > Vmid+ueps → upper bound active, x* s.t. mean(x*) = Vmid+ueps
 *     mean(a) < Vmid−leps → lower bound active, x* s.t. mean(x*) = Vmid−leps
 *
 * Test catalogue
 * ──────────────
 *  1. MMA — symmetric, upper bound active       (a=0.7, Vmid=0.4, eps=0.05)
 *  2. MMA — symmetric, lower bound active       (a=0.1, Vmid=0.4, eps=0.05)
 *  3. MMA — asymmetric leps≠ueps               (a=0.7, leps=0.02, ueps=0.10)
 *  4. MMA — two simultaneous relaxed equalities
 *  5. MMA — mixed: hard inequality + relaxed equality
 *  6. MMA — tolerance sweep eps=0.10, 0.03, 0.005
 *  7. SQ  — upper bound active  (SQ exact for quadratic, ≤ 5 iters)
 *  8. SQ  — lower bound active
 *  9. SQ  — asymmetric leps≠ueps
 * 10. SQ  — two simultaneous relaxed equalities
 * 11. MMA — GCMMA (UpdateGCMMA) with relaxed equality
 * 12. MMA — infeasible start (x₀ far below band)
 *
 * Build:  cmake --build build
 * Run:    ./build/test_relaxed_equalities_serial
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>

using namespace mfem;
using namespace mfem_mma;

static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// ── Quadratic objective + gradient ────────────────────────────────────────
static double QuadObj(const Vector& x, const Vector& a, Vector& df0)
{
    int n = x.Size();
    double f0 = 0;
    for (int j = 0; j < n; ++j) {
        double r = double(x(j)) - double(a(j));
        df0(j) = real_t(2.0 * r / n);
        f0 += r * r / n;
    }
    return f0;
}

// Analytic per-variable optimum: xj* = aj - mean(a) + Vtarget, clipped to [0.01,1].
static double AnalyticError(const Vector& x, const Vector& a, double Vtarget)
{
    int n = x.Size();
    double mean_a = 0;
    for (int j = 0; j < n; ++j) mean_a += double(a(j));
    mean_a /= n;
    double err = 0;
    for (int j = 0; j < n; ++j) {
        double xs = std::min(1.0, std::max(0.01, double(a(j)) - mean_a + Vtarget));
        err = std::max(err, std::abs(double(x(j)) - xs));
    }
    return err;
}

static double Mean(const Vector& x)
{
    double s = 0;
    for (int j = 0; j < x.Size(); ++j) s += double(x(j));
    return s / x.Size();
}

// ── Generic solver loop (MMA or SQ variant) ───────────────────────────────
// Returns final KKT residual. Fills x with the solution.
// h_func: lambda(x) → scalar h value (the equality function).
// Uses a SINGLE relaxed equality (n_eq=1) and no inequalities.
// leps, ueps: tolerance bounds.
template<typename Opt>
static real_t SolveRelaxed(
    Opt& opt, Vector& x,
    const Vector& a, const Vector& dh,
    const Vector& xmin, const Vector& xmax,
    double Vmid, double leps, double ueps,
    int max_iter, double kkt_tol)
{
    int n = x.Size();
    Vector df0(n), lv(1), uv(1);
    lv(0) = real_t(leps); uv(0) = real_t(ueps);

    real_t kkt = 1.0;
    for (int it = 0; it < max_iter && double(kkt) > kkt_tol
                                    && !std::isnan(double(kkt)); ++it)
    {
        double f0  = QuadObj(x, a, df0);
        double xm  = Mean(x);
        Vector h(1); h(0) = real_t(xm - Vmid);
        Vector fival = PackFivalRelaxed(Vector(0), h, lv, uv);
        Vector dh_arr[1] = {dh};
        PackedDfidx dfidx(nullptr, 0, dh_arr, 1);
        opt.Update(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);

        f0 = QuadObj(x, a, df0);
        xm = Mean(x);
        h(0) = real_t(xm - Vmid);
        fival = PackFivalRelaxed(Vector(0), h, lv, uv);
        kkt = opt.KKTresidual(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
    }
    return kkt;
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests 1 & 2: MMA symmetric — upper and lower bound active
// ═════════════════════════════════════════════════════════════════════════════
static void Test1_MMA_SymAbove()
{
    printf("\n── Test  1: MMA symmetric, upper bound active ──────────────\n");
    const int n = 200; const double Vmid = 0.4, eps = 0.05;
    // a=0.7 >> Vmid+eps=0.45 → upper bound active, x*=0.45
    Vector a(n), x(n), xmin(n), xmax(n), dh(n);
    a=real_t(0.7); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 1, x);
    real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, eps, eps, 1000, 1e-7);
    double xm = Mean(x), err = AnalyticError(x, a, Vmid+eps);
    printf("  kkt=%.2e  mean=%.5f(%.3f)  err=%.2e\n", double(kkt), xm, Vmid+eps, err);
    Check(kkt  < 1e-5,                    "MMA sym-above: KKT < 1e-5");
    Check(std::abs(xm-(Vmid+eps)) < 1e-3, "MMA sym-above: upper bound active");
    Check(err  < 1e-2,                    "MMA sym-above: analytic match");
}

static void Test2_MMA_SymBelow()
{
    printf("\n── Test  2: MMA symmetric, lower bound active ──────────────\n");
    const int n = 200; const double Vmid = 0.4, eps = 0.05;
    // a=0.1 << Vmid-eps=0.35 → lower bound active, x*=0.35
    Vector a(n), x(n), xmin(n), xmax(n), dh(n);
    a=real_t(0.1); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 1, x);
    real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, eps, eps, 1000, 1e-7);
    double xm = Mean(x), err = AnalyticError(x, a, Vmid-eps);
    printf("  kkt=%.2e  mean=%.5f(%.3f)  err=%.2e\n", double(kkt), xm, Vmid-eps, err);
    Check(kkt  < 1e-5,                    "MMA sym-below: KKT < 1e-5");
    Check(std::abs(xm-(Vmid-eps)) < 1e-3, "MMA sym-below: lower bound active");
    Check(err  < 1e-2,                    "MMA sym-below: analytic match");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3: MMA asymmetric (leps ≠ ueps)
// a=0.7, Vmid=0.4, leps=0.02, ueps=0.10.  Band: (0.38, 0.50).  x*=0.50.
// ═════════════════════════════════════════════════════════════════════════════
static void Test3_MMA_Asymmetric()
{
    printf("\n── Test  3: MMA asymmetric leps≠ueps ───────────────────────\n");
    const int n = 200; const double Vmid = 0.4, leps = 0.02, ueps = 0.10;
    Vector a(n), x(n), xmin(n), xmax(n), dh(n);
    a=real_t(0.7); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 1, x);
    real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, leps, ueps, 1000, 1e-7);
    double xm = Mean(x), h = xm - Vmid, err = AnalyticError(x, a, Vmid+ueps);
    printf("  kkt=%.2e  mean=%.5f  h=%.4f  band=(%.3f,%.3f)  err=%.2e\n",
           double(kkt), xm, h, -leps, ueps, err);
    Check(kkt  < 1e-5,        "MMA asym: KKT < 1e-5");
    Check(h   >= -leps-1e-3,  "MMA asym: h >= -leps");
    Check(h   <=  ueps+1e-3,  "MMA asym: h <=  ueps");
    Check(err  < 1e-2,        "MMA asym: upper bound active (analytic match)");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 4: MMA two simultaneous relaxed equalities
// Half 1: a=0.7, Vmid=0.40, eps=0.05 → upper active at 0.45
// Half 2: a=0.1, Vmid=0.50, eps=0.05 → lower active at 0.45
// ═════════════════════════════════════════════════════════════════════════════
static void Test4_MMA_Multiple()
{
    printf("\n── Test  4: MMA two relaxed equalities ─────────────────────\n");
    const int n = 200, half = n/2;
    const double Vmid1 = 0.4, Vmid2 = 0.5, eps = 0.05;
    Vector a(n), x(n), xmin(n), xmax(n), df0(n), dh1(n), dh2(n);
    for (int j=0;j<half;++j) { a(j)=real_t(0.7); dh1(j)=real_t(1.0/half); dh2(j)=0; }
    for (int j=half;j<n;++j) { a(j)=real_t(0.1); dh1(j)=0; dh2(j)=real_t(1.0/half); }
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 2, x);

    real_t kkt = 1.0;
    for (int it = 0; it < 1500 && double(kkt) > 1e-7 && !std::isnan(double(kkt)); ++it) {
        double f0 = QuadObj(x, a, df0);
        double s1=0, s2=0;
        for (int j=0;j<half;++j) s1 += double(x(j))/half;
        for (int j=half;j<n;++j) s2 += double(x(j))/half;
        Vector h(2); h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        Vector fival = PackFivalRelaxed(Vector(0), h, eps);
        Vector dh_arr[2] = {dh1, dh2}; PackedDfidx dfidx(nullptr, 0, dh_arr, 2);
        opt.Update(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
        f0 = QuadObj(x, a, df0);
        s1=s2=0;
        for (int j=0;j<half;++j) s1 += double(x(j))/half;
        for (int j=half;j<n;++j) s2 += double(x(j))/half;
        h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        fival = PackFivalRelaxed(Vector(0), h, eps);
        kkt = opt.KKTresidual(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
    }
    double s1=0, s2=0;
    for (int j=0;j<half;++j) s1 += double(x(j))/half;
    for (int j=half;j<n;++j) s2 += double(x(j))/half;
    printf("  kkt=%.2e  mean1=%.4f(0.45)  mean2=%.4f(0.45)\n", double(kkt), s1, s2);
    Check(kkt  < 1e-5,                     "MMA multi: KKT < 1e-5");
    Check(std::abs(s1-(Vmid1+eps)) < 1e-2, "MMA multi: eq1 upper bound active");
    Check(std::abs(s2-(Vmid2-eps)) < 1e-2, "MMA multi: eq2 lower bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 5: MMA mixed — hard inequality + relaxed equality
// min (1/n)Σ(xj-0.7)²  s.t.  mean(x)≤0.60  and  -eps<mean(x)-0.45<eps
// Band (0.40, 0.50). Unconstrained opt at mean=0.7 → upper relaxed bound active.
// x* = 0.50.
// ═════════════════════════════════════════════════════════════════════════════
static void Test5_MMA_Mixed()
{
    printf("\n── Test  5: MMA mixed ineq + relaxed equality ──────────────\n");
    const int n = 200; const double Vmid = 0.45, eps = 0.05, Vmax = 0.60;
    Vector a(n), x(n), xmin(n), xmax(n), df0(n), dg(n), dh(n);
    a=real_t(0.7); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) { dg(j)=real_t(1.0/n); dh(j)=real_t(1.0/n); }
    auto opt = MMAOptimizer::WithRelaxedEqualities(n, 1, 1, x);

    real_t kkt = 1.0;
    for (int it = 0; it < 1000 && double(kkt) > 1e-7 && !std::isnan(double(kkt)); ++it) {
        double f0 = QuadObj(x, a, df0);
        double xm = Mean(x);
        Vector fi(1); fi(0) = real_t(xm-Vmax);
        Vector h(1);   h(0) = real_t(xm-Vmid);
        Vector fival = PackFivalRelaxed(fi, h, eps);
        Vector dg_arr[1]={dg}, dh_arr[1]={dh};
        PackedDfidx dfidx(dg_arr, 1, dh_arr, 1);
        opt.Update(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
        f0 = QuadObj(x, a, df0); xm = Mean(x);
        fi(0)=real_t(xm-Vmax); h(0)=real_t(xm-Vmid);
        fival = PackFivalRelaxed(fi, h, eps);
        kkt = opt.KKTresidual(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
    }
    double xm = Mean(x), err = AnalyticError(x, a, Vmid+eps);
    printf("  kkt=%.2e  mean=%.5f  relax=(%.2f,%.2f)  ineq(<=%.2f)  err=%.2e\n",
           double(kkt), xm, Vmid-eps, Vmid+eps, Vmax, err);
    Check(kkt  < 1e-5,                    "MMA mixed: KKT < 1e-5");
    Check(xm   <= Vmax+1e-4,              "MMA mixed: hard inequality satisfied");
    Check(std::abs(xm-(Vmid+eps)) < 1e-2, "MMA mixed: relaxed upper bound active");
    Check(err  < 1e-2,                    "MMA mixed: analytic match");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 6: MMA tolerance sweep eps = 0.10, 0.03, 0.005
// a=0.7, Vmid=0.4.  Upper bound always active: mean(x*) = Vmid+eps.
// Checks: (a) in-band, (b) upper bound active, (c) tighter eps → smaller mean.
// ═════════════════════════════════════════════════════════════════════════════
static void Test6_MMA_ToleranceSweep()
{
    printf("\n── Test  6: MMA tolerance sweep ────────────────────────────\n");
    const int n = 200; const double Vmid = 0.4;
    Vector a(n); a = real_t(0.7);
    double prev_mean = -1;
    for (double eps : {0.10, 0.03, 0.005}) {
        Vector x(n), xmin(n), xmax(n), dh(n);
        x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
        for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
        auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 1, x);
        real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, eps, eps, 1000, 1e-7);
        double xm = Mean(x);
        printf("  eps=%.3f  mean=%.5f  |h|=%.3e  kkt=%.2e\n",
               eps, xm, std::abs(xm-Vmid), double(kkt));
        std::string tag = "eps=" + std::to_string(eps);
        Check(std::abs(xm-Vmid) <= eps+1e-3,        ("in-band: "+tag).c_str());
        Check(std::abs(xm-(Vmid+eps)) < 1e-2,        ("upper active: "+tag).c_str());
        if (prev_mean > 0)
            Check(xm <= prev_mean+1e-4, ("tighter eps → smaller mean: "+tag).c_str());
        prev_mean = xm;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 7 & 8: SQ symmetric — upper and lower bound active
// SQ is exact for quadratic objectives; converges in ≤ 5 outer iterations.
// ═════════════════════════════════════════════════════════════════════════════
static void Test7_SQ_SymAbove()
{
    printf("\n── Test  7: SQ symmetric, upper bound active ───────────────\n");
    const int n = 200; const double Vmid = 0.4, eps = 0.05;
    Vector a(n), x(n), xmin(n), xmax(n), dh(n);
    a=real_t(0.7); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = SQOptimizer::WithRelaxedEqualities(n, 0, 1, x);
    real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, eps, eps, 20, 1e-9);
    double xm = Mean(x), err = AnalyticError(x, a, Vmid+eps);
    printf("  kkt=%.2e  mean=%.5f(%.3f)  err=%.2e  iters=%d\n",
           double(kkt), xm, Vmid+eps, err, opt.NumIterations());
    Check(kkt  < 1e-7,                    "SQ sym-above: KKT < 1e-7 (exact for quadratic)");
    Check(std::abs(xm-(Vmid+eps)) < 1e-3, "SQ sym-above: upper bound active");
    Check(err  < 1e-3,                    "SQ sym-above: analytic match");
}

static void Test8_SQ_SymBelow()
{
    printf("\n── Test  8: SQ symmetric, lower bound active ───────────────\n");
    const int n = 200; const double Vmid = 0.4, eps = 0.05;
    Vector a(n), x(n), xmin(n), xmax(n), dh(n);
    a=real_t(0.1); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = SQOptimizer::WithRelaxedEqualities(n, 0, 1, x);
    real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, eps, eps, 20, 1e-9);
    double xm = Mean(x), err = AnalyticError(x, a, Vmid-eps);
    printf("  kkt=%.2e  mean=%.5f(%.3f)  err=%.2e  iters=%d\n",
           double(kkt), xm, Vmid-eps, err, opt.NumIterations());
    Check(kkt  < 1e-7,                    "SQ sym-below: KKT < 1e-7");
    Check(std::abs(xm-(Vmid-eps)) < 1e-3, "SQ sym-below: lower bound active");
    Check(err  < 1e-3,                    "SQ sym-below: analytic match");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 9: SQ asymmetric (leps ≠ ueps)
// ═════════════════════════════════════════════════════════════════════════════
static void Test9_SQ_Asymmetric()
{
    printf("\n── Test  9: SQ asymmetric leps≠ueps ────────────────────────\n");
    const int n = 200; const double Vmid = 0.4, leps = 0.02, ueps = 0.10;
    Vector a(n), x(n), xmin(n), xmax(n), dh(n);
    a=real_t(0.7); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = SQOptimizer::WithRelaxedEqualities(n, 0, 1, x);
    real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, leps, ueps, 20, 1e-9);
    double xm = Mean(x), h = xm-Vmid, err = AnalyticError(x, a, Vmid+ueps);
    printf("  kkt=%.2e  mean=%.5f  h=%.4f  band=(%.3f,%.3f)  err=%.2e\n",
           double(kkt), xm, h, -leps, ueps, err);
    Check(kkt  < 1e-7,        "SQ asym: KKT < 1e-7");
    Check(h   >= -leps-1e-4,  "SQ asym: h >= -leps");
    Check(h   <=  ueps+1e-4,  "SQ asym: h <=  ueps");
    Check(err  < 1e-3,        "SQ asym: upper bound active (analytic match)");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 10: SQ two simultaneous relaxed equalities
// ═════════════════════════════════════════════════════════════════════════════
static void Test10_SQ_Multiple()
{
    printf("\n── Test 10: SQ two relaxed equalities ──────────────────────\n");
    const int n = 200, half = n/2;
    const double Vmid1 = 0.4, Vmid2 = 0.5, eps = 0.05;
    Vector a(n), x(n), xmin(n), xmax(n), df0(n), dh1(n), dh2(n);
    for (int j=0;j<half;++j) { a(j)=real_t(0.7); dh1(j)=real_t(1.0/half); dh2(j)=0; }
    for (int j=half;j<n;++j) { a(j)=real_t(0.1); dh1(j)=0; dh2(j)=real_t(1.0/half); }
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    auto opt = SQOptimizer::WithRelaxedEqualities(n, 0, 2, x);

    real_t kkt = 1.0;
    for (int it = 0; it < 20 && double(kkt) > 1e-9 && !std::isnan(double(kkt)); ++it) {
        double f0 = QuadObj(x, a, df0);
        double s1=0, s2=0;
        for (int j=0;j<half;++j) s1 += double(x(j))/half;
        for (int j=half;j<n;++j) s2 += double(x(j))/half;
        Vector h(2); h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        Vector fival = PackFivalRelaxed(Vector(0), h, eps);
        Vector dh_arr[2] = {dh1, dh2}; PackedDfidx dfidx(nullptr, 0, dh_arr, 2);
        opt.Update(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
        f0 = QuadObj(x, a, df0);
        s1=s2=0;
        for (int j=0;j<half;++j) s1 += double(x(j))/half;
        for (int j=half;j<n;++j) s2 += double(x(j))/half;
        h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        fival = PackFivalRelaxed(Vector(0), h, eps);
        kkt = opt.KKTresidual(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
    }
    double s1=0, s2=0;
    for (int j=0;j<half;++j) s1 += double(x(j))/half;
    for (int j=half;j<n;++j) s2 += double(x(j))/half;
    printf("  kkt=%.2e  mean1=%.4f(0.45)  mean2=%.4f(0.45)  iters=%d\n",
           double(kkt), s1, s2, opt.NumIterations());
    Check(kkt  < 1e-7,                     "SQ multi: KKT < 1e-7");
    Check(std::abs(s1-(Vmid1+eps)) < 1e-3, "SQ multi: eq1 upper bound active");
    Check(std::abs(s2-(Vmid2-eps)) < 1e-3, "SQ multi: eq2 lower bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 11: GCMMA UpdateGCMMA with relaxed equality
// ═════════════════════════════════════════════════════════════════════════════
static void Test11_GCMMA()
{
    printf("\n── Test 11: GCMMA (UpdateGCMMA) relaxed equality ───────────\n");
    const int n = 200; const double Vmid = 0.4, eps = 0.05;
    Vector a(n), x(n), xmin(n), xmax(n), df0(n), dh(n);
    a=real_t(0.7); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 1, x);

    real_t kkt = 1.0;
    for (int it = 0; it < 1000 && double(kkt) > 1e-7 && !std::isnan(double(kkt)); ++it) {
        double f0 = QuadObj(x, a, df0);
        double xm = Mean(x);
        Vector h(1); h(0) = real_t(xm-Vmid);
        Vector fival = PackFivalRelaxed(Vector(0), h, eps);
        Vector dh_arr[1]={dh}; PackedDfidx dfidx(nullptr, 0, dh_arr, 1);
        opt.UpdateGCMMA(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
        f0 = QuadObj(x, a, df0); xm = Mean(x);
        h(0) = real_t(xm-Vmid);
        fival = PackFivalRelaxed(Vector(0), h, eps);
        kkt = opt.KKTresidual(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
    }
    double xm = Mean(x), err = AnalyticError(x, a, Vmid+eps);
    printf("  kkt=%.2e  mean=%.5f(%.3f)  err=%.2e\n", double(kkt), xm, Vmid+eps, err);
    Check(kkt  < 1e-5,                    "GCMMA: KKT < 1e-5");
    Check(std::abs(xm-(Vmid+eps)) < 1e-2, "GCMMA: upper bound active");
    Check(err  < 1e-2,                    "GCMMA: analytic match");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 12: MMA infeasible start (x₀ far outside the band)
// x₀=0.01, a=0.7, Vmid=0.4, eps=0.05.  Band (0.35, 0.45).  x*=0.45.
// ═════════════════════════════════════════════════════════════════════════════
static void Test12_InfeasibleStart()
{
    printf("\n── Test 12: MMA infeasible start ───────────────────────────\n");
    const int n = 200; const double Vmid = 0.4, eps = 0.05;
    Vector a(n), x(n), xmin(n), xmax(n), dh(n);
    a=real_t(0.7);
    x=real_t(0.01);   // mean=0.01 << Vmid-eps=0.35  (infeasible)
    xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 1, x);
    real_t kkt = SolveRelaxed(opt, x, a, dh, xmin, xmax, Vmid, eps, eps, 1500, 1e-7);
    double xm = Mean(x), h = xm-Vmid;
    printf("  kkt=%.2e  mean=%.5f  h=%.4e  band=(%.3f,%.3f)\n",
           double(kkt), xm, h, Vmid-eps, Vmid+eps);
    Check(kkt  < 1e-5,       "infeasible: KKT < 1e-5");
    Check(h   >= -eps-1e-3,  "infeasible: lower bound satisfied");
    Check(h   <=  eps+1e-3,  "infeasible: upper bound satisfied");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("╔══════════════════════════════════════════════════════════╗\n"
           "║  Relaxed equality test suite — serial                   ║\n"
           "╠══════════════════════════════════════════════════════════╣\n"
           "║  MMAOptimizer  (tests  1– 6, 11–12)                     ║\n"
           "║  SQOptimizer   (tests  7–10)                            ║\n"
           "║  Encoding: two inequality slots via PackFivalRelaxed()   ║\n"
           "╚══════════════════════════════════════════════════════════╝\n");

    Test1_MMA_SymAbove();
    Test2_MMA_SymBelow();
    Test3_MMA_Asymmetric();
    Test4_MMA_Multiple();
    Test5_MMA_Mixed();
    Test6_MMA_ToleranceSweep();
    Test7_SQ_SymAbove();
    Test8_SQ_SymBelow();
    Test9_SQ_Asymmetric();
    Test10_SQ_Multiple();
    Test11_GCMMA();
    Test12_InfeasibleStart();

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    if (g_nfail == 0)
        printf("║  All serial relaxed equality tests PASSED.              ║\n");
    else
        printf("║  %d serial relaxed equality test(s) FAILED.%-13s║\n", g_nfail, "");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    return g_nfail > 0 ? 1 : 0;
}
