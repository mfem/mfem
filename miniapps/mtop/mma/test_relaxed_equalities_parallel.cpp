/**
 * test_relaxed_equalities_parallel.cpp  —  Parallel relaxed equality test suite
 *
 * Tests PackFivalRelaxed() and WithRelaxedEqualities() on MMAOptimizerParallel
 * and SQOptimizerParallel (MPI, distributed mfem::Vector).
 *
 * The file also calls MMAOptimizer / SQOptimizer (serial classes) on rank 0
 * to verify that serial and parallel results agree on identical problems.
 * Running with 1 rank exercises the serial fallback of all parallel classes.
 *
 * Build:  cmake --build build
 * Run (serial):   ./build/test_relaxed_equalities_parallel
 * Run (parallel): mpirun -np 4 ./build/test_relaxed_equalities_parallel
 *
 * Test catalogue
 * ──────────────
 *  1. MMAOptimizerParallel — symmetric, upper bound active
 *  2. MMAOptimizerParallel — symmetric, lower bound active
 *  3. MMAOptimizerParallel — asymmetric leps≠ueps
 *  4. MMAOptimizerParallel — two relaxed equalities
 *  5. MMAOptimizerParallel — mixed: hard inequality + relaxed equality
 *  6. SQOptimizerParallel  — symmetric, upper bound active (SQ exact for quadratic)
 *  7. SQOptimizerParallel  — symmetric, lower bound active
 *  8. SQOptimizerParallel  — asymmetric leps≠ueps
 *  9. SQOptimizerParallel  — two relaxed equalities
 * 10. Serial/parallel agreement — same problem solved with both,
 *     verifying that xmean converges to the same value within 1e-4.
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>

using namespace mfem;
using namespace mfem_mma;

static int g_rank  = 0;
static int g_nrank = 1;
static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// ── MPI helpers ───────────────────────────────────────────────────────────
static double GSum(double v)
{
    double g;
    MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return g;
}

static std::pair<int,int> Distribute(int n)
{
    int b = n / g_nrank, r = n % g_nrank;
    return {b + (g_rank < r ? 1 : 0), g_rank*b + std::min(g_rank, r)};
}

// ── Local (per-rank) helpers ──────────────────────────────────────────────
static double LocalQuadObj(const Vector& x, double a_val, Vector& df0, int n_global)
{
    int nl = x.Size();
    double f0 = 0;
    for (int j = 0; j < nl; ++j) {
        double r = double(x(j)) - a_val;
        df0(j) = real_t(2.0 * r / n_global);
        f0 += r * r / n_global;
    }
    return f0;   // local contribution; caller does GSum
}

static double LocalMean(const Vector& x, int n_global)
{
    double s = 0;
    for (int j = 0; j < x.Size(); ++j) s += double(x(j));
    return GSum(s) / n_global;
}

// ═════════════════════════════════════════════════════════════════════════════
// Generic parallel solver loop: single relaxed equality, uniform a across ranks.
// Returns final global KKT.
// ═════════════════════════════════════════════════════════════════════════════
template<typename Opt>
static real_t SolveParallel(
    Opt& opt,
    Vector& x,           // local chunk (n_local elements)
    double a_val,        // uniform target value
    const Vector& dh,    // local dh (= 1/n_global for all j)
    const Vector& xmin, const Vector& xmax,
    int n_global,
    double Vmid, double leps, double ueps,
    int max_iter, double kkt_tol)
{
    int nl = x.Size();
    Vector df0(nl), lv(1), uv(1);
    lv(0) = real_t(leps); uv(0) = real_t(ueps);
    real_t kkt = 1.0;

    for (int it = 0; it < max_iter && double(kkt) > kkt_tol
                                    && !std::isnan(double(kkt)); ++it)
    {
        double f0_loc = LocalQuadObj(x, a_val, df0, n_global);
        double xm     = LocalMean(x, n_global);
        Vector h(1); h(0) = real_t(xm - Vmid);
        Vector fival  = PackFivalRelaxed(Vector(0), h, lv, uv);
        Vector dh_arr[1] = {dh};
        PackedDfidx dfidx(nullptr, 0, dh_arr, 1);
        opt.Update(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);

        f0_loc = LocalQuadObj(x, a_val, df0, n_global);
        xm = LocalMean(x, n_global);
        h(0) = real_t(xm - Vmid);
        fival = PackFivalRelaxed(Vector(0), h, lv, uv);
        kkt = opt.KKTresidual(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);
    }
    return kkt;
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests 1 & 2: MMAOptimizerParallel symmetric — upper and lower bound active
// ═════════════════════════════════════════════════════════════════════════════
static void Test1_ParMMA_SymAbove()
{
    if (g_rank==0) printf("\n── Test  1: par-MMA symmetric, upper bound active ──────────\n");
    const int n = 2000; const double Vmid = 0.4, eps = 0.05;
    // a=0.7, upper bound active at mean=0.45
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 1, x);
    real_t kkt = SolveParallel(opt, x, 0.7, dh, xmin, xmax, n, Vmid, eps, eps, 1000, 1e-7);
    double xm = LocalMean(x, n);
    if (g_rank==0) printf("  kkt=%.2e  mean=%.5f(%.3f)\n", double(kkt), xm, Vmid+eps);
    Check(kkt < 1e-5,                    "par-MMA sym-above: KKT < 1e-5");
    Check(std::abs(xm-(Vmid+eps)) < 1e-2, "par-MMA sym-above: upper bound active");
}

static void Test2_ParMMA_SymBelow()
{
    if (g_rank==0) printf("\n── Test  2: par-MMA symmetric, lower bound active ──────────\n");
    const int n = 2000; const double Vmid = 0.4, eps = 0.05;
    // a=0.1, lower bound active at mean=0.35.
    // Use a tighter KKT threshold (1e-9) to avoid early exit caused by stale
    // multipliers from the warm-start renormalization making KKT appear converged
    // before the iterates have reached the correct lower-band boundary.
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 1, x);
    real_t kkt = SolveParallel(opt, x, 0.1, dh, xmin, xmax, n, Vmid, eps, eps, 3000, 1e-9);
    double xm = LocalMean(x, n);
    if (g_rank==0) printf("  kkt=%.2e  mean=%.5f(%.3f)\n", double(kkt), xm, Vmid-eps);
    Check(kkt < 1e-5,                    "par-MMA sym-below: KKT < 1e-5");
    Check(std::abs(xm-(Vmid-eps)) < 1e-2, "par-MMA sym-below: lower bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3: MMAOptimizerParallel asymmetric (leps ≠ ueps)
// ═════════════════════════════════════════════════════════════════════════════
static void Test3_ParMMA_Asymmetric()
{
    if (g_rank==0) printf("\n── Test  3: par-MMA asymmetric leps≠ueps ───────────────────\n");
    const int n = 2000; const double Vmid = 0.4, leps = 0.02, ueps = 0.10;
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 1, x);
    real_t kkt = SolveParallel(opt, x, 0.7, dh, xmin, xmax, n, Vmid, leps, ueps, 1000, 1e-7);
    double xm = LocalMean(x, n), h = xm-Vmid;
    if (g_rank==0) printf("  kkt=%.2e  mean=%.5f  h=%.4f  band=(%.3f,%.3f)\n",
                          double(kkt), xm, h, -leps, ueps);
    Check(kkt < 1e-5,        "par-MMA asym: KKT < 1e-5");
    Check(h  >= -leps-1e-3,  "par-MMA asym: h >= -leps");
    Check(h  <=  ueps+1e-3,  "par-MMA asym: h <=  ueps");
    Check(std::abs(xm-(Vmid+ueps)) < 1e-2, "par-MMA asym: upper bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 4: MMAOptimizerParallel two simultaneous relaxed equalities
// Half 1 of each rank: a=0.7, Vmid=0.40 → upper at 0.45
// Half 2 of each rank: a=0.1, Vmid=0.50 → lower at 0.45
// ═════════════════════════════════════════════════════════════════════════════
static void Test4_ParMMA_Multiple()
{
    if (g_rank==0) printf("\n── Test  4: par-MMA two relaxed equalities ─────────────────\n");
    const int n = 2000, half_global = n/2;
    const double Vmid1 = 0.4, Vmid2 = 0.5, eps = 0.05;
    auto [nl, off] = Distribute(n);
    Vector a(nl), x(nl), xmin(nl), xmax(nl), df0(nl), dh1(nl), dh2(nl);
    // Assign a and gradient weights per variable based on global index
    for (int j = 0; j < nl; ++j) {
        int gj = off + j;
        a(j)   = real_t(gj < half_global ? 0.7 : 0.1);
        dh1(j) = real_t(gj < half_global ? 1.0/half_global : 0.0);
        dh2(j) = real_t(gj < half_global ? 0.0 : 1.0/half_global);
    }
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    auto opt = MMAOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 2, x);

    real_t kkt = 1.0;
    for (int it = 0; it < 1500 && double(kkt) > 1e-7 && !std::isnan(double(kkt)); ++it) {
        double f0_loc = 0;
        for (int j = 0; j < nl; ++j) {
            double r = double(x(j)) - double(a(j));
            df0(j) = real_t(2.0*r/n); f0_loc += r*r/n;
        }
        // Local sums for each half; global reduce gives the means
        double s1_loc=0, s2_loc=0;
        for (int j=0;j<nl;++j) {
            int gj=off+j;
            if (gj<half_global) s1_loc+=double(x(j));
            else                s2_loc+=double(x(j));
        }
        double s1=GSum(s1_loc)/half_global, s2=GSum(s2_loc)/half_global;
        Vector h(2); h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        Vector fival = PackFivalRelaxed(Vector(0), h, eps);
        Vector dh_arr[2]={dh1,dh2}; PackedDfidx dfidx(nullptr, 0, dh_arr, 2);
        opt.Update(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);

        f0_loc=0;
        for (int j=0;j<nl;++j) {
            double r=double(x(j))-double(a(j));
            df0(j)=real_t(2.0*r/n); f0_loc+=r*r/n;
        }
        s1_loc=s2_loc=0;
        for (int j=0;j<nl;++j) {
            int gj=off+j;
            if (gj<half_global) s1_loc+=double(x(j));
            else                s2_loc+=double(x(j));
        }
        s1=GSum(s1_loc)/half_global; s2=GSum(s2_loc)/half_global;
        h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        fival = PackFivalRelaxed(Vector(0), h, eps);
        kkt = opt.KKTresidual(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);
    }
    double s1_loc=0, s2_loc=0;
    for (int j=0;j<nl;++j) {
        int gj=off+j;
        if (gj<half_global) s1_loc+=double(x(j)); else s2_loc+=double(x(j));
    }
    double s1=GSum(s1_loc)/half_global, s2=GSum(s2_loc)/half_global;
    if (g_rank==0) printf("  kkt=%.2e  mean1=%.4f(0.45)  mean2=%.4f(0.45)\n",
                          double(kkt), s1, s2);
    Check(kkt < 1e-5,                     "par-MMA multi: KKT < 1e-5");
    Check(std::abs(s1-(Vmid1+eps)) < 1e-2, "par-MMA multi: eq1 upper bound active");
    Check(std::abs(s2-(Vmid2-eps)) < 1e-2, "par-MMA multi: eq2 lower bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 5: MMAOptimizerParallel mixed — hard inequality + relaxed equality
// ═════════════════════════════════════════════════════════════════════════════
static void Test5_ParMMA_Mixed()
{
    if (g_rank==0) printf("\n── Test  5: par-MMA mixed ineq + relaxed equality ──────────\n");
    const int n = 2000; const double Vmid = 0.45, eps = 0.05, Vmax = 0.60;
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), df0(nl), dg(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) { dg(j)=real_t(1.0/n); dh(j)=real_t(1.0/n); }
    auto opt = MMAOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 1, 1, x);

    real_t kkt = 1.0;
    for (int it = 0; it < 1000 && double(kkt) > 1e-7 && !std::isnan(double(kkt)); ++it) {
        double f0_loc = LocalQuadObj(x, 0.7, df0, n);
        double xm     = LocalMean(x, n);
        Vector fi(1); fi(0)=real_t(xm-Vmax);
        Vector h(1);   h(0)=real_t(xm-Vmid);
        Vector fival = PackFivalRelaxed(fi, h, eps);
        Vector dg_arr[1]={dg}, dh_arr[1]={dh};
        PackedDfidx dfidx(dg_arr, 1, dh_arr, 1);
        opt.Update(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);
        f0_loc=LocalQuadObj(x,0.7,df0,n); xm=LocalMean(x,n);
        fi(0)=real_t(xm-Vmax); h(0)=real_t(xm-Vmid);
        fival = PackFivalRelaxed(fi, h, eps);
        kkt = opt.KKTresidual(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);
    }
    double xm = LocalMean(x, n);
    if (g_rank==0) printf("  kkt=%.2e  mean=%.5f  relax=(%.2f,%.2f)  ineq(<=%.2f)\n",
                          double(kkt), xm, Vmid-eps, Vmid+eps, Vmax);
    Check(kkt < 1e-5,                    "par-MMA mixed: KKT < 1e-5");
    Check(xm  <= Vmax+1e-4,              "par-MMA mixed: hard inequality satisfied");
    Check(std::abs(xm-(Vmid+eps)) < 1e-2, "par-MMA mixed: relaxed upper bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests 6–7: SQOptimizerParallel symmetric — upper and lower bound active
// SQ is exact for quadratic objectives; converges in ≤ 5 outer iterations.
// ═════════════════════════════════════════════════════════════════════════════
static void Test6_ParSQ_SymAbove()
{
    if (g_rank==0) printf("\n── Test  6: par-SQ symmetric, upper bound active ───────────\n");
    const int n = 2000; const double Vmid = 0.4, eps = 0.05;
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);
    auto opt = SQOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 1, x);
    real_t kkt = SolveParallel(opt, x, 0.7, dh, xmin, xmax, n, Vmid, eps, eps, 20, 1e-9);
    double xm = LocalMean(x, n);
    if (g_rank==0) printf("  kkt=%.2e  mean=%.5f(%.3f)  iters=%d\n",
                          double(kkt), xm, Vmid+eps, opt.NumIterations());
    Check(kkt < 1e-7,                    "par-SQ sym-above: KKT < 1e-7 (exact for quadratic)");
    Check(std::abs(xm-(Vmid+eps)) < 1e-2, "par-SQ sym-above: upper bound active");
}

static void Test7_ParSQ_SymBelow()
{
    if (g_rank==0) printf("\n── Test  7: par-SQ symmetric, lower bound active ───────────\n");
    const int n = 2000; const double Vmid = 0.4, eps = 0.05;
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);
    auto opt = SQOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 1, x);
    real_t kkt = SolveParallel(opt, x, 0.1, dh, xmin, xmax, n, Vmid, eps, eps, 20, 1e-9);
    double xm = LocalMean(x, n);
    if (g_rank==0) printf("  kkt=%.2e  mean=%.5f(%.3f)  iters=%d\n",
                          double(kkt), xm, Vmid-eps, opt.NumIterations());
    Check(kkt < 1e-7,                    "par-SQ sym-below: KKT < 1e-7");
    Check(std::abs(xm-(Vmid-eps)) < 1e-2, "par-SQ sym-below: lower bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 8: SQOptimizerParallel asymmetric
// ═════════════════════════════════════════════════════════════════════════════
static void Test8_ParSQ_Asymmetric()
{
    if (g_rank==0) printf("\n── Test  8: par-SQ asymmetric leps≠ueps ────────────────────\n");
    const int n = 2000; const double Vmid = 0.4, leps = 0.02, ueps = 0.10;
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);
    auto opt = SQOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 1, x);
    real_t kkt = SolveParallel(opt, x, 0.7, dh, xmin, xmax, n, Vmid, leps, ueps, 20, 1e-9);
    double xm = LocalMean(x, n), h = xm-Vmid;
    if (g_rank==0) printf("  kkt=%.2e  mean=%.5f  h=%.4f  band=(%.3f,%.3f)\n",
                          double(kkt), xm, h, -leps, ueps);
    Check(kkt < 1e-7,        "par-SQ asym: KKT < 1e-7");
    Check(h  >= -leps-1e-3,  "par-SQ asym: h >= -leps");
    Check(h  <=  ueps+1e-3,  "par-SQ asym: h <=  ueps");
    Check(std::abs(xm-(Vmid+ueps)) < 1e-2, "par-SQ asym: upper bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 9: SQOptimizerParallel two simultaneous relaxed equalities
// ═════════════════════════════════════════════════════════════════════════════
static void Test9_ParSQ_Multiple()
{
    if (g_rank==0) printf("\n── Test  9: par-SQ two relaxed equalities ──────────────────\n");
    const int n = 2000, half_global = n/2;
    const double Vmid1 = 0.4, Vmid2 = 0.5, eps = 0.05;
    auto [nl, off] = Distribute(n);
    Vector a(nl), x(nl), xmin(nl), xmax(nl), df0(nl), dh1(nl), dh2(nl);
    for (int j = 0; j < nl; ++j) {
        int gj = off + j;
        a(j)   = real_t(gj < half_global ? 0.7 : 0.1);
        dh1(j) = real_t(gj < half_global ? 1.0/half_global : 0.0);
        dh2(j) = real_t(gj < half_global ? 0.0 : 1.0/half_global);
    }
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    auto opt = SQOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 2, x);

    real_t kkt = 1.0;
    for (int it = 0; it < 20 && double(kkt) > 1e-9 && !std::isnan(double(kkt)); ++it) {
        double f0_loc=0;
        for (int j=0;j<nl;++j) {
            double r=double(x(j))-double(a(j));
            df0(j)=real_t(2.0*r/n); f0_loc+=r*r/n;
        }
        double s1_loc=0, s2_loc=0;
        for (int j=0;j<nl;++j) {
            int gj=off+j;
            if (gj<half_global) s1_loc+=double(x(j)); else s2_loc+=double(x(j));
        }
        double s1=GSum(s1_loc)/half_global, s2=GSum(s2_loc)/half_global;
        Vector h(2); h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        Vector fival = PackFivalRelaxed(Vector(0), h, eps);
        Vector dh_arr[2]={dh1,dh2}; PackedDfidx dfidx(nullptr, 0, dh_arr, 2);
        opt.Update(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);
        f0_loc=0;
        for (int j=0;j<nl;++j) {
            double r=double(x(j))-double(a(j));
            df0(j)=real_t(2.0*r/n); f0_loc+=r*r/n;
        }
        s1_loc=s2_loc=0;
        for (int j=0;j<nl;++j) {
            int gj=off+j;
            if (gj<half_global) s1_loc+=double(x(j)); else s2_loc+=double(x(j));
        }
        s1=GSum(s1_loc)/half_global; s2=GSum(s2_loc)/half_global;
        h(0)=real_t(s1-Vmid1); h(1)=real_t(s2-Vmid2);
        fival = PackFivalRelaxed(Vector(0), h, eps);
        kkt = opt.KKTresidual(x, df0, real_t(GSum(f0_loc)), fival, dfidx.data(), xmin, xmax);
    }
    double s1_loc=0, s2_loc=0;
    for (int j=0;j<nl;++j) {
        int gj=off+j;
        if (gj<half_global) s1_loc+=double(x(j)); else s2_loc+=double(x(j));
    }
    double s1=GSum(s1_loc)/half_global, s2=GSum(s2_loc)/half_global;
    if (g_rank==0) printf("  kkt=%.2e  mean1=%.4f(0.45)  mean2=%.4f(0.45)  iters=%d\n",
                          double(kkt), s1, s2, opt.NumIterations());
    Check(kkt < 1e-7,                     "par-SQ multi: KKT < 1e-7");
    Check(std::abs(s1-(Vmid1+eps)) < 1e-2, "par-SQ multi: eq1 upper bound active");
    Check(std::abs(s2-(Vmid2-eps)) < 1e-2, "par-SQ multi: eq2 lower bound active");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 10: Serial / parallel agreement
//
// Both MMAOptimizer (serial, n=200) and MMAOptimizerParallel (parallel, n=200
// distributed) solve the SAME problem.  The converged means must agree to 1e-4.
// Run on rank 0 only for the serial part; barrier synchronises before comparing.
// ═════════════════════════════════════════════════════════════════════════════
static void Test10_SerialParallelAgreement()
{
    if (g_rank==0) printf("\n── Test 10: serial/parallel agreement ──────────────────────\n");
    const int n = 200; const double Vmid = 0.4, eps = 0.05;

    // ── Serial (MMAOptimizer on rank 0) ──────────────────────────────────
    double xm_serial = 0;
    if (g_rank == 0) {
        Vector a(n), x(n), xmin(n), xmax(n), df0(n), dh(n);
        a=real_t(0.7); x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
        for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);
        auto opt = MMAOptimizer::WithRelaxedEqualities(n, 0, 1, x);
        real_t kkt = 1.0;
        for (int it=0; it<1000 && double(kkt)>1e-7 && !std::isnan(double(kkt)); ++it) {
            double f0=0;
            for(int j=0;j<n;++j){ double r=double(x(j))-0.7; df0(j)=real_t(2.0*r/n); f0+=r*r/n; }
            double xm=0; for(int j=0;j<n;++j) xm+=double(x(j)); xm/=n;
            Vector h(1); h(0)=real_t(xm-Vmid);
            Vector fival=PackFivalRelaxed(Vector(0),h,eps);
            Vector dh_arr[1]={dh}; PackedDfidx dfidx(nullptr,0,dh_arr,1);
            opt.Update(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
            f0=0;
            for(int j=0;j<n;++j){ double r=double(x(j))-0.7; df0(j)=real_t(2.0*r/n); f0+=r*r/n; }
            xm=0; for(int j=0;j<n;++j) xm+=double(x(j)); xm/=n;
            h(0)=real_t(xm-Vmid);
            fival=PackFivalRelaxed(Vector(0),h,eps);
            kkt=opt.KKTresidual(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
        }
        xm_serial=0; for(int j=0;j<n;++j) xm_serial+=double(x(j)); xm_serial/=n;
    }
    MPI_Bcast(&xm_serial, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ── Parallel (MMAOptimizerParallel on all ranks) ──────────────────────
    auto [nl, off] = Distribute(n);
    Vector x(nl), xmin(nl), xmax(nl), dh(nl);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);
    auto opt = MMAOptimizerParallel::WithRelaxedEqualities(MPI_COMM_WORLD, nl, 0, 1, x);
    real_t kkt = SolveParallel(opt, x, 0.7, dh, xmin, xmax, n, Vmid, eps, eps, 1000, 1e-7);
    double xm_par = LocalMean(x, n);

    if (g_rank==0) printf("  serial mean=%.6f  parallel mean=%.6f  diff=%.2e\n",
                          xm_serial, xm_par, std::abs(xm_serial-xm_par));
    Check(std::abs(xm_serial-xm_par) < 1e-4, "serial/parallel agreement: mean diff < 1e-4");
    Check(kkt < 1e-5,                          "serial/parallel: parallel KKT < 1e-5");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nrank);

    if (g_rank == 0)
        printf("╔══════════════════════════════════════════════════════════╗\n"
               "║  Relaxed equality test suite — parallel                 ║\n"
               "║  (%2d rank(s))                                           ║\n"
               "╠══════════════════════════════════════════════════════════╣\n"
               "║  MMAOptimizerParallel   (tests 1–5)                     ║\n"
               "║  SQOptimizerParallel    (tests 6–9)                     ║\n"
               "║  Serial/parallel match  (test  10)                      ║\n"
               "╚══════════════════════════════════════════════════════════╝\n",
               g_nrank);

    Test1_ParMMA_SymAbove();
    Test2_ParMMA_SymBelow();
    Test3_ParMMA_Asymmetric();
    Test4_ParMMA_Multiple();
    Test5_ParMMA_Mixed();
    Test6_ParSQ_SymAbove();
    Test7_ParSQ_SymBelow();
    Test8_ParSQ_Asymmetric();
    Test9_ParSQ_Multiple();
    Test10_SerialParallelAgreement();

    if (g_rank == 0) {
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        if (g_nfail == 0)
            printf("║  All parallel relaxed equality tests PASSED.            ║\n");
        else
            printf("║  %d parallel relaxed equality test(s) FAILED.%-11s║\n", g_nfail, "");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail > 0 ? 1 : 0;
}
