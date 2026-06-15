/**
 * test_mma_unconstrained.cpp  —  Objective-only MMA tests (m=0)
 *
 * Tests mfem_mma::MMAOptimizer and MMAOptimizerParallel with zero
 * constraints across large problem sizes (n = 10 000 – 100 000).
 *
 * Three objective types:
 *
 *   1. Quadratic bowl:  f = Σ (xj − tj)²
 *      Exact optimum:   xj* = tj   (targets inside [xmin, xmax])
 *      Gradient:        dfj = 2(xj − tj)
 *
 *   2. Inverse sum:     f = Σ 1/xj
 *      Optimum at xmax: xj* = xmax  (gradient always negative)
 *      Gradient:        dfj = −1/xj²
 *
 *   3. Mixed separable: f = Σ_j [ α_j/xj + β_j * xj ]
 *      Interior optimum: xj* = sqrt(α_j/β_j)  clamped to [xmin,xmax]
 *      Gradient:         dfj = −α_j/xj² + β_j
 *
 * For each problem we measure:
 *   - KKT (projected gradient norm / n)
 *   - max pointwise error vs analytical optimum
 *   - wall-clock time per iteration
 *
 * Build:  cmake --build build
 * Run:    ./build/test_mma_unconstrained
 *         mpirun -np 4 ./build/test_mma_unconstrained
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <chrono>
#include <numeric>
#include <string>

using namespace mfem;
using namespace mfem_mma;
using Clock = std::chrono::steady_clock;

static int g_rank  = 0;
static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

static double GlobalSum(double v)
{
    double g; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return g;
}
static double GlobalMax(double v)
{
    double g; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return g;
}

#ifdef MFEM_USE_MPI
// Deterministic LCG pseudo-random [0,1) — same sequence on all ranks for
// reproducible global targets; seeded by global index so results are
// independent of rank count.
static double lcg(uint64_t& s)
{
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return double(s >> 33) / double(1ULL << 31);
}

static std::pair<int,int> Distribute(int n)
{
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    int base = n / nranks, rem = n % nranks;
    int nl = base + (g_rank < rem ? 1 : 0);
    int off = g_rank * base + std::min(g_rank, rem);
    return {nl, off};
}

// ============================================================
// Test 1 — Quadratic bowl  f = Σ(xj − tj)²
//   Exact interior optimum: xj* = tj,  tj ∈ (xmin, xmax)
// ============================================================
static void Test_QuadraticBowl(int n, bool use_gcmma = false)
{
    if (g_rank == 0)
        printf("\n--- QuadraticBowl (n=%d, m=0, %s) ---\n",
               n, use_gcmma ? "GCMMA" : "MMA");

    auto [nl, off] = Distribute(n);
    MPI_Comm comm  = MPI_COMM_WORLD;

    // Build deterministic targets: tj = 0.2 + 0.6 * u(j), u in [0,1)
    Vector x(nl), xmin(nl), xmax(nl), df0(nl), target(nl);
    xmin = 0.001; xmax = 1.0; x = 0.5;
    uint64_t seed = 12345ULL;
    // Advance seed to offset — cheap since we just need reproducibility
    uint64_t s = seed;
    for (int g = 0; g < off; ++g) lcg(s);            // skip to this rank's start
    for (int j = 0; j < nl; ++j)
        target(j) = real_t(0.2 + 0.6 * lcg(s));

    MMAOptimizerParallel opt(comm, nl, 0, x);      // m = 0
    double kkt = 1.0;
    int    it  = 0;

    auto t0 = Clock::now();
    for (; it < 200 && kkt > 1e-5; ++it) {
        // gradient of f = Σ(xj-tj)²:  dfj = 2*(xj - tj)
        double f0_loc = 0.0;
        for (int j = 0; j < nl; ++j) {
            double r = double(x(j)) - double(target(j));
            df0(j) = real_t(2.0 * r);
            f0_loc += r * r;
        }
        double f0 = GlobalSum(f0_loc);

        if (use_gcmma)
            opt.UpdateGCMMA(x, df0, f0, xmin, xmax);
        else
            opt.Update      (x, df0, f0, xmin, xmax);

        // KKT: projected gradient norm / n  (no constraints → just |df0|² / n)
        // Projected: if x==xmin and df0>0, pg=0; if x==xmax and df0<0, pg=0
        double pg2_loc = 0.0;
        const double tol_bnd = 1e-3;
        for (int j = 0; j < nl; ++j) {
            double g  = double(df0(j));
            double pg = g;
            if (double(x(j)) <= double(xmin(j)) + tol_bnd) pg = std::min(0.0, g);
            if (double(x(j)) >= double(xmax(j)) - tol_bnd) pg = std::max(0.0, g);
            pg2_loc += pg * pg;
        }
        kkt = GlobalSum(pg2_loc) / n;
        if (g_rank == 0 && it % 20 == 0)
            printf("  iter %3d: f0=%.4e  kkt=%.4e\n", it, f0, kkt);
    }
    auto t1  = Clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    // Error vs exact optimum
    double err_loc = 0.0;
    for (int j = 0; j < nl; ++j)
        err_loc = std::max(err_loc, std::abs(double(x(j)) - double(target(j))));
    double maxerr = GlobalMax(err_loc);

    if (g_rank == 0)
        printf("  Final: kkt=%.2e  max_err=%.2e  iters=%d  time=%.1fms"
               "  (%.2fms/iter)\n",
               kkt, maxerr, opt.GetIteration(), ms, ms/it);

    std::string tag = std::string("[") + (use_gcmma?"GCMMA":"MMA") +
                      ",n=" + std::to_string(n) + "]";
    Check(kkt < 1e-4,    (tag + " KKT < 1e-4").c_str());
    Check(maxerr < 0.01, (tag + " max_err < 0.01").c_str());
}

// ============================================================
// Test 2 — Inverse sum  f = Σ 1/xj
//   Optimum at upper bound: xj* = xmax = 1
//   Gradient: dfj = −1/xj²  (always negative → drives x to xmax)
// ============================================================
static void Test_InverseSum(int n)
{
    if (g_rank == 0)
        printf("\n--- InverseSum (n=%d, m=0) ---\n", n);

    auto [nl, off] = Distribute(n);
    MPI_Comm comm  = MPI_COMM_WORLD;

    Vector x(nl), xmin(nl), xmax(nl), df0(nl);
    xmin = 0.001; xmax = 1.0; x = 0.5;

    MMAOptimizerParallel opt(comm, nl, 0, x);
    double kkt = 1.0; int it = 0;

    auto t0 = Clock::now();
    for (; it < 200 && kkt > 1e-5; ++it) {
        double f0_loc = 0.0;
        for (int j = 0; j < nl; ++j) {
            double xj = double(x(j));
            df0(j) = real_t(-1.0 / (xj * xj));
            f0_loc += 1.0 / xj;
        }
        double f0 = GlobalSum(f0_loc);
        opt.Update(x, df0, f0, xmin, xmax);

        double pg2_loc = 0.0;
        for (int j = 0; j < nl; ++j) {
            double g  = double(df0(j));
            // x is being driven to xmax; only check lower-bound clamping
            double pg = (double(x(j)) >= double(xmax(j)) - 1e-3)
                        ? std::max(0.0, g) : g;
            pg2_loc += pg * pg;
        }
        kkt = GlobalSum(pg2_loc) / n;
        if (g_rank == 0 && it % 20 == 0)
            printf("  iter %3d: f0=%.4e  kkt=%.4e\n", it, f0, kkt);
    }
    auto t1  = Clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    // Optimum: all xj* = 1.0
    double xloc = 0.0;
    for (int j = 0; j < nl; ++j) xloc += double(x(j));
    double xmean = GlobalSum(xloc) / n;
    double err_loc = 0.0;
    for (int j = 0; j < nl; ++j)
        err_loc = std::max(err_loc, std::abs(double(x(j)) - 1.0));
    double maxerr = GlobalMax(err_loc);

    if (g_rank == 0)
        printf("  Final: xmean=%.6f (1.0)  kkt=%.2e  max_err=%.2e"
               "  iters=%d  time=%.1fms\n",
               xmean, kkt, maxerr, opt.GetIteration(), ms);

    std::string tag = "[InverseSum,n=" + std::to_string(n) + "]";
    Check(kkt < 1e-4,          (tag + " KKT < 1e-4").c_str());
    Check(std::abs(xmean-1.0) < 0.001, (tag + " mean(x) near 1.0").c_str());
    Check(maxerr < 0.01,       (tag + " max_err < 0.01").c_str());
}

// ============================================================
// Test 3 — Mixed separable  f = Σ_j [ α_j/xj + β_j*xj ]
//   Interior optimum: xj* = sqrt(α_j/β_j),  clamped to [xmin, xmax]
//   Gradient: dfj = −α_j/xj² + β_j
// ============================================================
static void Test_MixedSeparable(int n)
{
    if (g_rank == 0)
        printf("\n--- MixedSeparable (n=%d, m=0) ---\n", n);

    auto [nl, off] = Distribute(n);
    MPI_Comm comm  = MPI_COMM_WORLD;

    // Build coefficients: α_j in [0.5,2], β_j in [0.5,2]
    // Optimum: x* = sqrt(α/β) in [sqrt(0.5/2), sqrt(2/0.5)] = [0.5, 2.0]
    // Clamp to [0.001, 1.0]: targets in [0.5, 1.0]
    Vector x(nl), xmin(nl), xmax(nl), df0(nl), alpha(nl), beta(nl), xstar(nl);
    xmin = 0.001; xmax = 1.0; x = 0.5;

    uint64_t s = 98765ULL;
    for (int g = 0; g < off; ++g) { lcg(s); lcg(s); }
    for (int j = 0; j < nl; ++j) {
        double a = 0.5 + 1.5 * lcg(s);
        double b = 0.5 + 1.5 * lcg(s);
        alpha(j) = real_t(a);
        beta (j) = real_t(b);
        double xs = std::sqrt(a / b);
        xstar(j) = real_t(std::max(0.001, std::min(1.0, xs)));
    }

    MMAOptimizerParallel opt(comm, nl, 0, x);
    double kkt = 1.0; int it = 0;

    auto t0 = Clock::now();
    for (; it < 200 && kkt > 1e-5; ++it) {
        double f0_loc = 0.0;
        for (int j = 0; j < nl; ++j) {
            double xj = double(x(j));
            double aj = double(alpha(j)), bj = double(beta(j));
            df0(j)  = real_t(-aj/(xj*xj) + bj);
            f0_loc += aj/xj + bj*xj;
        }
        double f0 = GlobalSum(f0_loc);
        opt.Update(x, df0, f0, xmin, xmax);

        double pg2_loc = 0.0;
        for (int j = 0; j < nl; ++j) {
            double g  = double(df0(j));
            double pg = g;
            if (double(x(j)) <= double(xmin(j)) + 1e-3) pg = std::min(0.0, g);
            if (double(x(j)) >= double(xmax(j)) - 1e-3) pg = std::max(0.0, g);
            pg2_loc += pg * pg;
        }
        kkt = GlobalSum(pg2_loc) / n;
        if (g_rank == 0 && it % 20 == 0)
            printf("  iter %3d: f0=%.4e  kkt=%.4e\n", it, f0, kkt);
    }
    auto t1  = Clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    double err_loc = 0.0;
    for (int j = 0; j < nl; ++j)
        err_loc = std::max(err_loc, std::abs(double(x(j)) - double(xstar(j))));
    double maxerr = GlobalMax(err_loc);

    if (g_rank == 0)
        printf("  Final: kkt=%.2e  max_err=%.2e  iters=%d  time=%.1fms"
               "  (%.2fms/iter)\n",
               kkt, maxerr, opt.GetIteration(), ms, ms/it);

    std::string tag = "[MixedSep,n=" + std::to_string(n) + "]";
    Check(kkt < 1e-4,    (tag + " KKT < 1e-4").c_str());
    Check(maxerr < 0.02, (tag + " max_err < 0.02").c_str());
}

// ============================================================
// Test 4 — Serial MMAOptimizer, QuadraticBowl, large n
// ============================================================
static void Test_Serial_QuadraticBowl(int n)
{
    if (g_rank != 0) return;   // single-process test
    printf("\n--- Serial QuadraticBowl (n=%d, m=0) ---\n", n);

    Vector x(n), xmin(n), xmax(n), df0(n), target(n);
    xmin = 0.001; xmax = 1.0; x = 0.5;
    uint64_t s = 12345ULL;
    for (int j = 0; j < n; ++j)
        target(j) = real_t(0.2 + 0.6 * lcg(s));

    MMAOptimizer opt(n, 0, x);
    double kkt = 1.0; int it = 0;

    auto t0 = Clock::now();
    for (; it < 200 && kkt > 1e-5; ++it) {
        double f0 = 0.0;
        for (int j = 0; j < n; ++j) {
            double r = double(x(j)) - double(target(j));
            df0(j) = real_t(2.0 * r);
            f0 += r * r;
        }
        opt.Update(x, df0, f0, xmin, xmax);

        double pg2 = 0.0;
        for (int j = 0; j < n; ++j) {
            double g  = double(df0(j));
            double pg = g;
            if (double(x(j)) <= double(xmin(j)) + 1e-3) pg = std::min(0.0, g);
            if (double(x(j)) >= double(xmax(j)) - 1e-3) pg = std::max(0.0, g);
            pg2 += pg * pg;
        }
        kkt = pg2 / n;
        if (it % 20 == 0)
            printf("  iter %3d: f0=%.4e  kkt=%.4e\n", it, f0, kkt);
    }
    double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    double maxerr = 0.0;
    for (int j = 0; j < n; ++j)
        maxerr = std::max(maxerr, std::abs(double(x(j)) - double(target(j))));

    printf("  Final: kkt=%.2e  max_err=%.2e  iters=%d  time=%.1fms"
           "  (%.2fms/iter)\n",
           kkt, maxerr, opt.GetIteration(), ms, ms/it);

    std::string tag = "[serial,n=" + std::to_string(n) + "]";
    Check(kkt < 1e-4,    (tag + " KKT < 1e-4").c_str());
    Check(maxerr < 0.01, (tag + " max_err < 0.01").c_str());
}

#endif // MFEM_USE_MPI

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (g_rank == 0)
        printf("=== MFEM MMA Unconstrained (m=0) test suite"
               "  (%d rank(s)) ===\n", nranks);

    // ── Serial tests (run on rank 0 only) ────────────────────────────────
    if (g_rank == 0)
        printf("\n── Serial MMAOptimizer ──────────────────────────────────\n");
    Test_Serial_QuadraticBowl(10000);
    Test_Serial_QuadraticBowl(50000);
    Test_Serial_QuadraticBowl(100000);
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef MFEM_USE_MPI
    // ── Parallel tests (all ranks) ───────────────────────────────────────
    if (g_rank == 0)
        printf("\n── Parallel MMAOptimizerParallel ────────────────────────\n");

    // Quadratic bowl — MMA and GCMMA
    Test_QuadraticBowl(10000,  false);
    Test_QuadraticBowl(50000,  false);
    Test_QuadraticBowl(100000, false);
    Test_QuadraticBowl(10000,  true);   // GCMMA
    Test_QuadraticBowl(50000,  true);

    // Inverse sum
    Test_InverseSum(10000);
    Test_InverseSum(100000);

    // Mixed separable
    Test_MixedSeparable(10000);
    Test_MixedSeparable(50000);
    Test_MixedSeparable(100000);

#endif // MFEM_USE_MPI

    if (g_rank == 0) {
        printf("\n========================================\n");
        if (g_nfail == 0) printf("All unconstrained tests PASSED.\n");
        else              printf("%d unconstrained test(s) FAILED.\n", g_nfail);
        printf("========================================\n");
    }
    MPI_Finalize();
    return g_nfail > 0 ? 1 : 0;
}
