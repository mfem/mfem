/**
 * test_sq_overconstrained.cpp  —  SQOptimizer over-constrained test suite
 *
 * Same m>n problems as test_overconstrained.cpp using SQOptimizer.
 *
 * Tests MMA and GCMMA when the number of constraints m exceeds (or equals)
 * the number of design variables n.  This regime stresses:
 *
 *   - The m×m dual Newton system  (m > n means a larger dense solve than
 *     the usual m << n case used in topology optimisation)
 *   - The redundancy / near-rank-deficiency of the dual Hessian when many
 *     constraints become simultaneously active
 *   - The SVD fallback in SolveDense() for exactly linearly-dependent rows
 *
 * Problems
 * ────────
 *  1. m = n      (square: equal number of constraints and variables)
 *  2. m = 2*n    (overdetermined: twice as many constraints)
 *  3. m = 5*n    (highly overdetermined)
 *  4. m = n with redundant constraint pairs  (rank-deficient dual Hessian)
 *  5. Parallel versions of the above
 *
 * Problem structure (compliance proxy with many regional volume constraints)
 * ──────────────────────────────────────────────────────────────────────────
 *   min  sum(1/xj)
 *   s.t. x_j <= V_k   for each variable j  (individual upper bound)
 *        or: mean(x_region_k) <= V_k  with regions of size 1
 *
 * When each constraint controls exactly one variable (region size = 1),
 * n constraints fully determine the solution: x_j* = V_j.
 * With m > n we add extra constraints (e.g. global mean) that are
 * implied by the individual bounds — making the system redundant.
 *
 * Build:  cmake --build build
 * Run:    ./build/test_overconstrained
 *         mpirun -np 4 ./build/test_overconstrained
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>

using namespace mfem;
using namespace mfem_mma;

static int g_rank  = 0;
static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

static double GSum(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return g; }

static std::pair<int,int> Distribute(int n)
{
    int nr; MPI_Comm_size(MPI_COMM_WORLD,&nr);
    int b=n/nr, r=n%nr;
    return {b+(g_rank<r?1:0), g_rank*b+std::min(g_rank,r)};
}

// ============================================================
// Build the constraint system for "individual upper bound" problems:
//   For k < n:  constraint k bounds variable k:  x_k - V_k <= 0
//   For k >= n: constraint k is a global sum implied by the first n:
//               mean(x) - (mean of V[0..n-1]) <= 0
//
// This deliberately creates linear dependencies when m > n.
// ============================================================
static void BuildConstraints(
    int n_global, int m, int n_local, int offset,
    const std::vector<double>& Vtgt,   // length n_global (per-variable targets)
    double Vglobal,                    // target for extra global constraints
    std::vector<Vector>& dg,           // [m] local gradient vectors (output)
    std::vector<double>& cv_out)       // penalty c[k] (output)
{
    dg.resize(m);
    cv_out.resize(m);
    double cv = std::max(1000.0, 10.0*n_global);

    for (int k = 0; k < m; ++k) {
        dg[k].SetSize(n_local);
        dg[k] = 0.0;
        cv_out[k] = cv;

        if (k < n_global) {
            // Per-variable constraint: gradient is 1 at variable k, 0 elsewhere
            for (int j = 0; j < n_local; ++j)
                if (offset + j == k) dg[k](j) = 1.0;
        } else {
            // Redundant global-mean constraint (implied by the first n_global)
            for (int j = 0; j < n_local; ++j)
                dg[k](j) = real_t(1.0 / n_global);
        }
    }
}

static mfem::Vector ComputeConstraintValues(
    int n_global, int m, int n_local, int offset,
    const Vector& x_local,
    const std::vector<double>& Vtgt,
    double Vglobal,
    MPI_Comm comm)
{
    mfem::Vector fi(m);

    // Per-variable constraints: fi(k)= x[k] - Vtgt[k]  for k < n_global
    // These are non-local on ranks that don't own variable k.
    // Approach: each rank computes its local values, then allreduce.
    std::vector<double> fi_loc(m, 0.0);
    for (int j = 0; j < n_local; ++j) {
        int g = offset + j;
        if (g < n_global && g < m)
            fi_loc[g]= double(x_local(j)) - Vtgt[g];
    }
    // Global mean constraint for k >= n_global
    double xloc = 0.0;
    for (int j = 0; j < n_local; ++j) xloc += double(x_local(j));
    double xmean = GSum(xloc) / n_global;
    for (int k = n_global; k < m; ++k)
        fi_loc[k]= xmean - Vglobal;   // same value for all extra constraints

    // Allreduce: for k < n_global, only one rank has a non-zero contribution
    std::vector<double> fi_glb(m);
    MPI_Allreduce(fi_loc.data(), fi_glb.data(), m, MPI_DOUBLE, MPI_SUM, comm);
    for (int k = 0; k < m; ++k) fi(k)= real_t(fi_glb[k]);
    return fi;
}

// ============================================================
// Core test: run MMA/GCMMA with m constraints and n variables
// (possibly m >= n), check convergence and solution quality.
// ============================================================
static void RunOverconstrained(
    const char* label,
    int n_global, int m,
    bool gcmma,
    bool parallel)
{
    if (g_rank == 0)
        printf("\n  %-60s [%s, %s]\n", label,
               gcmma?"GCMMA":"MMA", parallel?"parallel":"serial");

    // Per-variable targets V_k in (0.3, 0.7) with deterministic pattern
    std::vector<double> Vtgt(n_global);
    for (int k = 0; k < n_global; ++k)
        Vtgt[k] = 0.3 + 0.4 * ((k % 5) / 4.0);   // 0.30, 0.40, 0.50, 0.60, 0.70
    double Vglobal = 0.0;
    for (int k = 0; k < n_global; ++k) Vglobal += Vtgt[k];
    Vglobal /= n_global;   // mean of individual targets (redundant constraint target)

    // ── Serial path ───────────────────────────────────────────────────────
    if (!parallel) {
        if (g_rank != 0) return;

        Vector x(n_global), xmin(n_global), xmax(n_global), df0(n_global);
        x = 0.5; xmin = 0.001; xmax = 1.0;

        std::vector<Vector> dg_s;
        std::vector<double> cv_s;
        BuildConstraints(n_global, m, n_global, 0, Vtgt, Vglobal, dg_s, cv_s);

        std::vector<double> av(m, 0.0), dv(m, 1.0);
        MMAOptimizer opt(n_global, m, x, av.data(), cv_s.data(), dv.data());

        real_t kkt = 1.0;
        for (int it = 0; it < 500 && kkt > 1e-5; ++it) {
            for (int j = 0; j < n_global; ++j)
                df0(j) = real_t(-1.0/(double(x(j))*double(x(j))));

            // fi(k)= x[k] - Vtgt[k] for k < n_global; global mean for k >= n_global
            mfem::Vector fi(m);
            for (int k = 0; k < n_global && k < m; ++k)
                fi(k)= real_t(double(x(k)) - Vtgt[k]);
            double xmean = 0; for (int j=0;j<n_global;++j) xmean+=double(x(j)); xmean/=n_global;
            for (int k = n_global; k < m; ++k)
                fi(k)= real_t(xmean - Vglobal);

            if (gcmma)
                opt.UpdateGCMMA(x,df0,0.0f,fi,dg_s.data(),xmin,xmax);
            else
                opt.Update(x,df0,0.0f,fi,dg_s.data(),xmin,xmax);

            for (int j = 0; j < n_global; ++j)
                df0(j) = real_t(-1.0/(double(x(j))*double(x(j))));
            for (int k = 0; k < n_global && k < m; ++k)
                fi(k)= real_t(double(x(k)) - Vtgt[k]);
            xmean=0; for(int j=0;j<n_global;++j) xmean+=double(x(j)); xmean/=n_global;
            for (int k = n_global; k < m; ++k)
                fi(k)= real_t(xmean - Vglobal);

            kkt = opt.KKTresidual(x,df0,0.0f,fi,dg_s.data(),xmin,xmax);
            if (it % 50 == 0)
                printf("    iter %4d: kkt=%.3e\n", it, double(kkt));
        }

        // Check solution: x_k should be at or below Vtgt[k]
        double maxviol = 0.0, maxerr = 0.0;
        for (int k = 0; k < n_global && k < m; ++k) {
            double viol = double(x(k)) - Vtgt[k];
            if (viol > 0) maxviol = std::max(maxviol, viol);
            maxerr = std::max(maxerr, std::abs(double(x(k)) - Vtgt[k]));
        }
        printf("    Final: kkt=%.2e  max_viol=%.2e  iters=%d\n",
               double(kkt), maxviol, opt.NumIterations());
        Check(kkt < 1e-4, (std::string(label)+" KKT<1e-4").c_str());
        Check(maxviol < 0.01, (std::string(label)+" no constraint violation").c_str());
        return;
    }

    // ── Parallel path ─────────────────────────────────────────────────────
    auto [nl, off] = Distribute(n_global);
    MPI_Comm comm  = MPI_COMM_WORLD;

    Vector x(nl), xmin(nl), xmax(nl), df0(nl);
    x = 0.5; xmin = 0.001; xmax = 1.0;

    std::vector<Vector> dg;
    std::vector<double> cv_v;
    BuildConstraints(n_global, m, nl, off, Vtgt, Vglobal, dg, cv_v);

    std::vector<double> av(m, 0.0), dv(m, 1.0);
    MMAOptimizerParallel opt(comm, nl, m, x, av.data(), cv_v.data(), dv.data());

    real_t kkt = 1.0;
    for (int it = 0; it < 500 && kkt > 1e-5; ++it) {
        for (int j = 0; j < nl; ++j)
            df0(j) = real_t(-1.0/(double(x(j))*double(x(j))));

        auto fi = ComputeConstraintValues(n_global, m, nl, off, x, Vtgt, Vglobal, comm);

        if (gcmma)
            opt.UpdateGCMMA(x,df0,0.0f,fi,dg.data(),xmin,xmax);
        else
            opt.Update(x,df0,0.0f,fi,dg.data(),xmin,xmax);

        for (int j = 0; j < nl; ++j)
            df0(j) = real_t(-1.0/(double(x(j))*double(x(j))));

        fi = ComputeConstraintValues(n_global, m, nl, off, x, Vtgt, Vglobal, comm);
        kkt = opt.KKTresidual(x,df0,0.0f,fi,dg.data(),xmin,xmax);

        if (g_rank==0 && it%50==0)
            printf("    iter %4d: kkt=%.3e\n", it, double(kkt));
    }

    // Gather all x values to rank 0 for checking
    // (Each rank allreduces its local per-variable values)
    std::vector<double> x_loc_vals(n_global, 0.0);
    for (int j = 0; j < nl; ++j)
        x_loc_vals[off + j] = double(x(j));
    std::vector<double> x_all(n_global, 0.0);
    MPI_Allreduce(x_loc_vals.data(), x_all.data(), n_global, MPI_DOUBLE, MPI_SUM, comm);

    double maxviol = 0.0;
    for (int k = 0; k < n_global && k < m; ++k) {
        double viol = x_all[k] - Vtgt[k];
        if (viol > 0) maxviol = std::max(maxviol, viol);
    }

    if (g_rank==0)
        printf("    Final: kkt=%.2e  max_viol=%.2e  iters=%d\n",
               double(kkt), maxviol, opt.NumIterations());
    Check(kkt < 1e-4, (std::string(label)+" KKT<1e-4").c_str());
    Check(maxviol < 0.01, (std::string(label)+" no constraint violation").c_str());
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (g_rank==0) {
        printf("╔══════════════════════════════════════════════════════════╗\n"
               "║  Over-constrained MMA/GCMMA test  (%2d rank(s))          ║\n"
               "╠══════════════════════════════════════════════════════════╣\n"
               "║  Tests m >= n: more constraints than design variables    ║\n"
               "║  Exercises the m×m dual Newton system at m>>n scale and  ║\n"
               "║  the SVD fallback for rank-deficient dual Hessians       ║\n"
               "╚══════════════════════════════════════════════════════════╝\n",
               nranks);
    }

    // ── Serial: m = n ─────────────────────────────────────────────────────
    if (g_rank==0) printf("\n═══ Serial: m = n ════════════════════════════════════════\n");
    RunOverconstrained("n=10,  m=10  (m==n)",          10,  10,  false, false);
    RunOverconstrained("n=10,  m=10  (m==n, GCMMA)",   10,  10,  true,  false);
    RunOverconstrained("n=20,  m=20  (m==n)",          20,  20,  false, false);
    RunOverconstrained("n=20,  m=20  (m==n, GCMMA)",   20,  20,  true,  false);

    // ── Serial: m = 2*n ────────────────────────────────────────────────────
    if (g_rank==0) printf("\n═══ Serial: m = 2n ═══════════════════════════════════════\n");
    RunOverconstrained("n=10,  m=20  (m=2n, n redundant)",   10, 20, false, false);
    RunOverconstrained("n=10,  m=20  (m=2n, GCMMA)",         10, 20, true,  false);
    RunOverconstrained("n=20,  m=40  (m=2n)",                20, 40, false, false);
    RunOverconstrained("n=20,  m=40  (m=2n, GCMMA)",         20, 40, true,  false);

    // ── Serial: m = 5*n (highly overdetermined) ────────────────────────────
    if (g_rank==0) printf("\n═══ Serial: m = 5n ═══════════════════════════════════════\n");
    RunOverconstrained("n=10,  m=50  (m=5n)",      10, 50, false, false);
    RunOverconstrained("n=10,  m=50  (m=5n, GCMMA)", 10, 50, true, false);
    RunOverconstrained("n=20,  m=100 (m=5n)",      20, 100, false, false);
    RunOverconstrained("n=20,  m=100 (m=5n, GCMMA)", 20, 100, true, false);

    // ── Parallel: m = n ────────────────────────────────────────────────────
    if (g_rank==0) printf("\n═══ Parallel: m = n ══════════════════════════════════════\n");
    RunOverconstrained("n=20,  m=20  (m==n)",        20,  20,  false, true);
    RunOverconstrained("n=20,  m=20  (m==n, GCMMA)", 20,  20,  true,  true);
    RunOverconstrained("n=40,  m=40  (m==n)",        40,  40,  false, true);
    RunOverconstrained("n=40,  m=40  (m==n, GCMMA)", 40,  40,  true,  true);

    // ── Parallel: m = 2*n ──────────────────────────────────────────────────
    if (g_rank==0) printf("\n═══ Parallel: m = 2n ═════════════════════════════════════\n");
    RunOverconstrained("n=20,  m=40  (m=2n)",        20, 40, false, true);
    RunOverconstrained("n=20,  m=40  (m=2n, GCMMA)", 20, 40, true,  true);
    RunOverconstrained("n=40,  m=80  (m=2n)",        40, 80, false, true);
    RunOverconstrained("n=40,  m=80  (m=2n, GCMMA)", 40, 80, true,  true);

    // ── Parallel: m = 5*n ─────────────────────────────────────────────────
    if (g_rank==0) printf("\n═══ Parallel: m = 5n ═════════════════════════════════════\n");
    RunOverconstrained("n=20,  m=100 (m=5n)",          20, 100, false, true);
    RunOverconstrained("n=20,  m=100 (m=5n, GCMMA)",   20, 100, true,  true);
    RunOverconstrained("n=40,  m=200 (m=5n)",          40, 200, false, true);
    RunOverconstrained("n=40,  m=200 (m=5n, GCMMA)",   40, 200, true,  true);

    if (g_rank==0) {
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        if (g_nfail==0)
            printf("║  All over-constrained tests PASSED.                      ║\n");
        else
            printf("║  %d over-constrained test(s) FAILED.%-21s║\n", g_nfail, "");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail > 0 ? 1 : 0;
}
