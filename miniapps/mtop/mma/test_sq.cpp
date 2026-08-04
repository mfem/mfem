/**
 * test_sq.cpp  —  SQOptimizer / SQOptimizerParallel test suite
 *
 * Tests the separable quadratic (SQ) approximation optimiser on the same
 * canonical problems used for MMA, verifying the shared core interface and
 * comparing convergence on convex problems.
 *
 * Test catalogue
 * ──────────────
 *  1. Unconstrained quadratic  (n=100)
 *     min (1/n) Σ (xj - aj)²    → xj* = aj  (no bounds active)
 *
 *  2. Constrained: min (1/n) Σ 1/xj   s.t. mean(x) ≤ Vfrac
 *     Same as test_mma_serial P1 but using SQOptimizer.
 *
 *  3. SQOptimizer with equality  (WithEqualities factory)
 *     min (1/n) Σ (xj - aj)²   s.t. mean(x) = Vfrac
 *
 *  4. UpdateGCMMA API smoke test
 *     Unconstrained quadratic using the GCMMA convenience overload.
 *
 *  5. Parallel (SQOptimizerParallel)
 *     Same as Test 2 distributed across MPI ranks.
 *
 *  6. SQ vs MMA convergence comparison on convex problem
 *     Both should converge; SQ should need fewer iters on well-scaled problems.
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace mfem;
using namespace mfem_mma;

static int g_rank = 0;
static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

static double GSum(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return g; }

static double GMax(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); return g; }

static std::pair<int,int> Distribute(int n)
{
    int nr; MPI_Comm_size(MPI_COMM_WORLD,&nr);
    int b=n/nr, r=n%nr;
    return {b+(g_rank<r?1:0), g_rank*b+std::min(g_rank,r)};
}

// ── Test 1: unconstrained quadratic ──────────────────────────────────────
static void Test_Unconstrained()
{
    if (g_rank==0) printf("\n── Test 1: unconstrained quadratic  (SQOptimizer) ───\n");
    const int n = 100;
    Vector a(n), x(n), xmin(n), xmax(n), df0(n);
    for (int j=0;j<n;++j) a(j) = real_t(0.1 + 0.8*j/(n-1));
    x = real_t(0.5); xmin = real_t(0.05); xmax = real_t(0.95);

    SQOptimizer opt(n, 0);

    real_t kkt = 1.0;
    for (int it=0; it<200 && kkt>1e-9 && !std::isnan(double(kkt)); ++it) {
        for (int j=0;j<n;++j) df0(j) = real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0 = 0; for (int j=0;j<n;++j) f0 += std::pow(double(x(j))-double(a(j)),2)/n;
        opt.Update(x, df0, real_t(f0), xmin, xmax);
        f0=0;
        for (int j=0;j<n;++j) {
            double r=double(x(j))-double(a(j));
            df0(j)=real_t(2.0*r/n); f0+=r*r/n;
        }
        kkt = opt.KKTresidual(x, df0, real_t(f0), xmin, xmax);
    }

    double err = 0;
    for (int j=0;j<n;++j) err = std::max(err, std::abs(double(x(j))-double(a(j))));
    if (g_rank==0) printf("  kkt=%.2e  max_err_vs_analytic=%.2e  iters=%d\n",
                          double(kkt), err, opt.NumIterations());
    Check(kkt < 1e-8,  "converges: KKT < 1e-8");
    Check(err < 1e-3,  "matches analytic optimum");
}

// ── Test 2: inequality-constrained ───────────────────────────────────────
static void Test_Constrained()
{
    if (g_rank==0) printf("\n── Test 2: inequality constrained  (SQOptimizer) ────\n");
    const int n = 200;
    const double Vfrac = 0.4;
    Vector x(n), xmin(n), xmax(n), df0(n), dg(n);
    x = real_t(0.5); xmin = real_t(0.01); xmax = real_t(1.0);
    for (int j=0;j<n;++j) dg(j) = real_t(1.0/n);

    SQOptimizer opt(n, 1);

    real_t kkt = 1.0;
    for (int it=0; it<500 && kkt>1e-8 && !std::isnan(double(kkt)); ++it) {
        for (int j=0;j<n;++j) df0(j) = real_t(-1.0/(n*double(x(j))*double(x(j))));
        double f0 = 0; for (int j=0;j<n;++j) f0 += 1.0/(n*double(x(j)));
        double xm = 0; for (int j=0;j<n;++j) xm += double(x(j)); xm /= n;
        Vector fi(1); fi(0) = real_t(xm - Vfrac);
        Vector dg_arr[1] = {dg};
        opt.Update(x, df0, real_t(f0), fi, dg_arr, xmin, xmax);
        f0=0;
        for (int j=0;j<n;++j) {
            double xj=double(x(j));df0(j)=real_t(-1.0/(n*xj*xj));f0+=1.0/(n*xj);
        }
        xm = 0; for (int j=0;j<n;++j) xm += double(x(j)); xm /= n;
        fi(0) = real_t(xm - Vfrac);
        kkt = opt.KKTresidual(x, df0, real_t(f0), fi, dg_arr, xmin, xmax);
    }

    double xm = 0; for (int j=0;j<n;++j) xm += double(x(j)); xm /= n;
    if (g_rank==0) printf("  kkt=%.2e  xmean=%.6f(<=%.2f)  iters=%d\n",
                          double(kkt), xm, Vfrac, opt.NumIterations());
    Check(kkt < 1e-7,           "converges: KKT < 1e-7");
    Check(xm <= Vfrac + 1e-4,   "inequality constraint satisfied");
}

// ── Test 3: equality constraint ───────────────────────────────────────────
static void Test_Equality()
{
    if (g_rank==0) printf("\n── Test 3: equality constraint  (SQOptimizer) ───────\n");
    const int n = 200;
    const double Vfrac = 0.4;
    Vector a(n), x(n), xmin(n), xmax(n), df0(n), dh(n);
    for (int j=0;j<n;++j) a(j) = real_t(0.3 + 0.4*j/n);
    x = real_t(Vfrac); xmin = real_t(0.01); xmax = real_t(1.0);
    for (int j=0;j<n;++j) dh(j) = real_t(1.0/n);

    auto opt = SQOptimizer::WithEqualities(n, 0, 1);
    Check(opt.NumEqualities()  == 1, "SQ NumEqualities==1");
    Check(opt.NumConstraints() == 1, "SQ NumConstraints==1");

    real_t kkt = 1.0;
    for (int it=0; it<1000 && kkt>1e-9 && !std::isnan(double(kkt)); ++it) {
        for (int j=0;j<n;++j) df0(j) = real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0 = 0; for (int j=0;j<n;++j) f0 += std::pow(double(x(j))-double(a(j)),2)/n;
        double xm = 0; for (int j=0;j<n;++j) xm += double(x(j)); xm /= n;
        Vector h_eq(1); h_eq(0) = real_t(xm - Vfrac);
        Vector fival = PackFival(Vector(0), h_eq);
        Vector dh_arr[1] = {dh};
        PackedDfidx dfidx(nullptr, 0, dh_arr, 1);
        opt.Update(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
        f0=0;
        for (int j=0;j<n;++j) {
            double r=double(x(j))-double(a(j));
            df0(j)=real_t(2.0*r/n);f0+=r*r/n;
        }
        xm = 0; for (int j=0;j<n;++j) xm += double(x(j)); xm /= n;
        h_eq(0) = real_t(xm - Vfrac);
        fival = PackFival(Vector(0), h_eq);
        kkt = opt.KKTresidual(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);
    }

    double mean_a = 0; for (int j=0;j<n;++j) mean_a += double(a(j)); mean_a /= n;
    double err = 0, xm = 0;
    for (int j=0;j<n;++j) {
        err = std::max(err, std::abs(double(x(j)) - (double(a(j))-mean_a+Vfrac)));
        xm += double(x(j));
    }
    xm /= n;
    if (g_rank==0) printf("  kkt=%.2e  xmean=%.6f(%.2f)  err=%.2e  iters=%d\n",
                          double(kkt),xm,Vfrac,err,opt.NumIterations());
    Check(kkt < 1e-8,                "SQ equality: converges");
    Check(std::abs(xm-Vfrac)<1e-5,   "SQ equality: constraint satisfied");
    Check(err < 1e-2,                "SQ equality: matches analytic optimum");
}

// ── Test 4: UpdateGCMMA convenience-API smoke test ────────────────────────
static void Test_GCMMA()
{
    if (g_rank==0) printf("\n── Test 4: UpdateGCMMA API (unconstrained SQ) ────────\n");
    const int n = 100;
    Vector a(n), x(n), xmin(n), xmax(n), df0(n);
    for (int j=0;j<n;++j) a(j) = real_t(0.2 + 0.6*j/(n-1));
    x = real_t(0.5); xmin = real_t(0.05); xmax = real_t(0.95);

    SQOptimizer opt(n, 0);
    real_t kkt = 1.0;
    for (int it=0; it<200 && kkt>1e-9 && !std::isnan(double(kkt)); ++it) {
        for (int j=0;j<n;++j) df0(j) = real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0 = 0; for (int j=0;j<n;++j) f0 += std::pow(double(x(j))-double(a(j)),2)/n;
        opt.UpdateGCMMA(x, df0, real_t(f0), xmin, xmax);
        f0=0;
        for (int j=0;j<n;++j) {
            double r=double(x(j))-double(a(j));
            df0(j)=real_t(2.0*r/n);f0+=r*r/n;
        }
        kkt = opt.KKTresidual(x, df0, real_t(f0), xmin, xmax);
    }
    double err = 0;
    for (int j=0;j<n;++j) err = std::max(err, std::abs(double(x(j))-double(a(j))));
    if (g_rank==0) printf("  kkt=%.2e  err=%.2e  iters=%d\n",
                          double(kkt),err,opt.NumIterations());
    Check(kkt < 1e-8, "SQ GCMMA converges");
    Check(err < 1e-3, "SQ GCMMA matches analytic");
}

// ── Test 5: parallel ──────────────────────────────────────────────────────
// Tests SQOptimizerParallel with a large-n inequality-constrained problem:
//   min  (1/n) Σ (xj - aj)²    s.t.  mean(x) ≤ Vfrac
// Analytic optimum: xj* = clip(aj - mean(a) + Vfrac, xmin, xmax)
// Uses n=10000 so nl=2500 per rank (well-conditioned, matches real usage).
static void Test_Parallel()
{
    if (g_rank==0) printf("\n── Test 5: parallel (SQOptimizerParallel) ───────────\n");
    const int n = 10000;
    const double Vfrac = 0.4;
    MPI_Comm comm = MPI_COMM_WORLD;
    auto [nl, off] = Distribute(n);

    Vector x(nl), xmin(nl), xmax(nl), df0(nl), dg(nl), a_vec(nl);
    x = real_t(0.5); xmin = real_t(0.01); xmax = real_t(1.0);
    // Random target values (deterministic LCG, same sequence as test_sq_unconstrained)
    uint64_t s = 98765ULL;
    for (int g = 0; g < off; ++g) { s = s*6364136223846793005ULL + 1442695040888963407ULL; }
    for (int j = 0; j < nl; ++j) {
        s = s*6364136223846793005ULL + 1442695040888963407ULL;
        a_vec(j) = real_t(0.2 + 0.6*(double)(s >> 33) / (double)(1ULL << 31));
    }
    for (int j=0;j<nl;++j) dg(j) = real_t(1.0/n);

    double am_loc=0;for(int j=0;j<nl;++j)am_loc+=double(a_vec(j));
    double mean_a=GSum(am_loc)/n;

    SQOptimizerParallel opt(comm, nl, 1);

    real_t kkt = 1.0;
    double maxerr=1.0;
    for(int it=0;it<200&&(kkt>1e-7||maxerr>1e-3)&&
        !std::isnan(double(kkt));++it) {
        double f0_loc=0;
        for (int j=0;j<nl;++j) {
            double r=double(x(j))-double(a_vec(j));
            df0(j)=real_t(2.0*r/n); f0_loc+=r*r/n;
        }
        double f0=GSum(f0_loc);
        double xm_loc=0; for (int j=0;j<nl;++j) xm_loc+=double(x(j));
        double xm=GSum(xm_loc)/n;
        Vector fi(1); fi(0)=real_t(xm-Vfrac);
        Vector dg_arr[1]={dg};
        opt.Update(x,df0,real_t(f0),fi,dg_arr,xmin,xmax);
        // Recompute df0 and fi at updated x for correct KKT
        f0_loc=0;
        for (int j=0;j<nl;++j) {
            double r=double(x(j))-double(a_vec(j));
            df0(j)=real_t(2.0*r/n); f0_loc+=r*r/n;
        }
        f0=GSum(f0_loc);
        xm_loc=0; for (int j=0;j<nl;++j) xm_loc+=double(x(j));
        xm=GSum(xm_loc)/n;
        fi(0)=real_t(xm-Vfrac);
        kkt=opt.KKTresidual(x,df0,real_t(f0),fi,dg_arr,xmin,xmax);
        double err_loc=0;
        for(int j=0;j<nl;++j){
            double expected=std::max(double(xmin(j)),std::min(double(xmax(j)),
                                          double(a_vec(j))-mean_a+Vfrac));
            err_loc=std::max(err_loc,std::abs(double(x(j))-expected));
        }
        maxerr=GMax(err_loc);
    }

    double xm_loc=0; for (int j=0;j<nl;++j) xm_loc+=double(x(j));
    double xm=GSum(xm_loc)/n;
    if (g_rank==0)
        printf("  kkt=%.2e  xmean=%.6f(%.2f)  max_err=%.2e  iters=%d\n",
               double(kkt),xm,Vfrac,maxerr,opt.NumIterations());
    Check(kkt < 1e-5,       "SQ parallel converges");
    Check(std::abs(xm-Vfrac)<1e-4,"SQ parallel active constraint satisfied");
    Check(maxerr<1e-3,      "SQ parallel matches analytic optimum");
}

// ── Test 6: SQ vs MMA comparison ─────────────────────────────────────────
static void Test_Comparison()
{
    if (g_rank==0) printf("\n── Test 6: SQ vs MMA convergence comparison ─────────\n");
    const int n = 200;
    Vector xmin(n), xmax(n), df0(n), dg(n);
    xmin = real_t(0.01); xmax = real_t(0.99);
    for (int j=0;j<n;++j) dg(j) = real_t(1.0/n);
    const double Vfrac = 0.4;

    struct RunResult { int iters; double kkt; double xmean; };
    auto runOpt = [&](auto& opt, const char* name) -> RunResult {
        Vector x(n); x = real_t(0.5);
        real_t kkt = 1.0;
        int it = 0;
        for (; it<500 && kkt>1e-8 && !std::isnan(double(kkt)); ++it) {
            for (int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            double f0=0; for(int j=0;j<n;++j) f0+=1.0/(n*double(x(j)));
            double xm=0; for(int j=0;j<n;++j) xm+=double(x(j)); xm/=n;
            Vector fi(1); fi(0)=real_t(xm-Vfrac); Vector dg_arr[1]={dg};
            opt.Update(x, df0, real_t(f0), fi, dg_arr, xmin, xmax);
            f0=0;
            for(int j=0;j<n;++j){
                double xj=double(x(j));df0(j)=real_t(-1.0/(n*xj*xj));f0+=1.0/(n*xj);
            }
            xm=0; for(int j=0;j<n;++j) xm+=double(x(j)); xm/=n;
            fi(0)=real_t(xm-Vfrac);
            kkt=opt.KKTresidual(x,df0,real_t(f0),fi,dg_arr,xmin,xmax);
        }
        double xm=0;for(int j=0;j<n;++j)xm+=double(x(j));xm/=n;
        if (g_rank==0) printf("  %-8s  iters=%3d  kkt=%.2e  xmean=%.6f\n",
                              name,it,double(kkt),xm);
        return {it,double(kkt),xm};
    };

    Vector x0(n); x0 = real_t(0.5);
    MMAOptimizer mma(n, 1);
    SQOptimizer  sq (n, 1);
    RunResult mma_result=runOpt(mma,"MMA");
    RunResult sq_result =runOpt(sq,"SQ");

    Check(mma_result.iters<500&&mma_result.kkt<1e-7,"MMA converges within 500 iters");
    Check(sq_result.iters <500&&sq_result.kkt <1e-7,"SQ converges within 500 iters");
    Check(std::abs(mma_result.xmean-Vfrac)<1e-4,"MMA comparison solution feasible");
    Check(std::abs(sq_result.xmean-Vfrac)<1e-4,"SQ comparison solution feasible");
    // Both should converge; no strict ordering required (problem dependent)
    if(g_rank==0)printf("  SQ/MMA iter ratio: %.2f\n",
                        double(sq_result.iters)/mma_result.iters);
}

// ── main ──────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (g_rank==0)
        printf("╔══════════════════════════════════════════════════════════╗\n"
               "║  SQ (separable quadratic) optimiser test suite          ║\n"
               "║  (%2d rank(s))  —  Svanberg 2007 §5.1 approximation      ║\n"
               "╠══════════════════════════════════════════════════════════╣\n"
               "║  σ_j = 0.5*(xmax_j-xmin_j)  (default scaling)           ║\n"
               "║  Core API aligned with MMA serial and parallel classes   ║\n"
               "╚══════════════════════════════════════════════════════════╝\n",
               nranks);

    Test_Unconstrained();
    Test_Constrained();
    Test_Equality();
    Test_GCMMA();
    Test_Parallel();
    Test_Comparison();

    if (g_rank==0) {
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        if (g_nfail==0)
            printf("║  All SQ tests PASSED.                                    ║\n");
        else
            printf("║  %d SQ test(s) FAILED.%-37s║\n", g_nfail, "");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail > 0 ? 1 : 0;
}
