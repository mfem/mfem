/**
 * test_sq_equalities.cpp  —  SQOptimizer equality-constraint test suite
 * Same equality problems as test_equalities.cpp using SQOptimizer.  —  Equality-constraint MMA/GCMMA test suite
 *
 * Tests the WithEqualities() factory and the dual solver's handling of
 * free-sign (unconstrained) Lagrange multipliers for equality constraints.
 *
 * Equality convention
 * ───────────────────
 *   SQOptimizer::WithEqualities(n, n_ineq, n_eq, x) creates an optimiser
 *   with m = n_ineq + n_eq constraint slots.  The last n_eq multipliers are
 *   treated as free (unconstrained sign) by the dual IP loop.
 *
 *   fival layout:  [fi_ineq (n_ineq) | h_eq (n_eq)]
 *   dfidx layout:  [dfi_ineq (n_ineq rows) | dh_eq (n_eq rows)]
 *
 *   Use PackFival() and PackedDfidx for convenient packing.
 *
 * Test catalogue
 * ──────────────
 *  1. Pure equality  (n_ineq=0, n_eq=1)
 *     min  (1/n) Σ (xⱼ - aⱼ)²
 *     s.t. mean(x) = Vfrac
 *     Analytic optimum: xⱼ* = aⱼ - mean(a) + Vfrac  (shift so mean = Vfrac)
 *
 *  2. Multiple equalities (n_ineq=0, n_eq=2)
 *     min  (1/n) Σ (xⱼ - aⱼ)²
 *     s.t. mean(x[0..n/2-1]) = V1,  mean(x[n/2..n-1]) = V2
 *     Analytic optimum: independent shift of each half.
 *
 *  3. Mixed: one inequality + one equality
 *     min  (1/n) Σ 1/xⱼ
 *     s.t. max_j(x) ≤ 0.9  (inequality)
 *          mean(x) = Vfrac  (equality)
 *     Tests that multiplier signs are correct simultaneously.
 *
 *  4. GCMMA with equality  (same as Test 1 but UpdateGCMMA)
 *     Verifies that the conservatism inner loop also respects the free-sign
 *     multiplier correctly.
 *
 *  5. Parallel equality  (SQOptimizerParallel::WithEqualities)
 *     Same problem as Test 1 on distributed x, checks MPI consistency.
 *
 *  6. Equality forces feasibility  —  start at infeasible x₀ (mean ≠ Vfrac),
 *     verify the optimiser recovers feasibility and converges.
 *
 * Build:  cmake --build build
 * Run:    ./build/test_equalities
 *         mpirun -np 4 ./build/test_equalities
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
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
// Test 1: pure equality constraint
//
// min  f(x) = (1/n) Σ (xⱼ - aⱼ)²
// s.t. h(x) = mean(x) - Vfrac = 0
//
// Analytic optimum: xⱼ* = aⱼ - mean(a) + Vfrac
// Lagrange multiplier ν* = -2*(ν solves df/dxⱼ + ν/n = 0)
//   → ν* = 2*mean(a) - 2*Vfrac  (may be positive or negative)
// ============================================================
static void Test_PureEquality()
{
    if (g_rank==0) printf("\n── Test 1: pure equality  (n_ineq=0, n_eq=1) ─────────\n");

    const int n=200;
    const double Vfrac=0.4;

    // target profile: aⱼ = 0.3 + 0.4*j/n (mean ≈ 0.5)
    Vector a(n), x(n), xmin(n), xmax(n), df0(n), dh(n);
    for (int j=0;j<n;++j) a(j) = real_t(0.3+0.4*j/n);
    x=real_t(Vfrac);  xmin=real_t(0.01);  xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j) = real_t(1.0/n);

    auto opt = SQOptimizer::WithEqualities(n, /*n_ineq=*/0, /*n_eq=*/1, x);
    Check(opt.NumEqualities()  == 1, "NumEqualities()==1");
    Check(opt.NumInequalities()== 0, "NumInequalities()==0");
    Check(opt.NumConstraints() == 1, "NumConstraints()==1");

    real_t kkt=1.0;
    for (int it=0;it<1000&&kkt>1e-9 && !std::isnan(double(kkt));++it){
        // objective gradient: df/dxⱼ = 2*(xⱼ-aⱼ)/n
        for (int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0=0; for (int j=0;j<n;++j) f0+=double(df0(j))*double(x(j)-a(j))*0.5;

        // equality value: h = mean(x) - Vfrac
        double xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
        Vector h_eq(1); h_eq(0) = real_t(xmean-Vfrac);

        // pack (no inequalities)
        Vector fival = PackFival(Vector(0), h_eq);
        Vector dh_arr[1]={dh};
        PackedDfidx dfidx(nullptr, 0, dh_arr, 1);

        opt.Update(x, df0, real_t(f0), fival, dfidx.data(), xmin, xmax);

        // re-evaluate for KKT
        for (int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
        h_eq(0)=real_t(xmean-Vfrac);
        fival=PackFival(Vector(0),h_eq);
        kkt=opt.KKTresidual(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
    }

    // Check analytic solution: xⱼ* = aⱼ - mean(a) + Vfrac
    double mean_a=0; for (int j=0;j<n;++j) mean_a+=double(a(j)); mean_a/=n;
    double err=0, xmean=0;
    for (int j=0;j<n;++j){
        double xstar=double(a(j))-mean_a+Vfrac;
        err=std::max(err,std::abs(double(x(j))-xstar));
        xmean+=double(x(j));
    }
    xmean/=n;

    if (g_rank==0) printf("  kkt=%.2e  xmean=%.6f(%.2f)  max_err_vs_analytic=%.2e\n",
                          double(kkt), xmean, Vfrac, err);
    Check(kkt < 1e-5,              "converges: KKT < 1e-5");
    Check(std::abs(xmean-Vfrac)<1e-6, "equality satisfied: mean(x)=Vfrac");
    Check(err < 1e-2,              "solution matches analytic optimum");
}

// ============================================================
// Test 2: multiple equality constraints
//
// min  (1/n) Σ (xⱼ - aⱼ)²
// s.t. mean(x[0..n/2-1]) = V1
//      mean(x[n/2..n-1]) = V2
// ============================================================
static void Test_MultipleEqualities()
{
    if (g_rank==0) printf("\n── Test 2: multiple equalities  (n_ineq=0, n_eq=2) ───\n");

    const int n=200, half=n/2;
    const double V1=0.3, V2=0.6;

    Vector a(n), x(n), xmin(n), xmax(n), df0(n);
    for (int j=0;j<n;++j) a(j)=real_t(0.4+0.3*std::sin(3.14*j/n));
    x=real_t(0.45); xmin=real_t(0.01); xmax=real_t(1.0);

    // gradients of h1 and h2
    Vector dh1(n), dh2(n); dh1=0.0; dh2=0.0;
    for (int j=0;j<half;++j)   dh1(j)=real_t(1.0/half);
    for (int j=half;j<n;++j)   dh2(j)=real_t(1.0/half);

    auto opt = SQOptimizer::WithEqualities(n, 0, 2, x);

    real_t kkt=1.0;
    for (int it=0;it<1000&&kkt>1e-9 && !std::isnan(double(kkt));++it){
        for (int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0=0; for (int j=0;j<n;++j) f0+=std::pow(double(x(j))-double(a(j)),2)/n;

        double s1=0,s2=0;
        for (int j=0;j<half;++j)  s1+=double(x(j))/half;
        for (int j=half;j<n;++j)  s2+=double(x(j))/half;

        Vector h_eq(2); h_eq(0)=real_t(s1-V1); h_eq(1)=real_t(s2-V2);
        Vector fival=PackFival(Vector(0),h_eq);
        Vector dh_arr[2]={dh1,dh2};
        PackedDfidx dfidx(nullptr,0,dh_arr,2);

        opt.Update(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);

        for (int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        s1=s2=0;
        for (int j=0;j<half;++j) s1+=double(x(j))/half;
        for (int j=half;j<n;++j) s2+=double(x(j))/half;
        h_eq(0)=real_t(s1-V1); h_eq(1)=real_t(s2-V2);
        fival=PackFival(Vector(0),h_eq);
        kkt=opt.KKTresidual(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
    }

    double s1=0,s2=0;
    for (int j=0;j<half;++j) s1+=double(x(j))/half;
    for (int j=half;j<n;++j) s2+=double(x(j))/half;

    if (g_rank==0) printf("  kkt=%.2e  mean_lo=%.6f(%.2f)  mean_hi=%.6f(%.2f)\n",
                          double(kkt), s1, V1, s2, V2);
    Check(kkt<1e-5,                "converges: KKT < 1e-5");
    Check(std::abs(s1-V1)<1e-5,    "equality 1 satisfied: mean(x[0..n/2])==V1");
    Check(std::abs(s2-V2)<1e-5,    "equality 2 satisfied: mean(x[n/2..n])==V2");
}

// ============================================================
// Test 3: mixed — one inequality + one equality
//
// min  (1/n) Σ 1/xⱼ                  (separable convex)
// s.t. mean(x) ≤ Vfrac + 0.05        (inequality — slack constraint)
//      mean(x) = Vfrac                (equality  — exact target)
//
// The equality makes the inequality redundant at the optimum.
// Multiplier ν for the equality may be positive or negative.
// ============================================================
static void Test_MixedConstraints()
{
    if (g_rank==0) printf("\n── Test 3: mixed (n_ineq=1, n_eq=1) ──────────────────\n");

    // Non-degenerate mixed case:
    //   inequality: mean(x[n/2:]) <= Vfrac+0.05   (right half only)
    //   equality:   mean(x)       == Vfrac         (full domain)
    const int n=200;
    const double Vfrac=0.4;
    const int nhalf=n/2, nright=n-nhalf;

    Vector x(n),xmin(n),xmax(n),df0(n),dg(n),dh(n);
    x=real_t(0.5); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dg(j) = (j>=nhalf) ? real_t(1.0/nright) : real_t(0.0);
    for (int j=0;j<n;++j) dh(j) = real_t(1.0/n);

    auto opt = SQOptimizer::WithEqualities(n, 1, 1, x);
    Check(opt.NumEqualities()  ==1, "mixed: NumEqualities==1");
    Check(opt.NumInequalities()==1, "mixed: NumInequalities==1");
    Check(opt.NumConstraints() ==2, "mixed: NumConstraints==2");

    real_t kkt=1.0;
    for (int it=0;it<1000&&kkt>1e-9&&!std::isnan(double(kkt));++it){
        for (int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double f0=0; for (int j=0;j<n;++j) f0+=1.0/(n*double(x(j)));
        double xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
        double xright=0; for (int j=nhalf;j<n;++j) xright+=double(x(j)); xright/=nright;

        Vector fi_ineq(1); fi_ineq(0)=real_t(xright-(Vfrac+0.05));
        Vector h_eq(1);    h_eq(0)=real_t(xmean-Vfrac);
        Vector fival=PackFival(fi_ineq,h_eq);
        Vector dg_arr[1]={dg}, dh_arr[1]={dh};
        PackedDfidx dfidx(dg_arr,1,dh_arr,1);

        opt.Update(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);

        for (int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
        xright=0; for (int j=nhalf;j<n;++j) xright+=double(x(j)); xright/=nright;
        fi_ineq(0)=real_t(xright-(Vfrac+0.05));
        h_eq(0)=real_t(xmean-Vfrac);
        fival=PackFival(fi_ineq,h_eq);
        kkt=opt.KKTresidual(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
    }

    double xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
    double xright=0; for (int j=nhalf;j<n;++j) xright+=double(x(j)); xright/=nright;
    if (g_rank==0) printf("  kkt=%.2e  xmean=%.6f(%.2f)  xright<=%.2f: %s\n",
                          double(kkt),xmean,Vfrac,Vfrac+0.05,xright<=Vfrac+0.05+1e-4?"OK":"FAIL");
    Check(kkt<1e-5,                   "converges: KKT < 1e-5");
    Check(std::abs(xmean-Vfrac)<1e-4, "equality satisfied: mean(x)==Vfrac");
    Check(xright<=Vfrac+0.05+1e-4,    "inequality satisfied: mean(x[n/2:])<=Vfrac+0.05");
}

// ============================================================
// Test 4: GCMMA with equality constraint
// Same problem as Test 1 using UpdateGCMMA.
// ============================================================
static void Test_GCMMAEquality()
{
    if (g_rank==0) printf("\n── Test 4: GCMMA with equality ────────────────────────\n");

    const int n=200;
    const double Vfrac=0.35;

    Vector a(n),x(n),xmin(n),xmax(n),df0(n),dh(n);
    for (int j=0;j<n;++j) a(j)=real_t(0.2+0.5*j/n);
    x=real_t(Vfrac); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);

    auto opt = SQOptimizer::WithEqualities(n,0,1,x);

    real_t kkt=1.0;
    for (int it=0;it<1000&&kkt>1e-9 && !std::isnan(double(kkt));++it){
        for (int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0=0; for (int j=0;j<n;++j) f0+=std::pow(double(x(j))-double(a(j)),2)/n;
        double xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;

        Vector h_eq(1); h_eq(0)=real_t(xmean-Vfrac);
        Vector fival=PackFival(Vector(0),h_eq);
        Vector dh_arr[1]={dh};
        PackedDfidx dfidx(nullptr,0,dh_arr,1);

        opt.UpdateGCMMA(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);

        for (int j=0;j<n;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
        h_eq(0)=real_t(xmean-Vfrac);
        fival=PackFival(Vector(0),h_eq);
        kkt=opt.KKTresidual(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
    }

    double mean_a=0; for (int j=0;j<n;++j) mean_a+=double(a(j)); mean_a/=n;
    double err=0, xmean=0;
    for (int j=0;j<n;++j){
        err=std::max(err,std::abs(double(x(j))-(double(a(j))-mean_a+Vfrac)));
        xmean+=double(x(j));
    }
    xmean/=n;
    if (g_rank==0) printf("  kkt=%.2e  xmean=%.6f(%.2f)  err_vs_analytic=%.2e\n",
                          double(kkt), xmean, Vfrac, err);
    Check(kkt<1e-5,                   "GCMMA converges: KKT < 1e-5");
    Check(std::abs(xmean-Vfrac)<1e-6, "GCMMA: equality satisfied");
    Check(err<1e-2,                   "GCMMA: matches analytic optimum");
}

// ============================================================
// Test 5: parallel equality (SQOptimizerParallel::WithEqualities)
// ============================================================
static void Test_ParallelEquality()
{
    if (g_rank==0) printf("\n── Test 5: parallel equality ──────────────────────────\n");

    const int n=1000;
    const double Vfrac=0.45;
    MPI_Comm comm=MPI_COMM_WORLD;
    auto [nl,off]=Distribute(n);

    // target: aⱼ = sin(2π j/n)*0.3 + 0.5
    Vector a(nl),x(nl),xmin(nl),xmax(nl),df0(nl),dh(nl);
    for (int j=0;j<nl;++j) a(j)=real_t(0.3*std::sin(2*M_PI*(off+j)/n)+0.5);
    x=real_t(Vfrac); xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<nl;++j) dh(j)=real_t(1.0/n);

    auto opt=SQOptimizerParallel::WithEqualities(comm,nl,0,1,x);
    Check(opt.NumEqualities()==1,  "par: NumEqualities==1");
    Check(opt.NumConstraints()==1, "par: NumConstraints==1");

    real_t kkt=1.0;
    for (int it=0;it<1000&&kkt>1e-9 && !std::isnan(double(kkt));++it){
        for (int j=0;j<nl;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        double f0_loc=0; for (int j=0;j<nl;++j) f0_loc+=std::pow(double(x(j))-double(a(j)),2)/n;
        double xm_loc=0; for (int j=0;j<nl;++j) xm_loc+=double(x(j));
        double xmean=GSum(xm_loc)/n;

        Vector h_eq(1); h_eq(0)=real_t(xmean-Vfrac);
        Vector fival=PackFival(Vector(0),h_eq);
        Vector dh_arr[1]={dh};
        PackedDfidx dfidx(nullptr,0,dh_arr,1);

        opt.Update(x,df0,real_t(GSum(f0_loc)),fival,dfidx.data(),xmin,xmax);

        for (int j=0;j<nl;++j) df0(j)=real_t(2.0*(double(x(j))-double(a(j)))/n);
        f0_loc=0; for (int j=0;j<nl;++j) f0_loc+=std::pow(double(x(j))-double(a(j)),2)/n;
        xm_loc=0; for (int j=0;j<nl;++j) xm_loc+=double(x(j));
        xmean=GSum(xm_loc)/n;
        h_eq(0)=real_t(xmean-Vfrac);
        fival=PackFival(Vector(0),h_eq);
        kkt=opt.KKTresidual(x,df0,real_t(GSum(f0_loc)),fival,dfidx.data(),xmin,xmax);
    }

    double xm_loc=0; for (int j=0;j<nl;++j) xm_loc+=double(x(j));
    double xmean=GSum(xm_loc)/n;

    // analytic: xⱼ* = aⱼ - mean(a) + Vfrac
    double a_loc=0; for (int j=0;j<nl;++j) a_loc+=double(a(j));
    double mean_a=GSum(a_loc)/n;
    double err_loc=0; for (int j=0;j<nl;++j)
        err_loc=std::max(err_loc,std::abs(double(x(j))-(double(a(j))-mean_a+Vfrac)));
    double err=0; MPI_Allreduce(&err_loc,&err,1,MPI_DOUBLE,MPI_MAX,comm);

    if (g_rank==0) printf("  kkt=%.2e  xmean=%.6f(%.2f)  err_vs_analytic=%.2e\n",
                          double(kkt),xmean,Vfrac,err);
    Check(kkt<1e-5,                    "parallel: converges KKT<1e-5");
    Check(std::abs(xmean-Vfrac)<1e-5,  "parallel: equality satisfied");
    Check(err<3e-2,                    "parallel: matches analytic optimum");
}

// ============================================================
// Test 6: equality recovers feasibility from infeasible start
// Start at x₀ = 0.8 (mean = 0.8 ≠ Vfrac = 0.4).
// The first MMA step must move toward feasibility.
// ============================================================
static void Test_FeasibilityRecovery()
{
    if (g_rank==0) printf("\n── Test 6: feasibility recovery from infeasible start ─\n");

    const int n=100;
    const double Vfrac=0.4;

    Vector x(n),xmin(n),xmax(n),df0(n),dh(n);
    x=real_t(0.8);   // infeasible: mean=0.8 ≠ 0.4
    xmin=real_t(0.01); xmax=real_t(1.0);
    for (int j=0;j<n;++j) dh(j)=real_t(1.0/n);

    auto opt=SQOptimizer::WithEqualities(n,0,1,x);

    real_t kkt=1.0;
    for (int it=0;it<1000&&kkt>1e-9 && !std::isnan(double(kkt));++it){
        for (int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double f0=0; for (int j=0;j<n;++j) f0+=1.0/(n*double(x(j)));
        double xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;

        Vector h_eq(1); h_eq(0)=real_t(xmean-Vfrac);
        Vector fival=PackFival(Vector(0),h_eq);
        Vector dh_arr[1]={dh};
        PackedDfidx dfidx(nullptr,0,dh_arr,1);

        opt.Update(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);

        for (int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
        h_eq(0)=real_t(xmean-Vfrac);
        fival=PackFival(Vector(0),h_eq);
        kkt=opt.KKTresidual(x,df0,real_t(f0),fival,dfidx.data(),xmin,xmax);
    }

    double xmean=0; for (int j=0;j<n;++j) xmean+=double(x(j)); xmean/=n;
    if (g_rank==0) printf("  kkt=%.2e  xmean=%.6f(%.2f)\n",double(kkt),xmean,Vfrac);
    Check(kkt<1e-5,                    "feasibility recovery: KKT < 1e-5");
    Check(std::abs(xmean-Vfrac)<1e-5,  "feasibility recovery: equality satisfied");
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD,&nranks);

    if (g_rank==0)
        printf("╔══════════════════════════════════════════════════════════╗\n"
               "║  SQ equality-constraint test suite  (%2d rank(s))           ║\n"
               "╠══════════════════════════════════════════════════════════╣\n"
               "║  Tests WithEqualities(), PackFival(), PackedDfidx        ║\n"
               "║  SQ with equality constraints (SolveDualSQ) [v2]   ║\n"
               "╚══════════════════════════════════════════════════════════╝\n",
               nranks);

    Test_PureEquality();
    Test_MultipleEqualities();
    Test_MixedConstraints();
    Test_GCMMAEquality();
    Test_ParallelEquality();
    Test_FeasibilityRecovery();

    if (g_rank==0){
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        if (g_nfail==0)
            printf("║  All equality-constraint tests PASSED.                   ║\n");
        else
            printf("║  %d equality-constraint test(s) FAILED.%-19s║\n",g_nfail,"");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
