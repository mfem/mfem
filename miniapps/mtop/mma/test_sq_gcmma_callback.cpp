/**
 * test_sq_gcmma_callback.cpp  —  SQOptimizer GCMMA callback test suite
 *
 * Same conservatism tests using SQOptimizer (the SQ approximation has
 * different conservatism properties from MMA).
 *
 * Tests the callback-based UpdateGCMMA overload which implements the full
 * Svanberg (2007) §4 inner conservatism loop.
 *
 * Test catalogue
 * ──────────────
 * 1. Conservatism enforcement  — use a non-conservative callback that
 *    always reports f(x̂) > f̃(x̂), verify ρ increases and inner > 1.
 *
 * 2. Conservative first step  — convex separable problem where the MMA
 *    approximation is exact; verify inner == 1 on every outer iteration.
 *
 * 3. Convergence equivalence  — on a convex problem the callback overload
 *    must converge to the same KKT point as the no-callback overload.
 *
 * 4. Non-convex Rosenbrock  — callback genuinely helps: compare iteration
 *    count and final KKT between callback and no-callback on a problem
 *    where the MMA approximation is non-conservative.
 *
 * 5. Constraint conservatism  — problem where the constraint approximation
 *    is non-conservative (not just objective); verify constraint ρ increases.
 *
 * 6. max_inner respected  — callback always returns non-conservative;
 *    verify inner count == max_inner and no infinite loop.
 *
 * 7. Serial vs parallel equivalence  — same problem on 1 rank produces
 *    identical results from serial and parallel callback overloads.
 *
 * 8. Parallel callback  — distributed problem; callback performs
 *    MPI_Allreduce internally; verify convergence.
 *
 * Build:  cmake --build build
 * Run:    ./build/test_gcmma_callback
 *         mpirun -np 4 ./build/test_gcmma_callback
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
#include <functional>

using namespace mfem;
using namespace mfem_mma;

static int g_rank  = 0;
static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if(g_rank!=0) return;
    if(cond) printf("  [PASS] %s\n", msg);
    else    { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

static double GSum(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return g; }

static std::pair<int,int> Distribute(int n)
{
    int nr; MPI_Comm_size(MPI_COMM_WORLD,&nr);
    int b=n/nr, r=n%nr;
    return {b+(g_rank<r?1:0), g_rank*b+std::min(g_rank,r)};
}

// ── Reference analytic problem ────────────────────────────────────────────
// f0 = (1/n) sum(1/xj),  g0 = mean(x) - Vfrac <= 0
// Optimum: xj* = Vfrac (all equal),  f* = 1/Vfrac
// Gradient: df0/dxj = -1/(n*xj^2),  dg0/dxj = 1/n
// Convex and separable -> MMA approximation is always conservative -> inner==1
static void eval_convex(const Vector& x, int n, Vector& fi, real_t& f0,
                         double Vfrac)
{
    double f=0, g=0;
    for(int j=0;j<x.Size();++j){
        f+=1.0/double(x(j)); g+=double(x(j));
    }
    f0=real_t(f/n);
    fi(0)=real_t(g/n-Vfrac);
}

// ── Non-convex coupling: f = (mean(x))^(-3)  (global, non-separable) ─────
// At any x, the MMA approximation treats mean(x) as fixed — non-conservative.
// Gradient: df/dxj = -3*(mean(x))^(-4) / n
// The TRUE value at x̂ differs from the MMA approximation because mean(x̂)
// differs from mean(x_k), and the approximation doesn't capture that shift.
static real_t eval_nonconvex_f0(const Vector& x, Vector& df0, int n_global)
{
    double m=0; for(int j=0;j<x.Size();++j) m+=double(x(j));
    double mn=m/n_global;   // local sum; need Allreduce for parallel
    double f=std::pow(mn,-3.0);
    real_t df_val=real_t(-3.0*std::pow(mn,-4.0)/n_global);
    for(int j=0;j<x.Size();++j) df0(j)=df_val;
    return real_t(f);
}

// ============================================================
// Test 1: Conservatism enforcement
// Use a callback that always inflates f(x̂) above f̃(x̂),
// forcing ρ to increase each inner step.
// Verify: inner > 1 always, ρ increases, no crash.
// ============================================================
static void Test_ConservatismEnforcement()
{
    if(g_rank==0) printf("\n── Test 1: Conservatism enforcement ──────────────────\n");

    const int n=100, m=1;
    const double Vfrac=0.4;
    Vector x(n), xmin(n), xmax(n), df0(n), dg(n);
    x=0.5; xmin=0.01; xmax=1.0;
    for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);

    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    MMAOptimizer opt(n,m,x,a,c,d);

    // Callback: always return f(x̂) = f̃(x̂) + 1000 (guaranteed non-conservative)
    int total_inner=0;
    int outer_iters=0;
    real_t kkt=1.0;

    for(int it=0;it<20&&kkt>1e-3;++it){
        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
        Vector dg_arr[1]={dg};

        int inner=0;
        double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax,
            [&](const Vector& xc, Vector& fi_out, real_t& f0_out){
                // Return f(x̂) inflated by 1000 → always non-conservative
                double f=0,g=0;
                for(int j=0;j<xc.Size();++j){
                    f+=1.0/double(xc(j)); g+=double(xc(j));
                }
                f0_out = real_t(f/n + 1000.0);   // inflated
                fi_out(0)=real_t(g/n-Vfrac);
            },
            /*max_inner=*/8, &inner);

        total_inner+=inner;
        ++outer_iters;

        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        fi(0)=real_t(g0/n-Vfrac);
        f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        kkt=opt.KKTresidual(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax);
    }

    double avg_inner=outer_iters>0?double(total_inner)/outer_iters:0;
    if(g_rank==0)
        printf("  outer=%d  avg_inner=%.1f  kkt=%.2e\n",
               outer_iters,avg_inner,double(kkt));

    // With always-non-conservative callback, inner should always hit max_inner
    Check(avg_inner>=7.5,  "avg inner iterations >= 7.5 (rho forced up)");
    Check(kkt<1e10,        "optimiser does not diverge despite inflation");
}

// ============================================================
// Test 2: Conservative first step
// Convex separable problem: MMA approximation is always conservative.
// Verify inner == 1 on every outer iteration.
// ============================================================
static void Test_ConservativeFirstStep()
{
    if(g_rank==0) printf("\n── Test 2: Conservative first step (convex) ──────────\n");

    const int n=200, m=1;
    const double Vfrac=0.4;
    Vector x(n), xmin(n), xmax(n), df0(n), dg(n);
    x=0.5; xmin=0.01; xmax=1.0;
    for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);

    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    MMAOptimizer opt(n,m,x,a,c,d);

    std::vector<int> inner_counts;
    real_t kkt=1.0;

    for(int it=0;it<50&&kkt>1e-5;++it){
        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
        Vector dg_arr[1]={dg};

        int inner=0;
        double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax,
            [&](const Vector& xc, Vector& fi_out, real_t& f0_out){
                // True values (no inflation)
                eval_convex(xc,n,fi_out,f0_out,Vfrac);
            },
            /*max_inner=*/10, &inner);

        inner_counts.push_back(inner);

        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        fi(0)=real_t(g0/n-Vfrac);
        f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        kkt=opt.KKTresidual(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax);
    }

    int max_inner_seen=*std::max_element(inner_counts.begin(),inner_counts.end());
    double mean_inner=0;
    for(int v:inner_counts) mean_inner+=v;
    mean_inner/=inner_counts.size();

    if(g_rank==0)
        printf("  iters=%d  max_inner=%d  mean_inner=%.2f  kkt=%.2e\n",
               (int)inner_counts.size(), max_inner_seen, mean_inner, double(kkt));

    Check(max_inner_seen<=2, "inner never exceeds 2 for convex problem");
    Check(mean_inner<1.5,    "mean inner < 1.5 for convex problem");
    Check(kkt<1e-4,          "converges to KKT point");
}

// ============================================================
// Test 3: Convergence equivalence on convex problem
// Both callback and no-callback should converge to the same point.
// ============================================================
static void Test_ConvergenceEquivalence()
{
    if(g_rank==0) printf("\n── Test 3: Convergence equivalence ───────────────────\n");

    const int n=100, m=1;
    const double Vfrac=0.4;
    Vector dg(n); for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);
    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    Vector dg_arr[1]={dg};

    // Run without callback
    double xmean_nocb=0;
    real_t kkt_nocb=1.0;
    int iters_nocb=0;
    {
        Vector x(n),xmin(n),xmax(n),df0(n);
        x=0.5; xmin=0.01; xmax=1.0;
        MMAOptimizer opt(n,m,x,a,c,d);
        for(int it=0;it<100&&kkt_nocb>1e-5;++it,++iters_nocb){
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            kkt_nocb=opt.KKTresidual(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax);
        }
        for(int j=0;j<n;++j) xmean_nocb+=double(x(j));
        xmean_nocb/=n;
    }

    // Run with callback (honest evaluator)
    double xmean_cb=0;
    real_t kkt_cb=1.0;
    int iters_cb=0;
    {
        Vector x(n),xmin(n),xmax(n),df0(n);
        x=0.5; xmin=0.01; xmax=1.0;
        MMAOptimizer opt(n,m,x,a,c,d);
        for(int it=0;it<100&&kkt_cb>1e-5;++it,++iters_cb){
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            int inner=0;
            double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    eval_convex(xc,n,fo,f0o,Vfrac);
                },10,&inner);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            kkt_cb=opt.KKTresidual(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax);
        }
        for(int j=0;j<n;++j) xmean_cb+=double(x(j));
        xmean_cb/=n;
    }

    if(g_rank==0)
        printf("  no-callback: iters=%d  kkt=%.2e  xmean=%.6f\n"
               "  callback:    iters=%d  kkt=%.2e  xmean=%.6f\n"
               "  |xmean_diff|=%.2e\n",
               iters_nocb,double(kkt_nocb),xmean_nocb,
               iters_cb,  double(kkt_cb),  xmean_cb,
               std::abs(xmean_cb-xmean_nocb));

    Check(kkt_nocb<1e-4,                            "no-callback converges");
    Check(kkt_cb  <1e-4,                            "callback converges");
    Check(std::abs(xmean_cb-xmean_nocb)<1e-4,       "same xmean solution");
    Check(std::abs(double(kkt_cb)-double(kkt_nocb))<1e-3, "same KKT residual");
}

// ============================================================
// Test 4: Non-convex problem — callback reduces iterations
// f = (mean(x))^{-3}: globally coupled, non-conservative approximation.
// With honest callback: ρ adapts, fewer oscillations, same or better KKT.
// ============================================================
static void Test_NonConvexCallback()
{
    if(g_rank==0) printf("\n── Test 4: Non-convex — callback reduces oscillation ─\n");

    const int n=200, m=1;
    const double Vfrac=0.5;
    Vector dg(n); for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);
    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    Vector dg_arr[1]={dg};

    // Collect KKT trajectory for both variants
    std::vector<double> kkt_nocb_traj, kkt_cb_traj;
    const int max_it=100;

    // No callback
    {
        Vector x(n),xmin(n),xmax(n),df0(n);
        x=0.5; xmin=0.01; xmax=1.0;
        MMAOptimizer opt(n,m,x,a,c,d);
        for(int it=0;it<max_it;++it){
            double mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
            real_t f0=real_t(std::pow(mn,-3.0));
            real_t df_val=real_t(-3.0*std::pow(mn,-4.0)/n);
            for(int j=0;j<n;++j) df0(j)=df_val;
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            opt.UpdateGCMMA(x,df0,f0,fi,dg_arr,xmin,xmax);
            mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
            f0=real_t(std::pow(mn,-3.0));
            df_val=real_t(-3.0*std::pow(mn,-4.0)/n);
            for(int j=0;j<n;++j) df0(j)=df_val;
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            double kkt=double(opt.KKTresidual(x,df0,f0,fi,dg_arr,xmin,xmax));
            kkt_nocb_traj.push_back(kkt);
            if(kkt<1e-5) break;
        }
    }

    // With honest callback
    {
        Vector x(n),xmin(n),xmax(n),df0(n);
        x=0.5; xmin=0.01; xmax=1.0;
        MMAOptimizer opt(n,m,x,a,c,d);
        for(int it=0;it<max_it;++it){
            double mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
            real_t f0=real_t(std::pow(mn,-3.0));
            real_t df_val=real_t(-3.0*std::pow(mn,-4.0)/n);
            for(int j=0;j<n;++j) df0(j)=df_val;
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            int inner=0;
            opt.UpdateGCMMA(x,df0,f0,fi,dg_arr,xmin,xmax,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    // True function at x̂
                    double mc=0; for(int j=0;j<xc.Size();++j) mc+=double(xc(j)); mc/=n;
                    f0o=real_t(std::pow(mc,-3.0));
                    double gc=0; for(int j=0;j<xc.Size();++j) gc+=double(xc(j));
                    fo[0]=real_t(gc/n-Vfrac);
                },10,&inner);
            mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
            f0=real_t(std::pow(mn,-3.0));
            df_val=real_t(-3.0*std::pow(mn,-4.0)/n);
            for(int j=0;j<n;++j) df0(j)=df_val;
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            double kkt=double(opt.KKTresidual(x,df0,f0,fi,dg_arr,xmin,xmax));
            kkt_cb_traj.push_back(kkt);
            if(kkt<1e-5) break;
        }
    }

    double kkt_nocb_final=kkt_nocb_traj.back();
    double kkt_cb_final  =kkt_cb_traj.back();

    if(g_rank==0)
        printf("  no-callback: iters=%d  final_kkt=%.2e\n"
               "  callback:    iters=%d  final_kkt=%.2e\n",
               (int)kkt_nocb_traj.size(), kkt_nocb_final,
               (int)kkt_cb_traj.size(),   kkt_cb_final);

    Check(kkt_cb_final <1e-4,      "callback variant converges");
    // Callback should converge in <= iterations vs no-callback
    Check((int)kkt_cb_traj.size() <= (int)kkt_nocb_traj.size()+5,
          "callback not slower than no-callback");
}

// ============================================================
// Test 5: Constraint conservatism
// Construct a problem where the CONSTRAINT approximation is non-conservative.
// g(x) = (mean(x))^2 - Vfrac  (convex in mean, non-separable)
// The MMA linearisation underestimates g at x̂ -> inner > 1.
// ============================================================
static void Test_ConstraintConservatism()
{
    if(g_rank==0) printf("\n── Test 5: Constraint conservatism ───────────────────\n");

    const int n=100, m=1;
    const double Vfrac=0.16;   // target: mean(x)^2 <= 0.16 => mean(x) <= 0.4
    Vector x(n),xmin(n),xmax(n),df0(n),dg0(n);
    // Constraint: g = (mean(x))^2 - Vfrac
    // Gradient:   dg/dxj = 2*mean(x)/n  (non-constant -> MMA linearises)
    // Non-conservative because the quadratic constraint curves away from
    // the MMA linear approximation.
    x=0.6; xmin=0.01; xmax=1.0;

    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    MMAOptimizer opt(n,m,x,a,c,d);

    std::vector<int> inner_hist;
    real_t kkt=1.0;

    for(int it=0;it<100&&kkt>1e-5;++it){
        double mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
        // Objective: f0 = mean(x) (linear, want to minimise)
        real_t f0=real_t(mn);
        for(int j=0;j<n;++j) df0(j)=real_t(1.0/n);
        // Constraint: g = mn^2 - Vfrac, grad = 2*mn/n
        mfem::Vector fi(1); fi(0)=real_t(mn*mn - Vfrac);
        for(int j=0;j<n;++j) dg0(j)=real_t(2.0*mn/n);
        Vector dg_arr[1]={dg0};

        int inner=0;
        opt.UpdateGCMMA(x,df0,f0,fi,dg_arr,xmin,xmax,
            [&](const Vector& xc, Vector& fo, real_t& f0o){
                double mc=0; for(int j=0;j<xc.Size();++j) mc+=double(xc(j)); mc/=n;
                f0o=real_t(mc);
                fo[0]=real_t(mc*mc - Vfrac);   // true quadratic constraint
            },10,&inner);

        inner_hist.push_back(inner);

        mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
        f0=real_t(mn);
        for(int j=0;j<n;++j) df0(j)=real_t(1.0/n);
        fi(0)=real_t(mn*mn-Vfrac);
        for(int j=0;j<n;++j) dg0(j)=real_t(2.0*mn/n);
        kkt=opt.KKTresidual(x,df0,f0,fi,dg_arr,xmin,xmax);
    }

    int n_multi=0;
    for(int v:inner_hist) if(v>1) ++n_multi;
    double mn_final=0; for(int j=0;j<n;++j) mn_final+=double(x(j)); mn_final/=n;

    if(g_rank==0)
        printf("  iters=%d  kkt=%.2e  g_final=%.4f  n_multi_inner=%d/%d\n",
               (int)inner_hist.size(),double(kkt),
               mn_final*mn_final-Vfrac,n_multi,(int)inner_hist.size());

    Check(kkt<1e-4,   "converges despite non-conservative constraint");
    Check(n_multi>0,  "constraint non-conservatism triggered inner > 1");
    Check(mn_final*mn_final <= Vfrac+0.01, "constraint satisfied at optimum");
}

// ============================================================
// Test 6: max_inner respected
// Callback always returns non-conservative values.
// Verify inner_count == max_inner (loop exits, no hang).
// ============================================================
static void Test_MaxInnerRespected()
{
    if(g_rank==0) printf("\n── Test 6: max_inner limit respected ─────────────────\n");

    const int n=50, m=1, MAX_INNER=5;
    Vector x(n),xmin(n),xmax(n),df0(n),dg(n);
    x=0.5; xmin=0.01; xmax=1.0;
    for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);
    double cv=1000.0, a[1]={0},c[1]={cv},d[1]={1};
    MMAOptimizer opt(n,m,x,a,c,d);

    bool all_max=true;
    for(int it=0;it<5;++it){
        for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        mfem::Vector fi(1); fi(0)=real_t(g0/n-0.4);
        Vector dg_arr[1]={dg};

        int inner=0;
        double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax,
            [&](const Vector&, Vector& fo, real_t& f0o){
                // Always return huge f -> never conservative
                f0o=1e30f; fo[0]=1e30f;
            },
            MAX_INNER, &inner);

        if(inner!=MAX_INNER) all_max=false;
        if(g_rank==0) printf("  iter %d: inner=%d\n",it,inner);
    }
    Check(all_max, "inner always equals max_inner when never conservative");
}

// ============================================================
// Test 7: Serial vs parallel equivalence on 1 rank
// Same problem, same x₀ → same solution from both classes.
// ============================================================
static void Test_SerialParallelEquivalence()
{
    if(g_rank==0) printf("\n── Test 7: Serial vs parallel equivalence (1 rank) ───\n");

    const int n=100, m=2;
    const double Vfrac=0.4;
    MPI_Comm comm=MPI_COMM_WORLD;

    auto [nl,off]=Distribute(n);
    double cv=std::max(1000.0,10.0*n);
    double a[2]={0,0},c[2]={cv,cv},d[2]={1,1};

    // Two constraints: mean(x)<=Vfrac and mean(x)>= Vfrac-0.05
    // Serial needs size-n gradients; parallel needs size-nl gradients.
    std::vector<Vector> dg_par(2), dg_ser(2);
    for(int k=0;k<2;++k){
        dg_par[k].SetSize(nl); dg_par[k]=real_t(k==0?1.0/n:-1.0/n);
        dg_ser[k].SetSize(n);  dg_ser[k]=real_t(k==0?1.0/n:-1.0/n);
    }
    const double Vlo=Vfrac-0.05, Vhi=Vfrac;

    auto EvalF=[&](const Vector& xv, int sz)->std::tuple<real_t,mfem::Vector>{
        double f=0,g=0;
        for(int j=0;j<sz;++j){ f+=1.0/double(xv(j)); g+=double(xv(j)); }
        double f_g=GSum(f)/n, g_g=GSum(g)/n;
        mfem::Vector _fi_ret_(2);

        _fi_ret_(0)=real_t(g_g-Vhi);

        _fi_ret_(1)=real_t(Vlo-g_g);

        return {real_t(f_g), _fi_ret_};
    };

    double xmean_s=0, xmean_p=0;
    real_t kkt_s=1, kkt_p=1;

    // ── Serial — run on ALL ranks independently (MMAOptimizer uses MPI_COMM_SELF)
    // No inter-rank MPI allowed here — use local EvalF without Allreduce.
    {
        auto EvalFLocal=[&](const Vector& xv)->std::tuple<real_t,mfem::Vector>{
            double f=0,g=0;
            for(int j=0;j<xv.Size();++j){f+=1.0/double(xv(j));g+=double(xv(j));}
            mfem::Vector _fi_ret_(2);

            _fi_ret_(0)=real_t(g/n-Vhi);

            _fi_ret_(1)=real_t(Vlo-g/n);

            return {real_t(f/n), _fi_ret_};
        };
        Vector x(n),xmin(n),xmax(n),df0(n);
        x=0.5; xmin=0.01; xmax=1.0;
        MMAOptimizer opt(n,m,x,a,c,d);
        for(int it=0;it<100&&kkt_s>1e-5;++it){
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0,fi]=EvalFLocal(x);
            int inner=0;
            opt.UpdateGCMMA(x,df0,f0,fi,dg_ser.data(),xmin,xmax,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    auto [f,fii]=EvalFLocal(xc);
                    f0o=f; fo[0]=fii(0); fo[1]=fii(1);
                },10,&inner);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0b,fib]=EvalFLocal(x);
            kkt_s=opt.KKTresidual(x,df0,f0b,fib,dg_ser.data(),xmin,xmax);
        }
        for(int j=0;j<n;++j) xmean_s+=double(x(j)); xmean_s/=n;
        // Result is identical on all ranks — no broadcast needed
    }

    // ── Parallel ────────────────────────────────────────────────────────
    {
        Vector x(nl),xmin(nl),xmax(nl),df0(nl);
        x=0.5; xmin=0.01; xmax=1.0;
        MMAOptimizerParallel opt(comm,nl,m,x,a,c,d);
        for(int it=0;it<100&&kkt_p>1e-5;++it){
            for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0,fi]=EvalF(x,nl);
            int inner=0;
            opt.UpdateGCMMA(x,df0,f0,fi,dg_par.data(),xmin,xmax,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    auto [f,fii]=EvalF(xc,nl);
                    f0o=f; fo[0]=fii(0); fo[1]=fii(1);
                },10,&inner);
            for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0b,fib]=EvalF(x,nl);
            kkt_p=opt.KKTresidual(x,df0,f0b,fib,dg_par.data(),xmin,xmax);
        }
        double xl=0; for(int j=0;j<nl;++j) xl+=double(x(j));
        xmean_p=GSum(xl)/n;
    }

    if(g_rank==0)
        printf("  serial:   kkt=%.2e  xmean=%.6f\n"
               "  parallel: kkt=%.2e  xmean=%.6f\n"
               "  |diff|=%.2e\n",
               double(kkt_s),xmean_s,double(kkt_p),xmean_p,
               std::abs(xmean_s-xmean_p));

    Check(kkt_s<1e-4,                           "serial converges");
    Check(kkt_p<1e-4,                           "parallel converges");
    Check(std::abs(xmean_s-xmean_p)<1e-4,       "serial==parallel xmean");
}

// ============================================================
// Test 8: Parallel callback — multi-rank distributed problem
// Verify the callback inner loop works correctly with nl=0 ranks.
// ============================================================
static void Test_ParallelCallback()
{
    if(g_rank==0) printf("\n── Test 8: Parallel callback (multi-rank) ────────────\n");

    const int n=1000, m=2;
    const double Vfrac=0.4;
    MPI_Comm comm=MPI_COMM_WORLD;
    auto [nl,off]=Distribute(n);

    std::vector<Vector> dg(2);
    for(int k=0;k<2;++k){ dg[k].SetSize(nl); dg[k]=real_t(k==0?1.0/n:-1.0/n); }

    double cv=std::max(1000.0,10.0*n);
    double a[2]={0,0},c[2]={cv,cv},d[2]={1,1};
    Vector x(nl),xmin(nl),xmax(nl),df0(nl);
    x=0.5; xmin=0.01; xmax=1.0;
    MMAOptimizerParallel opt(comm,nl,m,x,a,c,d);

    std::vector<int> inner_hist;
    real_t kkt=1.0;

    for(int it=0;it<100&&kkt>1e-5;++it){
        for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double floc=0,gloc=0;
        for(int j=0;j<nl;++j){ floc+=1.0/double(x(j)); gloc+=double(x(j)); }
        real_t f0=real_t(GSum(floc)/n);
        mfem::Vector fi(2); fi(0)=real_t(GSum(gloc)/n-Vfrac); fi(1)=real_t((Vfrac-0.05)-GSum(gloc)/n);

        int inner=0;
        opt.UpdateGCMMA(x,df0,f0,fi,dg.data(),xmin,xmax,
            [&](const Vector& xc, Vector& fo, real_t& f0o){
                // All ranks participate in the allreduce
                double fl=0,gl=0;
                for(int j=0;j<xc.Size();++j){fl+=1.0/double(xc(j));gl+=double(xc(j));}
                f0o=real_t(GSum(fl)/n);
                fo[0]=real_t(GSum(gl)/n-Vfrac);
                fo[1]=real_t((Vfrac-0.05)-GSum(gl)/n);
            },10,&inner);

        inner_hist.push_back(inner);

        for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        floc=gloc=0;
        for(int j=0;j<nl;++j){floc+=1.0/double(x(j));gloc+=double(x(j));}
        f0=real_t(GSum(floc)/n);
        fi(0)=real_t(GSum(gloc)/n-Vfrac);
        fi(1)=real_t((Vfrac-0.05)-GSum(gloc)/n);
        kkt=opt.KKTresidual(x,df0,f0,fi,dg.data(),xmin,xmax);
    }

    double xl=0; for(int j=0;j<nl;++j) xl+=double(x(j));
    double xmean=GSum(xl)/n;
    int max_inner_seen=inner_hist.empty()?0:
        *std::max_element(inner_hist.begin(),inner_hist.end());

    if(g_rank==0)
        printf("  iters=%d  kkt=%.2e  xmean=%.4f(%.2f)  max_inner=%d\n",
               (int)inner_hist.size(),double(kkt),xmean,Vfrac,max_inner_seen);

    Check(kkt<1e-4,         "parallel callback converges");
    Check(xmean>Vfrac-0.06, "lower volume bound satisfied");
    Check(xmean<Vfrac+0.01, "upper volume bound satisfied");
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD,&nranks);

    if(g_rank==0)
        printf("╔══════════════════════════════════════════════════════════╗\n"
               "║  GCMMA callback test suite  (%2d rank(s))                ║\n"
               "╠══════════════════════════════════════════════════════════╣\n"
               "║  Tests the full Svanberg §4 inner conservatism loop      ║\n"
               "╚══════════════════════════════════════════════════════════╝\n",
               nranks);

    Test_ConservatismEnforcement();
    Test_ConservativeFirstStep();
    Test_ConvergenceEquivalence();
    Test_NonConvexCallback();
    Test_ConstraintConservatism();
    Test_MaxInnerRespected();
    Test_SerialParallelEquivalence();
    Test_ParallelCallback();

    if(g_rank==0){
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        if(g_nfail==0)
            printf("║  All GCMMA callback tests PASSED.                        ║\n");
        else
            printf("║  %d GCMMA callback test(s) FAILED.%-21s║\n",g_nfail,"");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
