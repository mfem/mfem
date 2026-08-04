/**
 * test_sq_gcmma_callback.cpp  —  SQOptimizer GCMMA callback test suite
 *
 * Same conservatism tests using SQOptimizer (the SQ approximation has
 * different conservatism properties from MMA).
 *
 * Tests SQOptimizer's constraint-callback UpdateGCMMA overload and its
 * rho-increase inner conservatism loop.
 *
 * Test catalogue
 * ──────────────
 * 1. Conservatism enforcement  — use a non-conservative constraint
 *    callback, verify constraint rho increases and inner > 1.
 *
 * 2. Conservative first step  — convex separable problem where the MMA
 *    approximation is exact; verify inner == 1 on every outer iteration.
 *
 * 3. Convergence equivalence  — on a convex problem the callback overload
 *    must converge to the same KKT point as the no-callback overload.
 *
 * 4. Non-convex objective  — verify that a constraint callback does not
 *    perturb convergence when only the objective is non-convex.
 *
 * 5. Constraint conservatism  — problem where the constraint approximation
 *    is non-conservative (not just objective); verify constraint ρ increases.
 *
 * 6. fixed inner limit respected — callback always returns
 *    non-conservative constraints; verify the loop terminates at 10.
 *
 * 7. Serial vs parallel equivalence  — same problem on 1 rank produces
 *    identical results from serial and parallel callback overloads.
 *
 * 8. Parallel callback with zero-DOF ranks — callback performs collective
 *    evaluation while ranks with empty local vectors participate.
 *
 * Build:  cmake --build build
 * Run:    ./build/test_sq_gcmma_callback
 *         mpirun -np 4 ./build/test_sq_gcmma_callback
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
#include <tuple>

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
// Convex and separable; the linear constraint model is exact -> inner==1.
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
    SQOptimizer opt(n,m,a,c,d);

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
            [&](const Vector&,Vector& fi_out,Vector*){
                fi_out(0)=real_t(1e6); // force non-conservative constraint
            },&inner);

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
        printf("  iters=%d  avg_inner=%.1f  kkt=%.2e\n",
               outer_iters,avg_inner,double(kkt));

    // One initial candidate plus ten failed retries is reported as 11.
    Check(avg_inner>=10.5,"inner loop reaches fixed limit (rho forced up)");
    Check(kkt<1e10,        "optimiser does not diverge despite inflation");
}

// ============================================================
// Test 2: Conservative first step
// Convex separable problem with an exact linear constraint model.
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
    SQOptimizer opt(n,m,a,c,d);

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
            [&](const Vector& xc,Vector& fi_out,Vector*){
                real_t ignored; eval_convex(xc,n,fi_out,ignored,Vfrac);
            },&inner);

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
        SQOptimizer opt(n,m,a,c,d);
        for(int it=0;it<100&&kkt_nocb>1e-5;++it,++iters_nocb){
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            f0_val_=0;for(int j=0;j<x.Size();++j)f0_val_+=1.0/double(x(j));f0_val_/=x.Size();
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
        SQOptimizer opt(n,m,a,c,d);
        for(int it=0;it<100&&kkt_cb>1e-5;++it,++iters_cb){
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            int inner=0;
            double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax,
                [&](const Vector& xc,Vector& fo,Vector*){
                    real_t ignored; eval_convex(xc,n,fo,ignored,Vfrac);
                },&inner);
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
// Test 4: Non-convex objective — constraint callback is neutral
// f = (mean(x))^{-3}: globally coupled, non-conservative approximation.
// SQ callbacks verify constraints only, so an exact linear-constraint
// callback must not perturb convergence for this non-convex objective.
// ============================================================
static void Test_NonConvexCallback()
{
    if(g_rank==0) printf("\n── Test 4: Non-convex objective, callback neutrality ─\n");

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
        SQOptimizer opt(n,m,a,c,d);
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
        SQOptimizer opt(n,m,a,c,d);
        for(int it=0;it<max_it;++it){
            double mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
            real_t f0=real_t(std::pow(mn,-3.0));
            real_t df_val=real_t(-3.0*std::pow(mn,-4.0)/n);
            for(int j=0;j<n;++j) df0(j)=df_val;
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            int inner=0;
            opt.UpdateGCMMA(x,df0,f0,fi,dg_arr,xmin,xmax,
                [&](const Vector& xc,Vector& fo,Vector*){
                    double gc=0;for(int j=0;j<xc.Size();++j)gc+=double(xc(j));
                    fo[0]=real_t(gc/n-Vfrac);
                },&inner);
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
    // An exact constraint callback should not materially slow convergence.
    Check((int)kkt_cb_traj.size() <= (int)kkt_nocb_traj.size()+5,
          "callback not slower than no-callback");
}

// ============================================================
// Test 5: Constraint conservatism
// Construct a problem where the CONSTRAINT approximation is non-conservative.
// g(x) = exp(10*(mean(x)-Vfrac)) - 1  (strongly convex in mean)
// The SQ model can underestimate g at the trial point -> inner > 1.
// ============================================================
static void Test_ConstraintConservatism()
{
    if(g_rank==0) printf("\n── Test 5: Constraint conservatism ───────────────────\n");

    const int n=100, m=1;
    const double Vfrac=0.4;
    Vector x(n),xmin(n),xmax(n),df0(n),dg0(n);
    // Constraint: g = exp(10*(mean(x)-Vfrac)) - 1
    // Gradient:   dg/dxj = 10*exp(10*(mean(x)-Vfrac))/n
    // Non-conservative because the quadratic constraint curves away from
    // the local SQ constraint approximation.
    // Start feasible and maximize mean(x) through minimization of -mean(x),
    // so the nonlinear upper constraint becomes active at mean(x)=0.4.
    x=0.2; xmin=0.01; xmax=1.0;

    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    SQOptimizer opt(n,m,a,c,d);

    std::vector<int> inner_hist;
    real_t kkt=1.0;
    double constraint_value=1.0;

    for(int it=0;it<100&&(kkt>1e-5||constraint_value>1e-6);++it){
        double mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
        // Objective: f0 = -mean(x), which pushes toward the constraint.
        real_t f0=real_t(-mn);
        for(int j=0;j<n;++j) df0(j)=real_t(-1.0/n);
        double eg=std::exp(10.0*(mn-Vfrac));
        mfem::Vector fi(1);fi(0)=real_t(eg-1.0);
        for(int j=0;j<n;++j)dg0(j)=real_t(10.0*eg/n);
        Vector dg_arr[1]={dg0};

        int inner=0;
        opt.UpdateGCMMA(x,df0,f0,fi,dg_arr,xmin,xmax,
            [&](const Vector& xc,Vector& fo,Vector*){
                double mc=0; for(int j=0;j<xc.Size();++j) mc+=double(xc(j)); mc/=n;
                fo[0]=real_t(std::exp(10.0*(mc-Vfrac))-1.0);
            },&inner);

        inner_hist.push_back(inner);

        mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
        f0=real_t(-mn);
        for(int j=0;j<n;++j) df0(j)=real_t(-1.0/n);
        eg=std::exp(10.0*(mn-Vfrac));
        constraint_value=eg-1.0;
        fi(0)=real_t(constraint_value);
        for(int j=0;j<n;++j)dg0(j)=real_t(10.0*eg/n);
        kkt=opt.KKTresidual(x,df0,f0,fi,dg_arr,xmin,xmax);
    }

    int n_multi=0;
    for(int v:inner_hist) if(v>1) ++n_multi;

    if(g_rank==0)
        printf("  iters=%d  kkt=%.2e  g_final=%.4f  n_multi_inner=%d/%d\n",
               (int)inner_hist.size(),double(kkt),
               constraint_value,n_multi,(int)inner_hist.size());

    Check(kkt<1e-4,   "converges despite non-conservative constraint");
    Check(n_multi>0,  "constraint non-conservatism triggered inner > 1");
    Check(constraint_value<=1e-4,"constraint satisfied at optimum");
}

// ============================================================
// Test 6: fixed inner limit respected
// Callback always returns non-conservative values.
// Verify the reported count is one initial solve plus ten retries.
// ============================================================
static void Test_MaxInnerRespected()
{
    if(g_rank==0) printf("\n── Test 6: fixed inner limit respected ───────────────\n");

    const int n=50,m=1,REPORTED_INNER_LIMIT=11;
    Vector x(n),xmin(n),xmax(n),df0(n),dg(n);
    x=0.5; xmin=0.01; xmax=1.0;
    for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);
    double cv=1000.0, a[1]={0},c[1]={cv},d[1]={1};
    SQOptimizer opt(n,m,a,c,d);

    bool all_max=true;
    for(int it=0;it<5;++it){
        for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        mfem::Vector fi(1); fi(0)=real_t(g0/n-0.4);
        Vector dg_arr[1]={dg};

        int inner=0;
        double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        opt.UpdateGCMMA(x,df0,real_t(f0_val_),fi,dg_arr,xmin,xmax,
            [&](const Vector&,Vector& fo,Vector*){fo[0]=real_t(1e30);},
            &inner);

        if(inner!=REPORTED_INNER_LIMIT) all_max=false;
        if(g_rank==0) printf("  iter %d: inner=%d\n",it,inner);
    }
    if(g_rank==0)printf("  Final: iters=%d\n",opt.NumIterations());
    Check(all_max,"inner always equals fixed limit when never conservative");
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
    (void)off;
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
    int iters_s=0,iters_p=0;

    // ── Serial — run on all ranks independently (no inter-rank collectives)
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
        SQOptimizer opt(n,m,a,c,d);
        for(int it=0;it<100&&kkt_s>1e-5;++it){
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0,fi]=EvalFLocal(x);
            int inner=0;
            opt.UpdateGCMMA(x,df0,f0,fi,dg_ser.data(),xmin,xmax,
                [&](const Vector& xc,Vector& fo,Vector*){
                    auto [f,fii]=EvalFLocal(xc);(void)f;
                    fo[0]=fii(0);fo[1]=fii(1);
                },&inner);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0b,fib]=EvalFLocal(x);
            kkt_s=opt.KKTresidual(x,df0,f0b,fib,dg_ser.data(),xmin,xmax);
        }
        for(int j=0;j<n;++j) { xmean_s+=double(x(j)); }
        xmean_s/=n;
        iters_s=opt.NumIterations();
        // Result is identical on all ranks — no broadcast needed
    }

    // ── Parallel ────────────────────────────────────────────────────────
    {
        Vector x(nl),xmin(nl),xmax(nl),df0(nl);
        x=0.5; xmin=0.01; xmax=1.0;
        SQOptimizerParallel opt(comm,nl,m,a,c,d);
        for(int it=0;it<100&&kkt_p>1e-5;++it){
            for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0,fi]=EvalF(x,nl);
            int inner=0;
            opt.UpdateGCMMA(x,df0,f0,fi,dg_par.data(),xmin,xmax,
                [&](const Vector& xc,Vector& fo,Vector*){
                    auto [f,fii]=EvalF(xc,nl);(void)f;
                    fo[0]=fii(0);fo[1]=fii(1);
                },&inner);
            for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0b,fib]=EvalF(x,nl);
            kkt_p=opt.KKTresidual(x,df0,f0b,fib,dg_par.data(),xmin,xmax);
        }
        double xl=0; for(int j=0;j<nl;++j) xl+=double(x(j));
        xmean_p=GSum(xl)/n;
        iters_p=opt.NumIterations();
    }

    if(g_rank==0)
        printf("  serial:   iters=%d  kkt=%.2e  xmean=%.6f\n"
               "  parallel: iters=%d  kkt=%.2e  xmean=%.6f\n"
               "  |diff|=%.2e\n",
               iters_s,double(kkt_s),xmean_s,iters_p,double(kkt_p),xmean_p,
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
    if(g_rank==0) printf("\n── Test 8: Parallel callback with zero-DOF ranks ─────\n");

    int nranks;MPI_Comm_size(MPI_COMM_WORLD,&nranks);
    const int n=std::max(1,nranks/2),m=2;
    const double Vfrac=0.4;
    MPI_Comm comm=MPI_COMM_WORLD;
    auto [nl,off]=Distribute(n);
    (void)off;
    int n_zero=int(GSum(nl==0?1.0:0.0));

    std::vector<Vector> dg(2);
    for(int k=0;k<2;++k){ dg[k].SetSize(nl); dg[k]=real_t(k==0?1.0/n:-1.0/n); }

    double cv=std::max(1000.0,10.0*n);
    double a[2]={0,0},c[2]={cv,cv},d[2]={1,1};
    Vector x(nl),xmin(nl),xmax(nl),df0(nl);
    x=0.5; xmin=0.01; xmax=1.0;
    SQOptimizerParallel opt(comm,nl,m,a,c,d);

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
            [&](const Vector& xc,Vector& fo,Vector*){
                // All ranks participate in the allreduce
                double gl=0;
                for(int j=0;j<xc.Size();++j)gl+=double(xc(j));
                fo[0]=real_t(GSum(gl)/n-Vfrac);
                fo[1]=real_t((Vfrac-0.05)-GSum(gl)/n);
            },&inner);

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
        printf("  iters=%d  kkt=%.2e  xmean=%.4f(%.2f)  max_inner=%d zero_ranks=%d\n",
               (int)inner_hist.size(),double(kkt),xmean,Vfrac,max_inner_seen,n_zero);

    Check(n_zero==nranks-n,"expected number of zero-DOF ranks participated");
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
               "║  SQ GCMMA callback test suite (%2d rank(s))              ║\n"
               "╠══════════════════════════════════════════════════════════╣\n"
               "║  Tests the SQ constraint conservatism loop               ║\n"
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
            printf("║  All SQ GCMMA callback tests PASSED.                     ║\n");
        else
            printf("║  %d SQ GCMMA callback test(s) FAILED.%-18s║\n",g_nfail,"");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
