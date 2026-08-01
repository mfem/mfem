/**
 * test_ccsa_gcmma_callback.cpp  —  GCMMA callback (full inner loop) test
 *                                   suite (CCSA)
 *
 * Tests the callback-based CCSAOptimizer::UpdateGCMMA() overload, which
 * implements the same full inner conservatism loop as
 * MMAOptimizer::UpdateGCMMA() (Svanberg 2007 §4 for MMA; [Note] Sec. 5.1 /
 * eq. 36 for the entropy-CCSA rho-increase rule CCSA uses instead). Problem
 * catalogue and pass/fail thresholds are carried over unchanged from
 * test_mma_gcmma_callback.cpp for direct comparability; since CCSA's rho
 * dynamics, while structurally analogous (same increase-on-violation rule,
 * same a/c/d z-variable mechanism), are not numerically identical to MMA's,
 * these thresholds have deliberately been kept loose in the original suite
 * and should still hold, but have not been verified against a real MFEM
 * build here.
 *
 * CCSAOptimizer/CCSAOptimizerParallel work directly on the LATENT variable
 * eta: UpdateGCMMA()/KKTresidual() take/return eta, not the physical
 * design, and take no xmin/xmax. The EvalCallback contract is UNCHANGED --
 * it is still invoked with the PHYSICAL trial point, since that is what a
 * real objective/constraint evaluator needs -- only the outer x argument
 * of UpdateGCMMA() itself is now latent. Every test below builds a
 * BoundsGeometry up front, constructs the optimiser with it, seeds
 * eta = opt.ToLatent(x0), and converts back via opt.ToPhysical(eta)
 * whenever it needs the physical point to evaluate df0/dx or fi outside
 * the callback.
 *
 * Test catalogue
 * ──────────────
 * 1. Conservatism enforcement — use an inflated callback to force rho
 *    retries, and verify retry counts remain within max_inner.
 *
 * 2. Convex callback  — verify convergence and bounded inner retries on a
 *    separable problem without assuming the entropy model is exact.
 *
 * 3. Convergence equivalence  — on a convex problem the callback overload
 *    must converge to the same KKT point as the no-callback overload.
 *
 * 4. Genuinely non-convex objective — verify callback convergence, rho
 *    retries, and a final KKT no worse than the no-callback overload.
 *
 * 5. Constraint conservatism  — problem where the constraint approximation
 *    is non-conservative (not just objective); verify constraint ρ increases.
 *
 * 6. max_inner respected  — callback always returns non-conservative;
 *    verify inner count == max_inner and no infinite loop.
 *
 * 7. Serial vs parallel-class equivalence — on MPI_COMM_SELF, the same
 *    problem produces identical results from the two optimizer classes.
 *
 * 8. Parallel callback — distributed problem; callback performs
 *    MPI_Allreduce internally; verify convergence.
 *
 * 9. Zero-local-DOF ranks — n=2 distributed across all ranks, so a normal
 *    four-rank run exercises empty local vectors and collective callbacks.
 *
 * Build:  cmake --build build   (links MMA_MFEM.cpp + CCSA_Bregman_MFEM.cpp)
 * Run:    ./build/test_ccsa_gcmma_callback
 *         mpirun -np 4 ./build/test_ccsa_gcmma_callback
 */

#include "CCSA_Bregman_MFEM.hpp"
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
// Convex and separable -> the CCSA entropy approximation is always
// conservative -> inner==1 (same as MMA's rational approximation here)
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
    Vector xmin(n), xmax(n), df0(n), dg(n);
    xmin=0.01; xmax=1.0;
    for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);

    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizer opt(n,m,bounds,a,c,d);
    Vector x0(n); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(n);

    // Callback: always return f(x̂) = f̃(x̂) + 1000 (guaranteed non-conservative)
    int total_inner=0;
    int outer_iters=0;
    int max_inner_seen=0;
    int multi_inner_updates=0;
    real_t kkt=1.0;

    for(int it=0;it<20&&kkt>1e-3;++it){
        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
        Vector dg_arr[1]={dg};

        int inner=0;
        double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        opt.UpdateGCMMA(eta,df0,real_t(f0_val_),fi,dg_arr,
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
        max_inner_seen=std::max(max_inner_seen,inner);
        if(inner>1) ++multi_inner_updates;
        ++outer_iters;

        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        fi(0)=real_t(g0/n-Vfrac);
        f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        kkt=opt.KKTresidual(eta,df0,real_t(f0_val_),fi,dg_arr);
    }

    double avg_inner=outer_iters>0?double(total_inner)/outer_iters:0;
    if(g_rank==0)
        printf("  outer=%d  avg_inner=%.1f  kkt=%.2e\n",
               outer_iters,avg_inner,double(kkt));

    Check(multi_inner_updates>0, "inflated callback triggers rho retries");
    Check(max_inner_seen<=8,     "inner count respects max_inner");
    Check(kkt<1e10,        "optimiser does not diverge despite inflation");
}

// ============================================================
// Test 2: Conservative first step
// Convex separable problem with an honest callback.  The entropy model is
// not exact for 1/x, so retries are allowed; verify bounded retries and
// convergence rather than assuming every update is conservative immediately.
// ============================================================
static void Test_ConservativeFirstStep()
{
    if(g_rank==0) printf("\n── Test 2: Conservative first step (convex) ──────────\n");

    const int n=200, m=1;
    const double Vfrac=0.4;
    Vector xmin(n), xmax(n), df0(n), dg(n);
    xmin=0.01; xmax=1.0;
    for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);

    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizer opt(n,m,bounds,a,c,d);
    Vector x0(n); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(n);

    std::vector<int> inner_counts;
    real_t kkt=1.0;

    for(int it=0;it<50&&kkt>1e-5;++it){
        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
        Vector dg_arr[1]={dg};

        int inner=0;
        double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        opt.UpdateGCMMA(eta,df0,real_t(f0_val_),fi,dg_arr,
            [&](const Vector& xc, Vector& fi_out, real_t& f0_out){
                // True values (no inflation)
                eval_convex(xc,n,fi_out,f0_out,Vfrac);
            },
            /*max_inner=*/10, &inner);

        inner_counts.push_back(inner);

        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j)
            df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        fi(0)=real_t(g0/n-Vfrac);
        f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        kkt=opt.KKTresidual(eta,df0,real_t(f0_val_),fi,dg_arr);
    }

    int max_inner_seen=*std::max_element(inner_counts.begin(),inner_counts.end());
    double mean_inner=0;
    for(int v:inner_counts) mean_inner+=v;
    mean_inner/=inner_counts.size();

    if(g_rank==0)
        printf("  iters=%d  max_inner=%d  mean_inner=%.2f  kkt=%.2e\n",
               (int)inner_counts.size(), max_inner_seen, mean_inner, double(kkt));

    Check(!inner_counts.empty(), "performed at least one update");
    Check(max_inner_seen<=10, "inner count respects max_inner");
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
    Vector xmin(n),xmax(n); xmin=0.01; xmax=1.0;
    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    Vector dg_arr[1]={dg};
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);

    // Run without callback
    double xmean_nocb=0;
    real_t kkt_nocb=1.0;
    int iters_nocb=0;
    {
        CCSAOptimizer opt(n,m,bounds,a,c,d);
        Vector x0(n); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(n), df0(n);
        for(int it=0;it<100&&kkt_nocb>1e-5;++it,++iters_nocb){
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            opt.UpdateGCMMA(eta,df0,real_t(f0_val_),fi,dg_arr);
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            kkt_nocb=opt.KKTresidual(eta,df0,real_t(f0_val_),fi,dg_arr);
        }
        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j) xmean_nocb+=double(x(j));
        xmean_nocb/=n;
    }

    // Run with callback (honest evaluator)
    double xmean_cb=0;
    real_t kkt_cb=1.0;
    int iters_cb=0;
    {
        CCSAOptimizer opt(n,m,bounds,a,c,d);
        Vector x0(n); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(n), df0(n);
        for(int it=0;it<100&&kkt_cb>1e-5;++it,++iters_cb){
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            int inner=0;
            double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            opt.UpdateGCMMA(eta,df0,real_t(f0_val_),fi,dg_arr,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    eval_convex(xc,n,fo,f0o,Vfrac);
                },10,&inner);
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
            kkt_cb=opt.KKTresidual(eta,df0,real_t(f0_val_),fi,dg_arr);
        }
        x = opt.ToPhysical(eta);
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
// Test 4: genuinely non-convex objective
// f = mean((x-0.7)^2 + 0.05*sin(12*x)); its second derivative
// 2-7.2*sin(12*x) changes sign on the design interval.
// ============================================================
static void Test_NonConvexCallback()
{
    if(g_rank==0) printf("\n── Test 4: Non-convex callback convergence ──────────\n");

    const int n=200, m=1;
    const double Vfrac=0.8;
    Vector dg(n); for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);
    Vector xmin(n),xmax(n); xmin=0.01; xmax=1.0;
    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    Vector dg_arr[1]={dg};
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);

    std::vector<double> kkt_nocb_traj, kkt_cb_traj;
    int cb_multi_inner=0;
    const int max_it=100;

    // No callback
    {
        CCSAOptimizer opt(n,m,bounds,a,c,d);
        Vector x0(n); x0=0.3;
        Vector eta = opt.ToLatent(x0);
        Vector x(n), df0(n);
        for(int it=0;it<max_it;++it){
            x = opt.ToPhysical(eta);
            double fv=0;
            for(int j=0;j<n;++j){double xj=double(x(j));fv+=(xj-0.7)*(xj-0.7)+0.05*std::sin(12*xj);df0(j)=real_t((2*(xj-0.7)+0.6*std::cos(12*xj))/n);}
            real_t f0=real_t(fv/n);
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            opt.UpdateGCMMA(eta,df0,f0,fi,dg_arr);
            x = opt.ToPhysical(eta);
            fv=0; for(int j=0;j<n;++j){double xj=double(x(j));fv+=(xj-0.7)*(xj-0.7)+0.05*std::sin(12*xj);df0(j)=real_t((2*(xj-0.7)+0.6*std::cos(12*xj))/n);}
            f0=real_t(fv/n);
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            double kkt=double(opt.KKTresidual(eta,df0,f0,fi,dg_arr));
            kkt_nocb_traj.push_back(kkt);
            if(kkt<1e-5) break;
        }
    }

    // With honest callback
    {
        CCSAOptimizer opt(n,m,bounds,a,c,d);
        Vector x0(n); x0=0.3;
        Vector eta = opt.ToLatent(x0);
        Vector x(n), df0(n);
        for(int it=0;it<max_it;++it){
            x = opt.ToPhysical(eta);
            double fv=0;
            for(int j=0;j<n;++j){double xj=double(x(j));fv+=(xj-0.7)*(xj-0.7)+0.05*std::sin(12*xj);df0(j)=real_t((2*(xj-0.7)+0.6*std::cos(12*xj))/n);}
            real_t f0=real_t(fv/n);
            double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            mfem::Vector fi(1); fi(0)=real_t(g0/n-Vfrac);
            int inner=0;
            opt.UpdateGCMMA(eta,df0,f0,fi,dg_arr,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    double fc=0; for(int j=0;j<xc.Size();++j){double xj=double(xc(j));fc+=(xj-0.7)*(xj-0.7)+0.05*std::sin(12*xj);}
                    f0o=real_t(fc/n);
                    double gc=0; for(int j=0;j<xc.Size();++j) gc+=double(xc(j));
                    fo[0]=real_t(gc/n-Vfrac);
                },10,&inner);
            if(inner>1) ++cb_multi_inner;
            x = opt.ToPhysical(eta);
            fv=0; for(int j=0;j<n;++j){double xj=double(x(j));fv+=(xj-0.7)*(xj-0.7)+0.05*std::sin(12*xj);df0(j)=real_t((2*(xj-0.7)+0.6*std::cos(12*xj))/n);}
            f0=real_t(fv/n);
            g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
            fi(0)=real_t(g0/n-Vfrac);
            double kkt=double(opt.KKTresidual(eta,df0,f0,fi,dg_arr));
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
    Check(cb_multi_inner>0,         "non-convex objective triggers rho retries");
    Check(std::isfinite(kkt_nocb_final) && kkt_cb_final<=kkt_nocb_final+1e-3,
          "callback final KKT is no worse than no-callback");
}

// ============================================================
// Test 5: Constraint conservatism
// Construct a problem where the CONSTRAINT approximation is non-conservative.
// g(x) = (mean(x))^2 - Vfrac  (convex in mean, non-separable)
// The CCSA linearisation-plus-curvature model underestimates g at x̂ ->
// inner > 1, same as it does for MMA's linear/rational model.
// ============================================================
static void Test_ConstraintConservatism()
{
    if(g_rank==0) printf("\n── Test 5: Constraint conservatism ───────────────────\n");

    const int n=100, m=1;
    const double Vfrac=0.16;   // target: mean(x)^2 <= 0.16 => mean(x) <= 0.4
    // Constraint: g = (mean(x))^2 - Vfrac
    // Gradient:   dg/dxj = 2*mean(x)/n  (non-constant -> linearised)
    // Non-conservative because the quadratic constraint curves away from
    // the separable model's linear term.
    Vector xmin(n),xmax(n); xmin=0.01; xmax=1.0;

    double cv=std::max(1000.0,10.0*n);
    double a[1]={0},c[1]={cv},d[1]={1};
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizer opt(n,m,bounds,a,c,d);
    // Start feasible and maximize mean(x), so trial steps move upward toward
    // the curved active boundary mean(x)=0.4.  The previous objective
    // minimized mean(x) from an infeasible point and moved away from the
    // constraint, so no constraint-conservatism retry was expected.
    Vector x0(n); x0=0.2;
    Vector eta = opt.ToLatent(x0);
    Vector x(n), df0(n), dg0(n);

    std::vector<int> inner_hist;
    real_t kkt=1.0;

    for(int it=0;it<100&&kkt>1e-5;++it){
        x = opt.ToPhysical(eta);
        double mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
        // Objective: f0 = -mean(x), i.e. maximize mean(x) toward the
        // quadratic constraint boundary.
        real_t f0=real_t(-mn);
        for(int j=0;j<n;++j) df0(j)=real_t(-1.0/n);
        // Constraint: g = mn^2 - Vfrac, grad = 2*mn/n
        mfem::Vector fi(1); fi(0)=real_t(mn*mn - Vfrac);
        for(int j=0;j<n;++j) dg0(j)=real_t(2.0*mn/n);
        Vector dg_arr[1]={dg0};

        int inner=0;
        opt.UpdateGCMMA(eta,df0,f0,fi,dg_arr,
            [&](const Vector& xc, Vector& fo, real_t& f0o){
                double mc=0; for(int j=0;j<xc.Size();++j) mc+=double(xc(j)); mc/=n;
                f0o=real_t(-mc);
                fo[0]=real_t(mc*mc - Vfrac);   // true quadratic constraint
            },10,&inner);

        inner_hist.push_back(inner);

        x = opt.ToPhysical(eta);
        mn=0; for(int j=0;j<n;++j) mn+=double(x(j)); mn/=n;
        f0=real_t(-mn);
        for(int j=0;j<n;++j) df0(j)=real_t(-1.0/n);
        fi(0)=real_t(mn*mn-Vfrac);
        for(int j=0;j<n;++j) dg0(j)=real_t(2.0*mn/n);
        kkt=opt.KKTresidual(eta,df0,f0,fi,dg_arr);
    }

    int n_multi=0;
    for(int v:inner_hist) if(v>1) ++n_multi;
    x = opt.ToPhysical(eta);
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
    Vector xmin(n),xmax(n),df0(n),dg(n);
    xmin=0.01; xmax=1.0;
    for(int j=0;j<n;++j) dg(j)=real_t(1.0/n);
    double cv=1000.0, a[1]={0},c[1]={cv},d[1]={1};
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizer opt(n,m,bounds,a,c,d);
    Vector x0(n); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(n);

    bool all_max=true;
    for(int it=0;it<5;++it){
        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        double g0=0; for(int j=0;j<n;++j) g0+=double(x(j));
        mfem::Vector fi(1); fi(0)=real_t(g0/n-0.4);
        Vector dg_arr[1]={dg};

        int inner=0;
        double f0_val_=0; for(int j=0;j<x.Size();++j) f0_val_+=1.0/double(x(j)); f0_val_/=x.Size();
        opt.UpdateGCMMA(eta,df0,real_t(f0_val_),fi,dg_arr,
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
// Test 7: Serial vs one-rank-parallel equivalence
// MPI_COMM_SELF isolates class differences from distributed reduction-order
// and adaptive-rho trajectory differences, which are covered by Test 8.
// ============================================================
static void Test_SerialParallelEquivalence()
{
    if(g_rank==0) printf("\n── Test 7: Serial vs parallel class (MPI_COMM_SELF) ──\n");

    const int n=100, m=2;
    const double Vfrac=0.4;
    double cv=std::max(1000.0,10.0*n);
    double a[2]={0,0},c[2]={cv,cv},d[2]={1,1};

    // Two constraints: mean(x)<=Vfrac and mean(x)>= Vfrac-0.05
    const double Vlo=Vfrac-0.05, Vhi=Vfrac;

    auto EvalFLocal=[&](const Vector& xv)->std::tuple<real_t,mfem::Vector>{
        double f=0,g=0;
        for(int j=0;j<xv.Size();++j){ f+=1.0/double(xv(j)); g+=double(xv(j)); }
        mfem::Vector _fi_ret_(2);
        _fi_ret_(0)=real_t(g/n-Vhi);
        _fi_ret_(1)=real_t(Vlo-g/n);
        return {real_t(f/n), _fi_ret_};
    };

    double xmean_s=0, xmean_p=0;
    real_t kkt_s=1, kkt_p=1;

    // ── Serial — run on ALL ranks independently (CCSAOptimizer uses a
    // serial-sentinel comm internally, not MPI_COMM_WORLD).
    // No inter-rank MPI allowed here — use local EvalF without Allreduce.
    {
        Vector xmin(n),xmax(n),df0(n);
        xmin=0.01; xmax=1.0;
        Vector dg_ser[2];
        for(int k=0;k<2;++k){ dg_ser[k].SetSize(n); dg_ser[k]=real_t(k==0?1.0/n:-1.0/n); }
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizer opt(n,m,bounds,a,c,d);
        Vector x0(n); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(n);
        for(int it=0;it<100&&kkt_s>1e-5;++it){
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0,fi]=EvalFLocal(x);
            int inner=0;
            opt.UpdateGCMMA(eta,df0,f0,fi,dg_ser,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    auto [f,fii]=EvalFLocal(xc);
                    f0o=f; fo[0]=fii(0); fo[1]=fii(1);
                },10,&inner);
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0b,fib]=EvalFLocal(x);
            kkt_s=opt.KKTresidual(eta,df0,f0b,fib,dg_ser);
        }
        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j) xmean_s+=double(x(j)); xmean_s/=n;
        // Result is identical on all ranks — no broadcast needed
    }

    // ── Parallel class on MPI_COMM_SELF ────────────────────────────────
    {
        Vector xmin(n),xmax(n),df0(n);
        xmin=0.01; xmax=1.0;
        Vector dg_par[2];
        for(int k=0;k<2;++k){ dg_par[k].SetSize(n); dg_par[k]=real_t(k==0?1.0/n:-1.0/n); }
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizerParallel opt(MPI_COMM_SELF,n,m,bounds,a,c,d);
        Vector x0(n); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(n);
        for(int it=0;it<100&&kkt_p>1e-5;++it){
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0,fi]=EvalFLocal(x);
            int inner=0;
            opt.UpdateGCMMA(eta,df0,f0,fi,dg_par,
                [&](const Vector& xc, Vector& fo, real_t& f0o){
                    auto [f,fii]=EvalFLocal(xc);
                    f0o=f; fo[0]=fii(0); fo[1]=fii(1);
                },10,&inner);
            x = opt.ToPhysical(eta);
            for(int j=0;j<n;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
            auto [f0b,fib]=EvalFLocal(x);
            kkt_p=opt.KKTresidual(eta,df0,f0b,fib,dg_par);
        }
        x = opt.ToPhysical(eta);
        for(int j=0;j<n;++j) xmean_p+=double(x(j));
        xmean_p/=n;
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
// Verify the distributed callback and its collectives; Test 9 separately
// forces nl=0 ranks under an ordinary four-rank launch.
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
    Vector xmin(nl),xmax(nl),df0(nl);
    xmin=0.01; xmax=1.0;
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizerParallel opt(comm,nl,m,bounds,a,c,d);
    Vector x0(nl); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(nl);

    auto EvalParallel=[&](const Vector& xv, Vector& fi_out, real_t& f0_out){
        double floc=0, xloc=0;
        for(int j=0;j<xv.Size();++j){floc+=1.0/double(xv(j));xloc+=double(xv(j));}
        const double f=GSum(floc)/n;
        const double mean=GSum(xloc)/n;
        f0_out=real_t(f);
        fi_out(0)=real_t(mean-Vfrac);
        fi_out(1)=real_t((Vfrac-0.05)-mean);
    };

    std::vector<int> inner_hist;
    real_t kkt=1.0;

    for(int it=0;it<100&&kkt>1e-5;++it){
        x = opt.ToPhysical(eta);
        for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        real_t f0;
        mfem::Vector fi(2);
        EvalParallel(x,fi,f0);

        int inner=0;
        opt.UpdateGCMMA(eta,df0,f0,fi,dg.data(),
            [&](const Vector& xc, Vector& fo, real_t& f0o){
                EvalParallel(xc,fo,f0o);
            },10,&inner);

        inner_hist.push_back(inner);

        x = opt.ToPhysical(eta);
        for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(n*double(x(j))*double(x(j))));
        EvalParallel(x,fi,f0);
        kkt=opt.KKTresidual(eta,df0,f0,fi,dg.data());
    }

    x = opt.ToPhysical(eta);
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
// Test 9: callback with zero-local-DOF ranks
// With n=2, any run using more than two ranks necessarily has ranks whose
// local vectors are empty; all ranks must still enter every collective.
// ============================================================
static void Test_ZeroDofCallback()
{
    if(g_rank==0) printf("\n── Test 9: Callback with zero-DOF ranks ──────────────\n");

    const int n=2, m=1;
    const double Vfrac=0.4;
    auto [nl,off]=Distribute(n); (void)off;
    Vector xmin(nl),xmax(nl),df0(nl),dg(nl);
    xmin=0.01; xmax=1.0; dg=real_t(1.0/n);
    double a[1]={0},c[1]={1000},d[1]={1};
    BoundsGeometry bounds=BoundsGeometry::TwoSided(xmin,xmax);
    CCSAOptimizerParallel opt(MPI_COMM_WORLD,nl,m,bounds,a,c,d);
    Vector x0(nl); x0=0.5;
    Vector eta=opt.ToLatent(x0), x(nl);
    real_t kkt=1.0;

    auto Eval=[&](const Vector& xv,Vector& fi,real_t& f0){
        double fl=0,xl=0;
        for(int j=0;j<xv.Size();++j){fl+=1.0/double(xv(j));xl+=double(xv(j));}
        f0=real_t(GSum(fl)/n);
        fi(0)=real_t(GSum(xl)/n-Vfrac);
    };

    for(int it=0;it<100&&kkt>1e-5;++it){
        x=opt.ToPhysical(eta);
        for(int j=0;j<nl;++j){double xj=double(x(j));df0(j)=real_t(-1.0/(n*xj*xj));}
        Vector fi(1); real_t f0; Eval(x,fi,f0);
        opt.UpdateGCMMA(eta,df0,f0,fi,&dg,
            [&](const Vector& xc,Vector& fo,real_t& f0o){Eval(xc,fo,f0o);},10);
        x=opt.ToPhysical(eta);
        for(int j=0;j<nl;++j){double xj=double(x(j));df0(j)=real_t(-1.0/(n*xj*xj));}
        Eval(x,fi,f0);
        kkt=opt.KKTresidual(eta,df0,f0,fi,&dg);
    }

    int zero_local=nl==0?1:0, zero_total=0, nr=0;
    MPI_Allreduce(&zero_local,&zero_total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Comm_size(MPI_COMM_WORLD,&nr);
    Check(zero_total==std::max(0,nr-n), "expected number of zero-DOF ranks participated");
    Check(kkt<1e-4, "zero-DOF callback case converges");
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
               "║  CCSA GCMMA callback test suite  (%2d rank(s))           ║\n"
               "╠══════════════════════════════════════════════════════════╣\n"
               "║  Tests the full inner conservatism loop (rho-increase)   ║\n"
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
    Test_ZeroDofCallback();

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
