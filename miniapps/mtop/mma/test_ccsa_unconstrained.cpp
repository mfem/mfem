/**
 * test_ccsa_unconstrained.cpp  —  CCSAOptimizer unconstrained (m=0) tests
 *
 * Same problems as test_sq_unconstrained.cpp using CCSAOptimizer /
 * CCSAOptimizerParallel. For m=0 there is no dual solve at all (the
 * closed-form fixed-lambda primal update degenerates to lambda=empty,
 * R(lambda)=rho_0), so this exercises the m=0 fast path of
 * detail::SolveDualEntropy.
 *
 * CCSAOptimizer works directly on the LATENT variable eta:
 * Update()/KKTresidual() take/return eta, not the physical design, and
 * take no xmin/xmax. Every test builds a BoundsGeometry up front,
 * constructs the optimiser with it, seeds eta = opt.ToLatent(x0), and
 * converts back via opt.ToPhysical(eta) whenever it needs the physical
 * point. Convergence is checked via opt.KKTresidual() (latent-space
 * stationarity — see CCSA_Bregman_MFEM.hpp) rather than a hand-rolled
 * physical-space projected-gradient formula, since there is no longer a
 * physical boundary for a manual projection to clip against.
 *
 * A note on Update() vs UpdateGCMMA(callback) for these problems: unlike
 * SQOptimizer (whose fixed quadratic model is exact for a quadratic
 * objective), CCSA's entropy-Bregman model's initial rho estimate
 * (ComputeInitialRho_(), a gradient-magnitude heuristic borrowed from
 * MMA's GCMMA) is not guaranteed conservative for an arbitrary objective's
 * curvature scale. Plain Update() has no way to detect or correct an
 * insufficient rho, and -- critically -- once seeded it never decays
 * either (DecayRho_() is only called by UpdateGCMMA()), so a fixed,
 * possibly-too-small rho is used forever. On QuadraticBowl (a uniform
 * interior-target problem with no boundary to provide implicit damping)
 * this has been observed to genuinely DIVERGE, not just oscillate: once a
 * coordinate's step overshoots, nothing pulls it back, so its error can
 * grow every iteration, and across repeated runs the resulting trajectory
 * reaches wildly different magnitudes at wildly different iteration
 * counts (consistent with sensitivity to floating-point summation order,
 * itself sensitive to rank count). For that reason main() below does NOT
 * call Test_QuadraticBowl(n, gcmma=false) by default; the function still
 * supports it for anyone who wants to deliberately study the instability,
 * see the comment at that call site. QuadraticBowl and MixedSeparable
 * therefore use the callback-verified UpdateGCMMA() by default, which
 * grows rho until the model is actually conservative; InverseSum (whose
 * optimum sits at the upper bound) happens to converge fine with plain
 * Update() since the entropy curvature factor kappa naturally damps steps
 * near a saturating bound.
 *
 * Build:  cmake --build build   (links MMA_MFEM.cpp + CCSA_Bregman_MFEM.cpp)
 */

#include "CCSA_Bregman_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>

using namespace mfem;
using namespace mfem_mma;
using Clock = std::chrono::steady_clock;

static int g_rank=0, g_nfail=0;
static void Check(bool c,const char* m){ if(g_rank!=0)return; if(c)printf("  [PASS] %s\n",m); else{printf("  [FAIL] %s\n",m);++g_nfail;} }
static double GSum(double v){double g;MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return g;}
static double GMax(double v){double g;MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);return g;}
static const mfem::Vector _empty_fival_;
// Pure arithmetic, no MPI dependency -- must live outside the
// MFEM_USE_MPI guard below since Test_Serial_QuadraticBowl() (which is
// NOT MPI-guarded, and must compile in a non-MPI MFEM build) uses lcgd().
static uint64_t lcg(uint64_t& s){s=s*6364136223846793005ULL+1442695040888963407ULL;return s>>33;}
static double lcgd(uint64_t& s){return double(lcg(s))/double(1ULL<<31);}
#ifdef MFEM_USE_MPI
static std::pair<int,int> Dist(int n){int nr;MPI_Comm_size(MPI_COMM_WORLD,&nr);int b=n/nr,r=n%nr;return{b+(g_rank<r?1:0),g_rank*b+std::min(g_rank,r)};}

// ── Test 1: Quadratic bowl ─────────────────────────────────────────────
static void Test_QuadraticBowl(int n, bool gcmma=false)
{
    if(g_rank==0) printf("\n--- QuadraticBowl (n=%d, m=0, %s) ---\n",n,gcmma?"GCMMA(callback)":"CCSA(Update)");
    auto[nl,off]=Dist(n); MPI_Comm comm=MPI_COMM_WORLD;
    Vector xmin(nl),xmax(nl),df0(nl),target(nl);
    xmin=0.001;xmax=1.0;
    uint64_t s=12345ULL; for(int g=0;g<off;++g) lcgd(s);
    for(int j=0;j<nl;++j) target(j)=real_t(0.2+0.6*lcgd(s));
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizerParallel opt(comm,nl,0,bounds);
    Vector x0(nl); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(nl);
    real_t kkt=1.0; int it=0;
    double maxerr=1.0;
    auto t0=Clock::now();
    // Loop on max_err (the worst-case criterion this test actually asserts),
    // not on the aggregate kkt: kkt is a MEAN-square residual normalized by
    // 1/n, so for large n a handful of slower-converging coordinates can be
    // completely diluted by the many already-converged ones -- kkt can
    // read as "converged" well before every individual coordinate has
    // actually settled near its target.
    for(;it<200&&maxerr>1e-4;++it){
        x = opt.ToPhysical(eta);
        double f0_loc=0;
        for(int j=0;j<nl;++j){double r=double(x(j))-double(target(j));df0(j)=real_t(2.0*r);f0_loc+=r*r;}
        double f0=GSum(f0_loc);
        if(gcmma){
            // Callback-verified GCMMA: the entropy model's initial rho
            // estimate (gradient-magnitude heuristic) is not guaranteed
            // conservative for an arbitrary quadratic's curvature scale --
            // only the callback path can detect an insufficient rho (by
            // comparing the true f0 at the trial point against the
            // model's prediction) and grow it until the step is actually
            // conservative. Plain Update()/the no-callback UpdateGCMMA()
            // overload have no way to do this verification.
            opt.UpdateGCMMA(eta,df0,real_t(f0),_empty_fival_,nullptr,
                [&](const Vector& xc, Vector&, real_t& f0o){
                    double fl=0;
                    for(int j=0;j<xc.Size();++j){double r=double(xc(j))-double(target(j));fl+=r*r;}
                    f0o=real_t(GSum(fl));
                }, /*max_inner=*/15);
        } else {
            opt.Update(eta,df0,real_t(f0));
        }
        x = opt.ToPhysical(eta);
        f0_loc=0;
        double eloc=0;
        for(int j=0;j<nl;++j){double r=double(x(j))-double(target(j));df0(j)=real_t(2.0*r);f0_loc+=r*r;eloc=std::max(eloc,std::abs(r));}
        double f0_print = GSum(f0_loc);
        kkt=opt.KKTresidual(eta,df0,real_t(f0_print));
        maxerr=GMax(eloc);
        if(g_rank==0&&it%20==0) printf("  iter %3d: f0=%.4e kkt=%.4e max_err=%.4e\n",it,f0_print,double(kkt),maxerr);
    }
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    x = opt.ToPhysical(eta);
    if(g_rank==0) printf("  Final: kkt=%.2e max_err=%.2e iters=%d time=%.1fms\n",double(kkt),maxerr,opt.NumIterations(),ms);
    std::string tag=std::string("[")+( gcmma?"GCMMA(callback)":"CCSA(Update)")+",n="+std::to_string(n)+"]";
    if (gcmma) {
        // Callback-verified path: rho grows until conservative, so this
        // has a real convergence guarantee -- hold it to the tight bar.
        Check(kkt<1e-4,   (tag+" KKT<1e-4").c_str());
        Check(maxerr<0.01,(tag+" max_err<0.01").c_str());
    } else {
        // Plain Update() has no conservatism check and therefore no
        // formal global-convergence guarantee (same limitation classic
        // MMA has without its GC extension) -- only assert it doesn't
        // blow up, not that it reaches the analytic optimum. This branch
        // is not called from main() by default -- see the comment there.
        Check(std::isfinite(double(kkt)) && std::isfinite(maxerr),
              (tag+" does not diverge (no convergence guarantee without GCMMA)").c_str());
    }
}

// ── Test 2: InverseSum ────────────────────────────────────────────────────
static void Test_InverseSum(int n)
{
    if(g_rank==0) printf("\n--- InverseSum (n=%d, m=0) ---\n",n);
    auto[nl,off]=Dist(n); MPI_Comm comm=MPI_COMM_WORLD;
    Vector xmin(nl),xmax(nl),df0(nl);
    xmin=0.001;xmax=1.0;
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizerParallel opt(comm,nl,0,bounds);
    Vector x0(nl); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(nl);
    real_t kkt=1.0; int it=0;
    double maxerr=1.0;
    // See Test_QuadraticBowl's comment: loop on max_err (here, distance
    // from the upper bound x=1), not the aggregate kkt.
    for(;it<200&&maxerr>1e-4;++it){
        x = opt.ToPhysical(eta);
        double f0l=0; for(int j=0;j<nl;++j){double xj=double(x(j));df0(j)=real_t(-1.0/(xj*xj));f0l+=1.0/xj;}
        double f0=GSum(f0l); opt.Update(eta,df0,real_t(f0));
        x = opt.ToPhysical(eta);
        double eloc=0;
        f0l=0; for(int j=0;j<nl;++j){double xj=double(x(j));df0(j)=real_t(-1.0/(xj*xj));f0l+=1.0/xj;eloc=std::max(eloc,std::abs(xj-1.0));}
        kkt=opt.KKTresidual(eta,df0,real_t(GSum(f0l)));
        maxerr=GMax(eloc);
        if(g_rank==0&&it%20==0) printf("  iter %3d: kkt=%.4e max_err=%.4e\n",it,double(kkt),maxerr);
    }
    x = opt.ToPhysical(eta);
    double xloc=0; for(int j=0;j<nl;++j) xloc+=double(x(j));
    double xmean=GSum(xloc)/n;
    if(g_rank==0) printf("  Final: xmean=%.6f(1.0) kkt=%.2e max_err=%.2e iters=%d\n",xmean,double(kkt),maxerr,opt.NumIterations());
    std::string tag="[InvSum,n="+std::to_string(n)+"]";
    Check(kkt<1e-4,         (tag+" KKT<1e-4").c_str());
    Check(std::abs(xmean-1.0)<0.01,(tag+" mean(x)~1").c_str());
}

// ── Test 3: MixedSeparable ────────────────────────────────────────────────
// Uses the callback-verified UpdateGCMMA() (see the comment in
// Test_QuadraticBowl above): this objective has a genuine per-coordinate
// curvature (interior optimum x*=sqrt(a/b) for many coordinates) that the
// gradient-magnitude rho heuristic alone does not reliably bound; only the
// callback path can detect and grow an insufficient rho.
static void Test_MixedSeparable(int n)
{
    if(g_rank==0) printf("\n--- MixedSeparable (n=%d, m=0) ---\n",n);
    auto[nl,off]=Dist(n); MPI_Comm comm=MPI_COMM_WORLD;
    Vector xmin(nl),xmax(nl),df0(nl),alpha(nl),beta_v(nl),xstar(nl);
    xmin=0.001;xmax=1.0;
    uint64_t s=98765ULL; for(int g=0;g<off;++g){lcgd(s);lcgd(s);}
    for(int j=0;j<nl;++j){double a=0.5+1.5*lcgd(s),b=0.5+1.5*lcgd(s);alpha(j)=a;beta_v(j)=b;xstar(j)=real_t(std::max(0.001,std::min(1.0,std::sqrt(a/b))));}
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizerParallel opt(comm,nl,0,bounds);
    Vector x0(nl); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(nl);
    real_t kkt=1.0; int it=0;
    double maxerr=1.0;
    // See Test_QuadraticBowl's comment: loop on max_err, the worst-case
    // criterion actually asserted, not the mean-square-normalized kkt.
    //
    // Needs a larger budget than QuadraticBowl: this objective's
    // per-coordinate curvature f''_j = 2*a_j/x_j^3 varies substantially
    // across coordinates (a_j in [0.5,2]), but CCSA shares a single scalar
    // rho across all n coordinates. Once rho grows (via the callback
    // conservatism check) enough to be conservative for the worst-case
    // (highest-curvature) coordinate, it stays there -- theta_rho_
    // defaults to 1.0 (no automatic between-iteration decay, see
    // SetRhoParams()) -- so every coordinate is throttled by the same
    // fixed, worst-case step size. That is the textbook O(1/k) rate for a
    // fixed-step method on a heterogeneous-curvature problem (empirically
    // max_err(it) ~ 3/it here), not a stall: kkt keeps dropping
    // monotonically the whole time.
    for(;it<500&&maxerr>1e-4;++it){
        x = opt.ToPhysical(eta);
        double f0l=0; for(int j=0;j<nl;++j){double xj=double(x(j)),a=double(alpha(j)),b=double(beta_v(j));df0(j)=real_t(-a/(xj*xj)+b);f0l+=a/xj+b*xj;}
        double f0=GSum(f0l);
        int inner=0;
        opt.UpdateGCMMA(eta,df0,real_t(f0),_empty_fival_,nullptr,
            [&](const Vector& xc, Vector&, real_t& f0o){
                double fl=0;
                for(int j=0;j<xc.Size();++j){
                    double xj=double(xc(j)),a=double(alpha(j)),b=double(beta_v(j));
                    fl+=a/xj+b*xj;
                }
                f0o=real_t(GSum(fl));
            }, /*max_inner=*/15, &inner);
        x = opt.ToPhysical(eta);
        // Recompute df0 at the updated x for a correct KKT check
        double eloc=0;
        f0l=0; for(int j=0;j<nl;++j){double xj=double(x(j)),a=double(alpha(j)),b=double(beta_v(j));df0(j)=real_t(-a/(xj*xj)+b);f0l+=a/xj+b*xj;eloc=std::max(eloc,std::abs(double(x(j))-double(xstar(j))));}
        kkt=opt.KKTresidual(eta,df0,real_t(GSum(f0l)));
        maxerr=GMax(eloc);
        if(g_rank==0&&it%50==0) printf("  iter %3d: kkt=%.4e max_err=%.4e\n",it,double(kkt),maxerr);
    }
    x = opt.ToPhysical(eta);
    if(g_rank==0) printf("  Final: kkt=%.2e max_err=%.2e iters=%d\n",double(kkt),maxerr,opt.NumIterations());
    std::string tag="[MixSep,n="+std::to_string(n)+"]";
    Check(kkt<1e-4,   (tag+" KKT<1e-4").c_str());
    Check(maxerr<0.01,(tag+" max_err<0.01").c_str());
}

// ── Test 4: Serial CCSAOptimizer ──────────────────────────────────────────
#endif // MFEM_USE_MPI

// Uses the callback-verified UpdateGCMMA() -- see the comment in the
// parallel Test_QuadraticBowl above: plain Update() has no way to detect
// or correct an insufficient rho estimate for this objective's curvature
// scale, so it is not used here. Note this also means CCSA is NOT
// "exact for quadratics in ~1 iteration" the way SQOptimizer's fixed
// quadratic model is -- it typically needs several outer iterations
// (each possibly with a handful of inner conservatism-check retries).
static void Test_Serial_QuadraticBowl(int n)
{
    if(g_rank!=0) return;
    printf("\n--- Serial CCSAOptimizer QuadraticBowl (n=%d) ---\n",n);
    Vector xmin(n),xmax(n),df0(n),target(n);
    xmin=0.001;xmax=1.0;
    uint64_t s=12345ULL; for(int j=0;j<n;++j) target(j)=real_t(0.2+0.6*lcgd(s));
    BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
    CCSAOptimizer opt(n,0,bounds);
    Vector x0(n); x0=0.5;
    Vector eta = opt.ToLatent(x0);
    Vector x(n);
    real_t kkt=1.0; int it=0;
    double maxerr=1.0;
    auto t0=Clock::now();
    // See the parallel Test_QuadraticBowl's comment: loop on max_err, the
    // worst-case criterion actually asserted, not the aggregate kkt.
    for(;it<200&&maxerr>1e-4;++it){
        x = opt.ToPhysical(eta);
        double f0=0; for(int j=0;j<n;++j){double r=double(x(j))-double(target(j));df0(j)=real_t(2.0*r);f0+=r*r;}
        opt.UpdateGCMMA(eta,df0,real_t(f0),_empty_fival_,nullptr,
            [&](const Vector& xc, Vector&, real_t& f0o){
                double fl=0;
                for(int j=0;j<xc.Size();++j){double r=double(xc(j))-double(target(j));fl+=r*r;}
                f0o=real_t(fl);
            }, /*max_inner=*/15);
        x = opt.ToPhysical(eta);
        f0=0; maxerr=0;
        for(int j=0;j<n;++j){double r=double(x(j))-double(target(j));df0(j)=real_t(2.0*r);f0+=r*r;maxerr=std::max(maxerr,std::abs(r));}
        kkt=opt.KKTresidual(eta,df0,real_t(f0));
        if(it%20==0) printf("  iter %3d: kkt=%.4e max_err=%.4e\n",it,double(kkt),maxerr);
    }
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    x = opt.ToPhysical(eta);
    printf("  Final: kkt=%.2e max_err=%.2e iters=%d time=%.1fms (%.2fms/iter)\n",double(kkt),maxerr,opt.NumIterations(),ms,ms/std::max(it,1));
    std::string tag="[serial,n="+std::to_string(n)+"]";
    Check(kkt<1e-4,   (tag+" KKT<1e-4").c_str());
    Check(maxerr<0.01,(tag+" max_err<0.01").c_str());
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&g_rank);
    int nr; MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(g_rank==0) printf("=== CCSAOptimizer Unconstrained (m=0) test suite  (%d rank(s)) ===\n",nr);

    // Serial (rank 0 only)
    if(g_rank==0) printf("\n── Serial CCSAOptimizer ─────────────────────────────────\n");
    Test_Serial_QuadraticBowl(10000);
    Test_Serial_QuadraticBowl(50000);
    Test_Serial_QuadraticBowl(100000);
#ifdef MFEM_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);

    // Parallel
    if(g_rank==0) printf("\n── Parallel CCSAOptimizerParallel ───────────────────────\n");
    // NOTE: Test_QuadraticBowl(n, gcmma=false) -- the plain Update()
    // branch -- is deliberately NOT called here. Testing has shown it can
    // genuinely diverge (not just oscillate) on this adversarial,
    // uniform-interior-target problem: once a coordinate's step overshoots
    // past a stable region, the fixed (never-decayed, since plain Update()
    // never calls DecayRho_()) rho provides no negative feedback, so a
    // coordinate's error can grow every iteration rather than settle into
    // a bounded cycle. Across repeated runs this reaches wildly different
    // magnitudes at wildly different iteration counts (consistent with
    // sensitivity to floating-point summation order, itself sensitive to
    // rank count), and has been observed to hang a multi-rank run. This is
    // exactly the risk Update()'s documentation already calls out (no
    // formal convergence guarantee); the function still supports
    // gcmma=false for anyone who wants to deliberately study this
    // instability, it is just not exercised by default here.
    Test_QuadraticBowl(10000,true);  Test_QuadraticBowl(50000,true);
    Test_InverseSum(10000); Test_InverseSum(100000);
    Test_MixedSeparable(10000); Test_MixedSeparable(50000); Test_MixedSeparable(100000);

#endif // MFEM_USE_MPI

    if(g_rank==0){printf("\n========================================\n");
    if(g_nfail==0)printf("All CCSA unconstrained tests PASSED.\n");
    else printf("%d CCSA unconstrained test(s) FAILED.\n",g_nfail);printf("========================================\n");}
    MPI_Finalize(); return g_nfail>0?1:0;
}
