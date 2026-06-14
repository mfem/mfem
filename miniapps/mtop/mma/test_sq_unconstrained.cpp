/**
 * test_sq_unconstrained.cpp  —  SQOptimizer unconstrained (m=0) tests
 *
 * Same problems as test_mma_unconstrained.cpp using SQOptimizer /
 * SQOptimizerParallel.  For m=0 the SQ step is exact for quadratic
 * objectives (converges in 1 iteration) and provides a gradient step
 * with curvature scaling for non-quadratic objectives.
 */

#include "MMA_MFEM.hpp"
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
static uint64_t lcg(uint64_t& s){s=s*6364136223846793005ULL+1442695040888963407ULL;return s>>33;}
static double lcgd(uint64_t& s){return double(lcg(s))/double(1ULL<<31);}
static std::pair<int,int> Dist(int n){int nr;MPI_Comm_size(MPI_COMM_WORLD,&nr);int b=n/nr,r=n%nr;return{b+(g_rank<r?1:0),g_rank*b+std::min(g_rank,r)};}

// ── Test 1: Quadratic bowl (SQ exact in 1 iter for quadratic) ─────────────
static void Test_QuadraticBowl(int n, bool gcmma=false)
{
    if(g_rank==0) printf("\n--- QuadraticBowl (n=%d, m=0, %s) ---\n",n,gcmma?"GCMMA":"SQ");
    auto[nl,off]=Dist(n); MPI_Comm comm=MPI_COMM_WORLD;
    Vector x(nl),xmin(nl),xmax(nl),df0(nl),target(nl);
    xmin=0.001;xmax=1.0;x=0.5;
    uint64_t s=12345ULL; for(int g=0;g<off;++g) lcgd(s);
    for(int j=0;j<nl;++j) target(j)=real_t(0.2+0.6*lcgd(s));
    SQOptimizerParallel opt(comm,nl,0,x); double kkt=1.0; int it=0;
    auto t0=Clock::now();
    for(;it<200&&kkt>1e-5;++it){
        double f0_loc=0;
        for(int j=0;j<nl;++j){double r=double(x(j))-double(target(j));df0(j)=real_t(2.0*r/n);f0_loc+=r*r/n;}
        double f0=GSum(f0_loc);
        if(gcmma) opt.UpdateGCMMA(x,df0,f0,xmin,xmax); else opt.Update(x,df0,f0,xmin,xmax);
        double pg2=0; for(int j=0;j<nl;++j){double g=double(df0(j)),pg=g;if(double(x(j))<=double(xmin(j))+1e-3)pg=std::min(0.0,g);if(double(x(j))>=double(xmax(j))-1e-3)pg=std::max(0.0,g);pg2+=pg*pg;}
        kkt=GSum(pg2)/n;
        if(g_rank==0&&it%20==0) printf("  iter %3d: f0=%.4e kkt=%.4e\n",it,f0,kkt);
    }
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    double eloc=0; for(int j=0;j<nl;++j) eloc=std::max(eloc,std::abs(double(x(j))-double(target(j))));
    double maxerr=GMax(eloc);
    if(g_rank==0) printf("  Final: kkt=%.2e max_err=%.2e iters=%d time=%.1fms\n",kkt,maxerr,opt.NumIterations(),ms);
    std::string tag=std::string("[")+( gcmma?"GCMMA":"SQ")+",n="+std::to_string(n)+"]";
    Check(kkt<1e-4,  (tag+" KKT<1e-4").c_str());
    Check(maxerr<0.01,(tag+" max_err<0.01").c_str());
}

// ── Test 2: InverseSum ────────────────────────────────────────────────────
static void Test_InverseSum(int n)
{
    if(g_rank==0) printf("\n--- InverseSum (n=%d, m=0) ---\n",n);
    auto[nl,off]=Dist(n); MPI_Comm comm=MPI_COMM_WORLD;
    Vector x(nl),xmin(nl),xmax(nl),df0(nl);
    xmin=0.001;xmax=1.0;x=0.5;
    SQOptimizerParallel opt(comm,nl,0,x); double kkt=1.0; int it=0;
    for(;it<200&&kkt>1e-5;++it){
        double f0l=0; for(int j=0;j<nl;++j){double xj=double(x(j));df0(j)=real_t(-1.0/(xj*xj));f0l+=1.0/xj;}
        double f0=GSum(f0l); opt.Update(x,df0,f0,xmin,xmax);
        double pg2=0; for(int j=0;j<nl;++j){double g=double(df0(j)),pg=double(x(j))>=double(xmax(j))-1e-3?std::max(0.0,g):g;pg2+=pg*pg;}
        kkt=GSum(pg2)/n;
        if(g_rank==0&&it%20==0) printf("  iter %3d: kkt=%.4e\n",it,kkt);
    }
    double xloc=0; for(int j=0;j<nl;++j) xloc+=double(x(j));
    double xmean=GSum(xloc)/n;
    double eloc=0; for(int j=0;j<nl;++j) eloc=std::max(eloc,std::abs(double(x(j))-1.0));
    double maxerr=GMax(eloc);
    if(g_rank==0) printf("  Final: xmean=%.6f(1.0) kkt=%.2e max_err=%.2e iters=%d\n",xmean,kkt,maxerr,opt.NumIterations());
    std::string tag="[InvSum,n="+std::to_string(n)+"]";
    Check(kkt<1e-4,         (tag+" KKT<1e-4").c_str());
    Check(std::abs(xmean-1.0)<0.001,(tag+" mean(x)~1").c_str());
}

// ── Test 3: MixedSeparable ────────────────────────────────────────────────
static void Test_MixedSeparable(int n)
{
    if(g_rank==0) printf("\n--- MixedSeparable (n=%d, m=0) ---\n",n);
    auto[nl,off]=Dist(n); MPI_Comm comm=MPI_COMM_WORLD;
    Vector x(nl),xmin(nl),xmax(nl),df0(nl),alpha(nl),beta_v(nl),xstar(nl);
    xmin=0.001;xmax=1.0;x=0.5;
    uint64_t s=98765ULL; for(int g=0;g<off;++g){lcgd(s);lcgd(s);}
    for(int j=0;j<nl;++j){double a=0.5+1.5*lcgd(s),b=0.5+1.5*lcgd(s);alpha(j)=a;beta_v(j)=b;xstar(j)=real_t(std::max(0.001,std::min(1.0,std::sqrt(a/b))));}
    SQOptimizerParallel opt(comm,nl,0,x); double kkt=1.0; int it=0;
    for(;it<200&&kkt>1e-5;++it){
        double f0l=0; for(int j=0;j<nl;++j){double xj=double(x(j)),a=double(alpha(j)),b=double(beta_v(j));df0(j)=real_t((-a/(xj*xj)+b)/nl);f0l+=(a/xj+b*xj)/nl;}
        double f0=GSum(f0l); opt.Update(x,df0,f0,xmin,xmax);
        // Recompute df0 at the updated x for a correct KKT check
        for(int j=0;j<nl;++j){double xj=double(x(j)),a=double(alpha(j)),b=double(beta_v(j));df0(j)=real_t((-a/(xj*xj)+b)/nl);}
        double pg2=0; for(int j=0;j<nl;++j){double g=double(df0(j)),pg=g;if(double(x(j))<=double(xmin(j))+1e-3)pg=std::min(0.0,g);if(double(x(j))>=double(xmax(j))-1e-3)pg=std::max(0.0,g);pg2+=pg*pg;}
        kkt=GSum(pg2)/n;
        if(g_rank==0&&it%20==0) printf("  iter %3d: kkt=%.4e\n",it,kkt);
    }
    double eloc=0; for(int j=0;j<nl;++j) eloc=std::max(eloc,std::abs(double(x(j))-double(xstar(j))));
    double maxerr=GMax(eloc);
    if(g_rank==0) printf("  Final: kkt=%.2e max_err=%.2e iters=%d\n",kkt,maxerr,opt.NumIterations());
    std::string tag="[MixSep,n="+std::to_string(n)+"]";
    Check(kkt<1e-4,   (tag+" KKT<1e-4").c_str());
}

// ── Test 4: Serial SQOptimizer ────────────────────────────────────────────
static void Test_Serial_QuadraticBowl(int n)
{
    if(g_rank!=0) return;
    printf("\n--- Serial SQOptimizer QuadraticBowl (n=%d) ---\n",n);
    Vector x(n),xmin(n),xmax(n),df0(n),target(n);
    xmin=0.001;xmax=1.0;x=0.5;
    uint64_t s=12345ULL; for(int j=0;j<n;++j) target(j)=real_t(0.2+0.6*lcgd(s));
    SQOptimizer opt(n,0,x); double kkt=1.0; int it=0;
    auto t0=Clock::now();
    for(;it<200&&kkt>1e-5;++it){
        double f0=0; for(int j=0;j<n;++j){double r=double(x(j))-double(target(j));df0(j)=real_t(2.0*r/n);f0+=r*r/n;}
        opt.Update(x,df0,f0,xmin,xmax);
        double pg2=0; for(int j=0;j<n;++j){double g=double(df0(j)),pg=g;if(double(x(j))<=double(xmin(j))+1e-3)pg=std::min(0.0,g);if(double(x(j))>=double(xmax(j))-1e-3)pg=std::max(0.0,g);pg2+=pg*pg;}
        kkt=pg2/n;
        if(it%20==0) printf("  iter %3d: kkt=%.4e\n",it,kkt);
    }
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    double maxerr=0; for(int j=0;j<n;++j) maxerr=std::max(maxerr,std::abs(double(x(j))-double(target(j))));
    printf("  Final: kkt=%.2e max_err=%.2e iters=%d time=%.1fms (%.2fms/iter)\n",kkt,maxerr,opt.NumIterations(),ms,ms/std::max(it,1));
    // SQ is exact for quadratic — expect 1-2 iterations
    std::string tag="[serial,n="+std::to_string(n)+"]";
    Check(kkt<1e-4,   (tag+" KKT<1e-4").c_str());
    Check(maxerr<0.01,(tag+" max_err<0.01").c_str());
    Check(it<=10,     (tag+" converges fast (quadratic exact)").c_str());
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&g_rank);
    int nr; MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(g_rank==0) printf("=== SQOptimizer Unconstrained (m=0) test suite  (%d rank(s)) ===\n",nr);

    // Serial (rank 0 only)
    if(g_rank==0) printf("\n── Serial SQOptimizer ───────────────────────────────────\n");
    Test_Serial_QuadraticBowl(10000);
    Test_Serial_QuadraticBowl(50000);
    Test_Serial_QuadraticBowl(100000);
    MPI_Barrier(MPI_COMM_WORLD);

    // Parallel
    if(g_rank==0) printf("\n── Parallel SQOptimizerParallel ─────────────────────────\n");
    Test_QuadraticBowl(10000,false); Test_QuadraticBowl(50000,false); Test_QuadraticBowl(100000,false);
    Test_QuadraticBowl(10000,true);  Test_QuadraticBowl(50000,true);
    Test_InverseSum(10000); Test_InverseSum(100000);
    Test_MixedSeparable(10000); Test_MixedSeparable(50000); Test_MixedSeparable(100000);

    if(g_rank==0){printf("\n========================================\n");
    if(g_nfail==0)printf("All SQ unconstrained tests PASSED.\n");
    else printf("%d SQ unconstrained test(s) FAILED.\n",g_nfail);printf("========================================\n");}
    MPI_Finalize(); return g_nfail>0?1:0;
}
