/**
 * test_sq_parallel.cpp  —  SQOptimizerParallel test suite
 *
 * Same parallel problems as test_mma_parallel.cpp using SQOptimizerParallel.  —  Parallel MMAOptimizerParallel test suite
 *
 * Tests mfem_mma::SQOptimizerParallel (MPI, distributed mfem::Vector).
 * Covers: MMA Update, GCMMA UpdateGCMMA, all 8 problem types.
 * DOFs are distributed uniformly across all MPI ranks.
 *
 * Build:  cmake --build build
 * Run:    mpirun -np 4 ./build/test_mma_parallel
 *         ./build/test_mma_parallel        (1 rank — verifies serial-parallel match)
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

// ── uniform distribution of n_global DOFs across MPI_COMM_WORLD ─────────
static std::pair<int,int> Distribute(int n_global)
{
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    int base   = n_global / nranks;
    int rem    = n_global % nranks;
    int n_loc  = base + (g_rank < rem ? 1 : 0);
    int offset = g_rank * base + std::min(g_rank, rem);
    return {n_loc, offset};
}

static double GlobalSum(double v)
{
    double g; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return g;
}

// ============================================================
// Test 1/2 — Compliance proxy  (MMA and GCMMA)
//   min Σ1/xj   s.t. mean(x)≤Vfrac,   x∈[0.001,1]
// ============================================================
static void Test_ComplianceProxy(int n, double Vfrac, bool gcmma)
{
    if (g_rank==0)
        printf("\n--- ComplianceProxy (n=%d, Vfrac=%.2f, %s) ---\n",
               n, Vfrac, gcmma ? "GCMMA" : "MMA");

    auto [nl, off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(nl), xmin(nl), xmax(nl), df0(nl), dg(nl);
    x = 0.5; xmin = 0.001; xmax = 1.0; dg = real_t(1.0/n);

    MMAOptimizerParallel opt(comm, nl, 1, x);
    double kkt = 1.0;

    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        double gl = 0.0;
        for (int j=0;j<nl;++j) { df0(j)=real_t(-1.0/(double(x(j))*double(x(j)))); gl+=double(x(j)); }
        double g = GlobalSum(gl)/n - Vfrac;
        mfem::Vector fival(1);
        fival(0)=g;

        if (gcmma) {
            int inner;
            opt.UpdateGCMMA(x, df0, 0.0, fival, &dg, xmin, xmax, &inner);
        } else {
            opt.Update(x, df0, 0.0, fival, &dg, xmin, xmax);
        }

        gl=0; for(int j=0;j<nl;++j){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));gl+=double(x(j));}
        g=GlobalSum(gl)/n-Vfrac; fival(0)=g;
        kkt = opt.KKTresidual(x, df0, 0.0, fival, &dg, xmin, xmax);
        if (g_rank==0 && it%20==0)
            printf("  iter %3d: g=%.4e  kkt=%.4e\n", it, g, kkt);
    }

    double xloc=0; for(int j=0;j<nl;++j) xloc+=double(x(j));
    double xmean = GlobalSum(xloc)/n;
    if (g_rank==0)
        printf("  Final: xmean=%.6f (%.2f)  kkt=%.2e  iters=%d\n",
               xmean, Vfrac, kkt, opt.NumIterations());
    Check(kkt < 1e-4,                   "KKT < 1e-4");
    Check(std::abs(xmean-Vfrac) < 0.01, "Volume fraction satisfied");
}

// ============================================================
// Test 3 — Two block volume constraints
// ============================================================
static void Test_TwoConstraints(int n, double V1, double V2)
{
    if(g_rank==0) printf("\n--- TwoConstraints (n=%d, V1=%.2f, V2=%.2f) ---\n",n,V1,V2);
    auto [nl,off] = Distribute(n);
    const int n1=n/2, n2=n-n1;
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(nl),xmin(nl),xmax(nl),df0(nl),dg0(nl),dg1(nl);
    x=0.5;xmin=0.001;xmax=1.0;dg0=0.0;dg1=0.0;
    for(int j=0;j<nl;++j){
        int g=off+j;
        if(g<n1) dg0(j)=real_t(1.0/n1);
        else     dg1(j)=real_t(1.0/n2);
    }
    double cv=std::max(1000.0,10.0*n);
    double a2[2]={0,0},c2[2]={cv,cv},d2[2]={1,1};
    MMAOptimizerParallel opt(comm,nl,2,x,a2,c2,d2);
    Vector dg[2]={dg0,dg1};
    double kkt=1.0;

    for(int it=0;it<200&&kkt>1e-5;++it){
        double s0l=0,s1l=0;
        for(int j=0;j<nl;++j){
            int g=off+j; df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));
            if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));
        }
        double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
        mfem::Vector fival(2); fival(0)=s0/n1-V1; fival(1)=s1/n2-V2;
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        s0l=s1l=0;
        for(int j=0;j<nl;++j){int g=off+j;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));}
        s0=GlobalSum(s0l);s1=GlobalSum(s1l);
        fival(0)=s0/n1-V1;fival(1)=s1/n2-V2;
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax);
        if(g_rank==0&&it%20==0) printf("  iter %3d: g=[%.4e,%.4e] kkt=%.4e\n",it,fival(0),fival(1),kkt);
    }
    double s0l=0,s1l=0;
    for(int j=0;j<nl;++j){int g=off+j;if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));}
    double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
    if(g_rank==0) printf("  Final: mean1=%.6f(%.2f) mean2=%.6f(%.2f) kkt=%.2e\n",s0/n1,V1,s1/n2,V2,kkt);
    Check(kkt<1e-4,                "KKT < 1e-4");
    Check(std::abs(s0/n1-V1)<0.01, "Block 1 volume");
    Check(std::abs(s1/n2-V2)<0.01, "Block 2 volume");
}

// ============================================================
// Test 4 — Three block volume constraints
// ============================================================
static void Test_ThreeConstraints(int n, double V1, double V2, double V3)
{
    const int b1=n/3,b2=2*n/3;
    const int sz[3]={b1,b2-b1,n-b2};
    const double Vt[3]={V1,V2,V3};
    if(g_rank==0) printf("\n--- ThreeConstraints (n=%d, V=%.2f/%.2f/%.2f) ---\n",n,V1,V2,V3);

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(nl),xmin(nl),xmax(nl),df0(nl);
    Vector dg[3]; for(int i=0;i<3;++i){dg[i].SetSize(nl);dg[i]=0.0;}
    x=0.5;xmin=0.001;xmax=1.0;
    for(int j=0;j<nl;++j){int g=off+j;int blk=(g<b1)?0:(g<b2)?1:2;dg[blk](j)=real_t(1.0/sz[blk]);}

    double cv=std::max(1000.0,10.0*n);
    double a3[3]={0,0,0},c3[3]={cv,cv,cv},d3[3]={1,1,1};
    MMAOptimizerParallel opt(comm,nl,3,x,a3,c3,d3);
    double kkt=1.0;

    for(int it=0;it<200&&kkt>1e-5;++it){
        double sll[3]={0,0,0};
        for(int j=0;j<nl;++j){int g=off+j;int blk=(g<b1)?0:(g<b2)?1:2;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));sll[blk]+=double(x(j));}
        double sl[3]; for(int i=0;i<3;++i) MPI_Allreduce(&sll[i],&sl[i],1,MPI_DOUBLE,MPI_SUM,comm);
        mfem::Vector fival(3); for(int i=0;i<3;++i) fival(i)=sl[i]/sz[i]-Vt[i];
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        sll[0]=sll[1]=sll[2]=0;
        for(int j=0;j<nl;++j){int g=off+j;int blk=(g<b1)?0:(g<b2)?1:2;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));sll[blk]+=double(x(j));}
        for(int i=0;i<3;++i) MPI_Allreduce(&sll[i],&sl[i],1,MPI_DOUBLE,MPI_SUM,comm);
        for(int i=0;i<3;++i) fival(i)=sl[i]/sz[i]-Vt[i];
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax);
        if(g_rank==0&&it%20==0) printf("  iter %3d: g=[%.3e,%.3e,%.3e] kkt=%.4e\n",it,fival(0),fival(1),fival(2),kkt);
    }
    double sll[3]={0,0,0},sl[3];
    for(int j=0;j<nl;++j){int g=off+j;int blk=(g<b1)?0:(g<b2)?1:2;sll[blk]+=double(x(j));}
    for(int i=0;i<3;++i) MPI_Allreduce(&sll[i],&sl[i],1,MPI_DOUBLE,MPI_SUM,comm);
    if(g_rank==0) printf("  Final: means=[%.4f,%.4f,%.4f] kkt=%.2e\n",sl[0]/sz[0],sl[1]/sz[1],sl[2]/sz[2],kkt);
    Check(kkt<1e-4,"KKT < 1e-4");
    for(int i=0;i<3;++i)
        Check(std::abs(sl[i]/sz[i]-Vt[i])<0.01,
              (std::string("Block ")+std::to_string(i+1)+" volume").c_str());
}

// ============================================================
// Test 5 — Large n (parallel scaling)
// ============================================================
static void Test_ParallelScale(int n)
{
    if(g_rank==0) printf("\n--- ParallelScale (n=%d, m=1) ---\n",n);
    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(nl),xmin(nl),xmax(nl),df0(nl),dg(nl);
    x=0.5;xmin=0.001;xmax=1.0;dg=real_t(1.0/n);
    double cv=std::max(1000.0,10.0*n);
    double a1[1]={0},c1[1]={cv},d1[1]={1};
    MMAOptimizerParallel opt(comm,nl,1,x,a1,c1,d1);
    double kkt=1.0;

    for(int it=0;it<100&&kkt>1e-5;++it){
        double gl=0;
        for(int j=0;j<nl;++j){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));gl+=double(x(j));}
        double g=GlobalSum(gl)/n-0.4; mfem::Vector fival(1); fival(0)=g;
        opt.Update(x,df0,0.0,fival,&dg,xmin,xmax);
        gl=0;for(int j=0;j<nl;++j){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));gl+=double(x(j));}
        g=GlobalSum(gl)/n-0.4;fival(0)=g;
        kkt=opt.KKTresidual(x,df0,0.0,fival,&dg,xmin,xmax);
        if(g_rank==0&&it%10==0) printf("  iter %3d: g=%.4e kkt=%.4e\n",it,g,kkt);
    }
    double xl=0;for(int j=0;j<nl;++j)xl+=double(x(j));
    double xmean=GlobalSum(xl)/n;
    if(g_rank==0) printf("  Final: xmean=%.6f  kkt=%.2e  iters=%d\n",xmean,kkt,opt.NumIterations());
    Check(kkt<1e-4,                   "KKT < 1e-4");
    Check(std::abs(xmean-0.4)<0.01,   "Volume fraction satisfied");
}

// ============================================================
// Test 6 — Constraint switching
// ============================================================
static void Test_ConstraintSwitching()
{
    const int n=200,n1=n/2,n2=n-n1;
    if(g_rank==0){
        printf("\n--- ConstraintSwitching (n=%d, m=2) ---\n",n);
        printf("  left compliance->g0 ACTIVE; right material->g1 INACTIVE\n");
    }
    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(nl),xmin(nl),xmax(nl),df0(nl),dg0(nl),dg1(nl);
    x=0.5;xmin=0.001;xmax=1.0;dg0=0.0;dg1=0.0;
    for(int j=0;j<nl;++j){int g=off+j;if(g<n1)dg0(j)=real_t(1.0/n1);else dg1(j)=real_t(1.0/n2);}
    double cv=std::max(1000.0,10.0*n);
    double a2[2]={0,0},c2[2]={cv,cv},d2[2]={1,1};
    MMAOptimizerParallel opt(comm,nl,2,x,a2,c2,d2);
    Vector dg[2]={dg0,dg1};
    double kkt=1.0;

    for(int it=0;it<300&&kkt>1e-5;++it){
        double s0l=0,s1l=0;
        for(int j=0;j<nl;++j){
            int g=off+j;
            if(g<n1){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));s0l+=double(x(j));}
            else    {df0(j)=1.0;s1l+=double(x(j));}
        }
        double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
        mfem::Vector fival(2); fival(0)=s0/n1-0.5; fival(1)=s1/n2-0.5;
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        s0l=s1l=0;
        for(int j=0;j<nl;++j){int g=off+j;if(g<n1){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));s0l+=double(x(j));}else{df0(j)=1.0;s1l+=double(x(j));}}
        s0=GlobalSum(s0l);s1=GlobalSum(s1l);
        fival(0)=s0/n1-0.5;fival(1)=s1/n2-0.5;
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax);
        if(g_rank==0&&it%20==0){
            const auto& lam=opt.GetLambda();
            printf("  iter %3d: g=[%.4f,%.4f] lam=[%.3e,%.3e] kkt=%.4e\n",
                   it,fival(0),fival(1),lam[0],lam[1],kkt);
        }
    }
    double s0l=0,s1l=0;
    for(int j=0;j<nl;++j){int g=off+j;if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));}
    double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
    if(g_rank==0) printf("  Final: mean_left=%.5f(0.50) mean_right=%.5f(<0.1) kkt=%.2e\n",s0/n1,s1/n2,kkt);
    Check(kkt<1e-4,                   "KKT < 1e-4");
    Check(std::abs(s0/n1-0.5)<0.01,   "Left block at 0.5 (g0 active)");
    Check(s1/n2<0.1,                  "Right block near xmin (g1 inactive)");
}

// ============================================================
// Test 7 — 100 regional volume constraints
// ============================================================
static void Test_HundredConstraints()
{
    const int n=1000,m=100,region=n/m;
    if(g_rank==0) printf("\n--- HundredConstraints (n=%d, m=%d) ---\n",n,m);

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    std::vector<double> Vtgt(m);
    for(int k=0;k<m;++k) Vtgt[k]=(k%2==0)?0.3:0.6;

    std::vector<Vector> dg(m); for(int k=0;k<m;++k){dg[k].SetSize(nl);dg[k]=0.0;}
    Vector x(nl),xmin(nl),xmax(nl),df0(nl);
    x=0.5;xmin=0.001;xmax=1.0;
    for(int j=0;j<nl;++j){int g=off+j;int k=g/region;if(k<m)dg[k](j)=real_t(1.0/region);}

    double cv=std::max(1000.0,10.0*n);
    std::vector<double> av(m,0),cv_v(m,cv),dv(m,1);
    MMAOptimizerParallel opt(comm,nl,m,x,av.data(),cv_v.data(),dv.data());
    double kkt=1.0;

    for(int it=0;it<200&&kkt>1e-5;++it){
        std::vector<double> sll(m,0);
        for(int j=0;j<nl;++j){int g=off+j;int k=g/region;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));if(k<m)sll[k]+=double(x(j));}
        std::vector<double> sl(m);
        MPI_Allreduce(sll.data(),sl.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        mfem::Vector fival(m); for(int k=0;k<m;++k) fival(k)=sl[k]/region-Vtgt[k];
        opt.Update(x,df0,0.0,fival,dg.data(),xmin,xmax);
        std::fill(sll.begin(),sll.end(),0);
        for(int j=0;j<nl;++j){int g=off+j;int k=g/region;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));if(k<m)sll[k]+=double(x(j));}
        MPI_Allreduce(sll.data(),sl.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        for(int k=0;k<m;++k) fival(k)=sl[k]/region-Vtgt[k];
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg.data(),xmin,xmax);
        if(g_rank==0&&it%20==0){
            double gmax=*std::max_element(fival.begin(),fival.end());
            printf("  iter %3d: g_max=%.4e kkt=%.4e\n",it,gmax,kkt);
        }
    }
    std::vector<double> sll(m,0),sl(m);
    for(int j=0;j<nl;++j){int g=off+j;int k=g/region;if(k<m)sll[k]+=double(x(j));}
    MPI_Allreduce(sll.data(),sl.data(),m,MPI_DOUBLE,MPI_SUM,comm);
    if(g_rank==0){
        int nw=0;double me=0;
        for(int k=0;k<m;++k){double e=std::abs(sl[k]/region-Vtgt[k]);me=std::max(me,e);if(e>0.02)++nw;}
        printf("  Final: kkt=%.2e max_err=%.2e wrong=%d/%d iters=%d\n",kkt,me,nw,m,opt.NumIterations());
        Check(kkt<1e-4,"KKT < 1e-4");
        Check(me<0.02, "All 100 regions at target");
        Check(nw==0,   "No region violates target");
    } else {
        // Non-root ranks still need to check (count is global)
        // Only rank 0 checks to keep output clean
    }
}

// ============================================================
// Test 8 — SetAsymptotes API (parallel)
// ============================================================
static void Test_SetAsymptotes()
{
    const int n=200;
    if(g_rank==0) printf("\n--- SetAsymptotes (n=%d, m=1, custom asy) ---\n",n);
    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(nl),xmin(nl),xmax(nl),df0(nl),dg(nl);
    x=0.5;xmin=0.001;xmax=1.0;dg=real_t(1.0/n);
    MMAOptimizerParallel opt(comm,nl,1,x);
    opt.SetAsymptotes(0.3,0.65,1.08);
    double kkt=1.0;

    for(int it=0;it<300&&kkt>1e-5;++it){
        double gl=0;
        for(int j=0;j<nl;++j){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));gl+=double(x(j));}
        double g=GlobalSum(gl)/n-0.4; mfem::Vector fival(1); fival(0)=g;
        opt.Update(x,df0,0.0,fival,&dg,xmin,xmax);
        gl=0;for(int j=0;j<nl;++j){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));gl+=double(x(j));}
        g=GlobalSum(gl)/n-0.4;fival(0)=g;
        kkt=opt.KKTresidual(x,df0,0.0,fival,&dg,xmin,xmax);
    }
    double xl=0;for(int j=0;j<nl;++j)xl+=double(x(j));
    double xmean=GlobalSum(xl)/n;
    if(g_rank==0) printf("  Final: xmean=%.6f  kkt=%.2e\n",xmean,kkt);
    Check(kkt<1e-4,                 "KKT < 1e-4 with custom asymptotes");
    Check(std::abs(xmean-0.4)<0.01, "Volume fraction satisfied");
}

// ============================================================
// main
// ============================================================

// ============================================================
// Test: RedundantConstraints (parallel)
//   g2 = g0 + g1 exactly — rank-deficient dual Hessian.
// ============================================================
static void Test_RedundantConstraints()
{
    const int n = 200;
    if (g_rank==0) {
        printf("\n--- RedundantConstraints (n=%d, m=3, rank=2) ---\n", n);
        printf("  g2 = g0 + g1 (exactly linearly dependent)\n");
    }
    auto [nl, off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;
    const int n1 = n/2, n2 = n - n1;

    Vector x(nl),xmin(nl),xmax(nl),df0(nl);
    Vector dg[3]; for(int i=0;i<3;++i){dg[i].SetSize(nl);dg[i]=0.0;}
    x=0.5;xmin=0.001;xmax=1.0;
    for(int j=0;j<nl;++j){
        int g=off+j;
        if(g<n1)  dg[0](j)=real_t(1.0/n1);
        else      dg[1](j)=real_t(1.0/n2);
        dg[2](j) = dg[0](j)+dg[1](j);
    }
    double cv=std::max(1000.0,10.0*n);
    double a3[3]={0,0,0},c3[3]={cv,cv,cv},d3[3]={1,1,1};
    MMAOptimizerParallel opt(comm,nl,3,x,a3,c3,d3);
    double kkt=1.0;

    for(int it=0;it<300&&kkt>1e-5;++it){
        double s0l=0,s1l=0;
        for(int j=0;j<nl;++j){
            int g=off+j;
            df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));
            if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));
        }
        double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
        mfem::Vector fival(3); fival(0)=s0/n1-0.4; fival(1)=s1/n2-0.4; fival(2)=(s0+s1)/n-0.8;
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        s0l=s1l=0;
        for(int j=0;j<nl;++j){int g=off+j;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));}
        s0=GlobalSum(s0l);s1=GlobalSum(s1l);
        fival(0)=s0/n1-0.4;fival(1)=s1/n2-0.4;fival(2)=(s0+s1)/n-0.8;
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax);
        if(g_rank==0&&it%20==0)
            printf("  iter %3d: g=[%.3e,%.3e,%.3e] kkt=%.4e\n",
                   it,fival(0),fival(1),fival(2),kkt);
    }
    double s0l=0,s1l=0;
    for(int j=0;j<nl;++j){int g=off+j;if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));}
    double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
    double xmean=(s0+s1)/n;
    if(g_rank==0)
        printf("  Final: xmean=%.6f  m1=%.4f  m2=%.4f  kkt=%.2e  iters=%d\n",
               xmean,s0/n1,s1/n2,kkt,opt.NumIterations());
    Check(kkt<1e-4,               "KKT < 1e-4 with redundant constraints");
    Check(std::abs(s0/n1-0.4)<0.01,"Block 1 at target");
    Check(std::abs(s1/n2-0.4)<0.01,"Block 2 at target");
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if(g_rank==0)
        printf("=== MFEM MMA Parallel test suite (%d rank(s)) ===\n", nranks);

    // ── MMA ──────────────────────────────────────────────────────────────
    if(g_rank==0)
        printf("\n── MMA ──────────────────────────────────────────────────────\n");
    Test_ComplianceProxy(100, 0.4, false);
    Test_ComplianceProxy(50,  0.6, false);
    Test_TwoConstraints(500,  0.30, 0.50);
    Test_TwoConstraints(2000, 0.25, 0.60);
    Test_ThreeConstraints(2000, 0.30, 0.50, 0.40);
    Test_ParallelScale(1000);
    Test_ParallelScale(5000);
    Test_ConstraintSwitching();
    Test_HundredConstraints();
    Test_SetAsymptotes();

    Test_RedundantConstraints();

    // ── GCMMA ─────────────────────────────────────────────────────────────
    if(g_rank==0)
        printf("\n── GCMMA ────────────────────────────────────────────────────\n");
    Test_ComplianceProxy(100, 0.4, true);
    Test_ComplianceProxy(50,  0.6, true);
    Test_TwoConstraints(500,  0.30, 0.50);
    Test_ThreeConstraints(3000, 0.25, 0.45, 0.60);

    if(g_rank==0){
        printf("\n========================================\n");
        if(g_nfail==0) printf("All parallel tests PASSED.\n");
        else           printf("%d parallel test(s) FAILED.\n",g_nfail);
        printf("========================================\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
