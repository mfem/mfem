/**
 * test_ccsa_gcmma.cpp  —  no-callback UpdateGCMMA test suite (CCSA)
 *
 * Constrained cases test the single-step/no-callback UpdateGCMMA overloads.
 * Unconstrained nonlinear cases use honest callbacks because their initial
 * entropy curvature estimate is not guaranteed conservative.  Dedicated
 * callback-loop edge cases are covered by test_ccsa_gcmma_callback.cpp.
 * Problem catalogue mirrors test_mma_gcmma.cpp exactly:
 *
 *  1. Min-max via z-variable    (n=1,  m=2)
 *  2. Two block volumes         (n=500/2000, m=2)
 *  3. Three block volumes       (n=2000/3000, m=3)
 *  4. 100 regional volumes      (n=1000, m=100)
 *  5. Constraint switching      (n=200,  m=2)   — active vs inactive detection
 *  6. Large variable count      (n=10000/50000, m=1)
 *  7. Unconstrained (m=0)       (n=10000/100000)
 *
 * Each test runs both serial (rank 0) and parallel (all ranks).
 *
 * CCSAOptimizer/CCSAOptimizerParallel work directly on the LATENT variable
 * eta: UpdateGCMMA()/KKTresidual() take/return eta, not the physical
 * design, and take no xmin/xmax. Every test builds a BoundsGeometry up
 * front, constructs the optimiser with it, seeds eta = opt.ToLatent(x0)
 * from an initial physical guess, and converts back via opt.ToPhysical(eta)
 * whenever it needs the physical point.
 *
 * Build:  cmake --build build   (links MMA_MFEM.cpp + CCSA_Bregman_MFEM.cpp)
 * Run:    ./build/test_ccsa_gcmma              (1 rank, serial + parallel)
 *         mpirun -np 4 ./build/test_ccsa_gcmma (4 ranks, parallel path exercised)
 */

#include "CCSA_Bregman_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>
#include <chrono>

using namespace mfem;
using namespace mfem_mma;
using Clock = std::chrono::steady_clock;

// ── globals ──────────────────────────────────────────────────────────────
static int g_rank  = 0;
static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// ── MPI helpers ───────────────────────────────────────────────────────────
static std::pair<int,int> Distribute(int n)
{
    int nr; MPI_Comm_size(MPI_COMM_WORLD, &nr);
    int b = n/nr, r = n%nr;
    return { b + (g_rank < r ? 1 : 0),
             g_rank*b + std::min(g_rank, r) };
}
static double GSum(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return g; }
static double GMax(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); return g; }

// ── Section banner ────────────────────────────────────────────────────────
static void Banner(const char* title)
{
    if (g_rank==0) {
        printf("\n");
        printf("══════════════════════════════════════════════\n");
        printf("  %s\n", title);
        printf("══════════════════════════════════════════════\n");
    }
}

// ── Deterministic LCG for reproducible targets ────────────────────────────
static double lcg(uint64_t& s)
{
    s = s*6364136223846793005ULL + 1442695040888963407ULL;
    return double(s>>33) / double(1ULL<<31);
}

// ============================================================
// 1. Min-max via z-variable
//    min  max{(x-2)², (x+2)²},  x ∈ [-3,3]
//    Reformulated: min z  s.t.  h1(x)-z ≤ 0,  h2(x)-z ≤ 0
//    Optimum: x*=0,  z*=4
//
//    Tests the z-variable mechanism (ai=[1,1]), which CCSAOptimizer shares
//    unchanged with MMAOptimizer (same a/c/d elastic-penalty formulation).
// ============================================================
static void Test_MinMax()
{
    Banner("Min-Max (n=1, m=2, z-variable)");

    // ── serial ────────────────────────────────────────────────────────────
    if (g_rank==0) {
        printf("  [serial]\n");
        Vector xmin(1), xmax(1), df0(1), dh1(1), dh2(1);
        xmin(0)=-3.0; xmax(0)=3.0; df0(0)=0.0;

        double ai[2]={1,1}, ci[2]={1e4,1e4}, di[2]={1,1};
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizer opt(1,2,bounds,ai,ci,di);
        Vector x0(1); x0(0)=1.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(1);
        real_t kkt=1.0; std::vector<double> lam(2);
        int inner_total=0, iters=0;

        for (int it=0;it<200&&kkt>1e-5;++it,++iters) {
            x = opt.ToPhysical(eta);
            double xv=double(x(0));
            double h1=(xv-2)*(xv-2), h2=(xv+2)*(xv+2);
            dh1(0)=real_t(2*(xv-2)); dh2(0)=real_t(2*(xv+2));
            mfem::Vector fival(2); fival(0)=real_t(h1); fival(1)=real_t(h2);
            Vector dg[2]={dh1,dh2};
            int inner=0;
            opt.UpdateGCMMA(eta,df0,0.0f,fival,dg,&inner);
            inner_total+=inner;
            x = opt.ToPhysical(eta);
            xv=double(x(0)); h1=(xv-2)*(xv-2); h2=(xv+2)*(xv+2);
            double zh=std::max(h1,h2);
            fival(0)=real_t(h1-zh); fival(1)=real_t(h2-zh);
            dh1(0)=real_t(2*(xv-2)); dh2(0)=real_t(2*(xv+2));
            kkt=opt.KKTresidual(eta,df0,0.0f,fival,dg,lam.data());
            if(it%10==0)
                printf("    iter %3d: x=%.4f  h1=%.4f  h2=%.4f  kkt=%.3e\n",
                       it,double(x(0)),h1,h2,double(kkt));
        }
        x = opt.ToPhysical(eta);
        double xf=double(x(0));
        printf("  Final: x=%.6f  h1=%.4f  h2=%.4f  kkt=%.2e"
               "  iters=%d  avg_inner=%.1f\n",
               xf,(xf-2)*(xf-2),(xf+2)*(xf+2),double(kkt),
               iters, iters>0?double(inner_total)/iters:0);
        Check(kkt<1e-4,               "[serial] KKT < 1e-4");
        Check(std::abs(xf)<0.01,      "[serial] x near 0");
        Check(std::abs((xf-2)*(xf-2)-4)<0.1,"[serial] h1 near 4");
        Check(std::abs((xf+2)*(xf+2)-4)<0.1,"[serial] h2 near 4");
    }

    // ── parallel note ─────────────────────────────────────────────────────
    // Min-max has n=1 which is smaller than the rank count in typical
    // parallel runs.  This case remains a serial showcase for the z-variable;
    // zero-local-DOF callback coverage is provided by the dedicated n=2 test
    // in test_ccsa_gcmma_callback.cpp.
    if(g_rank==0) printf("  [parallel skipped — n=1 degenerate for multi-rank]\n");
}

// ============================================================
// 2. Two / Three block volume constraints
//    min sum(1/xj)  s.t. mean(x_block_k) <= Vk
//    Analytical optimum: xj* = Vk in each block
// ============================================================
static void Test_BlockVolumes(int n, const std::vector<double>& Vtgt)
{
    const int m = (int)Vtgt.size();
    if(g_rank==0)
        printf("\n--- BlockVolumes (n=%d, m=%d, V=[", n, m);
    if(g_rank==0){
        for(int i=0;i<m;++i) printf("%.2f%s",Vtgt[i],i+1<m?",":"");
        printf("]) ---\n");
    }

    // Block boundaries: divide n into m equal-ish chunks
    std::vector<int> bstart(m+1);
    bstart[0]=0;
    for(int k=0;k<m;++k) bstart[k+1] = bstart[k] + (n-bstart[k])/(m-k);
    std::vector<int> bsz(m);
    for(int k=0;k<m;++k) bsz[k]=bstart[k+1]-bstart[k];

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Local constraint gradients
    std::vector<Vector> dg(m);
    for(int k=0;k<m;++k){dg[k].SetSize(nl);dg[k]=0.0;}
    for(int j=0;j<nl;++j){
        int g=off+j;
        int blk=0; while(blk<m-1 && g>=bstart[blk+1]) ++blk;
        dg[blk](j) = real_t(1.0/bsz[blk]);
    }

    Vector xmin(nl),xmax(nl),df0(nl);
    xmin=0.001; xmax=1.0;

    double cv=std::max(1000.0,10.0*n);
    std::vector<double> av(m,0),cv_v(m,cv),dv(m,1);

    // ── serial ─────────────────────────────────────────────────────────
    if(g_rank==0) {
        printf("  [serial]\n");
        Vector xmins(n),xmaxs(n),df0s(n);
        xmins=0.001; xmaxs=1.0;
        std::vector<Vector> dgs(m);
        for(int k=0;k<m;++k){dgs[k].SetSize(n);dgs[k]=0.0;}
        for(int j=0;j<n;++j){
            int blk=0; while(blk<m-1 && j>=bstart[blk+1]) ++blk;
            dgs[blk](j)=real_t(1.0/bsz[blk]);
        }
        BoundsGeometry bounds_s = BoundsGeometry::TwoSided(xmins, xmaxs);
        CCSAOptimizer opt(n,m,bounds_s,av.data(),cv_v.data(),dv.data());
        Vector x0s(n); x0s=0.5;
        Vector eta_s = opt.ToLatent(x0s);
        Vector xs(n);
        real_t kkt=1.0;
        for(int it=0;it<300&&kkt>1e-5;++it){
            xs = opt.ToPhysical(eta_s);
            double f0=0;
            for(int j=0;j<n;++j){double xj=double(xs(j));df0s(j)=real_t(-1.0/(xj*xj));f0+=1.0/xj;}
            mfem::Vector fi(m);
            for(int k=0;k<m;++k){
                double s=0;
                for(int j=bstart[k];j<bstart[k+1];++j) s+=double(xs(j));
                fi(k)=real_t(s/bsz[k]-Vtgt[k]);
            }
            opt.UpdateGCMMA(eta_s,df0s,real_t(f0),fi,dgs.data());
            xs = opt.ToPhysical(eta_s);
            f0=0;
            for(int j=0;j<n;++j){double xj=double(xs(j));df0s(j)=real_t(-1.0/(xj*xj));f0+=1.0/xj;}
            for(int k=0;k<m;++k){
                double s=0;
                for(int j=bstart[k];j<bstart[k+1];++j) s+=double(xs(j));
                fi(k)=real_t(s/bsz[k]-Vtgt[k]);
            }
            kkt=opt.KKTresidual(eta_s,df0s,real_t(f0),fi,dgs.data());
            if(it%30==0){
                printf("    iter %3d:", it);
                for(int k=0;k<m;++k) printf(" g%d=%.3e",k,double(fi(k)));
                printf("  kkt=%.3e\n", double(kkt));
            }
        }
        xs = opt.ToPhysical(eta_s);
        printf("  Final:");
        for(int k=0;k<m;++k){
            double s=0;
            for(int j=bstart[k];j<bstart[k+1];++j) s+=double(xs(j));
            double mean=s/bsz[k];
            printf(" m%d=%.4f(%.2f)",k,mean,Vtgt[k]);
        }
        printf("  kkt=%.2e  iters=%d\n", double(kkt), opt.GetIteration());
        Check(kkt<1e-4, "[serial] KKT < 1e-4");
        for(int k=0;k<m;++k){
            std::string tag="[serial] Block "+std::to_string(k+1)+" volume";
            double s=0;
            for(int j=bstart[k];j<bstart[k+1];++j) s+=double(xs(j));
            Check(std::abs(s/bsz[k]-Vtgt[k])<0.01, tag.c_str());
        }
    }

    // ── parallel ──────────────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    if(g_rank==0) printf("  [parallel]\n");
    {
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizerParallel opt(comm,nl,m,bounds,av.data(),cv_v.data(),dv.data());
        Vector x0(nl); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(nl);
        real_t kkt=1.0;
        for(int it=0;it<300&&kkt>1e-5;++it){
            x = opt.ToPhysical(eta);
            double f0_loc=0;
            for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));
            for(int j=0;j<nl;++j) f0_loc+=1.0/double(x(j));
            std::vector<double> sl(m,0);
            for(int j=0;j<nl;++j){
                int g=off+j, blk=0;
                while(blk<m-1&&g>=bstart[blk+1]) ++blk;
                sl[blk]+=double(x(j));
            }
            std::vector<double> sg(m);
            MPI_Allreduce(sl.data(),sg.data(),m,MPI_DOUBLE,MPI_SUM,comm);
            mfem::Vector fi(m);
            for(int k=0;k<m;++k) fi(k)=real_t(sg[k]/bsz[k]-Vtgt[k]);
            double f0=GSum(f0_loc);
            opt.UpdateGCMMA(eta,df0,real_t(f0),fi,dg.data());
            x = opt.ToPhysical(eta);
            f0_loc=0;
            for(int j=0;j<nl;++j) df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));
            for(int j=0;j<nl;++j) f0_loc+=1.0/double(x(j));
            std::fill(sl.begin(),sl.end(),0);
            for(int j=0;j<nl;++j){
                int g=off+j, blk=0;
                while(blk<m-1&&g>=bstart[blk+1]) ++blk;
                sl[blk]+=double(x(j));
            }
            MPI_Allreduce(sl.data(),sg.data(),m,MPI_DOUBLE,MPI_SUM,comm);
            for(int k=0;k<m;++k) fi(k)=real_t(sg[k]/bsz[k]-Vtgt[k]);
            f0=GSum(f0_loc);
            kkt=opt.KKTresidual(eta,df0,real_t(f0),fi,dg.data());
            if(g_rank==0&&it%30==0){
                printf("    iter %3d:", it);
                for(int k=0;k<m;++k) printf(" g%d=%.3e",k,double(fi(k)));
                printf("  kkt=%.3e\n", double(kkt));
            }
        }
        x = opt.ToPhysical(eta);
        std::vector<double> sl(m,0),sg(m);
        for(int j=0;j<nl;++j){
            int g=off+j, blk=0;
            while(blk<m-1&&g>=bstart[blk+1]) ++blk;
            sl[blk]+=double(x(j));
        }
        MPI_Allreduce(sl.data(),sg.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        if(g_rank==0){
            printf("  Final:");
            for(int k=0;k<m;++k) printf(" m%d=%.4f(%.2f)",k,sg[k]/bsz[k],Vtgt[k]);
            printf("  kkt=%.2e  iters=%d\n",double(kkt),opt.GetIteration());
        }
        Check(kkt<1e-4,"[par] KKT < 1e-4");
        for(int k=0;k<m;++k){
            std::string tag="[par] Block "+std::to_string(k+1)+" volume";
            Check(std::abs(sg[k]/bsz[k]-Vtgt[k])<0.01, tag.c_str());
        }
    }
}

// ============================================================
// 3. 100 regional volume constraints  (m=100, n=1000)
//    Even regions: V=0.3,  odd regions: V=0.6
//    Tests the m×m dual Newton system for large m.
// ============================================================
static void Test_HundredConstraints()
{
    Banner("100 Regional Volume Constraints (n=1000, m=100)");
    const int n=1000, m=100, region=n/m;
    std::vector<double> Vtgt(m);
    for(int k=0;k<m;++k) Vtgt[k]=(k%2==0)?0.3:0.6;

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    std::vector<Vector> dg(m);
    for(int k=0;k<m;++k){dg[k].SetSize(nl);dg[k]=0.0;}
    for(int j=0;j<nl;++j){int g=off+j,k=g/region;if(k<m)dg[k](j)=real_t(1.0/region);}

    Vector xmin(nl),xmax(nl),df0(nl);
    xmin=0.001; xmax=1.0;
    double cv=std::max(1000.0,10.0*n);
    std::vector<double> av(m,0),cv_v(m,cv),dv(m,1);

    // ── serial ─────────────────────────────────────────────────────────
    if(g_rank==0){
        printf("  [serial]\n");
        Vector xmins(n),xmaxs(n),df0s(n);
        xmins=0.001;xmaxs=1.0;
        std::vector<Vector> dgs(m);
        for(int k=0;k<m;++k){dgs[k].SetSize(n);dgs[k]=0.0;}
        for(int j=0;j<n;++j){int k=j/region;dgs[k](j)=real_t(1.0/region);}
        BoundsGeometry bounds_s = BoundsGeometry::TwoSided(xmins, xmaxs);
        CCSAOptimizer opt(n,m,bounds_s,av.data(),cv_v.data(),dv.data());
        Vector x0s(n); x0s=0.5;
        Vector eta_s = opt.ToLatent(x0s);
        Vector xs(n);
        real_t kkt=1.0;
        auto t0=Clock::now();
        for(int it=0;it<300&&kkt>1e-5;++it){
            xs = opt.ToPhysical(eta_s);
            double f0=0; for(int j=0;j<n;++j){double xj=double(xs(j));df0s(j)=real_t(-1.0/(xj*xj));f0+=1.0/xj;}
            mfem::Vector fi(m);
            for(int k=0;k<m;++k){
                double s=0;
                for(int j=k*region;j<(k+1)*region;++j) s+=double(xs(j));
                fi(k)=real_t(s/region-Vtgt[k]);
            }
            opt.UpdateGCMMA(eta_s,df0s,real_t(f0),fi,dgs.data());
            xs = opt.ToPhysical(eta_s);
            f0=0; for(int j=0;j<n;++j){double xj=double(xs(j));df0s(j)=real_t(-1.0/(xj*xj));f0+=1.0/xj;}
            for(int k=0;k<m;++k){
                double s=0;
                for(int j=k*region;j<(k+1)*region;++j) s+=double(xs(j));
                fi(k)=real_t(s/region-Vtgt[k]);
            }
            kkt=opt.KKTresidual(eta_s,df0s,real_t(f0),fi,dgs.data());
            if(it%20==0){
                double gmax=*std::max_element(fi.begin(),fi.end());
                printf("    iter %3d: g_max=%.3e  kkt=%.3e\n",it,double(gmax),double(kkt));
            }
        }
        xs = opt.ToPhysical(eta_s);
        double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        int nw=0; double me=0;
        for(int k=0;k<m;++k){
            double s=0;
            for(int j=k*region;j<(k+1)*region;++j) s+=double(xs(j));
            double e=std::abs(s/region-Vtgt[k]); me=std::max(me,e); if(e>0.02)++nw;
        }
        printf("  Final: kkt=%.2e  max_err=%.2e  wrong=%d/%d  iters=%d  time=%.0fms\n",
               double(kkt),me,nw,m,opt.GetIteration(),ms);
        Check(kkt<1e-4,   "[serial] KKT < 1e-4");
        Check(me<0.02,    "[serial] All 100 regions at target");
        Check(nw==0,      "[serial] No region violates target");
    }

    // ── parallel ──────────────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    if(g_rank==0) printf("  [parallel]\n");
    {
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizerParallel opt(comm,nl,m,bounds,av.data(),cv_v.data(),dv.data());
        Vector x0(nl); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(nl);
        real_t kkt=1.0;
        auto t0=Clock::now();
        for(int it=0;it<300&&kkt>1e-5;++it){
            x = opt.ToPhysical(eta);
            double f0l=0;
            std::vector<double> sll(m,0),sgl(m);
            for(int j=0;j<nl;++j){int g=off+j,k=g/region;double xj=double(x(j));df0(j)=real_t(-1.0/(xj*xj));f0l+=1.0/xj;if(k<m)sll[k]+=xj;}
            MPI_Allreduce(sll.data(),sgl.data(),m,MPI_DOUBLE,MPI_SUM,comm);
            mfem::Vector fi(m);
            for(int k=0;k<m;++k) fi(k)=real_t(sgl[k]/region-Vtgt[k]);
            double f0=GSum(f0l); opt.UpdateGCMMA(eta,df0,real_t(f0),fi,dg.data());
            x = opt.ToPhysical(eta);
            f0l=0;
            std::fill(sll.begin(),sll.end(),0);
            for(int j=0;j<nl;++j){int g=off+j,k=g/region;double xj=double(x(j));df0(j)=real_t(-1.0/(xj*xj));f0l+=1.0/xj;if(k<m)sll[k]+=xj;}
            MPI_Allreduce(sll.data(),sgl.data(),m,MPI_DOUBLE,MPI_SUM,comm);
            for(int k=0;k<m;++k) fi(k)=real_t(sgl[k]/region-Vtgt[k]);
            f0=GSum(f0l); kkt=opt.KKTresidual(eta,df0,real_t(f0),fi,dg.data());
            if(g_rank==0&&it%20==0){
                double gmax=*std::max_element(fi.begin(),fi.end());
                printf("    iter %3d: g_max=%.3e  kkt=%.3e\n",it,double(gmax),double(kkt));
            }
        }
        x = opt.ToPhysical(eta);
        double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        std::vector<double> sll(m,0),sgl(m);
        for(int j=0;j<nl;++j){int g=off+j,k=g/region;if(k<m)sll[k]+=double(x(j));}
        MPI_Allreduce(sll.data(),sgl.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        if(g_rank==0){
            int nw=0;double me=0;
            for(int k=0;k<m;++k){double e=std::abs(sgl[k]/region-Vtgt[k]);me=std::max(me,e);if(e>0.02)++nw;}
            printf("  Final: kkt=%.2e  max_err=%.2e  wrong=%d/%d  iters=%d  time=%.0fms\n",
                   double(kkt),me,nw,m,opt.GetIteration(),ms);
            Check(kkt<1e-4, "[par] KKT < 1e-4");
            Check(me<0.02,  "[par] All 100 regions at target");
            Check(nw==0,    "[par] No region violates target");
        }
    }
}

// ============================================================
// 4. Constraint switching  (active vs inactive detection)
//    Left:  min sum(1/xj)  → gradient drives x UP → g0 ACTIVE
//    Right: min sum(xj)    → gradient drives x DOWN → g1 INACTIVE
//    Tests that GCMMA correctly identifies which constraints bind.
// ============================================================
static void Test_ConstraintSwitching()
{
    Banner("Constraint Switching (n=200, m=2)");
    const int n=200, n1=n/2, n2=n-n1;
    if(g_rank==0)
        printf("  left compliance -> g0 ACTIVE;  right material -> g1 INACTIVE\n");

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    // ── serial ─────────────────────────────────────────────────────────
    if(g_rank==0){
        printf("  [serial]\n");
        Vector xmins(n),xmaxs(n),df0s(n),dg0(n),dg1(n);
        xmins=0.001;xmaxs=1.0;dg0=0.0;dg1=0.0;
        for(int j=0;j<n1;++j) dg0(j)=real_t(1.0/n1);
        for(int j=n1;j<n;++j) dg1(j)=real_t(1.0/n2);
        double cv=std::max(1000.0,10.0*n);
        double a2[2]={0,0},c2[2]={cv,cv},d2[2]={1,1};
        BoundsGeometry bounds_s = BoundsGeometry::TwoSided(xmins, xmaxs);
        CCSAOptimizer opt(n,2,bounds_s,a2,c2,d2);
        Vector x0s(n); x0s=0.5;
        Vector eta_s = opt.ToLatent(x0s);
        Vector xs(n);
        Vector dgs[2]={dg0,dg1};
        real_t kkt=1.0; std::vector<double> lam(2);
        for(int it=0;it<400&&kkt>1e-5;++it){
            xs = opt.ToPhysical(eta_s);
            double s0=0,s1=0,f0=0;
            for(int j=0;j<n;++j){
                double xj=double(xs(j));
                if(j<n1){df0s(j)=real_t(-1.0/(xj*xj));s0+=xj;f0+=1.0/xj;}
                else{df0s(j)=1.0;s1+=xj;f0+=xj;}
            }
            mfem::Vector fival(2); fival(0)=real_t(s0/n1-0.5); fival(1)=real_t(s1/n2-0.5);
            opt.UpdateGCMMA(eta_s,df0s,real_t(f0),fival,dgs);
            xs = opt.ToPhysical(eta_s);
            s0=s1=f0=0;
            for(int j=0;j<n;++j){
                double xj=double(xs(j));
                if(j<n1){df0s(j)=real_t(-1.0/(xj*xj));s0+=xj;f0+=1.0/xj;}
                else{df0s(j)=1.0;s1+=xj;f0+=xj;}
            }
            fival(0)=real_t(s0/n1-0.5);fival(1)=real_t(s1/n2-0.5);
            kkt=opt.KKTresidual(eta_s,df0s,real_t(f0),fival,dgs,lam.data());
            if(it%40==0)
                printf("    iter %3d: g=[%.4f,%.4f]  lam=[%.2e,%.2e]  kkt=%.3e\n",
                       it,double(fival(0)),double(fival(1)),lam[0],lam[1],double(kkt));
        }
        xs = opt.ToPhysical(eta_s);
        double s0=0,s1=0;
        for(int j=0;j<n;++j){if(j<n1)s0+=double(xs(j));else s1+=double(xs(j));}
        printf("  Final: mean_left=%.5f(0.50)  mean_right=%.5f(<0.1)  kkt=%.2e\n",
               s0/n1,s1/n2,double(kkt));
        Check(kkt<1e-4,                   "[serial] KKT < 1e-4");
        Check(std::abs(s0/n1-0.5)<0.01,   "[serial] Left block at 0.5");
        Check(s1/n2<0.1,                  "[serial] Right block near xmin");
    }

    // ── parallel ──────────────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    if(g_rank==0) printf("  [parallel]\n");
    {
        Vector xmin(nl),xmax(nl),df0(nl),dg0(nl),dg1(nl);
        xmin=0.001;xmax=1.0;dg0=0.0;dg1=0.0;
        for(int j=0;j<nl;++j){
            int g=off+j;
            if(g<n1) dg0(j)=real_t(1.0/n1);
            else     dg1(j)=real_t(1.0/n2);
        }
        double cv=std::max(1000.0,10.0*n);
        double a2[2]={0,0},c2[2]={cv,cv},d2[2]={1,1};
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizerParallel opt(comm,nl,2,bounds,a2,c2,d2);
        Vector x0(nl); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(nl);
        Vector dg[2]={dg0,dg1};
        real_t kkt=1.0;
        for(int it=0;it<400&&kkt>1e-5;++it){
            x = opt.ToPhysical(eta);
            double s0l=0,s1l=0,f0l=0;
            for(int j=0;j<nl;++j){
                int g=off+j;
                double xj=double(x(j));
                if(g<n1){df0(j)=real_t(-1.0/(xj*xj));s0l+=xj;f0l+=1.0/xj;}
                else{df0(j)=1.0;s1l+=xj;f0l+=xj;}
            }
            double s0=GSum(s0l),s1=GSum(s1l);
            mfem::Vector fival(2); fival(0)=real_t(s0/n1-0.5); fival(1)=real_t(s1/n2-0.5);
            double f0=GSum(f0l); opt.UpdateGCMMA(eta,df0,real_t(f0),fival,dg);
            x = opt.ToPhysical(eta);
            s0l=s1l=f0l=0;
            for(int j=0;j<nl;++j){int g=off+j;double xj=double(x(j));if(g<n1){df0(j)=real_t(-1.0/(xj*xj));s0l+=xj;f0l+=1.0/xj;}else{df0(j)=1.0;s1l+=xj;f0l+=xj;}}
            s0=GSum(s0l);s1=GSum(s1l);
            fival(0)=real_t(s0/n1-0.5);fival(1)=real_t(s1/n2-0.5);
            f0=GSum(f0l); kkt=opt.KKTresidual(eta,df0,real_t(f0),fival,dg);
            if(g_rank==0&&it%40==0){
                const auto& lam=opt.GetLambda();
                printf("    iter %3d: g=[%.4f,%.4f]  lam=[%.2e,%.2e]  kkt=%.3e\n",
                       it,double(fival(0)),double(fival(1)),lam[0],lam[1],double(kkt));
            }
        }
        x = opt.ToPhysical(eta);
        double s0l=0,s1l=0;
        for(int j=0;j<nl;++j){int g=off+j;if(g<n1)s0l+=double(x(j));else s1l+=double(x(j));}
        double s0=GSum(s0l),s1=GSum(s1l);
        if(g_rank==0)
            printf("  Final: mean_left=%.5f(0.50)  mean_right=%.5f(<0.1)  kkt=%.2e\n",
                   s0/n1,s1/n2,double(kkt));
        Check(kkt<1e-4,                   "[par] KKT < 1e-4");
        Check(std::abs(s0/n1-0.5)<0.01,   "[par] Left block at 0.5");
        Check(s1/n2<0.1,                  "[par] Right block near xmin");
    }
}

// ============================================================
// 5. Large variable count  (n = 10k / 50k,  m=1)
//    min sum(1/xj)  s.t. mean(x) <= 0.4
//    Tests throughput and convergence robustness at scale.
// ============================================================
static void Test_LargeN(int n)
{
    Banner(("Large N (n=" + std::to_string(n) + ", m=1)").c_str());

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    // ── serial ─────────────────────────────────────────────────────────
    if(g_rank==0){
        printf("  [serial]\n");
        Vector xmins(n),xmaxs(n),df0s(n),dgs(n);
        xmins=0.001;xmaxs=1.0;dgs=real_t(1.0/n);
        double cv=std::max(1000.0,10.0*n);
        double a1[1]={0},c1[1]={cv},d1[1]={1};
        BoundsGeometry bounds_s = BoundsGeometry::TwoSided(xmins, xmaxs);
        CCSAOptimizer opt(n,1,bounds_s,a1,c1,d1);
        Vector x0s(n); x0s=0.5;
        Vector eta_s = opt.ToLatent(x0s);
        Vector xs(n);
        real_t kkt=1.0;
        auto t0=Clock::now();
        for(int it=0;it<200&&kkt>1e-5;++it){
            xs = opt.ToPhysical(eta_s);
            double g=0,f0=0;
            for(int j=0;j<n;++j){double xj=double(xs(j));df0s(j)=real_t(-1.0/(xj*xj));g+=xj;f0+=1.0/xj;}
            mfem::Vector fival(1); fival(0)=real_t(g/n-0.4);
            opt.UpdateGCMMA(eta_s,df0s,real_t(f0),fival,&dgs);
            xs = opt.ToPhysical(eta_s);
            g=f0=0;for(int j=0;j<n;++j){double xj=double(xs(j));df0s(j)=real_t(-1.0/(xj*xj));g+=xj;f0+=1.0/xj;}
            fival(0)=real_t(g/n-0.4);
            kkt=opt.KKTresidual(eta_s,df0s,real_t(f0),fival,&dgs);
            if(it%20==0) printf("    iter %3d: g=%.3e  kkt=%.3e\n",it,double(fival(0)),double(kkt));
        }
        xs = opt.ToPhysical(eta_s);
        double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        double xmean=0;for(int j=0;j<n;++j)xmean+=double(xs(j));xmean/=n;
        printf("  Final: xmean=%.6f(0.40)  kkt=%.2e  iters=%d  time=%.0fms\n",
               xmean,double(kkt),opt.GetIteration(),ms);
        Check(kkt<1e-4,                   "[serial] KKT < 1e-4");
        Check(std::abs(xmean-0.4)<0.01,   "[serial] Volume fraction satisfied");
    }

    // ── parallel ──────────────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    if(g_rank==0) printf("  [parallel]\n");
    {
        Vector xmin(nl),xmax(nl),df0(nl),dg(nl);
        xmin=0.001;xmax=1.0;dg=real_t(1.0/n);
        double cv=std::max(1000.0,10.0*n);
        double a1[1]={0},c1[1]={cv},d1[1]={1};
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizerParallel opt(comm,nl,1,bounds,a1,c1,d1);
        Vector x0(nl); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(nl);
        real_t kkt=1.0;
        auto t0=Clock::now();
        for(int it=0;it<200&&kkt>1e-5;++it){
            x = opt.ToPhysical(eta);
            double gl=0,f0l=0;
            for(int j=0;j<nl;++j){double xj=double(x(j));df0(j)=real_t(-1.0/(xj*xj));gl+=xj;f0l+=1.0/xj;}
            double g=GSum(gl)/n-0.4; mfem::Vector fival(1); fival(0)=g;
            double f0=GSum(f0l); opt.UpdateGCMMA(eta,df0,real_t(f0),fival,&dg);
            x = opt.ToPhysical(eta);
            gl=f0l=0;for(int j=0;j<nl;++j){double xj=double(x(j));df0(j)=real_t(-1.0/(xj*xj));gl+=xj;f0l+=1.0/xj;}
            g=GSum(gl)/n-0.4;fival(0)=g;
            f0=GSum(f0l); kkt=opt.KKTresidual(eta,df0,real_t(f0),fival,&dg);
            if(g_rank==0&&it%20==0) printf("    iter %3d: g=%.3e  kkt=%.3e\n",it,double(fival(0)),double(kkt));
        }
        x = opt.ToPhysical(eta);
        double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        double xl=0;for(int j=0;j<nl;++j)xl+=double(x(j));
        double xmean=GSum(xl)/n;
        if(g_rank==0)
            printf("  Final: xmean=%.6f(0.40)  kkt=%.2e  iters=%d  time=%.0fms\n",
                   xmean,double(kkt),opt.GetIteration(),ms);
        Check(kkt<1e-4,                   "[par] KKT < 1e-4");
        Check(std::abs(xmean-0.4)<0.01,   "[par] Volume fraction satisfied");
    }
}

// ============================================================
// 6. Unconstrained (m=0)
//    Two objective types:
//    (a) Quadratic bowl: min sum((x-t)^2),  x* = t  (interior)
//    (b) Mixed separable: min sum(a/x + b*x),  x* = sqrt(a/b)
// ============================================================
static void Test_Unconstrained(int n)
{
    static const mfem::Vector _local_empty_fival_;
    Banner(("Unconstrained m=0 (n=" + std::to_string(n) + ")").c_str());

    auto [nl,off] = Distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Build targets / coefficients (deterministic LCG)
    uint64_t st = 12345ULL, sc = 98765ULL;
    for(int g=0;g<off;++g){ lcg(st); lcg(sc); lcg(sc); }
    Vector target(nl), alpha_v(nl), beta_v(nl), xstar(nl);
    for(int j=0;j<nl;++j){
        target(j) = real_t(0.2 + 0.6*lcg(st));
        double a   = 0.5 + 1.5*lcg(sc);
        double b   = 0.5 + 1.5*lcg(sc);
        alpha_v(j) = real_t(a);
        beta_v(j)  = real_t(b);
        xstar(j)   = real_t(std::max(0.001, std::min(1.0, std::sqrt(a/b))));
    }

    // ── (a) Quadratic bowl ─────────────────────────────────────────────
    if(g_rank==0) printf("  [serial, quadratic bowl]\n");
    if(g_rank==0){
        uint64_t ss=12345ULL;
        Vector xmins(n),xmaxs(n),df0s(n),tgts(n);
        xmins=0.001;xmaxs=1.0;
        for(int j=0;j<n;++j) tgts(j)=real_t(0.2+0.6*lcg(ss));
        BoundsGeometry bounds_s = BoundsGeometry::TwoSided(xmins, xmaxs);
        CCSAOptimizer opt(n,0,bounds_s);
        Vector x0s(n); x0s=0.5;
        Vector eta_s = opt.ToLatent(x0s);
        Vector xs(n);
        real_t kkt=1.0; double maxerr=1.0;
        for(int it=0;it<500&&maxerr>1e-4;++it){
            xs = opt.ToPhysical(eta_s);
            maxerr=0;
            for(int j=0;j<n;++j){
                double r=double(xs(j))-double(tgts(j));
                df0s(j)=real_t(2.0*r);
                maxerr=std::max(maxerr,std::abs(r));
            }
            double f0=0; for(int j=0;j<n;++j){double r=double(xs(j))-double(tgts(j));f0+=r*r;}
            kkt=opt.KKTresidual(eta_s,df0s,real_t(f0),_local_empty_fival_,nullptr);
            opt.UpdateGCMMA(eta_s,df0s,real_t(f0),_local_empty_fival_,nullptr,
                [&](const Vector& xc,Vector&,real_t& f0o){double fc=0;for(int j=0;j<n;++j){double r=double(xc(j))-double(tgts(j));fc+=r*r;}f0o=real_t(fc);},15);
            if(it%20==0) printf("    iter %3d: kkt=%.3e\n",it,double(kkt));
        }
        xs = opt.ToPhysical(eta_s);
        maxerr=0;
        for(int j=0;j<n;++j) maxerr=std::max(maxerr,std::abs(double(xs(j))-double(tgts(j))));
        printf("  Final: maxerr=%.2e  kkt=%.2e  iters=%d\n",maxerr,double(kkt),opt.GetIteration());
        Check(kkt<1e-4,    "[serial,quad] KKT < 1e-4");
        Check(maxerr<0.01, "[serial,quad] max_err < 0.01");
    }

    // ── (a) Quadratic bowl parallel ────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    if(g_rank==0) printf("  [parallel, quadratic bowl]\n");
    {
        Vector xmin(nl),xmax(nl),df0(nl);
        xmin=0.001;xmax=1.0;
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizerParallel opt(comm,nl,0,bounds);
        Vector x0(nl); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(nl);
        real_t kkt=1.0; double maxerr=1.0;
        for(int it=0;it<500&&maxerr>1e-4;++it){
            x = opt.ToPhysical(eta);
            double errl=0;
            for(int j=0;j<nl;++j){
                double r=double(x(j))-double(target(j));
                df0(j)=real_t(2.0*r);
                errl=std::max(errl,std::abs(r));
            }
            maxerr=GMax(errl);
            double f0_loc=0;
            for(int j=0;j<nl;++j){double r=double(x(j))-double(target(j));f0_loc+=r*r;}
            double f0=GSum(f0_loc);
            kkt=opt.KKTresidual(eta,df0,real_t(f0),_local_empty_fival_,nullptr);
            opt.UpdateGCMMA(eta,df0,real_t(f0),_local_empty_fival_,nullptr,
                [&](const Vector& xc,Vector&,real_t& f0o){double fl=0;for(int j=0;j<xc.Size();++j){double r=double(xc(j))-double(target(j));fl+=r*r;}f0o=real_t(GSum(fl));},15);
            if(g_rank==0&&it%20==0) printf("    iter %3d: kkt=%.3e\n",it,double(kkt));
        }
        x = opt.ToPhysical(eta);
        double errl=0;for(int j=0;j<nl;++j) errl=std::max(errl,std::abs(double(x(j))-double(target(j))));
        maxerr=GMax(errl);
        if(g_rank==0)
            printf("  Final: maxerr=%.2e  kkt=%.2e  iters=%d\n",maxerr,double(kkt),opt.GetIteration());
        Check(kkt<1e-4,    "[par,quad] KKT < 1e-4");
        Check(maxerr<0.01, "[par,quad] max_err < 0.01");
    }

    // ── (b) Mixed separable parallel ───────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    if(g_rank==0) printf("  [parallel, mixed separable f=sum(a/x+b*x)]\n");
    {
        Vector xmin(nl),xmax(nl),df0(nl);
        xmin=0.001;xmax=1.0;
        BoundsGeometry bounds = BoundsGeometry::TwoSided(xmin, xmax);
        CCSAOptimizerParallel opt(comm,nl,0,bounds);
        Vector x0(nl); x0=0.5;
        Vector eta = opt.ToLatent(x0);
        Vector x(nl);
        real_t kkt=1.0; double maxerr=1.0;
        for(int it=0;it<500&&maxerr>1e-4;++it){
            x = opt.ToPhysical(eta);
            double f0_loc=0,errl=0;
            for(int j=0;j<nl;++j){
                double xj=double(x(j)), a=double(alpha_v(j)), b=double(beta_v(j));
                double g=-a/(xj*xj)+b; df0(j)=real_t(g);
                f0_loc+=a/xj+b*xj;
                errl=std::max(errl,std::abs(xj-double(xstar(j))));
            }
            maxerr=GMax(errl);
            double f0=GSum(f0_loc);
            kkt=opt.KKTresidual(eta,df0,real_t(f0),_local_empty_fival_,nullptr);
            opt.UpdateGCMMA(eta,df0,real_t(f0),_local_empty_fival_,nullptr,
                [&](const Vector& xc,Vector&,real_t& f0o){double fl=0;for(int j=0;j<xc.Size();++j){double xj=double(xc(j));fl+=double(alpha_v(j))/xj+double(beta_v(j))*xj;}f0o=real_t(GSum(fl));},15);
            if(g_rank==0&&it%20==0) printf("    iter %3d: kkt=%.3e\n",it,double(kkt));
        }
        x = opt.ToPhysical(eta);
        double errl=0;
        for(int j=0;j<nl;++j) errl=std::max(errl,std::abs(double(x(j))-double(xstar(j))));
        maxerr=GMax(errl);
        if(g_rank==0)
            printf("  Final: maxerr=%.2e  kkt=%.2e  iters=%d\n",maxerr,double(kkt),opt.GetIteration());
        Check(kkt<1e-4,    "[par,mixed] KKT < 1e-4");
        Check(maxerr<0.02, "[par,mixed] max_err < 0.02");
    }
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
        printf("╔══════════════════════════════════════════════╗\n"
               "║   CCSA UpdateGCMMA test suite (%2d rank(s))   ║\n"
               "╚══════════════════════════════════════════════╝\n\n",
               nranks);

    // ── 1. Min-max ──────────────────────────────────────────────────────
    Test_MinMax();

    // ── 2. Two block volumes ─────────────────────────────────────────────
    Test_BlockVolumes(500,  {0.30, 0.50});
    Test_BlockVolumes(2000, {0.25, 0.60});

    // ── 3. Three block volumes ───────────────────────────────────────────
    Test_BlockVolumes(2000, {0.30, 0.50, 0.40});
    Test_BlockVolumes(3000, {0.25, 0.45, 0.60});

    // ── 4. 100 regional constraints ──────────────────────────────────────
    Test_HundredConstraints();

    // ── 5. Constraint switching ──────────────────────────────────────────
    Test_ConstraintSwitching();

    // ── 6. Large variable count ──────────────────────────────────────────
    Test_LargeN(10000);
    Test_LargeN(50000);

    // ── 7. Unconstrained (m=0) ───────────────────────────────────────────
    Test_Unconstrained(10000);
    Test_Unconstrained(100000);

    if(g_rank==0){
        printf("\n╔══════════════════════════════════════════════╗\n");
        if(g_nfail==0)
            printf("║   All GCMMA tests PASSED.                    ║\n");
        else
            printf("║   %d GCMMA test(s) FAILED.%-20s║\n",g_nfail,"");
        printf("╚══════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
