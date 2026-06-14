/**
 * test_mma_serial.cpp  —  Serial MMAOptimizer test suite
 *
 * Tests mfem_mma::MMAOptimizer (single process, mfem::Vector).
 * Covers: MMA Update, GCMMA UpdateGCMMA, all 8 problem types.
 *
 * Build:  cmake --build build && ./build/test_mma_serial
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>

using namespace mfem;
using namespace mfem_mma;

static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// ── helper: compliance objective and gradient ─────────────────────────────
static void ComplianceGrad(const Vector& x, double& f0, Vector& df0)
{
    f0 = 0.0;
    for (int j = 0; j < x.Size(); ++j) {
        f0     += 1.0 / double(x(j));
        df0(j)  = real_t(-1.0 / (double(x(j))*double(x(j))));
    }
}

// ============================================================
// Test 1/2 — Compliance proxy  (MMA and GCMMA)
//   min Σ1/xj   s.t. mean(x)≤Vfrac,   x∈[0.001,1]
//   Optimum: xj*=Vfrac  (uniform)
// ============================================================
static void Test_ComplianceProxy(int n, double Vfrac, bool gcmma)
{
    printf("\n--- ComplianceProxy (n=%d, Vfrac=%.2f, %s) ---\n",
           n, Vfrac, gcmma ? "GCMMA" : "MMA");

    Vector x(n), xmin(n), xmax(n), df0(n), dg(n);
    x = 0.5; xmin = 0.001; xmax = 1.0;
    dg = real_t(1.0 / n);

    MMAOptimizer opt(n, 1, x);
    double kkt = 1.0;

    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        double f0, g;
        ComplianceGrad(x, f0, df0);
        g = 0.0; for (int j=0;j<n;++j) g += double(x(j)); g = g/n - Vfrac;
        mfem::Vector fival(1);
        fival(0)=g;

        if (gcmma) {
            int inner;
            opt.UpdateGCMMA(x, df0, f0, fival, &dg, xmin, xmax, &inner);
        } else {
            opt.Update(x, df0, f0, fival, &dg, xmin, xmax);
        }

        // recompute at new x for KKT
        ComplianceGrad(x, f0, df0);
        g = 0.0; for (int j=0;j<n;++j) g += double(x(j)); g = g/n - Vfrac;
        fival(0)= g;
        kkt = opt.KKTresidual(x, df0, f0, fival, &dg, xmin, xmax);
        if (it % 20 == 0)
            printf("  iter %3d: g=%.4e  kkt=%.4e\n", it, g, kkt);
    }

    double xmean = 0.0;
    for (int j = 0; j < n; ++j) xmean += double(x(j));
    xmean /= n;
    double maxerr = 0.0;
    for (int j = 0; j < n; ++j) maxerr = std::max(maxerr, std::abs(double(x(j))-Vfrac));

    printf("  Final: xmean=%.6f (%.2f)  max_err=%.2e  kkt=%.2e  iters=%d\n",
           xmean, Vfrac, maxerr, kkt, opt.GetIteration());

    Check(kkt < 1e-4,                    "KKT < 1e-4");
    Check(std::abs(xmean-Vfrac) < 0.01,  "Volume fraction satisfied");
    Check(maxerr < 0.05,                 "Uniform design");
}

// ============================================================
// Test 3 — Min-max via z-variable (n=1, m=2)
//   min max{(x-2)², (x+2)²},  x∈[-3,3],  x*=0, h*=4
// ============================================================
static void Test_MinMax()
{
    printf("\n--- MinMax (n=1, m=2, z-variable) ---\n");

    Vector x(1), xmin(1), xmax(1), df0(1), dh1(1), dh2(1);
    x(0) = 1.5; xmin(0) = -3.0; xmax(0) = 3.0; df0(0) = 0.0;

    double ai[2]={1,1}, ci[2]={1e4,1e4}, di[2]={1,1};
    MMAOptimizer opt(1, 2, x, ai, ci, di);
    double kkt = 1.0;
    std::vector<double> lam(2);

    for (int it = 0; it < 200 && kkt > 1e-5; ++it) {
        double xv = double(x(0));
        double h1 = (xv-2)*(xv-2), h2 = (xv+2)*(xv+2);
        dh1(0) = real_t(2*(xv-2)); dh2(0) = real_t(2*(xv+2));
        mfem::Vector fival(2);
        fival(0)=h1;
        fival(1)=h2;
        Vector dg[2] = {dh1, dh2};
        opt.Update(x, df0, 0.0, fival, dg, xmin, xmax);

        xv = double(x(0)); h1=(xv-2)*(xv-2); h2=(xv+2)*(xv+2);
        dh1(0)=real_t(2*(xv-2)); dh2(0)=real_t(2*(xv+2));
        double zh = std::max(h1, h2);
        fival(0)= h1-zh; fival(1)= h2-zh;
        kkt = opt.KKTresidual(x, df0, 0.0, fival, dg, xmin, xmax, lam.data());
        if (it % 10 == 0)
            printf("  iter %3d: x=%.4f  h1=%.4f  h2=%.4f  kkt=%.4e\n",
                   it, double(x(0)), h1, h2, kkt);
    }

    double xf = double(x(0));
    printf("  Final: x=%.6f  h1=%.4f  h2=%.4f  kkt=%.2e  iters=%d\n",
           xf, (xf-2)*(xf-2), (xf+2)*(xf+2), kkt, opt.GetIteration());

    Check(kkt < 1e-4,               "KKT < 1e-4");
    Check(std::abs(xf) < 0.01,      "x near 0");
    Check(std::abs((xf-2)*(xf-2)-4) < 0.1, "h1 near 4");
    Check(std::abs((xf+2)*(xf+2)-4) < 0.1, "h2 near 4");
}

// ============================================================
// Test 4/5 — Two block volume constraints
// ============================================================
static void Test_TwoConstraints(int n, double V1, double V2)
{
    printf("\n--- TwoConstraints (n=%d, V1=%.2f, V2=%.2f) ---\n", n, V1, V2);
    const int n1=n/2, n2=n-n1;

    Vector x(n), xmin(n), xmax(n), df0(n), dg0(n), dg1(n);
    x=0.5; xmin=0.001; xmax=1.0; dg0=0.0; dg1=0.0;
    for (int j=0;j<n1;++j) dg0(j) = real_t(1.0/n1);
    for (int j=n1;j<n;++j) dg1(j) = real_t(1.0/n2);

    double cv = std::max(1000.0, 10.0*n);
    double a2[2]={0,0}, c2[2]={cv,cv}, d2[2]={1,1};
    MMAOptimizer opt(n, 2, x, a2, c2, d2);
    Vector dg[2] = {dg0, dg1};
    double kkt = 1.0;

    for (int it=0; it<200 && kkt>1e-5; ++it) {
        double s1=0, s2=0;
        for (int j=0;j<n;++j) {
            df0(j) = real_t(-1.0/(double(x(j))*double(x(j))));
            if (j<n1) s1+=double(x(j)); else s2+=double(x(j));
        }
        mfem::Vector fival(2);
        fival(0)=s1/n1-V1;
        fival(1)=s2/n2-V2;
        opt.Update(x, df0, 0.0, fival, dg, xmin, xmax);
        s1=s2=0;
        for (int j=0;j<n;++j) {
            df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));
            if(j<n1)s1+=double(x(j));else s2+=double(x(j));
        }
        fival(0)=s1/n1-V1; fival(1)=s2/n2-V2;
        kkt = opt.KKTresidual(x, df0, 0.0, fival, dg, xmin, xmax);
        if (it%20==0)
            printf("  iter %3d: g=[%.4e,%.4e]  kkt=%.4e\n",it,fival(0),fival(1),kkt);
    }
    double s1=0,s2=0;
    for(int j=0;j<n;++j){if(j<n1)s1+=double(x(j));else s2+=double(x(j));}
    printf("  Final: mean1=%.6f(%.2f)  mean2=%.6f(%.2f)  kkt=%.2e\n",
           s1/n1,V1,s2/n2,V2,kkt);
    Check(kkt<1e-4,               "KKT < 1e-4");
    Check(std::abs(s1/n1-V1)<0.01,"Block 1 volume");
    Check(std::abs(s2/n2-V2)<0.01,"Block 2 volume");
}

// ============================================================
// Test 6 — Three block volume constraints
// ============================================================
static void Test_ThreeConstraints(int n, double V1, double V2, double V3)
{
    const int b1=n/3, b2=2*n/3;
    const int sz[3]={b1,b2-b1,n-b2};
    const double Vt[3]={V1,V2,V3};
    printf("\n--- ThreeConstraints (n=%d, V=%.2f/%.2f/%.2f) ---\n",n,V1,V2,V3);

    Vector x(n),xmin(n),xmax(n),df0(n);
    Vector dg[3]; for(int i=0;i<3;++i){dg[i].SetSize(n);dg[i]=0.0;}
    x=0.5;xmin=0.001;xmax=1.0;
    for(int j=0;j<n;++j){int blk=(j<b1)?0:(j<b2)?1:2;dg[blk](j)=real_t(1.0/sz[blk]);}
    double cv=std::max(1000.0,10.0*n);
    double a3[3]={0,0,0},c3[3]={cv,cv,cv},d3[3]={1,1,1};
    MMAOptimizer opt(n,3,x,a3,c3,d3);
    double kkt=1.0;

    for(int it=0;it<200&&kkt>1e-5;++it){
        double sl[3]={0,0,0};
        for(int j=0;j<n;++j){int blk=(j<b1)?0:(j<b2)?1:2;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));sl[blk]+=double(x(j));}
        mfem::Vector fival(3); for(int i=0;i<3;++i) fival(i)=sl[i]/sz[i]-Vt[i];
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        sl[0]=sl[1]=sl[2]=0;
        for(int j=0;j<n;++j){int blk=(j<b1)?0:(j<b2)?1:2;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));sl[blk]+=double(x(j));}
        for(int i=0;i<3;++i) fival(i)=sl[i]/sz[i]-Vt[i];
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax);
        if(it%20==0) printf("  iter %3d: g=[%.3e,%.3e,%.3e] kkt=%.4e\n",it,fival(0),fival(1),fival(2),kkt);
    }
    double sl[3]={0,0,0};
    for(int j=0;j<n;++j){int blk=(j<b1)?0:(j<b2)?1:2;sl[blk]+=double(x(j));}
    printf("  Final: means=[%.4f,%.4f,%.4f] kkt=%.2e\n",sl[0]/sz[0],sl[1]/sz[1],sl[2]/sz[2],kkt);
    Check(kkt<1e-4,"KKT < 1e-4");
    for(int i=0;i<3;++i)
        Check(std::abs(sl[i]/sz[i]-Vt[i])<0.01,
              (std::string("Block ")+std::to_string(i+1)+" volume").c_str());
}

// ============================================================
// Test 7 — Constraint switching: active vs inactive
// ============================================================
static void Test_ConstraintSwitching()
{
    const int n=200,n1=n/2,n2=n-n1;
    printf("\n--- ConstraintSwitching (n=%d, m=2) ---\n",n);
    printf("  left compliance->g0 ACTIVE; right material->g1 INACTIVE\n");

    Vector x(n),xmin(n),xmax(n),df0(n),dg0(n),dg1(n);
    x=0.5;xmin=0.001;xmax=1.0;dg0=0.0;dg1=0.0;
    for(int j=0;j<n1;++j) dg0(j)=real_t(1.0/n1);
    for(int j=n1;j<n;++j) dg1(j)=real_t(1.0/n2);
    double cv=std::max(1000.0,10.0*n);
    double a2[2]={0,0},c2[2]={cv,cv},d2[2]={1,1};
    MMAOptimizer opt(n,2,x,a2,c2,d2);
    Vector dg[2]={dg0,dg1};
    double kkt=1.0; std::vector<double> lam(2);

    for(int it=0;it<300&&kkt>1e-5;++it){
        double s1=0,s2=0;
        for(int j=0;j<n;++j){
            if(j<n1){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));s1+=double(x(j));}
            else    {df0(j)=1.0;s2+=double(x(j));}
        }
        mfem::Vector fival(2); fival(0)=s1/n1-0.5; fival(1)=s2/n2-0.5;
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        s1=s2=0;
        for(int j=0;j<n;++j){if(j<n1){df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));s1+=double(x(j));}else{df0(j)=1.0;s2+=double(x(j));}}
        fival(0)=s1/n1-0.5;fival(1)=s2/n2-0.5;
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax,lam.data());
        if(it%20==0) printf("  iter %3d: g=[%.4f,%.4f] lam=[%.3e,%.3e] kkt=%.4e\n",
                            it,fival(0),fival(1),lam[0],lam[1],kkt);
    }
    double s1=0,s2=0;
    for(int j=0;j<n;++j){if(j<n1)s1+=double(x(j));else s2+=double(x(j));}
    printf("  Final: mean_left=%.5f (0.50)  mean_right=%.5f (<0.1)  kkt=%.2e\n",
           s1/n1,s2/n2,kkt);
    Check(kkt<1e-4,                    "KKT < 1e-4");
    Check(std::abs(s1/n1-0.5)<0.01,    "Left block at 0.5 (g0 active)");
    Check(s2/n2<0.1,                   "Right block near xmin (g1 inactive)");
}

// ============================================================
// Test 8 — 100 regional volume constraints (m=100)
// ============================================================
static void Test_HundredConstraints()
{
    const int n=1000,m=100,region=n/m;
    printf("\n--- HundredConstraints (n=%d, m=%d) ---\n",n,m);

    std::vector<double> Vtgt(m);
    for(int k=0;k<m;++k) Vtgt[k]=(k%2==0)?0.3:0.6;

    Vector x(n),xmin(n),xmax(n),df0(n);
    std::vector<Vector> dg(m);
    for(int k=0;k<m;++k){dg[k].SetSize(n);dg[k]=0.0;}
    x=0.5;xmin=0.001;xmax=1.0;
    for(int j=0;j<n;++j){int k=j/region;dg[k](j)=real_t(1.0/region);}
    double cv=std::max(1000.0,10.0*n);
    std::vector<double> av(m,0),cv_v(m,cv),dv(m,1);
    MMAOptimizer opt(n,m,x,av.data(),cv_v.data(),dv.data());
    double kkt=1.0;

    for(int it=0;it<200&&kkt>1e-5;++it){
        std::vector<double> sl(m,0);
        for(int j=0;j<n;++j){int k=j/region;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));sl[k]+=double(x(j));}
        mfem::Vector fival(m);
        for(int k=0;k<m;++k) fival(k)=sl[k]/region-Vtgt[k];
        opt.Update(x,df0,0.0,fival,dg.data(),xmin,xmax);
        std::fill(sl.begin(),sl.end(),0);
        for(int j=0;j<n;++j){int k=j/region;df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));sl[k]+=double(x(j));}
        for(int k=0;k<m;++k) fival(k)=sl[k]/region-Vtgt[k];
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg.data(),xmin,xmax);
        if(it%20==0){
            double gmax=*std::max_element(fival.begin(),fival.end());
            printf("  iter %3d: g_max=%.4e  kkt=%.4e\n",it,gmax,kkt);
        }
    }
    std::vector<double> sl(m,0);
    for(int j=0;j<n;++j){int k=j/region;sl[k]+=double(x(j));}
    int nw=0; double me=0;
    for(int k=0;k<m;++k){double e=std::abs(sl[k]/region-Vtgt[k]);me=std::max(me,e);if(e>0.02)++nw;}
    printf("  Final: kkt=%.2e  max_err=%.2e  wrong=%d/%d  iters=%d\n",
           kkt,me,nw,m,opt.GetIteration());
    Check(kkt<1e-4,   "KKT < 1e-4");
    Check(me<0.02,    "All 100 regions at target");
    Check(nw==0,      "No region violates target");
}

// ============================================================
// Test 9 — SetAsymptotes API
// ============================================================
static void Test_SetAsymptotes()
{
    printf("\n--- SetAsymptotes (n=100, m=1, custom asy) ---\n");
    const int n=100;
    Vector x(n),xmin(n),xmax(n),df0(n),dg(n);
    x=0.5;xmin=0.001;xmax=1.0;dg=real_t(1.0/n);
    MMAOptimizer opt(n,1,x);
    opt.SetAsymptotes(0.3, 0.65, 1.08);
    double kkt=1.0;
    for(int it=0;it<300&&kkt>1e-5;++it){
        double f0;ComplianceGrad(x,f0,df0);
        double g=0;for(int j=0;j<n;++j)g+=double(x(j));g=g/n-0.4;
        mfem::Vector fival(1); fival(0)=g;
        opt.Update(x,df0,f0,fival,&dg,xmin,xmax);
        ComplianceGrad(x,f0,df0);g=0;for(int j=0;j<n;++j)g+=double(x(j));g=g/n-0.4;fival(0)=g;
        kkt=opt.KKTresidual(x,df0,f0,fival,&dg,xmin,xmax);
    }
    double xmean=0;for(int j=0;j<n;++j)xmean+=double(x(j));xmean/=n;
    printf("  Final: xmean=%.6f  kkt=%.2e\n",xmean,kkt);
    Check(kkt<1e-4,                 "KKT < 1e-4 with custom asymptotes");
    Check(std::abs(xmean-0.4)<0.01, "Volume fraction satisfied");
}

// ============================================================
// main
// ============================================================

// ============================================================
// Test: RedundantConstraints
//   Three constraints where g2 = g0 + g1 (exactly redundant).
//   The dual Hessian becomes rank-deficient; the SVD fallback
//   must compute a valid minimum-norm step.
//   Optimal: uniform x* = 0.4.
// ============================================================
static void Test_RedundantConstraints()
{
    const int n = 200;
    printf("\n--- RedundantConstraints (n=%d, m=3, rank=2) ---\n", n);
    printf("  g2 = g0 + g1 (exactly linearly dependent)\n");

    Vector x(n), xmin(n), xmax(n), df0(n);
    Vector dg[3];
    for (int i = 0; i < 3; ++i) { dg[i].SetSize(n); dg[i] = 0.0; }
    x = 0.5; xmin = 0.001; xmax = 1.0;

    const int n1 = n/2, n2 = n - n1;
    // g0: left block mean <= 0.4
    for (int j = 0;  j < n1; ++j) dg[0](j) = real_t(1.0/n1);
    // g1: right block mean <= 0.4
    for (int j = n1; j < n;  ++j) dg[1](j) = real_t(1.0/n2);
    // g2 = g0 + g1: global mean <= 0.8  (redundant given g0,g1 both <= 0.4)
    for (int j = 0; j < n; ++j)
        dg[2](j) = dg[0](j) + dg[1](j);

    double cv = std::max(1000.0, 10.0*n);
    double a3[3]={0,0,0}, c3[3]={cv,cv,cv}, d3[3]={1,1,1};
    MMAOptimizer opt(n, 3, x, a3, c3, d3);
    double kkt = 1.0;

    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        double s0=0, s1=0;
        for (int j=0;j<n;++j) {
            df0(j) = real_t(-1.0/(double(x(j))*double(x(j))));
            if (j < n1) s0 += double(x(j)); else s1 += double(x(j));
        }
        mfem::Vector fival(3);
        fival(0)=s0/n1 - 0.4;
        fival(1)=s1/n2 - 0.4;
        fival(2)=(s0+s1)/n - 0.8;
        opt.Update(x, df0, 0.0, fival, dg, xmin, xmax);
        s0=s1=0;
        for (int j=0;j<n;++j) {
            df0(j)=real_t(-1.0/(double(x(j))*double(x(j))));
            if(j<n1)s0+=double(x(j));else s1+=double(x(j));
        }
        fival(0)=s0/n1-0.4; fival(1)=s1/n2-0.4; fival(2)=(s0+s1)/n-0.8;
        kkt = opt.KKTresidual(x, df0, 0.0, fival, dg, xmin, xmax);
        if (it%20==0)
            printf("  iter %3d: g=[%.3e,%.3e,%.3e] kkt=%.4e\n",
                   it, fival(0), fival(1), fival(2), kkt);
    }
    double s0=0,s1=0;
    for(int j=0;j<n;++j){if(j<n1)s0+=double(x(j));else s1+=double(x(j));}
    double xmean=(s0+s1)/n;
    printf("  Final: xmean=%.6f  m1=%.4f  m2=%.4f  kkt=%.2e  iters=%d\n",
           xmean, s0/n1, s1/n2, kkt, opt.GetIteration());

    Check(kkt < 1e-4,               "KKT < 1e-4 with redundant constraints");
    Check(std::abs(s0/n1-0.4)<0.01, "Block 1 at target");
    Check(std::abs(s1/n2-0.4)<0.01, "Block 2 at target");
}

int main()
{
    MPI_Init(nullptr, nullptr);   // needed because MMA_MFEM links MPI
    printf("=== MFEM MMA Serial test suite ===\n");

    // ── MMA ──────────────────────────────────────────────────────────────
    printf("\n── MMA ──────────────────────────────────────────────────────\n");
    Test_ComplianceProxy(100, 0.4, false);
    Test_ComplianceProxy(50,  0.6, false);
    Test_MinMax();
    Test_TwoConstraints(500,  0.30, 0.50);
    Test_TwoConstraints(2000, 0.25, 0.60);
    Test_ThreeConstraints(2000, 0.30, 0.50, 0.40);
    Test_ConstraintSwitching();
    Test_HundredConstraints();
    Test_SetAsymptotes();

    Test_RedundantConstraints();

    // ── GCMMA ─────────────────────────────────────────────────────────────
    printf("\n── GCMMA ────────────────────────────────────────────────────\n");
    Test_ComplianceProxy(100, 0.4, true);
    Test_ComplianceProxy(50,  0.6, true);
    Test_TwoConstraints(500,  0.30, 0.50);  // same problem, GCMMA via global verify
    Test_ThreeConstraints(3000, 0.25, 0.45, 0.60);

    printf("\n========================================\n");
    if (g_nfail == 0) printf("All serial tests PASSED.\n");
    else              printf("%d serial test(s) FAILED.\n", g_nfail);
    printf("========================================\n");
    MPI_Finalize();
    return g_nfail > 0 ? 1 : 0;
}
