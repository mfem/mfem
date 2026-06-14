/**
 * test_mma_mfem.cpp  —  Tests for serial MMAOptimizer and
 *                        parallel MMAOptimizerParallel
 *
 * Build:
 *   cmake -B build -DMFEM_BUILD_DIR=<path>
 *   cmake --build build
 *   # Serial:
 *   ./build/test_mma_mfem
 *   # Parallel (requires MFEM built with MPI):
 *   mpirun -np 4 ./build/test_mma_mfem
 *
 * Tests mirror test_mma.cpp (PETSc suite) so numbers are comparable.
 * Each problem is run twice: once with MMAOptimizer (serial),
 * once with MMAOptimizerParallel (distributes DOFs across MPI ranks).
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

// ── globals ──────────────────────────────────────────────────────────────────
static int g_nfail = 0;
static int g_rank  = 0;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Distribute n_global DOFs across MPI_COMM_WORLD ranks.
// Returns (n_local, global_offset) for this rank.
static std::pair<int,int> distribute(int n_global)
{
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    int base = n_global / nranks;
    int rem  = n_global % nranks;
    int n_local = base + (g_rank < rem ? 1 : 0);
    int offset  = g_rank * base + std::min(g_rank, rem);
    return {n_local, offset};
}

// Global sum of a local double
static double GlobalSum(double v)
{
    double g; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return g;
}

// ============================================================
// Test: ComplianceProxy
//   min Σ 1/xj   s.t. mean(x) ≤ Vfrac,   xj ∈ [0.001,1]
//   Optimum: xj* = Vfrac
// ============================================================
static void Test_ComplianceProxy(int n, double Vfrac)
{
    if (g_rank==0) printf("\n--- ComplianceProxy (n=%d, Vfrac=%.2f) ---\n",n,Vfrac);

    auto [n_local, offset] = distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    // ── serial run (rank 0 only builds full problem) ──────────────────────
    if (g_rank == 0) {
        Vector x(n), xmin(n), xmax(n), df0(n), dg(n);
        x=0.5; xmin=0.001; xmax=1.0; dg=1.0/(real_t)n;
        MMAOptimizer opt(n, 1, x);
        double kkt=1.0;
        for (int it=0; it<300&&kkt>1e-5; ++it) {
            double g=0;
            for (int j=0;j<n;++j){df0(j)=-1.0/(x(j)*x(j));g+=x(j);}
            g=g/(double)n-Vfrac; mfem::Vector fival(1); fival(0)=g;
            opt.Update(x,df0,0.0,fival,&dg,xmin,xmax);
            for (int j=0;j<n;++j){df0(j)=-1.0/(x(j)*x(j));} g=0;
            for (int j=0;j<n;++j) g+=x(j); g=g/(double)n-Vfrac; fival(0)=g;
            kkt=opt.KKTresidual(x,df0,0.0,fival,&dg,xmin,xmax);
            if(it%20==0) printf("  [serial] iter %3d: g=%.3e kkt=%.3e\n",it,g,kkt);
        }
        double xmean=0; for(int j=0;j<n;++j) xmean+=x(j); xmean/=(double)n;
        printf("  [serial] Final: xmean=%.6f kkt=%.2e iters=%d\n",xmean,kkt,opt.GetIteration());
        Check(kkt<1e-4,                      "[serial] KKT < 1e-4");
        Check(std::abs(xmean-Vfrac)<0.01,    "[serial] Volume fraction satisfied");
    }

    // ── parallel run ─────────────────────────────────────────────────────
    Vector x_loc(n_local), xmin_loc(n_local), xmax_loc(n_local);
    Vector df0_loc(n_local), dg_loc(n_local);
    x_loc=0.5; xmin_loc=0.001; xmax_loc=1.0; dg_loc=1.0/(real_t)n;

    MMAOptimizerParallel opt(comm, n_local, 1, x_loc);
    double kkt=1.0;
    for (int it=0; it<300&&kkt>1e-5; ++it) {
        double gl=0;
        for(int j=0;j<n_local;++j){df0_loc(j)=-1.0/(x_loc(j)*x_loc(j));gl+=x_loc(j);}
        double gs=GlobalSum(gl); double g=gs/(double)n-Vfrac;
        mfem::Vector fival(1); fival(0)=g;
        opt.Update(x_loc,df0_loc,0.0,fival,&dg_loc,xmin_loc,xmax_loc);
        gl=0; for(int j=0;j<n_local;++j){df0_loc(j)=-1.0/(x_loc(j)*x_loc(j));gl+=x_loc(j);}
        gs=GlobalSum(gl); g=gs/(double)n-Vfrac; fival(0)=g;
        kkt=opt.KKTresidual(x_loc,df0_loc,0.0,fival,&dg_loc,xmin_loc,xmax_loc);
        if(g_rank==0&&it%20==0) printf("  [par  ] iter %3d: g=%.3e kkt=%.3e\n",it,g,kkt);
    }
    double xsum_loc=0; for(int j=0;j<n_local;++j) xsum_loc+=x_loc(j);
    double xmean=GlobalSum(xsum_loc)/(double)n;
    if(g_rank==0){
        printf("  [par  ] Final: xmean=%.6f kkt=%.2e iters=%d\n",xmean,kkt,opt.GetIteration());
        Check(kkt<1e-4,                   "[parallel] KKT < 1e-4");
        Check(std::abs(xmean-Vfrac)<0.01, "[parallel] Volume fraction satisfied");
    }
}

// ============================================================
// Test: TwoConstraints
//   min Σ1/xj,  g0:mean(left)≤V1,  g1:mean(right)≤V2
// ============================================================
static void Test_TwoConstraints(int n, double V1, double V2)
{
    if(g_rank==0) printf("\n--- TwoConstraints (n=%d, V1=%.2f, V2=%.2f) ---\n",n,V1,V2);

    auto [n_local, offset] = distribute(n);
    const int n1=n/2, n2=n-n1;
    MPI_Comm comm = MPI_COMM_WORLD;

    // local constraint gradients
    Vector dg0_loc(n_local), dg1_loc(n_local);
    dg0_loc=0.0; dg1_loc=0.0;
    for(int j=0;j<n_local;++j){
        int g=offset+j;
        if(g<n1) dg0_loc(j)=1.0/(real_t)n1;
        else     dg1_loc(j)=1.0/(real_t)n2;
    }

    Vector x(n_local), xmin(n_local), xmax(n_local), df0(n_local);
    x=0.5; xmin=0.001; xmax=1.0;
    Vector dg[2]={dg0_loc,dg1_loc};

    double cv=std::max(1000.0,10.0*n);
    double a2[2]={0,0},c2[2]={cv,cv},d2[2]={1,1};
    MMAOptimizerParallel opt(comm,n_local,2,x,a2,c2,d2);
    double kkt=1.0;

    for(int it=0;it<200&&kkt>1e-5;++it){
        double s0l=0,s1l=0;
        for(int j=0;j<n_local;++j){
            int g=offset+j; df0(j)=-1.0/(x(j)*x(j));
            if(g<n1) s0l+=x(j); else s1l+=x(j);
        }
        double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
        mfem::Vector fival(2); fival(0)=s0/(double)n1-V1; fival(1)=s1/(double)n2-V2;
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        s0l=s1l=0;
        for(int j=0;j<n_local;++j){int g=offset+j;df0(j)=-1.0/(x(j)*x(j));if(g<n1)s0l+=x(j);else s1l+=x(j);}
        s0=GlobalSum(s0l);s1=GlobalSum(s1l);
        fival(0)=s0/(double)n1-V1;fival(1)=s1/(double)n2-V2;
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax);
        if(g_rank==0&&it%20==0) printf("  iter %3d: g=[%.3e,%.3e] kkt=%.3e\n",it,fival(0),fival(1),kkt);
    }
    double s0l=0,s1l=0;
    for(int j=0;j<n_local;++j){int g=offset+j;if(g<n1)s0l+=x(j);else s1l+=x(j);}
    double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
    if(g_rank==0){
        printf("  Final: mean1=%.6f(%.2f) mean2=%.6f(%.2f) kkt=%.2e\n",
               s0/(double)n1,V1,s1/(double)n2,V2,kkt);
        Check(kkt<1e-4,"[par] KKT < 1e-4");
        Check(std::abs(s0/(double)n1-V1)<0.01,"[par] Block 1 volume");
        Check(std::abs(s1/(double)n2-V2)<0.01,"[par] Block 2 volume");
    }
}

// ============================================================
// Test: ParallelScale  —  large n, single constraint
// ============================================================
static void Test_ParallelScale(int n)
{
    if(g_rank==0) printf("\n--- ParallelScale (n=%d, m=1) ---\n",n);

    auto [n_local, offset] = distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector x(n_local),xmin(n_local),xmax(n_local),df0(n_local),dg(n_local);
    x=0.5;xmin=0.001;xmax=1.0;dg=1.0/(real_t)n;
    double cv=std::max(1000.0,10.0*n);
    double a1[1]={0},c1[1]={cv},d1[1]={1};
    MMAOptimizerParallel opt(comm,n_local,1,x,a1,c1,d1);
    double kkt=1.0;

    for(int it=0;it<100&&kkt>1e-5;++it){
        double gl=0;
        for(int j=0;j<n_local;++j){df0(j)=-1.0/(x(j)*x(j));gl+=x(j);}
        double gs=GlobalSum(gl);double g=gs/(double)n-0.4;
        mfem::Vector fival(1); fival(0)=g;
        opt.Update(x,df0,0.0,fival,&dg,xmin,xmax);
        gl=0;for(int j=0;j<n_local;++j){df0(j)=-1.0/(x(j)*x(j));gl+=x(j);}
        gs=GlobalSum(gl);g=gs/(double)n-0.4;fival(0)=g;
        kkt=opt.KKTresidual(x,df0,0.0,fival,&dg,xmin,xmax);
        if(g_rank==0&&it%10==0) printf("  iter %3d: g=%.3e kkt=%.3e\n",it,g,kkt);
    }
    double xl=0;for(int j=0;j<n_local;++j) xl+=x(j);
    double xmean=GlobalSum(xl)/(double)n;
    if(g_rank==0){
        printf("  Final: xmean=%.6f kkt=%.2e iters=%d\n",xmean,kkt,opt.GetIteration());
        Check(kkt<1e-4,"[par] KKT < 1e-4");
        Check(std::abs(xmean-0.4)<0.01,"[par] Volume fraction satisfied");
    }
}

// ============================================================
// Test: HundredConstraints  —  100 regional volumes
// ============================================================
static void Test_HundredConstraints()
{
    const int n=1000,m=100,region=n/m;
    if(g_rank==0) printf("\n--- HundredConstraints (n=%d, m=%d) ---\n",n,m);

    auto [n_local, offset] = distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    std::vector<double> Vtgt(m); for(int k=0;k<m;++k) Vtgt[k]=(k%2==0)?0.3:0.6;

    // local constraint gradients
    std::vector<Vector> dg(m);
    for(int k=0;k<m;++k){ dg[k].SetSize(n_local); dg[k]=0.0; }
    for(int j=0;j<n_local;++j){
        int g=offset+j, k=g/region;
        if(k<m) dg[k](j)=1.0/(real_t)region;
    }

    Vector x(n_local),xmin(n_local),xmax(n_local),df0(n_local);
    x=0.5;xmin=0.001;xmax=1.0;
    double cv=std::max(1000.0,10.0*n);
    std::vector<double> av(m,0),cv_v(m,cv),dv(m,1);
    MMAOptimizerParallel opt(comm,n_local,m,x,av.data(),cv_v.data(),dv.data());
    double kkt=1.0;

    for(int it=0;it<200&&kkt>1e-5;++it){
        std::vector<double> sl(m,0);
        for(int j=0;j<n_local;++j){
            int g=offset+j,k=g/region;
            df0(j)=-1.0/(x(j)*x(j));
            if(k<m) sl[k]+=x(j);
        }
        std::vector<double> sg(m);
        MPI_Allreduce(sl.data(),sg.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        mfem::Vector fival(m);
        for(int k=0;k<m;++k) fival(k)=sg[k]/(double)region-Vtgt[k];
        opt.Update(x,df0,0.0,fival,dg.data(),xmin,xmax);
        std::fill(sl.begin(),sl.end(),0);
        for(int j=0;j<n_local;++j){int g=offset+j,k=g/region;df0(j)=-1.0/(x(j)*x(j));if(k<m)sl[k]+=x(j);}
        MPI_Allreduce(sl.data(),sg.data(),m,MPI_DOUBLE,MPI_SUM,comm);
        for(int k=0;k<m;++k) fival(k)=sg[k]/(double)region-Vtgt[k];
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg.data(),xmin,xmax);
        if(g_rank==0&&it%20==0){
            double gmax=*std::max_element(fival.begin(),fival.end());
            printf("  iter %3d: g_max=%.3e kkt=%.3e\n",it,gmax,kkt);
        }
    }
    std::vector<double> sl(m,0),sg(m);
    for(int j=0;j<n_local;++j){int g=offset+j,k=g/region;if(k<m)sl[k]+=x(j);}
    MPI_Allreduce(sl.data(),sg.data(),m,MPI_DOUBLE,MPI_SUM,comm);
    if(g_rank==0){
        int nw=0; double me=0;
        for(int k=0;k<m;++k){double e=std::abs(sg[k]/(double)region-Vtgt[k]);me=std::max(me,e);if(e>0.02)++nw;}
        printf("  Final: kkt=%.2e max_err=%.2e wrong=%d/%d iters=%d\n",kkt,me,nw,m,opt.GetIteration());
        Check(kkt<1e-4,"[par] KKT < 1e-4");
        Check(me<0.02, "[par] All 100 regions at target");
        Check(nw==0,   "[par] No region violates target");
    }
}

// ============================================================
// Test: ConstraintSwitching  —  active vs inactive detection
// ============================================================
static void Test_ConstraintSwitching()
{
    const int n=200,n1=n/2,n2=n-n1;
    if(g_rank==0){
        printf("\n--- ConstraintSwitching (n=%d, m=2) ---\n",n);
        printf("  left compliance->g0 ACTIVE; right material->g1 INACTIVE\n");
    }

    auto [n_local, offset] = distribute(n);
    MPI_Comm comm = MPI_COMM_WORLD;

    Vector dg0(n_local),dg1(n_local); dg0=0.0;dg1=0.0;
    for(int j=0;j<n_local;++j){
        int g=offset+j;
        if(g<n1) dg0(j)=1.0/(real_t)n1;
        else     dg1(j)=1.0/(real_t)n2;
    }
    Vector x(n_local),xmin(n_local),xmax(n_local),df0(n_local);
    x=0.5;xmin=0.001;xmax=1.0;
    Vector dg[2]={dg0,dg1};
    double cv=std::max(1000.0,10.0*n);
    double a2[2]={0,0},c2[2]={cv,cv},d2[2]={1,1};
    MMAOptimizerParallel opt(comm,n_local,2,x,a2,c2,d2);
    double kkt=1.0;

    for(int it=0;it<300&&kkt>1e-5;++it){
        double s0l=0,s1l=0;
        for(int j=0;j<n_local;++j){
            int g=offset+j;
            if(g<n1){df0(j)=-1.0/(x(j)*x(j));s0l+=x(j);}
            else    {df0(j)=1.0;               s1l+=x(j);}
        }
        double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
        mfem::Vector fival(2); fival(0)=s0/(double)n1-0.5; fival(1)=s1/(double)n2-0.5;
        opt.Update(x,df0,0.0,fival,dg,xmin,xmax);
        s0l=s1l=0;
        for(int j=0;j<n_local;++j){int g=offset+j;if(g<n1){df0(j)=-1.0/(x(j)*x(j));s0l+=x(j);}else{df0(j)=1.0;s1l+=x(j);}}
        s0=GlobalSum(s0l);s1=GlobalSum(s1l);
        fival(0)=s0/(double)n1-0.5;fival(1)=s1/(double)n2-0.5;
        kkt=opt.KKTresidual(x,df0,0.0,fival,dg,xmin,xmax);
        if(g_rank==0&&it%20==0){
            const auto& lam=opt.GetLambda();
            printf("  iter %3d: g=[%.4f,%.4f] lam=[%.3e,%.3e] kkt=%.3e\n",
                   it,fival(0),fival(1),lam[0],lam[1],kkt);
        }
    }
    double s0l=0,s1l=0;
    for(int j=0;j<n_local;++j){int g=offset+j;if(g<n1)s0l+=x(j);else s1l+=x(j);}
    double s0=GlobalSum(s0l),s1=GlobalSum(s1l);
    if(g_rank==0){
        printf("  Final: mean_left=%.5f (0.50) mean_right=%.5f (<0.1) kkt=%.2e\n",
               s0/(double)n1,s1/(double)n2,kkt);
        Check(kkt<1e-4,                       "[par] KKT < 1e-4");
        Check(std::abs(s0/(double)n1-0.5)<0.01,"[par] Left block at 0.5");
        Check(s1/(double)n2<0.1,               "[par] Right block near xmin");
    }
}

// ============================================================
// Test: SetAsymptotes  API
// ============================================================
static void Test_SetAsymptotes()
{
    if(g_rank==0) printf("\n--- SetAsymptotes (serial, n=100) ---\n");
    if(g_rank!=0) return;

    const int n=100;
    Vector x(n),xmin(n),xmax(n),df0(n),dg(n);
    x=0.5;xmin=0.001;xmax=1.0;dg=1.0/(real_t)n;
    MMAOptimizer opt(n,1,x);
    opt.SetAsymptotes(0.3,0.65,1.08);
    double kkt=1.0;
    for(int it=0;it<300&&kkt>1e-5;++it){
        double g=0;
        for(int j=0;j<n;++j){df0(j)=-1.0/(x(j)*x(j));g+=x(j);}
        g=g/(double)n-0.4; mfem::Vector fival(1); fival(0)=g;
        opt.Update(x,df0,0.0,fival,&dg,xmin,xmax);
        g=0;for(int j=0;j<n;++j){df0(j)=-1.0/(x(j)*x(j));g+=x(j);}
        g=g/(double)n-0.4;fival(0)=g;
        kkt=opt.KKTresidual(x,df0,0.0,fival,&dg,xmin,xmax);
    }
    double xmean=0;for(int j=0;j<n;++j) xmean+=x(j);xmean/=(double)n;
    printf("  Final: xmean=%.6f kkt=%.2e\n",xmean,kkt);
    Check(kkt<1e-4,                    "[serial] KKT < 1e-4 with custom asymptotes");
    Check(std::abs(xmean-0.4)<0.01,    "[serial] Volume fraction satisfied");
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if(g_rank==0) printf("=== MFEM MMA test suite  (%d rank(s)) ===\n\n", nranks);

    Test_ComplianceProxy(100, 0.4);
    Test_ComplianceProxy(50,  0.6);
    Test_TwoConstraints(500,  0.3, 0.5);
    Test_TwoConstraints(2000, 0.25, 0.6);
    Test_ParallelScale(1000);
    Test_ParallelScale(5000);
    Test_ConstraintSwitching();
    Test_HundredConstraints();
    Test_SetAsymptotes();

    if(g_rank==0){
        printf("\n========================================\n");
        if(g_nfail==0) printf("All tests PASSED.\n");
        else           printf("%d test(s) FAILED.\n",g_nfail);
        printf("========================================\n");
    }
    MPI_Finalize();
    return g_nfail>0?1:0;
}
