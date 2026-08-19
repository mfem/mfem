/**
 * test_solvedense.cpp  —  Minimal reproducer for SolveDense SVD path
 *
 * Compile: g++ -O2 -o test_solvedense test_solvedense.cpp -llapack -lblas
 * Run:     ./test_solvedense
 *
 * Tests that SolveDense correctly handles the rank-1 equality Hessian
 * [[a,-a],[-a,a]] that arises in SolveDualSQ for pure equality constraints.
 */

#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>

extern "C" {
void dgesv_(int*,int*,double*,int*,int*,double*,int*,int*);
void dgelsd_(int*,int*,int*,double*,int*,double*,int*,
             double*,double*,int*,double*,int*,int*,int*);
}

// ── SolveDense (original) ─────────────────────────────────────────────────
static void SolveDense_orig(std::vector<double>& K, std::vector<double>& rhs, int m)
{
    if (m == 0) return;
    if (m == 1) { if (std::abs(K[0])<1e-300) K[0]=1e-300; rhs[0]/=K[0]; return; }
    {
        int nrhs=1, info=0;
        std::vector<int>    ipiv(m);
        std::vector<double> K_lu(K), rhs_lu(rhs);
        dgesv_(&m,&nrhs,K_lu.data(),&m,ipiv.data(),rhs_lu.data(),&m,&info);
        if (info==0) { rhs=rhs_lu; return; }
        if (info<0) throw std::runtime_error("dgesv bad arg");
    }
    {
        int nrhs=1, info=0, rank=0;
        double rcond = 2.2e-16 * m;   // ← ORIGINAL
        std::vector<double> svals(m), K_svd(K);
        int nlvl   = std::max(0,(int)std::ceil(std::log2(double(m)/25.0+1.0))+1);
        int liwork = std::max(1, 3*m*nlvl+11*m);
        std::vector<int> iwork(liwork);
        int lwork = -1; double wq;
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs.data(),&m,
                svals.data(),&rcond,&rank,&wq,&lwork,iwork.data(),&info);
        lwork = (info==0)?(int)wq:10*m*m;
        lwork = std::max(lwork,1);
        std::vector<double> work(lwork);
        K_svd = K;
        // NOTE: rhs is NOT reset here — potential issue if workspace query modifies it
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs.data(),&m,
                svals.data(),&rcond,&rank,work.data(),&lwork,iwork.data(),&info);
        printf("  [orig] dgelsd rank=%d svals=[%.3e,%.3e] rcond=%.3e\n",
               rank, svals[0], m>1?svals[1]:0.0, rcond);
    }
}

// ── SolveDense (fixed) ────────────────────────────────────────────────────
static void SolveDense_fixed(std::vector<double>& K, std::vector<double>& rhs, int m)
{
    if (m == 0) return;
    if (m == 1) { if (std::abs(K[0])<1e-300) K[0]=1e-300; rhs[0]/=K[0]; return; }
    {
        int nrhs=1, info=0;
        std::vector<int>    ipiv(m);
        std::vector<double> K_lu(K), rhs_lu(rhs);
        dgesv_(&m,&nrhs,K_lu.data(),&m,ipiv.data(),rhs_lu.data(),&m,&info);
        if (info==0) { rhs=rhs_lu; return; }
        if (info<0) throw std::runtime_error("dgesv bad arg");
    }
    {
        int nrhs=1, info=0, rank=0;
        double rcond = std::sqrt(2.2e-16);  // ← FIXED
        std::vector<double> svals(m), K_svd(K);
        std::vector<double> rhs_svd(rhs);   // ← copy rhs
        int nlvl   = std::max(0,(int)std::ceil(std::log2(double(m)/25.0+1.0))+1);
        int liwork = std::max(1, 3*m*nlvl+11*m);
        std::vector<int> iwork(liwork);
        int lwork = -1; double wq;
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs_svd.data(),&m,
                svals.data(),&rcond,&rank,&wq,&lwork,iwork.data(),&info);
        lwork = (info==0)?(int)wq:10*m*m;
        lwork = std::max(lwork,1);
        std::vector<double> work(lwork);
        K_svd = K;
        rhs_svd = rhs;   // ← reset rhs
        dgelsd_(&m,&m,&nrhs,K_svd.data(),&m,rhs_svd.data(),&m,
                svals.data(),&rcond,&rank,work.data(),&lwork,iwork.data(),&info);
        if (info==0||info>0) rhs=rhs_svd;
        printf("  [fixed] dgelsd rank=%d svals=[%.3e,%.3e] rcond=%.3e\n",
               rank, svals[0], m>1?svals[1]:0.0, rcond);
    }
}

int main()
{
    printf("=== SolveDense equality Hessian test ===\n\n");

    // The rank-1 Hessian from SolveDualSQ pure equality at iter=0
    // H = [[a,-a],[-a,a]] where a = -0.07425 (from the debug trace)
    double a = -0.07425;
    std::vector<double> H = {a, -a, -a, a};   // column-major: [[a,-a],[-a,a]]
    std::vector<double> g = {-0.397495, 0.396505};
    printf("Input H = [[%.5f,%.5f],[%.5f,%.5f]]\n", H[0],H[2],H[1],H[3]);
    printf("Input g = [%.6f, %.6f]\n", g[0], g[1]);
    printf("Expected solution: s ≈ [+2.67, -2.67]\n\n");

    // Test original
    {
        auto K=H; auto r=g;
        printf("ORIGINAL SolveDense:\n");
        printf("  rhs before query call: [%.6f, %.6f]\n", r[0], r[1]);
        SolveDense_orig(K, r, 2);
        printf("  result s = [%.4f, %.4f]  (correct: [+2.67, -2.67])\n", r[0], r[1]);
        double err = std::abs(r[0]-2.6734) + std::abs(r[1]+2.6734);
        printf("  error = %.2e  %s\n\n", err, err<0.01?"✓ CORRECT":"✗ WRONG");
    }

    // Test fixed
    {
        auto K=H; auto r=g;
        printf("FIXED SolveDense:\n");
        SolveDense_fixed(K, r, 2);
        printf("  result s = [%.4f, %.4f]  (correct: [+2.67, -2.67])\n", r[0], r[1]);
        double err = std::abs(r[0]-2.6734) + std::abs(r[1]+2.6734);
        printf("  error = %.2e  %s\n\n", err, err<0.01?"✓ CORRECT":"✗ WRONG");
    }

    // Test the larger explosion case (lam=-8e12 came from this)
    printf("=== Test with iter=1 Hessian (near-zero due to small D) ===\n");
    // At iter=1: xk=0.40, lam=[1.099,0.901], D much smaller → |a| smaller
    // This makes the near-zero eigenvalue even closer to machine zero
    double a2 = -0.01;  // approximate
    std::vector<double> H2 = {a2,-a2,-a2,a2};
    std::vector<double> g2 = {-0.001, 0.0009};
    {
        auto K=H2; auto r=g2;
        printf("ORIGINAL (a=%.4f, g=[%.4f,%.4f]):\n", a2, g2[0], g2[1]);
        SolveDense_orig(K,r,2);
        printf("  s=[%.2e, %.2e]\n", r[0], r[1]);
    }
    {
        auto K=H2; auto r=g2;
        printf("FIXED:\n");
        SolveDense_fixed(K,r,2);
        printf("  s=[%.2e, %.2e]\n", r[0], r[1]);
    }

    return 0;
}
