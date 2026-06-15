/**
 * @file test_lmg_serial.cpp
 * @brief Serial (CPU and GPU) LatentMirrorOptimizer test suite.
 *
 * Mirrors test_mma_serial.cpp / test_sq_serial.cpp in structure.
 * Uses BoundPartition for all primal<->latent conversions.
 * Tests are run twice: once CPU (UseDevice=false) and once GPU
 * (UseDevice=true, skipped when MFEM has no device backend).
 *
 * Tests:
 *  1.  BoxCompliance       – box [lo,hi]^n, Φ=Σ1/λᵢ, with Armijo.
 *  2.  LowerBoundOnly      – λ ≥ lo, no upper bound.
 *  3.  UpperBoundOnly      – λ ≤ hi, no lower bound; min Σ(λ−2)².
 *  4.  DiagonalMass        – TwoSided box, diagonal M≠I.
 *  5.  ArmijoCounts        – callback called; no Armijo violation.
 *  6.  HelperRoundTrip     – PrimalToLatent ∘ LatentToPrimal = id.
 *  7.  ClassifyBounds      – correct BoundType partitioning.
 *  8.  DefaultInit         – strict feasibility.
 *  9.  JacobianDiag        – positivity and FD check.
 */

#include "LatentMirrorOptimizer.hpp"
#include <mfem.hpp>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

using namespace mfem;
using namespace mfem_lmg;

// ── Utilities ─────────────────────────────────────────────────────────────

static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (cond) std::printf("  [PASS] %s\n", msg);
    else     { std::printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

static bool IsInfVal   (real_t v) { return  v >= real_t(std::numeric_limits<real_t>::infinity()); }
static bool IsNegInfVal(real_t v) { return  v <= real_t(-std::numeric_limits<real_t>::infinity()); }

static double CompliancePhi(const Vector& lam)
{
    // Must read on host; objective is host-side scalar.
    const real_t* lp = lam.HostRead();
    double phi = 0.0;
    for (int i = 0; i < lam.Size(); ++i) phi += 1.0/double(lp[i]);
    return phi;
}
static void ComplianceGrad(const Vector& lam, Vector& d)
{
    // Computed on device using forall.
    const bool ud = lam.UseDevice();
    d.UseDevice(ud);
    const real_t* lp = lam.Read();
    real_t*       dp = d.Write();
    mfem::forall_switch(ud, lam.Size(), [=] MFEM_HOST_DEVICE (int i){
        dp[i] = real_t(-1.0) / (lp[i]*lp[i]);
    });
}


// ── Test 1: BoxCompliance ─────────────────────────────────────────────────
static void Test_BoxCompliance(int n, real_t lo_val, real_t hi_val,
                                bool use_dev, const char* tag)
{
    std::printf("\n--- BoxCompliance (n=%d, lo=%.2f, hi=%.2f, %s) ---\n",
                n, double(lo_val), double(hi_val), tag);

    Vector lo(n), hi(n);
    lo.UseDevice(use_dev); hi.UseDevice(use_dev);
    lo = lo_val; hi = hi_val;

    BoundPartition part(lo, hi, lo);  // ref = lo (carries device flag)

    Vector lam_init(n); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo, hi, lam_init);

    Vector z(n); z.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z);

    LatentMirrorOptimizer opt(z, lo, hi);

    Vector lam(n), d(n);
    lam.UseDevice(use_dev); d.UseDevice(use_dev);

    double kkt = 1.0;
    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z, part, lam);
        const double phi = CompliancePhi(lam);
        ComplianceGrad(lam, d);

        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                phi_out = real_t(CompliancePhi(lt));
            });

        LatentToPrimal(z, part, lam);
        ComplianceGrad(lam, d);
        kkt = double(opt.StationarityResidual(z, d));
        if (it % 80 == 0)
            std::printf("  iter %3d: phi=%.4e  kkt=%.4e  ls=%d\n",
                        it, CompliancePhi(lam), kkt, opt.LastLineSearchSteps());
    }

    // Feasibility check on host.
    LatentToPrimal(z, part, lam);
    const real_t* lp = lam.HostRead();
    bool feasible = true;
    for (int i = 0; i < n; ++i) {
        if (!IsNegInfVal(lo_val) && lp[i] < lo_val - 1e-9) feasible = false;
        if (!IsInfVal   (hi_val) && lp[i] > hi_val + 1e-9) feasible = false;
    }
    std::printf("  Final: phi=%.4e  kkt=%.2e  iters=%d\n",
                CompliancePhi(lam), kkt, opt.NumIterations());
    Check(kkt < 1e-3,  (std::string("BoxCompliance KKT<1e-3 ")+tag).c_str());
    Check(feasible,    (std::string("BoxCompliance feasible " )+tag).c_str());
}


// ── Test 2: LowerBoundOnly ────────────────────────────────────────────────
static void Test_LowerBoundOnly(int n, bool use_dev, const char* tag)
{
    std::printf("\n--- LowerBoundOnly (n=%d, %s) ---\n", n, tag);

    const real_t lo_val = real_t(0.1);
    const real_t inf    = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(n), hi(n);
    lo.UseDevice(use_dev); hi.UseDevice(use_dev);
    lo = lo_val; hi = inf;

    BoundPartition part(lo, hi, lo);

    Vector lam_init(n); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo, hi, lam_init);
    Vector z(n); z.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z);

    LatentMirrorOptimizer opt(z, lo, hi);
    Vector lam(n), d(n);
    lam.UseDevice(use_dev); d.UseDevice(use_dev);

    double kkt = 1.0;
    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z, part, lam);
        const double phi = CompliancePhi(lam);
        ComplianceGrad(lam, d);
        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                p = real_t(CompliancePhi(lt));
            });
    }
    LatentToPrimal(z, part, lam);
    ComplianceGrad(lam, d);
    kkt = double(opt.StationarityResidual(z, d));

    const real_t* lp = lam.HostRead();
    bool feasible = true;
    for (int i = 0; i < n; ++i)
        if (double(lp[i]) < double(lo_val)-1e-9) feasible = false;

    std::printf("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(feasible,   (std::string("LowerBoundOnly feasible ")+tag).c_str());
    Check(kkt < 1e-3, (std::string("LowerBoundOnly KKT "    )+tag).c_str());
}


// ── Test 3: UpperBoundOnly ────────────────────────────────────────────────
static void Test_UpperBoundOnly(int n, bool use_dev, const char* tag)
{
    std::printf("\n--- UpperBoundOnly (n=%d, %s) ---\n", n, tag);

    const real_t hi_val = real_t(1.0);
    const real_t target = real_t(2.0);
    const real_t inf    = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(n), hi(n);
    lo.UseDevice(use_dev); hi.UseDevice(use_dev);
    lo = -inf; hi = hi_val;

    BoundPartition part(lo, hi, lo);

    Vector lam_init(n); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo, hi, lam_init);  // hi − 1
    Vector z(n); z.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z);

    LatentMirrorOptimizer opt(z, lo, hi);
    Vector lam(n), d(n);
    lam.UseDevice(use_dev); d.UseDevice(use_dev);

    double kkt = 1.0;
    for (int it = 0; it < 400 && kkt > 1e-6; ++it) {
        LatentToPrimal(z, part, lam);
        // Φ = Σ(λᵢ−target)²  on device.
        const bool ud = use_dev;
        const real_t tgt = target;
        const real_t* lp_r = lam.Read();
        real_t* dp = d.Write();
        mfem::forall_switch(ud, n, [=] MFEM_HOST_DEVICE (int i){
            dp[i] = real_t(2)*(lp_r[i]-tgt);
        });
        // Scalar phi on host.
        const real_t* lp_h = lam.HostRead();
        double phi = 0.0;
        for (int i = 0; i < n; ++i) {
            double v = double(lp_h[i])-double(target); phi += v*v;
        }
        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& p_out) {
                Vector lt(n); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                const real_t* lth = lt.HostRead();
                double lp2 = 0.0;
                for (int i=0;i<n;++i){double v=double(lth[i])-double(target);lp2+=v*v;}
                p_out = real_t(lp2);
            });
    }
    LatentToPrimal(z, part, lam);
    {
        const real_t tgt = target;
        const bool   ud  = use_dev;
        const real_t* lp_r = lam.Read();
        real_t* dp = d.Write();
        mfem::forall_switch(ud, n, [=] MFEM_HOST_DEVICE (int i){
            dp[i] = real_t(2)*(lp_r[i]-tgt);
        });
    }
    kkt = double(opt.StationarityResidual(z, d));

    const real_t* lp = lam.HostRead();
    bool feasible = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        if (double(lp[i]) > double(hi_val)+1e-9) feasible = false;
        maxerr = std::max(maxerr, std::abs(double(lp[i])-double(hi_val)));
    }
    std::printf("  Final: maxerr=%.2e  kkt=%.2e  iters=%d\n",
                maxerr, kkt, opt.NumIterations());
    Check(feasible,     (std::string("UpperBoundOnly feasible ")+tag).c_str());
    Check(maxerr < 0.01,(std::string("UpperBoundOnly at bound ")+tag).c_str());
    Check(kkt < 1e-3,   (std::string("UpperBoundOnly KKT "     )+tag).c_str());
}


// ── Test 4: DiagonalMass ──────────────────────────────────────────────────
static void Test_DiagonalMass(int n, bool use_dev, const char* tag)
{
    std::printf("\n--- DiagonalMass (n=%d, %s) ---\n", n, tag);

    Vector lo(n), hi(n);
    lo.UseDevice(use_dev); hi.UseDevice(use_dev);
    lo = real_t(1e-3); hi = real_t(1.0);

    BoundPartition part(lo, hi, lo);

    // M = diag(2); exact solve = DSmoother (Jacobi).
    SparseMatrix M_mat(n, n);
    for (int i = 0; i < n; ++i) M_mat.Add(i, i, 2.0);
    M_mat.Finalize();
    DSmoother M_inv(M_mat);

    Vector lam_init(n); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo, hi, lam_init);
    Vector z(n); z.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z);

    LatentMirrorOptimizer opt(z, lo, hi, &M_mat, &M_inv);

    Vector lam(n), d(n);
    lam.UseDevice(use_dev); d.UseDevice(use_dev);

    double kkt = 1.0;
    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z, part, lam);
        const double phi = CompliancePhi(lam);
        ComplianceGrad(lam, d);
        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                p = real_t(CompliancePhi(lt));
            });
    }
    LatentToPrimal(z, part, lam);
    ComplianceGrad(lam, d);
    kkt = double(opt.StationarityResidual(z, d));

    const real_t* lp = lam.HostRead();
    bool feasible = true;
    for (int i=0;i<n;++i)
        if(lp[i]<real_t(1e-3)-1e-9||lp[i]>real_t(1.0)+1e-9) feasible=false;

    std::printf("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(kkt < 1e-3, (std::string("DiagonalMass KKT "     )+tag).c_str());
    Check(feasible,   (std::string("DiagonalMass feasible " )+tag).c_str());
}


// ── Test 5: ArmijoCounts ─────────────────────────────────────────────────
static void Test_ArmijoCounts(bool use_dev, const char* tag)
{
    std::printf("\n--- ArmijoCounts (%s) ---\n", tag);

    const int n = 50;
    Vector lo(n), hi(n);
    lo.UseDevice(use_dev); hi.UseDevice(use_dev);
    lo = real_t(1e-3); hi = real_t(1.0);

    BoundPartition part(lo, hi, lo);

    Vector lam_init(n); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo, hi, lam_init);
    Vector z(n); z.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z);

    LatentMirrorOptimizer opt(z, lo, hi);
    opt.SetLineSearchParams(real_t(1e-4), real_t(0.5), 50);
    const real_t c1 = real_t(1e-4);

    Vector lam(n), d(n), lam_before(n);
    lam.UseDevice(use_dev); d.UseDevice(use_dev); lam_before.UseDevice(use_dev);

    int total_calls = 0, armijo_fails = 0;

    for (int it = 0; it < 60; ++it) {
        LatentToPrimal(z, part, lam);
        const double phi_before = CompliancePhi(lam);
        ComplianceGrad(lam, d);
        lam_before = lam;

        int calls_this = 0;
        opt.Update(z, d, real_t(phi_before),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                p = real_t(CompliancePhi(lt));
                ++calls_this;
            });
        total_calls += calls_this;

        LatentToPrimal(z, part, lam);
        const double phi_after = CompliancePhi(lam);

        // Armijo pairing on device.
        Vector dlam(n); dlam.UseDevice(use_dev);
        subtract(lam, lam_before, dlam);
        const double pairing = double(mfem::InnerProduct(d, dlam));
        const double rhs = phi_before + double(c1)*pairing;
        if (phi_after > rhs + 1e-10) ++armijo_fails;
    }

    std::printf("  Total eval_phi calls: %d  Armijo violations: %d\n",
                total_calls, armijo_fails);
    Check(total_calls > 0,   (std::string("ArmijoCounts callback ")+tag).c_str());
    Check(armijo_fails == 0, (std::string("ArmijoCounts no viol " )+tag).c_str());
}


// ── Test 6–9: unit tests (host-only; device-independent) ──────────────────

static void Test_HelperRoundTrip()
{
    std::printf("\n--- HelperRoundTrip ---\n");
    const int n = 8;
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(n), hi(n), lam_orig(n);
    lo(0)=-inf;        hi(0)=inf;         lam_orig(0)=real_t(1.5);
    lo(1)=-inf;        hi(1)=inf;         lam_orig(1)=real_t(-0.7);
    lo(2)=real_t(0.5); hi(2)=inf;         lam_orig(2)=real_t(1.2);
    lo(3)=real_t(-1.); hi(3)=inf;         lam_orig(3)=real_t(0.0);
    lo(4)=-inf;        hi(4)=real_t(2.0); lam_orig(4)=real_t(1.5);
    lo(5)=-inf;        hi(5)=real_t(0.0); lam_orig(5)=real_t(-0.3);
    lo(6)=real_t(0.0); hi(6)=real_t(1.0); lam_orig(6)=real_t(0.4);
    lo(7)=real_t(-2.); hi(7)=real_t(3.0); lam_orig(7)=real_t(0.8);

    BoundPartition part(lo, hi, lo);
    Vector z(n), lam_back(n);
    PrimalToLatent(lam_orig, part, z);
    LatentToPrimal(z, part, lam_back);

    const real_t* a = lam_orig.HostRead();
    const real_t* b = lam_back.HostRead();
    double maxerr = 0.0;
    for (int i=0;i<n;++i) maxerr = std::max(maxerr,std::abs(double(a[i])-double(b[i])));
    std::printf("  Max round-trip error: %.2e\n", maxerr);
    Check(maxerr < 1e-12, "RoundTrip error < 1e-12");
}

static void Test_ClassifyBounds()
{
    std::printf("\n--- ClassifyBounds ---\n");
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(4), hi(4);
    lo(0)=-inf;       hi(0)=inf;
    lo(1)=real_t(0);  hi(1)=inf;
    lo(2)=-inf;       hi(2)=real_t(1);
    lo(3)=real_t(0);  hi(3)=real_t(1);
    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);
    Check(types[0]==BoundType::Unbounded, "i=0 → Unbounded");
    Check(types[1]==BoundType::LowerOnly, "i=1 → LowerOnly");
    Check(types[2]==BoundType::UpperOnly, "i=2 → UpperOnly");
    Check(types[3]==BoundType::TwoSided,  "i=3 → TwoSided");

    // Also verify via BoundPartition counts.
    BoundPartition part(lo, hi, lo);
    Check(part.NumUnbounded()==1, "BoundPartition: 1 Unbounded");
    Check(part.NumLower()    ==1, "BoundPartition: 1 LowerOnly");
    Check(part.NumUpper()    ==1, "BoundPartition: 1 UpperOnly");
    Check(part.NumTwoSided() ==1, "BoundPartition: 1 TwoSided");
}

static void Test_DefaultInit()
{
    std::printf("\n--- DefaultPrimalInit ---\n");
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(4), hi(4), lam(4);
    lo(0)=-inf;        hi(0)=inf;
    lo(1)=real_t(2.0); hi(1)=inf;
    lo(2)=-inf;        hi(2)=real_t(-1.0);
    lo(3)=real_t(0.0); hi(3)=real_t(1.0);
    BoundPartition part(lo, hi, lo);
    DefaultPrimalInit(part, lo, hi, lam);
    const real_t* lp = lam.HostRead();
    Check(lp[0]==real_t(0),    "Unbounded → 0");
    Check(lp[1]==real_t(3.0),  "LowerOnly → l+1");
    Check(lp[2]==real_t(-2.0), "UpperOnly → u-1");
    Check(lp[3]==real_t(0.5),  "TwoSided  → (l+u)/2");
    Check(lp[1] > real_t(2.0), "LowerOnly strictly above l");
    Check(lp[2] < real_t(-1.0),"UpperOnly strictly below u");
    Check(lp[3] > real_t(0.0) && lp[3] < real_t(1.0), "TwoSided interior");
}

static void Test_JacobianDiag()
{
    std::printf("\n--- LatentJacobianDiag ---\n");
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(4), hi(4), lam(4);
    lo(0)=-inf;       hi(0)=inf;         lam(0)=real_t(1.0);
    lo(1)=real_t(0);  hi(1)=inf;         lam(1)=real_t(0.5);
    lo(2)=-inf;       hi(2)=real_t(1);   lam(2)=real_t(0.5);
    lo(3)=real_t(0);  hi(3)=real_t(1);   lam(3)=real_t(0.3);
    BoundPartition part(lo, hi, lo);
    Vector z(4), jac(4);
    PrimalToLatent(lam, part, z);
    LatentJacobianDiag(z, part, jac);

    const real_t* jp = jac.HostRead();
    bool all_pos = true;
    for (int i=0;i<4;++i) if(double(jp[i])<=0.0) all_pos=false;
    std::printf("  jac=[%.4f,%.4f,%.4f,%.4f]\n",
                double(jp[0]),double(jp[1]),double(jp[2]),double(jp[3]));
    Check(all_pos, "All Jacobian entries > 0");

    // FD check TwoSided (i=3).
    const real_t h = real_t(1e-5);
    Vector zp(4); zp=z; zp(3)+=h;
    Vector lamp(4); LatentToPrimal(zp, part, lamp);
    const real_t* zph = z.HostRead();
    const real_t* lp  = lam.HostRead();
    const real_t* lpp = lamp.HostRead();
    const real_t fd  = (lpp[3]-lp[3])/h;
    const real_t err = std::abs(double(fd-jp[3]));
    std::printf("  TwoSided FD: jac=%.6f fd=%.6f err=%.2e\n",
                double(jp[3]), double(fd), double(err));
    Check(err < 1e-4, "FD Jacobian TwoSided");
}


// ── Test: BoundarySafety ─────────────────────────────────────────────────
/**
 * Verifies that PrimalToLatent produces finite, clipped values even when
 * λ is exactly at or beyond a bound, and that LatentToPrimal + the optimizer
 * recover gracefully.
 */
static void Test_BoundarySafety()
{
    std::printf("\n--- BoundarySafety (lam at/beyond bounds) ---\n");

    const int    n   = 6;
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());

    // Mix of bound types with deliberately pathological primal values.
    Vector lo(n), hi(n), lam(n);
    lo(0) = real_t(0.0); hi(0) = inf;          // LowerOnly: λ at lower bound
    lo(1) = real_t(0.0); hi(1) = inf;          // LowerOnly: λ below lower bound
    lo(2) = -inf;        hi(2) = real_t(1.0);  // UpperOnly: λ at upper bound
    lo(3) = -inf;        hi(3) = real_t(1.0);  // UpperOnly: λ above upper bound
    lo(4) = real_t(0.0); hi(4) = real_t(1.0);  // TwoSided:  λ at lower bound
    lo(5) = real_t(0.0); hi(5) = real_t(1.0);  // TwoSided:  λ at upper bound

    lam(0) = real_t(0.0);   // exactly at lower bound → gap = 0
    lam(1) = real_t(-0.5);  // below lower bound → gap < 0
    lam(2) = real_t(1.0);   // exactly at upper bound → gap = 0
    lam(3) = real_t(1.5);   // above upper bound → gap < 0
    lam(4) = real_t(0.0);   // exactly at lower bound
    lam(5) = real_t(1.0);   // exactly at upper bound

    BoundPartition part(lo, hi, lo);

    // BoundSafetyCheck should report 6 violations (one per entry).
    const int nviol = BoundSafetyCheck(lam, part, lo, hi, real_t(0));
    std::printf("  BoundSafetyCheck violations (expect 6): %d\n", nviol);
    Check(nviol == 6, "BoundSafetyCheck counts 6 violations");

    // PrimalToLatent must not produce NaN or ±Inf.
    Vector z(n);
    PrimalToLatent(lam, part, z);
    const real_t* zp = z.HostRead();
    bool finite_z = true;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(double(zp[i]))) finite_z = false;
    }
    std::printf("  z = [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
                double(zp[0]), double(zp[1]), double(zp[2]),
                double(zp[3]), double(zp[4]), double(zp[5]));
    Check(finite_z, "PrimalToLatent: all z finite despite boundary λ");

    // z must be within the latent clipping range.
    // PrimalToLatent clamps the log argument to kPrimalMinGap = exp(-zmax),
    // so log(kPrimalMinGap) = -zmax exactly (hex float literal guarantees this).
    // Therefore |z| <= zmax holds with equality when lam is at a bound.
    const real_t zmax = real_t(sizeof(real_t) == 4 ? 15.0 : 40.0);
    bool clipped = true;
    for (int i = 0; i < n; ++i)
        if (std::abs(double(zp[i])) > double(zmax) * (1.0 + 1e-10)) clipped = false;
    Check(clipped, "PrimalToLatent: |z| <= zmax for boundary lam");

    // LatentToPrimal on clamped z must return strictly feasible λ.
    Vector lam_back(n);
    LatentToPrimal(z, part, lam_back);
    const real_t* lbp = lam_back.HostRead();
    const real_t* ld  = lo.HostRead();
    const real_t* ud  = hi.HostRead();
    bool feasible = true;
    for (int i = 0; i < n; ++i) {
        if (!IsNegInfVal(ld[i]) && lbp[i] < ld[i]) feasible = false;
        if (!IsInfVal   (ud[i]) && lbp[i] > ud[i]) feasible = false;
    }
    Check(feasible, "LatentToPrimal: λ feasible after clamped z");

    // Run a short optimization starting from the clamped z to verify
    // the optimizer itself does not crash or diverge.
    // Objective: Φ = Σ (λᵢ − mid)²  with mid chosen strictly inside each bound.
    Vector lo_box(4), hi_box(4), lam_box(4), z_box(4), d_box(4);
    lo_box = real_t(0.0); hi_box = real_t(1.0);
    // Start exactly at lower bound → tests recovery from degenerate init.
    lam_box = real_t(0.0);
    BoundPartition part_box(lo_box, hi_box, lo_box);
    PrimalToLatent(lam_box, part_box, z_box);   // safe clamp applied here

    LatentMirrorOptimizer opt(z_box, lo_box, hi_box);
    const real_t target = real_t(0.4);
    double kkt = 1.0;
    for (int it = 0; it < 200 && kkt > 1e-5; ++it) {
        LatentToPrimal(z_box, part_box, lam_box);
        double phi = 0.0;
        for (int i = 0; i < 4; ++i) {
            double v = double(lam_box(i)) - double(target);
            phi      += v * v;
            d_box(i)  = real_t(2.0 * v);
        }
        opt.Update(z_box, d_box, real_t(phi),
            [&](const Vector& zt, real_t& p) {
                Vector lt(4); LatentToPrimal(zt, part_box, lt);
                double lp2 = 0.0;
                for (int i=0;i<4;++i){double v=double(lt(i))-double(target);lp2+=v*v;}
                p = real_t(lp2);
            });
    }
    LatentToPrimal(z_box, part_box, lam_box);
    for (int i = 0; i < 4; ++i)
        d_box(i) = real_t(2.0*(double(lam_box(i))-double(target)));
    kkt = double(opt.StationarityResidual(z_box, d_box));
    std::printf("  Recovery optimization: kkt=%.2e  iters=%d\n",
                kkt, opt.NumIterations());
    Check(kkt < 1e-3, "BoundarySafety: optimizer converges after boundary init");
}


// ── main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
#ifdef MFEM_USE_MPI
    MPI_Init(&argc, &argv);
#endif

    // Determine if a GPU backend is available.
    const bool have_device = mfem::Device::IsAvailable();
    std::printf("=== LatentMirrorOptimizer serial test suite ===\n");
    std::printf("    GPU device available: %s\n\n", have_device ? "yes" : "no");

    // ── Helper unit tests (device-independent, run once) ──────────────────
    std::printf("── Helper unit tests ─────────────────────────────────────\n");
    Test_ClassifyBounds();
    Test_DefaultInit();
    Test_HelperRoundTrip();
    Test_JacobianDiag();

    // ── Optimizer tests: CPU ───────────────────────────────────────────────
    std::printf("\n── Boundary safety tests ────────────────────────────────────\n");
    Test_BoundarySafety();

    std::printf("\n── Optimizer tests (CPU) ─────────────────────────────────\n");
    Test_BoxCompliance(100, real_t(1e-3), real_t(1.0), false, "CPU");
    Test_BoxCompliance(50,  real_t(0.2),  real_t(0.8), false, "CPU-narrow");
    Test_LowerBoundOnly(80, false, "CPU");
    Test_UpperBoundOnly(80, false, "CPU");
    Test_DiagonalMass(100,  false, "CPU");
    Test_ArmijoCounts(false, "CPU");

    // ── Optimizer tests: GPU (if available) ───────────────────────────────
    if (have_device) {
        mfem::Device device("cuda");  // or "hip" / "occa-cuda" per MFEM build
        std::printf("\n── Optimizer tests (GPU) ─────────────────────────────────\n");
        Test_BoxCompliance(100, real_t(1e-3), real_t(1.0), true, "GPU");
        Test_BoxCompliance(50,  real_t(0.2),  real_t(0.8), true, "GPU-narrow");
        Test_LowerBoundOnly(80, true, "GPU");
        Test_UpperBoundOnly(80, true, "GPU");
        Test_DiagonalMass(100,  true, "GPU");
        Test_ArmijoCounts(true, "GPU");
    } else {
        std::printf("\n  (GPU tests skipped — no device backend)\n");
    }

    std::printf("\n========================================\n");
    if (g_nfail == 0) std::printf("All serial LMG tests PASSED.\n");
    else              std::printf("%d serial LMG test(s) FAILED.\n", g_nfail);
    std::printf("========================================\n");

#ifdef MFEM_USE_MPI
    MPI_Finalize();
#endif
    return g_nfail > 0 ? 1 : 0;
}
