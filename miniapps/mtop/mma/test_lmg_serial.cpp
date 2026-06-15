/**
 * @file test_lmg_serial.cpp
 * @brief Serial LatentMirrorOptimizer test suite.
 *
 * Mirrors test_mma_serial.cpp / test_sq_serial.cpp in structure.
 * The user holds a latent vector z; the primal λ = T(z) is read via
 * LatentToPrimal.  The Armijo callback evaluates the true objective at
 * each trial latent point.
 *
 * Test catalogue:
 *  1.  BoxCompliance   – box [lo,hi]^n, compliance Φ=Σ1/λᵢ.
 *  2.  LowerBoundOnly  – λ ≥ lo > 0, no upper bound.
 *  3.  UpperBoundOnly  – λ ≤ hi < ∞, no lower bound; min Σ(λ−2)².
 *  4.  DiagonalMass    – TwoSided box with diagonal mass matrix M.
 *  5.  ArmijoCounts    – eval_phi called; no Armijo violation at accepted step.
 *  6.  HelperRoundTrip – PrimalToLatent ∘ LatentToPrimal = id.
 *  7.  ClassifyBounds  – correct partitioning.
 *  8.  DefaultInit     – strict feasibility.
 *  9.  JacobianDiag    – positivity and finite-difference check.
 */

#include "LatentMirrorOptimizer.hpp"
#include <mfem.hpp>
#include <cmath>
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

static bool IsInfVal(real_t v)
{ return v >= real_t(std::numeric_limits<real_t>::infinity()); }
static bool IsNegInfVal(real_t v)
{ return v <= real_t(-std::numeric_limits<real_t>::infinity()); }

// Compliance proxy:  Φ(λ) = Σᵢ 1/λᵢ,  dᵢ = −1/λᵢ²
static double CompliancePhi(const Vector& lam)
{
    double phi = 0.0;
    for (int i = 0; i < lam.Size(); ++i) phi += 1.0 / double(lam(i));
    return phi;
}
static void ComplianceGrad(const Vector& lam, Vector& d)
{
    for (int i = 0; i < lam.Size(); ++i)
        d(i) = real_t(-1.0 / (double(lam(i)) * double(lam(i))));
}


// ── Test 1: BoxCompliance ─────────────────────────────────────────────────
static void Test_BoxCompliance(int n, real_t lo_val, real_t hi_val,
                                const char* tag = "")
{
    std::printf("\n--- BoxCompliance (n=%d, lo=%.2f, hi=%.2f %s) ---\n",
                n, double(lo_val), double(hi_val), tag);

    Vector lo(n), hi(n);
    lo = lo_val; hi = hi_val;
    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);

    // Start at midpoint → latent.
    Vector lam_init(n);
    DefaultPrimalInit(lo, hi, types, lam_init);
    Vector z(n);
    PrimalToLatent(lam_init, lo, hi, types, z);

    LatentMirrorOptimizer opt(z, lo, hi);

    Vector lam(n), d(n);
    double kkt = 1.0;
    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z, lo, hi, types, lam);
        const double phi = CompliancePhi(lam);
        ComplianceGrad(lam, d);

        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n);
                LatentToPrimal(zt, lo, hi, types, lt);
                phi_out = real_t(CompliancePhi(lt));
            });

        LatentToPrimal(z, lo, hi, types, lam);
        ComplianceGrad(lam, d);
        kkt = double(opt.StationarityResidual(z, d));
        if (it % 60 == 0)
            std::printf("  iter %3d: phi=%.4e  kkt=%.4e  ls=%d\n",
                        it, CompliancePhi(lam), kkt, opt.LastLineSearchSteps());
    }

    // Feasibility.
    bool feasible = true;
    for (int i = 0; i < n; ++i) {
        if (!IsNegInfVal(lo_val) && lam(i) < lo_val - 1e-9) feasible = false;
        if (!IsInfVal   (hi_val) && lam(i) > hi_val + 1e-9) feasible = false;
    }

    std::printf("  Final: phi=%.4e  kkt=%.2e  iters=%d\n",
                CompliancePhi(lam), kkt, opt.NumIterations());
    Check(kkt < 1e-3,  (std::string("BoxCompliance KKT<1e-3 ")+tag).c_str());
    Check(feasible,    (std::string("BoxCompliance feasible ")+tag).c_str());
}


// ── Test 2: LowerBoundOnly ────────────────────────────────────────────────
static void Test_LowerBoundOnly(int n)
{
    std::printf("\n--- LowerBoundOnly (n=%d) ---\n", n);

    const real_t lo_val = real_t(0.1);
    const real_t inf    = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(n), hi(n);
    lo = lo_val; hi = inf;

    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);

    Vector lam_init(n);
    DefaultPrimalInit(lo, hi, types, lam_init);  // lo + 1
    Vector z(n);
    PrimalToLatent(lam_init, lo, hi, types, z);

    LatentMirrorOptimizer opt(z, lo, hi);

    Vector lam(n), d(n);
    double kkt = 1.0;
    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z, lo, hi, types, lam);
        const double phi = CompliancePhi(lam);
        ComplianceGrad(lam, d);
        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n); LatentToPrimal(zt, lo, hi, types, lt);
                p = real_t(CompliancePhi(lt));
            });
    }
    LatentToPrimal(z, lo, hi, types, lam);
    ComplianceGrad(lam, d);
    kkt = double(opt.StationarityResidual(z, d));

    bool feasible = true;
    for (int i = 0; i < n; ++i)
        if (double(lam(i)) < double(lo_val) - 1e-9) feasible = false;

    std::printf("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(feasible,    "LowerBoundOnly: λ ≥ lo");
    Check(kkt < 1e-3,  "LowerBoundOnly: KKT < 1e-3");
}


// ── Test 3: UpperBoundOnly ────────────────────────────────────────────────
static void Test_UpperBoundOnly(int n)
{
    std::printf("\n--- UpperBoundOnly (n=%d) ---\n", n);

    // min Σ(λᵢ−2)²  s.t. λᵢ ≤ 1.  Optimal: λ*=1.
    const real_t hi_val = real_t(1.0);
    const real_t target = real_t(2.0);
    const real_t inf    = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(n), hi(n);
    lo = -inf; hi = hi_val;

    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);
    Vector lam_init(n);
    DefaultPrimalInit(lo, hi, types, lam_init);  // hi - 1 = 0
    Vector z(n);
    PrimalToLatent(lam_init, lo, hi, types, z);

    LatentMirrorOptimizer opt(z, lo, hi);

    Vector lam(n), d(n);
    double kkt = 1.0;
    for (int it = 0; it < 400 && kkt > 1e-6; ++it) {
        LatentToPrimal(z, lo, hi, types, lam);
        double phi = 0.0;
        for (int i = 0; i < n; ++i) {
            double v = double(lam(i)) - double(target);
            phi   += v * v;
            d(i)   = real_t(2.0 * v);
        }
        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n); LatentToPrimal(zt, lo, hi, types, lt);
                double lp = 0.0;
                for (int i = 0; i < n; ++i) {
                    double v = double(lt(i)) - double(target); lp += v*v;
                }
                p = real_t(lp);
            });
    }
    LatentToPrimal(z, lo, hi, types, lam);
    for (int i = 0; i < n; ++i)
        d(i) = real_t(2.0 * (double(lam(i)) - double(target)));
    kkt = double(opt.StationarityResidual(z, d));

    bool feasible = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        if (double(lam(i)) > double(hi_val) + 1e-9) feasible = false;
        maxerr = std::max(maxerr, std::abs(double(lam(i)) - double(hi_val)));
    }
    std::printf("  Final: maxerr=%.2e  kkt=%.2e  iters=%d\n",
                maxerr, kkt, opt.NumIterations());
    Check(feasible,     "UpperBoundOnly: λ ≤ hi");
    Check(maxerr < 0.01,"UpperBoundOnly: λ* at upper bound");
    Check(kkt < 1e-3,   "UpperBoundOnly: KKT < 1e-3");
}


// ── Test 4: DiagonalMass ──────────────────────────────────────────────────
static void Test_DiagonalMass(int n)
{
    std::printf("\n--- DiagonalMass (n=%d) ---\n", n);

    Vector lo(n), hi(n);
    lo = real_t(1e-3); hi = real_t(1.0);
    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);

    // M = diag(2, ..., 2); exact solve = diag(0.5).
    SparseMatrix M_mat(n, n);
    for (int i = 0; i < n; ++i) M_mat.Add(i, i, 2.0);
    M_mat.Finalize();
    DSmoother M_inv(M_mat);

    Vector lam_init(n);
    DefaultPrimalInit(lo, hi, types, lam_init);
    Vector z(n);
    PrimalToLatent(lam_init, lo, hi, types, z);

    LatentMirrorOptimizer opt(z, lo, hi, &M_mat, &M_inv);

    Vector lam(n), d(n);
    double kkt = 1.0;
    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z, lo, hi, types, lam);
        const double phi = CompliancePhi(lam);
        ComplianceGrad(lam, d);
        opt.Update(z, d, real_t(phi),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n); LatentToPrimal(zt, lo, hi, types, lt);
                p = real_t(CompliancePhi(lt));
            });
        if (it % 60 == 0) {
            LatentToPrimal(z, lo, hi, types, lam);
            ComplianceGrad(lam, d);
            std::printf("  iter %3d: kkt=%.4e  ls=%d\n",
                        it, double(opt.StationarityResidual(z,d)),
                        opt.LastLineSearchSteps());
        }
    }
    LatentToPrimal(z, lo, hi, types, lam);
    ComplianceGrad(lam, d);
    kkt = double(opt.StationarityResidual(z, d));

    bool feasible = true;
    for (int i = 0; i < n; ++i)
        if (lam(i) < lo(i)-1e-9 || lam(i) > hi(i)+1e-9) feasible = false;

    std::printf("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(kkt < 1e-3, "DiagonalMass: KKT < 1e-3");
    Check(feasible,   "DiagonalMass: feasibility with M≠I");
}


// ── Test 5: ArmijoCounts ─────────────────────────────────────────────────
static void Test_ArmijoCounts()
{
    std::printf("\n--- ArmijoCounts (n=50) ---\n");

    const int n = 50;
    Vector lo(n), hi(n);
    lo = real_t(1e-3); hi = real_t(1.0);
    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);

    Vector lam_init(n); DefaultPrimalInit(lo, hi, types, lam_init);
    Vector z(n); PrimalToLatent(lam_init, lo, hi, types, z);

    LatentMirrorOptimizer opt(z, lo, hi);
    opt.SetLineSearchParams(real_t(1e-4), real_t(0.5), 50);
    const real_t c1 = real_t(1e-4);

    Vector lam(n), d(n);
    int total_calls  = 0;
    int armijo_fails = 0;

    for (int it = 0; it < 60; ++it) {
        LatentToPrimal(z, lo, hi, types, lam);
        const double phi_before = CompliancePhi(lam);
        ComplianceGrad(lam, d);

        Vector lam_before = lam;
        int calls_this = 0;

        opt.Update(z, d, real_t(phi_before),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n); LatentToPrimal(zt, lo, hi, types, lt);
                p = real_t(CompliancePhi(lt));
                ++calls_this;
            });
        total_calls += calls_this;

        // Check Armijo at the accepted step.
        LatentToPrimal(z, lo, hi, types, lam);
        const double phi_after = CompliancePhi(lam);
        Vector dlam(n); subtract(lam, lam_before, dlam);
        // (dᵏ)ᵀ Δλ  with dᵏ = d (M=I, so Mg = g = d).
        const double pairing = double(mfem::InnerProduct(d, dlam));
        const double rhs = phi_before + double(c1) * pairing;
        if (phi_after > rhs + 1e-10) ++armijo_fails;
    }

    std::printf("  Total eval_phi calls: %d  Armijo violations: %d\n",
                total_calls, armijo_fails);
    Check(total_calls > 0,   "ArmijoCounts: callback was called");
    Check(armijo_fails == 0, "ArmijoCounts: no Armijo violations");
}


// ── Test 6: HelperRoundTrip ───────────────────────────────────────────────
static void Test_HelperRoundTrip()
{
    std::printf("\n--- HelperRoundTrip (4 bound types) ---\n");

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

    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);
    Vector z(n), lam_back(n);
    PrimalToLatent(lam_orig, lo, hi, types, z);
    LatentToPrimal(z, lo, hi, types, lam_back);

    double maxerr = 0.0;
    for (int i = 0; i < n; ++i)
        maxerr = std::max(maxerr,
                          std::abs(double(lam_orig(i))-double(lam_back(i))));
    std::printf("  Max round-trip error: %.2e\n", maxerr);
    Check(maxerr < 1e-12, "RoundTrip error < 1e-12");
}


// ── Test 7: ClassifyBounds ────────────────────────────────────────────────
static void Test_ClassifyBounds()
{
    std::printf("\n--- ClassifyBounds ---\n");
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(4), hi(4);
    lo(0)=-inf;        hi(0)=inf;
    lo(1)=real_t(0);   hi(1)=inf;
    lo(2)=-inf;        hi(2)=real_t(1);
    lo(3)=real_t(0);   hi(3)=real_t(1);
    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);
    Check(types[0]==BoundType::Unbounded, "i=0 → Unbounded");
    Check(types[1]==BoundType::LowerOnly, "i=1 → LowerOnly");
    Check(types[2]==BoundType::UpperOnly, "i=2 → UpperOnly");
    Check(types[3]==BoundType::TwoSided,  "i=3 → TwoSided");
}


// ── Test 8: DefaultInit ───────────────────────────────────────────────────
static void Test_DefaultInit()
{
    std::printf("\n--- DefaultPrimalInit ---\n");
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(4), hi(4), lam(4);
    lo(0)=-inf;        hi(0)=inf;
    lo(1)=real_t(2.0); hi(1)=inf;
    lo(2)=-inf;        hi(2)=real_t(-1.0);
    lo(3)=real_t(0.0); hi(3)=real_t(1.0);
    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);
    DefaultPrimalInit(lo, hi, types, lam);
    Check(lam(0)==real_t(0),    "Unbounded → 0");
    Check(lam(1)==real_t(3.0),  "LowerOnly → l+1");
    Check(lam(2)==real_t(-2.0), "UpperOnly → u-1");
    Check(lam(3)==real_t(0.5),  "TwoSided  → (l+u)/2");
    Check(lam(1) > lo(1),       "LowerOnly strictly above l");
    Check(lam(2) < hi(2),       "UpperOnly strictly below u");
    Check(lam(3) > lo(3) && lam(3) < hi(3), "TwoSided strictly inside");
}


// ── Test 9: JacobianDiag ─────────────────────────────────────────────────
static void Test_JacobianDiag()
{
    std::printf("\n--- LatentJacobianDiag ---\n");
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(4), hi(4), lam(4), z(4), jac(4);
    lo(0)=-inf;        hi(0)=inf;         lam(0)=real_t(1.0);
    lo(1)=real_t(0);   hi(1)=inf;         lam(1)=real_t(0.5);
    lo(2)=-inf;        hi(2)=real_t(1);   lam(2)=real_t(0.5);
    lo(3)=real_t(0);   hi(3)=real_t(1);   lam(3)=real_t(0.3);
    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);
    PrimalToLatent(lam, lo, hi, types, z);
    LatentJacobianDiag(z, lo, hi, types, jac);

    bool all_pos = true;
    for (int i = 0; i < 4; ++i)
        if (double(jac(i)) <= 0.0) all_pos = false;
    std::printf("  jac = [%.4f, %.4f, %.4f, %.4f]\n",
                double(jac(0)),double(jac(1)),double(jac(2)),double(jac(3)));
    Check(all_pos, "All Jacobian entries strictly positive");

    // FD check for TwoSided (i=3).
    const real_t h = real_t(1e-5);
    Vector zp(4); zp = z; zp(3) += h;
    Vector lamp(4); LatentToPrimal(zp, lo, hi, types, lamp);
    const real_t fd  = (lamp(3) - lam(3)) / h;
    const real_t err = std::abs(double(fd - jac(3)));
    std::printf("  TwoSided FD check: jac=%.6f fd=%.6f err=%.2e\n",
                double(jac(3)), double(fd), double(err));
    Check(err < 1e-4, "FD Jacobian check for TwoSided");
}


// ── main ──────────────────────────────────────────────────────────────────
int main()
{
#ifdef MFEM_USE_MPI
    MPI_Init(nullptr, nullptr);
#endif

    std::printf("=== LatentMirrorOptimizer serial test suite ===\n\n");

    std::printf("── Helper unit tests ────────────────────────────────────\n");
    Test_ClassifyBounds();
    Test_DefaultInit();
    Test_HelperRoundTrip();
    Test_JacobianDiag();

    std::printf("\n── Optimizer tests ──────────────────────────────────────\n");
    Test_BoxCompliance(100, real_t(1e-3), real_t(1.0), "(standard)");
    Test_BoxCompliance(50,  real_t(0.2),  real_t(0.8), "(narrower bounds)");
    Test_LowerBoundOnly(80);
    Test_UpperBoundOnly(80);
    Test_DiagonalMass(100);
    Test_ArmijoCounts();

    std::printf("\n========================================\n");
    if (g_nfail == 0) std::printf("All serial LMG tests PASSED.\n");
    else              std::printf("%d serial LMG test(s) FAILED.\n", g_nfail);
    std::printf("========================================\n");

#ifdef MFEM_USE_MPI
    MPI_Finalize();
#endif
    return g_nfail > 0 ? 1 : 0;
}
