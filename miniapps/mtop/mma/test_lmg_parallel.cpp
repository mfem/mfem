/**
 * @file test_lmg_parallel.cpp
 * @brief MPI-parallel (+ optional GPU) LatentMirrorOptimizerParallel tests.
 *
 * Tests:
 *  1.  HelperRoundTrip  – PrimalToLatent ∘ LatentToPrimal = id, all ranks.
 *  2.  BoxCompliance    – two-sided box, compliance Φ=Σ1/λᵢ, Armijo.
 *  3.  MixedBounds      – all four bound types distributed across ranks.
 *  4.  DiagonalMass     – local diagonal M≠I, global inner products.
 *  5.  ArmijoCounts     – callback called; no Armijo violation at accepted step.
 *
 * Each test is run CPU-only; if a GPU device is present tests 2–5 are
 * re-run with UseDevice(true) on all vectors.
 *
 * Compiled only when MFEM_USE_MPI is defined.
 */

#include "LatentMirrorOptimizer.hpp"

#ifdef MFEM_USE_MPI

#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

using namespace mfem;
using namespace mfem_lmg;

// ── Globals ───────────────────────────────────────────────────────────────
static int g_nfail = 0;
static int g_rank  = 0;
static int g_nranks = 1;

static bool IsInfVal   (real_t v){ return  v >= real_t(std::numeric_limits<real_t>::infinity()); }
static bool IsNegInfVal(real_t v){ return  v <= real_t(-std::numeric_limits<real_t>::infinity()); }

static void Print0(const char* fmt, ...)
{
    if (g_rank != 0) return;
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
}

/** All-rank AND reduction; failure printed on rank 0. */
static void Check(bool cond_local, const char* msg)
{
    int ok_local  = cond_local ? 1 : 0;
    int ok_global = 0;
    MPI_Allreduce(&ok_local, &ok_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (g_rank == 0) {
        if (ok_global) std::printf("  [PASS] %s\n", msg);
        else          { std::printf("  [FAIL] %s\n", msg); ++g_nfail; }
    }
    if (!ok_global) ++g_nfail;
}

static int LocalSize(int ng, int r, int nr)
{ return ng/nr + (r < ng%nr ? 1 : 0); }

// ── Objective helpers ─────────────────────────────────────────────────────

/** Global compliance Φ = Σ 1/λᵢ via MPI_Allreduce. */
static double GlobalCompliance(const Vector& lam_local)
{
    const real_t* lp = lam_local.HostRead();
    double local_phi = 0.0;
    for (int i = 0; i < lam_local.Size(); ++i) local_phi += 1.0/double(lp[i]);
    double global_phi = 0.0;
    MPI_Allreduce(&local_phi, &global_phi, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_phi;
}

/** Local compliance gradient on device. */
static void LocalComplianceGrad(const Vector& lam_local, Vector& d_local)
{
    const bool ud = lam_local.UseDevice();
    d_local.UseDevice(ud);
    const real_t* lp = lam_local.Read();
    real_t*       dp = d_local.Write();
    mfem::forall_switch(ud, lam_local.Size(), [=] MFEM_HOST_DEVICE (int i){
        dp[i] = real_t(-1.0)/(lp[i]*lp[i]);
    });
}


// ── Test 1: HelperRoundTrip ───────────────────────────────────────────────
static void Test_HelperRoundTrip()
{
    Print0("\n--- HelperRoundTrip (per-rank, 4 bound types) ---\n");

    const int    n   = 4;
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(n), hi(n), lam_orig(n);
    lo(0)=-inf;        hi(0)=inf;          lam_orig(0)=real_t(1.0+0.1*g_rank);
    lo(1)=real_t(0.5); hi(1)=inf;          lam_orig(1)=real_t(1.0+0.1*g_rank);
    lo(2)=-inf;        hi(2)=real_t(2.0);  lam_orig(2)=real_t(1.5-0.1*g_rank);
    lo(3)=real_t(0.0); hi(3)=real_t(1.0);  lam_orig(3)=real_t(0.4+0.02*g_rank);

    BoundPartition part(lo, hi, lo);
    Vector z(n), lam_back(n);
    PrimalToLatent(lam_orig, part, z);
    LatentToPrimal(z, part, lam_back);

    const real_t* a = lam_orig.HostRead();
    const real_t* b = lam_back.HostRead();
    double loc_maxerr = 0.0;
    for (int i=0;i<n;++i)
        loc_maxerr = std::max(loc_maxerr, std::abs(double(a[i])-double(b[i])));
    double glb_maxerr = 0.0;
    MPI_Allreduce(&loc_maxerr, &glb_maxerr, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    Print0("  Global max round-trip error: %.2e\n", glb_maxerr);
    Check(glb_maxerr < 1e-12, "HelperRoundTrip: error < 1e-12 all ranks");
}


// ── Test 2: BoxCompliance ─────────────────────────────────────────────────
static void Test_BoxCompliance(bool use_dev, const char* tag)
{
    Print0("\n--- BoxCompliance (%d ranks, %s) ---\n", g_nranks, tag);

    const int N_global = 100;
    const int n_loc    = LocalSize(N_global, g_rank, g_nranks);

    const real_t lo_val = real_t(0.01 + 0.01*g_rank);
    const real_t hi_val = real_t(1.0);

    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc.UseDevice(use_dev); hi_loc.UseDevice(use_dev);
    lo_loc = lo_val; hi_loc = hi_val;

    BoundPartition part(lo_loc, hi_loc, lo_loc);

    Vector lam_init(n_loc); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo_loc, hi_loc, lam_init);
    Vector z_loc(n_loc); z_loc.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);

    Vector lam_loc(n_loc), d_loc(n_loc);
    lam_loc.UseDevice(use_dev); d_loc.UseDevice(use_dev);
    double kkt = 1.0;

    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z_loc, part, lam_loc);
        const double phi = GlobalCompliance(lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);

        opt.Update(z_loc, d_loc, real_t(phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n_loc); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                phi_out = real_t(GlobalCompliance(lt));
            });

        LatentToPrimal(z_loc, part, lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
        if (it % 80 == 0)
            Print0("  iter %3d: kkt=%.4e  ls=%d\n",
                   it, kkt, opt.LastLineSearchSteps());
    }

    const real_t* lp = lam_loc.HostRead();
    bool feasible = true;
    for (int i=0;i<n_loc;++i) {
        if(lp[i] < double(lo_val)-1e-9) feasible = false;
        if(lp[i] > double(hi_val)+1e-9) feasible = false;
    }
    Print0("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(kkt < 1e-3, (std::string("BoxCompliance KKT "     )+tag).c_str());
    Check(feasible,   (std::string("BoxCompliance feasible " )+tag).c_str());
}


// ── Test 3: MixedBounds ───────────────────────────────────────────────────
static void Test_MixedBounds(bool use_dev, const char* tag)
{
    Print0("\n--- MixedBounds (%d ranks, %s) ---\n", g_nranks, tag);

    const int    n_loc = 50;
    const real_t inf   = real_t(std::numeric_limits<real_t>::infinity());

    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc.UseDevice(use_dev); hi_loc.UseDevice(use_dev);

    // Assign by rank (cycling).
    switch (g_rank % 4) {
    case 0: lo_loc = real_t(0.1);  hi_loc = real_t(2.0);  break; // TwoSided
    case 1: lo_loc = real_t(0.5);  hi_loc = inf;           break; // LowerOnly
    case 2: lo_loc = -inf;         hi_loc = real_t(3.0);   break; // UpperOnly
    case 3: lo_loc = -inf;         hi_loc = inf;            break; // Unbounded
    }

    BoundPartition part(lo_loc, hi_loc, lo_loc);

    Vector lam_init(n_loc); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo_loc, hi_loc, lam_init);

    // Nudge toward interior.
    {
        real_t* lp = lam_init.HostReadWrite();
        const real_t* ld = lo_loc.HostRead();
        const real_t* ud = hi_loc.HostRead();
        for (int i=0;i<n_loc;++i) {
            lp[i] = real_t(1.5);
            if (!IsNegInfVal(ld[i]) && lp[i] <= ld[i]) lp[i] = ld[i]+real_t(0.01);
            if (!IsInfVal   (ud[i]) && lp[i] >= ud[i]) lp[i] = ud[i]-real_t(0.01);
        }
    }

    Vector z_loc(n_loc); z_loc.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);

    Vector lam_loc(n_loc), d_loc(n_loc);
    lam_loc.UseDevice(use_dev); d_loc.UseDevice(use_dev);
    double kkt = 1.0;

    for (int it = 0; it < 400 && kkt > 1e-5; ++it) {
        LatentToPrimal(z_loc, part, lam_loc);
        // Φ = Σ(λᵢ−1)²; gradient = 2(λᵢ−1) on device.
        {
            const bool   ud = use_dev;
            const real_t* lp = lam_loc.Read();
            real_t* dp = d_loc.Write();
            mfem::forall_switch(ud, n_loc, [=] MFEM_HOST_DEVICE (int i){
                dp[i] = real_t(2)*(lp[i]-real_t(1));
            });
        }
        // Global Φ via host read + Allreduce.
        const real_t* lp_h = lam_loc.HostRead();
        double loc_phi = 0.0;
        for (int i=0;i<n_loc;++i){ double v=double(lp_h[i])-1.0; loc_phi+=v*v; }
        double glb_phi = 0.0;
        MPI_Allreduce(&loc_phi, &glb_phi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        opt.Update(z_loc, d_loc, real_t(glb_phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n_loc); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                const real_t* lth = lt.HostRead();
                double lp2 = 0.0;
                for (int i=0;i<n_loc;++i){ double v=double(lth[i])-1.0; lp2+=v*v; }
                double gp2 = 0.0;
                MPI_Allreduce(&lp2, &gp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                phi_out = real_t(gp2);
            });

        LatentToPrimal(z_loc, part, lam_loc);
        {
            const bool ud = use_dev;
            const real_t* lp = lam_loc.Read();
            real_t* dp = d_loc.Write();
            mfem::forall_switch(ud, n_loc, [=] MFEM_HOST_DEVICE (int i){
                dp[i] = real_t(2)*(lp[i]-real_t(1));
            });
        }
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
        if (it % 80 == 0) Print0("  iter %3d: kkt=%.4e\n", it, kkt);
    }

    LatentToPrimal(z_loc, part, lam_loc);
    const real_t* lp = lam_loc.HostRead();
    const real_t* ld = lo_loc.HostRead();
    const real_t* ud = hi_loc.HostRead();
    bool feasible = true;
    for (int i=0;i<n_loc;++i) {
        if (!IsNegInfVal(ld[i]) && lp[i] < ld[i]-1e-9) feasible = false;
        if (!IsInfVal   (ud[i]) && lp[i] > ud[i]+1e-9) feasible = false;
    }
    Print0("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(kkt < 1e-3, (std::string("MixedBounds KKT "     )+tag).c_str());
    Check(feasible,   (std::string("MixedBounds feasible " )+tag).c_str());
}


// ── Test 4: DiagonalMass ──────────────────────────────────────────────────
static void Test_DiagonalMass(bool use_dev, const char* tag)
{
    Print0("\n--- DiagonalMass (%d ranks, %s) ---\n", g_nranks, tag);

    const int N_global = 80;
    const int n_loc    = LocalSize(N_global, g_rank, g_nranks);

    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc.UseDevice(use_dev); hi_loc.UseDevice(use_dev);
    lo_loc = real_t(0.05); hi_loc = real_t(1.0);

    BoundPartition part(lo_loc, hi_loc, lo_loc);

    SparseMatrix M_mat(n_loc, n_loc);
    for (int i=0;i<n_loc;++i)
        M_mat.Add(i,i, 1.0+0.5*double(i)/double(std::max(n_loc-1,1)));
    M_mat.Finalize();
    DSmoother M_inv(M_mat);

    Vector lam_init(n_loc); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo_loc, hi_loc, lam_init);
    Vector z_loc(n_loc); z_loc.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc,
                                      &M_mat, &M_inv);
    Vector lam_loc(n_loc), d_loc(n_loc);
    lam_loc.UseDevice(use_dev); d_loc.UseDevice(use_dev);
    double kkt = 1.0;

    for (int it=0; it<300 && kkt>1e-5; ++it) {
        LatentToPrimal(z_loc, part, lam_loc);
        const double phi = GlobalCompliance(lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);
        opt.Update(z_loc, d_loc, real_t(phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n_loc); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                phi_out = real_t(GlobalCompliance(lt));
            });
        LatentToPrimal(z_loc, part, lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
        if (it%80==0) Print0("  iter %3d: kkt=%.4e\n", it, kkt);
    }

    const real_t* lp = lam_loc.HostRead();
    bool feasible = true;
    for (int i=0;i<n_loc;++i)
        if(lp[i]<real_t(0.05)-1e-9||lp[i]>real_t(1.0)+1e-9) feasible=false;
    Print0("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(kkt < 1e-3, (std::string("DiagonalMass KKT "     )+tag).c_str());
    Check(feasible,   (std::string("DiagonalMass feasible " )+tag).c_str());
}


// ── Test 5: ArmijoCounts ─────────────────────────────────────────────────
static void Test_ArmijoCounts(bool use_dev, const char* tag)
{
    Print0("\n--- ArmijoCounts (%d ranks, %s) ---\n", g_nranks, tag);

    const int n_loc = 30;
    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc.UseDevice(use_dev); hi_loc.UseDevice(use_dev);
    lo_loc = real_t(0.05); hi_loc = real_t(1.0);

    BoundPartition part(lo_loc, hi_loc, lo_loc);

    Vector lam_init(n_loc); lam_init.UseDevice(use_dev);
    DefaultPrimalInit(part, lo_loc, hi_loc, lam_init);
    Vector z_loc(n_loc); z_loc.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);
    opt.SetLineSearchParams(real_t(1e-4), real_t(0.5), 50);
    const real_t c1 = real_t(1e-4);

    Vector lam_loc(n_loc), d_loc(n_loc), lam_before(n_loc);
    lam_loc.UseDevice(use_dev); d_loc.UseDevice(use_dev);
    lam_before.UseDevice(use_dev);

    int total_calls = 0, armijo_fails = 0;

    for (int it=0; it<60; ++it) {
        LatentToPrimal(z_loc, part, lam_loc);
        const double phi_before = GlobalCompliance(lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);
        lam_before = lam_loc;

        int calls_this = 0;
        opt.Update(z_loc, d_loc, real_t(phi_before),
            [&](const Vector& zt, real_t& p) {
                Vector lt(n_loc); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                p = real_t(GlobalCompliance(lt));
                ++calls_this;
            });
        total_calls += calls_this;

        // Verify Armijo at accepted step.
        LatentToPrimal(z_loc, part, lam_loc);
        const double phi_after = GlobalCompliance(lam_loc);

        Vector dlam(n_loc); dlam.UseDevice(use_dev);
        subtract(lam_loc, lam_before, dlam);

        // Global pairing = Σ_ranks  dᵀ Δλ  (M=I so Mg=d).
        double loc_pair  = double(mfem::InnerProduct(d_loc, dlam));
        double glb_pair  = 0.0;
        MPI_Allreduce(&loc_pair, &glb_pair, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        const double rhs = phi_before + double(c1)*glb_pair;
        if (phi_after > rhs + 1e-10) ++armijo_fails;
    }

    Print0("  Total eval_phi calls: %d  Armijo violations: %d\n",
           total_calls, armijo_fails);
    Check(total_calls > 0,   (std::string("ArmijoCounts callback ")+tag).c_str());
    Check(armijo_fails == 0, (std::string("ArmijoCounts no viol " )+tag).c_str());
}


// ── main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    const bool have_device = mfem::Device::IsAvailable();
    Print0("=== LatentMirrorOptimizerParallel test suite ===\n");
    Print0("    Ranks: %d   GPU device: %s\n\n",
           g_nranks, have_device ? "yes" : "no");

    // Helper unit test (rank-local, no device).
    Test_HelperRoundTrip();

    // ── CPU ───────────────────────────────────────────────────────────────
    Print0("\n── Optimizer tests (CPU) ────────────────────────────────────\n");
    Test_BoxCompliance(false, "CPU");
    Test_MixedBounds  (false, "CPU");
    Test_DiagonalMass (false, "CPU");
    Test_ArmijoCounts (false, "CPU");

    // ── GPU (if available) ────────────────────────────────────────────────
    if (have_device) {
        // Enable device on rank 0 (other ranks do likewise).
        if (g_rank == 0) {
            mfem::Device device("cuda");
        }
        Print0("\n── Optimizer tests (GPU) ────────────────────────────────────\n");
        Test_BoxCompliance(true, "GPU");
        Test_MixedBounds  (true, "GPU");
        Test_DiagonalMass (true, "GPU");
        Test_ArmijoCounts (true, "GPU");
    } else {
        Print0("\n  (GPU tests skipped — no device backend)\n");
    }

    // Global failure count (de-duplicate across ranks).
    int global_nfail = 0;
    MPI_Allreduce(&g_nfail, &global_nfail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    global_nfail /= g_nranks;

    Print0("\n========================================\n");
    if (global_nfail == 0) Print0("All parallel LMG tests PASSED.\n");
    else                   Print0("%d parallel LMG test(s) FAILED.\n",
                                   global_nfail);
    Print0("========================================\n");

    MPI_Finalize();
    return global_nfail > 0 ? 1 : 0;
}

#else  // !MFEM_USE_MPI

#include <cstdio>
int main()
{
    std::printf("test_lmg_parallel: MFEM_USE_MPI not defined, skipping.\n");
    return 0;
}

#endif // MFEM_USE_MPI
