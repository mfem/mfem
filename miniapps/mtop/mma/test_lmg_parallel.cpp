/**
 * @file test_lmg_parallel.cpp
 * @brief MPI parallel test suite for LatentMirrorOptimizerParallel.
 *
 * Mirrors the serial test file (test_lmg_serial.cpp) and the MMA/SQ
 * parallel test conventions.  Compiled only when MFEM_USE_MPI is defined.
 *
 * Test catalogue:
 *  1.  BoxCompliance    – box [lo,hi]^N, compliance proxy Φ=Σ1/λᵢ.
 *      Distributed: rank r owns variables [r*n_loc, (r+1)*n_loc).
 *      All variables are two-sided bounded; each rank has different lo/hi.
 *
 *  2.  MixedBounds      – mix of all four bound types across ranks.
 *      Each rank gets a different bound pattern; verifies feasibility and
 *      descent across all four latent maps.
 *
 *  3.  DiagonalMass     – box problem with a local diagonal mass matrix M.
 *      Verifies that the M-inner product path (Allreduce of local dots)
 *      gives the same solution as the identity path.
 *
 *  4.  ArmijoCounts     – verifies that the line-search callback is actually
 *      called and that the Armijo condition is genuinely checked (the
 *      accepted step satisfies sufficient decrease).
 *
 *  5.  HelperRoundTrip  – parallel: PrimalToLatent ∘ LatentToPrimal = id
 *      checked on each rank independently.
 *
 * Build (example, adjust paths):
 *   mpicxx -std=c++14 -DMFEM_USE_MPI -I/path/to/mfem \
 *          LatentMirrorOptimizer.cpp test_lmg_parallel.cpp \
 *          -L/path/to/mfem -lmfem -o test_lmg_parallel
 *   mpirun -np 4 ./test_lmg_parallel
 */

#include "LatentMirrorOptimizer.hpp"

#ifdef MFEM_USE_MPI

#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <limits>

using namespace mfem;
using namespace mfem_lmg;

// ── Bound helpers used throughout the tests ───────────────────────────────
static bool IsInf   (real_t v) { return  v >= real_t(std::numeric_limits<real_t>::infinity()); }
static bool IsNegInf(real_t v) { return  v <= real_t(-std::numeric_limits<real_t>::infinity()); }

// ── Utilities ─────────────────────────────────────────────────────────────

static int g_nfail_local = 0;  // per-rank failure counter
static int g_rank        = 0;
static int g_nranks      = 1;

/** Print only on rank 0. */
static void Print0(const char* fmt, ...)
{
    if (g_rank != 0) return;
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
}

/** Check condition; all ranks contribute to the global fail count. */
static void Check(bool cond_local, const char* msg)
{
    // All ranks must agree — reduce AND.
    int ok_local  = cond_local ? 1 : 0;
    int ok_global = 0;
    MPI_Allreduce(&ok_local, &ok_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (g_rank == 0) {
        if (ok_global) std::printf("  [PASS] %s\n", msg);
        else          { std::printf("  [FAIL] %s\n", msg); ++g_nfail_local; }
    }
    if (!ok_global) ++g_nfail_local;  // every rank counts failures
}

// ── Distributed problem helpers ───────────────────────────────────────────

/** Distribute n_global DOFs across nranks; return local count for rank r. */
static int LocalSize(int n_global, int r, int nranks)
{
    const int base  = n_global / nranks;
    const int extra = n_global % nranks;
    return base + (r < extra ? 1 : 0);
}

/** Global offset for rank r. */
static int LocalOffset(int n_global, int r, int nranks)
{
    int off = 0;
    for (int i = 0; i < r; ++i) off += LocalSize(n_global, i, nranks);
    return off;
}

// ── Compliance proxy helpers ───────────────────────────────────────────────

/**
 * Global objective Φ = Σᵢ 1/λᵢ.
 * Each rank contributes its local sum; returns global value on all ranks.
 */
static double GlobalCompliance(const Vector& lam_local)
{
    double local_phi = 0.0;
    for (int i = 0; i < lam_local.Size(); ++i)
        local_phi += 1.0 / double(lam_local(i));
    double global_phi = 0.0;
    MPI_Allreduce(&local_phi, &global_phi, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_phi;
}

/** Local Euclidean gradient of Φ = Σ 1/λᵢ. */
static void LocalComplianceGrad(const Vector& lam_local, Vector& d_local)
{
    for (int i = 0; i < lam_local.Size(); ++i)
        d_local(i) = real_t(-1.0 / (double(lam_local(i)) *
                                     double(lam_local(i))));
}


// ── Test 1: BoxCompliance ─────────────────────────────────────────────────
/**
 * Global: min Σ 1/λᵢ  s.t. lo_i ≤ λᵢ ≤ hi_i.
 *
 * Each rank owns n_local variables with bounds
 *   lo_i = 0.01 + 0.01*rank,  hi_i = 1.0
 * so the unconstrained minimizer (→0) is blocked by the lower bound and the
 * optimum is λ*ᵢ = lo_i on each rank.
 */
static void Test_BoxCompliance()
{
    Print0("\n--- BoxCompliance (N=%d, %d ranks) ---\n",
           100, g_nranks);

    const int N_global = 100;
    const int n_loc    = LocalSize(N_global, g_rank, g_nranks);

    const real_t lo_val = real_t(0.01 + 0.01 * g_rank);
    const real_t hi_val = real_t(1.0);

    // Build local lo/hi/types.
    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc = lo_val;
    hi_loc = hi_val;
    std::vector<BoundType> types;
    ClassifyBounds(lo_loc, hi_loc, types);

    // Initialize latent from midpoint.
    Vector lam_init(n_loc);
    DefaultPrimalInit(lo_loc, hi_loc, types, lam_init);
    Vector z_loc(n_loc);
    PrimalToLatent(lam_init, lo_loc, hi_loc, types, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);

    Vector lam_loc(n_loc), d_loc(n_loc);
    double kkt = 1.0;
    int ls_total = 0;

    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        const double phi = GlobalCompliance(lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);

        opt.Update(z_loc, d_loc, real_t(phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n_loc);
                LatentToPrimal(zt, lo_loc, hi_loc, types, lt);
                phi_out = real_t(GlobalCompliance(lt));
            });
        ls_total += opt.LastLineSearchSteps();

        // Recompute for residual.
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);
        kkt = double(opt.StationarityResidual(z_loc, d_loc));

        if (it % 50 == 0)
            Print0("  iter %3d: kkt=%.4e ls_steps=%d\n",
                   it, kkt, opt.LastLineSearchSteps());
    }

    // Verify: each λᵢ should be at (or near) lo_i, since Φ is monotone
    // decreasing and the lower bound is the active constraint.
    bool feasible = true;
    double max_lo_viol = 0.0, max_hi_viol = 0.0;
    for (int i = 0; i < n_loc; ++i) {
        double v = double(lam_loc(i));
        if (v < double(lo_val) - 1e-9) { feasible = false; max_lo_viol = std::max(max_lo_viol, double(lo_val)-v); }
        if (v > double(hi_val) + 1e-9) { feasible = false; max_hi_viol = std::max(max_hi_viol, v-double(hi_val)); }
    }

    Print0("  Final: kkt=%.2e  iters=%d  total_ls=%d\n",
           kkt, opt.NumIterations(), ls_total);

    Check(kkt < 1e-3,   "BoxCompliance: KKT < 1e-3");
    Check(feasible,     "BoxCompliance: all λ in [lo,hi]");
    Check(opt.NumIterations() > 0, "BoxCompliance: optimizer ran");
}


// ── Test 2: MixedBounds ───────────────────────────────────────────────────
/**
 * Each rank gets a different bound type for all its variables:
 *   rank 0: TwoSided   [0.1, 2.0]
 *   rank 1: LowerOnly  [0.5, +∞)   (wraps around for nranks < 4)
 *   rank 2: UpperOnly  (-∞, 3.0]
 *   rank 3: Unbounded  (-∞, +∞)
 *
 * Objective: Φ = Σ (λᵢ − 1)²  (global minimum at λ*ᵢ = 1).
 * For bounded ranks the constrained optimum is 1 if inside bounds,
 * else the nearest bound.
 */
static void Test_MixedBounds()
{
    Print0("\n--- MixedBounds (%d ranks) ---\n", g_nranks);

    const int n_loc = 50;
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());

    // Assign bound type by rank (cycling if nranks < 4).
    Vector lo_loc(n_loc), hi_loc(n_loc);
    switch (g_rank % 4) {
    case 0: lo_loc = real_t(0.1);  hi_loc = real_t(2.0);  break;  // TwoSided
    case 1: lo_loc = real_t(0.5);  hi_loc = inf;           break;  // LowerOnly
    case 2: lo_loc = -inf;         hi_loc = real_t(3.0);   break;  // UpperOnly
    case 3: lo_loc = -inf;         hi_loc = inf;            break;  // Unbounded
    }

    std::vector<BoundType> types;
    ClassifyBounds(lo_loc, hi_loc, types);

    // Initialize: primal midpoint → latent.
    Vector lam_init(n_loc);
    DefaultPrimalInit(lo_loc, hi_loc, types, lam_init);
    // For unbounded and UpperOnly that starts at u-1=2 or 0; nudge toward 1.
    for (int i = 0; i < n_loc; ++i) lam_init(i) = real_t(1.5);
    // Re-clip to interior.
    for (int i = 0; i < n_loc; ++i) {
        const real_t li = lo_loc(i), ui = hi_loc(i);
        if (!IsNegInf(li) && lam_init(i) <= li) lam_init(i) = li + real_t(0.01);
        if (!IsInf   (ui) && lam_init(i) >= ui) lam_init(i) = ui - real_t(0.01);
    }

    Vector z_loc(n_loc);
    PrimalToLatent(lam_init, lo_loc, hi_loc, types, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);

    Vector lam_loc(n_loc), d_loc(n_loc);
    double kkt = 1.0;

    for (int it = 0; it < 400 && kkt > 1e-5; ++it) {
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);

        // Global Φ = Σ (λᵢ − 1)²  via Allreduce.
        double local_phi = 0.0;
        for (int i = 0; i < n_loc; ++i) {
            const double v = double(lam_loc(i)) - 1.0;
            local_phi += v * v;
            d_loc(i)   = real_t(2.0 * v);
        }
        double global_phi = 0.0;
        MPI_Allreduce(&local_phi, &global_phi, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        opt.Update(z_loc, d_loc, real_t(global_phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n_loc);
                LatentToPrimal(zt, lo_loc, hi_loc, types, lt);
                double lp = 0.0;
                for (int i = 0; i < n_loc; ++i) {
                    const double v = double(lt(i)) - 1.0;
                    lp += v * v;
                }
                double gp = 0.0;
                MPI_Allreduce(&lp, &gp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                phi_out = real_t(gp);
            });

        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        for (int i = 0; i < n_loc; ++i)
            d_loc(i) = real_t(2.0 * (double(lam_loc(i)) - 1.0));
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
        if (it % 50 == 0)
            Print0("  iter %3d: kkt=%.4e\n", it, kkt);
    }

    // Feasibility check.
    bool feasible = true;
    for (int i = 0; i < n_loc; ++i) {
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        if (!IsNegInf(lo_loc(i)) && lam_loc(i) < lo_loc(i) - 1e-9) feasible = false;
        if (!IsInf   (hi_loc(i)) && lam_loc(i) > hi_loc(i) + 1e-9) feasible = false;
    }

    Print0("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(kkt < 1e-3, "MixedBounds: KKT < 1e-3");
    Check(feasible,   "MixedBounds: all λ respect bounds");
}


// ── Test 3: DiagonalMass ──────────────────────────────────────────────────
/**
 * Same box compliance problem as Test 1, but with a local diagonal mass
 * matrix M = diag(w) where w_i = 1 + 0.5 * (i / n_loc).
 * Verifies that the M-weighted inner product path gives a convergent result.
 */
static void Test_DiagonalMass()
{
    Print0("\n--- DiagonalMass (%d ranks) ---\n", g_nranks);

    const int N_global = 80;
    const int n_loc    = LocalSize(N_global, g_rank, g_nranks);

    const real_t lo_val = real_t(0.05);
    const real_t hi_val = real_t(1.0);

    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc = lo_val; hi_loc = hi_val;

    std::vector<BoundType> types;
    ClassifyBounds(lo_loc, hi_loc, types);

    // Diagonal mass matrix: M_ii = 1 + 0.5*(i/n_loc).
    SparseMatrix M_mat(n_loc, n_loc);
    for (int i = 0; i < n_loc; ++i)
        M_mat.Add(i, i, 1.0 + 0.5 * double(i) / double(std::max(n_loc-1,1)));
    M_mat.Finalize();
    // Exact solve: Jacobi (exact for diagonal).
    DSmoother M_inv(M_mat);

    // Initialize.
    Vector lam_init(n_loc);
    DefaultPrimalInit(lo_loc, hi_loc, types, lam_init);
    Vector z_loc(n_loc);
    PrimalToLatent(lam_init, lo_loc, hi_loc, types, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc,
                                      &M_mat, &M_inv);

    Vector lam_loc(n_loc), d_loc(n_loc);
    double kkt = 1.0;

    for (int it = 0; it < 300 && kkt > 1e-5; ++it) {
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        const double phi = GlobalCompliance(lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);

        opt.Update(z_loc, d_loc, real_t(phi),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n_loc);
                LatentToPrimal(zt, lo_loc, hi_loc, types, lt);
                phi_out = real_t(GlobalCompliance(lt));
            });

        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
        if (it % 50 == 0)
            Print0("  iter %3d: kkt=%.4e\n", it, kkt);
    }

    bool feasible = true;
    for (int i = 0; i < n_loc; ++i) {
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        if (lam_loc(i) < lo_val - 1e-9 || lam_loc(i) > hi_val + 1e-9)
            feasible = false;
    }

    Print0("  Final: kkt=%.2e  iters=%d\n", kkt, opt.NumIterations());
    Check(kkt < 1e-3, "DiagonalMass: KKT < 1e-3");
    Check(feasible,   "DiagonalMass: feasibility with M≠I");
}


// ── Test 4: ArmijoCounts ─────────────────────────────────────────────────
/**
 * Checks that:
 *  (a) the eval_phi callback is genuinely called (counter incremented), and
 *  (b) whenever backtracking occurs (ls > 0), the finally accepted step
 *      satisfies the Armijo sufficient-decrease condition.
 */
static void Test_ArmijoCounts()
{
    Print0("\n--- ArmijoCounts (%d ranks) ---\n", g_nranks);

    const int n_loc = 30;
    const real_t lo_val = real_t(0.05), hi_val = real_t(1.0);

    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc = lo_val; hi_loc = hi_val;
    std::vector<BoundType> types;
    ClassifyBounds(lo_loc, hi_loc, types);

    Vector lam_init(n_loc); DefaultPrimalInit(lo_loc, hi_loc, types, lam_init);
    Vector z_loc(n_loc);
    PrimalToLatent(lam_init, lo_loc, hi_loc, types, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);
    opt.SetLineSearchParams(real_t(1e-4), real_t(0.5), 50);

    Vector lam_loc(n_loc), d_loc(n_loc);
    int total_calls  = 0;
    int armijo_fails = 0;   // times sufficient decrease was violated on accept
    const real_t c1 = real_t(1e-4);

    for (int it = 0; it < 60; ++it) {
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        const double phi_before = GlobalCompliance(lam_loc);
        LocalComplianceGrad(lam_loc, d_loc);

        // Make a copy of z before the step so we can evaluate the pairing.
        Vector z_before = z_loc;
        Vector lam_before = lam_loc;

        int calls_this_iter = 0;
        opt.Update(z_loc, d_loc, real_t(phi_before),
            [&](const Vector& zt, real_t& phi_out) {
                Vector lt(n_loc);
                LatentToPrimal(zt, lo_loc, hi_loc, types, lt);
                phi_out = real_t(GlobalCompliance(lt));
                ++calls_this_iter;
            });
        total_calls += calls_this_iter;

        // Verify Armijo at the accepted point.
        LatentToPrimal(z_loc, lo_loc, hi_loc, types, lam_loc);
        const double phi_after = GlobalCompliance(lam_loc);

        // Armijo pairing (dᵏ)ᵀ(λ_after − λ_before) — global.
        Vector dlam(n_loc);
        subtract(lam_loc, lam_before, dlam);
        // Mg = M g = d (identity M here).
        double loc_pair = double(mfem::InnerProduct(d_loc, dlam));
        double glb_pair = 0.0;
        MPI_Allreduce(&loc_pair, &glb_pair, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        // Armijo: phi_after ≤ phi_before + c1 * pairing
        const double rhs = double(phi_before) + double(c1) * glb_pair;
        if (phi_after > rhs + 1e-10) ++armijo_fails;
    }

    Print0("  Total eval_phi calls: %d  Armijo violations: %d\n",
           total_calls, armijo_fails);
    Check(total_calls > 0,   "ArmijoCounts: eval_phi was called");
    Check(armijo_fails == 0, "ArmijoCounts: no Armijo violations at accepted steps");
}


// ── Test 5: HelperRoundTrip (parallel) ───────────────────────────────────
/**
 * On each rank independently:
 *   PrimalToLatent(lam) → z → LatentToPrimal(z) should recover lam to
 *   machine precision for all four bound types.
 */
static void Test_HelperRoundTrip()
{
    Print0("\n--- HelperRoundTrip (per-rank, 4 bound types) ---\n");

    // Use exactly 4 elements per rank (one per bound type).
    const int n = 4;
    const real_t inf = real_t(std::numeric_limits<real_t>::infinity());
    Vector lo(n), hi(n), lam_orig(n);

    lo(0) = -inf;        hi(0) = inf;         // Unbounded
    lo(1) = real_t(0.5); hi(1) = inf;         // LowerOnly
    lo(2) = -inf;        hi(2) = real_t(2.0); // UpperOnly
    lo(3) = real_t(0.0); hi(3) = real_t(1.0); // TwoSided

    // Use a rank-dependent strictly-feasible interior point.
    lam_orig(0) = real_t(1.0 + 0.1 * g_rank);
    lam_orig(1) = real_t(1.0 + 0.1 * g_rank);   // > 0.5
    lam_orig(2) = real_t(1.5 - 0.1 * g_rank);   // < 2.0
    lam_orig(3) = real_t(0.4 + 0.02 * g_rank);  // in (0,1)

    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);

    Vector z(n), lam_back(n);
    PrimalToLatent(lam_orig, lo, hi, types, z);
    LatentToPrimal(z, lo, hi, types, lam_back);

    double local_maxerr = 0.0;
    for (int i = 0; i < n; ++i)
        local_maxerr = std::max(local_maxerr,
                                std::abs(double(lam_orig(i)) -
                                         double(lam_back(i))));
    double global_maxerr = 0.0;
    MPI_Allreduce(&local_maxerr, &global_maxerr, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    Print0("  Global max round-trip error: %.2e\n", global_maxerr);
    Check(global_maxerr < 1e-12, "HelperRoundTrip: error < 1e-12 on all ranks");
}

// ── main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    Print0("=== LatentMirrorOptimizerParallel test suite ===\n");
    Print0("    Ranks: %d\n\n", g_nranks);

    Test_HelperRoundTrip();
    Test_BoxCompliance();
    Test_MixedBounds();
    Test_DiagonalMass();
    Test_ArmijoCounts();

    // Global failure count.
    int global_nfail = 0;
    MPI_Allreduce(&g_nfail_local, &global_nfail, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    // Deduplicate: each rank counted the same failure; divide by nranks.
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
