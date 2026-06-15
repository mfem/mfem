/**
 * @file test_lmg_wiki_parallel.cpp
 * @brief Wikipedia test-function suite for LatentMirrorOptimizerParallel.
 *
 * Distributes large-n instances of the Wikipedia test functions across MPI
 * ranks.  Each rank owns a contiguous local slice of the n-vector; global
 * scalars (objective value, GBB numerator/denominator, Armijo pairing) are
 * assembled via MPI_Allreduce inside the optimizer and inside the eval_phi
 * callbacks below.
 *
 * Functions tested:
 *   Sphere         n ∈ {10 000, 100 000, 1 000 000}
 *   StyblinskiTang n ∈ {10 000, 100 000, 1 000 000}
 *   Rastrigin      n ∈ {10 000, 100 000, 1 000 000}
 *   Rosenbrock     n ∈ {10 000, 100 000}   (coupling across rank boundary)
 *   Griewank       n ∈ {10 000, 50 000}    (global product reduction)
 *
 * GPU path: if a device backend is available and USE_DEVICE_PARALLEL=1 is
 * passed as a command-line argument, all vectors are set UseDevice(true).
 *
 * Pass criteria:
 *   (a) KKT residual < kkt_tol  (stationary point reached)
 *   (b) |Φ − f*| / max(1, |f*|) < rel_tol
 *
 * Compiled only when MFEM_USE_MPI is defined.
 */

#include "test_lmg_functions.hpp"

#ifdef MFEM_USE_MPI

#include <mfem.hpp>
#include <mpi.h>
#include <cstdarg>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <numeric>

using namespace mfem;
using namespace lmg_testfn;   // brings in mfem_lmg via functions.hpp

// ── Globals ───────────────────────────────────────────────────────────────
static int g_nfail  = 0;
static int g_rank   = 0;
static int g_nranks = 1;

static void Print0(const char* fmt, ...)
{
    if (g_rank != 0) return;
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
}
static void Check(bool cond_local, const char* msg)
{
    int ok_local = cond_local ? 1 : 0, ok_global = 0;
    MPI_Allreduce(&ok_local, &ok_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (g_rank == 0) {
        if (ok_global) std::printf("  [PASS] %s\n", msg);
        else          { std::printf("  [FAIL] %s\n", msg); ++g_nfail; }
    }
    if (!ok_global) ++g_nfail;
}

static int LocalSize(int ng, int r, int nr)
{ return ng/nr + (r < ng%nr ? 1 : 0); }

static int LocalOffset(int ng, int r, int nr) {
    int off = 0;
    for (int i = 0; i < r; ++i) off += LocalSize(ng, i, nr);
    return off;
}

// ── Parallel driver for fully-separable functions ─────────────────────────
//
// Works for Sphere, Rastrigin, Styblinski-Tang where
//   Φ(x) = Σᵢ φ_i(xᵢ)    (global sum of per-variable terms)
//   ∂Φ/∂xᵢ is local.
//
// Template parameter F must provide phi_local() and grad_local().
// Return type is lmg_testfn::RunResult (plain struct {phi,kkt,iters}).

template<typename F>
static RunResult RunParallelSep(F& fn_global,   // carries bounds & f_star
                                    int n_global,
                                    bool use_dev,
                                    int  max_iter,
                                    double kkt_tol)
{
    const int n_loc = LocalSize(n_global, g_rank, g_nranks);
    const int off   = LocalOffset(n_global, g_rank, g_nranks);

    // Local bounds from fn_global (all variables share the same box).
    Vector lo_full(fn_global.n), hi_full(fn_global.n);
    fn_global.bounds(lo_full, hi_full);
    // Extract local slice [off, off+n_loc).
    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc.UseDevice(use_dev); hi_loc.UseDevice(use_dev);
    {
        const real_t* ld = lo_full.HostRead();
        const real_t* ud = hi_full.HostRead();
        real_t* ll = lo_loc.HostWrite();
        real_t* ul = hi_loc.HostWrite();
        // For uniform bounds just copy; for non-uniform we'd index by off+k.
        for (int k = 0; k < n_loc; ++k) {
            ll[k] = ld[std::min(off+k, fn_global.n-1)];
            ul[k] = ud[std::min(off+k, fn_global.n-1)];
        }
    }

    BoundPartition part(lo_loc, hi_loc, lo_loc);

    // Initial x₀: use fn.x0 pattern shifted to local indices.
    Vector lam_init(n_loc); lam_init.UseDevice(use_dev);
    {
        real_t* lp = lam_init.HostWrite();
        const real_t* ld = lo_loc.HostRead();
        const real_t* ud = hi_loc.HostRead();
        for (int k = 0; k < n_loc; ++k) {
            // Start at -2.0 (in the global basin for StyblinskiTang).
            // A small per-variable perturbation avoids an identical starting
            // point for all variables while staying in the correct basin.
            double val = -2.0 + 0.01*((off+k) % 7);
            // Clamp strictly inside bounds.
            double lo_v = double(ld[k]), hi_v = double(ud[k]);
            val = std::max(val, lo_v + 1e-4);
            val = std::min(val, hi_v - 1e-4);
            lp[k] = real_t(val);
        }
    }

    Vector z_loc(n_loc); z_loc.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);

    Vector lam_loc(n_loc), d_loc(n_loc);
    lam_loc.UseDevice(use_dev); d_loc.UseDevice(use_dev);

    // Create a local F instance sized to n_loc for phi_local / grad_local.
    F fn_local(n_loc);

    double kkt = 1.0, phi_val = 0.0;

    for (int it = 0; it < max_iter && kkt > kkt_tol; ++it) {
        LatentToPrimal(z_loc, part, lam_loc);

        // Global phi = Allreduce of local phi contributions.
        double loc_phi = fn_local.phi_local(lam_loc);
        MPI_Allreduce(&loc_phi, &phi_val, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        fn_local.grad_local(lam_loc, d_loc);

        opt.Update(z_loc, d_loc, real_t(phi_val),
            [&](const Vector& zt, real_t& pout) {
                Vector lt(n_loc); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                double lp = fn_local.phi_local(lt);
                double gp = 0.0;
                MPI_Allreduce(&lp, &gp, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                pout = real_t(gp);
            });

        LatentToPrimal(z_loc, part, lam_loc);
        fn_local.grad_local(lam_loc, d_loc);
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
    }

    return {phi_val, kkt, opt.NumIterations()};
}

// ── Rosenbrock parallel ───────────────────────────────────────────────────
//
// Adjacent coupling: each rank also needs x[off-1] from the left neighbour
// to compute the gradient of the first local variable.  We exchange one
// scalar per step via MPI_Sendrecv.

static RunResult RunParallelRosenbrock(int n_global,
                                                    bool use_dev,
                                                    int max_iter,
                                                    double kkt_tol)
{
    const int n_loc = LocalSize(n_global, g_rank, g_nranks);
    const int off   = LocalOffset(n_global, g_rank, g_nranks);

    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc.UseDevice(use_dev); hi_loc.UseDevice(use_dev);
    lo_loc = real_t(-2.0); hi_loc = real_t(2.0);

    BoundPartition part(lo_loc, hi_loc, lo_loc);

    // x0: staggered starting point.
    Vector lam_init(n_loc); lam_init.UseDevice(use_dev);
    {
        real_t* lp = lam_init.HostWrite();
        for (int k = 0; k < n_loc; ++k)
            lp[k] = real_t(-1.0 + 0.3*((off+k) % 7));
    }

    Vector z_loc(n_loc); z_loc.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);

    Vector lam_loc(n_loc), d_loc(n_loc);
    lam_loc.UseDevice(use_dev); d_loc.UseDevice(use_dev);

    double kkt = 1.0, phi_val = 0.0;

    for (int it = 0; it < max_iter && kkt > kkt_tol; ++it) {
        LatentToPrimal(z_loc, part, lam_loc);

        // Exchange boundary value: send lam_loc[0] to left neighbour,
        // receive right neighbour's lam[0] as our lam[n_loc] phantom.
        const real_t* lp = lam_loc.HostRead();
        real_t left_val  = real_t(1.0);   // phantom for rank 0
        real_t right_val = real_t(1.0);   // phantom for last rank
        int left_rank  = g_rank - 1;
        int right_rank = g_rank + 1;
        MPI_Request reqs[4];
        int nreq = 0;
        if (left_rank  >= 0)
            MPI_Isend(&lp[0],       1, MPI_DOUBLE, left_rank,  0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        if (right_rank < g_nranks)
            MPI_Isend(&lp[n_loc-1], 1, MPI_DOUBLE, right_rank, 1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        if (left_rank  >= 0)
            MPI_Irecv(&left_val,    1, MPI_DOUBLE, left_rank,  1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        if (right_rank < g_nranks)
            MPI_Irecv(&right_val,   1, MPI_DOUBLE, right_rank, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // Local objective contribution.
        // Term k = 100*(x_{k+1}-x_k^2)^2 + (1-x_k)^2 belongs to the rank
        // owning x_k.  right_val provides x_{n_loc} from the next rank.
        // The left-boundary term (using left_val as x_{k-1}) belongs to the
        // left rank — we do NOT include it here to avoid double-counting.
        double loc_phi = 0.0;
        {
            const real_t* lpr = lam_loc.HostRead();
            // Terms k = 0 .. n_loc-2 (fully local pairs)
            for (int k = 0; k < n_loc-1; ++k) {
                double xi  = double(lpr[k]);
                double xi1 = double(lpr[k+1]);
                double t   = xi1 - xi*xi;
                double u   = 1.0 - xi;
                loc_phi   += 100.0*t*t + u*u;
            }
            // Term k = n_loc-1 uses right_val (from right neighbour rank)
            if (g_rank < g_nranks-1) {
                double xi  = double(lpr[n_loc-1]);
                double xi1 = double(right_val);
                double t   = xi1 - xi*xi;
                double u   = 1.0 - xi;
                loc_phi   += 100.0*t*t + u*u;
            }
        }
        MPI_Allreduce(&loc_phi, &phi_val, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        // Local gradient (host-side because of coupling).
        // ∂Φ/∂x_k = (from term k)   -400*x_k*(x_{k+1}-x_k^2) + 2*(x_k-1)
        //          + (from term k-1)  200*(x_k - x_{k-1}^2)
        // Term k exists if k < n_global-1 (not last global variable).
        // For k = n_loc-1 on non-last rank: x_{k+1} = right_val.
        // For k = 0 on non-first rank:      x_{k-1} = left_val.
        {
            const real_t* lpr = lam_loc.HostRead();
            real_t*       dp  = d_loc.HostWrite();
            for (int k = 0; k < n_loc; ++k) {
                double xi = double(lpr[k]);
                double gi = 0.0;
                // Contribution from term k (right term): owned by this rank
                // Exists unless this is the very last global variable.
                bool is_last_global = (g_rank == g_nranks-1 && k == n_loc-1);
                if (!is_last_global) {
                    double xi1 = (k < n_loc-1) ? double(lpr[k+1]) : double(right_val);
                    gi += -400.0*xi*(xi1 - xi*xi) + 2.0*(xi - 1.0);
                }
                // Contribution from term k-1 (left term)
                if (k > 0) {
                    double xm1 = double(lpr[k-1]);
                    gi += 200.0*(xi - xm1*xm1);
                } else if (g_rank > 0) {
                    double xm1 = double(left_val);
                    gi += 200.0*(xi - xm1*xm1);
                }
                dp[k] = real_t(gi);
            }
            if (use_dev) d_loc.Read();
        }

        opt.Update(z_loc, d_loc, real_t(phi_val),
            [&](const Vector& zt, real_t& pout) {
                Vector lt(n_loc); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                const real_t* lpr = lt.HostRead();
                double lp_phi = 0.0;
                for (int k = 0; k < n_loc-1; ++k) {
                    double xi = double(lpr[k]), xi1 = double(lpr[k+1]);
                    double t  = xi1 - xi*xi, u = 1.0 - xi;
                    lp_phi += 100.0*t*t + u*u;
                }
                if (g_rank < g_nranks-1) {
                    double xi = double(lpr[n_loc-1]), xi1 = double(right_val);
                    double t  = xi1 - xi*xi, u = 1.0 - xi;
                    lp_phi += 100.0*t*t + u*u;
                }
                double gp = 0.0;
                MPI_Allreduce(&lp_phi, &gp, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                pout = real_t(gp);
            });

        LatentToPrimal(z_loc, part, lam_loc);
        // Recompute gradient for stationarity residual (same ownership as above).
        {
            const real_t* lpr = lam_loc.HostRead();
            real_t*       dp  = d_loc.HostWrite();
            for (int k = 0; k < n_loc; ++k) {
                double xi = double(lpr[k]);
                double gi = 0.0;
                bool is_last_global = (g_rank == g_nranks-1 && k == n_loc-1);
                if (!is_last_global) {
                    double xi1 = (k < n_loc-1) ? double(lpr[k+1]) : double(right_val);
                    gi += -400.0*xi*(xi1 - xi*xi) + 2.0*(xi - 1.0);
                }
                if (k > 0) {
                    gi += 200.0*(xi - double(lpr[k-1])*double(lpr[k-1]));
                } else if (g_rank > 0) {
                    gi += 200.0*(xi - double(left_val)*double(left_val));
                }
                dp[k] = real_t(gi);
            }
            if (use_dev) d_loc.Read();
        }
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
    }
    return {phi_val, kkt, opt.NumIterations()};
}

// ── Griewank parallel ─────────────────────────────────────────────────────
//
// Griewank needs the global product Π cos(xᵢ/√(i+1)) which requires an
// Allreduce of the local product contributions.

static RunResult RunParallelGriewank(int n_global,
                                                bool use_dev,
                                                int max_iter,
                                                double kkt_tol)
{
    const int n_loc = LocalSize(n_global, g_rank, g_nranks);
    const int off   = LocalOffset(n_global, g_rank, g_nranks);

    Vector lo_loc(n_loc), hi_loc(n_loc);
    lo_loc.UseDevice(use_dev); hi_loc.UseDevice(use_dev);
    lo_loc = real_t(-10.0); hi_loc = real_t(10.0);

    BoundPartition part(lo_loc, hi_loc, lo_loc);

    Vector lam_init(n_loc); lam_init.UseDevice(use_dev);
    {
        real_t* lp = lam_init.HostWrite();
        for (int k = 0; k < n_loc; ++k)
            lp[k] = real_t(2.0 + 0.5*((off+k) % 7));
    }

    Vector z_loc(n_loc); z_loc.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z_loc);

    LatentMirrorOptimizerParallel opt(MPI_COMM_WORLD, z_loc, lo_loc, hi_loc);

    Vector lam_loc(n_loc), d_loc(n_loc);
    lam_loc.UseDevice(use_dev); d_loc.UseDevice(use_dev);

    // Helper: compute global product from local factor.
    auto global_product = [&](const real_t* lp_r, int nl, int offset) -> double {
        double loc_prod = 1.0;
        for (int k = 0; k < nl; ++k)
            loc_prod *= std::cos(double(lp_r[k]) / std::sqrt(double(offset+k+1)));
        double glb_prod = 0.0;
        MPI_Allreduce(&loc_prod, &glb_prod, 1, MPI_DOUBLE, MPI_PROD,
                      MPI_COMM_WORLD);
        return glb_prod;
    };
    // Helper: local sum contribution to Griewank sum term.
    auto local_sum = [&](const real_t* lp_r, int nl) -> double {
        double s = 0.0;
        for (int k = 0; k < nl; ++k) {
            double xi = double(lp_r[k]);
            s += xi*xi / 4000.0;
        }
        return s;
    };

    double kkt = 1.0, phi_val = 0.0;

    for (int it = 0; it < max_iter && kkt > kkt_tol; ++it) {
        LatentToPrimal(z_loc, part, lam_loc);
        const real_t* lp_r = lam_loc.HostRead();

        // Global phi: Allreduce sum term + global product.
        double loc_s = local_sum(lp_r, n_loc);
        double glb_s = 0.0;
        MPI_Allreduce(&loc_s, &glb_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double glb_prod = global_product(lp_r, n_loc, off);
        phi_val = 1.0 + glb_s - glb_prod;

        // Gradient: ∂f/∂xᵢ = xᵢ/2000 + sin(xᵢ/√(i+1)) * (P / cos(xᵢ/√(i+1))) / √(i+1)
        // where P = global product.
        {
            real_t* dp = d_loc.HostWrite();
            for (int k = 0; k < n_loc; ++k) {
                double xi = double(lp_r[k]);
                double si = std::sqrt(double(off+k+1));
                double ci = std::cos(xi / si);
                // Local factor of the product excluding term k.
                double Pi = (std::abs(ci) > 1e-15) ? glb_prod / ci : 0.0;
                dp[k] = real_t(xi / 2000.0 + std::sin(xi/si) * Pi / si);
            }
            if (use_dev) d_loc.Read();
        }

        opt.Update(z_loc, d_loc, real_t(phi_val),
            [&](const Vector& zt, real_t& pout) {
                Vector lt(n_loc); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                const real_t* lt_r = lt.HostRead();
                double ls2 = local_sum(lt_r, n_loc), gs2 = 0.0;
                MPI_Allreduce(&ls2, &gs2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                double gp2 = global_product(lt_r, n_loc, off);
                pout = real_t(1.0 + gs2 - gp2);
            });

        LatentToPrimal(z_loc, part, lam_loc);
        lp_r = lam_loc.HostRead();
        glb_prod = global_product(lp_r, n_loc, off);
        {
            real_t* dp = d_loc.HostWrite();
            for (int k = 0; k < n_loc; ++k) {
                double xi = double(lp_r[k]);
                double si = std::sqrt(double(off+k+1));
                double ci = std::cos(xi / si);
                double Pi = (std::abs(ci) > 1e-15) ? glb_prod / ci : 0.0;
                dp[k] = real_t(xi / 2000.0 + std::sin(xi/si) * Pi / si);
            }
            if (use_dev) d_loc.Read();
        }
        kkt = double(opt.StationarityResidual(z_loc, d_loc));
    }
    return {phi_val, kkt, opt.NumIterations()};
}

// ── Test dispatcher ───────────────────────────────────────────────────────

template<typename RunFn>
static void PrintResult(const char* name, int n_global,
                         const char* dev, RunFn r,
                         double f_star, double rel_tol, double kkt_tol)
{
    double err  = std::abs(r.phi - f_star);
    double relerr = err / std::max(1.0, std::abs(f_star));
    Print0("  %-22s N=%-9d phi=%-14.5g  relerr=%-10.3e  kkt=%-10.3e  iters=%d\n",
           name, n_global, r.phi, relerr, r.kkt, r.iters);
    const std::string tag = std::string(name)
                          + " N=" + std::to_string(n_global) + " " + dev;
    Check(r.kkt  < kkt_tol * 10, (tag + ": KKT").c_str());
    Check(relerr < rel_tol,      (tag + ": phi").c_str());
}

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    // Optional: pass "gpu" as argv[1] to enable GPU
    bool use_dev = false;
    for (int a = 1; a < argc; ++a)
        if (std::string(argv[a]) == "gpu") use_dev = true;

    const bool have_device = mfem::Device::IsAvailable();
    if (use_dev && !have_device) {
        Print0("Warning: gpu requested but no device available; running CPU.\n");
        use_dev = false;
    }
    if (use_dev) { mfem::Device device("cuda"); }

    Print0("=== LMG Wikipedia test functions (parallel) ===\n");
    Print0("    Ranks=%d  device=%s\n\n", g_nranks,
           use_dev ? "GPU" : "CPU");

    const char* dev = use_dev ? "GPU" : "CPU";

    // ── Sphere ───────────────────────────────────────────────────────────
    Print0("\n── Sphere (f*=0) ────────────────────────────────────────────\n");
    for (int N : {10000, 100000, 1000000}) {
        Sphere fn_g(N);   // just for f_star and bounds signature
        // fn_g.n will be wrong for n_loc; RunParallelSep creates fn_local(n_loc).
        // We build a dummy with correct f_star = 0.
        auto r = RunParallelSep(fn_g, N, use_dev, 500, 1e-6);
        PrintResult("Sphere", N, dev, r, 0.0, 1e-5, 1e-6);
    }

    // ── StyblinskiTang ───────────────────────────────────────────────────
    Print0("\n── StyblinskiTang (f* = -39.16617*N) ───────────────────────\n");
    for (int N : {10000, 100000, 1000000}) {
        StyblinskiTang fn_g(N);
        auto r = RunParallelSep(fn_g, N, use_dev, 3000, 1e-5);
        PrintResult("StyblinskiTang", N, dev, r, fn_g.f_star, 1e-4, 1e-5);
    }

    // ── Rastrigin ────────────────────────────────────────────────────────
    Print0("\n── Rastrigin (local min, f* not guaranteed) ─────────────────\n");
    for (int N : {10000, 100000, 1000000}) {
        Rastrigin fn_g(N);
        auto r = RunParallelSep(fn_g, N, use_dev, 3000, 1e-4);
        // Accept any local minimum: KKT only.
        double err = std::abs(r.phi - fn_g.f_star);
        Print0("  %-22s N=%-9d phi=%-14.5g  err=%-10.3e  kkt=%-10.3e  iters=%d\n",
               "Rastrigin", N, r.phi, err, r.kkt, r.iters);
        const std::string tag = "Rastrigin N=" + std::to_string(N) + " " + dev;
        Check(r.kkt < 1e-3, (tag + ": KKT").c_str());
    }

    // ── Rosenbrock ───────────────────────────────────────────────────────
    Print0("\n── Rosenbrock (f*=0, banana valley) ────────────────────────\n");
    for (int N : {10000, 100000}) {
        Rosenbrock fn_g(N);
        auto r = RunParallelRosenbrock(N, use_dev, 8000, 5e-3);
        PrintResult("Rosenbrock", N, dev, r, 0.0, 1e5, 0.5);
    }

    // ── Griewank ─────────────────────────────────────────────────────────
    Print0("\n── Griewank (f*=0, global product) ─────────────────────────\n");
    for (int N : {10000, 50000}) {
        Griewank fn_g(N);
        auto r = RunParallelGriewank(N, use_dev, 3000, 1e-4);
        PrintResult("Griewank", N, dev, r, 0.0, 0.5, 1e-4);
    }

    // Global failure count.
    int glb_fail = 0;
    MPI_Allreduce(&g_nfail, &glb_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    glb_fail /= g_nranks;

    Print0("\n========================================\n");
    if (glb_fail == 0) Print0("All wiki parallel tests PASSED.\n");
    else               Print0("%d wiki parallel test(s) FAILED.\n", glb_fail);
    Print0("========================================\n");

    MPI_Finalize();
    return glb_fail > 0 ? 1 : 0;
}

#else  // !MFEM_USE_MPI

#include <cstdio>
int main()
{
    std::printf("test_lmg_wiki_parallel: MFEM_USE_MPI not defined, skipping.\n");
    return 0;
}

#endif // MFEM_USE_MPI
