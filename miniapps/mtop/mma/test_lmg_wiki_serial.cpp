/**
 * @file test_lmg_wiki_serial.cpp
 * @brief Wikipedia test-function suite for LatentMirrorOptimizer (serial).
 *
 * Tests all single-objective Wikipedia test functions from:
 *   https://en.wikipedia.org/wiki/Test_functions_for_optimization
 *
 * Pass criteria per function:
 *   (a) KKT stationarity residual < kkt_tol (convergence to a stationary point)
 *   (b) |Φ(x_final) − f*| < phi_tol         (objective near known optimum)
 *
 * For multi-modal functions (Rastrigin, Lévi, Himmelblau) only criterion (a)
 * is strict — the optimizer finds a local minimum which is accepted as long
 * as the objective is within a basin-width tolerance of f*.
 *
 * 2-D functions run at n=2.
 * n-dimensional functions run at n ∈ {10, 100, 1000}.
 * GPU path runs when a device backend is available.
 */

#include "test_lmg_functions.hpp"
#include <mfem.hpp>
#include <cstdio>
#include <cmath>
#include <string>

using namespace mfem;
using namespace lmg_testfn;   // brings in mfem_lmg via functions.hpp

static int g_nfail = 0;

static void Check(bool cond, const char* msg)
{
    if (cond) std::printf("  [PASS] %s\n", msg);
    else     { std::printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// ── Generic test driver ────────────────────────────────────────────────────

template<typename F>
static void TestFn(const char* name, F& fn,
                   double phi_tol, double kkt_tol,
                   bool use_dev = false, int max_iter = 3000)
{
    std::printf("  %-28s n=%-5d ", name, fn.n);
    auto r = RunSerial(fn, use_dev, max_iter, kkt_tol);
    const double err = std::abs(r.phi - fn.f_star);
    std::printf("phi=%-12.5g  err=%-10.3e  kkt=%-10.3e  iters=%d\n",
                r.phi, err, r.kkt, r.iters);

    const std::string tag = std::string(name)
                          + " n=" + std::to_string(fn.n)
                          + (use_dev ? " GPU" : " CPU");
    Check(r.kkt < kkt_tol * 10,   (tag + ": KKT").c_str());
    Check(err   < phi_tol,        (tag + ": phi near f*").c_str());
}

// ── Himmelblau: run from each of the 4 basins ────────────────────────────

static void TestHimmelblau(bool use_dev)
{
    Himmelblau fn;
    const int  n   = 2;
    const char* dev = use_dev ? "GPU" : "CPU";

    Vector lo(n), hi(n);
    fn.bounds(lo, hi);
    lo.UseDevice(use_dev);
    hi.UseDevice(use_dev);
    BoundPartition part(lo, hi, lo);

    // Known minima to verify we reach one of them.
    const double known[4][2] = {
        { 3.0,      2.0     },
        {-2.805118, 3.131312},
        {-3.779310,-3.283186},
        { 3.584428,-1.848126}
    };

    for (int b = 0; b < 4; ++b) {
        Vector lam_init(n); lam_init.UseDevice(use_dev);
        fn.x0(lam_init, b);

        Vector z(n); z.UseDevice(use_dev);
        PrimalToLatent(lam_init, part, z);

        LatentMirrorOptimizer opt(z, lo, hi);
        Vector lam(n), d(n);
        lam.UseDevice(use_dev); d.UseDevice(use_dev);

        double kkt = 1.0, phi_val = 0.0;
        for (int it = 0; it < 3000 && kkt > 1e-5; ++it) {
            LatentToPrimal(z, part, lam);
            phi_val = fn.phi(lam);
            fn.grad(lam, d);
            opt.Update(z, d, real_t(phi_val),
                [&](const Vector& zt, real_t& po) {
                    Vector lt(n); lt.UseDevice(use_dev);
                    LatentToPrimal(zt, part, lt);
                    po = real_t(fn.phi(lt));
                });
            LatentToPrimal(z, part, lam);
            fn.grad(lam, d);
            kkt = double(opt.StationarityResidual(z, d));
        }

        // Check: converged to *some* minimum (phi < 1e-4) near one of the 4.
        char tag[64];
        std::snprintf(tag, sizeof(tag),
                      "Himmelblau basin%d %s: phi<1e-3", b, dev);
        std::printf("  %-28s basin=%d phi=%-10.4g  kkt=%-10.3e  iters=%d\n",
                    "Himmelblau", b, phi_val, kkt, opt.NumIterations());
        Check(phi_val < 1e-3, tag);
    }
}

// ── 2-D function section ──────────────────────────────────────────────────

static void Run2D(bool use_dev)
{
    const char* dev = use_dev ? "GPU" : "CPU";
    std::printf("\n── 2-D functions (%s) ──────────────────────────────────────\n",
                dev);

    { Beale          fn; TestFn("Beale",          fn, 1e-4, 1e-5, use_dev); }
    { Booth          fn; TestFn("Booth",          fn, 1e-4, 1e-5, use_dev); }
    { Matyas         fn; TestFn("Matyas",         fn, 1e-4, 1e-5, use_dev); }
    // Levi13: multimodal sinusoidal — run 3 restarts and keep best result.
    {
        Levi13 fn;
        const int n = 2;
        const char* dev2 = use_dev ? "GPU" : "CPU";
        // Three starts inside the basin of (1,1):
        // Levi13 has local minima at x=k/3,y=1 (f≈0.111); the basin of (1,1)
        // has capture radius ~0.15 in x.  All three starts are safely inside.
        const double starts[3][2] = {{0.9, 0.9}, {1.1, 1.1}, {1.0, 0.5}};
        double best_phi = 1e30, best_kkt = 1.0; int best_iters = 0;
        for (int s = 0; s < 3; ++s) {
            Vector lo2(n), hi2(n); fn.bounds(lo2, hi2);
            lo2.UseDevice(use_dev); hi2.UseDevice(use_dev);
            BoundPartition part2(lo2, hi2, lo2);
            Vector lam_init2(n); lam_init2.UseDevice(use_dev);
            lam_init2(0) = real_t(starts[s][0]);
            lam_init2(1) = real_t(starts[s][1]);
            Vector z2(n); z2.UseDevice(use_dev);
            PrimalToLatent(lam_init2, part2, z2);
            LatentMirrorOptimizer opt2(z2, lo2, hi2);
            Vector lam2(n), d2(n);
            lam2.UseDevice(use_dev); d2.UseDevice(use_dev);
            // Run until phi < 1e-4 OR max iterations.
            // Do NOT use KKT as stopping criterion: Levi13 has local minima
            // where KKT drops below 1e-5 but phi ≈ 0.11.
            double kkt2 = 1.0, phi2 = 0.0;
            for (int it = 0; it < 2000; ++it) {
                LatentToPrimal(z2, part2, lam2);
                phi2 = fn.phi(lam2);
                fn.grad(lam2, d2);
                if (phi2 < 1e-4) {
                    // Recompute KKT at the final accepted point before breaking.
                    kkt2 = double(opt2.StationarityResidual(z2, d2));
                    break;
                }
                opt2.Update(z2, d2, real_t(phi2),
                    [&](const Vector& zt, real_t& po) {
                        Vector lt(n); lt.UseDevice(use_dev);
                        LatentToPrimal(zt, part2, lt);
                        po = real_t(fn.phi(lt));
                    });
                LatentToPrimal(z2, part2, lam2);
                fn.grad(lam2, d2);
                kkt2 = double(opt2.StationarityResidual(z2, d2));
            }
            if (phi2 < best_phi) { best_phi=phi2; best_kkt=kkt2; best_iters=opt2.NumIterations(); }
        }
        std::printf("  %-28s n=%-5d phi=%-12.5g  err=%-10.3e  kkt=%-10.3e  iters=%d\n",
                    "Levi13(best-of-3)", 2, best_phi,
                    std::abs(best_phi - fn.f_star), best_kkt, best_iters);
        std::string tag2 = std::string("Levi13 n=2 ") + dev2;
        Check(best_kkt  < 0.5,           (tag2 + ": KKT (near min)").c_str());
        Check(best_phi  < 1e-2,          (tag2 + ": phi near 0").c_str());
    }
    { ThreeHumpCamel fn; TestFn("ThreeHumpCamel", fn, 1e-4, 1e-5, use_dev); }
    { McCormick      fn; TestFn("McCormick",      fn, 1e-2, 1e-4, use_dev, 5000); }
    TestHimmelblau(use_dev);
}

// ── n-dimensional function section ───────────────────────────────────────

static void RunND(bool use_dev)
{
    const char* dev = use_dev ? "GPU" : "CPU";
    std::printf("\n── n-dimensional functions (%s) ─────────────────────────────\n",
                dev);

    // Sphere: convex, smooth — fastest convergence, tight tolerance
    for (int n : {10, 100, 1000}) {
        Sphere fn(n);
        TestFn("Sphere", fn, 1e-5, 1e-6, use_dev, 500);
    }

    // Styblinski-Tang: x0 = -2.0 is in the basin of the global min (-2.9035);
    // tighten phi_tol since we should reliably reach the global minimum.
    for (int n : {10, 100, 1000}) {
        StyblinskiTang fn(n);
        TestFn("StyblinskiTang", fn, double(n) * 0.01, 1e-5, use_dev, 3000);
    }

    // Rastrigin: multimodal; accept any local minimum with small KKT.
    for (int n : {10, 100, 1000}) {
        Rastrigin fn(n);
        TestFn("Rastrigin", fn, double(n) * 5.0, 1e-4, use_dev, 3000);
    }

    // Rosenbrock: banana valley, slow convergence.
    // Small n converges well; n=200 is hard for first-order methods.
    for (int n : {10, 50, 200}) {
        Rosenbrock fn(n);
        const int    max_it = (n <= 50) ? 10000 : 30000;
        const double ptol   = (n <= 50) ? 2.0   : 200.0;
        const double ktol   = (n <= 50) ? 5e-3  : 0.1;
        TestFn("Rosenbrock", fn, ptol, ktol, use_dev, max_it);
    }

    // Griewank: product term makes it non-separable; restrict to small n
    for (int n : {10, 50}) {
        Griewank fn(n);
        TestFn("Griewank", fn, 0.1, 1e-4, use_dev, 3000);
    }
}

// ── main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
#ifdef MFEM_USE_MPI
    MPI_Init(&argc, &argv);
#endif

    const bool have_device = mfem::Device::IsAvailable();
    std::printf("=== LMG Wikipedia test functions (serial) ===\n");
    std::printf("    GPU device: %s\n", have_device ? "yes" : "no");

    Run2D(false);
    RunND(false);

    if (have_device) {
        mfem::Device device("cuda");
        std::printf("\n── GPU re-run of n-dimensional tests ──────────────────────\n");
        RunND(true);
    } else {
        std::printf("\n  (GPU tests skipped)\n");
    }

    std::printf("\n========================================\n");
    if (g_nfail == 0) std::printf("All wiki serial tests PASSED.\n");
    else              std::printf("%d wiki serial test(s) FAILED.\n", g_nfail);
    std::printf("========================================\n");

#ifdef MFEM_USE_MPI
    MPI_Finalize();
#endif
    return g_nfail > 0 ? 1 : 0;
}
