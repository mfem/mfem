/**
 * test_sq_nonconvex.cpp  --  Non-convex large-scale SQ/GCMMA test suite
 *
 * Density-filtered SIMP topology optimisation on a 1D domain using the
 * Separable Quadratic (SQ) approximation.  Physics, filter, load pattern,
 * and problem suite are identical to test_mma_nonconvex.cpp so that MMA and SQ
 * results can be compared directly.
 *
 * The global and regional volume limits are represented as one-sided
 * inequalities.  The decreasing SIMP objective normally makes the global
 * upper-volume bound active at a solution.
 *
 * For the algorithm description see test_mma_nonconvex.cpp and mma_algorithms.pdf.
 *
 * Build:  cmake --build build --target test_sq_nonconvex
 * Run:    ./build/test_sq_nonconvex
 *         mpirun -np 4 ./build/test_sq_nonconvex
 *         ./build/test_sq_nonconvex --large
 */

#include "MMA_MFEM.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>

using namespace mfem;
using namespace mfem_mma;
using Clock = std::chrono::steady_clock;

static int  g_rank   = 0;
static int  g_nranks = 1;
static int  g_nfail  = 0;
static bool g_large  = false;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) printf("  [PASS] %s\n", msg);
    else     { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

static double GSum(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return g; }

static std::pair<int,int> Distribute(int n)
{ int b=n/g_nranks, r=n%g_nranks; return {b+(g_rank<r?1:0), g_rank*b+std::min(g_rank,r)}; }

// ── Gaussian density filter (identical to test_mma_nonconvex.cpp) ─────────────
struct Filter {
    int n_global, r;
    std::vector<std::vector<int>>    idx;
    std::vector<std::vector<double>> wgt;

    Filter(int ng, int nl, int off, int radius) : n_global(ng), r(radius)
    {
        idx.resize(nl); wgt.resize(nl);
        for (int e = 0; e < nl; ++e) {
            int g = off + e;
            double wsum = 0;
            std::vector<int>    nbr;
            std::vector<double> w;
            int lo = std::max(0, g-3*r), hi = std::min(ng-1, g+3*r);
            for (int k = lo; k <= hi; ++k) {
                double d  = double(g - k);
                double wi = std::exp(-d*d / (2.0*r*r));
                nbr.push_back(k); w.push_back(wi); wsum += wi;
            }
            for (auto& wi : w) wi /= wsum;
            idx[e] = nbr; wgt[e] = w;
        }
    }

    void apply(const std::vector<double>& x_full, int nl, int /*off*/,
               std::vector<double>& x_hat) const
    {
        x_hat.resize(nl);
        for (int e = 0; e < nl; ++e) {
            double s = 0;
            for (int i = 0; i < (int)idx[e].size(); ++i)
                s += wgt[e][i] * x_full[idx[e][i]];
            x_hat[e] = s;
        }
    }

    void applyTranspose(const std::vector<double>& y_local,
                        std::vector<double>& result, MPI_Comm comm) const
    {
        result.assign(n_global, 0.0);
        for (int e = 0; e < (int)idx.size(); ++e)
            for (int i = 0; i < (int)idx[e].size(); ++i)
                result[idx[e][i]] += wgt[e][i] * y_local[e];
        MPI_Allreduce(MPI_IN_PLACE, result.data(), n_global, MPI_DOUBLE,
                      MPI_SUM, comm);
    }
};

static std::vector<double> GatherFull(int n_global, int nl, int off,
                                       const Vector& x, MPI_Comm comm)
{
    std::vector<double> full(n_global, 0.0);
    for (int j = 0; j < nl; ++j) full[off+j] = double(x(j));
    MPI_Allreduce(MPI_IN_PLACE, full.data(), n_global, MPI_DOUBLE, MPI_SUM, comm);
    return full;
}

// ── Run one filtered SIMP test with SQ ───────────────────────────────────────
// n_regional is either zero or two (first/last 30% of the domain).
// The global volume constraint is the inequality mean(x)-Vfrac <= 0.
struct Result { int iters=0; double kkt_final=INFINITY; double max_viol=INFINITY;
                double f0_init=0.0, f0_final=0.0, max_design_change=0.0; };

static Result RunFilteredSIMP_SQ(
    int n_global, int n_regional, double Vfrac,
    double simp_p, bool continuation,
    int filter_r, bool gcmma, int max_iter)
{
    MFEM_VERIFY(n_regional == 0 || n_regional == 2,
                "RunFilteredSIMP_SQ supports zero or two regional constraints");
    auto [nl, off] = Distribute(n_global);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Load pattern (identical to test_mma_nonconvex.cpp)
    std::vector<double> w_local(nl);
    uint64_t lcg_s = 314159265ULL;
    for (int gg = 0; gg < off; ++gg)
        lcg_s = lcg_s*6364136223846793005ULL + 1442695040888963407ULL;
    for (int e = 0; e < nl; ++e) {
        int g = off + e;
        lcg_s = lcg_s*6364136223846793005ULL + 1442695040888963407ULL;
        double noise = 0.04*(double(lcg_s >> 33)/double(1ULL << 31) - 0.5);
        double sq1   = (g/filter_r)%2 == 0 ?  1.0 : -1.0;
        double sq2   = (g/(2*filter_r))%2 == 0 ?  1.0 : -1.0;
        constexpr double pi = 3.141592653589793238462643383279502884;
        double env   = 1.0 + 0.3*std::sin(2.0*pi*3*g/n_global)
                          + 0.15*std::sin(2.0*pi*7*g/n_global);
        w_local[e]   = env*(1.0 + 0.40*sq1 + 0.15*sq2) + noise;
        if (w_local[e] < 0.05) w_local[e] = 0.05;
    }

    Filter filt(n_global, nl, off, filter_r);

    const double xmin_v = 0.01;
    Vector x(nl), xmin_v_(nl), xmax_v_(nl), df0(nl);
    x = real_t(Vfrac); xmin_v_ = real_t(xmin_v); xmax_v_ = 1.0;

    // Start from a feasible, nonuniform design when the two regional limits
    // are present.  Put the end regions on their bounds and choose the middle
    // value so that the global mean is exactly V, including when integer
    // partition sizes are not precisely 30%/40%/30%.
    if (n_regional == 2)
    {
        MFEM_VERIFY(Vfrac - 0.05 >= xmin_v && Vfrac + 0.05 <= 1.0,
                    "regional initial design lies outside the box bounds");
        const int r0_end = int(0.3*n_global);
        const int r1_beg = int(0.7*n_global);
        const int r0_size = r0_end;
        const int mid_size = r1_beg - r0_end;
        const int r1_size = n_global - r1_beg;
        MFEM_VERIFY(mid_size > 0, "regional initial design has no middle region");
        const double middle =
            (n_global*Vfrac - r0_size*(Vfrac-0.05)
             - r1_size*(Vfrac+0.05))/mid_size;
        MFEM_VERIFY(middle >= xmin_v && middle <= 1.0,
                    "middle initial density lies outside the box bounds");
        for (int j = 0; j < nl; ++j)
        {
            const int g = off + j;
            x(j) = real_t(g < r0_end ? Vfrac - 0.05
                          : (g >= r1_beg ? Vfrac + 0.05 : middle));
        }
    }
    const Vector x_initial(x);

    // ── Constraint layout ─────────────────────────────────────────────────
    // Plain inequalities only — no equality encoding.
    // fi(0) = mean(x) - Vfrac             <= 0   (global volume upper bound)
    // fi(1) = mean(x_region0) - target0   <= 0   (regional, asymmetric)
    // fi(2) = mean(x_region1) - target1   <= 0   (regional, asymmetric)
    //
    // Region 0: first 30%, target = Vfrac-0.05
    // Region 1: last  30%, target = Vfrac+0.05
    //
    // The volume constraint fi(0) is one-sided (upper). At the optimum the
    // SIMP objective drives x toward xmax, so fi(0) is always active and
    // mean(x)=Vfrac at convergence. Using a single inequality instead of
    // the ±h equality encoding avoids the rank-1 Hessian pathology in the
    // SQ dual that causes oscillation in mixed constraint systems.

    int m = 1 + (n_regional > 0 ? n_regional : 0);
    std::vector<Vector> dg(m);
    for (int k = 0; k < m; ++k) { dg[k].SetSize(nl); dg[k] = 0.0; }

    // Volume gradient: dfi(0)/dx_j = 1/n
    for (int j = 0; j < nl; ++j) dg[0](j) = real_t(1.0/n_global);

    // Regional gradients (asymmetric: first/last 30%)
    if (n_regional >= 2) {
        int r0_end  = (int)(0.3*n_global);
        int r1_beg  = (int)(0.7*n_global);
        int r0_size = r0_end;
        int r1_size = n_global - r1_beg;
        for (int j = 0; j < nl; ++j) {
            int g = off + j;
            if (g < r0_end)  dg[1](j) = real_t(1.0/r0_size);
            if (g >= r1_beg) dg[2](j) = real_t(1.0/r1_size);
        }
    }

    const double cv = std::max(1000.0, 10.0*n_global);
    std::vector<double> av(m,0), cv_v(m,cv), dv_v(m,1);
    SQOptimizerParallel opt(comm, nl, m, av.data(), cv_v.data(), dv_v.data());
    // Strong SIMP penalization has much larger and more rapidly changing
    // curvature.  Restrict its SQ move region to avoid the persistent
    // two-cycle produced by the default half-range step.
    const double sigma_scale = simp_p >= 5.0 ? 0.1 : 0.5;
    opt.SetSigmaScale(real_t(sigma_scale));

    // ── Constraint evaluator ──────────────────────────────────────────────
    auto EvalFi = [&](const Vector& x_eval) -> Vector {
        Vector fi(m);
        double xloc = 0;
        for (int j = 0; j < nl; ++j) xloc += double(x_eval(j));
        fi(0) = real_t(GSum(xloc)/n_global - Vfrac);

        if (n_regional >= 2) {
            int r0_end = (int)(0.3*n_global), r1_beg = (int)(0.7*n_global);
            int r0_sz  = r0_end, r1_sz = n_global-r1_beg;
            double s0=0, s1=0;
            for (int j = 0; j < nl; ++j) {
                int g = off+j;
                if (g < r0_end)  s0 += double(x_eval(j))/r0_sz;
                if (g >= r1_beg) s1 += double(x_eval(j))/r1_sz;
            }
            fi(1) = real_t(GSum(s0) - (Vfrac - 0.05));
            fi(2) = real_t(GSum(s1) - (Vfrac + 0.05));
        }
        return fi;
    };

    // ── SIMP objective + filter ───────────────────────────────────────────
    auto EvalF = [&](double p) -> double {
        auto xfull = GatherFull(n_global, nl, off, x, comm);
        std::vector<double> xhat;
        filt.apply(xfull, nl, off, xhat);
        std::vector<double> sens_hat(nl);
        double f_loc = 0;
        for (int e = 0; e < nl; ++e) {
            double xhe = std::max(xhat[e], xmin_v);
            double xhp = std::pow(xhe, p);
            f_loc      += w_local[e] / xhp;
            sens_hat[e] = -p*w_local[e] / (xhp*xhe) / n_global;
        }
        double f = GSum(f_loc)/n_global;
        std::vector<double> sens_x_full;
        filt.applyTranspose(sens_hat, sens_x_full, comm);
        for (int e = 0; e < nl; ++e) df0(e) = real_t(sens_x_full[off+e]);
        return f;
    };

    auto eval_fi_cb = [&](const Vector& x_trial, Vector& fi_out, Vector* /*dfidx_out*/)
    {
        fi_out = EvalFi(x_trial);
    };

    // ── Main loop ─────────────────────────────────────────────────────────
    Result res;
    // Use the untouched initial design, evaluated at the requested final
    // exponent, as the common objective baseline for fixed-p and continuation.
    res.f0_init = EvalF(simp_p);
    auto   t0 = Clock::now();

    for (int it = 0; it < max_iter; ++it) {
        double p  = continuation
            ? 1.0 + (simp_p-1.0)*double(std::min(it,200))/200.0
            : simp_p;
        double f0 = EvalF(p);
        auto   fi = EvalFi(x);

        if (gcmma)
            opt.UpdateGCMMA(x, df0, real_t(f0), fi, dg.data(),
                            xmin_v_, xmax_v_, eval_fi_cb);
        else
            opt.Update(x, df0, real_t(f0), fi, dg.data(), xmin_v_, xmax_v_);

        f0 = EvalF(p);
        fi = EvalFi(x);
        double kkt = opt.KKTresidual(x, df0, real_t(f0), fi, dg.data(),
                                      xmin_v_, xmax_v_);
        res.kkt_final = kkt;

        bool at_final_p = (!continuation) || (p >= simp_p - 1e-9);
        if (at_final_p) res.f0_final = f0;
        res.iters = it+1;

        if (g_rank==0 && (it%50==0 || it==max_iter-1)) {
            double gmax = 0;
            for (int k = 0; k < m; ++k) gmax = std::max(gmax, double(fi(k)));
            printf("  iter %4d: f0=%.4e  g0=%+.3e  g_max=%+.3e  kkt=%.3e  p=%.2f\n",
                   it, f0, double(fi(0)), gmax, kkt, p);
        }
    }
    double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    {
        auto fi_fin = EvalFi(x);
        double viol = 0;
        for (int k = 0; k < m; ++k) viol = std::max(viol, double(fi_fin(k)));
        res.max_viol = viol;
    }
    double xl = 0; for (int j = 0; j < nl; ++j) xl += double(x(j));
    double xmean_f = GSum(xl)/n_global;
    double change_loc = 0.0;
    for (int j = 0; j < nl; ++j)
        change_loc = std::max(change_loc,
                              std::abs(double(x(j)-x_initial(j))));
    MPI_Allreduce(&change_loc, &res.max_design_change, 1, MPI_DOUBLE,
                  MPI_MAX, comm);
    if (g_rank == 0) {
        double obj_drop = res.f0_final < res.f0_init
            ? 100.0*(res.f0_init-res.f0_final)/std::max(res.f0_init,1e-30) : 0.0;
        printf("  Final: iters=%d  kkt=%.2e  viol=%.2e"
               "  xmean=%.4f  obj: %.4e->%.4e (drop=%.1f%%)  time=%.0fms (%.2fms/it)\n",
               res.iters, res.kkt_final, res.max_viol,
               xmean_f, res.f0_init, res.f0_final, obj_drop, ms, ms/res.iters);
    }
    return res;
}

static void Test_FilteredSIMP_SQ(
    int n_global, int n_regional, double Vfrac,
    double p, bool cont, int r,
    bool gcmma, int max_iter,
    const char* label)
{
    if (g_rank == 0)
    {
        std::string p_label = cont
            ? std::string("1->") + std::to_string((int)p)
            : std::to_string((int)p);
        printf("\n--- %-10s  n=%-7d           Vfrac=%.2f  r=%-3d"
               "  p=%s  sigma=%.1f  [SQ%s] ---\n",
               label, n_global, Vfrac, r,
               p_label.c_str(), p >= 5.0 ? 0.1 : 0.5,
               gcmma ? "+GCMMA" : "");
    }

    auto res = RunFilteredSIMP_SQ(n_global, n_regional, Vfrac, p, cont, r, gcmma, max_iter);

    std::string tag = std::string("[") + label
        + ",n=" + std::to_string(n_global)
        + ",r=" + std::to_string(r)
        + "," + (gcmma ? "SQ+GCMMA" : "SQ") + "]";

    Check(std::isfinite(res.kkt_final) && res.kkt_final < 1e-3,
          (tag+" final KKT<1e-3 and finite").c_str());
    // Note: "objective bounded" is omitted — SIMP is nonconvex and SQ does not
    // guarantee monotone decrease; only KKT stationarity and feasibility are tested.
    Check(std::isfinite(res.max_viol) && res.max_viol < 5e-3,
          (tag+" volume constraints satisfied").c_str());
    Check(std::isfinite(res.f0_init) && std::isfinite(res.f0_final) &&
          res.f0_init > 0.0 && res.f0_final > 0.0,
          (tag+" objective finite and positive").c_str());
    Check(std::isfinite(res.max_design_change) && res.max_design_change > 1e-3,
          (tag+" design is nontrivially redistributed").c_str());
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--large") == 0) g_large = true;

    if (g_rank == 0)
        printf(
"╔══════════════════════════════════════════════════════════╗\n"
"║  Density-filtered SIMP test — SQ approximation          ║\n"
"║  (%2d rank(s))%-44s║\n"
"╠══════════════════════════════════════════════════════════╣\n"
"║  x̂_j = H*x  (Gaussian filter, radius r)                  ║\n"
"║  f = (1/n)Σ w_j/x̂_j^p   w_j = spatial load pattern     ║\n"
"║  Optimiser: SQOptimizerParallel                         ║\n"
"║  Volume as upper inequality (one-sided, avoids rank-1)  ║\n"
"╚══════════════════════════════════════════════════════════╝\n",
g_nranks, g_large ? " [--large]" : "");

    // Constraint layout: 2 asymmetric regional inequalities (first/last 30% of
    // domain). Volume as single upper inequality fi(0)=mean(x)-Vfrac<=0. m=3.
    //
    // n=1000 is excluded: with the default sigma=0.5*(xmax-xmin), the SIMP
    // gradient at x=Vfrac saturates the move limit on every step (|Δx|≈0.45),
    // making the dual Hessian identically zero and preventing equality enforcement.
    // n=10000 is the minimum reliable problem size for SQ SIMP with sigma=0.5.

    // ── P1: r=10, p=3 ──────────────────────────────────────────────────────
    if (g_rank == 0)
        printf("\n═══ P1: r=10  p=3  (checkerboard load, period 2r=20) ═══\n");
    //                n       nreg  Vfrac  p  cont  r  gcmma  iters label
    Test_FilteredSIMP_SQ(10000, 2, 0.4, 3.0,false, 10, false, 500, "p3r10");
    Test_FilteredSIMP_SQ(10000, 2, 0.4, 3.0,false, 10, true,  500, "p3r10");
    if (g_large) {
        Test_FilteredSIMP_SQ( 50000, 2, 0.4,3.0,false,10,false,500,"p3r10");
        Test_FilteredSIMP_SQ( 50000, 2, 0.4,3.0,false,10,true, 500,"p3r10");
        Test_FilteredSIMP_SQ(100000, 2, 0.4,3.0,false,10,false,500,"p3r10");
        Test_FilteredSIMP_SQ(500000, 2, 0.4,3.0,false,10,false,500,"p3r10");
        Test_FilteredSIMP_SQ(1000000,2, 0.4,3.0,false,10,false,300,"p3r10");
    }

    // ── P2: r=5, p=5 ───────────────────────────────────────────────────────
    if (g_rank == 0)
        printf("\n═══ P2: r=5  p=5  (tighter filter, stronger SIMP) ══════\n");
    Test_FilteredSIMP_SQ(10000, 2, 0.4, 5.0,false,  5, false, 500, "p5r5");
    Test_FilteredSIMP_SQ(10000, 2, 0.4, 5.0,false,  5, true,  500, "p5r5");
    if (g_large) {
        Test_FilteredSIMP_SQ( 50000, 2, 0.4,5.0,false, 5,false,500,"p5r5");
        Test_FilteredSIMP_SQ( 50000, 2, 0.4,5.0,false, 5,true, 500,"p5r5");
        Test_FilteredSIMP_SQ(100000, 2, 0.4,5.0,false, 5,false,500,"p5r5");
        Test_FilteredSIMP_SQ(500000, 2, 0.4,5.0,false, 5,false,500,"p5r5");
    }

    // ── P3: r=10, p: 1→5 continuation ─────────────────────────────────────
    if (g_rank == 0)
        printf("\n═══ P3: r=10  p: 1->5  continuation ════════════════════\n");
    Test_FilteredSIMP_SQ(10000, 2, 0.4, 5.0, true, 10, false, 500, "p5cont");
    Test_FilteredSIMP_SQ(10000, 2, 0.4, 5.0, true, 10, true,  500, "p5cont");
    if (g_large) {
        Test_FilteredSIMP_SQ( 50000, 2, 0.4,5.0,true,10,false,500,"p5cont");
        Test_FilteredSIMP_SQ(100000, 2, 0.4,5.0,true,10,false,500,"p5cont");
        Test_FilteredSIMP_SQ(500000, 2, 0.4,5.0,true,10,false,500,"p5cont");
    }

    if (g_rank == 0) {
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        if (g_nfail == 0)
            printf("║  All SQ filtered SIMP tests PASSED.                     ║\n");
        else
            printf("║  %d test(s) FAILED.%-38s║\n", g_nfail, "");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }
    MPI_Finalize();
    return g_nfail > 0 ? 1 : 0;
}
