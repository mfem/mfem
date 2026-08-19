/**
 * @file test_lmg_functions.hpp
 * @brief Wikipedia test functions for optimization, adapted for the
 *        LatentMirrorOptimizer interface.
 *
 * Each function exposes:
 *   - bounds lo, hi  (host vectors)
 *   - a strictly-interior starting point x0
 *   - phi(lam)    : scalar objective (host read)
 *   - grad(lam,d) : Euclidean gradient written into d (device forall)
 *   - known minimum value f* and (for 2-D) minimizer x*
 *
 * References: https://en.wikipedia.org/wiki/Test_functions_for_optimization
 *
 * All functions are separable-friendly (suitable for bound-constrained
 * gradient methods) with well-defined box domains.  Non-separable functions
 * (Rosenbrock, Griewank, etc.) are included because they stress the GBB
 * curvature estimation.
 *
 * Conventions:
 *   - "serial"   functions work on a single Vector of length n.
 *   - "parallel" variants split the n variables across ranks; the caller
 *     provides the local slice and a global-reduction lambda for the scalar
 *     objective terms that couple variables across the whole vector.
 */

#pragma once

#include "LatentMirrorOptimizer.hpp"
#include <mfem.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lmg_testfn {

using namespace mfem_lmg;  // BoundPartition, PrimalToLatent, etc.

// ── Helper: device-aware forall gradient skeleton ─────────────────────────

using mfem::real_t;
using mfem::Vector;
using mfem::forall_switch;

// ============================================================
//  1. Sphere  f(x) = Σ xᵢ²   min = 0 at x* = 0
//     Domain: unbounded (use [-10,10] for tests)
// ============================================================
struct Sphere {
    int    n;
    double f_star = 0.0;

    explicit Sphere(int n_) : n(n_) {}

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(n); hi.SetSize(n);
        lo = real_t(-10); hi = real_t(10);
    }
    void x0(Vector& x) const {
        x.SetSize(n);
        real_t* xp = x.HostWrite();
        for (int i = 0; i < n; ++i) xp[i] = real_t(3.0 + 0.1*i);  // away from 0
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += double(lp[i])*double(lp[i]);
        return s;
    }
    void grad(const Vector& lam, Vector& d) const {
        const bool ud = lam.UseDevice();
        d.UseDevice(ud);
        const real_t* lp = lam.Read();
        real_t*       dp = d.Write();
        forall_switch(ud, n, [=] MFEM_HOST_DEVICE (int i){
            dp[i] = real_t(2)*lp[i];
        });
    }
    // For parallel: each rank owns a local slice; phi is the sum of local phis.
    double phi_local(const Vector& lam_local) const { return phi(lam_local); }
    void   grad_local(const Vector& lam_local, Vector& d_local) const {
        grad(lam_local, d_local);
    }
};


// ============================================================
//  2. Rastrigin  f(x) = An + Σ [xᵢ² - A cos(2π xᵢ)]  A=10
//     min = 0 at x* = 0    Domain: [-5.12, 5.12]^n
// ============================================================
struct Rastrigin {
    int    n;
    double f_star = 0.0;
    static constexpr double A = 10.0;

    explicit Rastrigin(int n_) : n(n_) {}

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(n); hi.SetSize(n);
        lo = real_t(-5.12); hi = real_t(5.12);
    }
    void x0(Vector& x) const {
        x.SetSize(n);
        real_t* xp = x.HostWrite();
        for (int i = 0; i < n; ++i) xp[i] = real_t(1.0 + 0.3*(i%5));
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double s = A * n;
        for (int i = 0; i < n; ++i) {
            double xi = double(lp[i]);
            s += xi*xi - A*std::cos(2*M_PI*xi);
        }
        return s;
    }
    void grad(const Vector& lam, Vector& d) const {
        const bool ud = lam.UseDevice();
        d.UseDevice(ud);
        const real_t* lp = lam.Read();
        real_t*       dp = d.Write();
        const real_t  twopiA = real_t(2*M_PI*A);
        forall_switch(ud, n, [=] MFEM_HOST_DEVICE (int i){
            dp[i] = real_t(2)*lp[i] + twopiA*std::sin(real_t(2*M_PI)*lp[i]);
        });
    }
    double phi_local(const Vector& lam_local) const {
        // Each rank contributes A*n_local + local sum; caller adds A*n_global - A*n_local
        const real_t* lp = lam_local.HostRead();
        const int     nl = lam_local.Size();
        double s = A * nl;   // local An term
        for (int i = 0; i < nl; ++i) {
            double xi = double(lp[i]);
            s += xi*xi - A*std::cos(2*M_PI*xi);
        }
        return s;
    }
    void grad_local(const Vector& lam_local, Vector& d_local) const {
        grad(lam_local, d_local);
    }
};


// ============================================================
//  3. Rosenbrock  f(x) = Σᵢ₌₁ⁿ⁻¹ [100(xᵢ₊₁-xᵢ²)² + (1-xᵢ)²]
//     min = 0 at x* = (1,...,1)   Domain: [-2,2]^n (practical)
//
//  NOTE: Rosenbrock couples adjacent variables, so in the parallel
//  case ranks need to communicate boundary values.  We provide a
//  local gradient that assumes each rank holds a contiguous slice
//  [offset, offset+n_local) and the left-boundary xᵢ₋₁ is passed
//  in as a scalar "left_val" (0 on rank 0 is approximated by 1).
// ============================================================
struct Rosenbrock {
    int    n;
    double f_star = 0.0;

    explicit Rosenbrock(int n_) : n(n_) {}

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(n); hi.SetSize(n);
        lo = real_t(-2); hi = real_t(2);
    }
    void x0(Vector& x) const {
        x.SetSize(n);
        real_t* xp = x.HostWrite();
        for (int i = 0; i < n; ++i) xp[i] = real_t(-1.0 + 0.3*(i%7));
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double s = 0.0;
        for (int i = 0; i < n-1; ++i) {
            double xi = double(lp[i]), xi1 = double(lp[i+1]);
            double t  = xi1 - xi*xi;
            double u  = 1.0 - xi;
            s += 100.0*t*t + u*u;
        }
        return s;
    }
    void grad(const Vector& lam, Vector& d) const {
        // Host-side because of the coupling between adjacent elements
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        dp[0] = real_t(0);
        for (int i = 0; i < n-1; ++i) {
            double xi  = double(lp[i]);
            double xi1 = double(lp[i+1]);
            double t   = xi1 - xi*xi;
            // ∂f/∂xᵢ  contribution from term i:
            //   200(xi1 - xi²)(-2xi) + 2(xi - 1)
            double gi = -400.0*xi*t + 2.0*(xi - 1.0);
            // ∂f/∂xᵢ  contribution from term i-1 (if i>0):
            //   200(xi - x_{i-1}²)
            if (i > 0) {
                double xm1 = double(lp[i-1]);
                gi += 200.0*(xi - xm1*xm1);
            }
            dp[i] = real_t(gi);
            // Last variable: ∂f/∂xₙ = 200(xₙ - xₙ₋₁²)
            if (i == n-2) dp[n-1] = real_t(200.0*(xi1 - xi*xi));
        }
        if (n == 1) dp[0] = real_t(0);
        // Sync to device if needed
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
//  4. Styblinski-Tang  f(x) = Σᵢ (xᵢ⁴ - 16xᵢ² + 5xᵢ) / 2
//     min ≈ -39.16617 n  at x* = (-2.903534,...,-2.903534)
//     Domain: [-5, 5]^n
// ============================================================
struct StyblinskiTang {
    int    n;
    double f_star;

    explicit StyblinskiTang(int n_)
        : n(n_), f_star(-39.16617 * n_) {}

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(n); hi.SetSize(n);
        lo = real_t(-5); hi = real_t(5);
    }
    void x0(Vector& x) const {
        // Start at -2.0: in the basin of the global minimum at -2.9035.
        // The local minimum at +2.75 has f ≈ -25.0 (not -39.17), so
        // starting near +3 would miss the global optimum.
        x.SetSize(n);
        real_t* xp = x.HostWrite();
        for (int i = 0; i < n; ++i) xp[i] = real_t(-2.0);
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            double xi = double(lp[i]);
            s += xi*xi*xi*xi - 16.0*xi*xi + 5.0*xi;
        }
        return 0.5*s;
    }
    void grad(const Vector& lam, Vector& d) const {
        const bool ud = lam.UseDevice();
        d.UseDevice(ud);
        const real_t* lp = lam.Read();
        real_t*       dp = d.Write();
        forall_switch(ud, n, [=] MFEM_HOST_DEVICE (int i){
            double xi = double(lp[i]);
            dp[i] = real_t(0.5*(4.0*xi*xi*xi - 32.0*xi + 5.0));
        });
    }
    double phi_local(const Vector& lam_local) const { return phi(lam_local); }
    void   grad_local(const Vector& lam_local, Vector& d_local) const {
        grad(lam_local, d_local);
    }
};


// ============================================================
//  5. Griewank  f(x) = 1 + Σxᵢ²/4000 - ΠP_i(xᵢ)
//     P_i(xi) = cos(xi / sqrt(i))
//     min = 0 at x* = 0    Domain: [-600,600]^n (use [-10,10])
//
//  NOTE: The product term couples all variables; we split phi into
//  a "sum part" (separable, parallelisable) and a "product part"
//  (global reduction needed in parallel).
// ============================================================
struct Griewank {
    int    n;
    double f_star = 0.0;

    explicit Griewank(int n_) : n(n_) {}

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(n); hi.SetSize(n);
        lo = real_t(-10); hi = real_t(10);
    }
    void x0(Vector& x) const {
        x.SetSize(n);
        real_t* xp = x.HostWrite();
        for (int i = 0; i < n; ++i) xp[i] = real_t(2.0 + 0.5*(i%7));
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double sum = 0.0, prod = 1.0;
        for (int i = 0; i < n; ++i) {
            double xi = double(lp[i]);
            sum  += xi*xi / 4000.0;
            prod *= std::cos(xi / std::sqrt(double(i+1)));
        }
        return 1.0 + sum - prod;
    }
    // Gradient: ∂f/∂xᵢ = xᵢ/2000 + sin(xᵢ/√i) * P / cos(xᵢ/√i)
    //                   = xᵢ/2000 + sin(xᵢ/√i) * Π_{j≠i} cos(xⱼ/√j) * 1/√i
    // Host-side because the product needs to be computed first.
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        // Compute product P = Π cos(xᵢ/√i)
        double prod = 1.0;
        for (int i = 0; i < n; ++i)
            prod *= std::cos(double(lp[i]) / std::sqrt(double(i+1)));
        for (int i = 0; i < n; ++i) {
            double xi  = double(lp[i]);
            double si  = std::sqrt(double(i+1));
            double ci  = std::cos(xi / si);
            double P_i = (std::abs(ci) > 1e-15) ? prod / ci : 0.0; // P without factor i
            dp[i] = real_t(xi / 2000.0 + std::sin(xi / si) * P_i / si);
        }
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
//  6. Beale (2-D only)
//     f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
//     min = 0 at (3, 0.5)   Domain: [-4.5, 4.5]²
// ============================================================
struct Beale {
    int n = 2;
    double f_star = 0.0;
    double x_star[2] = {3.0, 0.5};

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(2); hi.SetSize(2);
        lo = real_t(-4.5); hi = real_t(4.5);
    }
    void x0(Vector& x) const {
        x.SetSize(2);
        x(0) = real_t(1.0); x(1) = real_t(-1.0);
    }
    double phi(const Vector& lam) const {
        double x = double(lam.HostRead()[0]);
        double y = double(lam.HostRead()[1]);
        double a = 1.5   - x + x*y;
        double b = 2.25  - x + x*y*y;
        double c = 2.625 - x + x*y*y*y;
        return a*a + b*b + c*c;
    }
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        double x = double(lp[0]), y = double(lp[1]);
        double a = 1.5   - x + x*y;
        double b = 2.25  - x + x*y*y;
        double c = 2.625 - x + x*y*y*y;
        dp[0] = real_t(2*a*(-1+y) + 2*b*(-1+y*y) + 2*c*(-1+y*y*y));
        dp[1] = real_t(2*a*(x) + 2*b*(2*x*y) + 2*c*(3*x*y*y));
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
//  7. Booth (2-D only)
//     f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
//     min = 0 at (1, 3)   Domain: [-10, 10]²
// ============================================================
struct Booth {
    int n = 2;
    double f_star = 0.0;
    double x_star[2] = {1.0, 3.0};

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(2); hi.SetSize(2);
        lo = real_t(-10); hi = real_t(10);
    }
    void x0(Vector& x) const {
        x.SetSize(2);
        x(0) = real_t(-5.0); x(1) = real_t(-5.0);
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double x = double(lp[0]), y = double(lp[1]);
        double a = x + 2*y - 7;
        double b = 2*x + y - 5;
        return a*a + b*b;
    }
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        double x = double(lp[0]), y = double(lp[1]);
        double a = x + 2*y - 7;
        double b = 2*x + y - 5;
        dp[0] = real_t(2*a + 4*b);
        dp[1] = real_t(4*a + 2*b);
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
//  8. Matyas (2-D only)
//     f(x,y) = 0.26(x² + y²) - 0.48xy
//     min = 0 at (0, 0)   Domain: [-10, 10]²
// ============================================================
struct Matyas {
    int n = 2;
    double f_star = 0.0;
    double x_star[2] = {0.0, 0.0};

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(2); hi.SetSize(2);
        lo = real_t(-10); hi = real_t(10);
    }
    void x0(Vector& x) const {
        x.SetSize(2);
        x(0) = real_t(5.0); x(1) = real_t(-5.0);
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double x = double(lp[0]), y = double(lp[1]);
        return 0.26*(x*x + y*y) - 0.48*x*y;
    }
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        double x = double(lp[0]), y = double(lp[1]);
        dp[0] = real_t(0.52*x - 0.48*y);
        dp[1] = real_t(0.52*y - 0.48*x);
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
//  9. Lévi N.13 (2-D only)
//     f(x,y) = sin²(3πx) + (x-1)²(1+sin²(3πy)) + (y-1)²(1+sin²(2πy))
//     min = 0 at (1, 1)   Domain: [-10, 10]²
// ============================================================
struct Levi13 {
    int n = 2;
    double f_star = 0.0;
    double x_star[2] = {1.0, 1.0};

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(2); hi.SetSize(2);
        lo = real_t(-10); hi = real_t(10);
    }
    void x0(Vector& x) const {
        // Start at (0.9, 0.9): inside the basin of the global minimum (1,1).
        // Levi13 has sinusoidal ridges creating local minima at x=k/3, y=1;
        // the nearest ones are at x≈0.667 and x≈1.333 with f≈0.111.
        // Any start with |x-1|<~0.15 and |y-1|<~0.4 reliably converges to (1,1).
        x.SetSize(2);
        x(0) = real_t(0.9); x(1) = real_t(0.9);
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double x = double(lp[0]), y = double(lp[1]);
        double s3px = std::sin(3*M_PI*x);
        double s3py = std::sin(3*M_PI*y);
        double s2py = std::sin(2*M_PI*y);
        return s3px*s3px
             + (x-1)*(x-1)*(1 + s3py*s3py)
             + (y-1)*(y-1)*(1 + s2py*s2py);
    }
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        double x = double(lp[0]), y = double(lp[1]);
        double s3px = std::sin(3*M_PI*x), c3px = std::cos(3*M_PI*x);
        double s3py = std::sin(3*M_PI*y), c3py = std::cos(3*M_PI*y);
        double s2py = std::sin(2*M_PI*y), c2py = std::cos(2*M_PI*y);
        dp[0] = real_t(2*3*M_PI*s3px*c3px
                + 2*(x-1)*(1 + s3py*s3py));
        dp[1] = real_t((x-1)*(x-1)*2*3*M_PI*s3py*c3py
                + 2*(y-1)*(1 + s2py*s2py)
                + (y-1)*(y-1)*2*2*M_PI*s2py*c2py);
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
// 10. Himmelblau (2-D only)
//     f(x,y) = (x² + y - 11)² + (x + y² - 7)²
//     4 minima = 0  Domain: [-5, 5]²
// ============================================================
struct Himmelblau {
    int n = 2;
    double f_star = 0.0;

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(2); hi.SetSize(2);
        lo = real_t(-5); hi = real_t(5);
    }
    // 4 basins — start from different points to hit different minima
    void x0(Vector& x, int basin = 0) const {
        x.SetSize(2);
        const double starts[4][2] = {{2,2},{-2,2},{-3,-3},{2,-3}};
        x(0) = real_t(starts[basin%4][0]);
        x(1) = real_t(starts[basin%4][1]);
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double x = double(lp[0]), y = double(lp[1]);
        double a = x*x + y - 11;
        double b = x + y*y - 7;
        return a*a + b*b;
    }
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        double x = double(lp[0]), y = double(lp[1]);
        double a = x*x + y - 11;
        double b = x + y*y - 7;
        dp[0] = real_t(4*x*a + 2*b);
        dp[1] = real_t(2*a + 4*y*b);
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
// 11. Three-hump camel (2-D only)
//     f(x,y) = 2x² - 1.05x⁴ + x⁶/6 + xy + y²
//     min = 0 at (0,0)   Domain: [-5, 5]²
// ============================================================
struct ThreeHumpCamel {
    int n = 2;
    double f_star = 0.0;
    double x_star[2] = {0.0, 0.0};

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(2); hi.SetSize(2);
        lo = real_t(-5); hi = real_t(5);
    }
    void x0(Vector& x) const {
        x.SetSize(2);
        x(0) = real_t(-2.0); x(1) = real_t(2.0);
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double x = double(lp[0]), y = double(lp[1]);
        return 2*x*x - 1.05*x*x*x*x + x*x*x*x*x*x/6.0 + x*y + y*y;
    }
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        double x = double(lp[0]), y = double(lp[1]);
        dp[0] = real_t(4*x - 4.2*x*x*x + x*x*x*x*x + y);
        dp[1] = real_t(x + 2*y);
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
// 12. McCormick (2-D only)
//     f(x,y) = sin(x+y) + (x-y)² - 1.5x + 2.5y + 1
//     min ≈ -1.9133 at (-0.54719, -1.54719)
//     Domain: -1.5 ≤ x ≤ 4,  -3 ≤ y ≤ 4
// ============================================================
struct McCormick {
    int n = 2;
    double f_star = -1.9133;
    double x_star[2] = {-0.54719, -1.54719};

    void bounds(Vector& lo, Vector& hi) const {
        lo.SetSize(2); hi.SetSize(2);
        lo(0) = real_t(-1.5); hi(0) = real_t(4.0);
        lo(1) = real_t(-3.0); hi(1) = real_t(4.0);
    }
    void x0(Vector& x) const {
        // Start near the global minimum at (-0.54719, -1.54719).
        // (2,2) lies in a different sinusoidal basin.
        x.SetSize(2);
        x(0) = real_t(-0.5); x(1) = real_t(-1.5);
    }
    double phi(const Vector& lam) const {
        const real_t* lp = lam.HostRead();
        double x = double(lp[0]), y = double(lp[1]);
        return std::sin(x+y) + (x-y)*(x-y) - 1.5*x + 2.5*y + 1.0;
    }
    void grad(const Vector& lam, Vector& d) const {
        const real_t* lp = lam.HostRead();
        real_t*       dp = d.HostWrite();
        double x = double(lp[0]), y = double(lp[1]);
        dp[0] = real_t(std::cos(x+y) + 2*(x-y) - 1.5);
        dp[1] = real_t(std::cos(x+y) - 2*(x-y) + 2.5);
        if (d.UseDevice()) d.Read();
    }
};


// ============================================================
//  Generic optimizer driver — runs LatentMirrorOptimizer on any function F
//  Returns: {final_phi, kkt, n_iters}
// ============================================================

struct RunResult { double phi; double kkt; int iters; };

template<typename F>
RunResult RunSerial(F& fn, bool use_dev = false,
                   int max_iter = 2000, double kkt_tol = 1e-5)
{
    const int n = fn.n;

    Vector lo(n), hi(n);
    fn.bounds(lo, hi);
    lo.UseDevice(use_dev);
    hi.UseDevice(use_dev);

    std::vector<BoundType> types;
    ClassifyBounds(lo, hi, types);
    BoundPartition part(lo, hi, lo);

    // Strictly feasible x0
    Vector lam_init(n); lam_init.UseDevice(use_dev);
    fn.x0(lam_init);
    // Clamp strictly inside the box (handles exact-boundary x0).
    {
        real_t* lp = lam_init.HostReadWrite();
        const real_t* ld = lo.HostRead();
        const real_t* ud = hi.HostRead();
        for (int i = 0; i < n; ++i) {
            if (types[i] == BoundType::LowerOnly ||
                types[i] == BoundType::TwoSided)
                lp[i] = std::max(lp[i], ld[i] + real_t(1e-4));
            if (types[i] == BoundType::UpperOnly ||
                types[i] == BoundType::TwoSided)
                lp[i] = std::min(lp[i], ud[i] - real_t(1e-4));
        }
    }

    Vector z(n); z.UseDevice(use_dev);
    PrimalToLatent(lam_init, part, z);

    LatentMirrorOptimizer opt(z, lo, hi);

    Vector lam(n), d(n);
    lam.UseDevice(use_dev); d.UseDevice(use_dev);

    double kkt = 1.0, phi_val = 0.0;
    for (int it = 0; it < max_iter && kkt > kkt_tol; ++it) {
        LatentToPrimal(z, part, lam);
        phi_val = fn.phi(lam);
        fn.grad(lam, d);
        opt.Update(z, d, real_t(phi_val),
            [&](const Vector& zt, real_t& pout) {
                Vector lt(n); lt.UseDevice(use_dev);
                LatentToPrimal(zt, part, lt);
                pout = real_t(fn.phi(lt));
            });
        LatentToPrimal(z, part, lam);
        fn.grad(lam, d);
        kkt = double(opt.StationarityResidual(z, d));
    }
    return {phi_val, kkt, opt.NumIterations()};
}

} // namespace lmg_testfn
