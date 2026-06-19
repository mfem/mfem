/**
 * test_packfival.cpp  --  Unit tests for PackFival and PackedDfidx
 *
 * These are pure header-level helpers (no solver calls needed) that pack
 * inequality and equality constraint data into the internal +-h layout used
 * by WithEqualities().
 *
 * Layout convention (documented in MMA_MFEM.hpp):
 *
 *   fival  : [ fi_ineq(0..n_ineq-1) | +h_eq(0..n_eq-1) | -h_eq(0..n_eq-1) ]
 *   dfidx  : same row order, with the -h rows sign-flipped
 *
 * Test catalogue
 * --------------
 *  1. PackFival -- pure equality (n_ineq=0, n_eq=1)
 *  2. PackFival -- pure equality (n_ineq=0, n_eq=2)
 *  3. PackFival -- mixed (n_ineq=2, n_eq=1)
 *  4. PackFival -- mixed (n_ineq=1, n_eq=3)
 *  5. PackFival -- empty (n_ineq=0, n_eq=0)
 *  6. PackedDfidx -- pure equality (n_ineq=0, n_eq=1)
 *  7. PackedDfidx -- mixed (n_ineq=2, n_eq=2)
 *  8. PackedDfidx -- sign flip does not alias the original gradient
 *  9. WithEqualities() -- NumConstraints / NumEqualities / NumInequalities
 * 10. WithEqualities() -- Update does not crash with correctly sized fival
 * 11. Round-trip -- PackFival drives a mean equality constraint to convergence
 *
 * Build:
 *   cmake --build build --target test_packfival
 * Run:
 *   ./build/test_packfival
 */

#include "MMA_MFEM.hpp"
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace mfem_mma;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static int g_rank  = 0;
static int g_nfail = 0;
static int g_npass = 0;

static void Check(bool cond, const char* msg)
{
    if (g_rank != 0) return;
    if (cond) { printf("  [PASS] %s\n", msg); ++g_npass; }
    else      { printf("  [FAIL] %s\n", msg); ++g_nfail; }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static mfem::Vector ConstVec(int n, double val)
{
    mfem::Vector v(n);
    v = mfem::real_t(val);
    return v;
}

static mfem::Vector FromList(std::initializer_list<double> vals)
{
    mfem::Vector v((int)vals.size());
    int i = 0;
    for (double d : vals) v(i++) = mfem::real_t(d);
    return v;
}

static bool allclose(const mfem::Vector& v, double expected, double tol = 1e-14)
{
    for (int j = 0; j < v.Size(); ++j)
        if (std::abs(double(v(j)) - expected) > tol) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Test 1: PackFival -- pure equality, n_ineq=0, n_eq=1
// ---------------------------------------------------------------------------
static void Test1()
{
    if (g_rank==0) printf("\n-- Test 1: PackFival (n_ineq=0, n_eq=1) -----------\n");

    mfem::Vector fi_ineq(0);
    mfem::Vector h_eq = FromList({0.07});

    mfem::Vector fival = PackFival(fi_ineq, h_eq);

    Check(fival.Size() == 2,                                   "size = 2");
    Check(std::abs(double(fival(0)) -  0.07) < 1e-14,         "fival[0] = +h");
    Check(std::abs(double(fival(1)) - (-0.07)) < 1e-14,        "fival[1] = -h");
}

// ---------------------------------------------------------------------------
// Test 2: PackFival -- pure equality, n_ineq=0, n_eq=2
// ---------------------------------------------------------------------------
static void Test2()
{
    if (g_rank==0) printf("\n-- Test 2: PackFival (n_ineq=0, n_eq=2) -----------\n");

    mfem::Vector fi_ineq(0);
    mfem::Vector h_eq = FromList({0.1, -0.05});

    mfem::Vector fival = PackFival(fi_ineq, h_eq);

    Check(fival.Size() == 4,                                   "size = 4");
    Check(std::abs(double(fival(0)) -  0.10) < 1e-14,         "fival[0] = +h[0]");
    Check(std::abs(double(fival(1)) - (-0.05)) < 1e-14,        "fival[1] = +h[1]");
    Check(std::abs(double(fival(2)) - (-0.10)) < 1e-14,        "fival[2] = -h[0]");
    Check(std::abs(double(fival(3)) -  0.05) < 1e-14,         "fival[3] = -h[1]");
}

// ---------------------------------------------------------------------------
// Test 3: PackFival -- mixed, n_ineq=2, n_eq=1
// ---------------------------------------------------------------------------
static void Test3()
{
    if (g_rank==0) printf("\n-- Test 3: PackFival (n_ineq=2, n_eq=1) -----------\n");

    mfem::Vector fi_ineq = FromList({-0.3, 0.15});
    mfem::Vector h_eq    = FromList({0.08});

    mfem::Vector fival = PackFival(fi_ineq, h_eq);

    Check(fival.Size() == 4,                                   "size = n_ineq+2*n_eq = 4");
    Check(std::abs(double(fival(0)) - (-0.3)) < 1e-14,         "fival[0] = fi[0]");
    Check(std::abs(double(fival(1)) -  0.15) < 1e-14,          "fival[1] = fi[1]");
    Check(std::abs(double(fival(2)) -  0.08) < 1e-14,          "fival[2] = +h");
    Check(std::abs(double(fival(3)) - (-0.08)) < 1e-14,        "fival[3] = -h");
}

// ---------------------------------------------------------------------------
// Test 4: PackFival -- mixed, n_ineq=1, n_eq=3
// ---------------------------------------------------------------------------
static void Test4()
{
    if (g_rank==0) printf("\n-- Test 4: PackFival (n_ineq=1, n_eq=3) -----------\n");

    mfem::Vector fi_ineq = FromList({-0.2});
    mfem::Vector h_eq    = FromList({0.1, -0.2, 0.3});

    mfem::Vector fival = PackFival(fi_ineq, h_eq);

    // Expected: [-0.2 | 0.1 -0.2 0.3 | -0.1 0.2 -0.3]
    Check(fival.Size() == 7,                                    "size = 1+2*3 = 7");
    Check(std::abs(double(fival(0)) - (-0.2)) < 1e-14,          "fival[0] = fi[0]");
    Check(std::abs(double(fival(1)) -  0.1)  < 1e-14,           "fival[1] = +h[0]");
    Check(std::abs(double(fival(2)) - (-0.2)) < 1e-14,          "fival[2] = +h[1]");
    Check(std::abs(double(fival(3)) -  0.3)  < 1e-14,           "fival[3] = +h[2]");
    Check(std::abs(double(fival(4)) - (-0.1)) < 1e-14,          "fival[4] = -h[0]");
    Check(std::abs(double(fival(5)) -  0.2)  < 1e-14,           "fival[5] = -h[1]");
    Check(std::abs(double(fival(6)) - (-0.3)) < 1e-14,          "fival[6] = -h[2]");
}

// ---------------------------------------------------------------------------
// Test 5: PackFival -- empty (n_ineq=0, n_eq=0)
// ---------------------------------------------------------------------------
static void Test5()
{
    if (g_rank==0) printf("\n-- Test 5: PackFival (n_ineq=0, n_eq=0) -----------\n");

    mfem::Vector fi_ineq(0), h_eq(0);
    mfem::Vector fival = PackFival(fi_ineq, h_eq);

    Check(fival.Size() == 0, "size = 0");
}

// ---------------------------------------------------------------------------
// Test 6: PackedDfidx -- pure equality, n_ineq=0, n_eq=1
// ---------------------------------------------------------------------------
static void Test6()
{
    if (g_rank==0) printf("\n-- Test 6: PackedDfidx (n_ineq=0, n_eq=1) ---------\n");

    const int n = 6;
    mfem::Vector dh(n);
    for (int j = 0; j < n; ++j) dh(j) = mfem::real_t(j + 1);  // [1,2,3,4,5,6]

    PackedDfidx packed(nullptr, 0, &dh, 1);
    Check(packed.size() == 2, "size = 0 + 2*1 = 2 rows");

    const mfem::Vector* rows = packed.data();
    bool plus_ok = true, minus_ok = true;
    for (int j = 0; j < n; ++j) {
        if (std::abs(double(rows[0](j)) - (j+1)) > 1e-14) plus_ok  = false;
        if (std::abs(double(rows[1](j)) + (j+1)) > 1e-14) minus_ok = false;
    }
    Check(plus_ok,  "row 0 = +dh");
    Check(minus_ok, "row 1 = -dh");
}

// ---------------------------------------------------------------------------
// Test 7: PackedDfidx -- mixed, n_ineq=2, n_eq=2
// ---------------------------------------------------------------------------
static void Test7()
{
    if (g_rank==0) printf("\n-- Test 7: PackedDfidx (n_ineq=2, n_eq=2) ---------\n");

    const int n = 4;
    mfem::Vector dfi0 = ConstVec(n,  1.0);
    mfem::Vector dfi1 = ConstVec(n,  2.0);
    mfem::Vector dh0  = ConstVec(n,  3.0);
    mfem::Vector dh1  = ConstVec(n, -5.0);

    mfem::Vector dfi_arr[2] = {dfi0, dfi1};
    mfem::Vector dh_arr[2]  = {dh0, dh1};

    PackedDfidx packed(dfi_arr, 2, dh_arr, 2);
    Check(packed.size() == 6, "size = 2 + 2*2 = 6 rows");

    const mfem::Vector* rows = packed.data();
    // Expected: dfi0, dfi1, +dh0, +dh1, -dh0, -dh1
    Check(allclose(rows[0],  1.0), "row 0 = dfi[0]");
    Check(allclose(rows[1],  2.0), "row 1 = dfi[1]");
    Check(allclose(rows[2],  3.0), "row 2 = +dh[0]");
    Check(allclose(rows[3], -5.0), "row 3 = +dh[1]");
    Check(allclose(rows[4], -3.0), "row 4 = -dh[0]");
    Check(allclose(rows[5],  5.0), "row 5 = -dh[1]");
}

// ---------------------------------------------------------------------------
// Test 8: PackedDfidx -- sign-flipped rows are independent copies
// ---------------------------------------------------------------------------
static void Test8()
{
    if (g_rank==0) printf("\n-- Test 8: PackedDfidx no aliasing -----------------\n");

    const int n = 5;
    mfem::Vector dh(n);
    for (int j = 0; j < n; ++j) dh(j) = mfem::real_t(j + 1);

    PackedDfidx packed(nullptr, 0, &dh, 1);

    // Mutate the original; the packed rows must not change.
    dh = 0.0;

    const mfem::Vector* rows = packed.data();
    bool plus_ok = true, minus_ok = true;
    for (int j = 0; j < n; ++j) {
        if (std::abs(double(rows[0](j)) - (j+1)) > 1e-14) plus_ok  = false;
        if (std::abs(double(rows[1](j)) + (j+1)) > 1e-14) minus_ok = false;
    }
    Check(plus_ok,  "row 0 unchanged after mutating source");
    Check(minus_ok, "row 1 unchanged after mutating source");
}

// ---------------------------------------------------------------------------
// Test 9: WithEqualities() -- counter queries
// ---------------------------------------------------------------------------
static void Test9()
{
    if (g_rank==0) printf("\n-- Test 9: WithEqualities counts -------------------\n");

    const int n = 10;
    mfem::Vector x = ConstVec(n, 0.5);

    { // MMA pure equality
        auto opt = MMAOptimizer::WithEqualities(n, 0, 1, x);
        Check(opt.NumEqualities()   == 1, "MMA(0,1): NumEqualities==1");
        Check(opt.NumInequalities() == 0, "MMA(0,1): NumInequalities==0");
        Check(opt.NumConstraints()  == 1, "MMA(0,1): NumConstraints==1");
    }
    { // MMA mixed
        auto opt = MMAOptimizer::WithEqualities(n, 2, 3, x);
        Check(opt.NumEqualities()   == 3, "MMA(2,3): NumEqualities==3");
        Check(opt.NumInequalities() == 2, "MMA(2,3): NumInequalities==2");
        Check(opt.NumConstraints()  == 5, "MMA(2,3): NumConstraints==5");
    }
    { // SQ pure equality
        auto opt = SQOptimizer::WithEqualities(n, 0, 2, x);
        Check(opt.NumEqualities()   == 2, "SQ(0,2): NumEqualities==2");
        Check(opt.NumInequalities() == 0, "SQ(0,2): NumInequalities==0");
        Check(opt.NumConstraints()  == 2, "SQ(0,2): NumConstraints==2");
    }
    { // SQ mixed
        auto opt = SQOptimizer::WithEqualities(n, 1, 2, x);
        Check(opt.NumEqualities()   == 2, "SQ(1,2): NumEqualities==2");
        Check(opt.NumInequalities() == 1, "SQ(1,2): NumInequalities==1");
        Check(opt.NumConstraints()  == 3, "SQ(1,2): NumConstraints==3");
    }
}

// ---------------------------------------------------------------------------
// Test 10: WithEqualities() -- Update does not crash with correct fival size
// ---------------------------------------------------------------------------
static void Test10()
{
    if (g_rank==0) printf("\n-- Test 10: WithEqualities Update no-crash ---------\n");

    const int n = 20, n_ineq = 1, n_eq = 2;
    mfem::Vector x    = ConstVec(n, 0.5);
    mfem::Vector xmin = ConstVec(n, 0.01);
    mfem::Vector xmax = ConstVec(n, 1.0);
    mfem::Vector df0  = ConstVec(n, 0.0);

    mfem::Vector fi_ineq = FromList({-0.1});
    mfem::Vector h_eq    = FromList({0.0, 0.0});
    mfem::Vector fival   = PackFival(fi_ineq, h_eq);

    Check(fival.Size() == n_ineq + 2*n_eq, "fival.Size() == n_ineq + 2*n_eq");

    mfem::Vector dfi_arr[1]; dfi_arr[0] = ConstVec(n, 0.0);
    mfem::Vector dh_arr[2];
    dh_arr[0] = ConstVec(n, 1.0/n);
    dh_arr[1] = ConstVec(n, 1.0/n);
    PackedDfidx dfidx(dfi_arr, n_ineq, dh_arr, n_eq);

    Check(dfidx.size() == n_ineq + 2*n_eq, "dfidx.size() == n_ineq + 2*n_eq");

    auto opt = MMAOptimizer::WithEqualities(n, n_ineq, n_eq, x);
    bool threw = false;
    try {
        opt.Update(x, df0, 0.0, fival, dfidx.data(), xmin, xmax);
    } catch (...) { threw = true; }
    Check(!threw, "Update does not throw");

    double kkt = double(opt.KKTresidual(x, df0, 0.0, fival, dfidx.data(), xmin, xmax));
    Check(std::isfinite(kkt) && kkt >= 0.0, "KKTresidual is finite and >= 0");
}

// ---------------------------------------------------------------------------
// Test 11: Round-trip -- PackFival drives mean(x)=Vfrac to convergence
// ---------------------------------------------------------------------------
static void Test11()
{
    if (g_rank==0) printf("\n-- Test 11: round-trip mean(x)=Vfrac --------------\n");

    const int    n     = 100;
    const double Vfrac = 0.4;

    mfem::Vector x    = ConstVec(n, 0.5);   // mean(x)=0.5, target=0.4
    mfem::Vector xmin = ConstVec(n, 0.01);
    mfem::Vector xmax = ConstVec(n, 1.0);
    mfem::Vector df0(n);

    // Gradient of equality: dh/dx_j = 1/n
    mfem::Vector dh = ConstVec(n, 1.0/n);
    PackedDfidx dfidx(nullptr, 0, &dh, 1);

    mfem::Vector fi_ineq(0);

    auto opt = MMAOptimizer::WithEqualities(n, 0, 1, x);

    double kkt = 1.0;
    for (int it = 0; it < 100 && kkt > 1e-6; ++it) {
        // Objective: min (1/n)*sum((x-Vfrac)^2)
        double f0 = 0.0;
        for (int j = 0; j < n; ++j) {
            double xj = double(x(j));
            df0(j) = mfem::real_t(2.0*(xj - Vfrac)/n);
            f0 += (xj - Vfrac)*(xj - Vfrac)/n;
        }
        // h(x) = mean(x) - Vfrac
        double mean = 0.0;
        for (int j = 0; j < n; ++j) mean += double(x(j));
        mean /= n;
        mfem::Vector h_eq = FromList({mean - Vfrac});
        mfem::Vector fival = PackFival(fi_ineq, h_eq);

        opt.Update(x, df0, f0, fival, dfidx.data(), xmin, xmax);
        kkt = double(opt.KKTresidual(x, df0, f0, fival, dfidx.data(), xmin, xmax));
    }

    double final_mean = 0.0;
    for (int j = 0; j < n; ++j) final_mean += double(x(j));
    final_mean /= n;

    if (g_rank==0)
        printf("  kkt=%.2e  mean(x)=%.6f (target %.2f)\n", kkt, final_mean, Vfrac);

    Check(kkt < 1e-5,                             "KKT < 1e-5 after 100 iters");
    Check(std::abs(final_mean - Vfrac) < 1e-5,    "mean(x) == Vfrac");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    if (g_rank==0)
        printf("=== PackFival / PackedDfidx unit tests ===\n");

    Test1();
    Test2();
    Test3();
    Test4();
    Test5();
    Test6();
    Test7();
    Test8();
    Test9();
    Test10();
    Test11();

    if (g_rank==0) {
        printf("\n==========================================\n");
        if (g_nfail == 0)
            printf("All %d test(s) PASSED.\n", g_npass);
        else
            printf("%d test(s) FAILED, %d passed.\n", g_nfail, g_npass);
        printf("==========================================\n");
    }

    MPI_Finalize();
    return (g_nfail > 0) ? 1 : 0;
}
