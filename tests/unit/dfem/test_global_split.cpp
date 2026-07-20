#include "mfem.hpp"
#include "unit_tests.hpp"
#include "../../../linalg/tensor_arrays.hpp"
#include <cstdio>

#ifdef MFEM_USE_ENZYME

namespace enzyme_test_monolithic
{

using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

struct CubicQFunction
{
   void operator()(tensor_array<const dscalar_t> &x,
                   tensor_array<const dscalar_t> &coef,
                   tensor_array<dscalar_t> &y) const
   {
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      { y(q) = coef(q) * x(q) * x(q) * x(q); });
   }
};

// Enzyme is invoked on a plain function pointer, so this wrapper performs the
// same tensor-array flow DFEM would normally do before calling qfunctions.
template <int N>
void qfunction_wrapper(const dscalar_t *x, dscalar_t *y, const dscalar_t *coef)
{
   auto x_t = make_tensor_array(x, N);
   auto coef_t = make_tensor_array(coef, N);
   auto y_t = make_tensor_array(y, N);

   CubicQFunction qf;
   qf(x_t, coef_t, y_t);
}

} // namespace enzyme_test_monolithic

namespace enzyme_test_split
{

using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

struct CubicQFunctionWithScratch
{
   void operator()(tensor_array<const dscalar_t> &x,
                   tensor_array<const dscalar_t> &coef,
                   tensor_array<dscalar_t> &scratch,
                   tensor_array<dscalar_t> &y) const
   {
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      { scratch(q) = x(q); });

      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      { scratch(q) = scratch(q) * x(q); });

      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      { y(q) = coef(q) * scratch(q) * x(q); });
   }
};

template <int N>
void qfunction_wrapper(const dscalar_t *x, dscalar_t *y, const dscalar_t *coef,
                       dscalar_t *scratch)
{
   auto x_t = make_tensor_array(x, N);
   auto coef_t = make_tensor_array(coef, N);
   auto scratch_t = make_tensor_array(scratch, N);
   auto y_t = make_tensor_array(y, N);

   CubicQFunctionWithScratch qf;
   qf(x_t, coef_t, scratch_t, y_t);
}

inline void print_results(const mfem::Vector &x,
                          const mfem::Vector &coef,
                          const mfem::Vector &y,
                          const mfem::Vector &yd,
                          const mfem::Vector &scratch,
                          const mfem::Vector &scratchd)
{
   std::printf("Function: y = coef * x^3\n");
   std::printf("%3s %10s %10s %10s %10s %10s\n",
               "q", "y", "yd", "yd_ex", "sc", "scd");

   for (int q = 0; q < x.Size(); q++)
   {
      const double exact_yd = 3.0 * coef[q] * x[q] * x[q];

      std::printf("%3d %10.4g %10.4g %10.4g %10.4g %10.4g\n",
                  q, y[q], yd[q], exact_yd, scratch[q], scratchd[q]);
   }
}

} // namespace enzyme_test_split

TEST_CASE("Minimal Enzyme qfunction example",
          "[Enzyme][GPU][Global-Monolithic]")
{
   constexpr int N = 100;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);

   for (int i = 0; i < N; i++)
   {
      x(i) = i + 1.0; // nnz derivative
      xd(i) = 1.0;    // seed dx/dt = 1 to recover dy/dx
      y(i) = 0.0;
      yd(i) = 0.0;
      coef(i) = 0.5 + 0.25 * i; // makes coefficient per-entry
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();

   __enzyme_fwddiff<void>((void *)enzyme_test_monolithic::qfunction_wrapper<N>,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_const, coef_d,
                          enzyme_runtime_activity);

   yd.HostRead();
   for (int q = 0; q < N; q++)
   {
      const double exact = 3.0 * coef[q] * x[q] * x[q];
      REQUIRE(yd[q] == MFEM_Approx(exact));
   }
}

TEST_CASE("Enzyme qfunction with split scratch chain",
          "[Enzyme][GPU][Global-Split]")
{
   constexpr int N = 100;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N), scratch(N), scratchd(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);
   scratch.UseDevice(true);
   scratchd.UseDevice(true);

   for (int i = 0; i < N; i++)
   {
      x(i) = i + 1.0;
      xd(i) = 1.0;
      y(i) = 0.0;
      yd(i) = 0.0;
      coef(i) = 0.5 + 0.25 * i;
      scratch(i) = 0.0;
      scratchd(i) = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto scratch_d = scratch.ReadWrite();
   auto scratchd_d = scratchd.ReadWrite();

   __enzyme_fwddiff<void>((void *)enzyme_test_split::qfunction_wrapper<N>,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_const, coef_d,
                          enzyme_dup, scratch_d, scratchd_d,
                          enzyme_runtime_activity);

   y.HostRead();
   yd.HostRead();
   scratch.HostRead();
   scratchd.HostRead();

   for (int q = 0; q < N; q++)
   {
      const double exact_yd = 3.0 * coef[q] * x[q] * x[q];
      const double exact_scratch = x[q] * x[q];
      const double exact_scratchd = 2.0 * x[q];

      REQUIRE(yd[q] == MFEM_Approx(exact_yd));
      REQUIRE(scratch[q] == MFEM_Approx(exact_scratch));
      REQUIRE(scratchd[q] == MFEM_Approx(exact_scratchd));
   }

   enzyme_test_split::print_results(x, coef, y, yd, scratch, scratchd);
}

#endif
