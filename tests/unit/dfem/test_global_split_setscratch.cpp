// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"
#include "../../../linalg/tensor_arrays.hpp"

#ifdef MFEM_USE_ENZYME

namespace enzyme_test_setscratch
{

using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

struct CubicQFunctionWithMemberScratch
{
   const dscalar_t *coef = nullptr;
   dscalar_t *scratch = nullptr;

   MFEM_HOST_DEVICE
   void SetCoef(const dscalar_t *coef_)
   {
      coef = coef_;
   }

   MFEM_HOST_DEVICE
   void SetScratch(dscalar_t *scratch_)
   {
      scratch = scratch_;
   }

   void operator()(tensor_array<const dscalar_t> &x,
                   tensor_array<dscalar_t> &y) const
   {
      auto coef_t = make_tensor_array(coef, x.size());
      auto scratch_t = make_tensor_array(scratch, x.size());

      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         scratch_t(q) = x(q);
      });

      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         scratch_t(q) = scratch_t(q) * x(q);
      });

      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         y(q) = coef_t(q) * scratch_t(q) * x(q);
      });
   }
};

template <int N>
void qfunction_wrapper(const dscalar_t *x, dscalar_t *y,
                      const dscalar_t *coef, dscalar_t *scratch)
{
   auto x_t = make_tensor_array(x, N);
   auto y_t = make_tensor_array(y, N);

   CubicQFunctionWithMemberScratch qf;
   qf.SetCoef(coef);
   qf.SetScratch(scratch);
   qf(x_t, y_t);
}

} // namespace enzyme_test_setscratch

TEST_CASE("Enzyme qfunction with SetScratch member", "[Enzyme][GPU][Global-SetScratch]")
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
      scratch(i) = -1.0; // Pre-seed to show external memory gets reused/overwritten.
      scratchd(i) = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto scratch_d = scratch.ReadWrite();
   auto scratchd_d = scratchd.ReadWrite();

   __enzyme_fwddiff<void>((void *)enzyme_test_setscratch::qfunction_wrapper<N>,
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
   std::printf("Function: y = coef * x^3\n");
   std::printf("Exact (for xd = 1): yd = 3*coef*x^2, scratch = x^2, scratchd = 2*x\n");
   std::printf("%-6s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n",
               "q", "y", "yd", "yd_exact", "scratch", "scratch_exact",
               "scratchd", "scratchd_exact");
   for (int q = 0; q < N; q++)
   {
       const double exact_yd = 3.0 * coef[q] * x[q] * x[q];
       const double exact_scratch = x[q] * x[q];
       const double exact_scratchd = 2.0 * x[q];

       std::printf("%-6d%-18.10g%-18.10g%-18.10g%-18.10g%-18.10g%-18.10g%-18.10g\n",
                   q, y[q], yd[q], exact_yd, scratch[q], exact_scratch,
                   scratchd[q], exact_scratchd);
   }
}

#endif
