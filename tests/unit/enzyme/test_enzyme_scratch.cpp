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

struct LinearQFunctionWithExternalScratch
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
         scratch_t(q) *= 2.0;
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

template <int N>
void qfunction_apply(const CubicQFunctionWithMemberScratch &qf,
                           const dscalar_t *x, dscalar_t *y)
{
   auto x_t = make_tensor_array(x, N);
   auto y_t = make_tensor_array(y, N);

   qf(x_t, y_t);
}

template <int N>
void qfunction_apply_qdata(const LinearQFunctionWithExternalScratch &qf,
                           const dscalar_t *x, dscalar_t *y)
{
   auto x_t = make_tensor_array(x, N);
   auto y_t = make_tensor_array(y, N);

   qf(x_t, y_t);
}

inline void print_results(const char *label,
                          const mfem::Vector &x,
                          const mfem::Vector &coef,
                          const mfem::Vector &y,
                          const mfem::Vector &yd,
                          const mfem::Vector &scratch,
                          const mfem::Vector &scratchd,
                          const bool qf_const = false)
{
   std::printf("%s\n", label);
   if (qf_const)
   {
      std::printf("%3s %10s %10s %10s %10s %10s %10s\n",
                  "q", "y", "yd", "yd_ex", "yd_obs", "sc", "scd");
   }
   else
   {
      std::printf("%3s %10s %10s %10s %10s %10s\n",
                  "q", "y", "yd", "yd_ex", "sc", "scd");
   }

   for (int q = 0; q < x.Size(); q++)
   {
      const double exact_yd = 3.0 * coef[q] * x[q] * x[q];
      const double observed_yd = coef[q] * x[q] * x[q];

      if (qf_const)
      {
         std::printf("%3d %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g\n",
                     q, y[q], yd[q], exact_yd, observed_yd,
                     scratch[q], scratchd[q]);
      }
      else
      {
         std::printf("%3d %10.4g %10.4g %10.4g %10.4g %10.4g\n",
                     q, y[q], yd[q], exact_yd, scratch[q], scratchd[q]);
      }
   }
}

inline void print_qdata_results(const char *label,
                                const mfem::Vector &x,
                                const mfem::Vector &coef,
                                const mfem::Vector &qdata,
                                const mfem::Vector &y,
                                const mfem::Vector &yd)
{
   std::printf("%s\n", label);
   std::printf("%3s %10s %10s %10s %10s\n",
               "q", "y", "yd", "yd_ex", "qd_out");

   for (int q = 0; q < x.Size(); q++)
   {
      const double exact_yd = coef[q] * qdata[q];
      std::printf("%3d %10.4g %10.4g %10.4g %10.4g\n",
                  q, y[q], yd[q], exact_yd, qdata[q]);
   }
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
   enzyme_test_setscratch::print_results("Function: y = coef * x^3",
                                         x, coef, y, yd, scratch, scratchd);
}

TEST_CASE("Enzyme qfunction with SetScratch member and qf dup", "[Enzyme][GPU][Global-SetScratch-QFDup]")
{
   constexpr int N = 100;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N), coefd(N), scratch(N), scratchd(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);
   coefd.UseDevice(true);
   scratch.UseDevice(true);
   scratchd.UseDevice(true);

   for (int i = 0; i < N; i++)
   {
      x(i) = i + 1.0;
      xd(i) = 1.0;
      y(i) = 0.0;
      yd(i) = 0.0;
      coef(i) = 0.5 + 0.25 * i;
      coefd(i) = 0.0;
      scratch(i) = -1.0;
      scratchd(i) = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto coefd_d = coefd.ReadWrite();
   auto scratch_d = scratch.ReadWrite();
   auto scratchd_d = scratchd.ReadWrite();

   enzyme_test_setscratch::CubicQFunctionWithMemberScratch qf;
   enzyme_test_setscratch::CubicQFunctionWithMemberScratch qf_d;
   qf.SetCoef(coef_d);
   qf.SetScratch(scratch_d);
   qf_d.SetCoef(coefd_d);
   qf_d.SetScratch(scratchd_d);

   __enzyme_fwddiff<void>((void *)enzyme_test_setscratch::qfunction_apply<N>,
                          enzyme_dup, &qf, &qf_d,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
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
   enzyme_test_setscratch::print_results("Function: y = coef * x^3 (qf const), yd = 3 * coef * x^2, scratch = x^2, scratchd = 2 * x",
                                         x, coef, y, yd, scratch, scratchd);
}

TEST_CASE("Enzyme qfunction with SetScratch member and qf const",
          "[Enzyme][GPU][Global-SetScratch-QFConst]")
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
      scratch(i) = -1.0;
      scratchd(i) = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto scratch_d = scratch.ReadWrite();

   enzyme_test_setscratch::CubicQFunctionWithMemberScratch qf;
   qf.SetCoef(coef_d);
   qf.SetScratch(scratch_d);

   __enzyme_fwddiff<void>((void *)enzyme_test_setscratch::qfunction_apply<N>,
                          enzyme_const, &qf,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_runtime_activity);

   y.HostRead();
   yd.HostRead();
   scratch.HostRead();
   scratchd.HostRead();

   for (int q = 0; q < N; q++)
   {
      const double exact_yd = 3.0 * coef[q] * x[q] * x[q];
      const double observed_yd = coef[q] * x[q] * x[q];
      const double exact_scratch = x[q] * x[q];

      REQUIRE(yd[q] == MFEM_Approx(observed_yd));
      REQUIRE(yd[q] != MFEM_Approx(exact_yd));
      REQUIRE(scratch[q] == MFEM_Approx(exact_scratch));
      REQUIRE(scratchd[q] == MFEM_Approx(0.0));
   }
   enzyme_test_setscratch::print_results("Function: y = coef * x^3 (qf const), yd = 3 * coef * x^2, scratch = x^2, scratchd = 2 * x",
                                         x, coef, y, yd, scratch, scratchd,
                                         true);
}

TEST_CASE("Enzyme qfunction with external qdata-like scratch and qf const",
          "[Enzyme][GPU][Global-SetScratch-ExtQData]")
{
   constexpr int N = 100;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N), qdata_seed(N), qdata(N), qdatad(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);
   qdata.UseDevice(true);
   qdatad.UseDevice(true);

   for (int i = 0; i < N; i++)
   {
      x(i) = i + 1.0;
      xd(i) = 1.0;
      y(i) = 0.0;
      yd(i) = 0.0;
      coef(i) = 0.5 + 0.25 * i;
      qdata_seed(i) = 1.0 + 0.5 * i;
      qdata(i) = qdata_seed(i); // Mimic external qdata provided from outside.
      qdatad(i) = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto qdata_d = qdata.ReadWrite();

   enzyme_test_setscratch::LinearQFunctionWithExternalScratch qf;
   qf.SetCoef(coef_d);
   qf.SetScratch(qdata_d);

   __enzyme_fwddiff<void>((void *)enzyme_test_setscratch::qfunction_apply_qdata<N>,
                          enzyme_const, &qf,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_runtime_activity);

   y.HostRead();
   yd.HostRead();
   qdata.HostRead();
   qdatad.HostRead();

   for (int q = 0; q < N; q++)
   {
      const double exact_qdata = 2.0 * qdata_seed[q];
      const double exact_y = coef[q] * exact_qdata * x[q];
      const double exact_yd = coef[q] * exact_qdata;

      REQUIRE(qdata[q] == MFEM_Approx(exact_qdata));
      REQUIRE(y[q] == MFEM_Approx(exact_y));
      REQUIRE(yd[q] == MFEM_Approx(exact_yd));
      REQUIRE(qdatad[q] == MFEM_Approx(0.0));
   }
   enzyme_test_setscratch::print_qdata_results(
      "Function: scratch <- 2*scratch; y = coef * scratch * x (qf const)",
      x, coef, qdata, y, yd);
}

#endif
