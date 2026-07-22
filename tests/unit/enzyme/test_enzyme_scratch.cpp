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
                          const mfem::Vector &scratchd)
{
   std::printf("%s\n", label);
   std::printf("%3s %10s %10s %10s %10s %10s\n",
               "q", "y", "yd", "yd_ex", "sc", "scd");
   for (int q = 0; q < x.Size(); q++)
   {
      const double exact_yd = 3.0 * coef[q] * x[q] * x[q];
      std::printf("%3d %10.4g %10.4g %10.4g %10.4g %10.4g\n",
                  q, y[q], yd[q], exact_yd, scratch[q], scratchd[q]);
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

TEST_CASE("Enzyme qfunction with SetScratch member",
          "[Enzyme][GPU][Global-SetScratch]")
{
   constexpr int N = 10;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N), scratch_seed(N), scratch(N),
        scratchd(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);
   scratch.UseDevice(true);
   scratchd.UseDevice(true);
   auto x_w = x.HostWrite();
   auto xd_w = xd.HostWrite();
   auto y_w = y.HostWrite();
   auto yd_w = yd.HostWrite();
   auto coef_w = coef.HostWrite();
   auto scratch_w = scratch.HostWrite();
   auto scratchd_w = scratchd.HostWrite();
   for (int i = 0; i < N; i++)
   {
      x_w[i] = i + 1.0;
      xd_w[i] = 1.0;
      y_w[i] = 0.0;
      yd_w[i] = 0.0;
      coef_w[i] = 0.5 + 0.25 * i;
      scratch_w[i] = -1.0; // Pre-seed to show external memory gets reused/overwritten.
      scratchd_w[i] = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto scratch_d = scratch.ReadWrite();
   auto scratchd_d = scratchd.ReadWrite();

   __enzyme_fwddiff<void>((void *)qfunction_wrapper<N>,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_const, coef_d,
                          enzyme_dup, scratch_d, scratchd_d,
                          enzyme_runtime_activity);

   const real_t *x_h = x.HostRead();
   const real_t *coef_h = coef.HostRead();
   const real_t *yd_h = yd.HostRead();
   const real_t *scratch_h = scratch.HostRead();
   const real_t *scratchd_h = scratchd.HostRead();

   for (int q = 0; q < N; q++)
   {
      const double exact_yd = 3.0 * coef_h[q] * x_h[q] * x_h[q];
      const double exact_scratch = x_h[q] * x_h[q];
      const double exact_scratchd = 2.0 * x_h[q];

      REQUIRE(yd_h[q] == MFEM_Approx(exact_yd));
      REQUIRE(scratch_h[q] == MFEM_Approx(exact_scratch));
      REQUIRE(scratchd_h[q] == MFEM_Approx(exact_scratchd));
   }
   if (verbose_tests)
   {
      print_results("Function: y = coef * x^3",
                                            x, coef, y, yd, scratch, scratchd);
   }
}

TEST_CASE("Enzyme qfunction with SetScratch member and qf dup",
          "[Enzyme][GPU][Global-SetScratch-QFDup]")
{
   constexpr int N = 10;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N), coefd(N), scratch(N),
        scratchd(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);
   coefd.UseDevice(true);
   scratch.UseDevice(true);
   scratchd.UseDevice(true);
   auto x_w = x.HostWrite();
   auto xd_w = xd.HostWrite();
   auto y_w = y.HostWrite();
   auto yd_w = yd.HostWrite();
   auto coef_w = coef.HostWrite();
   auto coefd_w = coefd.HostWrite();
   auto scratch_w = scratch.HostWrite();
   auto scratchd_w = scratchd.HostWrite();

   for (int i = 0; i < N; i++)
   {
      x_w[i] = i + 1.0;
      xd_w[i] = 1.0;
      y_w[i] = 0.0;
      yd_w[i] = 0.0;
      coef_w[i] = 0.5 + 0.25 * i;
      coefd_w[i] = 0.0;
      scratch_w[i] = -1.0;
      scratchd_w[i] = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto coefd_d = coefd.ReadWrite();
   auto scratch_d = scratch.ReadWrite();
   auto scratchd_d = scratchd.ReadWrite();

   CubicQFunctionWithMemberScratch qf;
   CubicQFunctionWithMemberScratch qf_d;
   qf.SetCoef(coef_d);
   qf.SetScratch(scratch_d);
   qf_d.SetCoef(coefd_d);
   qf_d.SetScratch(scratchd_d);

   __enzyme_fwddiff<void>((void *)qfunction_apply<N>,
                          enzyme_dup, &qf, &qf_d,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_runtime_activity);

   const real_t *x_h = x.HostRead();
   const real_t *coef_h = coef.HostRead();
   const real_t *yd_h = yd.HostRead();
   const real_t *scratch_h = scratch.HostRead();
   const real_t *scratchd_h = scratchd.HostRead();

   for (int q = 0; q < N; q++)
   {
      const double exact_yd = 3.0 * coef_h[q] * x_h[q] * x_h[q];
      const double exact_scratch = x_h[q] * x_h[q];
      const double exact_scratchd = 2.0 * x_h[q];

      REQUIRE(yd_h[q] == MFEM_Approx(exact_yd));
      REQUIRE(scratch_h[q] == MFEM_Approx(exact_scratch));
      REQUIRE(scratchd_h[q] == MFEM_Approx(exact_scratchd));
   }
   if (verbose_tests)
   {
      print_results("Function: y = coef * x^3 (qf dup), yd = 3 * coef * x^2, scratch = x^2, scratchd = 2 * x",
                                            x, coef, y, yd, scratch, scratchd);
   }
}

TEST_CASE("Enzyme qfunction with SetScratch member and qf const",
          "[Enzyme][GPU][Global-SetScratch-QFConst]")
{
   constexpr int N = 10;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N), scratch_seed(N), scratch(N),
        scratchd(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);
   scratch.UseDevice(true);
   scratchd.UseDevice(true);
   auto x_w = x.HostWrite();
   auto xd_w = xd.HostWrite();
   auto y_w = y.HostWrite();
   auto yd_w = yd.HostWrite();
   auto coef_w = coef.HostWrite();
   auto scratch_w = scratch.HostWrite();
   auto scratchd_w = scratchd.HostWrite();

   for (int i = 0; i < N; i++)
   {
      x_w[i] = i + 1.0;
      xd_w[i] = 1.0;
      y_w[i] = 0.0;
      yd_w[i] = 0.0;
      coef_w[i] = 0.5 + 0.25 * i;
      scratch_seed(i) = 1.0 + 0.5 * i;
      scratch_w[i] = scratch_seed(i);
      scratchd_w[i] = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto scratch_d = scratch.ReadWrite();

   LinearQFunctionWithExternalScratch qf;
   qf.SetCoef(coef_d);
   qf.SetScratch(scratch_d);

   __enzyme_fwddiff<void>((void *)qfunction_apply_qdata<N>,
                          enzyme_const, &qf,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_runtime_activity);

   const real_t *x_h = x.HostRead();
   const real_t *coef_h = coef.HostRead();
   const real_t *scratch_seed_h = scratch_seed.HostRead();
   const real_t *y_h = y.HostRead();
   const real_t *yd_h = yd.HostRead();
   const real_t *scratch_h = scratch.HostRead();
   const real_t *scratchd_h = scratchd.HostRead();

   if (verbose_tests)
   {
      print_qdata_results(
         "Function: scratch <- 2*scratch; y = coef * scratch * x (qf const)",
         x, coef, scratch, y, yd);
   }

   for (int q = 0; q < N; q++)
   {
      const double exact_scratch = 2.0 * scratch_seed_h[q];
      const double exact_y = coef_h[q] * exact_scratch * x_h[q];
      const double exact_yd = coef_h[q] * exact_scratch;

      REQUIRE(y_h[q] == MFEM_Approx(exact_y));
      REQUIRE(yd_h[q] == MFEM_Approx(exact_yd));
      REQUIRE(scratch_h[q] == MFEM_Approx(exact_scratch));
      REQUIRE(scratchd_h[q] == MFEM_Approx(0.0));
   }
}

TEST_CASE("Enzyme qfunction with external qdata-like scratch and qf const",
          "[Enzyme][GPU][Global-SetScratch-ExtQData]")
{
   constexpr int N = 10;
   mfem::Vector x(N), xd(N), y(N), yd(N), coef(N), qdata_seed(N), qdata(N),
        qdatad(N);
   x.UseDevice(true);
   xd.UseDevice(true);
   y.UseDevice(true);
   yd.UseDevice(true);
   coef.UseDevice(true);
   qdata.UseDevice(true);
   qdatad.UseDevice(true);
   auto x_w = x.HostWrite();
   auto xd_w = xd.HostWrite();
   auto y_w = y.HostWrite();
   auto yd_w = yd.HostWrite();
   auto coef_w = coef.HostWrite();
   auto qdata_w = qdata.HostWrite();
   auto qdatad_w = qdatad.HostWrite();

   for (int i = 0; i < N; i++)
   {
      x_w[i] = i + 1.0;
      xd_w[i] = 1.0;
      y_w[i] = 0.0;
      yd_w[i] = 0.0;
      coef_w[i] = 0.5 + 0.25 * i;
      qdata_seed(i) = 1.0 + 0.5 * i;
      qdata_w[i] = qdata_seed(i); // Mimic external qdata provided from outside.
      qdatad_w[i] = 0.0;
   }

   auto x_d = x.Read();
   auto xd_d = xd.ReadWrite();
   auto y_d = y.ReadWrite();
   auto yd_d = yd.ReadWrite();
   auto coef_d = coef.Read();
   auto qdata_d = qdata.ReadWrite();

   LinearQFunctionWithExternalScratch qf;
   qf.SetCoef(coef_d);
   qf.SetScratch(qdata_d);

   __enzyme_fwddiff<void>((void *)qfunction_apply_qdata<N>,
                          enzyme_const, &qf,
                          enzyme_dup, x_d, xd_d,
                          enzyme_dup, y_d, yd_d,
                          enzyme_runtime_activity);

   const real_t *x_h = x.HostRead();
   const real_t *coef_h = coef.HostRead();
   const real_t *qdata_seed_h = qdata_seed.HostRead();
   const real_t *y_h = y.HostRead();
   const real_t *yd_h = yd.HostRead();
   const real_t *qdata_h = qdata.HostRead();
   const real_t *qdatad_h = qdatad.HostRead();

   for (int q = 0; q < N; q++)
   {
      const double exact_qdata = 2.0 * qdata_seed_h[q];
      const double exact_y = coef_h[q] * exact_qdata * x_h[q];
      const double exact_yd = coef_h[q] * exact_qdata;

      REQUIRE(qdata_h[q] == MFEM_Approx(exact_qdata));
      REQUIRE(y_h[q] == MFEM_Approx(exact_y));
      REQUIRE(yd_h[q] == MFEM_Approx(exact_yd));
      REQUIRE(qdatad_h[q] == MFEM_Approx(0.0));
   }
   if (verbose_tests)
   {
      print_qdata_results(
         "Function: scratch <- 2*scratch; y = coef * scratch * x (qf const)",
         x, coef, qdata, y, yd);
   }
}

#endif
