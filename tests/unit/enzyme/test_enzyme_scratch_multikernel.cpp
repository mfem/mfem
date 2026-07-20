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

namespace enzyme_test_multikernel
{

using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

// -----------------------------------------------------------------------
// Qfunction split into three kernels, each meant to be launched via its
// own __enzyme_fwddiff call. scratch is a member pointer, mirroring
// CubicQFunctionWithMemberScratch from the single-call test suite, but
// here each mfem::forall loop is its own top-level kernel rather than
// three loops fused inside one operator().
// -----------------------------------------------------------------------
struct CubicQFunctionThreeKernels
{
   const dscalar_t *coef = nullptr;
   dscalar_t *scratch = nullptr;

   MFEM_HOST_DEVICE
   void SetCoef(const dscalar_t *coef_) { coef = coef_; }

   MFEM_HOST_DEVICE
   void SetScratch(dscalar_t *scratch_) { scratch = scratch_; }

   void Kernel1(tensor_array<const dscalar_t> &x) const
   {
      auto scratch_t = make_tensor_array(scratch, x.size());
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         scratch_t(q) = x(q);
      });
   }

   void Kernel2(tensor_array<const dscalar_t> &x) const
   {
      auto scratch_t = make_tensor_array(scratch, x.size());
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         scratch_t(q) = scratch_t(q) * x(q);
      });
   }

   void Kernel3(tensor_array<const dscalar_t> &x,
                tensor_array<dscalar_t> &y) const
   {
      auto coef_t = make_tensor_array(coef, x.size());
      auto scratch_t = make_tensor_array(scratch, x.size());
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         y(q) = coef_t(q) * scratch_t(q) * x(q);
      });
   }
};

// -----------------------------------------------------------------------
// Same three-kernel decomposition, but scratch is passed explicitly as a
// kernel argument rather than routed through the qfunction object.
// -----------------------------------------------------------------------
struct CubicQFunctionThreeKernelsExplicitScratch
{
   const dscalar_t *coef = nullptr;

   MFEM_HOST_DEVICE
   void SetCoef(const dscalar_t *coef_) { coef = coef_; }

   void Kernel1(tensor_array<const dscalar_t> &x,
                tensor_array<dscalar_t> &scratch_t) const
   {
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         scratch_t(q) = x(q);
      });
   }

   void Kernel2(tensor_array<const dscalar_t> &x,
                tensor_array<dscalar_t> &scratch_t) const
   {
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         scratch_t(q) = scratch_t(q) * x(q);
      });
   }

   void Kernel3(tensor_array<const dscalar_t> &x,
                tensor_array<const dscalar_t> &scratch_t,
                tensor_array<dscalar_t> &y) const
   {
      auto coef_t = make_tensor_array(coef, x.size());
      mfem::forall<UseEnzyme>(x.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         y(q) = coef_t(q) * scratch_t(q) * x(q);
      });
   }
};

// -----------------------------------------------------------------------
// Wrapper functions: one per kernel, each becomes its own __enzyme_fwddiff
// call site. Member-scratch variants take qf by const ref (activity of qf
// itself is fixed by the enzyme_const/enzyme_dup annotation at the call).
// -----------------------------------------------------------------------
template <int N>
void kernel1_member(const CubicQFunctionThreeKernels &qf, const dscalar_t *x)
{
   auto x_t = make_tensor_array(x, N);
   qf.Kernel1(x_t);
}

template <int N>
void kernel2_member(const CubicQFunctionThreeKernels &qf, const dscalar_t *x)
{
   auto x_t = make_tensor_array(x, N);
   qf.Kernel2(x_t);
}

template <int N>
void kernel3_member(const CubicQFunctionThreeKernels &qf, const dscalar_t *x,
                    dscalar_t *y)
{
   auto x_t = make_tensor_array(x, N);
   auto y_t = make_tensor_array(y, N);
   qf.Kernel3(x_t, y_t);
}

template <int N>
void kernel1_explicit(const CubicQFunctionThreeKernelsExplicitScratch &qf,
                      const dscalar_t *x, dscalar_t *scratch)
{
   auto x_t = make_tensor_array(x, N);
   auto s_t = make_tensor_array(scratch, N);
   qf.Kernel1(x_t, s_t);
}

template <int N>
void kernel2_explicit(const CubicQFunctionThreeKernelsExplicitScratch &qf,
                      const dscalar_t *x, dscalar_t *scratch)
{
   auto x_t = make_tensor_array(x, N);
   auto s_t = make_tensor_array(scratch, N);
   qf.Kernel2(x_t, s_t);
}

template <int N>
void kernel3_explicit(const CubicQFunctionThreeKernelsExplicitScratch &qf,
                      const dscalar_t *x, const dscalar_t *scratch,
                      dscalar_t *y)
{
   auto x_t = make_tensor_array(x, N);
   auto s_t = make_tensor_array(scratch, N);
   auto y_t = make_tensor_array(y, N);
   qf.Kernel3(x_t, s_t, y_t);
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

} // namespace enzyme_test_multikernel

// =========================================================================
// TEST 1 (BROKEN): three separate __enzyme_fwddiff calls, qf marked
// enzyme_const in all three. scratch's tangent computed in Kernel1 cannot
// escape that call (const qf gives Enzyme no caller-visible shadow to
// write into), so Kernel2/Kernel3 silently see zero contribution from it.
// yd comes out wrong (missing the factor-of-3 term).
// =========================================================================
TEST_CASE("Enzyme multi-kernel qfunction with qf const across separate calls",
          "[Enzyme][MultiKernel][QFConst]")
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

   enzyme_test_multikernel::CubicQFunctionThreeKernels qf;
   qf.SetCoef(coef_d);
   qf.SetScratch(scratch_d);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel1_member<N>,
      enzyme_const, &qf,
      enzyme_dup, x_d, xd_d,
      enzyme_runtime_activity);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel2_member<N>,
      enzyme_const, &qf,
      enzyme_dup, x_d, xd_d,
      enzyme_runtime_activity);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel3_member<N>,
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

      // yd is wrong-by-construction: the x^2 term's derivative never
      // survives the boundary between Kernel1/Kernel2's separate calls.
      REQUIRE(yd[q] == MFEM_Approx(observed_yd));
      REQUIRE(yd[q] != MFEM_Approx(exact_yd));
      // Primal scratch is still correct -- only the tangent is lost.
      REQUIRE(scratch[q] == MFEM_Approx(exact_scratch));
      REQUIRE(scratchd[q] == MFEM_Approx(0.0));
   }
   enzyme_test_multikernel::print_results(
      "Multi-kernel, qf const (BROKEN): yd missing scratch's tangent contribution",
      x, coef, y, yd, scratch, scratchd, true);
}

// =========================================================================
// TEST 2 (WORKING): scratch/scratchd passed as explicit enzyme_dup kernel
// arguments across all three separate calls. qf stays enzyme_const since
// it holds no differentiable state directly.
// =========================================================================
TEST_CASE("Enzyme multi-kernel qfunction with explicit scratch buffers dup'd",
          "[Enzyme][MultiKernel][ExplicitScratch]")
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
   auto scratchd_d = scratchd.ReadWrite();

   enzyme_test_multikernel::CubicQFunctionThreeKernelsExplicitScratch qf;
   qf.SetCoef(coef_d);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel1_explicit<N>,
      enzyme_const, &qf,
      enzyme_dup, x_d, xd_d,
      enzyme_dup, scratch_d, scratchd_d,
      enzyme_runtime_activity);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel2_explicit<N>,
      enzyme_const, &qf,
      enzyme_dup, x_d, xd_d,
      enzyme_dup, scratch_d, scratchd_d,
      enzyme_runtime_activity);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel3_explicit<N>,
      enzyme_const, &qf,
      enzyme_dup, x_d, xd_d,
      enzyme_dup, scratch_d, scratchd_d,
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
   enzyme_test_multikernel::print_results(
      "Multi-kernel, explicit scratch dup'd (WORKING)",
      x, coef, y, yd, scratch, scratchd);
}

// =========================================================================
// TEST 3 (WORKING): qf itself marked enzyme_dup across all three separate
// calls, with a real scratchd buffer wired up via SetScratch on qf_d, so
// the shadow is reachable through qf_d.scratch between calls.
// =========================================================================
TEST_CASE("Enzyme multi-kernel qfunction with qf dup and member scratch_d",
          "[Enzyme][MultiKernel][QFDup]")
{
   constexpr int N = 100;
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

   enzyme_test_multikernel::CubicQFunctionThreeKernels qf;
   enzyme_test_multikernel::CubicQFunctionThreeKernels qf_d;
   qf.SetCoef(coef_d);
   qf.SetScratch(scratch_d);
   qf_d.SetCoef(coefd_d);
   qf_d.SetScratch(
      scratchd_d);   // real shadow memory -- required for this to work

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel1_member<N>,
      enzyme_dup, &qf, &qf_d,
      enzyme_dup, x_d, xd_d,
      enzyme_runtime_activity);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel2_member<N>,
      enzyme_dup, &qf, &qf_d,
      enzyme_dup, x_d, xd_d,
      enzyme_runtime_activity);

   __enzyme_fwddiff<void>(
      (void *)enzyme_test_multikernel::kernel3_member<N>,
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
   enzyme_test_multikernel::print_results(
      "Multi-kernel, qf dup with member scratch_d (WORKING)",
      x, coef, y, yd, scratch, scratchd);
}

#endif