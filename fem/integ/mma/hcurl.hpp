// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

/** @file hcurl.hpp
    H(curl) PA MMA — QFns + thin ApplyTensor / ApplySimplex callers.
*/

#include "../../bilininteg.hpp"
#include "form/form.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal::mma::form
{

/** H(curl) mass at Q: y = A * u (Piola-mapped vector, DIM components). */
template <int DIM, bool SYM = true>
struct HcurlMass
{
   static constexpr bool symmetric_pa = SYM;

   MFEM_HOST_DEVICE void operator()(const grad_t<DIM> &u, grad_t<DIM> &y,
                                    const tensor<real_t, DIM, DIM> &A) const
   {
      y = A * u;
   }
};

template <int DIM, bool SYM>
struct qfn_traits<HcurlMass<DIM, SYM>>
   : VecEvalEvalQFnTraits<DIM, SYM, true, true> {};

/** Curl-curl at Q: y = A * curl(u). 2D: scalar A; 3D: 3×3 A. */
template <int DIM, bool SYM = true>
struct CurlCurlQFn;

template <bool SYM>
struct CurlCurlQFn<2, SYM>
{
   static constexpr bool symmetric_pa = SYM;

   MFEM_HOST_DEVICE void operator()(const curl_t<2> &u, curl_t<2> &y,
                                    real_t A) const
   {
      y[0] = A * u[0];
   }
};

template <bool SYM>
struct CurlCurlQFn<3, SYM>
{
   static constexpr bool symmetric_pa = SYM;

   MFEM_HOST_DEVICE void operator()(const curl_t<3> &u, curl_t<3> &y,
                                    const tensor<real_t, 3, 3> &A) const
   {
      y = A * u;
   }
};

template <bool SYM>
struct qfn_traits<CurlCurlQFn<2, SYM>> : CurlCurlQFnTraits<2, SYM> {};
template <bool SYM>
struct qfn_traits<CurlCurlQFn<3, SYM>> : CurlCurlQFnTraits<3, SYM> {};

} // namespace internal::mma::form

namespace internal
{

template <int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void MmaHcurlMassApplyTensors(
   const int NE, const bool symmetric,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::HcurlMass;
   if (symmetric)
   {
      ApplyTensor<HcurlMass<DIM, true>, DIM, T_D1D, T_Q1D>(
         NE, bo, bc, bot, bct, pa_data, x, y, d1d, q1d);
   }
   else
   {
      ApplyTensor<HcurlMass<DIM, false>, DIM, T_D1D, T_Q1D>(
         NE, bo, bc, bot, bct, pa_data, x, y, d1d, q1d);
   }
}

inline void MmaHcurlMassApplyTensors2D(
   const int NE, const bool symmetric, const bool /*scalar_coeff*/,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int /*test_d1d*/, const int q1d)
{
   MmaHcurlMassApplyTensors<2>(NE, symmetric, bo, bc, bot, bct,
                               pa_data, x, y, d1d, q1d);
}

inline void MmaHcurlMassApplyTensors3D(
   const int NE, const bool symmetric, const bool /*scalar_coeff*/,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int /*test_d1d*/, const int q1d)
{
   MmaHcurlMassApplyTensors<3>(NE, symmetric, bo, bc, bot, bct,
                               pa_data, x, y, d1d, q1d);
}

template <int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void MmaCurlCurlApplyTensors(
   const int NE, const bool symmetric,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Array<real_t> &gc, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::CurlCurlQFn;
   if (symmetric)
   {
      ApplyTensor<CurlCurlQFn<DIM, true>, DIM, T_D1D, T_Q1D>(
         NE, bo, bc, bot, bct, gc, gct, pa_data, x, y, d1d, q1d);
   }
   else
   {
      ApplyTensor<CurlCurlQFn<DIM, false>, DIM, T_D1D, T_Q1D>(
         NE, bo, bc, bot, bct, gc, gct, pa_data, x, y, d1d, q1d);
   }
}

inline void MmaCurlCurlApplyTensors2D(
   const int d1d, const int q1d, const bool symmetric, const int NE,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Array<real_t> &gc, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y, const bool /*use_abs*/)
{
   MmaCurlCurlApplyTensors<2>(NE, symmetric, bo, bc, bot, bct, gc, gct,
                              pa_data, x, y, d1d, q1d);
}

inline void MmaCurlCurlApplyTensors3D(
   const int d1d, const int q1d, const bool symmetric, const int NE,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Array<real_t> &gc, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y, const bool /*use_abs*/)
{
   MmaCurlCurlApplyTensors<3>(NE, symmetric, bo, bc, bot, bct, gc, gct,
                              pa_data, x, y, d1d, q1d);
}

inline void MmaHcurlMassApplySimplex(const int dim, const int NE, const int nd,
                                    const int nq, const int d1d,
                                    const int sdim,
                                    const bool symmetric,
                                    const Array<real_t> &B,
                                    const Vector &pa_data,
                                    const Vector &x, Vector &y)
{
   using mma::form::ApplySimplexRegistered;
   using mma::form::HcurlMass;
   MFEM_VERIFY(dim == sdim, "");
   MFEM_VERIFY(nd * NE == x.Size(), "");
   if (dim == 2)
   {
      if (symmetric)
      {
         ApplySimplexRegistered<HcurlMass<2, true>, 2>(
            d1d, nq, NE, B, pa_data, x, y);
      }
      else
      {
         ApplySimplexRegistered<HcurlMass<2, false>, 2>(
            d1d, nq, NE, B, pa_data, x, y);
      }
   }
   else
   {
      if (symmetric)
      {
         ApplySimplexRegistered<HcurlMass<3, true>, 3>(
            d1d, nq, NE, B, pa_data, x, y);
      }
      else
      {
         ApplySimplexRegistered<HcurlMass<3, false>, 3>(
            d1d, nq, NE, B, pa_data, x, y);
      }
   }
}

inline void MmaCurlCurlApplySimplex(const int dim, const int NE, const int nd,
                                   const int nq, const int d1d,
                                   const int curl_dim,
                                   const bool symmetric,
                                   const Array<real_t> &C,
                                   const Vector &pa_data,
                                   const Vector &x, Vector &y)
{
   using mma::form::ApplySimplexRegistered;
   using mma::form::CurlCurlQFn;
   (void)nd; (void)curl_dim;
   if (dim == 2)
   {
      if (symmetric)
      {
         ApplySimplexRegistered<CurlCurlQFn<2, true>, 2>(
            d1d, nq, NE, C, pa_data, x, y);
      }
      else
      {
         ApplySimplexRegistered<CurlCurlQFn<2, false>, 2>(
            d1d, nq, NE, C, pa_data, x, y);
      }
   }
   else
   {
      if (symmetric)
      {
         ApplySimplexRegistered<CurlCurlQFn<3, true>, 3>(
            d1d, nq, NE, C, pa_data, x, y);
      }
      else
      {
         ApplySimplexRegistered<CurlCurlQFn<3, false>, 3>(
            d1d, nq, NE, C, pa_data, x, y);
      }
   }
}

} // namespace internal

/// \endcond

} // namespace mfem
