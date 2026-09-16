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
    H(curl) PA MMA — QFns + tensor/simplex apply for VectorFEMass / CurlCurl.
*/

#include "../../bilininteg.hpp"
#include "form/form.hpp"
#include "../bilininteg_hcurl_kernels.hpp"

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
struct qfn_traits<HcurlMass<DIM, SYM>> : VecEvalEvalQFnTraits<DIM, SYM> {};

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

/** Tensor H(curl) mass — matches PAHcurlMassApply* (oracle path). */
inline void MmaHcurlMassApplyTensors2D(
   const int NE, const bool symmetric, const bool scalar_coeff,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int test_d1d, const int q1d)
{
   PAHcurlMassApply2D(NE, symmetric, scalar_coeff, bo, bc, bot, bct,
                      pa_data, x, y, d1d, test_d1d, q1d);
}

inline void MmaHcurlMassApplyTensors3D(
   const int NE, const bool symmetric, const bool scalar_coeff,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int test_d1d, const int q1d)
{
   PAHcurlMassApply3D(NE, symmetric, scalar_coeff, bo, bc, bot, bct,
                      pa_data, x, y, d1d, test_d1d, q1d);
}

/** Tensor curl-curl — matches PACurlCurlApply*. */
inline void MmaCurlCurlApplyTensors2D(
   const int d1d, const int q1d, const bool symmetric, const int NE,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Array<real_t> &gc, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y, const bool use_abs)
{
   PACurlCurlApply2D(d1d, q1d, symmetric, NE, bo, bc, bot, bct, gc, gct,
                     pa_data, x, y, use_abs);
}

inline void MmaCurlCurlApplyTensors3D(
   const int d1d, const int q1d, const bool symmetric, const int NE,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Array<real_t> &gc, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y, const bool use_abs)
{
   PACurlCurlApply3D(d1d, q1d, symmetric, NE, bo, bc, bot, bct, gc, gct,
                     pa_data, x, y, use_abs);
}

/** Dense simplex H(curl) mass / curl-curl apply (host). */
void MmaHcurlMassApplySimplex(const int dim, const int NE, const int nd,
                              const int nq, const int sdim,
                              const bool symmetric,
                              const Array<real_t> &B, // (nq, nd, sdim) ref vshape
                              const Vector &pa_data, // (nq, ncomp, NE)
                              const Vector &x, Vector &y);

void MmaCurlCurlApplySimplex(const int dim, const int NE, const int nd,
                             const int nq, const int curl_dim,
                             const bool symmetric,
                             const Array<real_t> &C, // (nq, nd, curl_dim) ref
                             const Vector &pa_data,
                             const Vector &x, Vector &y);

} // namespace internal

/// \endcond

} // namespace mfem
