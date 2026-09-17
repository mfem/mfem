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

/** @file hdiv.hpp
    H(div) PA MMA — QFns + thin ApplyTensor / ApplySimplex callers.
*/

#include "../../bilininteg.hpp"
#include "form/form.hpp"
#include "hcurl.hpp" // MmaHcurlMassApplySimplex shared dense vec path via ApplySimplex

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal::mma::form
{

/** H(div) mass at Q: y = A * u (Piola-mapped vector). */
template <int DIM, bool SYM = true>
struct HdivMass
{
   static constexpr bool symmetric_pa = SYM;

   MFEM_HOST_DEVICE void operator()(const grad_t<DIM> &u, grad_t<DIM> &y,
                                    const tensor<real_t, DIM, DIM> &A) const
   {
      y = A * u;
   }
};

template <int DIM, bool SYM>
struct qfn_traits<HdivMass<DIM, SYM>>
   : VecEvalEvalQFnTraits<DIM, SYM, false, false> {};

/** Div-div at Q: y = d * div(u). */
struct DivDivQFn
{
   MFEM_HOST_DEVICE void operator()(const div_t &u, div_t &y, real_t d) const
   {
      y = d * u;
   }
};

template <>
struct qfn_traits<DivDivQFn> : DivDivQFnTraits {};

} // namespace internal::mma::form

namespace internal
{

template <int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void MmaHdivMassApplyTensors(
   const int NE, const bool symmetric,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::HdivMass;
   if (symmetric)
   {
      ApplyTensor<HdivMass<DIM, true>, DIM, T_D1D, T_Q1D>(
         NE, bo, bc, bot, bct, pa_data, x, y, d1d, q1d);
   }
   else
   {
      ApplyTensor<HdivMass<DIM, false>, DIM, T_D1D, T_Q1D>(
         NE, bo, bc, bot, bct, pa_data, x, y, d1d, q1d);
   }
}

inline void MmaHdivMassApplyTensors2D(
   const int NE, const bool symmetric, const bool /*scalar_coeff*/,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int /*test_d1d*/, const int q1d)
{
   MmaHdivMassApplyTensors<2>(NE, symmetric, bo, bc, bot, bct,
                              pa_data, x, y, d1d, q1d);
}

inline void MmaHdivMassApplyTensors3D(
   const int NE, const bool symmetric, const bool /*scalar_coeff*/,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int /*test_d1d*/, const int q1d)
{
   MmaHdivMassApplyTensors<3>(NE, symmetric, bo, bc, bot, bct,
                              pa_data, x, y, d1d, q1d);
}

template <int DIM, int T_D1D = 0, int T_Q1D = 0>
inline void MmaDivDivApplyTensors(
   const int NE,
   const Array<real_t> &bo, const Array<real_t> &gc,
   const Array<real_t> &bot, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::DivDivQFn;
   ApplyTensor<DivDivQFn, DIM, T_D1D, T_Q1D>(
      NE, bo, gc, bot, gct, pa_data, x, y, d1d, q1d);
}

inline void MmaDivDivApplyTensors2D(
   const int d1d, const int q1d, const int NE,
   const Array<real_t> &bo, const Array<real_t> &gc,
   const Array<real_t> &bot, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y)
{
   MmaDivDivApplyTensors<2>(NE, bo, gc, bot, gct, pa_data, x, y, d1d, q1d);
}

inline void MmaDivDivApplyTensors3D(
   const int d1d, const int q1d, const int NE,
   const Array<real_t> &bo, const Array<real_t> &gc,
   const Array<real_t> &bot, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y)
{
   MmaDivDivApplyTensors<3>(NE, bo, gc, bot, gct, pa_data, x, y, d1d, q1d);
}

inline void MmaHdivMassApplySimplex(const int dim, const int NE, const int nd,
                                   const int nq, const int d1d,
                                   const int sdim,
                                   const bool symmetric,
                                   const Array<real_t> &B,
                                   const Vector &pa_data,
                                   const Vector &x, Vector &y)
{
   using mma::form::ApplySimplexRegistered;
   using mma::form::HdivMass;
   MFEM_VERIFY(dim == sdim, "");
   (void)nd;
   if (dim == 2)
   {
      if (symmetric)
      {
         ApplySimplexRegistered<HdivMass<2, true>, 2>(
            d1d, nq, NE, B, pa_data, x, y);
      }
      else
      {
         ApplySimplexRegistered<HdivMass<2, false>, 2>(
            d1d, nq, NE, B, pa_data, x, y);
      }
   }
   else
   {
      if (symmetric)
      {
         ApplySimplexRegistered<HdivMass<3, true>, 3>(
            d1d, nq, NE, B, pa_data, x, y);
      }
      else
      {
         ApplySimplexRegistered<HdivMass<3, false>, 3>(
            d1d, nq, NE, B, pa_data, x, y);
      }
   }
}

inline void MmaDivDivApplySimplex(const int dim, const int NE, const int nd,
                                 const int nq, const int d1d,
                                 const Array<real_t> &Div,
                                 const Vector &pa_data,
                                 const Vector &x, Vector &y)
{
   using mma::form::ApplySimplexRegistered;
   using mma::form::DivDivQFn;
   (void)nd;
   if (dim == 2)
   {
      ApplySimplexRegistered<DivDivQFn, 2>(d1d, nq, NE, Div, pa_data, x, y);
   }
   else
   {
      ApplySimplexRegistered<DivDivQFn, 3>(d1d, nq, NE, Div, pa_data, x, y);
   }
}

} // namespace internal

/// \endcond

} // namespace mfem
