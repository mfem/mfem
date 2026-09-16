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
    H(div) PA MMA — QFns + tensor/simplex apply for VectorFEMass / DivDiv.
*/

#include "../../bilininteg.hpp"
#include "form/form.hpp"

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
struct qfn_traits<HdivMass<DIM, SYM>> : VecEvalEvalQFnTraits<DIM, SYM> {};

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

/** Owned tensor H(div) mass — Bo/Bc sum-fact PA apply (not a PAHdiv* wrap). */
void MmaHdivMassApplyTensors2D(
   const int NE, const bool symmetric, const bool scalar_coeff,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int test_d1d, const int q1d);

void MmaHdivMassApplyTensors3D(
   const int NE, const bool symmetric, const bool scalar_coeff,
   const Array<real_t> &bo, const Array<real_t> &bc,
   const Array<real_t> &bot, const Array<real_t> &bct,
   const Vector &pa_data, const Vector &x, Vector &y,
   const int d1d, const int test_d1d, const int q1d);

/** Owned tensor div-div — Bo/Gc sum-fact PA apply (not a PADivDiv* wrap). */
void MmaDivDivApplyTensors2D(
   const int d1d, const int q1d, const int NE,
   const Array<real_t> &bo, const Array<real_t> &gc,
   const Array<real_t> &bot, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y);

void MmaDivDivApplyTensors3D(
   const int d1d, const int q1d, const int NE,
   const Array<real_t> &bo, const Array<real_t> &gc,
   const Array<real_t> &bot, const Array<real_t> &gct,
   const Vector &pa_data, const Vector &x, Vector &y);

void MmaHdivMassApplySimplex(const int dim, const int NE, const int nd,
                             const int nq, const int sdim,
                             const bool symmetric,
                             const Array<real_t> &B,
                             const Vector &pa_data,
                             const Vector &x, Vector &y);

void MmaDivDivApplySimplex(const int NE, const int nd, const int nq,
                           const Array<real_t> &Div,
                           const Vector &pa_data,
                           const Vector &x, Vector &y);

} // namespace internal

/// \endcond

} // namespace mfem
