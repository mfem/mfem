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
#pragma once

/** @file apply_tensor.hpp
    Unified tensor-product Apply entry: form::ApplyTensor<QFn, DIM, D1D, Q1D>.

    Dispatches on qfn_traits:
      - Eval×Eval → sum-fact Interp path (tensor_eval.hpp)
      - Grad×Grad → sum-fact Grad path (tensor_grad.hpp)

    Integrator-agnostic: pass any QFn defined under fem/integ/.
*/

#include "fields.hpp"
#include "tensor_eval.hpp"
#include "tensor_grad.hpp"

#include <type_traits>

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::form
{

// ---------------------------------------------------------------------------
// Eval×Eval — B basis only (mass-like)
// ---------------------------------------------------------------------------

/** Tensor Eval apply. bt is accepted for API symmetry with registration; unused. */
template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<!qfn_traits<QFn>::trial_is_grad, void>
ApplyTensor(const int NE,
            const Array<real_t> &b,
            const Array<real_t> &bt,
            const Vector &d,
            const Vector &x,
            Vector &y,
            const int d1d = 0,
            const int q1d = 0)
{
   TensorEvalApply<QFn, DIM, D1D, Q1D>(NE, b, bt, d, x, y, d1d, q1d);
}

// ---------------------------------------------------------------------------
// Grad×Grad — B/G bases + runtime symmetric PA layout (diffusion-like)
// ---------------------------------------------------------------------------

/** Tensor Grad apply. `symmetric` selects packed PA layout at runtime. */
template <typename QFn, int DIM, int D1D = 0, int Q1D = 0>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_grad, void>
ApplyTensor(const int NE,
            const bool symmetric,
            const Array<real_t> &b,
            const Array<real_t> &g,
            const Array<real_t> &bt,
            const Array<real_t> &gt,
            const Vector &d,
            const Vector &x,
            Vector &y,
            const int d1d = 0,
            const int q1d = 0)
{
   TensorGradApply<QFn, DIM, D1D, Q1D>(
      NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
}

} // namespace mfem::internal::mma::form

/// \endcond
