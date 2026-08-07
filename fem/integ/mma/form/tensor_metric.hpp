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

/** @file tensor_metric.hpp
    Generic PA metric pack + Grad QFn apply (integrator-agnostic).
*/

#include "fields.hpp"
#include "../../../../linalg/tensor.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::form
{

using mfem::future::tensor;

template <int DIM, bool SYM>
MFEM_HOST_DEVICE inline void PackPaMetric(tensor<real_t, DIM, DIM> &A,
                                          const real_t *O)
{
   if constexpr (DIM == 2)
   {
      const real_t O11 = O[0], O21 = O[1];
      if constexpr (SYM)
      {
         const real_t O22 = O[2];
         A(0, 0) = O11; A(0, 1) = O21;
         A(1, 0) = O21; A(1, 1) = O22;
      }
      else
      {
         const real_t O12 = O[2], O22 = O[3];
         A(0, 0) = O11; A(0, 1) = O12;
         A(1, 0) = O21; A(1, 1) = O22;
      }
   }
   else
   {
      const real_t O11 = O[0], O12 = O[1], O13 = O[2];
      if constexpr (SYM)
      {
         const real_t O22 = O[3], O23 = O[4], O33 = O[5];
         A(0, 0) = O11; A(0, 1) = O12; A(0, 2) = O13;
         A(1, 0) = O12; A(1, 1) = O22; A(1, 2) = O23;
         A(2, 0) = O13; A(2, 1) = O23; A(2, 2) = O33;
      }
      else
      {
         const real_t O21 = O[3], O22 = O[4], O23 = O[5];
         const real_t O31 = O[6], O32 = O[7], O33 = O[8];
         A(0, 0) = O11; A(0, 1) = O12; A(0, 2) = O13;
         A(1, 0) = O21; A(1, 1) = O22; A(1, 2) = O23;
         A(2, 0) = O31; A(2, 1) = O32; A(2, 2) = O33;
      }
   }
}

/** Apply Grad×Grad QFn at one qp: g[] in/out, O packed PA. */
template <int DIM, bool SYM, typename QFn>
MFEM_HOST_DEVICE inline void ApplyGradQFnVec(QFn qfn, real_t *g, const real_t *O)
{
   grad_t<DIM> u, y;
   for (int c = 0; c < DIM; ++c) { u[c] = g[c]; }
   tensor<real_t, DIM, DIM> A{};
   PackPaMetric<DIM, SYM>(A, O);
   qfn(u, y, A);
   for (int c = 0; c < DIM; ++c) { g[c] = y[c]; }
}

/** Device smem: planes g_in/g_out[c * plane_ld + q], D(q,c,e). */
template <int DIM, bool SYM, typename QFn, typename TD>
MFEM_HOST_DEVICE inline
void ApplyGradQFnSmem(QFn qfn, real_t *g_in, real_t *g_out, const int plane_ld,
                      TD D, const int e, const int Q1D,
                      const int tid, const int stride)
{
   constexpr int PA = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   const int nq = (DIM == 2) ? Q1D * Q1D : Q1D * Q1D * Q1D;
   for (int q = tid; q < nq; q += stride)
   {
      real_t g[DIM];
      for (int c = 0; c < DIM; ++c) { g[c] = g_in[c * plane_ld + q]; }
      real_t O[PA];
      for (int c = 0; c < PA; ++c) { O[c] = D(q, c, e); }
      ApplyGradQFnVec<DIM, SYM>(qfn, g, O);
      for (int c = 0; c < DIM; ++c) { g_out[c * plane_ld + q] = g[c]; }
   }
}

} // namespace mfem::internal::mma::form

/// \endcond
