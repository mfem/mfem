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

/** @file simplex_derham.hpp
    De Rham simplex PA engines + ApplySimplex overloads (dense value/curl/div).
    Shared-basis path uses mode Gemm/GemmT; per-element bases use dense forall.
    Included at end of simplex.hpp.
*/

#include "../mode/dispatch.hpp"
#include "../../../../general/forall.hpp"
#include "../../../../linalg/dtensor.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::form
{
namespace detail
{

inline void SimplexVecEvalApply(const int dim, const int NE, const int nd,
                              const int nq, const int sdim,
                              const bool symmetric,
                              const Array<real_t> &B,
                              const Vector &pa_data,
                              const Vector &x, Vector &y)
{
   // B: (nq, nd, sdim[, NE]) physical vector shapes (Piola already applied)
   // pa_data: (nq, ncomp, NE) physical mass metric Q*w*|J|
   MFEM_VERIFY(B.Size() > 0, "Simplex H(curl) mass MMA requires Q-space bases");
   const int ncomp = symmetric ? (sdim * (sdim + 1)) / 2 : sdim * sdim;
   const bool per_elem_B = (B.Size() == nq * nd * sdim * NE);
   const auto Bb4 = Reshape(B.Read(), nq, nd, sdim, per_elem_B ? NE : 1);
   const auto D = Reshape(pa_data.Read(), nq, ncomp, NE);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int be = per_elem_B ? e : 0;
      for (int q = 0; q < nq; ++q)
      {
         real_t u[3] = {};
         for (int c = 0; c < sdim; ++c)
         {
            real_t s = 0.0;
            for (int i = 0; i < nd; ++i) { s += Bb4(q, i, c, be) * X(i, e); }
            u[c] = s;
         }

         real_t v[3] = {};
         if (sdim == 2)
         {
            const real_t O11 = D(q, 0, e);
            const real_t O21 = D(q, 1, e);
            const real_t O12 = symmetric ? O21 : D(q, 2, e);
            const real_t O22 = symmetric ? D(q, 2, e) : D(q, 3, e);
            v[0] = O11 * u[0] + O12 * u[1];
            v[1] = O21 * u[0] + O22 * u[1];
         }
         else
         {
            real_t A[3][3];
            if (symmetric)
            {
               A[0][0] = D(q, 0, e);
               A[1][0] = A[0][1] = D(q, 1, e);
               A[2][0] = A[0][2] = D(q, 2, e);
               A[1][1] = D(q, 3, e);
               A[2][1] = A[1][2] = D(q, 4, e);
               A[2][2] = D(q, 5, e);
            }
            else
            {
               for (int i = 0; i < 3; ++i)
                  for (int j = 0; j < 3; ++j)
                  {
                     A[i][j] = D(q, i * 3 + j, e);
                  }
            }
            for (int i = 0; i < 3; ++i)
            {
               v[i] = A[i][0] * u[0] + A[i][1] * u[1] + A[i][2] * u[2];
            }
         }

         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int c = 0; c < sdim; ++c) { s += Bb4(q, i, c, be) * v[c]; }
            Y(i, e) += s;
         }
      }
   });
}

inline void SimplexCurlCurlApply(const int dim, const int NE, const int nd,
                             const int nq, const int curl_dim,
                             const bool /*symmetric*/,
                             const Array<real_t> &C,
                             const Vector &pa_data,
                             const Vector &x, Vector &y)
{
   MFEM_VERIFY(C.Size() > 0, "Simplex curl-curl MMA requires Q-space bases");
   // C: (nq, nd, curl_dim[, NE]) reference curl shapes
   const bool per_elem_C = (C.Size() == nq * nd * curl_dim * NE);
   const auto Cc = Reshape(C.Read(), nq, nd, curl_dim, per_elem_C ? NE : 1);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   if (dim == 2)
   {
      MFEM_VERIFY(curl_dim == 1, "");
      const auto D = Reshape(pa_data.Read(), nq, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         const int ce = per_elem_C ? e : 0;
         for (int q = 0; q < nq; ++q)
         {
            real_t curl = 0.0;
            for (int i = 0; i < nd; ++i) { curl += Cc(q, i, 0, ce) * X(i, e); }
            curl *= D(q, e);
            for (int i = 0; i < nd; ++i) { Y(i, e) += Cc(q, i, 0, ce) * curl; }
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3 && curl_dim == 3, "");
   const int ncomp = pa_data.Size() / (nq * NE);
   const auto D = Reshape(pa_data.Read(), nq, ncomp, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int ce = per_elem_C ? e : 0;
      for (int q = 0; q < nq; ++q)
      {
         real_t u[3] = {};
         for (int c = 0; c < 3; ++c)
         {
            for (int i = 0; i < nd; ++i) { u[c] += Cc(q, i, c, ce) * X(i, e); }
         }
         real_t v[3] = {};
         if (ncomp == 6)
         {
            const real_t a00 = D(q, 0, e), a10 = D(q, 1, e), a20 = D(q, 2, e);
            const real_t a11 = D(q, 3, e), a21 = D(q, 4, e), a22 = D(q, 5, e);
            v[0] = a00 * u[0] + a10 * u[1] + a20 * u[2];
            v[1] = a10 * u[0] + a11 * u[1] + a21 * u[2];
            v[2] = a20 * u[0] + a21 * u[1] + a22 * u[2];
         }
         else
         {
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
               {
                  v[i] += D(q, i * 3 + j, e) * u[j];
               }
         }
         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int c = 0; c < 3; ++c) { s += Cc(q, i, c, ce) * v[c]; }
            Y(i, e) += s;
         }
      }
   });
}

inline void SimplexDivDivApply(const int NE, const int nd, const int nq,
                           const Array<real_t> &Div,
                           const Vector &pa_data,
                           const Vector &x, Vector &y)
{
   const auto Dd = Reshape(Div.Read(), nq, nd);
   const auto D = Reshape(pa_data.Read(), nq, NE);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < nq; ++q)
      {
         real_t div = 0.0;
         for (int i = 0; i < nd; ++i) { div += Dd(q, i) * X(i, e); }
         div *= D(q, e);
         for (int i = 0; i < nd; ++i) { Y(i, e) += Dd(q, i) * div; }
      }
   });
}


} // namespace detail

// ---------------------------------------------------------------------------
// ApplySimplex — Vector FE mass (multi-plane value basis)
// ---------------------------------------------------------------------------

template <typename QFn, int DIM, int D1D, int QND>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_vec_eval, void>
ApplySimplex(const int NE,
             const Array<real_t> &basis,
             const Vector &d,
             const Vector &x,
             Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::spatial_dim == DIM, "QFn DIM must match ApplySimplex DIM");
   constexpr bool SYM = Tr::symmetric_pa;
   // QND is the specialized nq; nd from x layout / basis
   const int nq = QND;
   const int sdim = DIM;
   const int nd = [&]() {
      const int ncomp_planes = sdim;
      if (basis.Size() == nq * (x.Size()/NE) * ncomp_planes * NE ||
          basis.Size() == nq * (x.Size()/NE) * ncomp_planes)
      {
         return x.Size() / NE;
      }
      return x.Size() / NE;
   }();
   detail::SimplexVecEvalApply(DIM, NE, nd, nq, sdim, SYM, basis, d, x, y);
}

template <typename QFn, int DIM>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_vec_eval, void>
ApplySimplex(const int NE,
             const Array<real_t> &basis,
             const Vector &d,
             const Vector &x,
             Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::spatial_dim == DIM, "");
   constexpr bool SYM = Tr::symmetric_pa;
   const int nd = x.Size() / NE;
   // Infer nq from pa_data: ncomp*nq*NE
   const int ncomp = SYM ? (DIM * (DIM + 1)) / 2 : DIM * DIM;
   MFEM_VERIFY(d.Size() % (ncomp * NE) == 0, "pa_data size");
   const int nq = d.Size() / (ncomp * NE);
   detail::SimplexVecEvalApply(DIM, NE, nd, nq, DIM, SYM, basis, d, x, y);
}

// ---------------------------------------------------------------------------
// ApplySimplex — Curl×Curl
// ---------------------------------------------------------------------------

template <typename QFn, int DIM, int D1D, int QND>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_curl, void>
ApplySimplex(const int NE,
             const Array<real_t> &curl_basis,
             const Vector &d,
             const Vector &x,
             Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::spatial_dim == DIM, "");
   constexpr bool SYM = Tr::symmetric_pa;
   const int nd = x.Size() / NE;
   const int curl_dim = curl_t<DIM>::curl_dim;
   detail::SimplexCurlCurlApply(DIM, NE, nd, QND, curl_dim, SYM,
                                curl_basis, d, x, y);
}

template <typename QFn, int DIM>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_curl, void>
ApplySimplex(const int NE,
             const Array<real_t> &curl_basis,
             const Vector &d,
             const Vector &x,
             Vector &y)
{
   using Tr = qfn_traits<QFn>;
   static_assert(Tr::spatial_dim == DIM, "");
   constexpr bool SYM = Tr::symmetric_pa;
   const int nd = x.Size() / NE;
   const int curl_dim = curl_t<DIM>::curl_dim;
   int nq;
   if (DIM == 2)
   {
      MFEM_VERIFY(d.Size() % NE == 0, "");
      nq = d.Size() / NE;
   }
   else
   {
      const int ncomp = SYM ? 6 : 9;
      MFEM_VERIFY(d.Size() % (ncomp * NE) == 0, "");
      nq = d.Size() / (ncomp * NE);
   }
   detail::SimplexCurlCurlApply(DIM, NE, nd, nq, curl_dim, SYM,
                                curl_basis, d, x, y);
}

// ---------------------------------------------------------------------------
// ApplySimplex — Div×Div
// ---------------------------------------------------------------------------

template <typename QFn, int DIM, int D1D, int QND>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_div, void>
ApplySimplex(const int NE,
             const Array<real_t> &div_basis,
             const Vector &d,
             const Vector &x,
             Vector &y)
{
   const int nd = x.Size() / NE;
   detail::SimplexDivDivApply(NE, nd, QND, div_basis, d, x, y);
}

template <typename QFn, int DIM>
inline std::enable_if_t<qfn_traits<QFn>::trial_is_div, void>
ApplySimplex(const int NE,
             const Array<real_t> &div_basis,
             const Vector &d,
             const Vector &x,
             Vector &y)
{
   const int nd = x.Size() / NE;
   MFEM_VERIFY(d.Size() % NE == 0, "");
   const int nq = d.Size() / NE;
   detail::SimplexDivDivApply(NE, nd, nq, div_basis, d, x, y);
}

} // namespace mfem::internal::mma::form

/// \endcond
