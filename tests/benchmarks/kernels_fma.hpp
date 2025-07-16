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

#include "../../config/config.hpp"
#include "../../config/tconfig.hpp" // MFEM_ALWAYS_INLINE
// #include "../../linalg/dtensor.hpp"
// #include "../../linalg/tensor.hpp"
// #include "../../general/forall.hpp" // MFEM_UNROLL
#include "./kernels_vd.hpp"

using namespace mfem::kernels::internal::vd;

namespace mfem::kernels::internal::fma
{

// Grad1X
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE void
ContractX3d(const int D1D, const int Q1D,
            real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
            const real_t (*B)[MQ1],
            const DeviceTensor<5, const real_t> &XE,
            const int vd, const int d, const int e)
{
   const int tz = MFEM_THREAD_ID(z);

   MFEM_FOREACH_THREAD(b, y, D1D)
   {
      MFEM_FOREACH_THREAD(a, x, D1D)
      {
         // MFEM_UNROLL(Q1D)
         for (int k = 0; k < Q1D; ++k)
         {
            double u = 0.0;
            // MFEM_UNROLL(D1D)
            for (int c = 0; c < D1D; ++c) { u += B[c][k] * XE(c,b,a, vd,e); }
            smem[tz][vd][d][k][b][a] = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// Grad1Y
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE void
ContractY3d(const int D1D, const int Q1D,
            real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
            const real_t (*B)[MQ1],
            regs3d_vd_t<VDIM,DIM,MQ1> &r_q,
            const int vd, const int d)
{
   const int tz = MFEM_THREAD_ID(z);

   MFEM_FOREACH_THREAD(k, y, Q1D)
   {
      MFEM_FOREACH_THREAD(a, x, D1D)
      {
         // MFEM_UNROLL(D1D)
         for (int b = 0; b < D1D; ++b) { r_q[k][a][b][vd][d] = smem[tz][vd][d][k][b][a]; }
         // MFEM_UNROLL(Q1D)
         for (int j = 0; j < Q1D; ++j)
         {
            double u = 0.0;
            // MFEM_UNROLL(D1D)
            for (int b = 0; b < D1D; ++b) { u += B[b][j] * r_q[k][a][b][vd][d]; }
            smem[tz][vd][d][k][j][a] = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}


// Grad1Z
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE
void ContractZ3d(const int D1D, const int Q1D,
                 real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
                 const real_t (*B)[MQ1],
                 regs3d_vd_t<VDIM,DIM,MQ1> &r_q,
                 const int vd, const int d)
{
   const int tz = MFEM_THREAD_ID(z);

   MFEM_FOREACH_THREAD(k,y,Q1D)
   {
      MFEM_FOREACH_THREAD(j,x,Q1D)
      {
         // MFEM_UNROLL(D1D)
         for (int a=0; a<D1D; ++a) { r_q[k][j][a][vd][d] = smem[tz][vd][d][k][j][a]; }
         // MFEM_UNROLL(Q1D)
         for (int i=0; i<Q1D; ++i)
         {
            double u = 0.0;
            // MFEM_UNROLL(D1D)
            for (int a=0; a<D1D; ++a) { u += B[a][i] * r_q[k][j][a][vd][d]; }
            smem[tz][vd][d][k][j][i] = u;
         }
      }
   }
   MFEM_SYNC_THREAD;

   // Flush
   // MFEM_FOREACH_THREAD(j,y,Q1D)
   // {
   //    MFEM_FOREACH_THREAD(i,x,Q1D)
   //    {
   //       MFEM_UNROLL(Q1D)
   //       for (int k = 0; k < Q1D; ++k) { r_q[k] = 0.0; }
   //    }
   // }
   // MFEM_SYNC_THREAD;
}

/// 3D vector gradient, with component
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE
void Grad3d(const int D1D, const int Q1D,
            real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
            const real_t (*B)[MQ1],
            const real_t (*G)[MQ1],
            regs3d_vd_t<VDIM,DIM,MQ1> &r_q,
            const DeviceTensor<5, const real_t> &XE,
            const int e)
{
   for (int vd = 0; vd < VDIM; vd++)
   {
      for (int d = 0; d < DIM; d++)
      {
         const real_t (*Bx)[MQ1] = (d == 0) ? G : B;
         const real_t (*By)[MQ1] = (d == 1) ? G : B;
         const real_t (*Bz)[MQ1] = (d == 2) ? G : B;
         ContractX3d(D1D, Q1D, smem, Bx,  XE, vd, d, e);
         ContractY3d(D1D, Q1D, smem, By, r_q, vd, d);
         ContractZ3d(D1D, Q1D, smem, Bz, r_q, vd, d);
      }
   }
}

// GradZT
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE
void ContractZT3d(const int D1D, const int Q1D,
                  real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
                  const real_t (*B)[MQ1],
                  regs3d_vd_t<VDIM,DIM,MQ1> &r_q,
                  const int vd, const int d)
{
   const int tz = MFEM_THREAD_ID(z);

   MFEM_FOREACH_THREAD(j,y,Q1D)
   {
      MFEM_FOREACH_THREAD(i,x,Q1D)
      {
         MFEM_UNROLL(D1D)
         for (int c=0; c<D1D; ++c)
         {
            double u = 0.0;
            MFEM_UNROLL(Q1D)
            for (int k=0; k<Q1D; ++k) { u += B[c][k] * r_q[k][j][i][vd][d]; }
            smem[tz][vd][d][c][j][i] = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// GradYT
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE void
ContractYT3d(const int D1D, const int Q1D,
             real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
             const real_t (*B)[MQ1],
             regs3d_vd_t<VDIM,DIM,MQ1> &r_q,
             const int vd, const int d)
{
   const int tz = MFEM_THREAD_ID(z);

   MFEM_FOREACH_THREAD(c,y,D1D)
   {
      MFEM_FOREACH_THREAD(i,x,Q1D)
      {
         MFEM_UNROLL(Q1D)
         for (int j=0; j<Q1D; ++j) { r_q[c][i][j][vd][d] = smem[tz][vd][d][c][j][i]; }
         MFEM_UNROLL(D1D)
         for (int b=0; b<D1D; ++b)
         {
            double u = 0.0;
            MFEM_UNROLL(Q1D)
            for (int j=0; j<Q1D; ++j) { u += B[b][j] * r_q[c][i][j][vd][d]; }
            smem[tz][vd][d][c][b][i] = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// GradXT
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE void
ContractXT3d(const int D1D, const int Q1D,
             real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
             const real_t (*B)[MQ1],
             regs3d_vd_t<VDIM,DIM,MQ1> &r_q,
             const int vd, const int d)
{
   const int tz = MFEM_THREAD_ID(z);
   MFEM_FOREACH_THREAD(c,y,D1D)
   {
      MFEM_FOREACH_THREAD(b,x,D1D)
      {
         MFEM_UNROLL(Q1D)
         for (int i=0; i<Q1D; ++i) { r_q[c][b][i][vd][d] = smem[tz][vd][d][c][b][i]; }
         MFEM_UNROLL(D1D)
         for (int a=0; a<D1D; ++a)
         {
            double u = 0.0;
            MFEM_UNROLL(Q1D)
            for (int i=0; i<Q1D; ++i) { u += B[a][i] * r_q[c][b][i][vd][d]; }
            smem[tz][vd][d][c][b][a] = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D vector transposed gradient
template <int VDIM, int DIM, int MQ1, int NBZ>
inline MFEM_ALWAYS_INLINE MFEM_HOST_DEVICE
void GradTranspose3d(const int d1d, const int q1d,
                     real_t (&smem)[NBZ][VDIM][DIM][MQ1][MQ1][MQ1],
                     const real_t (*B)[MQ1],
                     const real_t (*G)[MQ1],
                     regs3d_vd_t<VDIM,DIM,MQ1> &r_q,
                     const DeviceTensor<5, real_t> &YE,
                     const int e)
{
   for (int vd = 0; vd < VDIM; vd++)
   {
      for (int d = 0; d < DIM; d++)
      {
         const real_t (*Bx)[MQ1] = (d == 0) ? G : B;
         const real_t (*By)[MQ1] = (d == 1) ? G : B;
         const real_t (*Bz)[MQ1] = (d == 2) ? G : B;
         ContractZT3d(d1d, q1d, smem, Bz, r_q, vd, d);
         ContractYT3d(d1d, q1d, smem, By, r_q, vd, d);
         ContractXT3d(d1d, q1d, smem, Bx, r_q, vd, d);
      }
   }
}

} // namespace mfem::kernels::internal::fma
