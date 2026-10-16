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

#ifndef MFEM_QUADINTERP_DET_HPP
#define MFEM_QUADINTERP_DET_HPP

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../fem/kernels.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{
namespace quadrature_interpolator
{

inline void Det1D(const int NE,
                  const real_t *b,
                  const real_t *g,
                  const real_t *x,
                  real_t *y,
                  const int d1d,
                  const int q1d,
                  Vector *d_buff = nullptr)
{
   MFEM_CONTRACT_VAR(b);
   MFEM_CONTRACT_VAR(d_buff);
   const auto G = Reshape(g, q1d, d1d);
   const auto X = Reshape(x, d1d, NE);

   auto Y = Reshape(y, q1d, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < q1d; q++)
      {
         real_t u = 0.0;
         for (int d = 0; d < d1d; d++)
         {
            u += G(q, d) * X(d, e);
         }
         Y(q, e) = u;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void Det2D(const int NE,
                  const real_t *b,
                  const real_t *g,
                  const real_t *x,
                  real_t *y,
                  const int d1d = 0,
                  const int q1d = 0,
                  Vector *d_buff = nullptr)
{
   MFEM_CONTRACT_VAR(d_buff);
   static constexpr int SDIM = 2;
   static constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x,  D1D, D1D, SDIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      MFEM_SHARED real_t XY[SDIM][NBZ][MD1*MD1];
      MFEM_SHARED real_t DQ[2*SDIM][NBZ][MD1*MQ1];
      MFEM_SHARED real_t QQ[2*SDIM][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            real_t J[4];
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,QQ,J);
            Y(qx,qy,e) = kernels::Det<2>(J);
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void Det2DSurface(const int NE,
                         const real_t *b,
                         const real_t *g,
                         const real_t *x,
                         real_t *y,
                         const int d1d = 0,
                         const int q1d = 0,
                         Vector *d_buff = nullptr)
{
   MFEM_CONTRACT_VAR(d_buff);

   static constexpr int SDIM = 3;
   static constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x,  D1D, D1D, SDIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int tidz = MFEM_THREAD_ID(z);

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      MFEM_SHARED real_t XYZ[SDIM][NBZ][MD1*MD1];
      MFEM_SHARED real_t DQ[2*SDIM][NBZ][MD1*MQ1];

      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      // Load XYZ components
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            for (int d = 0; d < SDIM; ++d)
            {
               XYZ[d][tidz][dx + dy*D1D] = X(dx,dy,d,e);
            }
         }
      }
      MFEM_SYNC_THREAD;

      ConstDeviceMatrix B_mat(BG[0], D1D, Q1D);
      ConstDeviceMatrix G_mat(BG[1], D1D, Q1D);

      // x contraction
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            for (int d = 0; d < SDIM; ++d)
            {
               real_t u = 0.0;
               real_t v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t xval = XYZ[d][tidz][dx + dy*D1D];
                  u += xval * G_mat(dx,qx);
                  v += xval * B_mat(dx,qx);
               }
               DQ[d][tidz][dy + qx*D1D] = u;
               DQ[3 + d][tidz][dy + qx*D1D] = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // y contraction and determinant computation
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            real_t J_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int d = 0; d < SDIM; ++d)
            {
               for (int dy = 0; dy < D1D; ++dy)
               {
                  J_[d] += DQ[d][tidz][dy + qx*D1D] * B_mat(dy,qy);
                  J_[3 + d] += DQ[3 + d][tidz][dy + qx*D1D] * G_mat(dy,qy);
               }
            }
            DeviceTensor<2> J(J_, 3, 2);
            const real_t E = J(0,0)*J(0,0) + J(1,0)*J(1,0) + J(2,0)*J(2,0);
            const real_t F = J(0,0)*J(0,1) + J(1,0)*J(1,1) + J(2,0)*J(2,1);
            const real_t G = J(0,1)*J(0,1) + J(1,1)*J(1,1) + J(2,1)*J(2,1);
            Y(qx,qy,e) = std::sqrt(E*G - F*F);
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0, bool SMEM = true>
inline void Det3D(const int NE,
                  const real_t *b,
                  const real_t *g,
                  const real_t *x,
                  real_t *y,
                  const int d1d = 0,
                  const int q1d = 0,
                  Vector *d_buff = nullptr) // used only with SMEM = false
{
   constexpr int DIM = 3;
   static constexpr int GRID = SMEM ? 0 : 128;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x, D1D, D1D, D1D, DIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, Q1D, NE);

   real_t *GM = nullptr;
   if (!SMEM)
   {
      const DeviceDofQuadLimits &limits = DeviceDofQuadLimits::Get();
      const int max_q1d = T_Q1D ? T_Q1D : limits.MAX_Q1D;
      const int max_d1d = T_D1D ? T_D1D : limits.MAX_D1D;
      const int max_qd = std::max(max_q1d, max_d1d);
      const int mem_size = max_qd * max_qd * max_qd * 9;
      d_buff->SetSize(2*mem_size*GRID);
      GM = d_buff->Write();
   }

   mfem::forall_3D_grid(NE, Q1D, Q1D, Q1D, GRID, [=] MFEM_HOST_DEVICE (int e)
   {
      static constexpr int MQ1 = T_Q1D ? T_Q1D :
                                 (SMEM ? DofQuadLimits::MAX_DET_1D : DofQuadLimits::MAX_Q1D);
      static constexpr int MD1 = T_D1D ? T_D1D :
                                 (SMEM ? DofQuadLimits::MAX_DET_1D : DofQuadLimits::MAX_D1D);
      static constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;
      static constexpr int MSZ = MDQ * MDQ * MDQ * 9;

      const int bid = MFEM_BLOCK_ID(x);
      MFEM_SHARED real_t BG[2][MQ1*MD1];
      MFEM_SHARED real_t SM0[SMEM?MSZ:1];
      MFEM_SHARED real_t SM1[SMEM?MSZ:1];
      real_t *lm0 = SMEM ? SM0 : GM + MSZ*bid;
      real_t *lm1 = SMEM ? SM1 : GM + MSZ*(GRID+bid);
      real_t (*DDD)[MD1*MD1*MD1] = (real_t (*)[MD1*MD1*MD1]) (lm0);
      real_t (*DDQ)[MD1*MD1*MQ1] = (real_t (*)[MD1*MD1*MQ1]) (lm1);
      real_t (*DQQ)[MD1*MQ1*MQ1] = (real_t (*)[MD1*MQ1*MQ1]) (lm0);
      real_t (*QQQ)[MQ1*MQ1*MQ1] = (real_t (*)[MQ1*MQ1*MQ1]) (lm1);

      kernels::internal::LoadX<MD1>(e,D1D,X,DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t J[9];
               kernels::internal::PullGrad<MQ1>(Q1D, qx,qy,qz, QQQ, J);
               Y(qx,qy,qz,e) = kernels::Det<3>(J);
            }
         }
      }
   });
}

} // namespace quadrature_interpolator
} // namespace internal

/// @cond Suppress_Doxygen_warnings

template<int DIM, int SDIM, int D1D, int Q1D>
QuadratureInterpolator::DetKernelType
QuadratureInterpolator::DetKernels::Kernel()
{
   if (DIM == 1) { return internal::quadrature_interpolator::Det1D; }
   else if (DIM == 2 && SDIM == 2) { return internal::quadrature_interpolator::Det2D<D1D, Q1D>; }
   else if (DIM == 2 && SDIM == 3) { return internal::quadrature_interpolator::Det2DSurface<D1D, Q1D>; }
   else if (DIM == 3) { return internal::quadrature_interpolator::Det3D<D1D, Q1D>; }
   else { MFEM_ABORT(""); }
}

/// @endcond

} // namespace mfem

#endif // MFEM_QUADINTERP_DET_HPP
