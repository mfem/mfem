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

#ifndef MFEM_BILININTEG_DIFFUSION_KERNELS_HPP
#define MFEM_BILININTEG_DIFFUSION_KERNELS_HPP

#include "../kernel_dispatch.hpp"
#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT
namespace internal
{

void PADiffusionSetup(const int dim,
                      const int sdim,
                      const int D1D,
                      const int Q1D,
                      const int coeffDim,
                      const int NE,
                      const Array<real_t> &W,
                      const Vector &J,
                      const Vector &C,
                      Vector &D);

// PA Diffusion Assemble 2D f
template<int T_SDIM>
void PADiffusionSetup2D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const Array<real_t> &w,
                        const Vector &j,
                        const Vector &c,
                        Vector &d);

// PA Diffusion Assemble 3D kernel
void PADiffusionSetup3D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const Array<real_t> &w,
                        const Vector &j,
                        const Vector &c,
                        Vector &d);

#ifdef MFEM_USE_OCCA
// OCCA 2D Assemble kernel
void OccaPADiffusionSetup2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<real_t> &W,
                            const Vector &J,
                            const Vector &C,
                            Vector &op);

// OCCA 3D Assemble kernel
void OccaPADiffusionSetup3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<real_t> &W,
                            const Vector &J,
                            const Vector &C,
                            Vector &op);
#endif // MFEM_USE_OCCA

void PADiffusionAssembleDiagonal(const int dim,
                                 const int D1D,
                                 const int Q1D,
                                 const int NE,
                                 const bool symm,
                                 const Array<real_t> &B,
                                 const Array<real_t> &G,
                                 const Vector &D,
                                 Vector &Y);

// PA Diffusion Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionDiagonal2D(const int NE,
                                  const bool symmetric,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   // note the different shape for D, if this is a symmetric matrix we only
   // store necessary entries
   auto D = Reshape(d.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      // gradphi \cdot Q \gradphi has four terms
      real_t QD0[MQ1][MD1];
      real_t QD1[MQ1][MD1];
      real_t QD2[MQ1][MD1];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            QD0[qx][dy] = 0.0;
            QD1[qx][dy] = 0.0;
            QD2[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int q = qx + qy * Q1D;
               const real_t D00 = D(q,0,e);
               const real_t D10 = D(q,1,e);
               const real_t D01 = symmetric ? D10 : D(q,2,e);
               const real_t D11 = symmetric ? D(q,2,e) : D(q,3,e);
               QD0[qx][dy] += B(qy, dy) * B(qy, dy) * D00;
               QD1[qx][dy] += B(qy, dy) * G(qy, dy) * (D01 + D10);
               QD2[qx][dy] += G(qy, dy) * G(qy, dy) * D11;
            }
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Y(dx,dy,e) += G(qx, dx) * G(qx, dx) * QD0[qx][dy];
               Y(dx,dy,e) += G(qx, dx) * B(qx, dx) * QD1[qx][dy];
               Y(dx,dy,e) += B(qx, dx) * B(qx, dx) * QD2[qx][dy];
            }
         }
      }
   });
}

namespace diffusion
{
constexpr int ipow(int x, int p) { return p == 0 ? 1 : x*ipow(x, p-1); }
constexpr int D11(int x) { return (11 - x)/2; }
constexpr int D10(int x) { return (10 - x)/2; }
constexpr int NBZApply(int D1D)
{
   return ipow(2, D11(D1D) >= 0 ? D11(D1D) : 0);
}
constexpr int NBZDiagonal(int D1D)
{
   return ipow(2, D10(D1D) >= 0 ? D10(D1D) : 0);
}
}

// Shared memory PA Diffusion Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionDiagonal2D(const int NE,
                                      const bool symmetric,
                                      const Array<real_t> &b_,
                                      const Array<real_t> &g_,
                                      const Vector &d_,
                                      Vector &y_,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   static constexpr int T_NBZ = diffusion::NBZDiagonal(T_D1D);
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);
   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_SHARED real_t BG[2][MQ1*MD1];
      real_t (*B)[MD1] = (real_t (*)[MD1]) (BG+0);
      real_t (*G)[MD1] = (real_t (*)[MD1]) (BG+1);
      MFEM_SHARED real_t QD[3][NBZ][MD1][MQ1];
      real_t (*QD0)[MD1] = (real_t (*)[MD1])(QD[0] + tidz);
      real_t (*QD1)[MD1] = (real_t (*)[MD1])(QD[1] + tidz);
      real_t (*QD2)[MD1] = (real_t (*)[MD1])(QD[2] + tidz);
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            QD0[qx][dy] = 0.0;
            QD1[qx][dy] = 0.0;
            QD2[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int q = qx + qy * Q1D;
               const real_t D00 = D(q,0,e);
               const real_t D10 = D(q,1,e);
               const real_t D01 = symmetric ? D10 : D(q,2,e);
               const real_t D11 = symmetric ? D(q,2,e) : D(q,3,e);
               const real_t By = B[qy][dy];
               const real_t Gy = G[qy][dy];
               const real_t BBy = By * By;
               const real_t BGy = By * Gy;
               const real_t GGy = Gy * Gy;
               QD0[qx][dy] += BBy * D00;
               QD1[qx][dy] += BGy * (D01 + D10);
               QD2[qx][dy] += GGy * D11;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t Bx = B[qx][dx];
               const real_t Gx = G[qx][dx];
               const real_t BBx = Bx * Bx;
               const real_t BGx = Bx * Gx;
               const real_t GGx = Gx * Gx;
               Y(dx,dy,e) += GGx * QD0[qx][dy];
               Y(dx,dy,e) += BGx * QD1[qx][dy];
               Y(dx,dy,e) += BBx * QD2[qx][dy];
            }
         }
      }
   });
}

// PA Diffusion Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionDiagonal3D(const int NE,
                                  const bool symmetric,
                                  const Array<real_t> &b,
                                  const Array<real_t> &g,
                                  const Vector &d,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Q = Reshape(d.Read(), Q1D*Q1D*Q1D, symmetric ? 6 : 9, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t QQD[MQ1][MQ1][MD1];
      real_t QDD[MQ1][MD1][MD1];
      for (int i = 0; i < DIM; ++i)
      {
         for (int j = 0; j < DIM; ++j)
         {
            // first tensor contraction, along z direction
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     QQD[qx][qy][dz] = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const int q = qx + (qy + qz * Q1D) * Q1D;
                        const int ksym = j >= i ?
                                         3 - (3-i)*(2-i)/2 + j:
                                         3 - (3-j)*(2-j)/2 + i;
                        const int k = symmetric ? ksym : (i*DIM) + j;
                        const real_t O = Q(q,k,e);
                        const real_t Bz = B(qz,dz);
                        const real_t Gz = G(qz,dz);
                        const real_t L = i==2 ? Gz : Bz;
                        const real_t R = j==2 ? Gz : Bz;
                        QQD[qx][qy][dz] += L * O * R;
                     }
                  }
               }
            }
            // second tensor contraction, along y direction
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dz = 0; dz < D1D; ++dz)
               {
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     QDD[qx][dy][dz] = 0.0;
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const real_t By = B(qy,dy);
                        const real_t Gy = G(qy,dy);
                        const real_t L = i==1 ? Gy : By;
                        const real_t R = j==1 ? Gy : By;
                        QDD[qx][dy][dz] += L * QQD[qx][qy][dz] * R;
                     }
                  }
               }
            }
            // third tensor contraction, along x direction
            for (int dz = 0; dz < D1D; ++dz)
            {
               for (int dy = 0; dy < D1D; ++dy)
               {
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t Bx = B(qx,dx);
                        const real_t Gx = G(qx,dx);
                        const real_t L = i==0 ? Gx : Bx;
                        const real_t R = j==0 ? Gx : Bx;
                        Y(dx, dy, dz, e) += L * QDD[qx][dy][dz] * R;
                     }
                  }
               }
            }
         }
      }
   });
}

// Shared memory PA Diffusion Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionDiagonal3D(const int NE,
                                      const bool symmetric,
                                      const Array<real_t> &b_,
                                      const Array<real_t> &g_,
                                      const Vector &d_,
                                      Vector &y_,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D*Q1D, symmetric ? 6 : 9, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_SHARED real_t BG[2][MQ1*MD1];
      real_t (*B)[MD1] = (real_t (*)[MD1]) (BG+0);
      real_t (*G)[MD1] = (real_t (*)[MD1]) (BG+1);
      MFEM_SHARED real_t QQD[MQ1][MQ1][MD1];
      MFEM_SHARED real_t QDD[MQ1][MD1][MD1];
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int i = 0; i < DIM; ++i)
      {
         for (int j = 0; j < DIM; ++j)
         {
            // first tensor contraction, along z direction
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(dz,z,D1D)
                  {
                     QQD[qx][qy][dz] = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const int q = qx + (qy + qz * Q1D) * Q1D;
                        const int ksym = j >= i ?
                                         3 - (3-i)*(2-i)/2 + j:
                                         3 - (3-j)*(2-j)/2 + i;
                        const int k = symmetric ? ksym : (i*DIM) + j;
                        const real_t O = D(q,k,e);
                        const real_t Bz = B[qz][dz];
                        const real_t Gz = G[qz][dz];
                        const real_t L = i==2 ? Gz : Bz;
                        const real_t R = j==2 ? Gz : Bz;
                        QQD[qx][qy][dz] += L * O * R;
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // second tensor contraction, along y direction
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  MFEM_FOREACH_THREAD(dy,y,D1D)
                  {
                     QDD[qx][dy][dz] = 0.0;
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const real_t By = B[qy][dy];
                        const real_t Gy = G[qy][dy];
                        const real_t L = i==1 ? Gy : By;
                        const real_t R = j==1 ? Gy : By;
                        QDD[qx][dy][dz] += L * QQD[qx][qy][dz] * R;
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // third tensor contraction, along x direction
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               MFEM_FOREACH_THREAD(dy,y,D1D)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1D)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t Bx = B[qx][dx];
                        const real_t Gx = G[qx][dx];
                        const real_t L = i==0 ? Gx : Bx;
                        const real_t R = j==0 ? Gx : Bx;
                        Y(dx, dy, dz, e) += L * QDD[qx][dy][dz] * R;
                     }
                  }
               }
            }
         }
      }
   });
}

void PADiffusionApply(const int dim,
                      const int D1D,
                      const int Q1D,
                      const int NE,
                      const bool symm,
                      const Array<real_t> &B,
                      const Array<real_t> &G,
                      const Array<real_t> &Bt,
                      const Array<real_t> &Gt,
                      const Vector &D,
                      const Vector &X,
                      Vector &Y);

#ifdef MFEM_USE_OCCA
// OCCA PA Diffusion Apply 2D kernel
void OccaPADiffusionApply2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<real_t> &B,
                            const Array<real_t> &G,
                            const Array<real_t> &Bt,
                            const Array<real_t> &Gt,
                            const Vector &D,
                            const Vector &X,
                            Vector &Y);

// OCCA PA Diffusion Apply 3D kernel
void OccaPADiffusionApply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<real_t> &B,
                            const Array<real_t> &G,
                            const Array<real_t> &Bt,
                            const Array<real_t> &Gt,
                            const Vector &D,
                            const Vector &X,
                            Vector &Y);
#endif // MFEM_USE_OCCA

// PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApply2D(const int NE,
                               const bool symmetric,
                               const Array<real_t> &b_,
                               const Array<real_t> &g_,
                               const Array<real_t> &bt_,
                               const Array<real_t> &gt_,
                               const Vector &d_,
                               const Vector &x_,
                               Vector &y_,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b_.Read(), Q1D, D1D);
   auto G = Reshape(g_.Read(), Q1D, D1D);
   auto Bt = Reshape(bt_.Read(), D1D, Q1D);
   auto Gt = Reshape(gt_.Read(), D1D, Q1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto X = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t grad[max_Q1D][max_Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t gradX[max_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s = X(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * B(qx,dx);
               gradX[qx][1] += s * G(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t wy  = B(qy,dy);
            const real_t wDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }
      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;

            const real_t O11 = D(q,0,e);
            const real_t O21 = D(q,1,e);
            const real_t O12 = symmetric ? O21 : D(q,2,e);
            const real_t O22 = symmetric ? D(q,2,e) : D(q,3,e);

            const real_t gradX = grad[qy][qx][0];
            const real_t gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O21 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t gradX[max_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t gX = grad[qy][qx][0];
            const real_t gY = grad[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t wx  = Bt(dx,qx);
               const real_t wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t wy  = Bt(dy,qy);
            const real_t wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// Shared memory PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionApply2D(const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &b_,
                                   const Array<real_t> &g_,
                                   const Array<real_t> &bt_,
                                   const Array<real_t> &gt_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   static constexpr int T_NBZ = diffusion::NBZApply(T_D1D);
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);
   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_SHARED real_t sBG[2][MQ1*MD1];
      real_t (*B)[MD1] = (real_t (*)[MD1]) (sBG+0);
      real_t (*G)[MD1] = (real_t (*)[MD1]) (sBG+1);
      real_t (*Bt)[MQ1] = (real_t (*)[MQ1]) (sBG+0);
      real_t (*Gt)[MQ1] = (real_t (*)[MQ1]) (sBG+1);
      MFEM_SHARED real_t Xz[NBZ][MD1][MD1];
      MFEM_SHARED real_t GD[2][NBZ][MD1][MQ1];
      MFEM_SHARED real_t GQ[2][NBZ][MD1][MQ1];
      real_t (*X)[MD1] = (real_t (*)[MD1])(Xz + tidz);
      real_t (*DQ0)[MD1] = (real_t (*)[MD1])(GD[0] + tidz);
      real_t (*DQ1)[MD1] = (real_t (*)[MD1])(GD[1] + tidz);
      real_t (*QQ0)[MD1] = (real_t (*)[MD1])(GQ[0] + tidz);
      real_t (*QQ1)[MD1] = (real_t (*)[MD1])(GQ[1] + tidz);
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X[dy][dx] = x(dx,dy,e);
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][dy] = b(q,dy);
               G[q][dy] = g(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            real_t u = 0.0;
            real_t v = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t coords = X[dy][dx];
               u += B[qx][dx] * coords;
               v += G[qx][dx] * coords;
            }
            DQ0[dy][qx] = u;
            DQ1[dy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            real_t u = 0.0;
            real_t v = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               u += DQ1[dy][qx] * B[qy][dy];
               v += DQ0[dy][qx] * G[qy][dy];
            }
            QQ0[qy][qx] = u;
            QQ1[qy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int q = (qx + ((qy) * Q1D));
            const real_t O11 = D(q,0,e);
            const real_t O21 = D(q,1,e);
            const real_t O12 = symmetric ? O21 : D(q,2,e);
            const real_t O22 = symmetric ? D(q,2,e) : D(q,3,e);
            const real_t gX = QQ0[qy][qx];
            const real_t gY = QQ1[qy][qx];
            QQ0[qy][qx] = (O11 * gX) + (O12 * gY);
            QQ1[qy][qx] = (O21 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[dy][q] = b(q,dy);
               Gt[dy][q] = g(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            real_t u = 0.0;
            real_t v = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += Gt[dx][qx] * QQ0[qy][qx];
               v += Bt[dx][qx] * QQ1[qy][qx];
            }
            DQ0[qy][dx] = u;
            DQ1[qy][dx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            real_t u = 0.0;
            real_t v = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += DQ0[qy][dx] * Bt[dy][qy];
               v += DQ1[qy][dx] * Gt[dy][qy];
            }
            Y(dx,dy,e) += (u + v);
         }
      }
   });
}

/* This function computes the action of the diffusion integrator for the Bernstein basis on triangles.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C2 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F2 and roughly corresponding to Algorithm 3 of [1]).

   [1] Ainsworth, M., Andriamaro, G., & Davydov, O. (2011). Bernstein–Bézier finite elements
       of arbitrary order and optimal assembly procedures. SIAM Journal on Scientific Computing,
       33(6), 3087-3109.
   */
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApplyTriangle(const int NE,
                                     const bool symmetric,
                                     const Array<int> &lex_map_,
                                     const Array<int> &forward_map2d_,
                                     const Array<int> &inverse_map2d_,
                                     const Array<int> &inverse_map3d_,
                                     const Array<real_t> &ga1_,
                                     const Array<real_t> &ga2_,
                                     const Array<real_t> &ga3_, // unused in 2D...
                                     const Vector &d_,
                                     const Vector &x_,
                                     Vector &y_,
                                     const int d1d = 0,
                                     const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int BASIS_DIM = D1D * (D1D+1) / 2;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   const auto lex_map = lex_map_.Read();
   const auto Ga1 = ConstDeviceMatrix(ga1_.Read(), Q1D, D1D-1);
   const auto Ga2 = ConstDeviceCube(ga2_.Read(), Q1D, D1D-1, D1D-1);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   const auto X = Reshape(x_.Read(), BASIS_DIM, NE);
   auto Y = Reshape(y_.ReadWrite(), BASIS_DIM, NE);

   int p2 = (D1D-1) * (D1D-1);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      for (int idx = 0; idx < BASIS_DIM; idx++)
      {
         Y(idx, e) = 0.0;
      }

      // should be using stack memory here like quad version, but this seems faster...
      real_t *cin = new real_t[2 * (D1D-1) * (D1D-1)] {0};
      real_t *C1 = new real_t[2 * (D1D-1) * Q1D] {0};
      real_t *C2 = new real_t[2 * Q1D * Q1D] {0};

      // cin contains the vector coefficient
      //    cin_{\beta} = \sum_{k=1}^{3} \nabla\lambda_{k} * X_{\beta + e_{k}},
      // where \lambda_{k} are the standard barycentric coordinates and e_{k}
      // is the unit vector with nonzero value in entry k. C2 will contain the
      // value of the Bernstein polynomial
      //    \sum_{\beta} cin_{\beta} * B_{\beta}^{p-1}(\Phi(t1,t2)),
      // where \Phi is the Duffy transform and (t1,t2) is a Stroud quadrature node
      // in the unit square.
      for (int dy = 0; dy < D1D-1; ++dy)
      {
         for (int dx = 0; dx < D1D-dy-1; ++dx)
         {
            // k=0, component 0
            int idx = lex_map[dx + D1D*(dy+1)];
            const int dydx = 2*(dy + (D1D-1)*dx);
            cin[dydx] += X(idx, e);

            // // k=1, component 0
            // idx = lex_map[(dx+1) + D1D*dy];
            // cin[0 + 2*(dy + (D1D-1)*dx)] += X(idx, e) * 0.0;

            // k=2, component 0
            idx = lex_map[dx + D1D*dy];
            cin[dydx] -= X(idx, e);

            // // k=0, component 1
            // idx = lex_map[dx + D1D*(dy+1)];
            // cin[1 + 2*(dy + (D1D-1)*dx)] += X(idx, e) * 0.0;

            // k=1, component 1
            idx = lex_map[(dx+1) + D1D*dy];
            cin[1 + dydx] += X(idx, e);

            // k=2, component 1
            idx = lex_map[dx + D1D*dy];
            cin[1 + dydx] -= X(idx, e);
         }
      }

      // C1 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the first spatial dimension
      for (int iL = 0; iL < Q1D; iL++)
      {
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            const int a1iL2 = 2*(iL + Q1D*a1);
            for (int aL = 0; aL < D1D-a1-1; aL++)
            {
               const int a1aL2 = 2*(a1 + (D1D-1)*aL);
               const real_t Gai = Ga2(iL, a1, aL);
               C1[a1iL2] += cin[a1aL2] * Gai;
               C1[1 + a1iL2] += cin[1 + a1aL2] * Gai;
            }
         }
      }

      // C2 contains the Bernstein polynomial on a triangle with coefficients cin evaluated at
      // all of the Stroud quadrature nodes. E.g. if (t1,t2) is a Stroud node, then
      //    C2[i,j] = \sum_{\alpha} cin_{\alpha} * B_{\alpha}^{p-1}(\Phi(t1,t2)),
      // where \Phi is the Duffy transform.
      for (int iL = 0; iL < Q1D; iL++)
      {
         for (int aL = 0; aL < D1D-1; aL++)
         {
            const real_t Gai = Ga1(iL, aL);
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const int i2iL2 = 2*(i2 + Q1D*iL);
               const int i2aL2 = 2*(i2 + Q1D*aL);
               C2[i2iL2] += C1[i2aL2] * Gai;
               C2[1 + i2iL2] += C1[1 + i2aL2] * Gai;
            }
         }
      }

      // now evaluate the Bernstein moments
      real_t *fin = new real_t[2 * Q1D * Q1D];
      real_t *F1 = new real_t[2 * (D1D-1) * Q1D] {0};
      real_t *F2 = new real_t[2 * (D1D-1) * (D1D-1)] {0};

      // fin contains (B_{K})^{-1} * D(x) * (B_{K})^{-T} * C2(x). the result stored in F2
      // will be all Bernstein moments
      //    \int_{K} B_{\alpha}^{p-1}(x) * fin(x) dx.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t O11 = D(qy, qx, 0, e);
            const real_t O21 = D(qy, qx, 1, e);
            const real_t O12 = symmetric ? O21 : D(qy, qx, 2, e);
            const real_t O22 = symmetric ? D(qy, qx, 2, e) : D(qy, qx, 3, e);

            const int qxqy2 = 2*(qx + Q1D*qy);
            fin[qxqy2] = O11 * C2[qxqy2] + O12 * C2[1 + qxqy2];
            fin[1 + qxqy2] = O21 * C2[qxqy2] + O22 * C2[1 + qxqy2];
         }
      }


      // F1 computes the Bernstein moment over the first ragged tensor dimension.
      for (int iL = 0; iL < Q1D; iL++)
      {
         for (int aL = 0; aL < D1D-1; aL++)
         {
            const real_t Gai = Ga1(iL, aL);
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const int i2iL2 = 2*(i2 + Q1D*iL);
               const int i2aL2 = 2*(i2 + Q1D*aL);
               F1[i2aL2] += fin[i2iL2] * Gai;
               F1[1 + i2aL2] += fin[1 + i2iL2] * Gai;
            }
         }
      }

      // F2 computes the Bernstein moment over the second/last ragged tensor dimension.
      for (int iL = 0; iL < Q1D; iL++)
      {
         for (int a1 = 0; a1 < D1D-1; a1++)
         {
            const int a1iL2 = 2*(iL + Q1D*a1);
            for (int aL = 0; aL < D1D-a1-1; aL++)
            {
               const int a1aL2 = 2*(aL + (D1D-1)*a1);
               const real_t Gai = Ga2(iL, a1, aL);
               F2[a1aL2] += F1[a1iL2] * Gai;
               F2[1 + a1aL2] += F1[1 + a1iL2] * Gai;
            }
         }
      }

      // compute contributions to local RHS. we have
      //       Y_{\alpha + e_{k}} = p^{2} * \nabla\lambda_{k} * F2_{\alpha}
      // where \lambda_{k} is the kth barycentric coordinate and
      // e_{k} is the unit vector with nonzero entry k, for k=1,2,3.
      for (int a1 = 0; a1 < D1D-1; ++a1)
      {
         for (int a2 = 0; a2 < D1D-a1-1; ++a2)
         {
            // k=0
            int idx = lex_map[a2 + D1D*(a1+1)];
            const int a2a1 = 2*(a2 + (D1D-1)*a1);
            Y(idx,e) += p2 * F2[a2a1];

            // k=1
            idx = lex_map[(a2+1) + D1D*a1];
            Y(idx,e) += p2 * F2[1 + a2a1];

            // k=2
            idx = lex_map[a2 + D1D*a1];
            Y(idx,e) -= p2 * (F2[a2a1] + F2[1 + a2a1]);
         }
      }

      delete[] cin;
      delete[] C1;
      delete[] C2;
      delete[] fin;
      delete[] F1;
      delete[] F2;
   });
}

// PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApply3D(const int NE,
                               const bool symmetric,
                               const Array<real_t> &b,
                               const Array<real_t> &g,
                               const Array<real_t> &bt,
                               const Array<real_t> &gt,
                               const Vector &d_,
                               const Vector &x_,
                               Vector &y_,
                               int d1d = 0, int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D*Q1D, symmetric ? 6 : 9, NE);
   auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t grad[max_Q1D][max_Q1D][max_Q1D][3];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t gradXY[max_Q1D][max_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t gradX[max_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = X(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy  = B(qy,dy);
               const real_t wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx  = gradX[qx][0];
                  const real_t wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz  = B(qz,dz);
            const real_t wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }
      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + (qy + qz * Q1D) * Q1D;
               const real_t O11 = D(q,0,e);
               const real_t O12 = D(q,1,e);
               const real_t O13 = D(q,2,e);
               const real_t O21 = symmetric ? O12 : D(q,3,e);
               const real_t O22 = symmetric ? D(q,3,e) : D(q,4,e);
               const real_t O23 = symmetric ? D(q,4,e) : D(q,5,e);
               const real_t O31 = symmetric ? O13 : D(q,6,e);
               const real_t O32 = symmetric ? O23 : D(q,7,e);
               const real_t O33 = symmetric ? D(q,5,e) : D(q,8,e);
               const real_t gradX = grad[qz][qy][qx][0];
               const real_t gradY = grad[qz][qy][qx][1];
               const real_t gradZ = grad[qz][qy][qx][2];
               grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[qz][qy][qx][1] = (O21*gradX)+(O22*gradY)+(O23*gradZ);
               grad[qz][qy][qx][2] = (O31*gradX)+(O32*gradY)+(O33*gradZ);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t gradXY[max_D1D][max_D1D][3];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t gradX[max_D1D][3];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t gX = grad[qz][qy][qx][0];
               const real_t gY = grad[qz][qy][qx][1];
               const real_t gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t wx  = Bt(dx,qx);
                  const real_t wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t wy  = Bt(dy,qy);
               const real_t wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const real_t wz  = Bt(dz,qz);
            const real_t wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
}

// Shared memory PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPADiffusionApply3D(const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &b_,
                                   const Array<real_t> &g_,
                                   const Array<real_t> &,
                                   const Array<real_t> &,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      MFEM_SHARED real_t sBG[2][MQ1*MD1];
      real_t (*B)[MD1] = (real_t (*)[MD1]) (sBG+0);
      real_t (*G)[MD1] = (real_t (*)[MD1]) (sBG+1);
      real_t (*Bt)[MQ1] = (real_t (*)[MQ1]) (sBG+0);
      real_t (*Gt)[MQ1] = (real_t (*)[MQ1]) (sBG+1);
      MFEM_SHARED real_t sm0[3][MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[3][MDQ*MDQ*MDQ];
      real_t (*X)[MD1][MD1]    = (real_t (*)[MD1][MD1]) (sm0+2);
      real_t (*DDQ0)[MD1][MQ1] = (real_t (*)[MD1][MQ1]) (sm0+0);
      real_t (*DDQ1)[MD1][MQ1] = (real_t (*)[MD1][MQ1]) (sm0+1);
      real_t (*DQQ0)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) (sm1+0);
      real_t (*DQQ1)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) (sm1+1);
      real_t (*DQQ2)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) (sm1+2);
      real_t (*QQQ0)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) (sm0+0);
      real_t (*QQQ1)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) (sm0+1);
      real_t (*QQQ2)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) (sm0+2);
      real_t (*QQD0)[MQ1][MD1] = (real_t (*)[MQ1][MD1]) (sm1+0);
      real_t (*QQD1)[MQ1][MD1] = (real_t (*)[MQ1][MD1]) (sm1+1);
      real_t (*QQD2)[MQ1][MD1] = (real_t (*)[MQ1][MD1]) (sm1+2);
      real_t (*QDD0)[MD1][MD1] = (real_t (*)[MD1][MD1]) (sm0+0);
      real_t (*QDD1)[MD1][MD1] = (real_t (*)[MD1][MD1]) (sm0+1);
      real_t (*QDD2)[MD1][MD1] = (real_t (*)[MD1][MD1]) (sm0+2);
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               B[qx][dy] = b(qx,dy);
               G[qx][dy] = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0, v = 0.0;
               MFEM_UNROLL(MD1)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t coords = X[dz][dy][dx];
                  u += coords * B[qx][dx];
                  v += coords * G[qx][dx];
               }
               DDQ0[dz][dy][qx] = u;
               DDQ1[dz][dy][qx] = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MD1)
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DDQ1[dz][dy][qx] * B[qy][dy];
                  v += DDQ0[dz][dy][qx] * G[qy][dy];
                  w += DDQ0[dz][dy][qx] * B[qy][dy];
               }
               DQQ0[dz][qy][qx] = u;
               DQQ1[dz][qy][qx] = v;
               DQQ2[dz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u += DQQ0[dz][qy][qx] * B[qz][dz];
                  v += DQQ1[dz][qy][qx] * B[qz][dz];
                  w += DQQ2[dz][qy][qx] * G[qz][dz];
               }
               const real_t O11 = d(qx,qy,qz,0,e);
               const real_t O12 = d(qx,qy,qz,1,e);
               const real_t O13 = d(qx,qy,qz,2,e);
               const real_t O21 = symmetric ? O12 : d(qx,qy,qz,3,e);
               const real_t O22 = symmetric ? d(qx,qy,qz,3,e) : d(qx,qy,qz,4,e);
               const real_t O23 = symmetric ? d(qx,qy,qz,4,e) : d(qx,qy,qz,5,e);
               const real_t O31 = symmetric ? O13 : d(qx,qy,qz,6,e);
               const real_t O32 = symmetric ? O23 : d(qx,qy,qz,7,e);
               const real_t O33 = symmetric ? d(qx,qy,qz,5,e) : d(qx,qy,qz,8,e);
               const real_t gX = u;
               const real_t gY = v;
               const real_t gZ = w;
               QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
               QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ);
               QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               Bt[dy][qx] = b(qx,dy);
               Gt[dy][qx] = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               real_t u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MQ1)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ0[qz][qy][qx] * Gt[dx][qx];
                  v += QQQ1[qz][qy][qx] * Bt[dx][qx];
                  w += QQQ2[qz][qy][qx] * Bt[dx][qx];
               }
               QQD0[qz][qy][dx] = u;
               QQD1[qz][qy][dx] = v;
               QQD2[qz][qy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               real_t u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += QQD0[qz][qy][dx] * Bt[dy][qy];
                  v += QQD1[qz][qy][dx] * Gt[dy][qy];
                  w += QQD2[qz][qy][dx] * Bt[dy][qy];
               }
               QDD0[qz][dy][dx] = u;
               QDD1[qz][dy][dx] = v;
               QDD2[qz][dy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               real_t u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += QDD0[qz][dy][dx] * Bt[dz][qz];
                  v += QDD1[qz][dy][dx] * Bt[dz][qz];
                  w += QDD2[qz][dy][dx] * Gt[dz][qz];
               }
               y(dx,dy,dz,e) += (u + v + w);
            }
         }
      }
   });
}

/* This function computes the action of the diffusion integrator for the Bernstein basis on tetrahedrons.
   The key components are an O(p^{d+1}) routine for evaluating the Bernstein polynomial
   \sum_{\alpha} c_{\alpha} B_{\alpha}^{p}(x) simultaneously at all quadrature points x
   (stored in the array C3 and roughly corresponding to Algorithm 1 of [1])and an O(p^{d+1})
   routine for evaluating the Bernstein moments \int_{K} f(x) * B_{\alpha}^{p}(x) dx for all
   \alpha (stored in the array F3 and roughly corresponding to Algorithm 3 of [1]).

   [1] Ainsworth, M., Andriamaro, G., & Davydov, O. (2011). Bernstein–Bézier finite elements
       of arbitrary order and optimal assembly procedures. SIAM Journal on Scientific Computing,
       33(6), 3087-3109.
   */
  template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApplyTetrahedron(const int NE,
                                        const bool symmetric,
                                        const Array<int> &lex_map_,
                                        const Array<int> &forward_map2d_,
                                        const Array<int> &inverse_map2d_,
                                        const Array<int> &inverse_map3d_,
                                        const Array<real_t> &ga1_,
                                        const Array<real_t> &ga2_,
                                        const Array<real_t> &ga3_,
                                        const Vector &d_,
                                        const Vector &x_,
                                        Vector &y_,
                                        const int d1d = 0,
                                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int BASIS_DIM2D = D1D * (D1D+1) / 2;
   const int BASIS_DIM3D = D1D * (D1D+1) * (D1D+2) / 6;
   const int BASIS_DIM2D_DIFF = (D1D-1) * D1D / 2;
   const int BASIS_DIM3D_DIFF = (D1D-1) * D1D * (D1D+1) / 6;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   const auto lex_map = lex_map_.Read();
   const auto forward_map2d = forward_map2d_.Read();
   const auto inverse_map2d = inverse_map2d_.Read();
   const auto inverse_map3d = inverse_map3d_.Read();
   const auto Ga1 = ConstDeviceMatrix(ga1_.Read(), Q1D, D1D-1);
   const auto Ga2 = ConstDeviceMatrix(ga2_.Read(), Q1D, BASIS_DIM2D_DIFF);
   const auto Ga3 = ConstDeviceMatrix(ga3_.Read(), Q1D, BASIS_DIM3D_DIFF);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   const auto X = Reshape(x_.Read(), BASIS_DIM3D, NE);
   auto Y = Reshape(y_.ReadWrite(), BASIS_DIM3D, NE);

   const int p2 = (D1D-1) * (D1D-1);

   // real_t *C1 = new real_t[(int) 3 * BASIS_DIM2D_DIFF* Q1D] {0};
   // real_t *C2 = new real_t[3 * (D1D-1) * Q1D * Q1D] {0};
   // real_t *C3 = new real_t[3 * Q1D * Q1D * Q1D] {0};

   // real_t *F1 = new real_t[3 * (D1D-1) * Q1D * Q1D] {0};
   // real_t *F2 = new real_t[(int) 3 * BASIS_DIM2D_DIFF * Q1D] {0};

   // std::cout << "Printing from Apply Tetrahedron" << std::endl;
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      constexpr int basis_dim2d = (int) 3 * (max_D1D-1) * (max_D1D) / 2;
      real_t C1[(int) 3 * basis_dim2d * max_Q1D];
      real_t C2[3 * (max_D1D-1) * max_Q1D * max_Q1D];
      real_t C3[3 * max_Q1D * max_Q1D * max_Q1D];
      real_t F1[3 * (max_D1D-1) * max_Q1D * max_Q1D];
      real_t F2[(int) 3 * basis_dim2d * max_Q1D];

      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               const int q = 3*(i1 + Q1D*(i2 + Q1D*i3));
               C3[q] = 0.0;
               C3[1+q] = 0.0;
               C3[2+q] = 0.0;
            }
            for (int a1 = 0; a1 < D1D-1; a1++)
            {
               const int q = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
               C2[q] = 0.0;
               C2[1+q] = 0.0;
               C2[2+q] = 0.0;
               F1[q] = 0.0;
               F1[1+q] = 0.0;
               F1[2+q] = 0.0;
            }
         }
         
         for (int a = 0; a < BASIS_DIM2D_DIFF; a++)
         {
            const int q = 3*(a + BASIS_DIM2D_DIFF*i3);
            C1[q] = 0.0;
            C1[1+q] = 0.0;
            C1[2+q] = 0.0;
            F2[q] = 0.0;
            F2[1+q] = 0.0;
            F2[2+q] = 0.0;
         }
      }

      // C1 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the first spatial dimension
      for (int a = 0; a < BASIS_DIM3D_DIFF; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + (D1D-1)*a1];  

         // aggregate input vector
         real_t u = 0.0, v = 0.0, w = 0.0;
         // k=3, component 0
         int idx = lex_map[a3 + D1D*(a2 + D1D*a1)];
         u -= X(idx, e);

         // k=3, component 1
         v -= X(idx, e);

         // k=3, component 2
         w -= X(idx, e);

         // k=2, component 2
         idx = lex_map[a3+1 + D1D*(a2 + D1D*a1)];
         w += X(idx, e);

         // k=1, component 1 (not computed because \nabla\lambda_{k}
         // component is 0)
         idx = lex_map[a3 + D1D*(a2+1 + D1D*a1)];
         v += X(idx, e);

         // k=0, component 0
         idx = lex_map[a3 + D1D*(a2 + D1D*(a1 + 1))];
         u += X(idx, e);

         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const int a1a2i3 = 3*(i3 + Q1D*a_2d);
            const real_t Gai = Ga3(i3,a);

            C1[a1a2i3] += u * Gai; // cin can be stored outside this inner loop...
            C1[1 + a1a2i3] += v * Gai;
            C1[2 + a1a2i3] += w * Gai;
         }
      }

      // C2 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the second spatial dimension
      for (int a = 0; a < BASIS_DIM2D_DIFF; a++)
      {
         const int a1 = inverse_map2d[a];
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const int a1a2i3 = 3*(i3 + Q1D*a);
            const real_t C1x = C1[a1a2i3];
            const real_t C1y = C1[1 + a1a2i3];
            const real_t C1z = C1[2 + a1a2i3];
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const real_t Gai = Ga2(i2,a);

               const int a1i2i3 = 3*(i2 + Q1D*(i3 + Q1D*a1));
               C2[a1i2i3] += C1x * Gai;
               C2[1 + a1i2i3] += C1y * Gai;
               C2[2 + a1i2i3] += C1z * Gai;
            }
         }
      }

      for (int a1 = 0; a1 < D1D-1; a1++)
      {
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const int a1i2i3 = 3*(i2 + Q1D*(i3 + Q1D*a1));
               const real_t C2x = C2[a1i2i3];
               const real_t C2y = C2[1 + a1i2i3];
               const real_t C2z = C2[2 + a1i2i3];
               for (int i1 = 0; i1 < Q1D; i1++)
               {
                  const real_t Gai = Ga1(i1,a1);
                  const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
                  C3[i1i2i3] += C2x * Gai;
                  C3[1 + i1i2i3] += C2y * Gai;
                  C3[2 + i1i2i3] += C2z * Gai;
               }
            }
         }
      }

      // F1 computes the Bernstein moment over the first ragged tensor dimension.
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            // const int i1i2 = Q1D*(i2 + Q1D*i1);
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               const real_t O11 = D(i1,i2,i3,0,e);
               const real_t O12 = D(i1,i2,i3,1,e);
               const real_t O13 = D(i1,i2,i3,2,e);
               const real_t O21 = symmetric ? O12 : D(i1,i2,i3,3,e);
               const real_t O22 = symmetric ? D(i1,i2,i3,3,e) : D(i1,i2,i3,4,e);
               const real_t O23 = symmetric ? D(i1,i2,i3,4,e) : D(i1,i2,i3,5,e);
               const real_t O31 = symmetric ? O13 : D(i1,i2,i3,6,e);
               const real_t O32 = symmetric ? O23 : D(i1,i2,i3,7,e);
               const real_t O33 = symmetric ? D(i1,i2,i3,5,e) : D(i1,i2,i3,8,e);

               const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
               real_t gX = C3[i1i2i3];
               real_t gY = C3[1 + i1i2i3];
               real_t gZ = C3[2 + i1i2i3];
   
               const real_t fin1 = O11 * gX + O12 * gY + O13 * gZ;
               const real_t fin2 = O21 * gX + O22 * gY + O23 * gZ;
               const real_t fin3 = O31 * gX + O32 * gY + O33 * gZ;
               for (int a1 = 0; a1 < D1D-1; a1++)
               {
                  const real_t Gai = Ga1(i1, a1);
                  const int a1i2i3 = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
                  F1[a1i2i3] += fin1 * Gai;
                  F1[1 + a1i2i3] += fin2 * Gai;
                  F1[2 + a1i2i3] += fin3 * Gai;
               }
            }
         }
      }

      // F2 computes the Bernstein moment over the second ragged tensor dimension.
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int a = 0; a < BASIS_DIM2D_DIFF; a++)
            {
               const int a1 = inverse_map2d[a];
               const real_t Gai = Ga2(i2,a);

               const int a1a2i3 = 3*(a + BASIS_DIM2D_DIFF*i3);
               const int a1i2i3 = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
               F2[a1a2i3] += F1[a1i2i3] * Gai;
               F2[1 + a1a2i3] += F1[1 + a1i2i3] * Gai;
               F2[2 + a1a2i3] += F1[2 + a1i2i3] * Gai;
            }
         }
      }

      for (int a = 0; a < BASIS_DIM3D_DIFF; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + (D1D-1)*a1];  

         real_t u = 0.0, v = 0.0, w = 0.0;
         // const int a1a2a3 = 3*(a3 + a1a2);
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            // const int idx = i3 + Q1D*a;
            const real_t Gai = Ga3(i3,a);
            const int a1a2i3 = 3*(a_2d + BASIS_DIM2D_DIFF*i3);
            u += F2[a1a2i3] * Gai;
            v += F2[1 + a1a2i3] * Gai;
            w += F2[2 + a1a2i3] * Gai; 
         }

         // k=3
         int idx = lex_map[a3 + D1D*(a2 + D1D*a1)];
         Y(idx,e) -= p2 * (u + v + w);
         
         // k=0
         idx = lex_map[a3 + D1D*(a2 + D1D*(a1+1))];
         Y(idx,e) += p2 * u;

         // k=1
         idx = lex_map[a3 + D1D*(a2+1 + D1D*a1)];
         Y(idx,e) += p2 * v;

         // k=2
         idx = lex_map[a3+1 + D1D*(a2 + D1D*a1)];
         Y(idx,e) += p2 * w;
      }

      // // memset(cin, 0, sizeof(real_t) * (int) 3 * (D1D-1) * (D1D) * (D1D+1) / 6);
      // memset(C1, 0, sizeof(real_t) * (int) 3 * BASIS_DIM2D_DIFF * Q1D);
      // memset(C2, 0, sizeof(real_t) * 3 * (D1D-1) * Q1D * Q1D);
      // memset(C3, 0, sizeof(real_t) * 3 * Q1D * Q1D * Q1D);

      // // memset(fin, 0, sizeof(real_t) * 3 * Q1D * Q1D * Q1D);
      // memset(F1, 0, sizeof(real_t) * 3 * (D1D-1) * Q1D * Q1D);
      // memset(F2, 0, sizeof(real_t) * (int) 3 * BASIS_DIM2D_DIFF * Q1D);
      // // memset(F3, 0, sizeof(real_t) * (int) 3 * (D1D-1) * (D1D) * (D1D+1) / 6);
   });

   // delete[] C1;
   // delete[] C2;
   // delete[] C3;
   // delete[] F1;
   // delete[] F2;
}

} // namespace internal

namespace
{
using ApplyKernelType = DiffusionIntegrator::ApplyKernelType;
using ApplySimplexKernelType = DiffusionIntegrator::ApplySimplexKernelType;
using DiagonalKernelType = DiffusionIntegrator::DiagonalKernelType;
}

template<int DIM, int T_D1D, int T_Q1D>
ApplyKernelType DiffusionIntegrator::ApplyPAKernels::Kernel()
{
   if (DIM == 2) { return internal::SmemPADiffusionApply2D<T_D1D,T_Q1D>; }
   else if (DIM == 3) { return internal::SmemPADiffusionApply3D<T_D1D, T_Q1D>; }
   else { MFEM_ABORT(""); }
}

inline
ApplyKernelType DiffusionIntegrator::ApplyPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2) { return internal::PADiffusionApply2D; }
   else if (DIM == 3) { return internal::PADiffusionApply3D; }
   else { MFEM_ABORT(""); }
}

template<int DIM, int T_D1D, int T_Q1D>
ApplySimplexKernelType DiffusionIntegrator::ApplySimplexPAKernels::Kernel()
{
   if (DIM == 2) { return internal::PADiffusionApplyTriangle<T_D1D,T_Q1D>; }
   else if (DIM == 3) { return internal::PADiffusionApplyTetrahedron<T_D1D,T_Q1D>; }
   else { MFEM_ABORT(""); }
}

inline
ApplySimplexKernelType DiffusionIntegrator::ApplySimplexPAKernels::Fallback(
   int DIM, int, int)
{
   if (DIM == 2) { return internal::PADiffusionApplyTriangle; }
   else if (DIM == 3) { return internal::PADiffusionApplyTetrahedron; }
   else { MFEM_ABORT(""); }
}

template<int DIM, int D1D, int Q1D>
DiagonalKernelType DiffusionIntegrator::DiagonalPAKernels::Kernel()
{
   if (DIM == 2) { return internal::SmemPADiffusionDiagonal2D<D1D,Q1D>; }
   else if (DIM == 3) { return internal::SmemPADiffusionDiagonal3D<D1D, Q1D>; }
   else { MFEM_ABORT(""); }
}

inline DiagonalKernelType
DiffusionIntegrator::DiagonalPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2) { return internal::PADiffusionDiagonal2D; }
   else if (DIM == 3) { return internal::PADiffusionDiagonal3D; }
   else { MFEM_ABORT(""); }
}
/// \endcond DO_NOT_DOCUMENT

} // namespace mfem


#endif
