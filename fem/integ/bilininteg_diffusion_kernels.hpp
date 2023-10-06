// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

namespace mfem
{

namespace internal
{

void PADiffusionSetup(const int dim,
                      const int sdim,
                      const int D1D,
                      const int Q1D,
                      const int coeffDim,
                      const int NE,
                      const Array<double> &W,
                      const Vector &J,
                      const Vector &C,
                      Vector &D);

// PA Diffusion Assemble 2D kernel
template<int T_SDIM>
void PADiffusionSetup2D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const Array<double> &w,
                        const Vector &j,
                        const Vector &c,
                        Vector &d);

// PA Diffusion Assemble 3D kernel
void PADiffusionSetup3D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const Array<double> &w,
                        const Vector &j,
                        const Vector &c,
                        Vector &d);

#ifdef MFEM_USE_OCCA
// OCCA 2D Assemble kernel
void OccaPADiffusionSetup2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &W,
                            const Vector &J,
                            const Vector &C,
                            Vector &op);

// OCCA 3D Assemble kernel
void OccaPADiffusionSetup3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &W,
                            const Vector &J,
                            const Vector &C,
                            Vector &op);
#endif // MFEM_USE_OCCA

void PADiffusionAssembleDiagonal(const int dim,
                                 const int D1D,
                                 const int Q1D,
                                 const int NE,
                                 const bool symm,
                                 const Array<double> &B,
                                 const Array<double> &G,
                                 const Vector &D,
                                 Vector &Y);

// PA Diffusion Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionDiagonal2D(const int NE,
                                  const bool symmetric,
                                  const Array<double> &b,
                                  const Array<double> &g,
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
      double QD0[MQ1][MD1];
      double QD1[MQ1][MD1];
      double QD2[MQ1][MD1];
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
               const double D00 = D(q,0,e);
               const double D10 = D(q,1,e);
               const double D01 = symmetric ? D10 : D(q,2,e);
               const double D11 = symmetric ? D(q,2,e) : D(q,3,e);
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

// Shared memory PA Diffusion Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
inline void SmemPADiffusionDiagonal2D(const int NE,
                                      const bool symmetric,
                                      const Array<double> &b_,
                                      const Array<double> &g_,
                                      const Vector &d_,
                                      Vector &y_,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
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
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_SHARED double BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG+0);
      double (*G)[MD1] = (double (*)[MD1]) (BG+1);
      MFEM_SHARED double QD[3][NBZ][MD1][MQ1];
      double (*QD0)[MD1] = (double (*)[MD1])(QD[0] + tidz);
      double (*QD1)[MD1] = (double (*)[MD1])(QD[1] + tidz);
      double (*QD2)[MD1] = (double (*)[MD1])(QD[2] + tidz);
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
               const double D00 = D(q,0,e);
               const double D10 = D(q,1,e);
               const double D01 = symmetric ? D10 : D(q,2,e);
               const double D11 = symmetric ? D(q,2,e) : D(q,3,e);
               const double By = B[qy][dy];
               const double Gy = G[qy][dy];
               const double BBy = By * By;
               const double BGy = By * Gy;
               const double GGy = Gy * Gy;
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
               const double Bx = B[qx][dx];
               const double Gx = G[qx][dx];
               const double BBx = Bx * Bx;
               const double BGx = Bx * Gx;
               const double GGx = Gx * Gx;
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
                                  const Array<double> &b,
                                  const Array<double> &g,
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
      double QQD[MQ1][MQ1][MD1];
      double QDD[MQ1][MD1][MD1];
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
                        const double O = Q(q,k,e);
                        const double Bz = B(qz,dz);
                        const double Gz = G(qz,dz);
                        const double L = i==2 ? Gz : Bz;
                        const double R = j==2 ? Gz : Bz;
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
                        const double By = B(qy,dy);
                        const double Gy = G(qy,dy);
                        const double L = i==1 ? Gy : By;
                        const double R = j==1 ? Gy : By;
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
                        const double Bx = B(qx,dx);
                        const double Gx = G(qx,dx);
                        const double L = i==0 ? Gx : Bx;
                        const double R = j==0 ? Gx : Bx;
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
                                      const Array<double> &b_,
                                      const Array<double> &g_,
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
      MFEM_SHARED double BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG+0);
      double (*G)[MD1] = (double (*)[MD1]) (BG+1);
      MFEM_SHARED double QQD[MQ1][MQ1][MD1];
      MFEM_SHARED double QDD[MQ1][MD1][MD1];
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
                        const double O = D(q,k,e);
                        const double Bz = B[qz][dz];
                        const double Gz = G[qz][dz];
                        const double L = i==2 ? Gz : Bz;
                        const double R = j==2 ? Gz : Bz;
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
                        const double By = B[qy][dy];
                        const double Gy = G[qy][dy];
                        const double L = i==1 ? Gy : By;
                        const double R = j==1 ? Gy : By;
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
                        const double Bx = B[qx][dx];
                        const double Gx = G[qx][dx];
                        const double L = i==0 ? Gx : Bx;
                        const double R = j==0 ? Gx : Bx;
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
                      const Array<double> &B,
                      const Array<double> &G,
                      const Array<double> &Bt,
                      const Array<double> &Gt,
                      const Vector &D,
                      const Vector &X,
                      Vector &Y);

#ifdef MFEM_USE_OCCA
// OCCA PA Diffusion Apply 2D kernel
void OccaPADiffusionApply2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &B,
                            const Array<double> &G,
                            const Array<double> &Bt,
                            const Array<double> &Gt,
                            const Vector &D,
                            const Vector &X,
                            Vector &Y);

// OCCA PA Diffusion Apply 3D kernel
void OccaPADiffusionApply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &B,
                            const Array<double> &G,
                            const Array<double> &Bt,
                            const Array<double> &Gt,
                            const Vector &D,
                            const Vector &X,
                            Vector &Y);
#endif // MFEM_USE_OCCA

// PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApply2D(const int NE,
                               const bool symmetric,
                               const Array<double> &b_,
                               const Array<double> &g_,
                               const Array<double> &bt_,
                               const Array<double> &gt_,
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

      double grad[max_Q1D][max_Q1D][2];
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
         double gradX[max_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = X(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * B(qx,dx);
               gradX[qx][1] += s * G(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
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

            const double O11 = D(q,0,e);
            const double O21 = D(q,1,e);
            const double O12 = symmetric ? O21 : D(q,2,e);
            const double O22 = symmetric ? D(q,2,e) : D(q,3,e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O21 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double gradX[max_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double wx  = Bt(dx,qx);
               const double wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy  = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// Shared memory PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
inline void SmemPADiffusionApply2D(const int NE,
                                   const bool symmetric,
                                   const Array<double> &b_,
                                   const Array<double> &g_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
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
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      MFEM_SHARED double GQ[2][NBZ][MD1][MQ1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + tidz);
      double (*DQ0)[MD1] = (double (*)[MD1])(GD[0] + tidz);
      double (*DQ1)[MD1] = (double (*)[MD1])(GD[1] + tidz);
      double (*QQ0)[MD1] = (double (*)[MD1])(GQ[0] + tidz);
      double (*QQ1)[MD1] = (double (*)[MD1])(GQ[1] + tidz);
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
            double u = 0.0;
            double v = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double coords = X[dy][dx];
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
            double u = 0.0;
            double v = 0.0;
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
            const double O11 = D(q,0,e);
            const double O21 = D(q,1,e);
            const double O12 = symmetric ? O21 : D(q,2,e);
            const double O22 = symmetric ? D(q,2,e) : D(q,3,e);
            const double gX = QQ0[qy][qx];
            const double gY = QQ1[qy][qx];
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
            double u = 0.0;
            double v = 0.0;
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
            double u = 0.0;
            double v = 0.0;
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

// PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PADiffusionApply3D(const int NE,
                               const bool symmetric,
                               const Array<double> &b,
                               const Array<double> &g,
                               const Array<double> &bt,
                               const Array<double> &gt,
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
      double grad[max_Q1D][max_Q1D][max_Q1D][3];
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
         double gradXY[max_Q1D][max_Q1D][3];
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
            double gradX[max_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = X(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
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
               const double O11 = D(q,0,e);
               const double O12 = D(q,1,e);
               const double O13 = D(q,2,e);
               const double O21 = symmetric ? O12 : D(q,3,e);
               const double O22 = symmetric ? D(q,3,e) : D(q,4,e);
               const double O23 = symmetric ? D(q,4,e) : D(q,5,e);
               const double O31 = symmetric ? O13 : D(q,6,e);
               const double O32 = symmetric ? O23 : D(q,7,e);
               const double O33 = symmetric ? D(q,5,e) : D(q,8,e);
               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];
               grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[qz][qy][qx][1] = (O21*gradX)+(O22*gradY)+(O23*gradZ);
               grad[qz][qy][qx][2] = (O31*gradX)+(O32*gradY)+(O33*gradZ);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double gradXY[max_D1D][max_D1D][3];
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
            double gradX[max_D1D][3];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double gX = grad[qz][qy][qx][0];
               const double gY = grad[qz][qy][qx][1];
               const double gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double wx  = Bt(dx,qx);
                  const double wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy  = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
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
            const double wz  = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
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
                                   const Array<double> &b_,
                                   const Array<double> &g_,
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
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double sm0[3][MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[3][MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);
      double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
      double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
      double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);
      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
      double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1);
      double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2);
      double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
      double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1);
      double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);
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
               double u = 0.0, v = 0.0;
               MFEM_UNROLL(MD1)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double coords = X[dz][dy][dx];
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
               double u = 0.0, v = 0.0, w = 0.0;
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
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u += DQQ0[dz][qy][qx] * B[qz][dz];
                  v += DQQ1[dz][qy][qx] * B[qz][dz];
                  w += DQQ2[dz][qy][qx] * G[qz][dz];
               }
               const double O11 = d(qx,qy,qz,0,e);
               const double O12 = d(qx,qy,qz,1,e);
               const double O13 = d(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : d(qx,qy,qz,3,e);
               const double O22 = symmetric ? d(qx,qy,qz,3,e) : d(qx,qy,qz,4,e);
               const double O23 = symmetric ? d(qx,qy,qz,4,e) : d(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : d(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : d(qx,qy,qz,7,e);
               const double O33 = symmetric ? d(qx,qy,qz,5,e) : d(qx,qy,qz,8,e);
               const double gX = u;
               const double gY = v;
               const double gZ = w;
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
               double u = 0.0, v = 0.0, w = 0.0;
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
               double u = 0.0, v = 0.0, w = 0.0;
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
               double u = 0.0, v = 0.0, w = 0.0;
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

} // namespace internal

} // namespace mfem

#endif
