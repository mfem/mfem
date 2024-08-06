// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG_MASS_KERNELS_HPP
#define MFEM_BILININTEG_MASS_KERNELS_HPP

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

// PA Mass Diagonal 1D kernel
static void PAMassAssembleDiagonal1D(const int NE,
                                     const Array<real_t> &b,
                                     const Vector &d,
                                     Vector &y,
                                     const int D1D,
                                     const int Q1D)
{
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            Y(dx, e) += B(qx, dx) * B(qx, dx) * D(qx, e);
         }
      }
   });
}

MFEM_HOST_DEVICE inline
void PAMassApply1D_Element(const int e,
                           const int NE,
                           const real_t *b_,
                           const real_t *bt_,
                           const real_t *d_,
                           const real_t *x_,
                           real_t *y_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   auto B = ConstDeviceMatrix(b_, Q1D, D1D);
   auto Bt = ConstDeviceMatrix(bt_, D1D, Q1D);
   auto D = ConstDeviceMatrix(d_, Q1D, NE);
   auto X = ConstDeviceMatrix(x_, D1D, NE);
   auto Y = DeviceMatrix(y_, D1D, NE);

   real_t XQ[DofQuadLimits::MAX_Q1D];
   for (int qx = 0; qx < Q1D; ++qx)
   {
      XQ[qx] = 0.0;
   }
   for (int dx = 0; dx < D1D; ++dx)
   {
      const real_t s = X(dx,e);
      for (int qx = 0; qx < Q1D; ++qx)
      {
         XQ[qx] += B(qx,dx)*s;
      }
   }
   for (int qx = 0; qx < Q1D; ++qx)
   {
      const double q = XQ[qx]*D(qx,e);
      for (int dx = 0; dx < D1D; ++dx)
      {
         Y(dx,e) += Bt(dx,qx) * q;
      }
   }
}

// PA Mass Apply 1D kernel
static void PAMassApply1D(const int NE,
                          const Array<real_t> &b_,
                          const Array<real_t> &bt_,
                          const Vector &d_,
                          const Vector &x_,
                          Vector &y_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = b_.Read();
   const auto Bt = bt_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApply1D_Element(e, NE, B, Bt, D, X, Y, d1d, q1d);
   });
}

// PA Mass Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassAssembleDiagonal2D(const int NE,
                                     const Array<real_t> &b,
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
   auto D = Reshape(d.Read(), Q1D, Q1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t QD[MQ1][MD1];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            QD[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               QD[qx][dy] += B(qy, dy) * B(qy, dy) * D(qx, qy, e);
            }
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Y(dx,dy,e) += B(qx, dx) * B(qx, dx) * QD[qx][dy];
            }
         }
      }
   });
}

namespace mass
{
constexpr int ipow(int x, int p) { return p == 0 ? 1 : x*ipow(x, p-1); }
constexpr int D(int D1D) { return (11 - D1D) / 2; }
constexpr int NBZ(int D1D)
{
   return ipow(2, D(D1D) >= 0 ? D(D1D) : 0);
}
}

// Shared memory PA Mass Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassAssembleDiagonal2D(const int NE,
                                         const Array<real_t> &b_,
                                         const Vector &d_,
                                         Vector &y_,
                                         const int d1d = 0,
                                         const int q1d = 0)
{
   static constexpr int T_NBZ = mass::NBZ(T_D1D);
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);
   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_SHARED real_t B[MQ1][MD1];
      MFEM_SHARED real_t QDZ[NBZ][MQ1][MD1];
      real_t (*QD)[MD1] = (real_t (*)[MD1])(QDZ + tidz);
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            QD[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               QD[qx][dy] += B[qy][dy] * B[qy][dy] * D(qx, qy, e);
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
               // might need absolute values on next line
               Y(dx,dy,e) += B[qx][dx] * B[qx][dx] * QD[qx][dy];
            }
         }
      }
   });
}

// PA Mass Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassAssembleDiagonal3D(const int NE,
                                     const Array<real_t> &b,
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
   auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t QQD[MQ1][MQ1][MD1];
      real_t QDD[MQ1][MD1][MD1];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               QQD[qx][qy][dz] = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  QQD[qx][qy][dz] += B(qz, dz) * B(qz, dz) * D(qx, qy, qz, e);
               }
            }
         }
      }
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               QDD[qx][dy][dz] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  QDD[qx][dy][dz] += B(qy, dy) * B(qy, dy) * QQD[qx][qy][dz];
               }
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               real_t t = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  t += B(qx, dx) * B(qx, dx) * QDD[qx][dy][dz];
               }
               Y(dx, dy, dz, e) += t;
            }
         }
      }
   });
}

// Shared memory PA Mass Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassAssembleDiagonal3D(const int NE,
                                         const Array<real_t> &b_,
                                         const Vector &d_,
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
   auto D = Reshape(d_.Read(), Q1D, Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      MFEM_SHARED real_t B[MQ1][MD1];
      MFEM_SHARED real_t QQD[MQ1][MQ1][MD1];
      MFEM_SHARED real_t QDD[MQ1][MD1][MD1];
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               QQD[qx][qy][dz] = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  QQD[qx][qy][dz] += B[qz][dz] * B[qz][dz] * D(qx, qy, qz, e);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               QDD[qx][dy][dz] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  QDD[qx][dy][dz] += B[qy][dy] * B[qy][dy] * QQD[qx][qy][dz];
               }
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
               real_t t = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  t += B[qx][dx] * B[qx][dx] * QDD[qx][dy][dz];
               }
               Y(dx, dy, dz, e) += t;
            }
         }
      }
   });
}

#ifdef MFEM_USE_OCCA
// OCCA PA Mass Apply 2D kernel
void OccaPAMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &B,
                       const Array<real_t> &Bt,
                       const Vector &D,
                       const Vector &X,
                       Vector &Y);

// OCCA PA Mass Apply 3D kernel
void OccaPAMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &B,
                       const Array<real_t> &Bt,
                       const Vector &D,
                       const Vector &X,
                       Vector &Y);
#endif // MFEM_USE_OCCA

template <bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void PAMassApply2D_Element(const int e,
                           const int NE,
                           const real_t *b_,
                           const real_t *bt_,
                           const real_t *d_,
                           const real_t *x_,
                           real_t *y_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   auto B = ConstDeviceMatrix(b_, Q1D, D1D);
   auto Bt = ConstDeviceMatrix(bt_, D1D, Q1D);
   auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   auto X = ConstDeviceCube(x_, D1D, D1D, NE);
   auto Y = DeviceCube(y_, D1D, D1D, NE);

   if (!ACCUMULATE)
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx, dy, e) = 0.0;
         }
      }
   }

   constexpr int max_D1D = DofQuadLimits::MAX_D1D;
   constexpr int max_Q1D = DofQuadLimits::MAX_Q1D;
   real_t sol_xy[max_Q1D][max_Q1D];
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] = 0.0;
      }
   }
   for (int dy = 0; dy < D1D; ++dy)
   {
      real_t sol_x[max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         sol_x[qy] = 0.0;
      }
      for (int dx = 0; dx < D1D; ++dx)
      {
         const real_t s = X(dx,dy,e);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] += B(qx,dx)* s;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         const real_t d2q = B(qy,dy);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] += d2q * sol_x[qx];
         }
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] *= D(qx,qy,e);
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      real_t sol_x[max_D1D];
      for (int dx = 0; dx < D1D; ++dx)
      {
         sol_x[dx] = 0.0;
      }
      for (int qx = 0; qx < Q1D; ++qx)
      {
         const real_t s = sol_xy[qy][qx];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] += Bt(dx,qx) * s;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         const real_t q2d = Bt(dy,qy);
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx,dy,e) += q2d * sol_x[dx];
         }
      }
   }
}

template<int T_D1D, int T_Q1D, int T_NBZ, bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void SmemPAMassApply2D_Element(const int e,
                               const int NE,
                               const real_t *b_,
                               const real_t *d_,
                               const real_t *x_,
                               real_t *y_,
                               int d1d = 0,
                               int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

   auto b = ConstDeviceMatrix(b_, Q1D, D1D);
   auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   auto x = ConstDeviceCube(x_, D1D, D1D, NE);
   auto Y = DeviceCube(y_, D1D, D1D, NE);

   const int tidz = MFEM_THREAD_ID(z);

   MFEM_SHARED real_t BBt[MQ1*MD1];
   real_t (*B)[MD1] = (real_t (*)[MD1]) BBt;
   real_t (*Bt)[MQ1] = (real_t (*)[MQ1]) BBt;
   MFEM_SHARED real_t sm0[NBZ][MDQ*MDQ];
   MFEM_SHARED real_t sm1[NBZ][MDQ*MDQ];
   real_t (*X)[MD1] = (real_t (*)[MD1]) (sm0 + tidz);
   real_t (*DQ)[MQ1] = (real_t (*)[MQ1]) (sm1 + tidz);
   real_t (*QQ)[MQ1] = (real_t (*)[MQ1]) (sm0 + tidz);
   real_t (*QD)[MD1] = (real_t (*)[MD1]) (sm1 + tidz);


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
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t dq = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            dq += X[dy][dx] * B[qx][dx];
         }
         DQ[dy][qx] = dq;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t qq = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            qq += DQ[dy][qx] * B[qy][dy];
         }
         QQ[qy][qx] = qq * D(qx, qy, e);
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
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         real_t dq = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            dq += QQ[qy][qx] * Bt[dx][qx];
         }
         QD[qy][dx] = dq;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         real_t dd = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            dd += (QD[qy][dx] * Bt[dy][qy]);
         }
         if (ACCUMULATE)
         {
            Y(dx, dy, e) += dd;
         }
         else
         {
            Y(dx, dy, e) = dd;
         }
      }
   }
}

template <bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void PAMassApply3D_Element(const int e,
                           const int NE,
                           const real_t *b_,
                           const real_t *bt_,
                           const real_t *d_,
                           const real_t *x_,
                           real_t *y_,
                           const int d1d,
                           const int q1d)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   auto B = ConstDeviceMatrix(b_, Q1D, D1D);
   auto Bt = ConstDeviceMatrix(bt_, D1D, Q1D);
   auto D = DeviceTensor<4,const real_t>(d_, Q1D, Q1D, Q1D, NE);
   auto X = DeviceTensor<4,const real_t>(x_, D1D, D1D, D1D, NE);
   auto Y = DeviceTensor<4,real_t>(y_, D1D, D1D, D1D, NE);

   if (!ACCUMULATE)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx, dy, dz, e) = 0.0;
            }
         }
      }
   }

   constexpr int max_D1D = DofQuadLimits::MAX_D1D;
   constexpr int max_Q1D = DofQuadLimits::MAX_Q1D;
   real_t sol_xyz[max_Q1D][max_Q1D][max_Q1D];
   for (int qz = 0; qz < Q1D; ++qz)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xyz[qz][qy][qx] = 0.0;
         }
      }
   }
   for (int dz = 0; dz < D1D; ++dz)
   {
      real_t sol_xy[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t sol_x[max_Q1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] = 0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s = X(dx,dy,dz,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B(qx,dx) * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t wy = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += wy * sol_x[qx];
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         const real_t wz = B(qz,dz);
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
            }
         }
      }
   }
   for (int qz = 0; qz < Q1D; ++qz)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xyz[qz][qy][qx] *= D(qx,qy,qz,e);
         }
      }
   }
   for (int qz = 0; qz < Q1D; ++qz)
   {
      real_t sol_xy[max_D1D][max_D1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_xy[dy][dx] = 0;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t sol_x[max_D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t s = sol_xyz[qz][qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t wy = Bt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] += wy * sol_x[dx];
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         const real_t wz = Bt(dz,qz);
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
            }
         }
      }
   }
}

template<int T_D1D, int T_Q1D, bool ACCUMULATE = true>
MFEM_HOST_DEVICE inline
void SmemPAMassApply3D_Element(const int e,
                               const int NE,
                               const real_t *b_,
                               const real_t *d_,
                               const real_t *x_,
                               real_t *y_,
                               const int d1d = 0,
                               const int q1d = 0)
{
   constexpr int D1D = T_D1D ? T_D1D : d1d;
   constexpr int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

   auto b = ConstDeviceMatrix(b_, Q1D, D1D);
   auto d = DeviceTensor<4,const real_t>(d_, Q1D, Q1D, Q1D, NE);
   auto x = DeviceTensor<4,const real_t>(x_, D1D, D1D, D1D, NE);
   auto y = DeviceTensor<4,real_t>(y_, D1D, D1D, D1D, NE);

   MFEM_SHARED real_t sDQ[MQ1*MD1];
   real_t (*B)[MD1] = (real_t (*)[MD1]) sDQ;
   real_t (*Bt)[MQ1] = (real_t (*)[MQ1]) sDQ;
   MFEM_SHARED real_t sm0[MDQ*MDQ*MDQ];
   MFEM_SHARED real_t sm1[MDQ*MDQ*MDQ];
   real_t (*X)[MD1][MD1]   = (real_t (*)[MD1][MD1]) sm0;
   real_t (*DDQ)[MD1][MQ1] = (real_t (*)[MD1][MQ1]) sm1;
   real_t (*DQQ)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) sm0;
   real_t (*QQQ)[MQ1][MQ1] = (real_t (*)[MQ1][MQ1]) sm1;
   real_t (*QQD)[MQ1][MD1] = (real_t (*)[MQ1][MD1]) sm0;
   real_t (*QDD)[MD1][MD1] = (real_t (*)[MD1][MD1]) sm1;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            X[dz][dy][dx] = x(dx,dy,dz,e);
         }
      }
      MFEM_FOREACH_THREAD(dx,x,Q1D)
      {
         B[dx][dy] = b(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dx = 0; dx < D1D; ++dx)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += X[dz][dy][dx] * B[qx][dx];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            DDQ[dz][dy][qx] = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dy = 0; dy < D1D; ++dy)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] += DDQ[dz][dy][qx] * B[qy][dy];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            DQQ[dz][qy][qx] = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         real_t u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; qz++)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] += DQQ[dz][qy][qx] * B[qz][dz];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; qz++)
         {
            QQQ[qz][qy][qx] = u[qz] * d(qx,qy,qz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(di,y,D1D)
   {
      MFEM_FOREACH_THREAD(q,x,Q1D)
      {
         Bt[di][q] = b(q,di);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         real_t u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQQ[qz][qy][qx] * Bt[dx][qx];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQD[qz][qy][dx] = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         real_t u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qy = 0; qy < Q1D; ++qy)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQD[qz][qy][dx] * Bt[dy][qy];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QDD[qz][dy][dx] = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         real_t u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += QDD[qz][dy][dx] * Bt[dz][qz];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            if (ACCUMULATE)
            {
               y(dx,dy,dz,e) += u[dz];
            }
            else
            {
               y(dx,dy,dz,e) = u[dz];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// PA Mass Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassApply2D(const int NE,
                          const Array<real_t> &b_,
                          const Array<real_t> &bt_,
                          const Vector &d_,
                          const Vector &x_,
                          Vector &y_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = b_.Read();
   const auto Bt = bt_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApply2D_Element(e, NE, B, Bt, D, X, Y, d1d, q1d);
   });
}

// Shared memory PA Mass Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApply2D(const int NE,
                              const Array<real_t> &b_,
                              const Array<real_t> &bt_,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   MFEM_CONTRACT_VAR(bt_);
   static constexpr int T_NBZ = mass::NBZ(T_D1D);
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   const auto b = b_.Read();
   const auto D = d_.Read();
   const auto x = x_.Read();
   auto Y = y_.ReadWrite();
   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassApply2D_Element<T_D1D,T_Q1D,T_NBZ>(
         e, NE, b, D, x, Y, d1d, q1d);
   });
}

// PA Mass Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassApply3D(const int NE,
                          const Array<real_t> &b_,
                          const Array<real_t> &bt_,
                          const Vector &d_,
                          const Vector &x_,
                          Vector &y_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = b_.Read();
   const auto Bt = bt_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApply3D_Element(e, NE, B, Bt, D, X, Y, d1d, q1d);
   });
}

// Shared memory PA Mass Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAMassApply3D(const int NE,
                              const Array<real_t> &b_,
                              const Array<real_t> &bt_,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   MFEM_CONTRACT_VAR(bt_);
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int max_q1d = T_Q1D ? T_Q1D : DeviceDofQuadLimits::Get().MAX_Q1D;
   const int max_d1d = T_D1D ? T_D1D : DeviceDofQuadLimits::Get().MAX_D1D;
   MFEM_VERIFY(D1D <= max_d1d, "");
   MFEM_VERIFY(Q1D <= max_q1d, "");
   auto b = b_.Read();
   auto d = d_.Read();
   auto x = x_.Read();
   auto y = y_.ReadWrite();
   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassApply3D_Element<T_D1D,T_Q1D>(e, NE, b, d, x, y, d1d, q1d);
   });
}

} // namespace internal

namespace
{
using ApplyKernelType = MassIntegrator::ApplyKernelType;
using DiagonalKernelType = MassIntegrator::DiagonalKernelType;
}

template<int DIM, int T_D1D, int T_Q1D>
ApplyKernelType MassIntegrator::ApplyPAKernels::Kernel()
{
   if (DIM == 1) { return internal::PAMassApply1D; }
   else if (DIM == 2) { return internal::SmemPAMassApply2D<T_D1D,T_Q1D>; }
   else if (DIM == 3) { return internal::SmemPAMassApply3D<T_D1D, T_Q1D>; }
   else { MFEM_ABORT(""); }
}

inline ApplyKernelType MassIntegrator::ApplyPAKernels::Fallback(
   int DIM, int, int)
{
   if (DIM == 1) { return internal::PAMassApply1D; }
   else if (DIM == 2) { return internal::PAMassApply2D; }
   else if (DIM == 3) { return internal::PAMassApply3D; }
   else { MFEM_ABORT(""); }
}

template<int DIM, int T_D1D, int T_Q1D>
DiagonalKernelType MassIntegrator::DiagonalPAKernels::Kernel()
{
   if (DIM == 1) { return internal::PAMassAssembleDiagonal1D; }
   else if (DIM == 2) { return internal::SmemPAMassAssembleDiagonal2D<T_D1D,T_Q1D>; }
   else if (DIM == 3) { return internal::SmemPAMassAssembleDiagonal3D<T_D1D, T_Q1D>; }
   else { MFEM_ABORT(""); }
}

inline DiagonalKernelType MassIntegrator::DiagonalPAKernels::Fallback(
   int DIM, int, int)
{
   if (DIM == 1) { return internal::PAMassAssembleDiagonal1D; }
   else if (DIM == 2) { return internal::PAMassAssembleDiagonal2D; }
   else if (DIM == 3) { return internal::PAMassAssembleDiagonal3D; }
   else { MFEM_ABORT(""); }
}

} // namespace mfem

#endif
