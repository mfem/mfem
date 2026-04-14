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

/// \cond DO_NOT_DOCUMENT

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

template <bool ACCUMULATE = true>
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

   if (!ACCUMULATE)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         Y(dx, e) = 0.0;
      }
   }

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
   const auto b = b_.Read();
   const auto d = d_.Read();
   const auto x = x_.Read();
   auto y = y_.ReadWrite();
   mfem::forall_2D<T_Q1D*T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassApply3D_Element<T_D1D,T_Q1D>(e, NE, b, d, x, y, d1d, q1d);
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void EAMassAssemble1D(const int NE,
                             const Array<real_t> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   const auto B = Reshape(basis.Read(), Q1D, D1D);
   const auto D = Reshape(padata.Read(), Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(), D1D, D1D, NE);
   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         real_t r_Bi[MQ1];
         for (int q = 0; q < Q1D; q++) { r_Bi[q] = B(q,i1); }
         MFEM_FOREACH_THREAD(j1,y,D1D)
         {
            real_t r_Bj[MQ1];
            for (int q = 0; q < Q1D; q++) { r_Bj[q] = B(q,j1); }

            real_t val = 0.0;
            for (int k1 = 0; k1 < Q1D; ++k1)
            {
               val += r_Bi[k1] * r_Bj[k1] * D(k1, e);
            }
            if (add)
            {
               M(i1, j1, e) += val;
            }
            else
            {
               M(i1, j1, e) = val;
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void EAMassAssemble2D(const int NE,
                             const Array<real_t> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(), D1D, D1D, D1D, D1D,
                    NE);
   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t r_B[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
         }
      }
      MFEM_SHARED real_t s_D[MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            s_D[k1][k2] = D(k1,k2,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            for (int j1 = 0; j1 < D1D; ++j1)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  real_t val = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        val += r_B[k1][i1] * r_B[k1][j1]
                               * r_B[k2][i2] * r_B[k2][j2]
                               * s_D[k1][k2];
                     }
                  }
                  if (add)
                  {
                     M(i1, i2, j1, j2, e) += val;
                  }
                  else
                  {
                     M(i1, i2, j1, j2, e) = val;
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
inline void EAMassAssemble3D(const int NE,
                             const Array<real_t> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(), D1D, D1D, D1D, D1D,
                    D1D, D1D, NE);
   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int DQ = T_D1D * T_Q1D;

      // For quadratic and lower it's better to use registers but for higher-order you start to
      // spill and it's better to use shared memory
      constexpr bool USE_REG = DQ != 0 && DQ <= 12;
      constexpr int MD1r = USE_REG ? MD1 : 1;
      constexpr int MQ1r = USE_REG ? MQ1 : 1;
      constexpr int MD1s = USE_REG ? 1 : MD1;
      constexpr int MQ1s = USE_REG ? 1 : MQ1;

      MFEM_SHARED real_t s_B[MQ1s][MD1s];
      real_t r_B[MQ1r][MD1r];
      real_t (*l_B)[MD1] = nullptr;
      if (USE_REG)
      {
         for (int d = 0; d < D1D; d++)
         {
            for (int q = 0; q < Q1D; q++)
            {
               r_B[q][d] = B(q,d);
            }
         }
         l_B = (real_t (*)[MD1])r_B;
      }
      else
      {
         if (MFEM_THREAD_ID(z) == 0)
         {
            MFEM_FOREACH_THREAD(d,x,D1D)
            {
               MFEM_FOREACH_THREAD(q,y,Q1D)
               {
                  s_B[q][d] = B(q,d);
               }
            }
         }
         l_B = (real_t (*)[MD1])s_B;
      }

      MFEM_SHARED real_t s_D[MQ1][MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            MFEM_FOREACH_THREAD(k3,z,Q1D)
            {
               s_D[k1][k2][k3] = D(k1,k2,k3,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j3 = 0; j3 < D1D; ++j3)
                     {
                        real_t val = 0.0;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 val += l_B[k1][i1] * l_B[k1][j1]
                                        * l_B[k2][i2] * l_B[k2][j2]
                                        * l_B[k3][i3] * l_B[k3][j3]
                                        * s_D[k1][k2][k3];
                              }
                           }
                        }
                        if (add)
                        {
                           M(i1, i2, i3, j1, j2, j3, e) += val;
                        }
                        else
                        {
                           M(i1, i2, i3, j1, j2, j3, e) = val;
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

template <bool UPPER, int T_D1D = 0, int T_Q1D = 0>
inline void EAMassAssembleTriangular1D(const int NE,
                                       const Array<real_t> &basis,
                                       const Vector &padata,
                                       Vector &eadata,
                                       const bool add,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(),
                    TriPackMatrix<TriangularPart::UPPER>::PackedSize(D1D), NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      for (int i1 = 0; i1 < D1D; ++i1)
      {
         const int j_begin = UPPER ? i1 : 0;
         const int j_end = UPPER ? D1D : i1 + 1;
         for (int j1 = j_begin; j1 < j_end; ++j1)
         {
            real_t val = 0.0;
            for (int k1 = 0; k1 < Q1D; ++k1)
            {
               val += B(k1, i1) * B(k1, j1) * D(k1, e);
            }
            int idx = 0;
            if constexpr (UPPER)
            {
               idx = TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i1, j1, D1D);
            }
            else
            {
               idx = TriPackMatrix<TriangularPart::LOWER>::LowerIndex(i1, j1);
            }
            if (add)
            {
               M(idx, e) += val;
            }
            else
            {
               M(idx, e) = val;
            }
         }
      }
   });
}

template <int T_D1D, int T_Q1D, int T_COLB, int T_NT>
inline void EAMassAssembleTriangular2D_UpperBlockCols(
   const int NE,
   const Array<real_t> &basis,
   const Vector &padata,
   Vector &eadata,
   const bool add,
   const int d1d = 0,
   const int q1d = 0);

template <bool UPPER, int T_D1D = 0, int T_Q1D = 0>
inline void EAMassAssembleTriangular2D(const int NE,
                                       const Array<real_t> &basis,
                                       const Vector &padata,
                                       Vector &eadata,
                                       const bool add,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   const int ndofs = D1D*D1D;
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(),
                    TriPackMatrix<TriangularPart::UPPER>::PackedSize(ndofs), NE);

   if constexpr (UPPER && T_D1D > 0 && T_Q1D > 0)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         return EAMassAssembleTriangular2D_UpperBlockCols<T_D1D, T_Q1D, 4, 32>(
                   NE, basis, padata, eadata, add, d1d, q1d);
      }
   }

   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      const int ndofs = D1D*D1D;
      real_t r_B[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
         }
      }
      MFEM_SHARED real_t s_D[MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            s_D[k1][k2] = D(k1,k2,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            const int row = i1 + D1D*i2;
            for (int j2 = 0; j2 < D1D; ++j2)
            {
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  const int col = j1 + D1D*j2;
                  if ((UPPER && row > col) || (!UPPER && row < col))
                  {
                     continue;
                  }
                  real_t val = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        val += r_B[k1][i1] * r_B[k1][j1]
                               * r_B[k2][i2] * r_B[k2][j2]
                               * s_D[k1][k2];
                     }
                  }
                  int idx = 0;
                  if constexpr (UPPER)
                  {
                     idx = TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, col, ndofs);
                  }
                  else
                  {
                     idx = TriPackMatrix<TriangularPart::LOWER>::LowerIndex(row, col);
                  }
                  if (add)
                  {
                     M(idx, e) += val;
                  }
                  else
                  {
                     M(idx, e) = val;
                  }
               }
            }
         }
      }
   });
}

template <int T_D1D, int T_Q1D, int T_COLB = 4, int T_NT = 32>
inline void EAMassAssembleTriangular2D_UpperBlockCols(
   const int NE,
   const Array<real_t> &basis,
   const Vector &padata,
   Vector &eadata,
   const bool add,
   const int,
   const int)
{
   static_assert(T_D1D > 0 && T_Q1D > 0, "");
   // Specialized upper-packed quad mass assembly using block-column
   // sum-factorization with MFEM's packed upper indexing.
   constexpr int D1D = T_D1D;
   constexpr int Q1D = T_Q1D;
   constexpr int COLB = T_COLB;
   constexpr int NT = T_NT;
   constexpr int ND = D1D*D1D;
   constexpr int NQ = Q1D*Q1D;

   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(),
                    TriPackMatrix<TriangularPart::UPPER>::PackedSize(ND), NE);

   mfem::forall_3D_grid(NE, NT, 1, 1, 0, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tid = MFEM_THREAD_ID(x);

      MFEM_SHARED real_t s_B[Q1D][D1D];
      MFEM_SHARED real_t uW[NQ*COLB];
      MFEM_SHARED real_t t1[D1D*Q1D*COLB];

      for (int qb = tid; qb < Q1D*D1D; qb += NT)
      {
         const int q = qb % Q1D;
         const int d = qb / Q1D;
         s_B[q][d] = B(q, d);
      }
      MFEM_SYNC_THREAD;

      for (int j0 = 0; j0 < ND; j0 += COLB)
      {
         const int b = (j0 + COLB <= ND) ? COLB : (ND - j0);
         int j1[COLB], j2[COLB];
         for (int c = 0; c < COLB; ++c)
         {
            if (c < b)
            {
               const int jj = j0 + c;
               j1[c] = jj % D1D;
               j2[c] = jj / D1D;
            }
         }

         for (int q = tid; q < NQ; q += NT)
         {
            const int q1 = q % Q1D;
            const int q2 = q / Q1D;
            const real_t Dq = D(q1, q2, e);

            for (int c = 0; c < b; ++c)
            {
               uW[q + NQ*c] = s_B[q1][j1[c]] * s_B[q2][j2[c]] * Dq;
            }
         }
         MFEM_SYNC_THREAD;

         constexpr int T1S = D1D*Q1D;
         for (int a = tid; a < T1S; a += NT)
         {
            const int i1 = a % D1D;
            const int q2 = a / D1D;

            for (int c = 0; c < b; ++c)
            {
               real_t sum = 0.0;
               for (int q1 = 0; q1 < Q1D; ++q1)
               {
                  const int q = q1 + Q1D*q2;
                  sum += s_B[q1][i1] * uW[q + NQ*c];
               }
               t1[a + T1S*c] = sum;
            }
         }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < b; ++c)
         {
            const int col = j0 + c;
            const int jj1 = j1[c];
            const int jj2 = j2[c];

            for (int i2 = 0; i2 < jj2; ++i2)
            {
               for (int i1 = tid; i1 < D1D; i1 += NT)
               {
                  real_t sum = 0.0;
                  for (int q2 = 0; q2 < Q1D; ++q2)
                  {
                     const int a1 = i1 + D1D*q2;
                     sum += s_B[q2][i2] * t1[a1 + T1S*c];
                  }
                  const int row = i1 + D1D*i2;
                  const int idx =
                     TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, col, ND);
                  if (add) { M(idx, e) += sum; }
                  else { M(idx, e) = sum; }
               }
               MFEM_SYNC_THREAD;
            }

            {
               const int i2 = jj2;
               for (int i1 = tid; i1 <= jj1; i1 += NT)
               {
                  real_t sum = 0.0;
                  for (int q2 = 0; q2 < Q1D; ++q2)
                  {
                     const int a1 = i1 + D1D*q2;
                     sum += s_B[q2][i2] * t1[a1 + T1S*c];
                  }
                  const int row = i1 + D1D*i2;
                  const int idx =
                     TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, col, ND);
                  if (add) { M(idx, e) += sum; }
                  else { M(idx, e) = sum; }
               }
               MFEM_SYNC_THREAD;
            }
         }
      }
   });
}

template <int T_D1D, int T_Q1D, int T_COLB, int T_NT = 32>
inline void EAMassAssembleTriangular3D_UpperBlockCols_Impl(
   const int NE,
   const Array<real_t> &basis,
   const Vector &padata,
   Vector &eadata,
   const bool add,
   const int,
   const int)
{
   static_assert(T_D1D > 0 && T_Q1D > 0, "");
   // Specialized upper-packed hex mass assembly using block-column
   // sum-factorization. This matches the structure of the standalone CUDA
   // kernel, but stores entries with MFEM's packed upper indexing.
   constexpr int D1D = T_D1D;
   constexpr int Q1D = T_Q1D;
   constexpr int COLB = T_COLB;
   constexpr int NT = T_NT;
   constexpr int ND = D1D*D1D*D1D;
   constexpr int NQ = Q1D*Q1D*Q1D;

   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(),
                    TriPackMatrix<TriangularPart::UPPER>::PackedSize(ND), NE);

   mfem::forall_3D_grid(NE, NT, 1, 1, 0, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tid = MFEM_THREAD_ID(x);

      MFEM_SHARED real_t s_B[Q1D][D1D];
      MFEM_SHARED real_t uW[NQ*COLB];
      MFEM_SHARED real_t t1[D1D*Q1D*Q1D*COLB];
      MFEM_SHARED real_t t2[D1D*D1D*Q1D*COLB];

      for (int qb = tid; qb < Q1D*D1D; qb += NT)
      {
         const int q = qb % Q1D;
         const int d = qb / Q1D;
         s_B[q][d] = B(q, d);
      }
      MFEM_SYNC_THREAD;

      for (int j0 = 0; j0 < ND; j0 += COLB)
      {
         const int b = (j0 + COLB <= ND) ? COLB : (ND - j0);
         int j1[COLB], j2[COLB], j3[COLB];
         for (int c = 0; c < COLB; ++c)
         {
            if (c < b)
            {
               const int jj = j0 + c;
               j1[c] = jj % D1D;
               const int tmp = jj / D1D;
               j2[c] = tmp % D1D;
               j3[c] = tmp / D1D;
            }
         }

         for (int q = tid; q < NQ; q += NT)
         {
            const int q1 = q % Q1D;
            const int tmp = q / Q1D;
            const int q2 = tmp % Q1D;
            const int q3 = tmp / Q1D;
            const real_t Dq = D(q1, q2, q3, e);

            for (int c = 0; c < b; ++c)
            {
               uW[q + NQ*c] = s_B[q1][j1[c]] * s_B[q2][j2[c]]
                              * s_B[q3][j3[c]] * Dq;
            }
         }
         MFEM_SYNC_THREAD;

         constexpr int T1S = D1D*Q1D*Q1D;
         for (int a = tid; a < T1S; a += NT)
         {
            const int i1 = a % D1D;
            const int tmp = a / D1D;
            const int q2 = tmp % Q1D;
            const int q3 = tmp / Q1D;

            for (int c = 0; c < b; ++c)
            {
               real_t sum = 0.0;
               for (int q1 = 0; q1 < Q1D; ++q1)
               {
                  const int q = q1 + Q1D*(q2 + Q1D*q3);
                  sum += s_B[q1][i1] * uW[q + NQ*c];
               }
               t1[a + T1S*c] = sum;
            }
         }
         MFEM_SYNC_THREAD;

         constexpr int T2S = D1D*D1D*Q1D;
         for (int a = tid; a < T2S; a += NT)
         {
            const int i1 = a % D1D;
            const int tmp = a / D1D;
            const int i2 = tmp % D1D;
            const int q3 = tmp / D1D;

            for (int c = 0; c < b; ++c)
            {
               real_t sum = 0.0;
               for (int q2 = 0; q2 < Q1D; ++q2)
               {
                  const int a1 = i1 + D1D*(q2 + Q1D*q3);
                  sum += s_B[q2][i2] * t1[a1 + T1S*c];
               }
               t2[a + T2S*c] = sum;
            }
         }
         MFEM_SYNC_THREAD;

         for (int c = 0; c < b; ++c)
         {
            const int col = j0 + c;
            const int jj1 = j1[c];
            const int jj2 = j2[c];
            const int jj3 = j3[c];

            for (int i3 = 0; i3 < jj3; ++i3)
            {
               for (int a = tid; a < D1D*D1D; a += NT)
               {
                  const int i1 = a % D1D;
                  const int i2 = a / D1D;
                  real_t sum = 0.0;
                  for (int q3 = 0; q3 < Q1D; ++q3)
                  {
                     const int a2 = i1 + D1D*(i2 + D1D*q3);
                     sum += s_B[q3][i3] * t2[a2 + T2S*c];
                  }
                  const int row = i1 + D1D*(i2 + D1D*i3);
                  const int idx =
                     TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, col, ND);
                  if (add) { M(idx, e) += sum; }
                  else { M(idx, e) = sum; }
               }
               MFEM_SYNC_THREAD;
            }

            for (int i2 = 0; i2 < jj2; ++i2)
            {
               const int i3 = jj3;
               for (int i1 = tid; i1 < D1D; i1 += NT)
               {
                  real_t sum = 0.0;
                  for (int q3 = 0; q3 < Q1D; ++q3)
                  {
                     const int a2 = i1 + D1D*(i2 + D1D*q3);
                     sum += s_B[q3][i3] * t2[a2 + T2S*c];
                  }
                  const int row = i1 + D1D*(i2 + D1D*i3);
                  const int idx =
                     TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, col, ND);
                  if (add) { M(idx, e) += sum; }
                  else { M(idx, e) = sum; }
               }
               MFEM_SYNC_THREAD;
            }

            {
               const int i3 = jj3;
               const int i2 = jj2;
               for (int i1 = tid; i1 <= jj1; i1 += NT)
               {
                  real_t sum = 0.0;
                  for (int q3 = 0; q3 < Q1D; ++q3)
                  {
                     const int a2 = i1 + D1D*(i2 + D1D*q3);
                     sum += s_B[q3][i3] * t2[a2 + T2S*c];
                  }
                  const int row = i1 + D1D*(i2 + D1D*i3);
                  const int idx =
                     TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, col, ND);
                  if (add) { M(idx, e) += sum; }
                  else { M(idx, e) = sum; }
               }
               MFEM_SYNC_THREAD;
            }
         }
      }
   });
}

template <int T_D1D, int T_Q1D, int T_NT = 32>
inline void EAMassAssembleTriangular3D_UpperBlockCols(
   const int NE,
   const Array<real_t> &basis,
   const Vector &padata,
   Vector &eadata,
   const bool add,
   const int d1d = 0,
   const int q1d = 0)
{
   static_assert(T_D1D > 0 && T_Q1D > 0, "");

   constexpr int D1D = T_D1D;
   constexpr int Q1D = T_Q1D;
   constexpr int NQ = Q1D*Q1D*Q1D;
   constexpr int SharedBytesPerCol =
      sizeof(real_t)*(NQ + D1D*Q1D*Q1D + D1D*D1D*Q1D);
   constexpr int SharedBytesBase = sizeof(real_t)*(Q1D*D1D);
   constexpr int MaxSharedBytes = 48*1024;
   constexpr int COLB =
      (SharedBytesBase + 4*SharedBytesPerCol <= MaxSharedBytes) ? 4 :
      (SharedBytesBase + 2*SharedBytesPerCol <= MaxSharedBytes) ? 2 : 1;

   return EAMassAssembleTriangular3D_UpperBlockCols_Impl<T_D1D, T_Q1D, COLB, T_NT>(
             NE, basis, padata, eadata, add, d1d, q1d);
}

template <bool UPPER, int T_D1D = 0, int T_Q1D = 0>
inline void EAMassAssembleTriangular3D(const int NE,
                                       const Array<real_t> &basis,
                                       const Vector &padata,
                                       Vector &eadata,
                                       const bool add,
                                       const int d1d = 0,
                                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   const int ndofs = D1D*D1D*D1D;
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(),
                    TriPackMatrix<TriangularPart::UPPER>::PackedSize(ndofs), NE);

   if constexpr (UPPER && T_D1D > 0 && T_Q1D > 0)
   {
      // Use the sum-factorized packed-upper path when the tensor dimensions are
      // known at compile time. The generic path below handles lower storage and
      // dynamic-size cases.
      return EAMassAssembleTriangular3D_UpperBlockCols<T_D1D, T_Q1D>(
                NE, basis, padata, eadata, add, d1d, q1d);
   }
   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int DQ = T_D1D * T_Q1D;
      const int ndofs = D1D*D1D*D1D;

      constexpr bool USE_REG = DQ != 0 && DQ <= 12;
      constexpr int MD1r = USE_REG ? MD1 : 1;
      constexpr int MQ1r = USE_REG ? MQ1 : 1;
      constexpr int MD1s = USE_REG ? 1 : MD1;
      constexpr int MQ1s = USE_REG ? 1 : MQ1;

      MFEM_SHARED real_t s_B[MQ1s][MD1s];
      real_t r_B[MQ1r][MD1r];
      real_t (*l_B)[MD1] = nullptr;
      if (USE_REG)
      {
         for (int d = 0; d < D1D; d++)
         {
            for (int q = 0; q < Q1D; q++)
            {
               r_B[q][d] = B(q,d);
            }
         }
         l_B = (real_t (*)[MD1])r_B;
      }
      else
      {
         if (MFEM_THREAD_ID(z) == 0)
         {
            MFEM_FOREACH_THREAD(d,x,D1D)
            {
               MFEM_FOREACH_THREAD(q,y,Q1D)
               {
                  s_B[q][d] = B(q,d);
               }
            }
         }
         l_B = (real_t (*)[MD1])s_B;
      }

      MFEM_SHARED real_t s_D[MQ1][MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            MFEM_FOREACH_THREAD(k3,z,Q1D)
            {
               s_D[k1][k2][k3] = D(k1,k2,k3,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               const int row = i1 + D1D*(i2 + D1D*i3);
               for (int j3 = 0; j3 < D1D; ++j3)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j1 = 0; j1 < D1D; ++j1)
                     {
                        const int col = j1 + D1D*(j2 + D1D*j3);
                        if ((UPPER && row > col) || (!UPPER && row < col))
                        {
                           continue;
                        }
                        real_t val = 0.0;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 val += l_B[k1][i1] * l_B[k1][j1]
                                        * l_B[k2][i2] * l_B[k2][j2]
                                        * l_B[k3][i3] * l_B[k3][j3]
                                        * s_D[k1][k2][k3];
                              }
                           }
                        }
                        int idx = 0;
                        if constexpr (UPPER)
                        {
                           idx = TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, col, ndofs);
                        }
                        else
                        {
                           idx = TriPackMatrix<TriangularPart::LOWER>::LowerIndex(row, col);
                        }
                        if (add)
                        {
                           M(idx, e) += val;
                        }
                        else
                        {
                           M(idx, e) = val;
                        }
                     }
                  }
               }
            }
         }
      }
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
   if constexpr (DIM == 1) { return internal::PAMassApply1D; }
   else if constexpr (DIM == 2) { return internal::SmemPAMassApply2D<T_D1D,T_Q1D>; }
   else if constexpr (DIM == 3) { return internal::SmemPAMassApply3D<T_D1D, T_Q1D>; }
   MFEM_ABORT("");
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
   if constexpr (DIM == 1) { return internal::PAMassAssembleDiagonal1D; }
   else if constexpr (DIM == 2) { return internal::SmemPAMassAssembleDiagonal2D<T_D1D,T_Q1D>; }
   else if constexpr (DIM == 3) { return internal::SmemPAMassAssembleDiagonal3D<T_D1D, T_Q1D>; }
   MFEM_ABORT("");
}

inline DiagonalKernelType MassIntegrator::DiagonalPAKernels::Fallback(
   int DIM, int, int)
{
   if (DIM == 1) { return internal::PAMassAssembleDiagonal1D; }
   else if (DIM == 2) { return internal::PAMassAssembleDiagonal2D; }
   else if (DIM == 3) { return internal::PAMassAssembleDiagonal3D; }
   else { MFEM_ABORT(""); }
}
/// \endcond DO_NOT_DOCUMENT
} // namespace mfem

#endif
