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

#include "../../config/config.hpp"

#define MFEM_DEBUG_COLOR 50
#include "../../general/debug.hpp"

#include "../../general/array.hpp"
#include "../../general/backends.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"

namespace mfem
{

namespace internal
{

template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
void DynaSmemPADiffApply2DKern(const int NE,
                               const bool symmetric,
                               const Array<double> &b_,
                               const Array<double> &g_,
                               const Vector &d_,
                               const Vector &x_,
                               Vector &y_,
                               const int d1d = 0,
                               const int q1d = 0,
                               const int nbz = 0)
{
   const int d = T_D1D ? T_D1D : d1d;
   const int q = T_Q1D ? T_Q1D : q1d;
   const int z = T_NBZ ? T_NBZ : nbz;

   const auto b = Reshape(b_.Read(), q, d);
   const auto g = Reshape(g_.Read(), q, d);
   const auto D = Reshape(d_.Read(), q*q, symmetric ? 3 : 4, NE);
   const auto x = Reshape(x_.Read(), d, d, NE);

   auto Y = Reshape(y_.ReadWrite(), d, d, NE);

   const size_t smem_size = 2*q*d + z*(d*d + 2*d*q+ 2*q*q);

   mfem::forall_2D_batch(NE, q,q,z, smem_size,
                         [=] MFEM_HOST_DEVICE(int e, double *sm)
   {
      const int tz = MFEM_THREAD_ID(z);

      constexpr int QD = T_D1D*T_Q1D;
      auto sB = GetSmem<QD>(sm, q*d), sG = GetSmem<QD>(sm, q*d);
      DeviceMatrix B(sB, q,d), Bt(sB, d,q);
      DeviceMatrix G(sG, q,d), Gt(sG, d,q);

      auto sX = GetSmem<T_D1D*T_D1D*T_NBZ>(sm, d*d*z);
      auto sDQ = GetSmem<2*T_D1D*T_Q1D*T_NBZ>(sm, 2*d*q*z);
      auto sQQ = GetSmem<2*T_Q1D*T_Q1D*T_NBZ>(sm, 2*q*q*z);

      DeviceTensor<3> X(sX, d,d,z);
      DeviceTensor<4> DQ(sDQ, d,q,z,2), QQ(sQQ, q,q,z,2);

      MFEM_FOREACH_THREAD(dy,y,d)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            X(dx,dy,tz) = x(dx,dy,e);
         }
      }
      if (tz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               B(qx,dy) = b(qx,dy);
               G(qx,dy) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,d)
      {
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dx = 0; dx < d; ++dx)
            {
               const double coords = X(dx,dy,tz);
               u += B(qx,dx) * coords;
               v += G(qx,dx) * coords;
            }
            DQ(dy,qx,tz,0) = u;
            DQ(dy,qx,tz,1) = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,q)
      {
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dy = 0; dy < d; ++dy)
            {
               u += DQ(dy,qx,tz,1) * B(qy,dy);
               v += DQ(dy,qx,tz,0) * G(qy,dy);
            }
            QQ(qy,qx,tz,0) = u;
            QQ(qy,qx,tz,1) = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,q)
      {
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            const int f = (qx + ((qy) * q));
            const double O11 = D(f,0,e);
            const double O21 = D(f,1,e);
            const double O12 = symmetric ? O21 : D(f,2,e);
            const double O22 = symmetric ? D(f,2,e) : D(f,3,e);
            const double gX = QQ(qy,qx,tz,0);
            const double gY = QQ(qy,qx,tz,1);
            QQ(qy,qx,tz,0) = (O11 * gX) + (O12 * gY);
            QQ(qy,qx,tz,1) = (O21 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      if (tz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               Bt(dy,qx) = b(qx,dy);
               Gt(dy,qx) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,q)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qx = 0; qx < q; ++qx)
            {
               u += Gt(dx,qx) * QQ(qy,qx,tz,0);
               v += Bt(dx,qx) * QQ(qy,qx,tz,1);
            }
            DQ(dx,qy,tz,0) = u;
            DQ(dx,qy,tz,1) = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,d)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qy = 0; qy < q; ++qy)
            {
               u += DQ(dx,qy,tz,0) * Bt(dy,qy);
               v += DQ(dx,qy,tz,1) * Gt(dy,qy);
            }
            Y(dx,dy,e) += (u + v);
         }
      }
   });
}

void DynamicSmemPADiffusionApply2D(const int NE,
                                   const bool symm,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Vector &D,
                                   const Vector &X,
                                   Vector &Y,
                                   const int d,
                                   const int q,
                                   const int z)
{
   const int id = (d << 4) | q;

   static int cid = 0;
   if (cid != id) { dbg("NE:%d D1D:%d Q1D:%d",NE,d,q); cid = id; }

   switch (id)
   {
      case 0x22: return DynaSmemPADiffApply2DKern<2,2,16>(NE,symm,B,G,D,X,Y);
      case 0x23: return DynaSmemPADiffApply2DKern<2,3,16>(NE,symm,B,G,D,X,Y);
      case 0x33: return DynaSmemPADiffApply2DKern<3,3,16>(NE,symm,B,G,D,X,Y);
      case 0x34: return DynaSmemPADiffApply2DKern<3,4,8>(NE,symm,B,G,D,X,Y);
      case 0x44: return DynaSmemPADiffApply2DKern<4,4,8>(NE,symm,B,G,D,X,Y);
      case 0x45: return DynaSmemPADiffApply2DKern<4,5,4>(NE,symm,B,G,D,X,Y);
      //case 0x55: return DynaSmemPADiffApply2DKern<5,5,8>(NE,symm,B,G,D,X,Y);
      //case 0x66: return DynaSmemPADiffApply2DKern<6,6,4>(NE,symm,B,G,D,X,Y);
      //case 0x77: return DynaSmemPADiffApply2DKern<7,7,4>(NE,symm,B,G,D,X,Y);
      //case 0x88: return DynaSmemPADiffApply2DKern<8,8,2>(NE,symm,B,G,D,X,Y);
      //case 0x99: return DynaSmemPADiffApply2DKern<9,9,2>(NE,symm,B,G,D,X,Y);
      default: return DynaSmemPADiffApply2DKern(NE,symm,B,G,D,X,Y,d,q,z);
   }
   MFEM_ABORT("Unknown kernel: 0x"<<std::hex << id << std::dec);
}

template<int T_D1D = 0, int T_Q1D = 0>
void DynaSmemPADiffApply3DKern(const int NE,
                               const bool symmetric,
                               const Array<double> &b_,
                               const Array<double> &g_,
                               const Vector &d_,
                               const Vector &x_,
                               Vector &y_,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int d = T_D1D ? T_D1D : d1d;
   const int q = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), q, d);
   const auto g = Reshape(g_.Read(), q, d);
   const auto D = Reshape(d_.Read(), q, q, q, symmetric ? 6 : 9, NE);
   const auto x = Reshape(x_.Read(), d, d, d, NE);

   auto y = Reshape(y_.ReadWrite(), d, d, d, NE);

   const size_t smem_size = 2*q*d + 6*q*q*q;

   mfem::forall_3D(NE, q,q,q, smem_size, [=] MFEM_HOST_DEVICE (int e, double *sm)
   {
      constexpr int QD = T_D1D*T_Q1D;
      auto sB = GetSmem<QD>(sm, q*d), sG = GetSmem<QD>(sm, q*d);
      DeviceMatrix B(sB, q,d), Bt(sB, d,q);
      DeviceMatrix G(sG, q,d), Gt(sG, d,q);

      constexpr int QQQ = T_Q1D*T_Q1D*T_Q1D;
      auto sm0 = GetSmem<QQQ>(sm, q*q*q);
      auto sm1 = GetSmem<QQQ>(sm, q*q*q);
      auto sm2 = GetSmem<QQQ>(sm, q*q*q);
      auto sm3 = GetSmem<QQQ>(sm, q*q*q);
      auto sm4 = GetSmem<QQQ>(sm, q*q*q);
      auto sm5 = GetSmem<QQQ>(sm, q*q*q);

      DeviceCube X(sm0, d,d,d), qqq0(sm0, q,q,q), ddq0(sm0, d,d,q);
      DeviceCube qdd0(sm1, q,d,d), qqq1(sm1, q,q,q), ddq1(sm1, d,d,q);
      DeviceCube qdd1(sm2, q,d,d), qqq2(sm2, q,q,q), ddq2(sm2, d,d,q);
      DeviceCube qqd0(sm3, q,q,d), dqq0(sm3, d,q,q);
      DeviceCube qqd1(sm4, q,q,d), dqq1(sm4, d,q,q);
      DeviceCube qqd2(sm5, q,q,d), dqq2(sm5, d,q,q);

      MFEM_FOREACH_THREAD(dz,z,d)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               X(dz,dy,dx) = x(dx,dy,dz,e);
            }
         }
      }
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               B(qx,dy) = b(qx,dy);
               G(qx,dy) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               double u = 0.0, v = 0.0;
               MFEM_UNROLL_DEV_DISABLED
               for (int dx = 0; dx < d; ++dx)
               {
                  const double coords = X(dz,dy,dx);
                  u += coords * B(qx,dx);
                  v += coords * G(qx,dx);
               }
               qdd0(qx,dy,dz) = u;
               qdd1(qx,dy,dz) = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d)
      {
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DEV_DISABLED
               for (int dy = 0; dy < d; ++dy)
               {
                  u += qdd1(qx,dy,dz) * B(qy,dy);
                  v += qdd0(qx,dy,dz) * G(qy,dy);
                  w += qdd0(qx,dy,dz) * B(qy,dy);
               }
               qqd0(qx,qy,dz) = u;
               qqd1(qx,qy,dz) = v;
               qqd2(qx,qy,dz) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q)
      {
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DEV_DISABLED
               for (int dz = 0; dz < d; ++dz)
               {
                  u += qqd0(qx,qy,dz) * B(qz,dz);
                  v += qqd1(qx,qy,dz) * B(qz,dz);
                  w += qqd2(qx,qy,dz) * G(qz,dz);
               }
               const double O11 = D(qx,qy,qz,0,e);
               const double O12 = D(qx,qy,qz,1,e);
               const double O13 = D(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : D(qx,qy,qz,3,e);
               const double O22 = symmetric ? D(qx,qy,qz,3,e) : D(qx,qy,qz,4,e);
               const double O23 = symmetric ? D(qx,qy,qz,4,e) : D(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : D(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : D(qx,qy,qz,7,e);
               const double O33 = symmetric ? D(qx,qy,qz,5,e) : D(qx,qy,qz,8,e);
               const double gX = u;
               const double gY = v;
               const double gZ = w;
               qqq0(qx,qy,qz) = (O11*gX) + (O12*gY) + (O13*gZ);
               qqq1(qx,qy,qz) = (O21*gX) + (O22*gY) + (O23*gZ);
               qqq2(qx,qy,qz) = (O31*gX) + (O32*gY) + (O33*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               Bt(dy,qx) = b(qx,dy);
               Gt(dy,qx) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q)
      {
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DEV_DISABLED
               for (int qx = 0; qx < q; ++qx)
               {
                  u += qqq0(qx,qy,qz) * Gt(dx,qx);
                  v += qqq1(qx,qy,qz) * Bt(dx,qx);
                  w += qqq2(qx,qy,qz) * Bt(dx,qx);
               }
               dqq0(dx,qy,qz) = u;
               dqq1(dx,qy,qz) = v;
               dqq2(dx,qy,qz) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DEV_DISABLED
               for (int qy = 0; qy < q; ++qy)
               {
                  u += dqq0(dx,qy,qz) * Bt(dy,qy);
                  v += dqq1(dx,qy,qz) * Gt(dy,qy);
                  w += dqq2(dx,qy,qz) * Bt(dy,qy);
               }
               ddq0(dx,dy,qz) = u;
               ddq1(dx,dy,qz) = v;
               ddq2(dx,dy,qz) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DEV_DISABLED
               for (int qz = 0; qz < q; ++qz)
               {
                  u += ddq0(dx,dy,qz) * Bt(dz,qz);
                  v += ddq1(dx,dy,qz) * Bt(dz,qz);
                  w += ddq2(dx,dy,qz) * Gt(dz,qz);
               }
               y(dx,dy,dz,e) += (u + v + w);
            }
         }
      }
   });
}

void DynamicSmemPADiffusionApply3D(const int NE,
                                   const bool symm,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Vector &D,
                                   const Vector &X,
                                   Vector &Y,
                                   const int d,
                                   const int q)
{
   const int id = (d << 4) | q;

   //static int cid = 0;
   //if (cid != id) { dbg("NE:%d D1D:%d Q1D:%d",NE,d,q); cid = id; }

   // dynamic : SYCL, no instantiated kernels, smem_size to declare
   //  -5% [GPU] dynamic
   //   == [GPU] dynamic + JIT|switch
   //   == [CPU] dynamic + JIT|switch p>=3
   // -25% [CPU] slowdown if full dynamic p>=3

   // layout CPU good static/2x_lo,20%_ho dynamic, GPU better static / good dynamic ∀p

   switch (id)
   {
      case 0x22: return DynaSmemPADiffApply3DKern<2,2>(NE,symm,B,G,D,X,Y);
      case 0x23: return DynaSmemPADiffApply3DKern<2,3>(NE,symm,B,G,D,X,Y);
      case 0x33: return DynaSmemPADiffApply3DKern<3,3>(NE,symm,B,G,D,X,Y);
      case 0x34: return DynaSmemPADiffApply3DKern<3,4>(NE,symm,B,G,D,X,Y);
      case 0x44: return DynaSmemPADiffApply3DKern<4,4>(NE,symm,B,G,D,X,Y);
      case 0x45: return DynaSmemPADiffApply3DKern<4,5>(NE,symm,B,G,D,X,Y);
      case 0x56: return DynaSmemPADiffApply3DKern<5,6>(NE,symm,B,G,D,X,Y);
      case 0x67: return DynaSmemPADiffApply3DKern<6,7>(NE,symm,B,G,D,X,Y);
      case 0x78: return DynaSmemPADiffApply3DKern<7,8>(NE,symm,B,G,D,X,Y);
      default: return DynaSmemPADiffApply3DKern(NE,symm,B,G,D,X,Y,d,q);
   }
   MFEM_ABORT("Unknown kernel: 0x"<<std::hex << id << std::dec);
}

} // namespace internal

} // namespace mfem

