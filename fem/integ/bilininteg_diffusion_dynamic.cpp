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
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/tensor.hpp"
#include "../../linalg/ttensor.hpp"
#include "../../linalg/vector.hpp"
#include <cassert>

namespace mfem
{

namespace internal
{

#undef MFEM_SHARED_USE_CHAR
#define MFEM_SHARED_EXTRA_LOAD 0 // 64*1024


void DynamicSmemPADiffusionApply2DKernel(const int NE,
                                         const bool symmetric,
                                         const Array<double> &b_,
                                         const Array<double> &g_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_,
                                         const int d,
                                         const int q,
                                         const int z)
{
   const auto b = Reshape(b_.Read(), q, d);
   const auto g = Reshape(g_.Read(), q, d);
   const auto D = Reshape(d_.Read(), q*q, symmetric ? 3 : 4, NE);
   const auto x = Reshape(x_.Read(), d, d, NE);

   auto Y = Reshape(y_.ReadWrite(), d, d, NE);

   const size_t smem_size = (MFEM_SHARED_EXTRA_LOAD + 4*q*d + z*
                             (d*d + 2*d*q+ 2*q*q));
   //dbg("smem_size:%d", smem_size);

#ifdef MFEM_SHARED_USE_CHAR
   const size_t smem_size_char = sizeof(double) * smem_size;
   dbg("smem_size_char:%d", smem_size_char);
   mfem::internal::forall_2D_batch_smem<char>(NE, q,q,z, smem_size_char,
                                              [=] MFEM_HOST_DEVICE(int e, char *sm)
#else
   mfem::forall_2D_batch_smem(NE, q,q,z, smem_size,
                              [=] MFEM_HOST_DEVICE(int e, double *sm)
#endif
   {
      const int tz = MFEM_THREAD_ID(z);
      const decltype(sm) base = sm;

      mdsmem<3> X(sm, d,d,z);

      mdsmem<2> B(sm, q,d), G(sm, q,d);

      //assert(q==3 && d==3); // if using tensor, TMatrix for shared variables
      //MFEM_STATIC_SHARED_VAR(B, tensor<double,3,3>); // q==3, d==3 !!
      //MFEM_STATIC_SHARED_VAR(G, tensor<double,3,3>); // q==3, d==3 !!

      //MFEM_DYNAMIC_SHARED_VAR(B, sm, TMatrix<3,3>); // q==3, d==3 !!
      //MFEM_DYNAMIC_SHARED_VAR(B, sm, tensor<double,3,3>); // q==3, d==3 !!

      mdsmem<2> Bt(sm, d,q), Gt(sm, d,q);
      mdsmem<4> DQ(sm, d,q,z,2), QQ(sm, q,q,z,2);

      mdsmem<1> Extra(sm, MFEM_SHARED_EXTRA_LOAD);

      // can be less if there are some static shared
      assert(sm <= base + smem_size*sizeof(double)/sizeof(*sm));

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

   DynamicSmemPADiffusionApply2DKernel(NE,symm,B,G,D,X,Y,d,q,z);
}

template<int T_D1D = 0, int T_Q1D = 0>
void DynamicSmemPADiffusionApply3DKernel(const int NE,
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

   const size_t smem_size = 4*q*d + // B,Bt,G,Gt
                            3*d*d*q + // DDQs
                            3*d*q*q + // DQQs
                            3*q*q*q + // QQQs
                            MFEM_SHARED_EXTRA_LOAD;

#ifdef MFEM_SHARED_USE_CHAR
   const size_t smem_size_char = sizeof(double) * smem_size;
   mfem::internal::forall_3D_smem<char>(NE, q,q,q, smem_size_char,
                                        [=] MFEM_HOST_DEVICE(int e, char *sm)
#else
   forall_3D_smem(NE, q,q,q, smem_size, [=] MFEM_HOST_DEVICE (int e, double *sm)
#endif
   {
      //const decltype(sm) base = sm;

      mdsmem<2> B(sm, q,d), Bt(sm, d,q);
      mdsmem<2> G(sm, q,d), Gt(sm, d,q);

      mdsmem<3> DDQ0(sm, d,d,q), DDQ1(sm, d,d,q), DDQ2(sm, d,d,q);
      mdview<3> QDD0(DDQ0, q,d,d), QDD1(DDQ1, q,d,d), QDD2(DDQ2, q,d,d);
      mdview<3> X(DDQ2, d,d,d);

      mdsmem<3> QQD0(sm,   q,q,d), QQD1(sm,   q,q,d), QQD2(sm,   q,q,d);
      mdview<3> DQQ0(QQD0, d,q,q), DQQ1(QQD1, d,q,q), DQQ2(QQD2, d,q,q);

      mdsmem<3> QQQ0(sm, q,q,q), QQQ1(sm, q,q,q), QQQ2(sm, q,q,q);

      mdsmem<1> Extra(sm, MFEM_SHARED_EXTRA_LOAD);

      //assert(sm == base + smem_size*sizeof(double)/sizeof(*sm));

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
               MFEM_UNROLL_DISABLED
               for (int dx = 0; dx < d; ++dx)
               {
                  const double coords = X(dz,dy,dx);
                  u += coords * B(qx,dx);
                  v += coords * G(qx,dx);
               }
               DDQ0(dz,dy,qx) = u;
               DDQ1(dz,dy,qx) = v;
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
               MFEM_UNROLL_DISABLED
               for (int dy = 0; dy < d; ++dy)
               {
                  u += DDQ1(dz,dy,qx) * B(qy,dy);
                  v += DDQ0(dz,dy,qx) * G(qy,dy);
                  w += DDQ0(dz,dy,qx) * B(qy,dy);
               }
               DQQ0(dz,qy,qx) = u;
               DQQ1(dz,qy,qx) = v;
               DQQ2(dz,qy,qx) = w;
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
               MFEM_UNROLL_DISABLED
               for (int dz = 0; dz < d; ++dz)
               {
                  u += DQQ0(dz,qy,qx) * B(qz,dz);
                  v += DQQ1(dz,qy,qx) * B(qz,dz);
                  w += DQQ2(dz,qy,qx) * G(qz,dz);
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
               QQQ0(qz,qy,qx) = (O11*gX) + (O12*gY) + (O13*gZ);
               QQQ1(qz,qy,qx) = (O21*gX) + (O22*gY) + (O23*gZ);
               QQQ2(qz,qy,qx) = (O31*gX) + (O32*gY) + (O33*gZ);
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
               MFEM_UNROLL_DISABLED
               for (int qx = 0; qx < q; ++qx)
               {
                  u += QQQ0(qz,qy,qx) * Gt(dx,qx);
                  v += QQQ1(qz,qy,qx) * Bt(dx,qx);
                  w += QQQ2(qz,qy,qx) * Bt(dx,qx);
               }
               QQD0(qz,qy,dx) = u;
               QQD1(qz,qy,dx) = v;
               QQD2(qz,qy,dx) = w;
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
               MFEM_UNROLL_DISABLED
               for (int qy = 0; qy < q; ++qy)
               {
                  u += QQD0(qz,qy,dx) * Bt(dy,qy);
                  v += QQD1(qz,qy,dx) * Gt(dy,qy);
                  w += QQD2(qz,qy,dx) * Bt(dy,qy);
               }
               QDD0(qz,dy,dx) = u;
               QDD1(qz,dy,dx) = v;
               QDD2(qz,dy,dx) = w;
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
               MFEM_UNROLL_DISABLED
               for (int qz = 0; qz < q; ++qz)
               {
                  u += QDD0(qz,dy,dx) * Bt(dz,qz);
                  v += QDD1(qz,dy,dx) * Bt(dz,qz);
                  w += QDD2(qz,dy,dx) * Gt(dz,qz);
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

   static int cid = 0;
   if (cid != id) { dbg("NE:%d D1D:%d Q1D:%d",NE,d,q); cid = id; }

   switch (id)
   {
         // dynamic : SYCL, no instantiated kernels, smem_size to declare, +new smem tensors

         //     5 % [GPU] slowdown if full dynamic

         // 20~30 % [CPU] slowdown dynamic + JIT/switch

         //      2x [CPU] slowdown if full dynamic
#if 0
      case 0x23: return DynamicSmemPADiffusionApply3DKernel<2,3>(NE,symm,B,G,D,X,Y);
      case 0x34: return DynamicSmemPADiffusionApply3DKernel<3,4>(NE,symm,B,G,D,X,Y);
      case 0x45: return DynamicSmemPADiffusionApply3DKernel<4,5>(NE,symm,B,G,D,X,Y);
      case 0x56: return DynamicSmemPADiffusionApply3DKernel<5,6>(NE,symm,B,G,D,X,Y);
      case 0x67: return DynamicSmemPADiffusionApply3DKernel<6,7>(NE,symm,B,G,D,X,Y);
      case 0x78: return DynamicSmemPADiffusionApply3DKernel<7,8>(NE,symm,B,G,D,X,Y);
#endif
      default: return DynamicSmemPADiffusionApply3DKernel(NE,symm,B,G,D,X,Y,d,q);
   }
   MFEM_ABORT("Unknown kernel: 0x"<<std::hex << id << std::dec);
}

} // namespace internal

} // namespace mfem

