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


#define MFEM_DEBUG_COLOR 51
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

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
template<typename T, std::size_t UID>
MFEM_DEVICE inline T& StaticSharedMemoryVariable()
{
   MFEM_SHARED uint8_t smem alignas(alignof(T))[sizeof(T)];
   return *(reinterpret_cast<T*>(smem));
}
#define MFEM_STATIC_SHARED_VAR(var, ...) \
__VA_ARGS__& var = StaticSharedMemoryVariable<__VA_ARGS__, __COUNTER__>()
#else
#define MFEM_STATIC_SHARED_VAR(var, ...) __VA_ARGS__ var;
#endif

/**
 * @brief DynamicSmemPADiffusionApply2D
 * @param NE
 * @param symmetric
 * @param b_
 * @param g_
 * @param d_
 * @param x_
 * @param y_
 * @param D1D
 * @param Q1D
 * @param NBZ
 */
template<int D1D, int Q1D, int NBZ>
void StaticSmemPADiffusionApply2DKernel(const int NE,
                                        const bool symmetric,
                                        const Array<double> &b_,
                                        const Array<double> &g_,
                                        const Vector &d_,
                                        const Vector &x_,
                                        Vector &y_)
{
   dbg();
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   const auto x = Reshape(x_.Read(), D1D, D1D, NE);

   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE(int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      MFEM_STATIC_SHARED_VAR(tB, TMatrix<Q1D,D1D>); // operator(.,.)

      MFEM_STATIC_SHARED_VAR(B, tensor<double,Q1D,D1D>); // operator[]
      MFEM_STATIC_SHARED_VAR(G, tensor<double,Q1D,D1D>);

      MFEM_STATIC_SHARED_VAR(Bt, tensor<double,D1D,Q1D>);
      MFEM_STATIC_SHARED_VAR(Gt, tensor<double,D1D,Q1D>);

      MFEM_STATIC_SHARED_VAR(Xz, tensor<double,NBZ,D1D,D1D>);
      auto &X = Xz[tidz];

      MFEM_STATIC_SHARED_VAR(DQ, tensor<double,2,NBZ,D1D,Q1D>);
      auto &DQ0 = DQ[0][tidz], &DQ1 = DQ[1][tidz];

      MFEM_STATIC_SHARED_VAR(QQ, tensor<double,2,NBZ,Q1D,Q1D>);
      auto &QQ0 = QQ[0][tidz], &QQ1 = QQ[1][tidz];

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
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               B[qx][dy] = b(qx,dy);
               tB(qx,dy) = b(qx,dy);
               G[qx][dy] = g(qx,dy);
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
               //u += B[qx][dx] * coords;
               u += tB(qx,dx) * coords;
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
               //u += DQ1[dy][qx] * B[qy][dy];
               u += DQ1[dy][qx] * B(qy,dy);
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
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               Bt[dy][qx] = b(qx,dy);
               Gt[dy][qx] = g(qx,dy);
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
            DQ0[dx][qy] = u;
            DQ1[dx][qy] = v;
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
               u += DQ0[dx][qy] * Bt[dy][qy];
               v += DQ1[dx][qy] * Gt[dy][qy];
            }
            Y(dx,dy,e) += (u + v);
         }
      }
   });
}

void StaticSmemPADiffusionApply2D(const int NE,
                                  const bool symm,
                                  const Array<double> &B,
                                  const Array<double> &G,
                                  const Vector &D,
                                  const Vector &X,
                                  Vector &Y,
                                  const int D1D,
                                  const int Q1D,
                                  const int NBZ)
{
   dbg("NE:%d D1D:%d Q1D:%d",NE,D1D,Q1D);
   const int id = (D1D << 4) | Q1D;

   MFEM_VERIFY(id == 0x33 && NBZ == 16, "Wrong kernel!");
   StaticSmemPADiffusionApply2DKernel<3,3,16>(NE,symm,B,G,D,X,Y);
}

// Shared memory PA Diffusion Apply 3D kernel
template<int D1D = 0, int Q1D = 0>
void StaticSmemPADiffusionApply3DKernel(const int NE,
                                        const bool symmetric,
                                        const Array<double> &b_,
                                        const Array<double> &g_,
                                        const Vector &d_,
                                        const Vector &x_,
                                        Vector &y_)
{
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   const auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);

   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_STATIC_SHARED_VAR(B, tensor<double,Q1D,D1D>);
      MFEM_STATIC_SHARED_VAR(G, tensor<double,Q1D,D1D>);
      MFEM_STATIC_SHARED_VAR(Bt, tensor<double,D1D,Q1D>);
      MFEM_STATIC_SHARED_VAR(Gt, tensor<double,D1D,Q1D>);

      MFEM_STATIC_SHARED_VAR(sm0, tensor<double,3,Q1D,Q1D,Q1D>);
      MFEM_STATIC_SHARED_VAR(sm1, tensor<double,3,Q1D,Q1D,Q1D>);

      auto &X = sm0[2], &DDQ0 = sm0[0], &DDQ1 = sm0[1];
      auto &DQQ0 = sm1[0], &DQQ1 = sm1[1], &DQQ2 = sm1[2];
      auto &QQQ0 = sm0[0], &QQQ1 = sm0[1], &QQQ2 = sm0[2];
      auto &QQD0 = sm1[0], &QQD1 = sm1[1], &QQD2 = sm1[2];
      auto &QDD0 = sm0[0], &QDD1 = sm0[1], &QDD2 = sm0[2];

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
               MFEM_UNROLL(D1D)
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
               MFEM_UNROLL(D1D)
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
               MFEM_UNROLL(D1D)
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
               MFEM_UNROLL(Q1D)
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
               MFEM_UNROLL(Q1D)
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

void StaticSmemPADiffusionApply3D(const int NE,
                                  const bool symm,
                                  const Array<double> &B,
                                  const Array<double> &G,
                                  const Vector &D,
                                  const Vector &X,
                                  Vector &Y,
                                  const int D1D,
                                  const int Q1D)
{
   static int first = 0;
   if (first++ == 0) { dbg("NE:%d D1D:%d Q1D:%d",NE,D1D,Q1D); }

   const int id = (D1D << 4) | Q1D;

   switch (id)
   {
      //case 0x22: return StaticSmemPADiffusionApply3DKernel<2,2>(NE,symm,B,G,D,X,Y);
      case 0x23: return StaticSmemPADiffusionApply3DKernel<2,3>(NE,symm,B,G,D,X,Y);
      case 0x34: return StaticSmemPADiffusionApply3DKernel<3,4>(NE,symm,B,G,D,X,Y);
      case 0x45: return StaticSmemPADiffusionApply3DKernel<4,5>(NE,symm,B,G,D,X,Y);
      //case 0x46: return StaticSmemPADiffusionApply3DKernel<4,6>(NE,symm,B,G,D,X,Y);
      case 0x56: return StaticSmemPADiffusionApply3DKernel<5,6>(NE,symm,B,G,D,X,Y);
      //case 0x58: return StaticSmemPADiffusionApply3DKernel<5,8>(NE,symm,B,G,D,X,Y);
      case 0x67: return StaticSmemPADiffusionApply3DKernel<6,7>(NE,symm,B,G,D,X,Y);
      case 0x78: return StaticSmemPADiffusionApply3DKernel<7,8>(NE,symm,B,G,D,X,Y);
         //case 0x89: return StaticSmemPADiffusionApply3DKernel<8,9>(NE,symm,B,G,D,X,Y);
   }
   MFEM_ABORT("Unknown kernel: 0x"<<std::hex << id << std::dec);
}

} // namespace internal

} // namespace mfem
