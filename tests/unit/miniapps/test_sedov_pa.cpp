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

#include "test_sedov.hpp"

#include <cstring>
#include <unordered_map>
#include "linalg/kernels.hpp"

#if defined(MFEM_SEDOV_PA_MPI) && !defined(MFEM_USE_MPI)
#error "Cannot use MFEM_SEDOV_PA_MPI without MFEM_USE_MPI!"
#endif

using namespace mfem;

namespace mfem
{

struct QData
{
   Vector rho0DetJ0w;
   DenseTensor Jac0inv, stressJinvT;
   QData(int dim, int ne, int nq):
      rho0DetJ0w(ne * nq),
      Jac0inv(dim, dim, ne * nq),
      stressJinvT(ne * nq, dim, dim) { }
};

template<int DIM, int D1D, int Q1D, int L1D, int H1D, int NBZ =1> static
void kSmemForceMult2D(const int NE,
                      const Array<real_t> &B_,
                      const Array<real_t> &Bt_,
                      const Array<real_t> &Gt_,
                      const DenseTensor &sJit_,
                      const Vector &e_,
                      Vector &v_)
{
   constexpr real_t eps1 = std::numeric_limits<real_t>::epsilon();
   constexpr real_t eps2 = eps1*eps1;

   const auto b = Reshape(B_.Read(), Q1D, L1D);
   const auto bt = Reshape(Bt_.Read(), H1D, Q1D);
   const auto gt = Reshape(Gt_.Read(), H1D, Q1D);
   const auto sJit = Reshape(sJit_.Read(), Q1D, Q1D, NE, 2, 2);
   const auto energy = Reshape(e_.Read(), L1D, L1D, NE);

   auto velocity = Reshape(v_.Write(), D1D,D1D,2,NE);

   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED real_t B[Q1D][L1D];
      MFEM_SHARED real_t Bt[H1D][Q1D];
      MFEM_SHARED real_t Gt[H1D][Q1D];
      MFEM_SHARED real_t Ez[NBZ][L1D][L1D];
      auto E = (real_t (*)[L1D])(Ez + z);
      MFEM_SHARED real_t LQz[2][NBZ][H1D][Q1D];
      auto LQ0 = (real_t (*)[Q1D])(LQz[0] + z);
      auto LQ1 = (real_t (*)[Q1D])(LQz[1] + z);
      MFEM_SHARED real_t QQz[3][NBZ][Q1D][Q1D];
      auto QQ = (real_t (*)[Q1D])(QQz[0] + z);
      auto QQ0 = (real_t (*)[Q1D])(QQz[1] + z);
      auto QQ1 = (real_t (*)[Q1D])(QQz[2] + z);
      if (z == 0)
      {
         MFEM_FOREACH_THREAD_DIRECT(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(l,y,Q1D)
            {
               if (l < L1D) { B[q][l] = b(q,l); }
               if (l < H1D) { Bt[l][q] = bt(l,q); }
               if (l < H1D) { Gt[l][q] = gt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lx,x,L1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(ly,y,L1D)
         {
            E[lx][ly] = energy(lx,ly,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
         {
            real_t u = 0.0;
            for (int lx = 0; lx < L1D; ++lx)
            {
               u += B[qx][lx] * E[lx][ly];
            }
            LQ0[ly][qx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
         {
            real_t u = 0.0;
            for (int ly = 0; ly < L1D; ++ly)
            {
               u += B[qy][ly] * LQ0[ly][qx];
            }
            QQ[qy][qx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               const real_t esx = QQ[qy][qx] * sJit(qx,qy,e,0,c);
               const real_t esy = QQ[qy][qx] * sJit(qx,qy,e,1,c);
               QQ0[qy][qx] = esx;
               QQ1[qy][qx] = esy;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx,x,H1D)
            {
               real_t u = 0.0;
               real_t v = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += Gt[dx][qx] * QQ0[qy][qx];
                  v += Bt[dx][qx] * QQ1[qy][qx];
               }
               LQ0[dx][qy] = u;
               LQ1[dx][qy] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(dy,y,H1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx,x,H1D)
            {
               real_t u = 0.0;
               real_t v = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += LQ0[dx][qy] * Bt[dy][qy];
                  v += LQ1[dx][qy] * Gt[dy][qy];
               }
               velocity(dx,dy,c,e) = u + v;
            }
         }
         MFEM_SYNC_THREAD;
      }
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(dy,y,H1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx,x,H1D)
            {
               const real_t v = velocity(dx,dy,c,e);
               if (fabs(v) < eps2)
               {
                  velocity(dx,dy,c,e) = 0.0;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int DIM, int D1D, int Q1D, int L1D, int H1D> static
void kSmemForceMult3D(const int NE,
                      const Array<real_t> &B_,
                      const Array<real_t> &Bt_,
                      const Array<real_t> &Gt_,
                      const DenseTensor &sJit_,
                      const Vector &e_,
                      Vector &v_)
{
   constexpr real_t eps1 = std::numeric_limits<real_t>::epsilon();
   constexpr real_t eps2 = eps1*eps1;

   const auto b = Reshape(B_.Read(), Q1D, L1D);
   const auto bt = Reshape(Bt_.Read(), H1D, Q1D);
   const auto gt = Reshape(Gt_.Read(), H1D, Q1D);
   const auto sJit = Reshape(sJit_.Read(), Q1D, Q1D, Q1D, NE, 3, 3);
   const auto energy = Reshape(e_.Read(), L1D, L1D, L1D, NE);

   auto velocity = Reshape(v_.Write(), D1D, D1D, D1D, 3, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED real_t B[Q1D][L1D];
      MFEM_SHARED real_t Bt[H1D][Q1D];
      MFEM_SHARED real_t Gt[H1D][Q1D];
      MFEM_SHARED real_t E[L1D][L1D][L1D];
      MFEM_SHARED real_t sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED real_t sm1[3][Q1D*Q1D*Q1D];
      auto MMQ0 = (real_t (*)[D1D][Q1D]) (sm0+0);
      auto MMQ1 = (real_t (*)[D1D][Q1D]) (sm0+1);
      auto MMQ2 = (real_t (*)[D1D][Q1D]) (sm0+2);
      auto MQQ0 = (real_t (*)[Q1D][Q1D]) (sm1+0);
      auto MQQ1 = (real_t (*)[Q1D][Q1D]) (sm1+1);
      auto MQQ2 = (real_t (*)[Q1D][Q1D]) (sm1+2);
      MFEM_SHARED real_t QQQ[Q1D][Q1D][Q1D];
      auto QQQ0 = (real_t (*)[Q1D][Q1D]) (sm0+0);
      auto QQQ1 = (real_t (*)[Q1D][Q1D]) (sm0+1);
      auto QQQ2 = (real_t (*)[Q1D][Q1D]) (sm0+2);
      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(l,y,Q1D)
            {
               if (l < L1D) { B[q][l] = b(q,l); }
               if (l < H1D) { Bt[l][q] = bt(l,q); }
               if (l < H1D) { Gt[l][q] = gt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lx,x,L1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(lz,z,L1D)
            {
               E[lx][ly][lz] = energy(lx,ly,lz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               real_t u = 0.0;
               for (int lx = 0; lx < L1D; ++lx)
               {
                  u += B[qx][lx] * E[lx][ly][lz];
               }
               MMQ0[lz][ly][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               real_t u = 0.0;
               for (int ly = 0; ly < L1D; ++ly)
               {
                  u += B[qy][ly] * MMQ0[lz][ly][qx];
               }
               MQQ0[lz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               real_t u = 0.0;
               for (int lz = 0; lz < L1D; ++lz)
               {
                  u += B[qz][lz] * MQQ0[lz][qy][qx];
               }
               QQQ[qz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
               {
                  const real_t esx = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,0,c);
                  const real_t esy = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,1,c);
                  const real_t esz = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,2,c);
                  QQQ0[qz][qy][qx] = esx;
                  QQQ1[qz][qy][qx] = esy;
                  QQQ2[qz][qy][qx] = esz;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(hx,x,H1D)
               {
                  real_t u = 0.0;
                  real_t v = 0.0;
                  real_t w = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += Gt[hx][qx] * QQQ0[qz][qy][qx];
                     v += Bt[hx][qx] * QQQ1[qz][qy][qx];
                     w += Bt[hx][qx] * QQQ2[qz][qy][qx];
                  }
                  MQQ0[hx][qy][qz] = u;
                  MQQ1[hx][qy][qz] = v;
                  MQQ2[hx][qy][qz] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(hx,x,H1D)
               {
                  real_t u = 0.0;
                  real_t v = 0.0;
                  real_t w = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += MQQ0[hx][qy][qz] * Bt[hy][qy];
                     v += MQQ1[hx][qy][qz] * Gt[hy][qy];
                     w += MQQ2[hx][qy][qz] * Bt[hy][qy];
                  }
                  MMQ0[hx][hy][qz] = u;
                  MMQ1[hx][hy][qz] = v;
                  MMQ2[hx][hy][qz] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(hz,z,H1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(hx,x,H1D)
               {
                  real_t u = 0.0;
                  real_t v = 0.0;
                  real_t w = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += MMQ0[hx][hy][qz] * Bt[hz][qz];
                     v += MMQ1[hx][hy][qz] * Bt[hz][qz];
                     w += MMQ2[hx][hy][qz] * Gt[hz][qz];
                  }
                  velocity(hx,hy,hz,c,e) = u + v + w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(hz,z,H1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(hx,x,H1D)
               {
                  const real_t v = velocity(hx,hy,hz,c,e);
                  if (fabs(v) < eps2)
                  {
                     velocity(hx,hy,hz,c,e) = 0.0;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

static void kForceMult(const int DIM,
                       const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &B,
                       const Array<real_t> &Bt,
                       const Array<real_t> &Gt,
                       const DenseTensor &stressJinvT,
                       const Vector &e,
                       Vector &v)
{
   using fForceMult = decltype(&kSmemForceMult2D<2,3,4,2,3>);
   static std::unordered_map<int, fForceMult> call =
   {
      {0x234,&kSmemForceMult2D<2,3,4,2,3>},
      {0x334,&kSmemForceMult3D<3,3,4,2,3>},
   };
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   call[id](NE, B, Bt, Gt, stressJinvT, e, v);
}

template<int DIM, int D1D, int Q1D, int L1D, int H1D, int NBZ =1> static
void kSmemForceMultTranspose2D(const int NE,
                               const Array<real_t> &Bt_,
                               const Array<real_t> &B_,
                               const Array<real_t> &G_,
                               const DenseTensor &sJit_,
                               const Vector &v_,
                               Vector &e_)
{
   MFEM_VERIFY(D1D==H1D,"");
   const auto b = Reshape(B_.Read(), Q1D,H1D);
   const auto g = Reshape(G_.Read(), Q1D,H1D);
   const auto bt = Reshape(Bt_.Read(), L1D,Q1D);
   const auto sJit = Reshape(Read(sJit_.GetMemory(),
                                  Q1D*Q1D*NE*2*2), Q1D, Q1D, NE, 2, 2);
   const auto velocity = Reshape(v_.Read(), D1D,D1D,2,NE);
   auto energy = Reshape(e_.Write(), L1D, L1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED real_t Bt[L1D][Q1D];
      MFEM_SHARED real_t B[Q1D][H1D];
      MFEM_SHARED real_t G[Q1D][H1D];
      MFEM_SHARED real_t Vz[NBZ][D1D*D1D];
      auto V = (real_t (*)[D1D])(Vz + z);
      MFEM_SHARED real_t DQz[2][NBZ][D1D*Q1D];
      auto DQ0 = (real_t (*)[Q1D])(DQz[0] + z);
      auto DQ1 = (real_t (*)[Q1D])(DQz[1] + z);
      MFEM_SHARED real_t QQz[3][NBZ][Q1D*Q1D];
      auto QQ = (real_t (*)[Q1D])(QQz[0] + z);
      auto QQ0 = (real_t (*)[Q1D])(QQz[1] + z);
      auto QQ1 = (real_t (*)[Q1D])(QQz[2] + z);
      MFEM_SHARED real_t QLz[NBZ][Q1D*L1D];
      auto QL = (real_t (*)[L1D]) (QLz + z);
      if (z == 0)
      {
         MFEM_FOREACH_THREAD_DIRECT(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(h,y,Q1D)
            {
               if (h < H1D) { B[q][h] = b(q,h); }
               if (h < H1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
         {
            QQ[qy][qx] = 0.0;
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy,y,D1D)
            {
               V[dx][dy] = velocity(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               real_t u = 0.0;
               real_t v = 0.0;
               for (int dx = 0; dx < H1D; ++dx)
               {
                  const real_t input = V[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               real_t u = 0.0;
               real_t v = 0.0;
               for (int dy = 0; dy < H1D; ++dy)
               {
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               QQ0[qy][qx] = u;
               QQ1[qy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               const real_t esx = QQ0[qy][qx] * sJit(qx,qy,e,0,c);
               const real_t esy = QQ1[qy][qx] * sJit(qx,qy,e,1,c);
               QQ[qy][qx] += esx + esy;
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(lx,x,L1D)
         {
            real_t u = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += QQ[qy][qx] * Bt[lx][qx];
            }
            QL[qy][lx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(lx,x,L1D)
         {
            real_t u = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += QL[qy][lx] * Bt[ly][qy];
            }
            energy(lx,ly,e) = u;
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template<int DIM, int D1D, int Q1D, int L1D, int H1D> static
void kSmemForceMultTranspose3D(const int NE,
                               const Array<real_t> &Bt_,
                               const Array<real_t> &B_,
                               const Array<real_t> &G_,
                               const DenseTensor &sJit_,
                               const Vector &v_,
                               Vector &e_)
{
   MFEM_VERIFY(D1D==H1D,"");
   const auto b = Reshape(B_.Read(), Q1D,H1D);
   const auto g = Reshape(G_.Read(), Q1D,H1D);
   const auto bt = Reshape(Bt_.Read(), L1D,Q1D);
   const auto sJit = Reshape(Read(sJit_.GetMemory(),
                                  Q1D*Q1D*Q1D*NE*3*3), Q1D, Q1D, Q1D, NE, 3, 3);
   const auto velocity = Reshape(v_.Read(), D1D, D1D, D1D, 3, NE);
   auto energy = Reshape(e_.Write(), L1D, L1D, L1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED real_t Bt[L1D][Q1D];
      MFEM_SHARED real_t B[Q1D][H1D];
      MFEM_SHARED real_t G[Q1D][H1D];
      MFEM_SHARED real_t sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED real_t sm1[3][Q1D*Q1D*Q1D];
      auto V = (real_t (*)[D1D][D1D]) (sm0+0);
      auto MMQ0 = (real_t (*)[D1D][Q1D]) (sm0+1);
      auto MMQ1 = (real_t (*)[D1D][Q1D]) (sm0+2);
      auto MQQ0 = (real_t (*)[Q1D][Q1D]) (sm1+0);
      auto MQQ1 = (real_t (*)[Q1D][Q1D]) (sm1+1);
      auto MQQ2 = (real_t (*)[Q1D][Q1D]) (sm1+2);
      auto QQQ0 = (real_t (*)[Q1D][Q1D]) (sm0+0);
      auto QQQ1 = (real_t (*)[Q1D][Q1D]) (sm0+1);
      auto QQQ2 = (real_t (*)[Q1D][Q1D]) (sm0+2);
      MFEM_SHARED real_t QQQ[Q1D][Q1D][Q1D];
      if (z == 0)
      {
         MFEM_FOREACH_THREAD_DIRECT(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(h,y,Q1D)
            {
               if (h < H1D) { B[q][h] = b(q,h); }
               if (h < H1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               QQQ[qz][qy][qx] = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dz,z,D1D)
               {
                  V[dx][dy][dz] = velocity(dx,dy,dz,c,e);
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  real_t v = 0.0;
                  for (int dx = 0; dx < H1D; ++dx)
                  {
                     const real_t input = V[dx][dy][dz];
                     u += G[qx][dx] * input;
                     v += B[qx][dx] * input;
                  }
                  MMQ0[dz][dy][qx] = u;
                  MMQ1[dz][dy][qx] = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  real_t v = 0.0;
                  real_t w = 0.0;
                  for (int dy = 0; dy < H1D; ++dy)
                  {
                     u += MMQ0[dz][dy][qx] * B[qy][dy];
                     v += MMQ1[dz][dy][qx] * G[qy][dy];
                     w += MMQ1[dz][dy][qx] * B[qy][dy];
                  }
                  MQQ0[dz][qy][qx] = u;
                  MQQ1[dz][qy][qx] = v;
                  MQQ2[dz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  real_t v = 0.0;
                  real_t w = 0.0;
                  for (int dz = 0; dz < H1D; ++dz)
                  {
                     u += MQQ0[dz][qy][qx] * B[qz][dz];
                     v += MQQ1[dz][qy][qx] * B[qz][dz];
                     w += MQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  QQQ0[qz][qy][qx] = u;
                  QQQ1[qz][qy][qx] = v;
                  QQQ2[qz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
               {
                  const real_t esx = QQQ0[qz][qy][qx] * sJit(qx,qy,qz,e,0,c);
                  const real_t esy = QQQ1[qz][qy][qx] * sJit(qx,qy,qz,e,1,c);
                  const real_t esz = QQQ2[qz][qy][qx] * sJit(qx,qy,qz,e,2,c);
                  QQQ[qz][qy][qx] += esx + esy + esz;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(lx,x,L1D)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ[qz][qy][qx] * Bt[lx][qx];
               }
               MQQ0[qz][qy][lx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(lx,x,L1D)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += MQQ0[qz][qy][lx] * Bt[ly][qy];
               }
               MMQ0[qz][ly][lx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(lx,x,L1D)
            {
               real_t u = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += MMQ0[qz][ly][lx] * Bt[lz][qz];
               }
               energy(lx,ly,lz,e) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}


static void kForceMultTranspose(const int DIM,
                                const int D1D,
                                const int Q1D,
                                const int L1D,
                                const int H1D,
                                const int nzones,
                                const Array<real_t> &L2QuadToDof,
                                const Array<real_t> &H1DofToQuad,
                                const Array<real_t> &H1DofToQuadD,
                                const DenseTensor &stressJinvT,
                                const Vector &v,
                                Vector &e)
{
   using fForceMultTranspose = decltype(&kSmemForceMultTranspose2D<2,3,4,2,3>);
   static std::unordered_map<int, fForceMultTranspose> call =
   {
      {0x234,&kSmemForceMultTranspose2D<2,3,4,2,3>},
      {0x334,&kSmemForceMultTranspose3D<3,3,4,2,3>},
   };
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   call[id](nzones, L2QuadToDof, H1DofToQuad, H1DofToQuadD, stressJinvT, v, e);
}

class Force : public Operator
{
private:
   const int dim, nzones;
   const QData &quad_data;
   const typename T::FiniteElementSpace &h1fes, &l2fes;
   const Operator *h1restrict, *l2restrict;
   const IntegrationRule &integ_rule, &ir1D;
   const int D1D, Q1D, L1D, H1D;
   const int h1sz, l2sz;
   const DofToQuad *l2D2Q, *h1D2Q;
   mutable Vector gVecL2, gVecH1;
public:
   Force(const QData &qd,
         const typename T::FiniteElementSpace &h1f,
         const typename T::FiniteElementSpace &l2f,
         const IntegrationRule &ir) :
      dim(h1f.GetMesh()->Dimension()),
      nzones(h1f.GetMesh()->GetNE()),
      quad_data(qd),
      h1fes(h1f),
      l2fes(l2f),
      h1restrict(h1f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      l2restrict(l2f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      integ_rule(ir),
      ir1D(IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder())),
      D1D(h1fes.GetTypicalFE()->GetOrder()+1),
      Q1D(ir1D.GetNPoints()),
      L1D(l2fes.GetTypicalFE()->GetOrder()+1),
      H1D(h1fes.GetTypicalFE()->GetOrder()+1),
      h1sz(h1fes.GetVDim() * h1fes.GetTypicalFE()->GetDof() * nzones),
      l2sz(l2fes.GetTypicalFE()->GetDof() * nzones),
      l2D2Q(&l2fes.GetTypicalFE()->GetDofToQuad(integ_rule, DofToQuad::TENSOR)),
      h1D2Q(&h1fes.GetTypicalFE()->GetDofToQuad(integ_rule, DofToQuad::TENSOR)),
      gVecL2(l2sz),
      gVecH1(h1sz)
   {
      gVecL2.SetSize(l2sz);
      gVecH1.SetSize(h1sz);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      l2restrict->Mult(x, gVecL2);
      kForceMult(dim, D1D, Q1D, nzones,
                 l2D2Q->B, h1D2Q->Bt, h1D2Q->Gt,
                 quad_data.stressJinvT,
                 gVecL2, gVecH1);
      h1restrict->MultTranspose(gVecH1, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      h1restrict->Mult(x, gVecH1);
      kForceMultTranspose(dim, D1D, Q1D, L1D, H1D, nzones,
                          l2D2Q->Bt, h1D2Q->B, h1D2Q->G,
                          quad_data.stressJinvT,
                          gVecH1, gVecL2);
      l2restrict->MultTranspose(gVecL2, y);
   }
};

class Mass : public Operator
{
   typename T::BilinearForm bf;
   OperatorPtr massOperator;
   mutable int ess_tdofs_count;
   mutable Array<int> ess_tdofs;

public:
   Mass(Coefficient &Q,
        typename T::FiniteElementSpace &pfes,
        const IntegrationRule &ir) : Operator(pfes.GetTrueVSize()),
      bf(&pfes),
      ess_tdofs_count(0)
   {
      bf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      bf.AddDomainIntegrator(new mfem::MassIntegrator(Q, &ir));
      bf.Assemble();
      bf.FormSystemMatrix(Array<int> {}, massOperator);
   }

   void Mult(const Vector &X, Vector &Y) const override
   {
      massOperator->Mult(X, Y);
      if (ess_tdofs_count) { Y.SetSubVector(ess_tdofs, 0.0); }
   }

   void SetEssentialTrueDofs(Array<int> &dofs) const
   {
      ess_tdofs_count = dofs.Size();
      if (ess_tdofs.Size() == 0)
      {
         int global_ess_tdofs_count;
         SumReduce(&ess_tdofs_count, &global_ess_tdofs_count);
         MFEM_VERIFY(global_ess_tdofs_count>0, "!(global_ess_tdofs_count>0)");
         ess_tdofs.SetSize(global_ess_tdofs_count);
      }
      if (ess_tdofs_count == 0) { return; }
      ess_tdofs = dofs;
   }

   void EliminateRHS(Vector &B) const
   {
      if (ess_tdofs_count > 0) { B.SetSubVector(ess_tdofs, 0.0); }
   }
};

MFEM_HOST_DEVICE inline real_t smooth_step_01(real_t x, real_t eps)
{
   const real_t y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

template<int dim> MFEM_HOST_DEVICE static inline
void QBody(const int nzones, const int z,
           const int nqp, const int q,
           const real_t gamma,
           const bool use_viscosity,
           const real_t h0,
           const real_t h1order,
           const real_t cfl,
           const real_t infinity,
           real_t *Jinv,
           real_t *stress,
           real_t *sgrad_v,
           real_t *eig_val_data,
           real_t *eig_vec_data,
           real_t *compr_dir,
           real_t *Jpi,
           real_t *ph_dir,
           real_t *stressJiT,
           const real_t *d_weights,
           const real_t *d_Jacobians,
           const real_t *d_rho0DetJ0w,
           const real_t *d_e_quads,
           const real_t *d_grad_v_ext,
           const real_t *d_Jac0inv,
           real_t *d_dt_est,
           real_t *d_stressJinvT)
{
   constexpr int dim2 = dim*dim;
   real_t min_detJ = infinity;
   const int zq = z * nqp + q;
   const real_t weight =  d_weights[q];
   const real_t inv_weight = 1. / weight;
   const real_t *J = d_Jacobians + dim2*(nqp*z + q);
   const real_t detJ = kernels::Det<dim>(J);
   min_detJ = std::fmin(min_detJ,detJ);
   kernels::CalcInverse<dim>(J,Jinv);
   const real_t rho = inv_weight * d_rho0DetJ0w[zq] / detJ;
   const real_t e   = std::fmax((real_t) 0.0, d_e_quads[zq]);
   const real_t p   = (gamma - 1.0) * rho * e;
   const real_t sound_speed = std::sqrt(gamma * (gamma-1.0) * e);
   for (int k = 0; k < dim2; k+=1) { stress[k] = 0.0; }
   for (int d = 0; d < dim; d++) { stress[d*dim+d] = -p; }
   real_t visc_coeff = 0.0;
   if (use_viscosity)
   {
      const real_t *dV = d_grad_v_ext + dim2*(nqp*z + q);
      kernels::Mult(dim, dim, dim, dV, Jinv, sgrad_v);
      kernels::Symmetrize(dim,sgrad_v);
      if (dim==1)
      {
         eig_val_data[0] = sgrad_v[0];
         eig_vec_data[0] = 1.;
      }
      else
      {
         kernels::CalcEigenvalues<dim>(sgrad_v, eig_val_data, eig_vec_data);
      }
      for (int k=0; k<dim; k+=1) { compr_dir[k]=eig_vec_data[k]; }
      kernels::Mult(dim, dim, dim, J, d_Jac0inv+zq*dim*dim, Jpi);
      kernels::Mult(dim, dim, Jpi, compr_dir, ph_dir);
      const real_t ph_dir_nl2 = kernels::Norml2(dim,ph_dir);
      const real_t compr_dir_nl2 = kernels::Norml2(dim, compr_dir);
      const real_t h = h0 * ph_dir_nl2 / compr_dir_nl2;
      const real_t mu = eig_val_data[0];
      visc_coeff = 2.0 * rho * h * h * fabs(mu);
      const real_t eps = 1e-12;
      visc_coeff += 0.5 * rho * h * sound_speed *
                    (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
      kernels::Add(dim, dim, visc_coeff, stress, sgrad_v, stress);
   }
   const real_t sv = kernels::CalcSingularvalue<dim>(J, dim-1);
   const real_t h_min = sv / h1order;
   const real_t inv_h_min = 1. / h_min;
   const real_t inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
   const real_t inv_dt = sound_speed * inv_h_min
                         + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
   if (min_detJ < 0.0)
   {
      d_dt_est[zq] = 0.0;
   }
   else
   {
      if (inv_dt>0.0)
      {
         const real_t cfl_inv_dt = cfl / inv_dt;
         d_dt_est[zq] = std::fmin(d_dt_est[zq], cfl_inv_dt);
      }
   }
   kernels::MultABt(dim, dim, dim, stress, Jinv, stressJiT);
   for (int k=0; k<dim2; k+=1) { stressJiT[k] *= weight * detJ; }
   for (int vd = 0 ; vd < dim; vd++)
   {
      for (int gd = 0; gd < dim; gd++)
      {
         const int offset = zq + nqp*nzones*(gd+vd*dim);
         d_stressJinvT[offset] = stressJiT[vd+gd*dim];
      }
   }
}

template<int DIM, int Q1D, int DIM2 = DIM*DIM>
static inline void QKernel(const int nzones,
                           const int nqp,
                           const real_t gamma,
                           const bool use_viscosity,
                           const real_t h0,
                           const real_t h1order,
                           const real_t cfl,
                           const real_t infinity,
                           const Array<real_t> &weights,
                           const Vector &Jacobians,
                           const Vector &rho0DetJ0w,
                           const Vector &e_quads,
                           const Vector &grad_v_ext,
                           const DenseTensor &Jac0inv,
                           Vector &dt_est,
                           DenseTensor &stressJinvT)
{
   const auto d_weights = weights.Read();
   const auto d_Jacobians = Jacobians.Read();
   const auto d_rho0DetJ0w = rho0DetJ0w.Read();
   const auto d_e_quads = e_quads.Read();
   const auto d_grad_v_ext = grad_v_ext.Read();
   auto d_Jac0inv = Read(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWrite();
   auto d_stressJinvT = Write(stressJinvT.GetMemory(),
                              stressJinvT.TotalSize());
   if constexpr (DIM == 2)
   {
      mfem::forall_2D(nzones, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int z)
      {
         real_t Jinv[DIM2];
         real_t stress[DIM2];
         real_t sgrad_v[DIM2];
         real_t eig_val_data[DIM];
         real_t eig_vec_data[DIM2];
         real_t compr_dir[DIM];
         real_t Jpi[DIM2];
         real_t ph_dir[DIM];
         real_t stressJiT[DIM2];
         MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               QBody<DIM>(nzones, z, nqp, qx + qy * Q1D,
                          gamma, use_viscosity, h0, h1order, cfl, infinity,
                          Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
                          compr_dir,Jpi,ph_dir,stressJiT,
                          d_weights, d_Jacobians, d_rho0DetJ0w,
                          d_e_quads, d_grad_v_ext, d_Jac0inv,
                          d_dt_est, d_stressJinvT);
            }
         }
         MFEM_SYNC_THREAD;
      });
   }

   if constexpr (DIM == 3)
   {
      mfem::forall_3D(nzones, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int z)
      {
         real_t Jinv[DIM2];
         real_t stress[DIM2];
         real_t sgrad_v[DIM2];
         real_t eig_val_data[DIM];
         real_t eig_vec_data[DIM2];
         real_t compr_dir[DIM];
         real_t Jpi[DIM2];
         real_t ph_dir[DIM];
         real_t stressJiT[DIM2];
         MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
               {
                  QBody<DIM>(nzones, z, nqp, qx + Q1D * (qy + qz * Q1D),
                             gamma, use_viscosity, h0, h1order, cfl, infinity,
                             Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
                             compr_dir,Jpi,ph_dir,stressJiT,
                             d_weights, d_Jacobians, d_rho0DetJ0w,
                             d_e_quads, d_grad_v_ext, d_Jac0inv,
                             d_dt_est, d_stressJinvT);
               }
            }
         }
         MFEM_SYNC_THREAD;
      });
   }
}

template <int DIM>
class QUpdate
{
   const int dim, nq, ne;
   const bool use_viscosity;
   const real_t cfl, gamma;
   const IntegrationRule &ir;
   typename T::FiniteElementSpace &H1fes, &L2fes;
   const Operator *H1ER;
   const int vdim;
   Vector d_dt_est;
   Vector d_l2_e_quads_data;
   Vector d_h1_v_local_in, d_h1_grad_x_data, d_h1_grad_v_data;
   const QuadratureInterpolator *q1, *q2;
public:
   QUpdate(const int d, const int ne, const bool uv,
           const real_t c, const real_t g, const IntegrationRule &i,
           typename T::FiniteElementSpace &h1, typename T::FiniteElementSpace &l2):
      dim(d), nq(i.GetNPoints()), ne(ne), use_viscosity(uv), cfl(c), gamma(g),
      ir(i), H1fes(h1), L2fes(l2),
      H1ER(H1fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      vdim(H1fes.GetVDim()),
      d_dt_est(ne*nq),
      d_l2_e_quads_data(ne*nq),
      d_h1_v_local_in(nq*ne*vdim),
      d_h1_grad_x_data(nq*ne*vdim*vdim),
      d_h1_grad_v_data(nq*ne*vdim*vdim),
      q1(H1fes.GetQuadratureInterpolator(ir)),
      q2(L2fes.GetQuadratureInterpolator(ir)) { }

   void UpdateQuadratureData(const Vector &S,
                             const real_t h0,
                             QData &quad_data,
                             real_t &dt_est)
   {
      constexpr int nqp1D = 4;
      const int H1_size = H1fes.GetVSize();
      const real_t h1order = H1fes.GetElementOrder(0);
      const real_t infinity = std::numeric_limits<real_t>::infinity();

      GridFunction d_x, d_v, d_e;
      auto * S_p = const_cast<Vector*>(&S);

      d_x.MakeRef(&H1fes,*S_p, 0);
      H1ER->Mult(d_x, d_h1_v_local_in);
      q1->SetOutputLayout(QVectorLayout::byVDIM);
      q1->Derivatives(d_h1_v_local_in, d_h1_grad_x_data);

      d_v.MakeRef(&H1fes,*S_p, H1_size);
      H1ER->Mult(d_v, d_h1_v_local_in);
      q1->Derivatives(d_h1_v_local_in, d_h1_grad_v_data);

      d_e.MakeRef(&L2fes, *S_p, 2*H1_size);
      q2->SetOutputLayout(QVectorLayout::byVDIM);
      q2->Values(d_e, d_l2_e_quads_data);

      d_dt_est = dt_est;
      static std::unordered_map<int, decltype(&QKernel<2,4>)>
      qupdate = { {0x24,&QKernel<2,4>}, {0x34,&QKernel<3,4>} };

      const int id = (dim<<4) | nqp1D;
      qupdate[id](ne, nq, gamma, use_viscosity, h0,
                  h1order, cfl, infinity, ir.GetWeights(), d_h1_grad_x_data,
                  quad_data.rho0DetJ0w, d_l2_e_quads_data, d_h1_grad_v_data,
                  quad_data.Jac0inv, d_dt_est, quad_data.stressJinvT);

      dt_est = d_dt_est.Min();
   }
};

template <int DIM>
void ComputeRho0DetJ0(typename T::Mesh &mesh,
                      const IntegrationRule &ir,
                      typename T::FiniteElementSpace &l2_fes,
                      typename T::GridFunction &rho0,
                      QData &qdata)
{
   const int ne = mesh.GetNE();
   const int nq = ir.GetNPoints();
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int flags = GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh.GetGeometricFactors(ir, flags);
   Vector rho0Q(nq*ne);
   rho0Q.UseDevice(true);
   const QuadratureInterpolator *qi = l2_fes.GetQuadratureInterpolator(ir);
   qi->Values(rho0, rho0Q);
   const auto W = ir.GetWeights().Read();
   const auto R = Reshape(rho0Q.Read(), nq, ne);
   const auto J = Reshape(geom->J.Read(), nq, DIM, DIM, ne);
   const auto detJ = Reshape(geom->detJ.Read(), nq, ne);

   auto D = Reshape(qdata.rho0DetJ0w.Write(), nq, ne);
   auto invJ = Reshape(qdata.Jac0inv.Write(), DIM, DIM, nq, ne);

   if constexpr (DIM == 2)
   {
      mfem::forall_2D(ne, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               const int q = qx + qy * Q1D;
               const real_t J11 = J(q,0,0,e);
               const real_t J12 = J(q,1,0,e);
               const real_t J21 = J(q,0,1,e);
               const real_t J22 = J(q,1,1,e);
               const real_t det = detJ(q,e);
               D(q,e) =  W[q] * R(q,e) * det;
               const real_t r_idetJ = 1.0 / det;
               invJ(0,0,q,e) =  J22 * r_idetJ;
               invJ(1,0,q,e) = -J12 * r_idetJ;
               invJ(0,1,q,e) = -J21 * r_idetJ;
               invJ(1,1,q,e) =  J11 * r_idetJ;
            }
         }
      });
   }

   if constexpr (DIM == 3)
   {
      mfem::forall_3D(ne, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
               {
                  const int q = qx + (qy + qz * Q1D) * Q1D;
                  const real_t J11 = J(q,0,0,e), J12 = J(q,0,1,e), J13 = J(q,0,2,e);
                  const real_t J21 = J(q,1,0,e), J22 = J(q,1,1,e), J23 = J(q,1,2,e);
                  const real_t J31 = J(q,2,0,e), J32 = J(q,2,1,e), J33 = J(q,2,2,e);
                  const real_t det = detJ(q,e);
                  D(q,e) = W[q] * R(q,e) * det;
                  const real_t r_idetJ = 1.0 / det;
                  invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
                  invJ(1,0,q,e) = r_idetJ * ((J32 * J13)-(J33 * J12));
                  invJ(2,0,q,e) = r_idetJ * ((J12 * J23)-(J13 * J22));
                  invJ(0,1,q,e) = r_idetJ * ((J23 * J31)-(J21 * J33));
                  invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
                  invJ(2,1,q,e) = r_idetJ * ((J13 * J21)-(J11 * J23));
                  invJ(0,2,q,e) = r_idetJ * ((J21 * J32)-(J22 * J31));
                  invJ(1,2,q,e) = r_idetJ * ((J31 * J12)-(J32 * J11));
                  invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
               }
            }
         }
      });
   }
   qdata.rho0DetJ0w.HostRead();
}

template <int DIM>
class LagrangianHydroOperator : public LagrangianHydroBase<DIM>
{
   Force force;
   Mass v_mass, e_mass;

   mutable QData qdata;
   mutable QUpdate<DIM> qupdate;

   using B = LagrangianHydroBase<DIM>;
   using B::ir;

   void UpdateQuadratureData(const Vector &S) const final
   {
      if (B::quad_data_is_current) { return; }
      qupdate.UpdateQuadratureData(S, B::h0, qdata, B::dt_est);
      B::quad_data_is_current = true;
   }

public:
   LagrangianHydroOperator(Coefficient &rho_coeff, const int size,
                           typename T::FiniteElementSpace &h1,
                           typename T::FiniteElementSpace &l2,
                           typename T::Mesh &pmesh,
                           const Array<int> &essential_tdofs,
                           typename T::GridFunction &rho0,
                           const int source_type,
                           const real_t cfl,
                           const Coefficient &material,
                           const bool use_viscosity,
                           const real_t cgt,
                           const int cgiter,
                           real_t ftz,
                           const int order_q,
                           const real_t gamma,
                           int h1_basis_type):
      B(rho_coeff, size, h1, l2, pmesh, essential_tdofs, rho0, source_type,
        cfl, material, use_viscosity, cgt, cgiter, ftz, order_q, gamma, h1_basis_type),
      force(qdata, h1, l2, ir),
      v_mass(rho_coeff, B::H1c, ir),
      e_mass(rho_coeff, l2, ir),
      qdata(DIM, B::ne, ir.GetNPoints()),
      qupdate(DIM, B::ne, use_viscosity, cfl, gamma, ir, h1, l2)
   {
      ComputeRho0DetJ0<DIM>(pmesh, ir,l2, rho0, qdata);
      B::CG_VMass.SetOperator(v_mass);
      B::CG_EMass.SetOperator(e_mass);
   }

   void ForceMult(const Vector &S) const final { force.Mult(B::one, B::rhs); }

   void VMassSetup(const int c) const final
   {
      v_mass.SetEssentialTrueDofs(B::c_tdofs[c]);
      v_mass.EliminateRHS(B::B);
   }

   void ForceMultTranspose(const Vector &S, const Vector &v) const final
   {
      force.MultTranspose(v, B::e_rhs);
   }
};

} // namespace mfem

#ifdef MFEM_SEDOV_PA_MPI
TEST_CASE("Sedov", "[Sedov][Parallel]")
{
   sedov_tests<LagrangianHydroOperator>(Mpi::WorldRank());
}
#else
TEST_CASE("Sedov", "[Sedov]")
{
   sedov_tests<LagrangianHydroOperator>(0);
}
#endif
