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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "run_unit_tests.hpp"

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include <unordered_map>
#include <cstring>
#include "general/forall.hpp"
#include "linalg/kernels.hpp"

#if defined(MFEM_SEDOV_MPI) && !defined(MFEM_USE_MPI)
#error "Cannot use MFEM_SEDOV_MPI without MFEM_USE_MPI!"
#endif

#if defined(MFEM_USE_MPI) && defined(MFEM_SEDOV_MPI)
#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
#define PFesGetParMeshGetComm0(pfes) pfes.GetParMesh()->GetComm()
#else
#define HYPRE_BigInt int
#define ParMesh Mesh
#define GetParMesh GetMesh
#define GlobalTrueVSize GetVSize
#define ParBilinearForm BilinearForm
#define ParGridFunction GridFunction
#define ParFiniteElementSpace FiniteElementSpace
#define PFesGetParMeshGetComm(...)
#define PFesGetParMeshGetComm0(...) 0
#define MPI_Allreduce(src,dst,...) *dst = *src
#define MPI_Reduce(src, dst, n, T,...) *dst = *src
#endif

using namespace std;
using namespace mfem;

namespace mfem
{

static void v0(const Vector&, Vector &v) { v = 0.0; }
static real_t rho0(const Vector&) { return 1.0; }
static real_t gamma(const Vector&) { return 1.4; }

namespace hydrodynamics
{

struct QuadratureData
{
   DenseTensor Jac0inv, stressJinvT;
   Vector rho0DetJ0w;
   real_t h0, dt_est;
   QuadratureData(int dim, int nzones, int quads_per_zone)
      : Jac0inv(dim, dim, nzones * quads_per_zone),
        stressJinvT(nzones * quads_per_zone, dim, dim),
        rho0DetJ0w(nzones * quads_per_zone) { }
};

struct Tensors1D
{
   DenseMatrix HQshape1D, HQgrad1D, LQshape1D;
   Tensors1D(int H1order, int L2order, int nqp1D, bool bernstein_v)
      : HQshape1D(H1order + 1, nqp1D), HQgrad1D(H1order + 1, nqp1D),
        LQshape1D(L2order + 1, nqp1D)
   {
      const real_t *quad1D_pos =
         poly1d.GetPoints(nqp1D - 1, Quadrature1D::GaussLegendre);
      Poly_1D::Basis &basisH1 =
         poly1d.GetBasis(H1order, Quadrature1D::GaussLobatto);
      Vector col, grad_col;
      for (int q = 0; q < nqp1D; q++)
      {
         HQshape1D.GetColumnReference(q, col);
         HQgrad1D.GetColumnReference(q, grad_col);
         if (bernstein_v)
         {
            poly1d.CalcBernstein(H1order, quad1D_pos[q],
                                 col.GetData(), grad_col.GetData());
         }
         else { basisH1.Eval(quad1D_pos[q], col, grad_col); }
      }
      for (int q = 0; q < nqp1D; q++)
      {
         LQshape1D.GetColumnReference(q, col);
         poly1d.CalcBernstein(L2order, quad1D_pos[q], col);
      }
   }
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
   auto b = Reshape(B_.Read(), Q1D, L1D);
   auto bt = Reshape(Bt_.Read(), H1D, Q1D);
   auto gt = Reshape(Gt_.Read(), H1D, Q1D);
   auto sJit = Reshape(Read(sJit_.GetMemory(), Q1D*Q1D*NE*2*2),
                       Q1D,Q1D,NE,2,2);
   auto energy = Reshape(e_.Read(), L1D, L1D, NE);
   const real_t eps1 = std::numeric_limits<real_t>::epsilon();
   const real_t eps2 = eps1*eps1;
   auto velocity = Reshape(v_.Write(), D1D,D1D,2,NE);
   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED real_t B[Q1D][L1D];
      MFEM_SHARED real_t Bt[H1D][Q1D];
      MFEM_SHARED real_t Gt[H1D][Q1D];
      MFEM_SHARED real_t Ez[NBZ][L1D][L1D];
      real_t (*E)[L1D] = (real_t (*)[L1D])(Ez + z);
      MFEM_SHARED real_t LQz[2][NBZ][H1D][Q1D];
      real_t (*LQ0)[Q1D] = (real_t (*)[Q1D])(LQz[0] + z);
      real_t (*LQ1)[Q1D] = (real_t (*)[Q1D])(LQz[1] + z);
      MFEM_SHARED real_t QQz[3][NBZ][Q1D][Q1D];
      real_t (*QQ)[Q1D] = (real_t (*)[Q1D])(QQz[0] + z);
      real_t (*QQ0)[Q1D] = (real_t (*)[Q1D])(QQz[1] + z);
      real_t (*QQ1)[Q1D] = (real_t (*)[Q1D])(QQz[2] + z);
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
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            E[lx][ly] = energy(lx,ly,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
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
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const real_t esx = QQ[qy][qx] * sJit(qx,qy,e,0,c);
               const real_t esy = QQ[qy][qx] * sJit(qx,qy,e,1,c);
               QQ0[qy][qx] = esx;
               QQ1[qy][qx] = esy;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,H1D)
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
         MFEM_FOREACH_THREAD(dy,y,H1D)
         {
            MFEM_FOREACH_THREAD(dx,x,H1D)
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
         MFEM_FOREACH_THREAD(dy,y,H1D)
         {
            MFEM_FOREACH_THREAD(dx,x,H1D)
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
   auto b = Reshape(B_.Read(), Q1D, L1D);
   auto bt = Reshape(Bt_.Read(), H1D, Q1D);
   auto gt = Reshape(Gt_.Read(), H1D, Q1D);
   auto sJit = Reshape(Read(sJit_.GetMemory(), Q1D*Q1D*Q1D*NE*3*3),
                       Q1D,Q1D,Q1D,NE,3,3);
   auto energy = Reshape(e_.Read(), L1D, L1D, L1D, NE);
   const real_t eps1 = std::numeric_limits<real_t>::epsilon();
   const real_t eps2 = eps1*eps1;
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
      real_t (*MMQ0)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) (sm0+0);
      real_t (*MMQ1)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) (sm0+1);
      real_t (*MMQ2)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) (sm0+2);
      real_t (*MQQ0)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm1+0);
      real_t (*MQQ1)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm1+1);
      real_t (*MQQ2)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm1+2);
      MFEM_SHARED real_t QQQ[Q1D][Q1D][Q1D];
      real_t (*QQQ0)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm0+0);
      real_t (*QQQ1)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm0+1);
      real_t (*QQQ2)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm0+2);
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
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lz,z,L1D)
            {
               E[lx][ly][lz] = energy(lx,ly,lz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
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
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
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
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
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
         MFEM_FOREACH_THREAD(hz,z,H1D)
         {
            MFEM_FOREACH_THREAD(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
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
         MFEM_FOREACH_THREAD(hz,z,H1D)
         {
            MFEM_FOREACH_THREAD(hy,y,H1D)
            {
               MFEM_FOREACH_THREAD(hx,x,H1D)
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

typedef void (*fForceMult)(const int E,
                           const Array<real_t> &B,
                           const Array<real_t> &Bt,
                           const Array<real_t> &Gt,
                           const DenseTensor &stressJinvT,
                           const Vector &e,
                           Vector &v);

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
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   static std::unordered_map<int, fForceMult> call =
   {
      {0x234,&kSmemForceMult2D<2,3,4,2,3>},
      //{0x246,&kSmemForceMult2D<2,4,6,3,4>},
      //{0x258,&kSmemForceMult2D<2,5,8,4,5>},
      // 3D
      {0x334,&kSmemForceMult3D<3,3,4,2,3>},
      //{0x346,&kSmemForceMult3D<3,4,6,3,4>},
      //{0x358,&kSmemForceMult3D<3,5,8,4,5>},
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
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
   auto b = Reshape(B_.Read(), Q1D,H1D);
   auto g = Reshape(G_.Read(), Q1D,H1D);
   auto bt = Reshape(Bt_.Read(), L1D,Q1D);
   auto sJit = Reshape(Read(sJit_.GetMemory(), Q1D*Q1D*NE*2*2),
                       Q1D, Q1D, NE, 2, 2);
   auto velocity = Reshape(v_.Read(), D1D,D1D,2,NE);
   auto energy = Reshape(e_.Write(), L1D, L1D, NE);
   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED real_t Bt[L1D][Q1D];
      MFEM_SHARED real_t B[Q1D][H1D];
      MFEM_SHARED real_t G[Q1D][H1D];
      MFEM_SHARED real_t Vz[NBZ][D1D*D1D];
      real_t (*V)[D1D] = (real_t (*)[D1D])(Vz + z);
      MFEM_SHARED real_t DQz[2][NBZ][D1D*Q1D];
      real_t (*DQ0)[Q1D] = (real_t (*)[Q1D])(DQz[0] + z);
      real_t (*DQ1)[Q1D] = (real_t (*)[Q1D])(DQz[1] + z);
      MFEM_SHARED real_t QQz[3][NBZ][Q1D*Q1D];
      real_t (*QQ)[Q1D] = (real_t (*)[Q1D])(QQz[0] + z);
      real_t (*QQ0)[Q1D] = (real_t (*)[Q1D])(QQz[1] + z);
      real_t (*QQ1)[Q1D] = (real_t (*)[Q1D])(QQz[2] + z);
      MFEM_SHARED real_t QLz[NBZ][Q1D*L1D];
      real_t (*QL)[L1D] = (real_t (*)[L1D]) (QLz + z);
      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(h,y,Q1D)
            {
               if (h < H1D) { B[q][h] = b(q,h); }
               if (h < H1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            QQ[qy][qx] = 0.0;
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               V[dx][dy] = velocity(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const real_t esx = QQ0[qy][qx] * sJit(qx,qy,e,0,c);
               const real_t esy = QQ1[qy][qx] * sJit(qx,qy,e,1,c);
               QQ[qy][qx] += esx + esy;
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(lx,x,L1D)
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
      MFEM_FOREACH_THREAD(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD(lx,x,L1D)
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
   auto b = Reshape(B_.Read(), Q1D,H1D);
   auto g = Reshape(G_.Read(), Q1D,H1D);
   auto bt = Reshape(Bt_.Read(), L1D,Q1D);
   auto sJit = Reshape(Read(sJit_.GetMemory(), Q1D*Q1D*Q1D*NE*3*3),
                       Q1D, Q1D, Q1D, NE, 3, 3);
   auto velocity = Reshape(v_.Read(), D1D, D1D, D1D, 3, NE);
   auto energy = Reshape(e_.Write(), L1D, L1D, L1D, NE);
   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int z = MFEM_THREAD_ID(z);
      MFEM_SHARED real_t Bt[L1D][Q1D];
      MFEM_SHARED real_t B[Q1D][H1D];
      MFEM_SHARED real_t G[Q1D][H1D];
      MFEM_SHARED real_t sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED real_t sm1[3][Q1D*Q1D*Q1D];
      real_t (*V)[D1D][D1D]    = (real_t (*)[D1D][D1D]) (sm0+0);
      real_t (*MMQ0)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) (sm0+1);
      real_t (*MMQ1)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) (sm0+2);
      real_t (*MQQ0)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm1+0);
      real_t (*MQQ1)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm1+1);
      real_t (*MQQ2)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm1+2);
      real_t (*QQQ0)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm0+0);
      real_t (*QQQ1)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm0+1);
      real_t (*QQQ2)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) (sm0+2);
      MFEM_SHARED real_t QQQ[Q1D][Q1D][Q1D];
      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(h,y,Q1D)
            {
               if (h < H1D) { B[q][h] = b(q,h); }
               if (h < H1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
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
               QQQ[qz][qy][qx] = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  V[dx][dy][dz] = velocity(dx,dy,dz,c,e);
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
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
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
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
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
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
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
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
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
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
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

typedef void (*fForceMultTranspose)(const int nzones,
                                    const Array<real_t> &Bt,
                                    const Array<real_t> &B,
                                    const Array<real_t> &G,
                                    const DenseTensor &sJit,
                                    const Vector &v,
                                    Vector &e);

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
   MFEM_VERIFY(D1D==H1D,"D1D!=H1D");
   MFEM_VERIFY(L1D==D1D-1, "L1D!=D1D-1");
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   static std::unordered_map<int, fForceMultTranspose> call =
   {
      {0x234,&kSmemForceMultTranspose2D<2,3,4,2,3>},
      //{0x246,&kSmemForceMultTranspose2D<2,4,6,3,4>},
      //{0x258,&kSmemForceMultTranspose2D<2,5,8,4,5>},
      {0x334,&kSmemForceMultTranspose3D<3,3,4,2,3>},
      //{0x346,&kSmemForceMultTranspose3D<3,4,6,3,4>},
      //{0x358,&kSmemForceMultTranspose3D<3,5,8,4,5>}
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](nzones, L2QuadToDof, H1DofToQuad, H1DofToQuadD, stressJinvT, v, e);
}

class PAForceOperator : public Operator
{
private:
   const int dim, nzones;
   const QuadratureData &quad_data;
   const ParFiniteElementSpace &h1fes, &l2fes;
   const Operator *h1restrict, *l2restrict;
   const IntegrationRule &integ_rule, &ir1D;
   const int D1D, Q1D;
   const int L1D, H1D;
   const int h1sz, l2sz;
   const DofToQuad *l2D2Q, *h1D2Q;
   mutable Vector gVecL2, gVecH1;
public:
   PAForceOperator(const QuadratureData &qd,
                   const ParFiniteElementSpace &h1f,
                   const ParFiniteElementSpace &l2f,
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
      MFEM_ASSERT(h1f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC),"");
      MFEM_ASSERT(l2f.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC),"");
      gVecL2.SetSize(l2sz);
      gVecH1.SetSize(h1sz);
   }

   void Mult(const Vector &x, Vector &y) const override
   {

      l2restrict->Mult(x, gVecL2);
      kForceMult(dim, D1D, Q1D, nzones,
                 l2D2Q->B, h1D2Q->Bt, h1D2Q->Gt, quad_data.stressJinvT,
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

static void ComputeDiagonal2D(const int height, const int nzones,
                              const QuadratureData &quad_data,
                              const FiniteElementSpace &FESpace,
                              const Tensors1D *tensors1D,
                              Vector &diag)
{
   const TensorBasisElement *fe_H1 =
      dynamic_cast<const TensorBasisElement *>(FESpace.GetTypicalFE());
   const Array<int> &dof_map = fe_H1->GetDofMap();
   const DenseMatrix &HQs = tensors1D->HQshape1D;
   const int ndof1D = HQs.Height(), nqp1D = HQs.Width(), nqp = nqp1D * nqp1D;
   Vector dz(ndof1D * ndof1D);
   DenseMatrix HQ(ndof1D, nqp1D), D(dz.GetData(), ndof1D, ndof1D);
   Array<int> dofs;
   diag.SetSize(height);
   diag = 0.0;
   DenseMatrix HQs_sq(ndof1D, nqp1D);
   for (int i = 0; i < ndof1D; i++)
      for (int k = 0; k < nqp1D; k++)
      {
         HQs_sq(i, k) = HQs(i, k) * HQs(i, k);
      }
   for (int z = 0; z < nzones; z++)
   {
      DenseMatrix QQ(quad_data.rho0DetJ0w.GetData() + z*nqp, nqp1D, nqp1D);
      mfem::Mult(HQs_sq, QQ, HQ);
      MultABt(HQ, HQs_sq, D);
      FESpace.GetElementDofs(z, dofs);
      for (int j = 0; j < dz.Size(); j++)
      {
         diag[dofs[dof_map[j]]] += dz[j];
      }
   }
}

static void ComputeDiagonal3D(const int height, const int nzones,
                              const QuadratureData &quad_data,
                              const FiniteElementSpace &FESpace,
                              const Tensors1D *tensors1D,
                              Vector &diag)
{
   const TensorBasisElement *fe_H1 =
      dynamic_cast<const TensorBasisElement *>(FESpace.GetTypicalFE());
   const Array<int> &dof_map = fe_H1->GetDofMap();
   const DenseMatrix &HQs = tensors1D->HQshape1D;
   const int ndof1D = HQs.Height(), nqp1D = HQs.Width(),
             nqp = nqp1D * nqp1D * nqp1D;
   DenseMatrix HH_Q(ndof1D * ndof1D, nqp1D), Q_HQ(nqp1D, ndof1D*nqp1D);
   DenseMatrix H_HQ(HH_Q.GetData(), ndof1D, ndof1D*nqp1D);
   Vector dz(ndof1D * ndof1D * ndof1D);
   DenseMatrix D(dz.GetData(), ndof1D*ndof1D, ndof1D);
   Array<int> dofs;
   diag.SetSize(height);
   diag = 0.0;
   DenseMatrix HQs_sq(ndof1D, nqp1D);
   for (int i = 0; i < ndof1D; i++)
      for (int k = 0; k < nqp1D; k++)
      {
         HQs_sq(i, k) = HQs(i, k) * HQs(i, k);
      }
   for (int z = 0; z < nzones; z++)
   {
      DenseMatrix QQ_Q(quad_data.rho0DetJ0w.GetData() + z*nqp,
                       nqp1D * nqp1D, nqp1D);
      for (int k1 = 0; k1 < nqp1D; k1++)
      {
         for (int i2 = 0; i2 < ndof1D; i2++)
         {
            for (int k3 = 0; k3 < nqp1D; k3++)
            {
               Q_HQ(k1, i2 + ndof1D*k3) = 0.0;
               for (int k2 = 0; k2 < nqp1D; k2++)
               {
                  Q_HQ(k1, i2 + ndof1D*k3) +=
                     QQ_Q(k1 + nqp1D*k2, k3) * HQs_sq(i2, k2);
               }
            }
         }
      }
      mfem::Mult(HQs_sq, Q_HQ, H_HQ);
      MultABt(HH_Q, HQs_sq, D);
      FESpace.GetElementDofs(z, dofs);
      for (int j = 0; j < dz.Size(); j++)
      {
         diag[dofs[dof_map[j]]] += dz[j];
      }
   }
}

class PAMassOperator : public Operator
{
private:
#if defined(MFEM_USE_MPI) && defined(MFEM_SEDOV_MPI)
   const MPI_Comm comm;
#endif
   const int dim, nzones;
   const QuadratureData &quad_data;
   FiniteElementSpace &FESpace;
   ParBilinearForm pabf;
   int ess_tdofs_count;
   Array<int> ess_tdofs;
   OperatorPtr massOperator;
   Tensors1D *tensors1D;
public:
   PAMassOperator(Coefficient &Q,
                  const QuadratureData &qd,
                  ParFiniteElementSpace &pfes,
                  const IntegrationRule &ir,
                  Tensors1D *t1D) :
      Operator(pfes.GetTrueVSize()),
#if defined(MFEM_USE_MPI) && defined(MFEM_SEDOV_MPI)
      comm(PFesGetParMeshGetComm0(pfes)),
#endif
      dim(pfes.GetMesh()->Dimension()),
      nzones(pfes.GetMesh()->GetNE()),
      quad_data(qd),
      FESpace(pfes),
      pabf(&pfes),
      ess_tdofs_count(0),
      ess_tdofs(0),
      tensors1D(t1D)
   {
      pabf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      pabf.AddDomainIntegrator(new mfem::MassIntegrator(Q,&ir));
      pabf.Assemble();
      pabf.FormSystemMatrix(mfem::Array<int>(), massOperator);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // FIXME: why is 'x' being modified here (through 'X')?
      Vector X;
      X.NewMemoryAndSize(x.GetMemory(), x.Size(), false);
      if (ess_tdofs_count) { X.SetSubVector(ess_tdofs, 0.0); }
      massOperator->Mult(X, y);
      if (ess_tdofs_count) { y.SetSubVector(ess_tdofs, 0.0); }
   }

   void ComputeDiagonal2D(Vector &diag) const
   {
      return hydrodynamics::ComputeDiagonal2D(FESpace.GetVSize(), nzones,
                                              quad_data, FESpace, tensors1D,
                                              diag);
   }
   void ComputeDiagonal3D(Vector &diag) const
   {
      return hydrodynamics::ComputeDiagonal3D(FESpace.GetVSize(), nzones,
                                              quad_data, FESpace, tensors1D,
                                              diag);
   }

   const Operator *GetProlongation() const override
   { return FESpace.GetProlongationMatrix(); }

   const Operator *GetRestriction() const override
   { return FESpace.GetRestrictionMatrix(); }

   void SetEssentialTrueDofs(Array<int> &dofs)
   {
      ess_tdofs_count = dofs.Size();
      if (ess_tdofs.Size()==0)
      {
         int global_ess_tdofs_count;
         MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                       1, MPI_INT, MPI_SUM, comm);
         MFEM_VERIFY(global_ess_tdofs_count>0, "!(global_ess_tdofs_count>0)");
         ess_tdofs.SetSize(global_ess_tdofs_count);
      }
      if (ess_tdofs_count == 0)
      {
         return;
      }
      ess_tdofs = dofs;
   }

   void EliminateRHS(Vector &b) const
   {
      if (ess_tdofs_count > 0)
      {
         b.SetSubVector(ess_tdofs, 0.0);
      }
   }
};

class DiagonalSolver : public Solver
{
private:
   Vector diag;
   FiniteElementSpace &FESpace;
public:
   DiagonalSolver(FiniteElementSpace &fes)
      : Solver(fes.GetVSize()), diag(), FESpace(fes) { }
   void SetDiagonal(Vector &d)
   {
      const Operator *P = FESpace.GetProlongationMatrix();
      if (P == NULL) { diag = d; return; }
      diag.SetSize(P->Width());
      P->MultTranspose(d, diag);
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      const int N = x.Size();
      auto d_diag = diag.Read();
      auto d_x = x.Read();
      auto d_y = y.Write();
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { d_y[i] = d_x[i] / d_diag[i]; });
   }
   void SetOperator(const Operator&) override { }
};

struct TimingData
{
   StopWatch sw_cgH1, sw_cgL2, sw_force, sw_qdata;
   const HYPRE_BigInt L2dof;
   HYPRE_BigInt H1iter, L2iter, quad_tstep;
   TimingData(const HYPRE_BigInt l2d) :
      L2dof(l2d), H1iter(0), L2iter(0), quad_tstep(0) { }
};

class QUpdate
{
private:
   const int dim, NQ, NE;
   const bool use_viscosity;
   const real_t cfl, gamma;
   TimingData *timer;
   const IntegrationRule &ir;
   ParFiniteElementSpace &H1, &L2;
   const Operator *H1ER;
   const int vdim;
   Vector d_dt_est;
   Vector d_l2_e_quads_data;
   Vector d_h1_v_local_in, d_h1_grad_x_data, d_h1_grad_v_data;
   const QuadratureInterpolator *q1, *q2;
public:
   QUpdate(const int d, const int ne, const bool uv,
           const real_t c, const real_t g, TimingData *t,
           const IntegrationRule &i,
           ParFiniteElementSpace &h1, ParFiniteElementSpace &l2):
      dim(d), NQ(i.GetNPoints()), NE(ne), use_viscosity(uv), cfl(c), gamma(g),
      timer(t), ir(i), H1(h1), L2(l2),
      H1ER(H1.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      vdim(H1.GetVDim()),
      d_dt_est(NE*NQ),
      d_l2_e_quads_data(NE*NQ),
      d_h1_v_local_in(NQ*NE*vdim),
      d_h1_grad_x_data(NQ*NE*vdim*vdim),
      d_h1_grad_v_data(NQ*NE*vdim*vdim),
      q1(H1.GetQuadratureInterpolator(ir)),
      q2(L2.GetQuadratureInterpolator(ir)) { }

   void UpdateQuadratureData(const Vector &S,
                             bool &quad_data_is_current,
                             QuadratureData &quad_data,
                             const Tensors1D *tensors1D);
};

void ComputeRho0DetJ0AndVolume(const int dim,
                               const int NE,
                               const IntegrationRule &ir,
                               ParMesh *mesh,
                               ParFiniteElementSpace &l2_fes,
                               ParGridFunction &rho0,
                               QuadratureData &quad_data,
                               real_t &loc_area)
{
   const int NQ = ir.GetNPoints();
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int flags = GeometricFactors::JACOBIANS|GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(ir, flags);
   Vector rho0Q(NQ*NE);
   rho0Q.UseDevice(true);
   const QuadratureInterpolator *qi = l2_fes.GetQuadratureInterpolator(ir);
   qi->Values(rho0, rho0Q);
   auto W = ir.GetWeights().Read();
   auto R = Reshape(rho0Q.Read(), NQ, NE);
   auto J = Reshape(geom->J.Read(), NQ, dim, dim, NE);
   auto detJ = Reshape(geom->detJ.Read(), NQ, NE);
   auto V = Reshape(quad_data.rho0DetJ0w.Write(), NQ, NE);
   Memory<real_t> &Jinv_m = quad_data.Jac0inv.GetMemory();
   auto invJ = Reshape(Jinv_m.Write(Device::GetDeviceMemoryClass(),
                                    quad_data.Jac0inv.TotalSize()),
                       dim, dim, NQ, NE);
   Vector area(NE*NQ), one(NE*NQ);
   auto A = Reshape(area.Write(), NQ, NE);
   auto O = Reshape(one.Write(), NQ, NE);
   if (dim==2)
   {
      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const int q = qx + qy * Q1D;
               const real_t J11 = J(q,0,0,e);
               const real_t J12 = J(q,1,0,e);
               const real_t J21 = J(q,0,1,e);
               const real_t J22 = J(q,1,1,e);
               const real_t det = detJ(q,e);
               V(q,e) =  W[q] * R(q,e) * det;
               const real_t r_idetJ = 1.0 / det;
               invJ(0,0,q,e) =  J22 * r_idetJ;
               invJ(1,0,q,e) = -J12 * r_idetJ;
               invJ(0,1,q,e) = -J21 * r_idetJ;
               invJ(1,1,q,e) =  J11 * r_idetJ;
               A(q,e) = W[q] * det;
               O(q,e) = 1.0;
            }
         }
      });
   }
   else
   {
      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const int q = qx + (qy + qz * Q1D) * Q1D;
                  const real_t J11 = J(q,0,0,e), J12 = J(q,0,1,e), J13 = J(q,0,2,e);
                  const real_t J21 = J(q,1,0,e), J22 = J(q,1,1,e), J23 = J(q,1,2,e);
                  const real_t J31 = J(q,2,0,e), J32 = J(q,2,1,e), J33 = J(q,2,2,e);
                  const real_t det = detJ(q,e);
                  V(q,e) = W[q] * R(q,e) * det;
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
                  A(q,e) = W[q] * det;
                  O(q,e) = 1.0;
               }
            }
         }
      });
   }
   quad_data.rho0DetJ0w.HostRead();
   loc_area = area * one;
}

class TaylorCoefficient : public Coefficient
{
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);
      return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                  cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
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

template<int dim, int Q1D> static inline
void QKernel(const int nzones,
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
   auto d_weights = weights.Read();
   auto d_Jacobians = Jacobians.Read();
   auto d_rho0DetJ0w = rho0DetJ0w.Read();
   auto d_e_quads = e_quads.Read();
   auto d_grad_v_ext = grad_v_ext.Read();
   auto d_Jac0inv = Read(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWrite();
   auto d_stressJinvT = Write(stressJinvT.GetMemory(),
                              stressJinvT.TotalSize());
   if (dim==2)
   {
      mfem::forall_2D(nzones, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int z)
      {
         constexpr int DIM = dim;
         constexpr int DIM2 = dim*dim;
         real_t Jinv[DIM2];
         real_t stress[DIM2];
         real_t sgrad_v[DIM2];
         real_t eig_val_data[3];
         real_t eig_vec_data[9];
         real_t compr_dir[DIM];
         real_t Jpi[DIM2];
         real_t ph_dir[DIM];
         real_t stressJiT[DIM2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               QBody<dim>(nzones, z, nqp, qx + qy * Q1D,
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
   if (dim==3)
   {
      mfem::forall_3D(nzones, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int z)
      {
         constexpr int DIM = dim;
         constexpr int DIM2 = dim*dim;
         real_t Jinv[DIM2];
         real_t stress[DIM2];
         real_t sgrad_v[DIM2];
         real_t eig_val_data[3];
         real_t eig_vec_data[9];
         real_t compr_dir[DIM];
         real_t Jpi[DIM2];
         real_t ph_dir[DIM];
         real_t stressJiT[DIM2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  QBody<dim>(nzones, z, nqp, qx + Q1D * (qy + qz * Q1D),
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

void QUpdate::UpdateQuadratureData(const Vector &S,
                                   bool &quad_data_is_current,
                                   QuadratureData &quad_data,
                                   const Tensors1D *tensors1D)
{
   if (quad_data_is_current) { return; }
   timer->sw_qdata.Start();
   Vector* S_p = const_cast<Vector*>(&S);
   const int H1_size = H1.GetVSize();
   const int nqp1D = tensors1D->LQshape1D.Width();
   const real_t h1order = (real_t) H1.GetElementOrder(0);
   const real_t infinity = std::numeric_limits<real_t>::infinity();
   GridFunction d_x, d_v, d_e;
   d_x.MakeRef(&H1,*S_p, 0);
   H1ER->Mult(d_x, d_h1_v_local_in);
   q1->SetOutputLayout(QVectorLayout::byVDIM);
   q1->Derivatives(d_h1_v_local_in, d_h1_grad_x_data);
   d_v.MakeRef(&H1,*S_p, H1_size);
   H1ER->Mult(d_v, d_h1_v_local_in);
   q1->Derivatives(d_h1_v_local_in, d_h1_grad_v_data);
   d_e.MakeRef(&L2, *S_p, 2*H1_size);
   q2->SetOutputLayout(QVectorLayout::byVDIM);
   q2->Values(d_e, d_l2_e_quads_data);
   d_dt_est = quad_data.dt_est;
   const int id = (dim<<4) | nqp1D;
   typedef void (*fQKernel)(const int NE, const int NQ,
                            const real_t gamma, const bool use_viscosity,
                            const real_t h0, const real_t h1order,
                            const real_t cfl, const real_t infinity,
                            const Array<real_t> &weights,
                            const Vector &Jacobians, const Vector &rho0DetJ0w,
                            const Vector &e_quads, const Vector &grad_v_ext,
                            const DenseTensor &Jac0inv,
                            Vector &dt_est, DenseTensor &stressJinvT);
   static std::unordered_map<int, fQKernel> qupdate =
   {
      {0x24,&QKernel<2,4>}, //{0x26,&QKernel<2,6>}, {0x28,&QKernel<2,8>},
      {0x34,&QKernel<3,4>}, //{0x36,&QKernel<3,6>}, {0x38,&QKernel<3,8>}
   };
   if (!qupdate[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   qupdate[id](NE, NQ, gamma, use_viscosity, quad_data.h0,
               h1order, cfl, infinity, ir.GetWeights(), d_h1_grad_x_data,
               quad_data.rho0DetJ0w, d_l2_e_quads_data, d_h1_grad_v_data,
               quad_data.Jac0inv, d_dt_est, quad_data.stressJinvT);
   quad_data.dt_est = d_dt_est.Min();
   quad_data_is_current = true;
   timer->sw_qdata.Stop();
   timer->quad_tstep += NE;
}

class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1FESpace, &L2FESpace;
   mutable ParFiniteElementSpace H1compFESpace;
   const int H1Vsize;
   const int H1TVSize;
   const HYPRE_BigInt H1GTVSize;
   const int H1compTVSize;
   const int L2Vsize;
   const int L2TVSize;
   const HYPRE_BigInt L2GTVSize;
   Array<int> block_offsets;
   mutable ParGridFunction x_gf;
   const Array<int> &ess_tdofs;
   const int dim, nzones, l2dofs_cnt, h1dofs_cnt, source_type;
   const real_t cfl;
   const bool use_viscosity;
   const real_t cg_rel_tol;
   const int cg_max_iter;
   const real_t ftz_tol;
   Coefficient *material_pcf;
   mutable ParBilinearForm Mv;
   SparseMatrix Mv_spmat_copy;
   DenseTensor Me, Me_inv;
   const IntegrationRule &integ_rule;
   mutable QuadratureData quad_data;
   mutable bool quad_data_is_current, forcemat_is_assembled;
   Tensors1D T1D;
   mutable MixedBilinearForm Force;
   PAForceOperator *ForcePA;
   PAMassOperator *VMassPA, *EMassPA;
   mutable DiagonalSolver VMassPA_prec;
   CGSolver CG_VMass, CG_EMass, locCG;
   mutable TimingData timer;
   const real_t gamma;
   mutable QUpdate Q;
   mutable Vector X, B, one, rhs, e_rhs;
   mutable ParGridFunction rhs_c_gf, dvc_gf;
   mutable Array<int> c_tdofs[3];

   void UpdateQuadratureData(const Vector &S) const
   {
      return Q.UpdateQuadratureData(S, quad_data_is_current, quad_data, &T1D);
   }

public:
   LagrangianHydroOperator(Coefficient &rho_coeff,
                           const int size,
                           ParFiniteElementSpace &h1_fes,
                           ParFiniteElementSpace &l2_fes,
                           const Array<int> &essential_tdofs,
                           ParGridFunction &rho0,
                           const int source_type_,
                           const real_t cfl_,
                           Coefficient *material_,
                           const bool visc,
                           const real_t cgt,
                           const int cgiter,
                           real_t ftz,
                           const int order_q,
                           const real_t gm,
                           int h1_basis_type):
      TimeDependentOperator(size),
      H1FESpace(h1_fes), L2FESpace(l2_fes),
      H1compFESpace(h1_fes.GetParMesh(), h1_fes.FEColl(), 1),
      H1Vsize(H1FESpace.GetVSize()),
      H1TVSize(H1FESpace.GetTrueVSize()),
      H1GTVSize(H1FESpace.GlobalTrueVSize()),
      H1compTVSize(H1compFESpace.GetTrueVSize()),
      L2Vsize(L2FESpace.GetVSize()),
      L2TVSize(L2FESpace.GetTrueVSize()),
      L2GTVSize(L2FESpace.GlobalTrueVSize()),
      block_offsets(4),
      x_gf(&H1FESpace),
      ess_tdofs(essential_tdofs),
      dim(h1_fes.GetMesh()->Dimension()),
      nzones(h1_fes.GetMesh()->GetNE()),
      l2dofs_cnt(l2_fes.GetTypicalFE()->GetDof()),
      h1dofs_cnt(h1_fes.GetTypicalFE()->GetDof()),
      source_type(source_type_), cfl(cfl_),
      use_viscosity(visc),
      cg_rel_tol(cgt), cg_max_iter(cgiter),ftz_tol(ftz),
      material_pcf(material_),
      Mv(&h1_fes), Mv_spmat_copy(),
      Me(l2dofs_cnt, l2dofs_cnt, nzones),
      Me_inv(l2dofs_cnt, l2dofs_cnt, nzones),
      integ_rule(IntRules.Get(h1_fes.GetMesh()->GetTypicalElementGeometry(),
                              (order_q > 0) ? order_q :
                              3*h1_fes.GetElementOrder(0)
                              + l2_fes.GetElementOrder(0) - 1)),
      quad_data(dim, nzones, integ_rule.GetNPoints()),
      quad_data_is_current(false), forcemat_is_assembled(false),
      T1D(H1FESpace.GetTypicalFE()->GetOrder(), L2FESpace.GetTypicalFE()->GetOrder(),
          int(floor(0.7 + pow(integ_rule.GetNPoints(), 1.0 / dim))),
          h1_basis_type == BasisType::Positive),
      Force(&l2_fes, &h1_fes),
      VMassPA_prec(H1compFESpace),
      CG_VMass(PFesGetParMeshGetComm(H1FESpace)),
      CG_EMass(PFesGetParMeshGetComm(L2FESpace)),
      locCG(),
      timer(L2TVSize),
      gamma(gm),
      Q(dim, nzones, use_viscosity, cfl, gamma,
        &timer, integ_rule, H1FESpace, L2FESpace),
      X(H1compFESpace.GetTrueVSize()),
      B(H1compFESpace.GetTrueVSize()),
      one(L2Vsize),
      rhs(H1Vsize),
      e_rhs(L2Vsize),
      rhs_c_gf(&H1compFESpace),
      dvc_gf(&H1compFESpace)
   {
      block_offsets[0] = 0;
      block_offsets[1] = block_offsets[0] + H1Vsize;
      block_offsets[2] = block_offsets[1] + H1Vsize;
      block_offsets[3] = block_offsets[2] + L2Vsize;
      one.UseDevice(true);
      one = 1.0;
      ForcePA = new PAForceOperator(quad_data, h1_fes,l2_fes, integ_rule);
      VMassPA = new PAMassOperator(rho_coeff, quad_data, H1compFESpace,
                                   integ_rule, &T1D);
      EMassPA = new PAMassOperator(rho_coeff, quad_data, L2FESpace,
                                   integ_rule, &T1D);
      H1FESpace.GetParMesh()->GetNodes()->ReadWrite();
      const int bdr_attr_max = H1FESpace.GetMesh()->bdr_attributes.Max();
      Array<int> ess_bdr(bdr_attr_max);
      for (int c = 0; c < dim; c++)
      {
         ess_bdr = 0; ess_bdr[c] = 1;
         H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
         c_tdofs[c].Read();
      }
      X.UseDevice(true);
      B.UseDevice(true);
      rhs.UseDevice(true);
      e_rhs.UseDevice(true);
      GridFunctionCoefficient rho_coeff_gf(&rho0);
      real_t loc_area = 0.0, glob_area;
      int loc_z_cnt = nzones, glob_z_cnt;
      ParMesh *pm = H1FESpace.GetParMesh();
      ComputeRho0DetJ0AndVolume(dim, nzones, integ_rule,
                                H1FESpace.GetParMesh(),
                                l2_fes, rho0, quad_data, loc_area);
      MPI_Allreduce(&loc_area, &glob_area, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    pm->GetComm());
      MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
      switch (pm->GetTypicalElementGeometry())
      {
         case Geometry::SQUARE:
            quad_data.h0 = sqrt(glob_area / glob_z_cnt); break;
         case Geometry::CUBE:
            quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
         default: MFEM_ABORT("Unknown zone type!");
      }
      quad_data.h0 /= (real_t) H1FESpace.GetElementOrder(0);
      {
         Vector d;
         (dim == 2) ? VMassPA->ComputeDiagonal2D(d) : VMassPA->ComputeDiagonal3D(d);
         VMassPA_prec.SetDiagonal(d);
      }
      CG_VMass.SetPreconditioner(VMassPA_prec);
      CG_VMass.SetOperator(*VMassPA);
      CG_VMass.SetRelTol(cg_rel_tol);
      CG_VMass.SetAbsTol(0.0);
      CG_VMass.SetMaxIter(cg_max_iter);
      CG_VMass.SetPrintLevel(0);

      CG_EMass.SetOperator(*EMassPA);
      CG_EMass.iterative_mode = false;
      CG_EMass.SetRelTol(1e-8);
      CG_EMass.SetAbsTol(1e-8 * std::numeric_limits<real_t>::epsilon());
      CG_EMass.SetMaxIter(200);
      CG_EMass.SetPrintLevel(-1);
   }

   ~LagrangianHydroOperator() override
   {
      delete EMassPA;
      delete VMassPA;
      delete ForcePA;
   }

   void Mult(const Vector &S, Vector &dS_dt) const override
   {
      UpdateMesh(S);
      Vector* sptr = const_cast<Vector*>(&S);
      ParGridFunction v;
      const int VsizeH1 = H1FESpace.GetVSize();
      v.MakeRef(&H1FESpace, *sptr, VsizeH1);
      ParGridFunction dx;
      dx.MakeRef(&H1FESpace, dS_dt, 0);
      dx = v;
      SolveVelocity(S, dS_dt);
      SolveEnergy(S, v, dS_dt);
      quad_data_is_current = false;
   }

   MemoryClass GetMemoryClass() const override  { return Device::GetDeviceMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      ParGridFunction dv;
      dv.MakeRef(&H1FESpace, dS_dt, H1Vsize);
      dv = 0.0;
      timer.sw_force.Start();
      ForcePA->Mult(one, rhs);
      if (ftz_tol>0.0)
      {
         for (int i = 0; i < H1Vsize; i++)
         {
            if (fabs(rhs[i]) < ftz_tol)
            {
               rhs[i] = 0.0;
            }
         }
      }
      timer.sw_force.Stop();
      rhs.Neg();
      const int size = H1compFESpace.GetVSize();
      const Operator *Pconf = H1compFESpace.GetProlongationMatrix();
      const Operator *Rconf = H1compFESpace.GetRestrictionMatrix();
      PAMassOperator *kVMassPA = VMassPA;
      for (int c = 0; c < dim; c++)
      {
         dvc_gf.MakeRef(&H1compFESpace, dS_dt, H1Vsize + c*size);
         rhs_c_gf.MakeRef(&H1compFESpace, rhs, c*size);
         if (Pconf) { Pconf->MultTranspose(rhs_c_gf, B); }
         else { B = rhs_c_gf; }
         if (Rconf) { Rconf->Mult(dvc_gf, X); }
         else { X = dvc_gf; }
         kVMassPA->SetEssentialTrueDofs(c_tdofs[c]);
         kVMassPA->EliminateRHS(B);
         timer.sw_cgH1.Start();
         CG_VMass.Mult(B, X);
         timer.sw_cgH1.Stop();
         timer.H1iter += CG_VMass.GetNumIterations();
         if (Pconf) { Pconf->Mult(X, dvc_gf); }
         else { dvc_gf = X; }
         dvc_gf.GetMemory().SyncAlias(dS_dt.GetMemory(), dvc_gf.Size());
      }
   }

   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      ParGridFunction de;
      de.MakeRef(&L2FESpace, dS_dt, H1Vsize*2);
      de = 0.0;
      LinearForm *e_source = NULL;
      MFEM_VERIFY(source_type!=1,"");
      Array<int> l2dofs;
      timer.sw_force.Start();
      ForcePA->MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
      timer.sw_cgL2.Start();
      CG_EMass.Mult(e_rhs, de);
      timer.sw_cgL2.Stop();
      const int cg_num_iter = CG_EMass.GetNumIterations();
      timer.L2iter += (cg_num_iter==0) ? 1 : cg_num_iter;
      de.GetMemory().SyncAlias(dS_dt.GetMemory(), de.Size());
      delete e_source;
   }

   void UpdateMesh(const Vector &S) const
   {
      Vector* sptr = const_cast<Vector*>(&S);
      x_gf.MakeRef(&H1FESpace, *sptr, 0);
      H1FESpace.GetParMesh()->NewNodes(x_gf, false);
   }

   real_t GetTimeStepEstimate(const Vector &S) const
   {
      UpdateMesh(S);
      UpdateQuadratureData(S);
      real_t glob_dt_est;
      MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN,
                    H1FESpace.GetParMesh()->GetComm());
      return glob_dt_est;
   }

   void ResetTimeStepEstimate() const
   {
      quad_data.dt_est = std::numeric_limits<real_t>::infinity();
   }

   void ResetQuadratureData() const { quad_data_is_current = false; }
};
} // namespace hydrodynamics

int sedov(int myid, int argc, char *argv[])
{
   int dim = 3;
   const int problem = 1;
   const char *mesh_file = "none";
   int rs_levels = 0;
   const int rp_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 4;
   real_t t_final = 0.6;
   real_t cfl = 0.5;
   real_t cg_tol = 1e-14;
   real_t ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   real_t blast_energy = 0.25;
   real_t blast_position[] = {0.0, 0.0, 0.0};

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&cxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return -1;
   }
   Mesh mesh;
   if (strncmp(mesh_file, "none", 4))
   {
      mesh = Mesh::LoadFromFile(mesh_file, true, true);
      dim = mesh.Dimension();
   }
   else
   {
      if (dim == 2)
      {
         constexpr Element::Type QUAD = Element::QUADRILATERAL;
         mesh = Mesh::MakeCartesian2D(2, 2, QUAD, true);
         const int NBE = mesh.GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh.GetBdrElement(b);
            MFEM_ASSERT(bel->GetType() == Element::SEGMENT, "");
            const int attr = (b < NBE/2) ? 2 : 1;
            bel->SetAttribute(attr);
         }
      }
      if (dim == 3)
      {
         mesh = Mesh::MakeCartesian3D(2, 2, 2,Element::HEXAHEDRON);
         const int NBE = mesh.GetNBE();
         MFEM_ASSERT(NBE==24,"");
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh.GetBdrElement(b);
            MFEM_ASSERT(bel->GetType() == Element::QUADRILATERAL, "");
            const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
            bel->SetAttribute(attr);
         }
      }
   }
   dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   ParMesh *pmesh = NULL;
#if defined(MFEM_USE_MPI) && defined(MFEM_SEDOV_MPI)
   pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
#else
   pmesh = new Mesh(mesh);
#endif
   mesh.Clear();
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }
   int nzones = pmesh->GetNE(), nzones_min, nzones_max;
   MPI_Reduce(&nzones, &nzones_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&nzones, &nzones_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << nzones_min << " " << nzones_max << endl; }

   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
   Array<int> ess_tdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), tdofs1d;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
         ess_tdofs.Append(tdofs1d);
      }
   }
   ODESolver *ode_solver = new RK4Solver;
   const HYPRE_BigInt H1GTVSize = H1FESpace.GlobalTrueVSize();
   const HYPRE_BigInt L2GTVSize = L2FESpace.GlobalTrueVSize();
   const int H1Vsize = H1FESpace.GetVSize();
   const int L2Vsize = L2FESpace.GetVSize();
   if (myid == 0)
   {
      cout << "Number of local/global kinematic (position, velocity) dofs: "
           << H1Vsize << "/" << H1GTVSize << endl;
      cout << "Number of local/global specific internal energy dofs: "
           << L2Vsize << "/" << L2GTVSize << endl;
   }
   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + H1Vsize;
   true_offset[2] = true_offset[1] + H1Vsize;
   true_offset[3] = true_offset[2] + L2Vsize;
   BlockVector S(true_offset, Device::GetDeviceMemoryType());
   S.UseDevice(true);
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
   v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
   e_gf.MakeRef(&L2FESpace, S, true_offset[2]);
   pmesh->SetNodalGridFunction(&x_gf);
   x_gf.SyncAliasMemory(S);
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   v_gf.SyncAliasMemory(S);
   ParGridFunction rho(&L2FESpace);
   FunctionCoefficient rho_fct_coeff(rho0);
   ConstantCoefficient rho_coeff(1.0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
   l2_rho.ProjectCoefficient(rho_fct_coeff);
   rho.ProjectGridFunction(l2_rho);
   DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                            blast_position[2], blast_energy);
   l2_e.ProjectCoefficient(e_coeff);
   e_gf.ProjectGridFunction(l2_e);
   e_gf.SyncAliasMemory(S);
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma);
   mat_gf.ProjectCoefficient(mat_coeff);
   GridFunctionCoefficient *mat_gf_coeff = new GridFunctionCoefficient(&mat_gf);
   const int source = 0; bool visc = true;

   mfem::hydrodynamics::LagrangianHydroOperator oper(rho_coeff, S.Size(),
                                                     H1FESpace, L2FESpace,
                                                     ess_tdofs, rho, source,
                                                     cfl, mat_gf_coeff,
                                                     visc, cg_tol, cg_max_iter,
                                                     ftz_tol, order_q,
                                                     gamma(S),
                                                     H1FEC.GetBasisType());

   ode_solver->Init(oper);
   oper.ResetTimeStepEstimate();
   real_t t = 0.0, dt = oper.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   int checks = 0;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      oper.ResetTimeStepEstimate();
      ode_solver->Step(S, t, dt);
      steps++;
      const real_t dt_est = oper.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         dt *= 0.85;
         if (dt < numeric_limits<real_t>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         oper.ResetQuadratureData();
         if (myid == 0) { cout << "Repeating step " << ti << endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);
      pmesh->NewNodes(x_gf, false);
      if (last_step || (ti % vis_steps) == 0)
      {
         real_t loc_norm = e_gf * e_gf, tot_norm;
         MPI_Allreduce(&loc_norm, &tot_norm, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                       pmesh->GetComm());
         if (myid == 0)
         {
            const real_t sqrt_tot_norm = sqrt(tot_norm);
            cout << fixed;
            cout << "step " << setw(5) << ti
                 << ",\tt = " << setw(5) << setprecision(4) << t
                 << ",\tdt = " << setw(5) << setprecision(6) << dt
                 << ",\t|e| = " << setprecision(10)
                 << sqrt_tot_norm;
            cout << endl;
         }
      }
      REQUIRE(problem==1);
      real_t loc_norm = e_gf * e_gf, tot_norm;
      MPI_Allreduce(&loc_norm, &tot_norm, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    pmesh->GetComm());
      const real_t stm = sqrt(tot_norm);
      //printf("\n\033[33m%.15e\033[m", stm); fflush(0);
      REQUIRE((rs_levels==0 || rs_levels==1));
      REQUIRE(rp_levels==0);
      REQUIRE(order_v==2);
      REQUIRE(order_e==1);
      REQUIRE(ode_solver_type==4);
      REQUIRE(t_final==MFEM_Approx(0.6));
      REQUIRE(cfl==MFEM_Approx(0.5));
      REQUIRE(cg_tol==MFEM_Approx(1.e-14));
      if (dim==2)
      {
         const real_t p1_05[2] = {3.508254945225794e+00,
                                  1.403249766367977e+01
                                 };
         const real_t p1_15[2] = {2.756444596823211e+00,
                                  1.104093401469385e+01
                                 };
         if (ti==05) {checks++; REQUIRE(stm==MFEM_Approx(p1_05[rs_levels]));}
         if (ti==15) {checks++; REQUIRE(stm==MFEM_Approx(p1_15[rs_levels]));}
      }
      if (dim==3)
      {
         const real_t p1_05[2] = {1.339163718592567e+01,
                                  1.071277540097426e+02
                                 };
         const real_t p1_28[2] = {7.521073677398005e+00,
                                  5.985720905709158e+01
                                 };
         if (ti==05) {checks++; REQUIRE(stm==MFEM_Approx(p1_05[rs_levels]));}
         if (ti==28) {checks++; REQUIRE(stm==MFEM_Approx(p1_28[rs_levels]));}
      }
   }
   REQUIRE(checks==2);
   REQUIRE(ode_solver_type==4);
   steps *= 4;
   //oper.PrintTimingData(myid, steps, fom);
   delete ode_solver;
   delete pmesh;
   delete mat_gf_coeff;
   return 0;
}
} // namespace mfem

static int argn(const char *argv[], int argc =0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

static void sedov_tests(int myid)
{
   const char *argv2D[]= { "sedov_tests", "-d", "2", nullptr };
   REQUIRE(sedov(myid, argn(argv2D), const_cast<char**>(argv2D))==0);

   const char *argv2Drs1[]= { "sedov_tests", "-d", "2",
                              "-rs", "1", "-ms", "20",
                              nullptr
                            };
   REQUIRE(sedov(myid, argn(argv2Drs1), const_cast<char**>(argv2Drs1))==0);

   const char *argv3D[]= { "sedov_tests", "-d", "3", nullptr };
   REQUIRE(sedov(myid, argn(argv3D), const_cast<char**>(argv3D))==0);

   const char *argv3Drs1[]= { "sedov_tests", "-d", "3",
                              "-rs", "1", "-ms", "28",
                              nullptr
                            };
   REQUIRE(sedov(myid, argn(argv3Drs1), const_cast<char**>(argv3Drs1))==0);

}

#ifdef MFEM_SEDOV_MPI
TEST_CASE("Sedov", "[Sedov], [Parallel]")
{
   sedov_tests(Mpi::WorldRank());
}
#else
TEST_CASE("Sedov", "[Sedov]")
{
   sedov_tests(0);
}
#endif

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   std::cout << "\nThe Sedov unit tests are not supported in single"
             " precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

#ifdef MFEM_SEDOV_MPI
   mfem::Mpi::Init();
   mfem::Hypre::Init();
#endif
#ifdef MFEM_SEDOV_DEVICE
   Device device(MFEM_SEDOV_DEVICE);
#else
   Device device("cpu"); // make sure hypre runs on CPU, if possible
#endif
   device.Print();

#if defined(MFEM_SEDOV_MPI) && defined(MFEM_DEBUG) && defined(MFEM_SEDOV_DEVICE)
   if (HypreUsingGPU() && !strcmp(MFEM_SEDOV_DEVICE, "debug"))
   {
      cout << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this unit test\n"
           << "is NOT supported with the GPU version of hypre.\n\n";
      return MFEM_SKIP_RETURN_VALUE;
   }
#endif

#ifdef MFEM_SEDOV_MPI
   return RunCatchSession(argc, argv, {"[Parallel]"}, Root());
#else
   // Exclude parallel tests.
   return RunCatchSession(argc, argv, {"~[Parallel]"});
#endif
}
