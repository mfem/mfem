// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../general/debug.hpp"
#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

#ifdef MFEM_USE_SYCL

using namespace sycl;

namespace mfem
{

// Half of B and G are stored in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D.
static MFEM_HOST_DEVICE inline int qi(const int q, const int d, const int Q)
{
   return (q<=d) ? q : Q-1-q;
}

static MFEM_HOST_DEVICE inline int dj(const int q, const int d, const int D)
{
   return (q<=d) ? d : D-1-d;
}

static MFEM_HOST_DEVICE inline int qk(const int q, const int d, const int Q)
{
   return (q<=d) ? Q-1-q : q;
}

static MFEM_HOST_DEVICE inline int dl(const int q, const int d, const int D)
{
   return (q<=d) ? D-1-d : d;
}

static MFEM_HOST_DEVICE inline double sign(const int q, const int d)
{
   return (q<=d) ? -1.0 : 1.0;
}

// PA Diffusion Apply 3D kernel
template<int D1D, int Q1D>
static void NDPADiffusionApply3D(const int NE,
                                 const double * __restrict d_b,
                                 const double * __restrict d_g,
                                 const double * __restrict d_bt,
                                 const double * __restrict d_gt,
                                 const double * __restrict d_d,
                                 const double * __restrict d_x,
                                 double * __restrict d_y)
{
   dbg("D1D:%d, Q1D:%d",D1D,Q1D);
   constexpr size_t B_sz = Q1D*D1D;
   const size_t D_sz = Q1D*Q1D*Q1D * 6 * NE;
   const size_t X_sz = D1D*D1D*D1D * NE;
   const size_t Y_sz = D1D*D1D*D1D * NE;

   buffer<const double, 1> b_buf(d_b, range<1> {B_sz});
   buffer<const double, 1> g_buf(d_g, range<1> {B_sz});
   buffer<const double, 1> d_buf(d_d, range<1> {D_sz});
   buffer<const double, 1> x_buf(d_x, range<1> {X_sz});
   buffer<double, 1> y_buf(d_y, range<1> {Y_sz});
   {
      SYCL_KERNEL( // Q.submit
      {
         const auto b_ = b_buf.get_access<access::mode::read>(h);
         const auto g_ = g_buf.get_access<access::mode::read>(h);
         const auto d_ = d_buf.get_access<access::mode::read>(h);
         const auto x_ = x_buf.get_access<access::mode::read>(h);
         auto y_ = y_buf.get_access<access::mode::write>(h);

         const auto b = Reshape(b_, Q1D, D1D);
         const auto g = Reshape(g_, Q1D, D1D);
         const auto d = Reshape(d_, Q1D, Q1D, Q1D, 6, NE);
         const auto x = Reshape(x_, D1D, D1D, D1D, NE);
         auto y = Reshape(y_, D1D, D1D, D1D, NE);

         SYCL_FORALL_3D(e, 8/*NE*/, Q1D, Q1D, Q1D,
         {
            const int i = itm.get_local_id(0);
            const int j = itm.get_local_id(1);
            const int k = itm.get_local_id(2);
            if (i==0 && j==0 && k==0) { kout << "\033[31;1m"<<e<<" \033[m"; }
            //for (int i=itm.get_local_id(0); i<Q1D; i+=itm.get_local_range(0)) {}
            /*
                        constexpr int MQ1 = Q1D;
                        constexpr int MD1 = D1D;
                        constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
                        MFEM_SHARED double sBG[MQ1*MD1];
                        double (*B)[MD1] = (double (*)[MD1]) sBG;
                        double (*G)[MD1] = (double (*)[MD1]) sBG;
                        double (*Bt)[MQ1] = (double (*)[MQ1]) sBG;
                        double (*Gt)[MQ1] = (double (*)[MQ1]) sBG;
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

                        //SYCL_FOREACH();

                        //for (int i=itm.get_local_id(0); i<Q1D; i+=itm.get_local_range(0)) {}

                        SYCL_FOREACH_THREAD(dy,1,D1D)
                        {
                           SYCL_FOREACH_THREAD(dx,0,D1D)
                           {
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz)
                              {
                                 X[dz][dy][dx] = x(dx,dy,dz,e);
                              }
                           }
                           SYCL_FOREACH_THREAD(qx,0,Q1D)
                           {
                              const int i = qi(qx,dy,Q1D);
                              const int j = dj(qx,dy,D1D);
                              const int k = qk(qx,dy,Q1D);
                              const int l = dl(qx,dy,D1D);
                              B[i][j] = b(qx,dy);
                              G[k][l] = g(qx,dy) * sign(qx,dy);
                           }
                        }
                        SYCL_SYNC_THREAD;

                        SYCL_FOREACH_THREAD(dy,1,D1D)
                        {
                           SYCL_FOREACH_THREAD(qx,0,Q1D)
                           {
                              double u[D1D], v[D1D];
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = 0.0; }
                              MFEM_UNROLL(MD1)
                              for (int dx = 0; dx < D1D; ++dx)
                              {
                                 const int i = qi(qx,dx,Q1D);
                                 const int j = dj(qx,dx,D1D);
                                 const int k = qk(qx,dx,Q1D);
                                 const int l = dl(qx,dx,D1D);
                                 const double s = sign(qx,dx);
                                 MFEM_UNROLL(MD1)
                                 for (int dz = 0; dz < D1D; ++dz)
                                 {
                                    const double coords = X[dz][dy][dx];
                                    u[dz] += coords * B[i][j];
                                    v[dz] += coords * G[k][l] * s;
                                 }
                              }
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz)
                              {
                                 DDQ0[dz][dy][qx] = u[dz];
                                 DDQ1[dz][dy][qx] = v[dz];
                              }
                           }
                        }
                        SYCL_SYNC_THREAD;
                        SYCL_FOREACH_THREAD(qy,1,Q1D)
                        {
                           SYCL_FOREACH_THREAD(qx,0,Q1D)
                           {
                              double u[D1D], v[D1D], w[D1D];
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = w[dz] = 0.0; }
                              MFEM_UNROLL(MD1)
                              for (int dy = 0; dy < D1D; ++dy)
                              {
                                 const int i = qi(qy,dy,Q1D);
                                 const int j = dj(qy,dy,D1D);
                                 const int k = qk(qy,dy,Q1D);
                                 const int l = dl(qy,dy,D1D);
                                 const double s = sign(qy,dy);
                                 MFEM_UNROLL(MD1)
                                 for (int dz = 0; dz < D1D; dz++)
                                 {
                                    u[dz] += DDQ1[dz][dy][qx] * B[i][j];
                                    v[dz] += DDQ0[dz][dy][qx] * G[k][l] * s;
                                    w[dz] += DDQ0[dz][dy][qx] * B[i][j];
                                 }
                              }
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; dz++)
                              {
                                 DQQ0[dz][qy][qx] = u[dz];
                                 DQQ1[dz][qy][qx] = v[dz];
                                 DQQ2[dz][qy][qx] = w[dz];
                              }
                           }
                        }
                        SYCL_SYNC_THREAD;
                        SYCL_FOREACH_THREAD(qy,1,Q1D)
                        {
                           SYCL_FOREACH_THREAD(qx,0,Q1D)
                           {
                              double u[Q1D], v[Q1D], w[Q1D];
                              MFEM_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; qz++) { u[qz] = v[qz] = w[qz] = 0.0; }
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz)
                              {
                                 MFEM_UNROLL(MQ1)
                                 for (int qz = 0; qz < Q1D; qz++)
                                 {
                                    const int i = qi(qz,dz,Q1D);
                                    const int j = dj(qz,dz,D1D);
                                    const int k = qk(qz,dz,Q1D);
                                    const int l = dl(qz,dz,D1D);
                                    const double s = sign(qz,dz);
                                    u[qz] += DQQ0[dz][qy][qx] * B[i][j];
                                    v[qz] += DQQ1[dz][qy][qx] * B[i][j];
                                    w[qz] += DQQ2[dz][qy][qx] * G[k][l] * s;
                                 }
                              }
                              MFEM_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; qz++)
                              {
                                 const double O11 = d(qx,qy,qz,0,e);
                                 const double O12 = d(qx,qy,qz,1,e);
                                 const double O13 = d(qx,qy,qz,2,e);
                                 const double O21 = O12;
                                 const double O22 = d(qx,qy,qz,3,e);
                                 const double O23 = d(qx,qy,qz,4,e);
                                 const double O31 = O13;
                                 const double O32 = O23;
                                 const double O33 = d(qx,qy,qz,5,e);
                                 const double gX = u[qz];
                                 const double gY = v[qz];
                                 const double gZ = w[qz];
                                 QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
                                 QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ);
                                 QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);
                              }
                           }
                        }
                        SYCL_SYNC_THREAD;
                        SYCL_FOREACH_THREAD(d,1,D1D)
                        {
                           SYCL_FOREACH_THREAD(q,0,Q1D)
                           {
                              const int i = qi(q,d,Q1D);
                              const int j = dj(q,d,D1D);
                              const int k = qk(q,d,Q1D);
                              const int l = dl(q,d,D1D);
                              Bt[j][i] = b(q,d);
                              Gt[l][k] = g(q,d) * sign(q,d);
                           }
                        }
                        SYCL_SYNC_THREAD;
                        SYCL_FOREACH_THREAD(qy,1,Q1D)
                        {
                           SYCL_FOREACH_THREAD(dx,0,D1D)
                           {
                              double u[Q1D], v[Q1D], w[Q1D];
                              MFEM_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
                              MFEM_UNROLL(MQ1)
                              for (int qx = 0; qx < Q1D; ++qx)
                              {
                                 const int i = qi(qx,dx,Q1D);
                                 const int j = dj(qx,dx,D1D);
                                 const int k = qk(qx,dx,Q1D);
                                 const int l = dl(qx,dx,D1D);
                                 const double s = sign(qx,dx);
                                 MFEM_UNROLL(MQ1)
                                 for (int qz = 0; qz < Q1D; ++qz)
                                 {
                                    u[qz] += QQQ0[qz][qy][qx] * Gt[l][k] * s;
                                    v[qz] += QQQ1[qz][qy][qx] * Bt[j][i];
                                    w[qz] += QQQ2[qz][qy][qx] * Bt[j][i];
                                 }
                              }
                              MFEM_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz)
                              {
                                 QQD0[qz][qy][dx] = u[qz];
                                 QQD1[qz][qy][dx] = v[qz];
                                 QQD2[qz][qy][dx] = w[qz];
                              }
                           }
                        }
                        SYCL_SYNC_THREAD;
                        SYCL_FOREACH_THREAD(dy,1,D1D)
                        {
                           SYCL_FOREACH_THREAD(dx,0,D1D)
                           {
                              double u[Q1D], v[Q1D], w[Q1D];
                              MFEM_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
                              MFEM_UNROLL(MQ1)
                              for (int qy = 0; qy < Q1D; ++qy)
                              {
                                 const int i = qi(qy,dy,Q1D);
                                 const int j = dj(qy,dy,D1D);
                                 const int k = qk(qy,dy,Q1D);
                                 const int l = dl(qy,dy,D1D);
                                 const double s = sign(qy,dy);
                                 MFEM_UNROLL(MQ1)
                                 for (int qz = 0; qz < Q1D; ++qz)
                                 {
                                    u[qz] += QQD0[qz][qy][dx] * Bt[j][i];
                                    v[qz] += QQD1[qz][qy][dx] * Gt[l][k] * s;
                                    w[qz] += QQD2[qz][qy][dx] * Bt[j][i];
                                 }
                              }
                              MFEM_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz)
                              {
                                 QDD0[qz][dy][dx] = u[qz];
                                 QDD1[qz][dy][dx] = v[qz];
                                 QDD2[qz][dy][dx] = w[qz];
                              }
                           }
                        }
                        SYCL_SYNC_THREAD;
                        SYCL_FOREACH_THREAD(dy,1,D1D)
                        {
                           SYCL_FOREACH_THREAD(dx,0,D1D)
                           {
                              double u[D1D], v[D1D], w[D1D];
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz) { u[dz] = v[dz] = w[dz] = 0.0; }
                              MFEM_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz)
                              {
                                 MFEM_UNROLL(MD1)
                                 for (int dz = 0; dz < D1D; ++dz)
                                 {
                                    const int i = qi(qz,dz,Q1D);
                                    const int j = dj(qz,dz,D1D);
                                    const int k = qk(qz,dz,Q1D);
                                    const int l = dl(qz,dz,D1D);
                                    const double s = sign(qz,dz);
                                    u[dz] += QDD0[qz][dy][dx] * Bt[j][i];
                                    v[dz] += QDD1[qz][dy][dx] * Bt[j][i];
                                    w[dz] += QDD2[qz][dy][dx] * Gt[l][k] * s;
                                 }
                              }
                              MFEM_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz)
                              {
                                 y(dx,dy,dz,e) += 0.0;//(u[dz] + v[dz] + w[dz]);
                              }
                           }
                        }
                                             */
         }); // MFEM_FORALL
      }); // MFEM_KERNEL (Q.submit)
   }
}

// SYCL ND PA Diffusion Apply 3D kernel
void SyclNDPADiffusionApply3D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const bool symm,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Array<double> &gt,
                              const Vector &d,
                              const Vector &x,
                              Vector &y)
{
   MFEM_VERIFY(symm, "Only symmetric is supported!");

   const double *B = b.HostRead();
   const double *G = g.HostRead();
   const double *Bt = bt.HostRead();
   const double *Gt = gt.HostRead();
   const double *D = d.HostRead();
   const double *X = x.HostRead();
   double *Y = y.HostReadWrite();

   const int ID = (D1D << 4) | Q1D;

   switch (ID)
   {
      case 0x23: return NDPADiffusionApply3D<2,3>(NE,B,G,Bt,Gt,D,X,Y);
      /*
      case 0x34: return NDPADiffusionApply3D<3,4>(NE,B,G,Bt,Gt,D,X,Y);
      case 0x45: return NDPADiffusionApply3D<4,5>(NE,B,G,Bt,Gt,D,X,Y);
      case 0x56: return NDPADiffusionApply3D<5,6>(NE,B,G,Bt,Gt,D,X,Y);
      case 0x67: return NDPADiffusionApply3D<6,7>(NE,B,G,Bt,Gt,D,X,Y);
      case 0x78: return NDPADiffusionApply3D<7,8>(NE,B,G,Bt,Gt,D,X,Y);
      case 0x89: return NDPADiffusionApply3D<8,9>(NE,B,G,Bt,Gt,D,X,Y);
      case 0x9A: return NDPADiffusionApply3D<9,10>(NE,B,G,Bt,Gt,D,X,Y);
      */
      default:   MFEM_ABORT("Order D1D:"<<D1D<<", Q1D:"<<Q1D<<"!");
   }
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem

#endif // MFEM_USE_SYCL
