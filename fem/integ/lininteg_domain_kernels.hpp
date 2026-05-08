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

#ifndef MFEM_LININTEG_DOMAIN_KERNELS_HPP
#define MFEM_LININTEG_DOMAIN_KERNELS_HPP

#include "../../fem/kernels.hpp"
#include "../../general/forall.hpp"
#include "../fem.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
static void DLFEvalAssemble1D(const int vdim, const int ne, const int d,
                              const int q, const int map_type,
                              const int *markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   {
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_VERIFY(q <= Q, "");
      MFEM_VERIFY(d <= D, "");
   }

   const auto F = coeff.Read();
   const auto B = Reshape(b, q, d);
   const auto DETJ = Reshape(detj, q, ne);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F, vdim, 1, 1) : Reshape(F, vdim, q, ne);
   auto Y = Reshape(y, d, vdim, ne);

   mfem::forall_2D(ne, d, 1, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         return;
      } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sBt[Q * D];
      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D, Q>(d, q, B, sBt);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val = C(c, 0, 0);
         MFEM_FOREACH_THREAD(dx, x, d)
         {
            real_t u = 0;
            for (int qx = 0; qx < q; ++qx)
            {
               const real_t detJ =
                  (map_type == FiniteElement::VALUE) ? DETJ(qx, e) : 1.0;
               const real_t coeff_val = cst ? cst_val : C(c, qx, e);
               u += weights[qx] * coeff_val * detJ * Bt(dx, qx);
            }
            Y(dx, c, e) += u;
         }
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void DLFEvalAssemble2D(const int vdim, const int ne, const int d,
                              const int q, const int map_type,
                              const int *markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   {
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_VERIFY(q <= Q, "");
      MFEM_VERIFY(d <= D, "");
   }

   const auto F = coeff.Read();
   const auto B = Reshape(b, q, d);
   const auto DETJ = Reshape(detj, q, q, ne);
   const auto W = Reshape(weights, q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F, vdim, 1, 1, 1) : Reshape(F, vdim, q, q, ne);
   auto Y = Reshape(y, d, d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         return;
      } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sBt[Q * D];
      MFEM_SHARED real_t sQQ[Q * Q];
      MFEM_SHARED real_t sQD[Q * D];

      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D, Q>(d, q, B, sBt);

      const DeviceMatrix QQ(sQQ, q, q);
      const DeviceMatrix QD(sQD, q, d);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val = C(c, 0, 0, 0);
         MFEM_FOREACH_THREAD(x, x, q)
         {
            MFEM_FOREACH_THREAD(y, y, q)
            {
               const real_t detJ =
                  (map_type == FiniteElement::VALUE) ? DETJ(x, y, e) : 1.0;
               const real_t coeff_val = cst ? cst_val : C(c, x, y, e);
               QQ(y, x) = W(x, y) * coeff_val * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy, y, q)
         {
            MFEM_FOREACH_THREAD(dx, x, d)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < q; ++qx)
               {
                  u += QQ(qy, qx) * Bt(dx, qx);
               }
               QD(qy, dx) = u;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy, y, d)
         {
            MFEM_FOREACH_THREAD(dx, x, d)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < q; ++qy)
               {
                  u += QD(qy, dx) * Bt(dy, qy);
               }
               Y(dx, dy, c, e) += u;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void DLFEvalAssemble3D(const int vdim, const int ne, const int d,
                              const int q, const int map_type,
                              const int* markers, const real_t *b,
                              const real_t *detj, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   {
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      MFEM_VERIFY(q <= Q, "");
      MFEM_VERIFY(d <= D, "");
   }

   const auto F = coeff.Read();
   const auto B = Reshape(b, q, d);
   const auto DETJ = Reshape(detj, q, q, q, ne);
   const auto W = Reshape(weights, q, q, q);
   const bool cst_coeff = coeff.Size() == vdim;
   const auto C =
      cst_coeff ? Reshape(F, vdim, 1, 1, 1, 1) : Reshape(F, vdim, q, q, q, ne);

   auto Y = Reshape(y, d, d, d, vdim, ne);

   mfem::forall_2D(ne, q, q, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         return;
      } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQD = (Q >= D) ? Q : D;

      real_t u[D];

      MFEM_SHARED real_t sBt[Q * D];
      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D, Q>(d, q, B, sBt);

      MFEM_SHARED real_t sQQQ[MQD * MQD * MQD];
      const DeviceCube QQQ(sQQQ, MQD, MQD, MQD);

      for (int c = 0; c < vdim; ++c)
      {
         const real_t cst_val = C(c, 0, 0, 0, 0);
         MFEM_FOREACH_THREAD(x, x, q)
         {
            MFEM_FOREACH_THREAD(y, y, q)
            {
               for (int z = 0; z < q; ++z)
               {
                  const real_t detJ = (map_type == FiniteElement::VALUE)
                                      ? DETJ(x, y, z, e)
                                      : 1.0;
                  const real_t coeff_val =
                     cst_coeff ? cst_val : C(c, x, y, z, e);
                  QQQ(z, y, x) = W(x, y, z) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qx, x, q)
         {
            MFEM_FOREACH_THREAD(qy, y, q)
            {
               for (int dz = 0; dz < d; ++dz)
               {
                  u[dz] = 0.0;
               }
               for (int qz = 0; qz < q; ++qz)
               {
                  const real_t ZYX = QQQ(qz, qy, qx);
                  for (int dz = 0; dz < d; ++dz)
                  {
                     u[dz] += ZYX * Bt(dz, qz);
                  }
               }
               for (int dz = 0; dz < d; ++dz)
               {
                  QQQ(dz, qy, qx) = u[dz];
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz, y, d)
         {
            MFEM_FOREACH_THREAD(qx, x, q)
            {
               for (int dy = 0; dy < d; ++dy)
               {
                  u[dy] = 0.0;
               }
               for (int qy = 0; qy < q; ++qy)
               {
                  const real_t zYX = QQQ(dz, qy, qx);
                  for (int dy = 0; dy < d; ++dy)
                  {
                     u[dy] += zYX * Bt(dy, qy);
                  }
               }
               for (int dy = 0; dy < d; ++dy)
               {
                  QQQ(dz, dy, qx) = u[dy];
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz, y, d)
         {
            MFEM_FOREACH_THREAD(dy, x, d)
            {
               for (int dx = 0; dx < d; ++dx)
               {
                  u[dx] = 0.0;
               }
               for (int qx = 0; qx < q; ++qx)
               {
                  const real_t zyX = QQQ(dz, dy, qx);
                  for (int dx = 0; dx < d; ++dx)
                  {
                     u[dx] += zyX * Bt(dx, qx);
                  }
               }
               for (int dx = 0; dx < d; ++dx)
               {
                  Y(dx, dy, dz, c, e) += u[dx];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template <int DIM, int T_D1D, int T_Q1D>
DomainLFIntegrator::AssembleKernelType
DomainLFIntegrator::AssembleKernels::Kernel()
{
   if constexpr (DIM == 1) { return DLFEvalAssemble1D<T_D1D, T_Q1D>; }
   if constexpr (DIM == 2) { return DLFEvalAssemble2D<T_D1D, T_Q1D>; }
   if constexpr (DIM == 3) { return DLFEvalAssemble3D<T_D1D, T_Q1D>; }
   MFEM_ABORT("");
}

template <int T_D1D = 0, int T_Q1D = 0>
static void HdivDLFAssemble2D(const int ne, const Array<int> &markers,
                              const Vector &jac, const Array<real_t> &weights,
                              const Array<real_t> &testBO,
                              const Array<real_t> &testBC, const Vector &coeff,
                              Vector &y, const int d, const int q)
{
   MFEM_VERIFY(T_D1D || d <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Problem size too large.");
   MFEM_VERIFY(T_Q1D || q <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Problem size too large.");

   static constexpr int vdim = 2;
   const auto F = coeff.Read();
   const auto M = Reshape(markers.Read(), ne);
   const auto BO = Reshape(testBO.Read(), q, d-1);
   const auto BC = Reshape(testBC.Read(), q, d);
   const auto J = Reshape(jac.Read(), q, q, vdim, vdim, ne);
   const auto W = Reshape(weights.Read(), q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F,vdim,1,1,1) : Reshape(F,vdim,q,q,ne);
   auto Y = Reshape(y.Write(), 2*(d-1)*d, ne);

   mfem::forall_3D(ne, q, q, vdim, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;

      MFEM_SHARED real_t sBot[Q*D];
      MFEM_SHARED real_t sBct[Q*D];
      MFEM_SHARED real_t sQQ[vdim*Q*Q];
      MFEM_SHARED real_t sQD[vdim*Q*D];

      // Bo and Bc into shared memory
      const DeviceMatrix Bot(sBot, d-1, q);
      kernels::internal::LoadBt<D,Q>(d-1, q, BO, sBot);
      const DeviceMatrix Bct(sBct, d, q);
      kernels::internal::LoadBt<D,Q>(d, q, BC, sBct);

      const DeviceCube QQ(sQQ, q, q, vdim);
      const DeviceCube QD(sQD, q, d, vdim);

      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const real_t cst_val_0 = C(0,0,0,0);
         const real_t cst_val_1 = C(1,0,0,0);
         MFEM_FOREACH_THREAD(y,y,q)
         {
            MFEM_FOREACH_THREAD(x,x,q)
            {
               const real_t J0 = J(x,y,0,vd,e);
               const real_t J1 = J(x,y,1,vd,e);
               const real_t C0 = cst ? cst_val_0 : C(0,x,y,e);
               const real_t C1 = cst ? cst_val_1 : C(1,x,y,e);
               QQ(x,y,vd) = W(x,y)*(J0*C0 + J1*C1);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         DeviceMatrix Btx = (vd == 0) ? Bct : Bot;
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t qd = 0.0;
               for (int qx = 0; qx < q; ++qx)
               {
                  qd += QQ(qx,qy,vd) * Btx(dx,qx);
               }
               QD(dx,qy,vd) = qd;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         const int ny = (vd == 1) ? d : d-1;
         DeviceMatrix Bty = (vd == 1) ? Bct : Bot;
         DeviceTensor<4> Yxy(Y, nx, ny, vdim, ne);
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t dd = 0.0;
               for (int qy = 0; qy < q; ++qy)
               {
                  dd += QD(dx,qy,vd) * Bty(dy,qy);
               }
               Yxy(dx,dy,vd,e) += dd;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void HdivDLFAssemble3D(const int ne, const Array<int> &markers,
                              const Vector &jac, const Array<real_t> &weights,
                              const Array<real_t> &testBO,
                              const Array<real_t> &testBC, const Vector &coeff,
                              Vector &y, const int d, const int q)
{
   MFEM_VERIFY(T_D1D || d <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Problem size too large.");
   MFEM_VERIFY(T_Q1D || q <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Problem size too large.");

   static constexpr int vdim = 3;
   const auto F = coeff.Read();
   const auto M = Reshape(markers.Read(), ne);
   const auto BO = Reshape(testBO.Read(), q, d-1);
   const auto BC = Reshape(testBC.Read(), q, d);
   const auto J = Reshape(jac.Read(), q, q, q, vdim, vdim, ne);
   const auto W = Reshape(weights.Read(), q, q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F,vdim,1,1,1,1) : Reshape(F,vdim,q,q,q,ne);
   auto Y = Reshape(y.Write(), 3*(d-1)*(d-1)*d, ne);

   mfem::forall_3D(ne, q, q, vdim, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;

      MFEM_SHARED real_t sBot[Q*D];
      MFEM_SHARED real_t sBct[Q*D];

      // Bo and Bc into shared memory
      const DeviceMatrix Bot(sBot, d-1, q);
      kernels::internal::LoadB<D,Q>(d-1, q, BO, sBot);
      const DeviceMatrix Bct(sBct, d, q);
      kernels::internal::LoadB<D,Q>(d, q, BC, sBct);

      MFEM_SHARED real_t sm0[vdim*Q*Q*Q];
      MFEM_SHARED real_t sm1[vdim*Q*Q*Q];
      DeviceTensor<4> QQQ(sm1, q, q, q, vdim);
      DeviceTensor<4> DQQ(sm0, d, q, q, vdim);
      DeviceTensor<4> DDQ(sm1, d, d, q, vdim);

      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const real_t cst_val_0 = C(0,0,0,0,0);
         const real_t cst_val_1 = C(1,0,0,0,0);
         const real_t cst_val_2 = C(2,0,0,0,0);
         MFEM_FOREACH_THREAD(y,y,q)
         {
            MFEM_FOREACH_THREAD(x,x,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  const real_t J0 = J(x,y,z,0,vd,e);
                  const real_t J1 = J(x,y,z,1,vd,e);
                  const real_t J2 = J(x,y,z,2,vd,e);
                  const real_t C0 = cst ? cst_val_0 : C(0,x,y,z,e);
                  const real_t C1 = cst ? cst_val_1 : C(1,x,y,z,e);
                  const real_t C2 = cst ? cst_val_2 : C(2,x,y,z,e);
                  QQQ(x,y,z,vd) = W(x,y,z)*(J0*C0 + J1*C1 + J2*C2);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         DeviceMatrix Btx = (vd == 0) ? Bct : Bot;
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(Q)
               for (int qx = 0; qx < q; ++qx)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += QQQ(qx,qy,qz,vd) * Btx(dx,qx);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { DQQ(dx,qy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         const int ny = (vd == 1) ? d : d-1;
         DeviceMatrix Bty = (vd == 1) ? Bct : Bot;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(Q)
               for (int qy = 0; qy < q; ++qy)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += DQQ(dx,qy,qz,vd) * Bty(dy,qy);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz) { DDQ(dx,dy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,vdim)
      {
         const int nx = (vd == 0) ? d : d-1;
         const int ny = (vd == 1) ? d : d-1;
         const int nz = (vd == 2) ? d : d-1;
         DeviceTensor<5> Yxyz(Y, nx, ny, nz, vdim, ne);
         DeviceMatrix Btz = (vd == 2) ? Bct : Bot;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[D];
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  MFEM_UNROLL(D)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += DDQ(dx,dy,qz,vd) * Btz(dz,qz);
                  }
               }
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz) { Yxyz(dx,dy,dz,vd,e) += u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void HcurlDLFAssemble3D(const int ne, const Array<int> &markers,
                               const Vector &jac, const Array<real_t> &weights,
                               const Array<real_t> &testBO,
                               const Array<real_t> &testBC, const Vector &coeff,
                               Vector &y, const int d, const int q)
{
   MFEM_VERIFY(T_D1D || d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Problem size too large.");
   MFEM_VERIFY(T_Q1D || q <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Problem size too large.");
   MFEM_VERIFY(d <= q, "");

   constexpr int vdim = 3;
   const auto F = coeff.Read();
   const auto M = Reshape(markers.Read(), ne);
   const auto BO = Reshape(testBO.Read(), q, d-1);
   const auto BC = Reshape(testBC.Read(), q, d);
   const auto J = Reshape(jac.Read(), q, q, q, vdim, vdim, ne);
   const auto W = Reshape(weights.Read(), q, q, q);
   const bool cst = coeff.Size() == vdim;
   const auto C = cst ? Reshape(F,vdim,1,1,1,1) : Reshape(F,vdim,q,q,q,ne);
   auto Y = Reshape(y.Write(), 3*(d-1)*(d-1)*d, ne);

   mfem::forall_3D(ne, q, q, vdim, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         // ignore
         return;
      }

      constexpr int vdim = 3;
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;

      // D-1 could be zero, dummy have space for one
      MFEM_SHARED real_t sBot[D > 1 ? Q * (D - 1) : 1];
      MFEM_SHARED real_t sBct[Q * D];

      // Bo and Bc into shared memory
      const DeviceMatrix Bot(sBot, d - 1, q);
      // nvcc can't first-capture in an if constexpr
      auto Bo = BO;
      if constexpr (D > 1)
      {
         kernels::internal::LoadB<D - 1, Q>(d - 1, q, Bo, sBot);
      }
      const DeviceMatrix Bct(sBct, d, q);
      kernels::internal::LoadB<D, Q>(d, q, BC, sBct);

      MFEM_SHARED real_t sm0[vdim * Q * Q * Q];
      MFEM_SHARED real_t sm1[vdim * Q * Q * Q];
      DeviceTensor<4> QQQ(sm1, q, q, q, vdim);
      DeviceTensor<4> DQQ(sm0, d, q, q, vdim);
      DeviceTensor<4> DDQ(sm1, d, d, q, vdim);

      const real_t cst_val_0 = C(0, 0, 0, 0, 0);
      const real_t cst_val_1 = C(1, 0, 0, 0, 0);
      const real_t cst_val_2 = C(2, 0, 0, 0, 0);

      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         MFEM_FOREACH_THREAD(y, y, q)
         {
            MFEM_FOREACH_THREAD(x, x, q)
            {
               for (int z = 0; z < q; ++z)
               {
                  real_t curr[3];
                  curr[0] = cst ? cst_val_0 : C(0, x, y, z, e);
                  curr[1] = cst ? cst_val_1 : C(0, x, y, z, e);
                  curr[2] = cst ? cst_val_2 : C(0, x, y, z, e);

                  const real_t J11 = J(x, y, z, 0, 0, e);
                  const real_t J21 = J(x, y, z, 1, 0, e);
                  const real_t J31 = J(x, y, z, 2, 0, e);
                  const real_t J12 = J(x, y, z, 0, 1, e);
                  const real_t J22 = J(x, y, z, 1, 1, e);
                  const real_t J32 = J(x, y, z, 2, 1, e);
                  const real_t J13 = J(x, y, z, 0, 2, e);
                  const real_t J23 = J(x, y, z, 1, 2, e);
                  const real_t J33 = J(x, y, z, 2, 2, e);
                  // adj(J)
                  const real_t A11 = (J22 * J33) - (J23 * J32);
                  const real_t A12 = (J32 * J13) - (J12 * J33);
                  const real_t A13 = (J12 * J23) - (J22 * J13);
                  const real_t A21 = (J31 * J23) - (J21 * J33);
                  const real_t A22 = (J11 * J33) - (J13 * J31);
                  const real_t A23 = (J21 * J13) - (J11 * J23);
                  const real_t A31 = (J21 * J32) - (J31 * J22);
                  const real_t A32 = (J31 * J12) - (J11 * J32);
                  const real_t A33 = (J11 * J22) - (J12 * J21);
                  const real_t A[9] = {A11, A12, A13, A21, A22,
                                       A23, A31, A32, A33
                                      };
                  QQQ(x, y, z, vd) = W(x, y, z) * (A[vd * vdim] * curr[0] +
                                                   A[vd * vdim + 1] * curr[1] +
                                                   A[vd * vdim + 2] * curr[2]);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         const int nx = (vd == 0) ? d - 1 : d;
         DeviceMatrix Btx = (vd == 0) ? Bot : Bct;
         MFEM_FOREACH_THREAD(qy, y, q)
         {
            MFEM_FOREACH_THREAD(dx, x, nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  u[qz] = 0.0;
               }
               MFEM_UNROLL(Q)
               for (int qx = 0; qx < q; ++qx)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += QQQ(qx, qy, qz, vd) * Btx(dx, qx);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  DQQ(dx, qy, qz, vd) = u[qz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         const int nx = (vd == 0) ? d - 1 : d;
         const int ny = (vd == 1) ? d - 1 : d;
         DeviceMatrix Bty = (vd == 1) ? Bot : Bct;
         MFEM_FOREACH_THREAD(dy, y, ny)
         {
            MFEM_FOREACH_THREAD(dx, x, nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  u[qz] = 0.0;
               }
               MFEM_UNROLL(Q)
               for (int qy = 0; qy < q; ++qy)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += DQQ(dx, qy, qz, vd) * Bty(dy, qy);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  DDQ(dx, dy, qz, vd) = u[qz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         const int nx = (vd == 0) ? d - 1 : d;
         const int ny = (vd == 1) ? d - 1 : d;
         const int nz = (vd == 2) ? d - 1 : d;
         DeviceTensor<5> Yxyz(Y, nx, ny, nz, vdim, ne);
         DeviceMatrix Btz = (vd == 2) ? Bot : Bct;
         MFEM_FOREACH_THREAD(dy, y, ny)
         {
            MFEM_FOREACH_THREAD(dx, x, nx)
            {
               real_t u[D];
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz)
               {
                  u[dz] = 0.0;
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  MFEM_UNROLL(D)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += DDQ(dx, dy, qz, vd) * Btz(dz, qz);
                  }
               }
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz)
               {
                  Yxyz(dx, dy, dz, vd, e) += u[dz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template <FiniteElement::DerivType TestType, int DIM, int TEST_D1D, int Q1D>
VectorFEDomainLFIntegrator::AssembleKernelType
VectorFEDomainLFIntegrator::AssembleKernels::Kernel()
{
   if constexpr (TestType == FiniteElement::DIV)
   {
      if constexpr (DIM == 2)
      {
         return HdivDLFAssemble2D<TEST_D1D, Q1D>;
      }
      if constexpr (DIM == 3)
      {
         return HdivDLFAssemble3D<TEST_D1D, Q1D>;
      }
   }
   if constexpr (TestType == FiniteElement::CURL)
   {
      if constexpr (DIM == 3)
      {
         // TODO
         return HcurlDLFAssemble3D<TEST_D1D, Q1D>;
      }
   }
   MFEM_ABORT("");
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem

#endif // MFEM_LININTEG_DOMAIN_KERNELS_HPP
