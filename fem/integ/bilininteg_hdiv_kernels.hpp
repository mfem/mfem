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

#ifndef MFEM_BILININTEG_HDIV_KERNELS_HPP
#define MFEM_BILININTEG_HDIV_KERNELS_HPP

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

// Piola transformation in H(div): w = (1 / det (dF)) dF \hat{w}
// div w = (1 / det (dF)) \hat{div} \hat{w}

namespace mfem
{

namespace internal
{

// PA H(div) Mass Assemble 2D kernel
void PAHdivMassSetup2D(const int Q1D,
                       const int coeffDim,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff_,
                       Vector &op);

// PA H(div) Mass Assemble 3D kernel
void PAHdivMassSetup3D(const int Q1D,
                       const int coeffDim,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff_,
                       Vector &op);

// PA H(div) Mass Diagonal 2D kernel
void PAHdivMassAssembleDiagonal2D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const bool symmetric,
                                  const Array<real_t> &Bo_,
                                  const Array<real_t> &Bc_,
                                  const Vector &op_,
                                  Vector &diag_);

// PA H(div) Mass Diagonal 3D kernel
void PAHdivMassAssembleDiagonal3D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const bool symmetric,
                                  const Array<real_t> &Bo_,
                                  const Array<real_t> &Bc_,
                                  const Vector &op_,
                                  Vector &diag_);

void PAHdivMassApply(const int dim,
                     const int D1D,
                     const int Q1D,
                     const int NE,
                     const bool symmetric,
                     const Array<real_t> &Bo,
                     const Array<real_t> &Bc,
                     const Array<real_t> &Bot,
                     const Array<real_t> &Bct,
                     const Vector &op,
                     const Vector &x,
                     Vector &y);

// PA H(div) Mass Apply 2D kernel
void PAHdivMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const bool symmetric,
                       const Array<real_t> &Bo_,
                       const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_,
                       const Array<real_t> &Bct_,
                       const Vector &op_,
                       const Vector &x_,
                       Vector &y_);

// PA H(div) Mass Apply 3D kernel
void PAHdivMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const bool symmetric,
                       const Array<real_t> &Bo_,
                       const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_,
                       const Array<real_t> &Bct_,
                       const Vector &op_,
                       const Vector &x_,
                       Vector &y_);

// Shared memory PA H(div) Mass Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAHdivMassApply2D(const int NE,
                                  const bool symmetric,
                                  const Array<real_t> &Bo_,
                                  const Array<real_t> &Bc_,
                                  const Array<real_t> &Bot_,
                                  const Array<real_t> &Bct_,
                                  const Vector &op_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   MFEM_CONTRACT_VAR(Bot_);
   MFEM_CONTRACT_VAR(Bct_);

   static constexpr int VDIM = 2;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   const auto bc = Reshape(Bc_.Read(), Q1D, D1D);
   const auto D = Reshape(op_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   const auto x = Reshape(x_.Read(), D1D*(D1D-1), VDIM, NE);
   auto y = y_.ReadWrite();

   mfem::forall_3D(NE, Q1D, Q1D, VDIM, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t smo[MQ1*(MD1-1)];
      DeviceMatrix Bo(smo, D1D-1, Q1D);

      MFEM_SHARED real_t smc[MQ1*MD1];
      DeviceMatrix Bc(smc, D1D, Q1D);

      MFEM_SHARED real_t sm0[VDIM*MDQ*MDQ];
      MFEM_SHARED real_t sm1[VDIM*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1), VDIM);
      DeviceCube QD(sm1, Q1D, D1D, VDIM);
      DeviceCube QQ(sm0, Q1D, Q1D, VDIM);
      DeviceCube DQ(sm1, D1D, Q1D, VDIM);

      // Load X, Bo and Bc into shared memory
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               if (qx < D1D && dy < (D1D-1))
               {
                  X(qx + dy*D1D,vd) = x(qx+dy*D1D,vd,e);
               }
               if (tidz == 0)
               {
                  if (dy < (D1D-1)) { Bo(dy,qx) = bo(qx,dy); }
                  Bc(dy,qx) = bc(qx,dy);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply B operator
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceCube Xxy(X, nx, ny, VDIM);
         DeviceMatrix Bx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t dq = 0.0;
               for (int dx = 0; dx < nx; ++dx)
               {
                  dq += Xxy(dx,dy,vd) * Bx(dx,qx);
               }
               QD(qx,dy,vd) = dq;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t qq = 0.0;
               for (int dy = 0; dy < ny; ++dy)
               {
                  qq += QD(qx,dy,vd) * By(dy,qy);
               }
               QQ(qx,qy,vd) = qq;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply D operator
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const real_t Qx = QQ(qx,qy,0);
               const real_t Qy = QQ(qx,qy,1);

               const real_t D11 = D(qx,qy,0,e);
               const real_t D12 = D(qx,qy,1,e);
               const real_t D21 = symmetric ? D12 : D(qx,qy,2,e);
               const real_t D22 = symmetric ? D(qx,qy,2,e) : D(qx,qy,3,e);

               QQ(qx,qy,0) = D11*Qx + D12*Qy;
               QQ(qx,qy,1) = D21*Qx + D22*Qy;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         DeviceMatrix Btx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t qd = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  qd += QQ(qx,qy,vd) * Btx(dx,qx);
               }
               DQ(dx,qy,vd) = qd;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix Bty = (vd == 1) ? Bc : Bo;
         DeviceTensor<4> Yxy(y, nx, ny, VDIM, NE);
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t dd = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  dd += DQ(dx,qy,vd) * Bty(dy,qy);
               }
               Yxy(dx,dy,vd,e) += dd;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

// Shared memory PA H(div) Mass Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAHdivMassApply3D(const int NE,
                                  const bool symmetric,
                                  const Array<real_t> &Bo_,
                                  const Array<real_t> &Bc_,
                                  const Array<real_t> &Bot_,
                                  const Array<real_t> &Bct_,
                                  const Vector &op_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   MFEM_CONTRACT_VAR(Bot_);
   MFEM_CONTRACT_VAR(Bct_);

   static constexpr int VDIM = 3;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   const auto bc = Reshape(Bc_.Read(), Q1D, D1D);
   const auto D = Reshape(op_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   const auto x = Reshape(x_.Read(), D1D*(D1D-1)*(D1D-1), VDIM, NE);
   auto y = y_.ReadWrite();

   mfem::forall_3D(NE, Q1D, Q1D, VDIM, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t smo[MQ1*(MD1-1)];
      DeviceMatrix Bo(smo, D1D-1, Q1D);

      MFEM_SHARED real_t smc[MQ1*MD1];
      DeviceMatrix Bc(smc, D1D, Q1D);

      MFEM_SHARED real_t sm0[VDIM*MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[VDIM*MDQ*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1)*(D1D-1), VDIM);
      DeviceTensor<4> QDD(sm1, Q1D, D1D, D1D, VDIM);
      DeviceTensor<4> QQD(sm0, Q1D, Q1D, D1D, VDIM);
      DeviceTensor<4> QQQ(sm1, Q1D, Q1D, Q1D, VDIM);
      DeviceTensor<4> DQQ(sm0, D1D, Q1D, Q1D, VDIM);
      DeviceTensor<4> DDQ(sm1, D1D, D1D, Q1D, VDIM);

      // Load X into shared memory
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         MFEM_FOREACH_THREAD(dz,y,D1D-1)
         {
            MFEM_FOREACH_THREAD(dy,x,D1D-1)
            {
               MFEM_UNROLL(MD1)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  X(dx+(dy+dz*(D1D-1))*D1D,vd) = x(dx+(dy+dz*(D1D-1))*D1D,vd,e);
               }
            }
         }
      }
      // Load Bo and Bc into shared memory
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D-1)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bo(d,q) = bo(q,d);
            }
         }
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bc(d,q) = bc(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply B operator
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceTensor<4> Xxyz(X, nx, ny, nz, VDIM);
         DeviceMatrix Bx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[D1D];
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(MD1)
               for (int dx = 0; dx < nx; ++dx)
               {
                  MFEM_UNROLL(MD1)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += Xxyz(dx,dy,dz,vd) * Bx(dx,qx);
                  }
               }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { QDD(qx,dy,dz,vd) = u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[D1D];
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(MD1)
               for (int dy = 0; dy < ny; ++dy)
               {
                  MFEM_UNROLL(MD1)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += QDD(qx,dy,dz,vd) * By(dy,qy);
                  }
               }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { QQD(qx,qy,dz,vd) = u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix Bz = (vd == 2) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[Q1D];
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz)
               {
                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u[qz] += QQD(qx,qy,dz,vd) * Bz(dz,qz);
                  }
               }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { QQQ(qx,qy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply D operator
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const real_t Qx = QQQ(qx,qy,qz,0);
                  const real_t Qy = QQQ(qx,qy,qz,1);
                  const real_t Qz = QQQ(qx,qy,qz,2);

                  const real_t D11 = D(qx,qy,qz,0,e);
                  const real_t D12 = D(qx,qy,qz,1,e);
                  const real_t D13 = D(qx,qy,qz,2,e);
                  const real_t D21 = symmetric ? D12 : D(qx,qy,qz,3,e);
                  const real_t D22 = symmetric ? D(qx,qy,qz,3,e) : D(qx,qy,qz,4,e);
                  const real_t D23 = symmetric ? D(qx,qy,qz,4,e) : D(qx,qy,qz,5,e);
                  const real_t D31 = symmetric ? D13 : D(qx,qy,qz,6,e);
                  const real_t D32 = symmetric ? D23 : D(qx,qy,qz,7,e);
                  const real_t D33 = symmetric ? D(qx,qy,qz,5,e) : D(qx,qy,qz,8,e);

                  QQQ(qx,qy,qz,0) = D11*Qx + D12*Qy + D13*Qz;
                  QQQ(qx,qy,qz,1) = D21*Qx + D22*Qy + D23*Qz;
                  QQQ(qx,qy,qz,2) = D31*Qx + D32*Qy + D33*Qz;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         DeviceMatrix Btx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[Q1D];
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u[qz] += QQQ(qx,qy,qz,vd) * Btx(dx,qx);
                  }
               }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { DQQ(dx,qy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix Bty = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[Q1D];
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u[qz] += DQQ(dx,qy,qz,vd) * Bty(dy,qy);
                  }
               }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { DDQ(dx,dy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceTensor<5> Yxyz(y, nx, ny, nz, VDIM, NE);
         DeviceMatrix Btz = (vd == 2) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[D1D];
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  MFEM_UNROLL(MD1)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += DDQ(dx,dy,qz,vd) * Btz(dz,qz);
                  }
               }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { Yxyz(dx,dy,dz,vd,e) += u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

// PA H(div) div-div Assemble 2D kernel
void PADivDivSetup2D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     const Vector &j,
                     Vector &coeff_,
                     Vector &op);

// PA H(div) div-div Assemble 3D kernel
void PADivDivSetup3D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     const Vector &j,
                     Vector &coeff_,
                     Vector &op);

// PA H(div) div-div Diagonal 2D kernel
void PADivDivAssembleDiagonal2D(const int D1D,
                                const int Q1D,
                                const int NE,
                                const Array<real_t> &Bo_,
                                const Array<real_t> &Gc_,
                                const Vector &op_,
                                Vector &diag_);

// PA H(div) div-div Diagonal 3D kernel
void PADivDivAssembleDiagonal3D(const int D1D,
                                const int Q1D,
                                const int NE,
                                const Array<real_t> &Bo_,
                                const Array<real_t> &Gc_,
                                const Vector &op_,
                                Vector &diag_);

// PA H(div) div-div Apply 2D kernel
void PADivDivApply2D(const int D1D,
                     const int Q1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &Bot_,
                     const Array<real_t> &Gct_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_);

// PA H(div) div-div Apply 3D kernel
void PADivDivApply3D(const int D1D,
                     const int Q1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &Bot_,
                     const Array<real_t> &Gct_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_);

// PA H(div)-L2 Assemble 2D kernel
void PAHdivL2Setup2D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     Vector &coeff_,
                     Vector &op);

// PA H(div)-L2 Assemble 3D kernel
void PAHdivL2Setup3D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     Vector &coeff_,
                     Vector &op);

// PA H(div)-L2 Diagonal 2D kernel
void PAHdivL2AssembleDiagonal_ADAt_2D(const int D1D,
                                      const int Q1D,
                                      const int L2D1D,
                                      const int NE,
                                      const Array<real_t> &L2Bo_,
                                      const Array<real_t> &Gct_,
                                      const Array<real_t> &Bot_,
                                      const Vector &op_,
                                      const Vector &D_,
                                      Vector &diag_);

// PA H(div)-L2 Diagonal 3D kernel
void PAHdivL2AssembleDiagonal_ADAt_3D(const int D1D,
                                      const int Q1D,
                                      const int L2D1D,
                                      const int NE,
                                      const Array<real_t> &L2Bo_,
                                      const Array<real_t> &Gct_,
                                      const Array<real_t> &Bot_,
                                      const Vector &op_,
                                      const Vector &D_,
                                      Vector &diag_);

// PA H(div)-L2 Apply 2D kernel
void PAHdivL2Apply2D(const int D1D,
                     const int Q1D,
                     const int L2D1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &L2Bot_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_);

// PA H(div)-L2 Apply Transpose 2D kernel
void PAHdivL2ApplyTranspose2D(const int D1D,
                              const int Q1D,
                              const int L2D1D,
                              const int NE,
                              const Array<real_t> &L2Bo_,
                              const Array<real_t> &Gct_,
                              const Array<real_t> &Bot_,
                              const Vector &op_,
                              const Vector &x_,
                              Vector &y_);

// PA H(div)-L2 Apply 3D kernel
void PAHdivL2Apply3D(const int D1D,
                     const int Q1D,
                     const int L2D1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &L2Bot_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_);

// PA H(div)-L2 Apply Transpose 3D kernel
void PAHdivL2ApplyTranspose3D(const int D1D,
                              const int Q1D,
                              const int L2D1D,
                              const int NE,
                              const Array<real_t> &L2Bo_,
                              const Array<real_t> &Gct_,
                              const Array<real_t> &Bot_,
                              const Vector &op_,
                              const Vector &x_,
                              Vector &y_);

} // namespace internal

} // namespace mfem

#endif
