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

#ifndef MFEM_DGMASSINV_KERNELS_HPP
#define MFEM_DGMASSINV_KERNELS_HPP

#include "bilininteg_mass_pa.hpp"
#include "../linalg/kernels.hpp"
#include "kernels.hpp"

namespace mfem
{

namespace internal
{

void MakeReciprocal(int n, double *x)
{
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { x[i] = 1.0/x[i]; });
}

template <int DIM, int D1D, int Q1D>
MFEM_HOST_DEVICE inline
void DGMassApply(const int e,
                 const int NE,
                 const double *B,
                 const double *Bt,
                 const double *pa_data,
                 const double *x,
                 double *y,
                 const int d1d = 0,
                 const int q1d = 0)
{
   constexpr bool use_smem = (D1D > 0 && Q1D > 0);
   constexpr bool ACCUM = false;
   constexpr int NBZ = 1;
   if (use_smem)
   {
      // cannot specialize functions below with D1D or Q1D equal to zero
      // (this branch only runs with D1D and Q1D are both positive)
      constexpr int TD1D = D1D ? D1D : 1;
      constexpr int TQ1D = Q1D ? Q1D : 1;
      if (DIM == 2)
      {
         SmemPAMassApply2D_Element<TD1D,TQ1D,NBZ,ACCUM>(e, NE, B, pa_data, x, y);
      }
      else if (DIM == 3)
      {
         SmemPAMassApply3D_Element<TD1D,TQ1D,ACCUM>(e, NE, B, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT_KERNEL("Unsupported dimension.");
      }
   }
   else
   {
      if (DIM == 2)
      {
         PAMassApply2D_Element<ACCUM>(e, NE, B, Bt, pa_data, x, y, d1d, q1d);
      }
      else if (DIM == 3)
      {
         PAMassApply3D_Element<ACCUM>(e, NE, B, Bt, pa_data, x, y, d1d, q1d);
      }
      else
      {
         MFEM_ABORT_KERNEL("Unsupported dimension.");
      }
   }
}

MFEM_HOST_DEVICE inline
void DGMassPreconditioner(const int e,
                          const int NE,
                          const int ND,
                          const double *dinv,
                          const double *x,
                          double *y)
{
   const auto X = ConstDeviceMatrix(x, ND, NE);
   const auto D = ConstDeviceMatrix(dinv, ND, NE);
   auto Y = DeviceMatrix(y, ND, NE);

   const int tid = MFEM_THREAD_ID(x) + MFEM_THREAD_SIZE(x)*MFEM_THREAD_ID(y);
   const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

   for (int i = tid; i < ND; i += bxy)
   {
      Y(i, e) = D(i, e)*X(i, e);
   }
   MFEM_SYNC_THREAD;
}

MFEM_HOST_DEVICE inline
void DGMassAxpy(const int e,
                const int NE,
                const int ND,
                const double a,
                const double *x,
                const double b,
                const double *y,
                double *z)
{
   const auto X = ConstDeviceMatrix(x, ND, NE);
   const auto Y = ConstDeviceMatrix(y, ND, NE);
   auto Z = DeviceMatrix(z, ND, NE);

   const int tid = MFEM_THREAD_ID(x) + MFEM_THREAD_SIZE(x)*MFEM_THREAD_ID(y);
   const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

   for (int i = tid; i < ND; i += bxy)
   {
      Z(i, e) = a*X(i, e) + b*Y(i, e);
   }
   MFEM_SYNC_THREAD;
}

template <int NB>
MFEM_HOST_DEVICE inline
double DGMassDot(const int e,
                 const int NE,
                 const int ND,
                 const double *x,
                 const double *y)
{
   const auto X = ConstDeviceMatrix(x, ND, NE);
   const auto Y = ConstDeviceMatrix(y, ND, NE);

   const int tid = MFEM_THREAD_ID(x) + MFEM_THREAD_SIZE(x)*MFEM_THREAD_ID(y);
   const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

   MFEM_SHARED double s_dot[NB*NB];
   s_dot[tid] = 0.0;

   for (int i = tid; i < ND; i += bxy) { s_dot[tid] += X(i,e)*Y(i,e); }
   MFEM_SYNC_THREAD;

   if (bxy > 512 && tid + 512 < bxy) { s_dot[tid] += s_dot[tid + 512]; }
   MFEM_SYNC_THREAD;

   if (bxy > 256 && tid < 256 && tid + 256 < bxy) { s_dot[tid] += s_dot[tid + 256]; }
   MFEM_SYNC_THREAD;

   if (bxy > 128 && tid < 128 && tid + 128 < bxy) { s_dot[tid] += s_dot[tid + 128]; }
   MFEM_SYNC_THREAD;

   if (bxy > 64 && tid < 64 && tid + 64 < bxy) { s_dot[tid] += s_dot[tid + 64]; }
   MFEM_SYNC_THREAD;

   if (bxy > 32 && tid < 32 && tid + 32 < bxy) { s_dot[tid] += s_dot[tid + 32]; }
   MFEM_SYNC_THREAD;

   if (bxy > 16 && tid < 16 && tid + 16 < bxy) { s_dot[tid] += s_dot[tid + 16]; }
   MFEM_SYNC_THREAD;

   if (bxy > 8 && tid < 8 && tid + 8 < bxy) { s_dot[tid] += s_dot[tid + 8]; }
   MFEM_SYNC_THREAD;

   if (bxy > 4 && tid < 4 && tid + 4 < bxy) { s_dot[tid] += s_dot[tid + 4]; }
   MFEM_SYNC_THREAD;

   if (bxy > 2 && tid < 2 && tid + 2 < bxy) { s_dot[tid] += s_dot[tid + 2]; }
   MFEM_SYNC_THREAD;

   if (bxy > 1 && tid < 1 && tid + 1 < bxy) { s_dot[tid] += s_dot[tid + 1]; }
   MFEM_SYNC_THREAD;

   return s_dot[0];
}

template<int T_D1D = 0, int MAX_D1D = 0>
MFEM_HOST_DEVICE inline
void DGMassBasis2D(const int e,
                   const int NE,
                   const double *b_,
                   const double *x_,
                   double *y_,
                   const int d1d = 0)
{
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto b = Reshape(b_, D1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, NE);
   auto y = Reshape(y_, D1D, D1D, NE);

   MFEM_SHARED double sB[MD1*MD1];
   MFEM_SHARED double sm0[MD1*MD1];
   MFEM_SHARED double sm1[MD1*MD1];

   kernels::internal::LoadB<MD1,MD1>(D1D,D1D,b,sB);

   ConstDeviceMatrix B(sB, D1D,D1D);
   DeviceMatrix DD(sm0, MD1, MD1);
   DeviceMatrix DQ(sm1, MD1, MD1);
   DeviceMatrix QQ(sm0, MD1, MD1);

   kernels::internal::LoadX(e,D1D,x,DD);
   kernels::internal::EvalX(D1D,D1D,B,DD,DQ);
   kernels::internal::EvalY(D1D,D1D,B,DQ,QQ);
   MFEM_SYNC_THREAD; // sync here to allow in-place evaluations
   MFEM_FOREACH_THREAD(qy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,D1D)
      {
         y(qx,qy,e) = QQ(qx,qy);
      }
   }
   MFEM_SYNC_THREAD;
}

template<int T_D1D = 0, int MAX_D1D = 0>
MFEM_HOST_DEVICE inline
void DGMassBasis3D(const int e,
                   const int NE,
                   const double *b_,
                   const double *x_,
                   double *y_,
                   const int d1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto b = Reshape(b_, D1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, NE);
   auto y = Reshape(y_, D1D, D1D, D1D, NE);

   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

   MFEM_SHARED double sB[MD1*MD1];
   MFEM_SHARED double sm0[MD1*MD1*MD1];
   MFEM_SHARED double sm1[MD1*MD1*MD1];

   kernels::internal::LoadB<MD1,MD1>(D1D,D1D,b,sB);

   ConstDeviceMatrix B(sB, D1D,D1D);
   DeviceCube DDD(sm0, MD1,MD1,MD1);
   DeviceCube DDQ(sm1, MD1,MD1,MD1);
   DeviceCube DQQ(sm0, MD1,MD1,MD1);
   DeviceCube QQQ(sm1, MD1,MD1,MD1);

   kernels::internal::LoadX(e,D1D,x,DDD);
   kernels::internal::EvalX(D1D,D1D,B,DDD,DDQ);
   kernels::internal::EvalY(D1D,D1D,B,DDQ,DQQ);
   kernels::internal::EvalZ(D1D,D1D,B,DQQ,QQQ);
   MFEM_SYNC_THREAD; // sync here to allow in-place evaluation
   MFEM_FOREACH_THREAD(qz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qy,y,D1D)
      {
         for (int qx = 0; qx < D1D; ++qx)
         {
            y(qx,qy,qz,e) = QQQ(qz,qy,qx);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template<int DIM, int T_D1D = 0, int MAX_D1D = 0>
MFEM_HOST_DEVICE inline
void DGMassBasis(const int e,
                 const int NE,
                 const double *b_,
                 const double *x_,
                 double *y_,
                 const int d1d = 0)
{
   if (DIM == 2)
   {
      DGMassBasis2D<T_D1D, MAX_D1D>(e, NE, b_, x_, y_, d1d);
   }
   else if (DIM == 3)
   {
      DGMassBasis3D<T_D1D, MAX_D1D>(e, NE, b_, x_, y_, d1d);
   }
   else
   {
      MFEM_ABORT_KERNEL("Dimension not supported.");
   }
}

} // namespace internal

} // namespace mfem

#endif
