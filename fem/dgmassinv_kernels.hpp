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

#ifndef MFEM_DGMASSINV_KERNELS_HPP
#define MFEM_DGMASSINV_KERNELS_HPP

#include "../linalg/kernels.hpp"
#include "kernels.hpp"
#include "integ/bilininteg_mass_kernels.hpp"
#include "dgmassinv.hpp"

namespace mfem
{

namespace internal
{

template <int DIM, int D1D, int Q1D>
MFEM_HOST_DEVICE inline
void DGMassApply(const int e,
                 const int NE,
                 const real_t *B,
                 const real_t *Bt,
                 const real_t *pa_data,
                 const real_t *x,
                 real_t *y,
                 const int d1d = 0,
                 const int q1d = 0)
{
   constexpr bool use_smem = (D1D > 0 && Q1D > 0);
   constexpr bool ACCUM = false;
   constexpr int NBZ = 1;

   if (DIM == 1)
   {
      PAMassApply1D_Element<ACCUM>(e, NE, B, Bt, pa_data, x, y, d1d, q1d);
      return;
   }

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
                          const real_t *dinv,
                          const real_t *x,
                          real_t *y)
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
                const real_t a,
                const real_t *x,
                const real_t b,
                const real_t *y,
                real_t *z)
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
real_t DGMassDot(const int e,
                 const int NE,
                 const int ND,
                 const real_t *x,
                 const real_t *y)
{
   const auto X = ConstDeviceMatrix(x, ND, NE);
   const auto Y = ConstDeviceMatrix(y, ND, NE);

   const int tid = MFEM_THREAD_ID(x) + MFEM_THREAD_SIZE(x)*MFEM_THREAD_ID(y);
   const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

   MFEM_SHARED real_t s_dot[NB*NB];
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

template<int T_D1D = 0>
MFEM_HOST_DEVICE inline
void DGMassBasis1D(const int e,
                   const int NE,
                   const real_t *b_,
                   const real_t *x_,
                   real_t *y_,
                   const int d1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto b = Reshape(b_, D1D, D1D);
   const auto x = Reshape(x_, D1D, NE);
   auto y = Reshape(y_, D1D, NE);

   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   real_t Y[MD1];

   MFEM_FOREACH_THREAD(i,x,D1D)
   {
      real_t val = 0.0;
      for (int j = 0; j < D1D; ++j)
      {
         val += b(i,j)*x(j,e);
      }
      Y[i] = val;
   }
   MFEM_SYNC_THREAD;
   if (MFEM_THREAD_ID(y) == 0)
   {
      MFEM_FOREACH_THREAD(i,x,D1D)
      {
         y(i,e) = Y[i];
      }
   }
}

template<int T_D1D = 0>
MFEM_HOST_DEVICE inline
void DGMassBasis2D(const int e,
                   const int NE,
                   const real_t *b_,
                   const real_t *x_,
                   real_t *y_,
                   const int d1d = 0)
{
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto b = Reshape(b_, D1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, NE);
   auto y = Reshape(y_, D1D, D1D, NE);

   MFEM_SHARED real_t sB[MD1*MD1];
   MFEM_SHARED real_t sm0[MD1*MD1];
   MFEM_SHARED real_t sm1[MD1*MD1];

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

template<int T_D1D = 0>
MFEM_HOST_DEVICE inline
void DGMassBasis3D(const int e,
                   const int NE,
                   const real_t *b_,
                   const real_t *x_,
                   real_t *y_,
                   const int d1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto b = Reshape(b_, D1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, NE);
   auto y = Reshape(y_, D1D, D1D, D1D, NE);

   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

   MFEM_SHARED real_t sB[MD1*MD1];
   MFEM_SHARED real_t sm0[MD1*MD1*MD1];
   MFEM_SHARED real_t sm1[MD1*MD1*MD1];

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

template<int DIM, int T_D1D = 0>
MFEM_HOST_DEVICE inline
void DGMassBasis(const int e,
                 const int NE,
                 const real_t *b_,
                 const real_t *x_,
                 real_t *y_,
                 const int d1d = 0)
{
   if (DIM == 1)
   {
      DGMassBasis1D<T_D1D>(e, NE, b_, x_, y_, d1d);
   }
   else if (DIM == 2)
   {
      DGMassBasis2D<T_D1D>(e, NE, b_, x_, y_, d1d);
   }
   else if (DIM == 3)
   {
      DGMassBasis3D<T_D1D>(e, NE, b_, x_, y_, d1d);
   }
   else
   {
      MFEM_ABORT_KERNEL("Dimension not supported.");
   }
}

} // namespace internal

template<int DIM, int D1D, int Q1D>
void DGMassInverse::DGMassCGIteration(const Vector &b_, Vector &u_) const
{
   using namespace internal; // host/device kernel functions

   const int NE = fes.GetNE();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;

   const int ND = static_cast<int>(pow(d1d, DIM));

   const auto B = m->maps->B.Read();
   const auto Bt = m->maps->Bt.Read();
   const auto pa_data = m->pa_data.Read();
   const auto dinv = diag_inv.Read();
   auto r = r_.Write();
   auto d = d_.Write();
   auto z = z_.Write();
   auto u = u_.ReadWrite();

   const real_t RELTOL = rel_tol;
   const real_t ABSTOL = abs_tol;
   const int MAXIT = max_iter;
   const bool IT_MODE = iterative_mode;
   const bool CHANGE_BASIS = (d2q != nullptr);

   // b is the right-hand side (if no change of basis, this just points to the
   // incoming RHS vector, if we have to change basis, this points to the
   // internal b2 vector where we put the transformed RHS)
   const real_t *b;
   // the following are non-null if we have to change basis
   real_t *b2 = nullptr; // non-const access to b2
   const real_t *b_orig = nullptr; // RHS vector in "original" basis
   const real_t *d2q_B = nullptr; // matrix to transform initial guess
   const real_t *q2d_B = nullptr; // matrix to transform solution
   const real_t *q2d_Bt = nullptr; // matrix to transform RHS
   if (CHANGE_BASIS)
   {
      d2q_B = d2q->B.Read();
      q2d_B = B_.Read();
      q2d_Bt = Bt_.Read();

      b2 = b2_.Write();
      b_orig = b_.Read();
      b = b2;
   }
   else
   {
      b = b_.Read();
   }

   static constexpr int NB = Q1D ? Q1D : 1; // block size

   mfem::forall_2D(NE, NB, NB, [=] MFEM_HOST_DEVICE (int e)
   {
      // Perform change of basis if needed
      if (CHANGE_BASIS)
      {
         // Transform RHS
         DGMassBasis<DIM,D1D>(e, NE, q2d_Bt, b_orig, b2, d1d);
         if (IT_MODE)
         {
            // Transform initial guess
            DGMassBasis<DIM,D1D>(e, NE, d2q_B, u, u, d1d);
         }
      }

      const int tid = MFEM_THREAD_ID(x) + NB*MFEM_THREAD_ID(y);

      // Compute first residual
      if (IT_MODE)
      {
         DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, u, r, d1d, q1d);
         DGMassAxpy(e, NE, ND, 1.0, b, -1.0, r, r); // r = b - r
      }
      else
      {
         // if not in iterative mode, use zero initial guess
         const int BX = MFEM_THREAD_SIZE(x);
         const int BY = MFEM_THREAD_SIZE(y);
         const int bxy = BX*BY;
         const auto B = ConstDeviceMatrix(b, ND, NE);
         auto U = DeviceMatrix(u, ND, NE);
         auto R = DeviceMatrix(r, ND, NE);
         for (int i = tid; i < ND; i += bxy)
         {
            U(i, e) = 0.0;
            R(i, e) = B(i, e);
         }
         MFEM_SYNC_THREAD;
      }

      DGMassPreconditioner(e, NE, ND, dinv, r, z);
      DGMassAxpy(e, NE, ND, 1.0, z, 0.0, z, d); // d = z

      real_t nom = DGMassDot<NB>(e, NE, ND, d, r);
      if (nom < 0.0) { return; /* Not positive definite */ }
      real_t r0 = fmax(nom*RELTOL*RELTOL, ABSTOL*ABSTOL);
      if (nom <= r0) { return; /* Converged */ }

      DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d);
      real_t den = DGMassDot<NB>(e, NE, ND, z, d);
      if (den <= 0.0)
      {
         DGMassDot<NB>(e, NE, ND, d, d);
         // d2 > 0 => not positive definite
         if (den == 0.0) { return; }
      }

      // start iteration
      int i = 1;
      while (true)
      {
         const real_t alpha = nom/den;
         DGMassAxpy(e, NE, ND, 1.0, u, alpha, d, u); // u = u + alpha*d
         DGMassAxpy(e, NE, ND, 1.0, r, -alpha, z, r); // r = r - alpha*A*d

         DGMassPreconditioner(e, NE, ND, dinv, r, z);

         real_t betanom = DGMassDot<NB>(e, NE, ND, r, z);
         if (betanom < 0.0) { return; /* Not positive definite */ }
         if (betanom <= r0) { break; /* Converged */ }

         if (++i > MAXIT) { break; }

         const real_t beta = betanom/nom;
         DGMassAxpy(e, NE, ND, 1.0, z, beta, d, d); // d = z + beta*d
         DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d); // z = A d
         den = DGMassDot<NB>(e, NE, ND, d, z);
         if (den <= 0.0)
         {
            DGMassDot<NB>(e, NE, ND, d, d);
            // d2 > 0 => not positive definite
            if (den == 0.0) { break; }
         }
         nom = betanom;
      }

      if (CHANGE_BASIS)
      {
         DGMassBasis<DIM,D1D>(e, NE, q2d_B, u, u, d1d);
      }
   });
}

/// @cond Suppress_Doxygen_warnings

template <int DIM, int D1D, int Q1D>
inline DGMassInverse::CGKernelType DGMassInverse::CGKernels::Kernel()
{
   return &DGMassInverse::DGMassCGIteration<DIM,D1D,Q1D>;
}

inline DGMassInverse::CGKernelType DGMassInverse::CGKernels::Fallback(
   int dim, int, int)
{
   if (dim == 1) { return &DGMassInverse::DGMassCGIteration<1>; }
   else if (dim == 2) { return &DGMassInverse::DGMassCGIteration<2>; }
   else if (dim == 3) { return &DGMassInverse::DGMassCGIteration<3>; }
   else { MFEM_ABORT("Unsupported dimension."); }
}

/// @endcond

} // namespace mfem

#endif
