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

#pragma once

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/kernels.hpp"
#include "../kernels.hpp"

namespace mfem
{
namespace internal
{
namespace quadrature_interpolator
{

// Transpose gradient operation: integrate against shape function derivatives
// This is the adjoint of the Derivatives operation

template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS>
static void DerivativesTranspose1D(const int NE,
                                   const real_t *b_,
                                   const real_t *g_,
                                   const real_t *j_,
                                   const real_t *q_,
                                   real_t *e_,
                                   const int sdim,
                                   const int vdim,
                                   const int d1d,
                                   const int q1d)
{
   MFEM_CONTRACT_VAR(b_);
   const int SDIM = GRAD_PHYS ? sdim : 1;
   const auto g = Reshape(g_, q1d, d1d);
   const auto j = Reshape(j_, q1d, SDIM, NE);
   const auto q = Q_LAYOUT == QVectorLayout::byNODES ?
                  Reshape(q_, q1d, vdim, SDIM, NE):
                  Reshape(q_, vdim, SDIM, q1d, NE);
   auto e = Reshape(e_, d1d, vdim, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int el)
   {
      for (int c = 0; c < vdim; c++)
      {
         for (int d = 0; d < d1d; d++)
         {
            real_t u = 0.0;
            for (int qx = 0; qx < q1d; qx++)
            {
               // Load gradient from q-vector
               real_t dq[3] = {0.0, 0.0, 0.0};
               for (int s = 0; s < SDIM; ++s)
               {
                  if (Q_LAYOUT == QVectorLayout::byVDIM)  { dq[s] = q(c, s, qx, el); }
                  if (Q_LAYOUT == QVectorLayout::byNODES) { dq[s] = q(qx, c, s, el); }
               }

               // Apply inverse Jacobian transpose (adjoint of physical gradient)
               real_t du = dq[0];
               if (GRAD_PHYS)
               {
                  if (SDIM == 1) { du = dq[0] / j(qx, 0, el); }
                  else if (SDIM == 2)
                  {
                     const real_t Jloc[2] = {j(qx,0,el), j(qx,1,el)};
                     real_t Jinv[3];
                     kernels::CalcLeftInverse<2,1>(Jloc, Jinv);
                     du = Jinv[0]*dq[0] + Jinv[1]*dq[1];
                  }
                  else // SDIM == 3
                  {
                     const real_t Jloc[3] = {j(qx,0,el), j(qx,1,el), j(qx,2,el)};
                     real_t Jinv[3];
                     kernels::CalcLeftInverse<3,1>(Jloc, Jinv);
                     du = Jinv[0]*dq[0] + Jinv[1]*dq[1] + Jinv[2]*dq[2];
                  }
               }

               // Accumulate contribution (transpose of G matrix)
               u += g(qx, d) * du;
            }
            e(d, c, el) += u;
         }
      }
   });
}

template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int T_NBZ = 1>
static void DerivativesTranspose2D(const int NE,
                                   const real_t *b_,
                                   const real_t *g_,
                                   const real_t *j_,
                                   const real_t *q_,
                                   real_t *e_,
                                   const int sdim = 2,
                                   const int vdim = 0,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   const int SDIM = GRAD_PHYS ? sdim : 2;
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto j = Reshape(j_, Q1D, Q1D, SDIM, 2, NE);
   const auto q = Q_LAYOUT == QVectorLayout::byNODES ?
                  Reshape(q_, Q1D, Q1D, VDIM, SDIM, NE):
                  Reshape(q_, VDIM, SDIM, Q1D, Q1D, NE);
   auto e = Reshape(e_, D1D, D1D, VDIM, NE);

   mfem::forall_2D_batch(NE, D1D, D1D, NBZ, [=] MFEM_HOST_DEVICE (int el)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int tidz = MFEM_THREAD_ID(z);

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);
      DeviceMatrix B(BG[0], D1D, Q1D);
      DeviceMatrix G(BG[1], D1D, Q1D);

      MFEM_SHARED real_t sm0[NBZ][MDQ*MDQ];
      MFEM_SHARED real_t sm1[NBZ][MDQ*MDQ];

      DeviceMatrix QQ(sm0[tidz], MQ1, MQ1);
      DeviceMatrix DQ0(sm1[tidz], MD1, MQ1);
      DeviceMatrix DQ1(sm1[tidz], MD1, MQ1);  // Reuse sm1 after DQ0 is done
      DeviceMatrix DD(sm0[tidz], MD1, MD1);   // Reuse sm0 after QQ is done

      for (int c = 0; c < VDIM; c++)
      {
         // Load Q data and apply inverse Jacobian
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               // Load gradient components
               real_t dq[3] = {0.0, 0.0, 0.0};
               for (int d = 0; d < SDIM; ++d)
               {
                  if (Q_LAYOUT == QVectorLayout::byVDIM) { dq[d] = q(c, d, qx, qy, el); }
                  else { dq[d] = q(qx, qy, c, d, el); }
               }

               // Apply inverse Jacobian transpose (adjoint of physical gradient)
               real_t du[2] = {dq[0], dq[1]};
               if (GRAD_PHYS)
               {
                  if (SDIM == 2)
                  {
                     real_t Jloc[4], Jinv[4];
                     Jloc[0] = j(qx,qy,0,0,el);
                     Jloc[1] = j(qx,qy,1,0,el);
                     Jloc[2] = j(qx,qy,0,1,el);
                     Jloc[3] = j(qx,qy,1,1,el);
                     kernels::CalcInverse<2>(Jloc, Jinv);
                     const real_t U = Jinv[0]*dq[0] + Jinv[2]*dq[1];
                     const real_t V = Jinv[1]*dq[0] + Jinv[3]*dq[1];
                     du[0] = U;
                     du[1] = V;
                  }
                  else // SDIM == 3
                  {
                     real_t Jloc[6], Jinv[6];
                     Jloc[0] = j(qx,qy,0,0,el);
                     Jloc[1] = j(qx,qy,1,0,el);
                     Jloc[2] = j(qx,qy,2,0,el);
                     Jloc[3] = j(qx,qy,0,1,el);
                     Jloc[4] = j(qx,qy,1,1,el);
                     Jloc[5] = j(qx,qy,2,1,el);
                     kernels::CalcLeftInverse<3,2>(Jloc, Jinv);
                     const real_t U = Jinv[0]*dq[0] + Jinv[2]*dq[1] + Jinv[4]*dq[2];
                     const real_t V = Jinv[1]*dq[0] + Jinv[3]*dq[1] + Jinv[5]*dq[2];
                     du[0] = U;
                     du[1] = V;
                  }
               }
               QQ(qx, qy) = du[0];  // Store du/dx component
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in y-direction: QQ -> DQ0
         // (Transpose of d/dx which uses DQ1(dy,qx)*B(dy,qy))
         // Must produce DQ0(dy,qx) to match forward's DQ1 indexing
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += B(dy,qy) * QQ(qx,qy);
               }
               DQ0(dy,qx) = u;
            }
         }
         MFEM_SYNC_THREAD;

         // Apply G^T in x-direction: DQ0 -> DD
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += G(dx,qx) * DQ0(dy,qx);
               }
               DD(dx,dy) = u;
            }
         }
         MFEM_SYNC_THREAD;

         // Accumulate to output
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               e(dx,dy,c,el) += DD(dx,dy);
            }
         }
         MFEM_SYNC_THREAD;

         // Now process du/dy component
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               // Load gradient components
               real_t dq[3] = {0.0, 0.0, 0.0};
               for (int d = 0; d < SDIM; ++d)
               {
                  if (Q_LAYOUT == QVectorLayout::byVDIM) { dq[d] = q(c, d, qx, qy, el); }
                  else { dq[d] = q(qx, qy, c, d, el); }
               }

               // Apply inverse Jacobian transpose
               real_t du[2] = {dq[0], dq[1]};
               if (GRAD_PHYS)
               {
                  if (SDIM == 2)
                  {
                     real_t Jloc[4], Jinv[4];
                     Jloc[0] = j(qx,qy,0,0,el);
                     Jloc[1] = j(qx,qy,1,0,el);
                     Jloc[2] = j(qx,qy,0,1,el);
                     Jloc[3] = j(qx,qy,1,1,el);
                     kernels::CalcInverse<2>(Jloc, Jinv);
                     const real_t U = Jinv[0]*dq[0] + Jinv[2]*dq[1];
                     const real_t V = Jinv[1]*dq[0] + Jinv[3]*dq[1];
                     du[0] = U;
                     du[1] = V;
                  }
                  else // SDIM == 3
                  {
                     real_t Jloc[6], Jinv[6];
                     Jloc[0] = j(qx,qy,0,0,el);
                     Jloc[1] = j(qx,qy,1,0,el);
                     Jloc[2] = j(qx,qy,2,0,el);
                     Jloc[3] = j(qx,qy,0,1,el);
                     Jloc[4] = j(qx,qy,1,1,el);
                     Jloc[5] = j(qx,qy,2,1,el);
                     kernels::CalcLeftInverse<3,2>(Jloc, Jinv);
                     const real_t U = Jinv[0]*dq[0] + Jinv[2]*dq[1] + Jinv[4]*dq[2];
                     const real_t V = Jinv[1]*dq[0] + Jinv[3]*dq[1] + Jinv[5]*dq[2];
                     du[0] = U;
                     du[1] = V;
                  }
               }
               QQ(qx, qy) = du[1];  // Store du/dy component
            }
         }
         MFEM_SYNC_THREAD;

         // Apply G^T in y-direction: QQ -> DQ1
         // (Transpose of d/dy which uses DQ0(dy,qx)*G(dy,qy))
         // Must produce DQ1(dy,qx) to match forward's DQ0 indexing
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += G(dy,qy) * QQ(qx,qy);
               }
               DQ1(dy,qx) = u;
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in x-direction: DQ1 -> DD
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += B(dx,qx) * DQ1(dy,qx);
               }
               DD(dx,dy) = u;
            }
         }
         MFEM_SYNC_THREAD;

         // Accumulate to output
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               e(dx,dy,c,el) += DD(dx,dy);
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0>
static void DerivativesTranspose3D(const int NE,
                                   const real_t *b_,
                                   const real_t *g_,
                                   const real_t *j_,
                                   const real_t *q_,    // q_der
                                   real_t *e_,          // e_vec
                                   const int sdim = 3,
                                   const int vdim = 0,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   dbg();
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   dbg("VDIM:{}, Q1D:{}, NE:{}", VDIM, Q1D, NE);
   dbg("DIM*Q1D*Q1D*Q1D*NE: {}", VDIM*3*Q1D*Q1D*Q1D*NE);
   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto j = Reshape(j_, Q1D, Q1D, Q1D, 3, 3, NE);
   const auto q = Q_LAYOUT == QVectorLayout::byNODES ?
                  Reshape(q_, Q1D, Q1D, Q1D, VDIM, 3, NE):
                  Reshape(q_, VDIM, 3, Q1D, Q1D, Q1D, NE);
   auto e = Reshape(e_, D1D, D1D, D1D, VDIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int el)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_INTERP_1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_INTERP_1D;

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);
      DeviceMatrix B(BG[0], D1D, Q1D);
      DeviceMatrix G(BG[1], D1D, Q1D);

      MFEM_SHARED real_t sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED real_t sm1[3][MQ1*MQ1*MQ1];
      DeviceCube QQQ(sm0[0], MQ1, MQ1, MQ1);
      DeviceCube DQQ(sm1[0], MD1, MQ1, MQ1);
      DeviceCube DDQ(sm0[0], MD1, MD1, MQ1);
      DeviceCube DDD(sm1[0], MD1, MD1, MD1);

      for (int c = 0; c < VDIM; c++)
      {
         // Process du/dx component
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t dq[3];
                  if (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     dq[0] = q(c,0,qx,qy,qz,el);
                     dq[1] = q(c,1,qx,qy,qz,el);
                     dq[2] = q(c,2,qx,qy,qz,el);
                  }
                  else
                  {
                     dq[0] = q(qx,qy,qz,c,0,el);
                     dq[1] = q(qx,qy,qz,c,1,el);
                     dq[2] = q(qx,qy,qz,c,2,el);
                  }

                  real_t du[3] = {dq[0], dq[1], dq[2]};
                  if (GRAD_PHYS)
                  {
                     real_t Jloc[9], Jinv[9];
                     for (int col = 0; col < 3; col++)
                     {
                        for (int row = 0; row < 3; row++)
                        {
                           Jloc[row+3*col] = j(qx,qy,qz,row,col,el);
                        }
                     }
                     kernels::CalcInverse<3>(Jloc, Jinv);
                     const real_t U = Jinv[0]*dq[0] + Jinv[3]*dq[1] + Jinv[6]*dq[2];
                     const real_t V = Jinv[1]*dq[0] + Jinv[4]*dq[1] + Jinv[7]*dq[2];
                     const real_t W = Jinv[2]*dq[0] + Jinv[5]*dq[1] + Jinv[8]*dq[2];
                     du[0] = U; du[1] = V; du[2] = W;
                  }
                  QQQ(qx,qy,qz) = du[0];
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply G^T in x: QQQ -> DQQ (transpose of G⊗B⊗B)
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += G(dx,qx) * QQQ(qx,qy,qz);
                  }
                  DQQ(dx,qy,qz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in y: DQQ -> DDQ
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += B(dy,qy) * DQQ(dx,qy,qz);
                  }
                  DDQ(dx,dy,qz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in z: DDQ -> DDD
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += B(dz,qz) * DDQ(dx,dy,qz);
                  }
                  DDD(dx,dy,dz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Accumulate result
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  e(dx,dy,dz,c,el) += DDD(dx,dy,dz);
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Process du/dy component
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t dq[3];
                  if (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     dq[0] = q(c,0,qx,qy,qz,el);
                     dq[1] = q(c,1,qx,qy,qz,el);
                     dq[2] = q(c,2,qx,qy,qz,el);
                  }
                  else
                  {
                     dq[0] = q(qx,qy,qz,c,0,el);
                     dq[1] = q(qx,qy,qz,c,1,el);
                     dq[2] = q(qx,qy,qz,c,2,el);
                  }

                  real_t du[3] = {dq[0], dq[1], dq[2]};
                  if (GRAD_PHYS)
                  {
                     real_t Jloc[9], Jinv[9];
                     for (int col = 0; col < 3; col++)
                     {
                        for (int row = 0; row < 3; row++)
                        {
                           Jloc[row+3*col] = j(qx,qy,qz,row,col,el);
                        }
                     }
                     kernels::CalcInverse<3>(Jloc, Jinv);
                     const real_t U = Jinv[0]*dq[0] + Jinv[3]*dq[1] + Jinv[6]*dq[2];
                     const real_t V = Jinv[1]*dq[0] + Jinv[4]*dq[1] + Jinv[7]*dq[2];
                     const real_t W = Jinv[2]*dq[0] + Jinv[5]*dq[1] + Jinv[8]*dq[2];
                     du[0] = U; du[1] = V; du[2] = W;
                  }
                  QQQ(qx,qy,qz) = du[1];
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in x: QQQ -> DQQ (transpose of B⊗G⊗B)
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += B(dx,qx) * QQQ(qx,qy,qz);
                  }
                  DQQ(dx,qy,qz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply G^T in y: DQQ -> DDQ
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += G(dy,qy) * DQQ(dx,qy,qz);
                  }
                  DDQ(dx,dy,qz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in z: DDQ -> DDD
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += B(dz,qz) * DDQ(dx,dy,qz);
                  }
                  DDD(dx,dy,dz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Accumulate result
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  e(dx,dy,dz,c,el) += DDD(dx,dy,dz);
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Process du/dz component
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t dq[3];
                  if (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     dq[0] = q(c,0,qx,qy,qz,el);
                     dq[1] = q(c,1,qx,qy,qz,el);
                     dq[2] = q(c,2,qx,qy,qz,el);
                  }
                  else
                  {
                     dq[0] = q(qx,qy,qz,c,0,el);
                     dq[1] = q(qx,qy,qz,c,1,el);
                     dq[2] = q(qx,qy,qz,c,2,el);
                  }

                  real_t du[3] = {dq[0], dq[1], dq[2]};
                  if (GRAD_PHYS)
                  {
                     real_t Jloc[9], Jinv[9];
                     for (int col = 0; col < 3; col++)
                     {
                        for (int row = 0; row < 3; row++)
                        {
                           Jloc[row+3*col] = j(qx,qy,qz,row,col,el);
                        }
                     }
                     kernels::CalcInverse<3>(Jloc, Jinv);
                     const real_t U = Jinv[0]*dq[0] + Jinv[3]*dq[1] + Jinv[6]*dq[2];
                     const real_t V = Jinv[1]*dq[0] + Jinv[4]*dq[1] + Jinv[7]*dq[2];
                     const real_t W = Jinv[2]*dq[0] + Jinv[5]*dq[1] + Jinv[8]*dq[2];
                     du[0] = U; du[1] = V; du[2] = W;
                  }
                  QQQ(qx,qy,qz) = du[2];
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply G^T in z: QQQ -> DQQ (transpose of B⊗B⊗G)
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += G(dz,qz) * QQQ(qx,qy,qz);
                  }
                  //   DQQ(qx,qy,dz) = u; // 🔥🔥🔥
                  DQQ(dz,qy,qx) = u; // 🔥🔥🔥
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in y: DQQ -> DDQ
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     //  u += B(dy,qy) * DQQ(qx,qy,dz);
                     u += B(dy,qy) * DQQ(dz,qy,qx);
                  }
                  //   DDQ(qx,dy,dz) = u;
                  DDQ(dz,dy,qx) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Apply B^T in x: DDQ -> DDD
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  real_t u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     //  u += B(dx,qx) * DDQ(qx,dy,dz);
                     u += B(dx,qx) * DDQ(dz,dy,qx);
                  }
                  DDD(dx,dy,dz) = u;
               }
            }
         }
         MFEM_SYNC_THREAD;

         // Accumulate result
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  e(dx,dy,dz,c,el) += DDD(dx,dy,dz);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

} // namespace quadrature_interpolator
} // namespace internal

template<int DIM, QVectorLayout Q_LAYOUT, bool GRAD_PHYS, int VDIM, int D1D,
         int Q1D, int NBZ>
QuadratureInterpolator::GradTransposeKernelType
QuadratureInterpolator::GradTransposeKernels::Kernel()
{
   if (DIM == 1) { return internal::quadrature_interpolator::DerivativesTranspose1D<Q_LAYOUT, GRAD_PHYS>; }
   else if (DIM == 2) { return internal::quadrature_interpolator::DerivativesTranspose2D<Q_LAYOUT, GRAD_PHYS, VDIM, D1D, Q1D, NBZ>; }
   else if (DIM == 3) { return internal::quadrature_interpolator::DerivativesTranspose3D<Q_LAYOUT, GRAD_PHYS, VDIM, D1D, Q1D>; }
   else { MFEM_ABORT(""); }
}

} // namespace mfem
