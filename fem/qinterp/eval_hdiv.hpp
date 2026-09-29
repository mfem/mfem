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

// Internal header, included only by .cpp files.
// Template function implementations.

#ifndef MFEM_QUADINTERP_EVAL_HDIV_HPP
#define MFEM_QUADINTERP_EVAL_HDIV_HPP

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{
namespace quadrature_interpolator
{

// Evaluate values in H(div) on quads: DIM = SDIM = 2
// For RT(p): D1D = p + 2
template<QVectorLayout Q_LAYOUT, unsigned FLAGS, int T_D1D = 0, int T_Q1D = 0>
inline void EvalHDiv2D(const int NE,
                       const real_t *Bo_,
                       const real_t *Bc_,
                       const real_t *J_,
                       const real_t *x_,
                       real_t *y_,
                       const int d1d = 0,
                       const int q1d = 0)
{
   static constexpr int DIM = 2;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int M1D = (Q1D > D1D) ? Q1D : D1D;

   const auto bo = Reshape(Bo_, Q1D, D1D-1);
   const auto bc = Reshape(Bc_, Q1D, D1D);
   // J is used only when PHYS is true, otherwise J_ can be nullptr.
   const auto J = Reshape(J_, Q1D, Q1D, DIM, DIM, NE);
   const auto x = Reshape(x_, D1D*(D1D-1), DIM, NE);
   auto y = (FLAGS & (QuadratureInterpolator::VALUES |
                      QuadratureInterpolator::PHYSICAL_VALUES)) ?
            ((Q_LAYOUT == QVectorLayout::byNODES) ?
             Reshape(y_, Q1D, Q1D, DIM, NE) :
             Reshape(y_, DIM, Q1D, Q1D, NE)) :
            Reshape(y_, Q1D, Q1D, 1, NE);

   mfem::forall_3D(NE, M1D, M1D, DIM, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int M1D = (Q1D > D1D) ? Q1D : D1D;

      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t smo[MQ1*(MD1-1)];
      DeviceMatrix Bo(smo, D1D-1, Q1D);

      MFEM_SHARED real_t smc[MQ1*MD1];
      DeviceMatrix Bc(smc, D1D, Q1D);

      MFEM_SHARED real_t sm0[DIM*MDQ*MDQ];
      MFEM_SHARED real_t sm1[DIM*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1), DIM);
      DeviceCube QD(sm1, Q1D, D1D, DIM);
      DeviceCube QQ(sm0, Q1D, Q1D, DIM);

      // Load X, Bo and Bc into shared memory
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,M1D)
            {
               if (qx < D1D && dy < (D1D-1))
               {
                  X(qx + dy*D1D,vd) = x(qx+dy*D1D,vd,e);
               }
               if (tidz == 0 && qx < Q1D)
               {
                  if (dy < (D1D-1)) { Bo(dy,qx) = bo(qx,dy); }
                  Bc(dy,qx) = bc(qx,dy);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply B operator
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceCube Xxy(X, nx, ny, DIM);
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
      MFEM_FOREACH_THREAD(vd,z,DIM)
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
               if (FLAGS & (QuadratureInterpolator::PHYSICAL_VALUES |
                            QuadratureInterpolator::PHYSICAL_MAGNITUDES))
               {
                  QQ(qx,qy,vd) = qq;
               }
               else if (Q_LAYOUT == QVectorLayout::byNODES)
               {
                  y(qx,qy,vd,e) = qq;
               }
               else // Q_LAYOUT == QVectorLayout::byVDIM
               {
                  y(vd,qx,qy,e) = qq;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (FLAGS & (QuadratureInterpolator::PHYSICAL_VALUES |
                   QuadratureInterpolator::PHYSICAL_MAGNITUDES))
      {
         if (tidz == 0)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t u_ref[DIM], u_phys[DIM];
                  real_t J_loc[DIM*DIM];
                  // Piola transformation: u_phys = J/det(J) * u_ref
                  MFEM_UNROLL(DIM)
                  for (int d = 0; d < DIM; d++)
                  {
                     u_ref[d] = QQ(qx,qy,d);
                     MFEM_UNROLL(DIM)
                     for (int sd = 0; sd < DIM; sd++)
                     {
                        J_loc[sd+DIM*d] = J(qx,qy,sd,d,e);
                     }
                  }
                  const real_t detJ = kernels::Det<DIM>(J_loc);
                  kernels::Mult(DIM, DIM, J_loc, u_ref, u_phys);
                  kernels::Set(DIM, 1, 1_r/detJ, u_phys, u_phys);
                  if (FLAGS & QuadratureInterpolator::PHYSICAL_VALUES)
                  {
                     MFEM_UNROLL(DIM)
                     for (int sd = 0; sd < DIM; sd++)
                     {
                        if (Q_LAYOUT == QVectorLayout::byNODES)
                        {
                           y(qx,qy,sd,e) = u_phys[sd];
                        }
                        else // Q_LAYOUT == QVectorLayout::byVDIM
                        {
                           y(sd,qx,qy,e) = u_phys[sd];
                        }
                     }
                  }
                  else if (FLAGS & QuadratureInterpolator::PHYSICAL_MAGNITUDES)
                  {
                     y(qx,qy,0,e) = kernels::Norml2(DIM, u_phys);
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Evaluate values in H(div) on hexes: DIM = SDIM = 3
// For RT(p): D1D = p + 2
template<QVectorLayout Q_LAYOUT, unsigned FLAGS, int T_D1D = 0, int T_Q1D = 0>
inline void EvalHDiv3D(const int NE,
                       const real_t *Bo_,
                       const real_t *Bc_,
                       const real_t *J_,
                       const real_t *x_,
                       real_t *y_,
                       const int d1d = 0,
                       const int q1d = 0)
{
   static constexpr int DIM = 3;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto bo = Reshape(Bo_, Q1D, D1D-1);
   const auto bc = Reshape(Bc_, Q1D, D1D);
   // J is used only when PHYS is true, otherwise J_ can be nullptr.
   const auto J = Reshape(J_, Q1D, Q1D, Q1D, DIM, DIM, NE);
   const auto x = Reshape(x_, D1D*(D1D-1)*(D1D-1), DIM, NE);
   auto y = (FLAGS & (QuadratureInterpolator::VALUES |
                      QuadratureInterpolator::PHYSICAL_VALUES)) ?
            ((Q_LAYOUT == QVectorLayout::byNODES) ?
             Reshape(y_, Q1D, Q1D, Q1D, DIM, NE) :
             Reshape(y_, DIM, Q1D, Q1D, Q1D, NE)) :
            Reshape(y_, Q1D, Q1D, Q1D, 1, NE);

   mfem::forall_3D(NE, Q1D, Q1D, DIM, [=] MFEM_HOST_DEVICE (int e)
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

      MFEM_SHARED real_t sm0[DIM*MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[DIM*MDQ*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1)*(D1D-1), DIM);
      DeviceTensor<4> QDD(sm1, Q1D, D1D, D1D, DIM);
      DeviceTensor<4> QQD(sm0, Q1D, Q1D, D1D, DIM);
      DeviceTensor<4> QQQ(sm1, Q1D, Q1D, Q1D, DIM);

      // Load X into shared memory
      MFEM_FOREACH_THREAD(vd,z,DIM)
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
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceTensor<4> Xxyz(X, nx, ny, nz, DIM);
         DeviceMatrix Bx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MD1];
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
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MD1];
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
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix Bz = (vd == 2) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MQ1];
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
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  if (FLAGS & (QuadratureInterpolator::PHYSICAL_VALUES |
                               QuadratureInterpolator::PHYSICAL_MAGNITUDES))
                  {
                     QQQ(qx,qy,qz,vd) = u[qz];
                  }
                  else if (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     y(qx,qy,qz,vd,e) = u[qz];
                  }
                  else // Q_LAYOUT == QVectorLayout::byVDIM
                  {
                     y(vd,qx,qy,qz,e) = u[qz];
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (FLAGS & (QuadratureInterpolator::PHYSICAL_VALUES |
                   QuadratureInterpolator::PHYSICAL_MAGNITUDES))
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t u_ref[DIM], u_phys[DIM];
                  real_t J_loc[DIM*DIM];
                  // Piola transformation: u_phys = J/det(J) * u_ref
                  MFEM_UNROLL(DIM)
                  for (int d = 0; d < DIM; d++)
                  {
                     u_ref[d] = QQQ(qx,qy,qz,d);
                     MFEM_UNROLL(DIM)
                     for (int sd = 0; sd < DIM; sd++)
                     {
                        J_loc[sd+DIM*d] = J(qx,qy,qz,sd,d,e);
                     }
                  }
                  const real_t detJ = kernels::Det<DIM>(J_loc);
                  kernels::Mult(DIM, DIM, J_loc, u_ref, u_phys);
                  kernels::Set(DIM, 1, 1_r/detJ, u_phys, u_phys);
                  if (FLAGS & QuadratureInterpolator::PHYSICAL_VALUES)
                  {
                     MFEM_UNROLL(DIM)
                     for (int sd = 0; sd < DIM; sd++)
                     {
                        if (Q_LAYOUT == QVectorLayout::byNODES)
                        {
                           y(qx,qy,qz,sd,e) = u_phys[sd];
                        }
                        else // Q_LAYOUT == QVectorLayout::byVDIM
                        {
                           y(sd,qx,qy,qz,e) = u_phys[sd];
                        }
                     }
                  }
                  else if (FLAGS & QuadratureInterpolator::PHYSICAL_MAGNITUDES)
                  {
                     y(qx,qy,qz,0,e) = kernels::Norml2(DIM, u_phys);
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}


// Evaluate the reference or physical divergence in H(div) on quads.
// div u = du_x/dx + du_y/dy.
// For phys divergence div /= det(J).
template<unsigned FLAGS, int T_D1D = 0, int T_Q1D = 0>
inline void EvalHDivDiv2D(const int NE,
                          const real_t *Bo_,
                          const real_t *Gc_,
                          const real_t *detJ_,
                          const real_t *x_,
                          real_t *div_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   static constexpr int DIM = 2;
   constexpr bool PHYSICAL = FLAGS & QuadratureInterpolator::PHYSICAL_DERIVATIVES;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int M1D = (Q1D > D1D) ? Q1D : D1D;

   const auto bo = Reshape(Bo_, Q1D, D1D-1);
   const auto gc = Reshape(Gc_, Q1D, D1D);
   // detJ is used only when PHYSICAL is true, otherwise detJ_ can be nullptr.
   const auto detJ = Reshape(detJ_, Q1D, Q1D, NE);
   const auto x = Reshape(x_, D1D*(D1D-1), DIM, NE);
   auto div = Reshape(div_, Q1D, Q1D, NE);

   mfem::forall_3D(NE, M1D, M1D, DIM, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int M1D = (Q1D > D1D) ? Q1D : D1D;

      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t smo[MQ1*(MD1-1)];
      DeviceMatrix Bo(smo, D1D-1, Q1D);

      MFEM_SHARED real_t smc[MQ1*MD1];
      DeviceMatrix Gc(smc, D1D, Q1D);

      MFEM_SHARED real_t sm0[DIM*MDQ*MDQ];
      MFEM_SHARED real_t sm1[DIM*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1), DIM);
      DeviceCube QD(sm1, Q1D, D1D, DIM);
      DeviceCube QQ(sm0, Q1D, Q1D, DIM);

      // Load X, Bo and Gc into shared memory
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,M1D)
            {
               if (qx < D1D && dy < (D1D-1))
               {
                  X(qx + dy*D1D,vd) = x(qx+dy*D1D,vd,e);
               }
               if (tidz == 0 && qx < Q1D)
               {
                  if (dy < (D1D-1)) { Bo(dy,qx) = bo(qx,dy); }
                  Gc(dy,qx) = gc(qx,dy);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Contract x: the derivative for u_x, open basis for u_y
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceCube Xxy(X, nx, ny, DIM);
         DeviceMatrix Bx = (vd == 0) ? Gc : Bo;
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
      // Contract y: open basis for u_x, the derivative for u_y
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Gc : Bo;
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
      // Sum the components into the divergence
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t q_div = 0.0;
               MFEM_UNROLL(DIM)
               for (int vd = 0; vd < DIM; ++vd) { q_div += QQ(qx,qy,vd); }
               if constexpr (PHYSICAL) { q_div /= detJ(qx,qy,e); }
               div(qx,qy,e) = q_div;
            }
         }
      }
   });
}

// Evaluate the reference or physical divergence in H(div) on hexes.
// div u = du_x/dx + du_y/dy + du_z/dz.
template<unsigned FLAGS, int T_D1D = 0, int T_Q1D = 0>
inline void EvalHDivDiv3D(const int NE,
                          const real_t *Bo_,
                          const real_t *Gc_,
                          const real_t *detJ_,
                          const real_t *x_,
                          real_t *div_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   static constexpr int DIM = 3;
   constexpr bool PHYSICAL = FLAGS & QuadratureInterpolator::PHYSICAL_DERIVATIVES;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto bo = Reshape(Bo_, Q1D, D1D-1);
   const auto gc = Reshape(Gc_, Q1D, D1D);
   // detJ is used only when PHYSICAL is true, otherwise detJ_ can be nullptr.
   const auto detJ = Reshape(detJ_, Q1D, Q1D, Q1D, NE);
   const auto x = Reshape(x_, D1D*(D1D-1)*(D1D-1), DIM, NE);
   auto div = Reshape(div_, Q1D, Q1D, Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, DIM, [=] MFEM_HOST_DEVICE (int e)
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
      DeviceMatrix Gc(smc, D1D, Q1D);

      MFEM_SHARED real_t sm0[DIM*MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[DIM*MDQ*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1)*(D1D-1), DIM);
      DeviceTensor<4> QDD(sm1, Q1D, D1D, D1D, DIM);
      DeviceTensor<4> QQD(sm0, Q1D, Q1D, D1D, DIM);
      DeviceTensor<4> QQQ(sm1, Q1D, Q1D, Q1D, DIM);

      // Load X into shared memory
      MFEM_FOREACH_THREAD(vd,z,DIM)
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
      // Load Bo and Gc into shared memory
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
               Gc(d,q) = gc(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Contract x: the derivative for u_x, open basis for u_y and u_z
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceTensor<4> Xxyz(X, nx, ny, nz, DIM);
         DeviceMatrix Bx = (vd == 0) ? Gc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MD1];
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
      // Contract y: the derivative for u_y, open basis for u_x and u_z
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Gc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MD1];
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
      // Contract z: the derivative for u_z, open basis for u_x and u_y
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix Bz = (vd == 2) ? Gc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MQ1];
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
      // Sum the components into the divergence
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t q_div = 0.0;
               MFEM_UNROLL(DIM)
               for (int vd = 0; vd < DIM; ++vd) { q_div += QQQ(qx,qy,qz,vd); }
               if constexpr (PHYSICAL) { q_div /= detJ(qx,qy,qz,e); }
               div(qx,qy,qz,e) = q_div;
            }
         }
      }
   });
}

// Transpose of EvalHDiv2D and EvalHDivDiv2D on quads, accummulating to E-vec.

// NOTE: H(curl) not supported yet, but in principle it would differ in using
// Bo/Bc in opposite logic (closed for c != k).
// So maybe we may be able to modify the code to handle both H(div) and H(curl)?
template<QVectorLayout Q_LAYOUT, unsigned FLAGS, int T_D1D = 0, int T_Q1D = 0>
inline void EvalHDivTranspose2D(const int NE,
                                const real_t *Bo_,
                                const real_t *Bc_,
                                const real_t *detJ_,
                                const real_t *y_,
                                real_t *x_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   static constexpr int DIM = 2;
   constexpr bool DIV = FLAGS & (QuadratureInterpolator::DERIVATIVES |
                                 QuadratureInterpolator::PHYSICAL_DERIVATIVES);
   constexpr bool PHYSICAL = FLAGS & QuadratureInterpolator::PHYSICAL_DERIVATIVES;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int M1D = (Q1D > D1D) ? Q1D : D1D;

   const auto bo = Reshape(Bo_, Q1D, D1D-1);
   const auto bc = Reshape(Bc_, Q1D, D1D);
   // detJ is used only when PHYSICAL is true, otherwise detJ_ can be nullptr.
   const auto detJ = Reshape(detJ_, Q1D, Q1D, NE);
   const auto y = DIV ? Reshape(y_, Q1D, Q1D, 1, NE) :
                  ((Q_LAYOUT == QVectorLayout::byNODES) ?
                   Reshape(y_, Q1D, Q1D, DIM, NE) :
                   Reshape(y_, DIM, Q1D, Q1D, NE));
   auto x = Reshape(x_, D1D*(D1D-1), DIM, NE);

   mfem::forall_3D(NE, M1D, M1D, DIM, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int M1D = (Q1D > D1D) ? Q1D : D1D;

      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t smo[MQ1*(MD1-1)];
      DeviceMatrix Bo(smo, D1D-1, Q1D);

      MFEM_SHARED real_t smc[MQ1*MD1];
      DeviceMatrix Bc(smc, D1D, Q1D);

      MFEM_SHARED real_t sm0[DIM*MDQ*MDQ];
      MFEM_SHARED real_t sm1[DIM*MDQ*MDQ];
      DeviceCube QQ(sm0, Q1D, Q1D, DIM);
      DeviceCube QD(sm1, Q1D, D1D, DIM);

      // Load y, Bo and Bc into shared memory
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         MFEM_FOREACH_THREAD(qy,y,M1D)
         {
            MFEM_FOREACH_THREAD(qx,x,M1D)
            {
               if (qx < Q1D && qy < Q1D)
               {
                  real_t v;
                  if constexpr (DIV)
                  {
                     v = y(qx,qy,0,e);
                     if constexpr (PHYSICAL) { v /= detJ(qx,qy,e); }
                  }
                  else if (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     v = y(qx,qy,vd,e);
                  }
                  else // Q_LAYOUT == QVectorLayout::byVDIM
                  {
                     v = y(vd,qx,qy,e);
                  }
                  QQ(qx,qy,vd) = v;
               }

               if (tidz == 0 && qx < Q1D && qy < D1D)
               {
                  if (qy < (D1D-1)) { Bo(qy,qx) = bo(qx,qy); }
                  Bc(qy,qx) = bc(qx,qy);
               }
            }
         }
      }
      
      MFEM_SYNC_THREAD;

      // Contract y: open basis Bo for u_x, closed Bc for u_y
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t qd = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  qd += QQ(qx,qy,vd) * By(dy,qy);
               }
               QD(qx,dy,vd) = qd;
            }
         }
      }

      MFEM_SYNC_THREAD;
      
      // Contract x and add into the component blocks: closed basis for u_x, open for u_y
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix Bx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t dd = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  dd += QD(qx,dy,vd) * Bx(dx,qx);
               }
               x(dx + dy*nx,vd,e) += dd;
            }
         }
      }
   });
}

// Transpose of EvalHDiv3D and EvalHDivDiv3D on hexes.
template<QVectorLayout Q_LAYOUT, unsigned FLAGS, int T_D1D = 0, int T_Q1D = 0>
inline void EvalHDivTranspose3D(const int NE,
                                const real_t *Bo_,
                                const real_t *Bc_,
                                const real_t *detJ_,
                                const real_t *y_,
                                real_t *x_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   static constexpr int DIM = 3;
   constexpr bool DIV = FLAGS & (QuadratureInterpolator::DERIVATIVES |
                                 QuadratureInterpolator::PHYSICAL_DERIVATIVES);
   constexpr bool PHYSICAL = FLAGS & QuadratureInterpolator::PHYSICAL_DERIVATIVES;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto bo = Reshape(Bo_, Q1D, D1D-1);
   const auto bc = Reshape(Bc_, Q1D, D1D);
   // detJ is used only when PHYSICAL is true, otherwise detJ_ can be nullptr.
   const auto detJ = Reshape(detJ_, Q1D, Q1D, Q1D, NE);
   const auto y = DIV ? Reshape(y_, Q1D, Q1D, Q1D, 1, NE) :
                  ((Q_LAYOUT == QVectorLayout::byNODES) ?
                   Reshape(y_, Q1D, Q1D, Q1D, DIM, NE) :
                   Reshape(y_, DIM, Q1D, Q1D, Q1D, NE));
   auto x = Reshape(x_, D1D*(D1D-1)*(D1D-1), DIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, DIM, [=] MFEM_HOST_DEVICE (int e)
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

      MFEM_SHARED real_t sm0[DIM*MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[DIM*MDQ*MDQ*MDQ];
      DeviceTensor<4> QQQ(sm0, Q1D, Q1D, Q1D, DIM);
      DeviceTensor<4> QQD(sm1, Q1D, Q1D, D1D, DIM);
      DeviceTensor<4> QDD(sm0, Q1D, D1D, D1D, DIM);

      // Load y into shared memory
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  real_t v;
                  if constexpr (DIV)
                  {
                     v = y(qx,qy,qz,0,e);
                     if constexpr (PHYSICAL) { v /= detJ(qx,qy,qz,e); }
                  }
                  else if (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     v = y(qx,qy,qz,vd,e);
                  }
                  else // Q_LAYOUT == QVectorLayout::byVDIM
                  {
                     v = y(vd,qx,qy,qz,e);
                  }
                  QQQ(qx,qy,qz,vd) = v;
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
      // Contract z: closed basis for u_z, open basis for u_x and u_y
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix Bz = (vd == 2) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MD1];
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  MFEM_UNROLL(MD1)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += QQQ(qx,qy,qz,vd) * Bz(dz,qz);
                  }
               }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { QQD(qx,qy,dz,vd) = u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Contract y: closed basis for u_y, open basis for u_x and u_z
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u[MD1];
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  MFEM_UNROLL(MD1)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += QQD(qx,qy,dz,vd) * By(dy,qy);
                  }
               }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { QDD(qx,dy,dz,vd) = u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Contract x and add into the component blocks: closed basis for u_x, open basis for u_y and u_z
      MFEM_FOREACH_THREAD(vd,z,DIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix Bx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u[MD1];
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  MFEM_UNROLL(MD1)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += QDD(qx,dy,dz,vd) * Bx(dx,qx);
                  }
               }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz)
               {
                  x(dx + (dy + dz*ny)*nx,vd,e) += u[dz];
               }
            }
         }
      }
   });
}

} // namespace quadrature_interpolator
} // namespace internal

/// @cond Suppress_Doxygen_warnings

template<int DIM, QVectorLayout Q_LAYOUT, unsigned FLAGS, int D1D, int Q1D>
QuadratureInterpolator::TensorEvalHDivKernelType
QuadratureInterpolator::TensorEvalHDivKernels::Kernel()
{
   using namespace internal::quadrature_interpolator;
   static_assert(DIM == 2 || DIM == 3, "only DIM=2 and DIM=3 are implemented!");
   if constexpr (DIM == 2)
   {
      return EvalHDiv2D<Q_LAYOUT, FLAGS, D1D, Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return EvalHDiv3D<Q_LAYOUT, FLAGS, D1D, Q1D>;
   }
   MFEM_ABORT("only DIM=2 and DIM=3 are implemented!");
}

template<int DIM, unsigned FLAGS, int D1D, int Q1D>
QuadratureInterpolator::TensorEvalHDivDivKernelType
QuadratureInterpolator::TensorEvalHDivDivKernels::Kernel()
{
   using namespace internal::quadrature_interpolator;
   static_assert(DIM == 2 || DIM == 3, "only DIM=2 and DIM=3 are implemented!");
   if constexpr (DIM == 2)
   {
      return EvalHDivDiv2D<FLAGS, D1D, Q1D>;
   }
   else
   {
      return EvalHDivDiv3D<FLAGS, D1D, Q1D>;
   }
}

template<int DIM, QVectorLayout Q_LAYOUT, unsigned FLAGS, int D1D, int Q1D>
QuadratureInterpolator::TensorEvalHDivKernelType
QuadratureInterpolator::TensorEvalHDivTransposeKernels::Kernel()
{
   using namespace internal::quadrature_interpolator;
   static_assert(DIM == 2 || DIM == 3, "only DIM=2 and DIM=3 are implemented!");
   if constexpr (DIM == 2)
   {
      return EvalHDivTranspose2D<Q_LAYOUT, FLAGS, D1D, Q1D>;
   }
   else
   {
      return EvalHDivTranspose3D<Q_LAYOUT, FLAGS, D1D, Q1D>;
   }
}

/// @endcond

} // namespace mfem

#endif // MFEM_QUADINTERP_EVAL_HDIV_HPP
