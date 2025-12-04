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

#ifndef MFEM_BILININTEG_HCURL_KERNELS_HPP
#define MFEM_BILININTEG_HCURL_KERNELS_HPP

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

// Piola transformation in H(curl): w = dF^{-T} \hat{w}
// curl w = (1 / det (dF)) dF \hat{curl} \hat{w}

namespace mfem
{
/// \cond DO_NOT_DOCUMENT
namespace internal
{

// PA H(curl) Mass Diagonal 2D kernel
void PAHcurlMassAssembleDiagonal2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &bo,
                                   const Array<real_t> &bc,
                                   const Vector &pa_data,
                                   Vector &diag);

// PA H(curl) Mass Diagonal 3D kernel
void PAHcurlMassAssembleDiagonal3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &bo,
                                   const Array<real_t> &bc,
                                   const Vector &pa_data,
                                   Vector &diag);

// Shared memory PA H(curl) Mass Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAHcurlMassAssembleDiagonal3D(const int d1d,
                                              const int q1d,
                                              const int NE,
                                              const bool symmetric,
                                              const Array<real_t> &bo,
                                              const Array<real_t> &bc,
                                              const Vector &pa_data,
                                              Vector &diag)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t sBo[MQ1D][MD1D];
      MFEM_SHARED real_t sBc[MQ1D][MD1D];

      real_t op3[3];
      MFEM_SHARED real_t sop[3][MQ1D][MQ1D];

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               op3[0] = op(qx,qy,qz,0,e);
               op3[1] = op(qx,qy,qz,symmetric ? 3 : 4,e);
               op3[2] = op(qx,qy,qz,symmetric ? 5 : 8,e);
            }
         }
      }

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               sBc[q][d] = Bc(q,d);
               if (d < D1D-1)
               {
                  sBo[q][d] = Bo(q,d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      int osc = 0;
      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         real_t dxyz = 0.0;

         for (int qz=0; qz < Q1D; ++qz)
         {
            if (tidz == qz)
            {
               for (int i=0; i<3; ++i)
               {
                  sop[i][tidx][tidy] = op3[i];
               }
            }

            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               const real_t wz = ((c == 2) ? sBo[qz][dz] : sBc[qz][dz]);

               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const real_t wy = ((c == 1) ? sBo[qy][dy] : sBc[qy][dy]);

                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const real_t wx = ((c == 0) ? sBo[qx][dx] : sBc[qx][dx]);
                           dxyz += sop[c][qx][qy] * wx * wx * wy * wy * wz * wz;
                        }
                     }
                  }
               }
            }

            MFEM_SYNC_THREAD;
         }  // qz loop

         MFEM_FOREACH_THREAD(dz,z,D1Dz)
         {
            MFEM_FOREACH_THREAD(dy,y,D1Dy)
            {
               MFEM_FOREACH_THREAD(dx,x,D1Dx)
               {
                  D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += dxyz;
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // c loop
   }); // end of element loop
}

// PA H(curl) Mass Apply 2D kernel
void PAHcurlMassApply2D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const bool symmetric,
                        const Array<real_t> &bo,
                        const Array<real_t> &bc,
                        const Array<real_t> &bot,
                        const Array<real_t> &bct,
                        const Vector &pa_data,
                        const Vector &x,
                        Vector &y);

// PA H(curl) Mass Apply 3D kernel
void PAHcurlMassApply3D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const bool symmetric,
                        const Array<real_t> &bo,
                        const Array<real_t> &bc,
                        const Array<real_t> &bot,
                        const Array<real_t> &bct,
                        const Vector &pa_data,
                        const Vector &x,
                        Vector &y);

// Shared memory PA H(curl) Mass Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAHcurlMassApply3D(const int d1d,
                                   const int q1d,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &bo,
                                   const Array<real_t> &bc,
                                   const Array<real_t> &bot,
                                   const Array<real_t> &bct,
                                   const Vector &pa_data,
                                   const Vector &x,
                                   Vector &y)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const int dataSize = symmetric ? 6 : 9;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, dataSize, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t sBo[MQ1D][MD1D];
      MFEM_SHARED real_t sBc[MQ1D][MD1D];

      real_t op9[9];
      MFEM_SHARED real_t sop[9*MQ1D*MQ1D];
      MFEM_SHARED real_t mass[MQ1D][MQ1D][3];

      MFEM_SHARED real_t sX[MD1D][MD1D][MD1D];

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               for (int i=0; i<dataSize; ++i)
               {
                  op9[i] = op(qx,qy,qz,i,e);
               }
            }
         }
      }

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               sBc[q][d] = Bc(q,d);
               if (d < D1D-1)
               {
                  sBo[q][d] = Bo(q,d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int qz=0; qz < Q1D; ++qz)
      {
         int osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     sX[dz][dy][dx] = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  }
               }
            }
            MFEM_SYNC_THREAD;

            if (tidz == qz)
            {
               for (int i=0; i<dataSize; ++i)
               {
                  sop[i + (dataSize*tidx) + (dataSize*Q1D*tidy)] = op9[i];
               }

               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     real_t u = 0.0;

                     for (int dz = 0; dz < D1Dz; ++dz)
                     {
                        const real_t wz = (c == 2) ? sBo[qz][dz] : sBc[qz][dz];
                        for (int dy = 0; dy < D1Dy; ++dy)
                        {
                           const real_t wy = (c == 1) ? sBo[qy][dy] : sBc[qy][dy];
                           for (int dx = 0; dx < D1Dx; ++dx)
                           {
                              const real_t t = sX[dz][dy][dx];
                              const real_t wx = (c == 0) ? sBo[qx][dx] : sBc[qx][dx];
                              u += t * wx * wy * wz;
                           }
                        }
                     }

                     mass[qy][qx][c] = u;
                  } // qx
               } // qy
            } // tidz == qz

            osc += D1Dx * D1Dy * D1Dz;
            MFEM_SYNC_THREAD;
         } // c

         MFEM_SYNC_THREAD;  // Sync mass[qy][qx][d] and sop

         osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t dxyz = 0.0;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               const real_t wz = (c == 2) ? sBo[qz][dz] : sBc[qz][dz];

               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const real_t wy = (c == 1) ? sBo[qy][dy] : sBc[qy][dy];
                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const int os = (dataSize*qx) + (dataSize*Q1D*qy);
                           const int id1 = os + ((c == 0) ? 0 : ((c == 1) ? (symmetric ? 1 : 3) :
                                                                 (symmetric ? 2 : 6))); // O11, O21, O31
                           const int id2 = os + ((c == 0) ? 1 : ((c == 1) ? (symmetric ? 3 : 4) :
                                                                 (symmetric ? 4 : 7))); // O12, O22, O32
                           const int id3 = os + ((c == 0) ? 2 : ((c == 1) ? (symmetric ? 4 : 5) :
                                                                 (symmetric ? 5 : 8))); // O13, O23, O33

                           const real_t m_c = (sop[id1] * mass[qy][qx][0]) + (sop[id2] * mass[qy][qx][1]) +
                                              (sop[id3] * mass[qy][qx][2]);

                           const real_t wx = (c == 0) ? sBo[qx][dx] : sBc[qx][dx];
                           dxyz += m_c * wx * wy * wz;
                        }
                     }
                  }
               }
            }

            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += dxyz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         } // c loop
      } // qz
   }); // end of element loop
}

// PA H(curl) curl-curl Assemble 2D kernel
void PACurlCurlSetup2D(const int Q1D,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff,
                       Vector &op);

// PA H(curl) curl-curl Assemble 3D kernel
void PACurlCurlSetup3D(const int Q1D,
                       const int coeffDim,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff,
                       Vector &op);

// PA H(curl) curl-curl Diagonal 2D kernel
void PACurlCurlAssembleDiagonal2D(const int D1D,
                                  const int Q1D,
                                  const bool symmetric, // unused
                                  const int NE,
                                  const Array<real_t> &bo,
                                  const Array<real_t> &bc, // unused
                                  const Array<real_t> &go, // unused
                                  const Array<real_t> &gc,
                                  const Vector &pa_data,
                                  Vector &diag);

// PA H(curl) curl-curl Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PACurlCurlAssembleDiagonal3D(const int d1d,
                                         const int q1d,
                                         const bool symmetric,
                                         const int NE,
                                         const Array<real_t> &bo,
                                         const Array<real_t> &bc,
                                         const Array<real_t> &go,
                                         const Array<real_t> &gc,
                                         const Vector &pa_data,
                                         Vector &diag)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Go = Reshape(go.Read(), Q1D, D1D-1);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   const int s = symmetric ? 6 : 9;
   const int i11 = 0;
   const int i12 = 1;
   const int i13 = 2;
   const int i21 = symmetric ? i12 : 3;
   const int i22 = symmetric ? 3 : 4;
   const int i23 = symmetric ? 4 : 5;
   const int i31 = symmetric ? i13 : 6;
   const int i32 = symmetric ? i23 : 7;
   const int i33 = symmetric ? 5 : 8;

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
      // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      // For each c, we will keep 9 arrays for derivatives multiplied by the 9 entries of the 3x3 matrix (dF^T C dF),
      // which may be non-symmetric depending on a possibly non-symmetric matrix coefficient.

      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         real_t zt[MQ1D][MQ1D][MD1D][9][3];

         // z contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int i=0; i<s; ++i)
                  {
                     for (int d=0; d<3; ++d)
                     {
                        zt[qx][qy][dz][i][d] = 0.0;
                     }
                  }

                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t wz = ((c == 2) ? Bo(qz,dz) : Bc(qz,dz));
                     const real_t wDz = ((c == 2) ? Go(qz,dz) : Gc(qz,dz));

                     for (int i=0; i<s; ++i)
                     {
                        zt[qx][qy][dz][i][0] += wz * wz * op(qx,qy,qz,i,e);
                        zt[qx][qy][dz][i][1] += wDz * wz * op(qx,qy,qz,i,e);
                        zt[qx][qy][dz][i][2] += wDz * wDz * op(qx,qy,qz,i,e);
                     }
                  }
               }
            }
         }  // end of z contraction

         real_t yt[MQ1D][MD1D][MD1D][9][3][3];

         // y contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1Dz; ++dz)
            {
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int i=0; i<s; ++i)
                  {
                     for (int d=0; d<3; ++d)
                        for (int j=0; j<3; ++j)
                        {
                           yt[qx][dy][dz][i][d][j] = 0.0;
                        }
                  }

                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = ((c == 1) ? Bo(qy,dy) : Bc(qy,dy));
                     const real_t wDy = ((c == 1) ? Go(qy,dy) : Gc(qy,dy));

                     for (int i=0; i<s; ++i)
                     {
                        for (int d=0; d<3; ++d)
                        {
                           yt[qx][dy][dz][i][d][0] += wy * wy * zt[qx][qy][dz][i][d];
                           yt[qx][dy][dz][i][d][1] += wDy * wy * zt[qx][qy][dz][i][d];
                           yt[qx][dy][dz][i][d][2] += wDy * wDy * zt[qx][qy][dz][i][d];
                        }
                     }
                  }
               }
            }
         }  // end of y contraction

         // x contraction
         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     const real_t wDx = ((c == 0) ? Go(qx,dx) : Gc(qx,dx));

                     // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
                     // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
                     // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

                     /*
                       const double O11 = op(q,0,e);
                       const double O12 = op(q,1,e);
                       const double O13 = op(q,2,e);
                       const double O22 = op(q,3,e);
                       const double O23 = op(q,4,e);
                       const double O33 = op(q,5,e);
                     */

                     if (c == 0)
                     {
                        // (u_0)_{x_2} (O22 (u_0)_{x_2} - O23 (u_0)_{x_1}) - (u_0)_{x_1} (O32 (u_0)_{x_2} - O33 (u_0)_{x_1})
                        const real_t sumy = yt[qx][dy][dz][i22][2][0] - yt[qx][dy][dz][i23][1][1]
                                            - yt[qx][dy][dz][i32][1][1] + yt[qx][dy][dz][i33][0][2];

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += sumy * wx * wx;
                     }
                     else if (c == 1)
                     {
                        // (u_1)_{x_2} (O11 (u_1)_{x_2} - O13 (u_1)_{x_0}) + (u_1)_{x_0} (-O31 (u_1)_{x_2} + O33 (u_1)_{x_0})
                        const real_t d = (yt[qx][dy][dz][i11][2][0] * wx * wx)
                                         - ((yt[qx][dy][dz][i13][1][0] + yt[qx][dy][dz][i31][1][0]) * wDx * wx)
                                         + (yt[qx][dy][dz][i33][0][0] * wDx * wDx);

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += d;
                     }
                     else
                     {
                        // (u_2)_{x_1} (O11 (u_2)_{x_1} - O12 (u_2)_{x_0}) - (u_2)_{x_0} (O21 (u_2)_{x_1} - O22 (u_2)_{x_0})
                        const real_t d = (yt[qx][dy][dz][i11][0][2] * wx * wx)
                                         - ((yt[qx][dy][dz][i12][0][1] + yt[qx][dy][dz][i21][0][1]) * wDx * wx)
                                         + (yt[qx][dy][dz][i22][0][0] * wDx * wDx);

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += d;
                     }
                  }
               }
            }
         }  // end of x contraction

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

// Shared memory PA H(curl) curl-curl Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPACurlCurlAssembleDiagonal3D(const int d1d,
                                             const int q1d,
                                             const bool symmetric,
                                             const int NE,
                                             const Array<real_t> &bo,
                                             const Array<real_t> &bc,
                                             const Array<real_t> &go,
                                             const Array<real_t> &gc,
                                             const Vector &pa_data,
                                             Vector &diag)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Go = Reshape(go.Read(), Q1D, D1D-1);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   const int s = symmetric ? 6 : 9;
   const int i11 = 0;
   const int i12 = 1;
   const int i13 = 2;
   const int i21 = symmetric ? i12 : 3;
   const int i22 = symmetric ? 3 : 4;
   const int i23 = symmetric ? 4 : 5;
   const int i31 = symmetric ? i13 : 6;
   const int i32 = symmetric ? i23 : 7;
   const int i33 = symmetric ? 5 : 8;

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
      // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t sBo[MQ1D][MD1D];
      MFEM_SHARED real_t sBc[MQ1D][MD1D];
      MFEM_SHARED real_t sGo[MQ1D][MD1D];
      MFEM_SHARED real_t sGc[MQ1D][MD1D];

      real_t ope[9];
      MFEM_SHARED real_t sop[9][MQ1D][MQ1D];

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               for (int i=0; i<s; ++i)
               {
                  ope[i] = op(qx,qy,qz,i,e);
               }
            }
         }
      }

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               sBc[q][d] = Bc(q,d);
               sGc[q][d] = Gc(q,d);
               if (d < D1D-1)
               {
                  sBo[q][d] = Bo(q,d);
                  sGo[q][d] = Go(q,d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      int osc = 0;
      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         real_t dxyz = 0.0;

         for (int qz=0; qz < Q1D; ++qz)
         {
            if (tidz == qz)
            {
               for (int i=0; i<s; ++i)
               {
                  sop[i][tidx][tidy] = ope[i];
               }
            }

            MFEM_SYNC_THREAD;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               const real_t wz = ((c == 2) ? sBo[qz][dz] : sBc[qz][dz]);
               const real_t wDz = ((c == 2) ? sGo[qz][dz] : sGc[qz][dz]);

               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const real_t wy = ((c == 1) ? sBo[qy][dy] : sBc[qy][dy]);
                        const real_t wDy = ((c == 1) ? sGo[qy][dy] : sGc[qy][dy]);

                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const real_t wx = ((c == 0) ? sBo[qx][dx] : sBc[qx][dx]);
                           const real_t wDx = ((c == 0) ? sGo[qx][dx] : sGc[qx][dx]);

                           if (c == 0)
                           {
                              // (u_0)_{x_2} (O22 (u_0)_{x_2} - O23 (u_0)_{x_1}) - (u_0)_{x_1} (O32 (u_0)_{x_2} - O33 (u_0)_{x_1})

                              // (u_0)_{x_2} O22 (u_0)_{x_2}
                              dxyz += sop[i22][qx][qy] * wx * wx * wy * wy * wDz * wDz;

                              // -(u_0)_{x_2} O23 (u_0)_{x_1} - (u_0)_{x_1} O32 (u_0)_{x_2}
                              dxyz += -(sop[i23][qx][qy] + sop[i32][qx][qy]) * wx * wx * wDy * wy * wDz * wz;

                              // (u_0)_{x_1} O33 (u_0)_{x_1}
                              dxyz += sop[i33][qx][qy] * wx * wx * wDy * wDy * wz * wz;
                           }
                           else if (c == 1)
                           {
                              // (u_1)_{x_2} (O11 (u_1)_{x_2} - O13 (u_1)_{x_0}) + (u_1)_{x_0} (-O31 (u_1)_{x_2} + O33 (u_1)_{x_0})

                              // (u_1)_{x_2} O11 (u_1)_{x_2}
                              dxyz += sop[i11][qx][qy] * wx * wx * wy * wy * wDz * wDz;

                              // -(u_1)_{x_2} O13 (u_1)_{x_0} - (u_1)_{x_0} O31 (u_1)_{x_2}
                              dxyz += -(sop[i13][qx][qy] + sop[i31][qx][qy]) * wDx * wx * wy * wy * wDz * wz;

                              // (u_1)_{x_0} O33 (u_1)_{x_0})
                              dxyz += sop[i33][qx][qy] * wDx * wDx * wy * wy * wz * wz;
                           }
                           else
                           {
                              // (u_2)_{x_1} (O11 (u_2)_{x_1} - O12 (u_2)_{x_0}) - (u_2)_{x_0} (O21 (u_2)_{x_1} - O22 (u_2)_{x_0})

                              // (u_2)_{x_1} O11 (u_2)_{x_1}
                              dxyz += sop[i11][qx][qy] * wx * wx * wDy * wDy * wz * wz;

                              // -(u_2)_{x_1} O12 (u_2)_{x_0} - (u_2)_{x_0} O21 (u_2)_{x_1}
                              dxyz += -(sop[i12][qx][qy] + sop[i21][qx][qy]) * wDx * wx * wDy * wy * wz * wz;

                              // (u_2)_{x_0} O22 (u_2)_{x_0}
                              dxyz += sop[i22][qx][qy] * wDx * wDx * wy * wy * wz * wz;
                           }
                        }
                     }
                  }
               }
            }

            MFEM_SYNC_THREAD;
         }  // qz loop

         MFEM_FOREACH_THREAD(dz,z,D1Dz)
         {
            MFEM_FOREACH_THREAD(dy,y,D1Dy)
            {
               MFEM_FOREACH_THREAD(dx,x,D1Dx)
               {
                  D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += dxyz;
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // c loop
   }); // end of element loop
}

// PA H(curl) curl-curl Apply/AbsApply 2D kernel
void PACurlCurlApply2D(const int D1D,
                       const int Q1D,
                       const bool symmetric, // unused
                       const int NE,
                       const Array<real_t> &bo,
                       const Array<real_t> &bc, // unused
                       const Array<real_t> &bot,
                       const Array<real_t> &bct, // unused
                       const Array<real_t> &gc,
                       const Array<real_t> &gct,
                       const Vector &pa_data,
                       const Vector &x,
                       Vector &y,
                       const bool useAbs = false);

// PA H(curl) curl-curl Apply/AbsApply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PACurlCurlApply3D(const int d1d,
                              const int q1d,
                              const bool symmetric,
                              const int NE,
                              const Array<real_t> &bo,
                              const Array<real_t> &bc,
                              const Array<real_t> &bot,
                              const Array<real_t> &bct,
                              const Array<real_t> &gc,
                              const Array<real_t> &gct,
                              const Vector &pa_data,
                              const Vector &x,
                              Vector &y,
                              const bool useAbs = false)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk),
      // we get:
      // (\nabla\times u) \cdot (\nabla\times v)
      //     = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      real_t curl[MQ1D][MQ1D][MQ1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MQ1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy);
                  const real_t wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz);
               const real_t wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     if (useAbs)
                     {
                        // +(u_0)_{x_1}
                        curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;
                     }
                     else
                     {
                        // -(u_0)_{x_1}
                        curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;
                     }
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               real_t massY[MQ1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = Bc(qx,dx);
                  const real_t wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz);
               const real_t wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     if (useAbs)
                     {
                        // +(u_1)_{x_2}
                        curl[qz][qy][qx][0] += gradXY[qy][qx][1] * wDz;
                     }
                     else
                     {
                        // -(u_1)_{x_2}
                        curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz;
                     }
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            real_t gradYZ[MQ1D][MQ1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massZ[MQ1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy);
                  const real_t wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t wx = Bc(qx,dx);
               const real_t wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     if (useAbs)
                     {
                        // +(u_2)_{x_0}
                        curl[qz][qy][qx][1] += gradYZ[qz][qy][0] * wDx;
                     }
                     else
                     {
                        // -(u_2)_{x_0}
                        curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx;
                     }
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const real_t O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const real_t O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const real_t O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const real_t O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const real_t O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);

               const real_t c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const real_t c2 = (O21 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const real_t c3 = (O31 * curl[qz][qy][qx][0]) + (O32 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY12[MD1D][MD1D];
            real_t gradXY21[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MD1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const real_t wx = Bot(dx,qx);

                     massX[dx][0] += wx * curl[qz][qy][qx][1];
                     massX[dx][1] += wx * curl[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy);
                  const real_t wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz);
               const real_t wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     const int idx = dx + ((dy + (dz * D1Dy)) * D1Dx) + osc;
                     if (useAbs)
                     {
                        // (u_0)_{x_2} * (op * curl)_1 +
                        // (u_0)_{x_1} * (op * curl)_2
                        Y(idx, e) += (gradXY21[dy][dx] * wDz) +
                                     (gradXY12[dy][dx] * wz);
                     }
                     else
                     {
                        // (u_0)_{x_2} * (op * curl)_1 -
                        // (u_0)_{x_1} * (op * curl)_2
                        Y(idx, e) += (gradXY21[dy][dx] * wDz) -
                                     (gradXY12[dy][dx] * wz);
                     }
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY02[MD1D][MD1D];
            real_t gradXY20[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               real_t massY[MD1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const real_t wy = Bot(dy,qy);

                     massY[dy][0] += wy * curl[qz][qy][qx][2];
                     massY[dy][1] += wy * curl[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t wx = Bct(dx,qx);
                  const real_t wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz);
               const real_t wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const int idx = dx + ((dy + (dz * D1Dy)) * D1Dx) + osc;
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     if (useAbs)
                     {
                        // +(u_1)_{x_2} * (op * curl)_0 +
                        //  (u_1)_{x_0} * (op * curl)_2
                        Y(idx, e) += (gradXY20[dy][dx] * wDz) +
                                     (gradXY02[dy][dx] * wz);
                     }
                     else
                     {
                        // -(u_1)_{x_2} * (op * curl)_0 +
                        //  (u_1)_{x_0} * (op * curl)_2
                        Y(idx, e) += (-gradXY20[dy][dx] * wDz) +
                                     (gradXY02[dy][dx] * wz);
                     }
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            real_t gradYZ01[MD1D][MD1D];
            real_t gradYZ10[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massZ[MD1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const real_t wz = Bot(dz,qz);

                     massZ[dz][0] += wz * curl[qz][qy][qx][0];
                     massZ[dz][1] += wz * curl[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy);
                  const real_t wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t wx = Bct(dx,qx);
               const real_t wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const int idx = dx + ((dy + (dz * D1Dy)) * D1Dx) + osc;
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     if (useAbs)
                     {
                        // (u_2)_{x_1} * (op * curl)_0 +
                        // (u_2)_{x_0} * (op * curl)_1
                        Y(idx, e) += (gradYZ10[dz][dy] * wx) +
                                     (gradYZ01[dz][dy] * wDx);
                     }
                     else
                     {
                        // (u_2)_{x_1} * (op * curl)_0 -
                        // (u_2)_{x_0} * (op * curl)_1
                        Y(idx, e) += (gradYZ10[dz][dy] * wx) -
                                     (gradYZ01[dz][dy] * wDx);
                     }
                  }
               }
            }
         }  // loop qx
      }
   }); // end of element loop
}

// Shared memory PA H(curl) curl-curl Apply/AbsApply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPACurlCurlApply3D(const int d1d,
                                  const int q1d,
                                  const bool symmetric,
                                  const int NE,
                                  const Array<real_t> &bo,
                                  const Array<real_t> &bc,
                                  const Array<real_t> &bot,
                                  const Array<real_t> &bct,
                                  const Array<real_t> &gc,
                                  const Array<real_t> &gct,
                                  const Vector &pa_data,
                                  const Vector &x,
                                  Vector &y,
                                  const bool useAbs = false)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
   // (\nabla\times u) \cdot (\nabla\times v) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   const int s = symmetric ? 6 : 9;

   auto device_kernel = [=] MFEM_DEVICE (int e)
   {
      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t sBo[MD1D][MQ1D];
      MFEM_SHARED real_t sBc[MD1D][MQ1D];
      MFEM_SHARED real_t sGc[MD1D][MQ1D];

      real_t ope[9];
      MFEM_SHARED real_t sop[9][MQ1D][MQ1D];
      MFEM_SHARED real_t curl[MQ1D][MQ1D][3];

      MFEM_SHARED real_t sX[MD1D][MD1D][MD1D];

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               for (int i=0; i<s; ++i)
               {
                  ope[i] = op(qx,qy,qz,i,e);
               }
            }
         }
      }

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               sBc[d][q] = Bc(q,d);
               sGc[d][q] = Gc(q,d);
               if (d < D1D-1)
               {
                  sBo[d][q] = Bo(q,d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int qz=0; qz < Q1D; ++qz)
      {
         if (tidz == qz)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  for (int i=0; i<3; ++i)
                  {
                     curl[qy][qx][i] = 0.0;
                  }
               }
            }
         }

         int osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     sX[dz][dy][dx] = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  }
               }
            }
            MFEM_SYNC_THREAD;

            if (tidz == qz)
            {
               if (c == 0)
               {
                  for (int i=0; i<s; ++i)
                  {
                     sop[i][tidx][tidy] = ope[i];
                  }
               }

               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     real_t u = 0.0;
                     real_t v = 0.0;

                     // We treat x, y, z components separately for optimization specific to each.
                     if (c == 0) // x component
                     {
                        // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const real_t wz = sBc[dz][qz];
                           const real_t wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const real_t wy = sBc[dy][qy];
                              const real_t wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const real_t wx = sX[dz][dy][dx] * sBo[dx][qx];
                                 u += wx * wDy * wz;
                                 v += wx * wy * wDz;
                              }
                           }
                        }

                        curl[qy][qx][1] += v; // (u_0)_{x_2}
                        if (useAbs) { curl[qy][qx][2] += u; } // +(u_0)_{x_1}
                        else { curl[qy][qx][2] -= u; } // -(u_0)_{x_1}
                     }
                     else if (c == 1)  // y component
                     {
                        // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const real_t wz = sBc[dz][qz];
                           const real_t wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const real_t wy = sBo[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const real_t t = sX[dz][dy][dx];
                                 const real_t wx = t * sBc[dx][qx];
                                 const real_t wDx = t * sGc[dx][qx];

                                 u += wDx * wy * wz;
                                 v += wx * wy * wDz;
                              }
                           }
                        }

                        if (useAbs) { curl[qy][qx][0] += v; } // +(u_1)_{x_2}
                        else { curl[qy][qx][0] -= v; } // -(u_1)_{x_2}
                        curl[qy][qx][2] += u; // (u_1)_{x_0}
                     }
                     else // z component
                     {
                        // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const real_t wz = sBo[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const real_t wy = sBc[dy][qy];
                              const real_t wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const real_t t = sX[dz][dy][dx];
                                 const real_t wx = t * sBc[dx][qx];
                                 const real_t wDx = t * sGc[dx][qx];

                                 u += wDx * wy * wz;
                                 v += wx * wDy * wz;
                              }
                           }
                        }

                        curl[qy][qx][0] += v; // (u_2)_{x_1}
                        if (useAbs) { curl[qy][qx][1] += u; }// +(u_2)_{x_0}
                        else { curl[qy][qx][1] -= u; } // -(u_2)_{x_0}
                     }
                  } // qx
               } // qy
            } // tidz == qz

            osc += D1Dx * D1Dy * D1Dz;
            MFEM_SYNC_THREAD;
         } // c

         real_t dxyz1 = 0.0;
         real_t dxyz2 = 0.0;
         real_t dxyz3 = 0.0;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            const real_t wcz = sBc[dz][qz];
            const real_t wcDz = sGc[dz][qz];
            const real_t wz = (dz < D1D-1) ? sBo[dz][qz] : 0.0;

            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wcy = sBc[dy][qy];
                     const real_t wcDy = sGc[dy][qy];
                     const real_t wy = (dy < D1D-1) ? sBo[dy][qy] : 0.0;

                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t O11 = sop[0][qx][qy];
                        const real_t O12 = sop[1][qx][qy];
                        const real_t O13 = sop[2][qx][qy];
                        const real_t O21 = symmetric ? O12 : sop[3][qx][qy];
                        const real_t O22 = symmetric ? sop[3][qx][qy] : sop[4][qx][qy];
                        const real_t O23 = symmetric ? sop[4][qx][qy] : sop[5][qx][qy];
                        const real_t O31 = symmetric ? O13 : sop[6][qx][qy];
                        const real_t O32 = symmetric ? O23 : sop[7][qx][qy];
                        const real_t O33 = symmetric ? sop[5][qx][qy] : sop[8][qx][qy];

                        const real_t c1 = (O11 * curl[qy][qx][0]) + (O12 * curl[qy][qx][1]) +
                                          (O13 * curl[qy][qx][2]);
                        const real_t c2 = (O21 * curl[qy][qx][0]) + (O22 * curl[qy][qx][1]) +
                                          (O23 * curl[qy][qx][2]);
                        const real_t c3 = (O31 * curl[qy][qx][0]) + (O32 * curl[qy][qx][1]) +
                                          (O33 * curl[qy][qx][2]);

                        const real_t wcx = sBc[dx][qx];
                        const real_t wDx = sGc[dx][qx];

                        if (dx < D1D-1)
                        {
                           // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                           const real_t wx = sBo[dx][qx];
                           if (useAbs)
                           {
                              // (u_0)_{x_2} * (op * curl)_1 +
                              // (u_0)_{x_1} * (op * curl)_2
                              dxyz1 += (wx * c2 * wcy * wcDz) +
                                       (wx * c3 * wcDy * wcz);
                           }
                           else
                           {
                              // (u_0)_{x_2} * (op * curl)_1 -
                              // (u_0)_{x_1} * (op * curl)_2
                              dxyz1 += (wx * c2 * wcy * wcDz) -
                                       (wx * c3 * wcDy * wcz);
                           }
                        }

                        // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                        if (useAbs)
                        {
                           // +(u_1)_{x_2} * (op * curl)_0 +
                           //  (u_1)_{x_0} * (op * curl)_2
                           dxyz2 += (wy * c1 * wcx * wcDz) +
                                    (wy * c3 * wDx * wcz);
                        }
                        else
                        {
                           // -(u_1)_{x_2} * (op * curl)_0 +
                           //  (u_1)_{x_0} * (op * curl)_2
                           dxyz2 += (-wy * c1 * wcx * wcDz) +
                                    (wy * c3 * wDx * wcz);
                        }

                        // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                        if (useAbs)
                        {
                           // (u_2)_{x_1} * (op * curl)_0 +
                           // (u_2)_{x_0} * (op * curl)_1
                           dxyz3 += (wcDy * wz * c1 * wcx) +
                                    (wcy * wz * c2 * wDx);
                        }
                        else
                        {
                           // (u_2)_{x_1} * (op * curl)_0 -
                           // (u_2)_{x_0} * (op * curl)_1
                           dxyz3 += (wcDy * wz * c1 * wcx) -
                                    (wcy * wz * c2 * wDx);
                        }
                     } // qx
                  } // qy
               } // dx
            } // dy
         } // dz

         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  if (dx < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * (D1D-1)), e) += dxyz1;
                  }
                  if (dy < D1D-1)
                  {
                     Y(dx + ((dy + (dz * (D1D-1))) * D1D) + ((D1D-1)*D1D*D1D), e) += dxyz2;
                  }
                  if (dz < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * D1D) + (2*(D1D-1)*D1D*D1D), e) += dxyz3;
                  }
               }
            }
         }
      } // qz
   }; // end of element loop

   auto host_kernel = [&] MFEM_LAMBDA (int)
   {
      MFEM_ABORT_KERNEL("This kernel should only be used on GPU.");
   };

   ForallWrap<3>(true, NE, device_kernel, host_kernel, Q1D, Q1D, Q1D);
}

// PA H(curl)-L2 Assemble 2D kernel
void PAHcurlL2Setup2D(const int Q1D,
                      const int NE,
                      const Array<real_t> &w,
                      Vector &coeff,
                      Vector &op);

// PA H(curl)-L2 Assemble 3D kernel
void PAHcurlL2Setup3D(const int NQ,
                      const int coeffDim,
                      const int NE,
                      const Array<real_t> &w,
                      Vector &coeff,
                      Vector &op);

// PA H(curl)-L2 Apply 2D kernel
void PAHcurlL2Apply2D(const int D1D,
                      const int D1Dtest,
                      const int Q1D,
                      const int NE,
                      const Array<real_t> &bo,
                      const Array<real_t> &bot,
                      const Array<real_t> &bt,
                      const Array<real_t> &gc,
                      const Vector &pa_data,
                      const Vector &x,
                      Vector &y);

// PA H(curl)-L2 Apply Transpose 2D kernel
void PAHcurlL2ApplyTranspose2D(const int D1D,
                               const int D1Dtest,
                               const int Q1D,
                               const int NE,
                               const Array<real_t> &bo,
                               const Array<real_t> &bot,
                               const Array<real_t> &b,
                               const Array<real_t> &gct,
                               const Vector &pa_data,
                               const Vector &x,
                               Vector &y);

// PA H(curl)-L2 Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAHcurlL2Apply3D(const int d1d,
                             const int q1d,
                             const int coeffDim,
                             const int NE,
                             const Array<real_t> &bo,
                             const Array<real_t> &bc,
                             const Array<real_t> &bot,
                             const Array<real_t> &bct,
                             const Array<real_t> &gc,
                             const Vector &pa_data,
                             const Vector &x,
                             Vector &y)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using u = dF^{-T} \hat{u} and (\nabla\times u) F =
      // 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get:
      // (\nabla\times u) \cdot v
      //    = 1/det(dF) \hat{\nabla}\times\hat{u}^T dF^T dF^{-T} \hat{v}
      //    = 1/det(dF) \hat{\nabla}\times\hat{u}^T \hat{v}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      real_t curl[MQ1D][MQ1D][MQ1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MQ1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy);
                  const real_t wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz);
               const real_t wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t gradXY[MQ1D][MQ1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               real_t massY[MQ1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = Bc(qx,dx);
                  const real_t wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = Bc(qz,dz);
               const real_t wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            real_t gradYZ[MQ1D][MQ1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massZ[MQ1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = Bc(qy,dy);
                  const real_t wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const real_t wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t wx = Bc(qx,dx);
               const real_t wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(0,qx,qy,qz,e);
               if (coeffDim == 1)
               {
                  for (int c = 0; c < VDIM; ++c)
                  {
                     curl[qz][qy][qx][c] *= O11;
                  }
               }
               else
               {
                  const real_t O21 = op(1,qx,qy,qz,e);
                  const real_t O31 = op(2,qx,qy,qz,e);
                  const real_t O12 = op(3,qx,qy,qz,e);
                  const real_t O22 = op(4,qx,qy,qz,e);
                  const real_t O32 = op(5,qx,qy,qz,e);
                  const real_t O13 = op(6,qx,qy,qz,e);
                  const real_t O23 = op(7,qx,qy,qz,e);
                  const real_t O33 = op(8,qx,qy,qz,e);
                  const real_t curlX = curl[qz][qy][qx][0];
                  const real_t curlY = curl[qz][qy][qx][1];
                  const real_t curlZ = curl[qz][qy][qx][2];
                  curl[qz][qy][qx][0] = (O11*curlX)+(O12*curlY)+(O13*curlZ);
                  curl[qz][qy][qx][1] = (O21*curlX)+(O22*curlY)+(O23*curlZ);
                  curl[qz][qy][qx][2] = (O31*curlX)+(O32*curlY)+(O33*curlZ);
               }
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[MD1D][MD1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MD1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0.0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += curl[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Shared memory PA H(curl)-L2 Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAHcurlL2Apply3D(const int d1d,
                                 const int q1d,
                                 const int coeffDim,
                                 const int NE,
                                 const Array<real_t> &bo,
                                 const Array<real_t> &bc,
                                 const Array<real_t> &gc,
                                 const Vector &pa_data,
                                 const Vector &x,
                                 Vector &y)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   auto device_kernel = [=] MFEM_DEVICE (int e)
   {
      constexpr int VDIM = 3;
      constexpr int maxCoeffDim = 9;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t sBo[MD1D][MQ1D];
      MFEM_SHARED real_t sBc[MD1D][MQ1D];
      MFEM_SHARED real_t sGc[MD1D][MQ1D];

      real_t opc[maxCoeffDim];
      MFEM_SHARED real_t sop[maxCoeffDim][MQ1D][MQ1D];
      MFEM_SHARED real_t curl[MQ1D][MQ1D][3];

      MFEM_SHARED real_t sX[MD1D][MD1D][MD1D];

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               for (int i=0; i<coeffDim; ++i)
               {
                  opc[i] = op(i,qx,qy,qz,e);
               }
            }
         }
      }

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               sBc[d][q] = Bc(q,d);
               sGc[d][q] = Gc(q,d);
               if (d < D1D-1)
               {
                  sBo[d][q] = Bo(q,d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int qz=0; qz < Q1D; ++qz)
      {
         if (tidz == qz)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  for (int i=0; i<3; ++i)
                  {
                     curl[qy][qx][i] = 0.0;
                  }
               }
            }
         }

         int osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     sX[dz][dy][dx] = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  }
               }
            }
            MFEM_SYNC_THREAD;

            if (tidz == qz)
            {
               if (c == 0)
               {
                  for (int i=0; i<coeffDim; ++i)
                  {
                     sop[i][tidx][tidy] = opc[i];
                  }
               }

               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     real_t u = 0.0;
                     real_t v = 0.0;

                     // We treat x, y, z components separately for optimization specific to each.
                     if (c == 0) // x component
                     {
                        // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const real_t wz = sBc[dz][qz];
                           const real_t wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const real_t wy = sBc[dy][qy];
                              const real_t wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const real_t wx = sX[dz][dy][dx] * sBo[dx][qx];
                                 u += wx * wDy * wz;
                                 v += wx * wy * wDz;
                              }
                           }
                        }

                        curl[qy][qx][1] += v; // (u_0)_{x_2}
                        curl[qy][qx][2] -= u;  // -(u_0)_{x_1}
                     }
                     else if (c == 1)  // y component
                     {
                        // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const real_t wz = sBc[dz][qz];
                           const real_t wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const real_t wy = sBo[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const real_t t = sX[dz][dy][dx];
                                 const real_t wx = t * sBc[dx][qx];
                                 const real_t wDx = t * sGc[dx][qx];

                                 u += wDx * wy * wz;
                                 v += wx * wy * wDz;
                              }
                           }
                        }

                        curl[qy][qx][0] -= v; // -(u_1)_{x_2}
                        curl[qy][qx][2] += u; // (u_1)_{x_0}
                     }
                     else // z component
                     {
                        // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const real_t wz = sBo[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const real_t wy = sBc[dy][qy];
                              const real_t wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const real_t t = sX[dz][dy][dx];
                                 const real_t wx = t * sBc[dx][qx];
                                 const real_t wDx = t * sGc[dx][qx];

                                 u += wDx * wy * wz;
                                 v += wx * wDy * wz;
                              }
                           }
                        }

                        curl[qy][qx][0] += v; // (u_2)_{x_1}
                        curl[qy][qx][1] -= u; // -(u_2)_{x_0}
                     }
                  } // qx
               } // qy
            } // tidz == qz

            osc += D1Dx * D1Dy * D1Dz;
            MFEM_SYNC_THREAD;
         } // c

         real_t dxyz1 = 0.0;
         real_t dxyz2 = 0.0;
         real_t dxyz3 = 0.0;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            const real_t wcz = sBc[dz][qz];
            const real_t wz = (dz < D1D-1) ? sBo[dz][qz] : 0.0;

            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wcy = sBc[dy][qy];
                     const real_t wy = (dy < D1D-1) ? sBo[dy][qy] : 0.0;

                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t O11 = sop[0][qx][qy];
                        real_t c1, c2, c3;
                        if (coeffDim == 1)
                        {
                           c1 = O11 * curl[qy][qx][0];
                           c2 = O11 * curl[qy][qx][1];
                           c3 = O11 * curl[qy][qx][2];
                        }
                        else
                        {
                           const real_t O21 = sop[1][qx][qy];
                           const real_t O31 = sop[2][qx][qy];
                           const real_t O12 = sop[3][qx][qy];
                           const real_t O22 = sop[4][qx][qy];
                           const real_t O32 = sop[5][qx][qy];
                           const real_t O13 = sop[6][qx][qy];
                           const real_t O23 = sop[7][qx][qy];
                           const real_t O33 = sop[8][qx][qy];
                           c1 = (O11*curl[qy][qx][0])+(O12*curl[qy][qx][1])+(O13*curl[qy][qx][2]);
                           c2 = (O21*curl[qy][qx][0])+(O22*curl[qy][qx][1])+(O23*curl[qy][qx][2]);
                           c3 = (O31*curl[qy][qx][0])+(O32*curl[qy][qx][1])+(O33*curl[qy][qx][2]);
                        }

                        const real_t wcx = sBc[dx][qx];

                        if (dx < D1D-1)
                        {
                           const real_t wx = sBo[dx][qx];
                           dxyz1 += c1 * wx * wcy * wcz;
                        }

                        dxyz2 += c2 * wcx * wy * wcz;
                        dxyz3 += c3 * wcx * wcy * wz;
                     } // qx
                  } // qy
               } // dx
            } // dy
         } // dz

         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  if (dx < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * (D1D-1)), e) += dxyz1;
                  }
                  if (dy < D1D-1)
                  {
                     Y(dx + ((dy + (dz * (D1D-1))) * D1D) + ((D1D-1)*D1D*D1D), e) += dxyz2;
                  }
                  if (dz < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * D1D) + (2*(D1D-1)*D1D*D1D), e) += dxyz3;
                  }
               }
            }
         }
      } // qz
   }; // end of element loop

   auto host_kernel = [&] MFEM_LAMBDA (int)
   {
      MFEM_ABORT_KERNEL("This kernel should only be used on GPU.");
   };

   ForallWrap<3>(true, NE, device_kernel, host_kernel, Q1D, Q1D, Q1D);
}

// PA H(curl)-L2 Apply Transpose 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAHcurlL2ApplyTranspose3D(const int d1d,
                                      const int q1d,
                                      const int coeffDim,
                                      const int NE,
                                      const Array<real_t> &bo,
                                      const Array<real_t> &bc,
                                      const Array<real_t> &bot,
                                      const Array<real_t> &bct,
                                      const Array<real_t> &gct,
                                      const Vector &pa_data,
                                      const Vector &x,
                                      Vector &y)
{
   // See PAHcurlL2Apply3D for comments.
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      real_t mass[MQ1D][MQ1D][MQ1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t massXY[MQ1D][MQ1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MQ1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t O11 = op(0,qx,qy,qz,e);
               if (coeffDim == 1)
               {
                  for (int c = 0; c < VDIM; ++c)
                  {
                     mass[qz][qy][qx][c] *= O11;
                  }
               }
               else
               {
                  const real_t O12 = op(1,qx,qy,qz,e);
                  const real_t O13 = op(2,qx,qy,qz,e);
                  const real_t O21 = op(3,qx,qy,qz,e);
                  const real_t O22 = op(4,qx,qy,qz,e);
                  const real_t O23 = op(5,qx,qy,qz,e);
                  const real_t O31 = op(6,qx,qy,qz,e);
                  const real_t O32 = op(7,qx,qy,qz,e);
                  const real_t O33 = op(8,qx,qy,qz,e);
                  const real_t massX = mass[qz][qy][qx][0];
                  const real_t massY = mass[qz][qy][qx][1];
                  const real_t massZ = mass[qz][qy][qx][2];
                  mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
                  mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
                  mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
               }
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY12[MD1D][MD1D];
            real_t gradXY21[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MD1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const real_t wx = Bot(dx,qx);

                     massX[dx][0] += wx * mass[qz][qy][qx][1];
                     massX[dx][1] += wx * mass[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy);
                  const real_t wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz);
               const real_t wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradXY21[dy][dx] * wDz) - (gradXY12[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            real_t gradXY02[MD1D][MD1D];
            real_t gradXY20[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               real_t massY[MD1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const real_t wy = Bot(dy,qy);

                     massY[dy][0] += wy * mass[qz][qy][qx][2];
                     massY[dy][1] += wy * mass[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t wx = Bct(dx,qx);
                  const real_t wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = Bct(dz,qz);
               const real_t wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (-gradXY20[dy][dx] * wDz) + (gradXY02[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            real_t gradYZ01[MD1D][MD1D];
            real_t gradYZ10[MD1D][MD1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massZ[MD1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const real_t wz = Bot(dz,qz);

                     massZ[dz][0] += wz * mass[qz][qy][qx][0];
                     massZ[dz][1] += wz * mass[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = Bct(dy,qy);
                  const real_t wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t wx = Bct(dx,qx);
               const real_t wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradYZ10[dz][dy] * wx) - (gradYZ01[dz][dy] * wDx);
                  }
               }
            }
         }  // loop qx
      }
   });
}

// PA H(curl)-L2 Apply Transpose 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void SmemPAHcurlL2ApplyTranspose3D(const int d1d,
                                          const int q1d,
                                          const int coeffDim,
                                          const int NE,
                                          const Array<real_t> &bo,
                                          const Array<real_t> &bc,
                                          const Array<real_t> &gc,
                                          const Vector &pa_data,
                                          const Vector &x,
                                          Vector &y)
{
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   auto device_kernel = [=] MFEM_DEVICE (int e)
   {
      constexpr int VDIM = 3;
      constexpr int maxCoeffDim = 9;
      constexpr int MD1D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED real_t sBo[MD1D][MQ1D];
      MFEM_SHARED real_t sBc[MD1D][MQ1D];
      MFEM_SHARED real_t sGc[MD1D][MQ1D];

      real_t opc[maxCoeffDim];
      MFEM_SHARED real_t sop[maxCoeffDim][MQ1D][MQ1D];
      MFEM_SHARED real_t mass[MQ1D][MQ1D][3];

      MFEM_SHARED real_t sX[MD1D][MD1D][MD1D];

      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               for (int i=0; i<coeffDim; ++i)
               {
                  opc[i] = op(i,qx,qy,qz,e);
               }
            }
         }
      }

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               sBc[d][q] = Bc(q,d);
               sGc[d][q] = Gc(q,d);
               if (d < D1D-1)
               {
                  sBo[d][q] = Bo(q,d);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int qz=0; qz < Q1D; ++qz)
      {
         if (tidz == qz)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  for (int i=0; i<3; ++i)
                  {
                     mass[qy][qx][i] = 0.0;
                  }
               }
            }
         }

         int osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     sX[dz][dy][dx] = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  }
               }
            }
            MFEM_SYNC_THREAD;

            if (tidz == qz)
            {
               if (c == 0)
               {
                  for (int i=0; i<coeffDim; ++i)
                  {
                     sop[i][tidx][tidy] = opc[i];
                  }
               }

               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     real_t u = 0.0;

                     for (int dz = 0; dz < D1Dz; ++dz)
                     {
                        const real_t wz = (c == 2) ? sBo[dz][qz] : sBc[dz][qz];

                        for (int dy = 0; dy < D1Dy; ++dy)
                        {
                           const real_t wy = (c == 1) ? sBo[dy][qy] : sBc[dy][qy];

                           for (int dx = 0; dx < D1Dx; ++dx)
                           {
                              const real_t wx = sX[dz][dy][dx] * ((c == 0) ? sBo[dx][qx] : sBc[dx][qx]);
                              u += wx * wy * wz;
                           }
                        }
                     }

                     mass[qy][qx][c] += u;
                  } // qx
               } // qy
            } // tidz == qz

            osc += D1Dx * D1Dy * D1Dz;
            MFEM_SYNC_THREAD;
         } // c

         real_t dxyz1 = 0.0;
         real_t dxyz2 = 0.0;
         real_t dxyz3 = 0.0;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            const real_t wcz = sBc[dz][qz];
            const real_t wcDz = sGc[dz][qz];
            const real_t wz = (dz < D1D-1) ? sBo[dz][qz] : 0.0;

            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wcy = sBc[dy][qy];
                     const real_t wcDy = sGc[dy][qy];
                     const real_t wy = (dy < D1D-1) ? sBo[dy][qy] : 0.0;

                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t O11 = sop[0][qx][qy];
                        real_t c1, c2, c3;
                        if (coeffDim == 1)
                        {
                           c1 = O11 * mass[qy][qx][0];
                           c2 = O11 * mass[qy][qx][1];
                           c3 = O11 * mass[qy][qx][2];
                        }
                        else
                        {
                           const real_t O12 = sop[1][qx][qy];
                           const real_t O13 = sop[2][qx][qy];
                           const real_t O21 = sop[3][qx][qy];
                           const real_t O22 = sop[4][qx][qy];
                           const real_t O23 = sop[5][qx][qy];
                           const real_t O31 = sop[6][qx][qy];
                           const real_t O32 = sop[7][qx][qy];
                           const real_t O33 = sop[8][qx][qy];

                           c1 = (O11*mass[qy][qx][0])+(O12*mass[qy][qx][1])+(O13*mass[qy][qx][2]);
                           c2 = (O21*mass[qy][qx][0])+(O22*mass[qy][qx][1])+(O23*mass[qy][qx][2]);
                           c3 = (O31*mass[qy][qx][0])+(O32*mass[qy][qx][1])+(O33*mass[qy][qx][2]);
                        }

                        const real_t wcx = sBc[dx][qx];
                        const real_t wDx = sGc[dx][qx];

                        if (dx < D1D-1)
                        {
                           const real_t wx = sBo[dx][qx];
                           dxyz1 += (wx * c2 * wcy * wcDz) - (wx * c3 * wcDy * wcz);
                        }

                        dxyz2 += (-wy * c1 * wcx * wcDz) + (wy * c3 * wDx * wcz);

                        dxyz3 += (wcDy * wz * c1 * wcx) - (wcy * wz * c2 * wDx);
                     } // qx
                  } // qy
               } // dx
            } // dy
         } // dz

         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  if (dx < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * (D1D-1)), e) += dxyz1;
                  }
                  if (dy < D1D-1)
                  {
                     Y(dx + ((dy + (dz * (D1D-1))) * D1D) + ((D1D-1)*D1D*D1D), e) += dxyz2;
                  }
                  if (dz < D1D-1)
                  {
                     Y(dx + ((dy + (dz * D1D)) * D1D) + (2*(D1D-1)*D1D*D1D), e) += dxyz3;
                  }
               }
            }
         }
      } // qz
   }; // end of element loop

   auto host_kernel = [&] MFEM_LAMBDA (int)
   {
      MFEM_ABORT_KERNEL("This kernel should only be used on GPU.");
   };

   ForallWrap<3>(true, NE, device_kernel, host_kernel, Q1D, Q1D, Q1D);
}

} // namespace internal

template<int DIM, int T_D1D, int T_Q1D>
CurlCurlIntegrator::ApplyKernelType CurlCurlIntegrator::ApplyPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::PACurlCurlApply2D;
   }
   else if constexpr (DIM == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         return internal::SmemPACurlCurlApply3D<T_D1D, T_Q1D>;
      }
      else
      {
         return internal::PACurlCurlApply3D;
      }
   }
   MFEM_ABORT("");
}

template <int DIM, int T_D1D, int T_Q1D>
CurlCurlIntegrator::DiagonalKernelType
CurlCurlIntegrator::DiagonalPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::PACurlCurlAssembleDiagonal2D;
   }
   else if constexpr (DIM == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         return internal::SmemPACurlCurlAssembleDiagonal3D<T_D1D, T_Q1D>;
      }
      else
      {
         return internal::PACurlCurlAssembleDiagonal3D;
      }
   }
   MFEM_ABORT("");
}
/// \endcond DO_NOT_DOCUMENT
} // namespace mfem

#endif
