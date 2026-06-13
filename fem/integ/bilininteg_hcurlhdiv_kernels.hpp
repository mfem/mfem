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

#ifndef MFEM_BILININTEG_HCURLHDIV_KERNELS_HPP
#define MFEM_BILININTEG_HCURLHDIV_KERNELS_HPP

#include "../../config/config.hpp"
#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/vector.hpp"
#include "../bilininteg.hpp"

/// \cond DO_NOT_DOCUMENT
namespace mfem
{

namespace internal
{

// PA H(curl)-H(div) Mass Apply 2D kernel
void PAHcurlHdivMassSetup2D(const int Q1D,
                            const int coeffDim,
                            const int NE,
                            const bool transpose,
                            const Array<real_t> &w_,
                            const Vector &j,
                            Vector &coeff_,
                            Vector &op);

// PA H(curl)-H(div) Mass Assemble 3D kernel
void PAHcurlHdivMassSetup3D(const int Q1D,
                            const int coeffDim,
                            const int NE,
                            const bool transpose,
                            const Array<real_t> &w_,
                            const Vector &j,
                            Vector &coeff_,
                            Vector &op);

// PA H(curl)-H(div) Mass Apply 2D kernel
void PAHcurlHdivMassApply2D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const bool transpose,
                            const Array<real_t> &Bo_,
                            const Array<real_t> &Bc_,
                            const Array<real_t> &Bot_,
                            const Array<real_t> &Bct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_);

/// H(curl) test, H(div) trial
inline void
PAHcurlHdivMassApply2D(const int NE, const bool, const bool scalarCoeff,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int D1Dtest, const int Q1D)
{
   return PAHcurlHdivMassApply2D(D1D, D1Dtest, Q1D, NE, scalarCoeff, false,
                                 false, Bo_, Bc_, Bot_, Bct_, op_, x_, y_);
}

/// H(div) test, H(curl) trial
inline void
PAHdivHcurlMassApply2D(const int NE, const bool, const bool scalarCoeff,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int D1Dtest, const int Q1D)
{
   return PAHcurlHdivMassApply2D(D1D, D1Dtest, Q1D, NE, scalarCoeff, true,
                                 false, Bo_, Bc_, Bot_, Bct_, op_, x_, y_);
}

// PA H(curl)-H(div) Mass Apply 3D kernel
void PAHcurlHdivMassApply3D(const int D1D,
                            const int D1Dtest,
                            const int Q1D,
                            const int NE,
                            const bool scalarCoeff,
                            const bool trialHcurl,
                            const bool transpose,
                            const Array<real_t> &Bo_,
                            const Array<real_t> &Bc_,
                            const Array<real_t> &Bot_,
                            const Array<real_t> &Bct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_);

/// H(curl) test, H(div) trial
inline void
PAHcurlHdivMassApply3D(const int NE, const bool, const bool scalarCoeff,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int D1Dtest, const int Q1D)
{
   PAHcurlHdivMassApply3D(D1D, D1Dtest, Q1D, NE, scalarCoeff, false, false, Bo_,
                          Bc_, Bot_, Bct_, op_, x_, y_);
}

/// H(div) test, H(curl) trial
inline void
PAHdivHcurlMassApply3D(const int NE, const bool, const bool scalarCoeff,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int D1Dtest, const int Q1D)
{
   PAHcurlHdivMassApply3D(D1D, D1Dtest, Q1D, NE, scalarCoeff, true, false, Bo_,
                          Bc_, Bot_, Bct_, op_, x_, y_);
}

// PA H(curl)-H(div) Curl Apply 3D kernel
template<int T_D1D = 0, int T_D1D_TEST = 0, int T_Q1D = 0>
inline void PAHcurlHdivApply3D(const int d1d,
                               const int d1dtest,
                               const int q1d,
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
   MFEM_VERIFY(T_D1D_TEST || d1dtest <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1dtest > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int D1Dtest = T_D1D_TEST ? T_D1D_TEST : d1dtest;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1Dtest, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1Dtest-1)*(D1Dtest-1)*D1Dtest, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
      // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
      // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D :
                           DofQuadLimits::HCURL_MAX_D1D;  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int D1Dtest = T_D1D_TEST ? T_D1D_TEST : d1dtest;
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
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O22 = op(qx,qy,qz,3,e);
               const real_t O23 = op(qx,qy,qz,4,e);
               const real_t O33 = op(qx,qy,qz,5,e);

               const real_t c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const real_t c2 = (O12 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const real_t c3 = (O13 * curl[qz][qy][qx][0]) + (O23 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[MD1D][MD1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1Dtest : D1Dtest - 1;
            const int D1Dy = (c == 1) ? D1Dtest : D1Dtest - 1;
            const int D1Dx = (c == 0) ? D1Dtest : D1Dtest - 1;

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
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += curl[qz][qy][qx][c] *
                                  ((c == 0) ? Bct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Bct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// PA H(curl)-H(div) Curl Apply Transpose 3D kernel
template<int T_D1D = 0, int T_D1D_TEST = 0, int T_Q1D = 0>
inline void PAHcurlHdivApplyTranspose3D(const int d1d,
                                        const int d1dtest,
                                        const int q1d,
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
   MFEM_VERIFY(T_D1D || d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(T_D1D_TEST || d1dtest <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1dtest > HCURL_MAX_D1D");
   MFEM_VERIFY(T_Q1D || q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = T_D1D ? T_D1D : d1d;
   const int D1Dtest = T_D1D_TEST ? T_D1D_TEST : d1dtest;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1Dtest, Q1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), 3*(D1Dtest-1)*(D1Dtest-1)*D1Dtest, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
      // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
      // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      constexpr int VDIM = 3;
      constexpr int MD1D = T_D1D ? T_D1D :
                           DofQuadLimits::HCURL_MAX_D1D;  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D
      constexpr int MQ1D = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int D1Dtest = T_D1D_TEST ? T_D1D_TEST : d1dtest;
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
         const int D1Dz = (c == 2) ? D1Dtest : D1Dtest - 1;
         const int D1Dy = (c == 1) ? D1Dtest : D1Dtest - 1;
         const int D1Dx = (c == 0) ? D1Dtest : D1Dtest - 1;

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
                     massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
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
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O22 = op(qx,qy,qz,3,e);
               const real_t O23 = op(qx,qy,qz,4,e);
               const real_t O33 = op(qx,qy,qz,5,e);
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
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
   }); // end of element loop
}

namespace curlinterp
{
constexpr int NBZ3D(int ndof_o, int nquad_o, int mdq)
{
   if (ndof_o <= 0 || nquad_o <= 0)
   {
      return 1;
   }
   int ndof_c = ndof_o + 1;
   int nquad_c = nquad_o + 1;
   // z dimension is capped at 64 on nvidia and amd gpus
   int tmp = std::min((128 + mdq * mdq * mdq - 1) / (mdq * mdq * mdq), 64);
   int smem_req =
      sizeof(mfem::real_t) *
      ((3 * ndof_c * ndof_c * ndof_o + 2 * 2 * mdq * mdq * mdq) * tmp +
       ndof_c * nquad_o + ndof_c * nquad_c + ndof_o * nquad_o);
   // assume GPU has at least 48k shared memory
   return std::max(std::min(tmp, (48 * 1024 + smem_req - 1) / smem_req), 1);
}
}

template <int T_NDOF_O, int T_NQUAD_O>
void CurlInterpolatorApply3DSmem(const int ne, const int ndof_o,
                                 const int nquad_o, const Vector &pa,
                                 const Vector &x_, Vector &y_)
{
   constexpr int MND_O = T_NDOF_O ? T_NDOF_O : DofQuadLimits::HCURL_MAX_D1D - 1;
   constexpr int MNQ_O =
      T_NQUAD_O ? T_NQUAD_O : DofQuadLimits::HDIV_MAX_D1D - 1;
   constexpr int MNDQ = std::max(MND_O + 1, MNQ_O + 1);
   constexpr int TBATCH = curlinterp::NBZ3D(T_NDOF_O, T_NQUAD_O, MNDQ);
   MFEM_VERIFY(ndof_o <= MND_O, "Error: H(curl) order larger than supported");
   MFEM_VERIFY(nquad_o <= MNQ_O, "Error: H(div) order larger than supported");
   int mnq = std::max(ndof_o + 1, nquad_o + 1);
   auto pa_data = pa.Read();
   auto x_d = x_.Read();
   auto y_d = y_.ReadWrite();
   mfem::forall_2D_batch<MNDQ * MNDQ * MNDQ * TBATCH>(
      ne, mnq * mnq * mnq, 1, TBATCH, [=] MFEM_HOST_DEVICE(int e)
   {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      constexpr int nbz = TBATCH ? TBATCH : 1;
      int tidz = MFEM_THREAD_ID(z);
      // Make mnq a local variable since capturing would result in different
      // captures between host/device versions, and spuriously fails
      int mnq = std::max(ndof_o + 1, nquad_o + 1);
#else
      constexpr int nbz = 1;
      constexpr int tidz = 0;
#endif
      const int NDOF_O = T_NDOF_O ? T_NDOF_O : ndof_o;
      const int NQUAD_O = T_NQUAD_O ? T_NQUAD_O : nquad_o;
      const int NDOF_C = NDOF_O + 1;
      const int NQUAD_C = NQUAD_O + 1;
      MFEM_SHARED real_t
      sBG[(MND_O + 1) * MNQ_O + (MND_O + 1) * (MNQ_O + 1) + MND_O * MNQ_O];
      auto X_ = Reshape(x_d, 3 * NDOF_C * NDOF_C * NDOF_O, ne);
      auto Y = Reshape(y_d, 3 * NQUAD_C * NQUAD_O * NQUAD_O, ne);
      auto Gco = Reshape(sBG, NQUAD_O, NDOF_C);
      auto Bcc = Reshape(sBG + NDOF_C * NQUAD_O, NQUAD_C, NDOF_C);
      auto Boo =
         Reshape(sBG + NDOF_C * NQUAD_O + NDOF_C * NQUAD_C, NQUAD_O, NDOF_O);
      MFEM_SHARED real_t sX[3 * nbz * MND_O * (MND_O + 1) * (MND_O + 1)];
      MFEM_SHARED real_t sm0[nbz * 2 * MNDQ * MNDQ * MNDQ];
      MFEM_SHARED real_t sm1[nbz * 2 * MNDQ * MNDQ * MNDQ];

      real_t(*X)[nbz][MND_O * (MND_O + 1) * (MND_O + 1)] =
         (real_t(*)[nbz][MND_O * (MND_O + 1) * (MND_O + 1)])(sX);
      // shapes of buffers always use MNDQ to mitigate shared memory bank
      // conflicts
      real_t(*DDQ)[nbz][MNDQ][MNDQ][MNDQ] =
         (real_t(*)[nbz][MNDQ][MNDQ][MNDQ])(sm0);
      real_t(*DQQ)[nbz][MNDQ][MNDQ][MNDQ] =
         (real_t(*)[nbz][MNDQ][MNDQ][MNDQ])(sm1);
      real_t(*QQQ)[nbz][MNDQ][MNDQ][MNDQ] =
         (real_t(*)[nbz][MNDQ][MNDQ][MNDQ])(sm0);
      const int offset = NDOF_O * NDOF_C * NDOF_C;
      const int offsetq = NQUAD_C * NQUAD_O * NQUAD_O;
      MFEM_FOREACH_THREAD_DIRECT(ix, x, offset)
      {
         for (int dim = 0; dim < 3; ++dim)
         {
            X[dim][tidz][ix] = X_(ix + dim * offset, e);
         }
      }
      // load basis functions data
      if (tidz == 0)
      {
         auto npts = NDOF_C * NQUAD_O + NDOF_C * NQUAD_C + NDOF_O * NQUAD_O;
         MFEM_FOREACH_THREAD(ix, x, npts) { sBG[ix] = pa_data[ix]; }
      }
      MFEM_SYNC_THREAD;

      // x: Vz Bcc Gco Boo - Vy Bcc Boo Gco
      // threads assigned to mitigate bank conflicts
      // #define MFEM_HCURL_OFFSET
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, dy, dz, x, NQUAD_C, NDOF_C,
                                           NDOF_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dx = 0; dx < NDOF_C; ++dx)
         {
            u += X[2][tidz][dx + (dy + dz * NDOF_C) * NDOF_C] * Bcc(qx, dx);
         }
         DDQ[0][tidz][dz][dy][qx] = u;
      }
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, dy, dz, x, NQUAD_C, NDOF_O,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dx = 0; dx < NDOF_C; ++dx)
         {
            u += X[1][tidz][dx + (dy + dz * NDOF_O) * NDOF_C] * Bcc(qx, dx);
         }
         DDQ[1][tidz][dz][dy][qx] = u;
      }
      MFEM_SYNC_THREAD;

      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, dz, x, NQUAD_C, NQUAD_O,
                                           NDOF_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dy = 0; dy < NDOF_C; ++dy)
         {
            u += DDQ[0][tidz][dz][dy][qx] * Gco(qy, dy);
         }
         DQQ[0][tidz][dz][qy][qx] = u;
      }
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, dz, x, NQUAD_C, NQUAD_O,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dy = 0; dy < NDOF_O; ++dy)
         {
            u += DDQ[1][tidz][dz][dy][qx] * Boo(qy, dy);
         }
         DQQ[1][tidz][dz][qy][qx] = u;
      }
      MFEM_SYNC_THREAD;

      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, qz, x, NQUAD_C, NQUAD_O,
                                           NQUAD_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dz = 0; dz < NDOF_O; ++dz)
         {
            u += DQQ[0][tidz][dz][qy][qx] * Boo(qz, dz);
         }
         QQQ[0][tidz][qz][qy][qx] = u;
      }
      MFEM_SYNC_THREAD;
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, qz, x, NQUAD_C, NQUAD_O,
                                           NQUAD_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dz = 0; dz < NDOF_C; ++dz)
         {
            u += DQQ[1][tidz][dz][qy][qx] * Gco(qz, dz);
         }
         Y(qx + (qy + qz * NQUAD_O) * NQUAD_C, e) =
            QQQ[0][tidz][qz][qy][qx] - u;
      }
      MFEM_SYNC_THREAD;

      // y: Vx Boo Bcc Gco - Vz Gco Bcc Boo
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, dy, dz, x, NQUAD_O, NDOF_C,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dx = 0; dx < NDOF_O; ++dx)
         {
            u += X[0][tidz][dx + (dy + dz * NDOF_C) * NDOF_O] * Boo(qx, dx);
         }
         DDQ[0][tidz][dz][dy][qx] = u;
      }
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, dy, dz, x, NQUAD_O, NDOF_C,
                                           NDOF_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dx = 0; dx < NDOF_C; ++dx)
         {
            u += X[2][tidz][dx + (dy + dz * NDOF_C) * NDOF_C] * Gco(qx, dx);
         }
         DDQ[1][tidz][dz][dy][qx] = u;
      }
      MFEM_SYNC_THREAD;

      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, dz, x, NQUAD_O, NQUAD_C,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dy = 0; dy < NDOF_C; ++dy)
         {
            u += DDQ[0][tidz][dz][dy][qx] * Bcc(qy, dy);
         }
         DQQ[0][tidz][dz][qy][qx] = u;
      }
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, dz, x, NQUAD_O, NQUAD_C,
                                           NDOF_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dy = 0; dy < NDOF_C; ++dy)
         {
            u += DDQ[1][tidz][dz][dy][qx] * Bcc(qy, dy);
         }
         DQQ[1][tidz][dz][qy][qx] = u;
      }
      MFEM_SYNC_THREAD;

      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, qz, x, NQUAD_O, NQUAD_C,
                                           NQUAD_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dz = 0; dz < NDOF_C; ++dz)
         {
            u += DQQ[0][tidz][dz][qy][qx] * Gco(qz, dz);
         }
         QQQ[0][tidz][qz][qy][qx] = u;
      }
      MFEM_SYNC_THREAD;
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, qz, x, NQUAD_O, NQUAD_C,
                                           NQUAD_O, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dz = 0; dz < NDOF_O; ++dz)
         {
            u += DQQ[1][tidz][dz][qy][qx] * Boo(qz, dz);
         }
         Y(qx + (qy + qz * NQUAD_C) * NQUAD_O + offsetq, e) =
            QQQ[0][tidz][qz][qy][qx] - u;
      }
      MFEM_SYNC_THREAD;

      // z: Vy Gco Boo Bcc - Vx Boo Gco Bcc
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, dy, dz, x, NQUAD_O, NDOF_O,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dx = 0; dx < NDOF_C; ++dx)
         {
            u += X[1][tidz][dx + (dy + dz * NDOF_O) * NDOF_C] * Gco(qx, dx);
         }
         DDQ[0][tidz][dz][dy][qx] = u;
      }
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, dy, dz, x, NQUAD_O, NDOF_C,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dx = 0; dx < NDOF_O; ++dx)
         {
            u += X[0][tidz][dx + (dy + dz * NDOF_C) * NDOF_O] * Boo(qx, dx);
         }
         DDQ[1][tidz][dz][dy][qx] = u;
      }
      MFEM_SYNC_THREAD;

      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, dz, x, NQUAD_O, NQUAD_O,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dy = 0; dy < NDOF_O; ++dy)
         {
            u += DDQ[0][tidz][dz][dy][qx] * Boo(qy, dy);
         }
         DQQ[0][tidz][dz][qy][qx] = u;
      }
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, dz, x, NQUAD_O, NQUAD_O,
                                           NDOF_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dy = 0; dy < NDOF_C; ++dy)
         {
            u += DDQ[1][tidz][dz][dy][qx] * Gco(qy, dy);
         }
         DQQ[1][tidz][dz][qy][qx] = u;
      }
      MFEM_SYNC_THREAD;

      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, qz, x, NQUAD_O, NQUAD_O,
                                           NQUAD_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dz = 0; dz < NDOF_C; ++dz)
         {
            u += DQQ[0][tidz][dz][qy][qx] * Bcc(qz, dz);
         }
         QQQ[0][tidz][qz][qy][qx] = u;
      }
      MFEM_SYNC_THREAD;
      // threads assigned to mitigate bank conflicts
      MFEM_FOREACH_THREAD_DIRECT_3D_OFFSET(qx, qy, qz, x, NQUAD_O, NQUAD_O,
                                           NQUAD_C, mnq, mnq, mnq)
      {
         real_t u = 0;
         for (int dz = 0; dz < NDOF_C; ++dz)
         {
            u += DQQ[1][tidz][dz][qy][qx] * Bcc(qz, dz);
         }
         Y(qx + (qy + qz * NQUAD_O) * NQUAD_O + 2 * offsetq, e) =
            QQQ[0][tidz][qz][qy][qx] - u;
      }
      MFEM_SYNC_THREAD;
   });
}

template <int NDOF_O, int NQUAD_O>
void CurlInterpolatorTApply3DSmem(const int ne, const int ndof_o,
                                  const int nquad_o, const Vector &pa,
                                  const Vector &x, Vector &y)
{
   // TODO do half of the sum factorization cycle from D->Q
}

} // namespace internal

template <int DIM, int NDOF_O, int NQUAD_O>
CurlInterpolator::ApplyKernelType
CurlInterpolator::ApplyPAKernels::Kernel()
{
   if constexpr (DIM == 3)
   {
      return internal::CurlInterpolatorApply3DSmem<NDOF_O, NQUAD_O>;
   }
   MFEM_ABORT("Bad dimension!");
}

template <int DIM, int NDOF_O, int NQUAD_O>
CurlInterpolator::ApplyKernelType
CurlInterpolator::ApplyTPAKernels::Kernel()
{
   if constexpr (DIM == 3)
   {
      return internal::CurlInterpolatorTApply3DSmem<NDOF_O, NQUAD_O>;
   }
   MFEM_ABORT("Bad dimension!");
}

} // namespace mfem

/// \endcond DO_NOT_DOCUMENT

#endif
