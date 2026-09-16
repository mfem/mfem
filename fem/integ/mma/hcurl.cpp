// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "hcurl.hpp"
#include "mma.hpp"
#include "../../qfunction.hpp"
#include "../../fe/fe_nd.hpp"
#include "../../doftrans.hpp"

namespace mfem
{

namespace internal
{

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PAHcurlMassApply2D;
// not a call wrapper.
void MmaHcurlMassApplyTensors2D(const int NE, const bool symmetric,
                        [[maybe_unused]] const bool scalar_coeff,
                        const Array<real_t> &bo, const Array<real_t> &bc,
                        const Array<real_t> &bot, const Array<real_t> &bct,
                        const Vector &pa_data, const Vector &x, Vector &y,
                        const int D1D, [[maybe_unused]] const int TestD1D,
                        const int Q1D)
{
   MFEM_ASSERT(D1D == TestD1D,
               "Trial and Test space must have the same number of dofs");
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
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
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t O11 = op(qx,qy,0,e);
            const real_t O21 = op(qx,qy,1,e);
            const real_t O12 = symmetric ? O21 : op(qx,qy,2,e);
            const real_t O22 = symmetric ? op(qx,qy,2,e) : op(qx,qy,3,e);
            const real_t massX = mass[qy][qx][0];
            const real_t massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O21*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PAHcurlMassApply3D;
// not a call wrapper.
void MmaHcurlMassApplyTensors3D(const int NE, const bool symmetric,
                        [[maybe_unused]] const bool scalar_coeff,
                        const Array<real_t> &bo, const Array<real_t> &bc,
                        const Array<real_t> &bot, const Array<real_t> &bct,
                        const Vector &pa_data, const Vector &x, Vector &y,
                        const int D1D, [[maybe_unused]] const int TestD1D,
                        const int Q1D)
{
   MFEM_VERIFY(D1D == TestD1D,
               "Trial and test spaces must have same number of dofs");
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

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
            real_t massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[MAX_Q1D];
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
               const real_t O11 = op(qx,qy,qz,0,e);
               const real_t O12 = op(qx,qy,qz,1,e);
               const real_t O13 = op(qx,qy,qz,2,e);
               const real_t O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const real_t O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const real_t O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const real_t O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const real_t O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const real_t O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);
               const real_t massX = mass[qz][qy][qx][0];
               const real_t massY = mass[qz][qy][qx][1];
               const real_t massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t massXY[MAX_D1D][MAX_D1D];

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
                  massXY[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
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

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PACurlCurlApply2D;
// not a call wrapper.
void MmaCurlCurlApplyTensors2D(const int D1D, const int Q1D, const bool, const int NE,
                       const Array<real_t> &bo, const Array<real_t> &,
                       const Array<real_t> &bot, const Array<real_t> &,
                       const Array<real_t> &gc, const Array<real_t> &gct,
                       const Vector &pa_data, const Vector &x, Vector &y,
                       const bool useAbs)
{

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t curl[MAX_Q1D][MAX_Q1D];

      // curl[qy][qx] will be computed as du_y/dx - du_x/dy

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] = 0.0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int sign = useAbs ? 1 : -1;
               const real_t wy = (c == 0) ? (sign*Gc(qy,dy)) : Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  curl[qy][qx] += gradX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            real_t gradX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               gradX[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradX[dx] += curl[qy][qx] * ((c == 0) ? Bot(dx,qx) : Gct(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const int sign = useAbs ? 1 : -1;
               const real_t wy = (c == 0) ? (sign*Gct(dy,qy)) : Bot(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += gradX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PACurlCurlApply3D;
// not a call wrapper.
void MmaCurlCurlApplyTensors3D(const int d1d,
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
                              const bool useAbs)
{
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: d1d > HCURL_MAX_D1D");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: q1d > HCURL_MAX_Q1D");
   const int D1D = d1d;
   const int Q1D = q1d;

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
      constexpr int MD1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr int MQ1D = DofQuadLimits::HCURL_MAX_Q1D;
      const int D1D = d1d;
      const int Q1D = q1d;

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


void MmaHcurlMassApplySimplex(const int dim, const int NE, const int nd,
                              const int nq, const int sdim,
                              const bool symmetric,
                              const Array<real_t> &B,
                              const Vector &pa_data,
                              const Vector &x, Vector &y)
{
   // B: (nq, nd, sdim[, NE]) physical vector shapes (Piola already applied)
   // pa_data: (nq, ncomp, NE) physical mass metric Q*w*|J|
   MFEM_VERIFY(B.Size() > 0, "Simplex H(curl) mass MMA requires Q-space bases");
   const int ncomp = symmetric ? (sdim * (sdim + 1)) / 2 : sdim * sdim;
   const bool per_elem_B = (B.Size() == nq * nd * sdim * NE);
   const auto Bb4 = Reshape(B.Read(), nq, nd, sdim, per_elem_B ? NE : 1);
   const auto D = Reshape(pa_data.Read(), nq, ncomp, NE);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int be = per_elem_B ? e : 0;
      for (int q = 0; q < nq; ++q)
      {
         real_t u[3] = {};
         for (int c = 0; c < sdim; ++c)
         {
            real_t s = 0.0;
            for (int i = 0; i < nd; ++i) { s += Bb4(q, i, c, be) * X(i, e); }
            u[c] = s;
         }

         real_t v[3] = {};
         if (sdim == 2)
         {
            const real_t O11 = D(q, 0, e);
            const real_t O21 = D(q, 1, e);
            const real_t O12 = symmetric ? O21 : D(q, 2, e);
            const real_t O22 = symmetric ? D(q, 2, e) : D(q, 3, e);
            v[0] = O11 * u[0] + O12 * u[1];
            v[1] = O21 * u[0] + O22 * u[1];
         }
         else
         {
            real_t A[3][3];
            if (symmetric)
            {
               A[0][0] = D(q, 0, e);
               A[1][0] = A[0][1] = D(q, 1, e);
               A[2][0] = A[0][2] = D(q, 2, e);
               A[1][1] = D(q, 3, e);
               A[2][1] = A[1][2] = D(q, 4, e);
               A[2][2] = D(q, 5, e);
            }
            else
            {
               for (int i = 0; i < 3; ++i)
                  for (int j = 0; j < 3; ++j)
                  {
                     A[i][j] = D(q, i * 3 + j, e);
                  }
            }
            for (int i = 0; i < 3; ++i)
            {
               v[i] = A[i][0] * u[0] + A[i][1] * u[1] + A[i][2] * u[2];
            }
         }

         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int c = 0; c < sdim; ++c) { s += Bb4(q, i, c, be) * v[c]; }
            Y(i, e) += s;
         }
      }
   });
}

void MmaCurlCurlApplySimplex(const int dim, const int NE, const int nd,
                             const int nq, const int curl_dim,
                             const bool /*symmetric*/,
                             const Array<real_t> &C,
                             const Vector &pa_data,
                             const Vector &x, Vector &y)
{
   MFEM_VERIFY(C.Size() > 0, "Simplex curl-curl MMA requires Q-space bases");
   // C: (nq, nd, curl_dim[, NE]) reference curl shapes
   const bool per_elem_C = (C.Size() == nq * nd * curl_dim * NE);
   const auto Cc = Reshape(C.Read(), nq, nd, curl_dim, per_elem_C ? NE : 1);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   if (dim == 2)
   {
      MFEM_VERIFY(curl_dim == 1, "");
      const auto D = Reshape(pa_data.Read(), nq, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         const int ce = per_elem_C ? e : 0;
         for (int q = 0; q < nq; ++q)
         {
            real_t curl = 0.0;
            for (int i = 0; i < nd; ++i) { curl += Cc(q, i, 0, ce) * X(i, e); }
            curl *= D(q, e);
            for (int i = 0; i < nd; ++i) { Y(i, e) += Cc(q, i, 0, ce) * curl; }
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3 && curl_dim == 3, "");
   const int ncomp = pa_data.Size() / (nq * NE);
   const auto D = Reshape(pa_data.Read(), nq, ncomp, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int ce = per_elem_C ? e : 0;
      for (int q = 0; q < nq; ++q)
      {
         real_t u[3] = {};
         for (int c = 0; c < 3; ++c)
         {
            for (int i = 0; i < nd; ++i) { u[c] += Cc(q, i, c, ce) * X(i, e); }
         }
         real_t v[3] = {};
         if (ncomp == 6)
         {
            const real_t a00 = D(q, 0, e), a10 = D(q, 1, e), a20 = D(q, 2, e);
            const real_t a11 = D(q, 3, e), a21 = D(q, 4, e), a22 = D(q, 5, e);
            v[0] = a00 * u[0] + a10 * u[1] + a20 * u[2];
            v[1] = a10 * u[0] + a11 * u[1] + a21 * u[2];
            v[2] = a20 * u[0] + a21 * u[1] + a22 * u[2];
         }
         else
         {
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
               {
                  v[i] += D(q, i * 3 + j, e) * u[j];
               }
         }
         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int c = 0; c < 3; ++c) { s += Cc(q, i, c, ce) * v[c]; }
            Y(i, e) += s;
         }
      }
   });
}

} // namespace internal

namespace
{


/** Apply TransformDual on the nd-axis of a (nq,nd,ncomp[,NE]) basis so that
    B_eff @ x_E = B_native @ InvTransformPrimal(x_E) and the dual pullback
    matches FA TransformDual without an EA Mult path. */
void BakeNdDofTransformation(const FiniteElementSpace &fes,
                             Array<real_t> &B, int nq, int nd, int ncomp)
{
   const int NE = fes.GetNE();
   MFEM_VERIFY(B.Size() == nq * nd * ncomp * NE || B.Size() == nq * nd * ncomp,
               "unexpected simplex basis size");
   if (B.Size() == nq * nd * ncomp)
   {
      Array<real_t> B0 = B;
      B.SetSize(nq * nd * ncomp * NE);
      const auto Bin = Reshape(B0.HostRead(), nq, nd, ncomp);
      auto Bout = Reshape(B.HostWrite(), nq, nd, ncomp, NE);
      for (int e = 0; e < NE; ++e)
         for (int q = 0; q < nq; ++q)
            for (int c = 0; c < ncomp; ++c)
               for (int i = 0; i < nd; ++i)
               {
                  Bout(q, i, c, e) = Bin(q, i, c);
               }
   }
   // In-place transform on host memory
   auto Bb = Reshape(B.HostReadWrite(), nq, nd, ncomp, NE);
   Array<int> vdofs;
   Vector col(nd);
   for (int e = 0; e < NE; ++e)
   {
      DofTransformation *dt = fes.GetElementVDofs(e, vdofs);
      if (!dt) { continue; }
      for (int q = 0; q < nq; ++q)
         for (int c = 0; c < ncomp; ++c)
         {
            for (int i = 0; i < nd; ++i) { col(i) = Bb(q, i, c, e); }
            dt->TransformDual(col);
            for (int i = 0; i < nd; ++i) { Bb(q, i, c, e) = col(i); }
         }
   }
}

/** Pack physical ND vector shapes at IR into B: layout (nq, nd, sdim, NE). */
void BuildNdPhysVShape(const FiniteElementSpace &fes, const IntegrationRule &ir,
                       Array<real_t> &B)
{
   const FiniteElement &el = *fes.GetTypicalFE();
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   const int sdim = el.GetDim();
   const int NE = fes.GetNE();
   B.SetSize(nq * nd * sdim * NE);
   DenseMatrix vshape(nd, sdim);
   auto Bb = Reshape(B.HostWrite(), nq, nd, sdim, NE);
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcVShape(T, vshape); // physical Piola
         for (int i = 0; i < nd; ++i)
            for (int c = 0; c < sdim; ++c)
            {
               Bb(q, i, c, e) = vshape(i, c);
            }
      }
   }
}

/** Physical-space mass metric D = Q * w * |J| (FA VectorFEMass weight). */
void SetupHcurlMassPaSimplex(const FiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             Coefficient *Q,
                             DiagonalMatrixCoefficient *DQ,
                             MatrixCoefficient *MQ,
                             bool &symmetric,
                             Vector &pa_data)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const int NE = fes.GetNE();
   const int nq = ir.GetNPoints();
   const int symmDims = (dim * (dim + 1)) / 2;
   if (MQ && !dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      symmetric = false;
   }
   else
   {
      symmetric = true;
   }
   const int ncomp = symmetric ? symmDims : dim * dim;
   pa_data.SetSize(ncomp * nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, ncomp, NE);

   DenseMatrix M, A(dim);
   Vector Dvec;
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t wdet = ip.weight * T.Weight();
         A = 0.0;
         if (MQ)
         {
            MQ->Eval(M, T, ip);
            A = M;
            A *= wdet;
         }
         else if (DQ)
         {
            DQ->Eval(Dvec, T, ip);
            for (int i = 0; i < dim; ++i) { A(i, i) = Dvec(i) * wdet; }
         }
         else
         {
            const real_t qv = Q ? Q->Eval(T, ip) : real_t(1.0);
            for (int i = 0; i < dim; ++i) { A(i, i) = qv * wdet; }
         }

         if (symmetric)
         {
            if (dim == 2)
            {
               D(q, 0, e) = A(0, 0);
               D(q, 1, e) = A(1, 0);
               D(q, 2, e) = A(1, 1);
            }
            else
            {
               D(q, 0, e) = A(0, 0);
               D(q, 1, e) = A(1, 0);
               D(q, 2, e) = A(2, 0);
               D(q, 3, e) = A(1, 1);
               D(q, 4, e) = A(2, 1);
               D(q, 5, e) = A(2, 2);
            }
         }
         else
         {
            for (int i = 0; i < dim; ++i)
               for (int j = 0; j < dim; ++j)
               {
                  D(q, i * dim + j, e) = A(i, j);
               }
         }
      }
   }
}

/** Build reference curl shapes at IR: C(q,i,c). */
void BuildNdRefCurlShape(const FiniteElement &el, const IntegrationRule &ir,
                         Array<real_t> &C)
{
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   const int cdim = el.GetCurlDim();
   C.SetSize(nq * nd * cdim);
   DenseMatrix cshape(nd, cdim);
   auto Cc = Reshape(C.HostWrite(), nq, nd, cdim);
   for (int q = 0; q < nq; ++q)
   {
      el.CalcCurlShape(ir.IntPoint(q), cshape);
      for (int i = 0; i < nd; ++i)
         for (int c = 0; c < cdim; ++c)
         {
            Cc(q, i, c) = cshape(i, c);
         }
   }
}

void SetupCurlCurlPaSimplex(const FiniteElementSpace &fes,
                            const IntegrationRule &ir,
                            Coefficient *Q,
                            DiagonalMatrixCoefficient *DQ,
                            MatrixCoefficient *MQ,
                            bool &symmetric,
                            Vector &pa_data)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const int NE = fes.GetNE();
   const int nq = ir.GetNPoints();
   symmetric = true;
   if (MQ && !dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      symmetric = false;
   }

   if (dim == 2)
   {
      // D(q,e) = Q * w / |J|  so that (ref_curl)^T D (ref_curl) matches FA
      pa_data.SetSize(nq * NE, Device::GetMemoryType());
      auto D = Reshape(pa_data.HostWrite(), nq, NE);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation &T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);
            real_t coeff = 1.0;
            if (Q) { coeff = Q->Eval(T, ip); }
            else if (DQ)
            {
               Vector d(1);
               DQ->Eval(d, T, ip);
               coeff = d(0);
            }
            else if (MQ)
            {
               DenseMatrix M(1);
               MQ->Eval(M, T, ip);
               coeff = M(0, 0);
            }
            D(q, e) = coeff * ip.weight / T.Weight();
         }
      }
      return;
   }

   // 3D: phys_curl = J * curl_ref / |J|
   // FA: w*|J|*Q * phys·phys = w/|J| * curl_ref^T (J^T Q J) curl_ref
   const int ncomp = symmetric ? 6 : 9;
   pa_data.SetSize(ncomp * nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, ncomp, NE);
   DenseMatrix M, Qm(3), JJ(3), tmp(3), A(3);
   Vector Dv;
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         const DenseMatrix &J = T.Jacobian();
         Qm = 0.0;
         if (MQ)
         {
            MQ->Eval(M, T, ip);
            Qm = M;
         }
         else if (DQ)
         {
            DQ->Eval(Dv, T, ip);
            for (int i = 0; i < 3; ++i) { Qm(i, i) = Dv(i); }
         }
         else if (Q)
         {
            const real_t qv = Q->Eval(T, ip);
            for (int i = 0; i < 3; ++i) { Qm(i, i) = qv; }
         }
         else
         {
            for (int i = 0; i < 3; ++i) { Qm(i, i) = 1.0; }
         }
         // A = (w/|J|) * J^T Q J
         MultAtB(J, Qm, tmp); // J^T Q
         Mult(tmp, J, A);     // J^T Q J
         A *= (ip.weight / T.Weight());
         if (symmetric)
         {
            D(q, 0, e) = A(0, 0);
            D(q, 1, e) = A(1, 0);
            D(q, 2, e) = A(2, 0);
            D(q, 3, e) = A(1, 1);
            D(q, 4, e) = A(2, 1);
            D(q, 5, e) = A(2, 2);
         }
         else
         {
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
               {
                  D(q, i * 3 + j, e) = A(i, j);
               }
         }
      }
   }
}

} // namespace


void VectorFEMassIntegrator::AssembleSimplexMmaHcurlPA(
   const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::CURL, "");
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir_ptr = IntRule ? IntRule :
                                   &MassIntegrator::GetRule(el, el, Trans);
   const IntegrationRule &ir = *ir_ptr;
   nq = ir.GetNPoints();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = 0;
   mapsO = mapsC = mapsOtest = mapsCtest = nullptr;
   geom = nullptr;
   trial_fetype = test_fetype = FiniteElement::CURL;
   use_simplices_mma = true;
   use_tensors_mma = false;

   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = 0;

   BuildNdPhysVShape(fes, ir, simplex_B);
   if (el.GetDofTransformation() != nullptr)
   {
      BakeNdDofTransformation(fes, simplex_B, nq, simplex_nd, simplex_sdim);
   }
   SetupHcurlMassPaSimplex(fes, ir, Q, DQ, MQ, symmetric, pa_data);
}

void CurlCurlIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::CURL, "");
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir_ptr = IntRule ? IntRule :
                                   &MassIntegrator::GetRule(el, el, Trans);
   const IntegrationRule &ir = *ir_ptr;
   nq = ir.GetNPoints();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = 0;
   mapsO = mapsC = nullptr;
   geom = nullptr;
   use_simplices_mma = true;
   use_tensors_mma = false;

   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = el.GetCurlDim();

   BuildNdRefCurlShape(el, ir, simplex_B);
   if (el.GetDofTransformation() != nullptr)
   {
      BakeNdDofTransformation(fes, simplex_B, nq, simplex_nd, simplex_curl_dim);
   }
   SetupCurlCurlPaSimplex(fes, ir, Q, DQ, MQ, symmetric, pa_data);
}

} // namespace mfem
