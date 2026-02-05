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

#include "bilininteg_hcurl_kernels.hpp"

namespace mfem
{

namespace internal
{

void PAHcurlMassAssembleDiagonal2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &bo,
                                   const Array<real_t> &bc,
                                   const Vector &pa_data,
                                   Vector &diag)
{
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto D = Reshape(diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         real_t mass[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                  mass[qx] += wy * wy * ((c == 0) ? op(qx,qy,0,e) :
                                         op(qx,qy,symmetric ? 2 : 3, e));
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  D(dx + (dy * D1Dx) + osc, e) += mass[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

void PAHcurlMassAssembleDiagonal3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &bo,
                                   const Array<real_t> &bc,
                                   const Vector &pa_data,
                                   Vector &diag)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D,
               "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
               "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         const int opc = (c == 0) ? 0 : ((c == 1) ? (symmetric ? 3 : 4) :
                                         (symmetric ? 5 : 8));

         real_t mass[MAX_Q1D];

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const real_t wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);

                        mass[qx] += wy * wy * wz * wz * op(qx,qy,qz,opc,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += mass[qx] * wx * wx;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

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
                        Vector &y)
{
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
                        Vector &y)
{
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

void PACurlCurlSetup2D(const int Q1D,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff,
                       Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto C = Reshape(coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         const real_t J11 = J(q,0,0,e);
         const real_t J21 = J(q,1,0,e);
         const real_t J12 = J(q,0,1,e);
         const real_t J22 = J(q,1,1,e);
         const real_t detJ = (J11*J22)-(J21*J12);
         y(q,e) = W[q] * C(q,e) / detJ;
      }
   });
}

void PACurlCurlSetup3D(const int Q1D,
                       const int coeffDim,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff,
                       Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   const bool symmetric = (coeffDim != 9);
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto C = Reshape(coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), NQ, symmetric ? 6 : 9, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         const real_t J11 = J(q,0,0,e);
         const real_t J21 = J(q,1,0,e);
         const real_t J31 = J(q,2,0,e);
         const real_t J12 = J(q,0,1,e);
         const real_t J22 = J(q,1,1,e);
         const real_t J32 = J(q,2,1,e);
         const real_t J13 = J(q,0,2,e);
         const real_t J23 = J(q,1,2,e);
         const real_t J33 = J(q,2,2,e);
         const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                             J21 * (J12 * J33 - J32 * J13) +
                             J31 * (J12 * J23 - J22 * J13);

         const real_t c_detJ = W[q] / detJ;

         if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
         {
            // Set y to the 6 or 9 entries of J^T M J / det
            const real_t M11 = C(0, q, e);
            const real_t M12 = C(1, q, e);
            const real_t M13 = C(2, q, e);
            const real_t M21 = (!symmetric) ? C(3, q, e) : M12;
            const real_t M22 = (!symmetric) ? C(4, q, e) : C(3, q, e);
            const real_t M23 = (!symmetric) ? C(5, q, e) : C(4, q, e);
            const real_t M31 = (!symmetric) ? C(6, q, e) : M13;
            const real_t M32 = (!symmetric) ? C(7, q, e) : M23;
            const real_t M33 = (!symmetric) ? C(8, q, e) : C(5, q, e);

            // First compute R = MJ
            const real_t R11 = M11*J11 + M12*J21 + M13*J31;
            const real_t R12 = M11*J12 + M12*J22 + M13*J32;
            const real_t R13 = M11*J13 + M12*J23 + M13*J33;
            const real_t R21 = M21*J11 + M22*J21 + M23*J31;
            const real_t R22 = M21*J12 + M22*J22 + M23*J32;
            const real_t R23 = M21*J13 + M22*J23 + M23*J33;
            const real_t R31 = M31*J11 + M32*J21 + M33*J31;
            const real_t R32 = M31*J12 + M32*J22 + M33*J32;
            const real_t R33 = M31*J13 + M32*J23 + M33*J33;

            // Now set y to J^T R / det
            y(q,0,e) = c_detJ * (J11*R11 + J21*R21 + J31*R31); // 1,1
            const real_t Y12 = c_detJ * (J11*R12 + J21*R22 + J31*R32);
            y(q,1,e) = Y12; // 1,2
            y(q,2,e) = c_detJ * (J11*R13 + J21*R23 + J31*R33); // 1,3

            const real_t Y21 = c_detJ * (J12*R11 + J22*R21 + J32*R31);
            const real_t Y22 = c_detJ * (J12*R12 + J22*R22 + J32*R32);
            const real_t Y23 = c_detJ * (J12*R13 + J22*R23 + J32*R33);

            const real_t Y33 = c_detJ * (J13*R13 + J23*R23 + J33*R33);

            y(q,3,e) = symmetric ? Y22 : Y21; // 2,2 or 2,1
            y(q,4,e) = symmetric ? Y23 : Y22; // 2,3 or 2,2
            y(q,5,e) = symmetric ? Y33 : Y23; // 3,3 or 2,3

            if (!symmetric)
            {
               y(q,6,e) = c_detJ * (J13*R11 + J23*R21 + J33*R31); // 3,1
               y(q,7,e) = c_detJ * (J13*R12 + J23*R22 + J33*R32); // 3,2
               y(q,8,e) = Y33; // 3,3
            }
         }
         else  // Vector or scalar coefficient version
         {
            // Set y to the 6 entries of J^T D J / det^2
            const real_t D1 = C(0, q, e);
            const real_t D2 = coeffDim == 3 ? C(1, q, e) : D1;
            const real_t D3 = coeffDim == 3 ? C(2, q, e) : D1;

            y(q,0,e) = c_detJ * (D1*J11*J11 + D2*J21*J21 + D3*J31*J31); // 1,1
            y(q,1,e) = c_detJ * (D1*J11*J12 + D2*J21*J22 + D3*J31*J32); // 1,2
            y(q,2,e) = c_detJ * (D1*J11*J13 + D2*J21*J23 + D3*J31*J33); // 1,3
            y(q,3,e) = c_detJ * (D1*J12*J12 + D2*J22*J22 + D3*J32*J32); // 2,2
            y(q,4,e) = c_detJ * (D1*J12*J13 + D2*J22*J23 + D3*J32*J33); // 2,3
            y(q,5,e) = c_detJ * (D1*J13*J13 + D2*J23*J23 + D3*J33*J33); // 3,3
         }
      }
   });
}

void PACurlCurlAssembleDiagonal2D(const int D1D, const int Q1D, const bool,
                                  const int NE, const Array<real_t> &bo,
                                  const Array<real_t> &, const Array<real_t> &,
                                  const Array<real_t> &gc,
                                  const Vector &pa_data, Vector &diag)
{
   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto D = Reshape(diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         real_t t[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               t[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bo(qy,dy) : -Gc(qy,dy);
                  t[qx] += wy * wy * op(qx,qy,e);
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
                  D(dx + (dy * D1Dx) + osc, e) += t[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

void PACurlCurlApply2D(const int D1D, const int Q1D, const bool, const int NE,
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

void PAHcurlL2Setup2D(const int Q1D,
                      const int NE,
                      const Array<real_t> &w,
                      Vector &coeff,
                      Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto C = Reshape(coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         y(q,e) = W[q] * C(q,e);
      }
   });
}

void PAHcurlL2Setup3D(const int NQ,
                      const int coeffDim,
                      const int NE,
                      const Array<real_t> &w,
                      Vector &coeff,
                      Vector &op)
{
   auto W = w.Read();
   auto C = Reshape(coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), coeffDim, NQ, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         for (int c=0; c<coeffDim; ++c)
         {
            y(c,q,e) = W[q] * C(c,q,e);
         }
      }
   });
}

void PAHcurlL2Apply2D(const int D1D,
                      const int D1Dtest,
                      const int Q1D,
                      const int NE,
                      const Array<real_t> &bo,
                      const Array<real_t> &bot,
                      const Array<real_t> &bt,
                      const Array<real_t> &gc,
                      const Vector &pa_data,
                      const Vector &x, // trial = H(curl)
                      Vector &y)  // test = L2 or H1
{
   const int H1 = (D1Dtest == D1D);

   MFEM_VERIFY(y.Size() == NE*D1Dtest*D1Dtest, "Test vector of wrong dimension");

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1Dtest, D1Dtest, NE);

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
               const real_t wy = (c == 0) ? -Gc(qy,dy) : Bo(qy,dy);
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
         real_t sol_x[MAX_D1D];
         for (int dx = 0; dx < D1Dtest; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t s = curl[qy][qx];
            for (int dx = 0; dx < D1Dtest; ++dx)
            {
               sol_x[dx] += s * ((H1 == 1) ? Bt(dx,qx) : Bot(dx,qx));
            }
         }
         for (int dy = 0; dy < D1Dtest; ++dy)
         {
            const real_t wy = (H1 == 1) ? Bt(dy,qy) : Bot(dy,qy);

            for (int dx = 0; dx < D1Dtest; ++dx)
            {
               Y(dx,dy,e) += sol_x[dx] * wy;
            }
         }
      }  // loop qy
   }); // end of element loop
}

void PAHcurlL2ApplyTranspose2D(const int D1D,
                               const int D1Dtest,
                               const int Q1D,
                               const int NE,
                               const Array<real_t> &bo,
                               const Array<real_t> &bot,
                               const Array<real_t> &b,
                               const Array<real_t> &gct,
                               const Vector &pa_data,
                               const Vector &x, // trial = H(curl)
                               Vector &y)  // test = L2 or H1
{
   const int H1 = (D1Dtest == D1D);

   MFEM_VERIFY(x.Size() == NE*D1Dtest*D1Dtest, "Test vector of wrong dimension");

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), D1Dtest, D1Dtest, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HCURL_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HCURL_MAX_Q1D;

      real_t mass[MAX_Q1D][MAX_Q1D];

      // Zero-order term in L2 or H1 test space

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            mass[qy][qx] = 0.0;
         }
      }

      for (int dy = 0; dy < D1Dtest; ++dy)
      {
         real_t sol_x[MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < D1Dtest; ++dx)
         {
            const real_t s = X(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += s * ((H1 == 1) ? B(qx,dx) : Bo(qx,dx));
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t d2q = (H1 == 1) ? B(qy,dy) : Bo(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qy][qx] += d2q * sol_x[qx];
            }
         }
      }

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            mass[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         int osc = 0;

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
                  gradX[dx] += mass[qy][qx] * ((c == 0) ? Bot(dx,qx) : Gct(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 0) ? -Gct(dy,qy) : Bot(dy,qy);

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

} // namespace internal

} // namespace mfem
