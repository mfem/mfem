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

#include "hdiv.hpp"
#include "hcurl.hpp"
#include "mma.hpp"
#include "../../qfunction.hpp"
#include "../../fe/fe_rt.hpp"

namespace mfem
{

namespace internal
{

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PAHdivMassApply2D;
// not a call wrapper.
void MmaHdivMassApplyTensors2D(const int NE, const bool symmetric, const bool,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int TestD1D, const int Q1D)
{
   MFEM_VERIFY(D1D == TestD1D,
               "Trial and test spaces must have same number of dofs");
   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Bct = Reshape(Bct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

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
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = x(dx + (dy * D1Dx) + osc, e);
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
            const real_t O12 = op(qx,qy,1,e);
            const real_t O21 = symmetric ? O12 : op(qx,qy,2,e);
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
            const int D1Dx = (c == 1) ? D1D - 1 : D1D;
            const int D1Dy = (c == 0) ? D1D - 1 : D1D;

            real_t massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bct(dx,qx) :
                                                  Bot(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PAHdivMassApply3D;
// not a call wrapper.
void MmaHdivMassApplyTensors3D(const int NE, const bool symmetric, const bool,
                       const Array<real_t> &Bo_, const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_, const Array<real_t> &Bct_,
                       const Vector &op_, const Vector &x_, Vector &y_,
                       const int D1D, const int TestD1D, const int Q1D)
{
   MFEM_VERIFY(D1D == TestD1D,
               "Trial and test spaces must have same number of dofs");
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Bct = Reshape(Bct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto x = Reshape(x_.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t mass[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D][VDIM];

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
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t massXY[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t massX[DofQuadLimits::HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
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
         real_t massXY[DofQuadLimits::HDIV_MAX_D1D][DofQuadLimits::HDIV_MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D : D1D - 1;
            const int D1Dy = (c == 1) ? D1D : D1D - 1;
            const int D1Dx = (c == 0) ? D1D : D1D - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t massX[DofQuadLimits::HDIV_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] *
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
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PADivDivApply2D;
// not a call wrapper.
void MmaDivDivApplyTensors2D(const int D1D,
                     const int Q1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &Bot_,
                     const Array<real_t> &Gct_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_)
{
   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t div[MAX_Q1D][MAX_Q1D];

      // div[qy][qx] will be computed as du_x/dx + du_y/dy

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            div[qy][qx] = 0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = (c == 0) ? Bo(qy,dy) : Gc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  div[qy][qx] += gradX[qx] * wy;
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
            div[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dx = (c == 1) ? D1D - 1 : D1D;
            const int D1Dy = (c == 0) ? D1D - 1 : D1D;

            real_t gradX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               gradX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradX[dx] += div[qy][qx] * (c == 0 ? Gct(dx,qx) : Bot(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const real_t wy = (c == 0) ? Bot(dy,qy) : Gct(dy,qy);
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += gradX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

// Owned MMA tensor PA apply (Bo/Bc/Gc sum-fact). Algorithm matches stock PADivDivApply3D;
// not a call wrapper.
void MmaDivDivApplyTensors3D(const int D1D,
                     const int Q1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &Bot_,
                     const Array<real_t> &Gct_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      real_t div[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               div[qz][qy][qx] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            real_t aXY[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               real_t aX[DofQuadLimits::HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const real_t t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     aX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Gc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = aX[qx];
                     aXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const real_t wz = (c == 2) ? Gc(qz,dz) : Bo(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     div[qz][qy][qx] += aXY[qy][qx] * wz;
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
               div[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t aXY[DofQuadLimits::HDIV_MAX_D1D][DofQuadLimits::HDIV_MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D : D1D - 1;
            const int D1Dy = (c == 1) ? D1D : D1D - 1;
            const int D1Dx = (c == 0) ? D1D : D1D - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  aXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               real_t aX[DofQuadLimits::HDIV_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  aX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aX[dx] += div[qz][qy][qx] *
                               (c == 0 ? Gct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const real_t wy = (c == 1) ? Gct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aXY[dy][dx] += aX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const real_t wz = (c == 2) ? Gct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        aXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

void MmaHdivMassApplySimplex(const int dim, const int NE, const int nd,
                             const int nq, const int sdim,
                             const bool symmetric,
                             const Array<real_t> &B,
                             const Vector &pa_data,
                             const Vector &x, Vector &y)
{
   // Same dense vec-mass apply as H(curl); Piola is absorbed in pa_data.
   MmaHcurlMassApplySimplex(dim, NE, nd, nq, sdim, symmetric, B, pa_data, x, y);
}

void MmaDivDivApplySimplex(const int NE, const int nd, const int nq,
                           const Array<real_t> &Div,
                           const Vector &pa_data,
                           const Vector &x, Vector &y)
{
   const auto Dd = Reshape(Div.Read(), nq, nd);
   const auto D = Reshape(pa_data.Read(), nq, NE);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < nq; ++q)
      {
         real_t div = 0.0;
         for (int i = 0; i < nd; ++i) { div += Dd(q, i) * X(i, e); }
         div *= D(q, e);
         for (int i = 0; i < nd; ++i) { Y(i, e) += Dd(q, i) * div; }
      }
   });
}

} // namespace internal

namespace
{

void BuildRtRefVShape(const FiniteElement &el, const IntegrationRule &ir,
                      Array<real_t> &B)
{
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   const int sdim = el.GetDim();
   B.SetSize(nq * nd * sdim);
   DenseMatrix vshape(nd, sdim);
   auto Bb = Reshape(B.HostWrite(), nq, nd, sdim);
   for (int q = 0; q < nq; ++q)
   {
      el.CalcVShape(ir.IntPoint(q), vshape);
      for (int i = 0; i < nd; ++i)
         for (int c = 0; c < sdim; ++c)
         {
            Bb(q, i, c) = vshape(i, c);
         }
   }
}

void BuildRtRefDivShape(const FiniteElement &el, const IntegrationRule &ir,
                        Array<real_t> &Div)
{
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   Div.SetSize(nq * nd);
   Vector dshape(nd);
   auto Dd = Reshape(Div.HostWrite(), nq, nd);
   for (int q = 0; q < nq; ++q)
   {
      el.CalcDivShape(ir.IntPoint(q), dshape);
      for (int i = 0; i < nd; ++i) { Dd(q, i) = dshape(i); }
   }
}

/** H(div) mass: phys = (1/|J|) J û  → metric w/|J| * J^T Q J on ref shapes. */
void SetupHdivMassPaSimplex(const FiniteElementSpace &fes,
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
   symmetric = true;
   if (MQ && !dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      symmetric = false;
   }
   const int ncomp = symmetric ? symmDims : dim * dim;
   pa_data.SetSize(ncomp * nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, ncomp, NE);

   DenseMatrix Qm(dim), tmp(dim), A(dim), M;
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
            for (int i = 0; i < dim; ++i) { Qm(i, i) = Dv(i); }
         }
         else if (Q)
         {
            const real_t qv = Q->Eval(T, ip);
            for (int i = 0; i < dim; ++i) { Qm(i, i) = qv; }
         }
         else
         {
            for (int i = 0; i < dim; ++i) { Qm(i, i) = 1.0; }
         }
         // A = (w/|J|) * J^T Q J
         MultAtB(J, Qm, tmp);
         Mult(tmp, J, A);
         A *= (ip.weight / T.Weight());

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

void SetupDivDivPaSimplex(const FiniteElementSpace &fes,
                          const IntegrationRule &ir,
                          Coefficient *Q,
                          Vector &pa_data)
{
   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   const int nq = ir.GetNPoints();
   // phys_div = ref_div / |J|; FA weight*|J|*Q*phys^2 = Q*w/|J| * ref^2
   pa_data.SetSize(nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, NE);
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t coeff = Q ? Q->Eval(T, ip) : real_t(1.0);
         D(q, e) = coeff * ip.weight / T.Weight();
      }
   }
}

} // namespace

void VectorFEMassIntegrator::AssembleSimplexMmaHdivPA(
   const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::DIV, "");
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
   trial_fetype = test_fetype = FiniteElement::DIV;
   use_simplices_mma = true;
   use_tensors_mma = false;

   BuildRtRefVShape(el, ir, simplex_B);
   SetupHdivMassPaSimplex(fes, ir, Q, DQ, MQ, symmetric, pa_data);
   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = 0;
}

void DivDivIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::DIV, "");
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir_ptr = IntRule ? IntRule :
                                   &MassIntegrator::GetRule(el, el, Trans);
   const IntegrationRule &ir = *ir_ptr;
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = 0;
   mapsO = mapsC = nullptr;
   geom = nullptr;
   use_simplices_mma = true;
   use_tensors_mma = false;
   // DivDivIntegrator has no nq member — store via pa_data geometry only
   BuildRtRefDivShape(el, ir, simplex_B);
   SetupDivDivPaSimplex(fes, ir, Q, pa_data);
   simplex_nd = el.GetDof();
   simplex_nq = ir.GetNPoints();
   simplex_sdim = dim;
   simplex_curl_dim = 0;
}

} // namespace mfem
