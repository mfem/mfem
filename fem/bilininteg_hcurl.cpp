// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "libceed/mass.hpp"

using namespace std;

namespace mfem
{

void PAHcurlMassApply2D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const bool symmetric,
                        const Array<double> &bo,
                        const Array<double> &bc,
                        const Array<double> &bot,
                        const Array<double> &bct,
                        const Vector &pa_data,
                        const Vector &x,
                        Vector &y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][VDIM];

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
            double massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
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
            const double O11 = op(qx,qy,0,e);
            const double O21 = op(qx,qy,1,e);
            const double O12 = symmetric ? O21 : op(qx,qy,2,e);
            const double O22 = symmetric ? op(qx,qy,2,e) : op(qx,qy,3,e);
            const double massX = mass[qy][qx][0];
            const double massY = mass[qy][qx][1];
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

            double massX[MAX_D1D];
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
               const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

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

void PAHcurlMassAssembleDiagonal2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const bool symmetric,
                                   const Array<double> &bo,
                                   const Array<double> &bc,
                                   const Vector &pa_data,
                                   Vector &diag)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto D = Reshape(diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double mass[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                  mass[qx] += wy * wy * ((c == 0) ? op(qx,qy,0,e) :
                  op(qx,qy,symmetric ? 2 : 3, e));
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
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
                                   const Array<double> &bo,
                                   const Array<double> &bc,
                                   const Vector &pa_data,
                                   Vector &diag)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         const int opc = (c == 0) ? 0 : ((c == 1) ? (symmetric ? 3 : 4) :
         (symmetric ? 5 : 8));

         double mass[MAX_Q1D];

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);

                        mass[qx] += wy * wy * wz * wz * op(qx,qy,qz,opc,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += mass[qx] * wx * wx;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

template<int T_D1D, int T_Q1D>
void SmemPAHcurlMassAssembleDiagonal3D(const int D1D,
                                       const int Q1D,
                                       const int NE,
                                       const bool symmetric,
                                       const Array<double> &bo,
                                       const Array<double> &bc,
                                       const Vector &pa_data,
                                       Vector &diag)
{
   MFEM_VERIFY(D1D <= HCURL_MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= HCURL_MAX_Q1D, "Error: Q1D > MAX_Q1D");

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto D = Reshape(diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int VDIM = 3;
      constexpr int tD1D = T_D1D ? T_D1D : HCURL_MAX_D1D;
      constexpr int tQ1D = T_Q1D ? T_Q1D : HCURL_MAX_Q1D;

      MFEM_SHARED double sBo[tQ1D][tD1D];
      MFEM_SHARED double sBc[tQ1D][tD1D];

      double op3[3];
      MFEM_SHARED double sop[3][tQ1D][tQ1D];

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

         double dxyz = 0.0;

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
               const double wz = ((c == 2) ? sBo[qz][dz] : sBc[qz][dz]);

               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const double wy = ((c == 1) ? sBo[qy][dy] : sBc[qy][dy]);

                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const double wx = ((c == 0) ? sBo[qx][dx] : sBc[qx][dx]);
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

void PAHcurlMassApply3D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const bool symmetric,
                        const Array<double> &bo,
                        const Array<double> &bc,
                        const Array<double> &bot,
                        const Array<double> &bct,
                        const Vector &pa_data,
                        const Vector &x,
                        Vector &y)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

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
            double massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
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
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const double O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const double O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const double O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O21*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O31*massX)+(O32*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

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
               double massX[MAX_D1D];
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
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
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

template<int T_D1D, int T_Q1D>
void SmemPAHcurlMassApply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const bool symmetric,
                            const Array<double> &bo,
                            const Array<double> &bc,
                            const Array<double> &bot,
                            const Array<double> &bct,
                            const Vector &pa_data,
                            const Vector &x,
                            Vector &y)
{
   MFEM_VERIFY(D1D <= HCURL_MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= HCURL_MAX_Q1D, "Error: Q1D > MAX_Q1D");

   const int dataSize = symmetric ? 6 : 9;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, dataSize, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int VDIM = 3;
      constexpr int tD1D = T_D1D ? T_D1D : HCURL_MAX_D1D;
      constexpr int tQ1D = T_Q1D ? T_Q1D : HCURL_MAX_Q1D;

      MFEM_SHARED double sBo[tQ1D][tD1D];
      MFEM_SHARED double sBc[tQ1D][tD1D];

      double op9[9];
      MFEM_SHARED double sop[9*tQ1D*tQ1D];
      MFEM_SHARED double mass[tQ1D][tQ1D][3];

      MFEM_SHARED double sX[tD1D][tD1D][tD1D];

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
                     double u = 0.0;

                     for (int dz = 0; dz < D1Dz; ++dz)
                     {
                        const double wz = (c == 2) ? sBo[qz][dz] : sBc[qz][dz];
                        for (int dy = 0; dy < D1Dy; ++dy)
                        {
                           const double wy = (c == 1) ? sBo[qy][dy] : sBc[qy][dy];
                           for (int dx = 0; dx < D1Dx; ++dx)
                           {
                              const double t = sX[dz][dy][dx];
                              const double wx = (c == 0) ? sBo[qx][dx] : sBc[qx][dx];
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

            double dxyz = 0.0;

            MFEM_FOREACH_THREAD(dz,z,D1Dz)
            {
               const double wz = (c == 2) ? sBo[qz][dz] : sBc[qz][dz];

               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const double wy = (c == 1) ? sBo[qy][dy] : sBc[qy][dy];
                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const int os = (dataSize*qx) + (dataSize*Q1D*qy);
                           const int id1 = os + ((c == 0) ? 0 : ((c == 1) ? (symmetric ? 1 : 3) :
                                                                 (symmetric ? 2 : 6))); // O11, O21, O31
                           const int id2 = os + ((c == 0) ? 1 : ((c == 1) ? (symmetric ? 3 : 4) :
                                                                 (symmetric ? 4 : 7))); // O12, O22, O32
                           const int id3 = os + ((c == 0) ? 2 : ((c == 1) ? (symmetric ? 4 : 5) :
                                                                 (symmetric ? 5 : 8))); // O13, O23, O33

                           const double m_c = (sop[id1] * mass[qy][qx][0]) + (sop[id2] * mass[qy][qx][1]) +
                                              (sop[id3] * mass[qy][qx][2]);

                           const double wx = (c == 0) ? sBo[qx][dx] : sBc[qx][dx];
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

// PA H(curl) curl-curl assemble 2D kernel
static void PACurlCurlSetup2D(const int Q1D,
                              const int NE,
                              const Array<double> &w,
                              const Vector &j,
                              Vector &coeff,
                              Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto C = Reshape(coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double detJ = (J11*J22)-(J21*J12);
         y(q,e) = W[q] * C(q,e) / detJ;
      }
   });
}

// PA H(curl) curl-curl assemble 3D kernel
static void PACurlCurlSetup3D(const int Q1D,
                              const int coeffDim,
                              const int NE,
                              const Array<double> &w,
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

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);

         const double c_detJ = W[q] / detJ;

         if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
         {
            // Set y to the 6 or 9 entries of J^T M J / det
            const double M11 = C(0, q, e);
            const double M12 = C(1, q, e);
            const double M13 = C(2, q, e);
            const double M21 = (!symmetric) ? C(3, q, e) : M12;
            const double M22 = (!symmetric) ? C(4, q, e) : C(3, q, e);
            const double M23 = (!symmetric) ? C(5, q, e) : C(4, q, e);
            const double M31 = (!symmetric) ? C(6, q, e) : M13;
            const double M32 = (!symmetric) ? C(7, q, e) : M23;
            const double M33 = (!symmetric) ? C(8, q, e) : C(5, q, e);

            // First compute R = MJ
            const double R11 = M11*J11 + M12*J21 + M13*J31;
            const double R12 = M11*J12 + M12*J22 + M13*J32;
            const double R13 = M11*J13 + M12*J23 + M13*J33;
            const double R21 = M21*J11 + M22*J21 + M23*J31;
            const double R22 = M21*J12 + M22*J22 + M23*J32;
            const double R23 = M21*J13 + M22*J23 + M23*J33;
            const double R31 = M31*J11 + M32*J21 + M33*J31;
            const double R32 = M31*J12 + M32*J22 + M33*J32;
            const double R33 = M31*J13 + M32*J23 + M33*J33;

            // Now set y to J^T R / det
            y(q,0,e) = c_detJ * (J11*R11 + J21*R21 + J31*R31); // 1,1
            const double Y12 = c_detJ * (J11*R12 + J21*R22 + J31*R32);
            y(q,1,e) = Y12; // 1,2
            y(q,2,e) = c_detJ * (J11*R13 + J21*R23 + J31*R33); // 1,3

            const double Y21 = c_detJ * (J12*R11 + J22*R21 + J32*R31);
            const double Y22 = c_detJ * (J12*R12 + J22*R22 + J32*R32);
            const double Y23 = c_detJ * (J12*R13 + J22*R23 + J32*R33);

            const double Y33 = c_detJ * (J13*R13 + J23*R23 + J33*R33);

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
            const double D1 = C(0, q, e);
            const double D2 = coeffDim == 3 ? C(1, q, e) : D1;
            const double D3 = coeffDim == 3 ? C(2, q, e) : D1;

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

void CurlCurlIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetFE(0);

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*el, *el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   const int dimc = (dim == 3) ? 3 : 1;

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   const int MQsymmDim = MQ ? (MQ->GetWidth() * (MQ->GetWidth() + 1)) / 2 : 0;
   const int MQfullDim = MQ ? (MQ->GetHeight() * MQ->GetWidth()) : 0;
   const int MQdim = MQ ? (MQ->IsSymmetric() ? MQsymmDim : MQfullDim) : 0;
   const int coeffDim = MQ ? MQdim : (DQ ? DQ->GetVDim() : 1);

   symmetric = MQ ? MQ->IsSymmetric() : true;

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int ndata = (dim == 2) ? 1 : (symmetric ? symmDims : MQfullDim);
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   Vector coeff(coeffDim * ne * nq);
   coeff = 1.0;
   auto coeffh = Reshape(coeff.HostWrite(), coeffDim, nq, ne);
   if (Q || DQ || MQ)
   {
      Vector D(DQ ? coeffDim : 0);
      DenseMatrix M;
      Vector Msymm;
      if (MQ)
      {
         if (symmetric)
         {
            Msymm.SetSize(MQsymmDim);
         }
         else
         {
            M.SetSize(dimc);
         }
      }

      if (DQ)
      {
         MFEM_VERIFY(coeffDim == dimc, "");
      }
      if (MQ)
      {
         MFEM_VERIFY(coeffDim == MQdim, "");
         MFEM_VERIFY(MQ->GetHeight() == dimc && MQ->GetWidth() == dimc, "");
      }

      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            if (MQ)
            {
               if (MQ->IsSymmetric())
               {
                  MQ->EvalSymmetric(Msymm, *tr, ir->IntPoint(p));

                  for (int i=0; i<MQsymmDim; ++i)
                  {
                     coeffh(i, p, e) = Msymm[i];
                  }
               }
               else
               {
                  MQ->Eval(M, *tr, ir->IntPoint(p));

                  for (int i=0; i<dimc; ++i)
                     for (int j=0; j<dimc; ++j)
                     {
                        coeffh(j+(i*dimc), p, e) = M(i,j);
                     }
               }
            }
            else if (DQ)
            {
               DQ->Eval(D, *tr, ir->IntPoint(p));
               for (int i=0; i<coeffDim; ++i)
               {
                  coeffh(i, p, e) = D[i];
               }
            }
            else
            {
               coeffh(0, p, e) = Q->Eval(*tr, ir->IntPoint(p));
            }
         }
      }
   }

   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }

   if (dim == 3)
   {
      PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J, coeff,
                        pa_data);
   }
   else
   {
      PACurlCurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J, coeff, pa_data);
   }
}

static void PACurlCurlApply2D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &bo,
                              const Array<double> &bot,
                              const Array<double> &gc,
                              const Array<double> &gct,
                              const Vector &pa_data,
                              const Vector &x,
                              Vector &y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 2*(D1D-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D];

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
            double gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = X(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 0) ? -Gc(qy,dy) : Bo(qy,dy);
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

            double gradX[MAX_D1D];
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
               const double wy = (c == 0) ? -Gct(dy,qy) : Bot(dy,qy);

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

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PACurlCurlApply3D(const int D1D,
                              const int Q1D,
                              const bool symmetric,
                              const int NE,
                              const Array<double> &bo,
                              const Array<double> &bc,
                              const Array<double> &bot,
                              const Array<double> &bct,
                              const Array<double> &gc,
                              const Array<double> &gct,
                              const Vector &pa_data,
                              const Vector &x,
                              Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
   // (\nabla\times u) \cdot (\nabla\times v) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, (symmetric ? 6 : 9), NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
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
            double gradXY[MAX_Q1D][MAX_Q1D][2];
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
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
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
            double gradXY[MAX_Q1D][MAX_Q1D][2];
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
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
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
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
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
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

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
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : op(qx,qy,qz,3,e);
               const double O22 = symmetric ? op(qx,qy,qz,3,e) : op(qx,qy,qz,4,e);
               const double O23 = symmetric ? op(qx,qy,qz,4,e) : op(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : op(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : op(qx,qy,qz,7,e);
               const double O33 = symmetric ? op(qx,qy,qz,5,e) : op(qx,qy,qz,8,e);

               const double c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const double c2 = (O21 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const double c3 = (O31 * curl[qz][qy][qx][0]) + (O32 * curl[qz][qy][qx][1]) +
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
            double gradXY12[MAX_D1D][MAX_D1D];
            double gradXY21[MAX_D1D][MAX_D1D];

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
               double massX[MAX_D1D][2];
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
                     const double wx = Bot(dx,qx);

                     massX[dx][0] += wx * curl[qz][qy][qx][1];
                     massX[dx][1] += wx * curl[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
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
            double gradXY02[MAX_D1D][MAX_D1D];
            double gradXY20[MAX_D1D][MAX_D1D];

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
               double massY[MAX_D1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = Bot(dy,qy);

                     massY[dy][0] += wy * curl[qz][qy][qx][2];
                     massY[dy][1] += wy * curl[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double wx = Bct(dx,qx);
                  const double wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
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
            double gradYZ01[MAX_D1D][MAX_D1D];
            double gradYZ10[MAX_D1D][MAX_D1D];

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
               double massZ[MAX_D1D][2];
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
                     const double wz = Bot(dz,qz);

                     massZ[dz][0] += wz * curl[qz][qy][qx][0];
                     massZ[dz][1] += wz * curl[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double wx = Bct(dx,qx);
               const double wDx = Gct(dx,qx);

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

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void SmemPACurlCurlApply3D(const int D1D,
                                  const int Q1D,
                                  const bool symmetric,
                                  const int NE,
                                  const Array<double> &bo,
                                  const Array<double> &bc,
                                  const Array<double> &bot,
                                  const Array<double> &bct,
                                  const Array<double> &gc,
                                  const Array<double> &gct,
                                  const Vector &pa_data,
                                  const Vector &x,
                                  Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
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

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int VDIM = 3;

      MFEM_SHARED double sBo[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sBc[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sGc[MAX_D1D][MAX_Q1D];

      double ope[9];
      MFEM_SHARED double sop[9][MAX_Q1D][MAX_Q1D];
      MFEM_SHARED double curl[MAX_Q1D][MAX_Q1D][3];

      MFEM_SHARED double sX[MAX_D1D][MAX_D1D][MAX_D1D];

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
                     double u = 0.0;
                     double v = 0.0;

                     // We treat x, y, z components separately for optimization specific to each.
                     if (c == 0) // x component
                     {
                        // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const double wz = sBc[dz][qz];
                           const double wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBc[dy][qy];
                              const double wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double wx = sX[dz][dy][dx] * sBo[dx][qx];
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
                           const double wz = sBc[dz][qz];
                           const double wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBo[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double t = sX[dz][dy][dx];
                                 const double wx = t * sBc[dx][qx];
                                 const double wDx = t * sGc[dx][qx];

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
                           const double wz = sBo[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBc[dy][qy];
                              const double wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double t = sX[dz][dy][dx];
                                 const double wx = t * sBc[dx][qx];
                                 const double wDx = t * sGc[dx][qx];

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

         double dxyz1 = 0.0;
         double dxyz2 = 0.0;
         double dxyz3 = 0.0;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            const double wcz = sBc[dz][qz];
            const double wcDz = sGc[dz][qz];
            const double wz = (dz < D1D-1) ? sBo[dz][qz] : 0.0;

            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wcy = sBc[dy][qy];
                     const double wcDy = sGc[dy][qy];
                     const double wy = (dy < D1D-1) ? sBo[dy][qy] : 0.0;

                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const double O11 = sop[0][qx][qy];
                        const double O12 = sop[1][qx][qy];
                        const double O13 = sop[2][qx][qy];
                        const double O21 = symmetric ? O12 : sop[3][qx][qy];
                        const double O22 = symmetric ? sop[3][qx][qy] : sop[4][qx][qy];
                        const double O23 = symmetric ? sop[4][qx][qy] : sop[5][qx][qy];
                        const double O31 = symmetric ? O13 : sop[6][qx][qy];
                        const double O32 = symmetric ? O23 : sop[7][qx][qy];
                        const double O33 = symmetric ? sop[5][qx][qy] : sop[8][qx][qy];

                        const double c1 = (O11 * curl[qy][qx][0]) + (O12 * curl[qy][qx][1]) +
                                          (O13 * curl[qy][qx][2]);
                        const double c2 = (O21 * curl[qy][qx][0]) + (O22 * curl[qy][qx][1]) +
                                          (O23 * curl[qy][qx][2]);
                        const double c3 = (O31 * curl[qy][qx][0]) + (O32 * curl[qy][qx][1]) +
                                          (O33 * curl[qy][qx][2]);

                        const double wcx = sBc[dx][qx];
                        const double wDx = sGc[dx][qx];

                        if (dx < D1D-1)
                        {
                           // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                           // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                           const double wx = sBo[dx][qx];
                           dxyz1 += (wx * c2 * wcy * wcDz) - (wx * c3 * wcDy * wcz);
                        }

                        // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                        // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                        dxyz2 += (-wy * c1 * wcx * wcDz) + (wy * c3 * wDx * wcz);

                        // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                        // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
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
   }); // end of element loop
}

void CurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23: return SmemPACurlCurlApply3D<2,3>(dofs1D, quad1D, symmetric, ne,
                                                            mapsO->B, mapsC->B, mapsO->Bt,
                                                            mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y);
            case 0x34: return SmemPACurlCurlApply3D<3,4>(dofs1D, quad1D, symmetric, ne,
                                                            mapsO->B, mapsC->B, mapsO->Bt,
                                                            mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y);
            case 0x45: return SmemPACurlCurlApply3D<4,5>(dofs1D, quad1D, symmetric, ne,
                                                            mapsO->B,
                                                            mapsC->B, mapsO->Bt,
                                                            mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y);
            case 0x56: return SmemPACurlCurlApply3D<5,6>(dofs1D, quad1D, symmetric, ne,
                                                            mapsO->B, mapsC->B, mapsO->Bt,
                                                            mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y);
            default: return SmemPACurlCurlApply3D(dofs1D, quad1D, symmetric, ne, mapsO->B,
                                                     mapsC->B, mapsO->Bt, mapsC->Bt,
                                                     mapsC->G, mapsC->Gt, pa_data, x, y);
         }
      }
      else
         PACurlCurlApply3D(dofs1D, quad1D, symmetric, ne, mapsO->B, mapsC->B, mapsO->Bt,
                           mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y);
   }
   else if (dim == 2)
   {
      PACurlCurlApply2D(dofs1D, quad1D, ne, mapsO->B, mapsO->Bt,
                        mapsC->G, mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

static void PACurlCurlAssembleDiagonal2D(const int D1D,
                                         const int Q1D,
                                         const int NE,
                                         const Array<double> &bo,
                                         const Array<double> &gc,
                                         const Vector &pa_data,
                                         Vector &diag)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, NE);
   auto D = Reshape(diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double t[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               t[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : -Gc(qy,dy);
                  t[qx] += wy * wy * op(qx,qy,e);
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
                  D(dx + (dy * D1Dx) + osc, e) += t[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PACurlCurlAssembleDiagonal3D(const int D1D,
                                         const int Q1D,
                                         const bool symmetric,
                                         const int NE,
                                         const Array<double> &bo,
                                         const Array<double> &bc,
                                         const Array<double> &go,
                                         const Array<double> &gc,
                                         const Vector &pa_data,
                                         Vector &diag)
{
   constexpr static int VDIM = 3;
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

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

   MFEM_FORALL(e, NE,
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
      // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      // For each c, we will keep 9 arrays for derivatives multiplied by the 9 entries of the 3x3 matrix (dF^T C dF),
      // which may be non-symmetric depending on a possibly non-symmetric matrix coefficient.

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double zt[MAX_Q1D][MAX_Q1D][MAX_D1D][9][3];

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
                     const double wz = ((c == 2) ? Bo(qz,dz) : Bc(qz,dz));
                     const double wDz = ((c == 2) ? Go(qz,dz) : Gc(qz,dz));

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

         double yt[MAX_Q1D][MAX_D1D][MAX_D1D][9][3][3];

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
                     const double wy = ((c == 1) ? Bo(qy,dy) : Bc(qy,dy));
                     const double wDy = ((c == 1) ? Go(qy,dy) : Gc(qy,dy));

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
                     const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     const double wDx = ((c == 0) ? Go(qx,dx) : Gc(qx,dx));

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
                        const double sumy = yt[qx][dy][dz][i22][2][0] - yt[qx][dy][dz][i23][1][1]
                                            - yt[qx][dy][dz][i32][1][1] + yt[qx][dy][dz][i33][0][2];

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += sumy * wx * wx;
                     }
                     else if (c == 1)
                     {
                        // (u_1)_{x_2} (O11 (u_1)_{x_2} - O13 (u_1)_{x_0}) + (u_1)_{x_0} (-O31 (u_1)_{x_2} + O33 (u_1)_{x_0})
                        const double d = (yt[qx][dy][dz][i11][2][0] * wx * wx)
                                         - ((yt[qx][dy][dz][i13][1][0] + yt[qx][dy][dz][i31][1][0]) * wDx * wx)
                                         + (yt[qx][dy][dz][i33][0][0] * wDx * wDx);

                        D(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += d;
                     }
                     else
                     {
                        // (u_2)_{x_1} (O11 (u_2)_{x_1} - O12 (u_2)_{x_0}) - (u_2)_{x_0} (O21 (u_2)_{x_1} - O22 (u_2)_{x_0})
                        const double d = (yt[qx][dy][dz][i11][0][2] * wx * wx)
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

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void SmemPACurlCurlAssembleDiagonal3D(const int D1D,
                                             const int Q1D,
                                             const bool symmetric,
                                             const int NE,
                                             const Array<double> &bo,
                                             const Array<double> &bc,
                                             const Array<double> &go,
                                             const Array<double> &gc,
                                             const Vector &pa_data,
                                             Vector &diag)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

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

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
      // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      constexpr int VDIM = 3;

      MFEM_SHARED double sBo[MAX_Q1D][MAX_D1D];
      MFEM_SHARED double sBc[MAX_Q1D][MAX_D1D];
      MFEM_SHARED double sGo[MAX_Q1D][MAX_D1D];
      MFEM_SHARED double sGc[MAX_Q1D][MAX_D1D];

      double ope[9];
      MFEM_SHARED double sop[9][MAX_Q1D][MAX_Q1D];

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

         double dxyz = 0.0;

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
               const double wz = ((c == 2) ? sBo[qz][dz] : sBc[qz][dz]);
               const double wDz = ((c == 2) ? sGo[qz][dz] : sGc[qz][dz]);

               MFEM_FOREACH_THREAD(dy,y,D1Dy)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1Dx)
                  {
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const double wy = ((c == 1) ? sBo[qy][dy] : sBc[qy][dy]);
                        const double wDy = ((c == 1) ? sGo[qy][dy] : sGc[qy][dy]);

                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const double wx = ((c == 0) ? sBo[qx][dx] : sBc[qx][dx]);
                           const double wDx = ((c == 0) ? sGo[qx][dx] : sGc[qx][dx]);

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

void CurlCurlIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23: return SmemPACurlCurlAssembleDiagonal3D<2,3>(dofs1D, quad1D,
                                                                       symmetric, ne,
                                                                       mapsO->B, mapsC->B,
                                                                       mapsO->G, mapsC->G,
                                                                       pa_data, diag);
            case 0x34: return SmemPACurlCurlAssembleDiagonal3D<3,4>(dofs1D, quad1D,
                                                                       symmetric, ne,
                                                                       mapsO->B, mapsC->B,
                                                                       mapsO->G, mapsC->G,
                                                                       pa_data, diag);
            case 0x45: return SmemPACurlCurlAssembleDiagonal3D<4,5>(dofs1D, quad1D,
                                                                       symmetric, ne,
                                                                       mapsO->B, mapsC->B,
                                                                       mapsO->G, mapsC->G,
                                                                       pa_data, diag);
            case 0x56: return SmemPACurlCurlAssembleDiagonal3D<5,6>(dofs1D, quad1D,
                                                                       symmetric, ne,
                                                                       mapsO->B, mapsC->B,
                                                                       mapsO->G, mapsC->G,
                                                                       pa_data, diag);
            default: return SmemPACurlCurlAssembleDiagonal3D(dofs1D, quad1D, symmetric, ne,
                                                                mapsO->B, mapsC->B,
                                                                mapsO->G, mapsC->G,
                                                                pa_data, diag);
         }
      }
      else
         PACurlCurlAssembleDiagonal3D(dofs1D, quad1D, symmetric, ne,
                                      mapsO->B, mapsC->B,
                                      mapsO->G, mapsC->G,
                                      pa_data, diag);
   }
   else if (dim == 2)
   {
      PACurlCurlAssembleDiagonal2D(dofs1D, quad1D, ne,
                                   mapsO->B, mapsC->G, pa_data, diag);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

// Apply to x corresponding to DOF's in H^1 (trial), whose gradients are
// integrated against H(curl) test functions corresponding to y.
void PAHcurlH1Apply3D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &bc,
                      const Array<double> &gc,
                      const Array<double> &bot,
                      const Array<double> &bct,
                      const Vector &pa_data,
                      const Vector &x,
                      Vector &y)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

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

      for (int dz = 0; dz < D1D; ++dz)
      {
         double gradXY[MAX_Q1D][MAX_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double gradX[MAX_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = X(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * Bc(qx,dx);
                  gradX[qx][1] += s * Gc(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = Bc(qy,dy);
               const double wDy = Gc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx * wDy;
                  gradXY[qy][qx][2] += wx * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = Bc(qz,dz);
            const double wDz = Gc(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  mass[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  mass[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
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
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

         int osc = 0;

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
               double massX[MAX_D1D];
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
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
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

// Apply to x corresponding to DOF's in H^1 (trial), whose gradients are
// integrated against H(curl) test functions corresponding to y.
void PAHcurlH1Apply2D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &bc,
                      const Array<double> &gc,
                      const Array<double> &bot,
                      const Array<double> &bct,
                      const Vector &pa_data,
                      const Vector &x,
                      Vector &y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, 3, NE);
   auto X = Reshape(x.Read(), D1D, D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][VDIM];

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

      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[MAX_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = X(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * Bc(qx,dx);
               gradX[qx][1] += s * Gc(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = Bc(qy,dy);
            const double wDy = Gc(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx  = gradX[qx][0];
               const double wDx = gradX[qx][1];
               mass[qy][qx][0] += wDx * wy;
               mass[qy][qx][1] += wx * wDy;
            }
         }
      }

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O11 = op(qx,qy,0,e);
            const double O12 = op(qx,qy,1,e);
            const double O22 = op(qx,qy,2,e);
            const double massX = mass[qy][qx][0];
            const double massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O12*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         int osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            double massX[MAX_D1D];
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
               const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  Y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }
   }); // end of element loop
}

// PA H(curl) assemble kernel
void PAHcurlL2Setup(const int NQ,
                    const int coeffDim,
                    const int NE,
                    const Array<double> &w,
                    Vector &coeff,
                    Vector &op)
{
   auto W = w.Read();
   auto C = Reshape(coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), coeffDim, NQ, NE);

   MFEM_FORALL(e, NE,
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

void MixedVectorCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   mapsCtest = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsOtest = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;
   dofs1Dtest = mapsCtest->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   coeffDim = (DQ ? 3 : 1);

   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());

   Vector coeff(coeffDim * nq * ne);
   coeff = 1.0;
   auto coeffh = Reshape(coeff.HostWrite(), coeffDim, nq, ne);
   if (Q || DQ)
   {
      Vector V(coeffDim);
      if (DQ)
      {
         MFEM_VERIFY(DQ->GetVDim() == coeffDim, "");
      }

      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);

         for (int p=0; p<nq; ++p)
         {
            if (DQ)
            {
               DQ->Eval(V, *tr, ir->IntPoint(p));
               for (int i=0; i<coeffDim; ++i)
               {
                  coeffh(i, p, e) = V[i];
               }
            }
            else
            {
               coeffh(0, p, e) = Q->Eval(*tr, ir->IntPoint(p));
            }
         }
      }
   }

   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlL2Setup(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3 &&
            test_fel->GetOrder() == trial_fel->GetOrder())
   {
      PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J, coeff,
                        pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

// Apply to x corresponding to DOF's in H(curl) (trial), whose curl is
// integrated against H(curl) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlL2Apply3D(const int D1D,
                             const int Q1D,
                             const int coeffDim,
                             const int NE,
                             const Array<double> &bo,
                             const Array<double> &bc,
                             const Array<double> &bot,
                             const Array<double> &bct,
                             const Array<double> &gc,
                             const Vector &pa_data,
                             const Vector &x,
                             Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using u = dF^{-T} \hat{u} and (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
   // (\nabla\times u) \cdot v = 1/det(dF) \hat{\nabla}\times\hat{u}^T dF^T dF^{-T} \hat{v}
   // = 1/det(dF) \hat{\nabla}\times\hat{u}^T \hat{v}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
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
            double gradXY[MAX_Q1D][MAX_Q1D][2];
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
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
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
            double gradXY[MAX_Q1D][MAX_Q1D][2];
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
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
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
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
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
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

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
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] *= op(coeffDim == 3 ? c : 0, qx,qy,qz,e);
               }
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

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
               double massX[MAX_D1D];
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
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
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

// Apply to x corresponding to DOF's in H(curl) (trial), whose curl is
// integrated against H(curl) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void SmemPAHcurlL2Apply3D(const int D1D,
                                 const int Q1D,
                                 const int coeffDim,
                                 const int NE,
                                 const Array<double> &bo,
                                 const Array<double> &bc,
                                 const Array<double> &gc,
                                 const Vector &pa_data,
                                 const Vector &x,
                                 Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int VDIM = 3;
      constexpr int maxCoeffDim = 3;

      MFEM_SHARED double sBo[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sBc[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sGc[MAX_D1D][MAX_Q1D];

      double opc[maxCoeffDim];
      MFEM_SHARED double sop[maxCoeffDim][MAX_Q1D][MAX_Q1D];
      MFEM_SHARED double curl[MAX_Q1D][MAX_Q1D][3];

      MFEM_SHARED double sX[MAX_D1D][MAX_D1D][MAX_D1D];

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
                     double u = 0.0;
                     double v = 0.0;

                     // We treat x, y, z components separately for optimization specific to each.
                     if (c == 0) // x component
                     {
                        // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]

                        for (int dz = 0; dz < D1Dz; ++dz)
                        {
                           const double wz = sBc[dz][qz];
                           const double wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBc[dy][qy];
                              const double wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double wx = sX[dz][dy][dx] * sBo[dx][qx];
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
                           const double wz = sBc[dz][qz];
                           const double wDz = sGc[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBo[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double t = sX[dz][dy][dx];
                                 const double wx = t * sBc[dx][qx];
                                 const double wDx = t * sGc[dx][qx];

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
                           const double wz = sBo[dz][qz];

                           for (int dy = 0; dy < D1Dy; ++dy)
                           {
                              const double wy = sBc[dy][qy];
                              const double wDy = sGc[dy][qy];

                              for (int dx = 0; dx < D1Dx; ++dx)
                              {
                                 const double t = sX[dz][dy][dx];
                                 const double wx = t * sBc[dx][qx];
                                 const double wDx = t * sGc[dx][qx];

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

         double dxyz1 = 0.0;
         double dxyz2 = 0.0;
         double dxyz3 = 0.0;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            const double wcz = sBc[dz][qz];
            const double wz = (dz < D1D-1) ? sBo[dz][qz] : 0.0;

            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wcy = sBc[dy][qy];
                     const double wy = (dy < D1D-1) ? sBo[dy][qy] : 0.0;

                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const double O1 = sop[0][qx][qy];
                        const double O2 = (coeffDim == 3) ? sop[1][qx][qy] : O1;
                        const double O3 = (coeffDim == 3) ? sop[2][qx][qy] : O1;

                        const double c1 = O1 * curl[qy][qx][0];
                        const double c2 = O2 * curl[qy][qx][1];
                        const double c3 = O3 * curl[qy][qx][2];

                        const double wcx = sBc[dx][qx];

                        if (dx < D1D-1)
                        {
                           const double wx = sBo[dx][qx];
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
   }); // end of element loop
}

// Apply to x corresponding to DOF's in H(curl) (trial), whose curl is
// integrated against H(div) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlHdivApply3D(const int D1D,
                               const int D1Dtest,
                               const int Q1D,
                               const int NE,
                               const Array<double> &bo,
                               const Array<double> &bc,
                               const Array<double> &bot,
                               const Array<double> &bct,
                               const Array<double> &gc,
                               const Vector &pa_data,
                               const Vector &x,
                               Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
   // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
   // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1Dtest, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1Dtest-1)*(D1Dtest-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
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
            double gradXY[MAX_Q1D][MAX_Q1D][2];
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
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
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
            double gradXY[MAX_Q1D][MAX_Q1D][2];
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
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
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
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
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
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

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
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);

               const double c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const double c2 = (O12 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const double c3 = (O13 * curl[qz][qy][qx][0]) + (O23 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[HCURL_MAX_D1D][HCURL_MAX_D1D];  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D

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
                  massXY[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[HCURL_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0.0;
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
                  const double wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bct(dz,qz) : Bot(dz,qz);
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

void MixedVectorCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23: return SmemPAHcurlL2Apply3D<2,3>(dofs1D, quad1D, coeffDim, ne,
                                                           mapsO->B, mapsC->B,
                                                           mapsC->G, pa_data, x, y);
            case 0x34: return SmemPAHcurlL2Apply3D<3,4>(dofs1D, quad1D, coeffDim, ne,
                                                           mapsO->B, mapsC->B,
                                                           mapsC->G, pa_data, x, y);
            case 0x45: return SmemPAHcurlL2Apply3D<4,5>(dofs1D, quad1D, coeffDim, ne,
                                                           mapsO->B, mapsC->B,
                                                           mapsC->G, pa_data, x, y);
            case 0x56: return SmemPAHcurlL2Apply3D<5,6>(dofs1D, quad1D, coeffDim, ne,
                                                           mapsO->B, mapsC->B,
                                                           mapsC->G, pa_data, x, y);
            default: return SmemPAHcurlL2Apply3D(dofs1D, quad1D, coeffDim, ne, mapsO->B,
                                                    mapsC->B,
                                                    mapsC->G, pa_data, x, y);
         }
      }
      else
         PAHcurlL2Apply3D(dofs1D, quad1D, coeffDim, ne, mapsO->B, mapsC->B,
                          mapsO->Bt, mapsC->Bt, mapsC->G, pa_data, x, y);
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3)
      PAHcurlHdivApply3D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                         mapsC->B, mapsOtest->Bt, mapsCtest->Bt, mapsC->G,
                         pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   coeffDim = DQ ? 3 : 1;

   pa_data.SetSize(coeffDim * nq * ne, Device::GetMemoryType());

   Vector coeff(coeffDim * nq * ne);
   coeff = 1.0;
   auto coeffh = Reshape(coeff.HostWrite(), coeffDim, nq, ne);
   if (Q || DQ)
   {
      Vector V(coeffDim);
      if (DQ)
      {
         MFEM_VERIFY(DQ->GetVDim() == coeffDim, "");
      }

      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);

         for (int p=0; p<nq; ++p)
         {
            if (DQ)
            {
               DQ->Eval(V, *tr, ir->IntPoint(p));
               for (int i=0; i<coeffDim; ++i)
               {
                  coeffh(i, p, e) = V[i];
               }
            }
            else
            {
               coeffh(0, p, e) = Q->Eval(*tr, ir->IntPoint(p));
            }
         }
      }
   }

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   if (trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlL2Setup(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

// Apply to x corresponding to DOF's in H(curl) (trial), integrated against curl
// of H(curl) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlL2Apply3DTranspose(const int D1D,
                                      const int Q1D,
                                      const int coeffDim,
                                      const int NE,
                                      const Array<double> &bo,
                                      const Array<double> &bc,
                                      const Array<double> &bot,
                                      const Array<double> &bct,
                                      const Array<double> &gct,
                                      const Vector &pa_data,
                                      const Vector &x,
                                      Vector &y)
{
   // See PAHcurlL2Apply3D for comments.

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1D, Q1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

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
            double massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
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
               for (int c=0; c<VDIM; ++c)
               {
                  mass[qz][qy][qx][c] *= op(coeffDim == 3 ? c : 0, qx,qy,qz,e);
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
            double gradXY12[MAX_D1D][MAX_D1D];
            double gradXY21[MAX_D1D][MAX_D1D];

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
               double massX[MAX_D1D][2];
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
                     const double wx = Bot(dx,qx);

                     massX[dx][0] += wx * mass[qz][qy][qx][1];
                     massX[dx][1] += wx * mass[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
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
            double gradXY02[MAX_D1D][MAX_D1D];
            double gradXY20[MAX_D1D][MAX_D1D];

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
               double massY[MAX_D1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = Bot(dy,qy);

                     massY[dy][0] += wy * mass[qz][qy][qx][2];
                     massY[dy][1] += wy * mass[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double wx = Bct(dx,qx);
                  const double wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
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
            double gradYZ01[MAX_D1D][MAX_D1D];
            double gradYZ10[MAX_D1D][MAX_D1D];

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
               double massZ[MAX_D1D][2];
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
                     const double wz = Bot(dz,qz);

                     massZ[dz][0] += wz * mass[qz][qy][qx][0];
                     massZ[dz][1] += wz * mass[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double wx = Bct(dx,qx);
               const double wDx = Gct(dx,qx);

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

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void SmemPAHcurlL2Apply3DTranspose(const int D1D,
                                          const int Q1D,
                                          const int coeffDim,
                                          const int NE,
                                          const Array<double> &bo,
                                          const Array<double> &bc,
                                          const Array<double> &gc,
                                          const Vector &pa_data,
                                          const Vector &x,
                                          Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int VDIM = 3;
      constexpr int maxCoeffDim = 3;

      MFEM_SHARED double sBo[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sBc[MAX_D1D][MAX_Q1D];
      MFEM_SHARED double sGc[MAX_D1D][MAX_Q1D];

      double opc[maxCoeffDim];
      MFEM_SHARED double sop[maxCoeffDim][MAX_Q1D][MAX_Q1D];
      MFEM_SHARED double mass[MAX_Q1D][MAX_Q1D][3];

      MFEM_SHARED double sX[MAX_D1D][MAX_D1D][MAX_D1D];

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
                     double u = 0.0;

                     for (int dz = 0; dz < D1Dz; ++dz)
                     {
                        const double wz = (c == 2) ? sBo[dz][qz] : sBc[dz][qz];

                        for (int dy = 0; dy < D1Dy; ++dy)
                        {
                           const double wy = (c == 1) ? sBo[dy][qy] : sBc[dy][qy];

                           for (int dx = 0; dx < D1Dx; ++dx)
                           {
                              const double wx = sX[dz][dy][dx] * ((c == 0) ? sBo[dx][qx] : sBc[dx][qx]);
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

         double dxyz1 = 0.0;
         double dxyz2 = 0.0;
         double dxyz3 = 0.0;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            const double wcz = sBc[dz][qz];
            const double wcDz = sGc[dz][qz];
            const double wz = (dz < D1D-1) ? sBo[dz][qz] : 0.0;

            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wcy = sBc[dy][qy];
                     const double wcDy = sGc[dy][qy];
                     const double wy = (dy < D1D-1) ? sBo[dy][qy] : 0.0;

                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const double O1 = sop[0][qx][qy];
                        const double O2 = (coeffDim == 3) ? sop[1][qx][qy] : O1;
                        const double O3 = (coeffDim == 3) ? sop[2][qx][qy] : O1;

                        const double c1 = O1 * mass[qy][qx][0];
                        const double c2 = O2 * mass[qy][qx][1];
                        const double c3 = O3 * mass[qy][qx][2];

                        const double wcx = sBc[dx][qx];
                        const double wDx = sGc[dx][qx];

                        if (dx < D1D-1)
                        {
                           const double wx = sBo[dx][qx];
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
   }); // end of element loop
}

void MixedVectorWeakCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23: return SmemPAHcurlL2Apply3DTranspose<2,3>(dofs1D, quad1D, coeffDim,
                                                                    ne, mapsO->B, mapsC->B,
                                                                    mapsC->G, pa_data, x, y);
            case 0x34: return SmemPAHcurlL2Apply3DTranspose<3,4>(dofs1D, quad1D, coeffDim,
                                                                    ne, mapsO->B, mapsC->B,
                                                                    mapsC->G, pa_data, x, y);
            case 0x45: return SmemPAHcurlL2Apply3DTranspose<4,5>(dofs1D, quad1D, coeffDim,
                                                                    ne, mapsO->B, mapsC->B,
                                                                    mapsC->G, pa_data, x, y);
            case 0x56: return SmemPAHcurlL2Apply3DTranspose<5,6>(dofs1D, quad1D, coeffDim,
                                                                    ne, mapsO->B, mapsC->B,
                                                                    mapsC->G, pa_data, x, y);
            default: return SmemPAHcurlL2Apply3DTranspose(dofs1D, quad1D, coeffDim, ne,
                                                             mapsO->B, mapsC->B,
                                                             mapsC->G, pa_data, x, y);
         }
      }
      else
         PAHcurlL2Apply3DTranspose(dofs1D, quad1D, coeffDim, ne, mapsO->B, mapsC->B,
                                   mapsO->Bt, mapsC->Bt, mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

template void SmemPAHcurlMassAssembleDiagonal3D<0,0>(const int D1D,
                                                     const int Q1D,
                                                     const int NE,
                                                     const bool symmetric,
                                                     const Array<double> &bo,
                                                     const Array<double> &bc,
                                                     const Vector &pa_data,
                                                     Vector &diag);

template void SmemPAHcurlMassAssembleDiagonal3D<2,3>(const int D1D,
                                                     const int Q1D,
                                                     const int NE,
                                                     const bool symmetric,
                                                     const Array<double> &bo,
                                                     const Array<double> &bc,
                                                     const Vector &pa_data,
                                                     Vector &diag);

template void SmemPAHcurlMassAssembleDiagonal3D<3,4>(const int D1D,
                                                     const int Q1D,
                                                     const int NE,
                                                     const bool symmetric,
                                                     const Array<double> &bo,
                                                     const Array<double> &bc,
                                                     const Vector &pa_data,
                                                     Vector &diag);

template void SmemPAHcurlMassAssembleDiagonal3D<4,5>(const int D1D,
                                                     const int Q1D,
                                                     const int NE,
                                                     const bool symmetric,
                                                     const Array<double> &bo,
                                                     const Array<double> &bc,
                                                     const Vector &pa_data,
                                                     Vector &diag);

template void SmemPAHcurlMassAssembleDiagonal3D<5,6>(const int D1D,
                                                     const int Q1D,
                                                     const int NE,
                                                     const bool symmetric,
                                                     const Array<double> &bo,
                                                     const Array<double> &bc,
                                                     const Vector &pa_data,
                                                     Vector &diag);

template void SmemPAHcurlMassApply3D<0,0>(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const bool symmetric,
                                          const Array<double> &bo,
                                          const Array<double> &bc,
                                          const Array<double> &bot,
                                          const Array<double> &bct,
                                          const Vector &pa_data,
                                          const Vector &x,
                                          Vector &y);

template void SmemPAHcurlMassApply3D<2,3>(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const bool symmetric,
                                          const Array<double> &bo,
                                          const Array<double> &bc,
                                          const Array<double> &bot,
                                          const Array<double> &bct,
                                          const Vector &pa_data,
                                          const Vector &x,
                                          Vector &y);

template void SmemPAHcurlMassApply3D<3,4>(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const bool symmetric,
                                          const Array<double> &bo,
                                          const Array<double> &bc,
                                          const Array<double> &bot,
                                          const Array<double> &bct,
                                          const Vector &pa_data,
                                          const Vector &x,
                                          Vector &y);

template void SmemPAHcurlMassApply3D<4,5>(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const bool symmetric,
                                          const Array<double> &bo,
                                          const Array<double> &bc,
                                          const Array<double> &bot,
                                          const Array<double> &bct,
                                          const Vector &pa_data,
                                          const Vector &x,
                                          Vector &y);

template void SmemPAHcurlMassApply3D<5,6>(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const bool symmetric,
                                          const Array<double> &bo,
                                          const Array<double> &bc,
                                          const Array<double> &bot,
                                          const Array<double> &bct,
                                          const Vector &pa_data,
                                          const Vector &x,
                                          Vector &y);

} // namespace mfem
