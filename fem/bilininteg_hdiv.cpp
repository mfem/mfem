// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "libceed/mass.hpp"

using namespace std;


// Piola transformation in H(div): w = (1 / det (dF)) dF \hat{w}
// div w = (1 / det (dF)) \hat{div} \hat{w}

namespace mfem
{

// PA H(div) Mass Assemble 2D kernel
void PAHdivSetup2D(const int Q1D,
                   const int NE,
                   const Array<double> &w,
                   const Vector &j,
                   Vector &_coeff,
                   Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, 3, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double c_detJ = W[q] * coeff(q, e) / ((J11*J22)-(J21*J12));
         // (c/detJ) J^T J
         y(q,0,e) = c_detJ * (J11*J11 + J21*J21); // 1,1
         y(q,1,e) = c_detJ * (J11*J12 + J21*J22); // 1,2
         y(q,2,e) = c_detJ * (J12*J12 + J22*J22); // 2,2
      }
   });
}

// PA H(div) Mass Assemble 3D kernel
void PAHdivSetup3D(const int Q1D,
                   const int NE,
                   const Array<double> &w,
                   const Vector &j,
                   Vector &_coeff,
                   Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, 6, NE);

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
         const double c_detJ = W[q] * coeff(q, e) / detJ;
         // (c/detJ) J^T J
         y(q,0,e) = c_detJ * (J11*J11 + J21*J21 + J31*J31); // 1,1
         y(q,1,e) = c_detJ * (J12*J11 + J22*J21 + J32*J31); // 2,1
         y(q,2,e) = c_detJ * (J13*J11 + J23*J21 + J33*J31); // 3,1
         y(q,3,e) = c_detJ * (J12*J12 + J22*J22 + J32*J32); // 2,2
         y(q,4,e) = c_detJ * (J13*J12 + J23*J22 + J33*J32); // 3,2
         y(q,5,e) = c_detJ * (J13*J13 + J23*J23 + J33*J33); // 3,3
      }
   });
}

void PAHdivMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<double> &_Bo,
                       const Array<double> &_Bc,
                       const Array<double> &_Bot,
                       const Array<double> &_Bct,
                       const Vector &_op,
                       const Vector &_x,
                       Vector &_y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

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
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
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
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dx = (c == 1) ? D1D - 1 : D1D;
            const int D1Dy = (c == 0) ? D1D - 1 : D1D;

            double massX[MAX_D1D];
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
               const double wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);

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

void PAHdivMassAssembleDiagonal2D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const Array<double> &_Bo,
                                  const Array<double> &_Bc,
                                  const Vector &_op,
                                  Vector &_diag)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto diag = Reshape(_diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double mass[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  mass[qx] += wy*wy*((c == 0) ? op(qx,qy,0,e) : op(qx,qy,2,e));
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double val = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = (c == 0) ? Bc(qx,dx) : Bo(qx,dx);
                  val += mass[qx] * wx * wx;
               }
               diag(dx + (dy * D1Dx) + osc, e) += val;
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components
   }); // end of element loop
}

void PAHdivMassAssembleDiagonal3D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const Array<double> &_Bo,
                                  const Array<double> &_Bc,
                                  const Vector &_op,
                                  Vector &_diag)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto diag = Reshape(_diag.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         const int opc = (c == 0) ? 0 : ((c == 1) ? 3 : 5);

         double mass[HDIV_MAX_Q1D];

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const double wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
                        mass[qx] += wy * wy * wz * wz * op(qx,qy,qz,opc,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  double val = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = (c == 0) ? Bc(qx,dx) : Bo(qx,dx);
                     val += mass[qx] * wx * wx;
                  }
                  diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += val;
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void PAHdivMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<double> &_Bo,
                       const Array<double> &_Bc,
                       const Array<double> &_Bot,
                       const Array<double> &_Bct,
                       const Vector &_op,
                       const Vector &_x,
                       Vector &_y)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[HDIV_MAX_Q1D][HDIV_MAX_Q1D][HDIV_MAX_Q1D][VDIM];

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
            double massXY[HDIV_MAX_Q1D][HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
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
         double massXY[HDIV_MAX_D1D][HDIV_MAX_D1D];

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
               double massX[HDIV_MAX_D1D];
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

// PA H(div) div-div assemble 2D kernel
// NOTE: this is identical to PACurlCurlSetup3D
static void PADivDivSetup2D(const int Q1D,
                            const int NE,
                            const Array<double> &w,
                            const Vector &j,
                            Vector &_coeff,
                            Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
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
         y(q,e) = W[q] * coeff(q,e) / detJ;
      }
   });
}

static void PADivDivSetup3D(const int Q1D,
                            const int NE,
                            const Array<double> &w,
                            const Vector &j,
                            Vector &_coeff,
                            Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);

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
         y(q,e) = W[q] * coeff(q, e) / detJ;
      }
   });
}

static void PADivDivApply2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &_Bo,
                            const Array<double> &_Gc,
                            const Array<double> &_Bot,
                            const Array<double> &_Gct,
                            const Vector &_op,
                            const Vector &_x,
                            Vector &_y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double div[MAX_Q1D][MAX_Q1D];

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
            double gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 0) ? Bo(qy,dy) : Gc(qy,dy);
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

            double gradX[MAX_D1D];
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
               const double wy = (c == 0) ? Bot(dy,qy) : Gct(dy,qy);
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

static void PADivDivApply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &_Bo,
                            const Array<double> &_Gc,
                            const Array<double> &_Bot,
                            const Array<double> &_Gct,
                            const Vector &_op,
                            const Vector &_x,
                            Vector &_y)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double div[HDIV_MAX_Q1D][HDIV_MAX_Q1D][HDIV_MAX_Q1D];

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
            double aXY[HDIV_MAX_Q1D][HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double aX[HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     aX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Gc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = aX[qx];
                     aXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Gc(qz,dz) : Bo(qz,dz);
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
         double aXY[HDIV_MAX_D1D][HDIV_MAX_D1D];

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
               double aX[HDIV_MAX_D1D];
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
                  const double wy = (c == 1) ? Gct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aXY[dy][dx] += aX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Gct(dz,qz) : Bot(dz,qz);
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

void DivDivIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetFE(0);

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir = IntRule ? IntRule : &MassIntegrator::GetRule
                               (*el, *el, *mesh->GetElementTransformation(0));

   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   pa_data.SetSize(nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   if (el->GetDerivType() == mfem::FiniteElement::DIV && dim == 3)
   {
      PADivDivSetup3D(quad1D, ne, ir->GetWeights(), geom->J, coeff, pa_data);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::DIV && dim == 2)
   {
      PADivDivSetup2D(quad1D, ne, ir->GetWeights(), geom->J, coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void DivDivIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
      PADivDivApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->G,
                      mapsO->Bt, mapsC->Gt, pa_data, x, y);
   else if (dim == 2)
      PADivDivApply2D(dofs1D, quad1D, ne, mapsO->B, mapsC->G,
                      mapsO->Bt, mapsC->Gt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

static void PADivDivAssembleDiagonal2D(const int D1D,
                                       const int Q1D,
                                       const int NE,
                                       const Array<double> &_Bo,
                                       const Array<double> &_Gc,
                                       const Vector &_op,
                                       Vector &_diag)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto diag = Reshape(_diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         double div[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               div[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 0) ? Bo(qy,dy) : Gc(qy,dy);
                  div[qx] += wy * wy * op(qx,qy,e);
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double val = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = (c == 0) ? Gc(qx,dx) : Bo(qx,dx);
                  val += div[qx] * wx * wx;
               }
               diag(dx + (dy * D1Dx) + osc, e) += val;
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   });
}

static void PADivDivAssembleDiagonal3D(const int D1D,
                                       const int Q1D,
                                       const int NE,
                                       const Array<double> &_Bo,
                                       const Array<double> &_Gc,
                                       const Vector &_op,
                                       Vector &_diag)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, NE);
   auto diag = Reshape(_diag.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double a[HDIV_MAX_Q1D];

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  a[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = (c == 1) ? Gc(qy,dy) : Bo(qy,dy);

                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const double wz = (c == 2) ? Gc(qz,dz) : Bo(qz,dz);
                        a[qx] += wy * wy * wz * wz * op(qx,qy,qz,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  double val = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = (c == 0) ? Gc(qx,dx) : Bo(qx,dx);
                     val += a[qx] * wx * wx;
                  }
                  diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += val;
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void DivDivIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      PADivDivAssembleDiagonal3D(dofs1D, quad1D, ne,
                                 mapsO->B, mapsC->G, pa_data, diag);
   }
   else
   {
      PADivDivAssembleDiagonal2D(dofs1D, quad1D, ne,
                                 mapsO->B, mapsC->G, pa_data, diag);
   }
}

// PA H(div)-L2 (div u, p) assemble 2D kernel
static void PADivL2Setup2D(const int Q1D,
                           const int NE,
                           const Array<double> &w,
                           Vector &_coeff,
                           Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         y(q,e) = W[q] * coeff(q,e);
      }
   });
}

static void PADivL2Setup3D(const int Q1D,
                           const int NE,
                           const Array<double> &w,
                           Vector &_coeff,
                           Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         y(q,e) = W[q] * coeff(q, e);
      }
   });
}

void
VectorFEDivergenceIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                         const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and
   // scalar trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const NodalTensorFiniteElement *test_el =
      dynamic_cast<const NodalTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const IntegrationRule *ir = IntRule ? IntRule : &MassIntegrator::GetRule(
                                  *trial_el, *trial_el,
                                  *mesh->GetElementTransformation(0));

   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder() + 1, "");

   ne = trial_fes.GetNE();
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   L2mapsO = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   L2dofs1D = L2mapsO->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");
   if (dim == 2)
   {
      MFEM_VERIFY(nq == quad1D * quad1D, "");
   }
   else
   {
      MFEM_VERIFY(nq == quad1D * quad1D * quad1D, "");
   }

   pa_data.SetSize(nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   if (trial_el->GetDerivType() == mfem::FiniteElement::DIV && dim == 3)
   {
      PADivL2Setup3D(quad1D, ne, ir->GetWeights(), coeff, pa_data);
   }
   else if (trial_el->GetDerivType() == mfem::FiniteElement::DIV && dim == 2)
   {
      PADivL2Setup2D(quad1D, ne, ir->GetWeights(), coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

// Apply to x corresponding to DOF's in H(div) (trial), whose divergence is
// integrated against L_2 test functions corresponding to y.
static void PAHdivL2Apply3D(const int D1D,
                            const int Q1D,
                            const int L2D1D,
                            const int NE,
                            const Array<double> &_Bo,
                            const Array<double> &_Gc,
                            const Array<double> &_L2Bot,
                            const Vector &_op,
                            const Vector &_x,
                            Vector &_y)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto L2Bot = Reshape(_L2Bot.Read(), L2D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), L2D1D, L2D1D, L2D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double div[HDIV_MAX_Q1D][HDIV_MAX_Q1D][HDIV_MAX_Q1D];

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
            double aXY[HDIV_MAX_Q1D][HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double aX[HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     aX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Gc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     aXY[qy][qx] += aX[qx] * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Gc(qz,dz) : Bo(qz,dz);
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
         double aXY[HDIV_MAX_D1D][HDIV_MAX_D1D];

         for (int dy = 0; dy < L2D1D; ++dy)
         {
            for (int dx = 0; dx < L2D1D; ++dx)
            {
               aXY[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double aX[HDIV_MAX_D1D];
            for (int dx = 0; dx < L2D1D; ++dx)
            {
               aX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < L2D1D; ++dx)
               {
                  aX[dx] += div[qz][qy][qx] * L2Bot(dx,qx);
               }
            }
            for (int dy = 0; dy < L2D1D; ++dy)
            {
               const double wy = L2Bot(dy,qy);
               for (int dx = 0; dx < L2D1D; ++dx)
               {
                  aXY[dy][dx] += aX[dx] * wy;
               }
            }
         }

         for (int dz = 0; dz < L2D1D; ++dz)
         {
            const double wz = L2Bot(dz,qz);
            for (int dy = 0; dy < L2D1D; ++dy)
            {
               for (int dx = 0; dx < L2D1D; ++dx)
               {
                  y(dx,dy,dz,e) += aXY[dy][dx] * wz;
               }
            }
         }
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOF's in H(div) (trial), whose divergence is
// integrated against L_2 test functions corresponding to y.
static void PAHdivL2Apply2D(const int D1D,
                            const int Q1D,
                            const int L2D1D,
                            const int NE,
                            const Array<double> &_Bo,
                            const Array<double> &_Gc,
                            const Array<double> &_L2Bot,
                            const Vector &_op,
                            const Vector &_x,
                            Vector &_y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto L2Bot = Reshape(_L2Bot.Read(), L2D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), L2D1D, L2D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double div[MAX_Q1D][MAX_Q1D];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            div[qy][qx] = 0.0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double aX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aX[qx] += t * ((c == 0) ? Gc(qx,dx) : Bo(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 1) ? Gc(qy,dy) : Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  div[qy][qx] += aX[qx] * wy;
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
         double aX[MAX_D1D];
         for (int dx = 0; dx < L2D1D; ++dx)
         {
            aX[dx] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dx = 0; dx < L2D1D; ++dx)
            {
               aX[dx] += div[qy][qx] * L2Bot(dx,qx);
            }
         }
         for (int dy = 0; dy < L2D1D; ++dy)
         {
            const double wy = L2Bot(dy,qy);
            for (int dx = 0; dx < L2D1D; ++dx)
            {
               y(dx,dy,e) += aX[dx] * wy;
            }
         }
      }
   }); // end of element loop
}

static void PAHdivL2ApplyTranspose3D(const int D1D,
                                     const int Q1D,
                                     const int L2D1D,
                                     const int NE,
                                     const Array<double> &_L2Bo,
                                     const Array<double> &_Gct,
                                     const Array<double> &_Bot,
                                     const Vector &_op,
                                     const Vector &_x,
                                     Vector &_y)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto L2Bo = Reshape(_L2Bo.Read(), Q1D, L2D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), L2D1D, L2D1D, L2D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double div[HDIV_MAX_Q1D][HDIV_MAX_Q1D][HDIV_MAX_Q1D];

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

      for (int dz = 0; dz < L2D1D; ++dz)
      {
         double aXY[HDIV_MAX_Q1D][HDIV_MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aXY[qy][qx] = 0.0;
            }
         }

         for (int dy = 0; dy < L2D1D; ++dy)
         {
            double aX[HDIV_MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aX[qx] = 0.0;
            }

            for (int dx = 0; dx < L2D1D; ++dx)
            {
               const double t = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aX[qx] += t * L2Bo(qx,dx);
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = L2Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aXY[qy][qx] += aX[qx] * wy;
               }
            }
         }

         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz = L2Bo(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  div[qz][qy][qx] += aXY[qy][qx] * wz;
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
               div[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double aXY[HDIV_MAX_D1D][HDIV_MAX_D1D];

         int osc = 0;
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
               double aX[HDIV_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  aX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aX[dx] += div[qz][qy][qx] * ((c == 0) ? Gct(dx,qx) :
                                                  Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Gct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aXY[dy][dx] += aX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Gct(dz,qz) : Bot(dz,qz);
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

static void PAHdivL2ApplyTranspose2D(const int D1D,
                                     const int Q1D,
                                     const int L2D1D,
                                     const int NE,
                                     const Array<double> &_L2Bo,
                                     const Array<double> &_Gct,
                                     const Array<double> &_Bot,
                                     const Vector &_op,
                                     const Vector &_x,
                                     Vector &_y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto L2Bo = Reshape(_L2Bo.Read(), Q1D, L2D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), L2D1D, L2D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double div[MAX_Q1D][MAX_Q1D];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            div[qy][qx] = 0.0;
         }
      }

      for (int dy = 0; dy < L2D1D; ++dy)
      {
         double aX[MAX_Q1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            aX[qx] = 0.0;
         }

         for (int dx = 0; dx < L2D1D; ++dx)
         {
            const double t = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aX[qx] += t * L2Bo(qx,dx);
            }
         }

         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy = L2Bo(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               div[qy][qx] += aX[qx] * wy;
            }
         }
      }

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
         double aX[MAX_D1D];

         int osc = 0;
         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D : D1D - 1;
            const int D1Dx = (c == 0) ? D1D : D1D - 1;

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               aX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  aX[dx] += div[qy][qx] * ((c == 0) ? Gct(dx,qx) : Bot(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = (c == 0) ? Bot(dy,qy) : Gct(dy,qy);
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += aX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

void VectorFEDivergenceIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
      PAHdivL2Apply3D(dofs1D, quad1D, L2dofs1D, ne, mapsO->B, mapsC->G,
                      L2mapsO->Bt, pa_data, x, y);
   else if (dim == 2)
      PAHdivL2Apply2D(dofs1D, quad1D, L2dofs1D, ne, mapsO->B, mapsC->G,
                      L2mapsO->Bt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void VectorFEDivergenceIntegrator::AddMultTransposePA(const Vector &x,
                                                      Vector &y) const
{
   if (dim == 3)
      PAHdivL2ApplyTranspose3D(dofs1D, quad1D, L2dofs1D, ne, L2mapsO->B,
                               mapsC->Gt, mapsO->Bt, pa_data, x, y);
   else if (dim == 2)
      PAHdivL2ApplyTranspose2D(dofs1D, quad1D, L2dofs1D, ne, L2mapsO->B,
                               mapsC->Gt, mapsO->Bt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

static void PAHdivL2AssembleDiagonal_ADAt_3D(const int D1D,
                                             const int Q1D,
                                             const int L2D1D,
                                             const int NE,
                                             const Array<double> &_L2Bo,
                                             const Array<double> &_Gct,
                                             const Array<double> &_Bot,
                                             const Vector &_op,
                                             const Vector &_D,
                                             Vector &_diag)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto L2Bo = Reshape(_L2Bo.Read(), Q1D, L2D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, NE);
   auto D = Reshape(_D.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto diag = Reshape(_diag.ReadWrite(), L2D1D, L2D1D, L2D1D, NE);

   MFEM_FORALL(e, NE,
   {
      for (int rz = 0; rz < L2D1D; ++rz)
      {
         for (int ry = 0; ry < L2D1D; ++ry)
         {
            for (int rx = 0; rx < L2D1D; ++rx)
            {
               // Compute row (rx,ry,rz), assuming all contributions are from
               // a single element.

               double row[3*HDIV_MAX_D1D*(HDIV_MAX_D1D-1)*(HDIV_MAX_D1D-1)];
               double div[HDIV_MAX_Q1D][HDIV_MAX_Q1D][HDIV_MAX_Q1D];

               for (int i=0; i<3*D1D*(D1D - 1)*(D1D - 1); ++i)
               {
                  row[i] = 0;
               }

               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        div[qz][qy][qx] = op(qx,qy,qz,e) * L2Bo(qx,rx) *
                                          L2Bo(qy,ry) * L2Bo(qz,rz);
                     }
                  }
               }

               for (int qz = 0; qz < Q1D; ++qz)
               {
                  double aXY[HDIV_MAX_D1D][HDIV_MAX_D1D];

                  int osc = 0;
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
                        double aX[HDIV_MAX_D1D];
                        for (int dx = 0; dx < D1Dx; ++dx)
                        {
                           aX[dx] = 0;
                        }
                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           for (int dx = 0; dx < D1Dx; ++dx)
                           {
                              aX[dx] += div[qz][qy][qx] * ((c == 0) ? Gct(dx,qx)
                                                           : Bot(dx,qx));
                           }
                        }
                        for (int dy = 0; dy < D1Dy; ++dy)
                        {
                           const double wy = (c == 1) ? Gct(dy,qy) : Bot(dy,qy);
                           for (int dx = 0; dx < D1Dx; ++dx)
                           {
                              aXY[dy][dx] += aX[dx] * wy;
                           }
                        }
                     }

                     for (int dz = 0; dz < D1Dz; ++dz)
                     {
                        const double wz = (c == 2) ? Gct(dz,qz) : Bot(dz,qz);
                        for (int dy = 0; dy < D1Dy; ++dy)
                        {
                           for (int dx = 0; dx < D1Dx; ++dx)
                           {
                              row[dx + ((dy + (dz * D1Dy)) * D1Dx) + osc] +=
                                 aXY[dy][dx] * wz;
                           }
                        }
                     }

                     osc += D1Dx * D1Dy * D1Dz;
                  }  // loop c
               }  // loop qz

               double val = 0.0;
               for (int i=0; i<3*D1D*(D1D - 1)*(D1D - 1); ++i)
               {
                  val += row[i] * row[i] * D(i,e);
               }
               diag(rx,ry,rz,e) += val;
            }  // loop rx
         }  // loop ry
      }  // loop rz
   }); // end of element loop
}

static void PAHdivL2AssembleDiagonal_ADAt_2D(const int D1D,
                                             const int Q1D,
                                             const int L2D1D,
                                             const int NE,
                                             const Array<double> &_L2Bo,
                                             const Array<double> &_Gct,
                                             const Array<double> &_Bot,
                                             const Vector &_op,
                                             const Vector &_D,
                                             Vector &_diag)
{
   constexpr static int VDIM = 2;

   auto L2Bo = Reshape(_L2Bo.Read(), Q1D, L2D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto D = Reshape(_D.Read(), 2*(D1D-1)*D1D, NE);
   auto diag = Reshape(_diag.ReadWrite(), L2D1D, L2D1D, NE);

   MFEM_FORALL(e, NE,
   {
      for (int ry = 0; ry < L2D1D; ++ry)
      {
         for (int rx = 0; rx < L2D1D; ++rx)
         {
            // Compute row (rx,ry), assuming all contributions are from
            // a single element.

            double row[2*HDIV_MAX_D1D*(HDIV_MAX_D1D-1)];
            double div[HDIV_MAX_Q1D][HDIV_MAX_Q1D];

            for (int i=0; i<2*D1D*(D1D - 1); ++i)
            {
               row[i] = 0;
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  div[qy][qx] = op(qx,qy,e) * L2Bo(qx,rx) * L2Bo(qy,ry);
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               int osc = 0;
               for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
               {
                  const int D1Dy = (c == 1) ? D1D : D1D - 1;
                  const int D1Dx = (c == 0) ? D1D : D1D - 1;

                  double aX[HDIV_MAX_D1D];
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     aX[dx] = 0;
                  }
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     for (int dx = 0; dx < D1Dx; ++dx)
                     {
                        aX[dx] += div[qy][qx] * ((c == 0) ? Gct(dx,qx) :
                                                 Bot(dx,qx));
                     }
                  }

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = (c == 1) ? Gct(dy,qy) : Bot(dy,qy);

                     for (int dx = 0; dx < D1Dx; ++dx)
                     {
                        row[dx + (dy * D1Dx) + osc] += aX[dx] * wy;
                     }
                  }

                  osc += D1Dx * D1Dy;
               }  // loop c
            }  // loop qy

            double val = 0.0;
            for (int i=0; i<2*D1D*(D1D - 1); ++i)
            {
               val += row[i] * row[i] * D(i,e);
            }
            diag(rx,ry,e) += val;
         }  // loop rx
      }  // loop ry
   }); // end of element loop
}

void VectorFEDivergenceIntegrator::AssembleDiagonalPA_ADAt(const Vector &D,
                                                           Vector &diag)
{
   if (dim == 3)
      PAHdivL2AssembleDiagonal_ADAt_3D(dofs1D, quad1D, L2dofs1D, ne, L2mapsO->B,
                                       mapsC->Gt, mapsO->Bt, pa_data, D, diag);
   else if (dim == 2)
      PAHdivL2AssembleDiagonal_ADAt_2D(dofs1D, quad1D, L2dofs1D, ne, L2mapsO->B,
                                       mapsC->Gt, mapsO->Bt, pa_data, D, diag);
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

} // namespace mfem
