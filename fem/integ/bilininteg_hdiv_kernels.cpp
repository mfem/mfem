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

#include "bilininteg_hdiv_kernels.hpp"

namespace mfem
{

namespace internal
{

void PAHdivMassSetup2D(const int Q1D,
                       const int coeffDim,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff_,
                       Vector &op)
{
   const bool symmetric = (coeffDim != 4);
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto C = Reshape(coeff_.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), NQ, symmetric ? 3 : 4, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         const real_t J11 = J(q,0,0,e);
         const real_t J21 = J(q,1,0,e);
         const real_t J12 = J(q,0,1,e);
         const real_t J22 = J(q,1,1,e);
         const real_t c_detJ = W[q] / ((J11*J22)-(J21*J12));

         // (1/detJ) J^T C J
         if (coeffDim == 3 || coeffDim == 4) // Matrix coefficient
         {
            const real_t C11 = C(0,q,e);
            const real_t C12 = C(1,q,e);
            const real_t C21 = symmetric ? C12 : C(2,q,e);
            const real_t C22 = symmetric ? C(2,q,e) : C(3,q,e);
            const real_t R11 = C11*J11 + C12*J21;
            const real_t R21 = C21*J11 + C22*J21;
            const real_t R12 = C11*J12 + C12*J22;
            const real_t R22 = C21*J12 + C22*J22;

            y(q,0,e) = c_detJ * (J11*R11 + J21*R21); // 1,1
            y(q,1,e) = c_detJ * (J11*R12 + J21*R22); // 1,2

            if (symmetric)
            {
               y(q,2,e) = c_detJ * (J12*R12 + J22*R22); // 2,2
            }
            else
            {
               y(q,2,e) = c_detJ * (J12*R11 + J22*R21); // 2,1
               y(q,3,e) = c_detJ * (J12*R12 + J22*R22); // 2,2
            }
         }
         else // Vector or scalar coefficient
         {
            const real_t C1 = C(0,q,e);
            const real_t C2 = (coeffDim == 2 ? C(1,q,e) : C1);
            y(q,0,e) = c_detJ * (J11*C1*J11 + J21*C2*J21); // 1,1
            y(q,1,e) = c_detJ * (J11*C1*J12 + J21*C2*J22); // 1,2
            y(q,2,e) = c_detJ * (J12*C1*J12 + J22*C2*J22); // 2,2
         }
      }
   });
}

void PAHdivMassSetup3D(const int Q1D,
                       const int coeffDim,
                       const int NE,
                       const Array<real_t> &w,
                       const Vector &j,
                       Vector &coeff_,
                       Vector &op)
{
   const bool symmetric = (coeffDim != 9);
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto C = Reshape(coeff_.Read(), coeffDim, NQ, NE);
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

         // (1/detJ) J^T C J
         if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
         {
            real_t M[3][3];
            M[0][0] = C(0, q, e);
            M[0][1] = C(1, q, e);
            M[0][2] = C(2, q, e);
            M[1][0] = (!symmetric) ? C(3, q, e) : M[0][1];
            M[1][1] = (!symmetric) ? C(4, q, e) : C(3, q, e);
            M[1][2] = (!symmetric) ? C(5, q, e) : C(4, q, e);
            M[2][0] = (!symmetric) ? C(6, q, e) : M[0][2];
            M[2][1] = (!symmetric) ? C(7, q, e) : M[1][2];
            M[2][2] = (!symmetric) ? C(8, q, e) : C(5, q, e);

            int idx = 0;
            for (int i=0; i<3; ++i)
               for (int j = (symmetric ? i : 0); j<3; ++j)
               {
                  y(q,idx,e) = 0.0;
                  for (int k=0; k<3; ++k)
                  {
                     real_t MJ_kj = 0.0;
                     for (int l=0; l<3; ++l)
                     {
                        MJ_kj += M[k][l] * J(q,l,j,e);
                     }

                     y(q,idx,e) += J(q,k,i,e) * MJ_kj;
                  }

                  y(q,idx,e) *= c_detJ;
                  idx++;
               }
         }
         else  // Vector or scalar coefficient version
         {
            int idx = 0;
            for (int i=0; i<3; ++i)
               for (int j=i; j<3; ++j)
               {
                  y(q,idx,e) = 0.0;
                  for (int k=0; k<3; ++k)
                  {
                     y(q,idx,e) += J(q,k,i,e) * C(coeffDim == 3 ? k : 0, q, e) * J(q,k,j,e);
                  }

                  y(q,idx,e) *= c_detJ;
                  idx++;
               }
         }
      }
   });
}

void PAHdivMassAssembleDiagonal2D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const bool symmetric,
                                  const Array<real_t> &Bo_,
                                  const Array<real_t> &Bc_,
                                  const Vector &op_,
                                  Vector &diag_)
{
   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto diag = Reshape(diag_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            real_t mass[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  mass[qx] += wy*wy*((c == 0) ? op(qx,qy,0,e) : op(qx,qy,symmetric ? 2 : 3,e));
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               real_t val = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = (c == 0) ? Bc(qx,dx) : Bo(qx,dx);
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
                                  const bool symmetric,
                                  const Array<real_t> &Bo_,
                                  const Array<real_t> &Bc_,
                                  const Vector &op_,
                                  Vector &diag_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto diag = Reshape(diag_.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         const int opc = (c == 0) ? 0 : ((c == 1) ? (symmetric ? 3 : 4) :
                                         (symmetric ? 5 : 8));

         real_t mass[DofQuadLimits::HDIV_MAX_Q1D];

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const real_t wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
                        mass[qx] += wy * wy * wz * wz * op(qx,qy,qz,opc,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  real_t val = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = (c == 0) ? Bc(qx,dx) : Bo(qx,dx);
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

void PAHdivMassApply(const int dim,
                     const int D1D,
                     const int Q1D,
                     const int NE,
                     const bool symmetric,
                     const Array<real_t> &Bo,
                     const Array<real_t> &Bc,
                     const Array<real_t> &Bot,
                     const Array<real_t> &Bct,
                     const Vector &op,
                     const Vector &x,
                     Vector &y)
{
   const int id = (D1D << 4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x22: return SmemPAHdivMassApply2D<2,2>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x33: return SmemPAHdivMassApply2D<3,3>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x44: return SmemPAHdivMassApply2D<4,4>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x55: return SmemPAHdivMassApply2D<5,5>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         default: // fallback
            return PAHdivMassApply2D(D1D,Q1D,NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
      }
   }
   else if (dim == 3)
   {
      switch (id)
      {
         case 0x23: return SmemPAHdivMassApply3D<2,3>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x34: return SmemPAHdivMassApply3D<3,4>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x45: return SmemPAHdivMassApply3D<4,5>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x56: return SmemPAHdivMassApply3D<5,6>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x67: return SmemPAHdivMassApply3D<6,7>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         case 0x78: return SmemPAHdivMassApply3D<7,8>(NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
         default: // fallback
            return PAHdivMassApply3D(D1D,Q1D,NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);
      }
   }
}

void PAHdivMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const bool symmetric,
                       const Array<real_t> &Bo_,
                       const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_,
                       const Array<real_t> &Bct_,
                       const Vector &op_,
                       const Vector &x_,
                       Vector &y_)
{
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

void PAHdivMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const bool symmetric,
                       const Array<real_t> &Bo_,
                       const Array<real_t> &Bc_,
                       const Array<real_t> &Bot_,
                       const Array<real_t> &Bct_,
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

// NOTE: this is identical to PACurlCurlSetup2D
void PADivDivSetup2D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     const Vector &j,
                     Vector &coeff_,
                     Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(coeff_.Read(), NQ, NE);
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
         y(q,e) = W[q] * coeff(q,e) / detJ;
      }
   });
}

void PADivDivSetup3D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     const Vector &j,
                     Vector &coeff_,
                     Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(coeff_.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);

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
         y(q,e) = W[q] * coeff(q, e) / detJ;
      }
   });
}

void PADivDivAssembleDiagonal2D(const int D1D,
                                const int Q1D,
                                const int NE,
                                const Array<real_t> &Bo_,
                                const Array<real_t> &Gc_,
                                const Vector &op_,
                                Vector &diag_)
{
   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto diag = Reshape(diag_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dx = (c == 1) ? D1D - 1 : D1D;
         const int D1Dy = (c == 0) ? D1D - 1 : D1D;

         real_t div[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               div[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t wy = (c == 0) ? Bo(qy,dy) : Gc(qy,dy);
                  div[qx] += wy * wy * op(qx,qy,e);
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               real_t val = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t wx = (c == 0) ? Gc(qx,dx) : Bo(qx,dx);
                  val += div[qx] * wx * wx;
               }
               diag(dx + (dy * D1Dx) + osc, e) += val;
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   });
}

void PADivDivAssembleDiagonal3D(const int D1D,
                                const int Q1D,
                                const int NE,
                                const Array<real_t> &Bo_,
                                const Array<real_t> &Gc_,
                                const Vector &op_,
                                Vector &diag_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto diag = Reshape(diag_.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
               real_t a[DofQuadLimits::HDIV_MAX_Q1D];

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  a[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const real_t wy = (c == 1) ? Gc(qy,dy) : Bo(qy,dy);

                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const real_t wz = (c == 2) ? Gc(qz,dz) : Bo(qz,dz);
                        a[qx] += wy * wy * wz * wz * op(qx,qy,qz,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  real_t val = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const real_t wx = (c == 0) ? Gc(qx,dx) : Bo(qx,dx);
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

void PADivDivApply2D(const int D1D,
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

void PADivDivApply3D(const int D1D,
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

void PAHdivL2Setup2D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     Vector &coeff_,
                     Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto coeff = Reshape(coeff_.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         y(q,e) = W[q] * coeff(q,e);
      }
   });
}

void PAHdivL2Setup3D(const int Q1D,
                     const int NE,
                     const Array<real_t> &w,
                     Vector &coeff_,
                     Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto coeff = Reshape(coeff_.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         y(q,e) = W[q] * coeff(q, e);
      }
   });
}

void PAHdivL2AssembleDiagonal_ADAt_2D(const int D1D,
                                      const int Q1D,
                                      const int L2D1D,
                                      const int NE,
                                      const Array<real_t> &L2Bo_,
                                      const Array<real_t> &Gct_,
                                      const Array<real_t> &Bot_,
                                      const Vector &op_,
                                      const Vector &D_,
                                      Vector &diag_)
{
   constexpr static int VDIM = 2;

   auto L2Bo = Reshape(L2Bo_.Read(), Q1D, L2D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto D = Reshape(D_.Read(), 2*(D1D-1)*D1D, NE);
   auto diag = Reshape(diag_.ReadWrite(), L2D1D, L2D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int ry = 0; ry < L2D1D; ++ry)
      {
         for (int rx = 0; rx < L2D1D; ++rx)
         {
            // Compute row (rx,ry), assuming all contributions are from
            // a single element.

            real_t row[2*DofQuadLimits::HDIV_MAX_D1D*(DofQuadLimits::HDIV_MAX_D1D-1)];
            real_t div[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];

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

                  real_t aX[DofQuadLimits::HDIV_MAX_D1D];
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
                     const real_t wy = (c == 1) ? Gct(dy,qy) : Bot(dy,qy);

                     for (int dx = 0; dx < D1Dx; ++dx)
                     {
                        row[dx + (dy * D1Dx) + osc] += aX[dx] * wy;
                     }
                  }

                  osc += D1Dx * D1Dy;
               }  // loop c
            }  // loop qy

            real_t val = 0.0;
            for (int i=0; i<2*D1D*(D1D - 1); ++i)
            {
               val += row[i] * row[i] * D(i,e);
            }
            diag(rx,ry,e) += val;
         }  // loop rx
      }  // loop ry
   }); // end of element loop
}

void PAHdivL2AssembleDiagonal_ADAt_3D(const int D1D,
                                      const int Q1D,
                                      const int L2D1D,
                                      const int NE,
                                      const Array<real_t> &L2Bo_,
                                      const Array<real_t> &Gct_,
                                      const Array<real_t> &Bot_,
                                      const Vector &op_,
                                      const Vector &D_,
                                      Vector &diag_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto L2Bo = Reshape(L2Bo_.Read(), Q1D, L2D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto D = Reshape(D_.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto diag = Reshape(diag_.ReadWrite(), L2D1D, L2D1D, L2D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int rz = 0; rz < L2D1D; ++rz)
      {
         for (int ry = 0; ry < L2D1D; ++ry)
         {
            for (int rx = 0; rx < L2D1D; ++rx)
            {
               // Compute row (rx,ry,rz), assuming all contributions are from
               // a single element.

               real_t row[3*DofQuadLimits::HDIV_MAX_D1D*(DofQuadLimits::HDIV_MAX_D1D-1)*
                          (DofQuadLimits::HDIV_MAX_D1D-1)];
               real_t div[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];

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
                  real_t aXY[DofQuadLimits::HDIV_MAX_D1D][DofQuadLimits::HDIV_MAX_D1D];

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
                        real_t aX[DofQuadLimits::HDIV_MAX_D1D];
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
                              row[dx + ((dy + (dz * D1Dy)) * D1Dx) + osc] +=
                                 aXY[dy][dx] * wz;
                           }
                        }
                     }

                     osc += D1Dx * D1Dy * D1Dz;
                  }  // loop c
               }  // loop qz

               real_t val = 0.0;
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

// Apply to x corresponding to DOFs in H(div) (trial), whose divergence is
// integrated against L_2 test functions corresponding to y.
void PAHdivL2Apply2D(const int D1D,
                     const int Q1D,
                     const int L2D1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &L2Bot_,
                     const Vector &op_,
                     const Vector &x_,
                     Vector &y_)
{
   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto L2Bot = Reshape(L2Bot_.Read(), L2D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), L2D1D, L2D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t div[MAX_Q1D][MAX_Q1D];

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
            real_t aX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const real_t t = x(dx + (dy * D1Dx) + osc, e);
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
         real_t aX[MAX_D1D];
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
            const real_t wy = L2Bot(dy,qy);
            for (int dx = 0; dx < L2D1D; ++dx)
            {
               y(dx,dy,e) += aX[dx] * wy;
            }
         }
      }
   }); // end of element loop
}

void PAHdivL2ApplyTranspose2D(const int D1D,
                              const int Q1D,
                              const int L2D1D,
                              const int NE,
                              const Array<real_t> &L2Bo_,
                              const Array<real_t> &Gct_,
                              const Array<real_t> &Bot_,
                              const Vector &op_,
                              const Vector &x_,
                              Vector &y_)
{
   auto L2Bo = Reshape(L2Bo_.Read(), Q1D, L2D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), L2D1D, L2D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr static int VDIM = 2;
      constexpr static int MAX_D1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr static int MAX_Q1D = DofQuadLimits::HDIV_MAX_Q1D;

      real_t div[MAX_Q1D][MAX_Q1D];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            div[qy][qx] = 0.0;
         }
      }

      for (int dy = 0; dy < L2D1D; ++dy)
      {
         real_t aX[MAX_Q1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            aX[qx] = 0.0;
         }

         for (int dx = 0; dx < L2D1D; ++dx)
         {
            const real_t t = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aX[qx] += t * L2Bo(qx,dx);
            }
         }

         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t wy = L2Bo(qy,dy);
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
         real_t aX[MAX_D1D];

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
               const real_t wy = (c == 0) ? Bot(dy,qy) : Gct(dy,qy);
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

// Apply to x corresponding to DOFs in H(div) (trial), whose divergence is
// integrated against L_2 test functions corresponding to y.
void PAHdivL2Apply3D(const int D1D,
                     const int Q1D,
                     const int L2D1D,
                     const int NE,
                     const Array<real_t> &Bo_,
                     const Array<real_t> &Gc_,
                     const Array<real_t> &L2Bot_,
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
   auto L2Bot = Reshape(L2Bot_.Read(), L2D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), L2D1D, L2D1D, L2D1D, NE);

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
                     aXY[qy][qx] += aX[qx] * wy;
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

         for (int dy = 0; dy < L2D1D; ++dy)
         {
            for (int dx = 0; dx < L2D1D; ++dx)
            {
               aXY[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t aX[DofQuadLimits::HDIV_MAX_D1D];
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
               const real_t wy = L2Bot(dy,qy);
               for (int dx = 0; dx < L2D1D; ++dx)
               {
                  aXY[dy][dx] += aX[dx] * wy;
               }
            }
         }

         for (int dz = 0; dz < L2D1D; ++dz)
         {
            const real_t wz = L2Bot(dz,qz);
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

void PAHdivL2ApplyTranspose3D(const int D1D,
                              const int Q1D,
                              const int L2D1D,
                              const int NE,
                              const Array<real_t> &L2Bo_,
                              const Array<real_t> &Gct_,
                              const Array<real_t> &Bot_,
                              const Vector &op_,
                              const Vector &x_,
                              Vector &y_)
{
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto L2Bo = Reshape(L2Bo_.Read(), Q1D, L2D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), L2D1D, L2D1D, L2D1D, NE);
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

      for (int dz = 0; dz < L2D1D; ++dz)
      {
         real_t aXY[DofQuadLimits::HDIV_MAX_Q1D][DofQuadLimits::HDIV_MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aXY[qy][qx] = 0.0;
            }
         }

         for (int dy = 0; dy < L2D1D; ++dy)
         {
            real_t aX[DofQuadLimits::HDIV_MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               aX[qx] = 0.0;
            }

            for (int dx = 0; dx < L2D1D; ++dx)
            {
               const real_t t = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aX[qx] += t * L2Bo(qx,dx);
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = L2Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  aXY[qy][qx] += aX[qx] * wy;
               }
            }
         }

         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz = L2Bo(qz,dz);
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
         real_t aXY[DofQuadLimits::HDIV_MAX_D1D][DofQuadLimits::HDIV_MAX_D1D];

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
               real_t aX[DofQuadLimits::HDIV_MAX_D1D];
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

} // namespace internal

} // namespace mfem
