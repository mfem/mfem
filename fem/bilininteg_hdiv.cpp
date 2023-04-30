// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "qspace.hpp"

using namespace std;


// Piola transformation in H(div): w = (1 / det (dF)) dF \hat{w}
// div w = (1 / det (dF)) \hat{div} \hat{w}

namespace mfem
{

// PA H(div) Mass Assemble 2D kernel
void PAHdivSetup2D(const int Q1D,
                   const int coeffDim,
                   const int NE,
                   const Array<double> &w,
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
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double c_detJ = W[q] / ((J11*J22)-(J21*J12));

         // (1/detJ) J^T C J
         if (coeffDim == 3 || coeffDim == 4) // Matrix coefficient
         {
            const double C11 = C(0,q,e);
            const double C12 = C(1,q,e);
            const double C21 = symmetric ? C12 : C(2,q,e);
            const double C22 = symmetric ? C(2,q,e) : C(3,q,e);
            const double R11 = C11*J11 + C12*J21;
            const double R21 = C21*J11 + C22*J21;
            const double R12 = C11*J12 + C12*J22;
            const double R22 = C21*J12 + C22*J22;

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
            const double C1 = C(0,q,e);
            const double C2 = (coeffDim == 2 ? C(1,q,e) : C1);
            y(q,0,e) = c_detJ * (J11*C1*J11 + J21*C2*J21); // 1,1
            y(q,1,e) = c_detJ * (J11*C1*J12 + J21*C2*J22); // 1,2
            y(q,2,e) = c_detJ * (J12*C1*J12 + J22*C2*J22); // 2,2
         }
      }
   });
}

// PA H(div) Mass Assemble 3D kernel
void PAHdivSetup3D(const int Q1D,
                   const int coeffDim,
                   const int NE,
                   const Array<double> &w,
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
                             J21 * (J12 * J33 - J32 * J13) +
                             J31 * (J12 * J23 - J22 * J13);
         const double c_detJ = W[q] / detJ;

         // (1/detJ) J^T C J
         if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
         {
            double M[3][3];
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
                     double MJ_kj = 0.0;
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

void PAHdivMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const bool symmetric,
                       const Array<double> &Bo_,
                       const Array<double> &Bc_,
                       const Array<double> &Bot_,
                       const Array<double> &Bct_,
                       const Vector &op_,
                       const Vector &x_,
                       Vector &y_)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Bct = Reshape(Bct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
            const double O21 = symmetric ? O12 : op(qx,qy,2,e);
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

template<int T_D1D = 0, int T_Q1D = 0>
void SmemPAHdivMassApply2D(const int NE,
                           const bool symmetric,
                           const Array<double> &Bo_,
                           const Array<double> &Bc_,
                           const Array<double> &Bot_,
                           const Array<double> &Bct_,
                           const Vector &op_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   MFEM_CONTRACT_VAR(Bot_);
   MFEM_CONTRACT_VAR(Bct_);

   static constexpr int VDIM = 2;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   const auto bc = Reshape(Bc_.Read(), Q1D, D1D);
   const auto D = Reshape(op_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   const auto x = Reshape(x_.Read(), D1D*(D1D-1), VDIM, NE);
   auto y = y_.ReadWrite();

   mfem::forall_3D(NE, Q1D, Q1D, VDIM, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int MQ1 = T_Q1D ? T_Q1D : HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED double smo[MQ1*(MD1-1)];
      DeviceMatrix Bo(smo, D1D-1, Q1D);

      MFEM_SHARED double smc[MQ1*MD1];
      DeviceMatrix Bc(smc, D1D, Q1D);

      MFEM_SHARED double sm0[VDIM*MDQ*MDQ];
      MFEM_SHARED double sm1[VDIM*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1), VDIM);
      DeviceCube QD(sm1, Q1D, D1D, VDIM);
      DeviceCube QQ(sm0, Q1D, Q1D, VDIM);

      // Load X, Bo and Bc into shared memory
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               if (qx < D1D && dy < (D1D-1)) { X(qx + dy*D1D,vd) = x(qx+dy*D1D,vd,e); }
               if (tidz == 0)
               {
                  if (dy < (D1D-1)) { Bo(dy,qx) = bo(qx,dy); }
                  Bc(dy,qx) = bc(qx,dy);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply B operator
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceCube Xxy(X, nx, ny, VDIM);
         DeviceMatrix Bx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double dq = 0.0;
               for (int dx = 0; dx < nx; ++dx)
               {
                  dq += Xxy(dx,dy,vd) * Bx(dx,qx);
               }
               QD(qx,dy,vd) = dq;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double qq = 0.0;
               for (int dy = 0; dy < ny; ++dy)
               {
                  qq += QD(qx,dy,vd) * By(dy,qy);
               }
               QQ(qx,qy,vd) = qq;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply D operator
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double Qx = QQ(qx,qy,0);
               const double Qy = QQ(qx,qy,1);

               const double D11 = D(qx,qy,0,e);
               const double D12 = D(qx,qy,1,e);
               const double D21 = symmetric ? D12 : D(qx,qy,2,e);
               const double D22 = symmetric ? D(qx,qy,2,e) : D(qx,qy,3,e);

               QQ(qx,qy,0) = D11*Qx + D12*Qy;
               QQ(qx,qy,1) = D21*Qx + D22*Qy;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         DeviceMatrix Btx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               double qd = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  qd += QQ(qx,qy,vd) * Btx(dx,qx);
               }
               QD(dx,qy,vd) = qd;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix Bty = (vd == 1) ? Bc : Bo;
         DeviceTensor<4> Yxy(y, nx, ny, VDIM, NE);
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               double dd = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  dd += QD(dx,qy,vd) * Bty(dy,qy);
               }
               Yxy(dx,dy,vd,e) += dd;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

void PAHdivMassAssembleDiagonal2D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const bool symmetric,
                                  const Array<double> &Bo_,
                                  const Array<double> &Bc_,
                                  const Vector &op_,
                                  Vector &diag_)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, symmetric ? 3 : 4, NE);
   auto diag = Reshape(diag_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
                  mass[qx] += wy*wy*((c == 0) ? op(qx,qy,0,e) : op(qx,qy,symmetric ? 2 : 3,e));
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
                                  const bool symmetric,
                                  const Array<double> &Bo_,
                                  const Array<double> &Bc_,
                                  const Vector &op_,
                                  Vector &diag_)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
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
                       const bool symmetric,
                       const Array<double> &Bo_,
                       const Array<double> &Bc_,
                       const Array<double> &Bot_,
                       const Array<double> &Bct_,
                       const Vector &op_,
                       const Vector &x_,
                       Vector &y_)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
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

template<int T_D1D = 0, int T_Q1D = 0>
void SmemPAHdivMassApply3D(const int NE,
                           const bool symmetric,
                           const Array<double> &Bo_,
                           const Array<double> &Bc_,
                           const Array<double> &Bot_,
                           const Array<double> &Bct_,
                           const Vector &op_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   MFEM_CONTRACT_VAR(Bot_);
   MFEM_CONTRACT_VAR(Bct_);

   static constexpr int VDIM = 3;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   const auto bc = Reshape(Bc_.Read(), Q1D, D1D);
   const auto D = Reshape(op_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   const auto x = Reshape(x_.Read(), D1D*(D1D-1)*(D1D-1), VDIM, NE);
   auto y = y_.ReadWrite();

   mfem::forall_3D(NE, Q1D, Q1D, VDIM, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int MQ1 = T_Q1D ? T_Q1D : HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED double smo[MQ1*(MD1-1)];
      DeviceMatrix Bo(smo, D1D-1, Q1D);

      MFEM_SHARED double smc[MQ1*MD1];
      DeviceMatrix Bc(smc, D1D, Q1D);

      MFEM_SHARED double sm0[VDIM*MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[VDIM*MDQ*MDQ*MDQ];
      DeviceMatrix X(sm0, D1D*(D1D-1)*(D1D-1), VDIM);
      DeviceTensor<4> QDD(sm1, Q1D, D1D, D1D, VDIM);
      DeviceTensor<4> QQD(sm0, Q1D, Q1D, D1D, VDIM);
      DeviceTensor<4> QQQ(sm1, Q1D, Q1D, Q1D, VDIM);
      DeviceTensor<4> DQQ(sm0, D1D, Q1D, Q1D, VDIM);
      DeviceTensor<4> DDQ(sm1, D1D, D1D, Q1D, VDIM);

      // Load X into shared memory
      MFEM_FOREACH_THREAD(vd,z,VDIM)
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
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceTensor<4> Xxyz(X, nx, ny, nz, VDIM);
         DeviceMatrix Bx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[D1D];
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
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix By = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[D1D];
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
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceMatrix Bz = (vd == 2) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[Q1D];
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
      // Apply D operator
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const double Qx = QQQ(qx,qy,qz,0);
                  const double Qy = QQQ(qx,qy,qz,1);
                  const double Qz = QQQ(qx,qy,qz,2);

                  const double D11 = D(qx,qy,qz,0,e);
                  const double D12 = D(qx,qy,qz,1,e);
                  const double D13 = D(qx,qy,qz,2,e);
                  const double D21 = symmetric ? D12 : D(qx,qy,qz,3,e);
                  const double D22 = symmetric ? D(qx,qy,qz,3,e) : D(qx,qy,qz,4,e);
                  const double D23 = symmetric ? D(qx,qy,qz,4,e) : D(qx,qy,qz,5,e);
                  const double D31 = symmetric ? D13 : D(qx,qy,qz,6,e);
                  const double D32 = symmetric ? D23 : D(qx,qy,qz,7,e);
                  const double D33 = symmetric ? D(qx,qy,qz,5,e) : D(qx,qy,qz,8,e);

                  QQQ(qx,qy,qz,0) = D11*Qx + D12*Qy + D13*Qz;
                  QQQ(qx,qy,qz,1) = D21*Qx + D22*Qy + D23*Qz;
                  QQQ(qx,qy,qz,2) = D31*Qx + D32*Qy + D33*Qz;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         DeviceMatrix Btx = (vd == 0) ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               double u[Q1D];
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u[qz] += QQQ(qx,qy,qz,vd) * Btx(dx,qx);
                  }
               }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { DQQ(dx,qy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         DeviceMatrix Bty = (vd == 1) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               double u[Q1D];
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { u[qz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  MFEM_UNROLL(MQ1)
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u[qz] += DQQ(dx,qy,qz,vd) * Bty(dy,qy);
                  }
               }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz) { DDQ(dx,dy,qz,vd) = u[qz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd,z,VDIM)
      {
         const int nx = (vd == 0) ? D1D : D1D-1;
         const int ny = (vd == 1) ? D1D : D1D-1;
         const int nz = (vd == 2) ? D1D : D1D-1;
         DeviceTensor<5> Yxyz(y, nx, ny, nz, VDIM, NE);
         DeviceMatrix Btz = (vd == 2) ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               double u[D1D];
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { u[dz] = 0.0; }
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  MFEM_UNROLL(MD1)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += DDQ(dx,dy,qz,vd) * Btz(dz,qz);
                  }
               }
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < nz; ++dz) { Yxyz(dx,dy,dz,vd,e) += u[dz]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

void PAHdivMassApply(const int dim,
                     const int D1D,
                     const int Q1D,
                     const int NE,
                     const bool symmetric,
                     const Array<double> &Bo,
                     const Array<double> &Bc,
                     const Array<double> &Bot,
                     const Array<double> &Bct,
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

// PA H(div) div-div assemble 2D kernel
// NOTE: this is identical to PACurlCurlSetup3D
static void PADivDivSetup2D(const int Q1D,
                            const int NE,
                            const Array<double> &w,
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
                             J21 * (J12 * J33 - J32 * J13) +
                             J31 * (J12 * J23 - J22 * J13);
         y(q,e) = W[q] * coeff(q, e) / detJ;
      }
   });
}

static void PADivDivApply2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &Bo_,
                            const Array<double> &Gc_,
                            const Array<double> &Bot_,
                            const Array<double> &Gct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
                            const Array<double> &Bo_,
                            const Array<double> &Gc_,
                            const Array<double> &Bot_,
                            const Array<double> &Gct_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
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

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);

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
                                       const Array<double> &Bo_,
                                       const Array<double> &Gc_,
                                       const Vector &op_,
                                       Vector &diag_)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto diag = Reshape(diag_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
                                       const Array<double> &Bo_,
                                       const Array<double> &Gc_,
                                       const Vector &op_,
                                       Vector &diag_)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
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

static void PADivL2Setup3D(const int Q1D,
                           const int NE,
                           const Array<double> &w,
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

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);

   if (test_el->GetMapType() == FiniteElement::INTEGRAL)
   {
      const GeometricFactors *geom =
         mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS);
      coeff /= geom->detJ;
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

// Apply to x corresponding to DOFs in H(div) (trial), whose divergence is
// integrated against L_2 test functions corresponding to y.
static void PAHdivL2Apply3D(const int D1D,
                            const int Q1D,
                            const int L2D1D,
                            const int NE,
                            const Array<double> &Bo_,
                            const Array<double> &Gc_,
                            const Array<double> &L2Bot_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto L2Bot = Reshape(L2Bot_.Read(), L2D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 3*(D1D-1)*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), L2D1D, L2D1D, L2D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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

// Apply to x corresponding to DOFs in H(div) (trial), whose divergence is
// integrated against L_2 test functions corresponding to y.
static void PAHdivL2Apply2D(const int D1D,
                            const int Q1D,
                            const int L2D1D,
                            const int NE,
                            const Array<double> &Bo_,
                            const Array<double> &Gc_,
                            const Array<double> &L2Bot_,
                            const Vector &op_,
                            const Vector &x_,
                            Vector &y_)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   auto Gc = Reshape(Gc_.Read(), Q1D, D1D);
   auto L2Bot = Reshape(L2Bot_.Read(), L2D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(y_.ReadWrite(), L2D1D, L2D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
                                     const Array<double> &L2Bo_,
                                     const Array<double> &Gct_,
                                     const Array<double> &Bot_,
                                     const Vector &op_,
                                     const Vector &x_,
                                     Vector &y_)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
   constexpr static int VDIM = 3;

   auto L2Bo = Reshape(L2Bo_.Read(), Q1D, L2D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), L2D1D, L2D1D, L2D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 3*(D1D-1)*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
                                     const Array<double> &L2Bo_,
                                     const Array<double> &Gct_,
                                     const Array<double> &Bot_,
                                     const Vector &op_,
                                     const Vector &x_,
                                     Vector &y_)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HDIV_MAX_D1D;
   constexpr static int MAX_Q1D = HDIV_MAX_Q1D;

   auto L2Bo = Reshape(L2Bo_.Read(), Q1D, L2D1D);
   auto Gct = Reshape(Gct_.Read(), D1D, Q1D);
   auto Bot = Reshape(Bot_.Read(), D1D-1, Q1D);
   auto op = Reshape(op_.Read(), Q1D, Q1D, NE);
   auto x = Reshape(x_.Read(), L2D1D, L2D1D, NE);
   auto y = Reshape(y_.ReadWrite(), 2*(D1D-1)*D1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
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
                                             const Array<double> &L2Bo_,
                                             const Array<double> &Gct_,
                                             const Array<double> &Bot_,
                                             const Vector &op_,
                                             const Vector &D_,
                                             Vector &diag_)
{
   MFEM_VERIFY(D1D <= HDIV_MAX_D1D, "Error: D1D > HDIV_MAX_D1D");
   MFEM_VERIFY(Q1D <= HDIV_MAX_Q1D, "Error: Q1D > HDIV_MAX_Q1D");
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
                                             const Array<double> &L2Bo_,
                                             const Array<double> &Gct_,
                                             const Array<double> &Bot_,
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
