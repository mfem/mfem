// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Internal header, included only by .cpp files

#include "fem.hpp"

#include "../general/forall.hpp"
#include "../fem/kernels.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{

namespace linearform_extension
{

template<int D1D, int Q1D> static
void VectorDomainLFGradIntegratorAssemble2D(const int vdim,
                                            const bool byVDIM,
                                            const int ND,
                                            const int NE,
                                            const double *marks,
                                            const double *b,
                                            const double *g,
                                            const int *idx,
                                            const double *jacobians,
                                            const double *weights,
                                            const Vector &coeff,
                                            double* __restrict y)
{
   constexpr int DIM = 2;
   const int cdim = vdim == 1 ? DIM : vdim;

   const bool cst_coeff = coeff.Size() == cdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto G = Reshape(g, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,cdim/DIM,1,1,1):
                  Reshape(F,DIM,cdim/DIM,Q1D,Q1D,NE);

   auto Y = Reshape(y,
                    byVDIM ? vdim : ND,
                    byVDIM ? ND : vdim);


   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      if (M(e) < 1.0) { return; }

      MFEM_SHARED double sBG[2][D1D*Q1D];
      const DeviceMatrix Bt(sBG[0],D1D,Q1D);
      const DeviceMatrix Gt(sBG[1],D1D,Q1D);

      MFEM_SHARED double sm0[2][Q1D*Q1D];
      const DeviceMatrix QQ0(sm0[0],Q1D,Q1D);
      const DeviceMatrix QQ1(sm0[1],Q1D,Q1D);

      MFEM_SHARED double sm1[2][Q1D*Q1D];
      const DeviceMatrix DQ0(sm1[0],D1D,Q1D);
      const DeviceMatrix DQ1(sm1[1],D1D,Q1D);

      for (int c = 0; c < cdim/DIM; ++c)
      {
         const double cst_val0 = C(0,c,0,0,0);
         const double cst_val1 = C(1,c,0,0,0);

         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               double Jloc[4],Jinv[4];
               Jloc[0] = J(qx,qy,0,0,e);
               Jloc[1] = J(qx,qy,1,0,e);
               Jloc[2] = J(qx,qy,0,1,e);
               Jloc[3] = J(qx,qy,1,1,e);
               const double detJ = kernels::Det<2>(Jloc);
               kernels::CalcInverse<2>(Jloc, Jinv);
               const double weight = W(qx,qy);
               const double u = cst_coeff ? cst_val0 : C(0,c,qx,qy,e);
               const double v = cst_coeff ? cst_val1 : C(1,c,qx,qy,e);
               QQ0(qy,qx) = Jinv[0]*u + Jinv[2]*v;
               QQ1(qy,qx) = Jinv[1]*u + Jinv[3]*v;
               QQ0(qy,qx) *= weight * detJ;
               QQ1(qy,qx) *= weight * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::LoadBGt(D1D,Q1D,B,G,Bt,Gt);
         kernels::internal::Atomic2DGradTranspose(D1D,Q1D,Bt,Gt,
                                                  QQ0,QQ1,DQ0,DQ1,
                                                  I,Y,c,e,byVDIM);
      }
   });
}

template<int D1D, int Q1D> static
void VectorDomainLFGradIntegratorAssemble3D(const int vdim,
                                            const bool byVDIM,
                                            const int ND,
                                            const int NE,
                                            const double *marks,
                                            const double *b,
                                            const double *g,
                                            const int *idx,
                                            const double *jacobians,
                                            const double *weights,
                                            const Vector &coeff,
                                            double* __restrict y)
{
   constexpr int DIM = 3;
   const int cdim = vdim == 1 ? DIM : vdim;

   const bool cst_coeff = coeff.Size() == cdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, Q1D,D1D);
   const auto G = Reshape(g, Q1D,D1D);
   const auto J = Reshape(jacobians, Q1D,Q1D,Q1D,DIM,DIM,NE);
   const auto W = Reshape(weights, Q1D,Q1D,Q1D);
   const auto I = Reshape(idx, D1D,D1D,D1D, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,cdim/DIM,1,1,1,1):
                  Reshape(F,DIM,cdim/DIM,Q1D,Q1D,Q1D,NE);

   auto Y = Reshape(y,
                    byVDIM ? vdim : ND,
                    byVDIM ? ND : vdim);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      if (M(e) < 1.0) { return; }

      MFEM_SHARED double sBG[2][Q1D*D1D];
      const DeviceMatrix Bt(sBG[0],D1D,Q1D);
      const DeviceMatrix Gt(sBG[1],D1D,Q1D);
      kernels::internal::LoadBGt(D1D,Q1D,B,G,Bt,Gt);

      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      const DeviceCube QQ0(sm0[0],Q1D,Q1D,Q1D);
      const DeviceCube QQ1(sm0[1],Q1D,Q1D,Q1D);
      const DeviceCube QQ2(sm0[2],Q1D,Q1D,Q1D);

      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      const DeviceCube QD0(sm1[0],Q1D,Q1D,D1D);
      const DeviceCube QD1(sm1[1],Q1D,Q1D,D1D);
      const DeviceCube QD2(sm1[2],Q1D,Q1D,D1D);

      const DeviceCube DD0(sm0[0],Q1D,D1D,D1D);
      const DeviceCube DD1(sm0[1],Q1D,D1D,D1D);
      const DeviceCube DD2(sm0[2],Q1D,D1D,D1D);

      for (int c = 0; c < cdim/DIM; ++c)
      {
         const double cst_val_0 = C(0,c,0,0,0,0);
         const double cst_val_1 = C(1,c,0,0,0,0);
         const double cst_val_2 = C(2,c,0,0,0,0);

         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  double Jloc[9],Jinv[9];
                  for (int col = 0; col < 3; col++)
                  {
                     for (int row = 0; row < 3; row++)
                     {
                        Jloc[row+3*col] = J(qx,qy,qz,row,col,e);
                     }
                  }
                  const double detJ = kernels::Det<3>(Jloc);
                  kernels::CalcInverse<3>(Jloc, Jinv);
                  const double weight = W(qx,qy,qz);
                  const double u = cst_coeff ? cst_val_0 : C(0,c,qx,qy,qz,e);
                  const double v = cst_coeff ? cst_val_1 : C(1,c,qx,qy,qz,e);
                  const double w = cst_coeff ? cst_val_2 : C(2,c,qx,qy,qz,e);
                  QQ0(qz,qy,qx) = Jinv[0]*u + Jinv[3]*v + Jinv[6]*w;
                  QQ1(qz,qy,qx) = Jinv[1]*u + Jinv[4]*v + Jinv[7]*w;
                  QQ2(qz,qy,qx) = Jinv[2]*u + Jinv[5]*v + Jinv[8]*w;
                  QQ0(qz,qy,qx) *= weight * detJ;
                  QQ1(qz,qy,qx) *= weight * detJ;
                  QQ2(qz,qy,qx) *= weight * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Atomic3DGrad(D1D,Q1D,Bt,Gt,
                                         QQ0,QQ1,QQ2,
                                         QD0,QD1,QD2,
                                         DD0,DD1,DD2,
                                         I,Y,c,e,byVDIM);
      }
   });
}

} // namespace linearform_extension

} // namespace internal

} // namespace mfem
