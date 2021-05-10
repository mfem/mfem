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

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../fem/kernels.hpp"
#include "../../linalg/kernels.hpp"

using namespace mfem;

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

template<int T_D1D = 0, int T_Q1D = 0, int MAX_D1D = 0, int MAX_Q1D = 0>
static void Det2D(const int NE,
                  const double *b,
                  const double *g,
                  const double *x,
                  double *y,
                  const int vdim = 1,
                  const int d1d = 0,
                  const int q1d = 0)
{
   constexpr int DIM = 2;
   static constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x,  D1D, D1D, DIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double J[4];
            kernels::internal::PullGrad<MQ1,NBZ>(qx,qy,QQ,J);
            Y(qx,qy,e) = kernels::Det<2>(J);
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0, int MAX_D1D = 0, int MAX_Q1D = 0,
         bool SMEM = true>
static void Det3D(const int NE,
                  const double *b,
                  const double *g,
                  const double *x,
                  double *y,
                  const int vdim = 1,
                  const int d1d = 0,
                  const int q1d = 0,
                  Vector *d_buff = nullptr) // used only with SMEM = false
{
   constexpr int DIM = 3;
   static constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   static constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   static constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;
   static constexpr int MSZ = MDQ * MDQ * MDQ * 9;
   static constexpr int GRID = SMEM ? 0 : 128;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x, D1D, D1D, D1D, DIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, Q1D, NE);

   double *GM = nullptr;
   if (!SMEM)
   {
      d_buff->SetSize(2*MSZ*GRID);
      GM = d_buff->Write();
   }

   MFEM_FORALL_3D_GRID(e, NE, Q1D, Q1D, Q1D, GRID,
   {
      const int bid = MFEM_BLOCK_ID(x);
      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double SM0[SMEM?MSZ:1];
      MFEM_SHARED double SM1[SMEM?MSZ:1];
      double *lm0 = SMEM ? SM0 : GM + MSZ*bid;
      double *lm1 = SMEM ? SM1 : GM + MSZ*(GRID+bid);
      double (*DDD)[MD1*MD1*MD1] = (double (*)[MD1*MD1*MD1]) (lm0);
      double (*DDQ)[MD1*MD1*MQ1] = (double (*)[MD1*MD1*MQ1]) (lm1);
      double (*DQQ)[MD1*MQ1*MQ1] = (double (*)[MD1*MQ1*MQ1]) (lm0);
      double (*QQQ)[MQ1*MQ1*MQ1] = (double (*)[MQ1*MQ1*MQ1]) (lm1);

      kernels::internal::LoadX<MD1>(e,D1D,X,DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double J[9];
               kernels::internal::PullGrad<MQ1>(qx,qy,qz, QQQ, J);
               Y(qx,qy,qz,e) = kernels::Det<3>(J);
            }
         }
      }
   });
}

// Tensor-product evaluation of quadrature point determinants: dispatch
// function.
void TensorDeterminants(const int NE,
                        const int vdim,
                        const DofToQuad &maps,
                        const Vector &e_vec,
                        Vector &q_det,
                        Vector &d_buff)
{
   if (NE == 0) { return; }
   const int dim = maps.FE->GetDim();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const double *X = e_vec.Read();
   double *Y = q_det.Write();

   const int id = (vdim<<8) | (D1D<<4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x222: return Det2D<2,2>(NE,B,G,X,Y);
         case 0x223: return Det2D<2,3>(NE,B,G,X,Y);
         case 0x224: return Det2D<2,4>(NE,B,G,X,Y);
         case 0x226: return Det2D<2,6>(NE,B,G,X,Y);
         case 0x234: return Det2D<3,4>(NE,B,G,X,Y);
         case 0x236: return Det2D<3,6>(NE,B,G,X,Y);
         case 0x244: return Det2D<4,4>(NE,B,G,X,Y);
         case 0x246: return Det2D<4,6>(NE,B,G,X,Y);
         case 0x256: return Det2D<5,6>(NE,B,G,X,Y);
         default:
         {
            constexpr int MD = MAX_D1D;
            constexpr int MQ = MAX_Q1D;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                        << MQ << " 1D points are not supported!");
            Det2D<0,0,MD,MQ>(NE,B,G,X,Y,vdim,D1D,Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x324: return Det3D<2,4>(NE,B,G,X,Y);
         case 0x333: return Det3D<3,3>(NE,B,G,X,Y);
         case 0x335: return Det3D<3,5>(NE,B,G,X,Y);
         case 0x336: return Det3D<3,6>(NE,B,G,X,Y);
         default:
         {
            constexpr int MD = 6;
            constexpr int MQ = 6;
            // Highest orders that fit in shared mememory
            if (D1D <= MD && Q1D <= MQ)
            { return Det3D<0,0,MD,MQ>(NE,B,G,X,Y,vdim,D1D,Q1D); }
            // Last fall-back will use global memory
            return Det3D<0,0,MAX_D1D,MAX_Q1D,false>(
                      NE,B,G,X,Y,vdim,D1D,Q1D,&d_buff);
         }
      }
   }
   MFEM_ABORT("Kernel " << std::hex << id << std::dec << " not supported yet");
}

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem
