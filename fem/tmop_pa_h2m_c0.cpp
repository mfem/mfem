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

#include "tmop.hpp"
#include "tmop_pa.hpp"
#include "linearform.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_C0_2D,
                           const int NE,
                           const Array<double> &b_,
                           const Vector &h0_,
                           const Vector &r_,
                           Vector &c_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto H0 = Reshape(h0_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto R = Reshape(r_.Read(), D1D, D1D, DIM, NE);

   auto Y = Reshape(c_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int DIM = 2;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double B[MQ1*MD1];

      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[2][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,R,XY);
      kernels::LoadB<MD1,MQ1>(D1D,Q1D,b,B);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,B,XY,DQ);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            // Xh = X^T . Sh
            double Xh[2];
            kernels::PullEval<MQ1,NBZ>(qx,qy,QQ,Xh);

            double B[4];
            DeviceMatrix H(B,2,2);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  H(i,j) = H0(i,j,qx,qy,e);
               }
            }

            // p2 = B . Xh
            double p2[2];
            kernels::Mult(2,2,B,Xh,p2);
            kernels::PushEval<MQ1,NBZ>(qx,qy,p2,QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::LoadBt<MD1,MQ1>(D1D,Q1D,b,B);
      kernels::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,B,QQ,DQ);
      kernels::EvalYt<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ,Y,e);
   });
}

void TMOP_Integrator::AddMultGradPA_C0_2D(const Vector &X, const Vector &R,
                                          Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const Array<double> &B = PA.maps->B;
   const Vector &H0 = PA.H0;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultGradPA_Kernel_C0_2D,id,N,B,H0,R,C);
}

} // namespace mfem
