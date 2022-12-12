// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_SetupGradPA_C0_2D(const double lim_normal,
                            const ConstDeviceCube &LD,
                            const bool const_c0,
                            const DeviceTensor<3, const double> &C0,
                            const int NE,
                            const DeviceTensor<5, const double> &J,
                            const ConstDeviceMatrix &W,
                            const ConstDeviceMatrix &b,
                            const ConstDeviceMatrix &bld,
                            const DeviceTensor<4, const double> &X0,
                            const DeviceTensor<4, const double> &X1,
                            DeviceTensor<5> &H0,
                            const bool exp_lim,
                            const int d1d,
                            const int q1d,
                            const int max)
{
   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int DIM = 2;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double B[MQ1*MD1];
      MFEM_SHARED double BLD[MQ1*MD1];

      MFEM_SHARED double XY[NBZ][MD1*MD1];
      MFEM_SHARED double DQ[NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[NBZ][MQ1*MQ1];

      MFEM_SHARED double XY0[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ0[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ0[2][NBZ][MQ1*MQ1];

      MFEM_SHARED double XY1[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ1[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ1[2][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,LD,XY);
      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X0,XY0);
      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X1,XY1);

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,B);
      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,bld,BLD);

      kernels::internal::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BLD,XY,DQ);
      kernels::internal::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BLD,DQ,QQ);

      kernels::internal::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,B,XY0,DQ0);
      kernels::internal::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ0,QQ0);

      kernels::internal::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,B,XY1,DQ1);
      kernels::internal::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ1,QQ1);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = W(qx,qy) * detJtr;
            const double coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            const double weight_m = weight * lim_normal * coeff0;

            double D, grad_grad[4];
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ,D);
            const double dist = D; // GetValues, default comp set to 0

            if (!exp_lim)
            {
               // lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
               // d2.Diag(1.0 / (dist * dist), x.Size());
               const double c = 1.0 / (dist * dist);
               kernels::Diag<2>(c, grad_grad);
            }
            else
            {
               double p0[2], p1[2], tmp[2];
               kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ0,p0);
               kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ1,p1);
               kernels::Subtract<2>(1.0, p1, p0, tmp);
               double dsq = kernels::DistanceSquared<2>(p1,p0);
               double dist_squared = dist*dist;
               double dist_squared_squared = dist_squared*dist_squared;
               double f = exp(10.0*((dsq / dist_squared)-1.0));
               grad_grad[0] = ((400.0*tmp[0]*tmp[0]*f)/dist_squared_squared)+
                              (20.0*f/dist_squared);
               grad_grad[1] = (400.0*tmp[0]*tmp[1]*f)/dist_squared_squared;
               grad_grad[2] = grad_grad[1];
               grad_grad[3] = ((400.0*tmp[1]*tmp[1]*f)/dist_squared_squared)+
                              (20.0*f/dist_squared);
            }
            ConstDeviceMatrix gg(grad_grad,DIM,DIM);

            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  H0(i,j,qx,qy,e) = weight_m * gg(i,j);
               }
            }
         }
      }
   });
}

void TMOP_Integrator::AssembleGradPA_C0_2D(const Vector &x) const
{
   const int NE = PA.ne;
   constexpr int DIM = 2;
   const int D1D = PA.maps_lim->ndof;
   const int Q1D = PA.maps_lim->nqpt;

   const double ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const auto C0 = const_c0 ?
                   Reshape(PA.C0.Read(), 1, 1, 1) :
                   Reshape(PA.C0.Read(), Q1D, Q1D, NE);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), Q1D, Q1D);
   const auto B = Reshape(PA.maps->B.Read(), Q1D, D1D);
   const auto BLD = Reshape(PA.maps_lim->B.Read(), Q1D, D1D);
   const auto LD = Reshape(PA.LD.Read(), D1D, D1D, NE);
   const auto X0 = Reshape(PA.X0.Read(), D1D, D1D, DIM, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, DIM, NE);
   auto H0 = Reshape(PA.H0.Write(), DIM, DIM, Q1D, Q1D, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   decltype(&TMOP_SetupGradPA_C0_2D<>) ker = TMOP_SetupGradPA_C0_2D;
#ifndef MFEM_USE_JIT
   const int d=D1D, q=Q1D;
   if (d==2 && q==2) { ker = TMOP_SetupGradPA_C0_2D<2,2>; }
   if (d==2 && q==3) { ker = TMOP_SetupGradPA_C0_2D<2,3>; }
   if (d==2 && q==4) { ker = TMOP_SetupGradPA_C0_2D<2,4>; }
   if (d==2 && q==5) { ker = TMOP_SetupGradPA_C0_2D<2,5>; }
   if (d==2 && q==6) { ker = TMOP_SetupGradPA_C0_2D<2,6>; }

   if (d==3 && q==3) { ker = TMOP_SetupGradPA_C0_2D<3,3>; }
   if (d==3 && q==4) { ker = TMOP_SetupGradPA_C0_2D<3,4>; }
   if (d==3 && q==5) { ker = TMOP_SetupGradPA_C0_2D<3,5>; }
   if (d==3 && q==6) { ker = TMOP_SetupGradPA_C0_2D<3,6>; }

   if (d==4 && q==4) { ker = TMOP_SetupGradPA_C0_2D<4,4>; }
   if (d==4 && q==5) { ker = TMOP_SetupGradPA_C0_2D<4,5>; }
   if (d==4 && q==6) { ker = TMOP_SetupGradPA_C0_2D<4,6>; }

   if (d==5 && q==5) { ker = TMOP_SetupGradPA_C0_2D<5,5>; }
   if (d==5 && q==6) { ker = TMOP_SetupGradPA_C0_2D<5,6>; }
#endif
   ker(ln,LD,const_c0,C0,NE,J,W,B,BLD,X0,X,H0,exp_lim,D1D,Q1D,4);
}

} // namespace mfem
