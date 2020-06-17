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

#include "tmop.hpp"
#include "tmop_pa.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_C0_2D,
                           const double lim_normal,
                           const double dist,
                           const Vector &c0_,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const Vector &x0_,
                           const Vector &x_,
                           const Vector &r_,
                           Vector &c_,
                           const int d1d,
                           const int q1d)
{
   const bool const_c0 = c0_.Size() == 1;
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C0 = const_c0 ?
                   Reshape(c0_.Read(), 1, 1, 1) :
                   Reshape(c0_.Read(), Q1D, Q1D, NE);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto X0 = Reshape(x0_.Read(), D1D, D1D, DIM, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   const auto R = Reshape(r_.Read(), D1D, D1D, DIM, NE);
   //const auto H0 = Reshape(p_.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);
   auto Y = Reshape(c_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];

      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[2][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,R,XY);
      kernels::LoadBG<MD1,MQ1>(D1D, Q1D, b, g, BG);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double B[4];

            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = W(qx,qy) * detJtr;
            const double coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);

            // Jrt = Jtr^{-1}
            //double Jrt[4];
            //kernels::CalcInverse<2>(Jtr, Jrt);

            // Xh = X^T . Sh
            double Xh[2];
            kernels::PullEvalXY<MQ1,NBZ>(qx,qy,QQ,Xh);
            //A[0] = Xh[0]; A[1] = Xh[1];
            //A[2] = Xh[0]; A[3] = Xh[1];

            const double weight_m = weight * lim_normal * coeff0;
            //lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
            // d2.Diag(1.0 / (dist * dist), x.Size());
            const double c = 1.0 / (dist * dist);
            double grad_grad[4];
            kernels::Diag<2>(c, grad_grad);
            ConstDeviceMatrix gg(grad_grad,2,2);
            DeviceMatrix bb(B,2,2);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  bb(i,j) = weight_m * gg(i,j);
               }
            }
            double p2[2];
            // p2 = B . Xh
            kernels::Mult(2,2,B,Xh,p2);
            kernels::PushEvalXY<MQ1,NBZ>(qx,qy,p2,QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
      kernels::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::EvalYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
   });
}

void TMOP_Integrator::AddMultGradPA_C0_2D(const Vector &X, const Vector &R,
                                          Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   //const Vector &A0 = PA.A0;
   const Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;

   const double l = lim_normal;
   lim_dist->HostRead();
   MFEM_VERIFY(lim_dist, "Error");
   const double d = lim_dist->operator ()(0);

   if (KAddMultGradPA_Kernel_C0_2D.Find(id))
   {
      return KAddMultGradPA_Kernel_C0_2D.At(id)(l,d,C0,N,J,W,B,G,X0,X,R,C,0,0);
   }
   else
   {
      constexpr int T_MAX = 8;
      MFEM_VERIFY(D1D <= MAX_D1D && Q1D <= MAX_Q1D, "Max size error!");
      return AddMultGradPA_Kernel_C0_2D<0,0,T_MAX>(l,d,C0,N,J,W,B,G,X0,X,R,C,D1D,Q1D);
   }
}

} // namespace mfem
