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
#include "../linalg/dinvariants.hpp"
#include "../general/debug.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_C0_2D,
                           const Vector &x0_,
                           const Vector &x1_,
                           const double lim_normal,
                           const double dist,
                           const Vector &c0_,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           Vector &h_,
                           const int d1d,
                           const int q1d)
{
   dbg("");
   const bool const_c0 = c0_.Size() == 1;

   constexpr int DIM = 2;
   constexpr int NBZ = 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C0 = const_c0 ?
                   Reshape(c0_.Read(), 1, 1, 1) :
                   Reshape(c0_.Read(), Q1D, Q1D, NE);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto X0 = Reshape(x0_.Read(), D1D, D1D, DIM, NE);
   const auto X1 = Reshape(x1_.Read(), D1D, D1D, DIM, NE);

   auto H = Reshape(h_.Write(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];

      MFEM_SHARED double XY0[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ0[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ0[2][NBZ][MQ1*MQ1];

      MFEM_SHARED double XY1[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ1[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ1[2][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,X0,XY0);
      kernels::LoadX<MD1,NBZ>(e,D1D,X1,XY1);

      kernels::LoadBG<MD1,MQ1>(D1D, Q1D, b, g, BG);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY0,DQ0);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ0,QQ0);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY1,DQ1);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ1,QQ1);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = W(qx,qy) * detJtr;

            double p0[2], p1[2];
            const double coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            kernels::PullEvalXY<MQ1,NBZ>(qx,qy,QQ0,p0);
            kernels::PullEvalXY<MQ1,NBZ>(qx,qy,QQ1,p1);

            const double weight_m = weight * lim_normal * coeff0;
            //lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
            // d2.Diag(1.0 / (dist * dist), x.Size());
            const double c = 1.0 / (dist * dist);
            double grad_grad[4];
            kernels::Diag<2>(c, grad_grad);

            ConstDeviceMatrix gg(grad_grad,DIM,DIM);
            //printf("\n\033[33mweight_m:%.8e, dist:%.8e\033[m",weight_m,dist);
            //printf("\n\033[33m%f, %f, %f, %f\033[m",gg(0,0),gg(0,1),gg(1,0),gg(1,1));

            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  for (int r = 0; r < DIM; r++)
                  {
                     for (int c = 0; c < DIM; c++)
                     {
                        const double h = 1.0;//gg(r,c);
                        H(r,c,i,j,qx,qy,e) = weight_m * h;
                     }
                  }
               }
            }
         } // qx
      } // qy
   });
}

void TMOP_Integrator::AssembleGradPA_C0_2D(const Vector &X) const
{
   MFEM_ABORT("Not used!");
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;
   Vector &H0 = PA.A0;

   const double l = lim_normal;
   lim_dist->HostRead();
   MFEM_VERIFY(lim_dist, "Error");
   const double d = lim_dist->operator ()(0);

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_C0_2D, id, X0,X,l,d,C0,N,J,W,B,G,H0);
}

} // namespace mfem
