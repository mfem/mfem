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
#include "../general/debug.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_C0_3D,
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
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C0 = const_c0 ?
                   Reshape(c0_.Read(), 1, 1, 1, 1) :
                   Reshape(c0_.Read(), Q1D, Q1D, Q1D, NE);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto R = Reshape(r_.Read(), D1D, D1D, D1D, DIM, NE);
   auto Y = Reshape(c_.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];

      MFEM_SHARED double DDD[3][MD1*MD1*MD1];
      MFEM_SHARED double DDQ[3][MD1*MD1*MQ1];
      MFEM_SHARED double DQQ[3][MD1*MQ1*MQ1];
      MFEM_SHARED double QQQ[3][MQ1*MQ1*MQ1];

      kernels::LoadX<MD1>(e,D1D,R,DDD);
      kernels::LoadBG<MD1,MQ1>(D1D, Q1D, b, g, BG);

      kernels::EvalX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::EvalY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::EvalZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double *Jtr = &J(0,0,qx,qy,qz,e);
               const double detJtr = kernels::Det<DIM>(Jtr);
               const double weight = W(qx,qy,qz) * detJtr;
               const double coeff0 = const_c0 ? C0(0,0,0,0) : C0(qx,qy,qz,e);
               const double weight_m = weight * lim_normal * coeff0;

               // Xh = X^T . Sh
               double Xh[3];
               kernels::PullEvalXYZ<MQ1>(qx,qy,qz,QQQ,Xh);

               //lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
               // d2.Diag(1.0 / (dist * dist), x.Size());
               double B[9];
               const double c = 1.0 / (dist * dist);
               double grad_grad[9];
               kernels::Diag<3>(c, grad_grad);
               ConstDeviceMatrix gg(grad_grad,3,3);
               DeviceMatrix bb(B,3,3);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     bb(i,j) = weight_m * gg(i,j);
                  }
               }

               // p2 = B . Xh
               double p2[3];
               kernels::Mult(3,3,B,Xh,p2);
               kernels::PushEvalXYZ<MQ1>(qx,qy,qz,p2,QQQ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
      kernels::EvalXt<MD1,MQ1>(D1D,Q1D,BG,QQQ,DQQ);
      kernels::EvalYt<MD1,MQ1>(D1D,Q1D,BG,DQQ,DDQ);
      kernels::EvalZt<MD1,MQ1>(D1D,Q1D,BG,DDQ,Y,e);
   });
}

void TMOP_Integrator::AddMultGradPA_C0_3D(const Vector &X, const Vector &R,
                                          Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double ln = lim_normal;
   const double ld = lim_dist->HostRead()[0];
   const DenseTensor &J = PA.Jtr;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultGradPA_Kernel_C0_3D,id,
                           ln,ld,C0,N,J,W,B,G,X0,X,R,C);
}

} // namespace mfem
