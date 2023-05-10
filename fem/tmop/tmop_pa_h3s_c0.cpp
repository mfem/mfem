// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_Kernel_C0_3D,
                           const double lim_normal,
                           const Vector &lim_dist,
                           const Vector &c0_,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &bld_,
                           const Vector &x0_,
                           const Vector &x1_,
                           Vector &h0_,
                           const bool exp_lim,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const bool const_c0 = c0_.Size() == 1;
   const auto C0 = const_c0 ?
                   Reshape(c0_.Read(), 1, 1, 1, 1) :
                   Reshape(c0_.Read(), Q1D, Q1D, Q1D, NE);
   const auto LD = Reshape(lim_dist.Read(), D1D, D1D, D1D, NE);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto bld = Reshape(bld_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto X0 = Reshape(x0_.Read(), D1D, D1D, D1D, DIM, NE);
   const auto X1 = Reshape(x1_.Read(), D1D, D1D, D1D, DIM, NE);

   auto H0 = Reshape(h0_.Write(), DIM, DIM, Q1D, Q1D, Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED double B[MQ1*MD1];
      MFEM_SHARED double sBLD[MQ1*MD1];
      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,bld,sBLD);
      ConstDeviceMatrix BLD(sBLD, D1D, Q1D);

      MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
      DeviceCube DDD(sm0, MD1,MD1,MD1);
      DeviceCube DDQ(sm1, MD1,MD1,MQ1);
      DeviceCube DQQ(sm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(sm1, MQ1,MQ1,MQ1);

      MFEM_SHARED double DDD0[3][MD1*MD1*MD1];
      MFEM_SHARED double DDQ0[3][MD1*MD1*MQ1];
      MFEM_SHARED double DQQ0[3][MD1*MQ1*MQ1];
      MFEM_SHARED double QQQ0[3][MQ1*MQ1*MQ1];

      MFEM_SHARED double DDD1[3][MD1*MD1*MD1];
      MFEM_SHARED double DDQ1[3][MD1*MD1*MQ1];
      MFEM_SHARED double DQQ1[3][MD1*MQ1*MQ1];
      MFEM_SHARED double QQQ1[3][MQ1*MQ1*MQ1];

      kernels::internal::LoadX(e,D1D,LD,DDD);
      kernels::internal::LoadX<MD1>(e,D1D,X0,DDD0);
      kernels::internal::LoadX<MD1>(e,D1D,X1,DDD1);

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,B);

      kernels::internal::EvalX(D1D,Q1D,BLD,DDD,DDQ);
      kernels::internal::EvalY(D1D,Q1D,BLD,DDQ,DQQ);
      kernels::internal::EvalZ(D1D,Q1D,BLD,DQQ,QQQ);

      kernels::internal::EvalX<MD1,MQ1>(D1D,Q1D,B,DDD0,DDQ0);
      kernels::internal::EvalY<MD1,MQ1>(D1D,Q1D,B,DDQ0,DQQ0);
      kernels::internal::EvalZ<MD1,MQ1>(D1D,Q1D,B,DQQ0,QQQ0);

      kernels::internal::EvalX<MD1,MQ1>(D1D,Q1D,B,DDD1,DDQ1);
      kernels::internal::EvalY<MD1,MQ1>(D1D,Q1D,B,DDQ1,DQQ1);
      kernels::internal::EvalZ<MD1,MQ1>(D1D,Q1D,B,DQQ1,QQQ1);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double *Jtr = &J(0,0,qx,qy,qz,e);
               const double detJtr = kernels::Det<3>(Jtr);
               const double weight = W(qx,qy,qz) * detJtr;
               const double coeff0 = const_c0 ? C0(0,0,0,0) : C0(qx,qy,qz,e);
               const double weight_m = weight * lim_normal * coeff0;

               double D, p0[3], p1[3];
               kernels::internal::PullEval(qx,qy,qz,QQQ,D);
               kernels::internal::PullEval<MQ1>(Q1D,qx,qy,qz,QQQ0,p0);
               kernels::internal::PullEval<MQ1>(Q1D,qx,qy,qz,QQQ1,p1);

               const double dist = D; // GetValues, default comp set to 0

               // lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);

               double grad_grad[9];

               if (!exp_lim)
               {
                  // d2.Diag(1.0 / (dist * dist), x.Size());
                  const double c = 1.0 / (dist * dist);
                  kernels::Diag<3>(c, grad_grad);
               }
               else
               {
                  double tmp[3];
                  kernels::Subtract<3>(1.0, p1, p0, tmp);
                  double dsq = kernels::DistanceSquared<3>(p1,p0);
                  double dist_squared = dist*dist;
                  double dist_squared_squared = dist_squared*dist_squared;
                  double f = exp(10.0*((dsq / dist_squared)-1.0));
                  grad_grad[0] = ((400.0*tmp[0]*tmp[0]*f)/dist_squared_squared)+
                                 (20.0*f/dist_squared);
                  grad_grad[1] = (400.0*tmp[0]*tmp[1]*f)/dist_squared_squared;
                  grad_grad[2] = (400.0*tmp[0]*tmp[2]*f)/dist_squared_squared;
                  grad_grad[3] = grad_grad[1];
                  grad_grad[4] = ((400.0*tmp[1]*tmp[1]*f)/dist_squared_squared)+
                                 (20.0*f/dist_squared);
                  grad_grad[5] = (400.0*tmp[1]*tmp[2]*f)/dist_squared_squared;
                  grad_grad[6] = grad_grad[2];
                  grad_grad[7] = grad_grad[5];
                  grad_grad[8] = ((400.0*tmp[2]*tmp[2]*f)/dist_squared_squared)+
                                 (20.0*f/dist_squared);
               }
               ConstDeviceMatrix gg(grad_grad,DIM,DIM);

               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     H0(i,j,qx,qy,qz,e) = weight_m * gg(i,j);
                  }
               }
            }
         }
      }
   });
}

void TMOP_Integrator::AssembleGradPA_C0_3D(const Vector &X) const
{
   const int N = PA.ne;
   const int D1D = PA.maps_lim->ndof;
   const int Q1D = PA.maps_lim->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double ln = lim_normal;
   const Vector &LD = PA.LD;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &W = PA.ir->GetWeights();
   const Array<double> &B   = PA.maps->B;
   const Array<double> &BLD = PA.maps_lim->B;
   const Vector &C0 = PA.C0;
   const Vector &X0 = PA.X0;
   Vector &H0 = PA.H0;

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_Kernel_C0_3D,id,ln,LD,C0,N,J,W,B,BLD,X0,X,
                           H0,exp_lim);
}

} // namespace mfem
