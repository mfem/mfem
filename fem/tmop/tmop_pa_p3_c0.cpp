// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_C0_3D,
                           const real_t lim_normal,
                           const Vector &lim_dist,
                           const Vector &c0_,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<real_t> &w_,
                           const Array<real_t> &b_,
                           const Array<real_t> &bld_,
                           const Vector &x0_,
                           const Vector &x1_,
                           Vector &y_,
                           const bool exp_lim,
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
   const auto LD = Reshape(lim_dist.Read(), D1D, D1D, D1D, NE);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto bld = Reshape(bld_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto X0 = Reshape(x0_.Read(), D1D, D1D, D1D, DIM, NE);
   const auto X1 = Reshape(x1_.Read(), D1D, D1D, D1D, DIM, NE);

   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t B[MQ1*MD1];
      MFEM_SHARED real_t sBLD[MQ1*MD1];
      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,bld,sBLD);
      ConstDeviceMatrix BLD(sBLD, D1D, Q1D);

      MFEM_SHARED real_t sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED real_t sm1[MDQ*MDQ*MDQ];
      DeviceCube DDD(sm0, MD1,MD1,MD1);
      DeviceCube DDQ(sm1, MD1,MD1,MQ1);
      DeviceCube DQQ(sm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(sm1, MQ1,MQ1,MQ1);

      MFEM_SHARED real_t DDD0[3][MD1*MD1*MD1];
      MFEM_SHARED real_t DDQ0[3][MD1*MD1*MQ1];
      MFEM_SHARED real_t DQQ0[3][MD1*MQ1*MQ1];
      MFEM_SHARED real_t QQQ0[3][MQ1*MQ1*MQ1];

      MFEM_SHARED real_t DDD1[3][MD1*MD1*MD1];
      MFEM_SHARED real_t DDQ1[3][MD1*MD1*MQ1];
      MFEM_SHARED real_t DQQ1[3][MD1*MQ1*MQ1];
      MFEM_SHARED real_t QQQ1[3][MQ1*MQ1*MQ1];

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
               const real_t *Jtr = &J(0,0,qx,qy,qz,e);
               const real_t detJtr = kernels::Det<3>(Jtr);
               const real_t weight = W(qx,qy,qz) * detJtr;

               real_t D, p0[3], p1[3];
               const real_t coeff0 = const_c0 ? C0(0,0,0,0) : C0(qx,qy,qz,e);
               kernels::internal::PullEval(qx,qy,qz,QQQ,D);
               kernels::internal::PullEval<MQ1>(Q1D,qx,qy,qz,QQQ0,p0);
               kernels::internal::PullEval<MQ1>(Q1D,qx,qy,qz,QQQ1,p1);

               real_t d1[3];
               // Eval_d1 (Quadratic Limiter)
               // subtract(1.0 / (dist * dist), x, x0, d1);
               // z = a * (x - y)
               // grad = a * (x - x0)

               // Eval_d1 (Exponential Limiter)
               // real_t dist_squared = dist*dist;
               // subtract(20.0*exp(10.0*((x.DistanceSquaredTo(x0) / dist_squared) - 1.0)) /
               // dist_squared, x, x0, d1);
               // z = a * (x - y)
               // grad = a * (x - x0)
               const real_t dist = D; // GetValues, default comp set to 0
               real_t a = 0.0;
               const real_t w = weight * lim_normal * coeff0;
               const real_t dist_squared = dist * dist;

               if (!exp_lim)
               {
                  a =  1.0 / dist_squared;
               }
               else
               {
                  real_t dsq = kernels::DistanceSquared<3>(p1,p0) / dist_squared;
                  a = 20.0*exp(10.0*(dsq - 1.0))/dist_squared;
               }

               kernels::Subtract<3>(w*a, p1, p0, d1);
               kernels::internal::PushEval<MQ1>(Q1D,qx,qy,qz,d1,QQQ0);
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBt<MD1,MQ1>(D1D,Q1D,b,B);
      kernels::internal::EvalXt<MD1,MQ1>(D1D,Q1D,B,QQQ0,DQQ0);
      kernels::internal::EvalYt<MD1,MQ1>(D1D,Q1D,B,DQQ0,DDQ0);
      kernels::internal::EvalZt<MD1,MQ1>(D1D,Q1D,B,DDQ0,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_C0_3D(const Vector &X, Vector &Y) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const real_t ln = lim_normal;
   const Vector &LD = PA.LD;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W = PA.ir->GetWeights();
   const Array<real_t> &B = PA.maps->B;
   const Array<real_t> &BLD = PA.maps_lim->B;
   MFEM_VERIFY(PA.maps_lim->ndof == D1D, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == Q1D, "");
   const Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;
   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_C0_3D,id,ln,LD,C0,N,J,W,B,BLD,X0,X,Y,
                           exp_lim);
}

} // namespace mfem
