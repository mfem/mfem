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

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_C0_2D,
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

   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C0 = const_c0 ?
                   Reshape(c0_.Read(), 1, 1, 1) :
                   Reshape(c0_.Read(), Q1D, Q1D, NE);
   const auto LD = Reshape(lim_dist.Read(), D1D, D1D, NE);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto bld = Reshape(bld_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto X0 = Reshape(x0_.Read(), D1D, D1D, DIM, NE);
   const auto X1 = Reshape(x1_.Read(), D1D, D1D, DIM, NE);

   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED real_t B[MQ1*MD1];
      MFEM_SHARED real_t BLD[MQ1*MD1];

      MFEM_SHARED real_t XY[NBZ][MD1*MD1];
      MFEM_SHARED real_t DQ[NBZ][MD1*MQ1];
      MFEM_SHARED real_t QQ[NBZ][MQ1*MQ1];

      MFEM_SHARED real_t XY0[2][NBZ][MD1*MD1];
      MFEM_SHARED real_t DQ0[2][NBZ][MD1*MQ1];
      MFEM_SHARED real_t QQ0[2][NBZ][MQ1*MQ1];

      MFEM_SHARED real_t XY1[2][NBZ][MD1*MD1];
      MFEM_SHARED real_t DQ1[2][NBZ][MD1*MQ1];
      MFEM_SHARED real_t QQ1[2][NBZ][MQ1*MQ1];

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
            const real_t *Jtr = &J(0,0,qx,qy,e);
            const real_t detJtr = kernels::Det<2>(Jtr);
            const real_t weight = W(qx,qy) * detJtr;

            real_t ld, p0[2], p1[2];
            const real_t coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ,ld);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ0,p0);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ1,p1);

            const real_t dist = ld; // GetValues, default comp set to 0

            real_t d1[2];
            // Eval_d1 (Quadratic Limiter)
            // subtract(1.0 / (dist * dist), x, x0, d1);
            // z = a * (x - y)
            // grad = a * (x - x0)

            // Eval_d1 (Exponential Limiter)
            // double dist_squared = dist*dist;
            // subtract(20.0*exp(10.0*((x.DistanceSquaredTo(x0) / dist_squared) - 1.0)) /
            // dist_squared, x, x0, d1);
            // z = a * (x - y)
            // grad = a * (x - x0)

            real_t a = 0.0;
            const real_t w = weight * lim_normal * coeff0;
            const real_t dist_squared = dist * dist;

            if (!exp_lim)
            {
               a =  1.0 / dist_squared;
            }
            else
            {
               real_t dsq = kernels::DistanceSquared<2>(p1,p0) / dist_squared;
               a = 20.0*exp(10.0*(dsq - 1.0))/dist_squared;
            }
            kernels::Subtract<2>(w*a, p1, p0, d1);
            kernels::internal::PushEval<MQ1,NBZ>(Q1D,qx,qy,d1,QQ0);


         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::LoadBt<MD1,MQ1>(D1D,Q1D,b,B);
      kernels::internal::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,B,QQ0,DQ0);
      kernels::internal::EvalYt<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ0,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_C0_2D(const Vector &X, Vector &Y) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const real_t ln = lim_normal;
   const Vector &LD = PA.LD;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W   = PA.ir->GetWeights();
   const Array<real_t> &B   = PA.maps->B;
   const Array<real_t> &BLD = PA.maps_lim->B;
   MFEM_VERIFY(PA.maps_lim->ndof == D1D, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == Q1D, "");
   const Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;
   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_C0_2D,id,ln,LD,C0,N,J,W,B,BLD,X0,X,Y,
                           exp_lim);
}

} // namespace mfem
