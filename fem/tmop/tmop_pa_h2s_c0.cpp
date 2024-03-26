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

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_C0_2D,
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
                           Vector &h0_,
                           const bool exp_lim,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const bool const_c0 = c0_.Size() == 1;
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

   auto H0 = Reshape(h0_.Write(), DIM, DIM, Q1D, Q1D, NE);

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
            const real_t coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            const real_t weight_m = weight * lim_normal * coeff0;

            real_t D, p0[2], p1[2];
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ,D);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ0,p0);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ1,p1);

            const real_t dist = D; // GetValues, default comp set to 0

            // lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
            real_t grad_grad[4];

            if (!exp_lim)
            {
               // d2.Diag(1.0 / (dist * dist), x.Size());
               const real_t c = 1.0 / (dist * dist);
               kernels::Diag<2>(c, grad_grad);
            }
            else
            {
               real_t tmp[2];
               kernels::Subtract<2>(1.0, p1, p0, tmp);
               real_t dsq = kernels::DistanceSquared<2>(p1,p0);
               real_t dist_squared = dist*dist;
               real_t dist_squared_squared = dist_squared*dist_squared;
               real_t f = exp(10.0*((dsq / dist_squared)-1.0));
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

void TMOP_Integrator::AssembleGradPA_C0_2D(const Vector &X) const
{
   MFEM_CONTRACT_VAR(X);
   const int N = PA.ne;
   const int D1D = PA.maps_lim->ndof;
   const int Q1D = PA.maps_lim->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const real_t ln = lim_normal;
   const Vector &LD = PA.LD;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W   = PA.ir->GetWeights();
   const Array<real_t> &B   = PA.maps->B;
   const Array<real_t> &BLD = PA.maps_lim->B;
   const Vector &C0 = PA.C0;
   const Vector &X0 = PA.X0;
   Vector &H0 = PA.H0;

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_C0_2D,id,ln,LD,C0,N,J,W,B,BLD,X0,X,H0,
                           exp_lim);
}

} // namespace mfem
