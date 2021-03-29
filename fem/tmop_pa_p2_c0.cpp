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

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_C0_2D,
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
                           Vector &y_,
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

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
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

      kernels::LoadX<MD1,NBZ>(e,D1D,LD,XY);
      kernels::LoadX<MD1,NBZ>(e,D1D,X0,XY0);
      kernels::LoadX<MD1,NBZ>(e,D1D,X1,XY1);

      kernels::LoadB<MD1,MQ1>(D1D,Q1D,b,B);
      kernels::LoadB<MD1,MQ1>(D1D,Q1D,bld,BLD);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BLD,XY,DQ);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BLD,DQ,QQ);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,B,XY0,DQ0);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ0,QQ0);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,B,XY1,DQ1);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ1,QQ1);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = W(qx,qy) * detJtr;

            double ld, p0[2], p1[2];
            const double coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            kernels::PullEval<MQ1,NBZ>(qx,qy,QQ,ld);
            kernels::PullEval<MQ1,NBZ>(qx,qy,QQ0,p0);
            kernels::PullEval<MQ1,NBZ>(qx,qy,QQ1,p1);

            const double dist = ld; // GetValues, default comp set to 0

            double d1[2];
            // Eval_d1
            // subtract(1.0 / (dist * dist), x, x0, d1);
            // z = a * (x - y)
            // grad = a * (x - x0)
            const double a = 1.0 / (dist * dist);
            const double w = weight * lim_normal * coeff0;
            kernels::Subtract<2>(w*a, p1, p0, d1);
            kernels::PushEval<MQ1,NBZ>(qx,qy,d1,QQ0);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::LoadBt<MD1,MQ1>(D1D,Q1D,b,B);
      kernels::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,B,QQ0,DQ0);
      kernels::EvalYt<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ0,Y,e);
   });
}

void TMOP_Integrator::AddMultPA_C0_2D(const Vector &X, Vector &Y) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double ln = lim_normal;
   const Vector &LD = PA.LD;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &W   = PA.ir->GetWeights();
   const Array<double> &B   = PA.maps->B;
   const Array<double> &BLD = PA.maps_lim->B;
   const Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_C0_2D,id,ln,LD,C0,N,J,W,B,BLD,X0,X,Y);
}

} // namespace mfem
