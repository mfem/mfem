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

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_C0_2D,
                           const double lim_normal,
                           const Vector &lim_dist,
                           const Vector &c0_,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<double> &w_,
                           const Array<double> &bld_,
                           Vector &h0_,
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
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto bld = Reshape(bld_.Read(), Q1D, D1D);

   auto H0 = Reshape(h0_.Write(), DIM, DIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BLD[MQ1*MD1];

      MFEM_SHARED double XY[NBZ][MD1*MD1];
      MFEM_SHARED double DQ[NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,LD,XY);

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,bld,BLD);

      kernels::internal::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BLD,XY,DQ);
      kernels::internal::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BLD,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = W(qx,qy) * detJtr;
            const double coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            const double weight_m = weight * lim_normal * coeff0;

            double D;
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ,D);
            const double dist = D; // GetValues, default comp set to 0

            // lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
            // d2.Diag(1.0 / (dist * dist), x.Size());
            const double c = 1.0 / (dist * dist);
            double grad_grad[4];
            kernels::Diag<2>(c, grad_grad);
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
   const double ln = lim_normal;
   const Vector &LD = PA.LD;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &W   = PA.ir->GetWeights();
   const Array<double> &BLD = PA.maps_lim->B;
   const Vector &C0 = PA.C0;
   Vector &H0 = PA.H0;

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_C0_2D,id,ln,LD,C0,N,J,W,BLD,H0);
}

} // namespace mfem
