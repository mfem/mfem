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
void TMOP_SetupGradPA_C0_3D(const double lim_normal,
                            const DeviceTensor<4, const double> &LD,
                            const bool const_c0,
                            const DeviceTensor<4, const double> &C0,
                            const int NE,
                            const DeviceTensor<6, const double> &J,
                            const ConstDeviceCube &W,
                            const ConstDeviceMatrix &bld,
                            DeviceTensor<6> &H0,
                            const int d1d,
                            const int q1d,
                            const int max)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED double sBLD[MQ1*MD1];
      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,bld,sBLD);
      ConstDeviceMatrix BLD(sBLD, D1D, Q1D);

      MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
      DeviceCube DDD(sm0, MD1,MD1,MD1);
      DeviceCube DDQ(sm1, MD1,MD1,MQ1);
      DeviceCube DQQ(sm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(sm1, MQ1,MQ1,MQ1);

      kernels::internal::LoadX(e,D1D,LD,DDD);

      kernels::internal::EvalX(D1D,Q1D,BLD,DDD,DDQ);
      kernels::internal::EvalY(D1D,Q1D,BLD,DDQ,DQQ);
      kernels::internal::EvalZ(D1D,Q1D,BLD,DQQ,QQQ);

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

               double D;
               kernels::internal::PullEval(qx,qy,qz,QQQ,D);
               const double dist = D; // GetValues, default comp set to 0

               // lim_func->Eval_d2(p1, p0, d_vals(q), grad_grad);
               // d2.Diag(1.0 / (dist * dist), x.Size());
               const double c = 1.0 / (dist * dist);
               double grad_grad[9];
               kernels::Diag<3>(c, grad_grad);
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
   const int NE = PA.ne;
   constexpr int DIM = 3;
   const int D1D = PA.maps_lim->ndof;
   const int Q1D = PA.maps_lim->nqpt;

   const double ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const auto C0 = const_c0 ?
                   Reshape(PA.C0.Read(), 1, 1, 1, 1) :
                   Reshape(PA.C0.Read(), Q1D, Q1D, Q1D, NE);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), Q1D, Q1D, Q1D);
   const auto BLD = Reshape(PA.maps_lim->B.Read(), Q1D, D1D);
   const auto LD = Reshape(PA.LD.Read(), D1D, D1D, D1D, NE);
   auto H0 = Reshape(PA.H0.Write(), DIM, DIM, Q1D, Q1D, Q1D, NE);

#ifndef MFEM_USE_JIT
   decltype(&TMOP_SetupGradPA_C0_3D<>) ker = TMOP_SetupGradPA_C0_3D<>;

   const int d=D1D, q=Q1D;
   if (d == 2 && q==2) { ker = TMOP_SetupGradPA_C0_3D<2,2>; }
   if (d == 2 && q==3) { ker = TMOP_SetupGradPA_C0_3D<2,3>; }
   if (d == 2 && q==4) { ker = TMOP_SetupGradPA_C0_3D<2,4>; }
   if (d == 2 && q==5) { ker = TMOP_SetupGradPA_C0_3D<2,5>; }
   if (d == 2 && q==6) { ker = TMOP_SetupGradPA_C0_3D<2,6>; }

   if (d == 3 && q==3) { ker = TMOP_SetupGradPA_C0_3D<3,3>; }
   if (d == 3 && q==4) { ker = TMOP_SetupGradPA_C0_3D<4,4>; }
   if (d == 3 && q==5) { ker = TMOP_SetupGradPA_C0_3D<5,5>; }
   if (d == 3 && q==6) { ker = TMOP_SetupGradPA_C0_3D<6,6>; }

   if (d == 4 && q==4) { ker = TMOP_SetupGradPA_C0_3D<4,4>; }
   if (d == 4 && q==5) { ker = TMOP_SetupGradPA_C0_3D<4,5>; }
   if (d == 4 && q==6) { ker = TMOP_SetupGradPA_C0_3D<4,6>; }

   if (d == 5 && q==5) { ker = TMOP_SetupGradPA_C0_3D<5,5>; }
   if (d == 5 && q==6) { ker = TMOP_SetupGradPA_C0_3D<5,6>; }

   ker(ln,LD,const_c0,C0,NE,J,W,BLD,H0,D1D,Q1D,4);
#else
   TMOP_SetupGradPA_C0_3D(ln,LD,const_c0,C0,NE,J,W,BLD,H0,D1D,Q1D,4);
#endif
}

} // namespace mfem
