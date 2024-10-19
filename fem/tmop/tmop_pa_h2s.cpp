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

#include "tmop_pa_h2s.hpp"

namespace mfem
{

extern void TMOPAssembleGradPA_001(TMOPSetupGradPA2D &ker);
extern void TMOPAssembleGradPA_002(TMOPSetupGradPA2D &ker);
extern void TMOPAssembleGradPA_007(TMOPSetupGradPA2D &ker);
extern void TMOPAssembleGradPA_056(TMOPSetupGradPA2D &ker);
extern void TMOPAssembleGradPA_077(TMOPSetupGradPA2D &ker);
extern void TMOPAssembleGradPA_080(TMOPSetupGradPA2D &ker);
extern void TMOPAssembleGradPA_094(TMOPSetupGradPA2D &ker);

void TMOP_Integrator::AssembleGradPA_2D(const Vector &x) const
{
   const int mid = metric->Id();

   TMOPSetupGradPA2D ker(this, x);

   if (mid == 1)
   {
      return TMOPAssembleGradPA_001(ker);
   }
   if (mid == 2)
   {
      nconst int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      MFEM_SHARED double s_X[2][NBZ][MD1*MD1];
      MFEM_SHARED double s_DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double s_QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,s_X);
      kernels::internal::LoadBG<MD1,MQ1>(D1D, Q1D, b, g, s_BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D, Q1D, s_BG, s_X, s_DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D, Q1D, s_BG, s_DQ, s_QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = metric_normal * W(qx,qy) * detJtr;

            // Jrt = Jtr^{-1}
            double Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^t.DSh
            double Jpr[4];
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,s_QQ,Jpr);

            // Jpt = Jpr.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // metric->AssembleH
            if (mid ==  1) { EvalH_001(e,qx,qy,weight,Jpt,H); }
            if (mid ==  2) { EvalH_002(e,qx,qy,weight,Jpt,H); }
            if (mid ==  7) { EvalH_007(e,qx,qy,weight,Jpt,H); }
            if (mid == 77) { EvalH_077(e,qx,qy,weight,Jpt,H); }
            if (mid == 56) { EvalH_056(e,qx,qy,weight,Jpt,H); }
            if (mid == 80) { EvalH_080(e,qx,qy,weight,metric_data,Jpt,H); }
            if (mid == 94) { EvalH_094(e,qx,qy,weight,metric_data,Jpt,H); }
         } // qx
      } // qy
   });
}

void TMOP_Integrator::AssembleGradPA_2D(const Vector &X) const
{
   const int N = PA.ne;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double mn = metric_normal;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &W = PA.ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   Vector &H = PA.H;

   Array<double> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
}

}  // namespace mfem
