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
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_EnergyPA_2D(const double metric_normal,
                      const double gamma,
                      const int mid,
                      const int NE,
                      const DeviceTensor<5, const double> &J,
                      const ConstDeviceMatrix &W,
                      const ConstDeviceMatrix &B,
                      const ConstDeviceMatrix &G,
                      const DeviceTensor<4, const double> &X,
                      DeviceTensor<3> &E,
                      const int d1d,
                      const int q1d,
                      const int max)
{
   using Args = kernels::InvariantsEvaluator2D::Buffers;
   MFEM_VERIFY(mid == 1 || mid == 2 || mid == 7 || mid == 77 || mid == 80,
               "2D metric not yet implemented!");

   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

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
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,QQ,Jpr);

            // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
            double Jpt[4];
            kernels::Mult(2,2,2,Jpr,Jrt,Jpt);

            // metric->EvalW(Jpt);
            kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
            auto EvalW_001 = [&]() { return ie.Get_I1(); };
            auto EvalW_002 = [&]() { return 0.5 * ie.Get_I1b() - 1.0; };
            auto EvalW_007 = [&]()
            {
               return ie.Get_I1() * (1.0 + 1.0/ie.Get_I2()) - 4.0;
            };
            auto EvalW_077 = [&] ()
            {
               const double I2b = ie.Get_I2b();
               return 0.5*(I2b*I2b + 1./(I2b*I2b) - 2.);
            };
            auto EvalW_080 = [&]()
            {
               return (1.0 - gamma) * EvalW_002() + gamma * EvalW_077();
            };
            const double EvalW =
               mid ==  1 ? EvalW_001() :
               mid ==  2 ? EvalW_002() :
               mid ==  7 ? EvalW_007() :
               mid == 77 ? EvalW_077() :
               mid == 80 ? EvalW_080() : 0.0;

            E(qx,qy,e) = weight * EvalW;
         }
      }
   });
}

double TMOP_Integrator::GetLocalStateEnergyPA_2D(const Vector &x) const
{
   const int NE = PA.ne;
   constexpr int DIM = 2;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const double mn = metric_normal;

   double mp = 0.0;
   if (auto m = dynamic_cast<TMOP_Metric_080 *>(metric)) { mp = m->GetGamma(); }

   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto B = Reshape(PA.maps->B.Read(), Q1D, D1D);
   const auto G = Reshape(PA.maps->G.Read(), Q1D, D1D);
   const auto W = Reshape(PA.ir->GetWeights().Read(), Q1D, Q1D);
   const auto X = Reshape(x.Read(), D1D, D1D, DIM, NE);

   auto E = Reshape(PA.E.Write(), Q1D, Q1D, NE);

#ifndef MFEM_USE_JIT
   decltype(&TMOP_EnergyPA_2D<>) ker = TMOP_EnergyPA_2D<>;

   const int d=D1D, q=Q1D;
   if (d == 2 && q==2) { ker = TMOP_EnergyPA_2D<2,2>; }
   if (d == 2 && q==3) { ker = TMOP_EnergyPA_2D<2,3>; }
   if (d == 2 && q==4) { ker = TMOP_EnergyPA_2D<2,4>; }
   if (d == 2 && q==5) { ker = TMOP_EnergyPA_2D<2,5>; }
   if (d == 2 && q==6) { ker = TMOP_EnergyPA_2D<2,6>; }

   if (d == 3 && q==3) { ker = TMOP_EnergyPA_2D<3,3>; }
   if (d == 3 && q==4) { ker = TMOP_EnergyPA_2D<4,4>; }
   if (d == 3 && q==5) { ker = TMOP_EnergyPA_2D<5,5>; }
   if (d == 3 && q==6) { ker = TMOP_EnergyPA_2D<6,6>; }

   if (d == 4 && q==4) { ker = TMOP_EnergyPA_2D<4,4>; }
   if (d == 4 && q==5) { ker = TMOP_EnergyPA_2D<4,5>; }
   if (d == 4 && q==6) { ker = TMOP_EnergyPA_2D<4,6>; }

   if (d == 5 && q==5) { ker = TMOP_EnergyPA_2D<5,5>; }
   if (d == 5 && q==6) { ker = TMOP_EnergyPA_2D<5,6>; }

   ker(mn,mp,M,NE,J,W,B,G,X,E,D1D,Q1D,4);
#else
   TMOP_EnergyPA_2D(mn,mp,M,NE,J,W,B,G,X,E,D1D,Q1D,4);
#endif
   return PA.E * PA.O;
}

} // namespace mfem
