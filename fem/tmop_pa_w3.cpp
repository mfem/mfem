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
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dinvariants.hpp"

namespace mfem
{

// mu_302 = I1b * I2b / 9 - 1
static MFEM_HOST_DEVICE inline
double EvalW_302(const double *J)
{
   double B[9];
   kernels::InvariantsEvaluator3D ie(J,B);
   return ie.Get_I1b()*ie.Get_I2b()/9. - 1.;
}

// mu_303 = I1b/3 - 1
static MFEM_HOST_DEVICE inline
double EvalW_303(const double *J)
{
   double B[9];
   kernels::InvariantsEvaluator3D ie(J,B);
   return ie.Get_I1b()/3. - 1.;
}

// mu_321 = I1 + I2/I3 - 6
static MFEM_HOST_DEVICE inline
double EvalW_321(const double *J)
{
   double B[9];
   kernels::InvariantsEvaluator3D ie(J,B);
   return ie.Get_I1() + ie.Get_I2()/ie.Get_I3() - 6.0;
}

template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 0>
static double EnergyPA_3D(const int mid,
                          const int NE,
                          const DenseTensor &j_,
                          const Array<double> &w_,
                          const Array<double> &b_,
                          const Array<double> &g_,
                          const Vector &x_,
                          Vector &energy,
                          Vector &ones,
                          const int d1d = 0,
                          const int q1d = 0)
{
   MFEM_VERIFY(mid == 302 || mid == 303 || mid == 321 ,
               "3D metric not yet implemented!");

   constexpr int dim = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto J = Reshape(j_.Read(), dim, dim, Q1D, Q1D, Q1D, NE);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, dim, NE);

   auto E = Reshape(energy.Write(), Q1D, Q1D, Q1D, NE);
   auto O = Reshape(ones.Write(), Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      MFEM_SHARED double s_DDD[3][MD1*MD1*MD1];
      MFEM_SHARED double s_DDQ[6][MD1*MD1*MQ1];
      MFEM_SHARED double s_DQQ[9][MD1*MQ1*MQ1];
      MFEM_SHARED double s_QQQ[9][MQ1*MQ1*MQ1];

      kernels::LoadX<MD1>(e,D1D,X,s_DDD);
      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,s_BG);

      kernels::GradX<MD1,MQ1>(D1D,Q1D,s_BG,s_DDD,s_DDQ);
      kernels::GradY<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,s_DQQ);
      kernels::GradZ<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double *Jtr = &J(0,0,qx,qy,qz,e);
               const double detJtr = kernels::Det<3>(Jtr);
               const double weight = W(qx,qy,qz) * detJtr;

               // Jrt = Jtr^{-1}
               double Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               double Jpr[9];
               kernels::PullGradXYZ<MQ1>(qx,qy,qz, s_QQQ, Jpr);

               // Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               double Jpt[9];
               kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

               // metric->EvalW(Jpt);
               const double EvalW = mid == 302 ? EvalW_302(Jpt) :
               mid == 303 ? EvalW_303(Jpt) :
               mid == 321 ? EvalW_321(Jpt) :
               0.0;
               E(qx,qy,qz,e) = weight * EvalW;
               O(qx,qy,qz,e) = 1.0;
            }
         }
      }
   });
   return energy * ones;
}

double
TMOP_Integrator::GetGridFunctionEnergyPA_3D(const Vector &x) const
{
   MFEM_VERIFY(metric_normal == 1.0, "");

   const int N = PA.ne;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &X = PA.X;
   Vector &E = PA.E;
   Vector &O = PA.O;

   PA.elem_restrict_lex->Mult(x, PA.X);

   switch (id)
   {
      case 0x21: return EnergyPA_3D<2,1>(M,N,J,W,B,G,X,E,O);
      case 0x22: return EnergyPA_3D<2,2>(M,N,J,W,B,G,X,E,O);
      case 0x23: return EnergyPA_3D<2,3>(M,N,J,W,B,G,X,E,O);
      case 0x24: return EnergyPA_3D<2,4>(M,N,J,W,B,G,X,E,O);
      case 0x25: return EnergyPA_3D<2,5>(M,N,J,W,B,G,X,E,O);
      case 0x26: return EnergyPA_3D<2,6>(M,N,J,W,B,G,X,E,O);

      case 0x31: return EnergyPA_3D<3,1>(M,N,J,W,B,G,X,E,O);
      case 0x32: return EnergyPA_3D<3,2>(M,N,J,W,B,G,X,E,O);
      case 0x33: return EnergyPA_3D<3,3>(M,N,J,W,B,G,X,E,O);
      case 0x34: return EnergyPA_3D<3,4>(M,N,J,W,B,G,X,E,O);
      case 0x35: return EnergyPA_3D<3,5>(M,N,J,W,B,G,X,E,O);
      case 0x36: return EnergyPA_3D<3,6>(M,N,J,W,B,G,X,E,O);

      case 0x41: return EnergyPA_3D<4,1>(M,N,J,W,B,G,X,E,O);
      case 0x42: return EnergyPA_3D<4,2>(M,N,J,W,B,G,X,E,O);
      case 0x43: return EnergyPA_3D<4,3>(M,N,J,W,B,G,X,E,O);
      case 0x44: return EnergyPA_3D<4,4>(M,N,J,W,B,G,X,E,O);
      case 0x45: return EnergyPA_3D<4,5>(M,N,J,W,B,G,X,E,O);
      case 0x46: return EnergyPA_3D<4,6>(M,N,J,W,B,G,X,E,O);

      case 0x51: return EnergyPA_3D<5,1>(M,N,J,W,B,G,X,E,O);
      case 0x52: return EnergyPA_3D<5,2>(M,N,J,W,B,G,X,E,O);
      case 0x53: return EnergyPA_3D<5,3>(M,N,J,W,B,G,X,E,O);
      case 0x54: return EnergyPA_3D<5,4>(M,N,J,W,B,G,X,E,O);
      case 0x55: return EnergyPA_3D<5,5>(M,N,J,W,B,G,X,E,O);
      case 0x56: return EnergyPA_3D<5,6>(M,N,J,W,B,G,X,E,O);

      default:
      {
         constexpr int T_MAX = 4;
         MFEM_VERIFY(D1D <= T_MAX && Q1D <= T_MAX, "Max size error!");
         return EnergyPA_3D<0,0,T_MAX>(M,N,J,W,B,G,X,E,O,D1D,Q1D);
      }
   }
   return 0.0;
}

} // namespace mfem
