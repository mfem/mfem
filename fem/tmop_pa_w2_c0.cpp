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
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../general/debug.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(double, EnergyPA_C0_2D,
                           const double lim_normal,
                           const double dist,
                           const Vector &c0_,
                           const int NE,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const Vector &x0_,
                           const Vector &x1_,
                           Vector &energy,
                           Vector &ones,
                           const int d1d,
                           const int q1d)
{
   const bool const_c0 = c0_.Size() == 1;

   constexpr int dim = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C0 = const_c0 ?
                   Reshape(c0_.Read(), 1, 1, 1) :
                   Reshape(c0_.Read(), Q1D, Q1D, NE);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto X0 = Reshape(x0_.Read(), D1D, D1D, dim, NE);
   const auto X1 = Reshape(x1_.Read(), D1D, D1D, dim, NE);

   auto E = Reshape(energy.Write(), Q1D, Q1D, NE);
   auto O = Reshape(ones.Write(), Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];

      MFEM_SHARED double XY0[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ0[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ0[2][NBZ][MQ1*MQ1];

      MFEM_SHARED double XY1[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ1[2][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ1[2][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,X0,XY0);
      kernels::LoadX<MD1,NBZ>(e,D1D,X1,XY1);

      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY0,DQ0);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ0,QQ0);

      kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY1,DQ1);
      kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ1,QQ1);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double p0[2], p1[2];
            const double c0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            kernels::PullEvalXY<MQ1,NBZ>(qx,qy,QQ0,p0);
            kernels::PullEvalXY<MQ1,NBZ>(qx,qy,QQ1,p1);
            const double lf_eval_p_p0_dist =
            0.5 * kernels::DistanceSquared<2>(p1,p0) / (dist * dist);
            E(qx,qy,e) = lim_normal * lf_eval_p_p0_dist * c0;
            O(qx,qy,e) = 1.0;
         }
      }
   });
   return energy * ones;
}

double TMOP_Integrator::GetGridFunctionEnergyPA_C0_2D(const Vector &x) const
{
   dbg("");
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;
   Vector &X = PA.X;
   Vector &E = PA.E;
   Vector &O = PA.O;

   const double l = lim_normal;
   lim_dist->HostRead();
   MFEM_VERIFY(lim_dist, "Error");
   const double d = lim_dist->operator ()(0);

   PA.elem_restrict_lex->Mult(x, X);

   if (KEnergyPA_C0_2D.Find(id))
   {
      return KEnergyPA_C0_2D.At(id)(l,d,C0,N,B,G,X0,X,E,O,0,0);
   }
   else
   {
      constexpr int T_MAX = 8;
      MFEM_VERIFY(D1D <= T_MAX && Q1D <= T_MAX, "Max size error!");
      return EnergyPA_C0_2D<0,0,T_MAX>(l,d,C0,N,B,G,X0,X,E,O,D1D,Q1D);
   }
   return 0.0;
}

} // namespace mfem
