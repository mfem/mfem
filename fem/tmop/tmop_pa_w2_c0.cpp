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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(double, EnergyPA_C0_2D,
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
                           const Vector &ones,
                           Vector &energy,
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

   auto E = Reshape(energy.Write(), Q1D, Q1D, NE);

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
            double ld, p0[2], p1[2];
            const double *Jtr = &J(0,0,qx,qy,e);
            const double detJtr = kernels::Det<2>(Jtr);
            const double weight = W(qx,qy) * detJtr;
            const double coeff0 = const_c0 ? C0(0,0,0) : C0(qx,qy,e);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ,ld);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ0,p0);
            kernels::internal::PullEval<MQ1,NBZ>(Q1D,qx,qy,QQ1,p1);
            const double dist = ld; // GetValues, default comp set to 0
            const double id2 = 0.5 / (dist*dist);
            const double dsq = kernels::DistanceSquared<2>(p1,p0) * id2;
            E(qx,qy,e) = weight * lim_normal * dsq * coeff0;
         }
      }
   });
   return energy * ones;
}

double TMOP_Integrator::GetLocalStateEnergyPA_C0_2D(const Vector &X) const
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
   const Vector &O = PA.O;
   Vector &E = PA.E;

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_C0_2D,id,ln,LD,C0,N,J,W,B,BLD,X0,X,O,E);
}

} // namespace mfem
