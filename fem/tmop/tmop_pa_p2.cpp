// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tmop_pa_p2.hpp"

namespace mfem
{

struct MetricTMOP_1 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI1[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1));
      kernels::Set(2,2, 1.0, ie.Get_dI1(), P);
   }
};

struct MetricTMOP_2 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
      kernels::Set(2,2, 1./2., ie.Get_dI1b(), P);
   }
};

struct MetricTMOP_7 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI1[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1).dI2(dI2).dI2b(dI2b));
      const double I2 = ie.Get_I2();
      kernels::Add(2,2, 1.0 + 1.0 / I2, ie.Get_dI1(),
                   -ie.Get_I1() / (I2*I2), ie.Get_dI2(), P);
   }
};

struct MetricTMOP_56 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      // 0.5*(1 - 1/I2b^2)*dI2b
      double dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b));
      const double I2b = ie.Get_I2b();
      kernels::Set(2,2, 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
   }
};

struct MetricTMOP_77 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2(dI2).dI2b(dI2b));
      const double I2 = ie.Get_I2();
      kernels::Set(2,2, 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
   }
};

struct MetricTMOP_80 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      // w0 P_2 + w1 P_77
      double dI1b[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2(dI2).dI2b(dI2b));
      kernels::Set(2,2, w[0] * 0.5, ie.Get_dI1b(), P);
      const double I2 = ie.Get_I2();
      kernels::Add(2,2, w[1] * 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
   }
};

struct MetricTMOP_94 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      // w0 P_2 + w1 P_56
      double dI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie
      (Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
      kernels::Set(2,2, w[0] * 0.5, ie.Get_dI1b(), P);
      const double I2b = ie.Get_I2b();
      kernels::Add(2,2, w[1] * 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
   }
};

template<typename M, typename... Args>
static void Launch(const int d, const int q, Args&&... args)
{
   decltype(&TMOP_AddMultPA_2D<M>) ker = TMOP_AddMultPA_2D<M>;

   if (d==2 && q==2) { ker = TMOP_AddMultPA_2D<M,2,2>; }
   if (d==2 && q==3) { ker = TMOP_AddMultPA_2D<M,2,3>; }
   if (d==2 && q==4) { ker = TMOP_AddMultPA_2D<M,2,4>; }
   if (d==2 && q==5) { ker = TMOP_AddMultPA_2D<M,2,5>; }
   if (d==2 && q==6) { ker = TMOP_AddMultPA_2D<M,2,6>; }

   if (d==3 && q==3) { ker = TMOP_AddMultPA_2D<M,3,3>; }
   if (d==3 && q==4) { ker = TMOP_AddMultPA_2D<M,3,4>; }
   if (d==3 && q==5) { ker = TMOP_AddMultPA_2D<M,3,5>; }
   if (d==3 && q==6) { ker = TMOP_AddMultPA_2D<M,3,6>; }

   if (d==4 && q==4) { ker = TMOP_AddMultPA_2D<M,4,4>; }
   if (d==4 && q==5) { ker = TMOP_AddMultPA_2D<M,4,5>; }
   if (d==4 && q==6) { ker = TMOP_AddMultPA_2D<M,4,6>; }

   if (d==5 && q==5) { ker = TMOP_AddMultPA_2D<M,5,5>; }
   if (d==5 && q==6) { ker = TMOP_AddMultPA_2D<M,5,6>; }

   ker(std::forward<Args>(args)...,d,q,4);
}

void TMOP_Integrator::AddMultPA_2D(const Vector &x, Vector &y) const
{
   constexpr int DIM = 2;
   const double mn = metric_normal;
   const int NE = PA.ne, mid = metric->Id();
   const int d = PA.maps->ndof, q = PA.maps->nqpt;

   Array<double> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }
   const double *w = mp.Read();

   const auto J = Reshape(PA.Jtr.Read(), DIM,DIM, q,q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q,q);
   const auto B = Reshape(PA.maps->B.Read(), q,d);
   const auto G = Reshape(PA.maps->G.Read(), q,d);
   auto X = Reshape(x.Read(), d,d, DIM, NE);
   auto Y = Reshape(y.ReadWrite(), d,d, DIM, NE);

   if (mid == 1) { return Launch<MetricTMOP_1>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (mid == 2) { return Launch<MetricTMOP_2>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (mid == 7) { return Launch<MetricTMOP_7>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (mid == 56) { return Launch<MetricTMOP_56>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (mid == 77) { return Launch<MetricTMOP_77>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (mid == 80) { return Launch<MetricTMOP_80>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (mid == 94) { return Launch<MetricTMOP_94>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   MFEM_ABORT("Unsupported kernel!");
}

} // namespace mfem
