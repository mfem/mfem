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

#include "tmop_pa_p3.hpp"

namespace mfem
{

struct MetricTMOP_302 : MetricTMOPKer3D
{
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // (I1b/9)*dI2b + (I2b/9)*dI1b
      double B[9];
      double dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b) .dI3b(dI3b));
      const double alpha = ie.Get_I1b()/9.;
      const double beta = ie.Get_I2b()/9.;
      kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
   }
};

struct MetricTMOP_303 : MetricTMOPKer3D
{
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // dI1b/3
      double B[9];
      double dI1b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B).dI1b(dI1b).dI3b(dI3b));
      kernels::Set(3,3, 1./3., ie.Get_dI1b(), P);
   }
};

struct MetricTMOP_315 : MetricTMOPKer3D
{
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // 2*(I3b - 1)*dI3b
      double dI3b[9];
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b));
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Set(3,3, 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
   }
};

// P_318 = (I3b - 1/I3b^3)*dI3b.
// Uses the I3b form, as dI3 and ddI3 were not implemented at the time
struct MetricTMOP_318 : MetricTMOPKer3D
{
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      double dI3b[9];
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b));

      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Set(3,3, I3b - 1.0/(I3b * I3b * I3b), ie.Get_dI3b(sign_detJ), P);
   }
};

struct MetricTMOP_321 : MetricTMOPKer3D
{
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // dI1 + (1/I3)*dI2 - (2*I2/I3b^3)*dI3b
      double B[9];
      double dI1[9], dI2[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B).dI1(dI1).dI2(dI2).dI3b(dI3b));
      double sign_detJ;
      const double I3 = ie.Get_I3();
      const double alpha = 1.0/I3;
      const double beta = -2.*ie.Get_I2()/(I3*ie.Get_I3b(sign_detJ));
      kernels::Add(3,3, alpha, ie.Get_dI2(), beta, ie.Get_dI3b(sign_detJ), P);
      kernels::Add(3,3, ie.Get_dI1(), P);
   }
};

struct MetricTMOP_332 : MetricTMOPKer3D
{
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // w0 P_302 + w1 P_315
      double B[9];
      double dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
      const double alpha = w[0] * ie.Get_I1b()/9.;
      const double beta = w[0] * ie.Get_I2b()/9.;
      kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Add(3,3, w[1] * 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
   }
};

struct MetricTMOP_338 : MetricTMOPKer3D
{
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // w0 P_302 + w1 P_318
      double B[9];
      double dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt).B(B)
                                        .dI1b(dI1b)
                                        .dI2(dI2).dI2b(dI2b)
                                        .dI3b(dI3b));
      const double alpha = w[0] * ie.Get_I1b()/9.;
      const double beta = w[0]* ie.Get_I2b()/9.;
      kernels::Add(3,3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Add(3,3, w[1] * (I3b - 1.0/(I3b * I3b * I3b)),
                   ie.Get_dI3b(sign_detJ), P);
   }
};

template<typename M, typename... Args>
static void Launch(const int d, const int q, Args&&... args)
{
   decltype(&TMOP_AddMultPA_3D<M>) ker = TMOP_AddMultPA_3D<M>;

   if (d==2 && q==2) { ker = TMOP_AddMultPA_3D<M,2,2>; }
   if (d==2 && q==3) { ker = TMOP_AddMultPA_3D<M,2,3>; }
   if (d==2 && q==4) { ker = TMOP_AddMultPA_3D<M,2,4>; }
   if (d==2 && q==5) { ker = TMOP_AddMultPA_3D<M,2,5>; }
   if (d==2 && q==6) { ker = TMOP_AddMultPA_3D<M,2,6>; }

   if (d==3 && q==3) { ker = TMOP_AddMultPA_3D<M,3,3>; }
   if (d==3 && q==4) { ker = TMOP_AddMultPA_3D<M,3,4>; }
   if (d==3 && q==5) { ker = TMOP_AddMultPA_3D<M,3,5>; }
   if (d==3 && q==6) { ker = TMOP_AddMultPA_3D<M,3,6>; }

   if (d==4 && q==4) { ker = TMOP_AddMultPA_3D<M,4,4>; }
   if (d==4 && q==5) { ker = TMOP_AddMultPA_3D<M,4,5>; }
   if (d==4 && q==6) { ker = TMOP_AddMultPA_3D<M,4,6>; }

   if (d==5 && q==5) { ker = TMOP_AddMultPA_3D<M,5,5>; }
   if (d==5 && q==6) { ker = TMOP_AddMultPA_3D<M,5,6>; }

   ker(std::forward<Args>(args)...,d,q,4);
}

void TMOP_Integrator::AddMultPA_3D(const Vector &x, Vector &y) const
{
   constexpr int DIM = 3;
   const double mn = metric_normal;
   const int NE = PA.ne, M = metric->Id();
   const int d = PA.maps->ndof, q = PA.maps->nqpt;

   Array<double> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }
   const double *w = mp.Read();

   const auto J = Reshape(PA.Jtr.Read(), DIM,DIM, q,q,q, NE);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q,q,q);
   const auto B = Reshape(PA.maps->B.Read(), q,d);
   const auto G = Reshape(PA.maps->G.Read(), q,d);
   const auto X = Reshape(x.Read(), d,d,d, DIM, NE);
   auto Y = Reshape(y.ReadWrite(), d,d,d, DIM, NE);

   if (M == 302) { return Launch<MetricTMOP_302>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (M == 303) { return Launch<MetricTMOP_303>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (M == 315) { return Launch<MetricTMOP_315>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (M == 318) { return Launch<MetricTMOP_318>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (M == 321) { return Launch<MetricTMOP_321>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (M == 332) { return Launch<MetricTMOP_332>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   if (M == 338) { return Launch<MetricTMOP_338>(d,q,mn,w,NE,J,W,B,G,X,Y); }
   MFEM_ABORT("Unsupported kernel!");
}

} // namespace mfem
