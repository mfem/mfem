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

#include "tmop_pa_h3s.hpp"

namespace mfem
{

extern void KAssembleGradPA_3D_302(TMOP_SetupGradPA_3D &k, int d, int q);
extern void KAssembleGradPA_3D_303(TMOP_SetupGradPA_3D &k, int d, int q);
extern void KAssembleGradPA_3D_315(TMOP_SetupGradPA_3D &k, int d, int q);
extern void KAssembleGradPA_3D_318(TMOP_SetupGradPA_3D &k, int d, int q);
extern void KAssembleGradPA_3D_321(TMOP_SetupGradPA_3D &k, int d, int q);
extern void KAssembleGradPA_3D_332(TMOP_SetupGradPA_3D &k, int d, int q);
extern void KAssembleGradPA_3D_338(TMOP_SetupGradPA_3D &k, int d, int q);

void TMOP_Integrator::AssembleGradPA_3D(const Vector &x) const
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

   const auto B = Reshape(PA.maps->B.Read(), q,d);
   const auto G = Reshape(PA.maps->G.Read(), q,d);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q,q,q);
   const auto J = Reshape(PA.Jtr.Read(), DIM,DIM, q,q,q, NE);
   const auto X = Reshape(x.Read(), d,d,d, DIM, NE);
   auto H = Reshape(PA.H.Write(), DIM,DIM, DIM,DIM, q,q,q, NE);

   TMOP_SetupGradPA_3D ker(mn,w,NE,W,B,G,J,X,H,d,q,4);

   if (M == 302) { return KAssembleGradPA_3D_302(ker,d,q); }
   if (M == 303) { return KAssembleGradPA_3D_303(ker,d,q); }
   if (M == 315) { return KAssembleGradPA_3D_315(ker,d,q); }
   if (M == 318) { return KAssembleGradPA_3D_318(ker,d,q); }
   if (M == 321) { return KAssembleGradPA_3D_321(ker,d,q); }
   if (M == 332) { return KAssembleGradPA_3D_332(ker,d,q); }
   if (M == 338) { return KAssembleGradPA_3D_338(ker,d,q); }

   MFEM_ABORT("Unsupported TMOP metric " << M);
}

} // namespace mfem
