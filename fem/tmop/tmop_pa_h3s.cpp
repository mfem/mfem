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

extern void KAssembleGradPA_3D_302(TMOP_SetupGradPA_3D &k);
extern void KAssembleGradPA_3D_303(TMOP_SetupGradPA_3D &k);
extern void KAssembleGradPA_3D_315(TMOP_SetupGradPA_3D &k);
extern void KAssembleGradPA_3D_318(TMOP_SetupGradPA_3D &k);
extern void KAssembleGradPA_3D_321(TMOP_SetupGradPA_3D &k);
extern void KAssembleGradPA_3D_332(TMOP_SetupGradPA_3D &k);
extern void KAssembleGradPA_3D_338(TMOP_SetupGradPA_3D &k);

void TMOP_Integrator::AssembleGradPA_3D(const Vector &x) const
{
   TMOP_SetupGradPA_3D ker(this, x);
   const int mid = metric->Id();
   if (mid == 302) { return KAssembleGradPA_3D_302(ker); }
   if (mid == 303) { return KAssembleGradPA_3D_303(ker); }
   if (mid == 315) { return KAssembleGradPA_3D_315(ker); }
   if (mid == 318) { return KAssembleGradPA_3D_318(ker); }
   if (mid == 321) { return KAssembleGradPA_3D_321(ker); }
   if (mid == 332) { return KAssembleGradPA_3D_332(ker); }
   if (mid == 338) { return KAssembleGradPA_3D_338(ker); }
   MFEM_ABORT("Unsupported TMOP metric " << mid);
}

} // namespace mfem
