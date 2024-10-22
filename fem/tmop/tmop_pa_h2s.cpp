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

extern void TMOPAssembleGradPA_001(TMOPSetupGradPA2D &ker,
                                   const TMOP_Integrator *ti,
                                   const Vector &x);
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

   if (mid == 1) { return TMOPAssembleGradPA_001(ker, this, x); }
   if (mid == 2) { return TMOPAssembleGradPA_002(ker); }
   if (mid == 7) { return TMOPAssembleGradPA_007(ker); }
   if (mid == 56) { return TMOPAssembleGradPA_056(ker); }
   if (mid == 77) { return TMOPAssembleGradPA_077(ker); }
   if (mid == 80) { return TMOPAssembleGradPA_080(ker); }
   if (mid == 94) { return TMOPAssembleGradPA_094(ker); }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
}

} // namespace mfem
