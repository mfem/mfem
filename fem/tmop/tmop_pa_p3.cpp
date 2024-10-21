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

#include "tmop_pa_p3.hpp"

namespace mfem
{

extern void TMOPAddMultPA_302(TMOPAddMultPA3D &ker);
extern void TMOPAddMultPA_303(TMOPAddMultPA3D &ker);
extern void TMOPAddMultPA_315(TMOPAddMultPA3D &ker);
extern void TMOPAddMultPA_318(TMOPAddMultPA3D &ker);
extern void TMOPAddMultPA_321(TMOPAddMultPA3D &ker);
extern void TMOPAddMultPA_332(TMOPAddMultPA3D &ker);
extern void TMOPAddMultPA_338(TMOPAddMultPA3D &ker);

void TMOP_Integrator::AddMultPA_3D(const Vector &x, Vector &y) const
{
   const int mid = metric->Id();

   TMOPAddMultPA3D ker(this, x, y);

   if (mid == 302) { return TMOPAddMultPA_302(ker); }
   if (mid == 303) { return TMOPAddMultPA_303(ker); }
   if (mid == 315) { return TMOPAddMultPA_315(ker); }
   if (mid == 318) { return TMOPAddMultPA_318(ker); }
   if (mid == 321) { return TMOPAddMultPA_321(ker); }
   if (mid == 332) { return TMOPAddMultPA_332(ker); }
   if (mid == 338) { return TMOPAddMultPA_338(ker); }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
}

} // namespace mfem
