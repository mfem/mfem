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

#include "p2.hpp"

namespace mfem
{

extern void TMOPAddMultPA_001(TMOPAddMultPA2D &ker);
extern void TMOPAddMultPA_002(TMOPAddMultPA2D &ker);
extern void TMOPAddMultPA_007(TMOPAddMultPA2D &ker);
extern void TMOPAddMultPA_056(TMOPAddMultPA2D &ker);
extern void TMOPAddMultPA_077(TMOPAddMultPA2D &ker);
extern void TMOPAddMultPA_080(TMOPAddMultPA2D &ker);
extern void TMOPAddMultPA_094(TMOPAddMultPA2D &ker);

void TMOP_Integrator::AddMultPA_2D(const Vector &x, Vector &y) const
{
   const int mid = metric->Id();

   TMOPAddMultPA2D ker(this, x, y);

   if (mid == 1) { return TMOPAddMultPA_001(ker); }
   if (mid == 2) { return TMOPAddMultPA_002(ker); }
   if (mid == 7) { return TMOPAddMultPA_007(ker); }
   if (mid == 56) { return TMOPAddMultPA_056(ker); }
   if (mid == 77) { return TMOPAddMultPA_077(ker); }
   if (mid == 80) { return TMOPAddMultPA_080(ker); }
   if (mid == 94) { return TMOPAddMultPA_094(ker); }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
}

} // namespace mfem
