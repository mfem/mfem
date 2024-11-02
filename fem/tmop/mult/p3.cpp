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

#include "../pa.hpp"
#include "p3.hpp"

namespace mfem
{

void TMOP_Integrator::AddMultPA_3D(const Vector &x, Vector &y) const
{
   const int mid = metric->Id();

   TMOPAddMultPA3D ker(this, x, y);

   if (mid == 302) { return TMOPKernel<302>(ker); }
   if (mid == 303) { return TMOPKernel<303>(ker); }
   if (mid == 315) { return TMOPKernel<315>(ker); }
   if (mid == 318) { return TMOPKernel<318>(ker); }
   if (mid == 321) { return TMOPKernel<321>(ker); }
   if (mid == 332) { return TMOPKernel<332>(ker); }
   if (mid == 338) { return TMOPKernel<338>(ker); }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
}

} // namespace mfem
