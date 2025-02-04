// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "grad3.hpp"

namespace mfem
{

void TMOP_Integrator::AssembleGradPA_3D(const Vector &x) const
{
   const int mid = metric->Id();

   TMOPSetupGradPA3D ker(this, x);

   if (mid == 302) { return tmop::Kernel<302>(ker); }
   if (mid == 303) { return tmop::Kernel<303>(ker); }
   if (mid == 315) { return tmop::Kernel<315>(ker); }
   if (mid == 318) { return tmop::Kernel<318>(ker); }
   if (mid == 321) { return tmop::Kernel<321>(ker); }
   if (mid == 332) { return tmop::Kernel<332>(ker); }
   if (mid == 338) { return tmop::Kernel<338>(ker); }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
}

} // namespace mfem
