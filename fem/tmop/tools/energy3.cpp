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
#include "energy3.hpp"

namespace mfem
{

real_t TMOP_Integrator::GetLocalStateEnergyPA_3D(const Vector &x) const
{
   const int mid = metric->Id();

   TMOPEnergyPA3D ker(this, x);

   if (mid == 302) { tmop::Kernel<302>(ker); return ker.Energy(); }
   if (mid == 303) { tmop::Kernel<303>(ker); return ker.Energy(); }
   if (mid == 315) { tmop::Kernel<315>(ker); return ker.Energy(); }
   if (mid == 318) { tmop::Kernel<318>(ker); return ker.Energy(); }
   if (mid == 321) { tmop::Kernel<321>(ker); return ker.Energy(); }
   if (mid == 332) { tmop::Kernel<332>(ker); return ker.Energy(); }
   if (mid == 338) { tmop::Kernel<338>(ker); return ker.Energy(); }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
   return 0.0;
}

} // namespace mfem
