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
#include "energy2.hpp"

namespace mfem
{

real_t TMOP_Integrator::GetLocalStateEnergyPA_2D(const Vector &x) const
{
   const int mid = metric->Id();

   TMOPEnergyPA2D ker(this, x);

   if (mid == 1) { tmop::Kernel<1>(ker); return ker.Energy(); }
   if (mid == 2) { tmop::Kernel<2>(ker); return ker.Energy(); }
   if (mid == 7) { tmop::Kernel<7>(ker); return ker.Energy(); }
   if (mid == 56) { tmop::Kernel<56>(ker); return ker.Energy(); }
   if (mid == 77) { tmop::Kernel<77>(ker); return ker.Energy(); }
   if (mid == 80) { tmop::Kernel<80>(ker); return ker.Energy(); }
   if (mid == 94) { tmop::Kernel<94>(ker); return ker.Energy(); }

   MFEM_ABORT("Unsupported TMOP metric " << mid);
   return 0.0;
}

} // namespace mfem
