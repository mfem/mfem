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

#include "bilininteg_pa_simplices_mma.hpp"

#include "../../general/globals.hpp"

#include <cstring>

namespace mfem
{

namespace
{
bool force_simplex_positive_mma = false;
}

void ForceSimplexPositiveMMA(bool enable)
{
   force_simplex_positive_mma = enable;
}

bool GetForceSimplexPositiveMMA()
{
   if (force_simplex_positive_mma) { return true; }

   // Cached env lookup: MFEM_SIMPLEX_POSITIVE_MMA set (and not "0") forces MMA.
   static int env_mma = -1; // -1 unset, 0 no, 1 yes
   if (env_mma < 0)
   {
      const char *e = GetEnv("MFEM_SIMPLEX_POSITIVE_MMA");
      env_mma = (e && std::strcmp(e, "0") != 0) ? 1 : 0;
   }
   return env_mma == 1;
}

} // namespace mfem
