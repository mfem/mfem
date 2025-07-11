
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
#include "particlefunc.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{
real_t ParticleFunction::GetParticleData(int i, int comp)
{
   if (pspace.GetOrdering() == Ordering::byNODES)
   {
      return data[i+comp*pspace.GetNP()];
   }
   else
   {
      return data[comp+i*pspace.Dimension()];
   }
}

void ParticleFunction::GetParticleData(int i, Vector &v)
{
   v.SetSize(vdim);

   for (int c = 0; c < vdim; c++)
   {
      v[c] = GetParticleData(i,c);
   }

}

} // namespace mfem

#endif // MFEM_USE_GSLIB