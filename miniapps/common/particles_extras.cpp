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

#include "particles_extras.hpp"


namespace mfem
{
namespace common
{

void InitializeRandom(Particle &p, int seed, const Vector &pos_min, const Vector &pos_max)
{
   std::mt19937 gen(seed);
   std::uniform_real_distribution<> real_dist(0.0,1.0);

   for (int i = 0; i < p.GetSpaceDim(); i++)
   {
      p.GetCoords()[i] = pos_min[i] + (pos_max[i] - pos_min[i])*real_dist(gen);
   }

   for (int s = 0; s < p.GetNumScalars(); s++)
      p.GetScalar(s) = real_dist(gen);

   for (int v = 0; v < p.GetNumVectors(); v++)
   {
      for (int c = 0; c < p.GetVDim(v); c++)
      {
         p.GetVector(v)[c] = real_dist(gen);
      }
   }
}

} // namespace common

} // namespace mfem