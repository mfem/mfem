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

#include "particlespace.hpp"

#ifdef MFEM_USE_GSLIB

#include <random>

namespace mfem
{

ParticleSpace::ParticleSpace(int dim_, int num_particles, Ordering::Type ordering_, Mesh *mesh_, int seed)
: dim(dim_),
  ordering(ordering_),
  id_stride(1),
  id_counter(0),
  ids(num_particles),
  mesh(mesh_),
  finder(),
  coords(num_particles*dim_)
{
   Initialize(seed);
}

#ifdef MFEM_USE_MPI
ParticleSpace::ParticleSpace(MPI_Comm comm_, int dim_, int num_particles, Ordering::Type ordering_, Mesh *mesh_, int seed)
: dim(dim_),
  ordering(ordering_),
  id_stride([&](){int s; MPI_Comm_size(comm_, &s); return s; }()),
  id_counter([&]() { int r; MPI_Comm_rank(comm_, &r); return r; }()),
  ids(num_particles),
  mesh(mesh_),
  finder(comm_),
  coords(num_particles*dim_),
  comm(comm_)
{
   Initialize(seed);
}
#endif // MFEM_USE_MPI


void ParticleSpace::Initialize(int seed)
{
   // Setup FindPointsGSLIB if mesh was provided
   if (mesh)
   {
      MFEM_VERIFY(dim == mesh->SpaceDimension(), "Mesh spatial dimension must match provided particle dimension.");
      finder.Setup(*mesh);
   }

   // Initialize particle IDs
   for (int i = 0; i < ids.Size(); i++)
   {
      ids[i] = id_counter;
      id_counter += id_stride;
   }

   // Initialize particle coordinates randomly within Mesh bounding-box or unit volume
   Vector pos_min, pos_max;
   if (mesh)
   {
      mesh->GetBoundingBox(pos_min, pos_max);
   }
   else
   {
      pos_min.SetSize(dim); pos_min = 0.0;
      pos_max.SetSize(dim); pos_max = 1.0;
   }

   if (seed == 0)
   {
      seed = (int)time(0);
   }
   std::mt19937 gen(seed);
   std::uniform_real_distribution<> real_dist(0.0,1.0);

   for (int i = 0; i < GetNP(); i++)
   {
      real_t *dat;
      for (int d = 0; d < dim; d++)
      {
         dat = ordering == Ordering::byVDIM ? &coords[d+i*dim] : &coords[i+d*GetNP()];
         *dat = pos_min[d] + (pos_max[d] - pos_min[d])*real_dist(gen);
      }
   }

   if (mesh)
   {
      finder.FindPoints(coords);

   }
}


} // namespace mfem


#endif // MFEM_USE_GSLIB