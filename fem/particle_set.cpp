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

#include "particle_set.hpp"

namespace mfem
{

ParticleSet::ParticleSet(int dim, int num_fields, Ordering::Type ordering_)
   : ordering(ordering_),
     coords(dim),
     real_fields(num_fields)
{
   // Setup v_coords
   if (ordering == Ordering::byNODES) // XXX YYY ZZZ
   {
      // Initialize BlockVector w/ 3 blocks:
      mfem::Array<int> bOffsets(dim);
      v_coords = BlockVector(bOffsets);

      // Set blocks to refer to each coords Vector
      for (int i = 0; i < dim; i++)
      {
         v_coords.GetBlock(i).MakeRef(coords[i], 0);
      }
   }

   // byVDIM (XYZ XYZ XYZ) depends on GetNP() -- need to reset every SyncVCoords()
}

void ParticleSet::SyncVCoords()
{
   if (ordering == Ordering::byNODES)
   {
      v_coords.SyncFromBlocks();
   }
   else // byVDIM
   {
      // Re-initialize blocks every sync to have GetNP()*dim blocks
      int dim = coords.size();
      mfem::Array<int> bOffsets(GetNP()*dim);
      for (int i = 0; i < bOffsets.Size(); i++)
      {
         bOffsets[i] = i;
      }

      v_coords = BlockVector(bOffsets);

      for (int i = 0; i < v_coords.Size(); i++)
      {
         v_coords.GetBlock(i).MakeRef(coords[i % dim], i / dim, 1);
      }
   }
}

void ParticleSet::RandomInitialize(Mesh &m, int num_particles, int seed)
{
   int dim = coords.size();

   Vector pos_min, pos_max;

   m.GetBoundingBox(pos_min, pos_max);

   // Create randomized points using Vector
   Vector r_coords(dim*num_particles);
   r_coords.Randomize(seed);

   // Copy to coords
   for (int i = 0; i < dim; i++)
   {
      Vector comp(r_coords, i*num_particles, num_particles);
      coords[i] = comp;
   }

   // Scale based on min/max dimensions
   for (int i = 0; i < num_particles; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         coords[d][i] = pos_min[d] + coords[d][i] * (pos_max[d] - pos_min[d]);
      }
   }

   SyncVCoords();

   // Reset fields
   for (int i = 0; i < real_fields.size(); i++)
   {
      real_fields[i].SetSize(num_particles);
      real_fields[i] = 0.0;
   }

}

void ParticleSet::AddParticles(const Vector &in_coords,
                               const Vector* in_fields[])
{
   int dim = coords.size();
   int num_old = GetNP()/dim;
   int num_new = in_coords.Size()/dim;

   for (int d = 0; d < dim; d++)
   {
      Vector old_p = coords[d];
      coords[d].SetSize(num_old + num_new);
      coords[d].SetVector(old_p, 0);

      if (ordering == Ordering::byNODES)
      {
         Vector new_p(const_cast<Vector&>(in_coords), d*num_new, num_new);
         coords[d].SetVector(new_p, num_old);
      }
      else // byVDIM
      {
         for (int i = 0; i < num_new; i++)
         {
            coords[i % d][i+num_old] = in_coords[i];
         }
      }
   }

   for (int f = 0; f < GetNF(); f++)
   {
      const Vector *v_field = in_fields[f];
      Vector old_f = real_fields[f];
      real_fields[f].SetSize(num_old + num_new);
      real_fields[f].SetVector(old_f, 0);
      real_fields[f].SetVector(*v_field, num_old);
   }

   SyncVCoords();
}

void ParticleSet::RemoveParticles(const Array<int> &list)
{
   int dim = coords.size();
   int num_old = GetNP();

   // Sort the indices
   Array<int> sorted_list(list);
   sorted_list.Sort();

   int rm_count = 0;
   for (int i = sorted_list[rm_count]; i < num_old; i++)
   {
      if (rm_count < sorted_list.Size() && i == sorted_list[rm_count])
      {
         rm_count++;
      }
      else
      {
         // Shift elements rm_count
         for (int d = 0; d < dim; d++)
         {
            coords[d][i-rm_count] = coords[d][i];
         }

         for (int f = 0; f < GetNF(); f++)
         {
            real_fields[f][i-rm_count] = real_fields[f][i];
         }
      }
   }

   // Resize / remove tails
   for (int d = 0; d < dim; d++)
   {
      coords[d].SetSize(num_old - list.Size());
   }
   for (int f = 0; f < GetNF(); f++)
   {
      real_fields[f].SetSize(num_old - list.Size());
   }
}

void ParticleSet::UpdateParticlePositions(const Vector &new_coords)
{
   int dim = coords.size();

   if (ordering == Ordering::byNODES)
   {
      for (int d = 0; d < dim; d++)
      {
         Vector comp(const_cast<Vector&>(new_coords), d*GetNP(), GetNP());
         coords[d].SetVector(comp,0);
      }
   }
   else // byVDIM
   {
      for (int i = 0; i < GetNP(); i++)
      {
         coords[i % dim][i / dim] = new_coords[i];
      }
   }
}

} // namespace mfem