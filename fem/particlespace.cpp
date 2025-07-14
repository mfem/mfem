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

void ParticleSpace::Initialize(Mesh *mesh, int seed)
{
   // Initialize starting particle IDs
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

   Vector particle_coords(dim);
   for (int i = 0; i < GetNP(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         particle_coords[d] = pos_min[d] + (pos_max[d] - pos_min[d])*real_dist(gen);
      }
      coords.SetParticleData(i, particle_coords);
   }

   // Register mesh if it exists
   // FindPoints will be called using coords
   if (mesh)
   {
      RegisterMesh(*mesh);
   }
}

void ParticleSpace::AddParticles(const Vector &new_coords,
                                 const Array<int> &new_ids,
                                 Array<int> &new_idxs)
{
   MFEM_ASSERT(new_coords.Size() / dim == new_ids.Size(),
               "new_coords is not sized properly");

   new_idxs.SetSize(new_ids.Size());

   int old_np = ids.Size();
   int num_new = new_ids.Size();

   // Update IDs
   ids.AddParticles(num_new);
   for (int i = 0; i < num_new; i++)
   {
      new_idxs[i] = i + old_np;
      ids[new_idxs[i]] = new_ids[i];
   }
   
   // Update coordinates
   coords.AddParticles(num_new);
   UpdateCoords(new_idxs, new_coords); // FindPoints called...

   // Update all registered ParticleFunctions
   for (ParticleFunction &pf : all_funcs)
   {
      pf.AddParticles(num_new);
   }
}



ParticleSpace::ParticleSpace(int dim_, int num_particles,
                             Ordering::Type ordering_, Mesh *mesh_, int seed)
: dim(dim_),
   ordering(ordering_),
   id_stride(1),
   id_counter(0),
   ids(num_particles),
   coords(*this, dim)

{
   Initialize(mesh_, seed);
}

#ifdef MFEM_USE_MPI
ParticleSpace::ParticleSpace(MPI_Comm comm_, int dim_, int num_particles,
                             Ordering::Type ordering_, Mesh *mesh_, int seed)
:  dim(dim_),
   ordering(ordering_),
   id_stride([&]() {int s; MPI_Comm_size(comm_, &s); return s; }()),
   id_counter([&]() { int r; MPI_Comm_rank(comm_, &r); return r; }()),
   ids(num_particles),
   coords(*this, dim),
   comm(comm_)
{
   Initialize(mesh_, seed);
}
#endif // MFEM_USE_MPI


int ParticleSpace::RegisterMesh(Mesh &mesh_)
{
   MFEM_VERIFY(dim == mesh_.SpaceDimension(),
                  "Mesh spatial dimension must match provided particle dimension.");
   meshes.push_back(&mesh_);

#ifdef MFEM_USE_MPI
   finders.emplace_back(comm);
#else
   finders.push_back();
#endif // MFEM_USE_MPI

   finders.back().Setup(*meshes.back());
   finders.back().FindPoints(coords.GetVector(), GetOrdering());

   return meshes.size()-1;
}

void ParticleSpace::UpdateCoords(const Array<int> &indices, const Vector &updated_coords)
{
   coords.SetParticleData(indices, updated_coords);
   for (FindPointsGSLIB &finder : finders)
   {
      finder.FindPoints(coords.GetVector(), GetOrdering());
   }
}

// template<typename T>
// ParticleData<T>& ParticleSpace::CreateParticleData(int vdim, std::string name)
// {
//    all_arrs.emplace_back(GetNP(), GetNP(), GetOrdering(), vdim);

//    if (name == "")
//    {
//       name = "Array_" + std::to_string(all_arrs.size()-1);
//    }
//    all_arr_names.push_back(name);

//    return all_arrs.back();
// }

ParticleFunction& ParticleSpace::CreateParticleFunction(int vdim, std::string name)
{
   all_funcs.emplace_back(ParticleFunction(*this,vdim));

   if (name == "")
   {
      name = "Data_" + std::to_string(all_funcs.size()-1);
   }
   all_func_names.push_back(name);

   return all_funcs.back();
}

void ParticleSpace::AddParticles(const Vector &new_coords, Array<int> *new_indices)
{
   // Initialize IDs for new particles
   Array<int> new_ids(new_coords.Size()/dim);

   for (int i = 0; i < new_ids.Size(); i++)
   {
      new_ids[i] = id_counter;
      id_counter += id_stride;   
   }
   Array<int> *idxs;
   Array<int> temp;
   if (new_indices)
   {
      idxs = new_indices;
   }
   else
   {
      idxs = &temp;
   }

   AddParticles(new_coords, new_ids, *idxs);
}

void ParticleSpace::RemoveParticles(const Array<int> &indices)
{
   ids.RemoveParticles(indices);
   coords.RemoveParticles(indices);
   for (ParticleFunction &pf : all_funcs)
   {
      pf.RemoveParticles(indices);
   }
}

void ParticleSpace::RemoveLostParticles(int mesh_idx)
{
   const Array<unsigned int> code = finders[mesh_idx].GetCode();
   Array<int> rm_indices;
   for (int i = 0; i < code.Size(); i++)
   {
      if (code[i] == 2)
      {
         rm_indices.Append(i);
      }
   }
   RemoveParticles(rm_indices);

}


} // namespace mfem


#endif // MFEM_USE_GSLIB