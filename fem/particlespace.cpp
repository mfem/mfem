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

void ParticleSpace::PrintCSV(std:string fname, int precision)
{
   // TODO....
}

#ifdef MFEM_USE_MPI
void ParticleSpace::Redistribute(const Array<unsigned int> &rank_list)
{
   MFEM_ASSERT(rank_list.Size() == GetNP(), "rank_list.Size() != GetNP()");

   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Get particles to be transferred
   // (Avoid unnecessary copies of particle data into buffers)
   Array<int> send_idxs;
   Array<int> send_ranks;
   for (int i = 0; i < rank_list.Size(); i++)
   {
      if (rank != rank_list[i])
      {
         send_idxs.Append(i);
         send_ranks.Append(rank_list[i]);
      }
   }

   // Dispatch at runtime to use the correctly-sized static struct for gslib
   RuntimeDispatchTransfer(send_idxs, send_ranks, std::make_index_sequence<N_MAX+1>{});
}

template<std::size_t NData, std::size_t NFinder>
void ParticleSpace::Transfer(const Array<int> &send_idxs, const Array<int> &send_ranks);
{
   std::variant<pdata_t<NData>*, pdata_fdpts_t<NData, NFinder>*> pdata_arr_var;

   gslib::array gsl_arr;

   if (finder.size() > 0)
   {
      array_init(pdata_fdpts_t<NData, NFinder>, &gsl_arr, send_idxs.Size());
      pdata_arr_var = (pdata_fdpts_t<NData, NFinder>*) gsl_arr.ptr;
   }
   else
   {
      array_init(pdata_t<NData>, &gsl_arr, send_idxs.Size());
      pdata_arr_var = (pdata_t<NData>*) gsl_arr.ptr;
   }

   gsl_arr.n = send_idxs.Size();

   int rank; MPI_Comm_rank(comm, &rank);
   int size; MPI_Comm_size(comm, &size);
      
   // Set the data in pdata_arr
   std::visit(
   // Either a pdata_t<N>* or pdata_fdpts_t<N>*
   [&](auto &&pdata_arr)
   {
      using T = std::remove_pointer_t<std::decay_t<decltype(pdata_arr)>>; // Get pointee type (pdata_t<N> or pdata_fdpts_t<N>)
      constexpr bool send_fdpts_data = std::is_same_v<T, pdata_fdpts_t<NData, NFinder>>; // true if using pdata_fdpts_t<N>, false otherwise

      for (int i = 0; i < send_idxs.Size(); i++)
      {
         T &pdata = pdata_arr[i];

         pdata.id = ids[send_idxs[i]];
         pdata.proc = send_ranks[i];

         // Get copy of particle data
         // (TODO: skip this step... Copy directly from data to pdata!!)
         Particle p(meta);
         GetParticle(send_idxs[i], p);

         // Copy particle data into pdata
         for (int f = 0; f < totalFields; f++)
         {
            for (int c = 0; c < fieldVDims[f]; c++)
            {
               double* dat = &pdata.data[c + exclScanFieldVDims[f]];
               if (f == 0)
               {
                  *dat = static_cast<double>(p.GetCoords()[c]);
               }
               else if (f-1 < meta.NumProps())
               {
                  *dat = static_cast<double>(p.GetProperty(f-1));
               }
               else
               {
                  *dat = static_cast<double>(p.GetStateVar(f-1-meta.NumProps())[c]);
               }
            }
         }

         // If updating the FindPointsGSLIB object as well, get data from it + set into struct
         if constexpr (send_fdpts_data)
         {
            for (int d = 0; d < meta.SpaceDim(); d++)
            {
               pdata.rst[d] = finder->gsl_ref(send_idxs[i]*meta.SpaceDim()+d); // Stored byVDIM
               pdata.mfem_rst[d] = finder->gsl_mfem_ref(send_idxs[i]*meta.SpaceDim()+d); // Stored byVDIM
            }
            pdata.elem = finder->gsl_elem[send_idxs[i]];
            pdata.mfem_elem = finder->gsl_mfem_elem[send_idxs[i]];
            pdata.code = finder->gsl_code[send_idxs[i]];

            
         }

      }
      // Remove particles that will be transferred
      RemoveParticles(send_idxs);
      // GetNP() is now updated !!

      // Remove the elements to be sent from FindPointsGSLIB data structures
      // Maintain same ordering as coords post-RemoveParticles
      if constexpr (send_fdpts_data)
      {
         // TODO: Can probably optimize this better. Right now just copying all non-removed data to temp arr, then setting
         Array<unsigned int> rm_gsl_elem(GetNP());
         Array<unsigned int> rm_gsl_mfem_elem(GetNP());
         Array<unsigned int> rm_gsl_code(GetNP());
         Array<unsigned int> rm_gsl_proc(GetNP());

         Vector rm_gsl_ref(GetNP()*finder->dim);
         Vector rm_gsl_mfem_ref(GetNP()*finder->dim);

         int idx = 0;
         for (int i = 0; i < finder->points_cnt; i++) // points_cnt will be representative of the pre-redistribute point cnt on this rank
         {
            if (send_idxs.Find(i) == -1) // If particle at last i was NOT removed...
            {
               rm_gsl_elem[idx] = finder->gsl_elem[i];
               rm_gsl_mfem_elem[idx] = finder->gsl_mfem_elem[i];
               rm_gsl_code[idx] = finder->gsl_code[i];
               rm_gsl_proc[idx] = finder->gsl_proc[i];

               for (int d = 0; d < finder->dim; d++)
               {
                  rm_gsl_ref[idx*finder->dim+d] = finder->gsl_ref[i*finder->dim+d];
                  rm_gsl_mfem_ref[idx*finder->dim+d] = finder->gsl_mfem_ref[i*finder->dim+d];
               }
               idx++;
            }
         }


         finder->gsl_elem = rm_gsl_elem;
         finder->gsl_mfem_elem = rm_gsl_mfem_elem;
         finder->gsl_code = rm_gsl_code;
         finder->gsl_proc = rm_gsl_proc;
         finder->gsl_ref = rm_gsl_ref;
         finder->gsl_mfem_ref = rm_gsl_mfem_ref;
      }

      // Transfer particles
      sarray_transfer(T, &gsl_arr, proc, 0, cr.get());

      // Add received particles to this rank
      unsigned int recvd = gsl_arr.n;
      pdata_arr = (T*) gsl_arr.ptr;

      Vector add_gsl_ref;
      Vector add_gsl_mfem_ref;
      
      if constexpr (send_fdpts_data)
      {
         add_gsl_ref.SetSize((recvd+GetNP())*finder->dim);
         add_gsl_ref.SetVector(finder->gsl_ref, 0);

         add_gsl_mfem_ref.SetSize((recvd+GetNP())*finder->dim);
         add_gsl_mfem_ref.SetVector(finder->gsl_mfem_ref, 0);
      }

      for (int i = 0; i < recvd; i++)
      {
         T pdata = pdata_arr[i];
         if constexpr(std::is_same_v<real_t, double>)
         {
            // Create a particle, copy data from buffer to it, then add particle
            // TODO: Optimize by copying directly from received pdata to data!!
            Particle p(meta);

            p.GetCoords() = Vector(&pdata.data[0], meta.SpaceDim());

            for (int s = 0; s < meta.NumProps(); s++)
               p.GetProperty(s) = pdata.data[meta.SpaceDim() + s];

            for (int v = 0; v < meta.NumStateVars(); v++)
               p.GetStateVar(v) = Vector(&pdata.data[exclScanFieldVDims[1+meta.NumProps()+v]], meta.StateVDim(v));

            AddParticle(p, pdata.id);
         }
         else // need to copy from double to real_t if real_t is not double
         {
            // TODO
         }


         if constexpr (send_fdpts_data)
         {
            // Add new particle data 
            // IMPORTANT: Must make sure that order is correct / matches new Coords. We add received particle data to end so we add to end.
            finder->gsl_elem.Append(pdata.elem);
            finder->gsl_mfem_elem.Append(pdata.mfem_elem);
            finder->gsl_code.Append(pdata.code);
            finder->gsl_proc.Append(pdata.proc);
            
            add_gsl_ref.SetVector(Vector(pdata.rst, finder->dim), finder->gsl_ref.Size()+i*finder->dim);
            add_gsl_mfem_ref.SetVector(Vector(pdata.mfem_rst, finder->dim), finder->gsl_mfem_ref.Size()+i*finder->dim);

         }
      }

      if constexpr (send_fdpts_data)
      {
         finder->gsl_ref = add_gsl_ref;
         finder->gsl_mfem_ref = add_gsl_mfem_ref;

         // Lastly, update points_cnt
         finder->points_cnt = GetNP();
      }

   }, pdata_arr_var);
}

#endif // MFEM_USE_MPI


} // namespace mfem


#endif // MFEM_USE_GSLIB