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

#include "particleset.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
namespace gslib
{
extern "C"
{
#include <gslib.h>
} // extern C
} // namespace gslib
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

namespace mfem
{


void ParticleVector::GetParticleValues(int i, Vector &GetParticleValues) const
{
   pvals.SetSize(vdim);

   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         pvals[c] = data[i + c*GetNP()];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         pvals[c] = data[c + i*vdim];
      }
   }
}

void ParticleVector::GetParticleRefValues(int i, Vector &pref)
{
   MFEM_VERIFY(ordering == Ordering::byVDIM, "GetRefParticleField only valid when ordering byVDIM.");
   pref.MakeRef(data, i*vdim, vdim);
}

void ParticleVector::SetParticleValues(int i, const Vector &pvals)
{
   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         data[i + c*GetNP()] = pvals[c];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         data[c + i*vdim] = pvals[c];
      }
   }
}

real_t& ParticleVector::ParticleValue(int i, int comp)
{
   if (ordering == Ordering::byNODES)
   {
      return data[i + comp*GetNP()];
   }
   else
   {
      return data[comp + i*vdim];
   }
}

const real_t& ParticleVector::ParticleValue(int i, int comp) const
{
   if (ordering == Ordering::byNODES)
   {
      return data[i + comp*GetNP()];
   }
   else
   {
      return data[comp + i*vdim];
   }
}

Particle::Particle(int dim, const Array<int> &field_vdims)
: coords(dim),
  fields(field_vdims.Size())
{
   coords = 0.0;

   for (int i = 0; i < fields.Size(); i++)
   {
      fields[i] = std::make_unique<Vector>(field_vdims[i]);
      *fields[i] = 0.0;
   }
}

bool Particle::operator==(const Particle &rhs) const
{
   if (coords.Size() != rhs.coords.Size())
   {
      return false;
   }
   for (int d = 0; d < coords.Size(); d++)
   {
      if (coords[d] != rhs.coords[d])
      {
         return false;
      }
   }

   if (fields.Size() != rhs.fields.Size())
   {
      return false;
   }
   for (int f = 0; f < fields.Size(); f++)
   {
      if (fields[f]->Size() != rhs.fields[f]->Size())
      {
         return false;
      }
      for (int c = 0; c < fields[i].Size(); c++)
      {
         if ((*fields[f])[c] != (*rhs.fields[f])[c])
            return false;
      }
   }
   return true;
}

void Particle::Print(std::ostream &out) const
{
   out << "Coords: (";
   for (int d = 0; d < coords.Size(); d++)
   {
      out << coords[d] << ( (d+1 < coords.Size()) ? "," : ")\n");
   }
   for (int f = 0; f < fields.Size(); f++)
   {
      out << "Field " << f << ": (";
      for (int c = 0; c < fields[f]->Size(); c++)
         out << (*fields[f])[c] << ( (c+1 < fields[f]->Size()) ? "," : ")\n");
   }
}


Array<Ordering::Type> ParticleSet::GetOrderingArray(Ordering::Type o, int N)
{
   Array<Ordering::Type> ordering_arr(N);
   ordering_arr = o; 
   return std::move(ordering_array); 
}
std::string ParticleSet::GetDefaultFieldName(int i)
{
   return "Field_" + std::to_string(i);
}
Array<std::string> ParticleSet::GetFieldNameArray(int N)
{
   Array<std::string> names(N); 
   for (int i = 0; i < N; i++)
   {
      names[i] = GetDefaultFieldName(i);
   }
   return std::move(names)
}

#ifdef MFEM_USE_MPI
int ParticleSet::GetRank(MPI_Comm comm_)
{
   int r; MPI_Comm_rank(comm_, &r); 
   return r;
}
int ParticleSet::GetSize(MPI_Comm comm_)
{
   int s; MPI_Comm_rank(comm_, &s); 
   return s;
}
int ParticleSet::GetRankNumParticles(MPI_Comm comm_, int NP)
{
   return NP/GetSize(comm_) + r < (r < (num_particles % size) ? 1 : 0);
}
#endif // MFEM_USE_MPI

void ParticleSet::Reserve(int res, ParticleState &particles)
{
   particles.ids.Reserve(res);

   // Increase Vector capacity implicitly by resizing
   // TODO: should we just create a Vector::Reserve?
   for (int f = -1; f < particles.fields.Size(); f++)
   {
      ParticleVector &pv = (f == -1 ? particles.coords : *particles.fields[f]);

      int pv_res = res*pv.GetVDim();
      if (pv.Capacity() < pv_res)
      {
         ParticleVector pv_copy = pv;
         pv.SetSize(pv_res);
         pv.SetVector(pv_copy, 0);
      }
      // Else data_res less than existing capacity. Do nothing.
   }

}

void ParticleSet::AddParticles(const Array<int> &new_ids, ParticleState &particles, Array<int> *new_indices)
{
   int num_add = new_ids.Size();
   int old_np = particles.GetNP();
   int new_np = old_np + num_add;

   // Increase the capacity of all data without deleting existing
   Reserve(new_np, particles);

   // Set indices of new particles
   if (new_indices)
   {
      new_indices->SetSize(num_add);
      for (int i = 0; i < num_add; i++)
      {
         (*new_indices)[i] = particles.ids.Size() + i;
      }
   }
   // Add new ids
   particles.ids.Append(new_ids);

   // Update data
   for (int f = -1; f < particles.fields.Size(); f++)
   {
      ParticleVector &pv = (f == -1 ? particles.coords : *particles.fields[f]);

      int vdim = pv.GetVDim();

      // Resize all data to new capacity
      // Old data will not be deleted as capacity has been increased !
      pv.SetSize(new_np*vdim);

      // Properly add new elements based on ordering + default-initialize new particle data
      if (pv.GetOrdering() == Ordering::byNODES)
      {
         // Must shift entries for byNODES...
         for (int c = vdim-1; c > 0; c--)
         {
            for (int i = old_np-1; i >= 0; i--)
            {
               pv[i+c*new_np] = pv[i+c*old_np];
            }
            for (int i = old_np; i < new_np; i++)
            {
               pv[i+c*new_np] = 0.0;
            } 
         }
      }
      else // byVDIM
      {
         for (int i = old_np*vdim; i < new_np*vdim; i++)
         {
            pv[i] = 0.0;
         }
      }
   }
}

void ParticleSet::RemoveParticles(const Array<int> &list, ParticleState &particles)
{
   int num_rm = list.Size();
   int old_np = particles.GetNP();
   int new_np = old_np - num_rm;

   // Delete IDs
   particles.ids.DeleteAt(list);

   // Delete data
   for (int f = -1; f < particles.fields.Size(); f++)
   {
      ParticleVector &pv = (f == -1 ? particles.coords : *particles.fields[f]);

      int vdim = pv.GetVDim();

      // Convert list particle index array ("ldofs") to "vdofs"
      Array<int> v_list;
      v_list.Reserve(num_rm*vdim);
      if (pv.GetOrdering() == Ordering::byNODES)
      {
         for (int l = 0; l < list.Size(); l++)
         {
            for (int vd = 0; vd < vdim; vd++)
            {
               v_list.Append(Ordering::Map<Ordering::byNODES>(old_np, vdim, list[l], vd));
            }
         }
      }
      else
      {
         for (int l = 0; l < list.Size(); l++)
         {
            for (int vd = 0; vd < vdim; vd++)
            {
               v_list.Append(Ordering::Map<Ordering::byVDIM>(old_np, vdim, list[l], vd));
            }
         }
      }

      // Delete data
      pv.DeleteAt(v_list);
   }

}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

template<std::size_t N>
void ParticleSet::Transfer(const Array<int> &send_idxs, const Array<int> &send_ranks, FindPointsGSLIB *finder)
{
   std::variant<pdata_t<N>*, pdata_fdpts_t<N>*> pdata_arr_var;

   gslib::array gsl_arr;

   if (finder)
   {
      array_init(pdata_fdpts_t<N>, &gsl_arr, send_idxs.Size());
      pdata_arr_var = (pdata_fdpts_t<N>*) gsl_arr.ptr;
   }
   else
   {
      array_init(pdata_t<N>, &gsl_arr, send_idxs.Size());
      pdata_arr_var = (pdata_t<N>*) gsl_arr.ptr;
   }

   gsl_arr.n = send_idxs.Size();

   int rank = GetRank(comm);
   int size = GetSize(comm);
      
   // Set the data in pdata_arr
   std::visit(
   // Either a pdata_t<N>* or pdata_fdpts_t<N>*
   [&](auto &&pdata_arr)
   {
      using T = std::remove_pointer_t<std::decay_t<decltype(pdata_arr)>>; // Get pointee type (pdata_t<N> or pdata_fdpts_t<N>)
      constexpr bool send_fdpts_data = std::is_same_v<T, pdata_fdpts_t<N>>; // true if using pdata_fdpts_t<N>, false otherwise

      for (int i = 0; i < send_idxs.Size(); i++)
      {
         T &pdata = pdata_arr[i];

         pdata.id = active_state.ids[send_idxs[i]];
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

void ParticleSet::Redistribute(const Array<unsigned int> &rank_list, FindPointsGSLIB *finder)
{
   MFEM_ASSERT(rank_list.Size() == GetNP(), "rank_list.Size() != GetNP()");

   int rank = GetRank(comm);
   int size = GetSize(comm);

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
   RuntimeDispatchTransfer(send_idxs, send_ranks, std::make_index_sequence<N_MAX+1>{}, finder);

}
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

Particle ParticleSet::CreateParticle() const
{
   Array<int> field_vdims(active_state.fields.Size());
   for (int f = 0; f < field_vdims.Size(); f++)
   {
      data_vdims[f] = active_state.fields[f]->GetVDim();
   }

   Particle p(GetDim(), field_vdims);

   return std::move(p);
}

void ParticleSet::WriteToFile(const char* fname, const std::stringstream &ss_header, const std::stringstream &ss_data)
{

#ifdef MFEM_USE_MPI
   // Parallel:
   int rank = GetRank(comm);

   MPI_File_delete(fname, MPI_INFO_NULL); // delete old file if it exists
   MPI_File file;
   MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);


   // Print header
   if (rank == 0)
   {
      MPI_File_write_at(file, 0, ss_header.str().data(), ss_header.str().size(), MPI_CHAR, MPI_STATUS_IGNORE);
   }

   // Compute the data size in bytes
   MPI_Offset data_size = ss_data.str().size();
   MPI_Offset offset;

   // Compute the offsets using an exclusive scan
   MPI_Exscan(&data_size, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
   if (rank == 0)
   {
      offset = 0;
   }

   // Add offset from the header
   offset += ss_header.str().size();

   // Write data collectively
   MPI_File_write_at_all(file, offset, ss_data.str().data(), data_size, MPI_BYTE, MPI_STATUS_IGNORE);

   // Close file
   MPI_File_close(&file);

#else
   // Serial:
   std::ofstream ofs(fname);
   ofs << ss_header.str() << ss_data.str();
   ofs.close();

#endif // MFEM_USE_MPI

}

ParticleSet::ParticleSet(int id_stride_, int id_counter_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &data_vdims, const Array<Ordering::Type> &field_orderings, const Array<std::string> &field_names_)
: id_stride(id_stride_),
  id_counter(id_counter_)
{   
   // Initialize data
   for (int f = 0; f < field_vdims.Size(); f++)
   {
      AddField(field_vdims[f], field_orderings[f], field_names_[f]);
   }

   // Add num_particles
   Array<int> ids(num_particles);
   for (int i = 0; i < num_particles; i++)
   {
      ids[i] = id_counter;
      id_counter += id_stride;
   }
   AddParticles(ids, active_state);
}

ParticleSet::ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering)
: ParticleSet(1, 0, num_particles, dim, coords_ordering, Array<int>(), Array<Ordering::Type>(), Array<std::string>())
{

}
   
ParticleSet::ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, Ordering::Type all_ordering)
: ParticleSet(1, 0, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), GetDataNameArray(field_vdims.Size())) 
{

}

ParticleSet::ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &dataVDims, const Array<Ordering::Type> &field_orderings, const Array<std::string> &field_names_)
: ParticleSet(1, 0, num_particles, dim, coords_ordering, &field_vdims, &field_orderings, &field_names_)
{

}



#ifdef MFEM_USE_MPI

ParticleSet::ParticleSet(MPI_Comm comm_, int num_particles, int dim, Ordering::Type coords_ordering)
: ParticleSet(comm_, num_particles, dim, coords_ordering, Array<int>(), Array<Ordering::Type>(), Array<std::string>())
{

};

ParticleSet::ParticleSet(MPI_Comm comm_, int num_particles, int dim, const Array<int> &field_vdims, Ordering::Type all_ordering)
: ParticleSet(comm_, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), GetDataNameArray(field_vdims.Size()))
{

}

ParticleSet::ParticleSet(MPI_Comm comm_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<std::string> &field_names_)
: ParticleSet(GetSize(comm_), GetRank(comm_),
               GetRankNumParticles(comm_, num_particles),
               dim,
               coords_ordering,
               field_orderings,
               field_names_)
{
   comm = comm_;
   gsl_comm = std::make_unique<gslib::comm>();
   cr = std::make_unique<gslib::crystal>();
   comm_init(gsl_comm.get(), comm);
   crystal_init(cr.get(), gsl_comm.get());
}

#endif // MFEM_USE_MPI

ParticleVector& ParticleSet::AddField(int vdim, Ordering::Type field_ordering, std::string field_name)
{
   if (data_name == "")
   {
      data_name = GetDefaultFieldName(field_names.Size());
   }
   active_state.fields.Append(std::make_unique<ParticleVector>(GetNP(), vdim, data_ordering));
   inactive_state.fields.Append(std::make_unique<ParticleVector>(inactive_state.ids.Size(), vdim, data_ordering));

   field_names.Append(data_name);

   return *active_state.fields.Last();
}

void ParticleSet::AddParticle(const Particle &p)
{
   // Add the particle
   Array<int> idxs;
   AddParticles(Array<int>({id_counter}), active_state, &idxs);
   id_counter += id_stride;

   // Set the new particle data
   int idx = idxs[0];
   SetParticle(idx, p);
}

void ParticleSet::RemoveParticles(const Array<int> &list, bool delete)
{
   // If not deleting removed particles, first copy particles to inactive_state
   if (!delete)
   {
      Array<int> rm_ids(list.Size());
      for (int i = 0; i < rm_ids.Size(); i++)
      {
         rm_ids[i] = active_state.ids[i];
      }

      Array<int> inactive_idxs(list.Size());

      // Add padding to inactive_state for new particle data
      AddParticles(rm_ids, inactive_state, &inactive_idxs);

      // Set the newly-added particle data
      for (int i = 0; i < list.Size(); i++)
      {
         for (int f = -1; f < inactive_state.fields.Size(); f++)
         {
            ParticleVector &inactive_pv = (f == -1 ? inactive_state.coords : *inactive_state.fields[f]);
            ParticleVector &active_pv = (f == -1 ? active_state.coords : *active_state.fields[f]);
            
            for (int c = 0; c < pv.GetVDim(); c++)
            {
               inactive_pv.ParticleValue(inactive_idxs[i], c) = active_pv.ParticleValue(list[i], c);
            }
         }
      }
   }

   // Delete particles from active_state
   RemoveParticles(list, active_state);
}

Particle ParticleSet::GetParticle(int i) const
{
   Particle p = CreateParticle();

   Coords().GetParticleValues(i, p.Coords());
   
   for (int f = 0; f < active_state.fields.Size(); f++)
   {
      Field(f).GetParticleValues(i, p.Field(f));  
   }

   return std::move(p);
}

Particle ParticleSet::GetParticleRef(int i)
{
   Particle p = CreateParticle();

   Coords().GetParticleRefValues(i, p.Coords());

   for (int f = 0; f < active_state.fields.Size(); f++)
   {
      Field(f).GetParticleRefValues(i, p.Field(f));
   }

   return std::move(p);
}

void ParticleSet::SetParticle(int i, const Particle &p)
{
   Coords().SetParticleValues(i, p.Coords());

   for (int f = 0; f < active_state.fields.Size(); f++)
   {
      Field(f).SetParticleValues(i, p.Field(f));
   }
}

void ParticleSet::PrintCSV(const char *fname, int precision)
{
   std::stringstream ss_header;

   // Configure header:
   std::array<char, 3> ax = {'x', 'y', 'z'};

   ss_header << "id" << ",";

#ifdef MFEM_USE_MPI
   ss_header << "rank" << ",";
#endif // MFEM_USE_MPI

   for (int f = 0; f < totalFields; f++)
   {
      for (int c = 0; c < fieldVDims[f]; c++)
      {
         if (f == 0)
         {
            ss_header << ax[c];
         }
         else if (f-1 < meta.NumProps())
         {
            ss_header << "Property_" << f-1;
         }
         else
         {
            ss_header << "StateVariable_" << f-1-meta.NumProps() << "_" << c;
         }
         ss_header << ((f+1 == totalFields && c+1 == fieldVDims[f]) ? "\n" : ",");
      }
   }


   // Configure data
   std::stringstream ss_data;
   ss_data.precision(precision);
#ifdef MFEM_USE_MPI
   int rank = GetRank(comm);
#endif // MFEM_USE_MPI
   for (int i = 0; i < GetNP(); i++)
   {
      ss_data << ids[i] << ",";

#ifdef MFEM_USE_MPI
      ss_data << rank << ",";
#endif // MFEM_USE_MPI

      for (int f = 0; f < totalFields; f++)
      {
         for (int c = 0; c < fieldVDims[f]; c++)
         {
            real_t dat;
            if (ordering == Ordering::byNODES)
            {
               dat = data[i + (exclScanFieldVDims[f]+c)*GetNP()];
            }
            else
            {
               dat = data[c + fieldVDims[f]*i + exclScanFieldVDims[f]*GetNP()];
            }
            ss_data << dat;
            ss_data << ((f+1 == totalFields && c+1 == fieldVDims[f]) ? "\n" : ",");
         }
      }
   }

   // Write
   WriteToFile(fname, ss_header, ss_data);
}


} // namespace mfem
