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


void ParticleVector::GetParticleValues(int i, Vector &pvals) const
{
   pvals.SetSize(vdim);

   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         pvals[c] = Vector::operator[](i + c*GetNP());
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         pvals[c] = Vector::operator[](c + i*vdim);
      }
   }
}

void ParticleVector::GetParticleRefValues(int i, Vector &pref)
{
   MFEM_VERIFY(ordering == Ordering::byVDIM, "GetRefParticleField only valid when ordering byVDIM.");
   pref.MakeRef(*this, i*vdim, vdim);
}

void ParticleVector::SetParticleValues(int i, const Vector &pvals)
{
   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         Vector::operator[](i + c*GetNP()) = pvals[c];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         Vector::operator[](c + i*vdim) = pvals[c];
      }
   }
}

real_t& ParticleVector::ParticleValue(int i, int comp)
{
   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNP());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

const real_t& ParticleVector::ParticleValue(int i, int comp) const
{
   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNP());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

Particle::Particle(int dim, const Array<int> &field_vdims)
: coords(dim),
  fields()
{
   coords = 0.0;

   for (int i = 0; i < field_vdims.Size(); i++)
   {
      fields.emplace_back(field_vdims[i]);
      fields.back() = 0.0;
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

   if (fields.size() != rhs.fields.size())
   {
      return false;
   }
   for (int f = 0; f < fields.size(); f++)
   {
      if (fields[f].Size() != rhs.fields[f].Size())
      {
         return false;
      }
      for (int c = 0; c < fields[f].Size(); c++)
      {
         if (fields[f][c] != rhs.fields[f][c])
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
   for (int f = 0; f < fields.size(); f++)
   {
      out << "Field " << f << ": (";
      for (int c = 0; c < fields[f].Size(); c++)
         out << fields[f][c] << ( (c+1 < fields[f].Size()) ? "," : ")\n");
   }
}

Array<Ordering::Type> ParticleSet::GetOrderingArray(Ordering::Type o, int N)
{
   Array<Ordering::Type> ordering_arr(N);
   ordering_arr = o; 
   return std::move(ordering_arr); 
}
std::string ParticleSet::GetDefaultFieldName(int i)
{
   return "Field_" + std::to_string(i);
}
Array<const char*> ParticleSet::GetFieldNameArray(int N)
{
   Array<const char*> names(N); 
   for (int i = 0; i < N; i++)
   {
      names[i] = GetDefaultFieldName(i).c_str();
   }
   return std::move(names);
}


Array<int> ParticleSet::LDof2VDofs(int ndofs, int vdim, const Array<int> &ldofs, Ordering::Type o)
{
   // Convert list index array of "ldofs" to "vdofs"
   Array<int> v_list;
   v_list.Reserve(ldofs.Size()*vdim);
   if (o == Ordering::byNODES)
   {
      for (int l = 0; l < ldofs.Size(); l++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            v_list.Append(Ordering::Map<Ordering::byNODES>(ndofs, vdim, ldofs[l], vd));
         }
      }
   }
   else
   {
      for (int l = 0; l < ldofs.Size(); l++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            v_list.Append(Ordering::Map<Ordering::byVDIM>(ndofs, vdim, ldofs[l], vd));
         }
      }
   }

   return std::move(v_list);
}

#ifdef MFEM_USE_MPI
int ParticleSet::GetRank(MPI_Comm comm_)
{
   int r; MPI_Comm_rank(comm_, &r); 
   return r;
}
int ParticleSet::GetSize(MPI_Comm comm_)
{
   int s; MPI_Comm_size(comm_, &s); 
   return s;
}
int ParticleSet::GetRankNumParticles(MPI_Comm comm_, int NP)
{
   int rank = GetRank(comm_);
   int size = GetSize(comm_);
   return NP/size + (rank < (NP % size) ? 1 : 0);
}
#endif // MFEM_USE_MPI

void ParticleSet::ReserveParticles(int res, ParticleState &particles)
{
   particles.ids.Reserve(res);

   // Increase Vector capacity implicitly by resizing
   // TODO: should we just create a Vector::Reserve?
   for (int f = -1; f < particles.GetNF(); f++)
   {
      ParticleVector &pv = (f == -1 ? particles.coords : *particles.fields[f]);

      int pv_res = res*pv.GetVDim();
      if (pv_res > pv.Capacity())
      {
         ParticleVector pv_copy = pv;
         pv.SetSize(pv_res);
         pv.SetVector(pv_copy, 0);
         pv.SetSize(pv_copy.Size());
      }
      // Else pv_res is less than existing capacity. Do nothing (just like Array).
   }

}

void ParticleSet::AddParticles(const Array<int> &new_ids, ParticleState &particles, Array<int> *new_indices)
{
   int num_add = new_ids.Size();
   int old_np = particles.GetNP();
   int new_np = old_np + num_add;

   // Increase the capacity of all data without deleting existing
   ReserveParticles(new_np, particles);

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
   for (int f = -1; f < particles.GetNF(); f++)
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
   for (int f = -1; f < particles.GetNF(); f++)
   {
      ParticleVector &pv = (f == -1 ? particles.coords : *particles.fields[f]);

      int vdim = pv.GetVDim();

      // Convert list particle index array ("ldofs") to "vdofs" + delete data
      pv.DeleteAt(LDof2VDofs(old_np, vdim, list, pv.GetOrdering()));
   }

}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

template<std::size_t NData, std::size_t NFinder>
void ParticleSet::Transfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks, Array<FindPointsGSLIB*> finders)
{
   std::variant<pdata_t<NData>*, pdata_fdpts_t<NData, NFinder>*> pdata_arr_var;

   gslib::array gsl_arr;

   if (NFinder > 0)
   {
      // Use alias to mitigate common in template causing macro parsing issues
      using arr_type = pdata_fdpts_t<NData, NFinder>;
      array_init(arr_type, &gsl_arr, send_idxs.Size());
      pdata_arr_var = (pdata_fdpts_t<NData, NFinder>*) gsl_arr.ptr;
   }
   else
   {
      array_init(pdata_t<NData>, &gsl_arr, send_idxs.Size());
      pdata_arr_var = (pdata_t<NData>*) gsl_arr.ptr;
   }

   gsl_arr.n = send_idxs.Size();

   int rank = GetRank(comm);
   int size = GetSize(comm);
      
   // Set the data in pdata_arr
   std::visit(
   // Either a pdata_t<N>* or pdata_fdpts_t<N>*
   [&](auto &pdata_arr)
   {
      using T = std::remove_pointer_t<std::decay_t<decltype(pdata_arr)>>; // Get pointee type (pdata_t<N> or pdata_fdpts_t<N>)
      constexpr bool send_fdpts_data = std::is_same_v<T, pdata_fdpts_t<NData, NFinder>>; // true if using pdata_fdpts_t<N>, false otherwise

      for (int i = 0; i < send_idxs.Size(); i++)
      {
         T &pdata = pdata_arr[i];

         pdata.id = active_state.ids[send_idxs[i]];

         // Copy particle data directly into pdata
         int counter = 0;
         for (int f = -1; f < active_state.GetNF(); f++)
         {
            ParticleVector &pv = (f == -1 ? active_state.coords : *active_state.fields[f]);
            for (int c = 0; c < pv.GetVDim(); c++)
            {
               pdata.data[counter] = pv.ParticleValue(send_idxs[i], c);
               counter++;
            }
         }

         // If updating the FindPointsGSLIB objects as well, get data from it + set into struct
         if constexpr (send_fdpts_data)
         {
            for (int f = 0; f < finders.Size(); f++)
            {
               FindPointsGSLIB &finder = *finders[f];
               for (int d = 0; d < finder.dim; d++)
               {
                  // store byVDIM
                  pdata.rst[d+f*finder.dim] = finder.gsl_ref(send_idxs[i]*finder.dim+d); // Stored byVDIM
                  pdata.mfem_rst[d+f*finder.dim] = finder.gsl_mfem_ref(send_idxs[i]*finder.dim+d); // Stored byVDIM
               }
               pdata.elem[f] = finder.gsl_elem[send_idxs[i]];
               pdata.mfem_elem[f] = finder.gsl_mfem_elem[send_idxs[i]];
               pdata.code[f] = finder.gsl_code[send_idxs[i]];
               pdata.proc[f] = finder.gsl_proc[send_idxs[i]];
            }
         }

      }
      // Remove particles that will be transferred
      RemoveParticles(send_idxs, active_state);

      // Remove the elements to be sent from FindPointsGSLIB data structures
      // Maintain same ordering as coords post-RemoveParticles
      if constexpr (send_fdpts_data)
      {
         for (FindPointsGSLIB *finder : finders)
         {
            finder->gsl_elem.DeleteAt(send_idxs);
            finder->gsl_mfem_elem.DeleteAt(send_idxs);
            finder->gsl_code.DeleteAt(send_idxs);
            finder->gsl_proc.DeleteAt(send_idxs);
            Array<int> del_arr = LDof2VDofs(finder->points_cnt, finder->dim, send_idxs, Ordering::byVDIM);
            finder->gsl_ref.DeleteAt(del_arr);
            finder->gsl_mfem_ref.DeleteAt(del_arr);
         }
      }

      // Transfer particles
      sarray_transfer_ext(T, &gsl_arr, send_ranks.GetData(), sizeof(unsigned int), cr.get());

      // Add received particles to this rank
      // Received particles are added to end
      unsigned int recvd = gsl_arr.n;
      pdata_arr = (T*) gsl_arr.ptr;
      int inter_np = active_state.GetNP(); // pre-recvd NP (after remove)
      int new_np = inter_np + recvd;

      // Add data individually after reserving once
      ReserveParticles(new_np, active_state);
      if constexpr (send_fdpts_data)
      {
         for (FindPointsGSLIB *finder : finders)
         {
            finder->gsl_elem.Reserve(new_np);
            finder->gsl_mfem_elem.Reserve(new_np);
            finder->gsl_code.Reserve(new_np);
            finder->gsl_proc.Reserve(new_np);
            
            // TODO: Artificially increase capacity as in ReserveParticles
            // Ensures that increases in Vector size don't delete existing data
            if (finder->gsl_ref.Capacity() < new_np*finder->dim)
            {
               Vector gsl_ref_copy = finder->gsl_ref;
               finder->gsl_ref.SetSize(new_np*finder->dim);
               finder->gsl_ref.SetVector(gsl_ref_copy, 0);
               finder->gsl_ref.SetSize(gsl_ref_copy.Size());
            }
            if (finder->gsl_mfem_ref.Capacity() < new_np*finder->dim)
            {
               Vector gsl_mfem_ref_copy = finder->gsl_mfem_ref;
               finder->gsl_mfem_ref.SetSize(new_np*finder->dim);
               finder->gsl_mfem_ref.SetVector(gsl_mfem_ref_copy, 0);
               finder->gsl_mfem_ref.SetSize(gsl_mfem_ref_copy.Size());
            }
         }
      }

      // Add newly-recvd data directly to active state
      for (int i = 0; i < recvd; i++)
      {
         T pdata = pdata_arr[i];
         int id = pdata.id;
         
         Array<int> idx_temp;
         AddParticles(Array<int>({id}), active_state, &idx_temp);
         int new_idx = idx_temp[0]; // Get index of newly-added particle
   
         int counter = 0;
         for (int f = -1; f < active_state.GetNF(); f++)
         {
            ParticleVector &pv = (f == -1 ? active_state.coords : *active_state.fields[f]);
            for (int c = 0; c < pv.GetVDim(); c++)
            {
               pv.ParticleValue(new_idx, c) = pdata.data[counter];
               counter++;
            }
         }
         // Add recvd data to FindPointsGSLIB objects
         if constexpr (send_fdpts_data)
         {
            for (int f = 0; f < finders.Size(); f++)
            {
               FindPointsGSLIB *finder = finders[f];

               // Add new particle data 
               // IMPORTANT: Must make sure that order is correct / matches new Coords. We add received particle data to end so we add to end.
               finder->gsl_elem.Append(pdata.elem[f]);
               finder->gsl_mfem_elem.Append(pdata.mfem_elem[f]);
               finder->gsl_code.Append(pdata.code[f]);
               finder->gsl_proc.Append(pdata.proc[f]);
               
               // Increase size without losing data because we "reserved" earlier!
               finder->gsl_ref.SetSize(finder->gsl_ref.Size()+finder->dim);
               finder->gsl_mfem_ref.SetSize(finder->gsl_mfem_ref.Size()+finder->dim);

               // Set the new data byVDIM to the end
               finder->gsl_ref.SetVector(Vector(pdata.rst + f*finder->dim, finder->dim), finder->gsl_ref.Size()-finder->dim);
               finder->gsl_mfem_ref.SetVector(Vector(pdata.mfem_rst + f*finder->dim, finder->dim), finder->gsl_mfem_ref.Size()-finder->dim);
            }
         }
      }
      

      if constexpr (send_fdpts_data)
      {
         for (FindPointsGSLIB *finder : finders)
         {
            // Finally, update points_cnt
            finder->points_cnt = GetNP();
         }
      }

   }, pdata_arr_var);
}



#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

Particle ParticleSet::CreateParticle() const
{
   Array<int> field_vdims(active_state.GetNF());
   for (int f = 0; f < field_vdims.Size(); f++)
   {
      field_vdims[f] = active_state.fields[f]->GetVDim();
   }

   Particle p(GetDim(), field_vdims);

   return std::move(p);
}

void ParticleSet::WriteToFile(const char *fname, const std::stringstream &ss_header, const std::stringstream &ss_data)
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

ParticleSet::ParticleSet(int id_stride_, int id_counter_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_)
: id_stride(id_stride_),
  id_counter(id_counter_),
  active_state(dim, coords_ordering),
  inactive_state(dim, coords_ordering)
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
: ParticleSet(1, 0, num_particles, dim, coords_ordering, Array<int>(), Array<Ordering::Type>(), Array<const char*>())
{

}
   
ParticleSet::ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, Ordering::Type all_ordering)
: ParticleSet(1, 0, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), GetFieldNameArray(field_vdims.Size())) 
{

}

ParticleSet::ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, Ordering::Type all_ordering)
: ParticleSet(1, 0, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), field_names_) 
{

}

ParticleSet::ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_)
: ParticleSet(1, 0, num_particles, dim, coords_ordering, field_vdims, field_orderings, field_names_)
{

}



#ifdef MFEM_USE_MPI

ParticleSet::ParticleSet(MPI_Comm comm_, HYPRE_BigInt num_particles, int dim, Ordering::Type coords_ordering)
: ParticleSet(comm_, num_particles, dim, coords_ordering, Array<int>(), Array<Ordering::Type>(), Array<const char*>())
{

};

ParticleSet::ParticleSet(MPI_Comm comm_, HYPRE_BigInt num_particles, int dim, const Array<int> &field_vdims, Ordering::Type all_ordering)
: ParticleSet(comm_, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), GetFieldNameArray(field_vdims.Size()))
{

}

ParticleSet::ParticleSet(MPI_Comm comm_, HYPRE_BigInt num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, Ordering::Type all_ordering)
: ParticleSet(comm_, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), field_names_)
{

}

ParticleSet::ParticleSet(MPI_Comm comm_, HYPRE_BigInt num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_)
: ParticleSet(GetSize(comm_), GetRank(comm_),
               GetRankNumParticles(comm_, num_particles),
               dim,
               coords_ordering,
               field_vdims,
               field_orderings,
               field_names_)
{
   comm = comm_;
   gsl_comm = std::make_unique<gslib::comm>();
   cr = std::make_unique<gslib::crystal>();
   comm_init(gsl_comm.get(), comm);
   crystal_init(cr.get(), gsl_comm.get());
}


HYPRE_BigInt ParticleSet::GetGlobalNP() const
{
   HYPRE_BigInt total = GetNP();
   MPI_Allreduce(MPI_IN_PLACE, &total, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
   return total;
}

#endif // MFEM_USE_MPI

ParticleVector& ParticleSet::AddField(int vdim, Ordering::Type field_ordering, const char* field_name)
{
   if (!field_name)
   {
      field_name = GetDefaultFieldName(field_names.size()).c_str();
   }
   active_state.fields.emplace_back(std::make_unique<ParticleVector>(GetNP(), vdim, field_ordering));
   inactive_state.fields.emplace_back(std::make_unique<ParticleVector>(inactive_state.ids.Size(), vdim, field_ordering));

   field_names.emplace_back(field_name);

   return *active_state.fields.back();
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

void ParticleSet::RemoveParticles(const Array<int> &list, bool delete_particles)
{
   // If not deleting removed particles, first copy particles to inactive_state
   if (!delete_particles)
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
         for (int f = -1; f < inactive_state.GetNF(); f++)
         {
            ParticleVector &inactive_pv = (f == -1 ? inactive_state.coords : *inactive_state.fields[f]);
            ParticleVector &active_pv = (f == -1 ? active_state.coords : *active_state.fields[f]);
            
            for (int c = 0; c < active_pv.GetVDim(); c++)
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
   
   for (int f = 0; f < active_state.GetNF(); f++)
   {
      Field(f).GetParticleValues(i, p.Field(f));  
   }

   return std::move(p);
}

bool ParticleSet::ParticleRefValid() const
{
   if (active_state.coords.GetOrdering() == Ordering::byNODES)
   {
      return false;
   }
   for (int f = 0; f < active_state.GetNF(); f++)
   {
      if (active_state.fields[f]->GetOrdering() == Ordering::byNODES)
      {
         return false;
      }
   }
   return true;
}

Particle ParticleSet::GetParticleRef(int i)
{
   Particle p = CreateParticle();

   Coords().GetParticleRefValues(i, p.Coords());

   for (int f = 0; f < active_state.GetNF(); f++)
   {
      Field(f).GetParticleRefValues(i, p.Field(f));
   }

   return std::move(p);
}

void ParticleSet::SetParticle(int i, const Particle &p)
{
   Coords().SetParticleValues(i, p.Coords());

   for (int f = 0; f < active_state.GetNF(); f++)
   {
      Field(f).SetParticleValues(i, p.Field(f));
   }
}

void ParticleSet::PrintCSV(const char *fname, int precision)
{
   std::stringstream ss_header;

   // Configure header:
   std::array<char, 3> ax = {'X', 'Y', 'Z'};

   ss_header << "id" << ",";

#ifdef MFEM_USE_MPI
   ss_header << "rank" << ",";
#endif // MFEM_USE_MPI

   for (int f = -1; f < active_state.GetNF(); f++)
   {
      ParticleVector &pv = (f == -1 ? active_state.coords : *active_state.fields[f]);

      for (int c = 0; c < pv.GetVDim(); c++)
      {
         if (f == -1)
         {
            ss_header << ax[c];
         }
         else
         {
            ss_header << field_names[f] << (pv.GetVDim() > 1 ? "_" + std::to_string(c) : "");
         }
         ss_header << ((f+1 == active_state.GetNF() && c+1 == pv.GetVDim()) ? "\n" : ",");
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
      ss_data << active_state.ids[i] << ",";

#ifdef MFEM_USE_MPI
      ss_data << rank << ",";
#endif // MFEM_USE_MPI

      for (int f = -1; f < active_state.GetNF(); f++)
      {
         ParticleVector &pv = (f == -1 ? active_state.coords : *active_state.fields[f]);

         for (int c = 0; c < pv.GetVDim(); c++)
         {
            ss_data << pv.ParticleValue(i, c) << ((f+1 == active_state.GetNF() && c+1 == pv.GetVDim()) ? "\n" : ",");
         }
      }
   }

   // Write
   WriteToFile(fname, ss_header, ss_data);
}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
void ParticleSet::Redistribute(const Array<unsigned int> &rank_list, Array<FindPointsGSLIB*> finders)
{
   int rank = GetRank(comm);
   int size = GetSize(comm);

   // Get particles to be transferred
   // (Avoid unnecessary copies of particle data into and out of buffers)
   Array<unsigned int> send_idxs;
   Array<unsigned int> send_ranks;
   for (int i = 0; i < rank_list.Size(); i++)
   {
      if (rank != rank_list[i])
      {
         send_idxs.Append(i);
         send_ranks.Append(rank_list[i]);
      }
   }

   // Dispatch at runtime to use the correctly-sized static struct for gslib
   DispatchDataTransfer(send_idxs, send_ranks, finders, std::make_index_sequence<NDATA_MAX+1>{});
}
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

ParticleSet::~ParticleSet() = default;

} // namespace mfem
