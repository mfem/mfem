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

Particle::Particle(int dim, const Array<int> &field_vdims, int num_tags)
: coords(dim),
  fields(),
  tags()
{
   coords = 0.0;

   for (int f = 0; f < field_vdims.Size(); f++)
   {
      fields.emplace_back(field_vdims[f]);
      fields.back() = 0.0;
   }

   for (int t = 0; t < num_tags; t++)
   {
      tags.emplace_back(1);
      tags.back()[0] = 0;
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
   if (tags.size() != rhs.tags.size())
   {
      return false;
   }
   for (int t = 0; t < tags.size(); t++)
   {
      if (tags[t][0] != rhs.tags[t][0])
      {
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
   for (int t = 0; t < tags.size(); t++)
   {
      out << "Tag " << t << ": " << tags[t][0] << "\n";
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

std::string ParticleSet::GetDefaultTagName(int i)
{
   return "Tag_" + std::to_string(i);
}

Array<const char*> ParticleSet::GetEmptyNameArray(int N)
{
   Array<const char*> names(N); 
   for (int i = 0; i < N; i++)
   {
      names[i] = nullptr;
   }
   return std::move(names);
}

#ifdef MFEM_USE_MPI
unsigned int ParticleSet::GetRank(MPI_Comm comm_)
{
   int r; MPI_Comm_rank(comm_, &r); 
   return r;
}
unsigned int ParticleSet::GetSize(MPI_Comm comm_)
{
   int s; MPI_Comm_size(comm_, &s); 
   return s;
}
#endif // MFEM_USE_MPI

void ParticleSet::Reserve(int res)
{
   ids.Reserve(res);

   // Reserve fields
   for (int f = -1; f < GetNF(); f++)
   {
      MultiVector &pv = (f == -1 ? coords : *fields[f]);

      // ------------------------------------------------------------
      // Increase Vector capacity implicitly by resizing
      // TODO: should we just create a Vector::Reserve?
      //       Array::SetSize doesn't delete... Maybe a "GrowSize?"
      int pv_res = res*pv.GetVDim();
      if (pv_res > pv.Capacity())
      {
         MultiVector pv_copy = pv;
         pv.SetSize(pv_res);
         pv.SetVector(pv_copy, 0);
         pv.SetSize(pv_copy.Size());
      }
      // Else pv_res is less than existing capacity. Do nothing (just like Array).
      // ------------------------------------------------------------
   }

   // Reserve tags
   for (int t = 0; t < GetNT(); t++)
   {
      tags[t]->Reserve(res);
   }

}

void ParticleSet::AddParticles(const Array<unsigned int> &new_ids, Array<int> *new_indices)
{
   int num_add = new_ids.Size();
   int old_np = GetNP();
   int new_np = old_np + num_add;

   // Set indices of new particles
   if (new_indices)
   {
      new_indices->SetSize(num_add);
      for (int i = 0; i < num_add; i++)
      {
         (*new_indices)[i] = ids.Size() + i;
      }
   }
   // Add new ids
   ids.Append(new_ids);

   // Update data
   for (int f = -1; f < GetNF(); f++)
   {
      MultiVector &pv = (f == -1 ? coords : *fields[f]);
      pv.SetNumVectors(new_np); // does not delete existing data
   }

   // Update tags
   for (int t = 0; t < GetNT(); t++)
   {
      tags[t]->SetSize(new_np);
   }
}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

template<std::size_t NData, std::size_t NTag>
void ParticleSet::Transfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks)
{
   pdata_t<NData, NTag> *pdata_arr;
   gslib::array gsl_arr;

   // Use alias to mitigate comma in template causing macro parsing issues
   using arr_type = pdata_t<NData, NTag>;
   array_init(arr_type, &gsl_arr, send_idxs.Size());

   pdata_arr = (pdata_t<NData, NTag>*) gsl_arr.ptr;

   gsl_arr.n = send_idxs.Size();

   int rank = GetRank(comm);
   int size = GetSize(comm);
   
   for (int i = 0; i < send_idxs.Size(); i++)
   {
      pdata_t<NData, NTag> &pdata = pdata_arr[i];

      pdata.id = ids[send_idxs[i]];

      // Copy particle data directly into pdata
      int counter = 0;
      for (int f = -1; f < GetNF(); f++)
      {
         MultiVector &pv = (f == -1 ? coords : *fields[f]);
         for (int c = 0; c < pv.GetVDim(); c++)
         {
            pdata.data[counter] = pv(send_idxs[i], c);
            counter++;
         }
      }

      // Copy tags
      for (int t = 0; t < GetNT(); t++)
      {
         Array<int> &tag_arr = *tags[t];
         pdata.tags[t] = tag_arr[send_idxs[i]];
      }

   }
   // Remove particles that will be transferred
   RemoveParticles(send_idxs);

   // Transfer particles
   sarray_transfer_ext(arr_type, &gsl_arr, send_ranks.GetData(), sizeof(unsigned int), cr.get());

   // Add received particles to this rank
   // Received particles are added to end
   unsigned int recvd = gsl_arr.n;
   pdata_arr = (pdata_t<NData, NTag>*) gsl_arr.ptr;
   int inter_np = GetNP(); // pre-recvd NP (after remove)
   int new_np = inter_np + recvd;

   // Add data individually after reserving once
   Reserve(new_np);

   // Add newly-recvd data directly to active state
   for (int i = 0; i < recvd; i++)
   {
      pdata_t<NData, NTag> &pdata = pdata_arr[i];
      int id = pdata.id;
      
      Array<int> idx_temp;
      AddParticles(Array<int>({id}), &idx_temp);
      int new_idx = idx_temp[0]; // Get index of newly-added particle

      int counter = 0;
      for (int f = -1; f < GetNF(); f++)
      {
         MultiVector &pv = (f == -1 ? coords : *fields[f]);
         for (int c = 0; c < pv.GetVDim(); c++)
         {
            pv(new_idx, c) = pdata.data[counter];
            counter++;
         }
      }

      for (int t = 0; t < GetNT(); t++)
      {
         Array<int> &tag_arr = *tags[t];
         tag_arr[new_idx] = pdata.tags[t];
      }
   }
}



#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

Particle ParticleSet::CreateParticle() const
{
   Array<int> field_vdims(GetNF());
   for (int f = 0; f < field_vdims.Size(); f++)
   {
      field_vdims[f] = fields[f]->GetVDim();
   }

   Particle p(GetDim(), field_vdims, GetNT());

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

ParticleSet::ParticleSet(int id_stride_, int id_counter_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_)
: id_stride(id_stride_),
  id_counter(id_counter_),
  coords(dim, coords_ordering)
{   
   // Initialize fields
   for (int f = 0; f < field_vdims.Size(); f++)
   {
      AddField(field_vdims[f], field_orderings[f], field_names_[f]);
   }

   // Initialize tags
   for (int t = 0; t < num_tags; t++)
   {
      AddTag(tag_names_[t]);
   }

   // Add num_particles
   Array<int> ids(num_particles);
   for (int i = 0; i < num_particles; i++)
   {
      ids[i] = id_counter;
      id_counter += id_stride;
   }
   AddParticles(ids);
}

ParticleSet::ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering)
: ParticleSet(1, 0, num_particles, dim, coords_ordering, Array<int>(), Array<Ordering::Type>(), Array<const char*>(), 0, Array<const char*>())
{

}
   
ParticleSet::ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, int num_tags, Ordering::Type all_ordering)
: ParticleSet(1, 0, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), GetEmptyNameArray(field_vdims.Size()), num_tags, GetEmptyNameArray(num_tags)) 
{

}

ParticleSet::ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_, Ordering::Type all_ordering)
: ParticleSet(1, 0, num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), field_names_, num_tags, tag_names_) 
{

}

ParticleSet::ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names)
: ParticleSet(1, 0, num_particles, dim, coords_ordering, field_vdims, field_orderings, field_names_, num_tags, tag_names)
{

}



#ifdef MFEM_USE_MPI

ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, Ordering::Type coords_ordering)
: ParticleSet(comm_, rank_num_particles, dim, coords_ordering, Array<int>(), Array<Ordering::Type>(), Array<const char*>(), 0, Array<const char*>())
{

};

ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, const Array<int> &field_vdims, int num_tags, Ordering::Type all_ordering)
: ParticleSet(comm_, rank_num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), GetEmptyNameArray(field_vdims.Size()), num_tags, GetEmptyNameArray(num_tags))
{

}

ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_, Ordering::Type all_ordering)
: ParticleSet(comm_, rank_num_particles, dim, all_ordering, field_vdims, GetOrderingArray(all_ordering, field_vdims.Size()), field_names_, num_tags, tag_names_)
{

}

ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_)
: ParticleSet(GetSize(comm_), GetRank(comm_),
               rank_num_particles,
               dim,
               coords_ordering,
               field_vdims,
               field_orderings,
               field_names_,
               num_tags,
               tag_names_)
{
   comm = comm_;
   gsl_comm = std::make_unique<gslib::comm>();
   cr = std::make_unique<gslib::crystal>();
   comm_init(gsl_comm.get(), comm);
   crystal_init(cr.get(), gsl_comm.get());
}

unsigned int ParticleSet::GetGlobalNP() const
{
   unsigned int total = GetNP();
   MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED, MPI_SUM, comm);
   return total;
}

#endif // MFEM_USE_MPI

MultiVector& ParticleSet::AddField(int vdim, Ordering::Type field_ordering, const char* field_name)
{
   if (!field_name)
   {
      field_name = GetDefaultFieldName(field_names.size()).c_str();
   }
   fields.emplace_back(std::make_unique<MultiVector>(vdim, field_ordering, GetNP()));
   field_names.emplace_back(field_name);

   return *fields.back();
}

Array<int>& ParticleSet::AddTag(const char* tag_name)
{
   if (!tag_name)
   {
      tag_name = GetDefaultTagName(tag_names.size()).c_str();
   }
   tags.emplace_back(std::make_unique<Array<int>>(GetNP()));
   tag_names.emplace_back(tag_name);

   return *tags.back();
}

void ParticleSet::AddParticle(const Particle &p)
{
   // Add the particle
   Array<int> idxs;
   AddParticles(Array<int>({id_counter}), &idxs);
   id_counter += id_stride;

   // Set the new particle data
   int idx = idxs[0];
   SetParticle(idx, p);
}

void ParticleSet::AddParticles(int num_particles, Array<int> *new_indices)
{
   Array<int> ids(num_particles);
   for (int i = 0; i < num_particles; i++)
   {
      ids[i] = id_counter;
      id_counter += id_stride;
   }

   AddParticles(ids, new_indices);
}

void ParticleSet::RemoveParticles(const Array<int> &list)
{
   int num_rm = list.Size();
   int old_np = GetNP();
   int new_np = old_np - num_rm;

   // Delete IDs
   ids.DeleteAt(list);

   // Delete data
   for (int f = -1; f < GetNF(); f++)
   {
      MultiVector &pv = (f == -1 ? coords : *fields[f]);
      pv.DeleteVectorsAt(list);
   }

   // Delete tags
   for (int t = 0; t < GetNT(); t++)
   {
      tags[t]->DeleteAt(list);
   }

}

Particle ParticleSet::GetParticle(int i) const
{
   Particle p = CreateParticle();

   Coords().GetVectorValues(i, p.Coords());
   
   for (int f = 0; f < GetNF(); f++)
   {
      Field(f).GetVectorValues(i, p.Field(f));  
   }

   for (int t = 0; t < GetNT(); t++)
   {
      p.Tag(t) = Tag(t)[i];
   }

   return std::move(p);
}

bool ParticleSet::ParticleRefValid() const
{
   if (coords.GetOrdering() == Ordering::byNODES)
   {
      return false;
   }
   for (int f = 0; f < GetNF(); f++)
   {
      if (fields[f]->GetOrdering() == Ordering::byNODES)
      {
         return false;
      }
   }
   return true;
}

Particle ParticleSet::GetParticleRef(int i)
{
   Particle p = CreateParticle();

   Coords().GetVectorRef(i, p.Coords());

   for (int f = 0; f < GetNF(); f++)
   {
      Field(f).GetVectorRef(i, p.Field(f));
   }

   for (int t = 0; t < GetNT(); t++)
   {
      p.TagMemory(t).Delete();
      p.TagMemory(t).MakeAlias((*tags[t]).GetMemory(), i, 1);
   }

   return std::move(p);
}

void ParticleSet::SetParticle(int i, const Particle &p)
{
   Coords().SetVectorValues(i, p.Coords());

   for (int f = 0; f < GetNF(); f++)
   {
      Field(f).SetVectorValues(i, p.Field(f));
   }

   for (int t = 0; t < GetNT(); t++)
   {
      Tag(t)[i] = p.Tag(t);
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

   for (int f = -1; f < GetNF(); f++)
   {
      MultiVector &pv = (f == -1 ? coords : *fields[f]);

      for (int c = 0; c < pv.GetVDim(); c++)
      {
         if (f == -1)
         {
            ss_header << (c == 0 ? "" : ",") << ax[c];
         }
         else
         {
            ss_header << "," << field_names[f] << (pv.GetVDim() > 1 ? "_" + std::to_string(c) : "");
         }
      }
   }

   for (int t = 0; t < GetNT(); t++)
   {
      ss_header << "," << tag_names[t];
   }

   ss_header << "\n";


   // Configure data
   std::stringstream ss_data;
   ss_data.precision(precision);
#ifdef MFEM_USE_MPI
   int rank = GetRank(comm);
#endif // MFEM_USE_MPI
   for (int i = 0; i < GetNP(); i++)
   {
      ss_data << ids[i];

#ifdef MFEM_USE_MPI
      ss_data << "," << rank;
#endif // MFEM_USE_MPI

      for (int f = -1; f < GetNF(); f++)
      {
         MultiVector &pv = (f == -1 ? coords : *fields[f]);

         for (int c = 0; c < pv.GetVDim(); c++)
         {
            ss_data << "," << pv(i, c);
         }
      }
      for (int t = 0; t < GetNT(); t++)
      {
         ss_data << "," << (*tags[t])[i];
      }
      ss_data << "\n";

   }

   // Write
   WriteToFile(fname, ss_header, ss_data);
}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
void ParticleSet::Redistribute(const Array<unsigned int> &rank_list)
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
   DispatchDataTransfer(send_idxs, send_ranks, std::make_index_sequence<NDATA_MAX+1>{});
}
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

ParticleSet::~ParticleSet() = default;

} // namespace mfem
