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

// Ignore warnings from the gslib header (GCC version)
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace gslib
{
extern "C"
{
#include <gslib.h>
} // extern C
} // namespace gslib

#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB


namespace mfem
{

Particle::Particle(int dim, const Array<int> &field_vdims, int num_tags)
   : coords(dim), fields(), tags()
{
   coords = 0.0;

   fields.reserve(field_vdims.Size());
   for (int f = 0; f < field_vdims.Size(); f++)
   {
      fields.emplace_back(field_vdims[f]);
      fields.back() = 0.0;
   }

   tags.reserve(num_tags);
   for (int t = 0; t < num_tags; t++)
   {
      tags.emplace_back(1);
      tags.back()[0] = 0;
   }
}

void Particle::SetTagRef(int t, int *tag_data)
{
   MFEM_ASSERT(t >= 0 &&
               static_cast<size_t>(t) < tags.size(), "Invalid tag index");
   tags[t].MakeRef(tag_data, 1);
}

void Particle::SetFieldRef(int f, real_t *field_data)
{
   MFEM_ASSERT(f >= 0 &&
               static_cast<size_t>(f) < fields.size(), "Invalid field "
               "index");
   Vector temp(field_data, fields[f].Size());
   fields[f].MakeRef(temp, 0, fields[f].Size());
}

bool Particle::operator==(const Particle &rhs) const
{
   // Compare coordinate size and values
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
   // Compare fields vdim and values
   if (fields.size() != rhs.fields.size())
   {
      return false;
   }
   for (size_t f = 0; f < fields.size(); f++)
   {
      if (fields[f].Size() != rhs.fields[f].Size())
      {
         return false;
      }
      for (int c = 0; c < fields[f].Size(); c++)
      {
         if (fields[f][c] != rhs.fields[f][c])
         {
            return false;
         }
      }
   }
   // Compare tags size and values
   if (tags.size() != rhs.tags.size())
   {
      return false;
   }
   for (size_t t = 0; t < tags.size(); t++)
   {
      if (tags[t][0] != rhs.tags[t][0])
      {
         return false;
      }
   }

   return true;
}

void Particle::Print(std::ostream &os) const
{
   os << "Coords: (";
   for (int d = 0; d < coords.Size(); d++)
   {
      os << coords[d] << ( (d+1 < coords.Size()) ? "," : ")\n");
   }
   for (size_t f = 0; f < fields.size(); f++)
   {
      os << "Field " << f << ": (";
      for (int c = 0; c < fields[f].Size(); c++)
      {
         os << fields[f][c] << ( (c+1 < fields[f].Size()) ? "," : ")\n");
      }
   }
   for (size_t t = 0; t < tags.size(); t++)
   {
      os << "Tag " << t << ": " << tags[t][0] << "\n";
   }
}

Array<Ordering::Type> ParticleSet::GetOrderingArray(Ordering::Type o, int N)
{
   Array<Ordering::Type> ordering_arr(N);
   ordering_arr = o;
   return ordering_arr;
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
   names = nullptr;
   return names;
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
#endif // MFEM_USE_MPI

void ParticleSet::Reserve(int res)
{
   ids.Reserve(res);

   // Reserve fields
   for (int f = -1; f < GetNFields(); f++)
   {
      ParticleVector &pv = (f == -1 ? coords : *fields[f]);
      pv.Reserve(res*pv.GetVDim());
   }

   // Reserve tags
   for (int t = 0; t < GetNTags(); t++)
   {
      tags[t]->Reserve(res);
   }

}

const Array<int> ParticleSet::GetFieldVDims() const
{
   Array<int> field_vdims(GetNFields());
   for (int f = 0; f < GetNFields(); f++)
   {
      field_vdims[f] = Field(f).GetVDim();
   }
   return field_vdims;
}

void ParticleSet::AddParticles(const Array<IDType> &new_ids,
                               Array<int> *new_indices)
{
   int num_add = new_ids.Size();
   int old_np = GetNParticles();
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
   for (int f = -1; f < GetNFields(); f++)
   {
      ParticleVector &pv = (f == -1 ? coords : *fields[f]);
      pv.SetNumParticles(new_np); // does not delete existing data
   }

   // Update tags
   for (int t = 0; t < GetNTags(); t++)
   {
      tags[t]->SetSize(new_np);
   }
}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

/// \cond DO_NOT_DOCUMENT
template<size_t NBytes>
void ParticleSet::TransferParticlesImpl(ParticleSet &pset,
                                        const Array<int> &send_idxs,
                                        const Array<unsigned int> &send_ranks)
{
   struct pdata_t
   {
      alignas(real_t) std::array<std::byte, NBytes> data;
      IDType id;
   };

   int nreals = pset.GetFieldVDims().Sum() + pset.Coords().GetVDim();
   int ntags = pset.GetNTags();
   size_t nbytes = nreals*sizeof(real_t) + ntags*sizeof(int);
   MFEM_VERIFY(nbytes <= NBytes, "More data than can be packed.");

   using parr_t = pdata_t;
   gslib::array gsl_arr;
   parr_t *pdata_arr;
   array_init(parr_t, &gsl_arr, send_idxs.Size());
   pdata_arr = (parr_t*) gsl_arr.ptr;

   gsl_arr.n = send_idxs.Size();
   for (int i = 0; i < send_idxs.Size(); i++)
   {
      parr_t &pdata = pdata_arr[i];
      pdata.id = pset.GetIDs()[send_idxs[i]];

      // Copy particle data directly into pdata
      size_t counter = 0;
      for (int f = -1; f < pset.GetNFields(); f++)
      {
         ParticleVector &pv = (f == -1 ? pset.Coords() : pset.Field(f));
         for (int c = 0; c < pv.GetVDim(); c++)
         {
            std::memcpy(pdata.data.data() + counter, &pv(send_idxs[i], c),
                        sizeof(real_t));
            counter += sizeof(real_t);
         }
      }

      // Copy tags
      for (int t = 0; t < pset.GetNTags(); t++)
      {
         Array<int> &tag_arr = pset.Tag(t);
         std::memcpy(pdata.data.data() + counter, &tag_arr[send_idxs[i]],
                     sizeof(int));
         counter += sizeof(int);
      }
   }

   int nparticles = pset.GetNParticles();
   int nsend      = send_idxs.Size();

   // Transfer particles
   sarray_transfer_ext(parr_t, &gsl_arr, send_ranks.GetData(),
                       sizeof(unsigned int), pset.cr);

   // Make sure we have enough space for received particles
   int nrecv = (int) gsl_arr.n;
   int ndelete = nsend - nrecv;
   if (ndelete > 0)
   {
      // Remove unneeded particles
      auto datap = const_cast<int*>(send_idxs.GetData());
      Array<int> delete_idxs(datap + nrecv, ndelete);
      pset.RemoveParticles(delete_idxs);
   }
   else
   {
      pset.Reserve(nparticles-ndelete);
   }

   pdata_arr = (parr_t*) gsl_arr.ptr;

   // Add newly-recvd data directly to active state
   for (int i = 0; i < nrecv; i++)
   {
      parr_t &pdata = pdata_arr[i];
      IDType id = pdata.id;

      int new_loc_idx;
      if (i < nsend) // update existing particle
      {
         new_loc_idx = send_idxs[i];
         pset.UpdateID(new_loc_idx, id);
      }
      else
      {
         // add new particle
         Array<int> idx_temp;
         pset.AddParticles(Array<IDType>({id}), &idx_temp);
         new_loc_idx = idx_temp[0]; // Get index of newly-added particle
      }

      size_t counter = 0;
      for (int f = -1; f < pset.GetNFields(); f++)
      {
         ParticleVector &pv = (f == -1 ? pset.Coords() : pset.Field(f));
         for (int c = 0; c < pv.GetVDim(); c++)
         {
            real_t& val = pv(new_loc_idx, c);
            std::memcpy(&val, pdata.data.data() + counter, sizeof(real_t));
            counter += sizeof(real_t);
         }
      }

      for (int t = 0; t < pset.GetNTags(); t++)
      {
         Array<int> &tag_arr = pset.Tag(t);
         std::memcpy(&tag_arr[new_loc_idx],
                     pdata.data.data() + counter, sizeof(int));
         counter += sizeof(int);
      }
   }
   array_free(&gsl_arr);
}

template<size_t NBytes>
ParticleSet::TransferParticlesType ParticleSet::TransferParticles::Kernel()
{
   return &ParticleSet::TransferParticlesImpl<NBytes>;
}

ParticleSet::Kernels::Kernels()
{
   constexpr size_t sizd = sizeof(real_t);
   TransferParticles::Specialization<2*sizd>::Add();
   TransferParticles::Specialization<3*sizd>::Add();
   TransferParticles::Specialization<4*sizd>::Add();
   TransferParticles::Specialization<8*sizd>::Add();
   TransferParticles::Specialization<12*sizd>::Add();
   TransferParticles::Specialization<16*sizd>::Add();
   TransferParticles::Specialization<20*sizd>::Add();
   TransferParticles::Specialization<24*sizd>::Add();
   TransferParticles::Specialization<28*sizd>::Add();
   TransferParticles::Specialization<32*sizd>::Add();
   TransferParticles::Specialization<36*sizd>::Add();
   TransferParticles::Specialization<40*sizd>::Add();
}

auto ParticleSet::TransferParticles::Fallback(size_t bufsize)
-> ParticleSet::TransferParticlesType
{
   constexpr size_t sizd = sizeof(real_t);
   if (bufsize < 4*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<4*sizd>;
   }
   else if (bufsize < 8*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<8*sizd>;
   }
   else if (bufsize < 12*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<12*sizd>;
   }
   else if (bufsize < 16*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<16*sizd>;
   }
   else if (bufsize < 20*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<20*sizd>;
   }
   else if (bufsize < 24*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<24*sizd>;
   }
   else if (bufsize < 28*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<28*sizd>;
   }
   else if (bufsize < 32*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<32*sizd>;
   }
   else if (bufsize < 36*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<36*sizd>;
   }
   else if (bufsize < 40*sizd)
   {
      return &ParticleSet::TransferParticlesImpl<40*sizd>;
   }
   return &ParticleSet::TransferParticlesImpl<60*sizd>;
}
/// \endcond DO_NOT_DOCUMENT

void ParticleSet::Redistribute(const Array<unsigned int> &rank_list)
{
   MFEM_ASSERT(rank_list.Size() == GetNParticles(),
               "rank_list must be of size GetNParticles().");

   int rank = GetRank(comm);

   // Get particles to be transferred
   // (Avoid unnecessary copies of particle data into and out of buffers)
   Array<int> send_idxs;
   Array<unsigned int> send_ranks;
   send_idxs.Reserve(rank_list.Size());
   send_ranks.Reserve(rank_list.Size());
   for (int i = 0; i < rank_list.Size(); i++)
   {
      if (rank != static_cast<int>(rank_list[i]))
      {
         send_idxs.Append(i);
         send_ranks.Append(rank_list[i]);
      }
   }

   // Compute number of bytes of a single particle
   int nreals = GetFieldVDims().Sum() + coords.GetVDim();
   int ntags = GetNTags();
   size_t nbytes = nreals*sizeof(real_t) + ntags*sizeof(int);

   // Dispatch to appropriate redistribution function for this size
   TransferParticles::Run(nbytes, *this, send_idxs, send_ranks);
}

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

Particle ParticleSet::CreateParticle() const
{
   return Particle(GetDim(), GetFieldVDims(), GetNTags());
}

void ParticleSet::WriteToFile(const char *fname,
                              const std::stringstream &ss_header, const std::stringstream &ss_data)
{

#ifdef MFEM_USE_MPI
   // Parallel:
   int rank = GetRank(comm);

   MPI_File_delete(fname, MPI_INFO_NULL); // delete old file if it exists
   MPI_File file;
   int mpi_err = MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &file);
   MFEM_VERIFY(mpi_err == MPI_SUCCESS, "MPI_File_open failed.");

   // Print header
   if (rank == 0)
   {
      MPI_File_write_at(file, 0, ss_header.str().data(), ss_header.str().size(),
                        MPI_CHAR, MPI_STATUS_IGNORE);
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
   MPI_File_write_at_all(file, offset, ss_data.str().data(),
                         data_size, MPI_BYTE, MPI_STATUS_IGNORE);

   // Close file
   MPI_File_close(&file);
#else
   // Serial:
   std::ofstream ofs(fname);
   MFEM_VERIFY(ofs.is_open() && !ofs.fail(),
               "Error: Could not open file " << fname << " for writing.");
   ofs << ss_header.str() << ss_data.str();
   ofs.close();
#endif // MFEM_USE_MPI
}

ParticleSet::ParticleSet(int id_stride_, IDType id_counter_, int num_particles,
                         int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims,
                         const Array<Ordering::Type> &field_orderings,
                         const Array<const char*> &field_names_, int num_tags,
                         const Array<const char*> &tag_names_)
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
   Array<IDType> init_ids(num_particles);
   for (int i = 0; i < num_particles; i++)
   {
      init_ids[i] = id_counter;
      id_counter += id_stride;
   }
   AddParticles(init_ids);
}

bool ParticleSet::IsValidParticle(const Particle &p) const
{
   if (p.GetDim() != GetDim())
   {
      return false;
   }
   if (p.GetNFields() != GetNFields())
   {
      return false;
   }
   for (int f = 0; f < GetNFields(); f++)
   {
      if (p.GetFieldVDim(f) != Field(f).GetVDim())
      {
         return false;
      }
   }
   if (p.GetNTags() != GetNTags())
   {
      return false;
   }

   return true;

}

ParticleSet::ParticleSet(int num_particles, int dim,
                         Ordering::Type coords_ordering)
   : ParticleSet(1, 0, num_particles, dim, coords_ordering, Array<int>(),
                 Array<Ordering::Type>(), Array<const char*>(), 0,
                 Array<const char*>())
{

}

ParticleSet::ParticleSet(int num_particles, int dim,
                         const Array<int> &field_vdims, int num_tags,
                         Ordering::Type all_ordering)
   : ParticleSet(1, 0, num_particles, dim, all_ordering, field_vdims,
                 GetOrderingArray(all_ordering, field_vdims.Size()),
                 GetEmptyNameArray(field_vdims.Size()), num_tags,
                 GetEmptyNameArray(num_tags))
{
}

ParticleSet::ParticleSet(int num_particles, int dim,
                         const Array<int> &field_vdims, const Array<const
                         char*> &field_names_, int num_tags,
                         const Array<const char*> &tag_names_,
                         Ordering::Type all_ordering)
   : ParticleSet(1, 0, num_particles, dim, all_ordering, field_vdims,
                 GetOrderingArray(all_ordering, field_vdims.Size()),
                 field_names_, num_tags,
                 tag_names_)
{

}

ParticleSet::ParticleSet(int num_particles, int dim,
                         Ordering::Type coords_ordering,
                         const Array<int> &field_vdims,
                         const Array<Ordering::Type> &field_orderings,
                         const Array<const char*> &field_names_, int num_tags,
                         const Array<const char*> &tag_names_)
   : ParticleSet(1, 0, num_particles, dim, coords_ordering, field_vdims,
                 field_orderings, field_names_, num_tags, tag_names_)
{

}



#ifdef MFEM_USE_MPI
ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
                         Ordering::Type coords_ordering)
   : ParticleSet(comm_, rank_num_particles, dim, coords_ordering, Array<int>(),
                 Array<Ordering::Type>(), Array<const char*>(), 0,
                 Array<const char*>())
{

};

ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
                         const Array<int> &field_vdims, int num_tags,
                         Ordering::Type all_ordering)
   : ParticleSet(comm_, rank_num_particles, dim, all_ordering, field_vdims,
                 GetOrderingArray(all_ordering, field_vdims.Size()),
                 GetEmptyNameArray(field_vdims.Size()), num_tags,
                 GetEmptyNameArray(num_tags))
{

}

ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
                         const Array<int> &field_vdims, const Array<const
                         char*> &field_names_,
                         int num_tags, const Array<const char*> &tag_names_,
                         Ordering::Type all_ordering)
   : ParticleSet(comm_, rank_num_particles, dim, all_ordering, field_vdims,
                 GetOrderingArray(all_ordering, field_vdims.Size()),
                 field_names_, num_tags,
                 tag_names_)
{

}

ParticleSet::ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
                         Ordering::Type coords_ordering,
                         const Array<int> &field_vdims,
                         const Array<Ordering::Type> &field_orderings,
                         const Array<const char*> &field_names_, int num_tags,
                         const Array<const char*> &tag_names_)
   : ParticleSet(GetSize(comm_), (IDType)GetRank(comm_),
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
#ifdef MFEM_USE_GSLIB
   gsl_comm = new gslib::comm;
   cr       = new gslib::crystal;
   comm_init(gsl_comm, comm);
   crystal_init(cr, gsl_comm);
#endif // MFEM_USE_GSLIB
}
#endif // MFEM_USE_MPI

ParticleSet::IDType ParticleSet::GetGlobalNParticles() const
{
   IDType total = (IDType)GetNParticles();
#ifdef MFEM_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED_LONG_LONG,
                 MPI_SUM, comm);
#endif // MFEM_USE_MPI
   return total;
}

int ParticleSet::AddField(int vdim, Ordering::Type field_ordering,
                          const char* field_name)
{
   std::string field_name_str(field_name ? field_name : "");
   if (!field_name)
   {
      field_name_str = GetDefaultFieldName(field_names.size());
   }
   fields.emplace_back(std::make_unique<ParticleVector>(vdim, field_ordering,
                                                        GetNParticles()));
   field_names.emplace_back(field_name_str);

   return GetNFields() - 1;
}

int ParticleSet::AddTag(const char* tag_name)
{
   std::string tag_name_str(tag_name ? tag_name : "");
   if (!tag_name)
   {
      tag_name_str = GetDefaultTagName(tag_names.size());
   }
   tags.emplace_back(std::make_unique<Array<int>>(GetNParticles()));
   tag_names.emplace_back(tag_name_str);

   return GetNTags() - 1;
}

void ParticleSet::AddParticle(const Particle &p)
{
   MFEM_ASSERT(IsValidParticle(p),
               "Particle is incompatible with ParticleSet.");

   // Add the particle
   Array<int> idxs;
   AddParticles(Array<IDType>({id_counter}), &idxs);
   id_counter += id_stride;

   // Set the new particle data
   int idx = idxs[0];
   SetParticle(idx, p);
}

void ParticleSet::AddParticles(int num_particles, Array<int> *new_indices)
{
   Array<IDType> add_ids(num_particles);
   for (int i = 0; i < num_particles; i++)
   {
      add_ids[i] = id_counter;
      id_counter += id_stride;
   }

   AddParticles(add_ids, new_indices);
}

void ParticleSet::RemoveParticles(const Array<int> &list)
{
   // Delete IDs
   ids.DeleteAt(list);

   // Delete data
   for (int f = -1; f < GetNFields(); f++)
   {
      ParticleVector &pv = (f == -1 ? coords : *fields[f]);
      pv.DeleteParticles(list);
   }

   // Delete tags
   for (int t = 0; t < GetNTags(); t++)
   {
      tags[t]->DeleteAt(list);
   }
}

Particle ParticleSet::GetParticle(int i) const
{
   Particle p = CreateParticle();

   Coords().GetValues(i, p.Coords());

   for (int f = 0; f < GetNFields(); f++)
   {
      Field(f).GetValues(i, p.Field(f));
   }

   for (int t = 0; t < GetNTags(); t++)
   {
      p.Tag(t) = Tag(t)[i];
   }

   return p;
}

bool ParticleSet::IsParticleRefValid() const
{
   if (coords.GetOrdering() == Ordering::byNODES)
   {
      return false;
   }
   for (int f = 0; f < GetNFields(); f++)
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

   Coords().GetValuesRef(i, p.Coords());

   for (int f = 0; f < GetNFields(); f++)
   {
      MFEM_ASSERT(Field(f).GetOrdering() == Ordering::byVDIM,
                  "GetParticleRef only valid when all fields ordered byVDIM.");
      p.SetFieldRef(f, Field(f).GetData() + i*Field(f).GetVDim());
   }

   for (int t = 0; t < GetNTags(); t++)
   {
      p.SetTagRef(t, &(*tags[t])[i]);
   }

   return p;
}

void ParticleSet::SetParticle(int i, const Particle &p)
{
   MFEM_ASSERT(IsValidParticle(p),
               "Particle is incompatible with ParticleSet.");

   Coords().SetValues(i, p.Coords());

   for (int f = 0; f < GetNFields(); f++)
   {
      Field(f).SetValues(i, p.Field(f));
   }

   for (int t = 0; t < GetNTags(); t++)
   {
      Tag(t)[i] = p.Tag(t);
   }
}

void ParticleSet::PrintCSV(const char *fname, int precision)
{
   Array<int> all_field_idxs(GetNFields()), all_tag_idxs(GetNTags());

   for (int f = 0; f < GetNFields(); f++)
   {
      all_field_idxs[f] = f;
   }

   for (int t = 0; t < GetNTags(); t++)
   {
      all_tag_idxs[t] = t;
   }

   PrintCSV(fname, all_field_idxs, all_tag_idxs, precision);
}

void ParticleSet::PrintCSV(const char *fname, const Array<int> &field_idxs,
                           const Array<int> &tag_idxs, int precision)
{
   std::stringstream ss_header;

   // Configure header:
   ss_header << "id";

#ifdef MFEM_USE_MPI
   ss_header << ",rank";
#endif // MFEM_USE_MPI

   std::array<char, 3> ax = {'X', 'Y', 'Z'};
   for (int c = 0; c < coords.GetVDim(); c++)
   {
      ss_header << "," << ax[c];
   }

   for (int f = 0; f < field_idxs.Size(); f++)
   {
      ParticleVector &pv = *fields[field_idxs[f]];
      for (int c = 0; c < pv.GetVDim(); c++)
      {
         ss_header << "," << field_names[field_idxs[f]] <<
                   (pv.GetVDim() > 1 ? "_" + std::to_string(c) : "");
      }
   }

   for (int t = 0; t < tag_idxs.Size(); t++)
   {
      ss_header << "," << tag_names[tag_idxs[t]];
   }
   ss_header << "\n";

   // Configure data
   std::stringstream ss_data;
   ss_data.precision(precision);
#ifdef MFEM_USE_MPI
   int rank = GetRank(comm);
#endif // MFEM_USE_MPI
   for (int i = 0; i < GetNParticles(); i++)
   {
      ss_data << ids[i];
#ifdef MFEM_USE_MPI
      ss_data << "," << rank;
#endif // MFEM_USE_MPI

      for (int c = 0; c < coords.GetVDim(); c++)
      {
         ss_data << "," << coords(i, c);
      }
      for (int f = 0; f < field_idxs.Size(); f++)
      {
         ParticleVector &pv = *fields[field_idxs[f]];
         for (int c = 0; c < pv.GetVDim(); c++)
         {
            ss_data << "," << pv(i, c);
         }
      }
      for (int t = 0; t < tag_idxs.Size(); t++)
      {
         ss_data << "," << (*tags[tag_idxs[t]])[i];
      }
      ss_data << "\n";
   }

   // Write
   WriteToFile(fname, ss_header, ss_data);
}

ParticleSet::~ParticleSet()
{
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   if (gsl_comm)
   {
      if (!Mpi::IsFinalized())  // currently segfaults inside gslib otherwise
      {
         crystal_free(cr);
         comm_free(gsl_comm);
         delete gsl_comm;
         delete cr;
      }
   }
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB
}


} // namespace mfem
