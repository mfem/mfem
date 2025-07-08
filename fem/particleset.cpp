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


Particle::Particle(const ParticleMeta &pmeta)
: meta(pmeta),
  coords(pmeta.SpaceDim()),
  props(pmeta.NumProps()),
  state(pmeta.NumStateVars())
{

   for (int i = 0; i < state.size(); i++)
   {
      state[i].SetSize(pmeta.StateVDim(i));
      state[i] = 0.0;
   }
}

bool Particle::operator==(const Particle &rhs) const
{
   if (&meta.get() != &rhs.meta.get())
   {
      return false;
   }
   for (int d = 0; d < meta.get().SpaceDim(); d++)
   {
      if (coords[d] != rhs.coords[d])
         return false;
   }
   for (int s = 0; s < meta.get().NumProps(); s++)
   {
      if (props.at(s) != rhs.props.at(s))
         return false;
   }
   for (int v = 0; v < meta.get().NumStateVars(); v++)
   {
      for (int c = 0; c < meta.get().StateVDim(v); c++)
      {
         if (state.at(v)[c] != rhs.state.at(v)[c])
            return false;
      }
   }
   return true;
}

void Particle::Print(std::ostream &out) const
{
   out << "Coords: (";
   for (int d = 0; d < coords.Size(); d++)
      out << coords[d] << ( (d+1 < coords.Size()) ? "," : ")\n");
   for (int s = 0; s < props.size(); s++)
      out << "Property " << s << ": " << props.at(s) << "\n";
   for (int v = 0; v < state.size(); v++)
   {
      out << "State Variable " << v << ": (";
      for (int c = 0; c < state.at(v).Size(); c++)
         out << state.at(v)[c] << ( (c+1 < state.at(v).Size()) ? "," : ")\n");
   }
}

void ParticleSet::SyncVectors()
{
   // Reset Vector references to data
   for (int f = 0; f < totalFields; f++)
   {
      fields[f] = Vector(data.data() + GetNP()*exclScanFieldVDims[f], GetNP()*fieldVDims[f]);
   }
}

ParticleSet::ParticleSet(const ParticleMeta &meta_, Ordering::Type ordering_)
: ordering(ordering_),
  meta(meta_),
  totalFields(1+meta.NumProps()+meta.NumStateVars()),
  totalComps(meta.SpaceDim()+meta.NumProps()+meta.StateVDims().Sum()),
  fieldVDims(MakeFieldVDims()),
  exclScanFieldVDims(MakeExclScanFieldVDims()),
  id_stride(1),
  id_counter(0),
  fields(totalFields)
{

}


#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

ParticleSet::ParticleSet(MPI_Comm comm_, const ParticleMeta &meta_, Ordering::Type ordering_)
: ordering(ordering_),
  meta(meta_),
  totalFields(1+meta.NumProps()+meta.NumStateVars()),
  totalComps(meta.SpaceDim()+meta.NumProps()+meta.StateVDims().Sum()),
  fieldVDims(MakeFieldVDims()),
  exclScanFieldVDims(MakeExclScanFieldVDims()),
  id_stride([&](){int s; MPI_Comm_size(comm_, &s); return s; }()),
  id_counter([&]() { int r; MPI_Comm_rank(comm_, &r); return r; }()),
  fields(totalFields),
  comm(comm_),
  gsl_comm(std::make_unique<gslib::comm>()),
  cr(std::make_unique<gslib::crystal>())
{
   comm_init(gsl_comm.get(), comm);
   crystal_init(cr.get(), gsl_comm.get());
}

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB


void ParticleSet::AddParticle(const Particle &p, int id)
{
   MFEM_ASSERT(&p.GetMeta() == &meta, "Input particle metadata does not match the ParticleSet's!");

   int old_np = GetNP();

   if (ordering == Ordering::byNODES)
   {
      real_t dat;
      int offset = old_np;

      for (int f = 0; f < totalFields; f++)
      {
         for (int c = 0; c < fieldVDims[f]; c++)
         {
            if (f == 0) // If processing coord comps
            {
               dat = p.GetCoords()[c];
            }
            else if (f - 1 < meta.NumProps()) // Else if processing scalars
            {
               dat = p.GetProperty(f - 1);
            }
            else // Else processing vector comps
            {
               dat = p.GetStateVar(f - 1 - meta.NumProps())[c];
            }
            data.insert(data.begin() + offset, dat);
            offset += old_np + 1; // 1 to account for added data each loop iteration
         }
      }
   }
   else // byVDIM
   {
      const real_t* dat;
      for (int f = 0; f < totalFields; f++)
      {
         if (f == 0)
         {
            dat = p.GetCoords().GetData();
         }
         else if (f - 1 < meta.NumProps())
         {
            dat = &p.GetProperty(f-1);
         }
         else
         {
            dat = p.GetStateVar(f-1-meta.NumProps()).GetData();
         }
         data.insert(data.begin() + old_np*(exclScanFieldVDims[f] + fieldVDims[f]) + exclScanFieldVDims[f], dat, dat + fieldVDims[f]);
      }
   }

   ids.Append(id); // Add ID
   SyncVectors();
}

void ParticleSet::RemoveParticles(const Array<int> &list)
{
   if (list.Size() == 0)
      return;

   int old_np = GetNP();

   // Sort the indices
   Array<int> sorted_list(list);
   sorted_list.Sort();

   if (ordering == Ordering::byNODES)
   {
      int rm_count = 0;
      for (int i = sorted_list[0]; i < data.size(); i++)
      {
         if (i % GetNP() == sorted_list[rm_count % sorted_list.Size()])
         {
            rm_count++;
         }
         else
         {
            data[i-rm_count] = data[i]; // Shift elements rm_count
         }
      }
   }
   else // byVDIM
   {
      int rm_count = 0;

      int f = 0;
      for (int i = sorted_list[0]*fieldVDims[0]; i < data.size();  i++)
      {
         if (f + 1 < fieldVDims.Size() && i == exclScanFieldVDims[f+1]*GetNP())
         {
            f++;
         }

         int d_idx = (i - exclScanFieldVDims[f]*GetNP())/fieldVDims[f];
         int s_idx = ((rm_count - exclScanFieldVDims[f]*sorted_list.Size())/fieldVDims[f]);
         if (s_idx < sorted_list.Size() && d_idx == sorted_list[s_idx])
         {
            rm_count += fieldVDims[f];
            i += fieldVDims[f] - 1;
         }
         else
         {
            data[i-rm_count] = data[i];
         }
      }
   }

   // Remove old IDs
   int rm_idx = 0;
   for (int i = 0; i < old_np; i++)
   {
      if (rm_idx < sorted_list.Size() && i == sorted_list[rm_idx])
      {
         rm_idx++;
      }
      else
      {
         ids[i-rm_idx] = ids[i];
      }
   }

   // Resize / remove tails
   int num_new = old_np - list.Size();
   data.resize(num_new*totalComps);
   ids.SetSize(num_new);

   SyncVectors();

}


void ParticleSet::GetParticle(int i, Particle &p) const
{
   MFEM_ASSERT(&p.GetMeta() == &meta, "Input particle metadata does not match the ParticleSet's!");

   if (ordering == Ordering::byNODES)
   {
      real_t *dat;
      for (int f = 0; f < totalFields; f++)
      {
         for (int c = 0; c < fieldVDims[f]; c++)
         {
            if (f == 0)
            {
               dat = &p.GetCoords()[c];
            }
            else if (f-1 < meta.NumProps())
            {
               dat = &p.GetProperty(f-1);
            }
            else
            {
               dat = &p.GetStateVar(f-1-meta.NumProps())[c];
            }
            *dat = data[i+(c+exclScanFieldVDims[f])*GetNP()];
         }
      }
   }
   else // byVDIM
   {
      real_t *dat;
      for (int f = 0; f < totalFields; f++)
      {
         for (int c = 0; c < fieldVDims[f]; c++)
         {
            if (f == 0)
            {
               dat = &p.GetCoords()[c];
            }
            else if (f-1 < meta.NumProps())
            {
               dat = &p.GetProperty(f-1);
            }
            else
            {
               dat = &p.GetStateVar(f-1-meta.NumProps())[c];
            }
            *dat = data[c+i*fieldVDims[f]+exclScanFieldVDims[f]*GetNP()];
         }
      }
   }

}

void ParticleSet::SetParticle(int i, const Particle &p)
{
   MFEM_ASSERT(&p.GetMeta() == &meta, "Input particle metadata does not match the ParticleSet's!");

   if (ordering == Ordering::byNODES)
   {
      const real_t *dat;
      for (int f = 0; f < totalFields; f++)
      {
         for (int c = 0; c < fieldVDims[f]; c++)
         {
            if (f == 0)
            {
               dat = &p.GetCoords()[c];
            }
            else if (f-1 < meta.NumProps())
            {
               dat = &p.GetProperty(f-1);
            }
            else
            {
               dat = &p.GetStateVar(f-1-meta.NumProps())[c];
            }
            data[i+(c+exclScanFieldVDims[f])*GetNP()] = *dat;
         }
      }
   }
   else // byVDIM
   {
      const real_t *dat;
      for (int f = 0; f < totalFields; f++)
      {
         for (int c = 0; c < fieldVDims[f]; c++)
         {
            if (f == 0)
            {
               dat = &p.GetCoords()[c];
            }
            else if (f-1 < meta.NumProps())
            {
               dat = &p.GetProperty(f-1);
            }
            else
            {
               dat = &p.GetStateVar(f-1-meta.NumProps())[c];
            }
            data[c+i*fieldVDims[f]+exclScanFieldVDims[f]*GetNP()] = *dat;
         }
      }
   }
}

// void ParticleSet::PrintPoint3D(std::ostream &os)
// {
// #if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
//    MFEM_ABORT("PrintPoint3D not yet implemented in parallel");
// #else
//    // Write column headers
//    os << "x y z id\n";

//    // Write the data
//    for (int i = 0 ; i < GetNP(); i++)
//    {
//       for (int d = 0; d < 3; d++)
//       {
//          real_t coord;
//          if (ordering == Ordering::byNODES)
//          {
//             coord = (d < meta.SpaceDim()) ? data[i + d*GetNP()] : 0.0;
//          }
//          else
//          {
//             coord = (d < meta.SpaceDim()) ? data[d + i*meta.SpaceDim()] : 0.0;
//          }
//          os << ZeroSubnormal(coord) << " ";
//       }
//       os << ids[i] << "\n";
//    }
// #endif
// }

void ParticleSet::PrintHeader(std::ostream &os, bool inc_rank, const char *delimiter)
{
   std::array<char, 3> ax = {'x', 'y', 'z'};

   os << "id" << delimiter;
   if (inc_rank)
      os << "rank" << delimiter;

   for (int f = 0; f < totalFields; f++)
   {
      for (int c = 0; c < fieldVDims[f]; c++)
      {
         if (f == 0)
         {
            os << ax[c];
         }
         else if (f-1 < meta.NumProps())
         {
            os << "Property_" << f-1;
         }
         else
         {
            os << "StateVariable_" << f-1-meta.NumProps() << "_" << c;
         }
         os << ((f+1 == totalFields && c+1 == fieldVDims[f]) ? "\n" : delimiter);
      }
   }
}

void ParticleSet::PrintData(std::ostream &os, bool inc_header, const char *delimiter, int *rank)
{
   // Write column headers and data
   if (inc_header)
   {
      PrintHeader(os, rank, delimiter);
   }

   // Write data
   for (int i = 0; i < GetNP(); i++)
   {
      os << ids[i] << delimiter;
      if (rank)
         os << *rank << delimiter;
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
            os << dat;
            os << ((f+1 == totalFields && c+1 == fieldVDims[f]) ? "\n" : delimiter);
         }
      }
   }
}

void ParticleSet::Print(const char* fname, int precision, const char *delimiter)
{

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   // Parallel:
   int rank; MPI_Comm_rank(comm, &rank);

   MPI_File file;
   MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

   std::stringstream ss_header;
   ParticleSet::PrintHeader(ss_header, true, delimiter);
   std::string header = ss_header.str();

   // Print header
   if (rank == 0)
   {
      MPI_File_write_at(file, 0, header.data(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE);
   }

   // Get data for each rank
   std::stringstream ss;
   ss.precision(precision);
   ParticleSet::PrintData(ss, false, delimiter, &rank);

   // Compute the size in bytes
   std::string s_dat = ss.str();
   MPI_Offset dat_size = s_dat.size();
   MPI_Offset offset;

   // Compute the offsets using an exclusive scan
   MPI_Exscan(&dat_size, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
   if (rank == 0)
   {
      offset = 0;
   }

   // Add offset from the header
   offset += header.size();

   // Write data collectively
   MPI_File_write_at_all(file, offset, s_dat.data(), dat_size, MPI_BYTE, MPI_STATUS_IGNORE);

   // Close file
   MPI_File_close(&file);

#else
   // Serial:
   std::ofstream ofs(fname);
   ofs.precision(precision);
   PrintData(ofs, true, delimiter);

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

template<std::size_t N>
void ParticleSet::Transfer(const Array<int> &send_idxs, const Array<int> &send_ranks)
{
   gslib::array gsl_arr;
   array_init(pdata_t<N>, &gsl_arr, send_idxs.Size());
   pdata_t<N> *pdata_arr;
   pdata_arr = (pdata_t<N>*) gsl_arr.ptr;
   gsl_arr.n = send_idxs.Size();

   // Set the data in pdata_arr
   for (int i = 0; i < send_idxs.Size(); i++)
   {
      pdata_t<N> &pdata = pdata_arr[i];

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
   }


   // Remove particles that will be transferred
   RemoveParticles(send_idxs);

   // Transfer particles
   sarray_transfer(pdata_t<N>, &gsl_arr, proc, 1, cr.get());

   // Add received particles to this rank
   unsigned int recvd = gsl_arr.n;
   pdata_arr = (pdata_t<N>*) gsl_arr.ptr;

   for (int i = 0; i < recvd; i++)
   {
      pdata_t<N> pdata = pdata_arr[i];
      if constexpr(std::is_same_v<real_t, double>)
      {
         // Create a particle, copy data from buffer to it, then add particle
         // (TODO: Copy directly from received pdata to data!!)
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
   }
}

void ParticleSet::Redistribute(const Array<unsigned int> &rank_list)
{
   MFEM_ASSERT(rank_list.Size() == GetNP(), "rank_list.Size() != GetNP()");

   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Get particles to be transferred
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
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB


ParticleSet::~ParticleSet() = default;
} // namespace mfem