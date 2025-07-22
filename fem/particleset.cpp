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


void ParticleVector::GetParticleData(int i, Vector &pdata) const
{
   pdata.SetSize(vdim);

   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         pdata[c] = data[i + c*GetNP()];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         pdata[c] = data[c + i*vdim];
      }
   }
}

void ParticleVector::GetParticleRef(int i, Vector &pref)
{
   MFEM_VERIFY(ordering == Ordering::byVDIM, "GetParticleRef only valid when ordering byVDIM.");
   pref.MakeRef(data, i*vdim, vdim);
}

void ParticleVector::SetParticleData(int i, const Vector &pdata)
{
   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         data[i + c*GetNP()] = pdata[c];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         data[c + i*vdim] = pdata[c];
      }
   }
}

real_t& ParticleVector::ParticleData(int i, int comp)
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

const real_t& ParticleVector::ParticleData(int i, int comp) const
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

Particle::Particle(int dim, const Array<int> &dataVDims)
: coords(dim),
  data(dataVDims.Size())
{

   for (int i = 0; i < dataVDims.size(); i++)
   {
      data[i] = std::make_unique<Vector>(dataVDims[i]);
      *data[i] = 0.0;
   }
}

Particle::Particle(Vector *coords_, Vector *data_[], int numData)
: coords(coords_->GetData(), coords_->Size()),
  data(numData)
{
   for (int i = 0; i < numData; i++)
   {
      data[i] = std::make_unique<Vector>(data_[i]->GetData(), data_->Size());
   }
}

bool Particle::operator==(const Particle &rhs) const
{
   if (coords.Size() != &rhs.coords.Size())
   {
      return false;
   }
   for (int d = 0; d < coords.Size(); d++)
   {
      if (coords[d] != rhs.coords[d])
         return false;
   }
   if (data.Size() != rhs.data.Size())
   {
      return false;
   }
   for (int f = 0; f < data.Size(); f++)
   {
      if (data[i]->Size() != rhs.data[i]->Size())
      {
         return false;
      }
      for (int c = 0; c < data[i].Size(); c++)
      {
         if ((*data[i])[c] != *(rhs.data[i])[c])
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
   for (int f = 0; f < data.Size(); f++)
   {
      out << "Data " << 0 << ": (";
      for (int c = 0; c < data[i]->Size(); c++)
         out << (*data[i])[c] << ( (c+1 < data[i]->Size()) ? "," : ")\n");
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


void ParticleSet::PrintCSV(const char *fname, int precision)
{
   std::stringstream ss_header;

   // Configure header:
   std::array<char, 3> ax = {'x', 'y', 'z'};

   ss_header << "id" << ",";

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   ss_header << "rank" << ",";
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

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
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   int rank; MPI_Comm_rank(comm, &rank);
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB
   for (int i = 0; i < GetNP(); i++)
   {
      ss_data << ids[i] << ",";

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
      ss_data << rank << ",";
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

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

void ParticleSet::PrintPoint3D(const char *fname, int precision)
{

   // Configure header:
   std::stringstream ss_header;
   std::array<char, 3> ax = {'x', 'y', 'z'};
   for (int c = 0; c < fieldVDims[0]; c++)
   {
         ss_header << ax[c] << " ";
   }
   ss_header << "id\n";


   // Configure data
   std::stringstream ss_data;
   ss_data.precision(precision);
   for (int i = 0; i < GetNP(); i++)
   {
      for (int c = 0; c < fieldVDims[0]; c++)
      {
         real_t dat;
         if (ordering == Ordering::byNODES)
         {
            ss_data << data[i+c*GetNP()] << " ";
         }
         else
         {
            ss_data << data[c+i*fieldVDims[0]] << " ";
         }
      }
      ss_data << ids[i] << "\n";
   }

   // Write
   WriteToFile(fname, ss_header, ss_data);
}

void ParticleSet::WriteToFile(const char* fname, const std::stringstream &ss_header, const std::stringstream &ss_data)
{

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   // Parallel:
   int rank; MPI_Comm_rank(comm, &rank);

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

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

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

   int rank; MPI_Comm_rank(comm, &rank);
   int size; MPI_Comm_size(comm, &size);
      
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

void ParticleSet::Redistribute(const Array<unsigned int> &rank_list, FindPointsGSLIB *finder)
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
   RuntimeDispatchTransfer(send_idxs, send_ranks, std::make_index_sequence<N_MAX+1>{}, finder);

}
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB


ParticleSet::~ParticleSet() = default;
} // namespace mfem