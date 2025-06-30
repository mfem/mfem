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


namespace mfem
{

void Particle::Destroy()
{
   coords.Destroy();
   if (owning)
   {
      for (int s = 0; s < scalars.size(); s++)
         delete scalars[s];
   }
   scalars.resize(0);
   for (int v = 0; v < vectors.size(); v++)
   {
      vectors[v].Destroy();
   }
   vectors.resize(0);
}

void Particle::Copy(const Particle &p)
{
   Destroy();
   owning = p.owning;
   scalars.resize(p.GetNumScalars());
   vectors.resize(p.GetNumVectors());

   if (owning)
   {
      coords = Vector(p.coords);
      for (int s = 0; s < scalars.size(); s++)
         scalars[s] = new real_t(*p.scalars[s]);
      for (int v = 0; v < vectors.size(); v++)
         vectors[v] = Vector(p.vectors[v]);
   }
   else // data refers to external data
   {
      coords = Vector(p.coords.GetData(), p.GetSpaceDim());
      for (int s = 0; s < scalars.size(); s++)
         scalars[s] = p.scalars[s];
      for (int v = 0; v < vectors.size(); v++)
         vectors[v] = Vector(p.vectors[v].GetData(), p.GetVDim(v));
   }
}

void Particle::Steal(Particle &p)
{
   Destroy();
   owning = p.owning;
   coords = std::move(p.coords);
   scalars.resize(p.GetNumScalars());
   vectors.resize(p.GetNumVectors());

   for (int s = 0; s < scalars.size(); s++)
   {
      if (owning)
         scalars[s] = std::exchange(p.scalars[s], nullptr);
      else
         scalars[s] = p.scalars[s];
   }
      for (int v = 0; v < vectors.size(); v++)
      vectors[v] = std::move(p.vectors[v]);
}

Particle::Particle(int spaceDim, int numScalars, const Array<int> &vectorVDims)
: owning(true)
{
   coords.SetSize(spaceDim);
   coords = 0.0;

   // Initialize scalar ptrs
   scalars.resize(numScalars);
   for (int i = 0; i < scalars.size(); i++)
      scalars[i] = new real_t(0.0);

   // Initialize vectors
   vectors.resize(vectorVDims.Size());
   for (int i = 0; i < vectors.size(); i++)
   {
      vectors[i].SetSize(vectorVDims[i]);
      vectors[i] = 0.0;
   }
}

Particle::Particle(int spaceDim, int numScalars, const Array<int> &vectorVDims, real_t *in_coords, real_t *in_scalars[], real_t *in_vectors[])
: owning(false)
{
   coords = Vector(in_coords, spaceDim);

   scalars.resize(numScalars);
   for (int i = 0; i < scalars.size(); i++)
      scalars[i] = in_scalars[i];

   vectors.resize(vectorVDims.Size());
   for (int i = 0; i < vectors.size(); i++)
      vectors[i] = Vector(in_vectors[i], vectorVDims[i]);
}


bool Particle::operator==(const Particle &rhs) const
{
   bool equal = true;
   for (int d = 0; d < GetSpaceDim(); d++)
   {
      if (GetCoords()[d] != rhs.GetCoords()[d])
         equal = false;
   }
   for (int s = 0; s < GetNumScalars(); s++)
   {
      if (GetScalar(s) != rhs.GetScalar(s))
         equal = false;
   }
   for (int v = 0; v < GetNumVectors(); v++)
   {
      for (int c = 0; c < GetVDim(v); c++)
      {
         if (GetVector(v)[c] != rhs.GetVector(v)[c])
            equal = false;
      }
   }
   return equal;
}

void Particle::Print(std::ostream &out)
{
   out << "Coords: (";
   for (int d = 0; d < GetSpaceDim(); d++)
      out << GetCoords()[d] << ( (d+1 < GetSpaceDim()) ? "," : ")\n");
   for (int s = 0; s < GetNumScalars(); s++)
      out << "Scalar " << s << ": " << GetScalar(s) << "\n";
   for (int v = 0; v < GetNumVectors(); v++)
   {
      out << "Vector " << v << ": (";
      for (int c = 0; c < GetVDim(v); c++)
         out << GetVector(v)[c] << ( (c+1 < GetVDim(v)) ? "," : ")\n");
   }
}

template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::SyncVectors()
{
   // Reset Vector references to data
   for (int f = 0; f < TotalFields; f++)
   {
      fields[f] = Vector(data.data() + GetNP()*ExclScanFieldVDims[f], GetNP()*FieldVDims[f]);
   }
}

template<Ordering::Type VOrdering>
ParticleSet<VOrdering>::ParticleSet(int spaceDim, int numScalars, const Array<int> &vectorVDims)
: SpaceDim(spaceDim),
  NumScalars(numScalars),
  VectorVDims(vectorVDims),
  TotalFields(1+numScalars+vectorVDims.Size()),
  TotalComps(SpaceDim+NumScalars+vectorVDims.Sum()),
  FieldVDims(MakeFieldVDims()),
  ExclScanFieldVDims(MakeExclScanFieldVDims()),
  id_stride(1),
  id_counter(0),
  fields(vectorVDims.Size())
{

}


#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
template<Ordering::Type VOrdering>
ParticleSet<VOrdering>::ParticleSet(MPI_Comm comm_, int spaceDim, int numScalars, const Array<int> &vectorVDims)
: SpaceDim(spaceDim),
  NumScalars(numScalars),
  VectorVDims(vectorVDims),
  TotalFields(1+numScalars+vectorVDims.Size()),
  TotalComps(SpaceDim+NumScalars+vectorVDims.Sum()),
  FieldVDims(MakeFieldVDims()),
  ExclScanFieldVDims(MakeExclScanFieldVDims()),
  id_stride([&](){int s; MPI_Comm_size(comm_, &s); return s; }()),
  id_counter([&]() { int r; MPI_Comm_rank(comm_, &r); return r; }()),
  fields(vectorVDims.Size())
  comm(comm_)
{
   comm_init(&gsl_comm, comm);
   crystal_init(&cr, &gsl_comm);
}

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::AddParticle(const Particle &p, int id)
{
   int old_np = GetNP();

   if constexpr (VOrdering == Ordering::byNODES)
   {
      real_t dat;
      int offset = old_np;

      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            if (f == 0) // If processing coord comps
            {
               dat = p.GetCoords()[c];
            }
            else if (f - 1 < NumScalars) // Else if processing scalars
            {
               dat = p.GetScalar(f - 1);
            }
            else // Else processing vector comps
            {
               dat = p.GetVector(f - 1 - NumScalars)[c];
            }
            data.insert(data.begin() + offset, dat);
            offset += old_np + 1; // 1 to account for added data each loop iteration
         }
      }
   }
   else // byVDIM
   {
      const real_t* dat;
      for (int f = 0; f < TotalFields; f++)
      {
         if (f == 0)
         {
            dat = p.GetCoords().GetData();
         }
         else if (f - 1 < NumScalars)
         {
            dat = &p.GetScalar(f-1);
         }
         else
         {
            dat = p.GetVector(f-1-NumScalars).GetData();
         }
         data.insert(data.begin() + old_np*(ExclScanFieldVDims[f] + FieldVDims[f]) + ExclScanFieldVDims[f], dat, dat + FieldVDims[f]);
      }
   }

   ids.Append(id); // Add ID
   SyncVectors();
}

template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::RemoveParticles(const Array<int> &list)
{
   if (list.Size() == 0)
      return;

   int old_np = GetNP();

   // Sort the indices
   Array<int> sorted_list(list);
   sorted_list.Sort();

   if constexpr (VOrdering == Ordering::byNODES)
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
      for (int i = sorted_list[0]*FieldVDims[0]; i < data.size();  i++)
      {
         if (f + 1 < FieldVDims.Size() && i == ExclScanFieldVDims[f+1]*GetNP())
         {
            f++;
         }

         int d_idx = (i - ExclScanFieldVDims[f]*GetNP())/FieldVDims[f];
         int s_idx = ((rm_count - ExclScanFieldVDims[f]*sorted_list.Size())/FieldVDims[f]);
         if (s_idx < sorted_list.Size() && d_idx == sorted_list[s_idx])
         {
            rm_count += FieldVDims[f];
            i += FieldVDims[f] - 1;
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
   data.resize(num_new*TotalComps);
   ids.SetSize(num_new);

   SyncVectors();

}

template<Ordering::Type VOrdering>
template<Ordering::Type O, std::enable_if_t<O == Ordering::byVDIM, int>>
Particle ParticleSet<VOrdering>::GetParticleRef(int i)
{
   static_assert(VOrdering == Ordering::byVDIM, "GetParticleRef is only available when ordering is byVDIM.");

   real_t *coords = &data[i*SpaceDim];

   std::vector<real_t*> scalars;
   for (int s = 0; s < NumScalars; s++) scalars[s] = &data[(s+SpaceDim)*GetNP() + i];

   std::vector<real_t*> vectors;
   for (int v = 0; v < VectorVDims.Size(); v++) vectors[v] = &data[ExclScanFieldVDims[v+1+NumScalars]*GetNP() + i*VectorVDims[v]];

   // Construct particle before returning
   Particle p(SpaceDim, NumScalars, VectorVDims, coords, scalars.data(), vectors.data());

   return std::move(p);
}


template<Ordering::Type VOrdering>
Particle ParticleSet<VOrdering>::GetParticleData(int i) const
{
   if constexpr(VOrdering == Ordering::byNODES)
   {
      real_t *dat;
      Particle p(SpaceDim, NumScalars, VectorVDims);
      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            if (f == 0)
            {
               dat = &p.GetCoords()[c];
            }
            else if (f-1 < NumScalars)
            {
               dat = &p.GetScalar(f-1);
            }
            else
            {
               dat = &p.GetVector(f-1-NumScalars)[c];
            }
            *dat = data[i+(c+ExclScanFieldVDims[f])*GetNP()];
         }
      }
      return Particle(p);
   }
   else // byVDIM
   {
      return Particle(GetParticleRef(i));
   }

}

template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::PrintPoint3D(std::ostream &os)
{
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   MFEM_ABORT("PrintPoint3D not yet implemented in parallel");
#else
   // Write column headers
   os << "x y z id\n";

   // Write the data
   for (int i = 0 ; i < GetNP(); i++)
   {
      for (int d = 0; d < 3; d++)
      {
         real_t coord;
         if constexpr (VOrdering == Ordering::byNODES)
         {
            coord = (d < SpaceDim) ? data[i + d*GetNP()] : 0.0;
         }
         else
         {
            coord = (d < SpaceDim) ? data[d + i*SpaceDim] : 0.0;
         }
         os << ZeroSubnormal(coord) << " ";
      }
      os << ids[i] << "\n";
   }
#endif
}

template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::PrintCSVHeader(std::ostream &os, bool inc_rank)
{
   std::array<char, 3> ax = {'x', 'y', 'z'};

   os << "id,";
   if (inc_rank)
      os << "rank,";

   for (int f = 0; f < TotalFields; f++)
   {
      for (int c = 0; c < FieldVDims[f]; c++)
      {
         if (f == 0)
         {
            os << ax[c];
         }
         else if (f-1 < NumScalars)
         {
            os << "Scalar_" << f-1;
         }
         else
         {
            os << "Vector_" << f-1-NumScalars << "_" << c;
         }
         os << ((f+1 == TotalFields && c+1 == FieldVDims[f]) ? "\n" : ",");
      }
   }
}

template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::PrintCSV(std::ostream &os, bool inc_header, int *rank)
{
   // Write column headers and data
   if (inc_header)
   {
      PrintCSVHeader(os, rank);
   }

   // Write data
   for (int i = 0; i < GetNP(); i++)
   {
      os << ids[i] << ",";
      if (rank)
         os << *rank << ",";
      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            real_t dat;
            if constexpr (VOrdering == Ordering::byNODES)
            {
               dat = data[i + (ExclScanFieldVDims[f]+c)*GetNP()];
            }
            else
            {
               dat = data[c + FieldVDims[f]*i + ExclScanFieldVDims[f]*GetNP()];
            }
            os << dat;
            os << ((f+1 == TotalFields && c+1 == FieldVDims[f]) ? "\n" : ",");
         }
      }
   }
}

template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::PrintCSV(const char* fname, int precision)
{

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   // Parallel:
   int rank; MPI_Comm_rank(comm, &rank);

   MPI_File file;
   MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

   std::stringstream ss_header;
   ParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>::PrintCSVHeader(ss_header, true);
   std::string header = ss_header.str();

   // Print header
   if (rank == 0)
   {
      MPI_File_write_at(file, 0, header.data(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE);
   }

   // Get data for each rank
   std::stringstream ss;
   ss.precision(precision);
   ParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>::PrintCSV(ss, false, &rank);

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
   PrintCSV(ofs, true);

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
template<Ordering::Type VOrdering>
void ParticleSet<VOrdering>::Redistribute(const Array<unsigned int> &rank_list)
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

   // Initialize GSLIB array
   struct pdata_t
   {
      double data[SpaceDim + NumScalars + (VectorVDims + ... + 0)];
      unsigned int id;
      unsigned int proc;
   };
   gslib::array gsl_arr;
   array_init(pdata_t, &gsl_arr, send_idxs.Size());
   pdata_t *pdata_arr;
   pdata_arr = (pdata_t*) gsl_arr.ptr;
   gsl_arr.n = send_idxs.Size();

   // Set the data in pdata_arr
   // (For now, both make copies of the data for all VOrdering. One less copy for byVDIM.)
   for (int i = 0; i < send_idxs.Size(); i++)
   {
      Particle<SpaceDim, NumScalars, VectorVDims...> p;
      pdata_t &pdata = pdata_arr[i];

      pdata.id = ids[send_idxs[i]];
      pdata.proc = send_ranks[i];
      if constexpr (VOrdering == Ordering::byNODES)
      {
         p = GetParticleData(send_idxs[i]);
      }
      else
      {
         p = GetParticleRef(send_idxs[i]);
      }


      // Copy it into pdata
      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            double* dat = &pdata.data[c + ExclScanFieldVDims[f]];
            if (f == 0)
            {
               *dat = static_cast<double>(p.GetCoords()[c]);
            }
            else if (f-1 < NumScalars)
            {
               *dat = static_cast<double>(p.GetScalar(f-1));
            }
            else
            {
               *dat = static_cast<double>(p.GetVector(f-1-NumScalars)[c]);
            }
         }
      }
   }


   // Remove particles that will be transferred
   RemoveParticles(send_idxs);

   // Transfer particles
   sarray_transfer(pdata_t, &gsl_arr, proc, 1, &cr);

   // Add received particles to this rank
   unsigned int N_rec = gsl_arr.n;
   pdata_arr = (pdata_t*) gsl_arr.ptr;

   for (int i = 0; i < N_rec; i++)
   {
      pdata_t pdata = pdata_arr[i];
      if constexpr(std::is_same_v<real_t, double>)
      {
         double *coords = &pdata.data[0];
         double *scalars[NumScalars];
         double *vectors[sizeof...(VectorVDims)];

         for (int s = 0; s < NumScalars; s++)
            scalars[s] = &pdata.data[SpaceDim + s];

         for (int v = 0; v < sizeof...(VectorVDims); v++)
            vectors[v] = &pdata.data[SpaceDim + NumScalars + ExclScanFieldVDims[1+NumScalars+v]];

         Particle<SpaceDim, NumScalars, VectorVDims...> p(coords, scalars, vectors);

         AddParticle(p, pdata.id);
      }
      else // need to copy from real_t to double if real_t is not double
      {
         // TODO
      }
   }

}
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

template class ParticleSet<Ordering::byNODES>;
template class ParticleSet<Ordering::byVDIM>;

} // namespace mfem