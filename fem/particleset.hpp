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

#ifndef MFEM_PARTICLESET
#define MFEM_PARTICLESET

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "fespace.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
// Forward declare gslib structs to avoid including macro-filled header:
namespace gslib
{
extern "C"
{
struct comm;
struct crystal;
} //extern C
} // gslib
#endif

#include <vector>
#include <numeric>

namespace mfem
{

// -----------------------------------------------------------------------------------------------------
// Define Particle class

class Particle
{
protected:
   bool owning;

   Vector coords;
   std::vector<real_t*> scalars;
   std::vector<Vector> vectors;

   void Destroy();
   void Copy(const Particle &p);
   void Steal(Particle &p);

public:
   int GetSpaceDim() const { return coords.Size(); }
   int GetNumScalars() const { return scalars.size(); }
   int GetNumVectors() const { return vectors.size(); }
   int GetVDim(int v) const { return vectors[v].Size(); }
   const Array<int> GetVectorVDims() const
   {
      Array<int> vectorVDims(GetNumVectors());
      for (int i = 0; i < GetNumVectors(); i++)
         vectorVDims[i] = GetVDim(i);
      return vectorVDims;
   }

   /// Create a new particle which owns its own data
   Particle(int spaceDim, int numScalars, const Array<int> &vectorVDims);

   Particle(int spaceDim, int numScalars, std::initializer_list<int> vectorVDims)
   : Particle(spaceDim, numScalars, Array<int>(vectorVDims)) {};

   /// Create a new particle whose data references external data
   Particle(int spaceDim, int numScalars, const Array<int> &vectorVDims, real_t *in_coords, real_t *in_scalars[], real_t *in_vectors[]);

   /// Copy ctor
   Particle(const Particle &p) : Particle(p.GetSpaceDim(), p.GetNumScalars(), p.GetVectorVDims()) { Copy(p); }

   /// Move ctor
   Particle(Particle &&p) : Particle(p.GetSpaceDim(), p.GetNumScalars(), p.GetVectorVDims()) { Steal(p); }

   /// Copy assign
   Particle& operator=(const Particle &p) { Copy(p); return *this; }

   /// Move assign
   Particle& operator=(Particle &&p) { Steal(p); return *this; }

   Vector& GetCoords() { return coords; }

   real_t& GetScalar(int s) { return *scalars[s]; }

   Vector& GetVector(int v) { return vectors[v]; }

   const Vector& GetCoords() const { return coords; }

   const real_t& GetScalar(int s) const { return *scalars[s]; }

   const Vector& GetVector(int v) const { return vectors[v]; }

   bool operator==(const Particle &rhs) const;

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   ~Particle() { Destroy(); }

   void Print(std::ostream &out=mfem::out);
};

// -----------------------------------------------------------------------------------------------------
// Define ParticleSet

template<Ordering::Type VOrdering=Ordering::byVDIM>
class ParticleSet
{
protected:
   const int SpaceDim;
   const int NumScalars;
   const Array<int> VectorVDims; // VDims of vectors
   const int TotalFields; // Total number of fields / particle
   const int TotalComps; // Total comps / particle
   const Array<int> FieldVDims;
   const Array<int> ExclScanFieldVDims;

   const Array<int> MakeFieldVDims()
   {
      Array<int> temp(TotalFields);
      temp[0] = SpaceDim;
      for (int s = 0; s < NumScalars; s++) temp[1+s] = 1;
      for (int v = 0; v < VectorVDims.Size(); v++) temp[1+NumScalars+v] = VectorVDims[v];
      return temp;
   }
   const Array<int> MakeExclScanFieldVDims()
   {
      Array<int> temp(TotalFields);
      temp[0] = 0;
      for (int i = 1; i < TotalFields; i++)
      {
         temp[i] = temp[i-1] + FieldVDims[i-1];
      }
      return temp;
   }

   const std::size_t id_stride;
   std::size_t id_counter;

   std::vector<real_t> data; // Stores ALL particle data
   std::vector<Vector> fields; // User-facing Vectors, referencing data
   Array<unsigned int> ids; // Particle IDs

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   MPI_Comm comm;
   std::unique_ptr<gslib::comm> gsl_comm;
   std::unique_ptr<gslib::crystal> cr;
   template<std::size_t N>
   struct pdata_t
   {
      double data[N];
      unsigned int id, proc;
   };
   static constexpr std::size_t N_MAX = 100;

   template<std::size_t N>
   void Transfer(const Array<int> &send_idxs, const Array<int> &send_ranks);

   template<std::size_t... Ns>
   void RuntimeDispatchTransfer(const Array<int> &send_idxs, const Array<int> &send_ranks, std::index_sequence<Ns...>)
   {
      bool success = ( (TotalComps == Ns ? (Transfer<Ns>(send_idxs, send_ranks),true) : false) || ...);
      MFEM_ASSERT(success, "Particles with field size above 100 are currently not supported for redistributing. Please submit a PR to request a particular particle size above this.");
   }

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB


   /// Sync Vectors in \ref fields
   void SyncVectors();

   /// Add particle w/ given ID
   void AddParticle(const Particle &p, int id);

   void PrintCSVHeader(std::ostream &os, bool inc_rank);
   void PrintCSV(std::ostream &os, bool inc_header, int *rank=nullptr);

public:

   static constexpr Ordering::Type GetOrdering() { return VOrdering; }

   /// Serial constructor
   ParticleSet(int spaceDim, int numScalars, const Array<int> &vectorVDims);

   ParticleSet(int spaceDim, int numScalars, std::initializer_list<int> vectorVDims)
   : ParticleSet(spaceDim, numScalars, Array<int>(vectorVDims)) {};


#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

   /// Parallel constructor
   ParticleSet(MPI_Comm comm_, int spaceDim, int numScalars, const Array<int> &vectorVDims);

   ParticleSet(MPI_Comm comm_, int spaceDim, int numScalars, std::initializer_list<int> vectorVDims)
   : ParticleSet(comm_, spaceDim, numScalars, Array<int>(vectorVDims)) {};

   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<unsigned int> &rank_list);

   MPI_Comm GetComm() const { return comm; };
   
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   int GetSpaceDim() const { return SpaceDim; }
   int GetNumScalars() const { return NumScalars; }
   int GetNumVectors() const { return VectorVDims.Size(); }
   int GetVDim(int v) const { return VectorVDims[v]; }
   const Array<int> GetVectorVDims() const { return VectorVDims; }

   /// Reserve room for \p res particles. Can help to avoid re-allocation for adding + removing particles.
   void Reserve(int res) { data.reserve(res*TotalComps); ids.Reserve(res); }

   /// Get the number of particles currently held by this ParticleSet.
   int GetNP() const { return ids.Size(); }

   /// Add particle
   void AddParticle(const Particle &p)
   {
      AddParticle(p, id_counter);
      id_counter += id_stride;
   };

   /// Remove particle data specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get particle \p i referencing actual, data ONLY FOR Ordering byVDIM.
   template<Ordering::Type O = VOrdering, std::enable_if_t<O == Ordering::byVDIM, int> = 0>
   Particle GetParticleRef(int i);

   template<Ordering::Type O = VOrdering, std::enable_if_t<O == Ordering::byVDIM, int> = 0>
   const Particle GetParticleRef(int i) const { return const_cast<ParticleSet*>(this)->GetParticleRef(i); };

   /// Get copy of data in particle \p i .
   Particle GetParticleData(int i) const;

   const Array<unsigned int>& GetIDs() const { return ids; };

   Vector& GetSetCoords() { return fields[0]; }

   Vector& GetSetScalar(int s) { return fields[1+s]; }

   Vector& GetSetVector(int v) { return fields[1+NumScalars+v]; }

   const Vector& GetSetCoords() const { return fields[0]; }

   const Vector& GetSetScalar(int s) const { return fields[1+s]; }

   const Vector& GetSetVector(int v) const { return fields[1+NumScalars+v]; }

   /// Print in VisIt Point3D format (x y z ID)
   void PrintPoint3D(std::ostream &os);

   /// Print all data to CSV
   void PrintCSV(const char* fname, int precision=16);

   ~ParticleSet();

};

} // namespace mfem

#endif // MFEM_PARTICLESET