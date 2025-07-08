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
// Define ParticleMeta to hold particle metadata
// TODO: Make this a struct?

class ParticleMeta
{
private:
   const int spaceDim;
   const int numProps; // scalars intrinsic to particles
   const Array<int> stateVDims; // state variable (momentum, etc...) vdims

public:
   
   ParticleMeta(int spaceDim_, int numProps_, const Array<int> &stateVDims_)
   : spaceDim(spaceDim_), numProps(numProps_), stateVDims(stateVDims_) {}

   ParticleMeta(int spaceDim_, int numProps_, std::initializer_list<int> stateVDims_)
   : ParticleMeta(spaceDim_, numProps_, Array<int>(stateVDims_)) {}

   int SpaceDim() const { return spaceDim; }
   int NumProps() const { return numProps; }
   int NumStateVars() const { return stateVDims.Size(); }
   int StateVDim(int v) const { return stateVDims[v]; } // TODO: better name for this + below.. too similar
   const Array<int>& StateVDims() const { return stateVDims; }
};

// -----------------------------------------------------------------------------------------------------
// Define Particle class

class Particle
{
protected:
   std::reference_wrapper<const ParticleMeta> meta;

   Vector coords;
   std::vector<real_t> props;
   std::vector<Vector> state;

public:
   const ParticleMeta& GetMeta() const { return meta; }

   Particle(const ParticleMeta &pmeta);

   Vector& GetCoords() { return coords; }

   real_t& GetProperty(int s) { return props[s]; }

   Vector& GetStateVar(int v) { return state[v]; }

   const Vector& GetCoords() const { return coords; }
   
   const real_t& GetProperty(int s) const { return props[s]; }

   const Vector& GetStateVar(int v) const { return state[v]; }

   bool operator==(const Particle &rhs) const;

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   void Print(std::ostream &out=mfem::out) const;
};

// -----------------------------------------------------------------------------------------------------
// Define ParticleSet

class ParticleSet
{
protected:
   const Ordering::Type ordering;
   const ParticleMeta &meta;

   const int totalFields; // Total number of fields / particle
   const int totalComps; // Total comps / particle
   const Array<int> fieldVDims;
   const Array<int> exclScanFieldVDims;

   const Array<int> MakeFieldVDims()
   {
      Array<int> temp(totalFields);
      temp[0] = meta.SpaceDim();
      for (int s = 0; s < meta.NumProps(); s++) temp[1+s] = 1;
      for (int v = 0; v < meta.NumStateVars(); v++) temp[1+meta.NumProps()+v] = meta.StateVDim(v);
      return temp;
   }
   const Array<int> MakeExclScanFieldVDims()
   {
      Array<int> temp(totalFields);
      temp[0] = 0;
      for (int i = 1; i < totalFields; i++)
      {
         temp[i] = temp[i-1] + fieldVDims[i-1];
      }
      return temp;
   }

   const std::size_t id_stride;
   std::size_t id_counter;

   std::vector<real_t> data; /// Stores ALL actual particle data
   std::vector<Vector> fields; /// User-facing Vectors, always referencing actual data
   Array<unsigned int> ids; /// Particle IDs

   // TODO: consider below... (see member fxns too)
   //std::vector<Particle> particles; /// Optional internal AoS configuration of particles

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
      bool success = ( (totalComps == Ns ? (Transfer<Ns>(send_idxs, send_ranks),true) : false) || ...);
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

   Ordering::Type GetOrdering() const { return ordering; }

   const ParticleMeta& GetMeta() const { return meta; }
   /// Serial constructor
   ParticleSet(const ParticleMeta &meta_, Ordering::Type ordering_);

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

   /// Parallel constructor
   ParticleSet(MPI_Comm comm_, const ParticleMeta &meta_, Ordering::Type ordering_);

   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<unsigned int> &rank_list);

   MPI_Comm GetComm() const { return comm; };
   
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   /// Reserve room for \p res particles. Can help to avoid re-allocation for adding + removing particles.
   void Reserve(int res) { data.reserve(res*totalComps); ids.Reserve(res); }

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

   const Array<unsigned int>& GetIDs() const { return ids; };

   Vector& GetAllCoords() { return fields[0]; }

   Vector& GetAllProperty(int s) { return fields[1+s]; }

   Vector& GetAllStateVar(int v) { return fields[1+meta.NumProps()+v]; }

   const Vector& GetAllCoords() const { return fields[0]; }

   const Vector& GetAllProperty(int s) const { return fields[1+s]; }

   const Vector& GetAllStateVar(int v) const { return fields[1+meta.NumProps()+v]; }

   /// Copy data associated with particle at index \p i into \p p
   void GetParticle(int i, Particle &p) const;

   /// Set data for particle at index \p i with data from provided particle \p p
   void SetParticle(int i, const Particle &p);


/* TODO: consider below. Maybe helpful??? Maybe not???

   /// Set (optional) internal particle vector from actual set \ref data
   void SetParticleArray();

   /// Get reference to particle from (optional) internal particle vector
   Particle& GetParticleFromArray(int i) { return particles[i]; }

   /// Set particle set data from (optional) internal particle vector
   void SetFromParticleArray();
*/

   /// Print in VisIt Point3D format (x y z ID)
   void PrintPoint3D(std::ostream &os);

   /// Print all data to CSV
   void PrintCSV(const char* fname, int precision=16);

   ~ParticleSet();

};

} // namespace mfem

#endif // MFEM_PARTICLESET