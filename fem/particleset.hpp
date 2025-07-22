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
#include "gslib.hpp"

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

#include <numeric>

namespace mfem
{


class ParticleVector : public Vector
{
protected:
   const int vdim;
   const Ordering::Type ordering;

public:
   ParticleVector(int np, int vdim_, Ordering::Type ordering_)
   : Vector(np*vdim_), vdim(vdim_), ordering(ordering_) {}
   
   int GetVDim() const { return vdim; }

   Ordering::Type GetOrdering() const { return ordering; }

   int GetNP() const { return Size()/vdim; }

   void GetParticleData(int i, Vector &pdata) const;
   
   void GetParticleRef(int i, Vector &pref);

   void SetParticleData(int i, const Vector &pdata);

   real_t& ParticleData(int i, int comp=0);

   const real_t& ParticleData(int i, int comp=0) const;
}


// -----------------------------------------------------------------------------------------------------
// Define Particle class

class Particle
{
protected:

   Vector coords;
   Array<std::unique_ptr<Vector>> data;

public:
   
   // Owning ctor
   Particle(int dim, const Array<int> &data_vdims);

   // Ref ctor
   Particle(Vector *coords_, Vector *data_[], int num_data);

   Vector& Coords() { return coords; }

   const Vector& Coords() const { return coords; }

   real_t& Data(int f, int c=0) { return (*data[f])[c]; }

   const real_t& Data(int f, int c=0) { return (*data[f])[c]; }

   Vector& VectorData(int f) { return *data[f]; }

   const Vector& VectorData(int f) const { return *data[f]; }

   bool operator==(const Particle &rhs) const;

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   void Print(std::ostream &out=mfem::out) const;
};

// -----------------------------------------------------------------------------------------------------
// Define ParticleSet

class ParticleSet
{
private:

   static Array<Ordering::Type> GetOrderingArray(Ordering::Type o, int N)
   {
      Array<Ordering::Type> ordering_arr(data_vdims.Size());
      ordering_arr = o; 
      return std::move(ordering_array); 
   }
   static Array<std::string> GetDataNameArray(int N)
   {
      Array<std::string> names(data_vdims.Size()); 
      for (int i = 0; i < N; i++)
      {
         names[i] = "Data_" + std::to_string(i);
      }
      return std::move(names)
   }

#ifdef MFEM_USE_MPI
   static int GetRank(MPI_Comm comm_)
   {
      int r; MPI_Comm_rank(comm_, &r); 
      return r;
   }
   static int GetSize(MPI_Comm comm_)
   {
      int s; MPI_Comm_rank(comm_, &s); 
      return s;
   }
   static int GetRankNumParticles(MPI_Comm comm_, int NP)
   {
      return NP/GetSize(comm_) + r < (r < (num_particles % size) ? 1 : 0);
   }
#endif // MFEM_USE_MPI


protected:
   const std::size_t id_stride;
   std::size_t id_counter;

   Array<unsigned int> ids; /// Particle IDs
   ParticleVector coords;
   Array<std::unique_ptr<ParticleVector>> data;
   Array<std::string> names;

   ParticleSet(int id_stride_, int id_counter_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &data_vdims, const Array<Ordering::Type> &data_orderings, const Array<std::string> &data_names);

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
   ParticleSet(MPI_Comm comm_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &data_vdims, const Array<Ordering::Type> &data_orderings, const Array<std::string> &data_names);
#endif // MFEM_USE_MPI


#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   std::unique_ptr<gslib::comm> gsl_comm;
   std::unique_ptr<gslib::crystal> cr;
   template<std::size_t N>
   struct pdata_t
   {
      double data[N];
      unsigned int id, proc;
   };

   template<std::size_t N>
   struct pdata_fdpts_t
   {
      double data[N];
      double rst[3], mfem_rst[3]; // gslib ref coords , mfem reference coords
      unsigned int elem, mfem_elem, code; // gslib elem id, mfem elem id, and code
      unsigned int id, proc;
   };

   static constexpr std::size_t N_MAX = 100;

   template<std::size_t N>
   void Transfer(const Array<int> &send_idxs, const Array<int> &send_ranks, FindPointsGSLIB *finder);

   template<std::size_t... Ns>
   void RuntimeDispatchTransfer(const Array<int> &send_idxs, const Array<int> &send_ranks, std::index_sequence<Ns...>, FindPointsGSLIB *finder)
   {
      bool success = ( (totalComps == Ns ? (Transfer<Ns>(send_idxs, send_ranks, finder),true) : false) || ...);
      MFEM_ASSERT(success, "Particles with total components above 100 are currently not supported for redistributing. Please submit a PR to request a particular particle size above this.");
   }

   void Redistribute(const Array<unsigned int> &rank_list, FindPointsGSLIB *finder);

   
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB


   /// Add particle w/ given ID
   void AddParticle(const Particle &p, int id);

   void PrintHeader(std::ostream &os, bool inc_rank, const char *delimiter);
   void PrintData(std::ostream &os, bool inc_header, const char *delimiter, int *rank=nullptr);


   void WriteToFile(const char* fname, const std::stringstream &ss_header, const std::stringstream &ss_data);

public:

   // Serial constructors

   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering)
   : ParticleSet(1, 0, num_particles, dim, coords_ordering, Array<int>(), Array<Ordering::Type>(), Array<std::string>()) {}
   
   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &data_vdims, Ordering::Type data_ordering)
   : ParticleSet(1, 0, num_particles, dim, coords_ordering, data_vdims, GetOrderingArray(data_ordering, data_vdims.Size()), GetDataNameArray(data_vdims.Size())) {}

   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &dataVDims, const Array<Ordering::Type> &data_orderings, const Array<std::string> &data_names)
   : ParticleSet(1, 0, num_particles, dim, coords_ordering, &data_vdims, &data_orderings, &data_names) {};

#ifdef MFEM_USE_MPI

   // Parallel constructors
   
   ParticleSet(MPI_Comm comm_, int num_particles, int dim, Ordering::Type default_ordering);
   
   ParticleSet(MPI_Comm comm_, int dim, Ordering::Type default_ordering, const Array<int> &dataVDims);

   ParticleSet(MPI_Comm comm_, int dim, Ordering::Type coords_ordering, const Array<int> &dataVDims, const Array<Ordering::Type> &data_orderings, const Array<std::string> &data_names)

#endif // MFEM_USE_MPI

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<unsigned int> &rank_list)
   { Redistribute(rank_list, nullptr); }

   /// Redistribute points AND FindPointsGSLIB internal data in single comms.
   void Redistribute(FindPointsGSLIB &finder, bool findpts=false)
   { 
      if (findpts)
      {
         finder.FindPoints(GetAllCoords(), GetOrdering());
      }
      
      Redistribute(finder.GetProc(), &finder);
   }

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

   /// Print in VisIt Point3D format
   void PrintPoint3D(const char* fname, int precision=16);

   /// Print to CSV
   void PrintCSV(const char* fname, int precision=16);

   ~ParticleSet();

};

} // namespace mfem

#endif // MFEM_PARTICLESET