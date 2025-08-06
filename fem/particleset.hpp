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
#include <variant>

namespace mfem
{

// -----------------------------------------------------------------------------------------------------
// Define Particle class

class ParticleSet;


class Particle
{
   // Required only for setting tag Memory<int>::MakeAlias in GetParticleRef.
   // Can alternatively expose Memory<int>& TagMemory(int t) ....
   friend class ParticleSet;

protected:
   Vector coords;
   std::vector<Vector> fields;
   std::vector<Memory<int>> tags;
public:
   
   Particle(int dim, const Array<int> &field_vdims, int num_tags);

   int Dim() const { return coords.Size(); }

   int NumFields() const { return fields.size(); }

   int FieldVDim(int f) const { return fields[f].Size(); }

   int NumTags() const { return tags.size(); }

   Vector& Coords() { return coords; }

   const Vector& Coords() const { return coords; }

   real_t& FieldValue(int f, int c=0) { return fields[f][c]; }

   const real_t& FieldValue(int f, int c=0) const { return fields[f][c]; }

   Vector& Field(int f) { return fields[f]; }

   const Vector& Field(int f) const { return fields[f]; }

   int& Tag(int t) { return tags[t][0]; }

   const int& Tag(int t) const { return tags[t][0]; }

   bool operator==(const Particle &rhs) const;

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   void Print(std::ostream &out=mfem::out) const;

};

// -----------------------------------------------------------------------------------------------------
// Define ParticleSet

class ParticleSet
{
private:
   static Array<Ordering::Type> GetOrderingArray(Ordering::Type o, int N);
   static std::string GetDefaultFieldName(int i);
   static std::string GetDefaultTagName(int i);
   static Array<const char*> GetEmptyNameArray(int N);
   static Array<int> LDof2VDofs(int ndofs, int vdim, const Array<int> &ldofs, Ordering::Type o);
#ifdef MFEM_USE_MPI
   static unsigned int GetRank(MPI_Comm comm_);
   static unsigned int GetSize(MPI_Comm comm_);
#endif // MFEM_USE_MPI

protected:

   struct ParticleState
   {
      Array<unsigned int> ids; /// Particle IDs
      NodeFunction coords;
      std::vector<std::unique_ptr<NodeFunction>> fields;
      std::vector<std::unique_ptr<Array<int>>> tags;

      ParticleState(int dim, Ordering::Type coords_ordering)
      : coords(dim, coords_ordering) {}
      
      int GetNP() const { return ids.Size(); }
      int GetNF() const { return fields.size(); }
      int GetNT() const {return tags.size(); }
   };

   /// Increase capacity of data in \p particles w/o losing existing data
   static void ReserveParticles(int res, ParticleState &particles);

   /// Add size of \p new_ids particles to \p particles , and get indices of new particles in \p new_indices
   static void AddParticles(const Array<unsigned int> &new_ids, ParticleState &particles, Array<int> *new_indices=nullptr);

   /// Remove particles with indices \p list in \p particles
   static void RemoveParticles(const Array<int> &list, ParticleState &particles);


   const unsigned int id_stride;
   unsigned int id_counter;
   ParticleState active_state;
   ParticleState inactive_state;
   std::vector<std::string> field_names;
   std::vector<std::string> tag_names;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI


#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   std::unique_ptr<gslib::comm> gsl_comm;
   std::unique_ptr<gslib::crystal> cr;

   // If no FindPointsGSLIB data:
   template<std::size_t NData, std::size_t NTag>
   struct pdata_t
   {
      std::array<double, NData> data; // coords + fields
      std::array<int, NTag> tags; // tags
      unsigned int id;
   };

   static constexpr int NDATA_MAX = 50;
   static constexpr int NTAG_MAX = 3;

   template<std::size_t NData, std::size_t NTag>
   void Transfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks);

   template<std::size_t NData, std::size_t... NTags>
   void DispatchTagTransfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks, std::index_sequence<NTags...>)
   {
      bool success = ( (active_state.GetNT() == NTags ? (Transfer<NData, NTags>(send_idxs, send_ranks),true) : false) || ...);
      MFEM_ASSERT(success, "Redistributing with > " << NTAG_MAX << " tags is not supported. Please submit PR to request particular case with more.");
   }
   
   template<std::size_t... NDatas>
   void DispatchDataTransfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks, std::index_sequence<NDatas...>)
   {
      int total_comps = active_state.coords.GetVDim();
      for (std::unique_ptr<NodeFunction> &pv : active_state.fields)
      {
         total_comps += pv->GetVDim();
      }
      bool success = ( (total_comps == NDatas ? (DispatchTagTransfer<NDatas>(send_idxs, send_ranks, std::make_index_sequence<NTAG_MAX+1>{}),true) : false) || ...);
      MFEM_ASSERT(success, "Redistributing with > " << NDATA_MAX << " real_t components per particle is not supported. Please submit PR to request particular case with more.");
   }

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   Particle CreateParticle() const;

   void WriteToFile(const char *fname, const std::stringstream &ss_header, const std::stringstream &ss_data);

   ParticleSet(int id_stride_, int id_counter_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_);

public:

   // Serial constructors
   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering=Ordering::byVDIM);
   ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, int num_tags, Ordering::Type all_ordering=Ordering::byVDIM);
   ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names, Ordering::Type all_ordering=Ordering::byVDIM);
   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names);

#ifdef MFEM_USE_MPI

   // Parallel constructors
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, Ordering::Type coords_ordering=Ordering::byVDIM);
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, const Array<int> &field_vdims, int num_tags, Ordering::Type all_ordering=Ordering::byVDIM);
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names, Ordering::Type all_ordering=Ordering::byVDIM);
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names);

   MPI_Comm GetComm() const { return comm; };

   /// Get the global number of active particles
   unsigned int GetGlobalNP() const;

#endif // MFEM_USE_MPI

   int GetDim() const { return active_state.coords.GetVDim(); }

   const Array<unsigned int>& GetIDs() const { return active_state.ids; }

   NodeFunction& AddField(int vdim, Ordering::Type field_ordering=Ordering::byVDIM, const char* field_name=nullptr);

   Array<int>& AddTag(const char* tag_name=nullptr);

   /// Reserve room for \p res particles. Can help to avoid re-allocation for adding + removing particles.
   void Reserve(int res) { ReserveParticles(res, active_state); };

   /// Get the number of particles currently held by this ParticleSet.
   int GetNP() const { return active_state.GetNP(); }

   int GetNF() const { return active_state.GetNF(); }

   int GetNT() const { return active_state.GetNT(); }

   /// Add particle
   void AddParticle(const Particle &p);

   /// Remove particle data specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list, bool delete_particles=false);

   NodeFunction& Coords() { return active_state.coords; }

   const NodeFunction& Coords() const { return active_state.coords; }

   NodeFunction& Field(int f) { return *active_state.fields[f]; }

   const NodeFunction& Field(int f) const { return *active_state.fields[f]; }

   Array<int>& Tag(int t) { return *active_state.tags[t]; }

   const Array<int>& Tag(int t) const { return *active_state.tags[t]; }

   /// Get Particle with copy of data associated with particle \p i
   Particle GetParticle(int i) const;

   /// Returns true if coords and all fields are ordered byVDIM. False otherwise.
   bool ParticleRefValid() const;

   /// Get Particle that references data associated with particle \p i , only valid when \ref ParticleRefValid() is true 
   Particle GetParticleRef(int i);

   /// Set data for particle at index \p i with data from provided particle \p p
   void SetParticle(int i, const Particle &p);

   /// Print to CSV
   void PrintCSV(const char *fname, int precision=16);

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

   /// Redistribute particles onto ranks specified in \p rank_list .
   /// Optionally include array of FindPointsGSLIB objects to have their internal data transferred as well.
   void Redistribute(const Array<unsigned int> &rank_list);
   
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   // Destructor must be declared after inclusion of GSLIB header (so sizeof gslib::comm and gslib::crystal can be evaluated)
   ~ParticleSet();

};

} // namespace mfem

#endif // MFEM_PARTICLESET
