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

/// Container for data associated with a single particle. See \ref ParticleSet for more information.
class Particle
{
protected:
   Vector coords;
   std::vector<Vector> fields;
   std::vector<Memory<int>> tags;
public:
   
   /** @brief Construct a Particle instance
    *  @param[in] dim          Spatial dimension.
    *  @param[in] field_vdims  Vector dimensions of particle fields.
    *  @param[in] num_tags     Number of integer tags.
    */
   Particle(int dim, const Array<int> &field_vdims, int num_tags);

   /// Get the spatial dimension of this particle.
   int Dim() const { return coords.Size(); }

   /// Get the number of fields associated with this particle.
   int NumFields() const { return fields.size(); }

   /// Get the vector dimension of field \p f .
   int FieldVDim(int f) const { return fields[f].Size(); }

   /// Get the number of tags associated with this particle.
   int NumTags() const { return tags.size(); }

   /// Get reference to particle coordinates Vector.
   Vector& Coords() { return coords; }

   /// Get const reference to particle coordinates Vector.
   const Vector& Coords() const { return coords; }

   /// Get reference to field \p f , component \p c value.
   real_t& FieldValue(int f, int c=0) { return fields[f][c]; }

   /// Get const reference to field \p f , component \p c value.
   const real_t& FieldValue(int f, int c=0) const { return fields[f][c]; }

   /// Get reference to field \p f Vector.
   Vector& Field(int f) { return fields[f]; }

   /// Get const reference to field \p f Vector.
   const Vector& Field(int f) const { return fields[f]; }

   /// Get reference to tag \p t .
   int& Tag(int t) { return tags[t][0]; }

   /// Get const reference to tag \p t .
   const int& Tag(int t) const { return tags[t][0]; }

   /// Get reference to underlying memory object of tag \p t .
   Memory<int>& TagMemory(int t) { return tags[t]; }

   /// Particle equality operator.
   bool operator==(const Particle &rhs) const;

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   /// Print all particle data to \p out .
   void Print(std::ostream &out=mfem::out) const;

};

/** @brief ParticleSet initializes and manages data associated with particles.
 * 
 *  @details Particles are inherently initialized to have a position and an ID, and optionally can have any number of Vector (of arbitrary vdim) and scalar integer data associated with in the form of @b fields and @b tags respectively. All particle data are internally stored as a Struct-of-Arrays, as elaborated on below.
 * 
 *  @par Coordinates:
 *  %All particle coordinates are stored in a \ref MultiVector with vdim = spatial dimension, ordered either byNODES or byVDIM.
 * 
 *  @par IDs:
 *  %Each particle is assigned a unique global ID of type unsigned int. In parallel, IDs are initialized starting with @b rank and striding by @b size.
 * 
 *  @par Fields:
 *  %Fields represent scalar or Vector \ref real_t data to be associated with each particles, such as mass, momentum, or moment. For a given field, all particle data are stored in a single \ref MultiVector with a given vector dimension (1 for scalar data) and \ref Ordering::Type (byNODES or byVDIM) to specify how the data is organized in memory.
 * 
 *   @par Tags:
 *   %Tags represent an integer to be associated with each particle. For a given tag, all particle integers are stored in a single \ref Array<int>.
 * 
 */
class ParticleSet
{
private:
   static Array<Ordering::Type> GetOrderingArray(Ordering::Type o, int N);
   static std::string GetDefaultFieldName(int i);
   static std::string GetDefaultTagName(int i);
   static Array<const char*> GetEmptyNameArray(int N);
#ifdef MFEM_USE_MPI
   static unsigned int GetRank(MPI_Comm comm_);
   static unsigned int GetSize(MPI_Comm comm_);
#endif // MFEM_USE_MPI

protected:

   /// Array of particle IDs owned.
   Array<unsigned int> ids;

   /// All particle coordinates.
   MultiVector coords;
   
   /// All particle fields.
   std::vector<std::unique_ptr<MultiVector>> fields;

   /// All particle tags.
   std::vector<std::unique_ptr<Array<int>>> tags;

   /// Stride for IDs (when new particles are added).
   const unsigned int id_stride;

   /// Current ID to be assigned to the next particle added.
   unsigned int id_counter;

   /// Field names, to be written when \ref PrintCSV is called.
   std::vector<std::string> field_names;

   /// Tag names, to be written when \ref PrintCSV is called.
   std::vector<std::string> tag_names;

   /// Add size of \p new_ids particles, with ids \p new_ids and optionally get the indices of new particles in \p new_indices .
   void AddParticles(const Array<unsigned int> &new_ids, Array<int> *new_indices=nullptr);


#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI


#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   std::unique_ptr<gslib::comm> gsl_comm;
   std::unique_ptr<gslib::crystal> cr;

   /** @brief Buffer for individual particle data used in \ref Transfer.
    *  @tparam NData       Number of \ref real_t values for a particle.
    *  @tparam NTag        Number of tags (integer data) for a particle.
    */
   template<std::size_t NData, std::size_t NTag>
   struct pdata_t
   {
      std::array<double, NData> data; // coords + fields
      std::array<int, NTag> tags; // tags
      unsigned int id;
   };

   /// Maximum number of real_t values for a \ref pdata_t compiled, used in \ref Transfer.
   static constexpr int NDATA_MAX = 50;

   /// Maximum number of integer values for a \ref pdata_t compiled, used in \ref Transfer.
   static constexpr int NTAG_MAX = 3;


   /** @brief Transfer particle data with \p NData \ref real_t values and \p NTag integer values.
    * 
    *  @tparam NData       Number of \ref real_t values for a particle.
    *  @tparam NTag        Number of tags (integer data) for a particle.
    * 
    *  @param[in] send_idxs   Array of particles indices to send.
    *  @param[in] send_ranks  Array of ranks to send particles at \p send_idxs to.
    */
   template<std::size_t NData, std::size_t NTag>
   void Transfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks);

   template<std::size_t NData, std::size_t... NTags>
   void DispatchTagTransfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks, std::index_sequence<NTags...>)
   {
      bool success = ( (tags.size() == NTags ? (Transfer<NData, NTags>(send_idxs, send_ranks),true) : false) || ...);
      MFEM_ASSERT(success, "Redistributing with > " << NTAG_MAX << " tags is not supported. Please submit PR to request particular case with more.");
   }
   
   template<std::size_t... NDatas>
   void DispatchDataTransfer(const Array<unsigned int> &send_idxs, const Array<unsigned int> &send_ranks, std::index_sequence<NDatas...>)
   {
      int total_comps = coords.GetVDim();
      for (std::unique_ptr<MultiVector> &pv : fields)
      {
         total_comps += pv->GetVDim();
      }
      bool success = ( (total_comps == NDatas ? (DispatchTagTransfer<NDatas>(send_idxs, send_ranks, std::make_index_sequence<NTAG_MAX+1>{}),true) : false) || ...);
      MFEM_ASSERT(success, "Redistributing with > " << NDATA_MAX << " real_t components per particle is not supported. Please submit PR to request particular case with more.");
   }

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   /// Create a \ref Particle object with the same spatial dimension, number of fields and field vdims, and number of tags as this ParticleSet.
   Particle CreateParticle() const;

   /// Write string in \p ss_header , followed by \p ss_data , to a single file; compatible in parallel.
   void WriteToFile(const char *fname, const std::stringstream &ss_header, const std::stringstream &ss_data);

   bool IsValidParticle(const Particle &p) const;

   /** @brief Hidden main constructor of ParticleSet 
    * 
    *  @param[in] id_stride_          ID stride.
    *  @param[in] id_counter_         Starting ID counter.
    *  @param[in] num_particles       Number of particles to initialize at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates \ref MultiVector.
    *  @param[in] field_vdims         Array of field vector dimensions to register.
    *  @param[in] field_orderings     Array of field ordering types.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names. 
    */
   ParticleSet(int id_stride_, int id_counter_, int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_);

public:

   /** @brief Construct a serial ParticleSet.
    * 
    *  @param[in] num_particles       Number of particles to initialize at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates \ref MultiVector.
    */
   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering=Ordering::byVDIM);

   /** @brief Construct a serial ParticleSet with specified fields and tags at construction. More may be added later.
    * 
    *  @param[in] num_particles       Number of particles to initialize at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimensions to register.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] all_ordering        Ordering of coordinates and field \ref MultiVector s.
    */
   ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, int num_tags, Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Construct a serial ParticleSet with specified fields and tags at construction, with names ( \ref PrintCSV ). More may be added later.
    * 
    *  @param[in] num_particles       Number of particles to initialize at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimensions to register.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names. 
    *  @param[in] all_ordering        Ordering of coordinates and field \ref MultiVector s.
    */
   ParticleSet(int num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_, Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Comprehensive serial constructor of ParticleSet.
    * 
    *  @param[in] num_particles       Number of particles to initialize at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates \ref MultiVector.
    *  @param[in] field_vdims         Array of field vector dimensions to register.
    *  @param[in] field_orderings     Array of field ordering types.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names. 
    */
   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names_);

#ifdef MFEM_USE_MPI

   /** @brief Construct a parallel ParticleSet.
    * 
    *  @param[in] comm_               MPI communicator.
    *  @param[in] num_particles       Number of particles to initialize at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates \ref MultiVector.
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, Ordering::Type coords_ordering=Ordering::byVDIM);

   /** @brief Construct a parallel ParticleSet with specified fields and tags at construction. More may be added later.
    * 
    *  @param[in] comm_               MPI communicator.
    *  @param[in] rank_num_particles  Number of particles to initialize on this rank at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimensions to register.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] all_ordering        Ordering of coordinates and field \ref MultiVector s.
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, const Array<int> &field_vdims, int num_tags, Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Construct a parallel ParticleSet with specified fields and tags at construction, with names ( \ref PrintCSV ). More may be added later.
    * 
    *  @param[in] comm_               MPI communicator.
    *  @param[in] rank_num_particles  Number of particles to initialize on this rank at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimensions to register.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names. 
    *  @param[in] all_ordering        Ordering of coordinates and field \ref MultiVector s.
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, const Array<int> &field_vdims, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names, Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Comprehensive parallel constructor of ParticleSet.
    * 
    *  @param[in] comm_               MPI communicator.
    *  @param[in] rank_num_particles  Number of particles to initialize on this rank at construction.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates \ref MultiVector.
    *  @param[in] field_vdims         Array of field vector dimensions to register.
    *  @param[in] field_orderings     Array of field ordering types.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names. 
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim, Ordering::Type coords_ordering, const Array<int> &field_vdims, const Array<Ordering::Type> &field_orderings, const Array<const char*> &field_names_, int num_tags, const Array<const char*> &tag_names);

   /// Get the MPI communicator for this ParticleSet.
   MPI_Comm GetComm() const { return comm; };

   /// Get the global number of active particles across all ranks.
   unsigned int GetGlobalNP() const;

#endif // MFEM_USE_MPI

   /// Get the spatial dimension.
   int GetDim() const { return coords.GetVDim(); }

   /// Get the IDs of the active particles owned by this ParticleSet.
   const Array<unsigned int>& GetIDs() const { return ids; }

   /** @brief Add a field to the ParticleSet.
    * 
    *  @arg[in] vdim             Vector dimension of the field.
    *  @arg[in] field_ordering   Ordering::Type of the field.
    *  @arg[in] field_name       Name of the field.
    */
   MultiVector& AddField(int vdim, Ordering::Type field_ordering=Ordering::byVDIM, const char* field_name=nullptr);

   /** @brief Add a tag to the ParticleSet.
    * 
    *  @arg[in] field_name       Name of the tag.
    */
   Array<int>& AddTag(const char* tag_name=nullptr);

   /// Reserve room for \p res particles. Can help to avoid re-allocation for adding + removing particles.
   void Reserve(int res);

   /// Get the number of active particles currently held by this ParticleSet.
   int GetNP() const { return ids.Size(); }

   /// Get the number of fields registered to particles.
   int GetNF() const { return fields.size(); }

   /// Get the number of tags registered to particles
   int GetNT() const { return tags.size(); }

   /// Add a particle using \ref Particle .
   void AddParticle(const Particle &p);

   /// Add \p num_particles particles, and optionally get the indices of new particles in \p new_indices .
   void AddParticles(int num_particles, Array<int> *new_indices=nullptr);

   /// Remove particle data specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get a reference to the coordinates MultiVector.
   MultiVector& Coords() { return coords; }

   /// Get a const reference to the coordinates MultiVector.
   const MultiVector& Coords() const { return coords; }

   /// Get a reference to field \p f 's MultiVector.
   MultiVector& Field(int f) { return *fields[f]; }

   /// Get a const reference to field \p f 's MultiVector.
   const MultiVector& Field(int f) const { return *fields[f]; }

   /// Get a reference to tag \p t 's Array<int>.
   Array<int>& Tag(int t) { return *tags[t]; }

   /// Get a const reference to tag \p t 's Array<int>.
   const Array<int>& Tag(int t) const { return *tags[t]; }

   /// Get new \ref Particle object with copy of data associated with particle \p i .
   Particle GetParticle(int i) const;

   /** @brief Determine if \ref GetParticleRef is valid.
    * 
    * If coordinates and all fields are ordered byVDIM, then returns true. Otherwise, false.
    */
   bool ParticleRefValid() const;

   /** @brief Get Particle object whose member reference the actual data associated with particle \p i .
    * 
    * @see ParticleRefValid 
    */
   Particle GetParticleRef(int i);

   /// Set data for particle at index \p i with data from provided particle \p p
   void SetParticle(int i, const Particle &p);

   /// Print all particle data to a CSV file.
   void PrintCSV(const char *fname, int precision=16);

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

   /** @brief Redistribute particle data to \p rank_list
    * 
    *  @arg[in] rank_list    Array of size \ref GetNP denoting ultimate destination of particle data. Index = this rank means no data is moved.
    */
   void Redistribute(const Array<unsigned int> &rank_list);
   
#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   /** @brief Destructor
    * 
    * Destructor must be declared after inclusion of GSLIB header (so sizeof gslib::comm and gslib::crystal can be evaluated)
    */
   ~ParticleSet();

};

} // namespace mfem

#endif // MFEM_PARTICLESET
