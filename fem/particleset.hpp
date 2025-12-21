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
#include "gslib.hpp"
#include "kernel_dispatch.hpp"

namespace mfem
{

/** @brief  Container for data associated with a single particle.
 *
 *  @note This class mainly serves as a convenience interface to individual
 *  particle data from ParticleSet. We recommend seeing ParticleSet first.
 *
 *  @details As described in ParticleSet documentation, each particle has a
 *  position (\ref coords), arbitrary number of scalar or vector \ref real_t
 *  data (\ref fields), and arbitrary number of integers (\ref tags)
 *  associated with it.
 *
 *  \ref fields can thus hold data such as mass, momentum, and velocity, while
 *  \ref tags can hold integer data such as particle type, color, etc.
 *
 *  Each particle also has a unique global ID, but that is managed by the
 *  ParticleSet class and not stored in this Particle class. Simiarly, the names
 *  of the fields and tags, typically useful for output purposes, are managed by
 *  the ParticleSet class.
 *
 *
 * For clarity, we will use the particles below to illustrate the data layout
 * for \ref coords, \ref fields, and \ref tags
 *
 * @anchor sample_particle_data
 * @code
 * Particle_0: coords = (x0, y0),
 *             fields = {'mass'=m0, 'vel' = (vx0, vy0)},
 *               tags = {'type'=t0, 'color'=color0}
 * Particle_1: coords = (x1, y1),
 *             fields = {'mass'=m1, 'vel' = (vx1, vy1)},
 *               tags = {'type'=t1, 'color'=color1}
 * Particle_2: coords = (x2, y2),
 *             fields = {'mass'=m2, 'vel' = (vx2, vy2)},
 *               tags = {'type'=t2, 'color'=color2}
 * @endcode
 *
 */
class Particle
{
protected:
   /** @brief Spatial coordinates
    *
    *  @details For the \ref sample_particle_data, \ref coords would hold
    *  (x_i, y_i) for each particle i.
    */
   Vector coords;

   /** @brief A std::vector of Vector where each Vector holds data for a given
    *  field (e.g., mass, momentum or velocity) associated with the particle.
    *
    *  @details For the \ref sample_particle_data, \ref fields would be
    *  fields[0]=(m_i), fields[1]=(vx_i,vy_i) for each particle i.
    */
   std::vector<Vector> fields;

   /** @brief A std::vector of Array<int> where each Array<int> holds data
    *  for a given tag.
    *
    *  @details For the \ref sample_particle_data, \ref tags would be
    *  tags[0]=(type_i), tags[1]=(color_i) for each particle i. \n
    *
    *  @note An Array of length 1 is used for EACH tag, strictly for
    *  its owning/non-owning semantics (see Array<T>::MakeRef).
    */
   std::vector<Array<int>> tags;
public:
   /** @brief Construct a Particle instance.
    *  @param[in] dim          Spatial dimension (size of #coords).
    *  @param[in] field_vdims  Vector dimensions of particle fields.
    *  @param[in] num_tags     Number of integer tags.
    */
   Particle(int dim, const Array<int> &field_vdims, int num_tags);

   // Force default constructors and destructor
   Particle(const Particle&) = default;
   Particle& operator=(const Particle&) = default;
   Particle(Particle&&) = default;
   Particle& operator=(Particle&&) = default;
   ~Particle() = default;

   /// Get the spatial dimension of this particle.
   int GetDim() const { return coords.Size(); }

   /// Get the number of fields associated with this particle.
   int GetNFields() const { return fields.size(); }

   /// Get the vector dimension of field \p f .
   int GetFieldVDim(int f) const { return fields[f].Size(); }

   /// Get the number of tags associated with this particle.
   int GetNTags() const { return tags.size(); }

   /// Get reference to particle coordinates Vector.
   Vector& Coords() { return coords; }

   /// Get const reference to particle coordinates Vector.
   const Vector& Coords() const { return coords; }

   /// Get reference to field \p f , component \p c value.
   real_t& FieldValue(int f, int c=0)
   {
      MFEM_ASSERT(f >= 0 && static_cast<std::size_t>(f) < fields.size(),
                  "Invalid field index");
      MFEM_ASSERT(c >= 0 && c < fields[f].Size(),
                  "Invalid component index");
      return fields[f][c];
   }

   /// Get const reference to field \p f , component \p c value.
   const real_t& FieldValue(int f, int c=0) const
   {
      MFEM_ASSERT(f >= 0 && static_cast<std::size_t>(f) <  fields.size(),
                  "invalid field index");
      MFEM_ASSERT(c >= 0 && c < fields[f].Size(),
                  "invalid component index");
      return fields[f][c];
   }

   /// Get reference to field \p f Vector.
   Vector& Field(int f)
   {
      MFEM_ASSERT(f >= 0 && static_cast<std::size_t>(f) <  fields.size(),
                  "invalid field index");
      return fields[f];
   }

   /// Get const reference to field \p f Vector.
   const Vector& Field(int f) const
   {
      MFEM_ASSERT(f >= 0 && static_cast<std::size_t>(f) <  fields.size(),
                  "invalid field index");
      return fields[f];
   }

   /// Get reference to tag \p t .
   int& Tag(int t)
   {
      MFEM_ASSERT(t >= 0 && static_cast<std::size_t>(t) <  tags.size(),
                  "invalid tag index");
      return tags[t][0];
   }

   /// Get const reference to tag \p t .
   const int& Tag(int t) const
   {
      MFEM_ASSERT(t >= 0 && static_cast<std::size_t>(t) <  tags.size(),
                  "invalid tag index");
      return tags[t][0];
   }

   /// Set tag \p t to reference external data.
   void SetTagRef(int t, int *tag_data);

   /// Set field \p f to reference external data.
   void SetFieldRef(int f, real_t *field_data);

   /// Particle equality operator.
   bool operator==(const Particle &rhs) const;

   /// Particle inequality operator.
   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   /// Print all particle data to \p os.
   void Print(std::ostream &os=mfem::out) const;
};

/** @brief ParticleSet initializes and manages data associated with particles.
 *
 *  @details Particles are inherently initialized to have a position and an ID,
 *  and optionally can have any number of Vector (of arbitrary vdim) and scalar
 *  integer data in the form of @b fields and @b tags respectively. All particle
 *  data are internally stored in a Struct-of-Arrays fashion, as elaborated on
 *  below.
 *
 *  @par Coordinates:
 *  All particle coordinates are stored in a ParticleVector with vector
 *  dimension equal to the spatial dimension, ordered either byNODES or byVDIM.
 *  The ParticleVector \ref coords contains the coordinates of all particles.
 *
 *  @par IDs:
 *  Each particle is assigned a unique global ID of type IDType. In parallel,
 *  IDs are initialized starting with @b rank and striding by @b size. The IDs
 *  of all particles owned by this rank are stored in \ref ids.
 *
 *  @par Fields:
 *  Fields represent scalar or vector \ref real_t data to be associated with
 *  each particles, such as mass, momentum, or moment. For a given field, all
 *  particle data is stored in a single ParticleVector with a given
 *  vector dimension (1 for scalar data) and Ordering::Type (byNODES or
 *  byVDIM). The unique_ptrs to all the ParticleVectors are stored in the
 *  std::vector \ref fields.
 *
 *  @par Tags:
 *  Tags represent integers associated with each particle. For a given tag,
 *  all particle data are stored in a single Array<int>. The unique_ptrs to all
 *  the Array<int> is stored in the std::vector \ref tags.
 *
 *  @par Names:
 *  Each field and tag can optionally be given a name (string) to be used when
 *  printing particle data in CSV format using PrintCSV(). The names of all
 *  fields and tags are stored in the std::vectors \ref field_names and
 *  \ref tag_names, respectively.
 *
 *  @note We assume that all particles in a ParticleSet have the same number
 *  of fields and tags.
 *
 *  Following the example in the Particle class, we will use the
 *  particles below to illustrate the data layout for \ref coords, \ref ids,
 *  \ref fields, \ref tags, \ref field_names, and \ref tag_names.
 *  In each case, the name of the field and tag is enclosed in '...' for
 *  clarity. Additionally, we assume for this example that the particle
 *  coordinates and the 'vel' field are ordered byVDIM in their respective
 *  ParticleVector.
 *  @anchor sample_particleset_data
 *  @code
 *  Particle_0: id = id0, coords = (x0, y0),
 *              fields = {'mass'=m0, 'vel' = (vx0, vy0)},
 *              tags = {'type'=t0, 'color'=c0}
 *  Particle_1: id = id1, coords = (x1, y1),
 *              fields = {'mass'=m1, 'vel' = (vx1, vy1)},
 *              tags = {'type'=t1, 'color'=c1}
 *  Particle_2: id = id2, coords = (x2, y2),
 *              fields = {'mass'=m2, 'vel' = (vx2, vy2)},
 *              tags = {'type'=t2, 'color'=c2}
 *  @endcode
 */
class ParticleSet
{
public:
   using IDType = unsigned long long;
private:
   /// Constructs an Array of size N filled with Ordering::Type o.
   static Array<Ordering::Type> GetOrderingArray(Ordering::Type o, int N);

   /// Returns default field name for field index i. "Field_{i}"
   static std::string GetDefaultFieldName(int i);

   /// Returns default tag name for tag index i. "Tag_{i}"
   static std::string GetDefaultTagName(int i);

   /// Constructs an Array of size N filled with nullptr.
   static Array<const char*> GetEmptyNameArray(int N);

#ifdef MFEM_USE_MPI
   static int GetRank(MPI_Comm comm_);
   static int GetSize(MPI_Comm comm_);
#endif // MFEM_USE_MPI

protected:
   /// Stride for IDs (used internally when new particles are added).
   /** In parallel, this defaults to the number of MPI ranks. */
   const int id_stride;

   /// Current globally unique ID to be assigned to the next particle added.
   /** In parallel, this starts locally as the rank and increments with
    * id_stride, ensuring a global unique identifier whenever a particle is
    * added.
    */
   IDType id_counter;

   /** @brief Global unique IDs of particles owned by this rank.
    *
    *  @details For the \ref sample_particleset_data, \ref ids would be
    *  ids[0]=id0, ids[1]=id1, ids[2]=id2.
    */
   Array<IDType> ids;

   /** @brief Spatial coordinates of particles owned by this rank.
    *
    *  @details For the \ref sample_particleset_data, \ref coords would be
    *  coords=(x0,y0,x1,y1,x2,y2) assuming coords ordering is byVDIM.
    */
   ParticleVector coords;

   /** @brief All particle fields for particles owned by this rank.
    *
    *  @details For the \ref sample_particleset_data, \ref fields would be
    *  *fields[0]=(m0,m1,m2), *fields[1]=(vx0,vy0,vx1,vy1,vx2,vy2)
    *  assuming fields[1] ordering is byVDIM.
    */
   std::vector<std::unique_ptr<ParticleVector>> fields;

   /** @brief All particle tags for particles owned by this rank.
    *
    *  @details For the \ref sample_particleset_data, \ref tags would be
    *  *tags[0]=(t0,t1,t2), *tags[1]=(c0,c1,c2).
    */
   std::vector<std::unique_ptr<Array<int>>> tags;

   /** @brief Field names, to be written when PrintCSV() is called.
    *
    *  @details For the \ref sample_particleset_data, \ref field_names would be
    *  field_names[0]='mass', field_names[1]='vel'.
    */
   std::vector<std::string> field_names;

   /** @brief Tag names, to be written when PrintCSV() is called.
    *
    *  @details For the \ref sample_particleset_data, \ref tag_names would be
    *  tag_names[0]='type', tag_names[1]='color'.
    */
   std::vector<std::string> tag_names;

   /** @brief Add particles with global identifiers \p new_ids and
    *  optionally get the local indices of new particles in \p new_indices .
    *
    *  @details Note the data of new particles is uninitialized and must be
    *  set.
    */
   void AddParticles(const Array<IDType> &new_ids,
                     Array<int> *new_indices=nullptr);

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
   struct gslib::crystal *cr = nullptr;               // gslib's internal data
   struct gslib::comm *gsl_comm = nullptr;            // gslib's internal data

   /// \cond DO_NOT_DOCUMENT
   template<std::size_t NBytes>
   static void TransferParticlesImpl(ParticleSet &pset,
                                     const Array<int> &send_idxs,
                                     const Array<unsigned int> &send_ranks);

   using TransferParticlesType = void (*)(ParticleSet &pset,
                                          const Array<int> &send_idxs,
                                          const Array<unsigned int> &send_ranks);

   // Specialization parameter: NBytes
   MFEM_REGISTER_KERNELS(TransferParticles, TransferParticlesType, (size_t));
   friend TransferParticles;
   struct Kernels
   {
      Kernels();
   };
   /// \endcond

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   /** @brief  Update global ID of a particle.
    *
    *  @details This method updates the global ID of the particle at given
    *  local index after Redistribute().
    *
    *  @note This method must be used very carefully as it updates global
    *  ID of a particle.
    */
   void UpdateID(int local_idx, IDType new_global_id)
   { ids[local_idx] = new_global_id; }

   /** @brief Create a Particle object with the same spatial dimension,
    *  number of fields and field vdims, and number of tags as this ParticleSet.
    */
   Particle CreateParticle() const;

   /** @brief Write string in \p ss_header , followed by \p ss_data , to a
    *  single file; compatible in parallel.
    */
   void WriteToFile(const char *fname, const std::stringstream &ss_header,
                    const std::stringstream &ss_data);

   /** @brief Check if a particle could belong in this ParticleSet by
    *  comparing field and tag dimension.
    */
   bool IsValidParticle(const Particle &p) const;

   /** @brief Hidden main constructor of ParticleSet
    *
    *  @param[in] id_stride_          ID stride.
    *  @param[in] id_counter_         Starting ID counter.
    *  @param[in] num_particles       Number of particles to initialize.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates
    *  @param[in] field_vdims         Array of field vector dimensions
    *  @param[in] field_orderings     Array of field ordering types.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names.
    */
   ParticleSet(int id_stride_, IDType id_counter_, int num_particles, int dim,
               Ordering::Type coords_ordering, const Array<int> &field_vdims,
               const Array<Ordering::Type> &field_orderings,
               const Array<const char*> &field_names_, int num_tags,
               const Array<const char*> &tag_names_);

public:

   /** @brief Construct a serial ParticleSet.
    *
    *  @param[in] num_particles       Number of particles to initialize.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates.
    */
   ParticleSet(int num_particles, int dim,
               Ordering::Type coords_ordering=Ordering::byVDIM);

   /** @brief Construct a serial ParticleSet with specified fields and tags at
    *  construction.
    *
    *  @param[in] num_particles       Number of particles to initialize.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimensions.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] all_ordering        (Optional) Ordering of coordinates and
    *                                 field ParticleVector.
    */
   ParticleSet(int num_particles, int dim, const Array<int> &field_vdims,
               int num_tags, Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Construct a serial ParticleSet with specified fields and tags at
    *  construction, with names.
    *
    *  @param[in] num_particles       Number of particles to initialize.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimensions.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names.
    *  @param[in] all_ordering        (Optional) Ordering of coordinates and
    *                                 field ParticleVector.
    */
   ParticleSet(int num_particles, int dim, const Array<int> &field_vdims,
               const Array<const char*> &field_names_, int num_tags,
               const Array<const char*> &tag_names_,
               Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Comprehensive serial constructor of ParticleSet.
    *
    *  @param[in] num_particles       Number of particles to initialize.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates.
    *  @param[in] field_vdims         Array of field vector dimensions.
    *  @param[in] field_orderings     Array of field ordering types.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names.
    */
   ParticleSet(int num_particles, int dim, Ordering::Type coords_ordering,
               const Array<int> &field_vdims,
               const Array<Ordering::Type> &field_orderings,
               const Array<const char*> &field_names_, int num_tags,
               const Array<const char*> &tag_names_);

#ifdef MFEM_USE_MPI
   /** @brief Construct a parallel ParticleSet.
    *
    *  @param[in] comm_               MPI communicator.
    *  @param[in] rank_num_particles  Number of particles to initialize.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     (Optional) Ordering of coordinates.
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
               Ordering::Type coords_ordering=Ordering::byVDIM);

   /** @brief Construct a parallel ParticleSet with specified fields and tags
    *  at construction.
    *
    *  @param[in] comm_               MPI communicator.
    *  @param[in] rank_num_particles  # of particles to initialize on this rank.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimensions.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] all_ordering        (Optional) Ordering of coordinates and
    *                                 field ParticleVector.
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
               const Array<int> &field_vdims, int num_tags,
               Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Construct a parallel ParticleSet with specified fields and tags
    *  at construction, with names (for PrintCSV()).
    *
    *  @param[in] comm_               MPI communicator.
    *  @param[in] rank_num_particles  # of particles to initialize on this rank.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] field_vdims         Array of field vector dimension.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names.
    *  @param[in] all_ordering        (Optional) Ordering of coordinates and
    *                                 field ParticleVector.
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
               const Array<int> &field_vdims,
               const Array<const char*> &field_names_,
               int num_tags, const Array<const char*> &tag_names_,
               Ordering::Type all_ordering=Ordering::byVDIM);

   /** @brief Comprehensive parallel constructor of ParticleSet.
    *
    *  @param[in] comm_               MPI communicator.
    *  @param[in] rank_num_particles  # of particles to initialize on this rank.
    *  @param[in] dim                 Particle spatial dimension.
    *  @param[in] coords_ordering     Ordering of coordinates.
    *  @param[in] field_vdims         Array of field vector dimensions.
    *  @param[in] field_orderings     Array of field ordering types.
    *  @param[in] field_names_        Array of field names.
    *  @param[in] num_tags            Number of tags to register.
    *  @param[in] tag_names_          Array of tag names.
    */
   ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
               Ordering::Type coords_ordering, const Array<int> &field_vdims,
               const Array<Ordering::Type> &field_orderings,
               const Array<const char*> &field_names_, int num_tags,
               const Array<const char*> &tag_names_);

   /// Get the MPI communicator for this ParticleSet.
   MPI_Comm GetComm() const { return comm; };
#endif // MFEM_USE_MPI
   /// Get the global number of active particles across all ranks.
   IDType GetGlobalNParticles() const;

   /// Get the spatial dimension.
   int GetDim() const { return coords.GetVDim(); }

   /// Get the global IDs of the active particles owned by this ParticleSet.
   const Array<IDType>& GetIDs() const { return ids; }

   /** @brief Add a field to the ParticleSet.
    *
    *  @param[in] vdim             Vector dimension of the field.
    *  @param[in] field_ordering   (Optional) Ordering::Type of the field.
    *  @param[in] field_name       (Optional) Name of the field.
    *
    *  @return Index of the newly-added field.
    */
   int AddField(int vdim, Ordering::Type field_ordering=Ordering::byVDIM,
                const char* field_name=nullptr);

   /** @brief Add a field to the ParticleSet.
    *
    *  @details Same as AddField() but with different parameter order
    *  for convenience
    */
   int AddNamedField(int vdim, const char* field_name,
                     Ordering::Type field_ordering=Ordering::byVDIM)
   {
      return AddField(vdim, field_ordering, field_name);
   }

   /** @brief Add a tag to the ParticleSet.
    *
    *  @param[in] tag_name      (Optional) Name of the tag.
    *
    *  @return Index of the newly-added tag.
    */
   int AddTag(const char* tag_name=nullptr);

   /// Reserve memory for \p res particles.
   /** Can help to avoid re-allocation for adding + removing particles. */
   void Reserve(int res);

   /// Get the number of active particles currently held by this ParticleSet.
   int GetNParticles() const { return ids.Size(); }

   /// Get the number of fields registered to particles.
   int GetNFields() const { return fields.size(); }

   /// Get an Array<int> of the field vector-dimensions registered to particles.
   const Array<int> GetFieldVDims() const;

   /// Get Field vector-dimension
   int FieldVDim(int f) const { return fields[f]->GetVDim(); }

   /// Get the number of tags registered to particles.
   int GetNTags() const { return tags.size(); }

   /// Add a particle using Particle .
   void AddParticle(const Particle &p);

   /** @brief Add \p num_particles particles, and optionally get the local
    *  indices of new particles in \p new_indices .
    *
    *  @details The data of new particles is uninitialized and must be
    *  set.
    */
   void AddParticles(int num_particles, Array<int> *new_indices=nullptr);

   /// Remove particle data specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get a reference to the coordinates ParticleVector.
   ParticleVector& Coords() { return coords; }

   /// Get a const reference to the coordinates ParticleVector.
   const ParticleVector& Coords() const { return coords; }

   /// Get a reference to field \p f 's ParticleVector.
   ParticleVector& Field(int f) { return *fields[f]; }

   /// Get a const reference to field \p f 's ParticleVector.
   const ParticleVector& Field(int f) const { return *fields[f]; }

   /// Get a reference to tag \p t 's Array<int>.
   Array<int>& Tag(int t) { return *tags[t]; }

   /// Get a const reference to tag \p t 's Array<int>.
   const Array<int>& Tag(int t) const { return *tags[t]; }

   /** @brief Get new Particle object with copy of data associated with
       particle \p i . */
   Particle GetParticle(int i) const;

   /** @brief Get Particle object whose members reference the actual data
    *  associated with particle \p i in this ParticleSet.
    *
    *  @see IsParticleRefValid for when this method can be used.
    *
    *  @warning If particles are added, removed, or redistributed after
    *  invoking this, the returned Particle member references may be
    *  invalidated.
    */
   Particle GetParticleRef(int i);

   /** @brief Determine if GetParticleRef is valid.
    *
    * If coordinates and all fields are ordered byVDIM, then returns true.
    * Otherwise, false.
    */
   bool IsParticleRefValid() const;

   /// Set data for particle at index \p i with data from provided particle \p p
   void SetParticle(int i, const Particle &p);

   /** @brief Print all particle data to a comma-delimited CSV file.
    *
    *  The first row contains the header. We include the particle ID,
    *  owning rank (in parallel), coordinates, followed by all fields and
    *  tags.
    *
    *  The output can be visualized in Paraview by loading the csv files, and
    *  applying the "Table To Points" filter.
    */
   void PrintCSV(const char *fname, int precision=16);

   /** @brief Print only particle field and tags given by \p field_idxs and
       \p tag_idxs respectively to a CSV file. */
   void PrintCSV(const char *fname, const Array<int> &field_idxs,
                 const Array<int> &tag_idxs, int precision=16);

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

   /** @brief Redistribute particle data to \p rank_list

       @param[in] rank_list    Array of size GetNParticles() denoting ultimate
                               destination of particle data. Index = this rank
                               means no data is moved.
     */
   void Redistribute(const Array<unsigned int> &rank_list);

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB

   /// Destructor
   ~ParticleSet();
   ParticleSet(const ParticleSet&) = delete;
   ParticleSet& operator=(const ParticleSet&) = delete;
};

} // namespace mfem


#endif // MFEM_PARTICLESET
