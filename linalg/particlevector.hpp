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

#ifndef MFEM_PARTICLEVECTOR
#define MFEM_PARTICLEVECTOR

#include "ordering.hpp"
#include "vector.hpp"

namespace mfem
{

/** \brief ParticleVector carries vector data (of a given vector dimension) for
 *  an arbitrary number of particles. Data is stored contiguously in memory in
 *  an either
 *  byNODES (x0,x1,x2,...,xN,y0,y1,y2...yN,z0.....zN) or
 *  byVDIM (x0,y0,z0,...,xN,yN,zN) ordering,
 *  where N+1 is the number of particles.
 *  ParticleVector provides convenient methods for accessing and manipulating
 *  data for individual particles (e.g., \ref GetValues) or
 *  components across all particles (e.g., \ref GetComponents).
 *
 *  Note that, since ParticleVector inherits from Vector, all Vector operations
 *  (e.g., device support) are available. We do recommend use of the
 *  methods \ref SetNumParticles and \ref SetVDim for manipulating container
 *  size, instead of \ref Vector::SetSize, to ensure consistency.
 */
class ParticleVector : public Vector
{
protected:

   /// Vector dimension.
   int vdim;

   /// Ordering of Vector data in ParticleVector.
   Ordering::Type ordering;

   /// Re-allocate + copy memory. See Array::GrowSize.
   void GrowSize(int min_num_vectors, bool keep_data);

public:

   using Vector::operator=;
   using Vector::operator();

   ParticleVector() : vdim(1), ordering(Ordering::byNODES) {};

   /// Initialize an empty ParticleVector of vdim \p vdim_ with
   /// ordering \p ordering_.
   ParticleVector(int vdim_, Ordering::Type ordering_);

   /// Initialize a ParticleVector with \p num_particles vectors each of size
   /// \p vdim_ ordered \p ordering_.
   ParticleVector(int vdim_, Ordering::Type ordering_, int num_particles);

   /// Initialize a ParticleVector of vdim \p vdim_ with ordering
   /// \p ordering_ , initialized with copy of data in \p vec .
   ParticleVector(int vdim_, Ordering::Type ordering_, const Vector &vec);

   /// Get the Vector dimension of the ParticleVector.
   int GetVDim() const { return vdim; }

   /// Get the ordering of data in the ParticleVector.
   Ordering::Type GetOrdering() const { return ordering; }

   /// Get the number of particle data in the ParticleVector.
   int GetNumParticles() const { return Size()/vdim; }

   /// Get a copy of particle \p i 's data.
   void GetValues(int i, Vector &nvals) const;

   /** @brief For `GetOrdering` == Ordering::byVDIM, set \p nref to refer to
    *  particle \p i 's data.
    *
    *  @warning This method only works when ordering is Ordering::byVDIM, where
    *  an individual particle's data is stored contiguously in memory.
    */
   void GetValuesRef(int i, Vector &nref);

   /// Get a copy of component \p vd for all particle vector data.
   void GetComponents(int vd, Vector &comp);

   /** @brief For `GetOrdering` == Ordering::byNODES, set \p nref to refer to
    * component \p vd 's data.
    *
    *  @warning This method only works when ordering is Ordering::byNODES,
    *  where an individual component of all particle data is stored
    *  contiguously in memory.
    */
   void GetComponentsRef(int vd, Vector &nref);

   /// Set particle \p i 's data to \p nvals .
   void SetValues(int i, const Vector &nvals);

   /// Set component \p vd values for all particle data to \p comp .
   void SetComponents(int vd, const Vector &comp);

   /// Reference to particle \p i component \p comp value.
   real_t& operator()(int i, int comp);

   /// Const reference to particle \p i component \p comp value.
   const real_t& operator()(int i, int comp) const;

   /** @brief Remove particle data at \p indices.
    *
    *  @details The ParticleVector is resized appropriately, with existing data maintained.
    */
   void DeleteParticles(const Array<int> &indices);

   /** @brief Remove particle data at \p index.
    *
    *  @details The ParticleVector is resized appropriately, with existing data maintained.
    */
   void DeleteParticle(const int index)
   {
      Array<int> indices({index});
      DeleteParticles(indices);
   }

   /** @brief Set the vector dimension of the ParticleVector.
    *
    *  @details If \p keep_data is true, existing particle data in the
    *  ParticleVector is maintained with an updated vector dimension \p vdim_ .
    */
   void SetVDim(int vdim_, bool keep_data=true);

   /** @brief Set the ordering of the particle Vector data in ParticleVector.
    *
    *  @details If \p keep_data is true, existing particle data in the
    *  ParticleVector is reordered to \p ordering_ .
    */
   void SetOrdering(Ordering::Type ordering_, bool keep_data=true);

   /** @brief Set the number of particle Vector data to be held by the
    *  ParticleVector, keeping existing data.
    *
    * @details If \p keep_data is true, existing particle data in the
    * ParticleVector is maintained with an ordering-mindful resize. If
    * \p num_vectors * \ref GetVDim > \p Vector::Capacity , memory
    * is re-allocated.
    */
   void SetNumParticles(int num_vectors, bool keep_data=true);

};


} // namespace mfem


#endif // MFEM_PARTICLEVECTOR
