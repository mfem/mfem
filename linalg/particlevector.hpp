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

#include "../general/ordering.hpp"
#include "vector.hpp"

namespace mfem
{

/// ParticleVector carries data for an arbitrary number of Vectors of a given size/vdim.
class ParticleVector : public Vector
{
protected:

   /// Vector dimension.
   int vdim;

   /// Ordering of Vector data in ParticleVector.
   Ordering::Type ordering;

   /// Re-allocate + copy memory. See Array::GrowSize.
   void GrowSize(int min_num_vectors);

public:

   using Vector::operator=;
   using Vector::operator();

   ParticleVector() : vdim(1), ordering(Ordering::byNODES) {};

   /// Initialize an empty ParticleVector of vdim \p vdim_ with ordering \p ordering_.
   ParticleVector(int vdim_, Ordering::Type ordering_);

   /// Initialize a ParticleVector with \p num_particles vectors each of size \p vdim_ ordered \p ordering_.
   ParticleVector(int vdim_, Ordering::Type ordering_, int num_particles);

   /// Initialize a ParticleVector of vdim \p vdim_ with ordering \p ordering_ , initialized with copy of data in \p vec .
   ParticleVector(int vdim_, Ordering::Type ordering_, const Vector &vec);

   /// Get the Vector dimension of the ParticleVector.
   int GetVDim() const { return vdim; }

   /// Get the ordering of data in the ParticleVector.
   Ordering::Type GetOrdering() const { return ordering; }

   /// Get the number of particle data in the ParticleVector.
   int GetNumParticles() const { return Size()/vdim; }

   /// Get a copy of particle \p i 's data.
   void GetValues(int i, Vector &nvals) const;

   /** @brief For `GetOrdering` == Ordering::byVDIM, set \p nref to refer to particle \p i 's data.
    *
    *  @warning This method only works when ordering is Ordering::byVDIM, where an individual particle's data is stored contiguously in memory.
    */
   void GetValuesRef(int i, Vector &nref);

   /// Get a copy of all particle values for component \p vd.
   void GetComponents(int vd, Vector &comp);

   /** @brief For `GetOrdering` == Ordering::byNODES, set \p nref to refer to component \p vd 's data.
    *
    *  @warning This method only works when ordering is Ordering::byNODES, where an individual component of all particle data is stored contiguously in memory.
    */
   void GetComponentsRef(int vd, Vector &nref);

   /// Set particle \p i 's data to \p nvals .
   void SetValues(int i, const Vector &nvals);

   /// Set component \p vd values to \p comp .
   void SetComponents(int vd, const Vector &comp);

   /// Reference to particle \p i component \p comp value.
   real_t& operator()(int i, int comp);

   /// Const reference to particle \p i component \p comp value.
   const real_t& operator()(int i, int comp) const;

   /** @brief Remove particle data at \p indices.
    *
    *  @details The ParticleVector is resized appropriately.
    */
   void DeleteParticles(const Array<int> &indices);

   /// Set the vector dimension of the ParticleVector.
   void SetVDim(int vdim_);

   /** @brief Set the ordering of the particle Vector data in ParticleVector.
    *
    *  @details For \p ordering != \ref GetOrdering , particle data in the ParticleVector is reordered.
    */
   void SetOrdering(Ordering::Type ordering_);

   /** @brief Set the number of particles held by the ParticleVector, keeping existing data.
    *
    * @details If \p num_vectors * \ref GetVDim > \p Vector::Capacity , memory is re-allocated.
    */
   void SetNumParticles(int num_vectors);

};


} // namespace mfem


#endif // MFEM_PARTICLEVECTOR
