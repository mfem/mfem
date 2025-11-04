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

/// MultiVector carries data for an arbitrary number of Vectors of a given size/vdim.
class MultiVector : public Vector
{
protected:

   /// Vector dimension.
   int vdim;

   /// Ordering of Vector data in MultiVector.
   Ordering::Type ordering;

   /// Re-allocate + copy memory. See Array::GrowSize.
   void GrowSize(int min_num_vectors);

public:

   using Vector::operator=;
   using Vector::operator();

   MultiVector() : vdim(1), ordering(Ordering::byNODES) {};

   /// Initialize an empty MultiVector of vdim \p vdim_ with ordering \p ordering_.
   MultiVector(int vdim_, Ordering::Type ordering_);

   /// Initialize a MultiVector with \p num_vectors vectors each of size \p vdim_ ordered \p ordering_.
   MultiVector(int vdim_, Ordering::Type ordering_, int num_vectors);

   /// Initialize a MultiVector of vdim \p vdim_ with ordering \p ordering_ , initialized with copy of data in \p vec .
   MultiVector(int vdim_, Ordering::Type ordering_, const Vector &vec);

   /// Get the Vector dimension of the MultiVector.
   int GetVDim() const { return vdim; }

   /// Get the ordering of data in the MultiVector.
   Ordering::Type GetOrdering() const { return ordering; }

   /// Get the number of Vectors in the MultiVector.
   int GetNumVectors() const { return Size()/vdim; }

   /// Get a copy of Vector \p i 's data.
   void GetVectorValues(int i, Vector &nvals) const;

   /** @brief For `GetOrdering` == Ordering::byVDIM, set \p nref to refer to Vector \p i 's data.
    *
    *  @warning This method only works when ordering is Ordering::byVDIM, where an individual Vector's data is stored contiguously in memory.
    */
   void GetVectorRef(int i, Vector &nref);

   /// Get a copy of all values for component \p vd.
   void GetComponentValues(int vd, Vector &comp);

   /** @brief For `GetOrdering` == Ordering::byNODES, set \p nref to refer to component \p vd 's data.
    *
    *  @warning This method only works when ordering is Ordering::byNODES, where an individual component of all Vector data is stored contiguously in memory.
    */
   void GetComponentRef(int vd, Vector &nref);

   /// Set Vector \p i 's data to \p nvals .
   void SetVectorValues(int i, const Vector &nvals);

   /// Set component \p vd values to \p comp .
   void SetComponentValues(int vd, const Vector &comp);

   /// Reference to Vector \p i component \p comp value.
   real_t& operator()(int i, int comp);

   /// Const reference to Vector \p i component \p comp value.
   const real_t& operator()(int i, int comp) const;

   /** @brief Remove Vector s at \p indices.
    *
    *  @details The MultiVector is resized appropriately.
    */
   void DeleteVectorsAt(const Array<int> &indices);

   /// Set the vector dimension of the MultiVector.
   void SetVDim(int vdim_);

   /** @brief Set the ordering of the Vector data in MultiVector.
    *
    *  @details For \p ordering != \ref GetOrdering , Vector data in the MultiVector is reordered.
    */
   void SetOrdering(Ordering::Type ordering_);

   /** @brief Set the number of vectors held by the MultiVector, keeping existing Vectors.
    *
    * @details If \p num_vectors * \ref GetVDim > \p Vector::Capacity , memory is re-allocated.
    */
   void SetNumVectors(int num_vectors);

};


} // namespace mfem


#endif // MFEM_PARTICLEVECTOR
