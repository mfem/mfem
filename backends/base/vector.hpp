// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BACKENDS_BASE_VECTOR_HPP
#define MFEM_BACKENDS_BASE_VECTOR_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "../../general/scalars.hpp"
#include "array.hpp"

namespace mfem
{

/// Polymorphic vector - array of scalars.
template <typename T>
class PVector : public RefCounted
{
public:
   /** @brief Create a PVector. */
   /** The @a layout must be valid in the sense that layout != NULL and
       layout->HasEngine() == true. */
   PVector(PLayout &p_layout)
      : PArray(p_layout) { }

   template <typename derived_t>
   derived_t &As() { return *util::As<derived_t>(this); }

   template <typename derived_t>
   const derived_t &As() const { return *util::As<const derived_t>(this); }

   /// Get the current size of the array.
   virtual std::size_t Size() const = 0;

   /// Get the current layout of the array.
   virtual PLayout& GetLayout() const = 0;

   // TODO: Error handling ... handle errors at the Engine level, at the class
   //       level, or at the method level?

   // TODO: Asynchronous execution interface ...

   // TODO: Multi-vector interface ...


   /**
       @name Public virtual interface
    */
   ///@{

   /** @brief Create and return a new vector of the same dynamic type as this
       vector using the same layout with entries of type @a scalar_t.

       If @a copy_data is true, the contents of this vector is copied to the new
       vector; otherwise, the new vector remains uninitialized.

       If @a buffer is not NULL, return the vector data of the newly created
       object (in @a *buffer) , if it is stored as a contiguous array on the
       host; otherwise, set @a *buffer to NULL. */
   virtual DVector<T> Clone(bool copy_data) const = 0;

   /** @brief Compute and return the dot product of @a *this and @a x. In the
       case of an MPI-parallel vector, the result must be the MPI-global dot
       product. */
   /** Both vectors must have the same dynamic type and layout. */
   virtual T DotProduct(const PVector<T>& x) const = 0;

   // TODO: add reduction operations: min, max, sum

   /// Perform the operation @a *this = @a a @a x + @a b @a y.
   /** Rules:
       - the dynamic type of both @a x and @a y is the same as that of @a *this
       - if @a a == 0, neither @a x nor its data are accessed
       - if @a b == 0, neither @a y nor its data are accessed
       - @a x's data is never the same as @a y's data, unless @a a == 0, or
         @a b == 0
       - @a x's data or @a y's data may be the same as the data of @a *this
       - all accessed vectors, @a x, @a y, and @a *this have the same layout. */
   virtual void Axpby(const T& a, const PVector<T>& x, const T& b, const PVector<T>& y) = 0;

   ///@}
   // End: Virtual interface
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_VECTOR_HPP
