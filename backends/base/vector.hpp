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
class PVector : virtual public PArray
{
protected:
   /**
       @name Virtual interface
    */
   ///@{

   /** @brief Create and return a new array of the same dynamic type as this
       vector using the same layout and data type.

       Returns NULL if allocation fails.

       If @a copy_data is true, the contents of this vector is copied to the new
       vector; otherwise, the new vector remains uninitialized.

       If @a buffer is not NULL, return the vector data of the newly created
       object (in @a *buffer), if it is stored as a contiguous array on the
       host; otherwise, set @a *buffer to NULL. */
   virtual PVector *DoVectorClone(bool copy_data, void **buffer,
                                  int buffer_type_id) const = 0;

   /** @brief Compute and return the dot product of @a *this and @a x. In the
       case of an MPI-parallel vector, the result must be the MPI-global dot
       product. */
   /** Both vectors must have the same dynamic type and layout. */
   virtual void DoDotProduct(const PVector &x, void *result,
                             int result_type_id) const = 0;

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
   virtual void DoAxpby(const void *a, const PVector &x,
                        const void *b, const PVector &y,
                        int ab_type_id) = 0;

   ///@}
   // End: Virtual interface

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


   // TODO: Error handling ... handle errors at the Engine level, at the class
   //       level, or at the method level?

   // TODO: Asynchronous execution interface ...

   // TODO: Multi-vector interface ...


   /**
       @name Public virtual interface
    */
   ///@{

   /** @brief Create and return a new array (in @a *clone) of the same dynamic
       type as this array using the same layout and ItemSize().

       Set @a *clone to NULL if allocation fails.

       If @a copy_data is true, the contents of this array is copied to the new
       array; otherwise, the new array remains uninitialized.

       If @a buffer is not NULL, return the array data of the newly created
       object (in @a *buffer) , if it is stored as a contiguous array on the
       host; otherwise, set @a *buffer to NULL. */
   template <typename scalar_t>
   DVector Clone(bool copy_data, scalar_t **buffer) const
   {
      return DVector(DoVectorClone(copy_data, (void**)buffer,
                                   ScalarId<scalar_t>::value));
   }

   /// Compute and return the dot product of @a *this and @a x.
   /** Both vectors must have the same dynamic type and layout. */
   template <typename scalar_t>
   scalar_t DotProduct(const PVector &x) const
   {
      if (Size() == 0) { return scalar_t(0); }
      scalar_t result;
      DoDotProduct(x, &result, ScalarId<scalar_t>::value);
      return result;
   }

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
   template <typename scalar_t>
   void Axpby(const scalar_t &a, const PVector &x,
              const scalar_t &b, const PVector &y)
   { if (Size()) { DoAxpby(&a, x, &b, y, ScalarId<scalar_t>::value); } }

   ///@}
   // End: Virtual interface
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_VECTOR_HPP
