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

#ifndef MFEM_BACKENDS_BASE_ARRAY_HPP
#define MFEM_BACKENDS_BASE_ARRAY_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "layout.hpp"
#include "utils.hpp"

namespace mfem
{

/// Extension to the template class Array<T>
template <typename T>
class PArray : public RefCounted
{
public:
   /** @brief The @a layout parameter will be reference counted and therefore it
       should be dynamically allocated. */
   /** The @a layout must be valid in the sense that layout != NULL and
       layout->HasEngine() == true. */
   PArray(PLayout &p_layout)
      : layout(&p_layout)
   {
      MFEM_ASSERT(layout && layout->HasEngine(), "invalid layout");
   }

   virtual ~PArray() { }

   /// Get the current size of the array.
   virtual std::size_t Size() const = 0;

   /// Get the current layout of the array.
   virtual PLayout& GetLayout() const = 0;

   /// TODO
   /// Note: we cannot use static_cast for class PArray.
   template <typename derived_t>
   derived_t &As() { return dynamic_cast<derived_t&>(*this); }

   /// TODO
   /// Note: we cannot use static_cast for class PArray.
   template <typename derived_t>
   const derived_t &As() const { return dynamic_cast<const derived_t&>(*this); }


   // TODO: Error handling ... handle errors at the Engine level, at the class
   //       level, or at the method level?

   // TODO: Asynchronous execution interface ...


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
   virtual DArray<T> Clone(bool copy_data) const = 0;

   /// Resize the array, reallocating its data if necessary.
   /** If @a buffer is not NULL, return the array data (in @a *buffer), if it
       is stored as a contiguous array on the host; otherwise, set @a *buffer to
       NULL. Returns 0 on success and non-zero otherwise, e.g. if memory
       allocation fails.

       If the @a new_layout is not supported, a non-zero error code will be
       returned.

       The @a new_layout has to be valid, i.e. new_layout != NULL and
       new_layout->HasEngine() == true.

       @note If reallocation is performed, the previous content of the array is
       NOT copied to the new location. */
   virtual int Resize(PLayout &new_layout) = 0;

   /// Shortcut for Resize(*layout, buffer).
   /** This method is useful for updating the array after its layout is changed
       externally. */
   virtual int Update(T **buffer) = 0;

   /// Shortcut for layout->Resize(new_size) followed by Update()
   virtual int Resize(std::size_t new_size) = 0;

   /** @brief Get access to the contents of the array in host memory, as a
       contiguous array. */
   /** If the array data is stored as a contiguous array in host memory, return
       a pointer to it. Otherwise, copy the data to @a buffer (if @a buffer is
       not NULL) and return @a buffer.
       @note If not NULL, @a buffer is assumed to be of size greater than or
       equal to Size(). */
   virtual mfem::Array<T>* PullData(mfem::Array<T>* buffer) = 0;

   /** @brief Set all entries of the array to the (single) value pointed to by
       @a value_ptr. */
   virtual void Fill(const T &value) = 0;

   /** @brief Set all Size() entries of the array from the given contiguous
       array, @a src_buffer, on the host. */
   virtual void PushData(const mfem::Array<T>* src_buffer) = 0;

   /// Copy the data from @a src to @a *this.
   /** Both arrays must have the same dynamic type, layout, and entry type. */
   virtual void Assign(const PArray& src) = 0;

   ///@}
   // End: Virtual interface
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_ARRAY_HPP
