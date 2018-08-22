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

#ifndef MFEM_BACKENDS_PA_ARRAY_HPP
#define MFEM_BACKENDS_PA_ARRAY_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "layout.hpp"
#include "../base/array.hpp"
#ifdef __NVCC__
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

namespace mfem
{

namespace pa
{


template <Location Device>
struct ArrayType_t;

template <Location Device>
using ArrayType = typename ArrayType_t<Device>::type;

/**
*  Simple cpu backend array
*/
class HostArray : public virtual PArray
{
protected:
   //
   // Inherited fields
   //
   // DLayout layout;

   void* data;
   std::size_t size;

   //
   // Virtual interface
   //

   virtual HostArray* DoClone(bool copy_data, void **buffer,
                          std::size_t item_size) const;

   virtual int DoResize(PLayout &new_layout, void **buffer,
                        std::size_t item_size);

   virtual void* DoPullData(void *buffer, std::size_t item_size);

   virtual void DoFill(const void *value_ptr, std::size_t item_size);

   virtual void DoPushData(const void *src_buffer, std::size_t item_size);

   virtual void DoAssign(const PArray &src, std::size_t item_size);

   //
   // Auxiliary methods
   //

   inline int ResizeData(const HostLayout *lt, std::size_t item_size);

public:
   HostArray(HostLayout &lt, std::size_t item_size)
      : PArray(lt),
        data(lt.Alloc(lt.Size() * item_size)),
        size(lt.Size() * item_size)
   { }

   virtual ~HostArray() { delete [] static_cast<char*>(data); }

   /**
   *  An unsafe way to access the data, tries to provide a semblance of type safety.
   */
   template <typename T>
   T* GetTypedData() { return static_cast<T*>(data); }

   template <typename T>
   const T* GetTypedData() const { return static_cast<const T*>(data); }

   /**
   *  Overload the GetLayout of PArray to return Layout instead of PLayout to avoid to have to cast in the backend
   *  This is compliant with PArray's definition since Layout is a Covariant type of PLayout.
   */
   HostLayout &GetLayout() const
   { return *static_cast<HostLayout *>(layout.Get()); }
};

template <>
struct ArrayType_t<Host>{
   typedef HostArray type;
};

//
// Inline methods
//

inline int HostArray::ResizeData(const HostLayout *lt, std::size_t item_size)
{
   const std::size_t new_bytes = lt->Size() * item_size;
   if (size < new_bytes)
   {
      data = lt->Alloc(new_bytes);
      size = new_bytes;
   }
   return 0;
}

#ifdef __NVCC__

class CudaArray : public virtual PArray
{
private:
   void* data; //device_ptr

   virtual CudaArray* DoClone(bool copy_data, void** buffer,
                              std::size_t item_size) const;

   virtual int DoResize(PLayout& new_layout, void** buffer,
                        std::size_t item_size);

   virtual void* DoPullData(void* buffer, std::size_t item_size);

   virtual void DoFill(const void *value_ptr, std::size_t item_size);

   virtual void DoPushData(const void *src_buffer, std::size_t item_size);

   virtual void DoAssign(const PArray &src, std::size_t item_size);

public:
   CudaArray(CudaLayout& lt, std::size_t item_size)
      : PArray(lt)
   {
      int ierr = cudaMalloc(&data, item_size * lt.Size());
      //TODO check ierr
   }

   /**
   *  An unsafe way to access the data, tries to provide a semblance of type safety.
   */
   template <typename T>
   T* GetTypedData() { return static_cast<T*>(data); }

   template <typename T>
   const T* GetTypedData() const { return static_cast<const T*>(data); }
   
   CudaLayout &GetLayout() const
   { return *static_cast<CudaLayout*>(layout.Get()); }
};

template <>
struct ArrayType_t<CudaDevice>
{
   typedef CudaArray type;
};

#endif

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_ARRAY_HPP
