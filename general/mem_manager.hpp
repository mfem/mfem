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

#ifndef MFEM_MEM_MANAGER
#define MFEM_MEM_MANAGER

#include "globals.hpp"


namespace mfem
{

// Implementation of MFEM's lightweight host/device memory manager designed
// to work seamlessly with the okina device kernel interface.

/// The memory manager class
class MemManager
{
public:
   MemManager();
   ~MemManager();

   /// Adds an address in the map
   void *Insert(void *ptr, const std::size_t bytes);

   /// Remove the address from the map, as well as all the address' aliases
   void *Erase(void *ptr);

   /// Return a host or device address, coresponding to the Device::mode
   void *Ptr(void *ptr);
   const void *Ptr(const void *ptr);

   /// Data will be pushed/pulled before the copy happens on the H or the D
   void* Memcpy(void *dst, const void *src,
                std::size_t bytes, const bool async = false);

   /// Return the bytes of the memory region which base address is ptr
   std::size_t Bytes(const void *ptr);

   /// Return true if the registered pointer is on the host side
   bool IsOnHost(const void *ptr);

   /// Return true if the pointer has been registered
   bool IsKnown(const void *ptr);

   /// Return true if the pointer is an alias inside a registered memory region
   bool IsAlias(const void *ptr);

   /// Push the data to the device
   void Push(const void *ptr, const std::size_t bytes =0);

   /// Pull the data from the device
   void Pull(const void *ptr, const std::size_t bytes =0);

   /// Return the coresponding device pointer of ptr, allocating and moving the
   /// data if needed (used in OccaPtr)
   void *GetDevicePtr(const void *ptr);

   /// Registers external host pointer in the memory manager. The mm will manage
   /// the corresponding device pointer (but not the provided host pointer).
   template<class T>
   void RegisterHostPtr(T *ptr_host, const std::size_t size)
   {
      Insert(ptr_host, size*sizeof(T));
#ifdef MFEM_DEBUG
      RegisterCheck(ptr_host);
#endif
   }

   /// Registers external host and device pointers in the memory manager.
   template<class T>
   void RegisterHostAndDevicePtr(T *ptr_host, T *ptr_device,
                                 const std::size_t size, const bool host)
   {
      RegisterHostPtr(ptr_host, size);
      SetHostDevicePtr(ptr_host, ptr_device, host);
   }

   /// Set the host h_ptr, device d_ptr and mode host of the memory region just
   /// been registered with h_ptr (see RegisterHostAndDevicePtr)
   void SetHostDevicePtr(void *h_ptr, void *d_ptr, const bool host);

   /// Unregisters the host pointer from the memory manager. To be used with
   /// memory not allocated by the memory manager.
   template<class T>
   void UnregisterHostPtr(T *ptr) { Erase(ptr); }

   /// Check if pointer has been registered in the memory manager
   void RegisterCheck(void *ptr);

   /// Prints all pointers known by the memory manager
   void PrintPtrs(void);

   /// Copies all memory to the current memory space
   void GetAll(void);
};

extern bool MMReady;
extern MemManager MM;

class mm
{
public:

   /// Main malloc template function. Allocates n*size bytes and returns a
   /// pointer to the allocated memory.
   template<class T>
   static inline T *New(const std::size_t n)
   { return static_cast<T*>(MM.Insert(new T[n], n*sizeof(T))); }

   /// Frees the memory space pointed to by ptr, which must have been returned
   /// by a previous call to mm::New.
   template<class T>
   static inline void Delete(T *ptr)
   {
      static_assert(!std::is_void<T>::value, "Cannot Delete a void pointer. "
                    "Explicitly provide the correct type as a template parameter.");
      if (!ptr) { return; }
      delete [] ptr;
      if (!MMReady) { return; }
      MM.Erase(ptr);
   }

   /// Return a host or device address coresponding to current memory space
   template <class T>
   static inline T *Ptr(T *a) { return static_cast<T*>(MM.Ptr(a)); }

   /// Data will be pushed/pulled before the copy happens on the H or the D
   static void* Memcpy(void *dst, const void *src,
                       std::size_t bytes, const bool async = false)
   { return MM.Memcpy(dst, src, bytes, async); }

   /// Push the data to the device
   static void Push(const void *ptr, const std::size_t bytes =0)
   { return MM.Push(ptr, bytes); }

   /// Pull the data from the device
   static void Pull(const void *ptr, const std::size_t bytes =0)
   { return MM.Pull(ptr, bytes); }
};

} // namespace mfem

#endif // MFEM_MEM_MANAGER
