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

#ifndef MFEM_MEM_MANAGER_HPP
#define MFEM_MEM_MANAGER_HPP

#include "globals.hpp"

namespace mfem
{

// Implementation of MFEM's lightweight device/host memory manager designed to
// work seamlessly with the OCCA, RAJA, and other kernels supported by MFEM.

/// Memory types supported by the memory manager
struct Memory
{
   /** @brief In the documentation below, we use square brackets to indicate the
       if the type is for host or device memory. */
   enum Type
   {
      /// [host, device] No allocations or copies are possible.
      NONE      = 1 << 0,
      /// [host, device] Use std:: functions to allocate and free memory.
      STD       = 1 << 1,
      /// [device] Use the CUDA memory API through the runtime library.
      CUDA      = 1 << 2,
      /// [device] Use unified memory instead of host/device pointers.
      UNIFIED   = 1 << 3,
      /// [host] Align memory at 32 bytes.
      ALIGNED   = 1 << 4,
      /// [host] Protection against read and write accesses while the data is on
      /// the device. This protection is available when:
      ///    - MFEM_USE_CUDA and MFEM_DEBUG are set,
      ///    - MFEM_USE_MM is set in any other build, with the 'debug' device.
      DEBUG = 1 << 5
   };
};

/// The memory manager class
class MemoryManager
{
private:
   /// Allow to enable/disable the Ptr, Pull and Push functionalities
   /// New and Delete will still continue to register the pointers
   bool enabled;

   /// Allow to detect if a global memory manager instance exists
   static bool exists;

public:
   MemoryManager();
   ~MemoryManager();

   /// Adds an address in the map
   void *Insert(void *ptr, const std::size_t bytes);

   /// Remove the address from the map, as well as all its aliases
   void *Erase(void *ptr);

   /// Return true if the memory manager is used: pointers seen by mfem::New and
   /// mfem::Delete will be inserted in the ledger and erased from it
   static inline bool UsingMM()
   {
#ifdef MFEM_USE_MM
      return true;
#else
      return false;
#endif
   }

   /// Disable the memory manager: Ptr, Push and Pull will be no-op
   void Disable() { enabled = false; }

   /// Enable the memory manager: Ptr, Push and Pull wont be no-op
   void Enable() { enabled = true; }

   /// Return true if the memory manager is used and enabled
   bool IsEnabled() { return UsingMM() && enabled; }

   /// The opposite of IsEnabled().
   bool IsDisabled() { return !IsEnabled(); }

   /// Return true if a global memory manager instance exists
   static bool Exists() { return exists; }

   /** @brief Translates ptr to host or device address, depending on what
       backends are currently allowed by the Device class and on the ptr
       state. */
   void *Ptr(void *ptr);
   const void *Ptr(const void *ptr);
   void *PtrRW(void *ptr, const bool read = false);

   /// Data will be pushed/pulled before the copy happens on the H or the D
   void *Memcpy(void *dst, const void *src, std::size_t bytes);

   /// Return the bytes of the memory region which base address is ptr
   std::size_t Bytes(const void *ptr);

   /// Return true if the registered pointer is on the host side
   bool IsOnHost(const void *ptr);

   /// Return true if the pointer has been registered
   bool IsKnown(const void *ptr, const std::size_t bytes = 0);

   /// Return true if the pointer is an alias inside a registered memory region
   bool IsAlias(const void *ptr, const std::size_t bytes = 0);

   /// Push the data to the device
   void Push(const void *ptr, const std::size_t bytes);

   /// Pull the data from the device
   void Pull(const void *ptr, const std::size_t bytes);

   /// Enable the pointer to be able to be used on the device
   void DeviceEnable(const void *ptr);

   /// Enable read/write access of this memory
   void AccessEnable(const void *ptr, const std::size_t bytes);

   /// Change the types of the memory spaces
   void SetMemoryTypes(const Memory::Type host, const Memory::Type device);

   /// Return the corresponding device pointer of ptr, allocating and moving the
   /// data if needed (used in OccaPtr)
   void *GetDevicePtr(const void *ptr);

   /// Registers external host pointer in the memory manager which will manage
   /// the corresponding device pointer, but not the provided host pointer.
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

   void *New(void **ptr, const std::size_t bytes);
   void Delete(void *ptr);
};

/// The (single) global memory manager object
extern MemoryManager mm;

/// The global boolean allowing to run accelerated MFEM_FORALL
extern int parallel;

/// Main memory allocation template function. Allocates n*size bytes and returns
/// a pointer to the allocated memory.
template<class T>
inline T *New(const std::size_t n)
{
   T *ptr;
   const std::size_t bytes = n*sizeof(T);
   if (!MemoryManager::Exists()) { return ::new T[n]; }
   ptr = static_cast<T*>(mm.New((void**)&ptr, bytes));
   return static_cast<T*>(mm.Insert(ptr, bytes));
}

/// Frees the memory space pointed to by ptr, which must have been returned by a
/// previous call to mfem::New.
template<class T>
inline void Delete(T *ptr)
{
   static_assert(!std::is_void<T>::value, "Cannot Delete a void pointer. "
                 "Explicitly provide the correct type as a template parameter.");
   if (!ptr) { return; }
   if (!MemoryManager::Exists()) { delete [] ptr; return; }
   mm.Delete(ptr);
   mm.Erase(ptr);
}

/// Return a host or device address corresponding to current memory space
template <class T>
inline T *Ptr(T *a) { return static_cast<T*>(mm.Ptr(a)); }

/// Data will be pushed/pulled before the copy happens on the host or the device
inline void *Memcpy(void *dst, const void *src, std::size_t bytes)
{ return mm.Memcpy(dst, src, bytes); }

/// Push the data to the device
inline void Push(const void *ptr, const std::size_t bytes)
{ mm.Push(ptr, bytes); }

/// Pull the data from the device
inline void Pull(const void *ptr, const std::size_t bytes)
{ mm.Pull(ptr, bytes); }

/// Enable read/write access of this memory
inline void AccessEnable(const void *ptr, const std::size_t bytes)
{ mm.AccessEnable(ptr, bytes); }

/// Change the types of the each memory spaces
inline void SetMemoryTypes(Memory::Type host, Memory::Type device)
{ mm.SetMemoryTypes(host, device); }

/// Change the types of the memory spaces with the same given type
inline void SetMemoryTypes(Memory::Type both)
{ mm.SetMemoryTypes(both, both); }

/// Allow the pointer to run on devices
inline void DeviceEnable(const void *ptr)
{ mm.DeviceEnable(ptr); }

} // namespace mfem

#endif // MFEM_MEM_MANAGER_HPP
