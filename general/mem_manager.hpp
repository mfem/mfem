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

#ifndef MFEM_MEM_MANAGER_HPP
#define MFEM_MEM_MANAGER_HPP

#include "enzyme.hpp"
#include "globals.hpp"
#include "error.hpp"
#include <cstring> // std::memcpy
#include <type_traits> // std::is_const
#include <cstddef> // std::max_align_t

#ifdef MFEM_USE_MPI
// Enable internal hypre timing routines
#define HYPRE_TIMING
#include <HYPRE_utilities.h> // for HYPRE_GetMemoryLocation() and others
#if (21400 <= MFEM_HYPRE_VERSION) && (MFEM_HYPRE_VERSION < 21900)
#include <_hypre_utilities.h> // for HYPRE_MEMORY_HOST and others
#endif
#endif

namespace mfem
{

// Implementation of MFEM's lightweight device/host memory manager designed to
// work seamlessly with the OCCA, RAJA, and other kernels supported by MFEM.

/// Memory types supported by MFEM.
enum class MemoryType
{
   HOST,           ///< Host memory; using new[] and delete[]
   HOST_32,        ///< Host memory; aligned at 32 bytes
   HOST_64,        ///< Host memory; aligned at 64 bytes
   HOST_DEBUG,     ///< Host memory; allocated from a "host-debug" pool
   HOST_UMPIRE,    /**< Host memory; using an Umpire allocator which can be set
                        with MemoryManager::SetUmpireHostAllocatorName */
   HOST_PINNED,    ///< Host memory: pinned (page-locked)
   MANAGED,        /**< Managed memory; using CUDA or HIP *MallocManaged
                        and *Free */
   DEVICE,         ///< Device memory; using CUDA or HIP *Malloc and *Free
   DEVICE_DEBUG,   /**< Pseudo-device memory; allocated on host from a
                        "device-debug" pool */
   DEVICE_UMPIRE,  /**< Device memory; using an Umpire allocator which can be
                        set with MemoryManager::SetUmpireDeviceAllocatorName */
   DEVICE_UMPIRE_2, /**< Device memory; using a second Umpire allocator settable
                         with MemoryManager::SetUmpireDevice2AllocatorName */
   SIZE,           ///< Number of host and device memory types

   PRESERVE,       /**< Pseudo-MemoryType used as default value for MemoryType
                        parameters to request preservation of existing
                        MemoryType, e.g. in copy constructors. */
   DEFAULT         /**< Pseudo-MemoryType used as default value for MemoryType
                        parameters to request the use of the default host or
                        device MemoryType. */
};

/// Static casts to 'int' and sizes of some useful memory types.
constexpr int MemoryTypeSize = static_cast<int>(MemoryType::SIZE);
constexpr int HostMemoryType = static_cast<int>(MemoryType::HOST);
constexpr int HostMemoryTypeSize = static_cast<int>(MemoryType::DEVICE);
constexpr int DeviceMemoryType = static_cast<int>(MemoryType::MANAGED);
constexpr int DeviceMemoryTypeSize = MemoryTypeSize - DeviceMemoryType;

/// Memory type names, used during Device:: configuration.
extern MFEM_EXPORT const char *MemoryTypeName[MemoryTypeSize];

/// Memory classes identify sets of memory types.
/** This type is used by kernels that can work with multiple MemoryType%s.
 *  For example, kernels that can use DEVICE or MANAGED memory types should
 *  use MemoryClass::DEVICE for their inputs. */
enum class MemoryClass
{
   HOST,    /**< Memory types: { HOST, HOST_32, HOST_64, HOST_DEBUG,
                                 HOST_UMPIRE, HOST_PINNED, MANAGED } */
   HOST_32, ///< Memory types: { HOST_32, HOST_64, HOST_DEBUG }
   HOST_64, ///< Memory types: { HOST_64, HOST_DEBUG }
   DEVICE,  /**< Memory types: { DEVICE, DEVICE_DEBUG, DEVICE_UMPIRE,
                                 DEVICE_UMPIRE_2, MANAGED } */
   MANAGED  ///< Memory types: { MANAGED }
};

/// Return true if the given memory type is in MemoryClass::HOST.
inline bool IsHostMemory(MemoryType mt) { return mt <= MemoryType::MANAGED; }

/// Return true if the given memory type is in MemoryClass::DEVICE
inline bool IsDeviceMemory(MemoryType mt)
{
   return mt >= MemoryType::MANAGED && mt < MemoryType::SIZE;
}

/// Return a suitable MemoryType for a given MemoryClass.
MemoryType GetMemoryType(MemoryClass mc);

/// Return true iff the MemoryType @a mt is contained in the MemoryClass @a mc.
bool MemoryClassContainsType(MemoryClass mc, MemoryType mt);

/// Return a suitable MemoryClass from a pair of MemoryClass%es.
/** Note: this operation is commutative, i.e. a*b = b*a, associative, i.e.
    (a*b)*c = a*(b*c), and has an identity element: MemoryClass::HOST.

    Currently, the operation is defined as a*b := max(a,b) where the max
    operation is based on the enumeration ordering:

    HOST < HOST_32 < HOST_64 < DEVICE < MANAGED. */
MemoryClass operator*(MemoryClass mc1, MemoryClass mc2);

/// Class used by MFEM to store pointers to host and/or device memory.
/** The template class parameter, T, must be a plain-old-data (POD) type.

    In many respects this class behaves like a pointer:
    - When destroyed, a Memory object does NOT automatically delete any
      allocated memory.
    - Only the method Delete() will deallocate a Memory object.
    - Other methods that modify the object (e.g. New(), Wrap(), etc) will
      simply overwrite the old contents.
    In other aspects this class differs from a pointer:
    - Pointer arithmetic is not supported, MakeAlias() should be used instead.
    - Const Memory object does not allow modification of the content
      (unlike e.g. a const pointer).
    - Move constructor and assignment will transfer ownership flags, and
      Reset() the moved Memory object.
    - Copy constructor and assignment copy flags. This may result in two Memory
      objects owning the data which is an invalid state. This invalid state MUST
      be resolved by users manually using SetHostPtrOwner(),
      SetDevicePtrOwner(), or ClearOwnerFlags(). It is also possible to call
      Delete() on only one of the two Memory objects, however this is
      discouraged because it bypasses the internal ownership flags.
    - When moving or copying (between host and device) alias Memory objects
      and/or their base Memory objects, the consistency of memory flags have
      to be manually taken care of using either Sync() or SyncAlias(). Failure
      to do so will result in silent misuse of unsynchronized data.

    A Memory object stores up to two different pointers: one host pointer (with
    MemoryType from MemoryClass::HOST) and one device pointer (currently one of
    MemoryType: DEVICE, DEVICE_DEBUG, DEVICE_UMPIRE or MANAGED).

    A Memory object can hold (wrap) an externally allocated pointer with any
    given MemoryType.

    Access to the content of the Memory object can be requested with any given
    MemoryClass through the methods ReadWrite(), Read(), and Write().
    Requesting such access may result in additional (internally handled)
    memory allocation and/or memory copy.
    - When ReadWrite() is called, the returned pointer becomes the only
      valid pointer.
    - When Read() is called, the returned pointer becomes valid, however
      the other pointer (host or device) may remain valid as well.
    - When Write() is called, the returned pointer becomes the only valid
      pointer, however, unlike ReadWrite(), no memory copy will be performed.

    The host memory (pointer from MemoryClass::HOST) can be accessed through the
    inline methods: `operator[]()`, `operator*()`, the implicit conversion
    functions `operator T*()`, `operator const T*()`, and the explicit
    conversion template functions `operator U*()`,  `operator const U*()` (with
    any suitable type U). In certain cases, using these methods may have
    undefined behavior, e.g. if the host pointer is not currently valid. */
template <typename T>
class Memory
{
protected:
   friend class MemoryManager;
   friend void MemoryPrintFlags(unsigned flags);

   enum FlagMask: unsigned
   {
      // Workaround for use with headers that define REGISTERED as a macro,
      // e.g. nb30.h (which is included by Windows.h):
#ifndef REGISTERED
      REGISTERED    = 1 << 0, /**< The host pointer is registered with the
                                   MemoryManager */
#endif
      // Use the following identifier if REGISTERED is defined as a macro,
      // e.g. nb30.h (which is included by Windows.h):
      Registered    = 1 << 0, /**< The host pointer is registered with the
                                   MemoryManager */
      OWNS_HOST     = 1 << 1, ///< The host pointer will be deleted by Delete()
      OWNS_DEVICE   = 1 << 2, /**< The device pointer will be deleted by
                                   Delete() */
      OWNS_INTERNAL = 1 << 3, ///< Ownership flag for internal Memory data
      VALID_HOST    = 1 << 4, ///< Host pointer is valid
      VALID_DEVICE  = 1 << 5, ///< %Device pointer is valid
      USE_DEVICE    = 1 << 6, /**< Internal device flag, see e.g.
                                   Vector::UseDevice() */
      ALIAS         = 1 << 7  ///< Pointer is an alias
   };

   /// Pointer to host memory. Not owned.
   /** The type of the pointer is given by the field #h_mt; it can be any type
       from MemoryClass::HOST. */
   T *h_ptr;
   int capacity; ///< Size of the allocated memory
   MemoryType h_mt; ///< Host memory type
   mutable unsigned flags; ///< Bit flags defined from the #FlagMask enum
   // 'flags' is mutable so that it can be modified in Set{Host,Device}PtrOwner,
   // Copy{From,To}, {ReadWrite,Read,Write}.

public:
   /** Default constructor, sets the host pointer to nullptr and the metadata to
       meaningful default values. */
   Memory() { Reset(); }

   /// Copy constructor: default.
   Memory(const Memory &) = default;

   /** Move constructor. Sets the pointers and associated ownership of validity
       flags of @a *this to those of @a other. Resets @a other. */
   Memory(Memory &&other)
   {
      *this = other;
      other.Reset();
   }

   /// Copy-assignment operator: default.
   Memory &operator=(const Memory &) = default;

   /** Move assignment operator. Sets the pointers and associated ownership of
       validity flags of @a *this to those of @a other. Resets @a other. */
   Memory &operator=(Memory &&other)
   {
      // Guard self-assignment:
      if (this == &other) { return *this; }
      *this = other;
      other.Reset();
      return *this;
   }

   /// Allocate host memory for @a size entries.
   /** The allocation uses the current host memory type returned by
       MemoryManager::GetHostMemoryType(). */
   explicit Memory(int size) { New(size); }

   /// Creates a new empty Memory object with host MemoryType @a mt.
   explicit Memory(MemoryType mt) { Reset(mt); }

   /** @brief Allocate memory for @a size entries with the given MemoryType
       @a mt. */
   /** The newly allocated memory is not initialized, however the given
       MemoryType is still set as valid. */
   Memory(int size, MemoryType mt) { New(size, mt); }

   /** @brief Allocate memory for @a size entries with the given host MemoryType
       @a h_mt and device MemoryType @a d_mt. */
   /** The newly allocated memory is not initialized. The host pointer is set as
       valid. */
   Memory(int size, MemoryType h_mt, MemoryType d_mt) { New(size, h_mt, d_mt); }

   /** @brief Wrap an externally allocated host pointer, @a ptr with the current
       host memory type returned by MemoryManager::GetHostMemoryType(). */
   /** The parameter @a own determines whether @a ptr will be deleted when the
       method Delete() is called. */
   explicit Memory(T *ptr, int size, bool own) { Wrap(ptr, size, own); }

   /// Wrap an externally allocated pointer, @a ptr, of the given MemoryType.
   /** The new memory object will have the given MemoryType set as valid.

       The given @a ptr must be allocated appropriately for the given
       MemoryType.

       The parameter @a own determines whether @a ptr will be deleted when the
       method Delete() is called. */
   Memory(T *ptr, int size, MemoryType mt, bool own)
   { Wrap(ptr, size, mt, own); }

   /** @brief Alias constructor. Create a Memory object that points inside the
       Memory object @a base. */
   /** The new Memory object uses the same MemoryType(s) as @a base. */
   Memory(const Memory &base, int offset, int size)
   { MakeAlias(base, offset, size); }

   /// Destructor: default.
   /** @note The destructor will NOT delete the current memory. */
   ~Memory() = default;

   /// Swap without using move assignment, avoiding Reset() calls.
   void Swap(Memory &other)
   {
      Memory tmp(*this);
      *this = other;
      other = tmp;
   }

   /** @brief Return true if the host pointer is owned. Ownership indicates
       whether the pointer will be deleted by the method Delete(). */
   bool OwnsHostPtr() const { return flags & OWNS_HOST; }

   /** @brief Set/clear the ownership flag for the host pointer. Ownership
       indicates whether the pointer will be deleted by the method Delete(). */
   void SetHostPtrOwner(bool own) const
   { flags = own ? (flags | OWNS_HOST) : (flags & ~OWNS_HOST); }

   /** @brief Return true if the device pointer is owned. Ownership indicates
       whether the pointer will be deleted by the method Delete(). */
   bool OwnsDevicePtr() const { return flags & OWNS_DEVICE; }

   /** @brief Set/clear the ownership flag for the device pointer. Ownership
       indicates whether the pointer will be deleted by the method Delete(). */
   void SetDevicePtrOwner(bool own) const
   { flags = own ? (flags | OWNS_DEVICE) : (flags & ~OWNS_DEVICE); }

   /** @brief Clear the ownership flags for the host and device pointers, as
       well as any internal data allocated by the Memory object. */
   void ClearOwnerFlags() const
   { flags = flags & ~(OWNS_HOST | OWNS_DEVICE | OWNS_INTERNAL); }

   /// Read the internal device flag.
   bool UseDevice() const { return flags & USE_DEVICE; }

   /// Set the internal device flag.
   void UseDevice(bool use_dev) const
   { flags = use_dev ? (flags | USE_DEVICE) : (flags & ~USE_DEVICE); }

   /// Return the size of the allocated memory.
   int Capacity() const { return capacity; }

   /// Reset the memory to be empty, ensuring that Delete() will be a no-op.
   /** This is the Memory class equivalent to setting a pointer to NULL, see
       Empty().

       @note The current memory is NOT deleted by this method. */
   void Reset();

   /// Reset the memory and set the host memory type.
   void Reset(MemoryType host_mt);

   /// Return true if the Memory object is empty, see Reset().
   /** Default-constructed objects are uninitialized, so they are not guaranteed
       to be empty. */
   bool Empty() const { return h_ptr == NULL; }

   /** @brief Allocate host memory for @a size entries with the current host
       memory type returned by MemoryManager::GetHostMemoryType(). */
   /** @note The current memory is NOT deleted by this method. */
   inline void New(int size);

   /// Allocate memory for @a size entries with the given MemoryType.
   /** The newly allocated memory is not initialized, however the given
       MemoryType is still set as valid.

       When @a mt is a host type, the device MemoryType will be set later, if
       requested, using the dual type of @a mt, see
       MemoryManager::GetDualMemoryType().

       When @a mt is a device type, the host MemoryType will be set immediately
       to be the dual of @a mt, see MemoryManager::GetDualMemoryType().

       @note The current memory is NOT deleted by this method. */
   inline void New(int size, MemoryType mt);

   /** @brief Allocate memory for @a size entries with the given host MemoryType
       @a h_mt and device MemoryType @a d_mt. */
   /** The newly allocated memory is not initialized. The host pointer is set as
       valid.

       @note The current memory is NOT deleted by this method. */
   inline void New(int size, MemoryType h_mt, MemoryType d_mt);

   /** @brief Wrap an externally allocated host pointer, @a ptr with the current
       host memory type returned by MemoryManager::GetHostMemoryType(). */
   /** The parameter @a own determines whether @a ptr will be deleted when the
       method Delete() is called.

       @note The current memory is NOT deleted by this method. */
   inline void Wrap(T *ptr, int size, bool own);

   /// Wrap an externally allocated pointer, @a ptr, of the given MemoryType.
   /** The new memory object will have the given MemoryType set as valid.

       The given @a ptr must be allocated appropriately for the given
       MemoryType.

       The parameter @a own determines whether @a ptr will be deleted when the
       method Delete() is called.

       @note The current memory is NOT deleted by this method. */
   inline void Wrap(T *ptr, int size, MemoryType mt, bool own);

   /** Wrap an externally pair of allocated pointers, @a h_ptr and @a d_ptr,
       of the given host MemoryType @a h_mt. */
   /** The new memory object will have the device MemoryType set as valid unless
       specified otherwise by the parameters @a valid_host and @a valid_device.

       The given @a h_ptr and @a d_ptr must be allocated appropriately for the
       given host MemoryType and its dual device MemoryType as defined by
       MemoryManager::GetDualMemoryType().

       The parameter @a own determines whether both @a h_ptr and @a d_ptr will
       be deleted when the method Delete() is called.

       The parameters @a valid_host and @a valid_device determine which
       pointers, host and/or device, will be marked as valid; at least one of
       the two parameters must be set to true.

       @note Ownership can also be controlled by using the following methods:
         - ClearOwnerFlags,
         - SetHostPtrOwner,
         - SetDevicePtrOwner.

       @note The current memory is NOT deleted by this method. */
   inline void Wrap(T *h_ptr, T *d_ptr, int size, MemoryType h_mt, bool own,
                    bool valid_host = false, bool valid_device = true);

   /// Create a memory object that points inside the memory object @a base.
   /** The new Memory object uses the same MemoryType(s) as @a base.

       @note The current memory is NOT deleted by this method. */
   inline void MakeAlias(const Memory &base, int offset, int size);

   /// Set the device MemoryType to be used by the Memory object.
   /** If the specified @a d_mt is not a device MemoryType, i.e. not one of the
       types in MemoryClass::DEVICE, then this method will return immediately.

       If the device MemoryType has been previously set to a different type and
       the actual device memory has been allocated, this method will trigger an
       error. This method will not perform the actual device memory allocation,
       however, the allocation may already exist if the MemoryType is the same
       as the current one.

       If the Memory is an alias Memory, the device MemoryType of its base will
       be updated as described above. */
   inline void SetDeviceMemoryType(MemoryType d_mt);

   /** @brief Delete the owned pointers and reset the Memory object. */
   inline void Delete();

   /** @brief Delete the device pointer, if owned. If @a copy_to_host is true
       and the data is valid only on device, move it to host before deleting.
       Invalidates the device memory. */
   inline void DeleteDevice(bool copy_to_host = true);

   /// Array subscript operator for host memory.
   inline T &operator[](int idx);

   /// Array subscript operator for host memory, const version.
   inline const T &operator[](int idx) const;

   /// Direct access to the host memory as T* (implicit conversion).
   /** When the type T is const-qualified, this method can be used only if the
       host pointer is currently valid (the device pointer may be valid or
       invalid).

       When the type T is not const-qualified, this method can be used only if
       the host pointer is the only valid pointer.

       When the Memory is empty, this method can be used and it returns NULL. */
   inline operator T*();

   /// Direct access to the host memory as const T* (implicit conversion).
   /** This method can be used only if the host pointer is currently valid (the
       device pointer may be valid or invalid).

       When the Memory is empty, this method can be used and it returns NULL. */
   inline operator const T*() const;

   /// Direct access to the host memory via explicit typecast.
   /** A pointer to type T must be reinterpret_cast-able to a pointer to type U.
       In particular, this method cannot be used to cast away const-ness from
       the base type T.

       When the type U is const-qualified, this method can be used only if the
       host pointer is currently valid (the device pointer may be valid or
       invalid).

       When the type U is not const-qualified, this method can be used only if
       the host pointer is the only valid pointer.

       When the Memory is empty, this method can be used and it returns NULL. */
   template <typename U>
   inline explicit operator U*();

   /// Direct access to the host memory via explicit typecast, const version.
   /** A pointer to type T must be reinterpret_cast-able to a pointer to type
       const U.

       This method can be used only if the host pointer is currently valid (the
       device pointer may be valid or invalid).

       When the Memory is empty, this method can be used and it returns NULL. */
   template <typename U>
   inline explicit operator const U*() const;

   /// Get read-write access to the memory with the given MemoryClass.
   /** If only read or only write access is needed, then the methods
       Read() or Write() should be used instead of this method.

       The parameter @a size must not exceed the Capacity(). */
   inline T *ReadWrite(MemoryClass mc, int size);

   /// Get read-only access to the memory with the given MemoryClass.
   /** The parameter @a size must not exceed the Capacity(). */
   inline const T *Read(MemoryClass mc, int size) const;

   /// Get write-only access to the memory with the given MemoryClass.
   /** The parameter @a size must not exceed the Capacity().

       The contents of the returned pointer is undefined, unless it was
       validated by a previous call to Read() or ReadWrite() with
       the same MemoryClass. */
   inline T *Write(MemoryClass mc, int size);

   /// Copy the host/device pointer validity flags from @a other to @a *this.
   /** This method synchronizes the pointer validity flags of two Memory objects
       that use the same host/device pointers, or when @a *this is an alias
       (sub-Memory) of @a other. Typically, this method should be called after
       @a other is manipulated in a way that changes its pointer validity flags
       (e.g. it was moved from device to host memory). */
   inline void Sync(const Memory &other) const;

   /** @brief Update the alias Memory @a *this to match the memory location (all
       valid locations) of its base Memory, @a base. */
   /** This method is useful when alias Memory is moved and manipulated in a
       different memory space. Such operations render the pointer validity flags
       of the base incorrect. Calling this method will ensure that @a base is
       up-to-date. Note that this is achieved by moving/copying @a *this (if
       necessary), and not @a base. */
   inline void SyncAlias(const Memory &base, int alias_size) const;

   /** @brief Return a MemoryType that is currently valid. If both the host and
       the device pointers are currently valid, then the device memory type is
       returned. */
   inline MemoryType GetMemoryType() const;

   /// Return the host MemoryType of the Memory object.
   inline MemoryType GetHostMemoryType() const { return h_mt; }

   /** @brief Return the device MemoryType of the Memory object. If the device
       MemoryType is not set, return MemoryType::DEFAULT. */
   inline MemoryType GetDeviceMemoryType() const;

   /** @brief Return true if host pointer is valid */
   inline bool HostIsValid() const;

   /** @brief Return true if device pointer is valid */
   inline bool DeviceIsValid() const;

   /// Copy @a size entries from @a src to @a *this.
   /** The given @a size should not exceed the Capacity() of the source @a src
       and the destination, @a *this. */
   inline void CopyFrom(const Memory &src, int size);

   /// Copy @a size entries from the host pointer @a src to @a *this.
   /** The given @a size should not exceed the Capacity() of @a *this. */
   inline void CopyFromHost(const T *src, int size);

   /// Copy @a size entries from @a *this to @a dest.
   /** The given @a size should not exceed the Capacity() of @a *this and the
       destination, @a dest. */
   inline void CopyTo(Memory &dest, int size) const;

   /// Copy @a size entries from @a *this to the host pointer @a dest.
   /** The given @a size should not exceed the Capacity() of @a *this. */
   inline void CopyToHost(T *dest, int size) const;

   /// Print the internal flags.
   /** This method can be useful for debugging. It is explicitly instantiated
       for Memory<T> with T = int and T = real_t. */
   inline void PrintFlags() const;

   /// If both the host and the device data are valid, compare their contents.
   /** This method can be useful for debugging. It is explicitly instantiated
       for Memory<T> with T = int and T = real_t. */
   inline int CompareHostAndDevice(int size) const;

private:
   // GCC 4.8 workaround: max_align_t is not in std.
   static constexpr std::size_t def_align_bytes_()
   {
      using namespace std;
      return alignof(max_align_t);
   }
   static constexpr std::size_t def_align_bytes = def_align_bytes_();
   static constexpr std::size_t new_align_bytes =
      alignof(T) > def_align_bytes ? alignof(T) : def_align_bytes;

   template <std::size_t align_bytes, bool dummy = true> struct Alloc
   {
#if __cplusplus < 201703L
      static inline T *New(std::size_t)
      {
         // Generate an error in debug mode
         MFEM_ASSERT(false, "overaligned type cannot use MemoryType::HOST");
         return nullptr;
      }
#else
      static inline T *New(std::size_t size) { return new T[size]; }
#endif
   };

#if __cplusplus < 201703L
   template<bool dummy> struct Alloc<def_align_bytes,dummy>
   {
      static inline T *New(std::size_t size) { return new T[size]; }
   };
#endif

   // Shortcut for Alloc<new_align_bytes>::New(size)
   static inline T *NewHOST(std::size_t size)
   {
      return Alloc<new_align_bytes>::New(size);
   }
};


/** @brief Swap of Memory<T> objects for use with standard library algorithms.
    Also, used by mfem::Swap(). */
template <typename T>
void swap(Memory<T> &a, Memory<T> &b)
{
   a.Swap(b);
}


/** The MFEM memory manager class. Host-side pointers are inserted into this
    manager which keeps track of the associated device pointer, and where the
    data currently resides. */
class MFEM_EXPORT MemoryManager
{
private:

   typedef MemoryType MemType;
   typedef Memory<int> Mem;

   template <typename T> friend class Memory;

   /// Host memory type set during the Setup.
   MFEM_ENZYME_INACTIVE static MemoryType host_mem_type;

   /// Device memory type set during the Setup.
   MFEM_ENZYME_INACTIVE static MemoryType device_mem_type;

   /// Allow to detect if a global memory manager instance exists.
   MFEM_ENZYME_INACTIVE static bool exists;

   /// Return true if the global memory manager instance exists.
   static bool Exists() { return exists; }

   /// Array defining the dual MemoryType for each MemoryType
   /** The dual of a host MemoryType is a device MemoryType and vice versa: the
       dual of a device MemoryType is a host MemoryType. */
   MFEM_ENZYME_INACTIVE static MemoryType dual_map[MemoryTypeSize];

   /// Update the dual memory type of @a mt to be @a dual_mt.
   static void UpdateDualMemoryType(MemoryType mt, MemoryType dual_mt);

   /// True if Configure() was called.
   MFEM_ENZYME_INACTIVE static bool configured;

   /// Host and device allocator names for Umpire.
#ifdef MFEM_USE_UMPIRE
   static const char * h_umpire_name;
   static const char * d_umpire_name;
   static const char * d_umpire_2_name;
#endif

private: // Static methods used by the Memory<T> class

   /// Allocate and register a new pointer. Return the host pointer.
   /// h_tmp must be already allocated using new T[] if mt is a pure device
   /// memory type, e.g. CUDA (mt will not be HOST).
   static void *New_(void *h_tmp, size_t bytes, MemoryType mt, unsigned &flags);

   static void *New_(void *h_tmp, size_t bytes, MemoryType h_mt,
                     MemoryType d_mt, unsigned valid_flags, unsigned &flags);

   /// Register an external pointer of the given MemoryType.
   /// Return the host pointer.
   MFEM_ENZYME_INACTIVE static void *Register_(void *ptr, void *h_ptr,
                                               size_t bytes, MemoryType mt,
                                               bool own, bool alias, unsigned &flags);

   /// Register a pair of external host and device pointers
   static void Register2_(void *h_ptr, void *d_ptr, size_t bytes,
                          MemoryType h_mt, MemoryType d_mt,
                          bool own, bool alias, unsigned &flags,
                          unsigned valid_flags);

   /// Register an alias. Note: base_h_ptr may be an alias.
   static void Alias_(void *base_h_ptr, size_t offset, size_t bytes,
                      unsigned base_flags, unsigned &flags);

   static void SetDeviceMemoryType_(void *h_ptr, unsigned flags,
                                    MemoryType d_mt);

   /// Un-register and free memory identified by its host pointer.
   MFEM_ENZYME_FN_LIKE_FREE static void Delete_(void *h_ptr, MemoryType mt,
                                                unsigned flags);

   /// Free device memory identified by its host pointer
   static void DeleteDevice_(void *h_ptr, unsigned & flags);

   /// Check if the memory types given the memory class are valid
   static bool MemoryClassCheck_(MemoryClass mc, void *h_ptr,
                                 MemoryType h_mt, size_t bytes, unsigned flags);

   /// Return a pointer to the memory identified by the host pointer h_ptr for
   /// access with the given MemoryClass.
   MFEM_ENZYME_FN_LIKE_DYNCAST static void *ReadWrite_(void *h_ptr,
                                                       MemoryType h_mt, MemoryClass mc,
                                                       size_t bytes, unsigned &flags);

   MFEM_ENZYME_FN_LIKE_DYNCAST static const void *Read_(void *h_ptr,
                                                        MemoryType h_mt,  MemoryClass mc,
                                                        size_t bytes, unsigned &flags);

   MFEM_ENZYME_FN_LIKE_DYNCAST static void *Write_(void *h_ptr, MemoryType h_mt,
                                                   MemoryClass mc,
                                                   size_t bytes, unsigned &flags);

   static void SyncAlias_(const void *base_h_ptr, void *alias_h_ptr,
                          size_t alias_bytes, unsigned base_flags,
                          unsigned &alias_flags);

   /// Return the type the of the currently valid memory.
   /// If more than one types are valid, return a device type.
   MFEM_ENZYME_INACTIVE static MemoryType GetDeviceMemoryType_(void *h_ptr,
                                                               bool alias);

   /// Return the type the of the host memory.
   MFEM_ENZYME_INACTIVE static MemoryType GetHostMemoryType_(void *h_ptr);

   /// Verify that h_mt and h_ptr's h_mt (memory or alias) are equal.
   static void CheckHostMemoryType_(MemoryType h_mt, void *h_ptr, bool alias);

   /// Copy entries from valid memory type to valid memory type.
   ///  Both dest_h_ptr and src_h_ptr are registered host pointers.
   static void Copy_(void *dest_h_ptr, const void *src_h_ptr, size_t bytes,
                     unsigned src_flags, unsigned &dest_flags);

   /// Copy entries from valid memory type to host memory, where dest_h_ptr is
   /// not a registered host pointer and src_h_ptr is a registered host pointer.
   static void CopyToHost_(void *dest_h_ptr, const void *src_h_ptr,
                           size_t bytes, unsigned src_flags);

   /// Copy entries from host memory to valid memory type, where dest_h_ptr is a
   /// registered host pointer and src_h_ptr is not a registered host pointer.
   static void CopyFromHost_(void *dest_h_ptr, const void *src_h_ptr,
                             size_t bytes, unsigned &dest_flags);

   /// Check if the host pointer has been registered in the memory manager.
   static bool IsKnown_(const void *h_ptr);

   /** @brief Check if the host pointer has been registered as an alias in the
       memory manager. */
   static bool IsAlias_(const void *h_ptr);

   /// Compare the contents of the host and the device memory.
   static int CompareHostAndDevice_(void *h_ptr, size_t size, unsigned flags);

private:

   /// Insert a host address @a h_ptr and size *a bytes in the memory map to be
   /// managed.
   void Insert(void *h_ptr, size_t bytes, MemoryType h_mt,  MemoryType d_mt);

   /// Insert a device and the host addresses in the memory map
   void InsertDevice(void *d_ptr, void *h_ptr, size_t bytes,
                     MemoryType h_mt,  MemoryType d_mt);

   /// Insert an alias in the alias map
   void InsertAlias(const void *base_ptr, void *alias_ptr,
                    const size_t bytes, const bool base_is_alias);

   /// Erase an address from the memory map, as well as all its aliases
   void Erase(void *h_ptr, bool free_dev_ptr = true);

   /// Erase device memory for a given host address
   void EraseDevice(void *h_ptr);

   /// Erase an alias from the aliases map
   void EraseAlias(void *alias_ptr);

   /// Return the corresponding device pointer of h_ptr,
   /// allocating and moving the data if needed
   void *GetDevicePtr(const void *h_ptr, size_t bytes, bool copy_data);

   /// Return the corresponding device pointer of alias_ptr,
   /// allocating and moving the data if needed
   void *GetAliasDevicePtr(const void *alias_ptr, size_t bytes, bool copy_data);

   /// Return the corresponding host pointer of d_ptr,
   /// allocating and moving the data if needed
   void *GetHostPtr(const void *d_ptr, size_t bytes, bool copy_data);

   /// Return the corresponding host pointer of alias_ptr,
   /// allocating and moving the data if needed
   void *GetAliasHostPtr(const void *alias_ptr, size_t bytes, bool copy_data);

public:
   MemoryManager();
   ~MemoryManager();

   /// Initialize the memory manager.
   void Init();

   /// Return the dual MemoryType of the given one, @a mt.
   /** The default dual memory types are:

       memory type     | dual type
       --------------- | ---------
       HOST            | DEVICE
       HOST_32         | DEVICE
       HOST_64         | DEVICE
       HOST_DEBUG      | DEVICE_DEBUG
       HOST_UMPIRE     | DEVICE_UMPIRE
       HOST_PINNED     | DEVICE
       MANAGED         | MANAGED
       DEVICE          | HOST
       DEVICE_DEBUG    | HOST_DEBUG
       DEVICE_UMPIRE   | HOST_UMPIRE
       DEVICE_UMPIRE_2 | HOST_UMPIRE

       The dual types can be modified before device configuration using the
       method SetDualMemoryType() or by calling Device::SetMemoryTypes(). */
   static inline MemoryType GetDualMemoryType(MemoryType mt)
   { return dual_map[(int)mt]; }

   /// Set the dual memory type of @a mt to be @a dual_mt.
   /** This method can only be called before configuration, i.e. before calling
       Configure(), which is typically done during Device construction.

       One of the types must be a host MemoryType and the other must be a device
       MemoryType or both types must be the same host memory type. The latter
       case is only allowed for convenience in setting up pure host execution,
       so the actual dual is not updated. */
   static void SetDualMemoryType(MemoryType mt, MemoryType dual_mt);

   /** @brief Configure the Memory manager with given default host and device
       types. This method will be called when configuring a device.

       The host and device MemoryType%s, @a h_mt and @a d_mt, are set to be dual
       to each other. */
   void Configure(const MemoryType h_mt, const MemoryType d_mt);

#ifdef MFEM_USE_UMPIRE
   /// Set the host Umpire allocator name used with MemoryType::HOST_UMPIRE
   static void SetUmpireHostAllocatorName(const char * h_name) { h_umpire_name = h_name; }
   /// Set the device Umpire allocator name used with MemoryType::DEVICE_UMPIRE
   static void SetUmpireDeviceAllocatorName(const char * d_name) { d_umpire_name = d_name; }
   /// Set the device Umpire allocator name used with MemoryType::DEVICE_UMPIRE_2
   static void SetUmpireDevice2AllocatorName(const char * d_name) { d_umpire_2_name = d_name; }

   /// Get the host Umpire allocator name used with MemoryType::HOST_UMPIRE
   static const char * GetUmpireHostAllocatorName() { return h_umpire_name; }
   /// Get the device Umpire allocator name used with MemoryType::DEVICE_UMPIRE
   static const char * GetUmpireDeviceAllocatorName() { return d_umpire_name; }
   /// Get the device Umpire allocator name used with MemoryType::DEVICE_UMPIRE_2
   static const char * GetUmpireDevice2AllocatorName() { return d_umpire_2_name; }
#endif

   /// Free all the device memories
   void Destroy();

   /// Return true if the pointer is known by the memory manager
   bool IsKnown(const void *h_ptr) { return IsKnown_(h_ptr); }

   /// Return true if the pointer is known by the memory manager as an alias
   bool IsAlias(const void *h_ptr) { return IsAlias_(h_ptr); }

   /// Check if the host pointer has been registered in the memory manager
   void RegisterCheck(void *h_ptr);

   /// Prints all pointers known by the memory manager,
   /// returning the number of printed pointers
   int PrintPtrs(std::ostream &out = mfem::out);

   /// Prints all aliases known by the memory manager
   /// returning the number of printed pointers
   int PrintAliases(std::ostream &out = mfem::out);

   static MemoryType GetHostMemoryType() { return host_mem_type; }
   static MemoryType GetDeviceMemoryType() { return device_mem_type; }

#ifdef MFEM_USE_ENZYME
   static void myfree(void* mem, MemoryType MT, unsigned &flags)
   {
      MemoryManager::Delete_(mem, MT, flags);
   }
   __attribute__((used))
   inline static void* __enzyme_allocation_like1[4] = {(void*)static_cast<void*(*)(void*, size_t, MemoryType, unsigned&)>(MemoryManager::New_),
                                                       (void*)1, (void*)"-1,2,3", (void*)myfree
                                                      };
   __attribute__((used))
   inline static void* __enzyme_allocation_like2[4] = {(void*)static_cast<void*(*)(void*, size_t, MemoryType, MemoryType, unsigned, unsigned&)>(MemoryManager::New_),
                                                       (void*)1, (void*)"-1,2,4", (void*)MemoryManager::Delete_
                                                      };
#endif
};


#ifdef MFEM_USE_MPI

#if MFEM_HYPRE_VERSION < 21400
#define HYPRE_MEMORY_DEVICE (0)
#define HYPRE_MEMORY_HOST   (1)
#endif
#if MFEM_HYPRE_VERSION < 21900
typedef int HYPRE_MemoryLocation;
#endif

/// Return the configured HYPRE_MemoryLocation
inline HYPRE_MemoryLocation GetHypreMemoryLocation()
{
#if !defined(HYPRE_USING_GPU)
   return HYPRE_MEMORY_HOST;
#elif MFEM_HYPRE_VERSION < 23100
   return HYPRE_MEMORY_DEVICE;
#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
   if (!HYPRE_Initialized()) { return HYPRE_MEMORY_HOST; }
   HYPRE_MemoryLocation loc;
   HYPRE_GetMemoryLocation(&loc);
   return loc;
#endif
}

/// Return true if HYPRE is configured to use GPU
inline bool HypreUsingGPU()
{
#if !defined(HYPRE_USING_GPU)
   return false;
#elif MFEM_HYPRE_VERSION < 23100
   return true;
#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
   return GetHypreMemoryLocation() != HYPRE_MEMORY_HOST;
#endif
}

#endif // MFEM_USE_MPI


// Inline methods

template <typename T>
inline void Memory<T>::Reset()
{
   h_ptr = NULL;
   h_mt = MemoryManager::GetHostMemoryType();
   capacity = 0;
   flags = 0;
}

template <typename T>
inline void Memory<T>::Reset(MemoryType host_mt)
{
   h_ptr = NULL;
   h_mt = host_mt;
   capacity = 0;
   flags = 0;
}

template <typename T>
inline void Memory<T>::New(int size)
{
   capacity = size;
   flags = OWNS_HOST | VALID_HOST;
   h_mt = MemoryManager::GetHostMemoryType();
   h_ptr = (h_mt == MemoryType::HOST) ? NewHOST(size) :
           (T*)MemoryManager::New_(nullptr, size*sizeof(T), h_mt, flags);
}

template <typename T>
inline void Memory<T>::New(int size, MemoryType mt)
{
   capacity = size;
   const size_t bytes = size*sizeof(T);
   const bool mt_host = mt == MemoryType::HOST;
   if (mt_host) { flags = OWNS_HOST | VALID_HOST; }
   h_mt = IsHostMemory(mt) ? mt : MemoryManager::GetDualMemoryType(mt);
   T *h_tmp = (h_mt == MemoryType::HOST) ? NewHOST(size) : nullptr;
   h_ptr = (mt_host) ? h_tmp : (T*)MemoryManager::New_(h_tmp, bytes, mt, flags);
}

template <typename T>
inline void Memory<T>::New(int size, MemoryType host_mt, MemoryType device_mt)
{
   capacity = size;
   const size_t bytes = size*sizeof(T);
   this->h_mt = host_mt;
   T *h_tmp = (host_mt == MemoryType::HOST) ? NewHOST(size) : nullptr;
   h_ptr = (T*)MemoryManager::New_(h_tmp, bytes, host_mt, device_mt,
                                   VALID_HOST, flags);
}

template <typename T>
inline void Memory<T>::Wrap(T *ptr, int size, bool own)
{
   h_ptr = ptr;
   capacity = size;
   flags = (own ? OWNS_HOST : 0) | VALID_HOST;
   h_mt = MemoryManager::GetHostMemoryType();
#ifdef MFEM_DEBUG
   if (own && MemoryManager::Exists())
   {
      MemoryType h_ptr_mt = MemoryManager::GetHostMemoryType_(h_ptr);
      MFEM_VERIFY(h_mt == h_ptr_mt,
                  "h_mt = " << (int)h_mt << ", h_ptr_mt = " << (int)h_ptr_mt);
   }
#endif
   if (own && h_mt != MemoryType::HOST)
   {
      const size_t bytes = size*sizeof(T);
      MemoryManager::Register_(ptr, ptr, bytes, h_mt, own, false, flags);
   }
}

template <typename T>
inline void Memory<T>::Wrap(T *ptr, int size, MemoryType mt, bool own)
{
   capacity = size;
   if (IsHostMemory(mt))
   {
      h_mt = mt;
      h_ptr = ptr;
      if (mt == MemoryType::HOST || !own)
      {
         // Skip registration
         flags = (own ? OWNS_HOST : 0) | VALID_HOST;
         return;
      }
   }
   else
   {
      h_mt = MemoryManager::GetDualMemoryType(mt);
      h_ptr = (h_mt == MemoryType::HOST) ? NewHOST(size) : nullptr;
   }
   flags = 0;
   h_ptr = (T*)MemoryManager::Register_(ptr, h_ptr, size*sizeof(T), mt,
                                        own, false, flags);
}

template <typename T>
inline void Memory<T>::Wrap(T *h_ptr_, T *d_ptr, int size, MemoryType h_mt_,
                            bool own, bool valid_host, bool valid_device)
{
   h_mt = h_mt_;
   flags = 0;
   h_ptr = h_ptr_;
   capacity = size;
   MFEM_ASSERT(IsHostMemory(h_mt),"");
   MFEM_ASSERT(valid_host || valid_device,"");
   const size_t bytes = size*sizeof(T);
   const MemoryType d_mt = MemoryManager::GetDualMemoryType(h_mt);
   MemoryManager::Register2_(h_ptr, d_ptr, bytes, h_mt, d_mt,
                             own, false, flags,
                             valid_host*VALID_HOST|valid_device*VALID_DEVICE);
}

template <typename T>
inline void Memory<T>::MakeAlias(const Memory &base, int offset, int size)
{
   MFEM_ASSERT(0 <= offset, "invalid offset = " << offset);
   MFEM_ASSERT(0 <= size, "invalid size = " << size);
   MFEM_ASSERT(offset + size <= base.capacity,
               "invalid offset + size = " << offset + size
               << " > base capacity = " << base.capacity);
   capacity = size;
   h_mt = base.h_mt;
   h_ptr = base.h_ptr + offset;
   if (!(base.flags & Registered))
   {
      if (
#if !defined(HYPRE_USING_GPU)
         // If the following condition is true then MemoryManager::Exists()
         // should also be true:
         IsDeviceMemory(MemoryManager::GetDeviceMemoryType())
#elif MFEM_HYPRE_VERSION < 23100
         // When HYPRE_USING_GPU is defined and HYPRE < 2.31.0, we always
         // register the 'base' if the MemoryManager::Exists():
         MemoryManager::Exists()
#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
         IsDeviceMemory(MemoryManager::GetDeviceMemoryType()) ||
         (MemoryManager::Exists() && HypreUsingGPU())
#endif
      )
      {
         // Register 'base':
         MemoryManager::Register_(base.h_ptr, nullptr, base.capacity*sizeof(T),
                                  base.h_mt, base.flags & OWNS_HOST,
                                  base.flags & ALIAS, base.flags);
      }
      else
      {
         // Copy the flags from 'base', setting the ALIAS flag to true, and
         // setting both OWNS_HOST and OWNS_DEVICE to false:
         flags = (base.flags | ALIAS) & ~(OWNS_HOST | OWNS_DEVICE);
         return;
      }
   }
   const size_t s_bytes = size*sizeof(T);
   const size_t o_bytes = offset*sizeof(T);
   MemoryManager::Alias_(base.h_ptr, o_bytes, s_bytes, base.flags, flags);
}

template <typename T>
inline void Memory<T>::SetDeviceMemoryType(MemoryType d_mt)
{
   if (!IsDeviceMemory(d_mt)) { return; }
   if (!(flags & Registered))
   {
      MemoryManager::Register_(h_ptr, nullptr, capacity*sizeof(T), h_mt,
                               flags & OWNS_HOST, flags & ALIAS, flags);
   }
   MemoryManager::SetDeviceMemoryType_(h_ptr, flags, d_mt);
}

template <typename T>
inline void Memory<T>::Delete()
{
   const bool registered = flags & Registered;
   const bool mt_host = h_mt == MemoryType::HOST;
   const bool std_delete = !registered && mt_host;

   if (!std_delete)
   {
      MemoryManager::Delete_((void*)h_ptr, h_mt, flags);
   }

   if (mt_host)
   {
      if (flags & OWNS_HOST) { delete [] h_ptr; }
   }
   Reset(h_mt);
}

template <typename T>
inline void Memory<T>::DeleteDevice(bool copy_to_host)
{
   if (flags & Registered)
   {
      if (copy_to_host) { Read(MemoryClass::HOST, capacity); }
      MemoryManager::DeleteDevice_((void*)h_ptr, flags);
   }
}

template <typename T>
inline T &Memory<T>::operator[](int idx)
{
   MFEM_ASSERT((flags & VALID_HOST) && !(flags & VALID_DEVICE),
               "invalid host pointer access");
   return h_ptr[idx];
}

template <typename T>
inline const T &Memory<T>::operator[](int idx) const
{
   MFEM_ASSERT((flags & VALID_HOST), "invalid host pointer access");
   return h_ptr[idx];
}

template <typename T>
inline Memory<T>::operator T*()
{
   MFEM_ASSERT(Empty() ||
               ((flags & VALID_HOST) &&
                (std::is_const<T>::value || !(flags & VALID_DEVICE))),
               "invalid host pointer access");
   return h_ptr;
}

template <typename T>
inline Memory<T>::operator const T*() const
{
   MFEM_ASSERT(Empty() || (flags & VALID_HOST), "invalid host pointer access");
   return h_ptr;
}

template <typename T> template <typename U>
inline Memory<T>::operator U*()
{
   MFEM_ASSERT(Empty() ||
               ((flags & VALID_HOST) &&
                (std::is_const<U>::value || !(flags & VALID_DEVICE))),
               "invalid host pointer access");
   return reinterpret_cast<U*>(h_ptr);
}

template <typename T> template <typename U>
inline Memory<T>::operator const U*() const
{
   MFEM_ASSERT(Empty() || (flags & VALID_HOST), "invalid host pointer access");
   return reinterpret_cast<U*>(h_ptr);
}

template <typename T>
inline T *Memory<T>::ReadWrite(MemoryClass mc, int size)
{
   const size_t bytes = size * sizeof(T);
   if (!(flags & Registered))
   {
      if (mc == MemoryClass::HOST) { return h_ptr; }
      MemoryManager::Register_(h_ptr, nullptr, capacity*sizeof(T), h_mt,
                               flags & OWNS_HOST, flags & ALIAS, flags);
   }
   return (T*)MemoryManager::ReadWrite_(h_ptr, h_mt, mc, bytes, flags);
}

template <typename T>
inline const T *Memory<T>::Read(MemoryClass mc, int size) const
{
   const size_t bytes = size * sizeof(T);
   if (!(flags & Registered))
   {
      if (mc == MemoryClass::HOST) { return h_ptr; }
      MemoryManager::Register_(h_ptr, nullptr, capacity*sizeof(T), h_mt,
                               flags & OWNS_HOST, flags & ALIAS, flags);
   }
   return (const T*)MemoryManager::Read_(h_ptr, h_mt, mc, bytes, flags);
}

template <typename T>
inline T *Memory<T>::Write(MemoryClass mc, int size)
{
   const size_t bytes = size * sizeof(T);
   if (!(flags & Registered))
   {
      if (mc == MemoryClass::HOST) { return h_ptr; }
      MemoryManager::Register_(h_ptr, nullptr, capacity*sizeof(T), h_mt,
                               flags & OWNS_HOST, flags & ALIAS, flags);
   }
   return (T*)MemoryManager::Write_(h_ptr, h_mt, mc, bytes, flags);
}

template <typename T>
inline void Memory<T>::Sync(const Memory &other) const
{
   if (!(flags & Registered) && (other.flags & Registered))
   {
      MFEM_ASSERT(h_ptr == other.h_ptr &&
                  (flags & ALIAS) == (other.flags & ALIAS),
                  "invalid input");
      flags = (flags | Registered) & ~(OWNS_DEVICE | OWNS_INTERNAL);
   }
   flags = (flags & ~(VALID_HOST | VALID_DEVICE)) |
           (other.flags & (VALID_HOST | VALID_DEVICE));
}

template <typename T>
inline void Memory<T>::SyncAlias(const Memory &base, int alias_size) const
{
   // Assuming that if *this is registered then base is also registered.
   MFEM_ASSERT(!(flags & Registered) || (base.flags & Registered),
               "invalid base state");
   if (!(base.flags & Registered)) { return; }
   MemoryManager::SyncAlias_(base.h_ptr, h_ptr, alias_size*sizeof(T),
                             base.flags, flags);
}

template <typename T>
inline MemoryType Memory<T>::GetMemoryType() const
{
   if (h_ptr == nullptr || !(flags & VALID_DEVICE)) { return h_mt; }
   return MemoryManager::GetDeviceMemoryType_(h_ptr, flags & ALIAS);
}

template <typename T>
inline MemoryType Memory<T>::GetDeviceMemoryType() const
{
   if (!(flags & Registered)) { return MemoryType::DEFAULT; }
   return MemoryManager::GetDeviceMemoryType_(h_ptr, flags & ALIAS);
}

template <typename T>
inline bool Memory<T>::HostIsValid() const
{
   return flags & VALID_HOST ? true : false;
}

template <typename T>
inline bool Memory<T>::DeviceIsValid() const
{
   return flags & VALID_DEVICE ? true : false;
}

template <typename T>
inline void Memory<T>::CopyFrom(const Memory &src, int size)
{
   MFEM_VERIFY(src.capacity>=size && capacity>=size, "Incorrect size");
   if (size <= 0) { return; }
   if (!(flags & Registered) && !(src.flags & Registered))
   {
      if (h_ptr != src.h_ptr)
      {
         MFEM_ASSERT(h_ptr + size <= src.h_ptr || src.h_ptr + size <= h_ptr,
                     "data overlaps!");
         std::memcpy(h_ptr, src, size*sizeof(T));
      }
      // *this is not registered, so (flags & VALID_HOST) must be true
   }
   else
   {
      MemoryManager::Copy_(h_ptr, src.h_ptr, size*sizeof(T), src.flags, flags);
   }
}

template <typename T>
inline void Memory<T>::CopyFromHost(const T *src, int size)
{
   MFEM_VERIFY(capacity>=size, "Incorrect size");
   if (size <= 0) { return; }
   if (!(flags & Registered))
   {
      if (h_ptr != src)
      {
         MFEM_ASSERT(h_ptr + size <= src || src + size <= h_ptr,
                     "data overlaps!");
         std::memcpy(h_ptr, src, size*sizeof(T));
      }
      // *this is not registered, so (flags & VALID_HOST) must be true
   }
   else
   {
      MemoryManager::CopyFromHost_(h_ptr, src, size*sizeof(T), flags);
   }
}

template <typename T>
inline void Memory<T>::CopyTo(Memory &dest, int size) const
{
   dest.CopyFrom(*this, size);
}

template <typename T>
inline void Memory<T>::CopyToHost(T *dest, int size) const
{
   MFEM_VERIFY(capacity>=size, "Incorrect size");
   if (size <= 0) { return; }
   if (!(flags & Registered))
   {
      if (h_ptr != dest)
      {
         MFEM_ASSERT(h_ptr + size <= dest || dest + size <= h_ptr,
                     "data overlaps!");
         std::memcpy(dest, h_ptr, size*sizeof(T));
      }
   }
   else
   {
      MemoryManager::CopyToHost_(dest, h_ptr, size*sizeof(T), flags);
   }
}


/** @brief Print the state of a Memory object based on its internal flags.
    Useful in a debugger. See also Memory<T>::PrintFlags(). */
extern void MemoryPrintFlags(unsigned flags);


template <typename T>
inline void Memory<T>::PrintFlags() const
{
   MemoryPrintFlags(flags);
}

template <typename T>
inline int Memory<T>::CompareHostAndDevice(int size) const
{
   if (!(flags & VALID_HOST) || !(flags & VALID_DEVICE)) { return 0; }
   return MemoryManager::CompareHostAndDevice_(h_ptr, size*sizeof(T), flags);
}


/// The (single) global memory manager object
extern MFEM_EXPORT MemoryManager mm;

} // namespace mfem

#endif // MFEM_MEM_MANAGER_HPP
