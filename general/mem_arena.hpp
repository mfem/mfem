// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MEM_ARENA_HPP
#define MFEM_MEM_ARENA_HPP

#include "mem_manager.hpp"
#include "mem_internal.hpp"

#include <forward_list>
#include <vector>

namespace mfem
{

namespace internal
{

/// Chunk of memory used by the arena memory space.
class ArenaChunk
{
   /// The data used as a base pointer.
   Memory data;
   /// The offset (in bytes) of the next available memory in this chunk.
   size_t offset = 0;
   /// How many pointers have been allocated in this chunk.
   size_t ptr_count = 0;
   /// Is the chunk in the front of the list?
   bool front = true;
   /// Whether the memory has been freed (prevent double-free).
   bool dealloced = false;
   /// Pointers allocated in the chunk. Used to track out-of-order deallocation.
   std::vector<void*> ptr_stack;
public:
   /// Create a ArenaChunk with the given @a capacity.
   ArenaChunk(size_t capacity);
   /// Destroy a ArenaChunk (free the memory).
   ~ArenaChunk();
   /// @brief Return the available capacity (i.e. the largest vector that will
   /// fit in this chunk).
   size_t GetAvailableCapacity() const { return data.bytes - offset; }
   /// @brief Returns the original capacity of the chunk.
   ///
   /// If the chunk is not in the front of the list and all of its vectors are
   /// freed, it may deallocate its data, so the capacity becomes zero. The
   /// "original capacity" remains unchained.
   size_t GetCapacity() const { return data.bytes; }
   /// Return the data offset.
   size_t GetOffset() const { return offset; }
   /// Sets whether the chunk is in the front of the list
   void SetFront(bool front_) { front = front_; }
   /// Returns true if this chunk can fit a new vector of size @a n.
   bool HasCapacityFor(size_t n) const { return n <= GetAvailableCapacity(); }
   /// Returns true if this chunk is empty.
   bool IsEmpty() const { return ptr_stack.empty(); }
   /// Clear all deallocated pointers from the top of the stack.
   void ClearDeallocated();
   /// Allocates a buffer of size nbytes, returns the associated pointer.
   void *NewPointer(size_t nbytes);
   /// Return the device pointer associated with host pointer @a h_ptr.
   void *GetDevicePointer(void *h_ptr);
   /// Return the number of live pointers in the chunk.
   size_t GetPointerCount() const { return ptr_count; }
};

class ArenaHostMemorySpace : public HostMemorySpace
{
   /// Bytes to align allocations.
   static constexpr size_t alignment = 256;
   /// Chunks of storage.
   std::forward_list<internal::ArenaChunk> chunks;
   /// @brief Consolidate the chunks (merge consecutive empty chunks), and
   /// ensure that the front chunk has sufficient available capacity for a
   /// buffer of @a requested_size.
   void ConsolidateAndEnsureAvailable(size_t requested_size);
public:
   void Alloc(void **ptr, size_t nbytes) override;
   void Dealloc(Memory &mem) override;
};

class ArenaDeviceMemorySpace : public DeviceMemorySpace
{
public:
   void Alloc(Memory &base) override;
   void Dealloc(Memory &base) override;
   void *HtoD(void *dst, const void *src, size_t bytes) override;
   void *DtoD(void* dst, const void* src, size_t bytes) override;
   void *DtoH(void *dst, const void *src, size_t bytes) override;
};

} // namespace internal

} // namespace mfem

#endif
