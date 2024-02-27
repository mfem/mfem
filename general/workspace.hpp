// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_WORKSPACE_HPP
#define MFEM_WORKSPACE_HPP

#include "../linalg/vector.hpp"
#include <forward_list>

namespace mfem
{

// Forward declaration of internal::WorkspaceChunk
namespace internal { class WorkspaceChunk; }

/// @brief A vector used as a short-lived workspace for temporary calculations,
/// created with Workspace::NewVector.
///
/// A WorkspaceVector is created using the Workspace bump allocator. The
/// allocator can quickly and cheaply create new vectors, so that these vectors
/// can be created and destroyed in loops without incurring the memory
/// allocation and deallocation overhead.
///
/// WorkspaceVector%s should be used only for short-lived temporary storage; for
/// example, they are not intended to be stored as member data in other classes.
class WorkspaceVector : public Vector
{
   // using internal::WorkspaceChunk;
   friend class internal::WorkspaceChunk;

   /// The WorkspaceChunk containing the data for this vector.
   internal::WorkspaceChunk &chunk;

   /// Offset in the chunk.
   const int offset;

   /// Original size allocated.
   const int original_size;

   /// @brief Has this WorkspaceVector been moved from? If so, don't deallocate
   /// from its WorkspaceChunk in the destructor.
   bool moved_from = false;

   /// Private constructor, create with Workspace::NewVector() instead.
   WorkspaceVector(internal::WorkspaceChunk &chunk_, int offset_, int n);

public:
   /// @brief Move constructor. The moved-from WorkspaceVector has @a
   /// size_in_chunk set to zero.
   WorkspaceVector(WorkspaceVector &&other);

   /// No copy constructor.
   WorkspaceVector(const WorkspaceVector &other) = delete;

   /// Copy assignment: copy contents of vector, not metadata.
   WorkspaceVector& operator=(const WorkspaceVector &other)
   {
      Vector::operator=(other);
      return *this;
   }

   /// Cannot move to an existing WorkspaceVector.
   WorkspaceVector& operator=(WorkspaceVector &&other) = delete;

   // All other operator=, inherit from Vector
   using Vector::operator=;

   /// Destructor. Notifies the WorkspaceChunk that this vector has been freed.
   ~WorkspaceVector();
};

namespace internal
{

/// @brief A chunk of storage used to allocate WorkspaceVector%s.
///
/// This is an internal class used by Workspace.
class WorkspaceChunk
{
   /// The data used as a base for the WorkspaceVector%s.
   Vector data;

   /// The offset of currently allocated WorkspaceVector%s in thus chunk.
   int offset = 0;

   /// How many vectors have been allocated in this chunk.
   int vector_count = 0;

   /// Is the vector in the front of the list?
   bool front = true;

   /// The original capacity allocated.
   const int original_capacity;

public:
   /// Create a WorkspaceChunk with the given @a capacity.
   WorkspaceChunk(int capacity);

   /// @brief Return the available capacity (i.e. the largest vector that will
   /// fit in this chunk).
   int GetAvailableCapacity() const { return data.Size() - offset; }

   /// @brief Returns the original capacity of the chunk.
   ///
   /// If the chunk is not in the front of the list and all of its vectors are
   /// freed, it may deallocate its data, so the capacity becomes zero. The
   /// "original capacity" remains unchained.
   int GetOriginalCapacity() const { return original_capacity; }

   /// Return the data offset.
   int GetOffset() const { return offset; }

   /// Sets whether the chunk is in the front of the list
   void SetFront(bool front_) { front = front_; }

   /// Returns true if this chunk can fit a new vector of size @a n.
   bool HasCapacityFor(int n) const { return n <= GetAvailableCapacity(); }

   /// Returns true if this chunk is empty.
   bool IsEmpty() const { return vector_count == 0; }

   /// Note that a vector from this chunk has been deallocated.
   void FreeVector(const WorkspaceVector &v);

   /// Returns the backing data Vector.
   Vector &GetData() { return data; }

   /// Returns a new WorkspaceVector of size @a n.
   WorkspaceVector NewVector(int n);
};

}

/// @brief Storage for temporary, short-lived workspace vectors.
///
/// This class implements a simple bump allocator to quickly allocate and
/// deallocate workspace vectors without incurring the overhead of memory
/// allocation and deallocation.
class Workspace
{
   /// Chunks of storage to hold the vectors.
   std::forward_list<internal::WorkspaceChunk> chunks;

   /// Default constructor, private (singleton class).
   Workspace() = default;

   /// @brief Consolidate the chunks (merge consecutive empty chunks), and
   /// ensure that the front chunk has sufficient available capacity for a
   /// vector of @a requested_size.
   void ConsolidateAndEnsureAvailable(int requested_size);

   /// Return the singleton instance.
   static Workspace &Instance();

public:
   /// Return a new WorkspaceVector of the requested size.
   static WorkspaceVector NewVector(int n);

   /// Ensure that capacity of at least @a n is available for allocations.
   static void Reserve(int n);

   /// Clear all storage. Invalidates any existing WorkspaceVector%s.
   static void Clear();
};

} // namespace mfem

#endif
