// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <chrono>
#include <vector>

#include <map>

// #define MFEM_ENABLE_MEM_BENCH
// #define MFEM_ENABLE_MEM_OP_DEBUG

#ifdef MFEM_ENABLE_MEM_OP_DEBUG
#define MFEM_MEM_OP_DEBUG_ADD(OP_IDX, START, STOP, MSG)                        \
   mfem::internal::mem_op_debug_add(OP_IDX, START, STOP) << MSG << std::endl
#define MFEM_MEM_OP_DEBUG_REMOVE(OP_IDX, START, MSG)                           \
   mfem::internal::mem_op_debug_remove(OP_IDX, START) << MSG << std::endl
#define MFEM_MEM_OP_DEBUG_REMOVE2(OP_IDX, START, STOP, MSG)                    \
   mfem::internal::mem_op_debug_remove(OP_IDX, START, STOP) << MSG << std::endl

#define MFEM_MEM_OP_DEBUG(OP_IDX, MSG)                                         \
   mfem::internal::mem_op_debug(OP_IDX, 0) << MSG << std::endl

#define MFEM_MEM_OP_DEBUG_SYNC_ALIAS(OP_IDX, ASTART, BSTART, NBYTES)           \
   mfem::internal::mem_op_debug_sync_alias(OP_IDX, ASTART, BSTART, NBYTES)
#define MFEM_MEM_OP_DEBUG_USE(OP_IDX, START, STOP, MSG)                        \
   mfem::internal::mem_op_debug_use(OP_IDX, START, STOP) << MSG << std::endl

#define MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(OP_IDX, SRC_START, DST_START, NBYTES, \
                                         MSG, src_loc, dst_loc)                \
   mfem::internal::mem_op_debug_batch_mem_copy(OP_IDX, SRC_START, DST_START,   \
                                               NBYTES, src_loc, dst_loc)       \
      << MSG << std::endl
#else
#define MFEM_MEM_OP_DEBUG_ADD(OP_IDX, START, STOP, MSG)
#define MFEM_MEM_OP_DEBUG_REMOVE(OP_IDX, START, MSG)
#define MFEM_MEM_OP_DEBUG_REMOVE2(OP_IDX, START, STOP, MSG)
#define MFEM_MEM_OP_DEBUG(OP_IDX, MSG)
#define MFEM_MEM_OP_DEBUG_SYNC_ALIAS(OP_IDX, ASTART, BSTART, NBYTES)
#define MFEM_MEM_OP_DEBUG_USE(OP_IDX, START, STOP, MSG)
#define MFEM_MEM_OP_DEBUG_BATCH_MEM_COPY(OP_IDX, SRC_START, DST_START, NBYTES, \
                                         MSG, src_loc, dst_loc)
#endif

#ifdef MFEM_ENABLE_MEM_BENCH
#define MFEM_MEM_OP_BENCH_SCOPE_NAME(base, counter)base##counter
#define MFEM_MEM_OP_BENCH_SCOPE(OP_IDX, DO_SYNC)                               \
   mfem::internal::ScopeBench MFEM_MEM_OP_BENCH_SCOPE_NAME(                    \
      mem_op_bench_scope_var, __LINE__)(OP_IDX, DO_SYNC)
#else
#define MFEM_MEM_OP_BENCH_SCOPE(OP_IDX, DO_SYNC)
#endif

namespace mfem
{

// Implementation of MFEM's lightweight device/host memory manager designed to
// work seamlessly with the OCCA, RAJA, and other kernels supported by MFEM.

/// Memory types supported by MFEM.
enum class MemoryType
{
   HOST,           ///< Host memory; using new[] and delete[]
   TEMP_HOST,
   HOST_32,        ///< Host memory; aligned at 32 bytes
   TEMP_HOST_32,
   HOST_64,        ///< Host memory; aligned at 64 bytes
   TEMP_HOST_64,
   HOST_DEBUG,     ///< Host memory; allocated from a "host-debug" pool
   TEMP_HOST_DEBUG,
   HOST_UMPIRE,    /**< Host memory; using an Umpire allocator which can be set
                        with MemoryManager::SetUmpireHostAllocatorName */
   TEMP_HOST_UMPIRE, /** Currently maps to HOST_UMPIRE */
   HOST_PINNED,    ///< Host memory: pinned (page-locked)
   TEMP_HOST_PINNED,
   MANAGED,        /**< Managed memory; using CUDA or HIP *MallocManaged
                        and *Free */
   TEMP_MANAGED,
   MANAGED_UMPIRE, /**< Managed memory; using an Umpire allocator which can be
                        set with MemoryManager::SetUmpireManagedAllocatorName */
   TEMP_MANAGED_UMPIRE, /**< Managed memory; using an Umpire allocator which can be set
                             with MemoryManager::SetUmpireTempManagedAllocatorName */
   DEVICE,         ///< Device memory; using CUDA or HIP *Malloc and *Free
   TEMP_DEVICE,
   DEVICE_DEBUG,   /**< Pseudo-device memory; allocated on host from a
                        "device-debug" pool */
   TEMP_DEVICE_DEBUG,
   DEVICE_UMPIRE,  /**< Device memory; using an Umpire allocator which can be
                        set with MemoryManager::SetUmpireDeviceAllocatorName */
   TEMP_DEVICE_UMPIRE, /** Currently maps to DEVICE_UMPIRE_2 */
   DEVICE_UMPIRE_2 = TEMP_DEVICE_UMPIRE, /**< Device memory; using a second Umpire allocator
            settable with MemoryManager::SetUmpireDevice2AllocatorName */
   TEMP_DEVICE_UMPIRE_2 = TEMP_DEVICE_UMPIRE, /** Currently maps to DEVICE_UMPIRE_2 */
   SIZE,           ///< Number of host and device memory types

   PRESERVE,       /**< Pseudo-MemoryType used as default value for MemoryType
                        parameters to request preservation of existing
                        MemoryType, e.g. in copy constructors. */
   DEFAULT         /**< Pseudo-MemoryType used as default value for MemoryType
                        parameters to request the use of the default host or
                        device MemoryType. */
};

#ifdef MFEM_ENABLE_MEM_BENCH
namespace internal
{
struct BenchTimer
{
   std::chrono::high_resolution_clock timer;
   std::vector<size_t> call_counts;
   std::vector<std::chrono::high_resolution_clock::time_point> start_points;
   std::vector<std::chrono::high_resolution_clock::duration> durations;
   std::chrono::high_resolution_clock::time_point glob_start;

   static BenchTimer &Instance();

   BenchTimer();
   ~BenchTimer();
};

struct ScopeBench
{
   size_t idx;
   bool sync;
   ScopeBench(size_t i, bool do_sync);
   ~ScopeBench();
};
} // namespace internal
#endif

#ifdef MFEM_ENABLE_MEM_OP_DEBUG
namespace internal
{
struct mem_op_tracker
{
   using map_type = std::map<std::pair<const void *, const void *>,
         std::pair<size_t, size_t>>;
   map_type allocations;

   using key_type = typename map_type::iterator;

   size_t counter = 0;

   std::pair<size_t, size_t> add_allocation(const void *start,
                                            const void *stop);

   key_type find_containing(const void *start, const void *stop);

   key_type find_containing(const void *start);

   key_type find_allocation(const void *start, const void *stop);

   key_type find_allocation(const void *start);

   std::pair<size_t, size_t> remove_allocation(const void *start,
                                               const void *stop);

   std::pair<size_t, size_t> remove_allocation(const void *start);

   static mem_op_tracker &instance()
   {
      static mem_op_tracker res;
      return res;
   }
};

std::ostream &mem_op_debug(size_t idx, int);

size_t mem_op_debug(size_t idx);
std::ostream &mem_op_debug_add(size_t op_idx, const void *start,
                               const void *stop);
std::ostream &mem_op_debug_remove(size_t op_idx, const void *start);
std::ostream &mem_op_debug_remove(size_t op_idx, const void *start,
                                  const void *stop);
std::ostream &mem_op_debug_sync_alias(size_t op_idx, const void *astart,
                                      const void *bstart, size_t nbytes);
std::ostream &mem_op_debug_use(size_t op_idx, const void *start);
std::ostream &mem_op_debug_use(size_t op_idx, const void *start,
                               const void *stop);
std::ostream &mem_op_debug_batch_mem_copy(size_t op_idx, const void *src_start,
                                          const void *dst_start, size_t nbytes,
                                          MemoryType src_loc,
                                          MemoryType dst_loc);
std::string mem_op_debug_copy_type(MemoryType src_loc, MemoryType dst_loc);
} // namespace internal
#endif

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
inline bool IsHostMemory(MemoryType mt)
{
   return mt <= MemoryType::TEMP_MANAGED_UMPIRE;
}

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

#ifdef MFEM_USE_MPI

#if MFEM_HYPRE_VERSION < 21400
#define HYPRE_MEMORY_DEVICE (0)
#define HYPRE_MEMORY_HOST (1)
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

} // namespace mfem

#include "resource_manager.hpp"

#endif // MFEM_MEM_MANAGER_HPP
