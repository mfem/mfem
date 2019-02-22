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

#include "../config/config.hpp"

#if defined(MFEM_USE_UMPIRE)

#include "okina.hpp"
#include "mm.hpp"
#include "umpire.hpp"

namespace mfem
{

// ********** UmpireMemoryManager **********
// Register an address
void UmpireMemoryManager::insertAddress(void *ptr, const std::size_t bytes)
{
}

// Remove an address
void UmpireMemoryManager::removeAddress(void *ptr)
{
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(ptr);
   char* base_ptr = static_cast<char*>(rec->m_ptr);

   // Look it up in the map
   MapType::const_iterator iter = m_map.find(base_ptr);

   if (iter != m_map.end())
   {
      m_device.deallocate(iter->second);
   }
}

void* UmpireMemoryManager::getPtr(void *a)
{
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(a);
   char* base_ptr = static_cast<char*>(rec->m_ptr);

   // Look it up in the map
   MapType::const_iterator iter = m_map.find(base_ptr);

   // Calculate the offset
   const std::size_t offset = static_cast<char*>(a) - base_ptr;

   if (iter != m_map.end())
   {
      return static_cast<void*>(iter->second + offset);
   }
   else
   {
      char* d_ptr = m_map[base_ptr] = static_cast<char*>(m_device.allocate(rec->m_size));
      return d_ptr + offset;
   }
}

OccaMemory UmpireMemoryManager::getOccaPtr(const void *a)
{
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(const_cast<void*>(a));
   char* base_ptr = static_cast<char*>(rec->m_ptr);

   // Look it up in the map
   MapType::const_iterator iter = m_map.find(base_ptr);

   void* d_ptr = (iter != m_map.end()) ? iter->second : nullptr;

   return occaWrapMemory(config::GetOccaDevice(), d_ptr, rec->m_size);
}

void UmpireMemoryManager::pushData(const void *ptr, const std::size_t bytes)
{
   char* host_ptr = const_cast<char*>(static_cast<const char*>(ptr));
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(host_ptr);
   char* base_ptr = static_cast<char*>(rec->m_ptr);

   // Get the device pointer from the map offset by the same distance
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }

   std::size_t host_offset = host_ptr - base_ptr;
   char* dev_ptr = iter->second + host_offset;

   m_rm.copy(dev_ptr, host_ptr, bytes);
}

void UmpireMemoryManager::pullData(const void *ptr, const std::size_t bytes)
{
   char* host_ptr = const_cast<char*>(static_cast<const char*>(ptr));
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(host_ptr);
   char* base_ptr = static_cast<char*>(rec->m_ptr);

   // Get the device pointer from the map offset by the same distance
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }

   std::size_t host_offset = host_ptr - base_ptr;
   char* dev_ptr = iter->second + host_offset;

   m_rm.copy(host_ptr, dev_ptr, bytes);
}

void UmpireMemoryManager::copyData(void *dst, const void *src, std::size_t bytes, const bool async)
{
   if (async)
   {
      mfem_warning("Async transfers are not yet implemented. Will block for now.");
   }
   m_rm.copy(dst, const_cast<void*>(src), bytes);
}

UmpireMemoryManager::UmpireMemoryManager() :
   m_rm(umpire::ResourceManager::getInstance()),
   m_host(m_rm.makeAllocator<umpire::strategy::DynamicPool>("host_pool", m_rm.getAllocator("HOST"))),
   m_device(config::gpuEnabled() ?
            m_rm.makeAllocator<umpire::strategy::DynamicPool>("target_pool", m_rm.getAllocator("DEVICE")) :
            m_rm.makeAllocator<umpire::strategy::DynamicPool>("target_pool", m_rm.getAllocator("HOST"))) {}

} // namespace mfem

#endif // MFEM_USE_UMPIRE
