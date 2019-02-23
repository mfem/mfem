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

// **************************** UmpireMemoryManager ****************************

// *****************************************************************************
// * A dirty least significant bit is used in the second field of MapType.
// * It tells if a device address has been pulled on the host.
// *****************************************************************************
static inline char *FlushHostBit(char *ptr){
   return reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(ptr) & ~1ul);
}

static inline char *SetHostBit(char *ptr){
   return reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(ptr) | 1ul);
}

static inline char *ToggleHostBit(char **ptr){
   return *ptr=reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(*ptr) ^ 1ul);
}

static inline bool HasHostBit(const char *ptr){
   return (reinterpret_cast<uintptr_t>(ptr) & 1ul)==1ul;
}

// *****************************************************************************
// * Register an address
// *****************************************************************************
void UmpireMemoryManager::insertAddress(void *ptr, const std::size_t bytes)
{
}

// *****************************************************************************
// * Remove an address
// *****************************************************************************
void UmpireMemoryManager::removeAddress(void *ptr)
{
   const umpire::util::AllocationRecord *rec = m_rm.findAllocationRecord(ptr);
   char* base_ptr = static_cast<char*>(rec->m_ptr);
   // Look it up in the map
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter != m_map.end())
   {
      // if known, free on the device side
      m_device.deallocate(FlushHostBit(iter->second));
      // and remove it from the device host/device map
      m_map.erase(base_ptr);
   }
}

// *****************************************************************************
void* UmpireMemoryManager::getPtr(void *ptr)
{
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(ptr);
   char* base_ptr = static_cast<char*>(rec->m_ptr);
   const std::size_t bytes = rec->m_size;
   
   // Calculate the offset
   const std::size_t offset = static_cast<char*>(ptr) - base_ptr;
   assert(offset==0); // no alias seen yet
   
   // Look it up in the map
   MapType::iterator iter = m_map.find(base_ptr);
   const bool allocated_on_device = iter != m_map.end();

   // Get the device pointer, do the allocation if needed
   if (!allocated_on_device){
      char *d_ptr = static_cast<char*>(m_device.allocate(rec->m_size));
      // Assert we can use the host dirty bit
      MFEM_ASSERT(!(reinterpret_cast<uintptr_t>(d_ptr)%2),
                  "UmpireMemoryManager can not handle the returned address");
      // add it in our map
      m_map.emplace(base_ptr, SetHostBit(d_ptr));
      // refresh iter
      iter = m_map.find(base_ptr);
   }
   char *d_ptr = iter->second;
   
   // Get the states of our config and pointer
   const bool cpu = HasHostBit(d_ptr);
   const bool gpu = !cpu;
   const bool usingGpu = config::usingGpu();
   const bool usingCpu = !usingGpu;

   // CPU mode and pointer, nothing to do
   if (cpu && usingCpu) { return ptr; }

   // GPU mode and pointer, nothing to do
   if (gpu && usingGpu) { return d_ptr; }

   // CPU mode with a GPU pointer => Pull
   if (gpu && usingCpu)
   {
      cuMemcpyDtoH(ptr, d_ptr, bytes);
      ToggleHostBit(&iter->second);
      return ptr;
   }

   // Else push
   assert(cpu && usingGpu);
   d_ptr=ToggleHostBit(&iter->second);
   cuMemcpyHtoD(d_ptr, ptr, bytes);
   return d_ptr;
}

// *****************************************************************************
OccaMemory UmpireMemoryManager::getOccaPtr(const void *a)
{
   // Get the base pointer
   const umpire::util::AllocationRecord* rec =
      m_rm.findAllocationRecord(const_cast<void*>(a));
   char* base_ptr = static_cast<char*>(rec->m_ptr);
   // Look it up in the map
   MapType::const_iterator iter = m_map.find(base_ptr);
   void* d_ptr = (iter != m_map.end()) ? iter->second : nullptr;
   return occaWrapMemory(config::GetOccaDevice(), d_ptr, rec->m_size);
}

// *****************************************************************************
void UmpireMemoryManager::pushData(const void *ptr, const std::size_t bytes)
{
   dbg("\033[33m ptr %p \033[35m", ptr);
   assert(false);
   char* h_ptr = const_cast<char*>(static_cast<const char*>(ptr));
   // Get the base pointer
   const umpire::util::AllocationRecord *rec = m_rm.findAllocationRecord(h_ptr);
   char* base_ptr = static_cast<char*>(rec->m_ptr);
   // Get the device pointer from the map offset by the same distance
   const MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }
   const std::size_t offset = h_ptr - base_ptr;
   char* dev_ptr = iter->second + offset;
   m_rm.copy(dev_ptr, h_ptr, bytes);
}

// *****************************************************************************
void UmpireMemoryManager::pullData(const void *ptr, const std::size_t bytes)
{
   char* h_ptr = const_cast<char*>(static_cast<const char*>(ptr));
   // Get the base pointer
   const umpire::util::AllocationRecord *rec = m_rm.findAllocationRecord(h_ptr);
   char *base_ptr = static_cast<char*>(rec->m_ptr);
   const std::size_t base_bytes = rec->m_size;
   // Get the device pointer from the map offset by the same distance
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }
   const bool host = HasHostBit(iter->second);
   if (host) { return; }
   const std::size_t offset = h_ptr - base_ptr;
   char* dev_ptr = iter->second + offset;
   m_rm.copy(h_ptr, dev_ptr, bytes == 0 ? base_bytes : bytes);
}

// *****************************************************************************
void UmpireMemoryManager::copyData(void *dst, const void *src,
                                   std::size_t bytes, const bool async)
{
   if (async)
   {
      mfem_warning("Async copy are not yet implemented. Will block for now.");
   }
   m_rm.copy(mm::ptr(dst), (void*)mm::ptr(src), bytes);
}

// *****************************************************************************
UmpireMemoryManager::UmpireMemoryManager() :
   m_rm(umpire::ResourceManager::getInstance()),
   m_host(m_rm.makeAllocator<umpire::strategy::DynamicPool>
          ("host_pool", m_rm.getAllocator("HOST"))),
   m_device(m_rm.makeAllocator<umpire::strategy::DynamicPool>
            ("device_pool", m_rm.getAllocator("DEVICE"))) {}

} // namespace mfem

#endif // MFEM_USE_UMPIRE
