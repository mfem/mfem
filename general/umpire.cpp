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

// *****************************************************************************
// * Tests if ptr is a known address
// *****************************************************************************
/*static bool Known(const umpire::ResourceManager& rm,
                  const UmpireMemoryManager::MapType &map, char *ptr)
{
   const umpire::util::AllocationRecord *rec = rm.findAllocationRecord(ptr);
   assert(rec);
   char* base_ptr = static_cast<char*>(rec->m_ptr);
   assert(base_ptr);
   assert(base_ptr==ptr); // no alias
   dbg("\033[33m%p \033[35m", ptr);
   return true;
   //const UmpireMemoryManager::MapType::const_iterator found = map.find(ptr);
   //const bool known = found != map.end();
   //if (known) { return true; }
   //return false;
   }*/

// *****************************************************************************
// * Register an address
// *****************************************************************************
void UmpireMemoryManager::insertAddress(void *ptr, const std::size_t bytes)
{
   //dbg("\033[32m%p \033[35m(%ldb)", ptr, bytes);
   //const bool known = Known(m_rm, m_map, static_cast<char*>(ptr));
   //if (known)
   //{
   //mfem_error("Trying to insert a non-MM pointer!");
//}
   //MFEM_ASSERT(not known, "Trying to add already present address!");
   //m_map.emplace(ptr, ptr);
}

// *****************************************************************************
// * Remove an address
// *****************************************************************************
void UmpireMemoryManager::removeAddress(void *ptr)
{
   //dbg("\033[31m%p \033[35m", ptr);
   //const bool known = Known(m_rm, m_map, static_cast<char*>(ptr));
   //if (not known)
   //{
   //mfem_error("Trying to remove an unknown address!");
//}
   //MFEM_ASSERT(known, "Trying to remove an unknown address!");
   const umpire::util::AllocationRecord *rec = m_rm.findAllocationRecord(ptr);
   char* base_ptr = static_cast<char*>(rec->m_ptr);
   // Look it up in the map
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter != m_map.end())
   {
      // if known, free on the device side
      m_device.deallocate(iter->second.d_ptr);
      // and remove it from the device host/device map
      m_map.erase(base_ptr);
   }
}

// *****************************************************************************
void* UmpireMemoryManager::getPtr(void *ptr)
{
   //dbg("\033[31m%p \033[35m", ptr);
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(ptr);
   assert(rec);
   char* base_ptr = static_cast<char*>(rec->m_ptr);
   assert(base_ptr);
   const std::size_t bytes = rec->m_size;
   
   // Look it up in the map
   MapType::iterator iter = m_map.find(base_ptr);
   const bool known_in_device = iter != m_map.end();

   // Calculate the offset
   const std::size_t offset = static_cast<char*>(ptr) - base_ptr;
   assert(offset==0); // no alias

   // get device pointer, allocate if needed
   char *d_ptr = known_in_device ? iter->second.d_ptr :
      static_cast<char*>(m_device.allocate(rec->m_size));

   // add it in our map
   if (not known_in_device){
      m_map.emplace(base_ptr, d_ptr);
      //dbg("\033[33mnew device @%p => @%p", ptr, d_ptr);
      //dbg("\033[33mnew device offset: %d, bytes: %d", offset, bytes);
   }

   // refresh iter
   iter = m_map.find(base_ptr);
   const bool known = iter != m_map.end();
   assert(known);
   
   const bool host = known_in_device ? iter->second.host : true;
   const bool device = not host;
   const bool gpu = config::usingGpu();

   if (host && !gpu) { return ptr; }
   if (device && gpu) { return d_ptr; }
   if (device && !gpu) // Pull
   {
      iter->second.host = true;
      cuMemcpyDtoH(ptr, d_ptr, bytes);
      return ptr;
   }
   // Push
   cuMemcpyHtoD(d_ptr, ptr, bytes);
   iter->second.host = false;
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

   void* d_ptr = (iter != m_map.end()) ? iter->second.d_ptr : nullptr;

   return occaWrapMemory(config::GetOccaDevice(), d_ptr, rec->m_size);
}

// *****************************************************************************
void UmpireMemoryManager::pushData(const void *ptr, const std::size_t bytes)
{
   dbg("\033[33m ptr %p \033[35m", ptr);
   assert(false);
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
   char* dev_ptr = iter->second.d_ptr + host_offset;

   m_rm.copy(dev_ptr, host_ptr, bytes);
}

// *****************************************************************************
void UmpireMemoryManager::pullData(const void *ptr, const std::size_t bytes)
{
   //dbg("\033[33m ptr %p \033[35m", ptr);
   char* host_ptr = const_cast<char*>(static_cast<const char*>(ptr));
   // Get the base pointer
   const umpire::util::AllocationRecord *rec = m_rm.findAllocationRecord(host_ptr);
   assert(rec);
   char *base_ptr = static_cast<char*>(rec->m_ptr);
   const std::size_t base_bytes = rec->m_size;
   assert(base_ptr);

   // Get the device pointer from the map offset by the same distance
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }
   
   const bool host = iter->second.host;
   if (host) { return; }
   
   std::size_t host_offset = host_ptr - base_ptr;
   assert(host_offset==0);
   char* dev_ptr = iter->second.d_ptr + host_offset;

   cuMemcpyDtoH(host_ptr, dev_ptr, bytes == 0 ? base_bytes : bytes);
   //m_rm.copy(host_ptr, dev_ptr, bytes);
}

// *****************************************************************************
void UmpireMemoryManager::copyData(void *dst, const void *src,
                                   std::size_t bytes, const bool async)
{
   //dbg("\033[33m ptr %p => %p \033[35m", src, dst);
   if (config::usingCpu())
   {
      std::memcpy(dst, src, bytes);
   }
   else
   {
      const void *d_src = mm::ptr(src);
      void *d_dst = mm::ptr(dst);
      if (!async)
      {
         //cuMemcpyDtoD(d_dst, (void *)d_src, bytes);
         //m_rm.copy(dst, const_cast<void*>(src), bytes);
      }
      else
      {
         mfem_warning("Async transfers are not yet implemented. Will block for now.");
      }
      cuMemcpyDtoD(d_dst, (void *)d_src, bytes);
      //m_rm.copy(dst, const_cast<void*>(src), bytes);
   }
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
