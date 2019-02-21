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

#include "../general/okina.hpp"

#include <cassert>

namespace mfem
{

namespace mm
{

MemoryManager& getInstance() {
   static MemoryManager* s_instance = new MemoryManager();
   return *s_instance;
}

// TODO This wraps the d_ptr -- check if this works
// OccaMemory occaPtr(const void *a) {
// void *d_ptr = getInstance().getDevicePtr(a);
// return occaWrapMemory(config::GetOccaDevice(), d_ptr, bytes);
// }

void* ptr(void *a) {
   if (config::usingMM() && config::gpuEnabled() && config::gpuHasBeenEnabled())
   {
      return getInstance().getDevicePtr(a);
   }
   else
   {
      return a;
   }
}

const void* ptr(const void *a) {
   if (config::usingMM() && config::gpuEnabled() && config::gpuHasBeenEnabled())
   {
      return getInstance().getDevicePtr(const_cast<void*>(a));
   }
   else {
      return a;
   }
}

OccaMemory occaPtr(const void *a) {
   if (config::usingMM())
   {
      return getInstance().getOccaPointer(a);
   }
   else
   {
      OccaMemory o_ptr = occaWrapMemory(config::GetOccaDevice(), const_cast<void*>(a), 0);
      return o_ptr;
   }
}

void push(const void *ptr, const std::size_t bytes) {
   if (config::usingMM() && config::gpuEnabled() && config::gpuHasBeenEnabled())
   {
      getInstance().pushData(ptr, bytes);
   }
}

void pull(const void *ptr, const std::size_t bytes) {
   if (config::usingMM() && config::gpuEnabled() && config::gpuHasBeenEnabled())
   {
      getInstance().pullData(ptr, bytes);
   }
}

void memcpy(void *dst, const void *src,
            const std::size_t bytes, const bool async) {
   if (bytes > 0)
   {
      getInstance().copyData(dst, src, bytes, async);
   }
}

} // namespace mm


// ********** DefaultMemoryManager **********

// *****************************************************************************
// * Tests if ptr is a known address
// *****************************************************************************
static bool Known(const DefaultMemoryManager::ledger &maps, const void *ptr)
{
   const DefaultMemoryManager::memory_map::const_iterator found = maps.memories.find(ptr);
   const bool known = found != maps.memories.end();
   if (known) { return true; }
   return false;
}

// *****************************************************************************
// * Looks if ptr is an alias of one memory
// *****************************************************************************
static const void* IsAlias(const DefaultMemoryManager::ledger &maps, const void *ptr)
{
   MFEM_ASSERT(!Known(maps, ptr), "Ptr is an already known address!");
   for (DefaultMemoryManager::memory_map::const_iterator mem = maps.memories.begin();
        mem != maps.memories.end(); mem++)
   {
      const void *b_ptr = mem->first;
      if (b_ptr > ptr) { continue; }
      const void *end = (char*)b_ptr + mem->second.bytes;
      if (ptr < end) { return b_ptr; }
   }
   return NULL;
}

// *****************************************************************************
static const void* InsertAlias(DefaultMemoryManager::ledger &maps,
                               const void *base,
                               const void *ptr)
{
   DefaultMemoryManager::memory &mem = maps.memories.at(base);
   const std::size_t offset = (char *)ptr - (char *)base;
   const DefaultMemoryManager::alias *alias = new DefaultMemoryManager::alias{&mem, offset};
   maps.aliases[ptr] = alias;
   mem.aliases.push_back(alias);
   return ptr;
}

// *****************************************************************************
// * Tests if ptr is an alias address
// *****************************************************************************
static bool Alias(DefaultMemoryManager::ledger &maps, const void *ptr)
{
   const DefaultMemoryManager::alias_map::const_iterator found = maps.aliases.find(ptr);
   const bool alias = found != maps.aliases.end();
   if (alias) { return true; }
   const void *base = IsAlias(maps, ptr);
   if (!base) { return false; }
   InsertAlias(maps, base, ptr);
   return true;
}

// *****************************************************************************
/*static void debugMode(void)
{
   dbg("\033[1K\r%sMM %sHasBeenEnabled %sEnabled %sDisabled \
%sCPU %sGPU %sPA %sCUDA %sOCCA",
       config::usingMM()?"\033[32m":"\033[31m",
       config::gpuHasBeenEnabled()?"\033[32m":"\033[31m",
       config::gpuEnabled()?"\033[32m":"\033[31m",
       config::gpuDisabled()?"\033[32m":"\033[31m",
       config::usingCpu()?"\033[32m":"\033[31m",
       config::usingGpu()?"\033[32m":"\033[31m",
       config::usingPA()?"\033[32m":"\033[31m",
       config::usingCuda()?"\033[32m":"\033[31m",
       config::usingOcca()?"\033[32m":"\033[31m");
}*/

// *****************************************************************************
// * Adds an address
// *****************************************************************************
void DefaultMemoryManager::insertAddress(void *ptr, const std::size_t bytes)
{
   const bool known = Known(maps, ptr);
   MFEM_ASSERT(!known, "Trying to add already present address!");
   dbg("\033[33m%p \033[35m(%ldb)", ptr, bytes);
   maps.memories.emplace(ptr, memory(ptr, bytes));
}

// *****************************************************************************
// * Remove the address from the map, as well as all the address' aliases
// *****************************************************************************
void DefaultMemoryManager::removeAddress(void *ptr)
{
   const bool known = Known(maps, ptr);
   // if (!known) { BUILTIN_TRAP; }
   if (!known) { mfem_error("Trying to remove an unknown address!"); }
   MFEM_ASSERT(known, "Trying to remove an unknown address!");
   const memory &mem = maps.memories.at(ptr);
   dbg("\033[33m %p \033[35m(%ldb)", ptr, mem.bytes);
   for (const alias* const alias : mem.aliases)
   {
      maps.aliases.erase(alias);
      delete alias;
   }
   maps.memories.erase(ptr);
}

// *****************************************************************************
static void* PtrKnown(DefaultMemoryManager::ledger &maps, void *ptr)
{
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const bool host = base.host;
   const bool device = !host;
   const std::size_t bytes = base.bytes;
   const bool gpu = config::usingGpu();
   if (host && !gpu) { return ptr; }
   if (!base.d_ptr) { cuMemAlloc(&base.d_ptr, bytes); }
   if (device &&  gpu) { return base.d_ptr; }
   if (device && !gpu) // Pull
   {
      cuMemcpyDtoH(ptr, base.d_ptr, bytes);
      base.host = true;
      return ptr;
   }
   // Push
   assert(host && gpu);
   cuMemcpyHtoD(base.d_ptr, ptr, bytes);
   base.host = false;
   return base.d_ptr;
}

// *****************************************************************************
static void* PtrAlias(DefaultMemoryManager::ledger &maps, void *ptr)
{
   const bool gpu = config::usingGpu();
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   const DefaultMemoryManager::memory *base = alias->mem;
   const bool host = base->host;
   const bool device = !base->host;
   const std::size_t bytes = base->bytes;
   if (host && !gpu) { return ptr; }
   if (!base->d_ptr) { cuMemAlloc(&alias->mem->d_ptr, bytes); }
   void *a_ptr = (char*)base->d_ptr + alias->offset;
   if (device && gpu) { return a_ptr; }
   if (device && !gpu) // Pull
   {
      assert(base->d_ptr);
      cuMemcpyDtoH(base->h_ptr, base->d_ptr, bytes);
      alias->mem->host = true;
      return ptr;
   }
   // Push
   assert(host && gpu);
   cuMemcpyHtoD(base->d_ptr, base->h_ptr, bytes);
   alias->mem->host = false;
   return a_ptr;
}

// *****************************************************************************
// * Turn an address to the right host or device one
// *****************************************************************************
void* DefaultMemoryManager::getDevicePtr(void *ptr)
{
   if (Known(maps, ptr)) { return PtrKnown(maps, ptr); }
   const bool alias = Alias(maps, ptr);
   // if (!alias) { BUILTIN_TRAP; }
   if (!alias) { mfem_error("DefaultMemoryManager::Ptr"); }
   MFEM_ASSERT(alias, "Unknown address!");
   return PtrAlias(maps, ptr);
}


// *****************************************************************************
static OccaMemory occaMemory(DefaultMemoryManager::ledger &maps, const void *ptr)
{
   OccaDevice occaDevice = config::GetOccaDevice();
   const bool known = Known(maps, ptr);
   // if (!known) { BUILTIN_TRAP; }
   if (!known) { mfem_error("occaMemory"); }
   MFEM_ASSERT(known, "Unknown address!");
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   const std::size_t bytes = base.bytes;
   const bool gpu = config::usingGpu();
   const bool occa = config::usingOcca();
   MFEM_ASSERT(occa, "Using OCCA memory without OCCA mode!");
   if (!base.d_ptr)
   {
      base.host = false; // This address is no more on the host
      if (gpu)
      {
         cuMemAlloc(&base.d_ptr, bytes);
         void *stream = config::Stream();
         cuMemcpyHtoDAsync(base.d_ptr, base.h_ptr, bytes, stream);
      }
      else
      {
         base.o_ptr = occaDeviceMalloc(occaDevice, bytes);
         base.d_ptr = occaMemoryPtr(base.o_ptr);
         occaCopyFrom(base.o_ptr, base.h_ptr);
      }
   }
   if (gpu)
   {
      return occaWrapMemory(occaDevice, base.d_ptr, bytes);
   }
   return base.o_ptr;
}

// *****************************************************************************
OccaMemory DefaultMemoryManager::getOccaPointer(const void *ptr)
{
   return occaMemory(maps, ptr);
}

// *****************************************************************************
static void PushKnown(DefaultMemoryManager::ledger &maps, const void *ptr, const std::size_t bytes)
{
   DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   if (!base.d_ptr) { cuMemAlloc(&base.d_ptr, base.bytes); }
   cuMemcpyHtoD(base.d_ptr, ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PushAlias(const DefaultMemoryManager::ledger &maps, const void *ptr,
                      const std::size_t bytes)
{
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   cuMemcpyHtoD((char*)alias->mem->d_ptr + alias->offset, ptr, bytes);
}

// *****************************************************************************
void DefaultMemoryManager::pushData(const void *ptr, const std::size_t bytes)
{
   if (Known(maps, ptr)) { return PushKnown(maps, ptr, bytes); }
   assert(!config::usingOcca());
   const bool alias = Alias(maps, ptr);
   // if (!alias) { BUILTIN_TRAP; }
   if (!alias) { mfem_error("DefaultMemoryManager::push"); }
   MFEM_ASSERT(alias, "Unknown address!");
   return PushAlias(maps, ptr, bytes);
}

// *****************************************************************************
static void PullKnown(const DefaultMemoryManager::ledger &maps, const void *ptr,
                      const std::size_t bytes)
{
   const DefaultMemoryManager::memory &base = maps.memories.at(ptr);
   cuMemcpyDtoH(base.h_ptr, base.d_ptr, bytes == 0 ? base.bytes : bytes);
}

// *****************************************************************************
static void PullAlias(const DefaultMemoryManager::ledger &maps, const void *ptr,
                      const std::size_t bytes)
{
   const DefaultMemoryManager::alias *alias = maps.aliases.at(ptr);
   cuMemcpyDtoH((void *)ptr, (char*)alias->mem->d_ptr + alias->offset, bytes);
}

// *****************************************************************************
void DefaultMemoryManager::pullData(const void *ptr, const std::size_t bytes)
{
   if (Known(maps, ptr)) { return PullKnown(maps, ptr, bytes); }
   assert(!config::usingOcca());
   const bool alias = Alias(maps, ptr);
   // if (!alias) { BUILTIN_TRAP; }
   if (!alias) { mfem_error("DefaultMemoryManager::pull"); }
   MFEM_ASSERT(alias, "Unknown address!");
   return PullAlias(maps, ptr, bytes);
}

// *****************************************************************************
// __attribute__((unused)) // VS doesn't like this in Appveyor
/*static void Dump(const DefaultMemoryManager::ledger &maps)
{
   if (!getenv("DBG")) { return; }
   const DefaultMemoryManager::memory_map &mem = maps.memories;
   const DefaultMemoryManager::alias_map  &als = maps.aliases;
   std::size_t k = 0;
   for (DefaultMemoryManager::memory_map::const_iterator m = mem.begin(); m != mem.end(); m++)
   {
      const void *h_ptr = m->first;
      assert(h_ptr == m->second.h_ptr);
      const std::size_t bytes = m->second.bytes;
      const void *d_ptr = m->second.d_ptr;
      if (!d_ptr)
      {
         printf("\n[%ld] \033[33m%p \033[35m(%ld)", k, h_ptr, bytes);
      }
      else
      {
         printf("\n[%ld] \033[33m%p \033[35m (%ld) \033[32 -> %p",
                k, h_ptr, bytes, d_ptr);
      }
      fflush(0);
      k++;
   }
   k = 0;
   for (DefaultMemoryManager::alias_map::const_iterator a = als.begin(); a != als.end(); a++)
   {
      const void *ptr = a->first;
      const std::size_t offset = a->second->offset;
      const void *base = a->second->mem->h_ptr;
      printf("\n[%ld] \033[33m%p < (\033[37m%ld) < \033[33m%p",
             k, base, offset, ptr);
      fflush(0);
      k++;
   }
}*/

// *****************************************************************************
void DefaultMemoryManager::copyData(void *dst, const void *src,
                                    const std::size_t bytes, const bool async)
{
   GET_PTR(src);
   GET_PTR(dst);
   if (config::usingCpu())
   {
      std::memcpy(d_dst, d_src, bytes);
   }
   else if (!async)
   {
      cuMemcpyDtoD(d_dst, (void *)d_src, bytes);
   }
   else
   {
      cuMemcpyDtoDAsync(d_dst, (void *)d_src, bytes, config::Stream());
   }
}


// ********** UmpireMemoryManager **********
#if defined(MFEM_USE_UMPIRE)
// Register an address
void UmpireMemoryManager::insertAddress(void *ptr, const std::size_t bytes)
{
}

// Remove an address
void UmpireMemoryManager::removeAddress(void *ptr)
{
}

void* UmpireMemoryManager::getDevicePtr(void *a)
{
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(a);
   void* ptr = rec->m_ptr;

   // Look it up in the map
   MapType::const_iterator iter = m_map.find(ptr);

   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }
   return iter->second;
}

OccaMemory UmpireMemoryManager::getOccaPointer(const void *a)
{
   mfem_error("TBD");
}

void UmpireMemoryManager::pushData(const void *ptr, const std::size_t bytes)
{
   void* host_ptr = const_cast<void*>(ptr);
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(host_ptr);
   void* base_ptr = rec->m_ptr;

   // Get the device pointer from the map offset by the same distance
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }
   std::size_t host_offset = (char*)host_ptr - (char*)base_ptr;
   void *dev_ptr = static_cast<void*>((char*)iter->second + host_offset);

   m_rm.copy(host_ptr, dev_ptr, bytes);
}

void UmpireMemoryManager::pullData(const void *ptr, const std::size_t bytes)
{
   void* host_ptr = const_cast<void*>(ptr);
   // Get the base pointer
   const umpire::util::AllocationRecord* rec = m_rm.findAllocationRecord(host_ptr);
   void* base_ptr = rec->m_ptr;

   // Get the device pointer from the map offset by the same distance
   MapType::const_iterator iter = m_map.find(base_ptr);
   if (iter == m_map.end())
   {
      mfem_error("Could not find an allocation record assocated with address");
   }
   std::size_t host_offset = (char*)host_ptr - (char*)base_ptr;
   void *dev_ptr = static_cast<void*>((char*)iter->second + host_offset);

   m_rm.copy(dev_ptr, host_ptr, bytes);
}

void UmpireMemoryManager::copyData(void *dst, const void *src, std::size_t bytes, const bool async)
{
   mfem_error("TBD");
}

UmpireMemoryManager::UmpireMemoryManager() :
   m_rm(umpire::ResourceManager::getInstance()),
   m_host(m_rm.makeAllocator<umpire::strategy::DynamicPool>("host_pool", m_rm.getAllocator("HOST"))),
   m_device(config::gpuEnabled() ?
            m_rm.makeAllocator<umpire::strategy::DynamicPool>("target_pool", m_rm.getAllocator("DEVICE")) :
            m_rm.makeAllocator<umpire::strategy::DynamicPool>("target_pool", m_rm.getAllocator("HOST"))) {}

#endif


} // namespace mfem
