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

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
bool mm::Known(const void *adrs)
{
   const auto search = mng->find(adrs);
   const bool present = search != mng->end();
   return present;
}

// *****************************************************************************
// * Add an address only on the host
// *****************************************************************************
void* mm::Insert(void *h_adrs, const size_t size, const size_t size_of_T)
{
   if (not mng) { mng = new mm_t(); }
   const bool present = Known(h_adrs);
   if (present) { BUILTIN_TRAP; }
   MFEM_ASSERT(not present, "[ERROR] Trying to add already present address!");
   mm2dev_t &mm = mng->operator[](h_adrs);
   mm.host = true;
   mm.bytes = size*size_of_T;
   mm.h_adrs = h_adrs;
   mm.d_adrs = NULL;
   return mm.h_adrs;
}

// *****************************************************************************
// * Remove the address from the map
// *****************************************************************************
void *mm::Erase(void *adrs)
{
   const bool present = Known(adrs);
   if (not present) { BUILTIN_TRAP; }
   MFEM_ASSERT(present, "[ERROR] Trying to remove an unknown address!");
   mng->erase(adrs);
   return adrs;
}

// *****************************************************************************
// * Get an address from host or device
// *****************************************************************************
void* mm::Adrs(const void *adrs)
{
   const bool present = Known(adrs);
   if (not present) { BUILTIN_TRAP; }
   MFEM_ASSERT(present, "[ERROR] Trying to convert unknown address!");
   const bool cuda = config::Cuda();
   mm2dev_t &mm = mng->operator[](adrs);
   // Just return asked known host address if not in CUDA mode
   if (mm.host and not cuda) { return (void*) mm.h_adrs; }
   // If it hasn't been seen, alloc it in the device
   if (not mm.d_adrs)
   {
      MFEM_ASSERT(config::Nvcc(),"[ERROR] Trying to run without CUDA support!");
      const size_t bytes = mm.bytes;
      if (bytes>0) { cuMemAlloc(&mm.d_adrs, bytes); }
      void *stream = config::Stream();
      cuMemcpyHtoDAsync(mm.d_adrs, mm.h_adrs, bytes, stream);
      mm.host = false; // This address is no more on the host
   }
   return mm.d_adrs;
}

// *****************************************************************************
memory mm::Memory(const void *adrs)
{
   const bool present = Known(adrs);
   if (not present) { BUILTIN_TRAP; }
   MFEM_ASSERT(present, "[ERROR] Trying to convert unknown address!");
   const bool occa = config::Occa();
   MFEM_ASSERT(occa, "[ERROR] Using OCCA memory without OCCA mode!");
   mm2dev_t &mm = mng->operator[](adrs);
   const bool cuda = config::Cuda();
   const size_t bytes = mm.bytes;
   OccaDevice device = config::OccaDevice();
   if (not mm.d_adrs)
   {
      mm.host = false; // This address is no more on the host
      if (cuda)
      {
         cuMemAlloc(&mm.d_adrs, bytes);
         void *stream = config::Stream();
         cuMemcpyHtoDAsync(mm.d_adrs, mm.h_adrs, bytes, stream);
      }
      else
      {
         mm.o_adrs = occaDeviceMalloc(device, bytes);
         mm.d_adrs = occaMemoryPtr(mm.o_adrs);
         occaCopyFrom(mm.o_adrs, mm.h_adrs);
      }
   }
   if (cuda)
   {
      return occaWrapMemory(device, mm.d_adrs, bytes);
   }
   return mm.o_adrs;
}

// *****************************************************************************
void mm::Push(const void *adrs)
{
   MFEM_ASSERT(Known(adrs), "[ERROR] Trying to push an unknown address!");
   const mm2dev_t &mm = mng->operator[](adrs);
   if (mm.host) { return; }
   cuMemcpyHtoD(mm.d_adrs, mm.h_adrs, mm.bytes);
}

// *****************************************************************************
void mm::Pull(const void *adrs)
{
   MFEM_ASSERT(Known(adrs), "[ERROR] Trying to pull an unknown address!");
   const mm2dev_t &mm = mng->operator[](adrs);
   if (mm.host) { return; }
   if (config::Cuda())
   {
      cuMemcpyDtoH((void*)mm.h_adrs, mm.d_adrs, mm.bytes);
      return;
   }
   if (config::Occa())
   {
      occaCopyTo(Memory(adrs), (void*)mm.h_adrs);
      return;
   }
   MFEM_ASSERT(false, "[ERROR] Should not be there!");
}

// *****************************************************************************
void* mm::memcpy(void *dest, const void *src, size_t bytes)
{
   return mm::D2D(dest, src, bytes, false);
}

// ******************************************************************************
void* mm::H2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kH2D(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2H(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kD2H(dest, src, bytes, async);
}

// *****************************************************************************
void* mm::D2D(void *dest, const void *src, size_t bytes, const bool async)
{
   if (bytes==0) { return dest; }
   const bool cuda = config::Cuda();
   if (not cuda) { return std::memcpy(dest, src, bytes); }
   return mfem::kD2D(dest, src, bytes, async);
}

// *****************************************************************************
} // namespace mfem
