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

#include "mem_manager.hpp"

#include <algorithm> // std::max

namespace mfem
{
MemoryType GetMemoryType(MemoryClass mc)
{
   auto &inst = MemoryManager::Instance();
   switch (mc)
   {
      case MemoryClass::HOST:    return inst.GetHostMemoryType();
      case MemoryClass::HOST_32: return MemoryType::HOST_32;
      case MemoryClass::HOST_64: return MemoryType::HOST_64;
      case MemoryClass::DEVICE:  return inst.GetDeviceMemoryType();
      case MemoryClass::MANAGED: return MemoryType::MANAGED;
   }
   MFEM_VERIFY(false,"");
   return MemoryType::HOST;
}


bool MemoryClassContainsType(MemoryClass mc, MemoryType mt)
{
   switch (mc)
   {
      case MemoryClass::HOST:
         return IsHostMemory(mt);
      case MemoryClass::HOST_32:
         return (mt >= MemoryType::HOST_32 && mt < MemoryType::HOST_UMPIRE);
      case MemoryClass::HOST_64:
         return (mt >= MemoryType::HOST_64 && mt < MemoryType::HOST_UMPIRE);
      case MemoryClass::DEVICE:
         return IsDeviceMemory(mt);
      case MemoryClass::MANAGED:
         return (mt >= MemoryType::MANAGED && mt < MemoryType::DEVICE);
   }
   MFEM_ABORT("invalid MemoryClass");
   return false;
}


MemoryClass operator*(MemoryClass mc1, MemoryClass mc2)
{
   //          | HOST     HOST_32  HOST_64  DEVICE   MANAGED
   // ---------+---------------------------------------------
   //  HOST    | HOST     HOST_32  HOST_64  DEVICE   MANAGED
   //  HOST_32 | HOST_32  HOST_32  HOST_64  DEVICE   MANAGED
   //  HOST_64 | HOST_64  HOST_64  HOST_64  DEVICE   MANAGED
   //  DEVICE  | DEVICE   DEVICE   DEVICE   DEVICE   MANAGED
   //  MANAGED | MANAGED  MANAGED  MANAGED  MANAGED  MANAGED

   // Using the enumeration ordering:
   //    HOST < HOST_32 < HOST_64 < DEVICE < MANAGED,
   // the above table is simply: a*b = max(a,b).

   return std::max(mc1, mc2);
}


const char *MemoryTypeName[MemoryTypeSize] =
{
   "host-std",     "temp-host-std",     "host-32",        "temp-host-32",
   "host-64",      "temp-host-64",      "host-debug",     "temp-host-debug",
   "host-umpire",  "temp-host-umpire",  "host-pinned",    "temp-host-pinned",
#if defined(MFEM_USE_CUDA)
   "cuda-uvm",     "temp-cuda-uvm",     "cuda-uvm-umpire", "temp-cuda-uvm-umpire",
   "cuda",         "temp-cuda",
#elif defined(MFEM_USE_HIP)
   "hip-uvm",      "temp-hip-uvm",      "hip-uvm-umpire", "temp-hip-uvm-umpire",
   "hip",          "temp-hip",
#else
   "managed",        "temp-managed",
   "managed-umpire", "temp-managed-umpire",
   "device",         "temp-device",
#endif
   "device-debug", "temp-device-debug",
#if defined(MFEM_USE_CUDA)
   "cuda-umpire",  "temp-cuda-umpire"
#elif defined(MFEM_USE_HIP)
   "hip-umpire",   "temp-hip-umpire"
#else
   "device-umpire",  "temp-device-umpire",
#endif
};

} // namespace mfem
