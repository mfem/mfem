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

#ifndef MFEM_CONFIG_HPP
#define MFEM_CONFIG_HPP

namespace mfem
{

#ifndef __NVCC__
typedef int CUdevice;
typedef int CUcontext;
typedef void* CUstream;
#endif // __NVCC__

// *****************************************************************************
// * Config
// *****************************************************************************
class config
{
private:
   bool cuda = false;
   bool pa = false;
   bool sync = false;
   bool nvvp = false;
   int dev;
   int gpu_count;
   CUdevice cuDevice;
   CUcontext cuContext;
   CUstream *cuStream;
private:
   config() {}
   config(config const&);
   void operator=(config const&);
public:
   static config& Get()
   {
      static config singleton;
      return singleton;
   }
   // **************************************************************************
private:
   void cuDeviceSetup(const int dev =0);

public:
   // **************************************************************************
   constexpr static inline bool nvcc()
   {
#ifdef __NVCC__
      return true;
#else
      return false;
#endif
   }

   // **************************************************************************
   constexpr static inline bool occa()
   {
#ifdef __OCCA__
      return true;
#else
      return false;
#endif
   }

   // **************************************************************************
   inline bool Cuda() { return cuda; }
   inline void Cuda(const bool mode) { cuda = mode; }

   // **************************************************************************
   inline bool PA() { return pa; }
   inline void PA(const int mode) { pa = mode; }

   // **************************************************************************
   inline bool Sync(bool toggle=false) { return toggle?sync=!sync:sync; }

   // **************************************************************************
   inline bool Nvvp(bool toggle=false) { return toggle?nvvp=!nvvp:nvvp; }

   // **************************************************************************
   void Setup() { cuDeviceSetup(); }

   // **************************************************************************
   inline CUstream Stream() { return *cuStream; }
};

}

#endif // MFEM_CONFIG_HPP
