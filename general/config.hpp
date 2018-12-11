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

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * MFEM config class
// *****************************************************************************
class config
{
private:
   bool pa = false;
   bool cuda = false;
   bool occa = false;
   bool sync = false;
   bool nvvp = false;
   int dev;
   int gpu_count;
   CUdevice cuDevice;
   CUcontext cuContext;
   CUstream *cuStream = NULL;
   OccaDevice occaDevice;

private:
   // **************************************************************************
   config() {}
   config(config const&);
   void operator=(config const&);

private:
   // **************************************************************************
   static config& Get()
   {
      static config singleton;
      return singleton;
   }

private:
   // **************************************************************************
   void MfemDeviceSetup(const int device =0);
   void CudaDeviceSetup(const int device =0);
   void OccaDeviceSetup(const CUdevice cu_dev, const CUcontext cu_ctx);

public:
   // **************************************************************************
   static inline void DeviceSetup() { Get().MfemDeviceSetup(); }
   constexpr static inline bool Nvcc() { return NvccCompilerUsed(); }

   static inline bool PA() { return Get().pa; }
   static inline void SetPA(const bool mode) { Get().pa = mode; }

   static inline bool Cuda() { return Get().cuda; }
   static inline void SetCuda(const bool mode) { Get().cuda = mode; }
   static inline CUstream Stream() { return *Get().cuStream; }

   static inline bool Occa() { return Get().occa; }
   static inline void SetOcca(const bool mode) { Get().occa = mode; }
   static inline OccaDevice OccaDevice() { return Get().occaDevice; }
};

// *****************************************************************************
} // mfem

#endif // MFEM_CONFIG_HPP
