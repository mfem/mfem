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
   enum BACKENDS{CUDA, OCCA};
   enum MODES{HOST_MODE, DEVICE_MODE};
private:
   bool pa = false;
   MODES mode = HOST_MODE;
   bool cuda = false;
   bool occa = false;
   bool sync = false;
   bool nvvp = false;
   int dev = 0;
   int gpu_count = 0;
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
   constexpr static inline bool usingNvcc() { return usingNvccCompiler(); }

   // **************************************************************************
   
   static inline void setupDevice(const int dev =0) { Get().MfemDeviceSetup(dev); }
   static inline bool usingDevice() { return Get().gpu_count > 0; }
   static inline void SwitchToDevice(){ Get().mode = DEVICE_MODE; }
   static inline void SwitchToHost(){ Get().mode = HOST_MODE; }

   static inline bool usingPA() { return Get().pa; }
   static inline void usePA(const bool mode) { Get().pa = mode; }

   static inline bool usingCuda() { return Get().cuda; }
   static inline void useCuda() { Get().cuda = true; }
   static inline CUstream Stream() { return *Get().cuStream; }

   static inline bool usingOcca() { return Get().occa; }
   static inline void useOcca() { Get().occa = true; }
   static inline OccaDevice GetOccaDevice() { return Get().occaDevice; }
};

// *****************************************************************************
} // mfem

#endif // MFEM_CONFIG_HPP
