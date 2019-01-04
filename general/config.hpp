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
   //enum BACKENDS{CUDA, OCCA};
   enum MODES {CPU, GPU};
private:
   MODES mode;
   int dev = 0;
   int ngpu = -1;
   bool pa = false;
   bool cuda = false;
   bool occa = false;
   bool sync = false;
   bool nvvp = false;
   CUdevice cuDevice;
   CUstream *cuStream;
   CUcontext cuContext;
   OccaDevice occaDevice;

private:
   // **************************************************************************
   config(): mode{config::CPU} {}
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
   void MfemDeviceSetup(const int dev =0);
   void CudaDeviceSetup(const int dev =0);
   void OccaDeviceSetup(const CUdevice cu_dev, const CUcontext cu_ctx);

public:
   // **************************************************************************
   constexpr static inline bool usingMM()
   {
#ifdef MFEM_USE_MM
      return true;
#else
      return false;
#endif
   }

   // **************************************************************************
   static inline void enableGpu(const int dev =0) { Get().MfemDeviceSetup(dev); }
   static inline bool gpuEnabled() { return Get().ngpu > 0; }
   static inline bool gpuDisabled() { return Get().ngpu == 0; }
   static inline bool gpuHasBeenEnabled() { return Get().ngpu >= 0; }

   static inline bool usingGpu() { return gpuEnabled() && Get().mode == GPU; }
   static inline bool usingCpu() { return !usingGpu(); }

   static inline void SwitchToGpu() { Get().mode = config::GPU; }
   static inline void SwitchToCpu() { Get().mode = config::CPU; }


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
