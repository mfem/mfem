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
   enum MODES {HOST, DEVICE};
private:
   MODES mode;
   int dev = 0;
   int ngpu = -1;
   bool cuda = false;
   bool raja = false;
   bool occa = false;
   bool omp = false;
   bool sync = false;
   bool nvvp = false;
   CUdevice cuDevice;
   CUstream *cuStream;
   CUcontext cuContext;
   OccaDevice occaDevice;

private:
   // **************************************************************************
   config(): mode{config::HOST} {}
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
   void GpuDeviceSetup(const int dev);
   void MfemDeviceSetup(const int dev =0);
   void CudaDeviceSetup(const int dev =0);
   void RajaDeviceSetup(const int dev =0);
   void OccaDeviceSetup(const CUdevice cu_dev, const CUcontext cu_ctx);

public:
   // **************************************************************************
   constexpr static inline bool UsingMM()
   {
#ifdef MFEM_USE_MM
      return true;
#else
      return false;
#endif
   }

   // **************************************************************************
   static inline void EnableDevice(const int dev =0) { Get().MfemDeviceSetup(dev); }
   static inline bool DeviceEnabled() { return Get().ngpu > 0; }
   static inline bool DeviceDisabled() { return Get().ngpu == 0; }
   static inline bool DeviceHasBeenEnabled() { return Get().ngpu >= 0; }

   static inline bool UsingDevice() { return DeviceEnabled() && Get().mode == DEVICE; }
   static inline bool UsingHost() { return !UsingDevice(); }

   static inline void SwitchToDevice() { Get().mode = config::DEVICE; }
   static inline void SwitchToHost() { Get().mode = config::HOST; }

   static inline MODES GetMode() {return Get().mode; };
   static inline MODES DeviceMode() {return config::DEVICE;};
   static inline MODES HostMode() {return config::HOST;};

   static inline bool UsingCuda() { return Get().cuda; }
   static inline void UseCuda() { Get().cuda = true; }
   static inline CUstream Stream() { return *Get().cuStream; }

   static inline bool UsingOmp() { return Get().omp; }
   static inline void UseOmp() { Get().omp = true; }

   static inline bool UsingRaja() { return Get().raja; }
   static inline void UseRaja() { Get().raja = true; }

   static inline bool UsingOcca() { return Get().occa; }
   static inline void UseOcca() { Get().occa = true; }
   static inline OccaDevice GetOccaDevice() { return Get().occaDevice; }

   // **************************************************************************
   ~config();

};

// *****************************************************************************
} // mfem

#endif // MFEM_CONFIG_HPP
