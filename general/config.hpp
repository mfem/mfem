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
   CUstream *cuStream;
   OccaDevice occaDevice;

private:
   // **************************************************************************
   config() {}
   config(config const&);
   void operator=(config const&);
   // **************************************************************************
   static config& Singleton()
   {
      static config singleton;
      return singleton;
   }

private:
   // **************************************************************************
   void cudaDeviceSetup(const int =0);
   void occaDeviceSetup();
   void devSetup(const int =0);

private:
   // **************************************************************************
   inline bool GetOcca() { return occa; }
   inline void SetOcca(const bool mode) { occa = mode; }
   inline OccaDevice Device() { return occaDevice; }

   // **************************************************************************
   inline bool GetCuda() { return cuda; }
   void SetCuda(const bool);

   // **************************************************************************
   inline bool GetPA() { return pa; }
   inline void SetPA(const int mode) { pa = mode; }

   // **************************************************************************
   inline bool Sync(bool toggle=false) { return toggle?sync=!sync:sync; }
   inline bool Nvvp(bool toggle=false) { return toggle?nvvp=!nvvp:nvvp; }

   // **************************************************************************
   inline CUstream GetStream() { return *cuStream; }

public:
   // **************************************************************************
   // * Shortcuts
   // **************************************************************************
   static inline void Setup() { Singleton().devSetup(); }
   constexpr static inline bool Nvcc() { return cuNvcc(); }

   static inline bool PA() { return Singleton().GetPA(); }
   static inline void PA(const bool b) { Singleton().SetPA(b); }

   static inline bool Cuda() { return Singleton().GetCuda(); }
   static inline void Cuda(const bool b) { Singleton().SetCuda(b); }
   static inline CUstream Stream() { return Singleton().GetStream(); }

   static inline bool Occa() { return Singleton().GetOcca(); }
   static inline void Occa(const bool b) { Singleton().SetOcca(b); }
   static inline OccaDevice OccaGetDevice() { return Singleton().Device(); }
};

// *****************************************************************************
} // mfem

#endif // MFEM_CONFIG_HPP
