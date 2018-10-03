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
MFEM_NAMESPACE

#ifndef __NVCC__
typedef int CUdevice;
typedef int CUcontext;
typedef int CUstream;
#endif // __NVCC__

// *****************************************************************************
// * Config
// *****************************************************************************
class config{
private:
   bool cuda = false;
   bool uvm = false;
   int dev;
   int gpu_count;
   CUdevice cuDevice;
   CUcontext cuContext;
   CUstream *hStream;
private:
   config(){}
   config(config const&);
   void operator=(config const&);
public:
   static config& Get(){
      static config singleton;
      return singleton;
   }
   // **************************************************************************
private:
   void cuDeviceSetup(const int dev =0);

public:
   // **************************************************************************
   inline bool Cuda() { return cuda; }

   // **************************************************************************
   inline void Cuda(const bool flag) { cuda = flag; }

   // **************************************************************************
   inline bool Uvm() { return uvm; }

   // **************************************************************************
   void Setup();
   
   // **************************************************************************
   inline CUstream *Stream() { return hStream; }
};

// *****************************************************************************
MFEM_NAMESPACE_END

#endif // MFEM_CONFIG_HPP
