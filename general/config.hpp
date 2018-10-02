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
// * Config
// *****************************************************************************
class config{
private:
   bool cuda = false;
   bool uvm = false;
#ifdef __NVCC__
   int dev;
   int gpu_count;
   CUdevice cuDevice;
   CUcontext cuContext;
   CUstream *hStream;
#endif
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
   void Init(){
#ifdef __NVCC__
      gpu_count=0;    
      checkCudaErrors(cudaGetDeviceCount(&gpu_count));
      assert(gpu_count>0);   
      cuInit(0);
      dev = 0;
      cuDeviceGet(&cuDevice,dev); 
      cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
      hStream=new CUstream;
      cuStreamCreate(hStream, CU_STREAM_DEFAULT);
#endif
   }
   // **************************************************************************
   void Init(int argc, char *argv[]){
#ifdef __NVCC__
      gpu_count=0;    
      checkCudaErrors(cudaGetDeviceCount(&gpu_count));
      assert(gpu_count>0);   
      cuInit(0);
      dev = findCudaDevice(argc, (const char **)argv);
      cuDeviceGet(&cuDevice,dev); 
      cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
      hStream=new CUstream;
      cuStreamCreate(hStream, CU_STREAM_DEFAULT);
#endif
   }
   // **************************************************************************
   inline bool Cuda(const bool flag=false) {
      if (flag) {
         dbg("\033[32;7mSetting CUDA mode!");
         // Still move data to GPU, could be 'discretized'
         mfem::mm::Get().Cuda();
         cuda = true;
      }
      return cuda;
   }
   // **************************************************************************
   inline bool Uvm(const bool flag=false) {
      if (flag) {
         dbg("\033[32;7mSetting UVM mode!");
         uvm = true;
      }
      return uvm;
   }
#ifdef __NVCC__
   inline CUstream *Stream() { return hStream; }
#endif
};

#endif // MFEM_CONFIG_HPP
