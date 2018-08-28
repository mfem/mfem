// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// *************************************************************************
void* kmemcpy::rHtoH(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   //push();
   dbg("=\033[m");
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
   std::memcpy(dest,src,bytes);
   //pop();
   return dest;
}

// *************************************************************************
void* kmemcpy::rHtoD(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   //push();
   dbg(">\033[m");
   if (bytes==0){
      //pop();
      return dest;
   }
   assert(src); assert(dest);
   if (!config::Get().Cuda()) {
      //pop();
      return std::memcpy(dest,src,bytes);
   }
#ifdef __NVCC__
   if (!config::Get().Uvm())
   {
      checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)dest,src,bytes));
   }
   else { checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes)); }
#endif
   //pop();
   return dest;
}

// ***************************************************************************
void* kmemcpy::rDtoH(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   //push();
   dbg("<\033[m");
   if (bytes==0) { /*pop();*/ return dest; }
   assert(src); assert(dest);
   if (!config::Get().Cuda()) { /*pop();*/ return std::memcpy(dest,src,bytes); }
#ifdef __NVCC__
   if (!config::Get().Uvm())
   {
      checkCudaErrors(cuMemcpyDtoH(dest,(CUdeviceptr)src,bytes));
   }
   else { checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes)); }
#endif
   //pop();
   return dest;
}

// ***************************************************************************
void* kmemcpy::rDtoD(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   //push();
   dbg("=\033[m");
   if (bytes==0) { /*pop();*/ return dest; }
   assert(src); assert(dest);
   if (!config::Get().Cuda()) { /*pop();*/ return std::memcpy(dest,src,bytes); }
#ifdef __NVCC__
   if (!config::Get().Uvm())
   {
      if (!async)
      {
         checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
      }
      else
      {
         const CUstream s = *config::Get().Stream();
         checkCudaErrors(cuMemcpyDtoDAsync((CUdeviceptr)dest,(CUdeviceptr)src,bytes,s));
      }
   }
   else { checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes)); }
#endif
   //pop();
   return dest;
}

} // kernels

} // mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
