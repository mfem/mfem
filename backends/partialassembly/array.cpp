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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "array.hpp"
#include <cstring>
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif
// #include "cuda.h"
// #include "array.cu"

namespace mfem
{

namespace pa
{

////////////
// CpuArray

HostArray *HostArray::DoClone(bool copy_data, void **buffer,
                      std::size_t item_size) const
{
   HostLayout& lay = dynamic_cast<HostLayout&>(*layout);
   HostArray *new_array = new HostArray(lay, item_size);
   if (copy_data)
   {
      memcpy(new_array->data, data, size);
   }
   if (buffer)
   {
      *buffer = new_array->data;
   }
   return new_array;
}

int HostArray::DoResize(PLayout &new_layout, void **buffer,
                    std::size_t item_size)
{
   HostLayout *lt = static_cast<HostLayout *>(&new_layout);
   layout.Reset(lt); // Reset() checks if the pointer is the same
   int err = ResizeData(lt, item_size);
   if (!err && buffer)
   {
      *buffer = this->data;
   }
   return err;
}

void* HostArray::DoPullData(void *buffer, std::size_t item_size)
{
   //Always on host
   return data;
}

void HostArray::DoFill(const void *value_ptr, std::size_t item_size)
{
   char* this_data = GetTypedData<char>();
   for (std::size_t i = 0; i < size; i += item_size)
   {
      memcpy(this_data + i, value_ptr, item_size);
   }
}

void HostArray::DoPushData(const void *src_buffer, std::size_t item_size)
{
   if (src_buffer) memcpy(data, src_buffer, size);
}

void HostArray::DoAssign(const PArray &src, std::size_t item_size)
{
   // called only when Size() != 0
   const HostArray* src_array = dynamic_cast<const HostArray*>(&src);
   memcpy(data, src_array->data, size);
}

//////////////
// CudaArray

#ifdef __NVCC__

CudaArray* CudaArray::DoClone(bool copy_data, void **buffer,
                              std::size_t item_size) const
{
   CudaLayout& lay = dynamic_cast<CudaLayout&>(*layout);
   CudaArray* new_array = new CudaArray(lay, item_size);
   if (copy_data)
   {
      cudaMemcpy(new_array->data, data, lay.Size()*item_size, cudaMemcpyDeviceToDevice);
   }
   if (buffer)
   {
      *buffer = NULL;
   }
   return new_array;
}

int CudaArray::DoResize(PLayout& new_layout, void** buffer,
                        std::size_t item_size)
{
   CudaLayout* lay = dynamic_cast<CudaLayout*>(&new_layout);
   layout.Reset(lay);
   cudaFree(data);
   int ierr = cudaMalloc(&data, item_size * lay->Size());
   if(buffer)
   {
      *buffer = NULL;
   }
   return ierr;
}

void* CudaArray::DoPullData(void* buffer, std::size_t item_size)
{
   if (buffer)
   {
      cudaMemcpy(buffer, data, layout->Size()*item_size, cudaMemcpyDeviceToHost);
   }
   return buffer;
}


void CudaArray::DoPushData(const void* src_buffer, std::size_t item_size)
{
   if (src_buffer)
   {
      cudaMemcpy(data, src_buffer, layout->Size()*item_size, cudaMemcpyHostToDevice);
   }
}

void CudaArray::DoAssign(const PArray &src, std::size_t item_size)
{
   const CudaArray* src_array = dynamic_cast<const CudaArray*>(&src);
   cudaMemcpy(data, src_array->data, layout->Size()*item_size, cudaMemcpyDeviceToDevice);
}

template <typename T>
__global__ void fillKernel(const T value, const int size, T* data){
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx>=size)
      return;
   data[idx] = value;
}

void CudaArray::DoFill(const void* value_ptr, std::size_t item_size)
{
   const int bsize = 512;
   const int vecsize = layout->Size();
   int gridsize = vecsize/bsize;
   if (bsize*gridsize < vecsize)
        gridsize += 1;
   if(item_size==sizeof(int))
      fillKernel<int><<<gridsize,bsize>>>(*static_cast<const int*>(value_ptr), vecsize, static_cast<int*>(data));
   else if(item_size==sizeof(double))
      fillKernel<double><<<gridsize,bsize>>>(*static_cast<const double*>(value_ptr), vecsize, static_cast<double*>(data));
   else
      mfem_error("This type is not implemented");
}

#endif

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)
