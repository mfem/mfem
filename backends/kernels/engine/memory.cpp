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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// **************************************************************************
memory::memory(const std::size_t _bytes, const void *src):
   bytes(_bytes),
   data((char*)mfem::kernels::kmalloc<char>::operator new (bytes))
{
   nvtx_push();
   if (src)
   {
      data=(char*)src;
   }
   nvtx_pop();
}

// **************************************************************************
kernels::device memory::getDevice()
{
   return kernels::device();
}

// **************************************************************************
size_t memory::size() const
{
   return bytes;
}

// *****************************************************************************
void memory::copyFrom(memory &src, size_t b) const
{
   nvtx_push();
   //memcpy(data,src,b);
   mfem::kernels::kmemcpy::rHtoD(data,src,b);
   nvtx_pop();
}
void memory::copyFrom(memory &src)
{
   nvtx_push();
   //memcpy(data,src,bytes);
   mfem::kernels::kmemcpy::rHtoD(data,src,bytes);
   nvtx_pop();
}

// *****************************************************************************
void memory::copyFrom(const void *src, size_t b) const
{
   nvtx_push();
   //memcpy(data,src,b);
   mfem::kernels::kmemcpy::rHtoD(data,src,b);
   nvtx_pop();
}
void memory::copyFrom(const void *src)
{
   nvtx_push();
   //memcpy(data,src,bytes);
   assert(data);
   assert(src);
   mfem::kernels::kmemcpy::rHtoD(data,src,bytes);
   nvtx_pop();
}

// *****************************************************************************
void memory::copyTo(void *dest, size_t b) const
{
   nvtx_push();
   //memcpy(dest,data,b);
   mfem::kernels::kmemcpy::rDtoH(dest,data,b);
   nvtx_pop();
}
void memory::copyTo(void *dest)
{
   nvtx_push();
   //memcpy(dest,data,bytes);
   mfem::kernels::kmemcpy::rDtoH(dest,data,bytes);
   nvtx_pop();
}

// *****************************************************************************
void* memory::ptr() const
{
   return (void*)data;
}

// *****************************************************************************
memory memory::slice(const size_t offset,
                     const int bytes) const
{
   nvtx_push();
   assert(bytes>0);
   memory m = memory(bytes,NULL);
   dbg("\n\033[31;1;7m[memory::slice] rDtoD!");
   mfem::kernels::kmemcpy::rDtoD(m.data,data+offset,bytes);
   nvtx_pop();
   return m;
}

// *****************************************************************************
bool memory::operator == (const memory &m)
{
   nvtx_push();
   nvtx_pop();
   return (ptr() == m.ptr()) && (size() == m.size());
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
