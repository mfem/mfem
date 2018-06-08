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
//#include "raja.hpp"

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../engine/device.hpp"
#include "../engine/memory.hpp"
#include <assert.h>
#include "../raja.hpp"

namespace mfem
{

namespace raja
{
  memory::memory(const std::size_t _bytes,
                 const void *src)://mHandle(new memory_v()),
                                  bytes(_bytes),
                                  data(::new char[bytes]){
    assert(src==NULL);
  }
   
   raja::device memory::getDevice(){
     return raja::device();
  }

  size_t memory::size() const {
    return bytes;
  }
 
  void memory::copyFrom(memory &src) {
     memcpy(data,src,bytes);
 }
  
  void memory::copyFrom(const void *src) {
    memcpy(data,src,bytes);
  }
  
  void memory::copyTo(void *dest) {
    memcpy(dest,data,bytes);
  }
    
  void* memory::ptr() const {    
    return (void*)data;
  }
    
  memory memory::slice(const size_t offset,
                       const size_t bytes) const{
    MFEM_ABORT("FIXME");
    return memory();
  }
   
   bool memory::operator == (const memory &m){
      return (ptr() == m.ptr()) && (size() == m.size());
   }

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
