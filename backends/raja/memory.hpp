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
#ifndef MFEM_BACKENDS_RAJA_MEMORY_HPP
#define MFEM_BACKENDS_RAJA_MEMORY_HPP

namespace mfem
{

namespace raja
{
  
  class device;
  class device_v;
  
  
  class memory
  {
  private:
    size_t memory_size;
    device *dev;
  public:
 
    size_t size() const;
  
    void copyTo(void *dest);
  
    void copyFrom(memory &src);
  
    void copyFrom(const void*);
  
    device getDevice() const;
  
    void* ptr();

    device_v* getDHandle() const;
  
    memory slice(const size_t offset,
                 const size_t bytes = -1) const;

};

} // namespace mfem::raja

} // namespace mfem

#endif // MFEM_BACKENDS_RAJA_MEMORY_HPP
