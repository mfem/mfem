// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mnfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BACKENDS_RAJA_DEVICE_HPP
#define MFEM_BACKENDS_RAJA_DEVICE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include <cstddef>

namespace mfem
{

namespace raja
{
   class memory;
   
   // **************************************************************************
   class device_v {
      device_v();
      virtual ~device_v() = 0;
   };

   // ***************************************************************************
  class device  {
  private:
     mutable device_v *dHandle;
  public:
     device();
     ~device();
     device_v* getDHandle() const;
     bool hasSeparateMemorySpace();
     raja::memory malloc(const std::size_t,const void * = NULL);
  };

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_DEVICE_HPP
