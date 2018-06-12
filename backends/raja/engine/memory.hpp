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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{
class device;

// ***************************************************************************
class memory
{
public:
   std::size_t bytes;
   char *data;
public:
   memory(const std::size_t =0, const void* =NULL);

   size_t size() const;

   raja::device getDevice();

   void copyTo(void *dest);

   void copyFrom(memory &src);

   void copyFrom(const void*);

   void* ptr() const;

   inline operator double* () { return (double*)data; }

   inline operator const double* () const { return (const double*)data; }

   memory slice(const size_t offset,
                const size_t bytes = -1) const;

   inline char* operator[](const size_t i)
   {
      return data+i;
   }

   bool operator == (const memory &);
};


} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_MEMORY_HPP
