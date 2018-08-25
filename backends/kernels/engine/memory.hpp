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

#ifndef MFEM_BACKENDS_KERNELS_MEMORY_HPP
#define MFEM_BACKENDS_KERNELS_MEMORY_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
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

   kernels::device getDevice();

   void copyTo(void*);
   void copyTo(void*, size_t)const;

   void copyFrom(memory&);
   void copyFrom(memory&, size_t)const;

   void copyFrom(const void*);
   void copyFrom(const void*, size_t)const;

   void* ptr() const;

   inline operator double* () { return (double*)data; }

   inline operator const double* () const { return (const double*)data; }

   memory slice(const size_t offset,
                const int bytes = -1) const;

   inline char* operator[](const size_t i)
   {
      return data+i;
   }

   bool operator == (const memory &);
};


} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_MEMORY_HPP
