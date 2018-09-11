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
void Layout::Resize(std::size_t new_size)
{
   push();
   dbg("size=%d, new_size=%d",size,new_size);
   size = new_size;
   pop();
}

// **************************************************************************
void Layout::Resize(const mfem::Array<std::size_t> &offsets)
{
   push();
   dbg("Array");
   MFEM_ASSERT(offsets.Size() == 2,
               "multiple workers are not supported yet");
   assert(offsets.Size() == 2);
   size = offsets.Last();
   pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
