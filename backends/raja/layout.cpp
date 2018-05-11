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
#include "raja.hpp"

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{
  // ***************************************************************************
  const Engine& Layout::RajaEngine() const
  { return *static_cast<const Engine *>(engine.Get()); }

  // ***************************************************************************
  void Layout::Resize(std::size_t new_size)
  {
    size = new_size;
  }

  // ***************************************************************************
  void Layout::Resize(const mfem::Array<std::size_t> &offsets)
  {
    MFEM_ASSERT(offsets.Size() == 2,
                "multiple workers are not supported yet");
    size = offsets.Last();
  }

  // ***************************************************************************
  raja::memory Layout::Alloc(std::size_t bytes) const
  {
    return device::Get().malloc(bytes);
  }

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
