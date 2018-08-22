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

#include "layout.hpp"
#include "../../general/array.hpp"

namespace mfem
{

namespace pa
{

void HostLayout::Resize(std::size_t new_size)
{
	size = new_size;
}

void HostLayout::Resize(const mfem::Array<std::size_t> &offsets)
{
	MFEM_ASSERT(offsets.Size() == 2,
	            "multiple workers are not supported yet");
	size = offsets.Last();
}

#ifdef __NVCC__

void CudaLayout::Resize(std::size_t new_size)
{
	size = new_size;
}

void CudaLayout::Resize(const mfem::Array<std::size_t> &offsets)
{
	MFEM_ASSERT(offsets.Size() == 2,
	            "multiple workers are not supported yet");
	size = offsets.Last();
}

#endif

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)
