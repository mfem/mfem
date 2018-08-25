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

#include "backend.hpp"
#include "engine.hpp"

namespace mfem
{

namespace pa
{

bool Backend::Supports(const std::string &engine_spec) const
{
	return true;
}

mfem::Engine *Backend::Create(const std::string &engine_spec)
{
	mfem::Engine* engine = nullptr;
	if (engine_spec=="Host")
	{
		engine = new PAEngine<Host>(engine_spec);
	}
	#ifdef __NVCC__
	else if (engine_spec=="CudaDevice")
	{
		engine = new PAEngine<CudaDevice>(engine_spec);
	}
	#endif
	return engine;
}

#ifdef MFEM_USE_MPI
mfem::Engine *Backend::Create(MPI_Comm comm, const std::string &engine_spec)
{
	mfem::Engine* engine = nullptr;
	if (engine_spec=="Host")
	{
		engine = new PAEngine<Host>(comm, engine_spec);
	}
	#ifdef __NVCC__
	else if (engine_spec=="CudaDevice")
	{
		engine = new PAEngine<CudaDevice>(comm, engine_spec);
	}
	#endif
	return engine;
}
#endif


} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)