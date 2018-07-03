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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "backend.hpp"
#include "engine.hpp"

namespace mfem
{

namespace occa
{

bool Backend::Supports(const std::string &engine_spec) const
{
   // TODO: check if 'engine_spec' is valid OCCA string.
   return true;
}

mfem::Engine *Create(const std::string &engine_spec)
{
   return new Engine(engine_spec);
}

#ifdef MFEM_USE_MPI
mfem::Engine *Create(MPI_Comm comm, const std::string &engine_spec)
{
   return new Engine(comm, engine_spec);
}
#endif


} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
