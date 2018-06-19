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

#ifndef MFEM_BACKENDS_OMP_BACKEND_HPP
#define MFEM_BACKENDS_OMP_BACKEND_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

// Only the Backend and Engine classes should be exposed through "backend.hpp"
#include "../base/backend.hpp"

namespace mfem
{

namespace omp
{

class Backend : public mfem::Backend
{
public:
   virtual ~Backend();

   virtual bool Supports(const std::string &engine_spec) const;

   virtual mfem::Engine *Create(const std::string &engine_spec);

#ifdef MFEM_USE_MPI
   virtual mfem::Engine *Create(MPI_Comm comm, const std::string &engine_spec);
#endif
};

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_BACKEND_HPP
