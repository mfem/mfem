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

#ifndef MFEM_BACKENDS_BASE_BACKEND_HPP
#define MFEM_BACKENDS_BASE_BACKEND_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "memory_resource.hpp"
#include "engine.hpp"
#include "array.hpp"
#include "vector.hpp"
#include "fespace.hpp"
#include "pfespace.hpp"
#include "bilinearform.hpp"

#include <string>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

/// TODO
class Backend
{
public:
   /// TODO
   virtual ~Backend() { }

   /// TODO
   virtual bool Supports(const std::string &engine_spec) const = 0;

   /// TODO
   virtual Engine *Create(const std::string &engine_spec) = 0;

#ifdef MFEM_USE_MPI
   /// TODO
   virtual Engine *Create(MPI_Comm comm, const std::string &engine_spec) = 0;
#endif
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_BACKEND_HPP
