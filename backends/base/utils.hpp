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

#ifndef MFEM_BACKENDS_BASE_UTILS_HPP
#define MFEM_BACKENDS_BASE_UTILS_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "../../general/error.hpp"

namespace mfem
{

namespace util
{

//
// Inline methods
//

/// TODO: doxygen
template <typename derived_t, typename base_t>
inline derived_t *As(base_t *base_obj)
{
   MFEM_ASSERT(dynamic_cast<derived_t*>(base_obj) != NULL,
               "invalid object type");
   return static_cast<derived_t*>(base_obj);
}

/// TODO: doxygen
template <typename derived_t, typename base_t>
inline derived_t *Is(base_t *base_obj)
{
   return dynamic_cast<derived_t*>(base_obj);
}

} // namespace mfem::util

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_UTILS_HPP
