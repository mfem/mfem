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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "fespace.hpp"

namespace mfem
{

namespace omp
{

FiniteElementSpace::FiniteElementSpace(const Engine &e,
                                       mfem::FiniteElementSpace &fespace)
   : PFiniteElementSpace(e, fespace)
{
   lsize = 0;
   for (int e = 0; e < fespace.GetNE(); e++) { lsize += fespace.GetFE(e)->GetDof(); }
}

std::size_t FiniteElementSpace::GetLocalVSize() const { return lsize; }

/// Convert an E vector to L vector
void FiniteElementSpace::ToLVector(const Vector &e_vector, Vector &l_vector) const
{
   MFEM_ABORT("FIXME");
}

/// Covert an L vector to E vector
void FiniteElementSpace::ToEVector(const Vector &l_vector, Vector &e_vector) const
{
   MFEM_ABORT("FIXME");
}


} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
