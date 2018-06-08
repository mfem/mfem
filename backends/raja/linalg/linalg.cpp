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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{

namespace linalg
{

// ***************************************************************************
double dot(const raja::memory vec1, const raja::memory vec2)
{
   const double d = vector_dot(vec1.size()/sizeof(double),vec1,vec2);
   //printf("\n[dot] result=%f", d);fflush(0);
   return d;
}

} // namespace mfem::raja::linalg

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
