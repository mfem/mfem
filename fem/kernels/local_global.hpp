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

#ifndef MFEM_FEM_KERNELS_LOCAL_GLOBAL_HPP
#define MFEM_FEM_KERNELS_LOCAL_GLOBAL_HPP

namespace mfem
{
namespace kernels
{
namespace fem
{

// *****************************************************************************
void LocalToGlobal(const int NUM_VDIM,
                   const bool VDIM_ORDERING,
                   const int globalEntries,
                   const int localEntries,
                   const int* __restrict offsets,
                   const int* __restrict indices,
                   const double* __restrict localX,
                   double* __restrict globalX);

} // namespace fem
} // namespace kernels
} // namespace mfem

#endif // MFEM_FEM_KERNELS_LOCAL_GLOBAL_HPP
