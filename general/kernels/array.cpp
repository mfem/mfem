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

#include "../../general/okina.hpp"

namespace mfem
{
namespace kernels
{
namespace array
{

// *****************************************************************************
void Assign(const size_t N, const int *x, int *y)
{
   GET_PTR_T(y,int);
   GET_CONST_PTR_T(x,int);
   MFEM_FORALL(i, N, d_y[i] = d_x[i];);
}


} // namespace array
} // namespace kernels
} // namespace mfem
