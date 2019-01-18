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
using namespace std;

namespace mfem
{
namespace kernels
{
namespace mesh
{

// *****************************************************************************
const __device__ __constant__ double quad_children_init[2*4*4] =
{
   0.0,0.0, 0.5,0.0, 0.5,0.5, 0.0,0.5,
   0.5,0.0, 1.0,0.0, 1.0,0.5, 0.5,0.5,
   0.5,0.5, 1.0,0.5, 1.0,1.0, 0.5,1.0,
   0.0,0.5, 0.5,0.5, 0.5,1.0, 0.0,1.0
};

// *****************************************************************************
void QuadChildren(double *data)
{
   GET_ADRS(data);
   const double *d_quad_children_init = quad_children_init;
   const size_t N = 2*4*4;
   MFEM_FORALL(i, N,
   {
      d_data[i] = d_quad_children_init[i];
   });
}

} // namespace mesh
} // namespace kernels
} // namespace mfem
