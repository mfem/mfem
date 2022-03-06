// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_NODAL_INTERP
#define MFEM_LOR_NODAL_INTERP

#include "../../general/array.hpp"
#include "../../linalg/vector.hpp"

namespace mfem
{

template<int D1D, int Q1D>
void NodalInterpolation2D(const int NE,
                          const Vector &localL,
                          Vector &localH,
                          const Array<double> &B);
template<int D1D, int Q1D>
void NodalInterpolation3D(const int NE,
                          const Vector &localL,
                          Vector &localH,
                          const Array<double> &B);

}

#endif
