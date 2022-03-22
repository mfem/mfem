// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_TYPES_IMPL
#define MFEM_TENSOR_TYPES_IMPL

#include "pow.hpp"
#include "prod.hpp"

namespace mfem
{

/// 1D
/// Dynamic1dThreadLayout
constexpr int get_Dynamic1dThreadLayout_size(int MaxSize, int Rank)
{
   return Rank>2 ? pow(MaxSize,Rank-1) : 1;
}

/// Static1dThreadTensor
constexpr int get_Static1dThreadTensor_size(int Size0)
{
   return 1;
}

template <typename... Sizes>
constexpr int get_Static1dThreadTensor_size(int Size0, Sizes... sizes)
{
   return prod(sizes...);
}

/// 2D
/// Dynamic2dThreadLayout
constexpr int get_Dynamic2dThreadLayout_size(int MaxSize, int Rank)
{
   return Rank>2 ? pow(MaxSize,Rank-2) : 1;
}

/// Static2dThreadTensor
constexpr int get_Static2dThreadTensor_size(int Size0)
{
   return 1;
}

constexpr int get_Static2dThreadTensor_size(int Size0, int Size1)
{
   return 1;
}

template <typename... Sizes>
constexpr int get_Static2dThreadTensor_size(int Size0, int Size1, Sizes... sizes)
{
   return prod(sizes...);
}

/// 3D
/// Dynamic3dThreadLayout
constexpr int get_Dynamic3dThreadLayout_size(int MaxSize, int Rank)
{
   return Rank>3 ? pow(MaxSize,Rank-3) : 1;
}

/// Static3dThreadTensor
constexpr int get_Static3dThreadTensor_size(int Size0)
{
   return 1;
}

constexpr int get_Static3dThreadTensor_size(int Size0, int Size1)
{
   return 1;
}

constexpr int get_Static3dThreadTensor_size(int Size0, int Size1, int Size2)
{
   return 1;
}

template <typename... Sizes>
constexpr int get_Static3dThreadTensor_size(int Size0, int Size1, int Size2,
                                            Sizes... sizes)
{
   return prod(sizes...);
}

} // namespace mfem

#endif // MFEM_TENSOR_TYPES_IMPL
