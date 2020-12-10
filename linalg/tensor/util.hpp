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

#ifndef MFEM_TENSOR_UTIL
#define MFEM_TENSOR_UTIL

namespace mfem
{
// Getter for the N-th dimension value
template <int N, int... Dims>
struct Dim;

template <int Dim0, int... Dims>
struct Dim<0, Dim0, Dims...>
{
   static constexpr int val = Dim0;
};
template <int N, int Dim0, int... Dims>
struct Dim<N, Dim0, Dims...>
{
   static constexpr int val = Dim<N-1,Dims...>::val;
};

// Compute the product of a list of values
template <typename T>
constexpr T prod(T first) {
   return first;
}

template <typename T, typename... D>
constexpr T prod(T first, D... rest) {
   return first*prod(rest...);
}

template <typename T>
constexpr T pow(T x, unsigned int n)
{
   return n == 0 ? 1 : x * pow(x, n-1);
}

template <typename First, typename... Rest>
constexpr unsigned int rank()
{
   return 1 + rank<Rest...>();
}

} // mfem namespace

#endif // MFEM_TENSOR_UTIL
