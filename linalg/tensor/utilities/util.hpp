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

#include "../../../general/error.hpp"
#include "../../../general/forall.hpp"

namespace mfem
{

/// Getter for the N-th dimension value
template <int N, int... Dims>
struct get_value_v;

template <int Dim0, int... Dims>
struct get_value_v<0, Dim0, Dims...>
{
   static constexpr int value = Dim0;
};

template <int N, int Dim0, int... Dims>
struct get_value_v<N, Dim0, Dims...>
{
   static constexpr int value = get_value_v<N-1,Dims...>::value;
};

template <int N, int... Dims>
constexpr int get_value = get_value_v<N,Dims...>::value;

/// Get the last value
template <typename T> MFEM_HOST_DEVICE
T GetLast(T first)
{
   return first;
}

template <typename T, typename... Ts> MFEM_HOST_DEVICE
auto GetLast(T first, Ts... rest)
{
   return GetLast(rest...);
}

/// Does the same as sizeof...
template <int... Sizes>
constexpr int rank = sizeof...(Sizes);

/// IfThenElse
template <bool Cond, typename TrueType, typename FalseType>
struct IfThenElse_t
{
   using type = TrueType;
};

template <typename TrueType, typename FalseType>
struct IfThenElse_t<false, TrueType, FalseType>
{
   using type = FalseType;
};

template <bool Cond, typename TrueType, typename FalseType>
using IfThenElse = typename IfThenElse_t<Cond,TrueType,FalseType>::type;

} // mfem namespace

#endif // MFEM_TENSOR_UTIL
