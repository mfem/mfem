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

static constexpr int Dynamic = 0;
static constexpr int Error = -1;

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

/// Compute the product of a list of values
template <typename T>
constexpr T prod(T first) {
   return first;
}

template <typename T, typename... D>
constexpr T prod(T first, D... rest) {
   return first*prod(rest...);
}

/// Compute x to the power n
template <typename T>
constexpr T pow(T x, unsigned int n)
{
   return n == 0 ? 1 : x * pow(x, n-1);
}

/// Does the same as sizeof...
template <typename First, typename... Rest>
constexpr unsigned int rank()
{
   return 1 + rank<Rest...>();
}

/// A struct to store a list of integers
template <int... Vs>
struct int_list { };

/// Append two int_list together into another int_list
template <typename, typename>
struct append_t;

template <int... VLs, int... VRs>
struct append_t<int_list<VLs...>,int_list<VRs...>>
{
   using type = int_list<VLs...,VRs...>;
};

template <typename L, typename R>
using append = typename append_t<L,R>::type;

/// Create an int_list containing Val N times
template <int Val, int N>
struct int_repeat_t
{
   using type = append<
                  int_list<Val>,
                  typename int_repeat_t<Val,N-1>::type
                >;
};

template <int Val>
struct int_repeat_t<Val,0>
{
   using type = int_list<>;
};

template <int Val, int N>
using int_repeat = typename int_repeat_t<Val,N>::type;

/// Instatiate TT with T, i.e. TT<T>
template <template<int...> class TT, typename T>
struct instantiate_t;

template <template<int...> class TT, int... Vals>
struct instantiate_t<TT, int_list<Vals...>>
{
   using type = TT<Vals...>;
};

template <template<int...> class TT, typename T>
using instantiate = typename instantiate_t<TT,T>::type;

/// Append an int value to a type list
template<int, typename>
struct append_to_type_seq { };

template<int V, int... Vs, template<int...> class TT>
struct append_to_type_seq<V, TT<Vs...>>
{
   using type = TT<Vs..., V>;
};

/// Append the value V N times
template<int V, int N, template<int...> class TT>
struct repeat
{
   using type = typename
      append_to_type_seq<
         V,
         typename repeat<V, N-1, TT>::type
         >::type;
};

template<int V, template<int...> class TT>
struct repeat<V, 0, TT>
{
   using type = TT<>;
};

/// Append the value V1 N1 times, and then the value V2 N2 times.
template<int V1, int N1, int V2, int N2, template<int...> class TT>
struct rerepeat
{
   using type = typename append_to_type_seq<
      V1,
      typename rerepeat<V1, N1-1, V2, N2, TT>::type
      >::type;
};

template<int V1, int V2, int N2, template<int...> class TT>
struct rerepeat<V1, 0, V2, N2, TT>
{
   using type = typename
      append_to_type_seq<
         V2,
         typename rerepeat<V1, 0, V2, N2-1, TT>::type
         >::type;
};

template<int V1, int V2, template<int...> class TT>
struct rerepeat<V1, 0, V2, 0, TT>
{
   using type = TT<>;
};

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

} // mfem namespace

#endif // MFEM_TENSOR_UTIL
