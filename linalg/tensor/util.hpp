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
struct append_to_type_seq_t { };

template<int V, int... Vs, template<int...> class TT>
struct append_to_type_seq_t<V, TT<Vs...>>
{
   using type = TT<Vs..., V>;
};

template<int Val, typename T>
using append_to_type_seq = typename append_to_type_seq_t<Val, T>::type;

/// Append the value V N times
template<int V, int N, template<int...> class TT>
struct repeat
{
   using type = append_to_type_seq<V, typename repeat<V, N-1, TT>::type>;
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
   using type = append_to_type_seq<
      V1,
      typename rerepeat<V1, N1-1, V2, N2, TT>::type>;
};

template<int V1, int V2, int N2, template<int...> class TT>
struct rerepeat<V1, 0, V2, N2, TT>
{
   using type = append_to_type_seq<
      V2,
      typename rerepeat<V1, 0, V2, N2-1, TT>::type
      >;
};

template<int V1, int V2, template<int...> class TT>
struct rerepeat<V1, 0, V2, 0, TT>
{
   using type = TT<>;
};

/// Remove the Nth value in an int_list
template <int N, typename Tail, typename Head = int_list<>>
struct remove_t;

template <int N, int Val, int... TailVals, int... HeadVals>
struct remove_t<N, int_list<Val,TailVals...>, int_list<HeadVals...>>
{
   using type = typename remove_t<N-1,
                                  int_list<TailVals...>,
                                  int_list<HeadVals...,Val>>::type;
};

template <int Val, int... TailVals, int... HeadVals>
struct remove_t<0, int_list<Val,TailVals...>, int_list<HeadVals...>>
{
   using type = append< int_list<HeadVals...>, int_list<TailVals...> >;
};

template <int N, typename Vals>
using remove = typename remove_t<N, Vals>::type;

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

/// Layout utility classes

/// A Class to compute the real index from the multi-indices of a DynamicLayout
template <int Dim, int N = 1>
struct DynamicLayoutIndex
{
   template <typename... Args> MFEM_HOST_DEVICE inline
   static int eval(const int* sizes, int first, Args... args)
   {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      MFEM_ASSERT(first<sizes[N-1],"Trying to access out of boundary.");
#endif
      return first + sizes[N - 1] * DynamicLayoutIndex<Dim,N+1>::eval(sizes, args...);
   }
};

// Terminal case
template <int Dim>
struct DynamicLayoutIndex<Dim, Dim>
{
   MFEM_HOST_DEVICE inline
   static int eval(const int* sizes, int first)
   {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      MFEM_ASSERT(first<sizes[Dim-1],"Trying to access out of boundary.");
#endif
      return first;
   }
};

/// A class to initialize the size of a DynamicLayout
template <int Dim, int N = 1>
struct InitDynamicLayout
{
   template <typename... Args> MFEM_HOST_DEVICE inline
   static void result(int* sizes, int first, Args... args)
   {
      sizes[N - 1] = first;
      InitDynamicLayout<Dim,N+1>::result(sizes, args...);
   }

   template <typename Layout> MFEM_HOST_DEVICE inline
   static void result(int* sizes, const Layout &rhs)
   {
      sizes[N - 1] = rhs.template Size<N-1>();
      InitDynamicLayout<Dim,N+1>::result(sizes, rhs);
   }
};

// Terminal case
template <int Dim>
struct InitDynamicLayout<Dim, Dim>
{
   template <typename... Args> MFEM_HOST_DEVICE inline
   static void result(int* sizes, int first, Args... args)
   {
      sizes[Dim - 1] = first;
   }

   template <typename Layout> MFEM_HOST_DEVICE inline
   static void result(int* sizes, const Layout &rhs)
   {
      sizes[Dim - 1] = rhs.template Size<Dim-1>();
   }
};

//Compute the index inside a StaticLayout
template<int Cpt, int rank, int... Dims>
struct StaticIndex
{
   template <typename... Idx> MFEM_HOST_DEVICE inline
   static int eval(int first, Idx... args)
   {
      return first + get_value<Cpt-1,Dims...> * StaticIndex<Cpt+1, rank, Dims...>::eval(args...);
   }
};

template<int rank, int... Dims>
struct StaticIndex<rank,rank,Dims...>
{
   MFEM_HOST_DEVICE inline
   static int eval(int first)
   {
      return first;
   }
};

template<int... Dims>
struct StaticLayoutIndex
{
   template <typename... Idx> MFEM_HOST_DEVICE inline
   static int eval(Idx... args)
   {
      return StaticIndex<1,sizeof...(Dims),Dims...>::eval(args...);
   }
};

template<int... Dims>
struct StaticELayoutIndex
{
   template <typename... Idx> MFEM_HOST_DEVICE inline
   static int eval(Idx... args)
   {
      return StaticIndex<1,sizeof...(Dims)+1,Dims...>::eval(args...);
   }
};

// StaticELayoutSize
template <int StaticSize, int N, int... Sizes>
struct StaticELayoutSize
{
   MFEM_HOST_DEVICE inline
   static int eval(int last_size)
   {
      return get_value<N,Sizes...>;
   }
};

template <int StaticSize, int... Sizes>
struct StaticELayoutSize<StaticSize, StaticSize, Sizes...>
{
   MFEM_HOST_DEVICE inline
   static int eval(int last_size)
   {
      return last_size;
   }
};

template <int Rank>
class DynamicLayout;

template <int N, int Rank>
struct DynamicBlockLayoutSize
{
   MFEM_HOST_DEVICE inline
   static int eval(int size0, int size1, const DynamicLayout<Rank-2> &layout)
   {
      return layout.template Size<N-2>();
   }
};

template <int Rank>
struct DynamicBlockLayoutSize<0, Rank>
{
   MFEM_HOST_DEVICE inline
   static int eval(int size0, int size1, const DynamicLayout<Rank-2> &layout)
   {
      return size0;
   }
};

template <int Rank>
struct DynamicBlockLayoutSize<1, Rank>
{
   MFEM_HOST_DEVICE inline
   static int eval(int size0, int size1, const DynamicLayout<Rank-2> &layout)
   {
      return size1;
   }
};

} // mfem namespace

#endif // MFEM_TENSOR_UTIL
