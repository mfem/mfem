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

#ifndef MFEM_TENSOR_LAYOUT_IMPL
#define MFEM_TENSOR_LAYOUT_IMPL

#include "../../../general/error.hpp"
#include "../../../general/forall.hpp"

namespace mfem
{

/// Layout utility classes

/// A Class to compute the real index from the multi-indices of a DynamicLayout
template <int Dim, int N = 1>
struct DynamicLayoutIndex
{
   template <typename... Args> MFEM_HOST_DEVICE inline
   static int eval(const int* sizes, int first, Args... args)
   {
      MFEM_ASSERT_KERNEL(
         first<sizes[N-1],
         "Trying to access out of boundary.");
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
      MFEM_ASSERT_KERNEL(
         first<sizes[Dim-1],
         "Trying to access out of boundary.");
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
      constexpr int size = get_value<Cpt-1,Dims...>;
      MFEM_ASSERT_KERNEL(
         first<size,
         "Trying to access out of boundary.");
      return first + size * StaticIndex<Cpt+1, rank, Dims...>::eval(args...);
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

template <int N, int Rank>
struct Dynamic3dThreadLayoutSize
{
   MFEM_HOST_DEVICE inline
   static int eval(int size0, int size1, int size2,
                   const DynamicLayout<Rank-3> &layout)
   {
      return layout.template Size<N-3>();
   }
};

template <int Rank>
struct Dynamic3dThreadLayoutSize<0, Rank>
{
   MFEM_HOST_DEVICE inline
   static int eval(int size0, int size1, int size2,
                   const DynamicLayout<Rank-3> &layout)
   {
      return size0;
   }
};

template <int Rank>
struct Dynamic3dThreadLayoutSize<1, Rank>
{
   MFEM_HOST_DEVICE inline
   static int eval(int size0, int size1, int size2,
                   const DynamicLayout<Rank-3> &layout)
   {
      return size1;
   }
};

template <int Rank>
struct Dynamic3dThreadLayoutSize<2, Rank>
{
   MFEM_HOST_DEVICE inline
   static int eval(int size0, int size1, int size2,
                   const DynamicLayout<Rank-3> &layout)
   {
      return size2;
   }
};

} // mfem namespace

#endif // MFEM_TENSOR_LAYOUT_IMPL
