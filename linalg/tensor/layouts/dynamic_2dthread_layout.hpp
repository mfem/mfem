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

#ifndef MFEM_DYNAMIC_2DTHREAD_LAYOUT
#define MFEM_DYNAMIC_2DTHREAD_LAYOUT

#include "../../../general/error.hpp"
#include "dynamic_layout.hpp"
#include "layout_traits.hpp"
#include "../operators/get.hpp"

namespace mfem
{

template <int BatchSize, int... Sizes>
class SizedDynamic2dThreadLayout;

template <int Rank, int BatchSize>
using Dynamic2dThreadLayout = instantiate<SizedDynamic2dThreadLayout,
                                          append<
                                             int_list<BatchSize>,
                                             int_repeat<Dynamic,Rank> > >;

template <int BatchSize, int FirstSize, int SecondSize, int... Sizes>
class SizedDynamic2dThreadLayout<BatchSize,FirstSize,SecondSize,Sizes...>
{
private:
   const int size0;
   const int size1;
   SizedDynamicLayout<Sizes...> layout;
public:
   template <typename... Ss> MFEM_HOST_DEVICE inline
   SizedDynamic2dThreadLayout(int size0, int size1,  Ss... sizes)
   : size0(size0), size1(size1), layout(sizes...)
   {
      MFEM_ASSERT_KERNEL(
         FirstSize==Dynamic || FirstSize==size0,
         "Compilation time (%d) and runtime sizes (%d) must be the same.\n",
         FirstSize, size0);
      MFEM_ASSERT_KERNEL(
         SecondSize==Dynamic || SecondSize==size1,
         "Compilation time (%d) and runtime sizes (%d) must be the same.\n",
         SecondSize, size1);
      MFEM_ASSERT_KERNEL(
         size0<=MFEM_THREAD_SIZE(x),
         "The first dimension (%d) exceeds the number of x threads (%d).\n",
         size0, MFEM_THREAD_SIZE(x));
      MFEM_ASSERT_KERNEL(
         size1<=MFEM_THREAD_SIZE(y),
         "The second dimension (%d) exceeds the number of x threads (%d).\n",
         size1, MFEM_THREAD_SIZE(y));
      MFEM_ASSERT_KERNEL(
         BatchSize==MFEM_THREAD_SIZE(z),
         "The batchsize (%d) is not equal to the number of z threads (%d).\n",
         BatchSize, MFEM_THREAD_SIZE(z));
   }

   template <typename Layout> MFEM_HOST_DEVICE
   SizedDynamic2dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     size1(rhs.template Size<1>()),
     layout( Get<0>( 0, Get<0>( 0, rhs ) ) )
   {
      constexpr int Rank = sizeof...(Sizes) + 2;
      static_assert(
         Rank == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1, Idx... idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index (%d) must be equal to the x thread index (%d)"
         " when using SizedDynamic2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.\n",
         idx0, MFEM_THREAD_ID(x));
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index (%d) must be equal to the y thread index (%d)"
         " when using SizedDynamic2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.\n",
         idx1, MFEM_THREAD_ID(y));
      return layout.index(idx...);
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      constexpr int Rank = sizeof...(Sizes) + 2;
      static_assert(
         N>=0 && N<Rank,
         "Accessed size is higher than the rank of the Tensor.");
      return DynamicBlockLayoutSize<N,Rank>::eval(size0,size1,layout);
   }
};

template <int BatchSize, int FirstSize>
class SizedDynamic2dThreadLayout<BatchSize,FirstSize>
{
private:
   const int size0;
public:
   MFEM_HOST_DEVICE inline
   SizedDynamic2dThreadLayout(int size0)
   : size0(size0)
   {
      MFEM_ASSERT_KERNEL(
         FirstSize==Dynamic || FirstSize==size0,
         "Compilation time (%d) and runtime sizes (%d) must be the same.\n",
         FirstSize, size0);
      MFEM_ASSERT_KERNEL(
         size0<=MFEM_THREAD_SIZE(x),
         "The first dimension (%d) exceeds the number of x threads (%d).\n",
         size0, MFEM_THREAD_SIZE(x));
      MFEM_ASSERT_KERNEL(
         BatchSize==MFEM_THREAD_SIZE(z),
         "The batchsize (%d) is not equal to the number of z threads (%d).\n",
         BatchSize, MFEM_THREAD_SIZE(z));
   }

   template <typename Layout> MFEM_HOST_DEVICE
   SizedDynamic2dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>())
   {
      static_assert(
         1 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx==MFEM_THREAD_ID(x),
         "The first index (%d) must be equal to the x thread index (%d)"
         " when using SizedDynamic2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.\n",
         idx, MFEM_THREAD_ID(x));
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N==0,
         "Accessed size is higher than the rank of the Tensor.");
      return size0;
   }
};

template <int BatchSize, int FirstSize, int SecondSize>
class SizedDynamic2dThreadLayout<BatchSize,FirstSize,SecondSize>
{
private:
   const int size0;
   const int size1;
public:
   MFEM_HOST_DEVICE inline
   SizedDynamic2dThreadLayout(int size0, int size1)
   : size0(size0), size1(size1)
   {
      MFEM_ASSERT_KERNEL(
         FirstSize==Dynamic || FirstSize==size0,
         "Compilation time (%d) and runtime sizes (%d) must be the same.\n",
         FirstSize, size0);
      MFEM_ASSERT_KERNEL(
         SecondSize==Dynamic || SecondSize==size1,
         "Compilation time (%d) and runtime sizes (%d) must be the same.\n",
         SecondSize, size1);
      MFEM_ASSERT_KERNEL(
         size0<=MFEM_THREAD_SIZE(x),
         "The first dimension (%d) exceeds the number of x threads (%d).\n",
         size0, MFEM_THREAD_SIZE(x));
      MFEM_ASSERT_KERNEL(
         size1<=MFEM_THREAD_SIZE(y),
         "The second dimension (%d) exceeds the number of x threads (%d).\n",
         size1, MFEM_THREAD_SIZE(y));
      MFEM_ASSERT_KERNEL(
         BatchSize==MFEM_THREAD_SIZE(z),
         "The batchsize (%d) is not equal to the number of z threads (%d).\n",
         BatchSize, MFEM_THREAD_SIZE(z));
   }

   template <typename Layout> MFEM_HOST_DEVICE
   SizedDynamic2dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     size1(rhs.template Size<1>())
   {
      static_assert(
         2 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index (%d) must be equal to the x thread index (%d)"
         " when using SizedDynamic2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.\n",
         idx0, MFEM_THREAD_ID(x));
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index (%d) must be equal to the y thread index (%d)"
         " when using SizedDynamic2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.\n",
         idx1, MFEM_THREAD_ID(y));
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N>=0 && N<2,
         "Accessed size is higher than the rank of the Tensor.");
      return N==0? size0 : size1;
   }
};

// get_layout_rank
template <int BatchSize, int... Sizes>
struct get_layout_rank_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = sizeof...(Sizes);
};

// is_dynamic_layout
template <int BatchSize, int... Sizes>
struct is_dynamic_layout_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr bool value = true;
};

// is_2d_threaded_layout
template <int BatchSize, int... Sizes>
struct is_2d_threaded_layout_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr bool value = true;
};

// is_serial_layout_dim
template <int BatchSize, int... Sizes>
struct is_serial_layout_dim_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>, 0>
{
   static constexpr bool value = false;
};

template <int BatchSize, int... Sizes>
struct is_serial_layout_dim_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>, 1>
{
   static constexpr bool value = false;
};

// is_threaded_layout_dim
template <int BatchSize, int... Sizes>
struct is_threaded_layout_dim_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>, 0>
{
   static constexpr bool value = true;
};

template <int BatchSize, int... Sizes>
struct is_threaded_layout_dim_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>, 1>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int BatchSize, int... Sizes>
struct get_layout_size_v<N, SizedDynamic2dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = get_value<N,Sizes...>;
};

// get_layout_sizes
template <int BatchSize, int... Sizes>
struct get_layout_sizes_t<SizedDynamic2dThreadLayout<BatchSize,Sizes...>>
{
   using type = int_list<Sizes...>;
};

// get_layout_capacity
template <int BatchSize, int First, int Second, int... Rest>
struct get_layout_capacity_v<SizedDynamic2dThreadLayout<BatchSize,First,Second,Rest...>>
{
   static constexpr int value = get_layout_capacity<SizedDynamicLayout<Rest...>>;
};

template <int BatchSize, int First, int Second>
struct get_layout_capacity_v<SizedDynamic2dThreadLayout<BatchSize,First,Second>>
{
   static constexpr int value = 1;
};

template <int BatchSize, int First>
struct get_layout_capacity_v<SizedDynamic2dThreadLayout<BatchSize,First>>
{
   static constexpr int value = 1;
};

// get_layout_batch_size
template <int BatchSize, int... Sizes>
struct get_layout_batch_size_v<SizedDynamic2dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = BatchSize;
};

// get_layout_result_type
template <int BatchSize, int... Sizes>
struct get_layout_result_type_t<SizedDynamic2dThreadLayout<BatchSize,Sizes...>>
{
   template <int... mySizes>
   using type = SizedDynamic2dThreadLayout<BatchSize,mySizes...>;
};

} // namespace mfem

#endif // MFEM_DYNAMIC_2DTHREAD_LAYOUT
