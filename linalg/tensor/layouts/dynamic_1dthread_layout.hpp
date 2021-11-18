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

#ifndef MFEM_DYNAMIC_1DTHREAD_LAYOUT
#define MFEM_DYNAMIC_1DTHREAD_LAYOUT

#include "../../../general/error.hpp"
#include "dynamic_layout.hpp"
#include "layout_traits.hpp"

namespace mfem
{

template <int BatchSize, int... Sizes>
class SizedDynamic1dThreadLayout;

template <int Rank, int BatchSize>
using Dynamic1dThreadLayout = instantiate<SizedDynamic1dThreadLayout,
                                          append<
                                             int_list<BatchSize>,
                                             int_repeat<Dynamic,Rank> > >;

template <int BatchSize, int FirstSize, int... Sizes>
class SizedDynamic1dThreadLayout<BatchSize,FirstSize,Sizes...>
{
private:
   const int size0;
   SizedDynamicLayout<Sizes...> layout;
public:
   template <typename... Ss> MFEM_HOST_DEVICE inline
   SizedDynamic1dThreadLayout(int size0, Ss... sizes)
   : size0(size0), layout(sizes...)
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
   SizedDynamic1dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     layout(rhs.template Get<0>(0))
   {
      constexpr int Rank = sizeof...(Sizes) + 1;
      static_assert(
         Rank-1 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(int idx0, Idx... idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index (%d) must be equal to the x thread index (%d)"
         " when using SizedDynamic1dThreadLayout. Use shared memory"
         " to access values stored in a different thread.\n",
         idx0, MFEM_THREAD_ID(x));
      return layout.index(idx...);
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      constexpr int Rank = sizeof...(Sizes) + 1;
      static_assert(
         N>=0 && N<Rank,
         "Accessed size is higher than the rank of the Tensor.");
      return DynamicBlockLayoutSize<N,Rank>::eval(size0,layout);
   }
};

template <int BatchSize, int FirstSize>
class SizedDynamic1dThreadLayout<BatchSize, FirstSize>
{
private:
   const int size0;
public:
   MFEM_HOST_DEVICE inline
   SizedDynamic1dThreadLayout(int size0)
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
   SizedDynamic1dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>())
   {
      static_assert(
         1 == get_layout_rank<Layout>,
         "Can't copy-construct with a layout of different rank.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx==MFEM_THREAD_ID(x),
         "The first index (%d) must be equal to the x thread index (%d)"
         " when using SizedDynamic1dThreadLayout. Use shared memory"
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

// get_layout_rank
template <int BatchSize, int... Sizes>
struct get_layout_rank_v<SizedDynamic1dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = sizeof...(Sizes);
};

// is_dynamic_layout
template <int BatchSize, int... Sizes>
struct is_dynamic_layout_v<SizedDynamic1dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr bool value = true;
};

// is_1d_threaded_layout
template <int BatchSize, int... Sizes>
struct is_1d_threaded_layout_v<SizedDynamic1dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr bool value = true;
};

// is_serial_layout_dim
template <int BatchSize, int... Sizes>
struct is_serial_layout_dim_v<SizedDynamic1dThreadLayout<BatchSize,Sizes...>, 0>
{
   static constexpr bool value = false;
};

// is_threaded_layout_dim
template <int BatchSize, int... Sizes>
struct is_threaded_layout_dim_v<SizedDynamic1dThreadLayout<BatchSize,Sizes...>, 0>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int BatchSize, int... Sizes>
struct get_layout_size_v<N, SizedDynamic1dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = get_value<N,Sizes...>;
};

// get_layout_sizes
template <int BatchSize, int... Sizes>
struct get_layout_sizes_t<SizedDynamic1dThreadLayout<BatchSize,Sizes...>>
{
   using type = int_list<Sizes...>;
};

// get_layout_capacity
template <int BatchSize, int First, int... Rest>
struct get_layout_capacity_v<SizedDynamic1dThreadLayout<BatchSize,First,Rest...>>
{
   static constexpr int value = get_layout_capacity<SizedDynamicLayout<Rest...>>;
};

template <int BatchSize, int First>
struct get_layout_capacity_v<SizedDynamic1dThreadLayout<BatchSize,First>>
{
   static constexpr int value = 1;
};

// get_layout_batch_size
template <int BatchSize, int... Sizes>
struct get_layout_batch_size_v<SizedDynamic1dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = BatchSize;
};

// get_layout_result_type
template <int BatchSize, int... Sizes>
struct get_layout_result_type_t<SizedDynamic1dThreadLayout<BatchSize,Sizes...>>
{
   template <int... mySizes>
   using type = SizedDynamic1dThreadLayout<BatchSize,mySizes...>;
};

} // namespace mfem

#endif // MFEM_DYNAMIC_1DTHREAD_LAYOUT
