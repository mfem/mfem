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

template <int Rank, int BatchSize>
class Dynamic1dThreadLayout
{
private:
   const int size0;
   DynamicLayout<Rank-1> layout;
public:
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   Dynamic1dThreadLayout(int size0, Sizes... sizes)
   : size0(size0), layout(sizes...)
   {
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         BatchSize==MFEM_THREAD_SIZE(z),
         "The batchsize is not equal to the number of z threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Dynamic1dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     layout(rhs.template Get<0>(0))
   {
      static_assert(
         Rank-1 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(int idx0, Idx... idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Dynamic1dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return layout.index(idx...);
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N>=0 && N<Rank,
         "Accessed size is higher than the rank of the Tensor.");
      return DynamicBlockLayoutSize<N,Rank>::eval(size0,layout);
   }
};

template <int BatchSize>
class Dynamic1dThreadLayout<1,BatchSize>
{
private:
   const int size0;
public:
   MFEM_HOST_DEVICE inline
   Dynamic1dThreadLayout(int size0)
   : size0(size0)
   {
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         BatchSize==MFEM_THREAD_SIZE(z),
         "The batchsize is not equal to the number of z threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Dynamic1dThreadLayout(const Layout &rhs)
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
         "The first index must be equal to the x thread index"
         " when using Dynamic1dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
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
template <int Rank, int BatchSize>
struct get_layout_rank_v<Dynamic1dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = Rank;
};

// is_dynamic_layout
template <int Rank, int BatchSize>
struct is_dynamic_layout_v<Dynamic1dThreadLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};

// is_1d_threaded_layout
template <int Rank, int BatchSize>
struct is_1d_threaded_layout_v<Dynamic1dThreadLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};

// is_serial_layout_dim
template <int Rank, int BatchSize>
struct is_serial_layout_dim_v<Dynamic1dThreadLayout<Rank,BatchSize>, 0>
{
   static constexpr bool value = false;
};

// is_threaded_layout_dim
template <int Rank, int BatchSize>
struct is_threaded_layout_dim_v<Dynamic1dThreadLayout<Rank,BatchSize>, 0>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int Rank, int BatchSize>
struct get_layout_size_v<N, Dynamic1dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = Dynamic;
};

// get_layout_sizes
template <int Rank, int BatchSize>
struct get_layout_sizes_t<Dynamic1dThreadLayout<Rank,BatchSize>>
{
   using type = int_repeat<Dynamic,Rank>;
};

// get_layout_batch_size
template <int Rank, int BatchSize>
struct get_layout_batch_size_v<Dynamic1dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = BatchSize;
};

// get_layout_result_type
template <int Rank, int BatchSize>
struct get_layout_result_type<Dynamic1dThreadLayout<Rank,BatchSize>>
{
   template <int myRank>
   using type = Dynamic1dThreadLayout<myRank,BatchSize>;
};

} // namespace mfem

#endif // MFEM_DYNAMIC_1DTHREAD_LAYOUT
