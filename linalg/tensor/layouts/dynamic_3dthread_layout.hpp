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

#ifndef MFEM_DYNAMIC_3DTHREAD_LAYOUT
#define MFEM_DYNAMIC_3DTHREAD_LAYOUT

#include "../../../general/error.hpp"
#include "dynamic_layout.hpp"
#include "layout_traits.hpp"

namespace mfem
{

template <int Rank, int BatchSize>
class Dynamic3dThreadLayout
{
private:
   const int size0;
   const int size1;
   const int size2;
   DynamicLayout<Rank-3> layout;
public:
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   Dynamic3dThreadLayout(int size0, int size1, int size2, Sizes... sizes)
   : size0(size0), size1(size1), size2(size2), layout(sizes...)
   {
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         size1<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
      MFEM_ASSERT_KERNEL(
         size2<MFEM_THREAD_SIZE(z),
         "The third dimension exceeds the number of z threads.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1, int idx2, Idx... idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx2==MFEM_THREAD_ID(z),
         "The third index must be equal to the z thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return layout.index(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N>=0 && N<Rank,
         "Accessed size is higher than the rank of the Tensor.");
      return Dynamic3dThreadLayoutSize<N,Rank>::eval(size0,size1,size2,layout);
   }
};

template <int BatchSize>
class Dynamic3dThreadLayout<1,BatchSize>
{
private:
   const int size0;
public:
   MFEM_HOST_DEVICE inline
   Dynamic3dThreadLayout(int size0)
   : size0(size0)
   {
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return 0;
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N==0,
         "Accessed size is higher than the rank of the Tensor.");
      return size0;
   }
};

template <int BatchSize>
class Dynamic3dThreadLayout<2,BatchSize>
{
private:
   const int size0;
   const int size1;
public:
   MFEM_HOST_DEVICE inline
   Dynamic3dThreadLayout(int size0, int size1)
   : size0(size0), size1(size1)
   {
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         size1<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
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

template <int BatchSize>
class Dynamic3dThreadLayout<3,BatchSize>
{
private:
   const int size0;
   const int size1;
   const int size2;
public:
   MFEM_HOST_DEVICE inline
   Dynamic3dThreadLayout(int size0, int size1, int size2)
   : size0(size0), size1(size1), size2(size2)
   {
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         size1<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
      MFEM_ASSERT_KERNEL(
         size2<MFEM_THREAD_SIZE(z),
         "The third dimension exceeds the number of z threads.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1, int idx2) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx2==MFEM_THREAD_ID(z),
         "The third index must be equal to the z thread index"
         " when using Dynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return 0;
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N>=0 && N<3,
         "Accessed size is higher than the rank of the Tensor.");
      return N==0? size0 : (N==1? size1 : size2);
   }
};

// get_layout_rank
template <int Rank, int BatchSize>
struct get_layout_rank_v<Dynamic3dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = Rank;
};

// is_dynamic_layout
template <int Rank, int BatchSize>
struct is_dynamic_layout_v<Dynamic3dThreadLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};

// is_3d_threaded_layout
template <int Rank, int BatchSize>
struct is_3d_threaded_layout_v<Dynamic3dThreadLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};

// is_serial_layout_dim
template <int Rank, int BatchSize>
struct is_serial_layout_dim_v<Dynamic3dThreadLayout<Rank,BatchSize>, 0>
{
   static constexpr bool value = false;
};

template <int Rank, int BatchSize>
struct is_serial_layout_dim_v<Dynamic3dThreadLayout<Rank,BatchSize>, 1>
{
   static constexpr bool value = false;
};

template <int Rank, int BatchSize>
struct is_serial_layout_dim_v<Dynamic3dThreadLayout<Rank,BatchSize>, 2>
{
   static constexpr bool value = false;
};

// is_threaded_layout_dim
template <int Rank, int BatchSize>
struct is_threaded_layout_dim_v<Dynamic3dThreadLayout<Rank,BatchSize>, 0>
{
   static constexpr bool value = true;
};

template <int Rank, int BatchSize>
struct is_threaded_layout_dim_v<Dynamic3dThreadLayout<Rank,BatchSize>, 1>
{
   static constexpr bool value = true;
};

template <int Rank, int BatchSize>
struct is_threaded_layout_dim_v<Dynamic3dThreadLayout<Rank,BatchSize>, 2>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int Rank, int BatchSize>
struct get_layout_size_v<N, Dynamic3dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = Dynamic;
};

// get_layout_sizes
template <int Rank, int BatchSize>
struct get_layout_sizes_t<Dynamic3dThreadLayout<Rank,BatchSize>>
{
   using type = int_repeat<Dynamic,Rank>;
};

// get_layout_batch_size
template <int Rank, int BatchSize>
struct get_layout_batch_size_v<Dynamic3dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = BatchSize;
};

// get_layout_result_type
template <int Rank, int BatchSize>
struct get_layout_result_type<Dynamic3dThreadLayout<Rank,BatchSize>>
{
   template <int myRank>
   using type = Dynamic3dThreadLayout<myRank,BatchSize>;
};

} // namespace mfem

#endif // MFEM_DYNAMIC_3DTHREAD_LAYOUT
