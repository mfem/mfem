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

template <int BatchSize, int... Sizes>
class SizedDynamic3dThreadLayout;

template <int Rank, int BatchSize>
using Dynamic3dThreadLayout = instantiate<SizedDynamic3dThreadLayout,
                                          append<
                                             int_list<BatchSize>,
                                             int_repeat<Dynamic,Rank> > >;

template <int BatchSize, int FirstSize, int SecondSize, int ThirdSize, int... Sizes>
class SizedDynamic3dThreadLayout<BatchSize,FirstSize,SecondSize,ThirdSize,Sizes...>
{
private:
   const int size0;
   const int size1;
   const int size2;
   SizedDynamicLayout<Sizes...> layout;
public:
   template <typename... Ss> MFEM_HOST_DEVICE inline
   SizedDynamic3dThreadLayout(int size0, int size1, int size2, Ss... sizes)
   : size0(size0), size1(size1), size2(size2), layout(sizes...)
   {
      MFEM_ASSERT_KERNEL(
         FirstSize==Dynamic || FirstSize==size0,
         "Compilation time and runtime sizes must be the same.");
      MFEM_ASSERT_KERNEL(
         SecondSize==Dynamic || SecondSize==size1,
         "Compilation time and runtime sizes must be the same.");
      MFEM_ASSERT_KERNEL(
         ThirdSize==Dynamic || ThirdSize==size2,
         "Compilation time and runtime sizes must be the same.");
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

   template <typename Layout> MFEM_HOST_DEVICE
   SizedDynamic3dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     size1(rhs.template Size<1>()),
     size2(rhs.template Size<2>()),
     layout( Get<0>( 0, Get<0>( 0, Get<0>( 0, rhs ) ) ) )
   {
      constexpr int Rank = sizeof...(Sizes) + 3;
      static_assert(
         Rank == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1, int idx2, Idx... idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx2==MFEM_THREAD_ID(z),
         "The third index must be equal to the z thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return layout.index(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      constexpr int Rank = sizeof...(Sizes) + 3;
      static_assert(
         N>=0 && N<Rank,
         "Accessed size is higher than the rank of the Tensor.");
      return Dynamic3dThreadLayoutSize<N,Rank>::eval(size0,size1,size2,layout);
   }
};

template <int BatchSize, int FirstSize>
class SizedDynamic3dThreadLayout<BatchSize,FirstSize>
{
private:
   const int size0;
public:
   MFEM_HOST_DEVICE inline
   SizedDynamic3dThreadLayout(int size0)
   : size0(size0)
   {
      MFEM_ASSERT_KERNEL(
         FirstSize==Dynamic || FirstSize==size0,
         "Compilation time and runtime sizes must be the same.");
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   SizedDynamic3dThreadLayout(const Layout &rhs)
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
         "The first index must be equal to the x thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
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

template <int BatchSize, int FirstSize, int SecondSize>
class SizedDynamic3dThreadLayout<BatchSize,FirstSize,SecondSize>
{
private:
   const int size0;
   const int size1;
public:
   MFEM_HOST_DEVICE inline
   SizedDynamic3dThreadLayout(int size0, int size1)
   : size0(size0), size1(size1)
   {
      MFEM_ASSERT_KERNEL(
         FirstSize==Dynamic || FirstSize==size0,
         "Compilation time and runtime sizes must be the same.");
      MFEM_ASSERT_KERNEL(
         SecondSize==Dynamic || SecondSize==size1,
         "Compilation time and runtime sizes must be the same.");
      MFEM_ASSERT_KERNEL(
         size0<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         size1<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   SizedDynamic3dThreadLayout(const Layout &rhs)
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
         "The first index must be equal to the x thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
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

template <int BatchSize, int FirstSize, int SecondSize, int ThirdSize>
class SizedDynamic3dThreadLayout<BatchSize,FirstSize,SecondSize,ThirdSize>
{
private:
   const int size0;
   const int size1;
   const int size2;
public:
   MFEM_HOST_DEVICE inline
   SizedDynamic3dThreadLayout(int size0, int size1, int size2)
   : size0(size0), size1(size1), size2(size2)
   {
      MFEM_ASSERT_KERNEL(
         FirstSize==Dynamic || FirstSize==size0,
         "Compilation time and runtime sizes must be the same.");
      MFEM_ASSERT_KERNEL(
         SecondSize==Dynamic || SecondSize==size1,
         "Compilation time and runtime sizes must be the same.");
      MFEM_ASSERT_KERNEL(
         ThirdSize==Dynamic || ThirdSize==size2,
         "Compilation time and runtime sizes must be the same.");
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

   template <typename Layout> MFEM_HOST_DEVICE
   SizedDynamic3dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     size1(rhs.template Size<1>()),
     size2(rhs.template Size<2>())
   {
      static_assert(
         3 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1, int idx2) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx2==MFEM_THREAD_ID(z),
         "The third index must be equal to the z thread index"
         " when using SizedDynamic3dThreadLayout. Use shared memory"
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
template <int BatchSize, int... Sizes>
struct get_layout_rank_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = sizeof...(Sizes);
};

// is_dynamic_layout
template <int BatchSize, int... Sizes>
struct is_dynamic_layout_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr bool value = true;
};

// is_3d_threaded_layout
template <int BatchSize, int... Sizes>
struct is_3d_threaded_layout_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr bool value = true;
};

// is_serial_layout_dim
template <int BatchSize, int... Sizes>
struct is_serial_layout_dim_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>, 0>
{
   static constexpr bool value = false;
};

template <int BatchSize, int... Sizes>
struct is_serial_layout_dim_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>, 1>
{
   static constexpr bool value = false;
};

template <int BatchSize, int... Sizes>
struct is_serial_layout_dim_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>, 2>
{
   static constexpr bool value = false;
};

// is_threaded_layout_dim
template <int BatchSize, int... Sizes>
struct is_threaded_layout_dim_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>, 0>
{
   static constexpr bool value = true;
};

template <int BatchSize, int... Sizes>
struct is_threaded_layout_dim_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>, 1>
{
   static constexpr bool value = true;
};

template <int BatchSize, int... Sizes>
struct is_threaded_layout_dim_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>, 2>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int BatchSize, int... Sizes>
struct get_layout_size_v<N, SizedDynamic3dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = get_value<N,Sizes...>;
};

// get_layout_sizes
template <int BatchSize, int... Sizes>
struct get_layout_sizes_t<SizedDynamic3dThreadLayout<BatchSize,Sizes...>>
{
   using type = int_list<Sizes...>;
};

// get_layout_capacity
template <int BatchSize, int First, int Second, int Third, int... Rest>
struct get_layout_capacity_v<SizedDynamic3dThreadLayout<BatchSize,First,Second,Third,Rest...>>
{
   static constexpr int value = get_layout_capacity<SizedDynamicLayout<Rest...>>;
};

template <int BatchSize, int First, int Second, int Third>
struct get_layout_capacity_v<SizedDynamic3dThreadLayout<BatchSize,First,Second,Third>>
{
   static constexpr int value = 1;
};

template <int BatchSize, int First, int Second>
struct get_layout_capacity_v<SizedDynamic3dThreadLayout<BatchSize,First,Second>>
{
   static constexpr int value = 1;
};

template <int BatchSize, int First>
struct get_layout_capacity_v<SizedDynamic3dThreadLayout<BatchSize,First>>
{
   static constexpr int value = 1;
};

// get_layout_batch_size
template <int BatchSize, int... Sizes>
struct get_layout_batch_size_v<SizedDynamic3dThreadLayout<BatchSize,Sizes...>>
{
   static constexpr int value = BatchSize;
};

// get_layout_result_type
template <int BatchSize, int... Sizes>
struct get_layout_result_type_t<SizedDynamic3dThreadLayout<BatchSize,Sizes...>>
{
   template <int... mySizes>
   using type = SizedDynamic3dThreadLayout<BatchSize,mySizes...>;
};

} // namespace mfem

#endif // MFEM_DYNAMIC_3DTHREAD_LAYOUT
