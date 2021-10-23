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

#ifndef MFEM_STATIC_3DTHREAD_LAYOUT
#define MFEM_STATIC_3DTHREAD_LAYOUT

#include "static_layout.hpp"
#include "layout_traits.hpp"

namespace mfem
{

/// Layout using a thread cube to distribute data
template <int BatchSize, int... Dims>
class Static3dThreadLayout;

template <int BatchSize, int DimX>
class Static3dThreadLayout<BatchSize, DimX>
{
public:
   MFEM_HOST_DEVICE inline
   Static3dThreadLayout()
   {
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.")
   }

   MFEM_HOST_DEVICE inline
   Static3dThreadLayout(int size0)
   {
      MFEM_ASSERT_KERNEL(
         size0==DimX,
         "The runtime first dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Static3dThreadLayout(const Layout& rhs)
   {
      static_assert(
         1 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<0>() == DimX,
         "Layouts sizes don't match.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx0) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Static2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N==0,
         "Accessed size is higher than the rank of the Tensor.");
      return DimX;
   }
};

template <int BatchSize, int DimX, int DimY>
class Static3dThreadLayout<BatchSize, DimX, DimY>
{
public:
   MFEM_HOST_DEVICE inline
   Static3dThreadLayout()
   {
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         DimY<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
   }

   MFEM_HOST_DEVICE inline
   Static3dThreadLayout(int size0, int size1)
   {
      MFEM_ASSERT_KERNEL(
         size0==DimX,
         "The runtime first dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         size1==DimY,
         "The runtime second dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimY<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Static3dThreadLayout(const Layout& rhs)
   {
      static_assert(
         2 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<0>() == DimX,
         "Layouts sizes don't match.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<1>() == DimY,
         "Layouts sizes don't match.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Static2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using Static2dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<2,"Accessed size is higher than the rank of the Tensor.");
      return get_value<N,DimX,DimY>;
   }
};

template <int BatchSize, int DimX, int DimY, int DimZ>
class Static3dThreadLayout<BatchSize, DimX, DimY, DimZ>
{
public:
   MFEM_HOST_DEVICE inline
   Static3dThreadLayout()
   {
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         DimY<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
      MFEM_ASSERT_KERNEL(
         DimZ<MFEM_THREAD_SIZE(z),
         "The third dimension exceeds the number of z threads.");
   }

   MFEM_HOST_DEVICE inline
   Static3dThreadLayout(int size0, int size1, int size2)
   {
      MFEM_ASSERT_KERNEL(
         size0==DimX,
         "The runtime first dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         size1==DimY,
         "The runtime second dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimY<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
      MFEM_ASSERT_KERNEL(
         size2==DimZ,
         "The runtime third dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimZ<MFEM_THREAD_SIZE(z),
         "The third dimension exceeds the number of z threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Static3dThreadLayout(const Layout& rhs)
   {
      static_assert(
         3 == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<0>() == DimX,
         "Layouts sizes don't match.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<1>() == DimY,
         "Layouts sizes don't match.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<2>() == DimZ,
         "Layouts sizes don't match.");
   }

   MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1, int idx2) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Static3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using Static3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx2==MFEM_THREAD_ID(z),
         "The third index must be equal to the z thread index"
         " when using Static3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N>=0 && N<3,
         "Accessed size is higher than the rank of the Tensor.");
      return get_value<N,DimX,DimY,DimZ>;
   }
};

template <int BatchSize, int DimX, int DimY, int DimZ, int... Dims>
class Static3dThreadLayout<BatchSize, DimX, DimY, DimZ, Dims...>
{
private:
   StaticLayout<Dims...> layout;
public:
   MFEM_HOST_DEVICE inline
   Static3dThreadLayout()
   {
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         DimY<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
      MFEM_ASSERT_KERNEL(
         DimZ<MFEM_THREAD_SIZE(z),
         "The third dimension exceeds the number of z threads.");
   }

   template <typename... Sizes> MFEM_HOST_DEVICE inline
   Static3dThreadLayout(int size0, int size1, int size2, Sizes... sizes)
   : layout(sizes...)
   {
      MFEM_ASSERT_KERNEL(
         size0==DimX,
         "The runtime first dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimX<MFEM_THREAD_SIZE(x),
         "The first dimension exceeds the number of x threads.");
      MFEM_ASSERT_KERNEL(
         size1==DimY,
         "The runtime second dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimY<MFEM_THREAD_SIZE(y),
         "The second dimension exceeds the number of y threads.");
      MFEM_ASSERT_KERNEL(
         size2==DimZ,
         "The runtime third dimension is different to the compilation one.");
      MFEM_ASSERT_KERNEL(
         DimZ<MFEM_THREAD_SIZE(z),
         "The third dimension exceeds the number of z threads.");
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Static3dThreadLayout(const Layout& rhs)
   {
      static_assert(
         3 + sizeof...(Dims) == get_layout_rank<Layout>,
         "Can't copy-construct a layout of different rank.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<0>() == DimX,
         "Layouts sizes don't match.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<1>() == DimY,
         "Layouts sizes don't match.");
      MFEM_ASSERT_KERNEL(
         rhs.template size<2>() == DimZ,
         "Layouts sizes don't match.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(int idx0, int idx1, int idx2, Idx... idx) const
   {
      MFEM_ASSERT_KERNEL(
         idx0==MFEM_THREAD_ID(x),
         "The first index must be equal to the x thread index"
         " when using Static3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx1==MFEM_THREAD_ID(y),
         "The second index must be equal to the y thread index"
         " when using Static3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      MFEM_ASSERT_KERNEL(
         idx2==MFEM_THREAD_ID(z),
         "The third index must be equal to the z thread index"
         " when using Static3dThreadLayout. Use shared memory"
         " to access values stored in a different thread.");
      return layout.index(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<rank<DimX,DimY,DimZ,Dims...>,"Accessed size is higher than the rank of the Tensor.");
      return get_value<N,DimX,DimY,DimZ,Dims...>;
   }
};

// get_layout_rank
template <int BatchSize, int... Dims>
struct get_layout_rank_v<Static3dThreadLayout<BatchSize, Dims...>>
{
   static constexpr int value = sizeof...(Dims);
};

// is_static_layout
template <int BatchSize, int... Dims>
struct is_static_layout_v<Static3dThreadLayout<BatchSize,Dims...>>
{
   static constexpr bool value = true;
};

// is_3d_threaded_layout
template <int BatchSize, int... Dims>
struct is_3d_threaded_layout_v<Static3dThreadLayout<BatchSize,Dims...>>
{
   static constexpr bool value = true;
};

// is_serial_layout_dim
template <int BatchSize, int... Dims>
struct is_serial_layout_dim_v<Static3dThreadLayout<BatchSize,Dims...>, 0>
{
   static constexpr bool value = false;
};

template <int BatchSize, int... Dims>
struct is_serial_layout_dim_v<Static3dThreadLayout<BatchSize,Dims...>, 1>
{
   static constexpr bool value = false;
};

template <int BatchSize, int... Dims>
struct is_serial_layout_dim_v<Static3dThreadLayout<BatchSize,Dims...>, 2>
{
   static constexpr bool value = false;
};

// is_threaded_layout_dim
template <int BatchSize, int... Dims>
struct is_threaded_layout_dim_v<Static3dThreadLayout<BatchSize,Dims...>, 0>
{
   static constexpr bool value = true;
};

template <int BatchSize, int... Dims>
struct is_threaded_layout_dim_v<Static3dThreadLayout<BatchSize,Dims...>, 1>
{
   static constexpr bool value = true;
};

template <int BatchSize, int... Dims>
struct is_threaded_layout_dim_v<Static3dThreadLayout<BatchSize,Dims...>, 2>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int BatchSize, int... Dims>
struct get_layout_size_v<N, Static3dThreadLayout<BatchSize, Dims...>>
{
   static constexpr int value = get_value<N, Dims...>;
};

// get_layout_sizes
template <int BatchSize, int... Dims>
struct get_layout_sizes_t<Static3dThreadLayout<BatchSize, Dims...>>
{
   using type = int_list<Dims...>;
};

// get_layout_batch_size
template <int BatchSize, int... Dims>
struct get_layout_batch_size_v<Static3dThreadLayout<BatchSize, Dims...>>
{
   static constexpr int value = BatchSize;
};

// get_layout_capacity
template <int BatchSize, int DimX>
struct get_layout_capacity_v<Static3dThreadLayout<BatchSize, DimX>>
{
   static constexpr int value = BatchSize;
};

template <int BatchSize, int DimX, int DimY>
struct get_layout_capacity_v<Static3dThreadLayout<BatchSize, DimX, DimY>>
{
   static constexpr int value = BatchSize;
};

template <int BatchSize, int DimX, int DimY, int DimZ>
struct get_layout_capacity_v<Static3dThreadLayout<BatchSize, DimX, DimY, DimZ>>
{
   static constexpr int value = BatchSize;
};

template <int BatchSize, int DimX, int DimY, int DimZ, int... Dims>
struct get_layout_capacity_v<
   Static3dThreadLayout<BatchSize, DimX, DimY, DimZ, Dims...>>
{
   static constexpr int value = BatchSize * prod(Dims...);
};

// get_layout_result_type
template <int BatchSize, int... Sizes>
struct get_layout_result_type<Static3dThreadLayout<BatchSize,Sizes...>>
{
   template <int... Dims>
   using type = Static3dThreadLayout<BatchSize,Dims...>;
};

} // namespace mfem

#endif // MFEM_STATIC_3DTHREAD_LAYOUT
