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
   constexpr Static3dThreadLayout()
   {
      // TODO verify that DimX < BlockSizeX
   }

   MFEM_HOST_DEVICE inline
   constexpr Static3dThreadLayout(int size0)
   {
      // TODO Verify in debug that size0==DimX
      // TODO verify that size0 < BlockSizeX
      // TODO verify that BlockSizeZ == BatchSize
   }

   template <typename Layout> MFEM_HOST_DEVICE
   constexpr Static3dThreadLayout(const Layout& rhs)
   {
      // TODO verifications
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx0) const
   {
      // TODO verify that idx0 < DimX
      // TODO verify that idx0 == threadIdx.x
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N==0,"Accessed size is higher than the rank of the Tensor.");
      return DimX;
   }
};

template <int BatchSize, int DimX, int DimY>
class Static3dThreadLayout<BatchSize, DimX, DimY>
{
public:
   MFEM_HOST_DEVICE inline
   constexpr Static3dThreadLayout()
   {
      // TODO verify that DimX < BlockSizeX && DimY < BlockSizeY
   }

   MFEM_HOST_DEVICE inline
   constexpr Static3dThreadLayout(int size0, int size1)
   {
      // TODO Verify in debug that size0==DimX && size1==DimY
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that BlockSizeZ == BatchSize
   }

   template <typename Layout> MFEM_HOST_DEVICE
   constexpr Static3dThreadLayout(const Layout& rhs)
   {
      // TODO verifications
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
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
   constexpr Static3dThreadLayout()
   {
      // TODO verify that DimX < BlockSizeX && DimY < BlockSizeY
      // TODO verify that DimZ < BlockSizeZ
   }

   MFEM_HOST_DEVICE inline
   constexpr Static3dThreadLayout(int size0, int size1, int size2)
   {
      // TODO Verify in debug that size0==DimX && size1==DimY
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that size2 < BlockSizeZ
   }

   template <typename Layout> MFEM_HOST_DEVICE
   constexpr Static3dThreadLayout(const Layout& rhs)
   {
      // TODO verifications
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY && idx2 < DimZ
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y && idx2 == threadIdx.z
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<3,"Accessed size is higher than the rank of the Tensor.");
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
   constexpr Static3dThreadLayout()
   {
      // TODO verify that DimX < BlockSizeX && DimY < BlockSizeY
      // TODO verify that DimZ < BlockSizeZ
   }

   template <typename... Sizes> MFEM_HOST_DEVICE inline
   constexpr Static3dThreadLayout(int size0, int size1, int size2, Sizes... sizes)
   : layout(sizes...)
   {
      // TODO Verify in debug that size0==DimX && size1==DimY && size2==DimZ
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY && size2 < BlockSizeZ
   }

   template <typename Layout> MFEM_HOST_DEVICE
   constexpr Static3dThreadLayout(const Layout& rhs)
   {
      // TODO verifications
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, Idx... idx) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY && idx2 < DimZ
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y && idx2 == threadIdx.z
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
