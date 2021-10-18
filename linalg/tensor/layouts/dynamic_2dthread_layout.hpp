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

#include "dynamic_layout.hpp"
#include "layout_traits.hpp"

namespace mfem
{

template <int Rank, int BatchSize>
class Dynamic2dThreadLayout
{
private:
   const int size0;
   const int size1;
   DynamicLayout<Rank-2> layout;
public:
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   Dynamic2dThreadLayout(int size0, int size1,  Sizes... sizes)
   : size0(size0), size1(size1), layout(sizes...)
   {
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that BlockSizeZ == BatchSize
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Dynamic2dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     size1(rhs.template Size<1>()),
     layout(rhs.template Get<0>(0).template Get<0>(0))
   {
      // static_assert(Rank == get_layout_rank<Layout>,
      //               "Can't copy-construct with a layout of different rank.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, Idx... idx) const
   {
      // TODO verify that idx0 < size0 && idx1 < size1
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return layout.index(idx...);
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<Rank,"Accessed size is higher than the rank of the Tensor.");
      return DynamicBlockLayoutSize<N,Rank>::eval(size0,size1,layout);
   }
};

template <int BatchSize>
class Dynamic2dThreadLayout<1,BatchSize>
{
private:
   const int size0;
public:
   MFEM_HOST_DEVICE inline
   Dynamic2dThreadLayout(int size0)
   : size0(size0)
   {
      // TODO verify that size0 < BlockSizeX
      // TODO verify that BlockSizeZ == BatchSize
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Dynamic2dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>())
   {
      // static_assert(1 == get_layout_rank<Layout>,
      //               "Can't copy-construct with a layout of different rank.");
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx) const
   {
      // TODO verify that idx < DimX
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N==0,"Accessed size is higher than the rank of the Tensor.");
      return size0;
   }
};

template <int BatchSize>
class Dynamic2dThreadLayout<2,BatchSize>
{
private:
   const int size0;
   const int size1;
public:
   MFEM_HOST_DEVICE inline
   Dynamic2dThreadLayout(int size0, int size1)
   : size0(size0), size1(size1)
   {
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that BlockSizeZ == BatchSize
   }

   template <typename Layout> MFEM_HOST_DEVICE
   Dynamic2dThreadLayout(const Layout &rhs)
   : size0(rhs.template Size<0>()),
     size1(rhs.template Size<1>())
   {
      // static_assert(2 == get_layout_rank<Layout>,
      //               "Can't copy-construct with a layout of different rank.");
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1) const
   {
      // TODO verify that idx0 < size0 && idx1 < size1
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return 0;
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<2,"Accessed size is higher than the rank of the Tensor.");
      return N==0? size0 : size1;
   }
};

// get_layout_rank
template <int Rank, int BatchSize>
struct get_layout_rank_v<Dynamic2dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = Rank;
};

// is_dynamic_layout
template <int Rank, int BatchSize>
struct is_dynamic_layout_v<Dynamic2dThreadLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};

// is_2d_threaded_layout
template <int Rank, int BatchSize>
struct is_2d_threaded_layout_v<Dynamic2dThreadLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};

// is_serial_layout_dim
template <int Rank, int BatchSize>
struct is_serial_layout_dim_v<Dynamic2dThreadLayout<Rank,BatchSize>, 0>
{
   static constexpr bool value = false;
};

template <int Rank, int BatchSize>
struct is_serial_layout_dim_v<Dynamic2dThreadLayout<Rank,BatchSize>, 1>
{
   static constexpr bool value = false;
};

// is_threaded_layout_dim
template <int Rank, int BatchSize>
struct is_threaded_layout_dim_v<Dynamic2dThreadLayout<Rank,BatchSize>, 0>
{
   static constexpr bool value = true;
};

template <int Rank, int BatchSize>
struct is_threaded_layout_dim_v<Dynamic2dThreadLayout<Rank,BatchSize>, 1>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int Rank, int BatchSize>
struct get_layout_size_v<N, Dynamic2dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = Dynamic;
};

// get_layout_sizes
template <int Rank, int BatchSize>
struct get_layout_sizes_t<Dynamic2dThreadLayout<Rank,BatchSize>>
{
   using type = int_repeat<Dynamic,Rank>;
};

// get_layout_batch_size
template <int Rank, int BatchSize>
struct get_layout_batch_size_v<Dynamic2dThreadLayout<Rank, BatchSize>>
{
   static constexpr int value = BatchSize;
};

// get_layout_result_type
template <int Rank, int BatchSize>
struct get_layout_result_type<Dynamic2dThreadLayout<Rank,BatchSize>>
{
   template <int myRank>
   using type = Dynamic2dThreadLayout<myRank,BatchSize>;
};

} // namespace mfem

#endif // MFEM_DYNAMIC_2DTHREAD_LAYOUT
