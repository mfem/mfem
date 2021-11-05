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

#ifndef MFEM_RESTRICTED_LAYOUT
#define MFEM_RESTRICTED_LAYOUT

#include "../../../general/error.hpp"
#include "layout_traits.hpp"

namespace mfem
{

// TODO possible to write directly in a generic way?
// TODO throw an error if N>8?
template <int N, typename Layout>
class RestrictedLayout;

template <typename Layout>
class RestrictedLayout<0,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<0>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(Idx... idx) const
   {
      return layout.index(i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   int Size() const
   {
      return layout.template Size<M+1>();
   }
};

template <typename Layout>
class RestrictedLayout<1,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<1>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, Idx... idx) const
   {
      return layout.index(idx0,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<1?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<2,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<2>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, Idx... idx) const
   {
      return layout.index(idx0,idx1,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<2?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<3,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<3>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<3?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<4,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<4>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<4?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<5,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<5>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, int idx4, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<5?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<6,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<6>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, int idx4, int idx5, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,idx5,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<6?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<7,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<7>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, int idx4, int idx5, int idx6, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,idx5,idx6,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<7?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<8,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i)
   {
      MFEM_ASSERT_KERNEL(
         i<layout.template Size<8>(),
         "The RestrictedLayout is out of bounds.");
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3,int idx4, int idx5, int idx6, int idx7, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7,i,idx...);
   }

   template <int M> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      return layout.template Size<(M<8?M:M+1)>();
   }
};

// get_layout_rank
template <int I, typename Layout>
struct get_layout_rank_v<RestrictedLayout<I,Layout>>
{
   static constexpr int value = get_layout_rank<Layout> - 1;
};

// is_dynamic_layout
template <int N, typename Layout>
struct is_dynamic_layout_v<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_dynamic_layout<Layout>;
};

// is_static_layout
template <int N, typename Layout>
struct is_static_layout_v<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_static_layout<Layout>;
};

// is_serial_layout
template <int N, typename Layout>
struct is_serial_layout_v<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_serial_layout<Layout>;
};

// is_2d_threaded_layout
template <int N, typename Layout>
struct is_2d_threaded_layout_v<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_2d_threaded_layout<Layout>;
};

// is_3d_threaded_layout
template <int N, typename Layout>
struct is_3d_threaded_layout_v<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_3d_threaded_layout<Layout>;
};

// is_threaded_layout_dim
template <int N, int R, typename Layout>
struct is_threaded_layout_dim_v<RestrictedLayout<R,Layout>, N>
{
   static constexpr bool value = N<R?
                                 is_threaded_layout_dim<Layout,N>:
                                 is_threaded_layout_dim<Layout,N+1>;
};

// get_layout_size
template <int N, int I, typename Layout>
struct get_layout_size_v<N,RestrictedLayout<I,Layout>>
{
   static constexpr int value = get_layout_size<N+(N>=I),Layout>;
};

// get_layout_sizes
template <int N, typename Layout>
struct get_layout_sizes_t<RestrictedLayout<N,Layout>>
{
   using type = remove< N, get_layout_sizes<Layout> >;
};

// get_layout_batch_size
template <int N, typename Layout>
struct get_layout_batch_size_v<RestrictedLayout<N, Layout>>
{
   static constexpr int value = get_layout_batch_size<Layout>;
};

// get_layout_capacity
template <int N, typename Layout>
struct get_layout_capacity_v<RestrictedLayout<N,Layout>>
{
   static constexpr int capacity = get_layout_capacity<Layout>;
   static constexpr int sizeN = get_layout_size<N,Layout>;
   static constexpr int value = sizeN != Dynamic ?
                                         ( capacity / sizeN) :
                                         Dynamic;
};

// get_layout_result_type
template <int N, typename Layout>
struct get_layout_result_type< RestrictedLayout<N,Layout> >
: public get_restricted_layout_result_type<Layout>
{ };

} // namespace mfem

#endif // MFEM_RESTRICTED_LAYOUT
