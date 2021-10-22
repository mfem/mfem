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

#ifndef MFEM_LAYOUT_TRAITS
#define MFEM_LAYOUT_TRAITS

#include "../utilities/helper_constants.hpp"

namespace mfem
{

/////////////////
// Layout Traits

// get_layout_rank
template <typename Layout>
struct get_layout_rank_v;

template <typename Layout>
constexpr int get_layout_rank = get_layout_rank_v<Layout>::value;

// is_dynamic_layout
template <typename Layout>
struct is_dynamic_layout_v
{
   static constexpr bool value = false;
};

template <typename Layout>
constexpr bool is_dynamic_layout = is_dynamic_layout_v<Layout>::value;

// is_static_layout
template <typename Layout>
struct is_static_layout_v
{
   static constexpr bool value = false;
};

template <typename Layout>
constexpr bool is_static_layout = is_static_layout_v<Layout>::value;

// is_serial_layout
template <typename Layout>
struct is_serial_layout_v
{
   static constexpr bool value = false;
};

template <typename Layout>
constexpr bool is_serial_layout = is_serial_layout_v<Layout>::value;

// is_1d_threaded_layout
template <typename Layout>
struct is_1d_threaded_layout_v
{
   static constexpr bool value = false;
};

template <typename Layout>
constexpr bool is_1d_threaded_layout = is_1d_threaded_layout_v<Layout>::value;

// is_2d_threaded_layout
template <typename Layout>
struct is_2d_threaded_layout_v
{
   static constexpr bool value = false;
};

template <typename Layout>
constexpr bool is_2d_threaded_layout = is_2d_threaded_layout_v<Layout>::value;

// is_3d_threaded_layout
template <typename Layout>
struct is_3d_threaded_layout_v
{
   static constexpr bool value = false;
};

template <typename Layout>
constexpr bool is_3d_threaded_layout = is_3d_threaded_layout_v<Layout>::value;

// is_serial_layout_dim
template <typename Layout, int N>
struct is_serial_layout_dim_v
{
   static constexpr bool value = true;
};

template <typename Layout, int N>
constexpr bool is_serial_layout_dim = is_serial_layout_dim_v<Layout,N>::value;

// is_threaded_layout_dim
template <typename Layout, int N>
struct is_threaded_layout_dim_v
{
   static constexpr bool value = false;
};

template <typename Layout, int N>
constexpr bool is_threaded_layout_dim = is_threaded_layout_dim_v<Layout,N>::value;

// get_layout_size
template <int N, typename Layout>
struct get_layout_size_v;

template <int N, typename Layout>
constexpr int get_layout_size = get_layout_size_v<N,Layout>::value;

// get_layout_sizes
template <typename Layout>
struct get_layout_sizes_t;

template <typename Layout>
using get_layout_sizes = typename get_layout_sizes_t<Layout>::type;

// get_layout_batch_size
template <typename Layout>
struct get_layout_batch_size_v
{
   static constexpr int value = 1;
};

template <typename Layout>
constexpr int get_layout_batch_size = get_layout_batch_size_v<Layout>::value;

// get_layout_capacity
template <typename Layout>
struct get_layout_capacity_v
{
   static constexpr int rank = get_layout_rank<Layout>;
   static constexpr int value = pow(DynamicMaxSize, rank);
};

template <typename Layout>
constexpr int get_layout_capacity = get_layout_capacity_v<Layout>::value;

// get_layout_result_type
template <typename Layout>
struct get_layout_result_type;

template <typename Layout, typename Enable = void>
struct get_restricted_layout_result_type;

template <typename Layout>
struct get_restricted_layout_result_type<
   Layout,
   std::enable_if_t< is_static_layout<Layout> >
>
{
   template <int... Sizes>
   using type = typename get_layout_result_type<Layout>::template type<Sizes...>;
};

template <typename Layout>
struct get_restricted_layout_result_type<
   Layout,
   std::enable_if_t<is_dynamic_layout<Layout> >
>
{
   template <int Rank>
   using type = typename get_layout_result_type<Layout>::template type<Rank>;
};

} // namespace mfem

#endif // MFEM_LAYOUT_TRAITS
