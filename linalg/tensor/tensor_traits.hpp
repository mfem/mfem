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

#ifndef MFEM_TENSOR_TRAITS
#define MFEM_TENSOR_TRAITS

#include "containers/container_traits.hpp"
#include "layouts/layout_traits.hpp"
#include "containers/static_container.hpp"

namespace mfem
{

/// Forward declaration of the Tensor class
template <typename Container, typename Layout>
class Tensor;

/**
 * Tensor Traits:
 *    Traits can be seen as compilation time functions operating on types.
 * There is two types of traits, traits returning values, and traits returning
 * types. The following traits allow to analyze tensors at compilation.
*/

/// is_tensor
template <typename NotATensor>
struct is_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = true;
};

template <typename Tensor>
constexpr bool is_tensor = is_tensor_v<Tensor>::value;

/// get_tensor_rank
/** Trait to get the rank of a tensor at compilation.
    ex: `constexpr int Rank = get_tensor_rank<Tensor>;'
*/
template <typename NotATensor>
struct get_tensor_rank_v
{
   static constexpr int value = Error;
};

template <typename Container, typename Layout>
struct get_tensor_rank_v<Tensor<Container,Layout>>
{
   static constexpr int value = get_layout_rank<Layout>;
};

template <typename Tensor>
constexpr int get_tensor_rank = get_tensor_rank_v<Tensor>::value;

/// get_tensor_value_type
/** Return the type of values stored by the Tensor's container.
    ex: `using T = get_tensor_value_type<Tensor>;'
*/
template <typename Tensor>
struct get_tensor_value_type_t;

template <typename C, typename L>
struct get_tensor_value_type_t<Tensor<C,L>>
{
   using type = get_container_type<C>;
};

template <typename C, typename L>
struct get_tensor_value_type_t<const Tensor<C,L>>
{
   using type = get_container_type<C>;
};

template <typename Tensor>
using get_tensor_value_type = typename get_tensor_value_type_t<Tensor>::type;

template <typename Tensor>
using get_tensor_type = typename get_tensor_value_type_t<Tensor>::type;

/// is_dynamic_tensor
/** Return true if the tensor's layout is dynamically sized.
    ex: `constexpr bool is_dynamic = is_dynamic_tensor<Tensor>;'
*/
template <typename Tensor>
struct is_dynamic_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_dynamic_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_dynamic_layout<L>;
};

template <typename C, typename L>
struct is_dynamic_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_dynamic_layout<L>;
};

template <typename Tensor>
constexpr bool is_dynamic_tensor = is_dynamic_tensor_v<Tensor>::value;

/// is_static_tensor
/** Return true if the tensor's layout is statically sized.
*/
template <typename Tensor>
struct is_static_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_static_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_static_layout<L>;
};

template <typename C, typename L>
struct is_static_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_static_layout<L>;
};

template <typename Tensor>
constexpr bool is_static_tensor = is_static_tensor_v<Tensor>::value;

/// is_serial_tensor
/** Return true if the tensor's is not distributed over threads.
*/
template <typename Tensor>
struct is_serial_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_serial_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_serial_layout<L>;
};

template <typename C, typename L>
struct is_serial_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_serial_layout<L>;
};

template <typename Tensor>
constexpr bool is_serial_tensor = is_serial_tensor_v<Tensor>::value;

/// is_1d_threaded_tensor
/** Return true if the tensor's layout is 1d threaded.
*/
template <typename Tensor>
struct is_1d_threaded_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_1d_threaded_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_1d_threaded_layout<L>;
};

template <typename C, typename L>
struct is_1d_threaded_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_1d_threaded_layout<L>;
};

template <typename Tensor>
constexpr bool is_1d_threaded_tensor = is_1d_threaded_tensor_v<Tensor>::value;

/// is_2d_threaded_tensor
/** Return true if the tensor's layout is 2d threaded.
*/
template <typename Tensor>
struct is_2d_threaded_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_2d_threaded_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_2d_threaded_layout<L>;
};

template <typename C, typename L>
struct is_2d_threaded_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_2d_threaded_layout<L>;
};

template <typename Tensor>
constexpr bool is_2d_threaded_tensor = is_2d_threaded_tensor_v<Tensor>::value;

/// is_3d_threaded_tensor
/** Return true if the tensor's layout is 3d threaded.
*/
template <typename Tensor>
struct is_3d_threaded_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_3d_threaded_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_3d_threaded_layout<L>;
};

template <typename C, typename L>
struct is_3d_threaded_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_3d_threaded_layout<L>;
};

template <typename Tensor>
constexpr bool is_3d_threaded_tensor = is_3d_threaded_tensor_v<Tensor>::value;

/// is_serial_tensor_dim
/** Return true if the tensor's layout dimension N is not threaded.
*/
template <int N, typename NotATensor>
struct is_serial_tensor_dim_v;

template <int N, typename C, typename Layout>
struct is_serial_tensor_dim_v<N,Tensor<C, Layout>>
{
   static constexpr bool value = is_serial_layout_dim<Layout,N>;
};

template <int N, typename Tensor>
constexpr bool is_serial_tensor_dim = is_serial_tensor_dim_v<N,Tensor>::value;

/// is_threaded_tensor_dim
/** Return true if the tensor's layout dimension N is threaded.
*/
template <int N, typename NotATensor>
struct is_threaded_tensor_dim_v;

template <int N, typename C, typename Layout>
struct is_threaded_tensor_dim_v<N,Tensor<C, Layout>>
{
   static constexpr bool value = is_threaded_layout_dim<Layout,N>;
};

template <int N, typename Tensor>
constexpr bool is_threaded_tensor_dim = is_threaded_tensor_dim_v<N,Tensor>::value;

/// get_tensor_size
/** Return the compilation time size of the dimension N, returns Dynamic for
    dynamic sizes.
*/
template <int N, typename Tensor>
struct get_tensor_size_v;
// {
//    static constexpr int value = Error;
// };

template <int N, typename C, typename L>
struct get_tensor_size_v<N, Tensor<C,L>>
{
   static constexpr int value = get_layout_size<N, L>;
};

template <int N, typename C, typename L>
struct get_tensor_size_v<N, const Tensor<C,L>>
{
   static constexpr int value = get_layout_size<N, L>;
};

template <int N, typename Tensor>
constexpr int get_tensor_size = get_tensor_size_v<N, Tensor>::value;

// get_tensor_sizes
template <typename NotATensor>
struct get_tensor_sizes_t;

template <typename C, typename Layout>
struct get_tensor_sizes_t<Tensor<C,Layout>>
{
   using type = get_layout_sizes<Layout>;
};

template <typename Tensor>
using get_tensor_sizes = typename get_tensor_sizes_t<Tensor>::type;

/// get_tensor_batch_size
/** Return the tensor's batchsize, the batchsize being the number of elements
    treated per block of threads.
*/
template <typename Tensor>
struct get_tensor_batch_size_v
{
   static constexpr int value = Error;
};

template <typename C, typename L>
struct get_tensor_batch_size_v<Tensor<C,L>>
{
   static constexpr int value = get_layout_batch_size<L>;
};

template <typename C, typename L>
struct get_tensor_batch_size_v<const Tensor<C,L>>
{
   static constexpr int value = get_layout_batch_size<L>;
};

template <typename Tensor>
constexpr int get_tensor_batch_size = get_tensor_batch_size_v<Tensor>::value;

/// get_tensor_capacity
/** Return the number of values stored per thread.
*/
template <typename Tensor>
struct get_tensor_capacity_v
{
   static constexpr int value = Error;
};

template <typename C, typename L>
struct get_tensor_capacity_v<Tensor<C,L>>
{
   static constexpr int value = get_layout_capacity<L>;
};

template <typename C, typename L>
struct get_tensor_capacity_v<const Tensor<C,L>>
{
   static constexpr int value = get_layout_capacity<L>;
};

template <typename Tensor>
constexpr int get_tensor_capacity = get_tensor_capacity_v<Tensor>::value;

/// has_pointer_container
/** Return true if the tensor's container is a pointer type.
*/
template <typename Tensor>
struct has_pointer_container_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct has_pointer_container_v<Tensor<C,L>>
{
   static constexpr bool value = is_pointer_container<C>;
};

template <typename C, typename L>
struct has_pointer_container_v<const Tensor<C,L>>
{
   static constexpr bool value = is_pointer_container<C>;
};

template <typename Tensor>
constexpr bool has_pointer_container = has_pointer_container_v<Tensor>::value;

/// is_static_matrix
/** Return true if the tensor is a statically sized matrix.
*/
template <int N, int M, typename Tensor>
struct is_static_matrix_v
{
   static constexpr bool value = is_static_tensor<Tensor> &&
                                 get_tensor_rank<Tensor> == 2 &&
                                 get_tensor_size<0,Tensor> == N &&
                                 get_tensor_size<1,Tensor> == M;
};

template <int N, int M, typename Tensor>
constexpr bool is_static_matrix = is_static_matrix_v<N,M,Tensor>::value;

/// is_dynamic_matrix
/** Return true if the tensor is a dynamically sized matrix.
*/
template <typename Tensor>
struct is_dynamic_matrix_v
{
   static constexpr bool value = is_dynamic_tensor<Tensor> &&
                                 get_tensor_rank<Tensor> == 2;
};

template <typename Tensor>
constexpr bool is_dynamic_matrix = is_dynamic_matrix_v<Tensor>::value;


/// get_result_tensor
/** Return a tensor type with the given Sizes, or sized as the given Tensor if
    no sizes are provided. This is used to abstract the tensor result type of an
    unknown input tensor type, i.e. Allows to write algorithms which are
    agnostic of the input type Tensor.
    ex:
    ```
    using ResTensor = get_result_tensor<Tensor>;
    using SizedResTensor = get_result_tensor<Tensor,Size0,Size1>;
    ```
*/
template <typename NotATensor, int... Sizes>
struct get_result_tensor_t;

template <typename Tensor, int... Sizes>
using get_result_tensor = typename get_result_tensor_t<Tensor,Sizes...>::type;

template <typename Tensor, int... Dims>
using ResultTensor = get_result_tensor<Tensor,Dims...>;

// Sizes deduced
template <typename Container, typename Layout>
struct get_result_tensor_t<Tensor<Container,Layout>>
{
   using T = get_container_type<Container>;
   template <int... Sizes>
   using unsized_layout = get_layout_result_type<Layout,Sizes...>;
   using sizes = get_layout_sizes<Layout>;
   using layout = instantiate<unsized_layout,sizes>;
   using container = StaticContainer<T,get_layout_capacity<layout>>;
   using type = Tensor<container, layout>;
};

// Sizes given
template <typename Container, typename Layout, int... Sizes>
struct get_result_tensor_t<Tensor<Container,Layout>,Sizes...>
{
   using T = get_container_type<Container>;
   template <int... Dims>
   using unsized_layout = get_layout_result_type<Layout,Dims...>;
   using layout = unsized_layout<Sizes...>;
   using container = StaticContainer<T,get_layout_capacity<layout>>;
   using type = Tensor<container, layout>;
};

} // namespace mfem

#endif // MFEM_TENSOR_TRAITS
