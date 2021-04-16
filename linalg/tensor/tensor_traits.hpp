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

#include "../../general/backends.hpp"

namespace mfem
{

template <int Rank, typename T, typename Container, typename Layout>
class Tensor;

/////////////////
// Tensor Traits

// get_tensor_rank
template <typename Tensor>
struct get_tensor_rank_v
{
   static constexpr int value = Error;
};

template <int Rank, typename T, typename C, typename L>
struct get_tensor_rank_v<Tensor<Rank,T,C,L>>
{
   static constexpr int value = Rank;
};

template <int Rank, typename T, typename C, typename L>
struct get_tensor_rank_v<const Tensor<Rank,T,C,L>>
{
   static constexpr int value = Rank;
};

template <typename Tensor>
constexpr int get_tensor_rank = get_tensor_rank_v<Tensor>::value;

// get_tensor_value_type
template <typename Tensor>
struct get_tensor_value_type_t;

template <int Rank, typename T, typename C, typename L>
struct get_tensor_value_type_t<Tensor<Rank,T,C,L>>
{
   using type = T;
};

template <int Rank, typename T, typename C, typename L>
struct get_tensor_value_type_t<const Tensor<Rank,T,C,L>>
{
   using type = T;
};

template <typename Tensor>
using get_tensor_value_type = typename get_tensor_value_type_t<Tensor>::type;

// is_dynamic_tensor
template <typename Tensor>
struct is_dynamic_tensor_v
{
   static constexpr bool value = false;
};

template <int Rank, typename T, typename C, typename L>
struct is_dynamic_tensor_v<Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_dynamic_layout<L>::value;
};

template <int Rank, typename T, typename C, typename L>
struct is_dynamic_tensor_v<const Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_dynamic_layout<L>::value;
};

template <typename Tensor>
constexpr bool is_dynamic_tensor = is_dynamic_tensor_v<Tensor>::value;

// is_static_tensor
template <typename Tensor>
struct is_static_tensor_v
{
   static constexpr bool value = false;
};

template <int Rank, typename T, typename C, typename L>
struct is_static_tensor_v<Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_static_layout<L>::value;
};

template <int Rank, typename T, typename C, typename L>
struct is_static_tensor_v<const Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_static_layout<L>::value;
};

template <typename Tensor>
constexpr bool is_static_tensor = is_static_tensor_v<Tensor>::value;

// is_serial_tensor
template <typename Tensor>
struct is_serial_tensor_v
{
   static constexpr bool value = false;
};

template <int Rank, typename T, typename C, typename L>
struct is_serial_tensor_v<Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_serial_layout<L>::value;
};

template <int Rank, typename T, typename C, typename L>
struct is_serial_tensor_v<const Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_serial_layout<L>::value;
};

template <typename Tensor>
constexpr bool is_serial_tensor = is_serial_tensor_v<Tensor>::value;

// is_2d_threaded_tensor
template <typename Tensor>
struct is_2d_threaded_tensor_v
{
   static constexpr bool value = false;
};

template <int Rank, typename T, typename C, typename L>
struct is_2d_threaded_tensor_v<Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_2d_threaded_layout<L>::value;
};

template <int Rank, typename T, typename C, typename L>
struct is_2d_threaded_tensor_v<const Tensor<Rank,T,C,L>>
{
   static constexpr bool value = is_2d_threaded_layout<L>::value;
};

template <typename Tensor>
constexpr bool is_2d_threaded_tensor = is_2d_threaded_tensor_v<Tensor>::value;

// get_tensor_size
template <int N, typename Tensor>
struct get_tensor_size_v
{
   static constexpr int value = Error;
};

template <int N, int R, typename T, typename C, typename L>
struct get_tensor_size_v<N, Tensor<R,T,C,L>>
{
   static constexpr int value = get_layout_size<N, L>::value;
};

template <int N, int R, typename T, typename C, typename L>
struct get_tensor_size_v<N, const Tensor<R,T,C,L>>
{
   static constexpr int value = get_layout_size<N, L>::value;
};

template <int N, typename Tensor>
constexpr int get_tensor_size = get_tensor_size_v<N, Tensor>::value;

// get_tensor_batch_size
template <typename Tensor>
struct get_tensor_batch_size_v
{
   static constexpr int value = Error;
};

template <int Rank, typename T, typename C, typename L>
struct get_tensor_batch_size_v<Tensor<Rank,T,C,L>>
{
   static constexpr int value = get_layout_batch_size<L>::value;
};

template <int Rank, typename T, typename C, typename L>
struct get_tensor_batch_size_v<const Tensor<Rank,T,C,L>>
{
   static constexpr int value = get_layout_batch_size<L>::value;
};

template <typename Tensor>
constexpr int get_tensor_batch_size = get_tensor_batch_size_v<Tensor>::value;

// has_pointer_container
template <typename Tensor>
struct has_pointer_container_v
{
   static constexpr bool value = false;
};

template <int R, typename T, typename C, typename L>
struct has_pointer_container_v<Tensor<R,T,C,L>>
{
   static constexpr bool value = is_pointer_container<C>::value;
};

template <int R, typename T, typename C, typename L>
struct has_pointer_container_v<const Tensor<R,T,C,L>>
{
   static constexpr bool value = is_pointer_container<C>::value;
};

template <typename Tensor>
constexpr bool has_pointer_container = has_pointer_container_v<Tensor>::value;

// is_static_matrix
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

// is_dynamic_matrix
template <typename Tensor>
struct is_dynamic_matrix_v
{
   static constexpr bool value = is_dynamic_tensor<Tensor> &&
                                 get_tensor_rank<Tensor> == 2;
};

template <typename Tensor>
constexpr bool is_dynamic_matrix = is_dynamic_matrix_v<Tensor>::value;

} // namespace mfem

#endif // MFEM_TENSOR_TRAITS
