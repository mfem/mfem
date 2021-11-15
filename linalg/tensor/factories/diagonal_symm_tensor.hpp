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

#ifndef MFEM_DIAG_SYMM_TENSOR
#define MFEM_DIAG_SYMM_TENSOR

#include "../tensor.hpp"

namespace mfem
{

/// Represent a SRank+2*DRank symmetric Tensor, where SRank dims are symmetric.
template <int DRank, // The rank of diagonal values
          int SRank, // The rank of symmetric values
          typename Container,
          typename Layout>
class DiagonalSymmetricTensor: public Tensor<Container,Layout>
{
public:
   MFEM_HOST_DEVICE
   DiagonalSymmetricTensor(const Tensor<Container,Layout> &t)
   : Tensor<Container,Layout>(t)
   { }
};

template <int DRank, typename Tensor> MFEM_HOST_DEVICE inline
auto makeDiagonalSymmetricTensor(const Tensor &t)
{
   return DiagonalSymmetricTensor<DRank,
                                  get_tensor_rank<Tensor>-DRank,
                                  typename Tensor::container,
                                  typename Tensor::layout
                                 >(t);
}

/// DiagonalSymmetricTensor Traits

// is_diagonal_symmetric_tensor
template <typename NotADiagSymmTensor>
struct is_diagonal_symmetric_tensor_v
{
   static constexpr bool value = false;
};

template <int DRank, int Rank, typename Container, typename Layout>
struct is_diagonal_symmetric_tensor_v<DiagonalSymmetricTensor<DRank, Rank, Container, Layout>>
{
   static constexpr bool value = true;
};

template <typename Tensor>
constexpr bool is_diagonal_symmetric_tensor = is_diagonal_symmetric_tensor_v<Tensor>::value;

// get_diagonal_symmetric_tensor_rank
template <typename NotADiagSymmTensor>
struct get_diagonal_symmetric_tensor_rank_v
{
   static constexpr int value = Error;
};

template <int DRank, int Rank, typename Container, typename Layout>
struct get_diagonal_symmetric_tensor_rank_v<DiagonalSymmetricTensor<DRank, Rank, Container, Layout>>
{
   static constexpr int value = 2*DRank + 2*Rank;
};

template <typename Tensor>
constexpr int get_diagonal_symmetric_tensor_rank = get_diagonal_symmetric_tensor_rank_v<Tensor>::value;

// get_diagonal_symmetric_tensor_diagonal_rank
template <typename NotADiagSymmTensor>
struct get_diagonal_symmetric_tensor_diagonal_rank_v
{
   static constexpr int value = Error;
};

template <int DRank, int Rank, typename Container, typename Layout>
struct get_diagonal_symmetric_tensor_diagonal_rank_v<DiagonalSymmetricTensor<DRank, Rank, Container, Layout>>
{
   static constexpr int value = DRank;
};

template <typename Tensor>
constexpr int get_diagonal_symmetric_tensor_diagonal_rank = get_diagonal_symmetric_tensor_diagonal_rank_v<Tensor>::value;

// get_diagonal_symmetric_tensor_values_rank
template <typename NotADiagSymmTensor>
struct get_diagonal_symmetric_tensor_values_rank_v
{
   static constexpr int value = Error;
};

template <int DRank, int Rank, typename Container, typename Layout>
struct get_diagonal_symmetric_tensor_values_rank_v<DiagonalSymmetricTensor<DRank, Rank, Container, Layout>>
{
   static constexpr int value = Rank;
};

template <typename Tensor>
constexpr int get_diagonal_symmetric_tensor_values_rank = get_diagonal_symmetric_tensor_values_rank_v<Tensor>::value;

// get_tensor_size
template <int N, int DRank, int SRank, typename C, typename Layout>
struct get_tensor_size_v<N,DiagonalSymmetricTensor<DRank, SRank, C, Layout>>
{
   static constexpr int value = get_layout_size<N, Layout>;
};

// is_serial_tensor_dim
template <int N, int DRank, int VRank, typename C, typename Layout>
struct is_serial_tensor_dim_v<N,DiagonalSymmetricTensor<DRank, VRank, C, Layout>>
{
   static constexpr bool value = is_serial_layout_dim<Layout,N>;
};

// is_threaded_tensor_dim
template <int N, int DRank, int VRank, typename C, typename Layout>
struct is_threaded_tensor_dim_v<N,DiagonalSymmetricTensor<DRank, VRank, C, Layout>>
{
   static constexpr bool value = is_threaded_layout_dim<Layout,N>;
};

} // namespace mfem

#endif // MFEM_DIAG_SYMM_TENSOR
