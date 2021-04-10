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

#ifndef MFEM_DIAG_TENSOR
#define MFEM_DIAG_TENSOR

#include "tensor.hpp"

namespace mfem
{

/// Represent a Rank+2*DRank diagonal Tensor, where DRank is the diagonal rank.
template <int DRank, // The rank of diagonal values
          int Rank, // The rank of non-diagonal values
          typename T = double,
          typename Container = MemoryContainer<T>,
          typename Layout = DynamicLayout<Rank> >
class DiagonalTensor: public Tensor<DRank+Rank,T,Container,Layout>
{
public:
   DiagonalTensor(const Tensor<DRank+Rank,T,Container,Layout> &t)
   : Tensor<DRank+Rank,T,Container,Layout>(t)
   { }

   // TODO define a DRank accessor? probably not possible
   // private inheritance then?
};

template <int DRank, typename Tensor>
auto makeDiagonalTensor(const Tensor &t)
{
   return DiagonalTensor<DRank,
                         get_tensor_rank<Tensor>::value-DRank,
                         typename Tensor::type,
                         typename Tensor::container,
                         typename Tensor::layout
                        >(t);
}

/// DiagonalTensor Traits

// is_diagonal_tensor
template <typename Tensor>
struct is_diagonal_tensor
{
   static constexpr bool value = false;
};

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct is_diagonal_tensor<DiagonalTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr bool value = true;
};

// get_diagonal_tensor_rank
template <typename Tensor>
struct get_diagonal_tensor_rank;

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct get_diagonal_tensor_rank<DiagonalTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr int value = 2*DRank + Rank;
};

// get_diagonal_tensor_diagonal_rank
template <typename Tensor>
struct get_diagonal_tensor_diagonal_rank;

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct get_diagonal_tensor_diagonal_rank<DiagonalTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr int value = DRank;
};

// get_diagonal_tensor_values_rank
template <typename Tensor>
struct get_diagonal_tensor_values_rank;

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct get_diagonal_tensor_values_rank<DiagonalTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr int value = Rank;
};

/// Represent a SRank+2*DRank symmetric Tensor, where SRank dims are symmetric.
template <int DRank, // The rank of diagonal values
          int SRank, // The rank of symmetric values
          typename T = double,
          typename Container = MemoryContainer<T>,
          typename Layout = DynamicLayout<1> >
class DiagonalSymmetricTensor: public Tensor<DRank+SRank,T,Container,Layout>
{
public:
   DiagonalSymmetricTensor(const Tensor<DRank+SRank,T,Container,Layout> &t)
   : Tensor<DRank+SRank,T,Container,Layout>(t)
   { }

   // TODO define a DRank accessor? probably not possible
   // private inheritance then?
};

template <int DRank, typename Tensor>
auto makeDiagonalSymmetricTensor(const Tensor &t)
{
   return DiagonalSymmetricTensor<DRank,
                                  get_tensor_rank<Tensor>::value-DRank,
                                  typename Tensor::type,
                                  typename Tensor::container,
                                  typename Tensor::layout
                                 >(t);
}

/// DiagonalSymmetricTensor Traits

// is_diagonal_symmetric_tensor
template <typename Tensor>
struct is_diagonal_symmetric_tensor
{
   static constexpr bool value = false;
};

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct is_diagonal_symmetric_tensor<DiagonalSymmetricTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr bool value = true;
};

// get_diagonal_symmetric_tensor_rank
template <typename Tensor>
struct get_diagonal_symmetric_tensor_rank;

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct get_diagonal_symmetric_tensor_rank<DiagonalSymmetricTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr int value = 2*DRank + 2*Rank;
};

// get_diagonal_symmetric_tensor_diagonal_rank
template <typename Tensor>
struct get_diagonal_symmetric_tensor_diagonal_rank;

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct get_diagonal_symmetric_tensor_diagonal_rank<DiagonalSymmetricTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr int value = DRank;
};

// get_diagonal_symmetric_tensor_values_rank
template <typename Tensor>
struct get_diagonal_symmetric_tensor_values_rank;

template <int DRank, int Rank, typename T, typename Container, typename Layout>
struct get_diagonal_symmetric_tensor_values_rank<DiagonalSymmetricTensor<DRank, Rank, T, Container, Layout>>
{
   static constexpr int value = Rank;
};

// /// Represent a SRank+2*DRank symmetric Tensor, where SRank dims are symmetric.
// template <int DRank, // The rank of diagonal values
//           int SRank, // The rank of symmetric values
//           typename T = double,
//           typename Container = MemoryContainer<T>,
//           typename Layout = DynamicLayout<1> >
// class DiagonalSymmetricTensor: public Tensor<DRank+1,T,Container,Layout>
// {
// public:
//    // Storing the symmetric values linearly for the moment
//    DiagonalSymmetricTensor(const Tensor<DRank+1,T,Container,Layout> &t)
//    : Tensor<DRank+1,T,Container,Layout>(t)
//    { }

//    // TODO define a DRank accessor? probably not possible
//    // private inheritance then?
// };

// template <int DRank, typename Tensor>
// auto makeDiagonalSymmetricTensor(const Tensor &t)
// {
//    return DiagonalSymmetricTensor<DRank,
//                                   Tensor::rank-DRank,
//                                   typename Tensor::type,
//                                   typename Tensor::container,
//                                   typename Tensor::layout
//                                  >(t);
// }

} // namespace mfem

#endif // MFEM_DIAG_TENSOR
