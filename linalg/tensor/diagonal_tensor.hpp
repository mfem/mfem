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

/// Represent a 2*SRank+DRank symmetric Tensor, where SRank dims are symmetric.
template <int DRank, // The rank of diagonal values
          int SRank, // The rank of symmetric values
          typename T = double,
          typename Container = MemoryContainer<T>,
          typename Layout = DynamicLayout<SRank> >
class DiagonalTensor: public Tensor<DRank+SRank,T,Container,Layout>
{
public:
   DiagonalTensor(const Tensor<DRank+SRank,T,Container,Layout> &t)
   : Tensor<DRank+SRank,T,Container,Layout>(t)
   { }

   // TODO define a DRank accessor? probably not possible
   // private inheritance then?
};

template <int DRank, typename Tensor>
auto makeDiagonalTensor(const Tensor &t)
{
   return DiagonalTensor<DRank,
                         Tensor::rank-DRank,
                         typename Tensor::type,
                         typename Tensor::container,
                         typename Tensor::layout
                        >(t);
}

} // namespace mfem

#endif // MFEM_DIAG_TENSOR
