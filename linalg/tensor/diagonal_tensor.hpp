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

template <int Rank,
          typename T = double,
          typename Container = MemoryContainer<T>,
          typename Layout = DynamicLayout<Rank> >
class DiagonalTensor: public Tensor<Rank,T,Container,Layout>
{

};

} // namespace mfem

#endif // MFEM_DIAG_TENSOR
