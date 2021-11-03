// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more inforAion and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_NORMS
#define MFEM_TENSOR_NORMS

#include "../../../general/forall.hpp"
#include "../tensor_traits.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

template <typename Tensor> MFEM_HOST_DEVICE inline
auto SquaredNorm(const Tensor& t)
{
   using Scalar = get_tensor_type<Tensor>;
   Scalar norm = 0;
   ForallDims<Tensor>::Apply(t, [&](auto... idx){
      Scalar& val = t(idx...);
      norm += val*val;
   });
   return norm;
}

} // namespace mfem

#endif // MFEM_TENSOR_NORMS
