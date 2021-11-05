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

#ifndef MFEM_TENSOR_SCALAR_MULT
#define MFEM_TENSOR_SCALAR_MULT

#include "../tensor_traits.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

template <typename Scalar,
          typename Tensor,
          std::enable_if_t<
             is_tensor<Tensor> &&
             std::is_same<Scalar, get_tensor_type<Tensor>>::value,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Scalar &a, const Tensor &u)
{
   using Res = ResultTensor<Tensor>;
   Res v(GetLayout(u));
   ForallDims<Tensor>::ApplyBinOp(u, v, [&](auto... idx)
   {
      v(idx...) = a * v(idx...);
   });
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_SCALAR_MULT
