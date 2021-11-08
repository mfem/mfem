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

#ifndef MFEM_TENSOR_IDENTITY
#define MFEM_TENSOR_IDENTITY

#include "../../../general/forall.hpp"

namespace mfem
{

struct Identity { };

template <typename Tensor,
          std::enable_if_t<
             is_tensor<Tensor> &&
             !std::is_same<Tensor, ResultTensor<Tensor>>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Identity &I, const Tensor &u)
{
   using Result = get_identity_result_type<Tensor>;
   return Result(u);
}

template <typename Tensor,
          std::enable_if_t<
             is_tensor<Tensor> &&
             std::is_same<Tensor, ResultTensor<Tensor>>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Identity &I, const Tensor &u)
{
   return u;
}

} // namespace mfem

#endif // MFEM_TENSOR_IDENTITY
