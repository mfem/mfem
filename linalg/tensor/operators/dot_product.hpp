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

#ifndef MFEM_TENSOR_MAT_MULT
#define MFEM_TENSOR_MAT_MULT

#include "../../../general/forall.hpp"
#include "../tensor_traits.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

template <typename RHS,
          typename LHS,
          std::enable_if_t<
             get_tensor_rank<RHS> == get_tensor_rank<LHS> &&
             is_serial_tensor<RHS> &&
             is_serial_tensor<LHS>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto Dot(const LHS &lhs, const RHS &rhs)
{
   using Scalar = get_tensor_type<RHS>;
   Scalar res = 0;
   ForallDims<RHS>::ApplyBinOp(lhs, rhs, [&](auto... idx){
      res += lhs(idx...)*rhs(idx...);
   });
   return res;
}

template <typename RHS,
          typename LHS,
          std::enable_if_t<
             get_tensor_rank<RHS> == get_tensor_rank<LHS> &&
             (!is_serial_tensor<RHS> ||
              !is_serial_tensor<LHS>),
             bool> = true >
MFEM_HOST_DEVICE inline
auto Dot(const LHS &lhs, const RHS &rhs)
{
   using Scalar = get_tensor_type<RHS>;
   MFEM_SHARED Scalar res = 0;
   ForallDims<RHS>::ApplyBinOp(lhs, rhs, [&](auto... idx){
      AtomicAdd(&res, lhs(idx...)*rhs(idx...) );
   });
   return res;
}

} // namespace mfem

#endif // MFEM_TENSOR_MAT_MULT
