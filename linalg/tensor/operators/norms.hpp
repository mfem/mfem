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

#include "../../../general/backends.hpp"
#include "../tensor_traits.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

template <typename Tensor,
          std::enable_if_t<
             is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto SquaredNorm(const Tensor& t)
{
   using Scalar = get_tensor_type<Tensor>;
   Scalar norm = 0;
   ForallDims<Tensor>::Apply(t, [&](auto... idx){
      const Scalar& val = t(idx...);
      norm += val*val;
   });
   return norm;
}

template <typename Tensor,
          std::enable_if_t<
             !is_serial_tensor<Tensor>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto SquaredNorm(const Tensor& t)
{
   using Scalar = get_tensor_type<Tensor>;
   MFEM_SHARED Scalar res;
   if (MFEM_THREAD_ID(x)==0 && MFEM_THREAD_ID(y)==0 && MFEM_THREAD_ID(z)==0)
   {
      res = 0.0;
   }
   MFEM_SYNC_THREAD;
   Scalar norm = 0;
   ForallDims<Tensor>::Apply(t, [&](auto... idx){
      const Scalar& val = t(idx...);
      norm += val*val;
   });
   AtomicAdd(res, norm);
   MFEM_SYNC_THREAD;
   return res;
}

} // namespace mfem

#endif // MFEM_TENSOR_NORMS
