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

#ifndef MFEM_TENSOR_GET
#define MFEM_TENSOR_GET

#include "../containers/container_traits.hpp"
#include "../containers/view_container.hpp"
#include "../layouts/layout_traits.hpp"
#include "../layouts/restricted_layout.hpp"

namespace mfem
{

/// Lazy accessor for the sub-Tensor extracted from idx in Nth dimension.
template <int N, typename Container, typename Layout> MFEM_HOST_DEVICE inline
auto Get(int idx, Tensor<Container,Layout> &t)
{
   static_assert(N>=0 && N<get_layout_rank<Layout>,
      "Cannot access this dimension with Get");
   using T = get_container_type<Container>;
   using C = ViewContainer<T,Container>;
   using L = RestrictedLayout<N,Layout>;
   using RestrictedTensor = Tensor<C,L>;
   C data(t);
   L layout(idx,t);
   return RestrictedTensor(data,layout);
}

template <int N, typename Container, typename Layout> MFEM_HOST_DEVICE inline
auto Get(int idx, const Tensor<Container,Layout> &t)
{
   static_assert(N>=0 && N<get_layout_rank<Layout>,
      "Cannot access this dimension with Get");
   using T = get_container_type<Container>;
   using C = ConstViewContainer<T,Container>;
   using L = RestrictedLayout<N,Layout>;
   using RestrictedTensor = Tensor<C,L>;
   C data(t);
   L layout(idx,t);
   return RestrictedTensor(data,layout);
}

} // namespace mfem

#endif // MFEM_TENSOR_GET
