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

#ifndef MFEM_TENSOR_FOREACH
#define MFEM_TENSOR_FOREACH

#include "../../general/forall.hpp"
#include "tensor_traits.hpp"

namespace mfem
{

/// A structure that implements a specialized `for' loop for a specific dimension.
template <int N>
struct Foreach
{
   /// Apply a lambda function based on the dimensions of a Tensor.
   template <typename Tensor,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_serial_tensor_dim<Tensor, N>,
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void Apply(Tensor &t, Lambda &&func, Idx... idx)
   {
      for (int i = 0; i < t.template Size<N>(); i++)
      {
         func(i,idx...);
      }
   }

   template <typename Tensor,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_threaded_tensor_dim<Tensor, N>, // tensor_traits<Tensor>::is_serial_dim<2>
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void Apply(Tensor &t, Lambda &&func, Idx... idx)
   {
      MFEM_FOREACH_THREAD(i,y,t.template Size<N>())
      {
         func(i,idx...);
      }
   }
};

/// A structure that applies Foreach to all given dimensions.
template <int Dim, int... Dims>
struct Forall
{
   /// Apply a lambda function based on the dimensions of a Tensor.
   template <typename Tensor, typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void Apply(Tensor &t, Lambda &&func, Idx... idx)
   {
      Foreach<Dim>::Apply(
         t,
         [&](auto... i){
            Forall<Dims...>::Apply(t, func, i...);
         },
         idx...
      );
   }
};

template <int Dim>
struct Forall<Dim>
{
   /// Apply a lambda function based on the dimensions of a Tensor.
   template <typename Tensor, typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void Apply(Tensor &t, Lambda &&func, Idx... idx)
   {
      Foreach<Dim>::Apply(t, func, idx...);
   }
};

/// A structure applying Foreach to each dimension of the Tensor.
template <typename Tensor, int Dim = get_tensor_rank<Tensor>-1>
struct ForallDims
{
   /// Apply a lambda function based on the dimensions of a Tensor.
   template <typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void Apply(Tensor &t, Lambda &&func, Idx... idx)
   {
      Foreach<Dim>::Apply(
         t,
         [&](auto... i){
            ForallDims<Tensor,Dim-1>::Apply(t, func, i...);
         },
         idx...
      );
   }
};

template <typename Tensor>
struct ForallDims<Tensor, 0>
{
   /// Apply a lambda function based on the dimensions of a Tensor.
   template <typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void Apply(Tensor &t, Lambda &&func, Idx... idx)
   {
      Foreach<0>::Apply(t, func, idx...);
   }
};

} // namespace mfem

#endif // MFEM_TENSOR_FOREACH
