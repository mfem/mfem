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

/// A structure that implements an imbricated forall with for loops.
template <int N>
struct Foreach
{
   /// Apply an unary operator to each entry of a Tensor.
   template <typename TensorLHS, typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void ApplyUnOp(TensorLHS &lhs, Lambda &&func, Idx... idx)
   {
      for (int i = 0; i < lhs.template Size<N-1>(); i++)
      {
         Foreach<N-1>::ApplyUnOp(lhs,func,i,idx...);
      }
   }

   /// Apply a binary operator to each entry of a Tensor.
   template <typename TensorLHS, typename TensorRHS, typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void ApplyBinOp(TensorLHS &lhs, TensorRHS &rhs, Lambda &&func, Idx... idx)
   {
      for (int i = 0; i < lhs.template Size<N-1>(); i++)
      {
         Foreach<N-1>::ApplyBinOp(lhs,rhs,func,i,idx...);
      }
   }
};

template <>
struct Foreach<0>
{
   template <typename TensorLHS, typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void ApplyUnOp(TensorLHS &lhs, Lambda &&func, Idx... idx)
   {
      func(lhs,idx...);
   }

   template <typename TensorLHS, typename TensorRHS, typename Lambda, typename... Idx>
   MFEM_HOST_DEVICE inline
   static void ApplyBinOp(TensorLHS &lhs, TensorRHS &rhs, Lambda &&func, Idx... idx)
   {
      func(lhs,rhs,idx...);
   }
};

template <>
struct Foreach<1>
{
   template <typename TensorLHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_serial_tensor<TensorLHS>::value,
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyUnOp(TensorLHS &lhs, Lambda &&func, Idx... idx)
   {
      for (int i = 0; i < lhs.template Size<0>(); i++)
      {
         Foreach<0>::ApplyUnOp(lhs,func,i,idx...);
      }
   }

   template <typename TensorLHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_2d_threaded_tensor<TensorLHS>::value,
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyUnOp(TensorLHS &lhs, Lambda &&func, Idx... idx)
   {
      MFEM_FOREACH_THREAD(i,x,lhs.template Size<0>())
      {
         Foreach<0>::ApplyUnOp(lhs,func,i,idx...);
      }
   }

   template <typename TensorLHS,
             typename TensorRHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_serial_tensor<TensorLHS>::value &&
                is_serial_tensor<TensorRHS>::value,
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyBinOp(TensorLHS &lhs, TensorRHS &rhs, Lambda &&func, Idx... idx)
   {
      for (int i = 0; i < lhs.template Size<0>(); i++)
      {
         Foreach<0>::ApplyBinOp(lhs,rhs,func,i,idx...);
      }
   }

   template <typename TensorLHS,
             typename TensorRHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                (is_2d_threaded_tensor<TensorLHS>::value &&
                is_2d_threaded_tensor<TensorRHS>::value) ||
                (has_pointer_container<TensorLHS>::value &&
                is_2d_threaded_tensor<TensorRHS>::value) ||
                (is_2d_threaded_tensor<TensorLHS>::value &&
                has_pointer_container<TensorRHS>::value),
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyBinOp(TensorLHS &lhs, TensorRHS &rhs, Lambda &&func, Idx... idx)
   {
      MFEM_FOREACH_THREAD(i,x,lhs.template Size<0>())
      {
         Foreach<0>::ApplyBinOp(lhs,rhs,func,i,idx...);
      }
   }
};

template <>
struct Foreach<2>
{
   template <typename TensorLHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_serial_tensor<TensorLHS>::value,
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyUnOp(TensorLHS &lhs, Lambda &&func, Idx... idx)
   {
      for (int i = 0; i < lhs.template Size<1>(); i++)
      {
         Foreach<1>::ApplyUnOp(lhs,func,i,idx...);
      }
   }

   template <typename TensorLHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_2d_threaded_tensor<TensorLHS>::value,
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyUnOp(TensorLHS &lhs, Lambda &&func, Idx... idx)
   {
      MFEM_FOREACH_THREAD(i,y,lhs.template Size<1>())
      {
         Foreach<1>::ApplyUnOp(lhs,func,i,idx...);
      }
   }

   template <typename TensorLHS,
             typename TensorRHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                is_serial_tensor<TensorLHS>::value &&
                is_serial_tensor<TensorRHS>::value,
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyBinOp(TensorLHS &lhs, TensorRHS &rhs, Lambda &&func, Idx... idx)
   {
      for (int i = 0; i < lhs.template Size<1>(); i++)
      {
         Foreach<1>::ApplyBinOp(lhs,rhs,func,i,idx...);
      }
   }

   template <typename TensorLHS,
             typename TensorRHS,
             typename Lambda,
             typename... Idx,
             std::enable_if_t<
                (is_2d_threaded_tensor<TensorLHS>::value &&
                is_2d_threaded_tensor<TensorRHS>::value) ||
                (has_pointer_container<TensorLHS>::value &&
                is_2d_threaded_tensor<TensorRHS>::value) ||
                (is_2d_threaded_tensor<TensorLHS>::value &&
                has_pointer_container<TensorRHS>::value),
             bool> = true>
   MFEM_HOST_DEVICE inline
   static void ApplyBinOp(TensorLHS &lhs, TensorRHS &rhs, Lambda &&func, Idx... idx)
   {
      MFEM_FOREACH_THREAD(i,y,lhs.template Size<1>())
      {
         Foreach<1>::ApplyBinOp(lhs,rhs,func,i,idx...);
      }
   }
};

} // namespace mfem

#endif // MFEM_TENSOR_FOREACH
