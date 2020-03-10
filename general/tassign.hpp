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

#ifndef MFEM_TEMPLATE_ASSIGN
#define MFEM_TEMPLATE_ASSIGN

#include "../config/tconfig.hpp"
#include "../general/cuda.hpp"
#include "../general/hip.hpp"

namespace mfem
{

// Assignment operations

struct AssignOp
{
   enum Type
   {
      Set,   // a  = b
      Add,   // a += b
      Mult,  // a *= b
      Div,   // a /= b
      rDiv   // a  = b/a
   };
};

namespace internal
{

template <AssignOp::Type Op>
struct AssignOp_Impl;

template <>
struct AssignOp_Impl<AssignOp::Set>
{
   template <typename lvalue_t, typename rvalue_t>
   MFEM_HOST_DEVICE
   static inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
   {
      return (a = b);
   }
};

template <>
struct AssignOp_Impl<AssignOp::Add>
{
   template <typename lvalue_t, typename rvalue_t>
   MFEM_HOST_DEVICE
   static inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
   {
      MFEM_FLOPS_ADD(1);
      return (a += b);
   }
};

template <>
struct AssignOp_Impl<AssignOp::Mult>
{
   template <typename lvalue_t, typename rvalue_t>
   MFEM_HOST_DEVICE
   static inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
   {
      MFEM_FLOPS_ADD(1);
      return (a *= b);
   }
};

template <>
struct AssignOp_Impl<AssignOp::Div>
{
   template <typename lvalue_t, typename rvalue_t>
   MFEM_HOST_DEVICE
   static inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
   {
      MFEM_FLOPS_ADD(1);
      return (a /= b);
   }
};

template <>
struct AssignOp_Impl<AssignOp::rDiv>
{
   template <typename lvalue_t, typename rvalue_t>
   MFEM_HOST_DEVICE
   static inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
   {
      MFEM_FLOPS_ADD(1);
      return (a = b/a);
   }
};

} // namespace mfem::internal

template <AssignOp::Type Op, typename lvalue_t, typename rvalue_t>
MFEM_HOST_DEVICE
inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
{
   return internal::AssignOp_Impl<Op>::Assign(a, b);
}

} // namespace mfem

#endif // MFEM_TEMPLATE_ASSIGN
