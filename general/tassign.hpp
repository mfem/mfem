// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TEMPLATE_ASSIGN
#define MFEM_TEMPLATE_ASSIGN

#include "../config/tconfig.hpp"

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
   static inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
   {
      return (a = b);
   }
};

template <>
struct AssignOp_Impl<AssignOp::Add>
{
   template <typename lvalue_t, typename rvalue_t>
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
   static inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
   {
      MFEM_FLOPS_ADD(1);
      return (a = b/a);
   }
};

} // namespace mfem::internal

template <AssignOp::Type Op, typename lvalue_t, typename rvalue_t>
inline lvalue_t &Assign(lvalue_t &a, const rvalue_t &b)
{
   return internal::AssignOp_Impl<Op>::Assign(a, b);
}

} // namespace mfem

#endif // MFEM_TEMPLATE_ASSIGN
