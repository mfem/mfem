// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "operator.hpp"

#include "../../../linalg/vector.hpp"
#include "../../fespace.hpp"
#include "util.hpp"
#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
Operator::Operator(CeedOperator op)
{
   CeedSize in_len, out_len;
   int ierr = CeedOperatorGetActiveVectorLengths(op, &in_len, &out_len);
   PCeedChk(ierr);
   width = in_len;
   height = out_len;
   MFEM_VERIFY(width == in_len, "width overflow");
   MFEM_VERIFY(height == out_len, "height overflow");
#ifndef MFEM_USE_OPENMP
   oper = op;
   oper_t = nullptr;
   CeedVectorCreate(internal::ceed, width, &u);
   CeedVectorCreate(internal::ceed, height, &v);
#else
   thread_ops = {op};
   thread_ops_t = {nullptr};
   thread_u.resize(1);
   thread_v.resize(1);
   CeedVectorCreate(internal::ceed, width, &thread_u[0]);
   CeedVectorCreate(internal::ceed, height, &thread_v[0]);
#endif
}
#endif

Operator::~Operator()
{
#ifdef MFEM_USE_CEED
#ifndef MFEM_USE_OPENMP
   CeedOperatorDestroy(&oper);
   CeedOperatorDestroy(&oper_t);
   CeedVectorDestroy(&u);
   CeedVectorDestroy(&v);
#else
   #pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      CeedOperatorDestroy(&thread_ops[tid]);
      CeedOperatorDestroy(&thread_ops_t[tid]);
      CeedVectorDestroy(&thread_u[tid]);
      CeedVectorDestroy(&thread_v[tid]);
   }
#endif
#endif
}

namespace
{

#ifdef MFEM_USE_CEED
template <bool ADD>
inline void CeedAddMult(CeedOperator oper, CeedVector u,
                        CeedVector v, const mfem::Vector &x, mfem::Vector &y)
{
   if (!oper) { return; }  // No-op for an empty operator
   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if (Device::Allows(Backend::DEVICE_MASK) && mem == CEED_MEM_DEVICE)
   {
      x_ptr = x.Read();
      y_ptr = (!ADD) ? y.Write() : y.ReadWrite();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = (!ADD) ? y.HostWrite() : y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }

   CeedVectorSetArray(u, mem, CEED_USE_POINTER, const_cast<CeedScalar*>(x_ptr));
   CeedVectorSetArray(v, mem, CEED_USE_POINTER, y_ptr);
   if (!ADD)
   {
      CeedOperatorApply(oper, u, v, CEED_REQUEST_IMMEDIATE);
   }
   else
   {
      CeedOperatorApplyAdd(oper, u, v, CEED_REQUEST_IMMEDIATE);
   }
   CeedVectorTakeArray(u, mem, const_cast<CeedScalar**>(&x_ptr));
   CeedVectorTakeArray(v, mem, &y_ptr);
}

#ifdef MFEM_USE_OPENMP
inline void CeedAddMult(const std::vector<CeedOperator> &ops,
                        const std::vector<CeedVector> &u,
                        const std::vector<CeedVector> &v,
                        const mfem::Vector &x, mfem::Vector &y)
{
   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if (Device::Allows(Backend::DEVICE_MASK) && mem == CEED_MEM_DEVICE)
   {
      x_ptr = x.Read();
      y_ptr = y.ReadWrite();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }

   #pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      if (ops[tid])  // No-op for an empty operator
      {
         CeedScalar *u_ptr = const_cast<CeedScalar *>(x_ptr);
         CeedScalar *v_ptr = y_ptr;
         CeedVectorSetArray(u[tid], mem, CEED_USE_POINTER, u_ptr);
         CeedVectorSetArray(v[tid], mem, CEED_USE_POINTER, v_ptr);
         CeedOperatorApplyAdd(ops[tid], u[tid], v[tid], CEED_REQUEST_IMMEDIATE);
         CeedVectorTakeArray(u[tid], mem, &u_ptr);
         CeedVectorTakeArray(v[tid], mem, &v_ptr);
      }
   }
}
#endif
#endif

} // namespace

void Operator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
#ifdef MFEM_USE_CEED
#ifndef MFEM_USE_OPENMP
   CeedAddMult<false>(oper, u, v, x, y);
#else
   y = 0.0;
   CeedAddMult(thread_ops, thread_u, thread_v, x, y);
#endif
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void Operator::AddMult(const mfem::Vector &x, mfem::Vector &y,
                       const double a) const
{
#ifdef MFEM_USE_CEED
   MFEM_ASSERT(a == 1.0, "General coefficient case is not yet supported");
#ifndef MFEM_USE_OPENMP
   CeedAddMult<true>(oper, u, v, x, y);
#else
   CeedAddMult(thread_ops, thread_u, thread_v, x, y);
#endif
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void Operator::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{
#ifdef MFEM_USE_CEED
#ifndef MFEM_USE_OPENMP
   CeedAddMult<false>(oper_t, v, u, x, y);
#else
   y = 0.0;
   CeedAddMult(thread_ops_t, thread_v, thread_u, x, y);
#endif
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void Operator::AddMultTranspose(const mfem::Vector &x, mfem::Vector &y,
                                const double a) const
{
#ifdef MFEM_USE_CEED
   MFEM_ASSERT(a == 1.0, "General coefficient case is not yet supported");
#ifndef MFEM_USE_OPENMP
   CeedAddMult<true>(oper_t, v, u, x, y);
#else
   CeedAddMult(thread_ops_t, thread_v, thread_u, x, y);
#endif
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void Operator::GetDiagonal(mfem::Vector &diag) const
{
#ifdef MFEM_USE_CEED
   CeedScalar *d_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if (Device::Allows(Backend::DEVICE_MASK) && mem == CEED_MEM_DEVICE)
   {
      d_ptr = diag.ReadWrite();
   }
   else
   {
      d_ptr = diag.HostReadWrite();
      mem = CEED_MEM_HOST;
   }

#ifndef MFEM_USE_OPENMP
   CeedVectorSetArray(v, mem, CEED_USE_POINTER, d_ptr);
   CeedOperatorLinearAssembleAddDiagonal(oper, v, CEED_REQUEST_IMMEDIATE);
   CeedVectorTakeArray(v, mem, &d_ptr);
#else
   #pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      if (thread_ops[tid])  // No-op for an empty operator
      {
         CeedScalar *v_ptr = d_ptr;
         CeedVectorSetArray(thread_v[tid], mem, CEED_USE_POINTER, v_ptr);
         CeedOperatorLinearAssembleAddDiagonal(thread_ops[tid], thread_v[tid],
                                               CEED_REQUEST_IMMEDIATE);
         CeedVectorTakeArray(thread_v[tid], mem, &v_ptr);
      }
   }
#endif
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
