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

#ifndef MFEM_LIBCEED_OPERATOR
#define MFEM_LIBCEED_OPERATOR

#include "../../../linalg/operator.hpp"
#include "ceed.hpp"
#ifdef MFEM_USE_OPENMP
#include <vector>
#endif

namespace mfem
{

namespace ceed
{

/** A base class to represent a CeedOperator as an MFEM Operator. */
class Operator : public mfem::Operator
{
protected:
#ifdef MFEM_USE_CEED
#ifndef MFEM_USE_OPENMP
   CeedOperator oper, oper_t;
   CeedVector u, v;
#else
   std::vector<CeedOperator> thread_ops, thread_ops_t;
   std::vector<CeedVector> thread_u, thread_v;
#endif

#ifndef MFEM_USE_OPENMP
   Operator() : oper(nullptr), oper_t(nullptr), u(nullptr), v(nullptr) {}
#else
   Operator() {}
#endif
#endif

public:
#ifdef MFEM_USE_CEED
   /// This constructor takes ownership of the operator and will delete it
   Operator(CeedOperator op);

   operator CeedOperator() const
   {
#ifndef MFEM_USE_OPENMP
      return oper;
#else
      MFEM_VERIFY(thread_ops.size() == 1,
                  "Threaded ceed:Operator should access CeedOperators by thread index");
      return thread_ops[0];
#endif
   }
#ifdef MFEM_USE_OPENMP
   CeedOperator operator[](std::size_t i) const { return thread_ops[i]; }
   std::size_t Size() const { return thread_ops.size(); }
#endif
#endif

   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
   void AddMult(const mfem::Vector &x, mfem::Vector &y,
                const double a = 1.0) const override;
   void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;
   void AddMultTranspose(const mfem::Vector &x, mfem::Vector &y,
                         const double a = 1.0) const override;
   void GetDiagonal(mfem::Vector &diag) const;

   virtual ~Operator();
};

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_OPERATOR
