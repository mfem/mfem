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

namespace mfem
{

namespace ceed
{

/** A base class to represent a CeedOperator as an MFEM Operator. */
class Operator : public mfem::Operator
{
protected:
#ifdef MFEM_USE_CEED
   CeedOperator oper;
   CeedVector u, v;

   Operator() : oper(nullptr), u(nullptr), v(nullptr) { }
#endif

public:
#ifdef MFEM_USE_CEED
   /// This class takes ownership of op and will delete it
   Operator(CeedOperator op);
#endif
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
   void AddMult(const mfem::Vector &x, mfem::Vector &y,
                const double a = 1.0) const override;
   void GetDiagonal(mfem::Vector &diag) const;
   using mfem::Operator::SetupRAP;
   virtual ~Operator()
   {
#ifdef MFEM_USE_CEED
      CeedOperatorDestroy(&oper);
      CeedVectorDestroy(&u);
      CeedVectorDestroy(&v);
#endif
   }

#ifdef MFEM_USE_CEED
   CeedOperator& GetCeedOperator() { return oper; }
#endif
};

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_OPERATOR
