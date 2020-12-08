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

#ifndef MFEM_LIBCEED_OPERATOR
#define MFEM_LIBCEED_OPERATOR

#include "util.hpp"
#include "coefficient.hpp"

namespace mfem
{

/** A base class to represent an MFEM Operator with a CeedOperator. */
class MFEMCeedOperator
{
protected:
#ifdef MFEM_USE_CEED
   CeedOperator oper;
   CeedVector u, v;

   MFEMCeedOperator() : oper(nullptr), u(nullptr), v(nullptr) { }
#endif

public:
   void Mult(const Vector &x, Vector &y) const;
   void GetDiagonal(Vector &diag) const;
   virtual ~MFEMCeedOperator()
   {
#ifdef MFEM_USE_CEED
      CeedOperatorDestroy(&oper);
      CeedVectorDestroy(&u);
      CeedVectorDestroy(&v);
#endif
   }
};

/** The different evaluation modes available for PA and MF CeedIntegrator. */
enum class EvalMode { None, Interp, Grad, InterpAndGrad };

} // namespace mfem

#endif // MFEM_LIBCEED_OPERATOR
