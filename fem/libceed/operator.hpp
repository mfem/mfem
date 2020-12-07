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

#include <ceed.h>
#include "util.hpp"
#include "coefficient.hpp"

namespace mfem
{

namespace internal
{
extern Ceed ceed; // defined in device.cpp
extern CeedBasisMap basis_map;
extern CeedRestrMap restr_map;
}

/** A base class to represent an MFEM Operator with a CeedOperator. */
class MFEMCeedOperator
{
protected:
   CeedOperator oper;
   CeedVector u, v;

   MFEMCeedOperator() : oper(nullptr), u(nullptr), v(nullptr) { }

public:
   void Mult(const Vector &x, Vector &y) const;
   void GetDiagonal(Vector &diag) const;
   virtual ~MFEMCeedOperator()
   {
      CeedOperatorDestroy(&oper);
      CeedVectorDestroy(&u);
      CeedVectorDestroy(&v);
   }
};

/// A minimal BuildContext required by CeedIntegrator
struct BuildContext { CeedInt dim, space_dim, vdim; CeedScalar coeff[3]; };

enum class EvalMode { None, Interp, Grad, InterpAndGrad };

} // namespace mfem

#endif // MFEM_LIBCEED_OPERATOR
