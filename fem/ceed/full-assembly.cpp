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

#include "full-assembly.hpp"

#ifdef MFEM_USE_CEED
#include "util.hpp"

namespace mfem
{

namespace ceed
{

int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix **mat)
{
   int ierr;

   CeedInt nentries;
   CeedInt *rows, *cols;
   CeedVector values;
   const CeedScalar *vals;

   ierr = CeedOperatorLinearAssembleSymbolic(op, &nentries, &rows, &cols); CeedChk(ierr);
   ierr = CeedVectorCreate(internal::ceed, nentries, &values); CeedChk(ierr);
   ierr = CeedOperatorLinearAssemble(op, values); CeedChk(ierr);

   ierr = CeedVectorGetArrayRead(values, CEED_MEM_HOST, &vals); CeedChk(ierr);
   CeedElemRestriction er;
   ierr = CeedOperatorGetActiveElemRestriction(op, &er); CeedChk(ierr);
   CeedInt nnodes;
   ierr = CeedElemRestrictionGetLVectorSize(er, &nnodes); CeedChk(ierr);
   SparseMatrix *out = new SparseMatrix(nnodes, nnodes);
   for (int k = 0; k < nentries; ++k)
   {
      out->Add(rows[k], cols[k], vals[k]);
   }
   ierr = CeedVectorRestoreArrayRead(values, &vals); CeedChk(ierr);
   const int skip_zeros = 0;
   out->Finalize(skip_zeros);
   ierr = CeedVectorDestroy(&values); CeedChk(ierr);
   *mat = out;

   return 0;
}

} // namespace ceed

} // namespace mfem

#endif
