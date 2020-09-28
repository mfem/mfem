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

#include "ceedsolvers-interpolation.h"
#include "ceedsolvers-utility.h"

#include "../../general/forall.hpp"
using namespace mfem;


#ifdef MFEM_USE_CEED
#include <stdlib.h>

int CeedInterpolationCreate(Ceed ceed, CeedBasis basisctof,
                            CeedElemRestriction erestrictu_coarse,
                            CeedElemRestriction erestrictu_fine,
                            CeedInterpolation *interp) {
  int ierr;

  int height, width;
  ierr = CeedElemRestrictionGetLVectorSize(erestrictu_coarse, &width); CeedChk(ierr);
  CeedElemRestrictionGetLVectorSize(erestrictu_fine, &height); CeedChk(ierr);

  // interpolation qfunction
  const int bp3_ncompu = 1;
  CeedQFunction qf_restrict, qf_prolong;
  ierr = CeedQFunctionCreateIdentity(ceed, bp3_ncompu, CEED_EVAL_NONE,
                                     CEED_EVAL_INTERP, &qf_restrict); CeedChk(ierr);
  ierr = CeedQFunctionCreateIdentity(ceed, bp3_ncompu, CEED_EVAL_INTERP,
                                     CEED_EVAL_NONE, &qf_prolong); CeedChk(ierr);

  CeedVector c_fine_multiplicity;
  ierr = CeedVectorCreate(ceed, height, &c_fine_multiplicity); CeedChk(ierr);
  ierr = CeedVectorSetValue(c_fine_multiplicity, 0.0); CeedChk(ierr);

  // Create the restriction operator
  // Restriction - Fine to coarse
  CeedOperator op_interp, op_restrict;
  ierr = CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, &op_restrict); CeedChk(ierr);
  ierr = CeedOperatorSetField(op_restrict, "input", erestrictu_fine,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE); CeedChk(ierr);
  ierr = CeedOperatorSetField(op_restrict, "output", erestrictu_coarse,
                              basisctof, CEED_VECTOR_ACTIVE); CeedChk(ierr);

  // Interpolation - Coarse to fine
  // Create the prolongation operator
  ierr = CeedOperatorCreate(ceed, qf_prolong, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, &op_interp); CeedChk(ierr);
  CeedOperatorSetField(op_interp, "input", erestrictu_coarse,
                       basisctof, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_interp, "output", erestrictu_fine,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedVector fine_multiplicity_r;
  ierr = CeedElemRestrictionGetMultiplicity(
    erestrictu_fine, c_fine_multiplicity); CeedChk(ierr);
  ierr = CeedVectorCreate(ceed, height, &fine_multiplicity_r); CeedChk(ierr);

  CeedScalar* fine_r_data;
  const CeedScalar* fine_data;
  ierr = CeedVectorGetArray(fine_multiplicity_r, CEED_MEM_HOST,
                             &fine_r_data); CeedChk(ierr);
  ierr = CeedVectorGetArrayRead(c_fine_multiplicity, CEED_MEM_HOST,
                                 &fine_data); CeedChk(ierr);
  for (int i = 0; i < height; ++i) {
    fine_r_data[i] = 1.0 / fine_data[i];
  }

  ierr = CeedVectorRestoreArray(fine_multiplicity_r, &fine_r_data); CeedChk(ierr);
  ierr = CeedVectorRestoreArrayRead(c_fine_multiplicity, &fine_data); CeedChk(ierr);
  ierr = CeedVectorDestroy(&c_fine_multiplicity); CeedChk(ierr);

  CeedVector fine_work;
  ierr = CeedVectorCreate(ceed, height, &fine_work); CeedChk(ierr);

  // ierr = CeedCalloc(1, interp);
  *interp = (CeedInterpolation) calloc(1, sizeof(struct CeedInterpolation_private));
  (*interp)->ceed = ceed;
  (*interp)->qf_restrict = qf_restrict;
  (*interp)->qf_prolong = qf_prolong;
  (*interp)->op_interp = op_interp;
  (*interp)->op_restrict = op_restrict;
  (*interp)->fine_multiplicity_r = fine_multiplicity_r;
  (*interp)->fine_work = fine_work;

  return 0;
}

/// this is not implemented with reference counting etc.
int CeedInterpolationDestroy(CeedInterpolation *interp) {
  int ierr;

  ierr = CeedQFunctionDestroy(&(*interp)->qf_restrict); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*interp)->qf_prolong); CeedChk(ierr);
  ierr = CeedOperatorDestroy(&(*interp)->op_interp); CeedChk(ierr);
  ierr = CeedOperatorDestroy(&(*interp)->op_restrict); CeedChk(ierr);
  ierr = CeedVectorDestroy(&(*interp)->fine_multiplicity_r); CeedChk(ierr);
  ierr = CeedVectorDestroy(&(*interp)->fine_work); CeedChk(ierr);

  free(*interp);
  return 0;
}

/// @todo could use a CEED_REQUEST here
int CeedInterpolationInterpolate(CeedInterpolation interp,
                                 CeedVector in, CeedVector out)
{
  int ierr;

  ierr = CeedOperatorApply(interp->op_interp, in, out,
                           CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  ierr = CeedVectorPointwiseMult(out, interp->fine_multiplicity_r); CeedChk(ierr);

  return 0;
}

/// @todo could use a CEED_REQUEST here
/// @todo using MFEM_FORALL in this Ceed-like function is ugly
int CeedInterpolationRestrict(CeedInterpolation interp,
                              CeedVector in, CeedVector out)
{
  int ierr;

  int length;
  ierr = CeedVectorGetLength(in, &length); CeedChk(ierr);

  const CeedScalar *multiplicitydata, *indata;
  CeedScalar *workdata;
  CeedMemType mem;
  CeedGetPreferredMemType(interp->ceed, &mem);
  ierr = CeedVectorGetArrayRead(in, mem, &indata); CeedChk(ierr);
  ierr = CeedVectorGetArrayRead(interp->fine_multiplicity_r, mem,
                                &multiplicitydata); CeedChk(ierr);
  ierr = CeedVectorGetArray(interp->fine_work, mem, &workdata); CeedChk(ierr);
  MFEM_FORALL(i, length,
              {workdata[i] = indata[i] * multiplicitydata[i];});
  ierr = CeedVectorRestoreArrayRead(in, &indata); CeedChk(ierr);
  ierr = CeedVectorRestoreArrayRead(interp->fine_multiplicity_r, &multiplicitydata); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(interp->fine_work, &workdata); CeedChk(ierr);

  ierr = CeedOperatorApply(interp->op_restrict, interp->fine_work, out,
                           CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

  return 0;
}

#endif // MFEM_USE_CEED
