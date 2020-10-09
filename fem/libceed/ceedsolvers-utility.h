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

/*
  Some Ceed functions I wrote that I use more than once
  and find useful.
*/
#ifndef __CEEDUTILITY_H
#define __CEEDUTILITY_H

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED

#include <ceed.h>
#include <ceed-backend.h>

int CeedVectorPointwiseMult(CeedVector a, const CeedVector b);
int CeedOperatorGetActiveField(CeedOperator oper, CeedOperatorField *field);
int CeedOperatorGetOrder(CeedOperator oper, CeedInt * order);
int CeedOperatorGetActiveElemRestriction(CeedOperator oper,
                                         CeedElemRestriction* restr_out);
int CeedOperatorGetSize(CeedOperator oper, CeedInt * size);
int CeedOperatorGetActiveBasis(CeedOperator oper, CeedBasis *basis);

#endif // MFEM_USE_CEED

#endif // include guard
