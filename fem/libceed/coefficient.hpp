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

#ifndef MFEM_LIBCEED_COEFF
#define MFEM_LIBCEED_COEFF

#include <ceed.h>
#include "../../linalg/vector.hpp"

namespace mfem
{

class GridFunction;

enum class CeedCoeff { Const, Grid, Quad, VecConst, VecGrid, VecQuad };

struct CeedConstCoeff
{
   double val;
};

struct CeedGridCoeff
{
   const GridFunction* coeff;
   CeedBasis basis;
   CeedElemRestriction restr;
   CeedVector coeffVector;
};

struct CeedQuadCoeff
{
   Vector coeff;
   CeedElemRestriction restr;
   CeedVector coeffVector;
};

struct CeedVecConstCoeff
{
   double val[3];
};

struct CeedVecGridCoeff
{
   const GridFunction* coeff;
   CeedBasis basis;
   CeedElemRestriction restr;
   CeedVector coeffVector;
};

struct CeedVecQuadCoeff
{
   Vector coeff;
   CeedElemRestriction restr;
   CeedVector coeffVector;
};

class Mesh;
class IntegrationRule;
class Coefficient;
class VectorCoefficient;

/** @brief Identifies the type of coefficient of the Integrator to initialize
    accordingly the CeedData. */
// void InitCeedCoeff(Coefficient *Q, Mesh &mesh, const IntegrationRule &ir,
//                    CeedData *ptr);
void InitCeedCoeff(Coefficient *Q, Mesh &mesh, const IntegrationRule &ir,
                   CeedCoeff& coeff_type, void *&coeff);

/** @brief Identifies the type of vector coefficient of the Integrator to
    initialize accordingly the CeedData. */
// void InitCeedVecCoeff(VectorCoefficient *VQ, Mesh &mesh,
//                       const IntegrationRule &ir, CeedData *ptr);
void InitCeedVecCoeff(VectorCoefficient *VQ, Mesh &mesh,
                      const IntegrationRule &ir,
                      CeedCoeff& coeff_type, void *&coeff);

} // namespace mfem

#endif // MFEM_LIBCEED_COEFF