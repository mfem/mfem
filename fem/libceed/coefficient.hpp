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

#include "../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "util.hpp"
#include "../../linalg/vector.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../mesh/mesh.hpp"
#include "../gridfunc.hpp"
#include "../../linalg/dtensor.hpp"

namespace mfem
{

class GridFunction;

enum class CeedCoeff { Const, Grid, Quad, VecConst, VecGrid, VecQuad };

struct CeedConstCoeff
{
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
template <typename Context>
void InitCeedCoeff(Coefficient *Q, Mesh &mesh, const IntegrationRule &ir,
                   CeedCoeff& coeff_type, void *&ptr, Context &ctx)
{
   if ( Q == nullptr )
   {
      CeedConstCoeff *ceedCoeff = new CeedConstCoeff;
      coeff_type = CeedCoeff::Const;
      ctx.coeff = 1.0;
      ptr = static_cast<void*>(ceedCoeff);
   }
   else if (ConstantCoefficient *coeff = dynamic_cast<ConstantCoefficient*>(Q))
   {
      CeedConstCoeff *ceedCoeff = new CeedConstCoeff;
      coeff_type = CeedCoeff::Const;
      ctx.coeff = coeff->constant;
      ptr = static_cast<void*>(ceedCoeff);
   }
   else if (GridFunctionCoefficient* coeff =
               dynamic_cast<GridFunctionCoefficient*>(Q))
   {
      CeedGridCoeff *ceedCoeff = new CeedGridCoeff;
      ceedCoeff->coeff = coeff->GetGridFunction();
      InitCeedVector(*ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_type = CeedCoeff::Grid;
      ptr = static_cast<void*>(ceedCoeff);
   }
   else if (QuadratureFunctionCoefficient *cQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      CeedQuadCoeff *ceedCoeff = new CeedQuadCoeff;
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      const QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(&ir == &qFun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      ceedCoeff->coeff.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
      InitCeedVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_type = CeedCoeff::Quad;
      ptr = static_cast<void*>(ceedCoeff);
   }
   else
   {
      CeedQuadCoeff *ceedCoeff = new CeedQuadCoeff;
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      ceedCoeff->coeff.SetSize(nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), nq, ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation &T = *mesh.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C(q,e) = Q->Eval(T, ir.IntPoint(q));
         }
      }
      InitCeedVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_type = CeedCoeff::Quad;
      ptr = static_cast<void*>(ceedCoeff);
   }
}

/** @brief Identifies the type of vector coefficient of the Integrator to
    initialize accordingly the CeedData. */
template <typename Context>
void InitCeedVecCoeff(VectorCoefficient *VQ, Mesh &mesh,
                      const IntegrationRule &ir,
                      CeedCoeff& coeff_type, void *&ptr, Context &ctx)
{
   if (VectorConstantCoefficient *coeff =
          dynamic_cast<VectorConstantCoefficient*>(VQ))
   {
      const int vdim = coeff->GetVDim();
      const Vector &val = coeff->GetVec();
      CeedVecConstCoeff *ceedCoeff = new CeedVecConstCoeff;
      for (int i = 0; i < vdim; i++)
      {
         ctx.coeff[i] = val[i];
      }
      coeff_type = CeedCoeff::VecConst;
      ptr = static_cast<void*>(ceedCoeff);
   }
   else if (VectorGridFunctionCoefficient* coeff =
               dynamic_cast<VectorGridFunctionCoefficient*>(VQ))
   {
      CeedVecGridCoeff *ceedCoeff = new CeedVecGridCoeff;
      ceedCoeff->coeff = coeff->GetGridFunction();
      InitCeedVector(*ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_type = CeedCoeff::VecGrid;
      ptr = static_cast<void*>(ceedCoeff);
   }
   else if (VectorQuadratureFunctionCoefficient *cQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(VQ))
   {
      CeedVecQuadCoeff *ceedCoeff = new CeedVecQuadCoeff;
      const int ne = mesh.GetNE();
      const int dim = mesh.Dimension();
      const int nq = ir.GetNPoints();
      const QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == dim * nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(&ir == &qFun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      ceedCoeff->coeff.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
      InitCeedVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_type = CeedCoeff::VecQuad;
      ptr = static_cast<void*>(ceedCoeff);
   }
   else
   {
      CeedVecQuadCoeff *ceedCoeff = new CeedVecQuadCoeff;
      const int ne = mesh.GetNE();
      const int dim = mesh.Dimension();
      const int nq = ir.GetNPoints();
      ceedCoeff->coeff.SetSize(dim * nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), dim, nq, ne);
      DenseMatrix Q_ir;
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation &T = *mesh.GetElementTransformation(e);
         VQ->Eval(Q_ir, T, ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int i = 0; i < dim; ++i)
            {
               C(i,q,e) = Q_ir(i,q);
            }
         }
      }
      InitCeedVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_type = CeedCoeff::VecQuad;
      ptr = static_cast<void*>(ceedCoeff);
   }
}

} // namespace mfem

#endif

#endif // MFEM_LIBCEED_COEFF
