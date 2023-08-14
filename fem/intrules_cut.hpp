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

#ifndef MFEM_CUTINTRULES
#define MFEM_CUTINTRULES

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"
#include "coefficient.hpp"

namespace mfem
{

class SIntegrationRule : public Array<IntegrationPoint>
{
private:
   IsoparametricTransformation Trafo;
   Coefficient &LvlSet;
   int Order;
   int nBasis;
   int ElementNo;

   mutable Array<double> weights;

   void ComputeWeights();

   void Basis(const IntegrationPoint& ip, DenseMatrix& shape);
   void OrthoBasis(const IntegrationPoint& ip, DenseMatrix& shape);
   void mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN, int step);

public:
   SIntegrationRule(int q, ElementTransformation& Tr,
                    Coefficient &levelset);

   void Update(IsoparametricTransformation& Tr);

   int GetOrder() const { return Order; }
   int GetNPoints() const { return Size(); }
   IntegrationPoint &IntPoint(int i) { return (*this)[i]; }
   const IntegrationPoint &IntPoint(int i) const { return (*this)[i]; }

   int GetElement() { return ElementNo; }

   ~SIntegrationRule() { }
};

////////////////////////////////////////////////////////////////////////////////

class CutIntegrationRule : public Array<IntegrationPoint>
{
private:
   SIntegrationRule* SIR;
   IsoparametricTransformation Trafo;
   Coefficient &LvlSet;
   int Order;
   int nBasis;
   int ElementNo;
   Vector InteriorWeights;
#ifdef MFEM_USE_LAPACK
   DenseMatrixSVD* SVD;
#endif //MFEM_USE_LAPACK

   mutable Array<double> weights;

   void ComputeWeights();

   void Basis(const IntegrationPoint& ip, Vector& shape);
   void BasisAntiDerivative(const IntegrationPoint& ip, DenseMatrix& shape);

public:
   CutIntegrationRule(int q, ElementTransformation& Tr,
                      Coefficient &levelset);

   void Update(IsoparametricTransformation& Tr);

   SIntegrationRule* GetSurfaceIntegrationRule() { return SIR; }

   int GetOrder() const { return Order; }
   int GetNPoints() const { return Size(); }
   IntegrationPoint &IntPoint(int i) { return (*this)[i]; }
   const IntegrationPoint &IntPoint(int i) const { return (*this)[i]; }

   ~CutIntegrationRule();
};

}

#endif