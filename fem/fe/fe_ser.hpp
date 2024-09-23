// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FE_SERENDIPITY
#define MFEM_FE_SERENDIPITY

#include "fe_base.hpp"

namespace mfem
{

/// Arbitrary order H1 serendipity elements in 2D on a quad
class H1Ser_QuadrilateralElement : public ScalarFiniteElement
{
public:
   /// Construct the H1Ser_QuadrilateralElement of order @a p
   H1Ser_QuadrilateralElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;
   using FiniteElement::Project;
};


} // namespace mfem

#endif
