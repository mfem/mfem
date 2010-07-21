// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include <math.h>
#include "fem.hpp"

const DenseMatrix & IsoparametricTransformation::Jacobian()
{
   if (JacobianIsEvaluated)  return dFdx;

   dshape.SetSize(FElem->GetDof(), FElem->GetDim());
   dFdx.SetSize(PointMat.Height(), dshape.Width());

   FElem -> CalcDShape(*IntPoint, dshape);
   Mult(PointMat, dshape, dFdx);

   JacobianIsEvaluated = 1;

   return dFdx;
}

double IsoparametricTransformation::Weight()
{
   if (FElem->GetDim() == 0)
      return 1.0;
   if (WeightIsEvaluated)
      return Wght;
   Jacobian();
   WeightIsEvaluated = 1;
   return (Wght = dFdx.Weight());
}

int IsoparametricTransformation::OrderJ()
{
   switch (FElem->Space())
   {
   case FunctionSpace::Pk:
      return (FElem->GetOrder()-1);
   case FunctionSpace::Qk:
      return (FElem->GetOrder());
   default:
      mfem_error("IsoparametricTransformation::OrderJ()");
   }
   return 0;
}

int IsoparametricTransformation::OrderW()
{
   switch (FElem->Space())
   {
   case FunctionSpace::Pk:
      return (FElem->GetOrder() - 1) * FElem->GetDim();
   case FunctionSpace::Qk:
      return (FElem->GetOrder() * FElem->GetDim() - 1);
   default:
      mfem_error("IsoparametricTransformation::OrderW()");
   }
   return 0;
}

int IsoparametricTransformation::OrderGrad(const FiniteElement *fe)
{
   if (FElem->Space() == fe->Space())
   {
      int k = FElem->GetOrder();
      int d = FElem->GetDim();
      int l = fe->GetOrder();
      switch (fe->Space())
      {
      case FunctionSpace::Pk:
         return ((k-1)*(d-1)+(l-1));
      case FunctionSpace::Qk:
         return (k*(d-1)+(l-1));
      }
   }
   mfem_error("IsoparametricTransformation::OrderGrad(...)");
   return 0;
}

void IsoparametricTransformation::Transform (const IntegrationPoint &ip,
                                             Vector &trans)
{
   shape.SetSize(FElem->GetDof());
   trans.SetSize(PointMat.Height());

   FElem -> CalcShape(ip, shape);
   PointMat.Mult(shape, trans);
}

void IsoparametricTransformation::Transform (const IntegrationRule &ir,
                                             DenseMatrix &tr)
{
   int dof, n, dim, i, j, k;

   dim = PointMat.Height();
   dof = FElem->GetDof();
   n = ir.GetNPoints();

   shape.SetSize(dof);
   tr.SetSize(dim, n);

   for (j = 0; j < n; j++)
   {
      FElem -> CalcShape (ir.IntPoint(j), shape);
      for (i = 0; i < dim; i++)
      {
         tr(i, j) = 0.0;
         for (k = 0; k < dof; k++)
            tr(i, j) += PointMat(i, k) * shape(k);
      }
   }
}

void IntegrationPointTransformation::Transform (const IntegrationPoint &ip1,
                                                IntegrationPoint &ip2)
{
   double vec[3];
   Vector v (vec, Transf.GetPointMat().Height());

   Transf.Transform (ip1, v);
   ip2.x = vec[0];
   ip2.y = vec[1];
   ip2.z = vec[2];
}

void IntegrationPointTransformation::Transform (const IntegrationRule &ir1,
                                                IntegrationRule &ir2)
{
   int i, n;

   n = ir1.GetNPoints();
   for (i = 0; i < n; i++)
      Transform (ir1.IntPoint(i), ir2.IntPoint(i));
}
