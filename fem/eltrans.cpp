// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include <cmath>
#include "fem.hpp"

namespace mfem
{

ElementTransformation::ElementTransformation():
   JacobianIsEvaluated(0),
   WeightIsEvaluated(0),
   IntPoint(static_cast<IntegrationPoint *>(NULL)),
   Attribute(-1),
   ElementNo(-1)
{

}

void IsoparametricTransformation::SetIdentityTransformation(int GeomType)
{
   switch (GeomType)
   {
      case Geometry::POINT :       FElem = &PointFE; break;
      case Geometry::SEGMENT :     FElem = &SegmentFE; break;
      case Geometry::TRIANGLE :    FElem = &TriangleFE; break;
      case Geometry::SQUARE :      FElem = &QuadrilateralFE; break;
      case Geometry::TETRAHEDRON : FElem = &TetrahedronFE; break;
      case Geometry::CUBE :        FElem = &HexahedronFE; break;
      default:
         MFEM_ABORT("unknown Geometry::Type!");
   }
   int dim = FElem->GetDim();
   int dof = FElem->GetDof();
   const IntegrationRule &nodes = FElem->GetNodes();
   PointMat.SetSize(dim, dof);
   for (int j = 0; j < dof; j++)
   {
      nodes.IntPoint(j).Get(&PointMat(0,j), dim);
   }
}

const DenseMatrix & IsoparametricTransformation::Jacobian()
{
   if (JacobianIsEvaluated) { return dFdx; }

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
   {
      return 1.0;
   }
   if (WeightIsEvaluated)
   {
      return Wght;
   }
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
         {
            tr(i, j) += PointMat(i, k) * shape(k);
         }
      }
   }
}

void IsoparametricTransformation::Transform (const DenseMatrix &matrix,
                                             DenseMatrix &result)
{
   result.SetSize(PointMat.Height(), matrix.Width());

   IntegrationPoint ip;
   Vector col;

   for (int j = 0; j < matrix.Width(); j++)
   {
      ip.x = matrix(0, j);
      if (matrix.Height() > 1)
      {
         ip.y = matrix(1, j);
         if (matrix.Height() > 2)
         {
            ip.z = matrix(2, j);
         }
      }

      result.GetColumnReference(j, col);
      Transform(ip, col);
   }
}

int IsoparametricTransformation::TransformBack(const Vector &pt,
                                               IntegrationPoint &ip)
{
   const int    max_iter = 16;
   const double  ref_tol = 1e-15;
   const double phys_tol = 1e-15*pt.Normlinf();

   const int dim = FElem->GetDim();
   const int sdim = PointMat.Height();
   const int geom = FElem->GetGeomType();
   IntegrationPoint xip, prev_xip;
   double xd[3], yd[3], dxd[3], Jid[9];
   Vector x(xd, dim), y(yd, sdim), dx(dxd, dim);
   DenseMatrix Jinv(Jid, dim, sdim);
   bool hit_bdr = false, prev_hit_bdr;

   // Use the center of the element as initial guess
   xip = Geometries.GetCenter(geom);
   xip.Get(xd, dim); // xip -> x

   for (int it = 0; it < max_iter; it++)
   {
      // Newton iteration:    x := x + J(x)^{-1} [pt-F(x)]
      // or when dim != sdim: x := x + [J^t.J]^{-1}.J^t [pt-F(x)]
      Transform(xip, y);
      subtract(pt, y, y); // y = pt-y
      if (y.Normlinf() < phys_tol) { ip = xip; return 0; }
      SetIntPoint(&xip);
      CalcInverse(Jacobian(), Jinv);
      Jinv.Mult(y, dx);
      x += dx;
      prev_xip = xip;
      prev_hit_bdr = hit_bdr;
      xip.Set(xd, dim); // x -> xip
      // If xip is ouside project it on the boundary on the line segment
      // between prev_xip and xip
      hit_bdr = !Geometry::ProjectPoint(geom, prev_xip, xip);
      if (dx.Normlinf() < ref_tol) { ip = xip; return 0; }
      if (hit_bdr)
      {
         xip.Get(xd, dim); // xip -> x
         if (prev_hit_bdr)
         {
            prev_xip.Get(dxd, dim); // prev_xip -> dx
            subtract(x, dx, dx);    // dx = xip - prev_xip
            if (dx.Normlinf() < ref_tol) { return 1; }
         }
      }
   }
   ip = xip;
   return 2;
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
   {
      Transform (ir1.IntPoint(i), ir2.IntPoint(i));
   }
}

}
