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

// Implementation of Surface and Cutcell IntegrationRule(s) classes

#include "fem.hpp"
#include <cmath>

using namespace std;

#ifdef MFEM_USE_LAPACK

namespace mfem
{

SIntegrationRule::SIntegrationRule(int q, ElementTransformation& Tr,
                                   Coefficient &levelset)
   : Array<IntegrationPoint>(), Order(q), LvlSet(levelset)
{
   nBasis = 2 * (Order + 1) + (int)(Order * (Order + 1) / 2);
   Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

   // compute the quadrture points
   int qorder = 0;
   IntegrationRules irs(0, Quadrature1D::GaussLegendre);
   IntegrationRule ir = irs.Get(Trafo.GetGeometryType(), qorder);
   for (; ir.GetNPoints() <= nBasis; qorder++)
   {
      ir = irs.Get(Trafo.GetGeometryType(), qorder);
   }

   // set the quadrature points and zero weights
   SetSize(ir.GetNPoints());
   for (int ip = 0; ip < Size(); ip++)
   {
      IntPoint(ip).index = ip;
      IntegrationPoint &intp = IntPoint(ip);
      intp.x = ir.IntPoint(ip).x;
      intp.y = ir.IntPoint(ip).y;
      intp.weight = 0.;
   }

   // compute the weights for current element
   ComputeWeights2D();
}

void SIntegrationRule::ComputeWeights2D()
{
   Element* me = Trafo.mesh->GetElement(Trafo.ElementNo);

   DenseMatrix Mat(nBasis, GetNPoints());
   Mat = 0.;
   Vector RHS(nBasis);
   RHS = 0.;
   Vector Weights(GetNPoints());
   Weights = 0.;

   bool element_int = false;
   bool interior = true;
   Array<bool> edge_int;

   DenseMatrix PointA(me->GetNEdges(), Trafo.GetSpaceDim());
   DenseMatrix PointB(me->GetNEdges(), Trafo.GetSpaceDim());
   Vector edgelength(me->GetNEdges());

   Array<int> verts;
   Trafo.mesh->GetElementVertices(Trafo.ElementNo, verts);

   // find the edges that are intersected by he surface and inside the area
   for (int edge = 0; edge < me->GetNEdges(); edge++)
   {
      enum class Layout {inside, intersected, outside};
      Layout layout;

      const int* vert = me->GetEdgeVertices(edge);
      Vector pointA(Trafo.GetSpaceDim());
      Vector pointB(Trafo.GetSpaceDim());
      for(int d = 0; d < Trafo.GetSpaceDim(); d++)
      {
         pointA(d) = (Trafo.mesh->GetVertex(verts[vert[0]]))[d];
         pointB(d) = (Trafo.mesh->GetVertex(verts[vert[1]]))[d];
      }
      Vector edgevec(Trafo.GetSpaceDim());
      subtract(pointA, pointB, edgevec);
      edgelength(edge) = edgevec.Norml2();

      IntegrationPoint ipA;
      Trafo.TransformBack(pointA, ipA);
      IntegrationPoint ipB;
      Trafo.TransformBack(pointB, ipB);

      if (LvlSet.Eval(Trafo, ipA) < -1e-12
          || LvlSet.Eval(Trafo, ipB) < -1e-12)
      {
         interior = false;
      }

      if (LvlSet.Eval(Trafo, ipA) > -1e-12
          && LvlSet.Eval(Trafo, ipB) > -1e-12)
      {
         layout = Layout::inside;
      }
      else if (LvlSet.Eval(Trafo, ipA) > 1e-15
               && LvlSet.Eval(Trafo, ipB) <= 0.)
      {
         layout = Layout::intersected;
      }
      else if (LvlSet.Eval(Trafo, ipA) <= 0.
               && LvlSet.Eval(Trafo, ipB) > 1e-15)
      {
         layout = Layout::intersected;
         Vector temp(pointA.Size());
         temp = pointA;
         pointA = pointB;
         pointB = temp;
      }
      else
      {
         layout = Layout::outside;
      }

      if (layout == Layout::intersected)
      {
         Vector pointC(pointA.Size());
         Vector mid(pointA.Size());
         pointC = pointA;
         mid = pointC;
         mid += pointB;
         mid /= 2.;

         IntegrationPoint ip;
         Trafo.TransformBack(mid, ip);

         while (LvlSet.Eval(Trafo, ip) > 1e-12
                || LvlSet.Eval(Trafo, ip) < -1e-12)
         {
            if (LvlSet.Eval(Trafo, ip) > 1e-12)
            {
               pointC = mid;
            }
            else
            {
               pointB = mid;
            }

            mid = pointC;
            mid += pointB;
            mid /= 2.;
            Trafo.TransformBack(mid, ip);
         }
         pointB = mid;
      }

      PointA.SetRow(edge, pointA);
      PointB.SetRow(edge, pointB);

      if ((layout == Layout::inside || layout == Layout::intersected))
      {
         edge_int.Append(true);
      }
      else
      {
         edge_int.Append(false);
      }
   }

   // do integration over the edges
   for (int edge = 0; edge < me->GetNEdges(); edge++)
   {
      if (edge_int[edge] && !interior)
      {
         Vector point0(Trafo.GetSpaceDim());
         Vector point1(Trafo.GetSpaceDim());
         PointA.GetRow(edge, point0);
         PointB.GetRow(edge, point1);

         element_int = true;

         const IntegrationRule *ir2 = &IntRules.Get(Geometry::SEGMENT,
                                                    2*Order+1);

         Vector normal(Trafo.GetDimension());
         normal = 0.;
         if(edge == 0 || edge == 2)
            normal(1) = 1.;
         if(edge == 1 || edge == 3)
            normal(0) = 1.;
         if(edge == 0 || edge == 3)
            normal *= -1.;

         for (int ip = 0; ip < ir2->GetNPoints(); ip++)
         {
            Vector dist(Trafo.GetSpaceDim());
            dist = point1;
            dist -= point0;

            Vector point(Trafo.GetSpaceDim());
            point = dist;
            point *= ir2->IntPoint(ip).x;
            point += point0;

            IntegrationPoint intpoint;
            Trafo.TransformBack(point, intpoint);
            Trafo.SetIntPoint(&intpoint);
            DenseMatrix shapes;
            OrthoBasis2D(intpoint, shapes);
            Vector grad(Trafo.GetDimension());

            for (int dof = 0; dof < nBasis; dof++)
            {
               shapes.GetRow(dof, grad);
               // Add to RHS, scle by 1/length(edge)
               RHS(dof) -= (grad * normal) * ir2->IntPoint(ip).weight
                          * dist.Norml2() / edgelength(edge);
            }
         }
      }
   }

   Vector scale(Size());

   // do integration over the area for integral over interface
   if (element_int && !interior)
   {
      int elem = Trafo.ElementNo;
      H1_FECollection fec(9, 2);
      FiniteElementSpace fes(Trafo.mesh, &fec);
      GridFunction LevelSet(&fes);
      LevelSet.ProjectCoefficient(LvlSet);
      Trafo.mesh->GetElementTransformation(elem, &Trafo);

      const FiniteElement* fe = fes.GetFE(elem);
      Vector normal(Trafo.GetDimension());
      Vector normal2(Trafo.GetSpaceDim());
      Vector gradi(Trafo.GetDimension());
      DenseMatrix dshape(fe->GetDof(), Trafo.GetDimension());
      Array<int> dofs;
      fes.GetElementDofs(elem, dofs);

      for (int ip = 0; ip < GetNPoints(); ip++)
      {
         Trafo.SetIntPoint(&(IntPoint(ip)));
         LevelSet.GetGradient(Trafo, normal2);
         double normphys = normal2.Norml2();

         normal = 0.;
         fe->CalcDShape(IntPoint(ip), dshape);
         for(int dof = 0; dof < fe->GetDof(); dof++)
         {
            dshape.GetRow(dof, gradi);
            gradi *= LevelSet(dofs[dof]);
            normal += gradi;
         }
         double normref = normal.Norml2();
         normal *= (-1. / normal.Norml2());

         scale(ip) = normphys / normref;

         DenseMatrix shapes;
         OrthoBasis2D(IntPoint(ip), shapes);

         for (int dof = 0; dof < nBasis; dof++)
         {
            Vector grad(Trafo.GetSpaceDim());
            shapes.GetRow(dof, grad);
            Mat(dof, ip) =  (grad * normal);
         }
      }

      // solve the underdetermined linear system
      Vector temp(nBasis);
      Vector temp2(GetNPoints());
      DenseMatrixSVD SVD(Mat, 'A', 'A');
      SVD.Eval(Mat);
      SVD.LeftSingularvectors().MultTranspose(RHS, temp);
      temp2 = 0.;
      for (int i = 0; i < nBasis; i++)
         if (SVD.Singularvalue(i) > 1e-12)
         {
            temp2(i) = temp(i) / SVD.Singularvalue(i);
         }
      SVD.RightSingularvectors().MultTranspose(temp2, Weights);
   }

   // scale the weights
   for (int ip = 0; ip < GetNPoints(); ip++)
   {

      IntegrationPoint &intp = IntPoint(ip);
      intp.weight = Weights(ip) * scale(ip);
   }
}

void SIntegrationRule::Basis2D(const IntegrationPoint& ip, DenseMatrix& shape)
{
   shape.SetSize(nBasis, 2);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   for (int c = 0; c <= Order; c++)
   {
      Vector a(2);
      a = 0.;
      a(1) = pow(X(0), (double)(c));

      Vector b(2);
      b = 0.;
      b(0) = pow(X(1), (double)(c));

      shape.SetRow(2 * c, a);
      shape.SetRow(2 * c + 1, b);
   }

   Poly_1D poly;
   int count = 2 * Order + 2;
   for (int c = 1; c <= Order; c++)
   {
      const int* factorial = poly.Binom(c);
      for (int expo = c; expo > 0; expo--)
      {
         Vector a(2);
         a(0) = (double)(factorial[expo]) * pow(X(0), (double)(expo))
                *  pow(X(1), (double)(c - expo));
         a(1) = -1. * (double)(factorial[expo - 1])
                * pow(X(0), (double)(expo - 1))
                * pow(X(1), (double)(c - expo + 1));

         shape.SetRow(count, a);
         count++;
      }
   }
}

void SIntegrationRule::OrthoBasis2D(const IntegrationPoint& ip,
                                  DenseMatrix& shape)
{
   const IntegrationRule *ir = &IntRules.Get(Geometry::SQUARE, 2*Order+1);

   shape.SetSize(nBasis, 2);

   // evaluate basis inthe point
   DenseMatrix preshape(nBasis, 2);
   Basis2D(ip, shape);

   // evaluate basis for quadrature points
   DenseTensor shapeMFN(nBasis, 2, ir->GetNPoints());
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      DenseMatrix shapeN(nBasis, 2);
      Basis2D(ir->IntPoint(p), shapeN);
      for (int i = 0; i < nBasis; i++)
         for (int j = 0; j < 2; j++)
         {
            shapeMFN(i, j, p) = shapeN(i, j);
         }
   }

   // do modified Gram-Schmidt orthogonalization
   for (int count = 1; count < nBasis; count++)
   {
      mGSStep(shape, shapeMFN, count);
   }
}

void SIntegrationRule::mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN,
                               int step)
{
   const IntegrationRule *ir = &IntRules.Get(Geometry::SQUARE, 2*Order+1);

   for (int count = step; count < shape.Height(); count++)
   {
      double den = 0.;
      double num = 0.;

      for (int ip = 0; ip < ir->GetNPoints(); ip++)
      {
         Vector u(2);
         Vector v(2);

         shapeMFN(ip).GetRow(count, u);
         shapeMFN(ip).GetRow(step - 1, v);

         den += v * v * ir->IntPoint(ip).weight;
         num += u * v * ir->IntPoint(ip).weight;
      }

      double coeff = num / den;

      Vector s(2);
      Vector t(2);
      shape.GetRow(step - 1, s);
      shape.GetRow(count, t);
      s *= coeff;
      t += s;
      shape.SetRow(count, t);

      for (int ip = 0; ip < ir->GetNPoints(); ip++)
      {
         shapeMFN(ip).GetRow(step - 1, s);
         shapeMFN(ip).GetRow(count, t);
         s *= coeff;
         t += s;
         shapeMFN(ip).SetRow(count, t);
      }
   }
}

void SIntegrationRule::Update(IsoparametricTransformation& Tr)
{
   if (ElementNo != Tr.ElementNo)
   {
      ElementNo = Tr.ElementNo;
      Trafo = Tr;
      ComputeWeights2D();
   }
}

////////////////////////////////////////////////////////////////////////////////

CutIntegrationRule::CutIntegrationRule(int q, ElementTransformation& Tr,
                                       Coefficient &levelset)
   : Array<IntegrationPoint>(), Order(q), LvlSet(levelset),
     SIR(NULL), SVD(NULL)
{
   nBasis = (int)((q + 1) * (q + 2) / 2);
   Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

   // set surface integration rule
   SIR = new SIntegrationRule(q+1, Tr, levelset);

   // get the quadrature points
   int qorder = 0;
   int minIntPoints = 2 * (Order + 1) + (int)(Order * (Order + 1) / 2);
   IntegrationRules irs(0, Quadrature1D::GaussLegendre);
   IntegrationRule ir = irs.Get(Trafo.GetGeometryType(), qorder);
   for (; ir.GetNPoints() <= minIntPoints; qorder++)
   {
      ir = irs.Get(Trafo.GetGeometryType(), qorder);
   }

   // set the quadrature points and default weights
   SetSize(ir.GetNPoints());
   InteriorWeights.SetSize(ir.GetNPoints());
   for (int ip = 0; ip < Size(); ip++)
   {
      IntPoint(ip).index = ip;
      IntegrationPoint &intp = IntPoint(ip);
      intp.x = ir.IntPoint(ip).x;
      intp.y = ir.IntPoint(ip).y;
      intp.weight = 0.;
      InteriorWeights(ip) = ir.IntPoint(ip).weight;
   }

   // assamble the matrix
   DenseMatrix Mat(nBasis, Size());
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      Vector shape;
      Basis2D(ir.IntPoint(ip), shape);
      Mat.SetCol(ip, shape);
   }

   // compute the svd for the matrix
   SVD = new DenseMatrixSVD(Mat, 'A', 'A');
   SVD->Eval(Mat);

   // compute the weights for the current element
   ComputeWeights2D();
}

void CutIntegrationRule::ComputeWeights2D()
{
   Element* me = Trafo.mesh->GetElement(Trafo.ElementNo);

   Vector RHS(nBasis);
   RHS = 0.;
   Vector Weights(GetNPoints());
   Weights = 0.;

   bool element_int = false;
   bool interior = true;
   Array<bool> edge_int;

   DenseMatrix PointA(me->GetNEdges(), Trafo.GetSpaceDim());
   DenseMatrix PointB(me->GetNEdges(), Trafo.GetSpaceDim());
   Vector edgelength(me->GetNEdges());

   Array<int> verts;
   Trafo.mesh->GetElementVertices(Trafo.ElementNo, verts);

   // find the edges that are intersected by he surface and inside the area
   for (int edge = 0; edge < me->GetNEdges(); edge++)
   {
      enum class Layout {inside, intersected, outside};
      Layout layout;

      const int* vert = me->GetEdgeVertices(edge);
      Vector pointA(Trafo.GetSpaceDim());
      Vector pointB(Trafo.GetSpaceDim());
      for(int d = 0; d < Trafo.GetSpaceDim(); d++)
      {
         pointA(d) = (Trafo.mesh->GetVertex(verts[vert[0]]))[d];
         pointB(d) = (Trafo.mesh->GetVertex(verts[vert[1]]))[d];
      }
      Vector edgevec(Trafo.GetSpaceDim());
      subtract(pointA, pointB, edgevec);
      edgelength(edge) = edgevec.Norml2();

      IntegrationPoint ipA;
      Trafo.TransformBack(pointA, ipA);
      IntegrationPoint ipB;
      Trafo.TransformBack(pointB, ipB);

      if (LvlSet.Eval(Trafo, ipA) < -1e-12
          || LvlSet.Eval(Trafo, ipB) < -1e-12)
      {
         interior = false;
      }

      if (LvlSet.Eval(Trafo, ipA) > -1e-12
          && LvlSet.Eval(Trafo, ipB) > -1e-12)
      {
         layout = Layout::inside;
      }
      else if (LvlSet.Eval(Trafo, ipA) > 1e-15
               && LvlSet.Eval(Trafo, ipB) <= 0.)
      {
         layout = Layout::intersected;
      }
      else if (LvlSet.Eval(Trafo, ipA) <= 0.
               && LvlSet.Eval(Trafo, ipB) > 1e-15)
      {
         layout = Layout::intersected;
         Vector temp(pointA.Size());
         temp = pointA;
         pointA = pointB;
         pointB = temp;
      }
      else
      {
         layout = Layout::outside;
      }

      if (layout == Layout::intersected)
      {
         Vector pointC(pointA.Size());
         Vector mid(pointA.Size());
         pointC = pointA;
         mid = pointC;
         mid += pointB;
         mid /= 2.;

         IntegrationPoint ip;
         Trafo.TransformBack(mid, ip);

         while (LvlSet.Eval(Trafo, ip) > 1e-12
                || LvlSet.Eval(Trafo, ip) < -1e-12)
         {
            if (LvlSet.Eval(Trafo, ip) > 1e-12)
            {
               pointC = mid;
            }
            else
            {
               pointB = mid;
            }

            mid = pointC;
            mid += pointB;
            mid /= 2.;
            Trafo.TransformBack(mid, ip);
         }
         pointB = mid;
      }

      PointA.SetRow(edge, pointA);
      PointB.SetRow(edge, pointB);

      if ((layout == Layout::inside || layout == Layout::intersected))
      {
         edge_int.Append(true);
      }
      else
      {
         edge_int.Append(false);
      }
   }

   // do the integration over the edges
   for (int edge = 0; edge < me->GetNEdges(); edge++)
   {
      if (edge_int[edge] && !interior)
      {
         Vector point0(Trafo.GetSpaceDim());
         Vector point1(Trafo.GetSpaceDim());
         PointA.GetRow(edge, point0);
         PointB.GetRow(edge, point1);

         element_int = true;

         const IntegrationRule *ir2 = &IntRules.Get(Geometry::SEGMENT,
                                                    2*Order+1);

         Vector normal(Trafo.GetDimension());
         normal = 0.;
         if(edge == 0 || edge == 2)
            normal(1) = 1.;
         if(edge == 1 || edge == 3)
            normal(0) = 1.;
         if(edge == 0 || edge == 3)
            normal *= -1.;

         for (int ip = 0; ip < ir2->GetNPoints(); ip++)
         {
            Vector dist(Trafo.GetSpaceDim());
            dist = point1;
            dist -= point0;

            Vector point(Trafo.GetSpaceDim());
            point = dist;
            point *= ir2->IntPoint(ip).x;
            point += point0;

            IntegrationPoint intpoint;
            Trafo.TransformBack(point, intpoint);
            DenseMatrix shapes;
            BasisAntiDerivative2D(intpoint, shapes);
            Vector adiv(Trafo.GetDimension());

            for (int dof = 0; dof < nBasis; dof++)
            {
               shapes.GetRow(dof, adiv);
               // Add to RHS, scale by ??? length(edge) or 1/length(edge) and 1/2
               RHS(dof) += (adiv * normal) * ir2->IntPoint(ip).weight
                           * dist.Norml2() /edgelength(edge) * .5;
            }
         }
      }
   }

   // do the integration over the interface
   if (element_int && !interior)
   {
      int elem = Trafo.ElementNo;
      H1_FECollection fec(9, 2);
      FiniteElementSpace fes(Trafo.mesh, &fec);
      GridFunction LevelSet(&fes);
      LevelSet.ProjectCoefficient(LvlSet);
      Trafo.mesh->GetElementTransformation(elem, &Trafo);

      const FiniteElement* fe = fes.GetFE(elem);
      Vector normal(Trafo.GetDimension());
      Vector normal2(Trafo.GetDimension());
      Vector gradi(Trafo.GetDimension());
      DenseMatrix dshape(fe->GetDof(), Trafo.GetDimension());
      Array<int> dofs;
      fes.GetElementDofs(elem, dofs);

      for (int ip = 0; ip < GetNPoints(); ip++)
      {
         Trafo.SetIntPoint(&(IntPoint(ip)));
         LevelSet.GetGradient(Trafo, normal2);
         double normphys = normal2.Norml2();
         normal2 *= (-1. / normphys);

         normal = 0.;
         fe->CalcDShape(IntPoint(ip), dshape);
         for(int dof = 0; dof < fe->GetDof(); dof++)
         {
            dshape.GetRow(dof, gradi);
            gradi *= LevelSet(dofs[dof]);
            normal += gradi;
         }
         double normref = normal.Norml2();
         normal *= (-1. / normal.Norml2());

         double scale = normref / normphys;

         DenseMatrix shapes;
         BasisAntiDerivative2D(SIR->IntPoint(ip), shapes);

         for (int dof = 0; dof < nBasis; dof++)
         {
            Vector adiv(2);
            shapes.GetRow(dof, adiv);
            // Add to RHS, scale by scale/2
            RHS(dof) += (adiv * normal) * SIR->IntPoint(ip).weight
                          * .5 * scale;
         }
      }

      // solve the underdetermined linear system
      Vector temp(nBasis);
      Vector temp2(GetNPoints());
      temp2 = 0.;
      SVD->LeftSingularvectors().MultTranspose(RHS, temp);
      for (int i = 0; i < nBasis; i++)
         if (SVD->Singularvalue(i) > 1e-12)
         {
            temp2(i) = temp(i) / SVD->Singularvalue(i);
         }
      SVD->RightSingularvectors().MultTranspose(temp2, Weights);

      // scale the weights
      //Weights *= 1. / Trafo.Weight();
   }
   else if (interior)
   {
      Weights = InteriorWeights;
   }
   else
   {
      Weights = 0.;
   }

   for (int ip = 0; ip < GetNPoints(); ip++)
   {
      IntegrationPoint &intp = IntPoint(ip);
      intp.weight = Weights(ip);
   }
}

void CutIntegrationRule::Basis2D(const IntegrationPoint& ip, Vector& shape)
{
   shape.SetSize(nBasis);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   int count = 0;
   for (int c = 0; c <= Order; c++)
   {
      for (int expo = 0; expo <= c; expo++)
      {
         shape(count) = pow(X(0), (double)(expo))
                        * pow(X(1), (double)(c - expo));
         count++;
      }
   }
}

void CutIntegrationRule::BasisAntiDerivative2D(const IntegrationPoint& ip,
                                             DenseMatrix& shape)
{
   shape.SetSize(nBasis, 2);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   int count = 0;
   for (int c = 0; c <= Order; c++)
   {
      for (int expo = 0; expo <= c; expo++)
      {
         shape(count, 0) = .5 * pow(X(0), (double)(expo + 1))
                           * pow(X(1), (double)(c - expo))
                           / (double)(expo + 1);
         shape(count, 1) = .5 * pow(X(0), (double)(expo))
                           * pow(X(1), (double)(c - expo + 1))
                           / (double)(c - expo + 1);
         count++;
      }
   }
}

void CutIntegrationRule::Update(IsoparametricTransformation& Tr)
{
   if (ElementNo != Tr.ElementNo)
   {
      ElementNo = Tr.ElementNo;
      Trafo = Tr;
      SIR->Update(Tr);
      ComputeWeights2D();
   }
}

CutIntegrationRule::~CutIntegrationRule() { delete SVD; delete SIR; }

}

#endif //MFEM_USE_LAPACK