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
   : IntegrationRule(), LvlSet(levelset)
{
   SetOrder(q);
   Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);
   if(Trafo.GetDimension() == 2)
      nBasis = 2 * (GetOrder() + 1) + (int)(GetOrder() * (GetOrder() + 1) / 2);
   else if(Trafo.GetDimension() == 3)
   {
      if(GetOrder()== 0) { nBasis = 3; }
      else if(GetOrder()== 1) { nBasis = 11; }
      else if(GetOrder()== 2) { nBasis = 26; }
      else if(GetOrder()== 3) { nBasis = 50; }
      else if(GetOrder()== 4) { nBasis = 83; }
      else if(GetOrder()== 5) { nBasis = 133; }
      else if(GetOrder()== 6) { nBasis = 196; }
      else if(GetOrder()== 7) { nBasis = 276; }
      else if(GetOrder()== 8) { nBasis = 375; }
      else if(GetOrder()== 9) { nBasis = 495; }
      else if(GetOrder()== 10) { nBasis = 638; }
      else if(GetOrder()== 11) { nBasis = 806; }
      else if(GetOrder()== 12) { nBasis = 1001; }
   }

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
      intp.z = ir.IntPoint(ip).z;
      intp.weight = 0.;
   }

   Weights.SetSize(GetNPoints(), Trafo.mesh->GetNE());
   Weights = 0.;

   if(Trafo.GetDimension() == 3)
   {
      int nFBasis = 2 * (GetOrder() + 2)
                   + (int)((GetOrder() + 1) * (GetOrder() + 2) / 2);
      int fqorder = 0;
      IntegrationRule fir = irs.Get(Geometry::SQUARE, fqorder);
      for (; fir.GetNPoints() <= nFBasis; fqorder++)
      {
         fir = irs.Get(Geometry::SQUARE, fqorder);
      }

      FaceIP.SetSize(fir.GetNPoints());

      for (int ip = 0; ip < fir.GetNPoints(); ip++)
      {
         FaceIP[ip].index = ip;
         IntegrationPoint &intp = FaceIP[ip];
         intp.x = fir.IntPoint(ip).x;
         intp.y = fir.IntPoint(ip).y;
         intp.weight = 0.;
      }

      FaceWeights.SetSize(FaceIP.Size(), Trafo.mesh->GetNFaces());
      FaceWeights = 0.;
   }      

   // compute the weights for current element
   if(Trafo.GetDimension() == 2)
      ComputeWeights2D();
   else if(Trafo.GetDimension() == 3)
      ComputeWeights3D();
}

void SIntegrationRule::ComputeWeights2D()
{
   Mesh* mesh = Trafo.mesh;
   H1_FECollection fec(9, 2);
   FiniteElementSpace fes(Trafo.mesh, &fec);
   GridFunction LevelSet(&fes);
   LevelSet.ProjectCoefficient(LvlSet);

   for(int elem = 0; elem < mesh->GetNE(); elem++)
   {
      Element* me = mesh->GetElement(elem);
      mesh->GetElementTransformation(elem, &Trafo);

      DenseMatrix Mat(nBasis, GetNPoints());
      Mat = 0.;
      Vector RHS(nBasis);
      RHS = 0.;
      Vector ElemWeights(GetNPoints());
      ElemWeights = 0.;

      bool element_int = false;
      bool interior = true;
      Array<bool> edge_int;

      DenseMatrix PointA(me->GetNEdges(), Trafo.GetSpaceDim());
      DenseMatrix PointB(me->GetNEdges(), Trafo.GetSpaceDim());
      Vector edgelength(me->GetNEdges());

      Array<int> verts;
      mesh->GetElementVertices(elem, verts);

      // find the edges that are intersected by he surface and inside the area
      for (int edge = 0; edge < me->GetNEdges(); edge++)
      {
         enum class Layout {inside, intersected, outside};
         Layout layout;

         const int* vert = me->GetEdgeVertices(edge);
         Vector pointA(Trafo.GetSpaceDim());
         Vector pointB(Trafo.GetSpaceDim());
         for (int d = 0; d < Trafo.GetSpaceDim(); d++)
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
                  pointC = mid;
               else
                  pointB = mid;

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
            edge_int.Append(true);
         else
         edge_int.Append(false);
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
                                                       2*GetOrder()+1);

            Vector normal(Trafo.GetDimension());
            normal = 0.;
            if (edge == 0 || edge == 2)
               normal(1) = 1.;
            if (edge == 1 || edge == 3)
               normal(0) = 1.;
            if (edge == 0 || edge == 3)
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
            for (int dof = 0; dof < fe->GetDof(); dof++)
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
               temp2(i) = temp(i) / SVD.Singularvalue(i);
         SVD.RightSingularvectors().MultTranspose(temp2, ElemWeights);
      }

      // scale the weights
      for (int ip = 0; ip < GetNPoints(); ip++)
         Weights(ip, elem) = ElemWeights(ip) * scale(ip);
   }
}

void SIntegrationRule::ComputeWeights3D()
{
   Mesh* mesh = Trafo.mesh;
   H1_FECollection fec(9, 3);
   FiniteElementSpace fes(mesh, &fec);
   GridFunction LevelSet(&fes);
   LevelSet.ProjectCoefficient(LvlSet);

   for(int face = 0; face < mesh->GetNFaces(); face++)
   {
      Array<int> verts;
      mesh->GetFaceVertices(face, verts);
      Vector pointA(mesh->SpaceDimension());
      Vector pointB(mesh->SpaceDimension());
      Vector pointC(mesh->SpaceDimension());
      Vector pointD(mesh->SpaceDimension());
      for (int d = 0; d < mesh->SpaceDimension(); d++)
      {
         pointA(d) = (mesh->GetVertex(verts[0]))[d];
         pointB(d) = (mesh->GetVertex(verts[1]))[d];
         pointC(d) = (mesh->GetVertex(verts[2]))[d];
         pointD(d) = (mesh->GetVertex(verts[3]))[d];
      }

      Mesh local_mesh(2,4,1,0,3);
      local_mesh.AddVertex(pointA);
      local_mesh.AddVertex(pointB);
      local_mesh.AddVertex(pointC);
      local_mesh.AddVertex(pointD);
      local_mesh.AddQuad(0,1,2,3);
      local_mesh.FinalizeQuadMesh(1);
      IsoparametricTransformation faceTrafo;
      local_mesh.GetElementTransformation(0, &faceTrafo);

      CutIntegrationRule faceIntRule(GetOrder(), faceTrafo, LvlSet);
      faceIntRule.SetElement(0);

      for(int ip = 0; ip < faceIntRule.GetNPoints(); ip++)
         FaceWeights(ip, face) = faceIntRule.IntPoint(ip).weight;
   }

   for(int elem = 0; elem < mesh->GetNE(); elem++)
   {
      Element* me = Trafo.mesh->GetElement(Trafo.ElementNo);
      mesh->GetElementTransformation(elem, &Trafo);
      Trafo.SetIntPoint(&(IntPoint(0)));

      DenseMatrix Mat(nBasis, GetNPoints());
      Mat = 0.;
      Vector RHS(nBasis);
      RHS = 0.;
      Vector ElemWeights(GetNPoints());
      ElemWeights = 0.;

      bool element_int = false;
      bool interior = true;

      Array<int> verts;
      mesh->GetElementVertices(elem, verts);

      for(int face = 0; face < me->GetNFaces(); face++)
      {
         enum class Layout {inside, intersected, outside};
         Layout layout;

         const int* vert = me->GetFaceVertices(face);
         Vector pointA(Trafo.GetSpaceDim());
         Vector pointB(Trafo.GetSpaceDim());
         Vector pointC(Trafo.GetSpaceDim());
         Vector pointD(Trafo.GetSpaceDim());
         for (int d = 0; d < Trafo.GetSpaceDim(); d++)
         {
            pointA(d) = (Trafo.mesh->GetVertex(verts[vert[0]]))[d];
            pointB(d) = (Trafo.mesh->GetVertex(verts[vert[1]]))[d];
            pointC(d) = (Trafo.mesh->GetVertex(verts[vert[2]]))[d];
            pointD(d) = (Trafo.mesh->GetVertex(verts[vert[3]]))[d];
         }

         IntegrationPoint ipA;
         Trafo.TransformBack(pointA, ipA);
         IntegrationPoint ipB;
         Trafo.TransformBack(pointB, ipB);
         IntegrationPoint ipC;
         Trafo.TransformBack(pointC, ipC);
         IntegrationPoint ipD;
         Trafo.TransformBack(pointD, ipD);

         if (LvlSet.Eval(Trafo, ipA) < -1e-12
             || LvlSet.Eval(Trafo, ipB) < -1e-12
             || LvlSet.Eval(Trafo, ipC) < -1e-12
             || LvlSet.Eval(Trafo, ipD) < -1e-12)
         {
            interior = false;
         }

         if (LvlSet.Eval(Trafo, ipA) > -1e-12
             || LvlSet.Eval(Trafo, ipB) > -1e-12
             || LvlSet.Eval(Trafo, ipC) > -1e-12
             || LvlSet.Eval(Trafo, ipD) > -1e-12)
         {
            element_int = true;
         }
      
         Array<int> faces;
         Array<int> cor;
         mesh->GetElementFaces(elem, faces, cor);
         FaceElementTransformations* FTrans =
            Trafo.mesh->GetFaceElementTransformations(faces[face]);
         FTrans->SetIntPoint(&(FaceIP[0]));

         Vector normal(Trafo.GetDimension());
         normal = 0.;
         if (face == 0 || face == 5)
            normal(2) = 1.;
         if (face == 1 || face == 3)
            normal(1) = 1.;
         if (face == 2 || face == 4)
            normal(0) = 1.;
         if (face == 0 || face == 1 || face == 4)
            normal *= -1.;

         for(int ip = 0; ip < FaceIP.Size(); ip++)
         {
            DenseMatrix shape;
            Vector point(3);
            IntegrationPoint ipoint;
            FTrans->Transform(FaceIP[ip], point);
            Trafo.TransformBack(point, ipoint);
            OrthoBasis3D(ipoint, shape);

            for(int dof = 0; dof < nBasis; dof++)
            {
               Vector grad(Trafo.GetSpaceDim());
               shape.GetRow(dof, grad);
               RHS(dof) -= (grad * normal) * FaceWeights(ip, faces[face]);
            }
         }
      }

      Vector scale(Size());

      // do integration over the area for integral over interface
      if (element_int && !interior)
      {
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
            for (int dof = 0; dof < fe->GetDof(); dof++)
            {
               dshape.GetRow(dof, gradi);
               gradi *= LevelSet(dofs[dof]);
               normal += gradi;
            }
            double normref = normal.Norml2();
            normal *= (-1. / normal.Norml2());

            scale(ip) = normphys / normref;

            DenseMatrix shapes;
            OrthoBasis3D(IntPoint(ip), shapes);

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
               temp2(i) = temp(i) / SVD.Singularvalue(i);
         SVD.RightSingularvectors().MultTranspose(temp2, ElemWeights);
      }

      // scale the weights
      for (int ip = 0; ip < GetNPoints(); ip++)
         Weights(ip, elem) = ElemWeights(ip) * scale(ip);
   }
}

void SIntegrationRule::Basis2D(const IntegrationPoint& ip, DenseMatrix& shape)
{
   shape.SetSize(nBasis, 2);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   for (int c = 0; c <= GetOrder(); c++)
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
   int count = 2 * GetOrder()+ 2;
   for (int c = 1; c <= GetOrder(); c++)
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
   const IntegrationRule *ir = &IntRules.Get(Geometry::SQUARE, 2*GetOrder()+1);

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

void SIntegrationRule::OrthoBasis3D(const IntegrationPoint& ip,
                                    DenseMatrix& shape)
{
   Vector X(3);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;
   X(2) = -1. + 2. * ip.z;

   DivFreeBasis::Get3DBasis(X, shape, GetOrder());
}

void SIntegrationRule::mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN,
                               int step)
{
   const IntegrationRule *ir = &IntRules.Get(Geometry::SQUARE, 2*GetOrder()+1);

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
   Trafo = Tr;
   if(Trafo.GetDimension() == 2)
      ComputeWeights2D();
   else if(Trafo.GetDimension() == 3)
      ComputeWeights3D();
}

void SIntegrationRule::UpdateInterface(Coefficient& levelset)
{
   LvlSet = levelset;
   if(Trafo.GetDimension() == 2)
      ComputeWeights2D();
   else if(Trafo.GetDimension() == 3)
      ComputeWeights3D();
}

void SIntegrationRule::SetElement(int ElementNo)
{
   for(int ip = 0; ip < GetNPoints(); ip++)
   {
      IntegrationPoint &intp = IntPoint(ip);
      intp.weight = Weights(ip, ElementNo);
   }
}

////////////////////////////////////////////////////////////////////////////////

CutIntegrationRule::CutIntegrationRule(int q, ElementTransformation& Tr,
                                       Coefficient &levelset)
   : IntegrationRule(), LvlSet(levelset), SIR(NULL), SVD(NULL)
{
   SetOrder(q);
   Tr.mesh->GetElementTransformation(0, &Trafo);
   nBasis = 0;
   if(Trafo.GetDimension() == 2)
      nBasis = (int)((GetOrder() + 1) * (GetOrder() + 2) / 2);
   else if(Trafo.GetDimension() == 3)
      for(int p = 0; p <= GetOrder(); p++)
         nBasis +=(int)((p + 1) * (p + 2) / 2);

   // set surface integration rule
   SIR = new SIntegrationRule(GetOrder() + 1, Tr, levelset);

   // get the quadrature points
   int qorder = 0;
   IntegrationRules irs(0, Quadrature1D::GaussLegendre);
   IntegrationRule ir = irs.Get(Trafo.GetGeometryType(), qorder);
   for (; ir.GetNPoints() < SIR->GetNPoints(); qorder++)
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
      intp.z = ir.IntPoint(ip).z;
      intp.weight = 0.;
      InteriorWeights(ip) = ir.IntPoint(ip).weight;
   }

   Weights.SetSize(GetNPoints(), Trafo.mesh->GetNE());
   Weights = 0.;

   if(Trafo.GetDimension() == 3)
   {
      FaceIP = SIR->FaceIP;
      FaceWeights.SetSize(SIR->FaceWeights.Height(), SIR->FaceWeights.Width());
      FaceWeights = 0.;
   }

   // assamble the matrix
   DenseMatrix Mat(nBasis, Size());
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      Vector shape;
      if(Trafo.GetDimension() == 2)
         Basis2D(ir.IntPoint(ip), shape);
      else if(Trafo.GetDimension() == 3)
         Basis3D(ir.IntPoint(ip), shape);
      Mat.SetCol(ip, shape);
   }

   // compute the svd for the matrix
   SVD = new DenseMatrixSVD(Mat, 'A', 'A');
   SVD->Eval(Mat);

   // compute the weights for the current element
   if(Trafo.GetDimension() == 2)
      ComputeWeights2D();
   else if(Trafo.GetDimension() == 3)
      ComputeWeights3D();
}

void CutIntegrationRule::ComputeWeights2D()
{
   Mesh* mesh = Trafo.mesh;
   H1_FECollection fec(9, 2);
   FiniteElementSpace fes(Trafo.mesh, &fec);
   GridFunction LevelSet(&fes);
   LevelSet.ProjectCoefficient(LvlSet);

   for(int elem = 0; elem < mesh->GetNE(); elem++)
   {
      Element* me = mesh->GetElement(elem);
      mesh->GetElementTransformation(elem, &Trafo);

      SIR->SetElement(elem);

      Vector RHS(nBasis);
      RHS = 0.;
      Vector ElemWeights(GetNPoints());
      ElemWeights = 0.;

      bool element_int = false;
      bool interior = true;
      Array<bool> edge_int;

      DenseMatrix PointA(me->GetNEdges(), Trafo.GetSpaceDim());
      DenseMatrix PointB(me->GetNEdges(), Trafo.GetSpaceDim());
      Vector edgelength(me->GetNEdges());

      Array<int> verts;
      mesh->GetElementVertices(elem, verts);

      // find the edges that are intersected by he surface and inside the area
      for (int edge = 0; edge < me->GetNEdges(); edge++)
      {
         enum class Layout {inside, intersected, outside};
         Layout layout;

         const int* vert = me->GetEdgeVertices(edge);
         Vector pointA(Trafo.GetSpaceDim());
         Vector pointB(Trafo.GetSpaceDim());
         for (int d = 0; d < Trafo.GetSpaceDim(); d++)
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
                  pointC = mid;
               else
                  pointB = mid;

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
            edge_int.Append(true);
         else
            edge_int.Append(false);
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
                                                       2*GetOrder()+1);

            Vector normal(Trafo.GetDimension());
            normal = 0.;
            if (edge == 0 || edge == 2)
               normal(1) = 1.;
            if (edge == 1 || edge == 3)
               normal(0) = 1.;
            if (edge == 0 || edge == 3)
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
                             * dist.Norml2() / edgelength(edge);
               }
            }
         }
      }

      // do the integration over the interface
      if (element_int && !interior)
      {
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
            for (int dof = 0; dof < fe->GetDof(); dof++)
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

            SIR->SetElement(elem);

            for (int dof = 0; dof < nBasis; dof++)
            {
               Vector adiv(2);
               shapes.GetRow(dof, adiv);
               // Add to RHS, scale by scale/2
               RHS(dof) += (adiv * normal) * SIR->IntPoint(ip).weight * scale;
            }
         }

         // solve the underdetermined linear system
         Vector temp(nBasis);
         Vector temp2(GetNPoints());
         temp2 = 0.;
         SVD->LeftSingularvectors().MultTranspose(RHS, temp);
         for (int i = 0; i < nBasis; i++)
            if (SVD->Singularvalue(i) > 1e-12)
               temp2(i) = temp(i) / SVD->Singularvalue(i);
         SVD->RightSingularvectors().MultTranspose(temp2, ElemWeights);
      }
      else if (interior)
         ElemWeights = InteriorWeights;
      else
         ElemWeights = 0.;

      for (int ip = 0; ip < GetNPoints(); ip++)
         Weights(ip, elem) = ElemWeights(ip);
   }
}

void CutIntegrationRule::ComputeWeights3D()
{
   FaceWeights = SIR->FaceWeights;

   Mesh* mesh = Trafo.mesh;
   H1_FECollection fec(9, 3);
   FiniteElementSpace fes(mesh, &fec);
   GridFunction LevelSet(&fes);
   LevelSet.ProjectCoefficient(LvlSet);
   
   for(int elem = 0; elem < mesh->GetNE(); elem++)
   {
      Element* me = Trafo.mesh->GetElement(Trafo.ElementNo);
      mesh->GetElementTransformation(elem, &Trafo);
      Trafo.SetIntPoint(&(IntPoint(0)));

      DenseMatrix Mat(nBasis, GetNPoints());
      Mat = 0.;
      Vector RHS(nBasis);
      RHS = 0.;
      Vector ElemWeights(GetNPoints());
      ElemWeights = 0.;

      bool element_int = false;
      bool interior = true;

      Array<int> verts;
      mesh->GetElementVertices(elem, verts);

      for(int face = 0; face < me->GetNFaces(); face++)
      {
         enum class Layout {inside, intersected, outside};
         Layout layout;

         const int* vert = me->GetFaceVertices(face);
         Vector pointA(Trafo.GetSpaceDim());
         Vector pointB(Trafo.GetSpaceDim());
         Vector pointC(Trafo.GetSpaceDim());
         Vector pointD(Trafo.GetSpaceDim());
         for (int d = 0; d < Trafo.GetSpaceDim(); d++)
         {
            pointA(d) = (Trafo.mesh->GetVertex(verts[vert[0]]))[d];
            pointB(d) = (Trafo.mesh->GetVertex(verts[vert[1]]))[d];
            pointC(d) = (Trafo.mesh->GetVertex(verts[vert[2]]))[d];
            pointD(d) = (Trafo.mesh->GetVertex(verts[vert[3]]))[d];
         }

         IntegrationPoint ipA;
         Trafo.TransformBack(pointA, ipA);
         IntegrationPoint ipB;
         Trafo.TransformBack(pointB, ipB);
         IntegrationPoint ipC;
         Trafo.TransformBack(pointC, ipC);
         IntegrationPoint ipD;
         Trafo.TransformBack(pointD, ipD);

         if (LvlSet.Eval(Trafo, ipA) < -1e-12
             || LvlSet.Eval(Trafo, ipB) < -1e-12
             || LvlSet.Eval(Trafo, ipC) < -1e-12
             || LvlSet.Eval(Trafo, ipD) < -1e-12)
         {
            interior = false;
         }

         if (LvlSet.Eval(Trafo, ipA) > -1e-12
             || LvlSet.Eval(Trafo, ipB) > -1e-12
             || LvlSet.Eval(Trafo, ipC) > -1e-12
             || LvlSet.Eval(Trafo, ipD) > -1e-12)
         {
            element_int = true;
         }
      
         Array<int> faces;
         Array<int> cor;
         mesh->GetElementFaces(elem, faces, cor);
         FaceElementTransformations* FTrans =
            Trafo.mesh->GetFaceElementTransformations(faces[face]);
         FTrans->SetIntPoint(&(FaceIP[0]));

         Vector normal(Trafo.GetDimension());
         normal = 0.;
         if (face == 0 || face == 5)
            normal(2) = 1.;
         if (face == 1 || face == 3)
            normal(1) = 1.;
         if (face == 2 || face == 4)
            normal(0) = 1.;
         if (face == 0 || face == 1 || face == 4)
            normal *= -1.;

         for(int ip = 0; ip < FaceIP.Size(); ip++)
         {
            DenseMatrix shape;
            Vector point(3);
            IntegrationPoint ipoint;
            FTrans->Transform(FaceIP[ip], point);
            Trafo.TransformBack(point, ipoint);
            BasisAntiDerivative3D(ipoint, shape);

            for(int dof = 0; dof < nBasis; dof++)
            {
               Vector adiv(Trafo.GetSpaceDim());
               shape.GetRow(dof, adiv);
               RHS(dof) += (adiv * normal) * FaceWeights(ip, faces[face]);
                          //* FTrans->Weight() * pow(Trafo.Weight(), 1./3.);
            }
         }
      }

      // do integration over the area for integral over interface
      if (element_int && !interior)
      {
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
            for (int dof = 0; dof < fe->GetDof(); dof++)
            {
               dshape.GetRow(dof, gradi);
               gradi *= LevelSet(dofs[dof]);
               normal += gradi;
            }
            double normref = normal.Norml2();
            normal *= (-1. / normal.Norml2());

            double scale = normref / normphys;

            SIR->SetElement(elem);

            DenseMatrix shapes;
            BasisAntiDerivative3D(SIR->IntPoint(ip), shapes);

            for (int dof = 0; dof < nBasis; dof++)
            {
               Vector adiv(3);
               shapes.GetRow(dof, adiv);
               RHS(dof) += (adiv * normal) * SIR->IntPoint(ip).weight * scale;
            }
         }

         // solve the underdetermined linear system
         Vector temp(nBasis);
         Vector temp2(GetNPoints());
         SVD->LeftSingularvectors().MultTranspose(RHS, temp);
         temp2 = 0.;
         for (int i = 0; i < nBasis; i++)
            if (SVD->Singularvalue(i) > 1e-12)
               temp2(i) = temp(i) / SVD->Singularvalue(i);
         SVD->RightSingularvectors().MultTranspose(temp2, ElemWeights);
      }
      else if(interior)
         ElemWeights = InteriorWeights;
      else
         ElemWeights = 0.;

      for (int ip = 0; ip < GetNPoints(); ip++)
         Weights(ip, elem) = ElemWeights(ip);// / Trafo.Weight();
   }
}


void CutIntegrationRule::Basis2D(const IntegrationPoint& ip, Vector& shape)
{
   shape.SetSize(nBasis);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   int count = 0;
   for (int c = 0; c <= GetOrder(); c++)
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
   for (int c = 0; c <= GetOrder(); c++)
   {
      for (int expo = 0; expo <= c; expo++)
      {
         shape(count, 0) = .25 * pow(X(0), (double)(expo + 1))
                           * pow(X(1), (double)(c - expo))
                           / (double)(expo + 1);
         shape(count, 1) = .25 * pow(X(0), (double)(expo))
                           * pow(X(1), (double)(c - expo + 1))
                           / (double)(c - expo + 1);
         count++;
      }
   }
}

void CutIntegrationRule::Basis3D(const IntegrationPoint& ip, Vector& shape)
{
   shape.SetSize(nBasis);

   Vector X(3);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;
   X(2) = -1. + 2. * ip.z;

   int count = 0;
   for(int c = 0; c <= GetOrder(); c++)
      for(int expo = 0; expo <= c; expo++)
         for(int expo2 = 0; expo2 <= c - expo; expo2++)
         {
            shape(count) = pow(X(0), (double)(expo))
                          * pow(X(1), (double)(expo2))
                          * pow(X(2), (double)(c - expo - expo2));
            count++;
         }
}

void CutIntegrationRule::BasisAntiDerivative3D(const IntegrationPoint& ip,
                                               DenseMatrix& shape)
{
   shape.SetSize(nBasis, 3);

   Vector X(3);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;
   X(2) = -1. + 2. * ip.z;

   int count = 0;
   for(int c = 0; c <= GetOrder(); c++)
      for(int expo = 0; expo <= c; expo++)
         for(int expo2 = 0; expo2 <= c - expo; expo2++)
         {
            shape(count, 0) = pow(X(0), (double)(expo + 1))
                             * pow(X(1), (double)(expo2))
                             * pow(X(2), (double)(c - expo - expo2))
                             / (6. * (double)(expo + 1));
            shape(count, 1) = pow(X(0), (double)(expo))
                             * pow(X(1), (double)(expo2 + 1))
                             * pow(X(2), (double)(c - expo - expo2))
                             / (6. * (double)(expo2 + 1));;
            shape(count, 2) = pow(X(0), (double)(expo))
                             * pow(X(1), (double)(expo2))
                             * pow(X(2), (double)(c - expo - expo2 + 1))
                             / (6. * (double)(c - expo + expo2 + 1));;
            count++;
         }
}

void CutIntegrationRule::Update(IsoparametricTransformation& Tr)
{
   Trafo = Tr;
   SIR->Update(Tr);
   if(Trafo.GetDimension() == 2)
      ComputeWeights2D();
   else if(Trafo.GetDimension() == 3)
      ComputeWeights3D();
}

void CutIntegrationRule::UpdateInterface(Coefficient& levelset)
{
   LvlSet = levelset;
   SIR->UpdateInterface(LvlSet);
   if(Trafo.GetDimension() == 2)
      ComputeWeights2D();
   else if(Trafo.GetDimension() == 3)
      ComputeWeights3D();
}

void CutIntegrationRule::SetElement(int ElementNo)
{
   SIR->SetElement(ElementNo);

   for(int ip = 0; ip < GetNPoints(); ip++)
   {
      IntegrationPoint &intp = IntPoint(ip);
      intp.weight = Weights(ip, ElementNo);
   }
}

CutIntegrationRule::~CutIntegrationRule() { delete SVD; delete SIR; }

}

#endif //MFEM_USE_LAPACK