// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "cut_quadrature.hpp"
#include "cut_quadrature_moment.hpp"
#include <cmath>
#include <limits>

using namespace std;

namespace mfem
{

#ifdef MFEM_USE_LAPACK

real_t MomentFittingIntRules::Eval(
   ElementTransformation &transformation,
   const IntegrationPoint &point) const
{
   Vector coordinates(transformation.GetSpaceDim());
   transformation.Transform(point, coordinates);
   IntegrationPoint reference;
   reference.Set(coordinates.GetData(), coordinates.Size());
   return level_set_sign * level_set.Eval(reference);
}

void MomentFittingIntRules::EvalGradient(
   ElementTransformation &transformation,
   const IntegrationPoint &point, Vector &gradient) const
{
   Vector coordinates(transformation.GetSpaceDim());
   transformation.Transform(point, coordinates);
   IntegrationPoint reference;
   reference.Set(coordinates.GetData(), coordinates.Size());
   Vector reference_gradient;
   level_set.EvalGradient(reference, reference_gradient);
   transformation.SetIntPoint(&point);
   transformation.Jacobian().MultTranspose(reference_gradient, gradient);
   gradient *= level_set_sign;
}

void MomentFittingIntRules::InitSurface(int order, ElementTransformation& Tr)
{
   Init(order);
   dim = Tr.GetDimension();
   if (Tr.GetDimension() == 1)
   {
      nBasis = -1;
      IntegrationRules irs(0, Quadrature1D::GaussLegendre);
      ir = irs.Get(Geometry::SEGMENT, 0);
   }
   else
   {
      if (Tr.GetDimension() == 2)
      {
         nBasis = 2 * (Order + 1) + static_cast<int>(Order * (Order + 1) / 2);
      }
      else if (Tr.GetDimension() == 3)
      {
         if (Order== 0) { nBasis = 3; }
         else if (Order== 1) { nBasis = 11; }
         else if (Order== 2) { nBasis = 26; }
         else if (Order== 3) { nBasis = 50; }
         else if (Order== 4) { nBasis = 85; }
         else if (Order== 5) { nBasis = 133; }
         else if (Order== 6) { nBasis = 196; }
         else if (Order>= 7) { nBasis = 276; Order = 7; }
      }

      // compute the quadrature points
      int qorder = 0;
      IntegrationRules irs(0, Quadrature1D::GaussLegendre);
      ir = irs.Get(Tr.GetGeometryType(), qorder);
      for (; ir.GetNPoints() <= nBasis; qorder++)
      {
         ir = irs.Get(Tr.GetGeometryType(), qorder);
      }
   }
   if (fixed_points && Tr.GetDimension() > 1) { ir = *fixed_points; }
}

void MomentFittingIntRules::InitVolume(int order, ElementTransformation& Tr)
{
   order++;
   InitSurface(order, Tr);
   Order--;

   nBasisVolume = 0;
   if (Tr.GetDimension() == 1)
   {
      nBasisVolume = -1;
      IntegrationRules irs(0, Quadrature1D::GaussLegendre);
      ir = irs.Get(Geometry::SEGMENT, Order);
   }
   else
   {
      if (Tr.GetDimension() == 2)
      {
         nBasisVolume = (int)((Order + 1) * (Order + 2) / 2);
      }
      else if (Tr.GetDimension() == 3)
      {
         nBasisVolume = (int)((Order + 1) * (Order + 2) * (Order + 3) / 6);
      }

      // assemble the matrix
      DenseMatrix Mat(nBasisVolume, ir.GetNPoints());
      for (int ip = 0; ip < ir.GetNPoints(); ip++)
      {
         Vector shape;
         if (Tr.GetDimension() == 2)
         {
            Basis2D(ir.IntPoint(ip), shape);
         }
         else if (Tr.GetDimension() == 3)
         {
            Basis3D(ir.IntPoint(ip), shape);
         }

         Mat.SetCol(ip, shape);
      }

      // compute the SVD for the matrix
      VolumeSVD = new DenseMatrixSVD(Mat, 'A', 'A');
      VolumeSVD->Eval(Mat);
   }

}

void MomentFittingIntRules::ComputeFaceWeights(ElementTransformation& Tr)
{
   int elem = Tr.ElementNo;
   const Mesh *mesh = Tr.mesh;

   if (FaceIP.Size() == 0)
   {
      FaceWeightsComp.SetSize(mesh->GetNFaces());
      FaceWeightsComp = 0.;
   }

   const Element* me = mesh->GetElement(elem);
   IsoparametricTransformation Trafo;
   mesh->GetElementTransformation(elem, &Trafo);

   Array<int> faces;
   Array<int> cor;
   mesh->GetElementFaces(elem, faces, cor);

   for (int face = 0; face < me->GetNFaces(); face++)
   {
      if (FaceWeightsComp(faces[face]) == 0.)
      {
         FaceWeightsComp(faces[face]) = 1.;
         Array<int> verts;
         mesh->GetFaceVertices(faces[face], verts);
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

         // TODO - don't we lose the curvature with this local mesh setup?
         Mesh local_mesh(2,4,1,0,3);
         local_mesh.AddVertex(pointA);
         local_mesh.AddVertex(pointB);
         local_mesh.AddVertex(pointC);
         local_mesh.AddVertex(pointD);
         local_mesh.AddQuad(0,1,2,3);
         local_mesh.FinalizeQuadMesh(1);
         IsoparametricTransformation faceTrafo;
         local_mesh.GetElementTransformation(0, &faceTrafo);

         // The 3D face integrals are computed as 2D volumetric integrals.
         // The 2D face integrals are computed as 1D volumetric integrals.
         MomentFittingIntRules FaceRules(Order, level_set, level_set_sign);
         IntegrationRule FaceRule;
         FaceRules.GetVolumeIntegrationRule(faceTrafo, FaceRule);
         if (FaceIP.Size() != FaceRule.Size())
         {
            FaceIP.SetSize(FaceRule.Size());
            for (int ip = 0; ip < FaceRule.GetNPoints(); ip++)
            {
               FaceIP[ip].index = ip;
               IntegrationPoint &intp = FaceIP[ip];
               intp.x = FaceRule.IntPoint(ip).x;
               intp.y = FaceRule.IntPoint(ip).y;
               intp.weight = 0.;
            }

            FaceWeights.SetSize(FaceRule.GetNPoints(), mesh->GetNFaces());
            FaceWeights = 0.;

            FaceWeightsComp = 0.;
            FaceWeightsComp(faces[face]) = 1.;
         }

         for (int ip = 0; ip < FaceRule.GetNPoints(); ip++)
         {
            FaceWeights(ip, faces[face]) = FaceRule.IntPoint(ip).weight;
         }
      }
   }

   mesh->GetElementTransformation(elem, &Trafo);
}

void MomentFittingIntRules::ComputeSurfaceWeights1D(ElementTransformation& Tr)
{
   IntegrationPoint& intp = ir.IntPoint(0);

   IntegrationPoint ip0;
   ip0.x = 0.;
   IntegrationPoint ip1;
   ip1.x = 1.;
   Tr.SetIntPoint(&ip0);
   if (Eval(Tr, ip0) * Eval(Tr, ip1) < 0.)
   {
      IntegrationPoint ip2;
      ip2.x = .5;
      while (Eval(Tr, ip2) > tol_1
             || Eval(Tr, ip2) < -tol_1)
      {
         if (Eval(Tr, ip0) * Eval(Tr, ip2) < 0.)
         {
            ip1.x = ip2.x;
         }
         else
         {
            ip0.x = ip2.x;
         }

         ip2.x = (ip1.x + ip0.x) / 2.;
      }
      intp.x = ip2.x;
      intp.weight = 1. / Tr.Weight();
   }
   else if (Eval(Tr, ip0) > 0. && Eval(Tr, ip1) <= tol_1)
   {
      intp.x = 1.;
      intp.weight = 1. / Tr.Weight();
   }
   else if (Eval(Tr, ip1) > 0. && Eval(Tr, ip0) <= tol_1)
   {
      intp.x = 0.;
      intp.weight = 1. / Tr.Weight();
   }
   else
   {
      intp.x = .5;
      intp.weight = 0.;
   }
}

real_t MomentFittingIntRules::Bisect(ElementTransformation &Tr) const
{
   IntegrationPoint intp;

   IntegrationPoint ip0;
   ip0.x = 0.;
   IntegrationPoint ip1;
   ip1.x = 1.;
   Tr.SetIntPoint(&ip0);
   if (Eval(Tr, ip0) * Eval(Tr, ip1) < 0.)
   {
      IntegrationPoint ip2;
      ip2.x = .5;
      while (Eval(Tr, ip2) > 1e-12
             || Eval(Tr, ip2) < -1e-12)
      {
         if (Eval(Tr, ip0) * Eval(Tr, ip2) < 0.)
         {
            ip1.x = ip2.x;
         }
         else
         {
            ip0.x = ip2.x;
         }

         ip2.x = (ip1.x + ip0.x) / 2.;
      }
      intp.x = ip2.x;
      intp.weight = 1. / Tr.Weight();
   }
   else if (Eval(Tr, ip0) > 0. && Eval(Tr, ip1) <= 1e-12)
   {
      intp.x = 1.;
      intp.weight = 1. / Tr.Weight();
   }
   else if (Eval(Tr, ip1) > 0. && Eval(Tr, ip0) <= 1e-12)
   {
      intp.x = 0.;
      intp.weight = 1. / Tr.Weight();
   }
   else
   {
      intp.x = .5;
      intp.weight = 0.;
   }

   return intp.x;
}

void MomentFittingIntRules::ComputeVolumeWeights1D(ElementTransformation& Tr)
{
   IntegrationRules irs(0, Quadrature1D::GaussLegendre);
   IntegrationRule ir2 = irs.Get(Geometry::SEGMENT, ir.GetOrder());

   IntegrationPoint ip0;
   ip0.x = 0.;
   IntegrationPoint ip1;
   ip1.x = 1.;
   Tr.SetIntPoint(&ip0);
   if (Eval(Tr, ip0) * Eval(Tr, ip1) < 0.)
   {
      Vector tempX(ir.GetNPoints());
      real_t length;
      if (Eval(Tr, ip0) > 0.)
      {
         length = Bisect(Tr);
         for (int ip = 0; ip < ir.GetNPoints(); ip++)
         {
            IntegrationPoint &intp = ir.IntPoint(ip);
            intp.x = ir2.IntPoint(ip).x * length;
            intp.weight = ir2.IntPoint(ip).weight * length;
         }
      }
      else
      {
         length = 1. - Bisect(Tr);
         for (int ip = 0; ip < ir.GetNPoints(); ip++)
         {
            IntegrationPoint &intp = ir.IntPoint(ip);
            intp.x = Bisect(Tr) + ir2.IntPoint(ip).x * length;
            intp.weight = ir2.IntPoint(ip).weight * length;
         }
      }
   }
   else if (Eval(Tr, ip0) <= -tol_1
            || Eval(Tr, ip1) <= -tol_1)
   {
      for (int ip = 0; ip < ir.GetNPoints(); ip++)
      {
         IntegrationPoint &intp = ir.IntPoint(ip);
         intp.x = ir2.IntPoint(ip).x;
         intp.weight = 0.;
      }
   }
   else
   {
      ir = ir2;
   }
}

void MomentFittingIntRules::ComputeSurfaceWeights2D(ElementTransformation& Tr)
{
   int elem = Tr.ElementNo;
   const Mesh* mesh = Tr.mesh;

   const Element* me = mesh->GetElement(elem);
   IsoparametricTransformation Trafo;
   mesh->GetElementTransformation(elem, &Trafo);

   DenseMatrix Mat(nBasis, ir.GetNPoints());
   Mat = 0.;
   Vector RHS(nBasis);
   RHS = 0.;
   Vector ElemWeights(ir.GetNPoints());
   ElemWeights = 0.;

   bool element_int = false;
   bool interior = true;
   Array<bool> edge_int;

   DenseMatrix PointA(me->GetNEdges(), Trafo.GetSpaceDim());
   DenseMatrix PointB(me->GetNEdges(), Trafo.GetSpaceDim());
   Vector edgelength(me->GetNEdges());

   Array<int> verts;
   mesh->GetElementVertices(elem, verts);

   // find the edges that are intersected by the surface and inside the area
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

      if (Eval(Trafo, ipA) < -tol_1
          || Eval(Trafo, ipB) < -tol_1)
      {
         interior = false;
      }

      if (Eval(Trafo, ipA) > -tol_1
          && Eval(Trafo, ipB) > -tol_1)
      {
         layout = Layout::inside;
      }
      else if (Eval(Trafo, ipA) > tol_2
               && Eval(Trafo, ipB) <= 0.)
      {
         layout = Layout::intersected;
      }
      else if (Eval(Trafo, ipA) <= 0.
               && Eval(Trafo, ipB) > tol_2)
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

      // Store the end points of the (1D) intersected edge.
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

         while (Eval(Trafo, ip) > tol_1
                || Eval(Trafo, ip) < -tol_1)
         {
            if (Eval(Trafo, ip) > tol_1)
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

   // Integrate over the 1D edges.
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
         if (edge == 0 || edge == 2)
         {
            normal(1) = 1.;
         }
         if (edge == 1 || edge == 3)
         {
            normal(0) = 1.;
         }
         if (edge == 0 || edge == 3)
         {
            normal *= -1.;
         }

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
               RHS(dof) -= (grad * normal) * ir2->IntPoint(ip).weight
                           * dist.Norml2() / edgelength(edge);
            }
         }
      }
   }

   // do integration over the area for integral over interface
   if (element_int && !interior)
   {
      mesh->GetElementTransformation(elem, &Trafo);
      Vector normal(Trafo.GetDimension());

      for (int ip = 0; ip < ir.GetNPoints(); ip++)
      {
         EvalGradient(Trafo, ir.IntPoint(ip), normal);
         normal *= (-1. / normal.Norml2());

         DenseMatrix shapes;
         OrthoBasis2D(ir.IntPoint(ip), shapes);

         for (int dof = 0; dof < nBasis; dof++)
         {
            Vector grad(Trafo.GetSpaceDim());
            shapes.GetRow(dof, grad);
            Mat(dof, ip) =  (grad * normal);
         }
      }

      // solve the underdetermined linear system
      Vector temp(nBasis);
      Vector temp2(ir.GetNPoints());
      DenseMatrixSVD SVD(Mat, 'A', 'A');
      SVD.Eval(Mat);
      SVD.LeftSingularvectors().MultTranspose(RHS, temp);
      temp2 = 0.;
      for (int i = 0; i < nBasis; i++)
      {
         if (SVD.Singularvalue(i) > tol_1)
         {
            temp2(i) = temp(i) / SVD.Singularvalue(i);
         }
      }
      SVD.RightSingularvectors().MultTranspose(temp2, ElemWeights);
   }

   // save the weights
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      IntegrationPoint& intp = ir.IntPoint(ip);
      intp.weight = ElemWeights(ip);
   }

   mesh->GetElementTransformation(elem, &Trafo);
}

void MomentFittingIntRules::ComputeVolumeWeights2D(ElementTransformation& Tr,
                                                   const IntegrationRule* sir)
{
   int elem = Tr.ElementNo;
   const Mesh* mesh = Tr.mesh;

   const Element* me = mesh->GetElement(elem);
   IsoparametricTransformation Trafo;
   mesh->GetElementTransformation(elem, &Trafo);

   Vector RHS(nBasisVolume);
   RHS = 0.;
   Vector ElemWeights(ir.GetNPoints());
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

      if (Eval(Trafo, ipA) < -tol_1
          || Eval(Trafo, ipB) < -tol_1)
      {
         interior = false;
      }

      if (Eval(Trafo, ipA) > -tol_1
          && Eval(Trafo, ipB) > -tol_1)
      {
         layout = Layout::inside;
      }
      else if (Eval(Trafo, ipA) > tol_2
               && Eval(Trafo, ipB) <= 0.)
      {
         layout = Layout::intersected;
      }
      else if (Eval(Trafo, ipA) <= 0.
               && Eval(Trafo, ipB) > tol_2)
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

         while (Eval(Trafo, ip) > tol_1
                || Eval(Trafo, ip) < -tol_1)
         {
            if (Eval(Trafo, ip) > tol_1)
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
         if (edge == 0 || edge == 2)
         {
            normal(1) = 1.;
         }
         if (edge == 1 || edge == 3)
         {
            normal(0) = 1.;
         }
         if (edge == 0 || edge == 3)
         {
            normal *= -1.;
         }

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
            BasisAD2D(intpoint, shapes);
            Vector adiv(Trafo.GetDimension());

            for (int dof = 0; dof < nBasisVolume; dof++)
            {
               shapes.GetRow(dof, adiv);
               RHS(dof) += (adiv * normal) * ir2->IntPoint(ip).weight
                           * dist.Norml2() / edgelength(edge);
            }
         }
      }
   }

   // Integrate over the interface using the already computed surface rule, and
   // solve the linear system for the weights.
   if (element_int && !interior)
   {
      mesh->GetElementTransformation(elem, &Trafo);
      Vector normal(Trafo.GetDimension());

      for (int ip = 0; ip < sir->GetNPoints(); ip++)
      {
         EvalGradient(Trafo, sir->IntPoint(ip), normal);
         normal *= (-1. / normal.Norml2());

         DenseMatrix shapes;
         BasisAD2D(sir->IntPoint(ip), shapes);

         for (int dof = 0; dof < nBasisVolume; dof++)
         {
            Vector adiv(2);
            shapes.GetRow(dof, adiv);
            RHS(dof) += (adiv * normal) * sir->IntPoint(ip).weight;
         }
      }

      // solve the underdetermined linear system
      Vector temp(nBasisVolume);
      Vector temp2(ir.GetNPoints());
      temp2 = 0.;
      VolumeSVD->LeftSingularvectors().MultTranspose(RHS, temp);
      for (int i = 0; i < nBasisVolume; i++)
      {
         if (VolumeSVD->Singularvalue(i) > tol_1)
         {
            temp2(i) = temp(i) / VolumeSVD->Singularvalue(i);
         }
      }
      VolumeSVD->RightSingularvectors().MultTranspose(temp2, ElemWeights);
   }

   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      IntegrationPoint& intp = ir.IntPoint(ip);
      intp.weight = ElemWeights(ip);
   }

   if (interior)
   {
      int qorder = 0;
      IntegrationRules irs(0, Quadrature1D::GaussLegendre);
      IntegrationRule ir2 = irs.Get(Trafo.GetGeometryType(), qorder);
      for (; ir2.GetNPoints() < ir.GetNPoints(); qorder++)
      {
         ir2 = irs.Get(Trafo.GetGeometryType(), qorder);
      }
      ir = ir2;
   }

   mesh->GetElementTransformation(elem, &Trafo);
}

void MomentFittingIntRules::ComputeSurfaceWeights3D(ElementTransformation& Tr)
{
   ComputeFaceWeights(Tr);

   int elem = Tr.ElementNo;
   const Mesh* mesh = Tr.mesh;

   const Element* me = mesh->GetElement(elem);
   IsoparametricTransformation Trafo;
   mesh->GetElementTransformation(elem, &Trafo);

   DenseMatrix Mat(nBasis, ir.GetNPoints());
   Mat = 0.;
   Vector RHS(nBasis);
   RHS = 0.;
   Vector ElemWeights(ir.GetNPoints());
   ElemWeights = 0.;

   // Does the element have a positive vertex?
   bool element_int = false;
   // Are all element vertices positive?
   bool interior = true;

   Array<int> verts;
   mesh->GetElementVertices(elem, verts);

   for (int face = 0; face < me->GetNFaces(); face++)
   {
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

      if (Eval(Trafo, ipA) < -tol_1
          || Eval(Trafo, ipB) < -tol_1
          || Eval(Trafo, ipC) < -tol_1
          || Eval(Trafo, ipD) < -tol_1)
      {
         interior = false;
      }

      if (Eval(Trafo, ipA) > -tol_1
          || Eval(Trafo, ipB) > -tol_1
          || Eval(Trafo, ipC) > -tol_1
          || Eval(Trafo, ipD) > -tol_1)
      {
         element_int = true;
      }

      Array<int> faces;
      Array<int> cor;
      mesh->GetElementFaces(elem, faces, cor);

      IsoparametricTransformation Tr1, Tr2;
      FaceElementTransformations FTrans;
      Trafo.mesh->GetFaceElementTransformations(faces[face], FTrans, Tr1, Tr2);
      FTrans.SetIntPoint(&(FaceIP[0]));

      Vector normal(Trafo.GetDimension());
      normal = 0.;
      if (face == 0 || face == 5)
      {
         normal(2) = 1.;
      }
      if (face == 1 || face == 3)
      {
         normal(1) = 1.;
      }
      if (face == 2 || face == 4)
      {
         normal(0) = 1.;
      }
      if (face == 0 || face == 1 || face == 4)
      {
         normal *= -1.;
      }

      for (int ip = 0; ip < FaceIP.Size(); ip++)
      {
         DenseMatrix shape;
         Vector point(3);
         IntegrationPoint ipoint;
         FTrans.Transform(FaceIP[ip], point);
         Trafo.TransformBack(point, ipoint);
         OrthoBasis3D(ipoint, shape);

         for (int dof = 0; dof < nBasis; dof++)
         {
            Vector grad(Trafo.GetSpaceDim());
            shape.GetRow(dof, grad);
            RHS(dof) -= (grad * normal) * FaceWeights(ip, faces[face]);
         }
      }
   }

   // If the element is intersected, form the matrix and solve for the weights.
   if (element_int && !interior)
   {
      mesh->GetElementTransformation(elem, &Trafo);
      Vector normal(Trafo.GetDimension());

      // Form the matrix.
      for (int ip = 0; ip < ir.GetNPoints(); ip++)
      {
         EvalGradient(Trafo, ir.IntPoint(ip), normal);
         normal *= (-1. / normal.Norml2());

         DenseMatrix shapes;
         OrthoBasis3D(ir.IntPoint(ip), shapes);

         for (int dof = 0; dof < nBasis; dof++)
         {
            Vector grad(Trafo.GetSpaceDim());
            shapes.GetRow(dof, grad);
            Mat(dof, ip) =  (grad * normal);
         }
      }

      // solve the underdetermined linear system
      Vector temp(nBasis);
      Vector temp2(ir.GetNPoints());
      DenseMatrixSVD SVD(Mat, 'A', 'A');
      SVD.Eval(Mat);
      SVD.LeftSingularvectors().MultTranspose(RHS, temp);
      temp2 = 0.;
      for (int i = 0; i < nBasis; i++)
      {
         if (SVD.Singularvalue(i) > tol_1)
         {
            temp2(i) = temp(i) / SVD.Singularvalue(i);
         }
      }
      SVD.RightSingularvectors().MultTranspose(temp2, ElemWeights);
   }

   // scale the weights
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      IntegrationPoint& intp = ir.IntPoint(ip);
      intp.weight = ElemWeights(ip);
   }

   mesh->GetElementTransformation(elem, &Trafo);
}

void MomentFittingIntRules::ComputeVolumeWeights3D(ElementTransformation& Tr,
                                                   const IntegrationRule* sir)
{
   Order++;
   ComputeFaceWeights(Tr);
   Order--;

   int elem = Tr.ElementNo;
   const Mesh* mesh = Tr.mesh;

   const Element* me = mesh->GetElement(elem);
   IsoparametricTransformation Trafo;
   mesh->GetElementTransformation(elem, &Trafo);

   Vector RHS(nBasisVolume);
   RHS = 0.;
   Vector ElemWeights(ir.GetNPoints());
   ElemWeights = 0.;

   // Does the element have a positive vertex?
   bool element_int = false;
   // Are all element vertices positive?
   bool interior = true;

   Array<int> verts;
   mesh->GetElementVertices(elem, verts);

   for (int face = 0; face < me->GetNFaces(); face++)
   {
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

      if (Eval(Trafo, ipA) < -tol_1
          || Eval(Trafo, ipB) < -tol_1
          || Eval(Trafo, ipC) < -tol_1
          || Eval(Trafo, ipD) < -tol_1)
      {
         interior = false;
      }

      if (Eval(Trafo, ipA) > -tol_1
          || Eval(Trafo, ipB) > -tol_1
          || Eval(Trafo, ipC) > -tol_1
          || Eval(Trafo, ipD) > -tol_1)
      {
         element_int = true;
      }

      Array<int> faces;
      Array<int> cor;
      mesh->GetElementFaces(elem, faces, cor);

      IsoparametricTransformation Tr1, Tr2;
      FaceElementTransformations FTrans;
      Trafo.mesh->GetFaceElementTransformations(faces[face], FTrans, Tr1, Tr2);

      FTrans.SetIntPoint(&(FaceIP[0]));

      Vector normal(Trafo.GetDimension());
      normal = 0.;
      if (face == 0 || face == 5)
      {
         normal(2) = 1.;
      }
      if (face == 1 || face == 3)
      {
         normal(1) = 1.;
      }
      if (face == 2 || face == 4)
      {
         normal(0) = 1.;
      }
      if (face == 0 || face == 1 || face == 4)
      {
         normal *= -1.;
      }

      for (int ip = 0; ip < FaceIP.Size(); ip++)
      {
         DenseMatrix shape;
         Vector point(3);
         IntegrationPoint ipoint;
         FTrans.Transform(FaceIP[ip], point);
         Trafo.TransformBack(point, ipoint);
         BasisAD3D(ipoint, shape);

         for (int dof = 0; dof < nBasisVolume; dof++)
         {
            Vector adiv(Trafo.GetSpaceDim());
            shape.GetRow(dof, adiv);
            RHS(dof) += (adiv * normal) * FaceWeights(ip, faces[face]);
         }
      }
   }

   // If the element is intersected, integrate over the cut surface (using the
   // already computed rule) and solve the matrix for the weights.
   if (element_int && !interior)
   {
      mesh->GetElementTransformation(elem, &Trafo);
      Vector normal(Trafo.GetDimension());

      // Integrate over the cut surface using the already computed rule.
      for (int ip = 0; ip < sir->GetNPoints(); ip++)
      {
         EvalGradient(Trafo, sir->IntPoint(ip), normal);
         normal *= (-1. / normal.Norml2());

         DenseMatrix shapes;
         BasisAD3D(sir->IntPoint(ip), shapes);

         for (int dof = 0; dof < nBasisVolume; dof++)
         {
            Vector adiv(Trafo.GetSpaceDim());
            shapes.GetRow(dof, adiv);
            RHS(dof) +=  (adiv * normal) * sir->IntPoint(ip).weight;
         }
      }

      // solve the underdetermined linear system
      Vector temp(nBasisVolume);
      Vector temp2(ir.GetNPoints());
      VolumeSVD->LeftSingularvectors().MultTranspose(RHS, temp);
      temp2 = 0.;
      for (int i = 0; i < nBasisVolume; i++)
         if (VolumeSVD->Singularvalue(i) > tol_1)
         {
            temp2(i) = temp(i) / VolumeSVD->Singularvalue(i);
         }
      VolumeSVD->RightSingularvectors().MultTranspose(temp2, ElemWeights);
   }

   // scale the weights
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      IntegrationPoint& intp = ir.IntPoint(ip);
      intp.weight = ElemWeights(ip);
   }

   // Fully inside the subdomain -> standard integration.
   if (interior)
   {
      int qorder = 0;
      IntegrationRules irs(0, Quadrature1D::GaussLegendre);
      IntegrationRule ir2 = irs.Get(Trafo.GetGeometryType(), qorder);
      for (; ir2.GetNPoints() < ir.GetNPoints(); qorder++)
      {
         ir2 = irs.Get(Trafo.GetGeometryType(), qorder);
      }
      ir = ir2;
   }

   mesh->GetElementTransformation(elem, &Trafo);
}

void MomentFittingIntRules::DivFreeBasis2D(const IntegrationPoint& ip,
                                           DenseMatrix& shape)
{
   shape.SetSize(nBasis, 2);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   for (int c = 0; c <= Order; c++)
   {
      Vector a(2);
      a = 0.;
      a(1) = pow(X(0), (real_t)(c));

      Vector b(2);
      b = 0.;
      b(0) = pow(X(1), (real_t)(c));

      shape.SetRow(2 * c, a);
      shape.SetRow(2 * c + 1, b);
   }

   Poly_1D poly;
   int count = 2 * Order+ 2;
   for (int c = 1; c <= Order; c++)
   {
      const int* factorial = poly.Binom(c);
      for (int expo = c; expo > 0; expo--)
      {
         Vector a(2);
         a(0) = (real_t)(factorial[expo]) * pow(X(0), (real_t)(expo))
                *  pow(X(1), (real_t)(c - expo));
         a(1) = -1. * (real_t)(factorial[expo - 1])
                * pow(X(0), (real_t)(expo - 1))
                * pow(X(1), (real_t)(c - expo + 1));

         shape.SetRow(count, a);
         count++;
      }
   }
}

void MomentFittingIntRules::OrthoBasis2D(const IntegrationPoint& ip,
                                         DenseMatrix& shape)
{
   const IntegrationRule *ir_ = &IntRules.Get(Geometry::SQUARE, 2*Order+1);

   shape.SetSize(nBasis, 2);

   // evaluate basis in the point
   DenseMatrix preshape(nBasis, 2);
   DivFreeBasis2D(ip, shape);

   // evaluate basis for quadrature points
   DenseTensor shapeMFN(nBasis, 2, ir_->GetNPoints());
   for (int p = 0; p < ir_->GetNPoints(); p++)
   {
      DenseMatrix shapeN(nBasis, 2);
      DivFreeBasis2D(ir_->IntPoint(p), shapeN);
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

void MomentFittingIntRules::OrthoBasis3D(const IntegrationPoint& ip,
                                         DenseMatrix& shape)
{
   Vector X(3);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;
   X(2) = -1. + 2. * ip.z;

   DivFreeBasis::GetDivFree3DBasis(X, shape, Order);
}

void MomentFittingIntRules::mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN,
                                    int step)
{
   const IntegrationRule *ir_ = &IntRules.Get(Geometry::SQUARE, 2*Order+1);

   for (int count = step; count < shape.Height(); count++)
   {
      real_t den = 0.;
      real_t num = 0.;

      for (int ip = 0; ip < ir_->GetNPoints(); ip++)
      {
         Vector u(2);
         Vector v(2);

         shapeMFN(ip).GetRow(count, u);
         shapeMFN(ip).GetRow(step - 1, v);

         den += v * v * ir_->IntPoint(ip).weight;
         num += u * v * ir_->IntPoint(ip).weight;
      }

      real_t coeff = num / den;

      Vector s(2);
      Vector t(2);
      shape.GetRow(step - 1, s);
      shape.GetRow(count, t);
      s *= coeff;
      t += s;
      shape.SetRow(count, t);

      for (int ip = 0; ip < ir_->GetNPoints(); ip++)
      {
         shapeMFN(ip).GetRow(step - 1, s);
         shapeMFN(ip).GetRow(count, t);
         s *= coeff;
         t += s;
         shapeMFN(ip).SetRow(count, t);
      }
   }
}

void MomentFittingIntRules::Basis2D(const IntegrationPoint& ip, Vector& shape)
{
   shape.SetSize(nBasisVolume);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   int count = 0;
   for (int c = 0; c <= Order; c++)
   {
      for (int expo = 0; expo <= c; expo++)
      {
         shape(count) = pow(X(0), (real_t)(expo))
                        * pow(X(1), (real_t)(c - expo));
         count++;
      }
   }
}

void MomentFittingIntRules::BasisAD2D(const IntegrationPoint& ip,
                                      DenseMatrix& shape)
{
   shape.SetSize(nBasisVolume, 2);

   Vector X(2);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;

   int count = 0;
   for (int c = 0; c <= Order; c++)
   {
      for (int expo = 0; expo <= c; expo++)
      {
         shape(count, 0) = .25 * pow(X(0), (real_t)(expo + 1))
                           * pow(X(1), (real_t)(c - expo))
                           / (real_t)(expo + 1);
         shape(count, 1) = .25 * pow(X(0), (real_t)(expo))
                           * pow(X(1), (real_t)(c - expo + 1))
                           / (real_t)(c - expo + 1);
         count++;
      }
   }
}

void MomentFittingIntRules::Basis3D(const IntegrationPoint& ip, Vector& shape)
{
   shape.SetSize(nBasisVolume);

   Vector X(3);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;
   X(2) = -1. + 2. * ip.z;

   int count = 0;
   for (int c = 0; c <= Order; c++)
      for (int expo = 0; expo <= c; expo++)
         for (int expo2 = 0; expo2 <= c - expo; expo2++)
         {
            shape(count) = pow(X(0), (real_t)(expo))
                           * pow(X(1), (real_t)(expo2))
                           * pow(X(2), (real_t)(c - expo - expo2));
            count++;
         }
}

void MomentFittingIntRules::BasisAD3D(const IntegrationPoint& ip,
                                      DenseMatrix& shape)
{
   shape.SetSize(nBasisVolume, 3);

   Vector X(3);
   X(0) = -1. + 2. * ip.x;
   X(1) = -1. + 2. * ip.y;
   X(2) = -1. + 2. * ip.z;

   int count = 0;
   for (int c = 0; c <= Order; c++)
      for (int expo = 0; expo <= c; expo++)
         for (int expo2 = 0; expo2 <= c - expo; expo2++)
         {
            shape(count, 0) = pow(X(0), (real_t)(expo + 1))
                              * pow(X(1), (real_t)(expo2))
                              * pow(X(2), (real_t)(c - expo - expo2))
                              / (6. * (real_t)(expo + 1));
            shape(count, 1) = pow(X(0), (real_t)(expo))
                              * pow(X(1), (real_t)(expo2 + 1))
                              * pow(X(2), (real_t)(c - expo - expo2))
                              / (6. * (real_t)(expo2 + 1));;
            shape(count, 2) = pow(X(0), (real_t)(expo))
                              * pow(X(1), (real_t)(expo2))
                              * pow(X(2), (real_t)(c - expo - expo2 + 1))
                              / (6. * (real_t)(c - expo + expo2 + 1));;
            count++;
         }
}

void MomentFittingIntRules::Clear()
{
   dim = -1;
   nBasis = -1;
   nBasisVolume = -1;
   delete VolumeSVD;
   VolumeSVD = NULL;
   FaceIP.DeleteAll();
   FaceWeights = 0.;
   FaceWeightsComp = 0.;
}

void MomentFittingIntRules::GetSurfaceIntegrationRule(ElementTransformation& Tr,
                                                      IntegrationRule& result)
{
   if (nBasis == -1 || dim != Tr.GetDimension())
   {
      Clear();
      InitSurface(Order, Tr);
   }

   if (Tr.GetDimension() == 3)
   {
      FaceIP.DeleteAll();
      FaceWeights = 0.;
      FaceWeightsComp = 0.;
   }

   if (Tr.GetDimension() == 1)
   {
      ComputeSurfaceWeights1D(Tr);
   }
   else if (Tr.GetDimension() == 2)
   {
      ComputeSurfaceWeights2D(Tr);
   }
   else if (Tr.GetDimension() == 3)
   {
      ComputeSurfaceWeights3D(Tr);
   }

   result.SetSize(ir.GetNPoints());
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      result.IntPoint(ip).index = ip;
      IntegrationPoint &intp = result.IntPoint(ip);
      intp.x = ir.IntPoint(ip).x;
      intp.y = ir.IntPoint(ip).y;
      intp.z = ir.IntPoint(ip).z;
      intp.weight = ir.IntPoint(ip).weight;
   }
}

void MomentFittingIntRules::GetVolumeIntegrationRule(ElementTransformation& Tr,
                                                     IntegrationRule& result,
                                                     const IntegrationRule* sir)
{
   if (nBasis == -1 || nBasisVolume == -1 || dim != Tr.GetDimension())
   {
      Clear();
      InitVolume(Order, Tr);
   }

   if (Tr.GetDimension() == 3)
   {
      FaceIP.DeleteAll();
      FaceWeights = 0.;
      FaceWeightsComp = 0.;
   }

   IntegrationRule SIR;

   if (Tr.GetDimension() == 1)
   {
      Clear();
      InitVolume(Order, Tr);
   }
   else if (sir == NULL)
   {
      Order++;
      GetSurfaceIntegrationRule(Tr, SIR);
      Order--;
   }
   else if (sir->GetOrder() - 1 != ir.GetOrder())
   {
      Order++;
      GetSurfaceIntegrationRule(Tr, SIR);
      Order--;
   }
   else { SIR = *sir; }

   if (Tr.GetDimension() == 1)
   {
      ComputeVolumeWeights1D(Tr);
   }
   else if (Tr.GetDimension() == 2)
   {
      ComputeVolumeWeights2D(Tr, &SIR);
   }
   else if (Tr.GetDimension() == 3)
   {
      ComputeVolumeWeights3D(Tr, &SIR);
   }

   result.SetSize(ir.GetNPoints());
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      result.IntPoint(ip).index = ip;
      IntegrationPoint &intp = result.IntPoint(ip);
      intp.x = ir.IntPoint(ip).x;
      intp.y = ir.IntPoint(ip).y;
      intp.z = ir.IntPoint(ip).z;
      intp.weight = ir.IntPoint(ip).weight;
   }
}

#endif // MFEM_USE_LAPACK

namespace
{

CutQuadratureReport MomentFailure(CutQuadratureReport::Status status,
                                  const std::string &message)
{ return {status, message}; }

Mesh ReferenceMesh(Geometry::Type geometry)
{
   if (geometry == Geometry::SEGMENT)
   { return Mesh::MakeCartesian1D(1); }
   if (geometry == Geometry::SQUARE)
   {
      return Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                   true, 1.0, 1.0);
   }
   return Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                1.0, 1.0, 1.0);
}

int SurfaceBasisSize(Geometry::Type geometry, int order)
{
   if (geometry == Geometry::SEGMENT) { return 1; }
   if (geometry == Geometry::SQUARE)
   { return 2*(order + 1) + order*(order + 1)/2; }
   static const int sizes[] = {3, 11, 26, 50, 85, 133, 196, 276};
   return sizes[std::min(order, 7)];
}

bool HasNonzeroWeight(const IntegrationRule &rule)
{
   for (int index = 0; index < rule.GetNPoints(); ++index)
   { if (rule.IntPoint(index).weight != 0.0) { return true; } }
   return false;
}

#ifdef MFEM_USE_LAPACK
class MomentBackend : public CutQuadratureBackend
{
public:
   std::string Name() const override { return "moment"; }

   CutQuadratureReport CheckSupport(
      Geometry::Type geometry, const CutDomain &domain, int order,
      const IntegrationRule *fixed_points) const override
   {
      auto report = domain.Validate(geometry);
      if (!report.Succeeded()) { return report; }
      if (order < 0)
      {
         return MomentFailure(CutQuadratureReport::Status::InvalidInput,
                              "Negative quadrature order.");
      }
      if (order == 0)
      {
         return MomentFailure(CutQuadratureReport::Status::Unsupported,
                              "Moment fitting requires positive order.");
      }
      if (geometry != Geometry::SEGMENT && geometry != Geometry::SQUARE &&
          geometry != Geometry::CUBE)
      {
         return MomentFailure(CutQuadratureReport::Status::Unsupported,
                              "Moment fitting supports segments, quads, and hexes.");
      }
      if (domain.NumConstraints() > 1)
      {
         return MomentFailure(CutQuadratureReport::Status::Unsupported,
                              "Moment fitting supports one level-set constraint.");
      }
      if (geometry == Geometry::CUBE &&
          order + (domain.Codimension() ? 0 : 1) > 7)
      {
         return MomentFailure(CutQuadratureReport::Status::Unsupported,
                              "Moment-fitting order exceeds the 3D basis limit.");
      }
      if (fixed_points)
      {
         if (!domain.NumConstraints() || domain.Codimension() ||
             geometry == Geometry::SEGMENT)
         {
            return MomentFailure(CutQuadratureReport::Status::Unsupported,
                                 "Unsupported prescribed moment-fitting nodes.");
         }
         const int needed = SurfaceBasisSize(geometry, order + 1);
         if (fixed_points->GetNPoints() < needed)
         {
            return MomentFailure(CutQuadratureReport::Status::Unsupported,
                                 "Too few prescribed moment-fitting nodes.");
         }
      }
      return {};
   }

   CutQuadratureReport Generate(
      Geometry::Type geometry, const CutDomain &domain, int order,
      CutQuadrature &out,
      const IntegrationRule *fixed_points) const override
   {
      out.Clear();
      auto report = CheckSupport(geometry, domain, order, fixed_points);
      if (!report.Succeeded()) { return report; }
      if (!domain.NumConstraints())
      {
         out.rule = IntRules.Get(geometry, order);
         out.region = CutQuadrature::Region::Full;
         report.moment_degree = order;
         return report;
      }

      const auto &constraint = domain.GetConstraint(0);
      const real_t sign = constraint.sign == CutDomain::LevelSetSign::Negative ?
                          -1.0 : 1.0;
      Mesh mesh = ReferenceMesh(geometry);
      IsoparametricTransformation transformation;
      mesh.GetElementTransformation(0, &transformation);
      MomentFittingIntRules rules(order, constraint.level_set.get(), sign,
                                  fixed_points);
      if (constraint.sign == CutDomain::LevelSetSign::Zero)
      { rules.GetSurfaceIntegrationRule(transformation, out.rule); }
      else
      { rules.GetVolumeIntegrationRule(transformation, out.rule); }
      out.region = HasNonzeroWeight(out.rule) ? CutQuadrature::Region::Cut :
                   CutQuadrature::Region::Empty;
      report.moment_degree = order;
      return report;
   }
};
#endif

}

std::unique_ptr<CutQuadratureBackend> CreateMomentCutQuadratureBackend()
{
#ifdef MFEM_USE_LAPACK
   return std::unique_ptr<CutQuadratureBackend>(new MomentBackend);
#else
   return nullptr;
#endif
}

}

