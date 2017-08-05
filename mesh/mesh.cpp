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

// Implementation of data type mesh

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstring>
#include <ctime>

#ifdef MFEM_USE_GECKO
#include "graph.h"
#endif

using namespace std;

namespace mfem
{

void Mesh::GetElementJacobian(int i, DenseMatrix &J)
{
   int geom = GetElementBaseGeometry(i);
   ElementTransformation *eltransf = GetElementTransformation(i);
   eltransf->SetIntPoint(&Geometries.GetCenter(geom));
   Geometries.JacToPerfJac(geom, eltransf->Jacobian(), J);
}

double Mesh::GetElementSize(int i, int type)
{
   DenseMatrix J(Dim);
   GetElementJacobian(i, J);
   if (type == 0)
   {
      return pow(fabs(J.Det()), 1./Dim);
   }
   else if (type == 1)
   {
      return J.CalcSingularvalue(Dim-1);   // h_min
   }
   else
   {
      return J.CalcSingularvalue(0);   // h_max
   }
}

double Mesh::GetElementSize(int i, const Vector &dir)
{
   DenseMatrix J(Dim);
   Vector d_hat(Dim);
   GetElementJacobian(i, J);
   J.MultTranspose(dir, d_hat);
   return sqrt((d_hat * d_hat) / (dir * dir));
}

double Mesh::GetElementVolume(int i)
{
   ElementTransformation *et = GetElementTransformation(i);
   const IntegrationRule &ir = IntRules.Get(GetElementBaseGeometry(i),
                                            et->OrderJ());
   double volume = 0.0;
   for (int j = 0; j < ir.GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir.IntPoint(j);
      et->SetIntPoint(&ip);
      volume += ip.weight * et->Weight();
   }

   return volume;
}

// Similar to VisualizationSceneSolution3d::FindNewBox in GLVis
void Mesh::GetBoundingBox(Vector &min, Vector &max, int ref)
{
   min.SetSize(spaceDim);
   max.SetSize(spaceDim);

   for (int d = 0; d < spaceDim; d++)
   {
      min(d) = numeric_limits<double>::infinity();
      max(d) = -numeric_limits<double>::infinity();
   }

   if (Nodes == NULL)
   {
      double *coord;
      for (int i = 0; i < NumOfVertices; i++)
      {
         coord = GetVertex(i);
         for (int d = 0; d < spaceDim; d++)
         {
            if (coord[d] < min(d)) { min(d) = coord[d]; }
            if (coord[d] > max(d)) { max(d) = coord[d]; }
         }
      }
   }
   else
   {
      const bool use_boundary = false; // make this a parameter?
      int ne = use_boundary ? GetNBE() : GetNE();
      int fn, fo;
      DenseMatrix pointmat;
      RefinedGeometry *RefG;
      IntegrationRule eir;
      FaceElementTransformations *Tr;
      ElementTransformation *T;

      for (int i = 0; i < ne; i++)
      {
         if (use_boundary)
         {
            GetBdrElementFace(i, &fn, &fo);
            RefG = GlobGeometryRefiner.Refine(GetFaceBaseGeometry(fn), ref);
            Tr = GetFaceElementTransformations(fn, 5);
            eir.SetSize(RefG->RefPts.GetNPoints());
            Tr->Loc1.Transform(RefG->RefPts, eir);
            Tr->Elem1->Transform(eir, pointmat);
         }
         else
         {
            T = GetElementTransformation(i);
            RefG = GlobGeometryRefiner.Refine(GetElementBaseGeometry(i), ref);
            T->Transform(RefG->RefPts, pointmat);
         }
         for (int j = 0; j < pointmat.Width(); j++)
         {
            for (int d = 0; d < pointmat.Height(); d++)
            {
               if (pointmat(d,j) < min(d)) { min(d) = pointmat(d,j); }
               if (pointmat(d,j) > max(d)) { max(d) = pointmat(d,j); }
            }
         }
      }
   }
}

void Mesh::GetCharacteristics(double &h_min, double &h_max,
                              double &kappa_min, double &kappa_max,
                              Vector *Vh, Vector *Vk)
{
   int i, dim, sdim;
   DenseMatrix J;
   double h, kappa;

   dim = Dimension();
   sdim = SpaceDimension();
   J.SetSize(sdim, dim);

   if (Vh) { Vh->SetSize(NumOfElements); }
   if (Vk) { Vk->SetSize(NumOfElements); }

   h_min = kappa_min = numeric_limits<double>::infinity();
   h_max = kappa_max = -h_min;
   for (i = 0; i < NumOfElements; i++)
   {
      GetElementJacobian(i, J);
      h = pow(fabs(J.Weight()), 1.0/double(dim));
      kappa = (dim == sdim) ?
              J.CalcSingularvalue(0) / J.CalcSingularvalue(dim-1) : -1.0;
      if (Vh) { (*Vh)(i) = h; }
      if (Vk) { (*Vk)(i) = kappa; }

      if (h < h_min) { h_min = h; }
      if (h > h_max) { h_max = h; }
      if (kappa < kappa_min) { kappa_min = kappa; }
      if (kappa > kappa_max) { kappa_max = kappa; }
   }
}

void Mesh::PrintCharacteristics(Vector *Vh, Vector *Vk, std::ostream &out)
{
   double h_min, h_max, kappa_min, kappa_max;

   out << "Mesh Characteristics:";

   this->GetCharacteristics(h_min, h_max, kappa_min, kappa_max, Vh, Vk);

   if (Dim == 1)
   {
      out << '\n'
          << "Number of vertices : " << GetNV() << '\n'
          << "Number of elements : " << GetNE() << '\n'
          << "Number of bdr elem : " << GetNBE() << '\n'
          << "h_min              : " << h_min << '\n'
          << "h_max              : " << h_max << '\n';
   }
   else if (Dim == 2)
   {
      out << '\n'
          << "Number of vertices : " << GetNV() << '\n'
          << "Number of edges    : " << GetNEdges() << '\n'
          << "Number of elements : " << GetNE() << '\n'
          << "Number of bdr elem : " << GetNBE() << '\n'
          << "Euler Number       : " << EulerNumber2D() << '\n'
          << "h_min              : " << h_min << '\n'
          << "h_max              : " << h_max << '\n'
          << "kappa_min          : " << kappa_min << '\n'
          << "kappa_max          : " << kappa_max << '\n';
   }
   else if (Dim == 3)
   {
      out << '\n'
          << "Number of vertices : " << GetNV() << '\n'
          << "Number of edges    : " << GetNEdges() << '\n'
          << "Number of faces    : " << GetNFaces() << '\n'
          << "Number of elements : " << GetNE() << '\n'
          << "Number of bdr elem : " << GetNBE() << '\n'
          << "Euler Number       : " << EulerNumber() << '\n'
          << "h_min              : " << h_min << '\n'
          << "h_max              : " << h_max << '\n'
          << "kappa_min          : " << kappa_min << '\n'
          << "kappa_max          : " << kappa_max << '\n';
   }
   else
   {
	  cout << '\n'
		   << "Number of vertices : " << GetNV() << endl
		   << "Number of edges    : " << GetNEdges() << endl
		   << "Number of planars  : " << GetNPlanars() << endl
		   << "Number of faces    : " << GetNFaces() << endl
		   << "Number of elements : " << GetNE() << endl
		   << "Number of bdr elem : " << GetNBE() << endl
		   << "Euler Number       : " << EulerNumber4D() << endl
		   << "h_min              : " << h_min << endl
		   << "h_max              : " << h_max << endl
		   << "kappa_min          : " << kappa_min << endl
		   << "kappa_max          : " << kappa_max << endl
		   << endl;
   }
   out << '\n' << std::flush;
}

FiniteElement *Mesh::GetTransformationFEforElementType(int ElemType)
{
   switch (ElemType)
   {
      case Element::POINT :          return &PointFE;
      case Element::SEGMENT :        return &SegmentFE;
      case Element::TRIANGLE :       return &TriangleFE;
      case Element::QUADRILATERAL :  return &QuadrilateralFE;
      case Element::TETRAHEDRON :    return &TetrahedronFE;
      case Element::HEXAHEDRON :     return &HexahedronFE;
      case Element::PENTATOPE :      return &PentatopeFE;
      case Element::TESSERACT :      return &TesseractFE;
   }
   MFEM_ABORT("Unknown element type");
   return &TriangleFE;
}


void Mesh::GetElementTransformation(int i, IsoparametricTransformation *ElTr)
{
   ElTr->Attribute = GetAttribute(i);
   ElTr->ElementNo = i;
   if (Nodes == NULL)
   {
      GetPointMatrix(i, ElTr->GetPointMat());
      ElTr->SetFE(GetTransformationFEforElementType(GetElementType(i)));
   }
   else
   {
      DenseMatrix &pm = ElTr->GetPointMat();
      Array<int> vdofs;
      Nodes->FESpace()->GetElementVDofs(i, vdofs);

      int n = vdofs.Size()/spaceDim;
      pm.SetSize(spaceDim, n);
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < n; j++)
         {
            pm(k,j) = (*Nodes)(vdofs[n*k+j]);
         }
      }
      ElTr->SetFE(Nodes->FESpace()->GetFE(i));
   }
}

void Mesh::GetElementTransformation(int i, const Vector &nodes,
                                    IsoparametricTransformation *ElTr)
{
   ElTr->Attribute = GetAttribute(i);
   ElTr->ElementNo = i;
   DenseMatrix &pm = ElTr->GetPointMat();
   if (Nodes == NULL)
   {
      int       nv = elements[i]->GetNVertices();
      const int *v = elements[i]->GetVertices();
      int n = vertices.Size();
      pm.SetSize(spaceDim, nv);
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < nv; j++)
         {
            pm(k, j) = nodes(k*n+v[j]);
         }
      }
      ElTr->SetFE(GetTransformationFEforElementType(GetElementType(i)));
   }
   else
   {
      Array<int> vdofs;
      Nodes->FESpace()->GetElementVDofs(i, vdofs);
      int n = vdofs.Size()/spaceDim;
      pm.SetSize(spaceDim, n);
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < n; j++)
         {
            pm(k,j) = nodes(vdofs[n*k+j]);
         }
      }
      ElTr->SetFE(Nodes->FESpace()->GetFE(i));
   }
}

ElementTransformation *Mesh::GetElementTransformation(int i)
{
   GetElementTransformation(i, &Transformation);

   return &Transformation;
}

ElementTransformation *Mesh::GetBdrElementTransformation(int i)
{
   GetBdrElementTransformation(i, &FaceTransformation);
   return &FaceTransformation;
}

void Mesh::GetBdrElementTransformation(int i, IsoparametricTransformation* ElTr)
{
   ElTr->Attribute = GetBdrAttribute(i);
   ElTr->ElementNo = i; // boundary element number
   if (Nodes == NULL)
   {
      GetBdrPointMatrix(i, ElTr->GetPointMat());
      ElTr->SetFE(
         GetTransformationFEforElementType(GetBdrElementType(i)));
   }
   else
   {
      DenseMatrix &pm = ElTr->GetPointMat();
      Array<int> vdofs;
      Nodes->FESpace()->GetBdrElementVDofs(i, vdofs);
      int n = vdofs.Size()/spaceDim;
      pm.SetSize(spaceDim, n);
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < n; j++)
         {
            pm(k,j) = (*Nodes)(vdofs[n*k+j]);
         }
      }
      ElTr->SetFE(Nodes->FESpace()->GetBE(i));
   }
}

void Mesh::GetFaceTransformation(int FaceNo, IsoparametricTransformation *FTr)
{
   FTr->Attribute = (Dim == 1) ? 1 : faces[FaceNo]->GetAttribute();
   FTr->ElementNo = FaceNo;
   DenseMatrix &pm = FTr->GetPointMat();
   if (Nodes == NULL)
   {
      const int *v = (Dim == 1) ? &FaceNo : faces[FaceNo]->GetVertices();
      const int nv = (Dim == 1) ? 1 : faces[FaceNo]->GetNVertices();
      pm.SetSize(spaceDim, nv);
      for (int i = 0; i < spaceDim; i++)
      {
         for (int j = 0; j < nv; j++)
         {
            pm(i, j) = vertices[v[j]](i);
         }
      }
      FTr->SetFE(GetTransformationFEforElementType(GetFaceElementType(FaceNo)));
   }
   else // curved mesh
   {
      const FiniteElement *face_el = Nodes->FESpace()->GetFaceElement(FaceNo);
      if (face_el)
      {
         Array<int> vdofs;
         Nodes->FESpace()->GetFaceVDofs(FaceNo, vdofs);
         int n = vdofs.Size()/spaceDim;
         pm.SetSize(spaceDim, n);
         for (int i = 0; i < spaceDim; i++)
         {
            for (int j = 0; j < n; j++)
            {
               pm(i, j) = (*Nodes)(vdofs[n*i+j]);
            }
         }
         FTr->SetFE(face_el);
      }
      else // L2 Nodes (e.g., periodic mesh), go through the volume of Elem1
      {
         FaceInfo &face_info = faces_info[FaceNo];

         int face_geom = GetFaceGeometryType(FaceNo);
         int face_type = GetFaceElementType(FaceNo);

         GetLocalFaceTransformation(face_type,
                                    GetElementType(face_info.Elem1No),
                                    FaceElemTr.Loc1.Transf, face_info.Elem1Inf);
         // NOTE: FaceElemTr.Loc1 is overwritten here -- used as a temporary

         face_el = Nodes->FESpace()->GetTraceElement(face_info.Elem1No,
                                                     face_geom);

         IntegrationRule eir(face_el->GetDof());
         FaceElemTr.Loc1.Transform(face_el->GetNodes(), eir);
         // 'Transformation' is not used
         Nodes->GetVectorValues(Transformation, eir, pm);

         FTr->SetFE(face_el);
      }
   }
}

ElementTransformation *Mesh::GetFaceTransformation(int FaceNo)
{
   GetFaceTransformation(FaceNo, &FaceTransformation);
   return &FaceTransformation;
}

void Mesh::GetEdgeTransformation(int EdgeNo, IsoparametricTransformation *EdTr)
{
   if (Dim == 2)
   {
      GetFaceTransformation(EdgeNo, EdTr);
      return;
   }
   if (Dim == 1)
   {
      mfem_error("Mesh::GetEdgeTransformation not defined in 1D \n");
   }

   EdTr->Attribute = 1;
   EdTr->ElementNo = EdgeNo;
   DenseMatrix &pm = EdTr->GetPointMat();
   if (Nodes == NULL)
   {
      Array<int> v;
      GetEdgeVertices(EdgeNo, v);
      const int nv = 2;
      pm.SetSize(spaceDim, nv);
      for (int i = 0; i < spaceDim; i++)
      {
         for (int j = 0; j < nv; j++)
         {
            pm(i, j) = vertices[v[j]](i);
         }
      }
      EdTr->SetFE(GetTransformationFEforElementType(Element::SEGMENT));
   }
   else
   {
      const FiniteElement *edge_el = Nodes->FESpace()->GetEdgeElement(EdgeNo);
      if (edge_el)
      {
         Array<int> vdofs;
         Nodes->FESpace()->GetEdgeVDofs(EdgeNo, vdofs);
         int n = vdofs.Size()/spaceDim;
         pm.SetSize(spaceDim, n);
         for (int i = 0; i < spaceDim; i++)
         {
            for (int j = 0; j < n; j++)
            {
               pm(i, j) = (*Nodes)(vdofs[n*i+j]);
            }
         }
         EdTr->SetFE(edge_el);
      }
      else
      {
         MFEM_ABORT("Not implemented.");
      }
   }
}

ElementTransformation *Mesh::GetEdgeTransformation(int EdgeNo)
{
   GetEdgeTransformation(EdgeNo, &EdgeTransformation);
   return &EdgeTransformation;
}


void Mesh::GetLocalPtToSegTransformation(
   IsoparametricTransformation &Transf, int i)
{
   const IntegrationRule *SegVert;
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&PointFE);
   SegVert = Geometries.GetVertices(Geometry::SEGMENT);
   locpm.SetSize(1, 1);
   locpm(0, 0) = SegVert->IntPoint(i/64).x;
   //  (i/64) is the local face no. in the segment
   //  (i%64) is the orientation of the point (not used)
}

void Mesh::GetLocalSegToTriTransformation(
   IsoparametricTransformation &Transf, int i)
{
   const int *tv, *so;
   const IntegrationRule *TriVert;
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&SegmentFE);
   tv = tri_t::Edges[i/64];  //  (i/64) is the local face no. in the triangle
   so = seg_t::Orient[i%64]; //  (i%64) is the orientation of the segment
   TriVert = Geometries.GetVertices(Geometry::TRIANGLE);
   locpm.SetSize(2, 2);
   for (int j = 0; j < 2; j++)
   {
      locpm(0, so[j]) = TriVert->IntPoint(tv[j]).x;
      locpm(1, so[j]) = TriVert->IntPoint(tv[j]).y;
   }
}

void Mesh::GetLocalSegToQuadTransformation(
   IsoparametricTransformation &Transf, int i)
{
   const int *qv, *so;
   const IntegrationRule *QuadVert;
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&SegmentFE);
   qv = quad_t::Edges[i/64]; //  (i/64) is the local face no. in the quad
   so = seg_t::Orient[i%64]; //  (i%64) is the orientation of the segment
   QuadVert = Geometries.GetVertices(Geometry::SQUARE);
   locpm.SetSize(2, 2);
   for (int j = 0; j < 2; j++)
   {
      locpm(0, so[j]) = QuadVert->IntPoint(qv[j]).x;
      locpm(1, so[j]) = QuadVert->IntPoint(qv[j]).y;
   }
}

void Mesh::GetLocalTriToTetTransformation(
   IsoparametricTransformation &Transf, int i)
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&TriangleFE);
   //  (i/64) is the local face no. in the tet
   const int *tv = tet_t::FaceVert[i/64];
   //  (i%64) is the orientation of the tetrahedron face
   //         w.r.t. the face element
   const int *to = tri_t::Orient[i%64];
   const IntegrationRule *TetVert =
      Geometries.GetVertices(Geometry::TETRAHEDRON);
   locpm.SetSize(3, 3);
   for (int j = 0; j < 3; j++)
   {
      const IntegrationPoint &vert = TetVert->IntPoint(tv[to[j]]);
      locpm(0, j) = vert.x;
      locpm(1, j) = vert.y;
      locpm(2, j) = vert.z;
   }
}

void Mesh::GetLocalTetToPentaTransformation(
   IsoparametricTransformation &Transf, int i)
{
   // FIX ME: just copied TriToTet function and changed names
   std::cout << "Not implemented properly yet" << std::endl;
   /*
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&PentatopeFE);
   //  (i/64) is the local face no. in the penta
   const int *tv = pent_t::FaceVert[i/64];
   //  (i%64) is the orientation of the pentatope face
   //         w.r.t. the face element
   const int *to = tet_t::Orient[i%64];
   const IntegrationRule *PentaVert =
      Geometries.GetVertices(Geometry::PENTATOPE);
   locpm.SetSize(4, 4);
   for (int j = 0; j < 4; j++)
   {
      const IntegrationPoint &vert = PentaVert->IntPoint(tv[to[j]]);
      locpm(0, j) = vert.x;
      locpm(1, j) = vert.y;
      locpm(2, j) = vert.z;
      locpm(3, j) = vert.t;
   }
   */
}

void Mesh::GetLocalQuadToHexTransformation(
   IsoparametricTransformation &Transf, int i)
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&QuadrilateralFE);
   //  (i/64) is the local face no. in the hex
   const int *hv = hex_t::FaceVert[i/64];
   //  (i%64) is the orientation of the quad
   const int *qo = quad_t::Orient[i%64];
   const IntegrationRule *HexVert = Geometries.GetVertices(Geometry::CUBE);
   locpm.SetSize(3, 4);
   for (int j = 0; j < 4; j++)
   {
      const IntegrationPoint &vert = HexVert->IntPoint(hv[qo[j]]);
      locpm(0, j) = vert.x;
      locpm(1, j) = vert.y;
      locpm(2, j) = vert.z;
   }
}

void Mesh::GetLocalFaceTransformation(
   int face_type, int elem_type, IsoparametricTransformation &Transf, int inf)
{
   switch (face_type)
   {
      case Element::POINT:
         GetLocalPtToSegTransformation(Transf, inf);
         break;

      case Element::SEGMENT:
         if (elem_type == Element::TRIANGLE)
         {
            GetLocalSegToTriTransformation(Transf, inf);
         }
         else
         {
            MFEM_ASSERT(elem_type == Element::QUADRILATERAL, "");
            GetLocalSegToQuadTransformation(Transf, inf);
         }
         break;

      case Element::TRIANGLE:
         MFEM_ASSERT(elem_type == Element::TETRAHEDRON, "");
         GetLocalTriToTetTransformation(Transf, inf);
         break;

      case Element::QUADRILATERAL:
         MFEM_ASSERT(elem_type == Element::HEXAHEDRON, "");
         GetLocalQuadToHexTransformation(Transf, inf);
         break;

      case Element::TETRAHEDRON:
         MFEM_ASSERT(elem_type == Element::PENTATOPE, "");
         GetLocalTetToPentaTransformation(Transf, inf);
      break;
   }
}

FaceElementTransformations *Mesh::GetFaceElementTransformations(int FaceNo,
                                                                int mask)
{
   FaceInfo &face_info = faces_info[FaceNo];

   FaceElemTr.Elem1 = NULL;
   FaceElemTr.Elem2 = NULL;

   // setup the transformation for the first element
   FaceElemTr.Elem1No = face_info.Elem1No;
   if (mask & 1)
   {
      GetElementTransformation(FaceElemTr.Elem1No, &Transformation);
      FaceElemTr.Elem1 = &Transformation;
   }

   //  setup the transformation for the second element
   //     return NULL in the Elem2 field if there's no second element, i.e.
   //     the face is on the "boundary"
   FaceElemTr.Elem2No = face_info.Elem2No;
   if ((mask & 2) && FaceElemTr.Elem2No >= 0)
   {
#ifdef MFEM_DEBUG
      if (NURBSext && (mask & 1)) { MFEM_ABORT("NURBS mesh not supported!"); }
#endif
      GetElementTransformation(FaceElemTr.Elem2No, &Transformation2);
      FaceElemTr.Elem2 = &Transformation2;
   }

   // setup the face transformation
   FaceElemTr.FaceGeom = GetFaceGeometryType(FaceNo);
   FaceElemTr.Face = (mask & 16) ? GetFaceTransformation(FaceNo) : NULL;

   // setup Loc1 & Loc2
   int face_type = GetFaceElementType(FaceNo);
   if (mask & 4)
   {
      int elem_type = GetElementType(face_info.Elem1No);
      GetLocalFaceTransformation(face_type, elem_type,
                                 FaceElemTr.Loc1.Transf, face_info.Elem1Inf);
   }
   if ((mask & 8) && FaceElemTr.Elem2No >= 0)
   {
      int elem_type = GetElementType(face_info.Elem2No);
      GetLocalFaceTransformation(face_type, elem_type,
                                 FaceElemTr.Loc2.Transf, face_info.Elem2Inf);

      // NC meshes: prepend slave edge/face transformation to Loc2
      if (Nonconforming() && IsSlaveFace(face_info))
      {
         ApplyLocalSlaveTransformation(FaceElemTr.Loc2.Transf, face_info);

         if (face_type == Element::SEGMENT)
         {
            // flip Loc2 to match Loc1 and Face
            DenseMatrix &pm = FaceElemTr.Loc2.Transf.GetPointMat();
            std::swap(pm(0,0), pm(0,1));
            std::swap(pm(1,0), pm(1,1));
         }
      }
   }

   return &FaceElemTr;
}

bool Mesh::IsSlaveFace(const FaceInfo &fi) const
{
   return fi.NCFace >= 0 && nc_faces_info[fi.NCFace].Slave;
}

void Mesh::ApplyLocalSlaveTransformation(IsoparametricTransformation &transf,
                                         const FaceInfo &fi)
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix composition;
#else
   static DenseMatrix composition;
#endif
   MFEM_ASSERT(fi.NCFace >= 0, "");
   transf.Transform(*nc_faces_info[fi.NCFace].PointMatrix, composition);
   transf.GetPointMat() = composition;
}

FaceElementTransformations *Mesh::GetBdrFaceTransformations(int BdrElemNo)
{
   FaceElementTransformations *tr;
   int fn;
   if (Dim == 3)
   {
      fn = be_to_face[BdrElemNo];
   }
   else if (Dim == 2)
   {
      fn = be_to_edge[BdrElemNo];
   }
   else
   {
      fn = boundary[BdrElemNo]->GetVertices()[0];
   }
   // Check if the face is interior, shared, or non-conforming.
   if (FaceIsTrueInterior(fn) || faces_info[fn].NCFace >= 0)
   {
      return NULL;
   }
   tr = GetFaceElementTransformations(fn);
   tr->Face->Attribute = boundary[BdrElemNo]->GetAttribute();
   return tr;
}

void Mesh::GetFaceElements(int Face, int *Elem1, int *Elem2)
{
   *Elem1 = faces_info[Face].Elem1No;
   *Elem2 = faces_info[Face].Elem2No;
}

void Mesh::GetFaceInfos(int Face, int *Inf1, int *Inf2)
{
   *Inf1 = faces_info[Face].Elem1Inf;
   *Inf2 = faces_info[Face].Elem2Inf;
}

int Mesh::GetFaceGeometryType(int Face) const
{
   return (Dim == 1) ? Geometry::POINT : faces[Face]->GetGeometryType();
}

int Mesh::GetFaceElementType(int Face) const
{
   return (Dim == 1) ? Element::POINT : faces[Face]->GetType();
}

void Mesh::Init()
{
   // in order of declaration:
   Dim = spaceDim = 0;
   NumOfVertices = -1;
   NumOfElements = NumOfBdrElements = 0;
   NumOfPlanars = -1;
   NumOfEdges = NumOfFaces = 0;
   BaseGeom = BaseBdrGeom = -2; // invailid
   meshgen = 0;
   sequence = 0;
   Nodes = NULL;
   own_nodes = 1;
   NURBSext = NULL;
   ncmesh = NULL;
   last_operation = Mesh::NONE;
}

void Mesh::InitTables()
{
   el_to_edge =
      el_to_face = el_to_el = bel_to_edge = face_edge = edge_vertex = el_to_planar = bel_to_planar = NULL;
}

void Mesh::SetEmpty()
{
   // Members not touched by Init() or InitTables()
   Dim = spaceDim = 0;
   BaseGeom = BaseBdrGeom = -1;
   meshgen = 0;
   NumOfFaces = 0;

   Init();
   InitTables();
}

void Mesh::DestroyTables()
{
   delete el_to_edge;
   delete el_to_face;
   delete el_to_el;

   if (Dim >= 3)
   {
      delete bel_to_edge;
   }

   if (Dim == 4)
   {
	   delete el_to_planar;
	   delete bel_to_planar;
   }

   delete face_edge;
   delete edge_vertex;
}

void Mesh::DestroyPointers()
{
   if (own_nodes) { delete Nodes; }

   delete ncmesh;

   delete NURBSext;

   for (int i = 0; i < NumOfElements; i++)
   {
      FreeElement(elements[i]);
   }

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      FreeElement(boundary[i]);
   }

   for (int i = 0; i < faces.Size(); i++)
   {
      FreeElement(faces[i]);
   }

   for (int i = 0; i < planars.Size(); i++)
   {
      FreeElement(planars[i]);
   }

   DestroyTables();
}

void Mesh::Destroy()
{
   DestroyPointers();

   elements.DeleteAll();
   vertices.DeleteAll();
   boundary.DeleteAll();
   faces.DeleteAll();
   faces_info.DeleteAll();
   nc_faces_info.DeleteAll();
   be_to_edge.DeleteAll();
   be_to_face.DeleteAll();

   planars.DeleteAll();
   swappedElements.DeleteAll();
   swappedFaces.DeleteAll();


   // TODO:
   // IsoparametricTransformations
   // Transformation, Transformation2, FaceTransformation, EdgeTransformation;
   // FaceElementTransformations FaceElemTr;

   CoarseFineTr.Clear();

#ifdef MFEM_USE_MEMALLOC
   TetMemory.Clear();
#endif

   attributes.DeleteAll();
   bdr_attributes.DeleteAll();
}

void Mesh::SetAttributes()
{
   Array<int> attribs;

   attribs.SetSize(GetNBE());
   for (int i = 0; i < attribs.Size(); i++)
   {
      attribs[i] = GetBdrAttribute(i);
   }
   attribs.Sort();
   attribs.Unique();
   attribs.Copy(bdr_attributes);
   if (bdr_attributes.Size() > 0 && bdr_attributes[0] <= 0)
   {
      MFEM_WARNING("Non-positive attributes on the boundary!");
   }

   attribs.SetSize(GetNE());
   for (int i = 0; i < attribs.Size(); i++)
   {
      attribs[i] = GetAttribute(i);
   }
   attribs.Sort();
   attribs.Unique();
   attribs.Copy(attributes);
   if (attributes.Size() > 0 && attributes[0] <= 0)
   {
      MFEM_WARNING("Non-positive attributes in the domain!");
   }
}

void Mesh::InitMesh(int _Dim, int _spaceDim, int NVert, int NElem, int NBdrElem)
{
   SetEmpty();

   Dim = _Dim;
   spaceDim = _spaceDim;

   NumOfVertices = 0;
   vertices.SetSize(NVert);  // just allocate space for vertices

   NumOfElements = 0;
   elements.SetSize(NElem);  // just allocate space for Element *

   NumOfBdrElements = 0;
   boundary.SetSize(NBdrElem);  // just allocate space for Element *

   NumOfPlanars = 0;
   planars.SetSize(NumOfPlanars);// just allocate space for the planar Element *
}

void Mesh::InitBaseGeom()
{
   BaseGeom = BaseBdrGeom = -1;
   for (int i = 0; i < NumOfElements; i++)
   {
      int geom = elements[i]->GetGeometryType();
      if (geom != BaseGeom && BaseGeom >= 0)
      {
         BaseGeom = -1; break;
      }
      BaseGeom = geom;
   }
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      int geom = boundary[i]->GetGeometryType();
      if (geom != BaseBdrGeom && BaseBdrGeom >= 0)
      {
         BaseBdrGeom = -1; break;
      }
      BaseBdrGeom = geom;
   }
}

void Mesh::AddVertex(const double *x)
{
   double *y = vertices[NumOfVertices]();

   for (int i = 0; i < spaceDim; i++)
   {
      y[i] = x[i];
   }
   NumOfVertices++;
}

void Mesh::AddTri(const int *vi, int attr)
{
   elements[NumOfElements++] = new Triangle(vi, attr);
}

void Mesh::AddTriangle(const int *vi, int attr)
{
   elements[NumOfElements++] = new Triangle(vi, attr);
}

void Mesh::AddQuad(const int *vi, int attr)
{
   elements[NumOfElements++] = new Quadrilateral(vi, attr);
}

void Mesh::AddTet(const int *vi, int attr)
{
#ifdef MFEM_USE_MEMALLOC
   Tetrahedron *tet;
   tet = TetMemory.Alloc();
   tet->SetVertices(vi);
   tet->SetAttribute(attr);
   elements[NumOfElements++] = tet;
#else
   elements[NumOfElements++] = new Tetrahedron(vi, attr);
#endif
}

void Mesh::AddHex(const int *vi, int attr)
{
   elements[NumOfElements++] = new Hexahedron(vi, attr);
}

void Mesh::AddHexAsTets(const int *vi, int attr)
{
   static const int hex_to_tet[6][4] =
   {
      { 0, 1, 2, 6 }, { 0, 5, 1, 6 }, { 0, 4, 5, 6 },
      { 0, 2, 3, 6 }, { 0, 3, 7, 6 }, { 0, 7, 4, 6 }
   };
   int ti[4];

   for (int i = 0; i < 6; i++)
   {
      for (int j = 0; j < 4; j++)
      {
         ti[j] = vi[hex_to_tet[i][j]];
      }
      AddTet(ti, attr);
   }
}

void Mesh::AddTes(const int *vi, int attr)
{
   elements[NumOfElements++] = new Tesseract(vi, attr);
}

void Mesh::AddBdrSegment(const int *vi, int attr)
{
   boundary[NumOfBdrElements++] = new Segment(vi, attr);
}

void Mesh::AddBdrTriangle(const int *vi, int attr)
{
   boundary[NumOfBdrElements++] = new Triangle(vi, attr);
}

void Mesh::AddBdrQuad(const int *vi, int attr)
{
   boundary[NumOfBdrElements++] = new Quadrilateral(vi, attr);
}

void Mesh::AddBdrQuadAsTriangles(const int *vi, int attr)
{
   static const int quad_to_tri[2][3] = { { 0, 1, 2 }, { 0, 2, 3 } };
   int ti[3];

   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         ti[j] = vi[quad_to_tri[i][j]];
      }
      AddBdrTriangle(ti, attr);
   }
}

void Mesh::AddBdrHex(const int *vi, int attr)
{
	boundary[NumOfBdrElements++] = new Hexahedron(vi, attr);
}

void Mesh::GenerateBoundaryElements()
{
   int i, j;
   Array<int> &be2face = (Dim == 2) ? be_to_edge : be_to_face;

   // GenerateFaces();

   for (i = 0; i < boundary.Size(); i++)
   {
      FreeElement(boundary[i]);
   }

   if (Dim >= 3)
   {
      delete bel_to_edge;
      bel_to_edge = NULL;

      if (Dim==4)
      {
    	  delete bel_to_planar;
    	  bel_to_planar = NULL;
      }
   }

   // count the 'NumOfBdrElements'
   NumOfBdrElements = 0;
   for (i = 0; i < faces_info.Size(); i++)
   {
      if (faces_info[i].Elem2No < 0) { NumOfBdrElements++; }
   }

   boundary.SetSize(NumOfBdrElements);
   be2face.SetSize(NumOfBdrElements);
   for (j = i = 0; i < faces_info.Size(); i++)
   {
      if (faces_info[i].Elem2No < 0)
      {
         boundary[j] = faces[i]->Duplicate(this);
         be2face[j++] = i;
      }
   }
   // In 3D, 'bel_to_edge' is destroyed but it's not updated.
}

void Mesh::FinalizeCheck()
{
   MFEM_VERIFY(vertices.Size() == NumOfVertices,
               "incorrect number of vertices: preallocated: " << vertices.Size()
               << ", actually added: " << NumOfVertices);
   MFEM_VERIFY(elements.Size() == NumOfElements,
               "incorrect number of elements: preallocated: " << elements.Size()
               << ", actually added: " << NumOfElements);
   MFEM_VERIFY(boundary.Size() == NumOfBdrElements,
               "incorrect number of boundary elements: preallocated: "
               << boundary.Size() << ", actually added: " << NumOfBdrElements);
}

void Mesh::FinalizeTriMesh(int generate_edges, int refine, bool fix_orientation)
{
   FinalizeCheck();
   CheckElementOrientation(fix_orientation);

   if (refine)
   {
      MarkTriMeshForRefinement();
   }

   if (generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      GenerateFaces();
      CheckBdrElementOrientation();
   }
   else
   {
      NumOfEdges = 0;
   }

   NumOfFaces = 0;

   SetAttributes();

   BaseGeom = Geometry::TRIANGLE;
   BaseBdrGeom = Geometry::SEGMENT;

   meshgen = 1;
}

void Mesh::FinalizeQuadMesh(int generate_edges, int refine,
                            bool fix_orientation)
{
   FinalizeCheck();
   if (fix_orientation)
   {
      CheckElementOrientation(fix_orientation);
   }

   if (generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      GenerateFaces();
      CheckBdrElementOrientation();
   }
   else
   {
      NumOfEdges = 0;
   }

   NumOfFaces = 0;

   SetAttributes();

   BaseGeom = Geometry::SQUARE;
   BaseBdrGeom = Geometry::SEGMENT;

   meshgen = 2;
}


#ifdef MFEM_USE_GECKO
void Mesh::GetGeckoElementReordering(Array<int> &ordering)
{
   Gecko::Graph graph;

   // We will put some accesors in for these later
   Gecko::Functional *functional =
      new Gecko::FunctionalGeometric(); // ordering functional
   unsigned int iterations = 1;         // number of V cycles
   unsigned int window = 2;             // initial window size
   unsigned int period = 1;             // iterations between window increment
   unsigned int seed = 0;               // random number seed

   // Run through all the elements and insert the nodes in the graph for them
   for (int elemid = 0; elemid < GetNE(); ++elemid)
   {
      graph.insert();
   }

   // Run through all the elems and insert arcs to the graph for each element
   // face Indices in Gecko are 1 based hence the +1 on the insertion
   const Table &my_el_to_el = ElementToElementTable();
   for (int elemid = 0; elemid < GetNE(); ++elemid)
   {
      const int *neighid = my_el_to_el.GetRow(elemid);
      for (int i = 0; i < my_el_to_el.RowSize(elemid); ++i)
      {
         graph.insert(elemid + 1,  neighid[i] + 1);
      }
   }

   // Get the reordering from Gecko and copy it into the ordering Array<int>
   graph.order(functional, iterations, window, period, seed);
   ordering.DeleteAll();
   ordering.SetSize(GetNE());
   Gecko::Node::Index NE = GetNE();
   for (Gecko::Node::Index gnodeid = 1; gnodeid <= NE; ++gnodeid)
   {
      ordering[gnodeid - 1] = graph.rank(gnodeid);
   }

   delete functional;
}
#endif


void Mesh::ReorderElements(const Array<int> &ordering, bool reorder_vertices)
{
   if (NURBSext)
   {
      MFEM_WARNING("element reordering of NURBS meshes is not supported.");
      return;
   }
   if (ncmesh)
   {
      MFEM_WARNING("element reordering of non-conforming meshes is not"
                   " supported.");
      return;
   }
   MFEM_VERIFY(ordering.Size() == GetNE(), "invalid reordering array.")

   // Data members that need to be updated:

   // - elements   - reorder of the pointers and the vertex ids if reordering
   //                the vertices
   // - vertices   - if reordering the vertices
   // - boundary   - update the vertex ids, if reordering the vertices
   // - faces      - regenerate
   // - faces_info - regenerate

   // Deleted by DeleteTables():
   // - el_to_edge  - rebuild in 2D and 3D only
   // - el_to_face  - rebuild in 3D only
   // - bel_to_edge - rebuild in 3D only
   // - el_to_el    - no need to rebuild
   // - face_edge   - no need to rebuild
   // - edge_vertex - no need to rebuild

   // - be_to_edge  - 2D only
   // - be_to_face  - 3D only

   // - Nodes

   // Save the locations of the Nodes so we can rebuild them later
   Array<Vector*> old_elem_node_vals;
   FiniteElementSpace *nodes_fes = NULL;
   if (Nodes)
   {
      old_elem_node_vals.SetSize(GetNE());
      nodes_fes = Nodes->FESpace();
      Array<int> old_dofs;
      Vector vals;
      for (int old_elid = 0; old_elid < GetNE(); ++old_elid)
      {
         nodes_fes->GetElementVDofs(old_elid, old_dofs);
         Nodes->GetSubVector(old_dofs, vals);
         old_elem_node_vals[old_elid] = new Vector(vals);
      }
   }

   // Get the newly ordered elements
   Array<Element *> new_elements(GetNE());
   for (int old_elid = 0; old_elid < ordering.Size(); ++old_elid)
   {
      int new_elid = ordering[old_elid];
      new_elements[new_elid] = elements[old_elid];
   }
   mfem::Swap(elements, new_elements);
   new_elements.DeleteAll();

   if (reorder_vertices)
   {
      // Get the new vertex ordering permutation vectors and fill the new
      // vertices
      Array<int> vertex_ordering(GetNV());
      vertex_ordering = -1;
      Array<Vertex> new_vertices(GetNV());
      int new_vertex_ind = 0;
      for (int new_elid = 0; new_elid < GetNE(); ++new_elid)
      {
         int *elem_vert = elements[new_elid]->GetVertices();
         int nv = elements[new_elid]->GetNVertices();
         for (int vi = 0; vi < nv; ++vi)
         {
            int old_vertex_ind = elem_vert[vi];
            if (vertex_ordering[old_vertex_ind] == -1)
            {
               vertex_ordering[old_vertex_ind] = new_vertex_ind;
               new_vertices[new_vertex_ind] = vertices[old_vertex_ind];
               new_vertex_ind++;
            }
         }
      }
      mfem::Swap(vertices, new_vertices);
      new_vertices.DeleteAll();

      // Replace the vertex ids in the elements with the reordered vertex
      // numbers
      for (int new_elid = 0; new_elid < GetNE(); ++new_elid)
      {
         int *elem_vert = elements[new_elid]->GetVertices();
         int nv = elements[new_elid]->GetNVertices();
         for (int vi = 0; vi < nv; ++vi)
         {
            elem_vert[vi] = vertex_ordering[elem_vert[vi]];
         }
      }

      // Replace the vertex ids in the boundary with reordered vertex numbers
      for (int belid = 0; belid < GetNBE(); ++belid)
      {
         int *be_vert = boundary[belid]->GetVertices();
         int nv = boundary[belid]->GetNVertices();
         for (int vi = 0; vi < nv; ++vi)
         {
            be_vert[vi] = vertex_ordering[be_vert[vi]];
         }
      }
   }

   // Destroy tables that need to be rebuild
   DeleteTables();

   if (Dim > 1)
   {
      // generate el_to_edge, be_to_edge (2D), bel_to_edge (3D)
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   if (Dim > 2)
   {
      // generate el_to_face, be_to_face
      GetElementToFaceTable();
   }
   // Update faces and faces_info
   GenerateFaces();

   // Build the nodes from the saved locations if they were around before
   if (Nodes)
   {
      nodes_fes->Update();
      Array<int> new_dofs;
      for (int old_elid = 0; old_elid < GetNE(); ++old_elid)
      {
         int new_elid = ordering[old_elid];
         nodes_fes->GetElementVDofs(new_elid, new_dofs);
         Nodes->SetSubVector(new_dofs, *(old_elem_node_vals[old_elid]));
         delete old_elem_node_vals[old_elid];
      }
   }
}


void Mesh::MarkForRefinement()
{
   if (meshgen & 1)
   {
      if (Dim == 2)
      {
         MarkTriMeshForRefinement();
      }
      else if (Dim == 3)
      {
         DSTable v_to_v(NumOfVertices);
         GetVertexToVertexTable(v_to_v);
         MarkTetMeshForRefinement(v_to_v);
      }
   }
}

void Mesh::MarkTriMeshForRefinement()
{
   // Mark the longest triangle edge by rotating the indeces so that
   // vertex 0 - vertex 1 to be the longest element's edge.
   DenseMatrix pmat;
   for (int i = 0; i < NumOfElements; i++)
   {
      if (elements[i]->GetType() == Element::TRIANGLE)
      {
         GetPointMatrix(i, pmat);
         elements[i]->MarkEdge(pmat);
      }
   }
}

void Mesh::GetEdgeOrdering(DSTable &v_to_v, Array<int> &order)
{
   NumOfEdges = v_to_v.NumberOfEntries();
   order.SetSize(NumOfEdges);
   Array<Pair<double, int> > length_idx(NumOfEdges);

   for (int i = 0; i < NumOfVertices; i++)
   {
      for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
      {
         int j = it.Index();
         length_idx[j].one = GetLength(i, it.Column());
         length_idx[j].two = j;
      }
   }

   // Sort by increasing edge-length.
   length_idx.Sort();

   for (int i = 0; i < NumOfEdges; i++)
   {
      order[length_idx[i].two] = i;
   }
}

void Mesh::MarkTetMeshForRefinement(DSTable &v_to_v)
{
   // Mark the longest tetrahedral edge by rotating the indices so that
   // vertex 0 - vertex 1 is the longest edge in the element.
   Array<int> order;
   GetEdgeOrdering(v_to_v, order);

   for (int i = 0; i < NumOfElements; i++)
   {
      if (elements[i]->GetType() == Element::TETRAHEDRON)
      {
         elements[i]->MarkEdge(v_to_v, order);
      }
   }
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      if (boundary[i]->GetType() == Element::TRIANGLE)
      {
         boundary[i]->MarkEdge(v_to_v, order);
      }
   }
}

void Mesh::PrepareNodeReorder(DSTable **old_v_to_v, Table **old_elem_vert)
{
   if (*old_v_to_v && *old_elem_vert)
   {
      return;
   }

   FiniteElementSpace *fes = Nodes->FESpace();
   const FiniteElementCollection *fec = fes->FEColl();

   if (*old_v_to_v == NULL)
   {
      int num_edge_dofs = fec->DofForGeometry(Geometry::SEGMENT);
      if (num_edge_dofs > 0)
      {
         *old_v_to_v = new DSTable(NumOfVertices);
         GetVertexToVertexTable(*(*old_v_to_v));
      }
   }
   if (*old_elem_vert == NULL)
   {
      // assuming all elements have the same geometry
      int num_elem_dofs = fec->DofForGeometry(GetElementBaseGeometry(0));
      if (num_elem_dofs > 1)
      {
         *old_elem_vert = new Table;
         (*old_elem_vert)->MakeI(GetNE());
         for (int i = 0; i < GetNE(); i++)
         {
            (*old_elem_vert)->AddColumnsInRow(i, elements[i]->GetNVertices());
         }
         (*old_elem_vert)->MakeJ();
         for (int i = 0; i < GetNE(); i++)
         {
            (*old_elem_vert)->AddConnections(i, elements[i]->GetVertices(),
                                             elements[i]->GetNVertices());
         }
         (*old_elem_vert)->ShiftUpI();
      }
   }
}

void Mesh::DoNodeReorder(DSTable *old_v_to_v, Table *old_elem_vert)
{
   FiniteElementSpace *fes = Nodes->FESpace();
   const FiniteElementCollection *fec = fes->FEColl();
   int num_edge_dofs = fec->DofForGeometry(Geometry::SEGMENT);
   // assuming all faces have the same geometry
   int num_face_dofs =
      (Dim < 3) ? 0 : fec->DofForGeometry(GetFaceBaseGeometry(0));
   // assuming all elements have the same geometry
   int num_elem_dofs = fec->DofForGeometry(GetElementBaseGeometry(0));

   // reorder the Nodes
   Vector onodes = *Nodes;

   Array<int> old_dofs, new_dofs;
   int offset;
#ifdef MFEM_DEBUG
   int redges = 0;
#endif

   // vertex dofs do not need to be moved
   offset = NumOfVertices * fec->DofForGeometry(Geometry::POINT);

   // edge dofs:
   // edge enumeration may be different but edge orientation is the same
   if (num_edge_dofs > 0)
   {
      DSTable new_v_to_v(NumOfVertices);
      GetVertexToVertexTable(new_v_to_v);

      for (int i = 0; i < NumOfVertices; i++)
      {
         for (DSTable::RowIterator it(new_v_to_v, i); !it; ++it)
         {
            int old_i = (*old_v_to_v)(i, it.Column());
            int new_i = it.Index();
#ifdef MFEM_DEBUG
            if (old_i != new_i)
            {
               redges++;
            }
#endif
            old_dofs.SetSize(num_edge_dofs);
            new_dofs.SetSize(num_edge_dofs);
            for (int j = 0; j < num_edge_dofs; j++)
            {
               old_dofs[j] = offset + old_i * num_edge_dofs + j;
               new_dofs[j] = offset + new_i * num_edge_dofs + j;
            }
            fes->DofsToVDofs(old_dofs);
            fes->DofsToVDofs(new_dofs);
            for (int j = 0; j < old_dofs.Size(); j++)
            {
               (*Nodes)(new_dofs[j]) = onodes(old_dofs[j]);
            }
         }
      }
      offset += NumOfEdges * num_edge_dofs;
   }
#ifdef MFEM_DEBUG
   cout << "Mesh::DoNodeReorder : redges = " << redges << endl;
#endif

   // face dofs:
   // both enumeration and orientation of the faces may be different
   if (num_face_dofs > 0)
   {
      // generate the old face-vertex table using the unmodified 'faces'
      Table old_face_vertex;
      old_face_vertex.MakeI(NumOfFaces);
      for (int i = 0; i < NumOfFaces; i++)
      {
         old_face_vertex.AddColumnsInRow(i, faces[i]->GetNVertices());
      }
      old_face_vertex.MakeJ();
      for (int i = 0; i < NumOfFaces; i++)
         old_face_vertex.AddConnections(i, faces[i]->GetVertices(),
                                        faces[i]->GetNVertices());
      old_face_vertex.ShiftUpI();

      // update 'el_to_face', 'be_to_face', 'faces', and 'faces_info'
      STable3D *faces_tbl = GetElementToFaceTable(1);
      GenerateFaces();

      // loop over the old face numbers
      for (int i = 0; i < NumOfFaces; i++)
      {
         int *old_v = old_face_vertex.GetRow(i), *new_v;
         int new_i, new_or, *dof_ord;
         switch (old_face_vertex.RowSize(i))
         {
            case 3:
               new_i = (*faces_tbl)(old_v[0], old_v[1], old_v[2]);
               new_v = faces[new_i]->GetVertices();
               new_or = GetTriOrientation(old_v, new_v);
               dof_ord = fec->DofOrderForOrientation(Geometry::TRIANGLE, new_or);
               break;
            case 4:
            default:
               new_i = (*faces_tbl)(old_v[0], old_v[1], old_v[2], old_v[3]);
               new_v = faces[new_i]->GetVertices();
               new_or = GetQuadOrientation(old_v, new_v);
               dof_ord = fec->DofOrderForOrientation(Geometry::SQUARE, new_or);
               break;
         }

         old_dofs.SetSize(num_face_dofs);
         new_dofs.SetSize(num_face_dofs);
         for (int j = 0; j < num_face_dofs; j++)
         {
            old_dofs[j] = offset +     i * num_face_dofs + j;
            new_dofs[j] = offset + new_i * num_face_dofs + dof_ord[j];
            // we assumed the dofs are non-directional
            // i.e. dof_ord[j] is >= 0
         }
         fes->DofsToVDofs(old_dofs);
         fes->DofsToVDofs(new_dofs);
         for (int j = 0; j < old_dofs.Size(); j++)
         {
            (*Nodes)(new_dofs[j]) = onodes(old_dofs[j]);
         }
      }

      offset += NumOfFaces * num_face_dofs;
      delete faces_tbl;
   }

   // element dofs:
   // element orientation may be different
   if (num_elem_dofs > 1)
   {
      // matters when the 'fec' is
      // (this code is executed only for triangles/tets)
      // - Pk on triangles, k >= 4
      // - Qk on quads,     k >= 3
      // - Pk on tets,      k >= 5
      // - Qk on hexes,     k >= 3
      // - DG spaces
      // - ...

      // loop over all elements
      for (int i = 0; i < GetNE(); i++)
      {
         int *old_v = old_elem_vert->GetRow(i);
         int *new_v = elements[i]->GetVertices();
         int new_or, *dof_ord;
         int geom = elements[i]->GetGeometryType();
         switch (geom)
         {
            case Geometry::SEGMENT:
               new_or = (old_v[0] == new_v[0]) ? +1 : -1;
               break;
            case Geometry::TRIANGLE:
               new_or = GetTriOrientation(old_v, new_v);
               break;
            case Geometry::SQUARE:
               new_or = GetQuadOrientation(old_v, new_v);
               break;
            default:
               new_or = 0;
               cerr << "Mesh::DoNodeReorder : " << Geometry::Name[geom]
                    << " elements (" << fec->Name()
                    << " FE collection) are not supported yet!" << endl;
               mfem_error();
               break;
         }
         dof_ord = fec->DofOrderForOrientation(geom, new_or);
         if (dof_ord == NULL)
         {
            cerr << "Mesh::DoNodeReorder : FE collection '" << fec->Name()
                 << "' does not define reordering for " << Geometry::Name[geom]
                 << " elements!" << endl;
            mfem_error();
         }
         old_dofs.SetSize(num_elem_dofs);
         new_dofs.SetSize(num_elem_dofs);
         for (int j = 0; j < num_elem_dofs; j++)
         {
            // we assume the dofs are non-directional, i.e. dof_ord[j] is >= 0
            old_dofs[j] = offset + dof_ord[j];
            new_dofs[j] = offset + j;
         }
         fes->DofsToVDofs(old_dofs);
         fes->DofsToVDofs(new_dofs);
         for (int j = 0; j < old_dofs.Size(); j++)
         {
            (*Nodes)(new_dofs[j]) = onodes(old_dofs[j]);
         }

         offset += num_elem_dofs;
      }
   }

   // Update Tables, faces, etc
   if (Dim > 2)
   {
      if (num_face_dofs == 0)
      {
         // needed for FE spaces that have face dofs, even if
         // the 'Nodes' do not have face dofs.
         GetElementToFaceTable();
         GenerateFaces();
      }
      CheckBdrElementOrientation();
   }
   if (el_to_edge)
   {
      // update 'el_to_edge', 'be_to_edge' (2D), 'bel_to_edge' (3D)
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
      {
         // update 'faces' and 'faces_info'
         GenerateFaces();
         CheckBdrElementOrientation();
      }
   }
   Nodes->FESpace()->RebuildElementToDofTable();
}

void Mesh::FinalizeTetMesh(int generate_edges, int refine, bool fix_orientation)
{
   FinalizeCheck();
   CheckElementOrientation(fix_orientation);

   if (NumOfBdrElements == 0)
   {
      GetElementToFaceTable();
      GenerateFaces();
      GenerateBoundaryElements();
   }

   if (refine)
   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
      MarkTetMeshForRefinement(v_to_v);
   }

   GetElementToFaceTable();
   GenerateFaces();

   CheckBdrElementOrientation();

   if (generate_edges == 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   else
   {
      el_to_edge = NULL;  // Not really necessary -- InitTables was called
      bel_to_edge = NULL;
      NumOfEdges = 0;
   }

   SetAttributes();

   BaseGeom = Geometry::TETRAHEDRON;
   BaseBdrGeom = Geometry::TRIANGLE;

   meshgen = 1;
}

void Mesh::FinalizeHexMesh(int generate_edges, int refine, bool fix_orientation)
{
   FinalizeCheck();
   CheckElementOrientation(fix_orientation);

   GetElementToFaceTable();
   GenerateFaces();

   if (NumOfBdrElements == 0)
   {
      GenerateBoundaryElements();
   }

   CheckBdrElementOrientation();

   if (generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   else
   {
      NumOfEdges = 0;
   }

   SetAttributes();

   BaseGeom = Geometry::CUBE;
   BaseBdrGeom = Geometry::SQUARE;

   meshgen = 2;
}

void Mesh::FinalizeTopology()
{
   // Requirements: the following should be defined:
   //   1) Dim
   //   2) NumOfElements, elements
   //   3) NumOfBdrElements, boundary
   //   4) NumOfVertices
   // Optional:
   //   2) ncmesh may be defined
   //   3) el_to_edge may be allocated (it will be re-computed)

   FinalizeCheck();
   bool generate_edges = true;

   if (spaceDim == 0) { spaceDim = Dim; }
   if (ncmesh) { ncmesh->spaceDim = spaceDim; }

   InitBaseGeom();

   // set the mesh type ('meshgen')
   SetMeshGen();

   if (NumOfBdrElements == 0 && Dim > 2)
   {
      // in 3D, generate boundary elements before we 'MarkForRefinement'
	  if(Dim==3) GetElementToFaceTable();
	  else if(Dim==4)
	  {
		  GetElementToFaceTable4D();
	  }
      GenerateFaces();
      GenerateBoundaryElements();
   }
   else if (Dim == 1)
   {
      GenerateFaces();
   }

   // generate the faces
   if (Dim > 2)
   {
		  if(Dim==3) GetElementToFaceTable();
		  else if(Dim==4)
		  {
			  GetElementToFaceTable4D();
		  }
		  GenerateFaces();

		  if(Dim==4)
		  {
			 ReplaceBoundaryFromFaces();

		     GetElementToPlanarTable();
		     GeneratePlanars();

//			 GetElementToQuadTable4D();
//			 GenerateQuads4D();
		  }
   }
   else
   {
      NumOfFaces = 0;
   }

   // generate edges if requested
   if (Dim > 1 && generate_edges)
   {
      // el_to_edge may already be allocated (P2 VTK meshes)
      if (!el_to_edge) { el_to_edge = new Table; }
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
      {
         GenerateFaces(); // 'Faces' in 2D refers to the edges
         if (NumOfBdrElements == 0)
         {
            GenerateBoundaryElements();
         }
      }
   }
   else
   {
      NumOfEdges = 0;
   }

   if (ncmesh)
   {
      // tell NCMesh the numbering of edges/faces
      ncmesh->OnMeshUpdated(this);

      // update faces_info with NC relations
      GenerateNCFaceInfo();
   }

   // generate the arrays 'attributes' and 'bdr_attributes'
   SetAttributes();
}

void Mesh::Finalize(bool refine, bool fix_orientation)
{
   if (NURBSext || ncmesh)
   {
      MFEM_ASSERT(CheckElementOrientation(false) == 0, "");
      MFEM_ASSERT(CheckBdrElementOrientation() == 0, "");
      return;
   }

   // Requirements:
   //  1) FinilizeTopology() or equivalent was called
   //  2) if (Nodes == NULL), vertices must be defined
   //  3) if (Nodes != NULL), Nodes must be defined

   const bool check_orientation = true; // for regular elements, not boundary
   const bool curved = (Nodes != NULL);
   const bool may_change_topology =
      ( refine && (Dim > 1 && (meshgen & 1)) ) ||
      ( check_orientation && fix_orientation &&
        (Dim == 2 || (Dim == 3 && (meshgen & 1))) );

   DSTable *old_v_to_v = NULL;
   Table *old_elem_vert = NULL;

   if (curved && may_change_topology)
   {
      PrepareNodeReorder(&old_v_to_v, &old_elem_vert);
   }

   if (check_orientation)
   {
      // check and optionally fix element orientation
      CheckElementOrientation(fix_orientation);
   }
   if (refine)
   {
      MarkForRefinement();   // may change topology!
   }

   if (may_change_topology)
   {
      if (curved)
      {
         DoNodeReorder(old_v_to_v, old_elem_vert); // updates the mesh topology
         delete old_elem_vert;
         delete old_v_to_v;
      }
      else
      {
         FinalizeTopology(); // Re-computes some data unnecessarily.
      }

      // TODO: maybe introduce Mesh::NODE_REORDER operation and FESpace::
      // NodeReorderMatrix and do Nodes->Update() instead of DoNodeReorder?
   }

   // check and fix boundary element orientation
   CheckBdrElementOrientation();
}

void Mesh::FinalizeTesMesh(int generate_edges, int refine, bool fix_orientation)
{
   CheckElementOrientation(fix_orientation);

//   GetElementToFaceTable5D();

   GenerateFaces();

   if (NumOfBdrElements == 0)
      GenerateBoundaryElements();

   CheckBdrElementOrientation();

//   GetElementToTrigTable4D();
//   GenerateTrigs4D();
//
//   GetElementToQuadTable4D();
//   GenerateQuads4D();

   if(generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   else
      NumOfEdges = 0;

   SetAttributes();

   meshgen = 2;
}

void Mesh::Make4D(int nx, int ny, int nz, int nt, Element::Type type, int generate_edges,
               double sx, double sy, double sz, double st,
               bool generate_boundary, bool which_boundary[8],
               double shX, double shY, double shZ, double shT)
{
	int x, y, z, t;

	int NVert, NElem, NBdrElem;

	NVert = (nx+1) * (ny+1) * (nz+1) * (nt+1);
	NElem = nx * ny * nz * nt;
	NBdrElem = 0;
	if(generate_boundary) NBdrElem = 2 * (nx*ny*nz + nx*nz*nt + nx*ny*nt + ny*nz*nt);

	InitMesh(4, 4, NVert, NElem, NBdrElem);

	double coord[4];
	int ind[16];

	// Sets vertices and the corresponding coordinates
	for(t = 0; t<=nt; t++)
	{
		coord[3] = shT + ((double) t / nt) * st;
		for (z = 0; z <= nz; z++)
		{
		  coord[2] = shZ + ((double) z / nz) * sz;
		  for (y = 0; y <= ny; y++)
		  {
			 coord[1] = shY + ((double) y / ny) * sy;
			 for (x = 0; x <= nx; x++)
			 {
				coord[0] = shX + ((double) x / nx) * sx;
				AddVertex(coord);
			 }
		  }
		}
	}

  #define VTX4D(XC, YC, ZC, TC) ((XC)+((YC)+((ZC)+((TC)*(nz+1)))*(ny+1))*(nx+1))
   // Sets elements and the corresponding indices of vertices
   for(t = 0; t < nt; t++)
   {
	   for (z = 0; z < nz; z++)
	   {
		  for (y = 0; y < ny; y++)
		  {
			 for (x = 0; x < nx; x++)
			 {
				ind[0] = VTX4D(x  , y  , z  ,t  );
				ind[1] = VTX4D(x+1, y  , z  ,t  );
				ind[2] = VTX4D(x+1, y+1, z  ,t  );
				ind[3] = VTX4D(x  , y+1, z  ,t  );
				ind[4] = VTX4D(x  , y  , z+1,t  );
				ind[5] = VTX4D(x+1, y  , z+1,t  );
				ind[6] = VTX4D(x+1, y+1, z+1,t  );
				ind[7] = VTX4D(x  , y+1, z+1,t  );

				ind[8] = VTX4D(x  , y  , z  ,t+1);
				ind[9] = VTX4D(x+1, y  , z  ,t+1);
				ind[10] = VTX4D(x+1, y+1, z  ,t+1);
				ind[11] = VTX4D(x  , y+1, z  ,t+1);
				ind[12] = VTX4D(x  , y  , z+1,t+1);
				ind[13] = VTX4D(x+1, y  , z+1,t+1);
				ind[14] = VTX4D(x+1, y+1, z+1,t+1);
				ind[15] = VTX4D(x  , y+1, z+1,t+1);

				AddTes(ind, 1);
			 }
		  }
	   }
   }

   if(generate_boundary)
   {
   	   //x bottom
	   for(t = 0; t < nt; t++)
	   {
		   for (z = 0; z < nz; z++)
		   {
			  for (y = 0; y < ny; y++)
			  {
				ind[0] = VTX4D(0  , y  , z  ,t  );
				ind[1] = VTX4D(0  , y+1, z  ,t  );
				ind[3] = VTX4D(0  , y  , z+1,t  );
				ind[2] = VTX4D(0  , y+1, z+1,t  );
				ind[4] = VTX4D(0  , y  , z  ,t+1);
				ind[5] = VTX4D(0  , y+1, z  ,t+1);
				ind[7] = VTX4D(0  , y  , z+1,t+1);
				ind[6] = VTX4D(0  , y+1, z+1,t+1);

				AddBdrHex(ind, 2);
			  }
		   }
	   }
	   //x top
	   for(t = 0; t < nt; t++)
	   {
		   for (z = 0; z < nz; z++)
		   {
			  for (y = 0; y < ny; y++)
			  {
				ind[0] = VTX4D(nx, y  , z  ,t  );
				ind[1] = VTX4D(nx, y+1, z  ,t  );
				ind[3] = VTX4D(nx, y  , z+1,t  );
				ind[2] = VTX4D(nx, y+1, z+1,t  );
				ind[4] = VTX4D(nx, y  , z  ,t+1);
				ind[5] = VTX4D(nx, y+1, z  ,t+1);
				ind[7] = VTX4D(nx, y  , z+1,t+1);
				ind[6] = VTX4D(nx, y+1, z+1,t+1);

				AddBdrHex(ind, 3);
			  }
		   }
	   }

	   //y bottom
	   for(t = 0; t < nt; t++)
	   {
		   for (z = 0; z < nz; z++)
		   {
			 for (x = 0; x < nx; x++)
			 {
				ind[0] = VTX4D(x  , 0  , z  ,t  );
				ind[1] = VTX4D(x+1, 0  , z  ,t  );
				ind[3] = VTX4D(x  , 0  , z+1,t  );
				ind[2] = VTX4D(x+1, 0  , z+1,t  );
				ind[4] = VTX4D(x  , 0  , z  ,t+1);
				ind[5] = VTX4D(x+1, 0  , z  ,t+1);
				ind[7] = VTX4D(x  , 0  , z+1,t+1);
				ind[6] = VTX4D(x+1, 0  , z+1,t+1);

				AddBdrHex(ind, 4);
			 }

		   }
	   }
	   //y top
	   for(t = 0; t < nt; t++)
	   {
		   for (z = 0; z < nz; z++)
		   {
			 for (x = 0; x < nx; x++)
			 {
				ind[0] = VTX4D(x+1, ny, z  ,t  );
				ind[1] = VTX4D(x  , ny, z  ,t  );
				ind[3] = VTX4D(x+1, ny, z+1,t  );
				ind[2] = VTX4D(x  , ny, z+1,t  );
				ind[4] = VTX4D(x+1, ny, z  ,t+1);
				ind[5] = VTX4D(x  , ny, z  ,t+1);
				ind[7] = VTX4D(x+1, ny, z+1,t+1);
				ind[6] = VTX4D(x  , ny, z+1,t+1);

				AddBdrHex(ind, 5);
			 }

		   }
	   }

	   //z bottom
	   for(t = 0; t < nt; t++)
	   {
		  for (y = 0; y < ny; y++)
		  {
			 for (x = 0; x < nx; x++)
			 {
				ind[0] = VTX4D(x  , y  , 0  ,t  );
				ind[1] = VTX4D(x+1, y  , 0  ,t  );
				ind[2] = VTX4D(x+1, y+1, 0  ,t  );
				ind[3] = VTX4D(x  , y+1, 0  ,t  );
				ind[4] = VTX4D(x  , y  , 0  ,t+1);
				ind[5] = VTX4D(x+1, y  , 0  ,t+1);
				ind[6] = VTX4D(x+1, y+1, 0  ,t+1);
				ind[7] = VTX4D(x  , y+1, 0  ,t+1);

				AddBdrHex(ind, 6);
			 }
		  }
	   }

	   //z top
	   for(t = 0; t < nt; t++)
	   {
		  for (y = 0; y < ny; y++)
		  {
			 for (x = 0; x < nx; x++)
			 {
				ind[0] = VTX4D(x  , y  , nz,t  );
				ind[1] = VTX4D(x+1, y  , nz,t  );
				ind[2] = VTX4D(x+1, y+1, nz,t  );
				ind[3] = VTX4D(x  , y+1, nz,t  );
				ind[4] = VTX4D(x  , y  , nz,t+1);
				ind[5] = VTX4D(x+1, y  , nz,t+1);
				ind[6] = VTX4D(x+1, y+1, nz,t+1);
				ind[7] = VTX4D(x  , y+1, nz,t+1);

				AddBdrHex(ind, 7);
			 }
		  }
	   }


   	   //t bottom
	   for (z = 0; z < nz; z++)
	   {
		  for (y = 0; y < ny; y++)
		  {
			 for (x = 0; x < nx; x++)
			 {
				ind[0] = VTX4D(x  , y  , z  ,0  );
				ind[1] = VTX4D(x+1, y  , z  ,0  );
				ind[2] = VTX4D(x+1, y+1, z  ,0  );
				ind[3] = VTX4D(x  , y+1, z  ,0  );
				ind[4] = VTX4D(x  , y  , z+1,0  );
				ind[5] = VTX4D(x+1, y  , z+1,0  );
				ind[6] = VTX4D(x+1, y+1, z+1,0  );
				ind[7] = VTX4D(x  , y+1, z+1,0  );

				AddBdrHex(ind, 1);
			 }
		  }
	   }
   	   //t top
	   for (z = 0; z < nz; z++)
	   {
		  for (y = 0; y < ny; y++)
		  {
			 for (x = 0; x < nx; x++)
			 {
				ind[0] = VTX4D(x  , y  , z  ,nt  );
				ind[1] = VTX4D(x+1, y  , z  ,nt  );
				ind[2] = VTX4D(x+1, y+1, z  ,nt  );
				ind[3] = VTX4D(x  , y+1, z  ,nt  );
				ind[4] = VTX4D(x  , y  , z+1,nt  );
				ind[5] = VTX4D(x+1, y  , z+1,nt  );
				ind[6] = VTX4D(x+1, y+1, z+1,nt  );
				ind[7] = VTX4D(x  , y+1, z+1,nt  );

				AddBdrHex(ind, 8);
			 }
		  }
	   }
   }



   int refine = 1;
   bool fix_orientation = false;
   FinalizeTesMesh(generate_edges, refine, fix_orientation);
}

void Mesh::Make3D(int nx, int ny, int nz, Element::Type type,
                  int generate_edges, double sx, double sy, double sz)
{
   int x, y, z;

   int NVert, NElem, NBdrElem;

   NVert = (nx+1) * (ny+1) * (nz+1);
   NElem = nx * ny * nz;
   NBdrElem = 2*(nx*ny+nx*nz+ny*nz);
   if (type == Element::TETRAHEDRON)
   {
      NElem *= 6;
      NBdrElem *= 2;
   }

   InitMesh(3, 3, NVert, NElem, NBdrElem);

   double coord[3];
   int ind[8];

   // Sets vertices and the corresponding coordinates
   for (z = 0; z <= nz; z++)
   {
      coord[2] = ((double) z / nz) * sz;
      for (y = 0; y <= ny; y++)
      {
         coord[1] = ((double) y / ny) * sy;
         for (x = 0; x <= nx; x++)
         {
            coord[0] = ((double) x / nx) * sx;
            AddVertex(coord);
         }
      }
   }

#define VTX(XC, YC, ZC) ((XC)+((YC)+(ZC)*(ny+1))*(nx+1))

   // Sets elements and the corresponding indices of vertices
   for (z = 0; z < nz; z++)
   {
      for (y = 0; y < ny; y++)
      {
         for (x = 0; x < nx; x++)
         {
            ind[0] = VTX(x  , y  , z  );
            ind[1] = VTX(x+1, y  , z  );
            ind[2] = VTX(x+1, y+1, z  );
            ind[3] = VTX(x  , y+1, z  );
            ind[4] = VTX(x  , y  , z+1);
            ind[5] = VTX(x+1, y  , z+1);
            ind[6] = VTX(x+1, y+1, z+1);
            ind[7] = VTX(x  , y+1, z+1);
            if (type == Element::TETRAHEDRON)
            {
               AddHexAsTets(ind, 1);
            }
            else
            {
               AddHex(ind, 1);
            }
         }
      }
   }

   // Sets boundary elements and the corresponding indices of vertices
   // bottom, bdr. attribute 1
   for (y = 0; y < ny; y++)
      for (x = 0; x < nx; x++)
      {
         ind[0] = VTX(x  , y  , 0);
         ind[1] = VTX(x  , y+1, 0);
         ind[2] = VTX(x+1, y+1, 0);
         ind[3] = VTX(x+1, y  , 0);
         if (type == Element::TETRAHEDRON)
         {
            AddBdrQuadAsTriangles(ind, 1);
         }
         else
         {
            AddBdrQuad(ind, 1);
         }
      }
   // top, bdr. attribute 6
   for (y = 0; y < ny; y++)
      for (x = 0; x < nx; x++)
      {
         ind[0] = VTX(x  , y  , nz);
         ind[1] = VTX(x+1, y  , nz);
         ind[2] = VTX(x+1, y+1, nz);
         ind[3] = VTX(x  , y+1, nz);
         if (type == Element::TETRAHEDRON)
         {
            AddBdrQuadAsTriangles(ind, 6);
         }
         else
         {
            AddBdrQuad(ind, 6);
         }
      }
   // left, bdr. attribute 5
   for (z = 0; z < nz; z++)
      for (y = 0; y < ny; y++)
      {
         ind[0] = VTX(0  , y  , z  );
         ind[1] = VTX(0  , y  , z+1);
         ind[2] = VTX(0  , y+1, z+1);
         ind[3] = VTX(0  , y+1, z  );
         if (type == Element::TETRAHEDRON)
         {
            AddBdrQuadAsTriangles(ind, 5);
         }
         else
         {
            AddBdrQuad(ind, 5);
         }
      }
   // right, bdr. attribute 3
   for (z = 0; z < nz; z++)
      for (y = 0; y < ny; y++)
      {
         ind[0] = VTX(nx, y  , z  );
         ind[1] = VTX(nx, y+1, z  );
         ind[2] = VTX(nx, y+1, z+1);
         ind[3] = VTX(nx, y  , z+1);
         if (type == Element::TETRAHEDRON)
         {
            AddBdrQuadAsTriangles(ind, 3);
         }
         else
         {
            AddBdrQuad(ind, 3);
         }
      }
   // front, bdr. attribute 2
   for (x = 0; x < nx; x++)
      for (z = 0; z < nz; z++)
      {
         ind[0] = VTX(x  , 0, z  );
         ind[1] = VTX(x+1, 0, z  );
         ind[2] = VTX(x+1, 0, z+1);
         ind[3] = VTX(x  , 0, z+1);
         if (type == Element::TETRAHEDRON)
         {
            AddBdrQuadAsTriangles(ind, 2);
         }
         else
         {
            AddBdrQuad(ind, 2);
         }
      }
   // back, bdr. attribute 4
   for (x = 0; x < nx; x++)
      for (z = 0; z < nz; z++)
      {
         ind[0] = VTX(x  , ny, z  );
         ind[1] = VTX(x  , ny, z+1);
         ind[2] = VTX(x+1, ny, z+1);
         ind[3] = VTX(x+1, ny, z  );
         if (type == Element::TETRAHEDRON)
         {
            AddBdrQuadAsTriangles(ind, 4);
         }
         else
         {
            AddBdrQuad(ind, 4);
         }
      }

#if 0
   ofstream test_stream("debug.mesh");
   Print(test_stream);
   test_stream.close();
#endif

   int refine = 1;
   bool fix_orientation = true;

   if (type == Element::TETRAHEDRON)
   {
      FinalizeTetMesh(generate_edges, refine, fix_orientation);
   }
   else
   {
      FinalizeHexMesh(generate_edges, refine, fix_orientation);
   }
}

void Mesh::Make2D(int nx, int ny, Element::Type type, int generate_edges,
                  double sx, double sy)
{
   int i, j, k;

   SetEmpty();

   Dim = spaceDim = 2;

   // Creates quadrilateral mesh
   if (type == Element::QUADRILATERAL)
   {
      meshgen = 2;
      NumOfVertices = (nx+1) * (ny+1);
      NumOfElements = nx * ny;
      NumOfBdrElements = 2 * nx + 2 * ny;
      BaseGeom = Geometry::SQUARE;
      BaseBdrGeom = Geometry::SEGMENT;

      vertices.SetSize(NumOfVertices);
      elements.SetSize(NumOfElements);
      boundary.SetSize(NumOfBdrElements);

      double cx, cy;
      int ind[4];

      // Sets vertices and the corresponding coordinates
      k = 0;
      for (j = 0; j < ny+1; j++)
      {
         cy = ((double) j / ny) * sy;
         for (i = 0; i < nx+1; i++)
         {
            cx = ((double) i / nx) * sx;
            vertices[k](0) = cx;
            vertices[k](1) = cy;
            k++;
         }
      }

      // Sets elements and the corresponding indices of vertices
      k = 0;
      for (j = 0; j < ny; j++)
      {
         for (i = 0; i < nx; i++)
         {
            ind[0] = i + j*(nx+1);
            ind[1] = i + 1 +j*(nx+1);
            ind[2] = i + 1 + (j+1)*(nx+1);
            ind[3] = i + (j+1)*(nx+1);
            elements[k] = new Quadrilateral(ind);
            k++;
         }
      }

      // Sets boundary elements and the corresponding indices of vertices
      int m = (nx+1)*ny;
      for (i = 0; i < nx; i++)
      {
         boundary[i] = new Segment(i, i+1, 1);
         boundary[nx+i] = new Segment(m+i+1, m+i, 3);
      }
      m = nx+1;
      for (j = 0; j < ny; j++)
      {
         boundary[2*nx+j] = new Segment((j+1)*m, j*m, 4);
         boundary[2*nx+ny+j] = new Segment(j*m+nx, (j+1)*m+nx, 2);
      }
   }
   // Creates triangular mesh
   else if (type == Element::TRIANGLE)
   {
      meshgen = 1;
      NumOfVertices = (nx+1) * (ny+1);
      NumOfElements = 2 * nx * ny;
      NumOfBdrElements = 2 * nx + 2 * ny;
      BaseGeom = Geometry::TRIANGLE;
      BaseBdrGeom = Geometry::SEGMENT;

      vertices.SetSize(NumOfVertices);
      elements.SetSize(NumOfElements);
      boundary.SetSize(NumOfBdrElements);

      double cx, cy;
      int ind[3];

      // Sets vertices and the corresponding coordinates
      k = 0;
      for (j = 0; j < ny+1; j++)
      {
         cy = ((double) j / ny) * sy;
         for (i = 0; i < nx+1; i++)
         {
            cx = ((double) i / nx) * sx;
            vertices[k](0) = cx;
            vertices[k](1) = cy;
            k++;
         }
      }

      // Sets the elements and the corresponding indices of vertices
      k = 0;
      for (j = 0; j < ny; j++)
      {
         for (i = 0; i < nx; i++)
         {
            ind[0] = i + j*(nx+1);
            ind[1] = i + 1 + (j+1)*(nx+1);
            ind[2] = i + (j+1)*(nx+1);
            elements[k] = new Triangle(ind);
            k++;
            ind[1] = i + 1 + j*(nx+1);
            ind[2] = i + 1 + (j+1)*(nx+1);
            elements[k] = new Triangle(ind);
            k++;
         }
      }

      // Sets boundary elements and the corresponding indices of vertices
      int m = (nx+1)*ny;
      for (i = 0; i < nx; i++)
      {
         boundary[i] = new Segment(i, i+1, 1);
         boundary[nx+i] = new Segment(m+i+1, m+i, 3);
      }
      m = nx+1;
      for (j = 0; j < ny; j++)
      {
         boundary[2*nx+j] = new Segment((j+1)*m, j*m, 4);
         boundary[2*nx+ny+j] = new Segment(j*m+nx, (j+1)*m+nx, 2);
      }

      MarkTriMeshForRefinement();
   }
   else
   {
      MFEM_ABORT("Unsupported element type.");
   }

   CheckElementOrientation();

   if (generate_edges == 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      GenerateFaces();
      CheckBdrElementOrientation();
   }
   else
   {
      NumOfEdges = 0;
   }

   NumOfFaces = 0;

   attributes.Append(1);
   bdr_attributes.Append(1); bdr_attributes.Append(2);
   bdr_attributes.Append(3); bdr_attributes.Append(4);
}

void Mesh::Make1D(int n, double sx)
{
   int j, ind[1];

   SetEmpty();

   Dim = 1;
   spaceDim = 1;

   BaseGeom = Geometry::SEGMENT;
   BaseBdrGeom = Geometry::POINT;

   meshgen = 1;

   NumOfVertices = n + 1;
   NumOfElements = n;
   NumOfBdrElements = 2;
   vertices.SetSize(NumOfVertices);
   elements.SetSize(NumOfElements);
   boundary.SetSize(NumOfBdrElements);

   // Sets vertices and the corresponding coordinates
   for (j = 0; j < n+1; j++)
   {
      vertices[j](0) = ((double) j / n) * sx;
   }

   // Sets elements and the corresponding indices of vertices
   for (j = 0; j < n; j++)
   {
      elements[j] = new Segment(j, j+1, 1);
   }

   // Sets the boundary elements
   ind[0] = 0;
   boundary[0] = new Point(ind, 1);
   ind[0] = n;
   boundary[1] = new Point(ind, 2);

   NumOfEdges = 0;
   NumOfFaces = 0;

   GenerateFaces();

   attributes.Append(1);
   bdr_attributes.Append(1); bdr_attributes.Append(2);
}

Mesh::Mesh(const Mesh &mesh, bool copy_nodes)
{
   Dim = mesh.Dim;
   spaceDim = mesh.spaceDim;

   NumOfVertices = mesh.NumOfVertices;
   NumOfElements = mesh.NumOfElements;
   NumOfBdrElements = mesh.NumOfBdrElements;
   NumOfEdges = mesh.NumOfEdges;
   NumOfFaces = mesh.NumOfFaces;
   NumOfPlanars = mesh.NumOfPlanars;

   BaseGeom = mesh.BaseGeom;
   BaseBdrGeom = mesh.BaseBdrGeom;

   meshgen = mesh.meshgen;

   // Create the new Mesh instance without a record of its refinement history
   sequence = 0;
   last_operation = Mesh::NONE;

   // Duplicate the elements
   elements.SetSize(NumOfElements);
   for (int i = 0; i < NumOfElements; i++)
   {
      elements[i] = mesh.elements[i]->Duplicate(this);
   }

   // Copy the vertices
   MFEM_ASSERT(mesh.vertices.Size() == NumOfVertices, "internal MFEM error!");
   mesh.vertices.Copy(vertices);

   // Duplicate the boundary
   boundary.SetSize(NumOfBdrElements);
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      boundary[i] = mesh.boundary[i]->Duplicate(this);
   }

   // Copy the element-to-face Table, el_to_face
   el_to_face = (mesh.el_to_face) ? new Table(*mesh.el_to_face) : NULL;

   // Copy the boundary-to-face Array, be_to_face.
   mesh.be_to_face.Copy(be_to_face);

   // Copy the element-to-edge Table, el_to_edge
   el_to_edge = (mesh.el_to_edge) ? new Table(*mesh.el_to_edge) : NULL;

   // Copy the boudary-to-edge Table, bel_to_edge (3D)
   bel_to_edge = (mesh.bel_to_edge) ? new Table(*mesh.bel_to_edge) : NULL;

   // Copy the boudary-to-edge Array, be_to_edge (2D)
   mesh.be_to_edge.Copy(be_to_edge);

   el_to_planar = (mesh.el_to_planar) ? new Table(*mesh.el_to_planar) : NULL;
   bel_to_planar = (mesh.bel_to_planar) ? new Table(*mesh.bel_to_planar) : NULL;

   // Duplicate the faces and faces_info.
   faces.SetSize(mesh.faces.Size());
   for (int i = 0; i < faces.Size(); i++)
   {
      Element *face = mesh.faces[i]; // in 1D the faces are NULL
      faces[i] = (face) ? face->Duplicate(this) : NULL;
   }
   mesh.faces_info.Copy(faces_info);

   // Do NOT copy the element-to-element Table, el_to_el
   el_to_el = NULL;

   // Do NOT copy the face-to-edge Table, face_edge
   face_edge = NULL;

   // Copy the edge-to-vertex Table, edge_vertex
   edge_vertex = (mesh.edge_vertex) ? new Table(*mesh.edge_vertex) : NULL;

   // Copy the attributes and bdr_attributes
   mesh.attributes.Copy(attributes);
   mesh.bdr_attributes.Copy(bdr_attributes);

   // No support for NURBS meshes, yet. Need deep copy for NURBSExtension.
   MFEM_VERIFY(mesh.NURBSext == NULL,
               "copying NURBS meshes is not implemented");
   NURBSext = NULL;

   // Deep copy the NCMesh.
   ncmesh = mesh.ncmesh ? new NCMesh(*mesh.ncmesh) : NULL;

   // Duplicate the Nodes, including the FiniteElementCollection and the
   // FiniteElementSpace
   if (mesh.Nodes && copy_nodes)
   {
      FiniteElementSpace *fes = mesh.Nodes->FESpace();
      const FiniteElementCollection *fec = fes->FEColl();
      FiniteElementCollection *fec_copy =
         FiniteElementCollection::New(fec->Name());
      FiniteElementSpace *fes_copy =
         new FiniteElementSpace(this, fec_copy, fes->GetVDim(),
                                fes->GetOrdering());
      Nodes = new GridFunction(fes_copy);
      Nodes->MakeOwner(fec_copy);
      *Nodes = *mesh.Nodes;
      own_nodes = 1;
   }
   else
   {
      Nodes = mesh.Nodes;
      own_nodes = 0;
   }
}

Mesh::Mesh(const char *filename, int generate_edges, int refine,
           bool fix_orientation)
{
   // Initialization as in the default constructor
   SetEmpty();

   named_ifgzstream imesh(filename);
   if (!imesh)
   {
      // Abort with an error message.
      MFEM_ABORT("Mesh file not found: " << filename << '\n');
   }
   else
   {
      Load(imesh, generate_edges, refine, fix_orientation);
   }
}

Mesh::Mesh(std::istream &input, int generate_edges, int refine,
           bool fix_orientation)
{
   SetEmpty();
   Load(input, generate_edges, refine, fix_orientation);
}

void Mesh::ChangeVertexDataOwnership(double *vertex_data, int len_vertex_data,
                                     bool zerocopy)
{
   // A dimension of 3 is now required since we use mfem::Vertex objects as PODs
   // and these object have a hardcoded double[3] entry
   MFEM_VERIFY(len_vertex_data >= NumOfVertices * 3,
               "Not enough vertices in external array : "
               "len_vertex_data = "<< len_vertex_data << ", "
               "NumOfVertices * 3 = " << NumOfVertices * 3);
   // Allow multiple calls to this method with the same vertex_data
   if (vertex_data == (double *)(vertices.GetData()))
   {
      MFEM_ASSERT(!vertices.OwnsData(), "invalid ownership");
      return;
   }
   if (!zerocopy)
   {
      memcpy(vertex_data, vertices.GetData(),
             NumOfVertices * 3 * sizeof(double));
   }
   // Vertex is POD double[3]
   vertices.MakeRef(reinterpret_cast<Vertex*>(vertex_data), NumOfVertices);
}

Mesh::Mesh(double *_vertices, int num_vertices,
           int *element_indices, Geometry::Type element_type,
           int *element_attributes, int num_elements,
           int *boundary_indices, Geometry::Type boundary_type,
           int *boundary_attributes, int num_boundary_elements,
           int dimension, int space_dimension)
{
   if (space_dimension == -1)
   {
      space_dimension = dimension;
   }

   InitMesh(dimension, space_dimension, /*num_vertices*/ 0, num_elements,
            num_boundary_elements);

   int element_index_stride = Geometry::NumVerts[element_type];
   int boundary_index_stride = Geometry::NumVerts[boundary_type];

   // assuming Vertex is POD
   vertices.MakeRef(reinterpret_cast<Vertex*>(_vertices), num_vertices);
   NumOfVertices = num_vertices;

   for (int i = 0; i < num_elements; i++)
   {
      elements[i] = NewElement(element_type);
      elements[i]->SetVertices(element_indices + i * element_index_stride);
      elements[i]->SetAttribute(element_attributes[i]);
   }
   NumOfElements = num_elements;

   for (int i = 0; i < num_boundary_elements; i++)
   {
      boundary[i] = NewElement(boundary_type);
      boundary[i]->SetVertices(boundary_indices + i * boundary_index_stride);
      boundary[i]->SetAttribute(boundary_attributes[i]);
   }
   NumOfBdrElements = num_boundary_elements;

   FinalizeTopology();
}

Element *Mesh::NewElement(int geom)
{
   switch (geom)
   {
      case Geometry::POINT:     return (new Point);
      case Geometry::SEGMENT:   return (new Segment);
      case Geometry::TRIANGLE:  return (new Triangle);
      case Geometry::SQUARE:    return (new Quadrilateral);
      case Geometry::CUBE:      return (new Hexahedron);
      case Geometry::TETRAHEDRON:
#ifdef MFEM_USE_MEMALLOC
         return TetMemory.Alloc();
#else
         return (new Tetrahedron);
#endif
      case Geometry::PENTATOPE: return (new Pentatope);
   }

   return NULL;
}

Element *Mesh::ReadElementWithoutAttr(std::istream &input)
{
   int geom, nv, *v;
   Element *el;

   input >> geom;
   el = NewElement(geom);
   nv = el->GetNVertices();
   v  = el->GetVertices();
   for (int i = 0; i < nv; i++)
   {
      input >> v[i];
   }

   return el;
}

void Mesh::PrintElementWithoutAttr(const Element *el, std::ostream &out)
{
   out << el->GetGeometryType();
   const int nv = el->GetNVertices();
   const int *v = el->GetVertices();
   for (int j = 0; j < nv; j++)
   {
      out << ' ' << v[j];
   }
   out << '\n';
}

Element *Mesh::ReadElement(std::istream &input)
{
   int attr;
   Element *el;

   input >> attr;
   el = ReadElementWithoutAttr(input);
   el->SetAttribute(attr);

   return el;
}

void Mesh::PrintElement(const Element *el, std::ostream &out)
{
   out << el->GetAttribute() << ' ';
   PrintElementWithoutAttr(el, out);
}

void Mesh::SetMeshGen()
{
   meshgen = 0;
   for (int i = 0; i < NumOfElements; i++)
   {
      switch (elements[i]->GetType())
      {
         case Element::SEGMENT:
         case Element::TRIANGLE:
         case Element::TETRAHEDRON:
         case Element::PENTATOPE:
            meshgen |= 1; break;

         case Element::QUADRILATERAL:
         case Element::HEXAHEDRON:
         case Element::TESSERACT:
            meshgen |= 2;
      }
   }
}

void Mesh::Loader(std::istream &input, int generate_edges,
                  std::string parse_tag)
{
   int curved = 0, read_gf = 1;

   if (!input)
   {
      MFEM_ABORT("Input stream is not open");
   }

   Clear();

   string mesh_type;
   input >> ws;
   getline(input, mesh_type);
   filter_dos(mesh_type);

   // MFEM's native mesh formats
   bool mfem_v10 = (mesh_type == "MFEM mesh v1.0");
   bool mfem_v11 = (mesh_type == "MFEM mesh v1.1");
   bool mfem_v12 = (mesh_type == "MFEM mesh v1.2");
   if (mfem_v10 || mfem_v11 || mfem_v12) // MFEM's own mesh formats
   {
      // Formats mfem_v12 and newer have a tag indicating the end of the mesh
      // section in the stream. A user provided parse tag can also be provided
      // via the arguments. For example, if this is called from parallel mesh
      // object, it can indicate to read until parallel mesh section begins.
      if ( mfem_v12 && parse_tag.empty() )
      {
         parse_tag = "mfem_mesh_end";
      }
      ReadMFEMMesh(input, mfem_v11, curved);
   }
   else if (mesh_type == "linemesh") // 1D mesh
   {
      ReadLineMesh(input);
   }
   else if (mesh_type == "areamesh2" || mesh_type == "curved_areamesh2")
   {
      if (mesh_type == "curved_areamesh2")
      {
         curved = 1;
      }
      ReadNetgen2DMesh(input, curved);
   }
   else if (mesh_type == "NETGEN" || mesh_type == "NETGEN_Neutral_Format")
   {
      ReadNetgen3DMesh(input);
   }
   else if (mesh_type == "TrueGrid")
   {
      ReadTrueGridMesh(input);
   }
   else if (mesh_type == "# vtk DataFile Version 3.0" ||
            mesh_type == "# vtk DataFile Version 2.0") // VTK
   {
      ReadVTKMesh(input, curved, read_gf);
   }
   else if (mesh_type == "MFEM NURBS mesh v1.0")
   {
      ReadNURBSMesh(input, curved, read_gf);
   }
   else if (mesh_type == "MFEM INLINE mesh v1.0")
   {
      ReadInlineMesh(input, generate_edges);
      return; // done with inline mesh construction
   }
   else if (mesh_type == "$MeshFormat") // Gmsh
   {
      ReadGmshMesh(input);
   }
   else if
   ((mesh_type.size() > 2 &&
     mesh_type[0] == 'C' && mesh_type[1] == 'D' && mesh_type[2] == 'F') ||
    (mesh_type.size() > 3 &&
     mesh_type[1] == 'H' && mesh_type[2] == 'D' && mesh_type[3] == 'F'))
   {
      named_ifgzstream *mesh_input = dynamic_cast<named_ifgzstream *>(&input);
      if (mesh_input)
      {
#ifdef MFEM_USE_NETCDF
         ReadCubit(mesh_input->filename, curved, read_gf);
#else
         MFEM_ABORT("NetCDF support requires configuration with"
                    " MFEM_USE_NETCDF=YES");
         return;
#endif
      }
      else
      {
         MFEM_ABORT("Can not determine Cubit mesh filename!"
                    " Use mfem::named_ifgzstream for input.");
         return;
      }
   }
   else
   {
      MFEM_ABORT("Unknown input mesh format: " << mesh_type);
      return;
   }

   //for a 4d mesh sort the element and boundary element indices by the node numbers
   if(spaceDim==4)
   {
	   swappedElements.SetSize(NumOfElements);
	   DenseMatrix J(4,4);
	   for (int j = 0; j < NumOfElements; j++)
	   {
		   if (elements[j]->GetType() == Element::PENTATOPE)
		   {
			   int *v = elements[j]->GetVertices();
			   Sort5(v[0], v[1], v[2], v[3], v[4]);

			   GetElementJacobian(j, J);
			   if(J.Det() < 0.0)
			   {
				   swappedElements[j] = true;
				   Swap(v);
			   }else
			   {
				   swappedElements[j] = false;
			   }
		   }

	   }
       for (int j = 0; j < NumOfBdrElements; j++)
       {
		   if (boundary[j]->GetType() == Element::TETRAHEDRON)
		   {
			   int *v = boundary[j]->GetVertices();
			   Sort4(v[0], v[1], v[2], v[3]);
		   }
       }
   }



   // at this point the following should be defined:
   //  1) Dim
   //  2) NumOfElements, elements
   //  3) NumOfBdrElements, boundary
   //  4) NumOfVertices, with allocated space in vertices
   //  5) curved
   //  5a) if curved == 0, vertices must be defined
   //  5b) if curved != 0 and read_gf != 0,
   //         'input' must point to a GridFunction
   //  5c) if curved != 0 and read_gf == 0,
   //         vertices and Nodes must be defined
   // optional:
   //  1) el_to_edge may be allocated (as in the case of P2 VTK meshes)
   //  2) ncmesh may be allocated

   // FinalizeTopology() will:
   // - assume that generate_edges is true
   // - assume that refine is false
   // - does not check the orientation of regular and boundary elements
   FinalizeTopology();

   if (curved && read_gf)
   {
      Nodes = new GridFunction(this, input);
      own_nodes = 1;
      spaceDim = Nodes->VectorDim();
      if (ncmesh) { ncmesh->spaceDim = spaceDim; }
      // Set the 'vertices' from the 'Nodes'
      for (int i = 0; i < spaceDim; i++)
      {
         Vector vert_val;
         Nodes->GetNodalValues(vert_val, i+1);
         for (int j = 0; j < NumOfVertices; j++)
         {
            vertices[j](i) = vert_val(j);
         }
      }
   }

   // If a parse tag was supplied, keep reading the stream until the tag is
   // encountered.
   if (mfem_v12)
   {
      string line;
      do
      {
         skip_comment_lines(input, '#');
         MFEM_VERIFY(input.good(), "Required mesh-end tag not found");
         getline(input, line);
         filter_dos(line);
         // mfem v1.2 may not have parse_tag in it, e.g. if trying to read a
         // serial mfem v1.2 mesh as parallel with "mfem_serial_mesh_end" as
         // parse_tag. That's why, regardless of parse_tag, we stop reading if
         // we find "mfem_mesh_end" which is required by mfem v1.2 format.
         if (line == "mfem_mesh_end") { break; }
      }
      while (line != parse_tag);
   }

   // Finalize(...) should be called after this, if needed.
}

Mesh::Mesh(Mesh *mesh_array[], int num_pieces)
{
   int      i, j, ie, ib, iv, *v, nv;
   Element *el;
   Mesh    *m;

   SetEmpty();

   Dim = mesh_array[0]->Dimension();
   spaceDim = mesh_array[0]->SpaceDimension();

   BaseGeom = mesh_array[0]->BaseGeom;
   BaseBdrGeom = mesh_array[0]->BaseBdrGeom;

   if (mesh_array[0]->NURBSext)
   {
      // assuming the pieces form a partition of a NURBS mesh
      NURBSext = new NURBSExtension(mesh_array, num_pieces);

      NumOfVertices = NURBSext->GetNV();
      NumOfElements = NURBSext->GetNE();

      NURBSext->GetElementTopo(elements);

      // NumOfBdrElements = NURBSext->GetNBE();
      // NURBSext->GetBdrElementTopo(boundary);

      Array<int> lvert_vert, lelem_elem;

      // Here, for visualization purposes, we copy the boundary elements from
      // the individual pieces which include the interior boundaries.  This
      // creates 'boundary' array that is different from the one generated by
      // the NURBSExtension which, in particular, makes the boundary-dof table
      // invalid. This, in turn, causes GetBdrElementTransformation to not
      // function properly.
      NumOfBdrElements = 0;
      for (i = 0; i < num_pieces; i++)
      {
         NumOfBdrElements += mesh_array[i]->GetNBE();
      }
      boundary.SetSize(NumOfBdrElements);
      vertices.SetSize(NumOfVertices);
      ib = 0;
      for (i = 0; i < num_pieces; i++)
      {
         m = mesh_array[i];
         m->NURBSext->GetVertexLocalToGlobal(lvert_vert);
         m->NURBSext->GetElementLocalToGlobal(lelem_elem);
         // copy the element attributes
         for (j = 0; j < m->GetNE(); j++)
         {
            elements[lelem_elem[j]]->SetAttribute(m->GetAttribute(j));
         }
         // copy the boundary
         for (j = 0; j < m->GetNBE(); j++)
         {
            el = m->GetBdrElement(j)->Duplicate(this);
            v  = el->GetVertices();
            nv = el->GetNVertices();
            for (int k = 0; k < nv; k++)
            {
               v[k] = lvert_vert[v[k]];
            }
            boundary[ib++] = el;
         }
         // copy the vertices
         for (j = 0; j < m->GetNV(); j++)
         {
            vertices[lvert_vert[j]].SetCoords(m->SpaceDimension(),
                                              m->GetVertex(j));
         }
      }
   }
   else // not a NURBS mesh
   {
      NumOfElements    = 0;
      NumOfBdrElements = 0;
      NumOfVertices    = 0;
      for (i = 0; i < num_pieces; i++)
      {
         m = mesh_array[i];
         NumOfElements    += m->GetNE();
         NumOfBdrElements += m->GetNBE();
         NumOfVertices    += m->GetNV();
      }
      elements.SetSize(NumOfElements);
      boundary.SetSize(NumOfBdrElements);
      vertices.SetSize(NumOfVertices);
      ie = ib = iv = 0;
      for (i = 0; i < num_pieces; i++)
      {
         m = mesh_array[i];
         // copy the elements
         for (j = 0; j < m->GetNE(); j++)
         {
            el = m->GetElement(j)->Duplicate(this);
            v  = el->GetVertices();
            nv = el->GetNVertices();
            for (int k = 0; k < nv; k++)
            {
               v[k] += iv;
            }
            elements[ie++] = el;
         }
         // copy the boundary elements
         for (j = 0; j < m->GetNBE(); j++)
         {
            el = m->GetBdrElement(j)->Duplicate(this);
            v  = el->GetVertices();
            nv = el->GetNVertices();
            for (int k = 0; k < nv; k++)
            {
               v[k] += iv;
            }
            boundary[ib++] = el;
         }
         // copy the vertices
         for (j = 0; j < m->GetNV(); j++)
         {
            vertices[iv++].SetCoords(m->SpaceDimension(), m->GetVertex(j));
         }
      }
   }

   // set the mesh type ('meshgen')
   meshgen = 0;
   for (i = 0; i < num_pieces; i++)
   {
      meshgen |= mesh_array[i]->MeshGenerator();
   }

   // generate faces
   if (Dim > 2)
   {
      GetElementToFaceTable();
      GenerateFaces();
   }
   else
   {
      NumOfFaces = 0;
   }

   // generate edges
   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
      {
         GenerateFaces();   // 'Faces' in 2D refers to the edges
      }
   }
   else
   {
      NumOfEdges = 0;
   }

   // generate the arrays 'attributes' and ' bdr_attributes'
   SetAttributes();

   // copy the nodes (curvilinear meshes)
   GridFunction *g = mesh_array[0]->GetNodes();
   if (g)
   {
      Array<GridFunction *> gf_array(num_pieces);
      for (i = 0; i < num_pieces; i++)
      {
         gf_array[i] = mesh_array[i]->GetNodes();
      }
      Nodes = new GridFunction(this, gf_array, num_pieces);
      own_nodes = 1;
   }

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
   CheckBdrElementOrientation(false);
#endif
}

Mesh::Mesh(Mesh *orig_mesh, int ref_factor, int ref_type)
{
   Dim = orig_mesh->Dimension();
   MFEM_VERIFY(ref_factor > 1, "the refinement factor must be > 1");
   MFEM_VERIFY(ref_type == BasisType::ClosedUniform ||
               ref_type == BasisType::GaussLobatto, "invalid refinement type");
   MFEM_VERIFY(Dim == 2 || Dim == 3,
               "only implemented for Hexahedron and Quadrilateral elements in "
               "2D/3D");

   // Construct a scalar H1 FE space of order ref_factor and use its dofs as
   // the indices of the new, refined vertices.
   H1_FECollection rfec(ref_factor, Dim, ref_type);
   FiniteElementSpace rfes(orig_mesh, &rfec);

   int r_bndr_factor = ref_factor * (Dim == 2 ? 1 : ref_factor);
   int r_elem_factor = ref_factor * r_bndr_factor;

   int r_num_vert = rfes.GetNDofs();
   int r_num_elem = orig_mesh->GetNE() * r_elem_factor;
   int r_num_bndr = orig_mesh->GetNBE() * r_bndr_factor;

   InitMesh(Dim, orig_mesh->SpaceDimension(), r_num_vert, r_num_elem,
            r_num_bndr);

   // Set the number of vertices, set the actual coordinates later
   NumOfVertices = r_num_vert;
   // Add refined elements and set vertex coordinates
   Array<int> rdofs;
   DenseMatrix phys_pts;
   int max_nv = 0;
   for (int el = 0; el < orig_mesh->GetNE(); el++)
   {
      int geom = orig_mesh->GetElementBaseGeometry(el);
      int attrib = orig_mesh->GetAttribute(el);
      int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, ref_factor);

      max_nv = std::max(max_nv, nvert);
      rfes.GetElementDofs(el, rdofs);
      MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
      const FiniteElement *rfe = rfes.GetFE(el);
      orig_mesh->GetElementTransformation(el)->Transform(rfe->GetNodes(),
                                                         phys_pts);
      const int *c2h_map = rfec.GetDofMap(geom);
      for (int i = 0; i < phys_pts.Width(); i++)
      {
         vertices[rdofs[i]].SetCoords(spaceDim, phys_pts.GetColumn(i));
      }
      for (int j = 0; j < RG.RefGeoms.Size()/nvert; j++)
      {
         Element *elem = NewElement(geom);
         elem->SetAttribute(attrib);
         int *v = elem->GetVertices();
         for (int k = 0; k < nvert; k++)
         {
            int cid = RG.RefGeoms[k+nvert*j]; // local Cartesian index
            v[k] = rdofs[c2h_map[cid]];
         }
         AddElement(elem);
      }
   }
   // Add refined boundary elements
   for (int el = 0; el < orig_mesh->GetNBE(); el++)
   {
      int geom = orig_mesh->GetBdrElementBaseGeometry(el);
      int attrib = orig_mesh->GetBdrAttribute(el);
      int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, ref_factor);

      rfes.GetBdrElementDofs(el, rdofs);
      MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
      const int *c2h_map = rfec.GetDofMap(geom);
      for (int j = 0; j < RG.RefGeoms.Size()/nvert; j++)
      {
         Element *elem = NewElement(geom);
         elem->SetAttribute(attrib);
         int *v = elem->GetVertices();
         for (int k = 0; k < nvert; k++)
         {
            int cid = RG.RefGeoms[k+nvert*j]; // local Cartesian index
            v[k] = rdofs[c2h_map[cid]];
         }
         AddBdrElement(elem);
      }
   }

   FinalizeTopology();
   sequence = orig_mesh->GetSequence() + 1;
   last_operation = Mesh::REFINE;

   // Setup the data for the coarse-fine refinement transformations
   MFEM_VERIFY(BaseGeom != -1, "meshes with mixed elements are not supported");
   CoarseFineTr.point_matrices.SetSize(Dim, max_nv, r_elem_factor);
   CoarseFineTr.embeddings.SetSize(GetNE());
   if (orig_mesh->GetNE() > 0)
   {
      const int el = 0;
      int geom = orig_mesh->GetElementBaseGeometry(el);
      int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, ref_factor);
      const int *c2h_map = rfec.GetDofMap(geom);
      const IntegrationRule &r_nodes = rfes.GetFE(el)->GetNodes();
      for (int j = 0; j < RG.RefGeoms.Size()/nvert; j++)
      {
         DenseMatrix &Pj = CoarseFineTr.point_matrices(j);
         for (int k = 0; k < nvert; k++)
         {
            int cid = RG.RefGeoms[k+nvert*j]; // local Cartesian index
            const IntegrationPoint &ip = r_nodes.IntPoint(c2h_map[cid]);
            ip.Get(Pj.GetColumn(k), Dim);
         }
      }
   }
   for (int el = 0; el < GetNE(); el++)
   {
      Embedding &emb = CoarseFineTr.embeddings[el];
      emb.parent = el / r_elem_factor;
      emb.matrix = el % r_elem_factor;
   }

   MFEM_ASSERT(CheckElementOrientation(false) == 0, "");
   MFEM_ASSERT(CheckBdrElementOrientation(false) == 0, "");
}

void Mesh::KnotInsert(Array<KnotVector *> &kv)
{
   if (NURBSext == NULL)
   {
      mfem_error("Mesh::KnotInsert : Not a NURBS mesh!");
   }

   if (kv.Size() != NURBSext->GetNKV())
   {
      mfem_error("Mesh::KnotInsert : KnotVector array size mismatch!");
   }

   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->KnotInsert(kv);

   UpdateNURBS();
}

void Mesh::NURBSUniformRefinement()
{
   // do not check for NURBSext since this method is protected
   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->UniformRefinement();

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNURBS();
}

void Mesh::DegreeElevate(int t)
{
   if (NURBSext == NULL)
   {
      mfem_error("Mesh::DegreeElevate : Not a NURBS mesh!");
   }

   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->DegreeElevate(t);

   NURBSFECollection *nurbs_fec =
      dynamic_cast<NURBSFECollection *>(Nodes->OwnFEC());
   if (!nurbs_fec)
   {
      mfem_error("Mesh::DegreeElevate");
   }
   nurbs_fec->UpdateOrder(nurbs_fec->GetOrder() + t);

   UpdateNURBS();
}

void Mesh::UpdateNURBS()
{
   NURBSext->SetKnotsFromPatches();

   Dim = NURBSext->Dimension();
   spaceDim = Dim;

   if (NumOfElements != NURBSext->GetNE())
   {
      for (int i = 0; i < elements.Size(); i++)
      {
         FreeElement(elements[i]);
      }
      NumOfElements = NURBSext->GetNE();
      NURBSext->GetElementTopo(elements);
   }

   if (NumOfBdrElements != NURBSext->GetNBE())
   {
      for (int i = 0; i < boundary.Size(); i++)
      {
         FreeElement(boundary[i]);
      }
      NumOfBdrElements = NURBSext->GetNBE();
      NURBSext->GetBdrElementTopo(boundary);
   }

   Nodes->FESpace()->Update();
   Nodes->Update();
   NURBSext->SetCoordsFromPatches(*Nodes);

   if (NumOfVertices != NURBSext->GetNV())
   {
      NumOfVertices = NURBSext->GetNV();
      vertices.SetSize(NumOfVertices);
      int vd = Nodes->VectorDim();
      for (int i = 0; i < vd; i++)
      {
         Vector vert_val;
         Nodes->GetNodalValues(vert_val, i+1);
         for (int j = 0; j < NumOfVertices; j++)
         {
            vertices[j](i) = vert_val(j);
         }
      }
   }

   if (el_to_edge)
   {
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
      {
         GenerateFaces();
      }
   }

   if (el_to_face)
   {
      GetElementToFaceTable();
      GenerateFaces();
   }
}

void Mesh::LoadPatchTopo(std::istream &input, Array<int> &edge_to_knot)
{
   SetEmpty();

   int j;

   // Read MFEM NURBS mesh v1.0 format
   string ident;

   skip_comment_lines(input, '#');

   input >> ident; // 'dimension'
   input >> Dim;
   spaceDim = Dim;

   skip_comment_lines(input, '#');

   input >> ident; // 'elements'
   input >> NumOfElements;
   elements.SetSize(NumOfElements);
   for (j = 0; j < NumOfElements; j++)
   {
      elements[j] = ReadElement(input);
   }

   skip_comment_lines(input, '#');

   input >> ident; // 'boundary'
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (j = 0; j < NumOfBdrElements; j++)
   {
      boundary[j] = ReadElement(input);
   }

   skip_comment_lines(input, '#');

   input >> ident; // 'edges'
   input >> NumOfEdges;
   edge_vertex = new Table(NumOfEdges, 2);
   edge_to_knot.SetSize(NumOfEdges);
   for (j = 0; j < NumOfEdges; j++)
   {
      int *v = edge_vertex->GetRow(j);
      input >> edge_to_knot[j] >> v[0] >> v[1];
      if (v[0] > v[1])
      {
         edge_to_knot[j] = -1 - edge_to_knot[j];
      }
   }

   skip_comment_lines(input, '#');

   input >> ident; // 'vertices'
   input >> NumOfVertices;
   vertices.SetSize(0);

   InitBaseGeom();

   meshgen = 2;

   // generate the faces
   if (Dim > 2)
   {
      GetElementToFaceTable();
      GenerateFaces();
      if (NumOfBdrElements == 0)
      {
         GenerateBoundaryElements();
      }
      CheckBdrElementOrientation();
   }
   else
   {
      NumOfFaces = 0;
   }

   // generate edges
   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim < 3)
      {
         GenerateFaces();
         if (NumOfBdrElements == 0)
         {
            GenerateBoundaryElements();
         }
         CheckBdrElementOrientation();
      }
   }
   else
   {
      NumOfEdges = 0;
   }

   // generate the arrays 'attributes' and ' bdr_attributes'
   SetAttributes();
}

void XYZ_VectorFunction(const Vector &p, Vector &v)
{
   if (p.Size() >= v.Size())
   {
      for (int d = 0; d < v.Size(); d++)
      {
         v(d) = p(d);
      }
   }
   else
   {
      int d;
      for (d = 0; d < p.Size(); d++)
      {
         v(d) = p(d);
      }
      for ( ; d < v.Size(); d++)
      {
         v(d) = 0.0;
      }
   }
}

void Mesh::GetNodes(GridFunction &nodes) const
{
   if (Nodes == NULL || Nodes->FESpace() != nodes.FESpace())
   {
      const int newSpaceDim = nodes.FESpace()->GetVDim();
      VectorFunctionCoefficient xyz(newSpaceDim, XYZ_VectorFunction);
      nodes.ProjectCoefficient(xyz);
   }
   else
   {
      nodes = *Nodes;
   }
}

void Mesh::SetNodalFESpace(FiniteElementSpace *nfes)
{
   GridFunction *nodes = new GridFunction(nfes);
   SetNodalGridFunction(nodes, true);
}

void Mesh::SetNodalGridFunction(GridFunction *nodes, bool make_owner)
{
   GetNodes(*nodes);
   NewNodes(*nodes, make_owner);
}

const FiniteElementSpace *Mesh::GetNodalFESpace() const
{
   return ((Nodes) ? Nodes->FESpace() : NULL);
}

void Mesh::SetCurvature(int order, bool discont, int space_dim, int ordering)
{
   space_dim = (space_dim == -1) ? spaceDim : space_dim;
   FiniteElementCollection* nfec;
   if (discont)
   {
      const int type = 1; // Gauss-Lobatto points
      nfec = new L2_FECollection(order, Dim, type);
   }
   else
   {
      nfec = new H1_FECollection(order, Dim);
   }
   FiniteElementSpace* nfes = new FiniteElementSpace(this, nfec, space_dim,
                                                     ordering);
   SetNodalFESpace(nfes);
   Nodes->MakeOwner(nfec);
}

int Mesh::GetNumFaces() const
{
   switch (Dim)
   {
      case 1: return GetNV();
      case 2: return GetNEdges();
      case 3: return GetNFaces();
      case 4: return GetNFaces();
   }
   return 0;
}

#if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
static const char *fixed_or_not[] = { "fixed", "NOT FIXED" };
#endif

int Mesh::CheckElementOrientation(bool fix_it)
{
   int i, j, k, wo = 0, fo = 0, *vi = 0;
   double *v[4];

   if (Dim == 2 && spaceDim == 2)
   {
      DenseMatrix J(2, 2);

      for (i = 0; i < NumOfElements; i++)
      {
         if (Nodes == NULL)
         {
            vi = elements[i]->GetVertices();
            for (j = 0; j < 3; j++)
            {
               v[j] = vertices[vi[j]]();
            }
            for (j = 0; j < 2; j++)
               for (k = 0; k < 2; k++)
               {
                  J(j, k) = v[j+1][k] - v[0][k];
               }
         }
         else
         {
            // only check the Jacobian at the center of the element
            GetElementJacobian(i, J);
         }
         if (J.Det() < 0.0)
         {
            if (fix_it)
            {
               switch (GetElementType(i))
               {
                  case Element::TRIANGLE:
                     mfem::Swap(vi[0], vi[1]);
                     break;
                  case Element::QUADRILATERAL:
                     mfem::Swap(vi[1], vi[3]);
                     break;
               }
               fo++;
            }
            wo++;
         }
      }
   }

   if (Dim == 3)
   {
      DenseMatrix J(3, 3);

      for (i = 0; i < NumOfElements; i++)
      {
         vi = elements[i]->GetVertices();
         switch (GetElementType(i))
         {
            case Element::TETRAHEDRON:
               if (Nodes == NULL)
               {
                  for (j = 0; j < 4; j++)
                  {
                     v[j] = vertices[vi[j]]();
                  }
                  for (j = 0; j < 3; j++)
                     for (k = 0; k < 3; k++)
                     {
                        J(j, k) = v[j+1][k] - v[0][k];
                     }
               }
               else
               {
                  // only check the Jacobian at the center of the element
                  GetElementJacobian(i, J);
               }
               if (J.Det() < 0.0)
               {
                  wo++;
                  if (fix_it)
                  {
                     mfem::Swap(vi[0], vi[1]);
                     fo++;
                  }
               }
               break;

            case Element::HEXAHEDRON:
               // only check the Jacobian at the center of the element
               GetElementJacobian(i, J);
               if (J.Det() < 0.0)
               {
                  wo++;
                  if (fix_it)
                  {
                     // how?
                  }
               }
               break;
         }
      }
   }
#if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
   if (wo > 0)
      cout << "Elements with wrong orientation: " << wo << " / "
           << NumOfElements << " (" << fixed_or_not[(wo == fo) ? 0 : 1]
           << ")" << endl;
#endif
   return wo;
}

int Mesh::GetTriOrientation(const int *base, const int *test)
{
   // Static method.
   // This function computes the index 'j' of the permutation that transforms
   // test into base: test[tri_orientation[j][i]]=base[i].
   // tri_orientation = Geometry::Constants<Geometry::TRIANGLE>::Orient
   int orient;

   if (test[0] == base[0])
      if (test[1] == base[1])
      {
         orient = 0;   //  (0, 1, 2)
      }
      else
      {
         orient = 5;   //  (0, 2, 1)
      }
   else if (test[0] == base[1])
      if (test[1] == base[0])
      {
         orient = 1;   //  (1, 0, 2)
      }
      else
      {
         orient = 2;   //  (1, 2, 0)
      }
   else // test[0] == base[2]
      if (test[1] == base[0])
      {
         orient = 4;   //  (2, 0, 1)
      }
      else
      {
         orient = 3;   //  (2, 1, 0)
      }

#ifdef MFEM_DEBUG
   const int *aor = tri_t::Orient[orient];
   for (int j = 0; j < 3; j++)
      if (test[aor[j]] != base[j])
      {
         mfem_error("Mesh::GetTriOrientation(...)");
      }
#endif

   return orient;
}


int Mesh::GetQuadOrientation(const int *base, const int *test)
{
   int i;

   for (i = 0; i < 4; i++)
      if (test[i] == base[0])
      {
         break;
      }

#ifdef MFEM_DEBUG
   int orient;
   if (test[(i+1)%4] == base[1])
   {
      orient = 2*i;
   }
   else
   {
      orient = 2*i+1;
   }
   const int *aor = quad_t::Orient[orient];
   for (int j = 0; j < 4; j++)
      if (test[aor[j]] != base[j])
      {
         cerr << "Mesh::GetQuadOrientation(...)" << endl;
         cerr << " base = [";
         for (int k = 0; k < 4; k++)
         {
            cerr << " " << base[k];
         }
         cerr << " ]\n test = [";
         for (int k = 0; k < 4; k++)
         {
            cerr << " " << test[k];
         }
         cerr << " ]" << endl;
         mfem_error();
      }
#endif

   if (test[(i+1)%4] == base[1])
   {
      return 2*i;
   }

   return 2*i+1;
}

int Mesh::GetTetOrientation (const int * base, const int * test)
{
	int orient = -1;

	if(test[0] == base[0]) // 0, ...
	{
		if(test[1] == base[1]) // 0, 1, ...
		{
			if(test[2] == base[2]) orient = 0; // 0, 1, 2, 3
			else orient = 17; // 0, 1, 3, 2
		}
		else if(test[1] == base[2]) // 0, 2, ..
		{
			if(test[2] == base[1]) orient = 5;// 0, 2, 1, 3
			else orient = 6; // 0, 2, 3, 1
		}
		else if(test[1] == base[3]) // 0, 3, ..
		{
			if(test[2] == base[1]) orient = 12;// 0, 3, 1, 2
			else orient = 11; // 0, 3, 2, 1
		}
	}
	else if(test[0] == base[1]) // 1, ...
	{
		if(test[1] == base[0]) // 1, 0, ...
		{
			if(test[2] == base[2]) orient = 1; // 1, 0, 2, 3
			else orient = 16; // 1, 0, 3, 2
		}
		else if(test[1] == base[2]) // 1, 2, ..
		{
			if(test[2] == base[0]) orient = 2;// 1, 2, 0, 3
			else orient = 19; // 1, 2, 3, 0
		}
		else if(test[1] == base[3]) // 1, 3, ..
		{
			if(test[2] == base[0]) orient = 15;// 1, 3, 0, 2
			else orient = 20; // 1, 3, 2, 0
		}
	}
	else if(test[0] == base[2]) // 2, ...
	{
		if(test[1] == base[0]) // 2, 0, ...
		{
			if(test[2] == base[1]) orient = 4; // 2, 0, 1, 3
			else orient = 7; // 2, 0, 3, 1
		}
		else if(test[1] == base[1]) // 2, 1, ..
		{
			if(test[2] == base[0]) orient = 3;// 2, 1, 0, 3
			else orient = 18; // 2, 1, 3, 0
		}
		else if(test[1] == base[3]) // 2, 3, ..
		{
			if(test[2] == base[0]) orient = 8;// 2, 3, 0, 1
			else orient = 23; // 2, 3, 1, 0
		}
	}
	else if(test[0] == base[3]) // 3, ...
	{
		if(test[1] == base[0]) // 3, 0, ...
		{
			if(test[2] == base[1]) orient = 13; // 3, 0, 1, 2
			else orient = 10; // 3, 0, 2, 1
		}
		else if(test[1] == base[1]) // 3, 1, ..
		{
			if(test[2] == base[0]) orient = 14;// 3, 1, 0, 2
			else orient = 21; // 3, 1, 2, 0
		}
		else if(test[1] == base[2]) // 3, 2, ..
		{
			if(test[2] == base[0]) orient = 9;// 3, 2, 0, 1
			else orient = 22; // 3, 2, 1, 0
		}
	}

	return orient + 0;
}


int Mesh::GetHexOrientation(const int * base, const int * test)
{
	if(test[0] == base[0] && test[1] == base[1] && test[2] == base[2] && test[3] == base[3]
	                      && test[4] == base[4]
	                      && test[5] == base[5] && test[6] == base[6] && test[7] == base[7]) return 0;
	else return 1;
}


int Mesh::CheckBdrElementOrientation(bool fix_it)
{
   int i, wo = 0;

   if (Dim == 2)
   {
      for (i = 0; i < NumOfBdrElements; i++)
      {
         if (faces_info[be_to_edge[i]].Elem2No < 0) // boundary face
         {
            int *bv = boundary[i]->GetVertices();
            int *fv = faces[be_to_edge[i]]->GetVertices();
            if (bv[0] != fv[0])
            {
               if (fix_it)
               {
                  mfem::Swap<int>(bv[0], bv[1]);
               }
               wo++;
            }
         }
      }
   }

   if (Dim == 3)
   {
      int el, *bv, *ev;
      int v[4];

      for (i = 0; i < NumOfBdrElements; i++)
      {
         if (faces_info[be_to_face[i]].Elem2No < 0)
         {
            // boundary face
            bv = boundary[i]->GetVertices();
            el = faces_info[be_to_face[i]].Elem1No;
            ev = elements[el]->GetVertices();
            switch (GetElementType(el))
            {
               case Element::TETRAHEDRON:
               {
                  int *fv = faces[be_to_face[i]]->GetVertices();
                  int orientation; // orientation of the bdr. elem. w.r.t. the
                  // corresponding face element (that's the base)
                  orientation = GetTriOrientation(fv, bv);
                  if (orientation % 2)
                  {
                     // wrong orientation -- swap vertices 0 and 1 so that
                     //  we don't change the marked edge:  (0,1,2) -> (1,0,2)
                     if (fix_it)
                     {
                        mfem::Swap<int>(bv[0], bv[1]);
                        if (bel_to_edge)
                        {
                           int *be = bel_to_edge->GetRow(i);
                           mfem::Swap<int>(be[1], be[2]);
                        }
                     }
                     wo++;
                  }
               }
               break;

               case Element::HEXAHEDRON:
               {
                  int lf = faces_info[be_to_face[i]].Elem1Inf/64;
                  for (int j = 0; j < 4; j++)
                  {
                     v[j] = ev[hex_t::FaceVert[lf][j]];
                  }
                  if (GetQuadOrientation(v, bv) % 2)
                  {
                     if (fix_it)
                     {
                        mfem::Swap<int>(bv[0], bv[2]);
                        if (bel_to_edge)
                        {
                           int *be = bel_to_edge->GetRow(i);
                           mfem::Swap<int>(be[0], be[1]);
                           mfem::Swap<int>(be[2], be[3]);
                        }
                     }
                     wo++;
                  }
                  break;
               }
            }
         }
      }
   }
   // #if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
#ifdef MFEM_DEBUG
   if (wo > 0)
   {
      cout << "Boundary elements with wrong orientation: " << wo << " / "
           << NumOfBdrElements << " (" << fixed_or_not[fix_it ? 0 : 1]
           << ")" << endl;
   }
#endif
   return wo;
}

void Mesh::GetElementEdges(int i, Array<int> &edges, Array<int> &cor) const
{
   if (el_to_edge)
   {
      el_to_edge->GetRow(i, edges);
   }
   else
   {
      mfem_error("Mesh::GetElementEdges(...) element to edge table "
                 "is not generated.");
   }

   const int *v = elements[i]->GetVertices();
   const int ne = elements[i]->GetNEdges();
   cor.SetSize(ne);
   for (int j = 0; j < ne; j++)
   {
      const int *e = elements[i]->GetEdgeVertices(j);
      cor[j] = (v[e[0]] < v[e[1]]) ? (1) : (-1);
   }
}

void Mesh::GetBdrElementEdges(int i, Array<int> &edges, Array<int> &cor) const
{
   if (Dim == 2)
   {
      edges.SetSize(1);
      cor.SetSize(1);
      edges[0] = be_to_edge[i];
      const int *v = boundary[i]->GetVertices();
      cor[0] = (v[0] < v[1]) ? (1) : (-1);
   }
   else if (Dim >= 3)
   {
      if (bel_to_edge)
      {
         bel_to_edge->GetRow(i, edges);
      }
      else
      {
         mfem_error("Mesh::GetBdrElementEdges(...)");
      }

      const int *v = boundary[i]->GetVertices();
      const int ne = boundary[i]->GetNEdges();
      cor.SetSize(ne);
      for (int j = 0; j < ne; j++)
      {
         const int *e = boundary[i]->GetEdgeVertices(j);
         cor[j] = (v[e[0]] < v[e[1]]) ? (1) : (-1);
      }
   }
}

void Mesh::GetBdrElementPlanars(int i, Array<int> &pls, Array<int> &cor) const
{
	if(Dim == 4)
	{
		if (bel_to_planar)
		{
			bel_to_planar->GetRow(i, pls);
		}
		else
		{
			mfem_error("Mesh::GetBdrElementPlanars(...)");
		}

		int n = pls.Size();
		cor.SetSize(n);

		const int *v = boundary[i]->GetVertices();

		switch (boundary[i]->GetType())
		{
			case Element::TETRAHEDRON:
			{
				 cor.SetSize(4);
				 for (int j = 0; j < 4; j++)
				 {
					int* baseV = planars[pls[j]]->GetVertices();

					const int *fv = tet_t::FaceVert[j];
					int myTri[3] = { v[fv[0]], v[fv[1]], v[fv[2]] };
					cor[j] = GetTriOrientation(baseV, myTri);
				 }
				 break;
			}
			default:
			   mfem_error("Mesh::GetBdrElementPlanars(...) 2");
		}
	}
}

void Mesh::GetFaceEdges(int i, Array<int> &edges, Array<int> &o) const
{
   if (Dim == 2)
   {
      edges.SetSize(1);
      edges[0] = i;
      o.SetSize(1);
      const int *v = faces[i]->GetVertices();
      o[0] = (v[0] < v[1]) ? (1) : (-1);
   }

   if (Dim != 3 && Dim != 4)
   {
      return;
   }

   GetFaceEdgeTable(); // generate face_edge Table (if not generated)

   face_edge->GetRow(i, edges);

   const int *v = faces[i]->GetVertices();
   const int ne = faces[i]->GetNEdges();
   o.SetSize(ne);
   for (int j = 0; j < ne; j++)
   {
      const int *e = faces[i]->GetEdgeVertices(j);
      o[j] = (v[e[0]] < v[e[1]]) ? (1) : (-1);
   }
}

void Mesh::GetEdgeVertices(int i, Array<int> &vert) const
{
   // the two vertices are sorted: vert[0] < vert[1]
   // this is consistent with the global edge orientation
   // generate edge_vertex Table (if not generated)
   if (!edge_vertex) { GetEdgeVertexTable(); }
   edge_vertex->GetRow(i, vert);
}

void Mesh::GetPlanVertices(int i, Array<int> &vert) const
{
	planars[i]->GetVertices(vert);
}

Table *Mesh::GetFaceEdgeTable() const
{
   if (face_edge)
   {
      return face_edge;
   }

   if (Dim != 3 && Dim != 4)
   {
      return NULL;
   }

#ifdef MFEM_DEBUG
   if (faces.Size() != NumOfFaces)
   {
      mfem_error("Mesh::GetFaceEdgeTable : faces were not generated!");
   }
#endif

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);

   face_edge = new Table;
   GetElementArrayEdgeTable(faces, v_to_v, *face_edge);

   return (face_edge);
}

Table *Mesh::GetEdgeVertexTable() const
{
   if (edge_vertex)
   {
      return edge_vertex;
   }

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);

   int nedges = v_to_v.NumberOfEntries();
   edge_vertex = new Table(nedges, 2);
   for (int i = 0; i < NumOfVertices; i++)
   {
      for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
      {
         int j = it.Index();
         edge_vertex->Push(j, i);
         edge_vertex->Push(j, it.Column());
      }
   }
   edge_vertex->Finalize();

   return edge_vertex;
}

Table *Mesh::GetVertexToElementTable()
{
   int i, j, nv, *v;

   Table *vert_elem = new Table;

   vert_elem->MakeI(NumOfVertices);

   for (i = 0; i < NumOfElements; i++)
   {
      nv = elements[i]->GetNVertices();
      v  = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
      {
         vert_elem->AddAColumnInRow(v[j]);
      }
   }

   vert_elem->MakeJ();

   for (i = 0; i < NumOfElements; i++)
   {
      nv = elements[i]->GetNVertices();
      v  = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
      {
         vert_elem->AddConnection(v[j], i);
      }
   }

   vert_elem->ShiftUpI();

   return vert_elem;
}

Table *Mesh::GetFaceToElementTable() const
{
   Table *face_elem = new Table;

   face_elem->MakeI(faces_info.Size());

   for (int i = 0; i < faces_info.Size(); i++)
   {
      if (faces_info[i].Elem2No >= 0)
      {
         face_elem->AddColumnsInRow(i, 2);
      }
      else
      {
         face_elem->AddAColumnInRow(i);
      }
   }

   face_elem->MakeJ();

   for (int i = 0; i < faces_info.Size(); i++)
   {
      face_elem->AddConnection(i, faces_info[i].Elem1No);
      if (faces_info[i].Elem2No >= 0)
      {
         face_elem->AddConnection(i, faces_info[i].Elem2No);
      }
   }

   face_elem->ShiftUpI();

   return face_elem;
}

void Mesh::GetElementFaces(int i, Array<int> &fcs, Array<int> &cor)
const
{
   int n, j;

   if (el_to_face)
   {
      el_to_face->GetRow(i, fcs);
   }
   else
   {
      mfem_error("Mesh::GetElementFaces(...) : el_to_face not generated.");
   }

   n = fcs.Size();
   cor.SetSize(n);
   for (j = 0; j < n; j++)
      if (faces_info[fcs[j]].Elem1No == i)
      {
         cor[j] = faces_info[fcs[j]].Elem1Inf % 64;
      }
#ifdef MFEM_DEBUG
      else if (faces_info[fcs[j]].Elem2No == i)
      {
         cor[j] = faces_info[fcs[j]].Elem2Inf % 64;
      }
      else
      {
         mfem_error("Mesh::GetElementFaces(...) : 2");
      }
#else
      else
      {
         cor[j] = faces_info[fcs[j]].Elem2Inf % 64;
      }
#endif
}

void Mesh::GetBdrElementFace(int i, int *f, int *o) const
{
   const int *bv, *fv;

   *f = be_to_face[i];
   bv = boundary[i]->GetVertices();
   fv = faces[be_to_face[i]]->GetVertices();

   // find the orientation of the bdr. elem. w.r.t.
   // the corresponding face element (that's the base)
   switch (GetBdrElementType(i))
   {
      case Element::TRIANGLE:
         *o = GetTriOrientation(fv, bv);
         break;
      case Element::QUADRILATERAL:
         *o = GetQuadOrientation(fv, bv);
         break;
      case Element::TETRAHEDRON:
   	  *o = GetTetOrientation(fv, bv);
         break;
      case Element::HEXAHEDRON:
   	  *o = GetHexOrientation(fv, bv);
         break;
      default:
         mfem_error("Mesh::GetBdrElementFace(...) 2");
   }
}

void Mesh::GetElementPlanars(int i, Array<int> &pls, Array<int> &cor)
const
{
   int n;

   if (el_to_planar)
   {
      el_to_planar->GetRow(i, pls);
   }
   else
   {
      mfem_error("Mesh::GetElementPlanars(...) : el_to_planar not generated.");
   }

   n = pls.Size();
   cor.SetSize(n);

   const int *v = elements[i]->GetVertices();
   const int npls = elements[i]->GetNPlanars();

   cor.SetSize(npls);
   for (int j = 0; j < npls; j++)
   {
      const int *pl = elements[i]->GetPlanarsVertices(j);

      int* baseV = planars[pls[j]]->GetVertices();

      switch (planars[pls[j]]->GetType())
         {
            case Element::TRIANGLE:
            {
               int myTri[3] = { v[pl[0]], v[pl[1]], v[pl[2]] };
               cor[j] = GetTriOrientation(baseV, myTri);
               break;
            }
            case Element::QUADRILATERAL:
            {
                int myQuad[4] = { v[pl[0]], v[pl[1]], v[pl[2]], v[pl[3]] };
                cor[j] = GetQuadOrientation(baseV, myQuad);
                break;
            }
            default:
               mfem_error("Mesh::GetElementPlanars(...) 2");
         }
   }
}

int Mesh::GetFaceBaseGeometry(int i) const
{
   // Here, we assume all faces are of the same type
   switch (GetElementType(0))
   {
      case Element::SEGMENT:
         return Geometry::POINT;

      case Element::TRIANGLE:
      case Element::QUADRILATERAL:
         return Geometry::SEGMENT; // in 2D 'face' is an edge

      case Element::TETRAHEDRON:
         return Geometry::TRIANGLE;
      case Element::HEXAHEDRON:
         return Geometry::SQUARE;
      case Element::PENTATOPE:
            return Geometry::TETRAHEDRON;
      case Element::TESSERACT:
    	  	return Geometry::CUBE;
      default:
         mfem_error("Mesh::GetFaceBaseGeometry(...) #1");
   }
   return (-1);
}

int Mesh::GetPlanarBaseGeometry(int i) const
{
   // Here, we assume all planars are of the same type
   switch (GetElementType(0))
   {
      case Element::PENTATOPE:
            return Geometry::TRIANGLE;
      case Element::TESSERACT:
    	  	return Geometry::SQUARE;
      default:
         mfem_error("Mesh::GetPlanarBaseGeometry(...) #1");
   }
   return (-1);
}

int Mesh::GetBdrPlanarBaseGeometry(int i) const
{
   // Here, we assume all planars are of the same type
   switch (GetBdrElementType(0))
   {
	  case Element::TETRAHEDRON:
			return Geometry::TRIANGLE;
	  case Element::HEXAHEDRON:
			return Geometry::SQUARE;
	  default:
		 mfem_error("Mesh::GetBdrPlanarBaseGeometry(...) #1");
   }
   return (-1);
}

int Mesh::GetBdrElementEdgeIndex(int i) const
{
   switch (Dim)
   {
      case 1: return boundary[i]->GetVertices()[0];
      case 2: return be_to_edge[i];
      case 3: return be_to_face[i];
      case 4: // 4D case, "give it a try", seems to crush parelag::...:: generateTopology()
	      return be_to_face[i]; 
      default: mfem_error("Mesh::GetBdrElementEdgeIndex: invalid dimension!");
   }
   return -1;
}

void Mesh::GetBdrElementAdjacentElement(int bdr_el, int &el, int &info) const
{
   int fid = GetBdrElementEdgeIndex(bdr_el);
   const FaceInfo &fi = faces_info[fid];
   MFEM_ASSERT(fi.Elem1Inf%64 == 0, "internal error"); // orientation == 0
   const int *fv = (Dim > 1) ? faces[fid]->GetVertices() : NULL;
   const int *bv = boundary[bdr_el]->GetVertices();
   int ori;
   switch (GetBdrElementBaseGeometry(bdr_el))
   {
      case Geometry::POINT:    ori = 0; break;
      case Geometry::SEGMENT:  ori = (fv[0] == bv[0]) ? 0 : 1; break;
      case Geometry::TRIANGLE: ori = GetTriOrientation(fv, bv); break;
      case Geometry::SQUARE:   ori = GetQuadOrientation(fv, bv); break;
      default: MFEM_ABORT("boundary element type not implemented"); ori = 0;
   }
   el   = fi.Elem1No;
   info = fi.Elem1Inf + ori;
}

int Mesh::GetElementType(int i) const
{
   return elements[i]->GetType();
}

int Mesh::GetBdrElementType(int i) const
{
   return boundary[i]->GetType();
}

void Mesh::GetPointMatrix(int i, DenseMatrix &pointmat) const
{
   int k, j, nv;
   const int *v;

   v  = elements[i]->GetVertices();
   nv = elements[i]->GetNVertices();

   pointmat.SetSize(spaceDim, nv);
   for (k = 0; k < spaceDim; k++)
      for (j = 0; j < nv; j++)
      {
         pointmat(k, j) = vertices[v[j]](k);
      }
}

void Mesh::GetBdrPointMatrix(int i,DenseMatrix &pointmat) const
{
   int k, j, nv;
   const int *v;

   v  = boundary[i]->GetVertices();
   nv = boundary[i]->GetNVertices();

   pointmat.SetSize(spaceDim, nv);
   for (k = 0; k < spaceDim; k++)
      for (j = 0; j < nv; j++)
      {
         pointmat(k, j) = vertices[v[j]](k);
      }
}

double Mesh::GetLength(int i, int j) const
{
   const double *vi = vertices[i]();
   const double *vj = vertices[j]();
   double length = 0.;

   for (int k = 0; k < spaceDim; k++)
   {
      length += (vi[k]-vj[k])*(vi[k]-vj[k]);
   }

   return sqrt(length);
}

// static method
void Mesh::GetElementArrayEdgeTable(const Array<Element*> &elem_array,
                                    const DSTable &v_to_v, Table &el_to_edge)
{
   el_to_edge.MakeI(elem_array.Size());
   for (int i = 0; i < elem_array.Size(); i++)
   {
      el_to_edge.AddColumnsInRow(i, elem_array[i]->GetNEdges());
   }
   el_to_edge.MakeJ();
   for (int i = 0; i < elem_array.Size(); i++)
   {
      const int *v = elem_array[i]->GetVertices();
      const int ne = elem_array[i]->GetNEdges();
      for (int j = 0; j < ne; j++)
      {
         const int *e = elem_array[i]->GetEdgeVertices(j);
         el_to_edge.AddConnection(i, v_to_v(v[e[0]], v[e[1]]));
      }
   }
   el_to_edge.ShiftUpI();
}

void Mesh::GetVertexToVertexTable(DSTable &v_to_v) const
{
   if (edge_vertex)
   {
      for (int i = 0; i < edge_vertex->Size(); i++)
      {
         const int *v = edge_vertex->GetRow(i);
         v_to_v.Push(v[0], v[1]);
      }
   }
   else
   {
      for (int i = 0; i < NumOfElements; i++)
      {
         const int *v = elements[i]->GetVertices();
         const int ne = elements[i]->GetNEdges();
         for (int j = 0; j < ne; j++)
         {
            const int *e = elements[i]->GetEdgeVertices(j);
            v_to_v.Push(v[e[0]], v[e[1]]);
         }
      }
   }
}

int Mesh::GetElementToEdgeTable(Table & e_to_f, Array<int> &be_to_f)
{
   int i, NumberOfEdges;

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);

   NumberOfEdges = v_to_v.NumberOfEntries();

   // Fill the element to edge table
   GetElementArrayEdgeTable(elements, v_to_v, e_to_f);

   if (Dim == 2)
   {
      // Initialize the indices for the boundary elements.
      be_to_f.SetSize(NumOfBdrElements);
      for (i = 0; i < NumOfBdrElements; i++)
      {
         const int *v = boundary[i]->GetVertices();
         be_to_f[i] = v_to_v(v[0], v[1]);
      }
   }
   else if (Dim == 3 || Dim == 4)
   {
      if (bel_to_edge == NULL)
      {
         bel_to_edge = new Table;
      }
      GetElementArrayEdgeTable(boundary, v_to_v, *bel_to_edge);
   }
   else
   {
      mfem_error("1D GetElementToEdgeTable is not yet implemented.");
   }

   // Return the number of edges
   return NumberOfEdges;
}

const Table & Mesh::ElementToElementTable()
{
   if (el_to_el)
   {
      return *el_to_el;
   }

   int num_faces = GetNumFaces();
   // Note that, for ParNCMeshes, faces_info will contain also the ghost faces
   MFEM_ASSERT(faces_info.Size() >= num_faces, "faces were not generated!");

   Array<Connection> conn;
   conn.Reserve(2*faces_info.Size());

   for (int i = 0; i < faces_info.Size(); i++)
   {
      const FaceInfo &fi = faces_info[i];
      if (fi.Elem2No >= 0)
      {
         conn.Append(Connection(fi.Elem1No, fi.Elem2No));
         conn.Append(Connection(fi.Elem2No, fi.Elem1No));
      }
      else if (fi.Elem2Inf >= 0)
      {
         int nbr_elem_idx = NumOfElements - 1 - fi.Elem2No;
         conn.Append(Connection(fi.Elem1No, nbr_elem_idx));
         conn.Append(Connection(nbr_elem_idx, fi.Elem1No));
      }
   }

   conn.Sort();
   conn.Unique();
   el_to_el = new Table(NumOfElements, conn);

   return *el_to_el;
}

const Table & Mesh::ElementToFaceTable() const
{
   if (el_to_face == NULL)
   {
      mfem_error("Mesh::ElementToFaceTable()");
   }
   return *el_to_face;
}

const Table & Mesh::ElementToPlanTable() const
{
   if (el_to_planar == NULL)
   {
	  mfem_error("Mesh::ElementToPlanarTable()");
   }
   return *el_to_planar;
}

const Table & Mesh::ElementToEdgeTable() const
{
   if (el_to_edge == NULL)
   {
      mfem_error("Mesh::ElementToEdgeTable()");
   }
   return *el_to_edge;
}

void Mesh::AddPointFaceElement(int lf, int gf, int el)
{
   if (faces_info[gf].Elem1No == -1)  // this will be elem1
   {
      // faces[gf] = new Point(&gf);
      faces_info[gf].Elem1No  = el;
      faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
      faces_info[gf].Elem2No  = -1; // in case there's no other side
      faces_info[gf].Elem2Inf = -1; // face is not shared
   }
   else  //  this will be elem2
   {
      faces_info[gf].Elem2No  = el;
      faces_info[gf].Elem2Inf = 64 * lf + 1;
   }
}

void Mesh::AddSegmentFaceElement(int lf, int gf, int el, int v0, int v1)
{
   if (faces[gf] == NULL)  // this will be elem1
   {
      faces[gf] = new Segment(v0, v1);
      faces_info[gf].Elem1No  = el;
      faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
      faces_info[gf].Elem2No  = -1; // in case there's no other side
      faces_info[gf].Elem2Inf = -1; // face is not shared
   }
   else  //  this will be elem2
   {
      int *v = faces[gf]->GetVertices();
      faces_info[gf].Elem2No  = el;
      if ( v[1] == v0 && v[0] == v1 )
      {
         faces_info[gf].Elem2Inf = 64 * lf + 1;
      }
      else if ( v[0] == v0 && v[1] == v1 )
      {
         faces_info[gf].Elem2Inf = 64 * lf;
      }
      else
      {
         MFEM_ASSERT((v[1] == v0 && v[0] == v1)||
                     (v[0] == v0 && v[1] == v1), "");
      }
   }
}

void Mesh::AddTriangleFaceElement(int lf, int gf, int el,
                                  int v0, int v1, int v2)
{
   if (faces[gf] == NULL)  // this will be elem1
   {
      faces[gf] = new Triangle(v0, v1, v2);
      faces_info[gf].Elem1No  = el;
      faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
      faces_info[gf].Elem2No  = -1; // in case there's no other side
      faces_info[gf].Elem2Inf = -1; // face is not shared
   }
   else  //  this will be elem2
   {
      int orientation, vv[3] = { v0, v1, v2 };
      orientation = GetTriOrientation(faces[gf]->GetVertices(), vv);
      MFEM_ASSERT(orientation % 2 != 0, "");
      faces_info[gf].Elem2No  = el;
      faces_info[gf].Elem2Inf = 64 * lf + orientation;
   }
}

void Mesh::AddQuadFaceElement(int lf, int gf, int el,
                              int v0, int v1, int v2, int v3)
{
   if (faces_info[gf].Elem1No < 0)  // this will be elem1
   {
      faces[gf] = new Quadrilateral(v0, v1, v2, v3);
      faces_info[gf].Elem1No  = el;
      faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
      faces_info[gf].Elem2No  = -1; // in case there's no other side
      faces_info[gf].Elem2Inf = -1; // face is not shared
   }
   else  //  this will be elem2
   {
      int vv[4] = { v0, v1, v2, v3 };
      int oo = GetQuadOrientation(faces[gf]->GetVertices(), vv);
      MFEM_ASSERT(oo % 2 != 0, "");
      faces_info[gf].Elem2No  = el;
      faces_info[gf].Elem2Inf = 64 * lf + oo;
   }
}

void Mesh::AddTetrahedralFaceElement(int lf, int gf, int el,
                             int v0, int v1, int v2, int v3)
{
   if (faces[gf] == NULL)  // this will be elem1
   {
//	  ElementTransformation *eltransf = GetElementTransformation(el);
//	  double w = eltransf->SignedWeight();
//	  int oEl = 0; if(w < 0.0) oEl = 1;
//	  if(oEl==1) cout << "negative weight!" << endl;

	  faces[gf] = new Tetrahedron(v0, v1, v2, v3);
	  faces_info[gf].Elem1No  = el;
//	  faces_info[gf].Elem1Inf = 64 * lf+ lf%2 + oEl;
	  faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
	  faces_info[gf].Elem2No  = -1; // in case there's no other side
	  faces_info[gf].Elem2Inf = -1; // face is not shared
   }
   else  //  this will be elem2
   {
	  int orientation, vv[4] = { v0, v1, v2, v3 };
//	  orientation = GetTetOrientation(faces[gf]->GetVertices(), vv) + (faces_info[gf].Elem1Inf)%64;
	  orientation = GetTetOrientation(faces[gf]->GetVertices(), vv);

	  faces_info[gf].Elem2No  = el;
	  faces_info[gf].Elem2Inf = 64 * lf + orientation;
   }
}

void Mesh::AddHexahedralFaceElement(int lf, int gf, int el,
                             int v0, int v1, int v2, int v3,
                             int v4, int v5, int v6, int v7)
{
   if (faces[gf] == NULL)  // this will be elem1
   {
	  faces[gf] = new Hexahedron(v0, v1, v2, v3, v4, v5, v6, v7);
	  faces_info[gf].Elem1No  = el;
	  faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
	  faces_info[gf].Elem2No  = -1; // in case there's no other side
	  faces_info[gf].Elem2Inf = -1; // face is not shared
   }
   else  //  this will be elem2
   {
	  int orientation, vv[8] = { v0, v1, v2, v3, v4, v5, v6, v7 };
	  orientation =GetHexOrientation(faces[gf]->GetVertices(), vv);

	  faces_info[gf].Elem2No  = el;
	  faces_info[gf].Elem2Inf = 64 * lf + orientation;
   }
}


void Mesh::GenerateFaces()
{
   int i, nfaces = GetNumFaces();

   for (i = 0; i < faces.Size(); i++)
   {
      FreeElement(faces[i]);
   }

   swappedFaces.SetSize(nfaces);

   // (re)generate the interior faces and the info for them
   faces.SetSize(nfaces);
   faces_info.SetSize(nfaces);
   for (i = 0; i < nfaces; i++)
   {
      faces[i] = NULL;
      faces_info[i].Elem1No = -1;
      faces_info[i].NCFace = -1;
   }
   for (i = 0; i < NumOfElements; i++)
   {
      const int *v = elements[i]->GetVertices();
      const int *ef;
      if (Dim == 1)
      {
         AddPointFaceElement(0, v[0], i);
         AddPointFaceElement(1, v[1], i);
      }
      else if (Dim == 2)
      {
         ef = el_to_edge->GetRow(i);
         const int ne = elements[i]->GetNEdges();
         for (int j = 0; j < ne; j++)
         {
            const int *e = elements[i]->GetEdgeVertices(j);
            AddSegmentFaceElement(j, ef[j], i, v[e[0]], v[e[1]]);
         }
      }
      else if (Dim == 3)
      {
         ef = el_to_face->GetRow(i);
         switch (GetElementType(i))
         {
            case Element::TETRAHEDRON:
            {
               for (int j = 0; j < 4; j++)
               {
                  const int *fv = tet_t::FaceVert[j];
                  AddTriangleFaceElement(j, ef[j], i,
                                         v[fv[0]], v[fv[1]], v[fv[2]]);
               }
               break;
            }
            case Element::HEXAHEDRON:
            {
               for (int j = 0; j < 6; j++)
               {
                  const int *fv = hex_t::FaceVert[j];
                  AddQuadFaceElement(j, ef[j], i,
                                     v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
               }
               break;
            }
            default:
               MFEM_ABORT("Unexpected type of Element.");
         }
      }
      else if (Dim == 4)
      {
          ef = el_to_face->GetRow(i);
		  switch (GetElementType(i))
		  {
		    case Element::PENTATOPE:
		    {
		      bool swapped = swappedElements[i];
		      int tempv[5]; for(int j=0; j<5; j++) tempv[j] = v[j];
		      if(swapped) Swap(tempv);

		      int filter[5] = {0,1,2,3,4};
		      if(swapped)
		      {
				filter[3] = 4;
				filter[4] = 3;
		      }

		      for (int j = 0; j < 5; j++)
		      {
		    	bool swapFace = false;
		    	if((swapped && j%2==0) || (!swapped && j%2==1)) swapFace = true;

		    	if(faces[ef[filter[j]]]==NULL) swappedFaces[ef[filter[j]]] = swapFace;

		    	const int *fv = pent_t::FaceVert[j];
		    	if(swapFace)
		    	{
			      AddTetrahedralFaceElement(j, ef[filter[j]], i,
									 tempv[fv[1]], tempv[fv[0]], tempv[fv[2]], tempv[fv[3]]);
		    	}
		    	else
		    	{
			      AddTetrahedralFaceElement(j, ef[filter[j]], i,
					   tempv[fv[0]], tempv[fv[1]], tempv[fv[2]], tempv[fv[3]]);
		    	}

		      }

		      break;
		    }
		    case Element::TESSERACT:
		    for (int j = 0; j < 8; j++)
		    {
		      const int *fv = tess_t::FaceVert[j];
		      AddHexahedralFaceElement(j, ef[j], i,
									 v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]],
									v[fv[4]], v[fv[5]], v[fv[6]], v[fv[7]]);
		    }
		    break;
			#ifdef MFEM_DEBUG
			default:
			 MFEM_ABORT("Unexpected type of Element.");
			#endif
		  }
      }
   }
}

void Mesh::GeneratePlanars()
{
	for(int i = 0; i < planars.Size(); i++) FreeElement(planars[i]);

   // (re)generate the interior faces and the info for them
   planars.SetSize(NumOfPlanars);
   for(int i = 0; i < NumOfPlanars; i++) planars[i] = NULL;

   const int *fv;

   for(int i = 0; i < NumOfElements; i++)
   {
	  const int *v = elements[i]->GetVertices();
	  const int *ef;

	  ef = el_to_planar->GetRow(i);
	  if(GetElementType(i)==Element::PENTATOPE)
	   {
		  for (int j = 0; j < 10; j++)
		  {
			  if (planars[ef[j]] == NULL)
			  {
				 fv = pent_t::PlanarVert[j];
				 planars[ef[j]] = new Triangle(v[fv[0]], v[fv[1]], v[fv[2]]);
			  }
		  }
	   }
   }
}

void Mesh::GenerateNCFaceInfo()
{
   MFEM_VERIFY(ncmesh, "missing NCMesh.");

   for (int i = 0; i < faces_info.Size(); i++)
   {
      faces_info[i].NCFace = -1;
   }

   const NCMesh::NCList &list =
      (Dim == 2) ? ncmesh->GetEdgeList() : ncmesh->GetFaceList();

   nc_faces_info.SetSize(0);
   nc_faces_info.Reserve(list.masters.size() + list.slaves.size());

   int nfaces = GetNumFaces();

   // add records for master faces
   for (unsigned i = 0; i < list.masters.size(); i++)
   {
      const NCMesh::Master &master = list.masters[i];
      if (master.index >= nfaces) { continue; }

      faces_info[master.index].NCFace = nc_faces_info.Size();
      nc_faces_info.Append(NCFaceInfo(false, master.local, NULL));
      // NOTE: one of the unused members stores local face no. to be used below
   }

   // add records for slave faces
   for (unsigned i = 0; i < list.slaves.size(); i++)
   {
      const NCMesh::Slave &slave = list.slaves[i];
      if (slave.index >= nfaces || slave.master >= nfaces) { continue; }

      FaceInfo &slave_fi = faces_info[slave.index];
      FaceInfo &master_fi = faces_info[slave.master];
      NCFaceInfo &master_nc = nc_faces_info[master_fi.NCFace];

      slave_fi.NCFace = nc_faces_info.Size();
      nc_faces_info.Append(NCFaceInfo(true, slave.master, &slave.point_matrix));

      slave_fi.Elem2No = master_fi.Elem1No;
      slave_fi.Elem2Inf = 64 * master_nc.MasterFace; // get lf no. stored above
      // NOTE: orientation part of Elem2Inf is encoded in the point matrix
   }
}

STable3D *Mesh::GetFacesTable()
{
   STable3D *faces_tbl = new STable3D(NumOfVertices);
   for (int i = 0; i < NumOfElements; i++)
   {
      const int *v = elements[i]->GetVertices();
      switch (GetElementType(i))
      {
         case Element::TETRAHEDRON:
         {
            for (int j = 0; j < 4; j++)
            {
               const int *fv = tet_t::FaceVert[j];
               faces_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]);
            }
            break;
         }
         case Element::HEXAHEDRON:
         {
            // find the face by the vertices with the smallest 3 numbers
            // z = 0, y = 0, x = 1, y = 1, x = 0, z = 1
            for (int j = 0; j < 6; j++)
            {
               const int *fv = hex_t::FaceVert[j];
               faces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
            }
            break;
         }
         default:
            MFEM_ABORT("Unexpected type of Element.");
      }
   }
   return faces_tbl;
}

STable3D *Mesh::GetElementToFaceTable(int ret_ftbl)
{
   int i, *v;
   STable3D *faces_tbl;

   if (el_to_face != NULL)
   {
      delete el_to_face;
   }
   el_to_face = new Table(NumOfElements, 6);  // must be 6 for hexahedra
   faces_tbl = new STable3D(NumOfVertices);
   for (i = 0; i < NumOfElements; i++)
   {
      v = elements[i]->GetVertices();
      switch (GetElementType(i))
      {
         case Element::TETRAHEDRON:
         {
            for (int j = 0; j < 4; j++)
            {
               const int *fv = tet_t::FaceVert[j];
               el_to_face->Push(
                  i, faces_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]));
            }
            break;
         }
         case Element::HEXAHEDRON:
         {
            // find the face by the vertices with the smallest 3 numbers
            // z = 0, y = 0, x = 1, y = 1, x = 0, z = 1
            for (int j = 0; j < 6; j++)
            {
               const int *fv = hex_t::FaceVert[j];
               el_to_face->Push(
                  i, faces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]));
            }
            break;
         }
         default:
            MFEM_ABORT("Unexpected type of Element.");
      }
   }
   el_to_face->Finalize();
   NumOfFaces = faces_tbl->NumberOfElements();
   be_to_face.SetSize(NumOfBdrElements);
   for (i = 0; i < NumOfBdrElements; i++)
   {
      v = boundary[i]->GetVertices();
      switch (GetBdrElementType(i))
      {
         case Element::TRIANGLE:
         {
            be_to_face[i] = (*faces_tbl)(v[0], v[1], v[2]);
            break;
         }
         case Element::QUADRILATERAL:
         {
            be_to_face[i] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
            break;
         }
         default:
            MFEM_ABORT("Unexpected type of boundary Element.");
      }
   }

   if (ret_ftbl)
   {
      return faces_tbl;
   }
   delete faces_tbl;
   return NULL;
}

STable4D * Mesh::GetElementToFaceTable4D(int ret_ftbl)
{
   int i, *v;
   STable4D *faces_tbl;

   if(el_to_face != NULL) delete el_to_face;
   el_to_face = new Table(NumOfElements, 5);  // 5 faces for one pentatope
   faces_tbl = new STable4D(NumOfVertices);
   for (i = 0; i < NumOfElements; i++)
   {
	  v = elements[i]->GetVertices();

      //bool swapped = swappedElements[i];
	  int tempv[5]; for(int j=0; j<5; j++) tempv[j] = v[j];
      //if(swapped) Swap(tempv);

	  switch (GetElementType(i))
	  {
	  case Element::PENTATOPE:
		 for (int j = 0; j < 5; j++)
		 {
			const int *fv = pent_t::FaceVert[j];
			el_to_face->Push(
			   i, faces_tbl->Push(tempv[fv[0]], tempv[fv[1]], tempv[fv[2]], tempv[fv[3]]));
		 }
		 break;
#ifdef MFEM_DEBUG
	  default:
		 MFEM_ABORT("Unexpected type of Element.");
#endif
	  }
   }
   el_to_face->Finalize();
   NumOfFaces = faces_tbl->NumberOfElements();

//   cout << "num faces: " << NumOfFaces << endl << endl;

   be_to_face.SetSize(NumOfBdrElements);
   for (i = 0; i < NumOfBdrElements; i++)
   {
	  v = boundary[i]->GetVertices();
	  switch (GetBdrElementType(i))
	  {
	  case Element::TETRAHEDRON:
	  {
		 be_to_face[i] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
	  }
		 break;
#ifdef MFEM_DEBUG
	  default:
		 MFEM_ABORT("Unexpected type of boundary Element.");
#endif
	  }
   }

   if(ret_ftbl) return faces_tbl;
   delete faces_tbl;
   return NULL;
}

STable3D * Mesh::GetElementToPlanarTable(int ret_trigtbl)
{
   int i, *v;
   STable3D *trig_tbl;

   if(el_to_planar != NULL) delete el_to_planar;
   el_to_planar = new Table(NumOfElements, 24);  // 24 planars at most for a tesseract (pentatope only 10)
   trig_tbl = new STable3D(NumOfVertices);
   for (i = 0; i < NumOfElements; i++)
   {
	  v = elements[i]->GetVertices();
	  switch (GetElementType(i))
	  {
	   case Element::PENTATOPE:
		 for (int j = 0; j < 10; j++)
		 {
			const int *fv = pent_t::PlanarVert[j];
			el_to_planar->Push(i, trig_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]));
		 }
		 break;
#ifdef MFEM_DEBUG
	  default:
		 MFEM_ABORT("Unexpected type of Element.");
#endif
	  }
   }
   el_to_planar->Finalize();
   NumOfPlanars = trig_tbl->NumberOfElements();

   bel_to_planar = new Table(NumOfBdrElements, 6);  // 6 planars at most for cube
   for (i = 0; i < NumOfBdrElements; i++)
   {
	  v = boundary[i]->GetVertices();
	  switch (GetBdrElementType(i))
	  {
		  case Element::TETRAHEDRON:
			 for (int j = 0; j < 4; j++)
			 {
				const int *fv = tet_t::FaceVert[j];
				bel_to_planar->Push(i, (*trig_tbl)(v[fv[0]], v[fv[1]], v[fv[2]]));
			 }
			 break;
#ifdef MFEM_DEBUG
	  default:
		 MFEM_ABORT("Unexpected type of boundary Element.");
#endif
	  }
   }
   bel_to_planar->Finalize();

   if(ret_trigtbl) return trig_tbl;
   delete trig_tbl;
   return NULL;
}

void Mesh::ReorientTetMesh()
{
   int *v;

   if (Dim != 3 || !(meshgen & 1))
   {
      return;
   }

   DSTable *old_v_to_v = NULL;
   Table *old_elem_vert = NULL;

   if (Nodes)
   {
      PrepareNodeReorder(&old_v_to_v, &old_elem_vert);
   }

   for (int i = 0; i < NumOfElements; i++)
   {
      if (GetElementType(i) == Element::TETRAHEDRON)
      {
         v = elements[i]->GetVertices();

         Rotate3(v[0], v[1], v[2]);
         if (v[0] < v[3])
         {
            Rotate3(v[1], v[2], v[3]);
         }
         else
         {
            ShiftL2R(v[0], v[1], v[3]);
         }
      }
   }

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      if (GetBdrElementType(i) == Element::TRIANGLE)
      {
         v = boundary[i]->GetVertices();

         Rotate3(v[0], v[1], v[2]);
      }
   }

   if (!Nodes)
   {
      GetElementToFaceTable();
      GenerateFaces();
      if (el_to_edge)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }
   }
   else
   {
      DoNodeReorder(old_v_to_v, old_elem_vert);
      delete old_elem_vert;
      delete old_v_to_v;
   }
}

void Mesh::ReplaceBoundaryFromFaces()
{
	for (int i = 0; i < NumOfBdrElements; i++)
	{
		int faceID = be_to_face[i];
		int* vBnd = boundary[i]->GetVertices();
		int* vFce = faces[faceID]->GetVertices();

		int NVertices = boundary[i]->GetNVertices();

//		for(int k=0; k<NVertices; k++) cout << vBnd[k] << " "; cout << endl;
//		for(int k=0; k<NVertices; k++) cout << vFce[k] << " "; cout << endl << endl;

		for(int k=0; k<NVertices; k++) vBnd[k] = vFce[k];
	}
}

#ifdef MFEM_USE_MPI
#ifndef MFEM_USE_METIS_5
// METIS 4 prototypes
typedef int idxtype;
extern "C" {
   void METIS_PartGraphRecursive(int*, idxtype*, idxtype*, idxtype*, idxtype*,
                                 int*, int*, int*, int*, int*, idxtype*);
   void METIS_PartGraphKway(int*, idxtype*, idxtype*, idxtype*, idxtype*,
                            int*, int*, int*, int*, int*, idxtype*);
   void METIS_PartGraphVKway(int*, idxtype*, idxtype*, idxtype*, idxtype*,
                             int*, int*, int*, int*, int*, idxtype*);
}
#else
#include "metis.h"
#endif
#endif

int *Mesh::CartesianPartitioning(int nxyz[])
{
   int *partitioning;
   double pmin[3] = { numeric_limits<double>::infinity(),
                      numeric_limits<double>::infinity(),
                      numeric_limits<double>::infinity()
                    };
   double pmax[3] = { -numeric_limits<double>::infinity(),
                      -numeric_limits<double>::infinity(),
                      -numeric_limits<double>::infinity()
                    };
   // find a bounding box using the vertices
   for (int vi = 0; vi < NumOfVertices; vi++)
   {
      const double *p = vertices[vi]();
      for (int i = 0; i < spaceDim; i++)
      {
         if (p[i] < pmin[i]) { pmin[i] = p[i]; }
         if (p[i] > pmax[i]) { pmax[i] = p[i]; }
      }
   }

   partitioning = new int[NumOfElements];

   // determine the partitioning using the centers of the elements
   double ppt[3];
   Vector pt(ppt, spaceDim);
   for (int el = 0; el < NumOfElements; el++)
   {
      GetElementTransformation(el)->Transform(
         Geometries.GetCenter(GetElementBaseGeometry(el)), pt);
      int part = 0;
      for (int i = spaceDim-1; i >= 0; i--)
      {
         int idx = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
         if (idx < 0) { idx = 0; }
         if (idx >= nxyz[i]) { idx = nxyz[i]-1; }
         part = part * nxyz[i] + idx;
      }
      partitioning[el] = part;
   }

   return partitioning;
}

int *Mesh::GeneratePartitioning(int nparts, int part_method)
{
#ifdef MFEM_USE_MPI
   int i, *partitioning;

   ElementToElementTable();

   partitioning = new int[NumOfElements];

   if (nparts == 1)
   {
      for (i = 0; i < NumOfElements; i++)
      {
         partitioning[i] = 0;
      }
   }
   else
   {
      int *I, *J, n;
#ifndef MFEM_USE_METIS_5
      int wgtflag = 0;
      int numflag = 0;
      int options[5];
#else
      int ncon = 1;
      int err;
      int options[40];
#endif
      int edgecut;

      n = NumOfElements;
      I = el_to_el->GetI();
      J = el_to_el->GetJ();
#ifndef MFEM_USE_METIS_5
      options[0] = 0;
#else
      METIS_SetDefaultOptions(options);
      options[METIS_OPTION_CONTIG] = 1; // set METIS_OPTION_CONTIG
#endif

      // Sort the neighbor lists
      if (part_method >= 0 && part_method <= 2)
      {
         for (i = 0; i < n; i++)
         {
            // Sort in increasing order.
            // std::sort(J+I[i], J+I[i+1]);

            // Sort in decreasing order, as in previous versions of MFEM.
            std::sort(J+I[i], J+I[i+1], std::greater<int>());
         }
      }

      // This function should be used to partition a graph into a small
      // number of partitions (less than 8).
      if (part_method == 0 || part_method == 3)
      {
#ifndef MFEM_USE_METIS_5
         METIS_PartGraphRecursive(&n,
                                  (idxtype *) I,
                                  (idxtype *) J,
                                  (idxtype *) NULL,
                                  (idxtype *) NULL,
                                  &wgtflag,
                                  &numflag,
                                  &nparts,
                                  options,
                                  &edgecut,
                                  (idxtype *) partitioning);
#else
         err = METIS_PartGraphRecursive(&n,
                                        &ncon,
                                        I,
                                        J,
                                        (idx_t *) NULL,
                                        (idx_t *) NULL,
                                        (idx_t *) NULL,
                                        &nparts,
                                        (real_t *) NULL,
                                        (real_t *) NULL,
                                        options,
                                        &edgecut,
                                        partitioning);
         if (err != 1)
            mfem_error("Mesh::GeneratePartitioning: "
                       " error in METIS_PartGraphRecursive!");
#endif
      }

      // This function should be used to partition a graph into a large
      // number of partitions (greater than 8).
      if (part_method == 1 || part_method == 4)
      {
#ifndef MFEM_USE_METIS_5
         METIS_PartGraphKway(&n,
                             (idxtype *) I,
                             (idxtype *) J,
                             (idxtype *) NULL,
                             (idxtype *) NULL,
                             &wgtflag,
                             &numflag,
                             &nparts,
                             options,
                             &edgecut,
                             (idxtype *) partitioning);
#else
         err = METIS_PartGraphKway(&n,
                                   &ncon,
                                   I,
                                   J,
                                   (idx_t *) NULL,
                                   (idx_t *) NULL,
                                   (idx_t *) NULL,
                                   &nparts,
                                   (real_t *) NULL,
                                   (real_t *) NULL,
                                   options,
                                   &edgecut,
                                   partitioning);
         if (err != 1)
            mfem_error("Mesh::GeneratePartitioning: "
                       " error in METIS_PartGraphKway!");
#endif
      }

      // The objective of this partitioning is to minimize the total
      // communication volume
      if (part_method == 2 || part_method == 5)
      {
#ifndef MFEM_USE_METIS_5
         METIS_PartGraphVKway(&n,
                              (idxtype *) I,
                              (idxtype *) J,
                              (idxtype *) NULL,
                              (idxtype *) NULL,
                              &wgtflag,
                              &numflag,
                              &nparts,
                              options,
                              &edgecut,
                              (idxtype *) partitioning);
#else
         options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
         err = METIS_PartGraphKway(&n,
                                   &ncon,
                                   I,
                                   J,
                                   (idx_t *) NULL,
                                   (idx_t *) NULL,
                                   (idx_t *) NULL,
                                   &nparts,
                                   (real_t *) NULL,
                                   (real_t *) NULL,
                                   options,
                                   &edgecut,
                                   partitioning);
         if (err != 1)
            mfem_error("Mesh::GeneratePartitioning: "
                       " error in METIS_PartGraphKway!");
#endif
      }

#ifdef MFEM_DEBUG
      cout << "Mesh::GeneratePartitioning(...): edgecut = "
           << edgecut << endl;
#endif
   }

   if (el_to_el)
   {
      delete el_to_el;
   }
   el_to_el = NULL;

   // Check for empty partitionings (a "feature" in METIS)
   {
      Array< Pair<int,int> > psize(nparts);
      for (i = 0; i < nparts; i++)
      {
         psize[i].one = 0;
         psize[i].two = i;
      }

      for (i = 0; i < NumOfElements; i++)
      {
         psize[partitioning[i]].one++;
      }

      int empty_parts = 0;
      for (i = 0; i < nparts; i++)
         if (psize[i].one == 0)
         {
            empty_parts++;
         }

      // This code just split the largest partitionings in two.
      // Do we need to replace it with something better?
      if (empty_parts)
      {
         cerr << "Mesh::GeneratePartitioning returned " << empty_parts
              << " empty parts!" << endl;

         SortPairs<int,int>(psize, nparts);

         for (i = nparts-1; i > nparts-1-empty_parts; i--)
         {
            psize[i].one /= 2;
         }

         for (int j = 0; j < NumOfElements; j++)
            for (i = nparts-1; i > nparts-1-empty_parts; i--)
               if (psize[i].one == 0 || partitioning[j] != psize[i].two)
               {
                  continue;
               }
               else
               {
                  partitioning[j] = psize[nparts-1-i].two;
                  psize[i].one--;
               }
      }
   }

   return partitioning;

#else

   mfem_error("Mesh::GeneratePartitioning(...): "
              "MFEM was compiled without Metis.");

   return NULL;

#endif
}

/* required: 0 <= partitioning[i] < num_part */
void FindPartitioningComponents(Table &elem_elem,
                                const Array<int> &partitioning,
                                Array<int> &component,
                                Array<int> &num_comp)
{
   int i, j, k;
   int num_elem, *i_elem_elem, *j_elem_elem;

   num_elem    = elem_elem.Size();
   i_elem_elem = elem_elem.GetI();
   j_elem_elem = elem_elem.GetJ();

   component.SetSize(num_elem);

   Array<int> elem_stack(num_elem);
   int stack_p, stack_top_p, elem;
   int num_part;

   num_part = -1;
   for (i = 0; i < num_elem; i++)
   {
      if (partitioning[i] > num_part)
      {
         num_part = partitioning[i];
      }
      component[i] = -1;
   }
   num_part++;

   num_comp.SetSize(num_part);
   for (i = 0; i < num_part; i++)
   {
      num_comp[i] = 0;
   }

   stack_p = 0;
   stack_top_p = 0;  // points to the first unused element in the stack
   for (elem = 0; elem < num_elem; elem++)
   {
      if (component[elem] >= 0)
      {
         continue;
      }

      component[elem] = num_comp[partitioning[elem]]++;

      elem_stack[stack_top_p++] = elem;

      for ( ; stack_p < stack_top_p; stack_p++)
      {
         i = elem_stack[stack_p];
         for (j = i_elem_elem[i]; j < i_elem_elem[i+1]; j++)
         {
            k = j_elem_elem[j];
            if (partitioning[k] == partitioning[i])
            {
               if (component[k] < 0)
               {
                  component[k] = component[i];
                  elem_stack[stack_top_p++] = k;
               }
               else if (component[k] != component[i])
               {
                  mfem_error("FindPartitioningComponents");
               }
            }
         }
      }
   }
}

void Mesh::CheckPartitioning(int *partitioning)
{
   int i, n_empty, n_mcomp;
   Array<int> component, num_comp;
   const Array<int> _partitioning(partitioning, GetNE());

   ElementToElementTable();

   FindPartitioningComponents(*el_to_el, _partitioning, component, num_comp);

   n_empty = n_mcomp = 0;
   for (i = 0; i < num_comp.Size(); i++)
      if (num_comp[i] == 0)
      {
         n_empty++;
      }
      else if (num_comp[i] > 1)
      {
         n_mcomp++;
      }

   if (n_empty > 0)
   {
      cout << "Mesh::CheckPartitioning(...) :\n"
           << "The following subdomains are empty :\n";
      for (i = 0; i < num_comp.Size(); i++)
         if (num_comp[i] == 0)
         {
            cout << ' ' << i;
         }
      cout << endl;
   }
   if (n_mcomp > 0)
   {
      cout << "Mesh::CheckPartitioning(...) :\n"
           << "The following subdomains are NOT connected :\n";
      for (i = 0; i < num_comp.Size(); i++)
         if (num_comp[i] > 1)
         {
            cout << ' ' << i;
         }
      cout << endl;
   }
   if (n_empty == 0 && n_mcomp == 0)
      cout << "Mesh::CheckPartitioning(...) : "
           "All subdomains are connected." << endl;

   if (el_to_el)
   {
      delete el_to_el;
   }
   el_to_el = NULL;
}

// compute the coefficients of the polynomial in t:
//   c(0)+c(1)*t+...+c(d)*t^d = det(A+t*B)
// where A, B are (d x d), d=2,3
void DetOfLinComb(const DenseMatrix &A, const DenseMatrix &B, Vector &c)
{
   const double *a = A.Data();
   const double *b = B.Data();

   c.SetSize(A.Width()+1);
   switch (A.Width())
   {
      case 2:
      {
         // det(A+t*B) = |a0 a2|   / |a0 b2| + |b0 a2| \       |b0 b2|
         //              |a1 a3| + \ |a1 b3|   |b1 a3| / * t + |b1 b3| * t^2
         c(0) = a[0]*a[3]-a[1]*a[2];
         c(1) = a[0]*b[3]-a[1]*b[2]+b[0]*a[3]-b[1]*a[2];
         c(2) = b[0]*b[3]-b[1]*b[2];
      }
      break;

      case 3:
      {
         /*              |a0 a3 a6|
          * det(A+t*B) = |a1 a4 a7| +
          *              |a2 a5 a8|

          *     /  |b0 a3 a6|   |a0 b3 a6|   |a0 a3 b6| \
          *   + |  |b1 a4 a7| + |a1 b4 a7| + |a1 a4 b7| | * t +
          *     \  |b2 a5 a8|   |a2 b5 a8|   |a2 a5 b8| /

          *     /  |a0 b3 b6|   |b0 a3 b6|   |b0 b3 a6| \
          *   + |  |a1 b4 b7| + |b1 a4 b7| + |b1 b4 a7| | * t^2 +
          *     \  |a2 b5 b8|   |b2 a5 b8|   |b2 b5 a8| /

          *     |b0 b3 b6|
          *   + |b1 b4 b7| * t^3
          *     |b2 b5 b8|       */
         c(0) = (a[0] * (a[4] * a[8] - a[5] * a[7]) +
                 a[1] * (a[5] * a[6] - a[3] * a[8]) +
                 a[2] * (a[3] * a[7] - a[4] * a[6]));

         c(1) = (b[0] * (a[4] * a[8] - a[5] * a[7]) +
                 b[1] * (a[5] * a[6] - a[3] * a[8]) +
                 b[2] * (a[3] * a[7] - a[4] * a[6]) +

                 a[0] * (b[4] * a[8] - b[5] * a[7]) +
                 a[1] * (b[5] * a[6] - b[3] * a[8]) +
                 a[2] * (b[3] * a[7] - b[4] * a[6]) +

                 a[0] * (a[4] * b[8] - a[5] * b[7]) +
                 a[1] * (a[5] * b[6] - a[3] * b[8]) +
                 a[2] * (a[3] * b[7] - a[4] * b[6]));

         c(2) = (a[0] * (b[4] * b[8] - b[5] * b[7]) +
                 a[1] * (b[5] * b[6] - b[3] * b[8]) +
                 a[2] * (b[3] * b[7] - b[4] * b[6]) +

                 b[0] * (a[4] * b[8] - a[5] * b[7]) +
                 b[1] * (a[5] * b[6] - a[3] * b[8]) +
                 b[2] * (a[3] * b[7] - a[4] * b[6]) +

                 b[0] * (b[4] * a[8] - b[5] * a[7]) +
                 b[1] * (b[5] * a[6] - b[3] * a[8]) +
                 b[2] * (b[3] * a[7] - b[4] * a[6]));

         c(3) = (b[0] * (b[4] * b[8] - b[5] * b[7]) +
                 b[1] * (b[5] * b[6] - b[3] * b[8]) +
                 b[2] * (b[3] * b[7] - b[4] * b[6]));
      }
      break;

      default:
         mfem_error("DetOfLinComb(...)");
   }
}

// compute the real roots of
//   z(0)+z(1)*x+...+z(d)*x^d = 0,  d=2,3;
// the roots are returned in x, sorted in increasing order;
// it is assumed that x is at least of size d;
// return the number of roots counting multiplicity;
// return -1 if all z(i) are 0.
int FindRoots(const Vector &z, Vector &x)
{
   int d = z.Size()-1;
   if (d > 3 || d < 0)
   {
      mfem_error("FindRoots(...)");
   }

   while (z(d) == 0.0)
   {
      if (d == 0)
      {
         return (-1);
      }
      d--;
   }
   switch (d)
   {
      case 0:
      {
         return 0;
      }

      case 1:
      {
         x(0) = -z(0)/z(1);
         return 1;
      }

      case 2:
      {
         double a = z(2), b = z(1), c = z(0);
         double D = b*b-4*a*c;
         if (D < 0.0)
         {
            return 0;
         }
         if (D == 0.0)
         {
            x(0) = x(1) = -0.5 * b / a;
            return 2; // root with multiplicity 2
         }
         if (b == 0.0)
         {
            x(0) = -(x(1) = fabs(0.5 * sqrt(D) / a));
            return 2;
         }
         else
         {
            double t;
            if (b > 0.0)
            {
               t = -0.5 * (b + sqrt(D));
            }
            else
            {
               t = -0.5 * (b - sqrt(D));
            }
            x(0) = t / a;
            x(1) = c / t;
            if (x(0) > x(1))
            {
               Swap<double>(x(0), x(1));
            }
            return 2;
         }
      }

      case 3:
      {
         double a = z(2)/z(3), b = z(1)/z(3), c = z(0)/z(3);

         // find the real roots of x^3 + a x^2 + b x + c = 0
         double Q = (a * a - 3 * b) / 9;
         double R = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
         double Q3 = Q * Q * Q;
         double R2 = R * R;

         if (R2 == Q3)
         {
            if (Q == 0)
            {
               x(0) = x(1) = x(2) = - a / 3;
            }
            else
            {
               double sqrtQ = sqrt(Q);

               if (R > 0)
               {
                  x(0) = -2 * sqrtQ - a / 3;
                  x(1) = x(2) = sqrtQ - a / 3;
               }
               else
               {
                  x(0) = x(1) = - sqrtQ - a / 3;
                  x(2) = 2 * sqrtQ - a / 3;
               }
            }
            return 3;
         }
         else if (R2 < Q3)
         {
            double theta = acos(R / sqrt(Q3));
            double A = -2 * sqrt(Q);
            double x0, x1, x2;
            x0 = A * cos(theta / 3) - a / 3;
            x1 = A * cos((theta + 2.0 * M_PI) / 3) - a / 3;
            x2 = A * cos((theta - 2.0 * M_PI) / 3) - a / 3;

            /* Sort x0, x1, x2 */
            if (x0 > x1)
            {
               Swap<double>(x0, x1);
            }
            if (x1 > x2)
            {
               Swap<double>(x1, x2);
               if (x0 > x1)
               {
                  Swap<double>(x0, x1);
               }
            }
            x(0) = x0;
            x(1) = x1;
            x(2) = x2;
            return 3;
         }
         else
         {
            double A;
            if (R >= 0.0)
            {
               A = -pow(sqrt(R2 - Q3) + R, 1.0/3.0);
            }
            else
            {
               A =  pow(sqrt(R2 - Q3) - R, 1.0/3.0);
            }
            x(0) = A + Q / A - a / 3;
            return 1;
         }
      }
   }
   return 0;
}

void FindTMax(Vector &c, Vector &x, double &tmax,
              const double factor, const int Dim)
{
   const double c0 = c(0);
   c(0) = c0 * (1.0 - pow(factor, -Dim));
   int nr = FindRoots(c, x);
   for (int j = 0; j < nr; j++)
   {
      if (x(j) > tmax)
      {
         break;
      }
      if (x(j) >= 0.0)
      {
         tmax = x(j);
         break;
      }
   }
   c(0) = c0 * (1.0 - pow(factor, Dim));
   nr = FindRoots(c, x);
   for (int j = 0; j < nr; j++)
   {
      if (x(j) > tmax)
      {
         break;
      }
      if (x(j) >= 0.0)
      {
         tmax = x(j);
         break;
      }
   }
}

void Mesh::CheckDisplacements(const Vector &displacements, double &tmax)
{
   int nvs = vertices.Size();
   DenseMatrix P, V, DS, PDS(spaceDim), VDS(spaceDim);
   Vector c(spaceDim+1), x(spaceDim);
   const double factor = 2.0;

   // check for tangling assuming constant speed
   if (tmax < 1.0)
   {
      tmax = 1.0;
   }
   for (int i = 0; i < NumOfElements; i++)
   {
      Element *el = elements[i];
      int nv = el->GetNVertices();
      int *v = el->GetVertices();
      P.SetSize(spaceDim, nv);
      V.SetSize(spaceDim, nv);
      for (int j = 0; j < spaceDim; j++)
         for (int k = 0; k < nv; k++)
         {
            P(j, k) = vertices[v[k]](j);
            V(j, k) = displacements(v[k]+j*nvs);
         }
      DS.SetSize(nv, spaceDim);
      const FiniteElement *fe =
         GetTransformationFEforElementType(el->GetType());
      // check if  det(P.DShape+t*V.DShape) > 0 for all x and 0<=t<=1
      switch (el->GetType())
      {
         case Element::TRIANGLE:
         case Element::TETRAHEDRON:
         {
            // DS is constant
            fe->CalcDShape(Geometries.GetCenter(fe->GetGeomType()), DS);
            Mult(P, DS, PDS);
            Mult(V, DS, VDS);
            DetOfLinComb(PDS, VDS, c);
            if (c(0) <= 0.0)
            {
               tmax = 0.0;
            }
            else
            {
               FindTMax(c, x, tmax, factor, Dim);
            }
         }
         break;

         case Element::QUADRILATERAL:
         {
            const IntegrationRule &ir = fe->GetNodes();
            for (int j = 0; j < nv; j++)
            {
               fe->CalcDShape(ir.IntPoint(j), DS);
               Mult(P, DS, PDS);
               Mult(V, DS, VDS);
               DetOfLinComb(PDS, VDS, c);
               if (c(0) <= 0.0)
               {
                  tmax = 0.0;
               }
               else
               {
                  FindTMax(c, x, tmax, factor, Dim);
               }
            }
         }
         break;

         default:
            mfem_error("Mesh::CheckDisplacements(...)");
      }
   }
}

void Mesh::MoveVertices(const Vector &displacements)
{
   for (int i = 0, nv = vertices.Size(); i < nv; i++)
      for (int j = 0; j < spaceDim; j++)
      {
         vertices[i](j) += displacements(j*nv+i);
      }
}

void Mesh::GetVertices(Vector &vert_coord) const
{
   int nv = vertices.Size();
   vert_coord.SetSize(nv*spaceDim);
   for (int i = 0; i < nv; i++)
      for (int j = 0; j < spaceDim; j++)
      {
         vert_coord(j*nv+i) = vertices[i](j);
      }
}

void Mesh::SetVertices(const Vector &vert_coord)
{
   for (int i = 0, nv = vertices.Size(); i < nv; i++)
      for (int j = 0; j < spaceDim; j++)
      {
         vertices[i](j) = vert_coord(j*nv+i);
      }
}

void Mesh::GetNode(int i, double *coord)
{
   if (Nodes)
   {
      FiniteElementSpace *fes = Nodes->FESpace();
      for (int j = 0; j < spaceDim; j++)
      {
         coord[j] = (*Nodes)(fes->DofToVDof(i, j));
      }
   }
   else
   {
      for (int j = 0; j < spaceDim; j++)
      {
         coord[j] = vertices[i](j);
      }
   }
}

void Mesh::SetNode(int i, const double *coord)
{
   if (Nodes)
   {
      FiniteElementSpace *fes = Nodes->FESpace();
      for (int j = 0; j < spaceDim; j++)
      {
         (*Nodes)(fes->DofToVDof(i, j)) = coord[j];
      }
   }
   else
   {
      for (int j = 0; j < spaceDim; j++)
      {
         vertices[i](j) = coord[j];
      }

   }
}

void Mesh::MoveNodes(const Vector &displacements)
{
   if (Nodes)
   {
      (*Nodes) += displacements;
   }
   else
   {
      MoveVertices(displacements);
   }
}

void Mesh::GetNodes(Vector &node_coord) const
{
   if (Nodes)
   {
      node_coord = (*Nodes);
   }
   else
   {
      GetVertices(node_coord);
   }
}

void Mesh::SetNodes(const Vector &node_coord)
{
   if (Nodes)
   {
      (*Nodes) = node_coord;
   }
   else
   {
      SetVertices(node_coord);
   }
}

void Mesh::NewNodes(GridFunction &nodes, bool make_owner)
{
   if (own_nodes) { delete Nodes; }
   Nodes = &nodes;
   spaceDim = Nodes->FESpace()->GetVDim();
   own_nodes = (int)make_owner;

   if (NURBSext != nodes.FESpace()->GetNURBSext())
   {
      delete NURBSext;
      NURBSext = nodes.FESpace()->StealNURBSext();
   }
}

void Mesh::SwapNodes(GridFunction *&nodes, int &own_nodes_)
{
   mfem::Swap<GridFunction*>(Nodes, nodes);
   mfem::Swap<int>(own_nodes, own_nodes_);
   // TODO:
   // if (nodes)
   //    nodes->FESpace()->MakeNURBSextOwner();
   // NURBSext = (Nodes) ? Nodes->FESpace()->StealNURBSext() : NULL;
}

void Mesh::AverageVertices(int * indexes, int n, int result)
{
   int j, k;

   for (k = 0; k < spaceDim; k++)
   {
      vertices[result](k) = vertices[indexes[0]](k);
   }

   for (j = 1; j < n; j++)
      for (k = 0; k < spaceDim; k++)
      {
         vertices[result](k) += vertices[indexes[j]](k);
      }

   for (k = 0; k < spaceDim; k++)
   {
      vertices[result](k) *= (1.0 / n);
   }
}

void Mesh::UpdateNodes()
{
   if (Nodes)
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
   }
}

void Mesh::QuadUniformRefinement()
{
   int i, j, *v, vv[2], attr;
   const int *e;

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }

   int oedge = NumOfVertices;
   int oelem = oedge + NumOfEdges;

   vertices.SetSize(oelem + NumOfElements);

   for (i = 0; i < NumOfElements; i++)
   {
      v = elements[i]->GetVertices();

      AverageVertices(v, 4, oelem+i);

      e = el_to_edge->GetRow(i);

      vv[0] = v[0], vv[1] = v[1]; AverageVertices(vv, 2, oedge+e[0]);
      vv[0] = v[1], vv[1] = v[2]; AverageVertices(vv, 2, oedge+e[1]);
      vv[0] = v[2], vv[1] = v[3]; AverageVertices(vv, 2, oedge+e[2]);
      vv[0] = v[3], vv[1] = v[0]; AverageVertices(vv, 2, oedge+e[3]);
   }

   elements.SetSize(4 * NumOfElements);
   for (i = 0; i < NumOfElements; i++)
   {
      attr = elements[i]->GetAttribute();
      v = elements[i]->GetVertices();
      e = el_to_edge->GetRow(i);
      j = NumOfElements + 3 * i;

      elements[j+0] = new Quadrilateral(oedge+e[0], v[1], oedge+e[1],
                                        oelem+i, attr);
      elements[j+1] = new Quadrilateral(oelem+i, oedge+e[1], v[2],
                                        oedge+e[2], attr);
      elements[j+2] = new Quadrilateral(oedge+e[3], oelem+i, oedge+e[2],
                                        v[3], attr);

      v[1] = oedge+e[0];
      v[2] = oelem+i;
      v[3] = oedge+e[3];
   }

   boundary.SetSize(2 * NumOfBdrElements);
   for (i = 0; i < NumOfBdrElements; i++)
   {
      attr = boundary[i]->GetAttribute();
      v = boundary[i]->GetVertices();
      j = NumOfBdrElements + i;

      boundary[j] = new Segment(oedge+be_to_edge[i], v[1], attr);

      v[1] = oedge+be_to_edge[i];
   }

   static double quad_children[2*4*4] =
   {
      0.0,0.0, 0.5,0.0, 0.5,0.5, 0.0,0.5, // lower-left
      0.5,0.0, 1.0,0.0, 1.0,0.5, 0.5,0.5, // lower-right
      0.5,0.5, 1.0,0.5, 1.0,1.0, 0.5,1.0, // upper-right
      0.0,0.5, 0.5,0.5, 0.5,1.0, 0.0,1.0  // upper-left
   };

   CoarseFineTr.point_matrices.UseExternalData(quad_children, 2, 4, 4);
   CoarseFineTr.embeddings.SetSize(elements.Size());

   for (i = 0; i < elements.Size(); i++)
   {
      Embedding &emb = CoarseFineTr.embeddings[i];
      emb.parent = (i < NumOfElements) ? i : (i - NumOfElements) / 3;
      emb.matrix = (i < NumOfElements) ? 0 : (i - NumOfElements) % 3 + 1;
   }

   NumOfVertices    = oelem + NumOfElements;
   NumOfElements    = 4 * NumOfElements;
   NumOfBdrElements = 2 * NumOfBdrElements;
   NumOfFaces       = 0;

   if (el_to_edge != NULL)
   {
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      GenerateFaces();
   }

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
   CheckBdrElementOrientation(false);
#endif
}

void Mesh::HexUniformRefinement()
{
   int i;
   int * v;
   const int *e, *f;
   int vv[4];

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   if (el_to_face == NULL)
   {
      GetElementToFaceTable();
   }

   int oedge = NumOfVertices;
   int oface = oedge + NumOfEdges;
   int oelem = oface + NumOfFaces;

   vertices.SetSize(oelem + NumOfElements);
   for (i = 0; i < NumOfElements; i++)
   {
      MFEM_ASSERT(elements[i]->GetType() == Element::HEXAHEDRON,
                  "Element is not a hex!");
      v = elements[i]->GetVertices();

      AverageVertices(v, 8, oelem+i);

      f = el_to_face->GetRow(i);

      for (int j = 0; j < 6; j++)
      {
         for (int k = 0; k < 4; k++)
         {
            vv[k] = v[hex_t::FaceVert[j][k]];
         }
         AverageVertices(vv, 4, oface+f[j]);
      }

      e = el_to_edge->GetRow(i);

      for (int j = 0; j < 12; j++)
      {
         for (int k = 0; k < 2; k++)
         {
            vv[k] = v[hex_t::Edges[j][k]];
         }
         AverageVertices(vv, 2, oedge+e[j]);
      }
   }

   int attr, j;
   elements.SetSize(8 * NumOfElements);
   for (i = 0; i < NumOfElements; i++)
   {
      attr = elements[i]->GetAttribute();
      v = elements[i]->GetVertices();
      e = el_to_edge->GetRow(i);
      f = el_to_face->GetRow(i);
      j = NumOfElements + 7 * i;

      elements[j+0] = new Hexahedron(oedge+e[0], v[1], oedge+e[1], oface+f[0],
                                     oface+f[1], oedge+e[9], oface+f[2],
                                     oelem+i, attr);
      elements[j+1] = new Hexahedron(oface+f[0], oedge+e[1], v[2], oedge+e[2],
                                     oelem+i, oface+f[2], oedge+e[10],
                                     oface+f[3], attr);
      elements[j+2] = new Hexahedron(oedge+e[3], oface+f[0], oedge+e[2], v[3],
                                     oface+f[4], oelem+i, oface+f[3],
                                     oedge+e[11], attr);
      elements[j+3] = new Hexahedron(oedge+e[8], oface+f[1], oelem+i,
                                     oface+f[4], v[4], oedge+e[4], oface+f[5],
                                     oedge+e[7], attr);
      elements[j+4] = new Hexahedron(oface+f[1], oedge+e[9], oface+f[2],
                                     oelem+i, oedge+e[4], v[5], oedge+e[5],
                                     oface+f[5], attr);
      elements[j+5] = new Hexahedron(oelem+i, oface+f[2], oedge+e[10],
                                     oface+f[3], oface+f[5], oedge+e[5], v[6],
                                     oedge+e[6], attr);
      elements[j+6] = new Hexahedron(oface+f[4], oelem+i, oface+f[3],
                                     oedge+e[11], oedge+e[7], oface+f[5],
                                     oedge+e[6], v[7], attr);

      v[1] = oedge+e[0];
      v[2] = oface+f[0];
      v[3] = oedge+e[3];
      v[4] = oedge+e[8];
      v[5] = oface+f[1];
      v[6] = oelem+i;
      v[7] = oface+f[4];
   }

   boundary.SetSize(4 * NumOfBdrElements);
   for (i = 0; i < NumOfBdrElements; i++)
   {
      MFEM_ASSERT(boundary[i]->GetType() == Element::QUADRILATERAL,
                  "boundary Element is not a quad!");
      attr = boundary[i]->GetAttribute();
      v = boundary[i]->GetVertices();
      e = bel_to_edge->GetRow(i);
      f = & be_to_face[i];
      j = NumOfBdrElements + 3 * i;

      boundary[j+0] = new Quadrilateral(oedge+e[0], v[1], oedge+e[1],
                                        oface+f[0], attr);
      boundary[j+1] = new Quadrilateral(oface+f[0], oedge+e[1], v[2],
                                        oedge+e[2], attr);
      boundary[j+2] = new Quadrilateral(oedge+e[3], oface+f[0], oedge+e[2],
                                        v[3], attr);

      v[1] = oedge+e[0];
      v[2] = oface+f[0];
      v[3] = oedge+e[3];
   }

   static const double A = 0.0, B = 0.5, C = 1.0;
   static double hex_children[3*8*8] =
   {
      A,A,A, B,A,A, B,B,A, A,B,A, A,A,B, B,A,B, B,B,B, A,B,B,
      B,A,A, C,A,A, C,B,A, B,B,A, B,A,B, C,A,B, C,B,B, B,B,B,
      B,B,A, C,B,A, C,C,A, B,C,A, B,B,B, C,B,B, C,C,B, B,C,B,
      A,B,A, B,B,A, B,C,A, A,C,A, A,B,B, B,B,B, B,C,B, A,C,B,
      A,A,B, B,A,B, B,B,B, A,B,B, A,A,C, B,A,C, B,B,C, A,B,C,
      B,A,B, C,A,B, C,B,B, B,B,B, B,A,C, C,A,C, C,B,C, B,B,C,
      B,B,B, C,B,B, C,C,B, B,C,B, B,B,C, C,B,C, C,C,C, B,C,C,
      A,B,B, B,B,B, B,C,B, A,C,B, A,B,C, B,B,C, B,C,C, A,C,C
   };

   CoarseFineTr.point_matrices.UseExternalData(hex_children, 3, 8, 8);
   CoarseFineTr.embeddings.SetSize(elements.Size());

   for (i = 0; i < elements.Size(); i++)
   {
      Embedding &emb = CoarseFineTr.embeddings[i];
      emb.parent = (i < NumOfElements) ? i : (i - NumOfElements) / 7;
      emb.matrix = (i < NumOfElements) ? 0 : (i - NumOfElements) % 7 + 1;
   }

   NumOfVertices    = oelem + NumOfElements;
   NumOfElements    = 8 * NumOfElements;
   NumOfBdrElements = 4 * NumOfBdrElements;

   if (el_to_face != NULL)
   {
      GetElementToFaceTable();
      GenerateFaces();
   }

#ifdef MFEM_DEBUG
   CheckBdrElementOrientation(false);
#endif

   if (el_to_edge != NULL)
   {
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();
}

void Mesh::LocalRefinement(const Array<int> &marked_el, int type)
{
   int i, j, ind, nedges;
   Array<int> v;

   if (ncmesh)
   {
      MFEM_ABORT("Local and nonconforming refinements cannot be mixed.");
   }

   InitRefinementTransforms();

   if (Dim == 1) // --------------------------------------------------------
   {
      int cne = NumOfElements, cnv = NumOfVertices;
      NumOfVertices += marked_el.Size();
      NumOfElements += marked_el.Size();
      vertices.SetSize(NumOfVertices);
      elements.SetSize(NumOfElements);
      CoarseFineTr.embeddings.SetSize(NumOfElements);

      for (j = 0; j < marked_el.Size(); j++)
      {
         i = marked_el[j];
         Segment *c_seg = (Segment *)elements[i];
         int *vert = c_seg->GetVertices(), attr = c_seg->GetAttribute();
         int new_v = cnv + j, new_e = cne + j;
         AverageVertices(vert, 2, new_v);
         elements[new_e] = new Segment(new_v, vert[1], attr);
         vert[1] = new_v;

         CoarseFineTr.embeddings[i] = Embedding(i, 1);
         CoarseFineTr.embeddings[new_e] = Embedding(i, 2);
      }

      static double seg_children[3*2] = { 0.0,1.0, 0.0,0.5, 0.5,1.0 };
      CoarseFineTr.point_matrices.UseExternalData(seg_children, 1, 2, 3);

      GenerateFaces();

   } // end of 'if (Dim == 1)'
   else if (Dim == 2) // ---------------------------------------------------
   {
      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      nedges = v_to_v.NumberOfEntries();
      int *edge1  = new int[nedges];
      int *edge2  = new int[nedges];
      int *middle = new int[nedges];

      for (i = 0; i < nedges; i++)
      {
         edge1[i] = edge2[i] = middle[i] = -1;
      }

      for (i = 0; i < NumOfElements; i++)
      {
         elements[i]->GetVertices(v);
         for (j = 1; j < v.Size(); j++)
         {
            ind = v_to_v(v[j-1], v[j]);
            (edge1[ind] == -1) ? (edge1[ind] = i) : (edge2[ind] = i);
         }
         ind = v_to_v(v[0], v[v.Size()-1]);
         (edge1[ind] == -1) ? (edge1[ind] = i) : (edge2[ind] = i);
      }

      // 3. Do the red refinement.
      for (i = 0; i < marked_el.Size(); i++)
      {
         RedRefinement(marked_el[i], v_to_v, edge1, edge2, middle);
      }

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      do
      {
         need_refinement = 0;
         for (i = 0; i < nedges; i++)
         {
            if (middle[i] != -1 && edge1[i] != -1)
            {
               need_refinement = 1;
               GreenRefinement(edge1[i], v_to_v, edge1, edge2, middle);
            }
         }
      }
      while (need_refinement == 1);

      // 5. Update the boundary elements.
      int v1[2], v2[2], bisect, temp;
      temp = NumOfBdrElements;
      for (i = 0; i < temp; i++)
      {
         boundary[i]->GetVertices(v);
         bisect = v_to_v(v[0], v[1]);
         if (middle[bisect] != -1) // the element was refined (needs updating)
         {
            if (boundary[i]->GetType() == Element::SEGMENT)
            {
               v1[0] =           v[0]; v1[1] = middle[bisect];
               v2[0] = middle[bisect]; v2[1] =           v[1];

               boundary[i]->SetVertices(v1);
               boundary.Append(new Segment(v2, boundary[i]->GetAttribute()));
            }
            else
               mfem_error("Only bisection of segment is implemented"
                          " for bdr elem.");
         }
      }
      NumOfBdrElements = boundary.Size();

      // 6. Free the allocated memory.
      delete [] edge1;
      delete [] edge2;
      delete [] middle;

      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         GenerateFaces();
      }

   }
   else if (Dim == 3) // ---------------------------------------------------
   {
      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      nedges = v_to_v.NumberOfEntries();
      int *middle = new int[nedges];

      for (i = 0; i < nedges; i++)
      {
         middle[i] = -1;
      }

      // 3. Do the red refinement.
      int ii;
      switch (type)
      {
         case 1:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
            }
            break;
         case 2:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);

               Bisection(NumOfElements - 1, v_to_v, NULL, NULL, middle);
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
            }
            break;
         case 3:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);

               ii = NumOfElements - 1;
               Bisection(ii, v_to_v, NULL, NULL, middle);
               Bisection(NumOfElements - 1, v_to_v, NULL, NULL, middle);
               Bisection(ii, v_to_v, NULL, NULL, middle);

               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
               Bisection(NumOfElements-1, v_to_v, NULL, NULL, middle);
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
            }
            break;
      }

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      // int need_refinement, onoe, max_gen = 0;
      do
      {
         // int redges[2], type, flag;
         need_refinement = 0;
         // onoe = NumOfElements;
         // for (i = 0; i < onoe; i++)
         for (i = 0; i < NumOfElements; i++)
         {
            // ((Tetrahedron *)elements[i])->ParseRefinementFlag(redges, type, flag);
            // if (flag > max_gen)  max_gen = flag;
            if (elements[i]->NeedRefinement(v_to_v, middle))
            {
               need_refinement = 1;
               Bisection(i, v_to_v, NULL, NULL, middle);
            }
         }
      }
      while (need_refinement == 1);

      // cout << "Maximum generation: " << max_gen << endl;

      // 5. Update the boundary elements.
      do
      {
         need_refinement = 0;
         for (i = 0; i < NumOfBdrElements; i++)
            if (boundary[i]->NeedRefinement(v_to_v, middle))
            {
               need_refinement = 1;
               Bisection(i, v_to_v, middle);
            }
      }
      while (need_refinement == 1);

      // 6. Un-mark the Pf elements.
      int refinement_edges[2], type, flag;
      for (i = 0; i < NumOfElements; i++)
      {
         Tetrahedron* el = (Tetrahedron*) elements[i];
         el->ParseRefinementFlag(refinement_edges, type, flag);

         if (type == Tetrahedron::TYPE_PF)
         {
            el->CreateRefinementFlag(refinement_edges, Tetrahedron::TYPE_PU,
                                     flag);
         }
      }

      NumOfBdrElements = boundary.Size();

      // 7. Free the allocated memory.
      delete [] middle;

      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }
      if (el_to_face != NULL)
      {
         GetElementToFaceTable();
         GenerateFaces();
      }

   } //  end 'if (Dim == 3)'
   else if( Dim == 4 )
   {
	   // 1. Get table of vertex to vertex connections.
	   DSTable v_to_v(NumOfVertices);
	   GetVertexToVertexTable(v_to_v);

	   // 2. Get edge to element connections in arrays edge1 and edge2
	   nedges = v_to_v.NumberOfEntries();
	   int *middle = new int[nedges];

	   for (i = 0; i < nedges; i++)
	   {
		   middle[i] = -1;
	   }

	   // 3. Do the red refinement.
	   for(int i = 0; i < marked_el.Size(); i++)
	   {
		   RedRefinementPentatope(marked_el[i], v_to_v, middle);
	   }

	   // 4. Update the boundary elements.
	   for(int i = 0; i < NumOfBdrElements; i++)
		   if (boundary[i]->NeedRefinement(v_to_v, middle))
		   {
			   RedRefinementBoundaryTet(i, v_to_v, middle);
		   }
	   NumOfBdrElements = boundary.Size();

	   // 5. Free the allocated memory.
	   delete [] middle;

	   if (el_to_edge != NULL)
	   {
			NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
	   }
	   if (el_to_face != NULL)
	   {
		   GetElementToFaceTable4D();
		   GenerateFaces();

//   		 ReplaceBoundaryFromFaces();

		   GetElementToPlanarTable();
		   GeneratePlanars();
	  }
  }

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
#endif
}

void Mesh::NonconformingRefinement(const Array<Refinement> &refinements,
                                   int nc_limit)
{
   MFEM_VERIFY(!NURBSext, "Nonconforming refinement of NURBS meshes is "
               "not supported. Project the NURBS to Nodes first.");

   if (!ncmesh)
   {
      // start tracking refinement hierarchy
      ncmesh = new NCMesh(this);
   }

   if (!refinements.Size())
   {
      last_operation = Mesh::NONE;
      return;
   }

   // do the refinements
   ncmesh->MarkCoarseLevel();
   ncmesh->Refine(refinements);

   if (nc_limit > 0)
   {
      ncmesh->LimitNCLevel(nc_limit);
   }

   // create a second mesh containing the finest elements from 'ncmesh'
   Mesh* mesh2 = new Mesh(*ncmesh);
   ncmesh->OnMeshUpdated(mesh2);

   // now swap the meshes, the second mesh will become the old coarse mesh
   // and this mesh will be the new fine mesh
   Swap(*mesh2, false);
   delete mesh2;

   GenerateNCFaceInfo();

   last_operation = Mesh::REFINE;
   sequence++;

   if (Nodes) // update/interpolate curved mesh
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
   }
}

void Mesh::DerefineMesh(const Array<int> &derefinements)
{
   MFEM_VERIFY(ncmesh, "only supported for non-conforming meshes.");
   MFEM_VERIFY(!NURBSext, "Derefinement of NURBS meshes is not supported. "
               "Project the NURBS to Nodes first.");

   ncmesh->Derefine(derefinements);

   Mesh* mesh2 = new Mesh(*ncmesh);
   ncmesh->OnMeshUpdated(mesh2);

   Swap(*mesh2, false);
   delete mesh2;

   GenerateNCFaceInfo();

   last_operation = Mesh::DEREFINE;
   sequence++;

   if (Nodes) // update/interpolate mesh curvature
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
   }
}

bool Mesh::NonconformingDerefinement(Array<double> &elem_error,
                                     double threshold, int nc_limit, int op)
{
   const Table &dt = ncmesh->GetDerefinementTable();

   Array<int> level_ok;
   if (nc_limit > 0)
   {
      ncmesh->CheckDerefinementNCLevel(dt, level_ok, nc_limit);
   }

   Array<int> derefs;
   for (int i = 0; i < dt.Size(); i++)
   {
      if (nc_limit > 0 && !level_ok[i]) { continue; }

      const int* fine = dt.GetRow(i);
      int size = dt.RowSize(i);

      double error = 0.0;
      for (int j = 0; j < size; j++)
      {
         MFEM_VERIFY(fine[j] < elem_error.Size(), "");

         double err_fine = elem_error[fine[j]];
         switch (op)
         {
            case 0: error = std::min(error, err_fine); break;
            case 1: error += err_fine; break;
            case 2: error = std::max(error, err_fine); break;
         }
      }

      if (error < threshold) { derefs.Append(i); }
   }

   if (derefs.Size())
   {
      DerefineMesh(derefs);
      return true;
   }

   return false;
}

bool Mesh::DerefineByError(Array<double> &elem_error, double threshold,
                           int nc_limit, int op)
{
   // NOTE: the error array is not const because it will be expanded in parallel
   //       by ghost element errors
   if (Nonconforming())
   {
      return NonconformingDerefinement(elem_error, threshold, nc_limit, op);
   }
   else
   {
      MFEM_ABORT("Derefinement is currently supported for non-conforming "
                 "meshes only.");
      return false;
   }
}

bool Mesh::DerefineByError(const Vector &elem_error, double threshold,
                           int nc_limit, int op)
{
   Array<double> tmp(elem_error.Size());
   for (int i = 0; i < tmp.Size(); i++)
   {
      tmp[i] = elem_error(i);
   }
   return DerefineByError(tmp, threshold, nc_limit, op);
}


void Mesh::InitFromNCMesh(const NCMesh &ncmesh)
{
   Dim = ncmesh.Dimension();
   spaceDim = ncmesh.SpaceDimension();

   BaseGeom = ncmesh.GetElementGeometry();

   switch (BaseGeom)
   {
      case Geometry::TRIANGLE:
      case Geometry::SQUARE:
         BaseBdrGeom = Geometry::SEGMENT;
         break;
      case Geometry::CUBE:
         BaseBdrGeom = Geometry::SQUARE;
         break;
      default:
         BaseBdrGeom = -1;
   }

   DeleteTables();

   bool linear = (Nodes == NULL);
   ncmesh.GetMeshComponents(vertices, elements, boundary, linear);

   NumOfVertices = vertices.Size();
   NumOfElements = elements.Size();
   NumOfBdrElements = boundary.Size();

   SetMeshGen(); // set the mesh type ('meshgen')

   NumOfEdges = NumOfFaces = 0;

   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   if (Dim > 2)
   {
      GetElementToFaceTable();
   }
   GenerateFaces();
#ifdef MFEM_DEBUG
   CheckBdrElementOrientation(false);
#endif

   // NOTE: ncmesh->OnMeshUpdated() and GenerateNCFaceInfo() should be called
   // outside after this method.
}

Mesh::Mesh(const NCMesh &ncmesh)
{
   Init();
   InitTables();
   InitFromNCMesh(ncmesh);
   SetAttributes();
}

void Mesh::Swap(Mesh& other, bool non_geometry)
{
   mfem::Swap(Dim, other.Dim);
   mfem::Swap(spaceDim, other.spaceDim);

   mfem::Swap(NumOfVertices, other.NumOfVertices);
   mfem::Swap(NumOfElements, other.NumOfElements);
   mfem::Swap(NumOfBdrElements, other.NumOfBdrElements);
   mfem::Swap(NumOfEdges, other.NumOfEdges);
   mfem::Swap(NumOfFaces, other.NumOfFaces);

   mfem::Swap(meshgen, other.meshgen);

   mfem::Swap(elements, other.elements);
   mfem::Swap(vertices, other.vertices);
   mfem::Swap(boundary, other.boundary);
   mfem::Swap(faces, other.faces);
   mfem::Swap(faces_info, other.faces_info);
   mfem::Swap(nc_faces_info, other.nc_faces_info);

   mfem::Swap(el_to_edge, other.el_to_edge);
   mfem::Swap(el_to_face, other.el_to_face);
   mfem::Swap(el_to_el, other.el_to_el);
   mfem::Swap(be_to_edge, other.be_to_edge);
   mfem::Swap(bel_to_edge, other.bel_to_edge);
   mfem::Swap(be_to_face, other.be_to_face);
   mfem::Swap(face_edge, other.face_edge);
   mfem::Swap(edge_vertex, other.edge_vertex);

   mfem::Swap(attributes, other.attributes);
   mfem::Swap(bdr_attributes, other.bdr_attributes);

   if (non_geometry)
   {
      mfem::Swap(NURBSext, other.NURBSext);
      mfem::Swap(ncmesh, other.ncmesh);

      mfem::Swap(Nodes, other.Nodes);
      mfem::Swap(own_nodes, other.own_nodes);
   }
}

void Mesh::GetElementData(const Array<Element*> &elem_array, int geom,
                          Array<int> &elem_vtx, Array<int> &attr) const
{
   // protected method
   const int nv = Geometry::NumVerts[geom];
   int num_elems = 0;
   for (int i = 0; i < elem_array.Size(); i++)
   {
      if (elem_array[i]->GetGeometryType() == geom)
      {
         num_elems++;
      }
   }
   elem_vtx.SetSize(nv*num_elems);
   attr.SetSize(num_elems);
   elem_vtx.SetSize(0);
   attr.SetSize(0);
   for (int i = 0; i < elem_array.Size(); i++)
   {
      Element *el = elem_array[i];
      if (el->GetGeometryType() != geom) { continue; }

      Array<int> loc_vtx(el->GetVertices(), nv);
      elem_vtx.Append(loc_vtx);
      attr.Append(el->GetAttribute());
   }
}

void Mesh::UniformRefinement()
{
   if (NURBSext)
   {
      NURBSUniformRefinement();
   }
   else if (meshgen == 1 || ncmesh)
   {
      Array<int> elem_to_refine(GetNE());
      for (int i = 0; i < elem_to_refine.Size(); i++)
      {
         elem_to_refine[i] = i;
      }

      if (Conforming())
      {
         // In parallel we should set the default 2nd argument to -3 to indicate
         // uniform refinement.
         LocalRefinement(elem_to_refine);
      }
      else
      {
         GeneralRefinement(elem_to_refine, 1);
      }
   }
   else if (Dim == 2)
   {
      QuadUniformRefinement();
   }
   else if (Dim == 3)
   {
      HexUniformRefinement();
   }
   else
   {
      mfem_error("Mesh::UniformRefinement()");
   }
}

void Mesh::GeneralRefinement(const Array<Refinement> &refinements,
                             int nonconforming, int nc_limit)
{
   if (Dim == 1 || (Dim == 3 && meshgen & 1))
   {
      nonconforming = 0;
   }
   else if (nonconforming < 0)
   {
      // determine if nonconforming refinement is suitable
      int geom = GetElementBaseGeometry();
      if (geom == Geometry::CUBE || geom == Geometry::SQUARE)
      {
         nonconforming = 1;
      }
      else
      {
         nonconforming = 0;
      }
   }

   if (nonconforming || ncmesh != NULL)
   {
      // non-conforming refinement (hanging nodes)
      NonconformingRefinement(refinements, nc_limit);
   }
   else
   {
      Array<int> el_to_refine(refinements.Size());
      for (int i = 0; i < refinements.Size(); i++)
      {
         el_to_refine[i] = refinements[i].index;
      }

      // infer 'type' of local refinement from first element's 'ref_type'
      int type, rt = (refinements.Size() ? refinements[0].ref_type : 7);
      if (rt == 1 || rt == 2 || rt == 4)
      {
         type = 1; // bisection
      }
      else if (rt == 3 || rt == 5 || rt == 6)
      {
         type = 2; // quadrisection
      }
      else
      {
         type = 3; // octasection
      }

      // red-green refinement and bisection, no hanging nodes
      LocalRefinement(el_to_refine, type);
   }
}

void Mesh::GeneralRefinement(const Array<int> &el_to_refine, int nonconforming,
                             int nc_limit)
{
   Array<Refinement> refinements(el_to_refine.Size());
   for (int i = 0; i < el_to_refine.Size(); i++)
   {
      refinements[i] = Refinement(el_to_refine[i]);
   }
   GeneralRefinement(refinements, nonconforming, nc_limit);
}

void Mesh::EnsureNCMesh(bool triangles_nonconforming)
{
   MFEM_VERIFY(!NURBSext, "Cannot convert a NURBS mesh to an NC mesh. "
               "Project the NURBS to Nodes first.");

   if (!ncmesh)
   {
      if ((meshgen & 2) /* quads/hexes */ ||
          (triangles_nonconforming && BaseGeom == Geometry::TRIANGLE))
      {
         ncmesh = new NCMesh(this);
         ncmesh->OnMeshUpdated(this);
         GenerateNCFaceInfo();
      }
   }
}

void Mesh::RandomRefinement(double prob, bool aniso, int nonconforming,
                            int nc_limit)
{
   Array<Refinement> refs;
   for (int i = 0; i < GetNE(); i++)
   {
      if ((double) rand() / RAND_MAX < prob)
      {
         int type = 7;
         if (aniso)
         {
            type = (Dim == 3) ? (rand() % 7 + 1) : (rand() % 3 + 1);
         }
         refs.Append(Refinement(i, type));
      }
   }
   GeneralRefinement(refs, nonconforming, nc_limit);
}

void Mesh::RefineAtVertex(const Vertex& vert, double eps, int nonconforming)
{
   Array<int> v;
   Array<Refinement> refs;
   for (int i = 0; i < GetNE(); i++)
   {
      GetElementVertices(i, v);
      bool refine = false;
      for (int j = 0; j < v.Size(); j++)
      {
         double dist = 0.0;
         for (int l = 0; l < spaceDim; l++)
         {
            double d = vert(l) - vertices[v[j]](l);
            dist += d*d;
         }
         if (dist <= eps*eps) { refine = true; break; }
      }
      if (refine)
      {
         refs.Append(Refinement(i));
      }
   }
   GeneralRefinement(refs, nonconforming);
}

bool Mesh::RefineByError(const Array<double> &elem_error, double threshold,
                         int nonconforming, int nc_limit)
{
   MFEM_VERIFY(elem_error.Size() == GetNE(), "");
   Array<Refinement> refs;
   for (int i = 0; i < GetNE(); i++)
   {
      if (elem_error[i] > threshold)
      {
         refs.Append(Refinement(i));
      }
   }
   if (ReduceInt(refs.Size()))
   {
      GeneralRefinement(refs, nonconforming, nc_limit);
      return true;
   }
   return false;
}

bool Mesh::RefineByError(const Vector &elem_error, double threshold,
                         int nonconforming, int nc_limit)
{
   Array<double> tmp(const_cast<double*>(elem_error.GetData()),
                     elem_error.Size());
   return RefineByError(tmp, threshold, nonconforming, nc_limit);
}


void Mesh::Bisection(int i, const DSTable &v_to_v,
                     int *edge1, int *edge2, int *middle)
{
   int *vert;
   int v[2][4], v_new, bisect, t;
   Element *el = elements[i];
   Vertex V;

   t = el->GetType();
   if (t == Element::TRIANGLE)
   {
      Triangle *tri = (Triangle *) el;

      vert = tri->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v(vert[0], vert[1]);
      MFEM_ASSERT(bisect >= 0, "");

      if (middle[bisect] == -1)
      {
         v_new = NumOfVertices++;
         for (int d = 0; d < spaceDim; d++)
         {
            V(d) = 0.5 * (vertices[vert[0]](d) + vertices[vert[1]](d));
         }
         vertices.Append(V);

         // Put the element that may need refinement (because of this
         // bisection) in edge1, or -1 if no more refinement is needed.
         if (edge1[bisect] == i)
         {
            edge1[bisect] = edge2[bisect];
         }

         middle[bisect] = v_new;
      }
      else
      {
         v_new = middle[bisect];

         // This edge will require no more refinement.
         edge1[bisect] = -1;
      }

      // 2. Set the node indices for the new elements in v[0] and v[1] so that
      //    the  edge marked for refinement is between the first two nodes.
      v[0][0] = vert[2]; v[0][1] = vert[0]; v[0][2] = v_new;
      v[1][0] = vert[1]; v[1][1] = vert[2]; v[1][2] = v_new;

      tri->SetVertices(v[0]);   // changes vert[0..2] !!!

      Triangle* tri_new = new Triangle(v[1], tri->GetAttribute());
      elements.Append(tri_new);

      int tr = tri->GetTransform();
      tri_new->ResetTransform(tr);

      // record the sequence of refinements
      tri->PushTransform(4);
      tri_new->PushTransform(5);

      int coarse = FindCoarseElement(i);
      CoarseFineTr.embeddings[i].parent = coarse;
      CoarseFineTr.embeddings.Append(Embedding(coarse));

      // 3. edge1 and edge2 may have to be changed for the second triangle.
      if (v[1][0] < v_to_v.NumberOfRows() && v[1][1] < v_to_v.NumberOfRows())
      {
         bisect = v_to_v(v[1][0], v[1][1]);
         MFEM_ASSERT(bisect >= 0, "");

         if (edge1[bisect] == i)
         {
            edge1[bisect] = NumOfElements;
         }
         else if (edge2[bisect] == i)
         {
            edge2[bisect] = NumOfElements;
         }
      }
      NumOfElements++;
   }
   else if (t == Element::TETRAHEDRON)
   {
      int j, type, new_type, old_redges[2], new_redges[2][2], flag;
      Tetrahedron *tet = (Tetrahedron *) el;

      MFEM_VERIFY(tet->GetRefinementFlag() != 0,
                  "TETRAHEDRON element is not marked for refinement.");

      vert = tet->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v(vert[0], vert[1]);
      if (bisect == -1)
      {
         tet->ParseRefinementFlag(old_redges, type, flag);
         cerr << "Error in Bisection(...) of tetrahedron!" << endl
              << "   redge[0] = " << old_redges[0]
              << "   redge[1] = " << old_redges[1]
              << "   type = " << type
              << "   flag = " << flag << endl;
         mfem_error();
      }

      if (middle[bisect] == -1)
      {
         v_new = NumOfVertices++;
         for (j = 0; j < 3; j++)
         {
            V(j) = 0.5 * (vertices[vert[0]](j) + vertices[vert[1]](j));
         }
         vertices.Append(V);

         middle[bisect] = v_new;
      }
      else
      {
         v_new = middle[bisect];
      }

      // 2. Set the node indices for the new elements in v[2][4] so that
      //    the edge marked for refinement is between the first two nodes.
      tet->ParseRefinementFlag(old_redges, type, flag);

      v[0][3] = v_new;
      v[1][3] = v_new;
      new_redges[0][0] = 2;
      new_redges[0][1] = 1;
      new_redges[1][0] = 2;
      new_redges[1][1] = 1;
      int tr1 = -1, tr2 = -1;
      switch (old_redges[0])
      {
         case 2:
            v[0][0] = vert[0]; v[0][1] = vert[2]; v[0][2] = vert[3];
            if (type == Tetrahedron::TYPE_PF) { new_redges[0][1] = 4; }
            tr1 = 0;
            break;
         case 3:
            v[0][0] = vert[3]; v[0][1] = vert[0]; v[0][2] = vert[2];
            tr1 = 2;
            break;
         case 5:
            v[0][0] = vert[2]; v[0][1] = vert[3]; v[0][2] = vert[0];
            tr1 = 4;
      }
      switch (old_redges[1])
      {
         case 1:
            v[1][0] = vert[2]; v[1][1] = vert[1]; v[1][2] = vert[3];
            if (type == Tetrahedron::TYPE_PF) { new_redges[1][0] = 3; }
            tr2 = 1;
            break;
         case 4:
            v[1][0] = vert[1]; v[1][1] = vert[3]; v[1][2] = vert[2];
            tr2 = 3;
            break;
         case 5:
            v[1][0] = vert[3]; v[1][1] = vert[2]; v[1][2] = vert[1];
            tr2 = 5;
      }

      int attr = tet->GetAttribute();
      tet->SetVertices(v[0]);

#ifdef MFEM_USE_MEMALLOC
      Tetrahedron *tet2 = TetMemory.Alloc();
      tet2->SetVertices(v[1]);
      tet2->SetAttribute(attr);
#else
      Tetrahedron *tet2 = new Tetrahedron(v[1], attr);
#endif
      tet2->ResetTransform(tet->GetTransform());
      elements.Append(tet2);

      // record the sequence of refinements
      tet->PushTransform(tr1);
      tet2->PushTransform(tr2);

      int coarse = FindCoarseElement(i);
      CoarseFineTr.embeddings[i].parent = coarse;
      CoarseFineTr.embeddings.Append(Embedding(coarse));

      // 3. Set the bisection flag
      switch (type)
      {
         case Tetrahedron::TYPE_PU:
            new_type = Tetrahedron::TYPE_PF; break;
         case Tetrahedron::TYPE_PF:
            new_type = Tetrahedron::TYPE_A;  break;
         default:
            new_type = Tetrahedron::TYPE_PU;
      }

      tet->CreateRefinementFlag(new_redges[0], new_type, flag+1);
      tet2->CreateRefinementFlag(new_redges[1], new_type, flag+1);

      NumOfElements++;
   }
   else
   {
      MFEM_ABORT("Bisection for now works only for triangles & tetrahedra.");
   }
}

void Mesh::Bisection(int i, const DSTable &v_to_v, int *middle)
{
   int *vert;
   int v[2][3], v_new, bisect, t;
   Element *bdr_el = boundary[i];

   t = bdr_el->GetType();
   if (t == Element::TRIANGLE)
   {
      Triangle *tri = (Triangle *) bdr_el;

      vert = tri->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v(vert[0], vert[1]);
      MFEM_ASSERT(bisect >= 0, "");
      v_new = middle[bisect];
      MFEM_ASSERT(v_new != -1, "");

      // 2. Set the node indices for the new elements in v[0] and v[1] so that
      //    the  edge marked for refinement is between the first two nodes.
      v[0][0] = vert[2]; v[0][1] = vert[0]; v[0][2] = v_new;
      v[1][0] = vert[1]; v[1][1] = vert[2]; v[1][2] = v_new;

      tri->SetVertices(v[0]);

      boundary.Append(new Triangle(v[1], tri->GetAttribute()));

      NumOfBdrElements++;
   }
   else
   {
      MFEM_ABORT("Bisection of boundary elements works only for triangles!");
   }
}

void Mesh::RedRefinementPentatope(int i, const DSTable & v_to_v, int *middle)
{
	if (elements[i]->GetType() != Element::PENTATOPE) mfem_error("RedRefinementPentatope: Element must be a pentatope!");

	int w[5];
	int v_new[10], bisect[10];
	const int *ei;
	Vertex V;

	bool swapped = swappedElements[i];

	int *v = elements[i]->GetVertices();
	if(swappedElements[i]) Swap(v);

	for(int j = 0; j < 10; j++)
	{
		ei = elements[i]->GetEdgeVertices(j);
		bisect[j] = v_to_v(v[ei[0]],v[ei[1]]);

		if(middle[bisect[j]] == -1)
		{
			v_new[j] = NumOfVertices++;
			for (int d = 0; d < spaceDim; d++) V(d) = 0.5*(vertices[v[ei[0]]](d) + vertices[v[ei[1]]](d));
			vertices.Append(V);

			middle[bisect[j]] = v_new[j];
        }
        else
        {
           v_new[j] = middle[bisect[j]];
        }
	}

	int attr = elements[i]->GetAttribute();

//	w[0] = v[0]; w[1] = v_new[0]; w[2] = v_new[1]; w[3] = v_new[2]; w[4] = v_new[3]; elements.Append(new Pentatope(w, attr));
//	w[0] = v[1]; w[1] = v_new[0]; w[2] = v_new[4]; w[3] = v_new[5]; w[4] = v_new[6]; elements.Append(new Pentatope(w, attr));
//	w[0] = v[2]; w[1] = v_new[1]; w[2] = v_new[4]; w[3] = v_new[7]; w[4] = v_new[8]; elements.Append(new Pentatope(w, attr));
//	w[0] = v[3]; w[1] = v_new[2]; w[2] = v_new[5]; w[3] = v_new[7]; w[4] = v_new[9]; elements.Append(new Pentatope(w, attr));
//	w[0] = v[4]; w[1] = v_new[3]; w[2] = v_new[6]; w[3] = v_new[8]; w[4] = v_new[9]; elements.Append(new Pentatope(w, attr));

	bool mySwaped;
	w[0] = v[0];     w[1] = v_new[0]; w[2] = v_new[1]; w[3] = v_new[2]; w[4] = v_new[3]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[0]; w[1] = v[1];     w[2] = v_new[4]; w[3] = v_new[5]; w[4] = v_new[6]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[1]; w[1] = v_new[4]; w[2] = v[2];     w[3] = v_new[7]; w[4] = v_new[8]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[2]; w[1] = v_new[5]; w[2] = v_new[7]; w[3] = v[3];     w[4] = v_new[9]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[3]; w[1] = v_new[6]; w[2] = v_new[8]; w[3] = v_new[9]; w[4] = v[4];     mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);

	w[0] = v_new[0]; w[1] = v_new[1]; w[2] = v_new[4]; w[3] = v_new[5]; w[4] = v_new[6]; mySwaped = !swapped; if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[0]; w[1] = v_new[1]; w[2] = v_new[2]; w[3] = v_new[5]; w[4] = v_new[6]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[0]; w[1] = v_new[1]; w[2] = v_new[2]; w[3] = v_new[3]; w[4] = v_new[6]; mySwaped = !swapped; if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);

	w[0] = v_new[1]; w[1] = v_new[4]; w[2] = v_new[5]; w[3] = v_new[7]; w[4] = v_new[8]; mySwaped = !swapped; if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[1]; w[1] = v_new[4]; w[2] = v_new[5]; w[3] = v_new[6]; w[4] = v_new[8]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[1]; w[1] = v_new[2]; w[2] = v_new[5]; w[3] = v_new[7]; w[4] = v_new[8]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[1]; w[1] = v_new[2]; w[2] = v_new[5]; w[3] = v_new[6]; w[4] = v_new[8]; mySwaped = !swapped; if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[1]; w[1] = v_new[2]; w[2] = v_new[3]; w[3] = v_new[6]; w[4] = v_new[8]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);

	w[0] = v_new[2]; w[1] = v_new[5]; w[2] = v_new[7]; w[3] = v_new[8]; w[4] = v_new[9]; mySwaped = !swapped; if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[2]; w[1] = v_new[5]; w[2] = v_new[6]; w[3] = v_new[8]; w[4] = v_new[9]; mySwaped = swapped;  if(mySwaped) Swap(w); elements.Append(new Pentatope(w, attr)); swappedElements.Append(mySwaped);
	w[0] = v_new[2]; w[1] = v_new[3]; w[2] = v_new[6]; w[3] = v_new[8]; w[4] = v_new[9]; mySwaped = !swapped; if(mySwaped) Swap(w); elements[i]->SetVertices(w);             swappedElements[i] = mySwaped;

//	DenseMatrix J(4,4);
//	for(int k=0; k<16; k++)
//	{
//		int elID = NumOfElements + k;
//		if(k==15) elID = i;
//
//		GetElementJacobian(elID, J);
//		if(J.Det() < 0 && true)
//		{
//			cout << "Jacobian is negative!" << endl;
//		}
//	}

	NumOfElements += 15;

}

void Mesh::RedRefinementBoundaryTet(int i, const DSTable & v_to_v, int *middle)
{
	if (boundary[i]->GetType() != Element::TETRAHEDRON) mfem_error("RedRefinementBoundaryTet: Element must be a tetrahedron!");

	Array<int> vold;
	int w[4];
	int v_new[6], bisect[6];
	const int *ei;
	Vertex V;

//	   int geom = GetElementBaseGeometry(i);
//	   ElementTransformation *eltransf = GetBdrElementTransformation(i);
//	   eltransf->SetIntPoint(&Geometries.GetCenter(geom));
//	   DenseMatrix J = eltransf->Jacobian();
//	   Vector n(Dim); CalcOrtho(J, n);
//
//	   cout << n[0] << " " << n[1] << " " << n[2] << " " << n[3] << endl;



	bool swapped = swappedFaces[be_to_face[i]];
	int *v = boundary[i]->GetVertices();
	if(swapped) Swap(v);

//	cout << swapped << endl << " my computed " << endl;

	for(int j = 0; j < 6; j++)
	{
		ei = boundary[i]->GetEdgeVertices(j);
		bisect[j] = v_to_v(v[ei[0]],v[ei[1]]);

		if(middle[bisect[j]] == -1)
		{
			v_new[j] = NumOfVertices++;
			for (int d = 0; d < spaceDim; d++) V(d) = 0.5*(vertices[v[ei[0]]](d) + vertices[v[ei[1]]](d));
			vertices.Append(V);

			middle[bisect[j]] = v_new[j];
		}
		else
		{
		   v_new[j] = middle[bisect[j]];
		}
	}

	int attr = boundary[i]->GetAttribute();


	bool mySwaped;
	w[0] = v[0];     w[1] = v_new[0]; w[2] = v_new[1]; w[3] = v_new[2]; mySwaped = swapped; /*cout << mySwaped << endl;*/ if(mySwaped) Swap(w); boundary.Append(new Tetrahedron(w, attr));
	w[0] = v_new[0]; w[1] = v[1];     w[2] = v_new[3]; w[3] = v_new[4]; mySwaped = swapped; /*if(mySwaped) Swap(w); */ if(mySwaped) Swap(w); boundary.Append(new Tetrahedron(w, attr));
	w[0] = v_new[1]; w[1] = v_new[3]; w[2] = v[2];     w[3] = v_new[5]; mySwaped = swapped; /*if(mySwaped) Swap(w); */ if(mySwaped) Swap(w); boundary.Append(new Tetrahedron(w, attr));
	w[0] = v_new[2]; w[1] = v_new[4]; w[2] = v_new[5]; w[3] = v[3];     mySwaped = swapped; /*if(mySwaped) Swap(w); */if(mySwaped) Swap(w);  boundary.Append(new Tetrahedron(w, attr));

	w[0] = v_new[0]; w[1] = v_new[1]; w[2] = v_new[3]; w[3] = v_new[4]; mySwaped = !swapped;/*if(mySwaped) Swap(w); */ if(mySwaped) Swap(w); boundary.Append(new Tetrahedron(w, attr));
	w[0] = v_new[0]; w[1] = v_new[1]; w[2] = v_new[2]; w[3] = v_new[4]; mySwaped = swapped; /*if(mySwaped) Swap(w); */ if(mySwaped) Swap(w); boundary.Append(new Tetrahedron(w, attr));
	w[0] = v_new[1]; w[1] = v_new[3]; w[2] = v_new[4]; w[3] = v_new[5]; mySwaped = !swapped;/*if(mySwaped) Swap(w); */ if(mySwaped) Swap(w); boundary.Append(new Tetrahedron(w, attr));
	w[0] = v_new[1]; w[1] = v_new[2]; w[2] = v_new[4]; w[3] = v_new[5]; mySwaped = swapped; /*if(mySwaped) Swap(w); */ if(mySwaped) Swap(w); boundary[i]->SetVertices(w);

//	cout << endl;

	NumOfBdrElements += 7;
}

void Mesh::UniformRefinement(int i, const DSTable &v_to_v,
                             int *edge1, int *edge2, int *middle)
{
   Array<int> v;
   int j, v1[3], v2[3], v3[3], v4[3], v_new[3], bisect[3];
   Vertex V;

   if (elements[i]->GetType() == Element::TRIANGLE)
   {
      Triangle *tri0 = (Triangle*) elements[i];
      tri0->GetVertices(v);

      // 1. Get the indeces for the new vertices in array v_new
      bisect[0] = v_to_v(v[0],v[1]);
      bisect[1] = v_to_v(v[1],v[2]);
      bisect[2] = v_to_v(v[0],v[2]);
      MFEM_ASSERT(bisect[0] >= 0 && bisect[1] >= 0 && bisect[2] >= 0, "");

      for (j = 0; j < 3; j++)                // for the 3 edges fix v_new
      {
         if (middle[bisect[j]] == -1)
         {
            v_new[j] = NumOfVertices++;
            for (int d = 0; d < spaceDim; d++)
            {
               V(d) = (vertices[v[j]](d) + vertices[v[(j+1)%3]](d))/2.;
            }
            vertices.Append(V);

            // Put the element that may need refinement (because of this
            // bisection) in edge1, or -1 if no more refinement is needed.
            if (edge1[bisect[j]] == i)
            {
               edge1[bisect[j]] = edge2[bisect[j]];
            }

            middle[bisect[j]] = v_new[j];
         }
         else
         {
            v_new[j] = middle[bisect[j]];

            // This edge will require no more refinement.
            edge1[bisect[j]] = -1;
         }
      }

      // 2. Set the node indeces for the new elements in v1, v2, v3 & v4 so that
      //    the edges marked for refinement be between the first two nodes.
      v1[0] =     v[0]; v1[1] = v_new[0]; v1[2] = v_new[2];
      v2[0] = v_new[0]; v2[1] =     v[1]; v2[2] = v_new[1];
      v3[0] = v_new[2]; v3[1] = v_new[1]; v3[2] =     v[2];
      v4[0] = v_new[1]; v4[1] = v_new[2]; v4[2] = v_new[0];

      Triangle* tri1 = new Triangle(v1, tri0->GetAttribute());
      Triangle* tri2 = new Triangle(v2, tri0->GetAttribute());
      Triangle* tri3 = new Triangle(v3, tri0->GetAttribute());

      elements.Append(tri1);
      elements.Append(tri2);
      elements.Append(tri3);

      tri0->SetVertices(v4);

      // record the sequence of refinements
      unsigned code = tri0->GetTransform();
      tri1->ResetTransform(code);
      tri2->ResetTransform(code);
      tri3->ResetTransform(code);

      tri0->PushTransform(3);
      tri1->PushTransform(0);
      tri2->PushTransform(1);
      tri3->PushTransform(2);

      // set parent indices
      int coarse = FindCoarseElement(i);
      CoarseFineTr.embeddings[i] = Embedding(coarse);
      CoarseFineTr.embeddings.Append(Embedding(coarse));
      CoarseFineTr.embeddings.Append(Embedding(coarse));
      CoarseFineTr.embeddings.Append(Embedding(coarse));

      NumOfElements += 3;
   }
   else
   {
      MFEM_ABORT("Uniform refinement for now works only for triangles.");
   }
}

void Mesh::InitRefinementTransforms()
{
   // initialize CoarseFineTr
   CoarseFineTr.point_matrices.SetSize(0, 0, 0);
   CoarseFineTr.embeddings.SetSize(NumOfElements);
   for (int i = 0; i < NumOfElements; i++)
   {
      elements[i]->ResetTransform(0);
      CoarseFineTr.embeddings[i] = Embedding(i);
   }
}

int Mesh::FindCoarseElement(int i)
{
   int coarse;
   while ((coarse = CoarseFineTr.embeddings[i].parent) != i)
   {
      i = coarse;
   }
   return coarse;
}

const CoarseFineTransformations& Mesh::GetRefinementTransforms()
{
   MFEM_VERIFY(GetLastOperation() == Mesh::REFINE, "");

   if (ncmesh)
   {
      return ncmesh->GetRefinementTransforms();
   }

   if (!CoarseFineTr.point_matrices.SizeK())
   {
      if (BaseGeom == Geometry::TRIANGLE ||
          BaseGeom == Geometry::TETRAHEDRON)
      {
         std::map<unsigned, int> mat_no;
         mat_no[0] = 1; // identity

         // assign matrix indices to element transformations
         for (int i = 0; i < elements.Size(); i++)
         {
            int index = 0;
            unsigned code = elements[i]->GetTransform();
            if (code)
            {
               int &matrix = mat_no[code];
               if (!matrix) { matrix = mat_no.size(); }
               index = matrix-1;
            }
            CoarseFineTr.embeddings[i].matrix = index;
         }

         DenseTensor &pmats = CoarseFineTr.point_matrices;
         pmats.SetSize(Dim, Dim+1, mat_no.size());

         // calculate the point matrices used
         std::map<unsigned, int>::iterator it;
         for (it = mat_no.begin(); it != mat_no.end(); ++it)
         {
            if (BaseGeom == Geometry::TRIANGLE)
            {
               Triangle::GetPointMatrix(it->first, pmats(it->second-1));
            }
            else
            {
               Tetrahedron::GetPointMatrix(it->first, pmats(it->second-1));
            }
         }
      }
      else
      {
         MFEM_ABORT("Don't know how to construct CoarseFineTr.");
      }
   }

   // NOTE: quads and hexes already have trivial transformations ready
   return CoarseFineTr;
}

void Mesh::PrintXG(std::ostream &out) const
{
   MFEM_ASSERT(Dim==spaceDim, "2D Manifold meshes not supported");
   int i, j;
   Array<int> v;

   if (Dim == 2)
   {
      // Print the type of the mesh.
      if (Nodes == NULL)
      {
         out << "areamesh2\n\n";
      }
      else
      {
         out << "curved_areamesh2\n\n";
      }

      // Print the boundary elements.
      out << NumOfBdrElements << '\n';
      for (i = 0; i < NumOfBdrElements; i++)
      {
         boundary[i]->GetVertices(v);

         out << boundary[i]->GetAttribute();
         for (j = 0; j < v.Size(); j++)
         {
            out << ' ' << v[j] + 1;
         }
         out << '\n';
      }

      // Print the elements.
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         elements[i]->GetVertices(v);

         out << elements[i]->GetAttribute() << ' ' << v.Size();
         for (j = 0; j < v.Size(); j++)
         {
            out << ' ' << v[j] + 1;
         }
         out << '\n';
      }

      if (Nodes == NULL)
      {
         // Print the vertices.
         out << NumOfVertices << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            out << vertices[i](0);
            for (j = 1; j < Dim; j++)
            {
               out << ' ' << vertices[i](j);
            }
            out << '\n';
         }
      }
      else
      {
         out << NumOfVertices << '\n';
         Nodes->Save(out);
      }
   }
   else  // ===== Dim != 2 =====
   {
      if (Nodes)
      {
         mfem_error("Mesh::PrintXG(...) : Curved mesh in 3D");
      }

      if (meshgen == 1)
      {
         int nv;
         const int *ind;

         out << "NETGEN_Neutral_Format\n";
         // print the vertices
         out << NumOfVertices << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               out << ' ' << vertices[i](j);
            }
            out << '\n';
         }

         // print the elements
         out << NumOfElements << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            out << elements[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               out << ' ' << ind[j]+1;
            }
            out << '\n';
         }

         // print the boundary information.
         out << NumOfBdrElements << '\n';
         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            out << boundary[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               out << ' ' << ind[j]+1;
            }
            out << '\n';
         }
      }
      else if (meshgen == 2)  // TrueGrid
      {
         int nv;
         const int *ind;

         out << "TrueGrid\n"
             << "1 " << NumOfVertices << " " << NumOfElements
             << " 0 0 0 0 0 0 0\n"
             << "0 0 0 1 0 0 0 0 0 0 0\n"
             << "0 0 " << NumOfBdrElements << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
             << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
             << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

         for (i = 0; i < NumOfVertices; i++)
            out << i+1 << " 0.0 " << vertices[i](0) << ' ' << vertices[i](1)
                << ' ' << vertices[i](2) << " 0.0\n";

         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            out << i+1 << ' ' << elements[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               out << ' ' << ind[j]+1;
            }
            out << '\n';
         }

         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            out << boundary[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               out << ' ' << ind[j]+1;
            }
            out << " 1.0 1.0 1.0 1.0\n";
         }
      }
   }

   out << flush;
}

void Mesh::Printer(std::ostream &out, std::string section_delimiter) const
{
   int i, j;

   if (NURBSext)
   {
      // general format
      NURBSext->Print(out);
      out << '\n';
      Nodes->Save(out);

      // patch-wise format
      // NURBSext->ConvertToPatches(*Nodes);
      // NURBSext->Print(out);

      return;
   }

   out << (ncmesh ? "MFEM mesh v1.1\n" :
           section_delimiter.empty() ? "MFEM mesh v1.0\n" :
           "MFEM mesh v1.2\n");

   // optional
   out <<
       "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
       "# POINT       = 0\n"
       "# SEGMENT     = 1\n"
       "# TRIANGLE    = 2\n"
       "# SQUARE      = 3\n"
       "# TETRAHEDRON = 4\n"
       "# CUBE        = 5\n"
       "#\n";

   out << "\ndimension\n" << Dim
       << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], out);
   }

   out << "\nboundary\n" << NumOfBdrElements << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
   {
      PrintElement(boundary[i], out);
   }

   if (ncmesh)
   {
      out << "\nvertex_parents\n";
      ncmesh->PrintVertexParents(out);

      out << "\ncoarse_elements\n";
      ncmesh->PrintCoarseElements(out);
   }

   out << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      out << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            out << ' ' << vertices[i](j);
         }
         out << '\n';
      }
      out.flush();
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
   }

   if (!ncmesh && !section_delimiter.empty())
   {
      out << section_delimiter << endl; // only with format v1.2
   }
}

void Mesh::PrintTopo(std::ostream &out,const Array<int> &e_to_k) const
{
   int i;
   Array<int> vert;

   out << "MFEM NURBS mesh v1.0\n";

   // optional
   out <<
       "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
       "# SEGMENT     = 1\n"
       "# SQUARE      = 3\n"
       "# CUBE        = 5\n"
       "#\n";

   out << "\ndimension\n" << Dim
       << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], out);
   }

   out << "\nboundary\n" << NumOfBdrElements << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
   {
      PrintElement(boundary[i], out);
   }

   out << "\nedges\n" << NumOfEdges << '\n';
   for (i = 0; i < NumOfEdges; i++)
   {
      edge_vertex->GetRow(i, vert);
      int ki = e_to_k[i];
      if (ki < 0)
      {
         ki = -1 - ki;
      }
      out << ki << ' ' << vert[0] << ' ' << vert[1] << '\n';
   }
   out << "\nvertices\n" << NumOfVertices << '\n';
}

void Mesh::PrintVTK(std::ostream &out)
{
   out <<
       "# vtk DataFile Version 3.0\n"
       "Generated by MFEM\n"
       "ASCII\n"
       "DATASET UNSTRUCTURED_GRID\n";

   if (Nodes == NULL)
   {
      out << "POINTS " << NumOfVertices << " double\n";
      for (int i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         int j;
         for (j = 1; j < spaceDim; j++)
         {
            out << ' ' << vertices[i](j);
         }
         for ( ; j < 3; j++)
         {
            out << ' ' << 0.0;
         }
         out << '\n';
      }
   }
   else
   {
      Array<int> vdofs(3);
      out << "POINTS " << Nodes->FESpace()->GetNDofs() << " double\n";
      for (int i = 0; i < Nodes->FESpace()->GetNDofs(); i++)
      {
         vdofs.SetSize(1);
         vdofs[0] = i;
         Nodes->FESpace()->DofsToVDofs(vdofs);
         out << (*Nodes)(vdofs[0]);
         int j;
         for (j = 1; j < spaceDim; j++)
         {
            out << ' ' << (*Nodes)(vdofs[j]);
         }
         for ( ; j < 3; j++)
         {
            out << ' ' << 0.0;
         }
         out << '\n';
      }
   }

   int order = -1;
   if (Nodes == NULL)
   {
      int size = 0;
      for (int i = 0; i < NumOfElements; i++)
      {
         size += elements[i]->GetNVertices() + 1;
      }
      out << "CELLS " << NumOfElements << ' ' << size << '\n';
      for (int i = 0; i < NumOfElements; i++)
      {
         const int *v = elements[i]->GetVertices();
         const int nv = elements[i]->GetNVertices();
         out << nv;
         for (int j = 0; j < nv; j++)
         {
            out << ' ' << v[j];
         }
         out << '\n';
      }
      order = 1;
   }
   else
   {
      Array<int> dofs;
      int size = 0;
      for (int i = 0; i < NumOfElements; i++)
      {
         Nodes->FESpace()->GetElementDofs(i, dofs);
         size += dofs.Size() + 1;
      }
      out << "CELLS " << NumOfElements << ' ' << size << '\n';
      const char *fec_name = Nodes->FESpace()->FEColl()->Name();
      if (!strcmp(fec_name, "Linear") ||
          !strcmp(fec_name, "H1_2D_P1") ||
          !strcmp(fec_name, "H1_3D_P1"))
      {
         order = 1;
      }
      else if (!strcmp(fec_name, "Quadratic") ||
               !strcmp(fec_name, "H1_2D_P2") ||
               !strcmp(fec_name, "H1_3D_P2"))
      {
         order = 2;
      }
      if (order == -1)
      {
         cerr << "Mesh::PrintVTK : can not save '"
              << fec_name << "' elements!" << endl;
         mfem_error();
      }
      for (int i = 0; i < NumOfElements; i++)
      {
         Nodes->FESpace()->GetElementDofs(i, dofs);
         out << dofs.Size();
         if (order == 1)
         {
            for (int j = 0; j < dofs.Size(); j++)
            {
               out << ' ' << dofs[j];
            }
         }
         else if (order == 2)
         {
            const int *vtk_mfem;
            switch (elements[i]->GetGeometryType())
            {
               case Geometry::TRIANGLE:
               case Geometry::SQUARE:
                  vtk_mfem = vtk_quadratic_hex; break; // identity map
               case Geometry::TETRAHEDRON:
                  vtk_mfem = vtk_quadratic_tet; break;
               case Geometry::CUBE:
               default:
                  vtk_mfem = vtk_quadratic_hex; break;
            }
            for (int j = 0; j < dofs.Size(); j++)
            {
               out << ' ' << dofs[vtk_mfem[j]];
            }
         }
         out << '\n';
      }
   }

   out << "CELL_TYPES " << NumOfElements << '\n';
   for (int i = 0; i < NumOfElements; i++)
   {
      int vtk_cell_type = 5;
      if (order == 1)
      {
         switch (elements[i]->GetGeometryType())
         {
            case Geometry::TRIANGLE:     vtk_cell_type = 5;   break;
            case Geometry::SQUARE:       vtk_cell_type = 9;   break;
            case Geometry::TETRAHEDRON:  vtk_cell_type = 10;  break;
            case Geometry::CUBE:         vtk_cell_type = 12;  break;
         }
      }
      else if (order == 2)
      {
         switch (elements[i]->GetGeometryType())
         {
            case Geometry::TRIANGLE:     vtk_cell_type = 22;  break;
            case Geometry::SQUARE:       vtk_cell_type = 28;  break;
            case Geometry::TETRAHEDRON:  vtk_cell_type = 24;  break;
            case Geometry::CUBE:         vtk_cell_type = 29;  break;
         }
      }

      out << vtk_cell_type << '\n';
   }

   // write attributes
   out << "CELL_DATA " << NumOfElements << '\n'
       << "SCALARS material int\n"
       << "LOOKUP_TABLE default\n";
   for (int i = 0; i < NumOfElements; i++)
   {
      out << elements[i]->GetAttribute() << '\n';
   }
   out.flush();
}

void Mesh::PrintVTK(std::ostream &out, int ref, int field_data)
{
   int np, nc, size;
   RefinedGeometry *RefG;
   DenseMatrix pmat;

   out <<
       "# vtk DataFile Version 3.0\n"
       "Generated by MFEM\n"
       "ASCII\n"
       "DATASET UNSTRUCTURED_GRID\n";

   // additional dataset information
   if (field_data)
   {
      out << "FIELD FieldData 1\n"
          << "MaterialIds " << 1 << " " << attributes.Size() << " int\n";
      for (int i = 0; i < attributes.Size(); i++)
      {
         out << ' ' << attributes[i];
      }
      out << '\n';
   }

   // count the points, cells, size
   np = nc = size = 0;
   for (int i = 0; i < GetNE(); i++)
   {
      int geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      np += RefG->RefPts.GetNPoints();
      nc += RefG->RefGeoms.Size() / nv;
      size += (RefG->RefGeoms.Size() / nv) * (nv + 1);
   }
   out << "POINTS " << np << " double\n";
   // write the points
   for (int i = 0; i < GetNE(); i++)
   {
      RefG = GlobGeometryRefiner.Refine(
                GetElementBaseGeometry(i), ref, 1);

      GetElementTransformation(i)->Transform(RefG->RefPts, pmat);

      for (int j = 0; j < pmat.Width(); j++)
      {
         out << pmat(0, j) << ' ';
         if (pmat.Height() > 1)
         {
            out << pmat(1, j) << ' ';
            if (pmat.Height() > 2)
            {
               out << pmat(2, j);
            }
            else
            {
               out << 0.0;
            }
         }
         else
         {
            out << 0.0 << ' ' << 0.0;
         }
         out << '\n';
      }
   }

   // write the cells
   out << "CELLS " << nc << ' ' << size << '\n';
   np = 0;
   for (int i = 0; i < GetNE(); i++)
   {
      int geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      Array<int> &RG = RefG->RefGeoms;

      for (int j = 0; j < RG.Size(); )
      {
         out << nv;
         for (int k = 0; k < nv; k++, j++)
         {
            out << ' ' << np + RG[j];
         }
         out << '\n';
      }
      np += RefG->RefPts.GetNPoints();
   }
   out << "CELL_TYPES " << nc << '\n';
   for (int i = 0; i < GetNE(); i++)
   {
      int geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      Array<int> &RG = RefG->RefGeoms;
      int vtk_cell_type = 5;

      switch (geom)
      {
         case Geometry::SEGMENT:      vtk_cell_type = 3;   break;
         case Geometry::TRIANGLE:     vtk_cell_type = 5;   break;
         case Geometry::SQUARE:       vtk_cell_type = 9;   break;
         case Geometry::TETRAHEDRON:  vtk_cell_type = 10;  break;
         case Geometry::CUBE:         vtk_cell_type = 12;  break;
      }

      for (int j = 0; j < RG.Size(); j += nv)
      {
         out << vtk_cell_type << '\n';
      }
   }
   // write attributes (materials)
   out << "CELL_DATA " << nc << '\n'
       << "SCALARS material int\n"
       << "LOOKUP_TABLE default\n";
   for (int i = 0; i < GetNE(); i++)
   {
      int geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      int attr = GetAttribute(i);
      for (int j = 0; j < RefG->RefGeoms.Size(); j += nv)
      {
         out << attr << '\n';
      }
   }

   Array<int> coloring;
   srand((unsigned)time(0));
   double a = double(rand()) / (double(RAND_MAX) + 1.);
   int el0 = (int)floor(a * GetNE());
   GetElementColoring(coloring, el0);
   out << "SCALARS element_coloring int\n"
       << "LOOKUP_TABLE default\n";
   for (int i = 0; i < GetNE(); i++)
   {
      int geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      for (int j = 0; j < RefG->RefGeoms.Size(); j += nv)
      {
         out << coloring[i] + 1 << '\n';
      }
   }
   // prepare to write data
   out << "POINT_DATA " << np << '\n' << flush;
}

void Mesh::GetElementColoring(Array<int> &colors, int el0)
{
   int delete_el_to_el = (el_to_el) ? (0) : (1);
   const Table &el_el = ElementToElementTable();
   int num_el = GetNE(), stack_p, stack_top_p, max_num_col;
   Array<int> el_stack(num_el);

   const int *i_el_el = el_el.GetI();
   const int *j_el_el = el_el.GetJ();

   colors.SetSize(num_el);
   colors = -2;
   max_num_col = 1;
   stack_p = stack_top_p = 0;
   for (int el = el0; stack_top_p < num_el; el=(el+1)%num_el)
   {
      if (colors[el] != -2)
      {
         continue;
      }

      colors[el] = -1;
      el_stack[stack_top_p++] = el;

      for ( ; stack_p < stack_top_p; stack_p++)
      {
         int i = el_stack[stack_p];
         int num_nb = i_el_el[i+1] - i_el_el[i];
         if (max_num_col < num_nb + 1)
         {
            max_num_col = num_nb + 1;
         }
         for (int j = i_el_el[i]; j < i_el_el[i+1]; j++)
         {
            int k = j_el_el[j];
            if (colors[k] == -2)
            {
               colors[k] = -1;
               el_stack[stack_top_p++] = k;
            }
         }
      }
   }

   Array<int> col_marker(max_num_col);

   for (stack_p = 0; stack_p < stack_top_p; stack_p++)
   {
      int i = el_stack[stack_p], col;
      col_marker = 0;
      for (int j = i_el_el[i]; j < i_el_el[i+1]; j++)
      {
         col = colors[j_el_el[j]];
         if (col != -1)
         {
            col_marker[col] = 1;
         }
      }

      for (col = 0; col < max_num_col; col++)
         if (col_marker[col] == 0)
         {
            break;
         }

      colors[i] = col;
   }

   if (delete_el_to_el)
   {
      delete el_to_el;
      el_to_el = NULL;
   }
}

void Mesh::PrintWithPartitioning(int *partitioning, std::ostream &out,
                                 int elem_attr) const
{
   if (Dim != 3 && Dim != 2) { return; }

   int i, j, k, l, nv, nbe, *v;

   out << "MFEM mesh v1.0\n";

   // optional
   out <<
       "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
       "# POINT       = 0\n"
       "# SEGMENT     = 1\n"
       "# TRIANGLE    = 2\n"
       "# SQUARE      = 3\n"
       "# TETRAHEDRON = 4\n"
       "# CUBE        = 5\n"
       "#\n";

   out << "\ndimension\n" << Dim
       << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      out << int((elem_attr) ? partitioning[i]+1 : elements[i]->GetAttribute())
          << ' ' << elements[i]->GetGeometryType();
      nv = elements[i]->GetNVertices();
      v  = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
      {
         out << ' ' << v[j];
      }
      out << '\n';
   }
   nbe = 0;
   for (i = 0; i < faces_info.Size(); i++)
   {
      if ((l = faces_info[i].Elem2No) >= 0)
      {
         k = partitioning[faces_info[i].Elem1No];
         l = partitioning[l];
         if (k != l)
         {
            nbe++;
            if (!Nonconforming() || !IsSlaveFace(faces_info[i]))
            {
               nbe++;
            }
         }
      }
      else
      {
         nbe++;
      }
   }
   out << "\nboundary\n" << nbe << '\n';
   for (i = 0; i < faces_info.Size(); i++)
   {
      if ((l = faces_info[i].Elem2No) >= 0)
      {
         k = partitioning[faces_info[i].Elem1No];
         l = partitioning[l];
         if (k != l)
         {
            nv = faces[i]->GetNVertices();
            v  = faces[i]->GetVertices();
            out << k+1 << ' ' << faces[i]->GetGeometryType();
            for (j = 0; j < nv; j++)
            {
               out << ' ' << v[j];
            }
            out << '\n';
            if (!Nonconforming() || !IsSlaveFace(faces_info[i]))
            {
               out << l+1 << ' ' << faces[i]->GetGeometryType();
               for (j = nv-1; j >= 0; j--)
               {
                  out << ' ' << v[j];
               }
               out << '\n';
            }
         }
      }
      else
      {
         k = partitioning[faces_info[i].Elem1No];
         nv = faces[i]->GetNVertices();
         v  = faces[i]->GetVertices();
         out << k+1 << ' ' << faces[i]->GetGeometryType();
         for (j = 0; j < nv; j++)
         {
            out << ' ' << v[j];
         }
         out << '\n';
      }
   }
   out << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      out << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            out << ' ' << vertices[i](j);
         }
         out << '\n';
      }
      out.flush();
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
   }
}

void Mesh::PrintElementsWithPartitioning(int *partitioning,
                                         std::ostream &out,
                                         int interior_faces)
{
   MFEM_ASSERT(Dim == spaceDim, "2D Manifolds not supported\n");
   if (Dim != 3 && Dim != 2) { return; }

   int i, j, k, l, s;

   int nv;
   const int *ind;

   int *vcount = new int[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
   {
      vcount[i] = 0;
   }
   for (i = 0; i < NumOfElements; i++)
   {
      nv = elements[i]->GetNVertices();
      ind = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
      {
         vcount[ind[j]]++;
      }
   }

   int *voff = new int[NumOfVertices+1];
   voff[0] = 0;
   for (i = 1; i <= NumOfVertices; i++)
   {
      voff[i] = vcount[i-1] + voff[i-1];
   }

   int **vown = new int*[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
   {
      vown[i] = new int[vcount[i]];
   }

   // 2D
   if (Dim == 2)
   {
      int nv, nbe;
      int *ind;

      Table edge_el;
      Transpose(ElementToEdgeTable(), edge_el);

      // Fake printing of the elements.
      for (i = 0; i < NumOfElements; i++)
      {
         nv  = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         for (j = 0; j < nv; j++)
         {
            vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
      }

      for (i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      nbe = 0;
      for (i = 0; i < edge_el.Size(); i++)
      {
         const int *el = edge_el.GetRow(i);
         if (edge_el.RowSize(i) > 1)
         {
            k = partitioning[el[0]];
            l = partitioning[el[1]];
            if (interior_faces || k != l)
            {
               nbe += 2;
            }
         }
         else
         {
            nbe++;
         }
      }

      // Print the type of the mesh and the boundary elements.
      out << "areamesh2\n\n" << nbe << '\n';

      for (i = 0; i < edge_el.Size(); i++)
      {
         const int *el = edge_el.GetRow(i);
         if (edge_el.RowSize(i) > 1)
         {
            k = partitioning[el[0]];
            l = partitioning[el[1]];
            if (interior_faces || k != l)
            {
               Array<int> ev;
               GetEdgeVertices(i,ev);
               out << k+1; // attribute
               for (j = 0; j < 2; j++)
                  for (s = 0; s < vcount[ev[j]]; s++)
                     if (vown[ev[j]][s] == el[0])
                     {
                        out << ' ' << voff[ev[j]]+s+1;
                     }
               out << '\n';
               out << l+1; // attribute
               for (j = 1; j >= 0; j--)
                  for (s = 0; s < vcount[ev[j]]; s++)
                     if (vown[ev[j]][s] == el[1])
                     {
                        out << ' ' << voff[ev[j]]+s+1;
                     }
               out << '\n';
            }
         }
         else
         {
            k = partitioning[el[0]];
            Array<int> ev;
            GetEdgeVertices(i,ev);
            out << k+1; // attribute
            for (j = 0; j < 2; j++)
               for (s = 0; s < vcount[ev[j]]; s++)
                  if (vown[ev[j]][s] == el[0])
                  {
                     out << ' ' << voff[ev[j]]+s+1;
                  }
            out << '\n';
         }
      }

      // Print the elements.
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         nv  = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         out << partitioning[i]+1 << ' '; // use subdomain number as attribute
         out << nv << ' ';
         for (j = 0; j < nv; j++)
         {
            out << ' ' << voff[ind[j]]+vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
         out << '\n';
      }

      for (i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      // Print the vertices.
      out << voff[NumOfVertices] << '\n';
      for (i = 0; i < NumOfVertices; i++)
         for (k = 0; k < vcount[i]; k++)
         {
            for (j = 0; j < Dim; j++)
            {
               out << vertices[i](j) << ' ';
            }
            out << '\n';
         }
      out << flush;
      return;
   }

   //  Dim is 3
   if (meshgen == 1)
   {
      out << "NETGEN_Neutral_Format\n";
      // print the vertices
      out << voff[NumOfVertices] << '\n';
      for (i = 0; i < NumOfVertices; i++)
         for (k = 0; k < vcount[i]; k++)
         {
            for (j = 0; j < Dim; j++)
            {
               out << ' ' << vertices[i](j);
            }
            out << '\n';
         }

      // print the elements
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         out << partitioning[i]+1; // use subdomain number as attribute
         for (j = 0; j < nv; j++)
         {
            out << ' ' << voff[ind[j]]+vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
         out << '\n';
      }

      for (i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      // print the boundary information.
      int k, l, nbe;
      nbe = 0;
      for (i = 0; i < NumOfFaces; i++)
         if ((l = faces_info[i].Elem2No) >= 0)
         {
            k = partitioning[faces_info[i].Elem1No];
            l = partitioning[l];
            if (interior_faces || k != l)
            {
               nbe += 2;
            }
         }
         else
         {
            nbe++;
         }

      out << nbe << '\n';
      for (i = 0; i < NumOfFaces; i++)
         if ((l = faces_info[i].Elem2No) >= 0)
         {
            k = partitioning[faces_info[i].Elem1No];
            l = partitioning[l];
            if (interior_faces || k != l)
            {
               nv = faces[i]->GetNVertices();
               ind = faces[i]->GetVertices();
               out << k+1; // attribute
               for (j = 0; j < nv; j++)
                  for (s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem1No)
                     {
                        out << ' ' << voff[ind[j]]+s+1;
                     }
               out << '\n';
               out << l+1; // attribute
               for (j = nv-1; j >= 0; j--)
                  for (s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem2No)
                     {
                        out << ' ' << voff[ind[j]]+s+1;
                     }
               out << '\n';
            }
         }
         else
         {
            k = partitioning[faces_info[i].Elem1No];
            nv = faces[i]->GetNVertices();
            ind = faces[i]->GetVertices();
            out << k+1; // attribute
            for (j = 0; j < nv; j++)
               for (s = 0; s < vcount[ind[j]]; s++)
                  if (vown[ind[j]][s] == faces_info[i].Elem1No)
                  {
                     out << ' ' << voff[ind[j]]+s+1;
                  }
            out << '\n';
         }

      for (i = 0; i < NumOfVertices; i++)
      {
         delete [] vown[i];
      }
   }
   else if (meshgen == 2) // TrueGrid
   {
      // count the number of the boundary elements.
      int k, l, nbe;
      nbe = 0;
      for (i = 0; i < NumOfFaces; i++)
         if ((l = faces_info[i].Elem2No) >= 0)
         {
            k = partitioning[faces_info[i].Elem1No];
            l = partitioning[l];
            if (interior_faces || k != l)
            {
               nbe += 2;
            }
         }
         else
         {
            nbe++;
         }


      out << "TrueGrid\n"
          << "1 " << voff[NumOfVertices] << " " << NumOfElements
          << " 0 0 0 0 0 0 0\n"
          << "0 0 0 1 0 0 0 0 0 0 0\n"
          << "0 0 " << nbe << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
          << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
          << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

      for (i = 0; i < NumOfVertices; i++)
         for (k = 0; k < vcount[i]; k++)
            out << voff[i]+k << " 0.0 " << vertices[i](0) << ' '
                << vertices[i](1) << ' ' << vertices[i](2) << " 0.0\n";

      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         out << i+1 << ' ' << partitioning[i]+1; // partitioning as attribute
         for (j = 0; j < nv; j++)
         {
            out << ' ' << voff[ind[j]]+vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
         out << '\n';
      }

      for (i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      // boundary elements
      for (i = 0; i < NumOfFaces; i++)
         if ((l = faces_info[i].Elem2No) >= 0)
         {
            k = partitioning[faces_info[i].Elem1No];
            l = partitioning[l];
            if (interior_faces || k != l)
            {
               nv = faces[i]->GetNVertices();
               ind = faces[i]->GetVertices();
               out << k+1; // attribute
               for (j = 0; j < nv; j++)
                  for (s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem1No)
                     {
                        out << ' ' << voff[ind[j]]+s+1;
                     }
               out << " 1.0 1.0 1.0 1.0\n";
               out << l+1; // attribute
               for (j = nv-1; j >= 0; j--)
                  for (s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem2No)
                     {
                        out << ' ' << voff[ind[j]]+s+1;
                     }
               out << " 1.0 1.0 1.0 1.0\n";
            }
         }
         else
         {
            k = partitioning[faces_info[i].Elem1No];
            nv = faces[i]->GetNVertices();
            ind = faces[i]->GetVertices();
            out << k+1; // attribute
            for (j = 0; j < nv; j++)
               for (s = 0; s < vcount[ind[j]]; s++)
                  if (vown[ind[j]][s] == faces_info[i].Elem1No)
                  {
                     out << ' ' << voff[ind[j]]+s+1;
                  }
            out << " 1.0 1.0 1.0 1.0\n";
         }
   }

   out << flush;

   delete [] vcount;
   delete [] voff;
   delete [] vown;
}

void Mesh::PrintSurfaces(const Table & Aface_face, std::ostream &out) const
{
   int i, j;

   if (NURBSext)
   {
      mfem_error("Mesh::PrintSurfaces"
                 " NURBS mesh is not supported!");
      return;
   }

   out << "MFEM mesh v1.0\n";

   // optional
   out <<
       "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
       "# POINT       = 0\n"
       "# SEGMENT     = 1\n"
       "# TRIANGLE    = 2\n"
       "# SQUARE      = 3\n"
       "# TETRAHEDRON = 4\n"
       "# CUBE        = 5\n"
       "#\n";

   out << "\ndimension\n" << Dim
       << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], out);
   }

   out << "\nboundary\n" << Aface_face.Size_of_connections() << '\n';
   const int * const i_AF_f = Aface_face.GetI();
   const int * const j_AF_f = Aface_face.GetJ();

   for (int iAF=0; iAF < Aface_face.Size(); ++iAF)
      for (const int * iface = j_AF_f + i_AF_f[iAF]; iface < j_AF_f + i_AF_f[iAF+1];
           ++iface)
      {
         out << iAF+1 << ' ';
         PrintElementWithoutAttr(faces[*iface],out);
      }

   out << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      out << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            out << ' ' << vertices[i](j);
         }
         out << '\n';
      }
      out.flush();
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
   }
}

void Mesh::ScaleSubdomains(double sf)
{
   int i,j,k;
   Array<int> vert;
   DenseMatrix pointmat;
   int na = attributes.Size();
   double *cg = new double[na*spaceDim];
   int *nbea = new int[na];

   int *vn = new int[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
   {
      vn[i] = 0;
   }
   for (i = 0; i < na; i++)
   {
      for (j = 0; j < spaceDim; j++)
      {
         cg[i*spaceDim+j] = 0.0;
      }
      nbea[i] = 0;
   }

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementVertices(i, vert);
      for (k = 0; k < vert.Size(); k++)
      {
         vn[vert[k]] = 1;
      }
   }

   for (i = 0; i < NumOfElements; i++)
   {
      int bea = GetAttribute(i)-1;
      GetPointMatrix(i, pointmat);
      GetElementVertices(i, vert);

      for (k = 0; k < vert.Size(); k++)
         if (vn[vert[k]] == 1)
         {
            nbea[bea]++;
            for (j = 0; j < spaceDim; j++)
            {
               cg[bea*spaceDim+j] += pointmat(j,k);
            }
            vn[vert[k]] = 2;
         }
   }

   for (i = 0; i < NumOfElements; i++)
   {
      int bea = GetAttribute(i)-1;
      GetElementVertices (i, vert);

      for (k = 0; k < vert.Size(); k++)
         if (vn[vert[k]])
         {
            for (j = 0; j < spaceDim; j++)
               vertices[vert[k]](j) = sf*vertices[vert[k]](j) +
                                      (1-sf)*cg[bea*spaceDim+j]/nbea[bea];
            vn[vert[k]] = 0;
         }
   }

   delete [] cg;
   delete [] nbea;
   delete [] vn;
}

void Mesh::ScaleElements(double sf)
{
   int i,j,k;
   Array<int> vert;
   DenseMatrix pointmat;
   int na = NumOfElements;
   double *cg = new double[na*spaceDim];
   int *nbea = new int[na];

   int *vn = new int[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
   {
      vn[i] = 0;
   }
   for (i = 0; i < na; i++)
   {
      for (j = 0; j < spaceDim; j++)
      {
         cg[i*spaceDim+j] = 0.0;
      }
      nbea[i] = 0;
   }

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementVertices(i, vert);
      for (k = 0; k < vert.Size(); k++)
      {
         vn[vert[k]] = 1;
      }
   }

   for (i = 0; i < NumOfElements; i++)
   {
      int bea = i;
      GetPointMatrix(i, pointmat);
      GetElementVertices(i, vert);

      for (k = 0; k < vert.Size(); k++)
         if (vn[vert[k]] == 1)
         {
            nbea[bea]++;
            for (j = 0; j < spaceDim; j++)
            {
               cg[bea*spaceDim+j] += pointmat(j,k);
            }
            vn[vert[k]] = 2;
         }
   }

   for (i = 0; i < NumOfElements; i++)
   {
      int bea = i;
      GetElementVertices(i, vert);

      for (k = 0; k < vert.Size(); k++)
         if (vn[vert[k]])
         {
            for (j = 0; j < spaceDim; j++)
               vertices[vert[k]](j) = sf*vertices[vert[k]](j) +
                                      (1-sf)*cg[bea*spaceDim+j]/nbea[bea];
            vn[vert[k]] = 0;
         }
   }

   delete [] cg;
   delete [] nbea;
   delete [] vn;
}

void Mesh::Transform(void (*f)(const Vector&, Vector&))
{
   // TODO: support for different new spaceDim.
   if (Nodes == NULL)
   {
      Vector vold(spaceDim), vnew(NULL, spaceDim);
      for (int i = 0; i < vertices.Size(); i++)
      {
         for (int j = 0; j < spaceDim; j++)
         {
            vold(j) = vertices[i](j);
         }
         vnew.SetData(vertices[i]());
         (*f)(vold, vnew);
      }
   }
   else
   {
      GridFunction xnew(Nodes->FESpace());
      VectorFunctionCoefficient f_pert(spaceDim, f);
      xnew.ProjectCoefficient(f_pert);
      *Nodes = xnew;
   }
}

void Mesh::Transform(VectorCoefficient &deformation)
{
   MFEM_VERIFY(spaceDim == deformation.GetVDim(),
               "incompatible vector dimensions");
   if (Nodes == NULL)
   {
      LinearFECollection fec;
      FiniteElementSpace fes(this, &fec, spaceDim, Ordering::byVDIM);
      GridFunction xnew(&fes);
      xnew.ProjectCoefficient(deformation);
      for (int i = 0; i < NumOfVertices; i++)
         for (int d = 0; d < spaceDim; d++)
         {
            vertices[i](d) = xnew(d + spaceDim*i);
         }
   }
   else
   {
      GridFunction xnew(Nodes->FESpace());
      xnew.ProjectCoefficient(deformation);
      *Nodes = xnew;
   }
}

void Mesh::RemoveUnusedVertices()
{
   if (NURBSext || ncmesh) { return; }

   Array<int> v2v(GetNV());
   v2v = -1;
   for (int i = 0; i < GetNE(); i++)
   {
      Element *el = GetElement(i);
      int nv = el->GetNVertices();
      int *v = el->GetVertices();
      for (int j = 0; j < nv; j++)
      {
         v2v[v[j]] = 0;
      }
   }
   for (int i = 0; i < GetNBE(); i++)
   {
      Element *el = GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v2v[v[j]] = 0;
      }
   }
   int num_vert = 0;
   for (int i = 0; i < v2v.Size(); i++)
   {
      if (v2v[i] == 0)
      {
         vertices[num_vert] = vertices[i];
         v2v[i] = num_vert++;
      }
   }

   if (num_vert == v2v.Size()) { return; }

   Vector nodes_by_element;
   Array<int> vdofs;
   if (Nodes)
   {
      int s = 0;
      for (int i = 0; i < GetNE(); i++)
      {
         Nodes->FESpace()->GetElementVDofs(i, vdofs);
         s += vdofs.Size();
      }
      nodes_by_element.SetSize(s);
      s = 0;
      for (int i = 0; i < GetNE(); i++)
      {
         Nodes->FESpace()->GetElementVDofs(i, vdofs);
         Nodes->GetSubVector(vdofs, &nodes_by_element(s));
         s += vdofs.Size();
      }
   }
   vertices.SetSize(num_vert);
   NumOfVertices = num_vert;
   for (int i = 0; i < GetNE(); i++)
   {
      Element *el = GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   for (int i = 0; i < GetNBE(); i++)
   {
      Element *el = GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   DeleteTables();
   if (Dim > 1)
   {
      // generate el_to_edge, be_to_edge (2D), bel_to_edge (3D)
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   if (Dim > 2)
   {
      // generate el_to_face, be_to_face
      GetElementToFaceTable();
   }
   // Update faces and faces_info
   GenerateFaces();
   if (Nodes)
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
      int s = 0;
      for (int i = 0; i < GetNE(); i++)
      {
         Nodes->FESpace()->GetElementVDofs(i, vdofs);
         Nodes->SetSubVector(vdofs, &nodes_by_element(s));
         s += vdofs.Size();
      }
   }
}

void Mesh::RemoveInternalBoundaries()
{
   if (NURBSext || ncmesh) { return; }

   int num_bdr_elem = 0;
   int new_bel_to_edge_nnz = 0;
   for (int i = 0; i < GetNBE(); i++)
   {
      if (FaceIsInterior(GetBdrElementEdgeIndex(i)))
      {
         FreeElement(boundary[i]);
      }
      else
      {
         num_bdr_elem++;
         if (Dim == 3)
         {
            new_bel_to_edge_nnz += bel_to_edge->RowSize(i);
         }
      }
   }

   if (num_bdr_elem == GetNBE()) { return; }

   Array<Element *> new_boundary(num_bdr_elem);
   Array<int> new_be_to_edge, new_be_to_face;
   Table *new_bel_to_edge = NULL;
   new_boundary.SetSize(0);
   if (Dim == 2)
   {
      new_be_to_edge.Reserve(num_bdr_elem);
   }
   else if (Dim == 3)
   {
      new_be_to_face.Reserve(num_bdr_elem);
      new_bel_to_edge = new Table;
      new_bel_to_edge->SetDims(num_bdr_elem, new_bel_to_edge_nnz);
   }
   for (int i = 0; i < GetNBE(); i++)
   {
      if (!FaceIsInterior(GetBdrElementEdgeIndex(i)))
      {
         new_boundary.Append(boundary[i]);
         if (Dim == 2)
         {
            new_be_to_edge.Append(be_to_edge[i]);
         }
         else if (Dim == 3)
         {
            int row = new_be_to_face.Size();
            new_be_to_face.Append(be_to_face[i]);
            int *e = bel_to_edge->GetRow(i);
            int ne = bel_to_edge->RowSize(i);
            int *new_e = new_bel_to_edge->GetRow(row);
            for (int j = 0; j < ne; j++)
            {
               new_e[j] = e[j];
            }
            new_bel_to_edge->GetI()[row+1] = new_bel_to_edge->GetI()[row] + ne;
         }
      }
   }

   NumOfBdrElements = new_boundary.Size();
   mfem::Swap(boundary, new_boundary);

   if (Dim == 2)
   {
      mfem::Swap(be_to_edge, new_be_to_edge);
   }
   else if (Dim == 3)
   {
      mfem::Swap(be_to_face, new_be_to_face);
      delete bel_to_edge;
      bel_to_edge = new_bel_to_edge;
   }

   Array<int> attribs(num_bdr_elem);
   for (int i = 0; i < attribs.Size(); i++)
   {
      attribs[i] = GetBdrAttribute(i);
   }
   attribs.Sort();
   attribs.Unique();
   bdr_attributes.DeleteAll();
   attribs.Copy(bdr_attributes);
}

void Mesh::FreeElement(Element *E)
{
#ifdef MFEM_USE_MEMALLOC
   if (E)
   {
      if (E->GetType() == Element::TETRAHEDRON)
      {
         TetMemory.Free((Tetrahedron*) E);
      }
      else
      {
         delete E;
      }
   }
#else
   delete E;
#endif
}

std::ostream &operator<<(std::ostream &out, const Mesh &mesh)
{
   mesh.Print(out);
   return out;
}


NodeExtrudeCoefficient::NodeExtrudeCoefficient(const int dim, const int _n,
                                               const double _s)
   : VectorCoefficient(dim), n(_n), s(_s), tip(p, dim-1)
{
}

void NodeExtrudeCoefficient::Eval(Vector &V, ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   V.SetSize(vdim);
   T.Transform(ip, tip);
   V(0) = p[0];
   if (vdim == 2)
   {
      V(1) = s * ((ip.y + layer) / n);
   }
   else
   {
      V(1) = p[1];
      V(2) = s * ((ip.z + layer) / n);
   }
}


Mesh *Extrude1D(Mesh *mesh, const int ny, const double sy, const bool closed)
{
   if (mesh->Dimension() != 1)
   {
      cerr << "Extrude1D : Not a 1D mesh!" << endl;
      mfem_error();
   }

   int nvy = (closed) ? (ny) : (ny + 1);
   int nvt = mesh->GetNV() * nvy;

   Mesh *mesh2d;

   if (closed)
   {
      mesh2d = new Mesh(2, nvt, mesh->GetNE()*ny, mesh->GetNBE()*ny);
   }
   else
      mesh2d = new Mesh(2, nvt, mesh->GetNE()*ny,
                        mesh->GetNBE()*ny+2*mesh->GetNE());

   // vertices
   double vc[2];
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      vc[0] = mesh->GetVertex(i)[0];
      for (int j = 0; j < nvy; j++)
      {
         vc[1] = sy * (double(j) / ny);
         mesh2d->AddVertex(vc);
      }
   }
   // elements
   Array<int> vert;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const Element *elem = mesh->GetElement(i);
      elem->GetVertices(vert);
      const int attr = elem->GetAttribute();
      for (int j = 0; j < ny; j++)
      {
         int qv[4];
         qv[0] = vert[0] * nvy + j;
         qv[1] = vert[1] * nvy + j;
         qv[2] = vert[1] * nvy + (j + 1) % nvy;
         qv[3] = vert[0] * nvy + (j + 1) % nvy;

         mesh2d->AddQuad(qv, attr);
      }
   }
   // 2D boundary from the 1D boundary
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const Element *elem = mesh->GetBdrElement(i);
      elem->GetVertices(vert);
      const int attr = elem->GetAttribute();
      for (int j = 0; j < ny; j++)
      {
         int sv[2];
         sv[0] = vert[0] * nvy + j;
         sv[1] = vert[0] * nvy + (j + 1) % nvy;

         if (attr%2)
         {
            Swap<int>(sv[0], sv[1]);
         }

         mesh2d->AddBdrSegment(sv, attr);
      }
   }

   if (!closed)
   {
      // 2D boundary from the 1D elements (bottom + top)
      int nba = (mesh->bdr_attributes.Size() > 0 ?
                 mesh->bdr_attributes.Max() : 0);
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const Element *elem = mesh->GetElement(i);
         elem->GetVertices(vert);
         const int attr = nba + elem->GetAttribute();
         int sv[2];
         sv[0] = vert[0] * nvy;
         sv[1] = vert[1] * nvy;

         mesh2d->AddBdrSegment(sv, attr);

         sv[0] = vert[1] * nvy + ny;
         sv[1] = vert[0] * nvy + ny;

         mesh2d->AddBdrSegment(sv, attr);
      }
   }

   mesh2d->FinalizeQuadMesh(1, 0, false);

   GridFunction *nodes = mesh->GetNodes();
   if (nodes)
   {
      // duplicate the fec of the 1D mesh so that it can be deleted safely
      // along with its nodes, fes and fec
      FiniteElementCollection *fec2d = NULL;
      FiniteElementSpace *fes2d;
      const char *name = nodes->FESpace()->FEColl()->Name();
      string cname = name;
      if (cname == "Linear")
      {
         fec2d = new LinearFECollection;
      }
      else if (cname == "Quadratic")
      {
         fec2d = new QuadraticFECollection;
      }
      else if (cname == "Cubic")
      {
         fec2d = new CubicFECollection;
      }
      else if (!strncmp(name, "H1_", 3))
      {
         fec2d = new H1_FECollection(atoi(name + 7), 2);
      }
      else if (!strncmp(name, "L2_T", 4))
      {
         fec2d = new L2_FECollection(atoi(name + 10), 2, atoi(name + 4));
      }
      else if (!strncmp(name, "L2_", 3))
      {
         fec2d = new L2_FECollection(atoi(name + 7), 2);
      }
      else
      {
         delete mesh2d;
         cerr << "Extrude1D : The mesh uses unknown FE collection : "
              << cname << endl;
         mfem_error();
      }
      fes2d = new FiniteElementSpace(mesh2d, fec2d, 2);
      mesh2d->SetNodalFESpace(fes2d);
      GridFunction *nodes2d = mesh2d->GetNodes();
      nodes2d->MakeOwner(fec2d);

      NodeExtrudeCoefficient ecoeff(2, ny, sy);
      Vector lnodes;
      Array<int> vdofs2d;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         ElementTransformation &T = *mesh->GetElementTransformation(i);
         for (int j = ny-1; j >= 0; j--)
         {
            fes2d->GetElementVDofs(i*ny+j, vdofs2d);
            lnodes.SetSize(vdofs2d.Size());
            ecoeff.SetLayer(j);
            fes2d->GetFE(i*ny+j)->Project(ecoeff, T, lnodes);
            nodes2d->SetSubVector(vdofs2d, lnodes);
         }
      }
   }
   return mesh2d;
}

void Mesh::IntermeshInit( IntermediateMesh * intermesh, int dim, int nv, int ne, int nbdr, int with_gindices_flag)
{
    intermesh->dim = dim;
    intermesh->ne = ne;
    intermesh->nv = nv;
    intermesh->nbe = nbdr;

    intermesh->vertices = new double[nv * dim];
    intermesh->elements = new int[ne * (dim + 1)];
    intermesh->bdrelements = new int[nbdr * dim];
    intermesh->elattrs = new int[ne];
    intermesh->bdrattrs = new int[nbdr];

    if (with_gindices_flag != 0)
    {
        intermesh->withgindicesflag = 1;
        intermesh->vert_gindices = new int[nv];
    }
    else
        intermesh->withgindicesflag = 0;

    return;
}

void Mesh::IntermeshDelete( IntermediateMesh * intermesh_pt)
{
    delete [] intermesh_pt->vertices;
    delete [] intermesh_pt->elements;
    delete [] intermesh_pt->bdrelements;
    delete [] intermesh_pt->elattrs;
    delete [] intermesh_pt->bdrattrs;

    if ( intermesh_pt->withgindicesflag != 0)
        delete [] intermesh_pt->vert_gindices;

    delete intermesh_pt;

    return;
}

void Mesh::InterMeshPrint (IntermediateMesh * local_intermesh, int suffix, const char * filename)
{
    int dim = local_intermesh->dim;
    int ne = local_intermesh->ne;
    int nv = local_intermesh->nv;
    int nbe = local_intermesh->nbe;

    ofstream myfile;
    char csuffix[20];
    sprintf (csuffix, "_%d.intermesh", suffix);

    char fileoutput[250];
    strcpy (fileoutput, filename);
    strcat (fileoutput, csuffix);

    myfile.open (fileoutput);

    myfile << "elements: \n";
    myfile << ne << endl;
    for ( int i = 0; i < ne; ++i )
    {
        myfile << local_intermesh->elattrs[i] << " ";
        for ( int j = 0; j < dim + 1; ++j )
            myfile << local_intermesh->elements[i*(dim+1) + j] << " ";
        myfile << endl;
    }
    myfile << endl;

    myfile << "boundary: \n";
    myfile << nbe << endl;
    for ( int i = 0; i < nbe; ++i )
    {
        myfile << local_intermesh->bdrattrs[i] << " ";
        for ( int j = 0; j < dim; ++j )
            myfile << local_intermesh->bdrelements[i*dim + j] << " ";
        myfile << endl;
    }
    myfile << endl;

    myfile << "vertices: \n";
    myfile << nv << endl;
    int withgindicesflag = 0;
    if (local_intermesh->withgindicesflag != 0)
        withgindicesflag = 1;
    for ( int i = 0; i < nv; ++i )
    {
        for ( int j = 0; j < dim; ++j )
            myfile << local_intermesh->vertices[i*dim + j] << " ";
        if (withgindicesflag == 1)
            myfile << " gindex: " << local_intermesh->vert_gindices[i];
        myfile << endl;
    }
    myfile << endl;

    myfile.close();

    return;
}

// Takes the 4d mesh with elements, vertices and boundary already created
// and creates all the internal structure.
// Used inside the Mesh constructor.
// "refine" argument is added for handling 2D case, when refinement marker routines
// should be called before creating structures for shared entities which goes
// before the call to CreateInternal...()
// Probably for parallel mesh generator some tables are generated twice // FIX IT
void Mesh::CreateInternalMeshStructure (int refine)
{
    int j, curved = 0;
    //int refine = 1;
    bool fix_orientation = true;
    int generate_edges = 1;

    Nodes = NULL;
    own_nodes = 1;
    NURBSext = NULL;
    ncmesh = NULL;
    last_operation = Mesh::NONE;
    sequence = 0;

    InitTables();

    //for a 4d mesh sort the element and boundary element indices by the node numbers
    if(spaceDim==4)
    {
        swappedElements.SetSize(NumOfElements);
        DenseMatrix J(4,4);
        for (j = 0; j < NumOfElements; j++)
        {
            if (elements[j]->GetType() == Element::PENTATOPE)
            {
                int *v = elements[j]->GetVertices();
                Sort5(v[0], v[1], v[2], v[3], v[4]);

                GetElementJacobian(j, J);
                if(J.Det() < 0.0)
                {
                    swappedElements[j] = true;
                    Swap(v);
                }else
                {
                    swappedElements[j] = false;
                }
            }

        }
        for (j = 0; j < NumOfBdrElements; j++)
        {
            if (boundary[j]->GetType() == Element::TETRAHEDRON)
            {
                int *v = boundary[j]->GetVertices();
                Sort4(v[0], v[1], v[2], v[3]);
            }
        }
    }

    // at this point the following should be defined:
    //  1) Dim
    //  2) NumOfElements, elements
    //  3) NumOfBdrElements, boundary
    //  4) NumOfVertices, with allocated space in vertices
    //  5) curved
    //  5a) if curved == 0, vertices must be defined
    //  5b) if curved != 0 and read_gf != 0,
    //         'input' must point to a GridFunction
    //  5c) if curved != 0 and read_gf == 0,
    //         vertices and Nodes must be defined

    if (spaceDim == 0)
    {
       spaceDim = Dim;
    }

    InitBaseGeom();

    // set the mesh type ('meshgen')
    SetMeshGen();


    if (NumOfBdrElements == 0 && Dim > 2)
    {
       // in 3D, generate boundary elements before we 'MarkForRefinement'
       if(Dim==3) GetElementToFaceTable();
       else if(Dim==4)
       {
           GetElementToFaceTable4D();
       }
       GenerateFaces();
       GenerateBoundaryElements();
    }


    if (!curved)
    {
       // check and fix element orientation
       CheckElementOrientation(fix_orientation);

       if (refine)
       {
          MarkForRefinement();
       }
    }

    if (Dim == 1)
    {
       GenerateFaces();
    }

    // generate the faces
    if (Dim > 2)
    {
           if(Dim==3) GetElementToFaceTable();
           else if(Dim==4)
           {
               GetElementToFaceTable4D();
           }

           GenerateFaces();

           if(Dim==4)
           {
              ReplaceBoundaryFromFaces();

              GetElementToPlanarTable();
              GeneratePlanars();

 //			 GetElementToQuadTable4D();
 //			 GenerateQuads4D();
           }

       // check and fix boundary element orientation
       if ( !(curved && (meshgen & 1)) )
       {
          CheckBdrElementOrientation();
       }
    }
    else
    {
       NumOfFaces = 0;
    }

    // generate edges if requested
    if (Dim > 1 && generate_edges == 1)
    {
       // el_to_edge may already be allocated (P2 VTK meshes)
       if (!el_to_edge) { el_to_edge = new Table; }
       NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
       if (Dim == 2)
       {
          GenerateFaces(); // 'Faces' in 2D refers to the edges
          if (NumOfBdrElements == 0)
          {
             GenerateBoundaryElements();
          }
          // check and fix boundary element orientation
          if ( !(curved && (meshgen & 1)) )
          {
             CheckBdrElementOrientation();
          }
       }
    }
    else
    {
       NumOfEdges = 0;
    }

    //// generate the arrays 'attributes' and ' bdr_attributes'
    SetAttributes();

    return;
}

// computes elpartition array which is used for computing slice meshes over different time moments
// elpartition is the output
// elpartition stores for each time moment a vector of integer indices of the mesh elements which intersect
// with the corresponding time plane
void Mesh::Compute_elpartition (double t0, int Nmoments, double deltat, vector<vector<int> > & elpartition)
{
    bool verbose = false;
    int dim = Dimension();

    Element * el;
    int * vind;
    double * vcoords;
    double eltmin, eltmax;

    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        if (verbose)
            cout << "elind = " << elind << endl;
        el = GetElement(elind);
        vind = el->GetVertices();

        // computing eltmin and eltmax for an element = minimal and maximal time moments for each element
        eltmin = t0 + Nmoments * deltat;
        eltmax = 0.0;
        for (int vno = 0; vno < el->GetNVertices(); ++vno )
        {
            vcoords = GetVertex(vind[vno]);
            if ( vcoords[dim - 1] > eltmax )
                eltmax = vcoords[dim - 1];
            if ( vcoords[dim - 1] < eltmin )
                eltmin = vcoords[dim - 1];
        }


        if (verbose)
        {
            cout << "Special print: elind = " << elind << endl;
            for (int vno = 0; vno < el->GetNVertices(); ++vno )
            {
                cout << "vertex: ";
                vcoords = GetVertex(vind[vno]);
                for ( int coo = 0; coo < dim; ++coo )
                    cout << vcoords[coo] << " ";
                cout << endl;
            }

            cout << "eltmin = " << eltmin << " eltmax = " << eltmax << endl;
        }




        // deciding which time moments intersect the element if any
        //if ( (eltmin > t0 && eltmin < t0 + (Nmoments-1) * deltat) ||  (eltmax > t0 && eltmax < t0 + (Nmoments-1) * deltat))
        if ( (eltmax > t0 && eltmin < t0 + (Nmoments-1) * deltat))
        {
            if (verbose)
            {
                cout << "the element is intersected by some time moments" << endl;
                cout << "t0 = " << t0 << " deltat = " << deltat << endl;
                cout << fixed << setprecision(6);
                cout << "low bound = " << ceil( (max(eltmin,t0) - t0) / deltat  ) << endl;
                cout << "top bound = " << floor ((min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat) << endl;
                cout << "4isl for low = " << max(eltmin,t0) - t0 << endl;
                cout << "magic number for low = " << (max(eltmin,t0) - t0) / deltat << endl;
                cout << "magic number for top = " << (min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat << endl;
            }
            for ( int k = ceil( (max(eltmin,t0) - t0) / deltat  ); k <= floor ((min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat) ; ++k)
            {
                //if (myid == 0 )
                if (verbose)
                {
                    cout << "k = " << k << endl;
                }
                elpartition[k].push_back(elind);
            }
        }
        else
        {
            if (verbose)
                cout << "the element is not intersected by any time moments" << endl;
        }
    }

    // intermediate output
    /*
    for ( int i = 0; i < Nmoments; ++i)
    {
        cout << "moment " << i << ": time = " << t0 + i * deltat << endl;
        cout << "size for this partition = " << elpartition[i].size() << endl;
        for ( int j = 0; j < elpartition[i].size(); ++j)
            cout << "el: " << elpartition[i][j] << endl;
    }
    */
    return;
}

// computes number of slice cell vertexes, slice cell vertex indices and coordinates
// for a given element with index = elind.
// updates the edgemarkers and vertex_count correspondingly
// pvec defines the slice plane
void Mesh::computeSliceCell (int elind, vector<vector<double> > & pvec, vector<vector<double> > & ipoints, vector<int>& edgemarkers,
                             vector<vector<double> >& cellpnts, vector<int>& elvertslocal, int & nip, int & vertex_count )
{
    bool verbose = false; // probably should be a function argument
    int dim = Dimension();

    int * edgeindices;
    int edgenolen, edgeind;
    Array<int> edgev(2);
    double * v1, * v2;

    vector<vector<double> > edgeends(dim);
    edgeends[0].reserve(dim);
    edgeends[1].reserve(dim);

    DenseMatrix M(dim, dim);
    Vector sol(4), rh(4);

    vector<double> ip(dim);

    edgeindices = el_to_edge->GetRow(elind);
    edgenolen = el_to_edge->RowSize(elind);

    nip = 0;

    for ( int edgeno = 0; edgeno < edgenolen; ++edgeno)
    {
        // true mesh edge index
        edgeind = edgeindices[edgeno];

        if (verbose)
            cout << "edgeind " << edgeind << endl;
        if (edgemarkers[edgeind] == -2) // if this edge was not considered
        {
            GetEdgeVertices(edgeind, edgev);

            // vertex coordinates
            v1 = GetVertex(edgev[0]);
            v2 = GetVertex(edgev[1]);

            // vertex coordinates as vectors of doubles, edgeends 0 is lower in time coordinate than edgeends[1]
            if (v1[dim-1] < v2[dim-1])
            {
                for ( int coo = 0; coo < dim; ++coo)
                {
                    edgeends[0][coo] = v1[coo];
                    edgeends[1][coo] = v2[coo];
                }
            }
            else
            {
                for ( int coo = 0; coo < dim; ++coo)
                {
                    edgeends[0][coo] = v2[coo];
                    edgeends[1][coo] = v1[coo];
                }
            }


            if (verbose)
            {
                cout << "edge vertices:" << endl;
                for (int i = 0; i < 2; ++i)
                {
                    cout << "vert ";
                    for ( int coo = 0; coo < dim; ++coo)
                        cout << edgeends[i][coo] << " ";
                    cout << "   ";
                }
                cout << endl;
            }


            // creating the matrix for computing the intersection point
            for ( int i = 0; i < dim; ++i)
                for ( int j = 0; j < dim - 1; ++j)
                    M(i,j) = pvec[j + 1][i];
            for ( int i = 0; i < dim; ++i)
                M(i,dim - 1) = edgeends[0][i] - edgeends[1][i];

            /*
            cout << "M" << endl;
            M.Print();
            cout << "M.Det = " << M.Det() << endl;
            */

            if ( fabs(M.Det()) > MYZEROTOL )
            {
                M.Invert();

                // setting righthand side
                for ( int i = 0; i < dim; ++i)
                    rh[i] = edgeends[0][i] - pvec[0][i];

                // solving the system
                M.Mult(rh, sol);

                if ( sol[dim-1] > 0.0 - MYZEROTOL && sol[dim-1] <= 1.0 + MYZEROTOL)
                {
                    for ( int i = 0; i < dim; ++i)
                        ip[i] = edgeends[0][i] + sol[dim-1] * (edgeends[1][i] - edgeends[0][i]);

                    if (verbose)
                    {
                        cout << "intersection point for this edge: " << endl;
                        for ( int i = 0; i < dim; ++i)
                            cout << ip[i] << " ";
                        cout << endl;
                    }

                    ipoints.push_back(ip);
                    //vrtindices[momentind].push_back(vertex_count);
                    elvertslocal.push_back(vertex_count);
                    edgemarkers[edgeind] = vertex_count;
                    cellpnts.push_back(ip);
                    nip++;
                    vertex_count++;
                }
                else
                {
                    if (verbose)
                        cout << "Line but not edge intersects" << endl;
                    edgemarkers[edgeind] = -1;
                }

            }
            else
                if (verbose)
                    cout << "Edge is parallel" << endl;
        }
        else // the edge was already considered -> edgemarkers store the vertex index
        {
            if (edgemarkers[edgeind] >= 0)
            {
                elvertslocal.push_back(edgemarkers[edgeind]);
                cellpnts.push_back(ipoints[edgemarkers[edgeind]]);
                nip++;
            }
        }

        //cout << "tempvec.size = " << tempvec.size() << endl;

    } // end of loop over element edges

    return;
}

// outputs the slice mesh information in VTK format
void Mesh::outputSliceMeshVTK ( std::stringstream& fname, std::vector<std::vector<double> > & ipoints,
                                std::list<int> &celltypes, int cellstructsize, std::list<std::vector<int> > &elvrtindices)
{
    int dim = Dimension();
    // output in the vtk format for paraview
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);

    ofid << "# vtk DataFile Version 3.0" << endl;
    ofid << "Generated by MFEM" << endl;
    ofid << "ASCII" << endl;
    ofid << "DATASET UNSTRUCTURED_GRID" << endl;

    ofid << "POINTS " << ipoints.size() << " double" << endl;
    for (unsigned int vno = 0; vno < ipoints.size(); ++vno)
    {
        for ( int c = 0; c < dim - 1; ++c )
        {
            ofid << ipoints[vno][c] << " ";
        }
        if (dim == 3)
            ofid << ipoints[vno][dim - 1] << " ";
        ofid << endl;
    }

    ofid << "CELLS " << celltypes.size() << " " << cellstructsize << endl;
    std::list<int>::const_iterator iter;
    std::list<vector<int> >::const_iterator iter2;
    for (iter = celltypes.begin(), iter2 = elvrtindices.begin();
         iter != celltypes.end() && iter2 != elvrtindices.end()
         ; ++iter, ++iter2)
    {
        //cout << *it;
        int npoints;
        if (*iter == VTKTETRAHEDRON)
            npoints = 4;
        else if (*iter == VTKWEDGE)
            npoints = 6;
        else if (*iter == VTKQUADRIL)
            npoints = 4;
        else //(*iter == VTKTRIANGLE)
            npoints = 3;
        ofid << npoints << " ";

        for ( int i = 0; i < npoints; ++i)
            ofid << (*iter2)[i] << " ";
        ofid << endl;
    }

    ofid << "CELL_TYPES " << celltypes.size() << endl;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << *iter << endl;
    }

    // test lines for cell data
    ofid << "CELL_DATA " << celltypes.size() << endl;
    ofid << "SCALARS cekk_scalars double 1" << endl;
    ofid << "LOOKUP_TABLE default" << endl;
    int cnt = 0;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << cnt * 1.0 << endl;
        cnt++;
    }
    return;
}


// scalar product of two vectors (outputs 0 if vectors have different length)
double sprod(std::vector<double> vec1, std::vector<double> vec2)
{
    if (vec1.size() != vec2.size())
        return 0.0;
    double res = 0.0;
    for ( unsigned int c = 0; c < vec1.size(); ++c)
        res += vec1[c] * vec2[c];
    return res;
}
double l2Norm(std::vector<double> vec)
{
    return sqrt(sprod(vec,vec));
}

// compares pairs<int,double> with respect to the second (double) elements
bool intdComparison(const std::pair<int,double> &a,const std::pair<int,double> &b)
{
       return a.second>b.second;
}

// only first 2 coordinates of each element of Points is used (although now the
// input is 4 3-dimensional points but the last coordinate is time so it is not used
// because the slice is with t = const planes
// sorts in a counter-clock fashion required by VTK format for quadrilateral
// the main output is the permutation of the input points array
bool sortQuadril2d(std::vector<std::vector<double> > & Points, int * permutation)
{
    bool verbose = false;

    if (Points.size() != 4)
    {
        cout << "Error: sortQuadril2d should be called only for a vector storing 4 points" << endl;
        return false;
    }
    /*
    for ( int p = 0; p < Points.size(); ++p)
        if (Points[p].size() != 2)
        {
            cout << "Error: sortQuadril2d should be called only for a vector storing 4 2d-points" << endl;
            return false;
        }
    */

    /*
    cout << "Points inside sortQuadril2d() \n";
    for (int i = 0; i < 4; ++i)
    {
        cout << "vert " << i << ":";
        for ( int j = 0; j < 2; ++j)
            cout << Points[i][j] << " ";
        cout << endl;
    }
    */


    int argbottom = 0; // index of the the vertex with the lowest y-coordinate
    for (int p = 1; p < 4; ++p)
        if (Points[p][1] < Points[argbottom][1])
            argbottom = p;

    if (verbose)
        cout << "argbottom = " << argbottom << endl;

    // cosinuses of angles between radius vectors from vertex argbottom to the others and positive x-direction
    vector<pair<int, double> > cos(3);
    vector<vector<double> > radiuses(3);
    vector<double> xort(2);
    xort[0] = 1.0;
    xort[1] = 0.0;
    int cnt = 0;
    for (int p = 0; p < 4; ++p)
    {
        if (p != argbottom)
        {
            cos[cnt].first = p;
            for ( int c = 0; c < 2; ++c)
                radiuses[cnt].push_back(Points[p][c] - Points[argbottom][c]);
            cos[cnt].second = sprod(radiuses[cnt], xort) / l2Norm(radiuses[cnt]);
            cnt ++;
        }
    }

    //int permutation[4];
    permutation[0] = argbottom;

    std::sort(cos.begin(), cos.end(), intdComparison);

    for ( int i = 0; i < 3; ++i)
        permutation[1 + i] = cos[i].first;

    if (verbose)
    {
        cout << "permutation:" << endl;
        for (int i = 0; i < 4; ++i)
            cout << permutation[i] << " ";
        cout << endl;
    }

    // not needed actually. onlt for debugging. actually the output is the correct permutation
    /*
    vector<vector<double>> temp(4);
    for ( int p = 0; p < 4; ++p)
        for ( int i = 0; i < 3; ++i)
            temp[p].push_back(Points[permutation[p]][i]);

    for ( int p = 0; p < 4; ++p)
        for ( int i = 0; i < 3; ++i)
            Points[p][i] = temp[p][i];
    */
    return true;
}

// sorts the vertices in order for the points to form a proper vtk wedge
// first three vertices should be the base, with normal to (0,1,2)
// looking opposite to the direction of where the second base is.
// This ordering is required by VTK format for wedges, look
// in vtk wedge class definitio for explanations
// the main output is the permutation of the input vertexes array
bool sortWedge3d(std::vector<std::vector<double> > & Points, int * permutation)
{
    /*
    cout << "wedge points:" << endl;
    for ( int i = 0; i < Points.size(); ++i)
    {
        for ( int j = 0; j < Points[i].size(); ++j)
            cout << Points[i][j] << " ";
        cout << endl;
    }
    */

    vector<double> p1 = Points[0];
    int pn2 = -1;
    vector<int> pnum2;

    //bestimme die 2 quadrate
    for(unsigned int i=1; i<Points.size(); i++)
    {
        vector<double> dets;
        for(unsigned int k=1; k<Points.size()-1; k++)
        {
            for(unsigned int l=k+1; l<Points.size(); l++)
            {
                if(k!=i && l!=i)
                {
                    vector<double> Q1(3);
                    vector<double> Q2(3);
                    vector<double> Q3(3);

                    for ( int c = 0; c < 3; c++)
                        Q1[c] = p1[c] - Points[i][c];
                    for ( int c = 0; c < 3; c++)
                        Q2[c] = p1[c] - Points[k][c];
                    for ( int c = 0; c < 3; c++)
                        Q3[c] = p1[c] - Points[l][c];

                    //vector<double> Q1 = p1 - Points[i];
                    //vector<double> Q2 = p1 - Points[k];
                    //vector<double> Q3 = p1 - Points[l];

                    DenseMatrix MM(3,3);
                    MM(0,0) = Q1[0]; MM(0,1) = Q2[0]; MM(0,2) = Q3[0];
                    MM(1,0) = Q1[1]; MM(1,1) = Q2[1]; MM(1,2) = Q3[1];
                    MM(2,0) = Q1[2]; MM(2,1) = Q2[2]; MM(2,2) = Q3[2];
                    double determ = MM.Det();

                    dets.push_back(determ);
                }
            }
        }

        double max_ = 0; double min_ = fabs(dets[0]);
        for(unsigned int m=0; m<dets.size(); m++)
        {
            if(max_<fabs(dets[m])) max_ = fabs(dets[m]);
            if(min_>fabs(dets[m])) min_ = fabs(dets[m]);
        }

        //for ( int in = 0; in < dets.size(); ++in)
            //cout << "det = " << dets[in] << endl;

        if(max_!=0) for(unsigned int m=0; m<dets.size(); m++) dets[m] /= max_;

        //cout << "max_ = " << max_ << endl;

        int count = 0;
        vector<bool> el;
        for(unsigned int m=0; m<dets.size(); m++) { if(fabs(dets[m]) < 1e-8) { count++; el.push_back(true); } else el.push_back(false); }

        if(count==2)
        {
            for(unsigned int k=1, m=0; k<Points.size()-1; k++)
                for(unsigned int l=k+1; l<Points.size(); l++)
                {
                    if(k!=i && l!=i)
                    {
                        if(el[m]) { pnum2.push_back(k); pnum2.push_back(l); }
                        m++;
                    }

                }

            pn2 = i;
            break;
        }

        if(count == 0 || count > 2)
        {
            //cout << "count == 0 || count > 2" << endl;
            //cout << "count = " << count << endl;
            return false;
        }
    }

    if(pn2<0)
    {
        //cout << "pn2 < 0" << endl;
        return false;
    }


    vector<int> oben(3); oben[0] = pn2;
    vector<int> unten(3); unten[0] = 0;

    //winkel berechnen
    vector<double> pp1(3);
    vector<double> pp2(3);
    for ( int c = 0; c < 3; c++)
        pp1[c] = Points[0][c] - Points[pn2][c];
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[0]][c] - Points[pn2][c];
    //vector<double> pp1 = Points[0] - Points[pn2];
    //vector<double> pp2 = Points[pnum2[0]] - Points[pn2];
    double w1 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[1]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[1]]- Points[pn2];
    double w2 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));

    if(w1 < w2)  { oben[1] = pnum2[0]; unten[1] = pnum2[1]; }
    else{ oben[1] = pnum2[1]; unten[1] = pnum2[0]; }

    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[2]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[2]] - Points[pn2];
    w1 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[3]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[3]]- Points[pn2];
    w2 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));

    if(w1 < w2)  { oben[2] = pnum2[2]; unten[2] = pnum2[3]; }
    else{ oben[2] = pnum2[3]; unten[2] = pnum2[2]; }

    for(unsigned int i=0; i<unten.size(); i++) permutation[i] = unten[i];
    for(unsigned int i=0; i<oben.size(); i++)  permutation[i + unten.size()] = oben[i];

    //not needed since we actually need the permutation only
    /*
    vector<vector<double>> pointssort;
    for(unsigned int i=0; i<unten.size(); i++) pointssort.push_back(Points[unten[i]]);
    for(unsigned int i=0; i<oben.size(); i++) pointssort.push_back(Points[oben[i]]);

    for(unsigned int i=0; i<pointssort.size(); i++) Points[i] = pointssort[i];
    */

    return true;
}


// reorders the cell vertices so as to have the cell vertex ordering compatible with VTK format
// the output is the sorted elvertexes (which is also the input)
void reorder_cellvertices ( int dim, int nip, std::vector<std::vector<double> > & cellpnts, std::vector<int> & elvertexes)
{
    bool verbose = false;
    // used only for checking the orientation of tetrahedrons
    DenseMatrix Mtemp(3, 3);

    // special reordering of vertices is required for the vtk wedge, so that
    // vertices are added one base after another and not as a mix

    if (nip == 6)
    {

        /*
        cout << "Sorting the future wedge" << endl;
        cout << "Before sorting: " << endl;
        for (int i = 0; i < 6; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */


        // FIX IT: NOT TESTED AT ALL
        int permutation[6];
        if ( sortWedge3d (cellpnts, permutation) == false )
        {
            cout << "sortWedge returns false, possible bad behavior" << endl;
            return;
        }

        /*
        cout << "After sorting: " << endl;
        for (int i = 0; i < 6; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[permutation[i]][j] << " ";
            cout << endl;
        }
        */

        int temp[6];
        for ( int i = 0; i < 6; ++i)
            temp[i] = elvertexes[permutation[i]];
        for ( int i = 0; i < 6; ++i)
            elvertexes[i] = temp[i];


        double det = 0.0;

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,0) = (1.0/3.0)*(cellpnts[permutation[3]][i] + cellpnts[permutation[4]][i] + cellpnts[permutation[5]][i])
                    - cellpnts[permutation[0]][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,1) = cellpnts[permutation[2]][i] - cellpnts[permutation[0]][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,2) = cellpnts[permutation[1]][i] - cellpnts[permutation[0]][i];

        det = Mtemp.Det();

        if (verbose)
        {
            if (det < 0)
                cout << "orientation for wedge = negative" << endl;
            else if (det == 0.0)
                cout << "error for wedge: bad volume" << endl;
            else
                cout << "orientation for wedge = positive" << endl;
        }

        if (det < 0)
        {
            if (verbose)
                cout << "Have to swap the vertices to change the orientation of wedge" << endl;
            int tmp;
            tmp = elvertexes[1];
            elvertexes[1] = elvertexes[0];
            elvertexes[1] = tmp;
            //Swap(*(elvrtindices[momentind].end()));
            tmp = elvertexes[4];
            elvertexes[4] = elvertexes[3];
            elvertexes[4] = tmp;
        }

    }


    // positive orientation is required for vtk tetrahedron
    // normal to the plane with first three vertexes should poit towards the 4th vertex

    if (nip == 4 && dim == 4)
    {
        /*
        cout << "tetrahedra points" << endl;
        for (int i = 0; i < 4; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */

        double det = 0.0;

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,0) = cellpnts[3][i] - cellpnts[0][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,1) = cellpnts[2][i] - cellpnts[0][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,2) = cellpnts[1][i] - cellpnts[0][i];

        //Mtemp.Print();

        det = Mtemp.Det();

        if (verbose)
        {
            if (det < 0)
                cout << "orientation for tetra = negative" << endl;
            else if (det == 0.0)
                cout << "error for tetra: bad volume" << endl;
            else
                cout << "orientation for tetra = positive" << endl;
        }

        //return;

        if (det < 0)
        {
            if (verbose)
                cout << "Have to swap the vertices to change the orientation of tetrahedron" << endl;
            int tmp = elvertexes[1];
            elvertexes[1] = elvertexes[0];
            elvertexes[1] = tmp;
            //Swap(*(elvrtindices[momentind].end()));
        }

    }


    // in 2D case the vertices of a quadrilateral should be umbered in a counter-clock wise fashion
    if (nip == 4 && dim == 3)
    {
        /*
        cout << "Sorting the future quadrilateral" << endl;
        cout << "Before sorting: " << endl;
        for (int i = 0; i < nip; ++i)
        {
            cout << "vert " << elvertexes[i] << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */

        int permutation[4];
        sortQuadril2d(cellpnts, permutation);

        int temp[4];
        for ( int i = 0; i < 4; ++i)
            temp[i] = elvertexes[permutation[i]];
        for ( int i = 0; i < 4; ++i)
            elvertexes[i] = temp[i];

        /*
        cout << "After sorting: " << endl;
        for (int i = 0; i < nip; ++i)
        {
            cout << "vert " << elvertexes[i] << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[permutation[i]][j] << " ";
            cout << endl;
        }
        */

    }

    return;
}

// Computes and outputs in VTK format slice meshes of a given 3D or 4D mesh
// by time-like planes t = t0 + k * deltat, k = 0, ..., Nmoments - 1
// myid is used for creating different output files by different processes
// if the mesh is parallel
// usually it is reasonable to refer myid to the process id in the communicator
// so as to produce a correct output for parallel ParaView visualization
void Mesh::ComputeSlices(double t0, int Nmoments, double deltat, int myid)
{
    bool verbose = false;

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON )
    {
        //if (myid == 0)
            cout << "Mesh::ComputeSlices() is implemented only for pentatops "
                    "and tetrahedrons" << endl << flush;
        return;
    }

    int dim = Dimension();

    if (!el_to_edge)
    {
        el_to_edge = new Table;
        NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
    }

    // = -2 if not considered, -1 if considered, but does not intersected, index of this vertex in the new 3d mesh otherwise
    // refilled for each time moment
    vector<int> edgemarkers(GetNEdges());

    // stores indices of elements which are intersected by planes related to the time moments
    vector<vector<int> > elpartition(Nmoments);
    // can make it faster, if any estimates are known for how many elements are intersected by a single time plane
    //for ( int i = 0; i < Nmoments; ++i)
        //elpartition[i].reserve(100);

    // *************************************************************************
    // step 1 of x: loop over all elememnts and compute elpartition for all time
    // moments.
    // *************************************************************************

    Compute_elpartition (t0, Nmoments, deltat, elpartition);


    // *************************************************************************
    // step 2 of x: looping over time momemnts and slicing elements for each
    // given time moment, and outputs the resulting slice mesh in VTK format
    // *************************************************************************

    // slicing the elements, time moment over time moment
    int elind;

    vector<vector<double> > pvec(dim);
    for ( int i = 0; i < dim; ++i)
        pvec[i].reserve(dim);

    // used only for checking the orientation of tetrahedrons and quadrilateral vertexes reordering
    //DenseMatrix Mtemp(3, 3);

    // output data structures for vtk format
    // for each time moment holds a list with cell type for each cell
    vector<std::list<int> > celltypes(Nmoments);
    // for each time moment holds a list with vertex indices
    //vector<std::list<int>> vrtindices(Nmoments);
    // for each time moment holds a list with cell type for each cell
    vector<std::list<vector<int> > > elvrtindices(Nmoments);

    // number of integers in cell structure - for each cell 1 integer (number of vertices) +
    // + x integers (vertex indices)
    int cellstructsize;
    int vertex_count; // number of vertices in the slice mesh for a single time moment

    // loop over time moments
    for ( int momentind = 0; momentind < Nmoments; ++momentind )
    {
        if (verbose)
            cout << "Time moment " << momentind << ": time = " << t0 + momentind * deltat << endl;

        // refilling edgemarkers, resetting vertex_count and cellstructsize
        for ( int i = 0; i < GetNEdges(); ++i)
            edgemarkers[i] = -2;

        vertex_count = 0;
        cellstructsize = 0;

        vector<vector<double> > ipoints; // one of main arrays: all intersection points for a given time moment

        // vectors, defining the plane of the slice p0, p1, p2 (and p3 in 4D)
        // p0 is the time aligned vector for the given time moment
        // p1, p2 (and p3) - basis orts for the plane
        // pvec is {p0,p1,p2,p3} vector
        for ( int i = 0; i < dim; ++i)
            for ( int j = 0; j < dim; ++j)
                pvec[i][dim - 1 - j] = ( i == j ? 1.0 : 0.0);
        pvec[0][dim - 1] = t0 + momentind * deltat;

        // loop over elements intersected by the plane realted to a given time moment
        // here, elno = index in elpartition[momentind]
        for ( unsigned int elno = 0; elno < elpartition[momentind].size(); ++elno)
        //for ( int elno = 0; elno < 3; ++elno)
        {
            vector<int> tempvec;             // vertex indices for the cell of the slice mesh
            tempvec.reserve(6);
            vector<vector<double> > cellpnts; //points of the cell of the slice mesh
            cellpnts.reserve(6);

            // true mesh element index
            elind = elpartition[momentind][elno];
            //Element * el = GetElement(elind);

            if (verbose)
                cout << "Element: " << elind << endl;

            // computing number of intersection points, indices and coordinates for
            // local slice cell vertexes (cellpnts and tempvec)  and adding new intersection
            // points and changing edges markers for a given element elind
            // and plane defined by pvec
            int nip;
            computeSliceCell (elind, pvec, ipoints, edgemarkers, cellpnts, tempvec, nip, vertex_count);

            if ( (dim == 4 && (nip != 4 && nip != 6)) || (dim == 3 && (nip != 3 && nip != 4)) )
                cout << "Strange nip =  " << nip << " for elind = " << elind << ", time = " << t0 + momentind * deltat << endl;
            else
            {
                if (nip == 4) // tetrahedron in 3d or quadrilateral in 2d
                    if (dim == 4)
                        celltypes[momentind].push_back(VTKTETRAHEDRON);
                    else // dim == 3
                        celltypes[momentind].push_back(VTKQUADRIL);
                else if (nip == 6) // prism
                    celltypes[momentind].push_back(VTKWEDGE);
                else // nip == 3 = triangle
                    celltypes[momentind].push_back(VTKTRIANGLE);

                cellstructsize += nip + 1;

                elvrtindices[momentind].push_back(tempvec);

                // special reordering of cell vertices, required for the wedge,
                // tetrahedron and quadrilateral cells
                reorder_cellvertices (dim, nip, cellpnts, elvrtindices[momentind].back());

                if (verbose)
                    cout << "nip for the element = " << nip << endl;
            }

        } // end of loop over elements for a given time moment

        // intermediate output
        std::stringstream fname;
        fname << "slicemesh_"<< dim - 1 << "d_myid_" << myid << "_moment_" << momentind << ".vtk";
        outputSliceMeshVTK (fname, ipoints, celltypes[momentind], cellstructsize, elvrtindices[momentind]);


    } //end of loop over time moments

    // if not deleted here, gets segfault for more than two parallel refinements afterwards
    delete edge_vertex;
    edge_vertex = NULL;

    //

    return;
}

#ifdef MFEM_USE_MPI
// Converts a given ParMesh into a serial Mesh, and outputs the corresponding partioning as well.
Mesh::Mesh ( ParMesh& pmesh, int ** partioning)
{
    MPI_Comm comm = pmesh.GetComm();

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = pmesh.Dimension();
    int nvert_per_elem = dim + 1; // PENTATOPE and TETRAHEDRON case only
    int nvert_per_bdrelem = dim;  // PENTATOPE and TETRAHEDRON case only

    // step 1: extract local parmesh parts to the intermesh and
    // replace local vertex indices by the global indices

    BaseGeom = pmesh.BaseGeom;

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        if (myid == 0)
            cout << "This Mesh constructor works only for pentatops and tetrahedrons"
                 << endl << flush;
        return;
    }

    IntermediateMesh * local_intermesh = pmesh.ExtractMeshToInterMesh();

    FiniteElementCollection * lin_coll = new LinearFECollection;
    ParFiniteElementSpace * pspace = new ParFiniteElementSpace(&pmesh, lin_coll);

    int nv_global = pspace->GlobalTrueVSize(); // global number of vertices in the 4d mesh

    // writing the global vertex numbers inside the local IntermediateMesh(4d)
    int lvert;
    for ( int lvert = 0; lvert < local_intermesh->nv; ++lvert )
    {
        local_intermesh->vert_gindices[lvert] = pspace->GetGlobalTDofNumber(lvert);
    }

    //InterMeshPrint (local_intermesh, myid, "local_intermesh_inConvert");
    //MPI_Barrier(comm);

    // replacing local vertex indices by global indices from parFEspace
    // converting local to global vertex indices in elements
    for (int elind = 0; elind < local_intermesh->ne; ++elind)
    {
        for ( int j = 0; j < nvert_per_elem; ++j )
        {
            lvert = local_intermesh->elements[elind * nvert_per_elem + j];
            local_intermesh->elements[elind * nvert_per_elem + j] =
                    local_intermesh->vert_gindices[lvert];
        }
    }

    // converting local to global vertex indices in boundary elements
    for (int bdrelind = 0; bdrelind < local_intermesh->nbe; ++bdrelind)
    {
        for ( int j = 0; j < nvert_per_bdrelem; ++j )
        {
            lvert = local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j];
            local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j] =
                    local_intermesh->vert_gindices[lvert];
        }
    }

    delete lin_coll;
    delete pspace;


    // step 2: exchange local intermeshes between processors

    // 2.1: exchanging information about local sizes between processors
    // in order to set up mpi exchange parameters and allocate the future 4d mesh;

    // nvdg_global = sum of local number of vertices (without thinking that
    // some vertices are shared between processors)
    int nvdg_global, ne_global, nbe_global;

    int *recvcounts_el = new int[num_procs];
    MPI_Allgather( &(local_intermesh->ne), 1, MPI_INT, recvcounts_el, 1, MPI_INT, comm);

    int *rdispls_el = new int[num_procs];
    rdispls_el[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_el[i + 1] = rdispls_el[i] + recvcounts_el[i];

    ne_global = rdispls_el[num_procs - 1] + recvcounts_el[num_procs - 1];

    //cout << "ne_global = " << ne_global << endl;

    int * partioning_ = new int[ne_global];
    int elcount = 0;
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        for ( int el = 0; el < recvcounts_el[proc]; ++el )
        {
            partioning_[elcount++] = proc;
        }
    }

    *partioning = partioning_;

    int *recvcounts_be = new int[num_procs];

    MPI_Allgather( &(local_intermesh->nbe), 1, MPI_INT, recvcounts_be, 1, MPI_INT, comm);

    int *rdispls_be = new int[num_procs];

    rdispls_be[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_be[i + 1] = rdispls_be[i] + recvcounts_be[i];

    nbe_global = rdispls_be[num_procs - 1] + recvcounts_be[num_procs - 1];

    int *recvcounts_v = new int[num_procs];
    MPI_Allgather( &(local_intermesh->nv), 1, MPI_INT, recvcounts_v, 1, MPI_INT, comm);

    int *rdispls_v = new int[num_procs];
    rdispls_v[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_v[i + 1] = rdispls_v[i] + recvcounts_v[i];

    nvdg_global = rdispls_v[num_procs - 1] + recvcounts_v[num_procs - 1];

    MPI_Barrier(comm);

    Mesh::IntermediateMesh * intermesh_global = new IntermediateMesh;
    Mesh::IntermeshInit( intermesh_global, dim, nvdg_global, ne_global, nbe_global, 1);

    // 2.2: exchanging attributes, elements and vertices between processes using allgatherv

    // exchanging element attributes
    MPI_Allgatherv( local_intermesh->elattrs, local_intermesh->ne, MPI_INT,
                    intermesh_global->elattrs, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdr element attributes
    MPI_Allgatherv( local_intermesh->bdrattrs, local_intermesh->nbe, MPI_INT,
                    intermesh_global->bdrattrs, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging elements, changing recvcounts_el!!!
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_el[i] *= nvert_per_elem;
        rdispls_el[i] *= nvert_per_elem;
    }


    MPI_Allgatherv( local_intermesh->elements, (local_intermesh->ne)*nvert_per_elem, MPI_INT,
                    intermesh_global->elements, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdrelements, changing recvcounts_be!!!
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_be[i] *= nvert_per_bdrelem;
        rdispls_be[i] *= nvert_per_bdrelem;
    }


    MPI_Allgatherv( local_intermesh->bdrelements, (local_intermesh->nbe)*nvert_per_bdrelem, MPI_INT,
                    intermesh_global->bdrelements, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging global vertex indices
    MPI_Allgatherv( local_intermesh->vert_gindices, local_intermesh->nv, MPI_INT,
                    intermesh_global->vert_gindices, recvcounts_v, rdispls_v, MPI_INT, comm);

    // exchanging vertices : At the moment dg-type procedure = without considering presence
    // of shared vertices
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_v[i] *= dim;
        rdispls_v[i] *= dim;
    }

    MPI_Allgatherv( local_intermesh->vertices, (local_intermesh->nv)*dim, MPI_DOUBLE,
                    intermesh_global->vertices, recvcounts_v, rdispls_v, MPI_DOUBLE, comm);

    IntermeshDelete(local_intermesh);

    // step 3: load serial mesh4d from the created global intermesh4d

    InitMesh(dim,dim, nv_global, ne_global, nbe_global);

    // 3.1: creating the correct vertex array where each vertex is met only once
    // 3.1.1: cleaning up the vertices which are at the moment with multiple entries for shared
    // vertices

    int gindex;
    std::map<int, double*> vertices_unique; // map structure for storing only unique vertices

    // loop over all (with multiple entries) vertices, unique are added to the map object
    double * tempvert_map;
    for ( int i = 0; i < nvdg_global; ++i )
    {
        tempvert_map = new double[dim];
        for ( int j = 0; j < dim; j++ )
            tempvert_map[j] = intermesh_global->vertices[i * dim + j];
        gindex = intermesh_global->vert_gindices[i];
        vertices_unique[gindex] = tempvert_map;
    }

    // counting the final number of vertices. after that count_vert should be equal to nv_global
    int count_vert = 0;
    //for(auto const& ent : vertices_unique)
    //{
        //count_vert ++;
    //}
    // making compiler happy
    count_vert = vertices_unique.size();

    if ( count_vert != nv_global && myid == 0 )
    {
        cout << "Wrong number of vertices! Smth is probably wrong" << endl << flush;
    }

    // 3.1.2: creating the vertices array with taking care of shared vertices using
    // the std::map vertices_unique
    //delete [] intermesh_global->vertices;
    intermesh_global->nv = count_vert;
    //intermesh_global->vertices = new double[count_vert * dim];

    // now actual intermesh_global->vertices is:
    // right unique vertices + some vertices which are still alive after mpi transfer.
    // so we reuse the memory already allocated for vertices array with multiple entries.

    int tmp = 0;
    for(auto const& ent : vertices_unique)
    {
        for ( int j = 0; j < dim; j++)
        {
            intermesh_global->vertices[tmp*dim + j] = ent.second[j];
        }

        if ( tmp != ent.first )
            cout << "ERROR" << endl;
        tmp++;
    }

    vertices_unique.clear();

    //InterMeshPrint (intermesh_global, myid, "intermesh_reduced_inConvert");
    //MPI_Barrier(comm);

    // 3.2: loading created intermesh_global into a mfem mesh object
    // (temporarily copying the memory: FIX IT may be)
    if (dim == 4)
    {
        BaseGeom = Geometry::PENTATOPE;
        BaseBdrGeom = Geometry::TETRAHEDRON;
    }
    else // dim == 3
    {
        BaseGeom = Geometry::TETRAHEDRON;
        BaseBdrGeom = Geometry::TRIANGLE;
    }
    LoadMeshfromArrays( intermesh_global->nv, intermesh_global->vertices,
             intermesh_global->ne, intermesh_global->elements, intermesh_global->elattrs,
             intermesh_global->nbe, intermesh_global->bdrelements, intermesh_global->bdrattrs, dim );

    // 3.3 create the internal structure for mesh after el-s,, bdel-s and vertices have been loaded
    int refine = 1;
    CreateInternalMeshStructure(refine);

    IntermeshDelete (intermesh_global);

    MPI_Barrier (comm);
    return;
}
#endif

// Creates an IntermediateMesh whihc stores main arrays of the Mesh.
Mesh::IntermediateMesh * Mesh::ExtractMeshToInterMesh()
{
    int Dim = Dimension(), NumOfElements = GetNE(),
            NumOfBdrElements = GetNBE(),
            NumOfVertices = GetNV();

    if ( Dim != 4 && Dim != 3 )
    {
       cerr << "Wrong dimension in ExtractMeshToInterMesh(): " << Dim << endl;
       return NULL;
    }

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        cout << "ExtractMeshToInterMesh() is implemented only for pentatops and tetrahedrons" << endl;
        return NULL;
    }

    IntermediateMesh * intermesh = new IntermediateMesh;
    IntermeshInit( intermesh, Dim, NumOfVertices, NumOfElements, NumOfBdrElements, 1);

    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        Element * el = GetElement(elind);
        int * v = el->GetVertices();

        for ( int i = 0; i < Dim + 1; ++i )
            intermesh->elements[elind*(Dim+1) + i] = v[i];
        intermesh->elattrs[elind] = el->GetAttribute();
    }

    for ( int belind = 0; belind < GetNBE(); ++belind)
    {
        Element * el = GetBdrElement(belind);
        int * v = el->GetVertices();

        for ( int i = 0; i < Dim; ++i )
            intermesh->bdrelements[belind*Dim + i] = v[i];
        intermesh->bdrattrs[belind] = el->GetAttribute();
    }

    for ( int vind = 0; vind < GetNV(); ++vind)
    {
        double * coords = GetVertex(vind);

        for ( int i = 0; i < Dim; ++i )
            intermesh->vertices[vind*Dim + i] = coords[i];
    }

    return intermesh;
}


// Computes domain and boundary volumes, and checks,
// that faces and boundary elements lists are consistent with the actual element faces
int Mesh::MeshCheck (bool verbose)
{
    int dim = Dimension();

    if ( dim != 4 && dim != 3 && verbose )
    {
        cout << "Case dim != 3 or 4 is not supported in MeshCheck()" << endl;
        return -1;
    }

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON && verbose )
    {
        cout << "MeshCheck() is implemented only for pentatops and tetrahedrons" << endl;
        return -1;
    }

    if (verbose)
        cout << "Mesh checking:" << endl;

    // 2.5.0: assuming that vertices are fine, nothing is done for them

    // 2.5.1: volume check (+2.5.0) means that elements don't intersect
    // and no holes inside the domain are present
    double domain_volume = 0.0;
    double el_volume;
    double ** pointss = new double*[dim + 1];
    DenseMatrix VolumeEl;
    VolumeEl.SetSize(dim);

    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        Element * el = GetElement(elind);
        int * v = el->GetVertices();

        for ( int i = 0; i < dim + 1; ++i)
            pointss[i] = GetVertex(v[i]);

        for ( int i = 0; i < dim; ++i)
        {
            for ( int j = 0; j < dim; ++j)
            {
                VolumeEl.Elem(i,j) = pointss[i + 1][j] - pointss[0][j];
            }
        }

        el_volume = fabs (VolumeEl.Det() / factorial(dim)); //24 = 4!

        domain_volume += el_volume;
    }

    delete [] pointss;

    cout << "Domain volume from local mesh part = " << domain_volume << endl;

    // 2.5.2: Checking that faces are consistent
    int nbndface = 0, nintface = 0;

    for (int face = 0; face < GetNFaces(); ++face)
    {
       int el1, el2;
       GetFaceElements(face, &el1, &el2);

       //cout << "faceind = " << face << endl;
       //cout << "el indices: " << el1 << " and " << el2 << endl;

       if ( el1 == -1 || el2 == -1 )
           nbndface++;
       if ( el1 != -1 && el2 != -1 )
           nintface++;
    }

    //return 0;

    //cout << "nfaces = " << mesh.GetNFaces() << endl;
    //cout << "nbe = " << mesh.GetNBE() << endl;
    //cout << "nbndface = " << nbndface << ", nintface = " << nintface << endl;

    // 2.5.3: Checking the boundary volume
    double boundary_volume = 0.0;
    for ( int belind = 0; belind < GetNBE(); ++belind)
    {
        Element * el = GetBdrElement(belind);
        int * v = el->GetVertices();

        if (dim == 4)
        {
            double * point0 = GetVertex(v[0]);
            double * point1 = GetVertex(v[1]);
            double * point2 = GetVertex(v[2]);
            double * point3 = GetVertex(v[3]);

            double a1, a2, a3, a4, a5, a6;
            a1 = dist(point0, point1, dim);
            a2 = dist(point0, point2, dim);
            a3 = dist(point0, point3, dim);
            a4 = dist(point1, point2, dim);
            a5 = dist(point2, point3, dim);
            a6 = dist(point3, point1, dim);

            // formula from the webpage
            // http://keisan.casio.com/exec/system/1329962711
            el_volume = 0.0;
            el_volume += a1*a1*a5*a5*(a2*a2 + a3*a3 + a4*a4 + a6*a6 - a1*a1 - a5*a5);
            el_volume += a2*a2*a6*a6*(a1*a1 + a3*a3 + a4*a4 + a5*a5 - a2*a2 - a6*a6);
            el_volume += a3*a3*a4*a4*(a1*a1 + a2*a2 + a5*a5 + a6*a6 - a3*a3 - a4*a4);
            el_volume += - a1*a1*a2*a2*a4*a4 - a2*a2*a3*a3*a5*a5;
            el_volume += - a1*a1*a3*a3*a6*a6 - a4*a4*a5*a5*a6*a6;
            el_volume = el_volume/144.0;

            el_volume = sqrt(el_volume);
        }
        else // dim == 3
        {
            double * point0 = GetVertex(v[0]);
            double * point1 = GetVertex(v[1]);
            double * point2 = GetVertex(v[2]);

            double a1, a2, a3;
            a1 = dist(point0, point1, dim);
            a2 = dist(point0, point2, dim);
            a3 = dist(point1, point2, dim);

            // Heron's formula
            double halfp = 0.5 * (a1 + a2 + a3); //half of the perimeter
            el_volume = sqrt ( halfp * (halfp - a1) * (halfp - a2) * (halfp - a3) );
        }


        //cout << "bel_volume" << el_volume << endl;

        boundary_volume += el_volume;
    }

    cout << "Boundary volume from local mesh = " << boundary_volume << endl << flush;

    // 2.5.3: Checking faces using elements, brute-force type
    set<set<int> > BndElemSet;
    for ( int belind = 0; belind < GetNBE(); ++belind)
    {
        Element * el = GetBdrElement(belind);
        int * v = el->GetVertices();

        set<int> belset;

        for ( int i = 0; i < dim; ++i )
                belset.insert(v[i]);
        BndElemSet.insert(belset);
    }

    //cout << "BndElemSet size (how many different bdr elements in boundary) = " << BndElemSet.size() << endl;

    map<set<int>,int> FaceElemMap;
    int facecount = 0;
    int bndcountcheck = 0;
    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        Element * el = GetElement(elind);
        int * v = el->GetVertices();

        for ( int elface = 0; elface < dim + 1; ++elface)
        {
            set<int> faceset;

            for ( int i = 0; i < dim + 1; ++i )
                if (i != elface )
                    faceset.insert(v[i]);

            auto findset = FaceElemMap.find(faceset);
            if (findset != FaceElemMap.end() )
                FaceElemMap[faceset]++;
            else
            {
                FaceElemMap[faceset] = 1;
                facecount++;
            }

            auto findsetbel = BndElemSet.find(faceset);
            if (findsetbel != BndElemSet.end() )
                bndcountcheck++;
        }
    }

    //cout << "FaceElemMap: " << facecount << " faces" <<  endl;
    //cout << "Checking: bndcountcheck = " << bndcountcheck << endl;
    int bndmapcount = 0, intmapcount = 0;
    for(auto const& ent : FaceElemMap)
    {
        //for (int temp: ent.first)
            //cout << temp << " ";
        //cout << ": " << ent.second << endl;

        if (ent.second == 1)
            bndmapcount++;
        else if (ent.second == 2)
            intmapcount++;
        else
            cout << "ERROR: wrong intmapcount" << endl;
    }

    //cout << "Finally: bndmapcount = " << bndmapcount << ", intmapcount = " << intmapcount << endl;

    if ( bndmapcount != nbndface )
    {
        cout << "Something is wrong with bdr elements:" << endl;
        cout << "bndmapcount = " << bndmapcount << "must be equal to nbndface = " << nbndface << endl;
        return - 1;
    }
    if ( intmapcount != nintface )
    {
        cout << "Something is wrong with bdr elements:" << endl;
        cout << "intmapcount = " << intmapcount << "must be equal to nintface = " << nintface << endl;
        return - 1;
    }

    if (verbose)
        cout << "Bdr elements are consistent w.r.t elements!" << endl;

    return 0;
}


// Reads the elements, vertices and boundary from the input IntermediatMesh.
// It is like Load() in MFEM but for IntermediateMesh instead of an input stream.
// No internal mesh structures are initialized inside.
void Mesh::LoadMeshfromArrays( int nv, double * vertices, int ne, int * elements, int * elattrs,
                           int nbe, int * bdrelements, int * bdrattrs, int dim )
{
    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        cout << "LoadMeshfromArrays() is implemented only for pentatops and tetrahedrons" << endl;
        return;
    }
    int nvert_per_elem = dim + 1; // PENTATOPE and TETRAHEDRON case only
    int nvert_per_bdrelem = dim; // PENTATOPE and TETRAHEDRON case only

    Element * el;

    for (int j = 0; j < ne; j++)
    {
        if (dim == 4)
            el = new Pentatope(elements + j*nvert_per_elem);
        else // dim == 3
            el = new Tetrahedron(elements + j*nvert_per_elem);
        el->SetAttribute(elattrs[j]);

        AddElement(el);
    }

    for (int j = 0; j < nbe; j++)
    {
        if (dim == 4)
            el = new Tetrahedron(bdrelements + j*nvert_per_bdrelem);
        else // dim == 3
            el = new Triangle(bdrelements + j*nvert_per_bdrelem);
        el->SetAttribute(bdrattrs[j]);
        AddBdrElement(el);
    }

    for (int j = 0; j < nv; j++)
    {
        AddVertex(vertices + j * dim );
    }

    return;
}

// from a given base mesh (3d tetrahedrons or 2D triangles) produces a space-time mesh
// for a space-time cylinder with the given base, Nsteps * tau height in time
// enumeration of space-time vertices: time slab after time slab
// boundary attributes: 1 for t=0, 2 for lateral boundaries, 3 for t = tau*Nsteps
void Mesh::MeshSpaceTimeCylinder_onlyArrays ( Mesh& meshbase, double tau, int Nsteps,
                                              int bnd_method, int local_method)
{
    int DimBase = meshbase.Dimension(), NumOfBaseElements = meshbase.GetNE(),
            NumOfBaseBdrElements = meshbase.GetNBE(),
            NumOfBaseVertices = meshbase.GetNV();
    int NumOfSTElements, NumOfSTBdrElements, NumOfSTVertices;

    if ( DimBase != 3 && DimBase != 2 )
    {
        cerr << "Wrong dimension in MeshSpaceTimeCylinder(): " << DimBase << endl << flush;
        return;
    }

    if ( DimBase == 2 )
    {
        if ( local_method == 1 )
        {
            cerr << "This local method = " << local_method << " is not supported by case "
                                                     "dim = " << DimBase << endl << flush;
            return;
        }
    }

    int Dim = DimBase + 1;

    // for each base element and each time slab a space-time prism with base mesh element as a base
    // is decomposed into (Dim) simplices (tetrahedrons in 3d and pentatops in 4d);
    NumOfSTElements = NumOfBaseElements * Dim * Nsteps;
    NumOfSTVertices = NumOfBaseVertices * (Nsteps + 1); // no additional vertices inbetween time slabs so far
    // lateral 4d bdr faces (one for each 3d bdr face) + lower + upper bases
    // of the space-time cylinder
    NumOfSTBdrElements = NumOfBaseBdrElements * DimBase * Nsteps + 2 * NumOfBaseElements;

    // assuming that the 3D mesh contains elements of the same type = tetrahedrons
    int vert_per_base = meshbase.GetElement(0)->GetNVertices();
    int vert_per_prism = 2 * vert_per_base;
    int vert_per_latface = DimBase * 2;

    InitMesh(Dim,Dim,NumOfSTVertices,NumOfSTElements,NumOfSTBdrElements);

    Element * el;

    int * simplexes;
    if (local_method == 1 || local_method == 2)
    {
        simplexes = new int[Dim * (Dim + 1)]; // array for storing vertex indices for constructed simplices
    }
    else // local_method = 0
    {
        int nsliver = 5; //why 5? how many slivers can b created by qhull? maybe 0 if we don't joggle inside qhull but perturb the coordinates before?
        simplexes = new int[(Dim + nsliver) * (Dim + 1)]; // array for storing vertex indices for constructed simplices + probably sliver pentatopes
    }

    // stores indices of space-time element face vertices produced by qhull for all lateral faces
    // Used in local_method = 1 only.
    int * facesimplicesAll;
    if (local_method == 1 )
        facesimplicesAll = new int[DimBase * (DimBase + 1) * Dim ];

    Array<int> elverts_base;
    Array<int> elverts_prism;

    // temporary array for vertex indices of a pentatope face (used in local_method = 0 and 2)
    int * tempface = new int[Dim];
    int * temp = new int[Dim+1]; //temp array for simplex vertices in local_method = 1;

    // three arrays below are used only in local_method = 1
    Array2D<int> vert_to_vert_prism; // for a 4D prism
    // row ~ lateral face of the 4d prism
    // first 6 columns - indices of vertices belonging to the lateral face,
    // last 2 columns - indices of the rest 2 vertices of the prism
    Array2D<int> latfacets_struct;
    // coordinates of vertices of a lateral face of 4D prism
    double * vert_latface;
    // coordinates of vertices of a 3D base (triangle) of a lateral face of 4D prism
    double * vert_3Dlatface;
    if (local_method == 1)
    {
        vert_latface =  new double[Dim * vert_per_latface];
        vert_3Dlatface = new double[DimBase * vert_per_latface];
        latfacets_struct.SetSize(Dim, vert_per_prism);
        vert_to_vert_prism.SetSize(vert_per_prism, vert_per_prism);
    }

    // coordinates of vertices of the space-time prism
    double * elvert_coordprism = new double[Dim * vert_per_prism];

    char * qhull_flags;
    if (local_method == 0 || local_method == 1)
    {
        qhull_flags = new char[250];
        sprintf(qhull_flags, "qhull d Qbb");
    }

    int simplex_count = 0;
    Element * NewEl;
    Element * NewBdrEl;

    double * tempvert = new double[Dim];

    if (local_method < 0 && local_method > 2)
    {
        cout << "Local method = " << local_method << " is not supported" << endl << flush;
        return;
    }

    if ( bnd_method != 0 && bnd_method != 1)
    {
        cout << "Illegal value of bnd_method = " << bnd_method << " (must be 0 or 1)"
             << endl << flush;
        return;
    }

    Vector vert_coord3d(DimBase * meshbase.GetNV());
    meshbase.GetVertices(vert_coord3d);
    //printDouble2D(vert_coord3d, 10, Dim3D);

    // adding all space-time vertices to the mesh
    for ( int tslab = 0; tslab <= Nsteps; ++tslab)
    {
        // adding the vertices from the slab to the output space-time mesh
        for ( int vert = 0; vert < NumOfBaseVertices; ++vert)
        {
            for ( int j = 0; j < DimBase; ++j)
            {
                tempvert[j] = vert_coord3d[vert + j * NumOfBaseVertices];
                tempvert[Dim-1] = tau * tslab;
            }
            AddVertex(tempvert);
        }
    }

    delete [] tempvert;

    int * almostjogglers = new int[Dim];
    //int permutation[Dim];
    //vector<double*> lcoords(Dim);
    vector<vector<double> > lcoordsNew(Dim);

    // for each (of Dim) base mesh element faces stores 1 if it is at the boundary and 0 else
    int facebdrmarker[Dim];
    // std::set of the base mesh boundary elements. Using set allows one to perform a search
    // with O(log N_elem) operations
    std::set< std::vector<int> > BdrTriSet;
    Element * bdrel;

    Array<int> face_bndflags;
    if (bnd_method == 1)
    {
        if (Dim == 4)
            face_bndflags.SetSize(meshbase.GetNFaces());
        if (Dim == 3)
            face_bndflags.SetSize(meshbase.GetNEdges());
    }

    Table * localel_to_face;
    Array<int> localbe_to_face;

    // if = 0, a search algorithm is used for defining whether faces of a given base mesh element
    // are at the boundary.
    // if = 1, instead an array face_bndflags is used, which stores 0 and 1 depending on
    // whether the face is at the boundary, + el_to_face table which is usually already
    // generated for the base mesh
    //int bnd_method = 1;

    if (bnd_method == 0)
    {
        // putting base mesh boundary elements from base mesh structure to the set BdrTriSet
        for ( int boundelem = 0; boundelem < NumOfBaseBdrElements; ++boundelem)
        {
            //cout << "boundelem No. " << boundelem << endl;
            bdrel = meshbase.GetBdrElement(boundelem);
            int * bdrverts = bdrel->GetVertices();

            std::vector<int> buff (bdrverts, bdrverts+DimBase);
            std::sort (buff.begin(), buff.begin()+DimBase);

            BdrTriSet.insert(buff);
        }
        /*
        for (vector<int> temp : BdrTriSet)
        {
            cout << temp[0] << " " <<  temp[1] << " " << temp[2] << endl;
        }
        cout<<endl;
        */
    }
    else // bnd_method = 1
    {
        if (Dim == 4)
        {
            if (meshbase.el_to_face == NULL)
            {
                cout << "Have to built el_to_face" << endl;
                meshbase.GetElementToFaceTable(0);
            }
            localel_to_face = meshbase.el_to_face;
            localbe_to_face.MakeRef(meshbase.be_to_face);
        }
        if (Dim == 3)
        {
            if (meshbase.el_to_edge == NULL)
            {
                cout << "Have to built el_to_edge" << endl;
                meshbase.GetElementToEdgeTable(*(meshbase.el_to_edge), meshbase.be_to_edge);
            }
            localel_to_face = meshbase.el_to_edge;
            localbe_to_face.MakeRef(meshbase.be_to_edge);
        }

        //cout << "Special print" << endl;
        //cout << mesh3d.el_to_face(elind, facelind);
        //cout << "be_to_face" << endl;
        //mesh3d.be_to_face.Print();
        //localbe_to_face.Print();


        //cout << "nfaces = " << meshbase.GetNFaces();
        //cout << "nbe = " << meshbase.GetNBE() << endl;
        //cout << "boundary.size = " << mesh3d.boundary.Size() << endl;

        face_bndflags = -1;
        for ( int i = 0; i < meshbase.GetNBE(); ++i )
            //face_bndflags[meshbase.be_to_face[i]] = 1;
            face_bndflags[localbe_to_face[i]] = 1;

        //cout << "face_bndflags" << endl;
        //face_bndflags.Print();
    }

    int * ordering = new int [vert_per_base];
    //int antireordering[vert_per_base]; // used if bnd_method = 0 and local_method = 2
    Array<int> tempelverts(vert_per_base);

    // main loop creates space-time elements over all time slabs over all base mesh elements
    // loop over base mesh elements
    for ( int elind = 0; elind < NumOfBaseElements; elind++ )
    //for ( int elind = 0; elind < 1; ++elind )
    {
        //cout << "element " << elind << endl;

        el = meshbase.GetElement(elind);

        // 1. getting indices of base mesh element vertices and their coordinates in the prism
        el->GetVertices(elverts_base);

        //for ( int k = 0; k < elverts_base.Size(); ++k )
          //  cout << "elverts[" << k << "] = " << elverts_base[k] << endl;

        // for local_method 2 we need to reorder the local vertices of the prism to preserve
        // the the order in some global sense  = lexicographical order of the vertex coordinates
        if (local_method == 2)
        {
            // using elvert_coordprism as a temporary buffer for changing elverts_base
            for ( int vert = 0; vert < vert_per_base; ++vert)
            {
                for ( int j = 0; j < DimBase; ++j)
                {
                    elvert_coordprism[Dim * vert + j] =
                            vert_coord3d[elverts_base[vert] + j * NumOfBaseVertices];
                }
            }

            /*
             * old one
            for (int vert = 0; vert < Dim; ++vert)
                lcoords[vert] = elvert_coordprism + Dim * vert;

            sortingPermutation(DimBase, lcoords, ordering);

            cout << "ordering 1:" << endl;
            for ( int i = 0; i < vert_per_base; ++i)
                cout << ordering[i] << " ";
            cout << endl;
            */

            for (int vert = 0; vert < Dim; ++vert)
                lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                        elvert_coordprism + Dim * vert + DimBase);

            sortingPermutationNew(lcoordsNew, ordering);

            //cout << "ordering 2:" << endl;
            //for ( int i = 0; i < vert_per_base; ++i)
                //cout << ordering[i] << " ";
            //cout << endl;

            // UGLY: Fix it
            for ( int i = 0; i < vert_per_base; ++i)
                tempelverts[i] = elverts_base[ordering[i]];

            for ( int i = 0; i < vert_per_base; ++i)
                elverts_base[i] = tempelverts[i];
        }

        // 2. understanding which of the base mesh element faces (triangles) are at the boundary
        int local_nbdrfaces = 0;
        set<set<int> > LocalBdrs;
        if (bnd_method == 0) // in this case one looks in the set of base mesh boundary elements
        {
            vector<int> face(DimBase);
            for (int i = 0; i < Dim; ++i )
            {
                // should be consistent with lateral faces ordering in latfacet structure
                // if used with local_method = 1

                for ( int j = 0; j < DimBase; ++j)
                    face[j] = elverts_base[(i+j)%Dim];

                sort(face.begin(), face.begin()+DimBase);
                //cout << face[0] << " " <<  face[1] << " " << face[2] << endl;

                if (BdrTriSet.find(face) != BdrTriSet.end() )
                {
                    local_nbdrfaces++;
                    facebdrmarker[i] = 1;
                    set<int> face_as_set;

                    for ( int j = 0; j < DimBase; ++j)
                        face_as_set.insert((i+j)%Dim);

                    LocalBdrs.insert(face_as_set);
                }
                else
                    facebdrmarker[i] = 0;
            }

        } //end of if bnd_method == 0
        else // in this case one uses el_to_face and face_bndflags to check whether mesh base
             //face is at the boundary
        {
            int * faceinds = localel_to_face->GetRow(elind);
            Array<int> temp(DimBase);
            for ( int facelind = 0; facelind < Dim; ++facelind)
            {
                int faceind = faceinds[facelind];
                if (face_bndflags[faceind] == 1)
                {
                    meshbase.GetFaceVertices(faceind, temp);

                    set<int> face_as_set;
                    for ( int vert = 0; vert < DimBase; ++vert )
                        face_as_set.insert(temp[vert]);

                    LocalBdrs.insert(face_as_set);

                    local_nbdrfaces++;
                }

            } // end of loop over element faces

        }

        //cout << "Welcome the facebdrmarker" << endl;
        //printInt2D(facebdrmarker, 1, Dim);

        /*
        cout << "Welcome the LocalBdrs" << endl;
        for ( set<int> tempset: LocalBdrs )
        {
            cout << "element of LocalBdrs for el = " << elind << endl;
            for (int ind: tempset)
                cout << ind << " ";
            cout << endl;
        }
        */

        // 3. loop over all space-time slabs above a given mesh base element
        for ( int tslab = 0; tslab < Nsteps; ++tslab)
        {
            //cout << "tslab " << tslab << endl;

            //3.1 getting vertex indices for the space-time prism
            elverts_prism.SetSize(vert_per_prism);

            for ( int i = 0; i < vert_per_base; ++i)
            {
                elverts_prism[i] = elverts_base[i] + tslab * NumOfBaseVertices;
                elverts_prism[i + vert_per_base] = elverts_base[i] +
                        (tslab + 1) * NumOfBaseVertices;
            }
            //cout << "New elverts_prism" << endl;
            //elverts_prism.Print(cout, 10);
            //return;


            // 3.2 for the first time slab we add the base mesh elements in the lower base
            // to the space-time bdr elements
            if ( tslab == 0 )
            {
                //cout << "zero slab: adding boundary element:" << endl;
                if (Dim == 3)
                    NewBdrEl = new Triangle(elverts_prism);
                if (Dim == 4)
                    NewBdrEl = new Tetrahedron(elverts_prism);
                NewBdrEl->SetAttribute(1);
                AddBdrElement(NewBdrEl);
            }
            // 3.3 for the last time slab we add the base mesh elements in the upper base
            // to the space-time bdr elements
            if ( tslab == Nsteps - 1 )
            {
                //cout << "last slab: adding boundary element:" << endl;
                if (Dim == 3)
                    NewBdrEl = new Triangle(elverts_prism + vert_per_base);
                if (Dim == 4)
                    NewBdrEl = new Tetrahedron(elverts_prism + vert_per_base);
                NewBdrEl->SetAttribute(3);
                AddBdrElement(NewBdrEl);
            }

            if (local_method == 0 || local_method == 1)
            {
                // 3.4 setting vertex coordinates for space-time prism, lower base
                for ( int vert = 0; vert < vert_per_base; ++vert)
                {
                    for ( int j = 0; j < DimBase; ++j)
                        elvert_coordprism[Dim * vert + j] =
                                vert_coord3d[elverts_base[vert] + j * NumOfBaseVertices];
                    elvert_coordprism[Dim * vert + Dim-1] = tslab * tau;
                }

                //cout << "Welcome the vertex coordinates for the 4d prism base " << endl;
                //printDouble2D(elvert_coordprism, vert_per_base, Dim);

                /*
                 * old
                for (int vert = 0; vert < Dim; ++vert)
                    lcoords[vert] = elvert_coordprism + Dim * vert;


                //cout << "vector double * lcoords:" << endl;
                //for ( int i = 0; i < Dim; ++i)
                    //cout << "lcoords[" << i << "]: " << lcoords[i][0] << " " << lcoords[i][1] << " " << lcoords[i][2] << endl;

                sortingPermutation(DimBase, lcoords, permutation);
                */

                // here we compute the permutation "ordering" which preserves the geometric order of vertices
                // which is based on their coordinates comparison and compute jogglers for qhull
                // from the "ordering"

                for (int vert = 0; vert < Dim; ++vert)
                    lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                            elvert_coordprism + Dim * vert + DimBase);

                sortingPermutationNew(lcoordsNew, ordering);


                //cout << "Welcome the permutation:" << endl;
                //cout << ordering[0] << " " << ordering[1] << " " <<ordering[2] << " " << ordering[3] << endl;

                int joggle_coeff = 0;
                for ( int i = 0; i < Dim; ++i)
                    almostjogglers[ordering[i]] = joggle_coeff++;


                // 3.5 setting vertex coordinates for space-time prism, upper layer
                // Joggling is required for getting unique Delaunay tesselation and should be
                // the same for vertices shared between different elements or at least produce
                // the same Delaunay triangulation in the shared faces.
                // So here it is not exactly the same, but if joggle(vertex A) > joggle(vertex B)
                // on one element, then the same inequality will hold in another element which also has
                // vertices A and B.
                double joggle;
                for ( int vert = 0; vert < vert_per_base; ++vert)
                {
                    for ( int j = 0; j < DimBase; ++j)
                        elvert_coordprism[Dim * (vert_per_base + vert) + j] =
                                elvert_coordprism[Dim * vert + j];
                    joggle = 1.0e-2 * (almostjogglers[vert]);
                    //joggle = 1.0e-2 * elverts_prism[i + vert_per_base] * 1.0 / NumOf4DVertices;
                    //double joggle = 1.0e-2 * i;
                    elvert_coordprism[Dim * (vert_per_base + vert) + Dim-1] =
                            (tslab + 1) * tau * ( 1.0 + joggle );
                }

                //cout << "Welcome the vertex coordinates for the 4d prism" << endl;
                //printDouble2D(elvert_coordprism, 2 * vert_per_base, Dim);

                // 3.6 - 3.10: constructing new space-time simplices and space-time boundary elements
                if (local_method == 0)
                {
#ifdef WITH_QHULL
                    qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                    qhT *qh= &qh_qh;
                    int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                    double volumetol = 1.0e-8;
                    qhull_wrapper(simplexes, qh, elvert_coordprism, Dim, volumetol, qhull_flags);

                    qh_freeqhull(qh, !qh_ALL);
                    qh_memfreeshort(qh, &curlong, &totlong);
                    if (curlong || totlong)  /* could also check previous runs */
                    {
                      fprintf(stderr, "qhull internal warning (user_eg, #3): did not free %d bytes"
                                      " of long memory (%d pieces)\n", totlong, curlong);
                    }
#else
                        cout << "Wrong local method, WITH_QHULL flag was not set" << endl;
#endif
                } // end of if local_method = 0

                if (local_method == 1) // works only in 4D case. Just historically the first implementation
                {
                    setzero(&vert_to_vert_prism);

                    // 3.6 creating vert_to_vert for the prism before Delaunay
                    // (adding 4d prism edges)
                    for ( int i = 0; i < el->GetNEdges(); i++)
                    {
                        const int * edge = el->GetEdgeVertices(i);
                        //cout << "edge: " << edge[0] << " " << edge[1] << std::endl;
                        vert_to_vert_prism(edge[0], edge[1]) = 1;
                        vert_to_vert_prism(edge[1], edge[0]) = 1;
                        vert_to_vert_prism(edge[0] + vert_per_base, edge[1] + vert_per_base) = 1;
                        vert_to_vert_prism(edge[1] + vert_per_base, edge[0] + vert_per_base) = 1;
                    }

                    for ( int i = 0; i < vert_per_base; i++)
                    {
                        vert_to_vert_prism(i, i) = 1;
                        vert_to_vert_prism(i + vert_per_base, i + vert_per_base) = 1;
                        vert_to_vert_prism(i, i + vert_per_base) = 1;
                        vert_to_vert_prism(i + vert_per_base, i) = 1;
                    }

                    //cout << "vert_to_vert before delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);
                    //cout << endl;

                    // 3.7 creating latfacet structure (brute force), for 4D tetrahedron case
                    // indices are local w.r.t to the 4d prism!!!
                    latfacets_struct(0,0) = 0;
                    latfacets_struct(0,1) = 1;
                    latfacets_struct(0,2) = 2;
                    latfacets_struct(0,6) = 3;

                    latfacets_struct(1,0) = 1;
                    latfacets_struct(1,1) = 2;
                    latfacets_struct(1,2) = 3;
                    latfacets_struct(1,6) = 0;

                    latfacets_struct(2,0) = 2;
                    latfacets_struct(2,1) = 3;
                    latfacets_struct(2,2) = 0;
                    latfacets_struct(2,6) = 1;

                    latfacets_struct(3,0) = 3;
                    latfacets_struct(3,1) = 0;
                    latfacets_struct(3,2) = 1;
                    latfacets_struct(3,6) = 2;

                    for ( int i = 0; i < Dim; ++i)
                    {
                        latfacets_struct(i,3) = latfacets_struct(i,0) + vert_per_base;
                        latfacets_struct(i,4) = latfacets_struct(i,1) + vert_per_base;
                        latfacets_struct(i,5) = latfacets_struct(i,2) + vert_per_base;
                        latfacets_struct(i,7) = latfacets_struct(i,6) + vert_per_base;
                    }

                    //cout << "latfacets_struct (vertex indices)" << endl;
                    //printArr2DInt (&latfacets_struct);

                    //(*)const int * base_face = el->GetFaceVertices(i); // not implemented in MFEM for Tetrahedron ?!

                    int * tetrahedrons;
                    int shift = 0;


                    // 3.8 loop over lateral facets, creating Delaunay triangulations
                    for ( int latfacind = 0; latfacind < Dim; ++latfacind)
                    {
                        //cout << "latface = " << latfacind << endl;
                        for ( int vert = 0; vert < vert_per_latface ; ++vert )
                        {
                            //cout << "vert index = " << latfacets_struct(latfacind,vert) << endl;
                            for ( int coord = 0; coord < Dim; ++coord)
                            {
                                vert_latface[vert*Dim + coord] =
                                  elvert_coordprism[latfacets_struct(latfacind,vert) * Dim + coord];
                            }

                        }

                        //cout << "Welcome the vertices of a lateral face" << endl;
                        //printDouble2D(vert_latface, vert_per_latface, Dim);

                        // creating from 3Dprism in 4D a true 3D prism in 3D by change of
                        // coordinates = computing input argument vert_3Dlatface for qhull wrapper
                        // we know that the first three coordinated of a lateral face is actually
                        // a triangle, so we set the first vertex to be the origin,
                        // the first-to-second edge to be one of the axis
                        if ( Dim == 4 )
                        {
                            double x1, x2, x3, y1, y2, y3;
                            double dist12, dist13, dist23;
                            double area, h, p;

                            dist12 = dist(vert_latface, vert_latface+Dim , Dim);
                            dist13 = dist(vert_latface, vert_latface+2*Dim , Dim);
                            dist23 = dist(vert_latface+Dim, vert_latface+2*Dim , Dim);

                            p = 0.5 * (dist12 + dist13 + dist23);
                            area = sqrt (p * (p - dist12) * (p - dist13) * (p - dist23));
                            h = 2.0 * area / dist12;

                            x1 = 0.0;
                            y1 = 0.0;
                            x2 = dist12;
                            y2 = 0.0;
                            if ( dist13 - h < 0.0 )
                                if ( fabs(dist13 - h) > 1.0e-10)
                                {
                                    std::cout << "strange: dist13 = " << dist13 << " h = "
                                              << h << std::endl;
                                    return;
                                }
                                else
                                    x3 = 0.0;
                            else
                                x3 = sqrt(dist13*dist13 - h*h);
                            y3 = h;


                            // the time coordinate remains the same
                            for ( int vert = 0; vert < vert_per_latface ; ++vert )
                                vert_3Dlatface[vert*DimBase + 2] = vert_latface[vert*Dim + 3];

                            // first & fourth vertex
                            vert_3Dlatface[0*DimBase + 0] = x1;
                            vert_3Dlatface[0*DimBase + 1] = y1;
                            vert_3Dlatface[3*DimBase + 0] = x1;
                            vert_3Dlatface[3*DimBase + 1] = y1;

                            // second & fifth vertex
                            vert_3Dlatface[1*DimBase + 0] = x2;
                            vert_3Dlatface[1*DimBase + 1] = y2;
                            vert_3Dlatface[4*DimBase + 0] = x2;
                            vert_3Dlatface[4*DimBase + 1] = y2;

                            // third & sixth vertex
                            vert_3Dlatface[2*DimBase + 0] = x3;
                            vert_3Dlatface[2*DimBase + 1] = y3;
                            vert_3Dlatface[5*DimBase + 0] = x3;
                            vert_3Dlatface[5*DimBase + 1] = y3;
                        } //end of creating a true 3d prism

                        //cout << "Welcome the vertices of a lateral face in 3D" << endl;
                        //printDouble2D(vert_3Dlatface, vert_per_latface, Dim3D);

                        tetrahedrons = facesimplicesAll + shift;

#ifdef WITH_QHULL
                        qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                        qhT *qh= &qh_qh;
                        int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                        double volumetol = MYZEROTOL;
                        qhull_wrapper(tetrahedrons, qh, vert_3Dlatface, DimBase, volumetol, qhull_flags);

                        qh_freeqhull(qh, !qh_ALL);
                        qh_memfreeshort(qh, &curlong, &totlong);
                        if (curlong || totlong)  /* could also check previous runs */
                          cerr<< "qhull internal warning (user_eg, #3): did not free " << totlong
                          << "bytes of long memory (" << curlong << " pieces)" << endl;
#else
                        cout << "Wrong local method, WITH_QHULL flag was not set" << endl;
#endif
                        // convert local 3D prism (lateral face) vertex indices back to the
                        // 4D prism indices and adding boundary elements from tetrahedrins
                        // for lateral faces of the 4d prism ...
                        for ( int tetraind = 0; tetraind < DimBase; ++tetraind)
                        {
                            //cout << "tetraind = " << tetraind << endl;

                            for ( int vert = 0; vert < Dim; ++vert)
                            {
                                int temp = tetrahedrons[tetraind*Dim + vert];
                                tetrahedrons[tetraind*Dim + vert] = latfacets_struct(latfacind, temp);
                            }

                            if ( bnd_method == 0 )
                            {
                                if ( facebdrmarker[latfacind] == 1 )
                                {
                                    //cout << "lateral facet " << latfacind << " is at the boundary: adding bnd element" << endl;

                                    tempface[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    tempface[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    tempface[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    tempface[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    // wrong because indices in tetrahedrons are local to 4d prism
                                    //NewBdrTri = new Tetrahedron(tetrahedrons + tetraind*Dim);

                                    NewBdrEl = new Tetrahedron(tempface);
                                    NewBdrEl->SetAttribute(2);
                                    AddBdrElement(NewBdrEl);

                                }
                            }
                            else // bnd_method = 1
                            {
                                set<int> latface3d_set;
                                for ( int i = 0; i < DimBase; ++i)
                                    latface3d_set.insert(elverts_prism[latfacets_struct(latfacind,i)] % NumOfBaseVertices);

                                // checking whether a face is at the boundary of 3d mesh
                                if ( LocalBdrs.find(latface3d_set) != LocalBdrs.end())
                                {
                                    // converting local indices to global indices and
                                    // adding the new boundary element
                                    tempface[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    tempface[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    tempface[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    tempface[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    NewBdrEl = new Tetrahedron(tempface);
                                    NewBdrEl->SetAttribute(2);
                                    AddBdrElement(NewBdrEl);
                                }
                            }



                         } //end of loop over tetrahedrons for a given lateral face

                        shift += DimBase * (DimBase + 1);

                        //return;
                    } // end of loop over lateral faces

                    // 3.9 adding the new edges from created tetrahedrons into the vert_to_vert
                    for ( int k = 0; k < Dim; ++k )
                        for (int i = 0; i < DimBase; ++i )
                        {
                            int vert0 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 0];
                            int vert1 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 1];
                            int vert2 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 2];
                            int vert3 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 3];

                            vert_to_vert_prism(vert0, vert1) = 1;
                            vert_to_vert_prism(vert1, vert0) = 1;

                            vert_to_vert_prism(vert0, vert2) = 1;
                            vert_to_vert_prism(vert2, vert0) = 1;

                            vert_to_vert_prism(vert0, vert3) = 1;
                            vert_to_vert_prism(vert3, vert0) = 1;

                            vert_to_vert_prism(vert1, vert2) = 1;
                            vert_to_vert_prism(vert2, vert1) = 1;

                            vert_to_vert_prism(vert1, vert3) = 1;
                            vert_to_vert_prism(vert3, vert1) = 1;

                            vert_to_vert_prism(vert2, vert3) = 1;
                            vert_to_vert_prism(vert3, vert2) = 1;
                        }

                    //cout << "vert_to_vert after delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);

                    int count_penta = 0;

                    // 3.10 creating finally 4d pentatopes:
                    // take a tetrahedron related to a lateral face, find out which of the rest
                    // 2 vertices of the 4d prism (one is not) is connected to all vertices of
                    // tetrahedron, and get a pentatope from tetrahedron + this vertex
                    // If pentatope is new, add it to the final structure
                    // To make checking for new pentatopes easy, reoder the pentatope indices
                    // in the default std order

                    for ( int tetraind = 0; tetraind < DimBase * Dim; ++tetraind)
                    {
                        // creating a pentatop temp
                        int latface_ind = tetraind / DimBase;
                        for ( int vert = 0; vert < Dim; vert++ )
                            temp[vert] = facesimplicesAll[tetraind * Dim + vert];

                        //cout << "tetrahedron" << endl;
                        //printInt2D(temp,1,4); // tetrahedron

                        bool isconnected = true;
                        for ( int vert = 0; vert < 4; ++vert)
                            if (vert_to_vert_prism(temp[vert],
                                                   latfacets_struct(latface_ind,6)) == 0)
                                isconnected = false;

                        if ( isconnected == true)
                            temp[4] = latfacets_struct(latface_ind,6);
                        else
                        {
                            bool isconnectedCheck = true;
                            for ( int vert = 0; vert < 4; ++vert)
                                if (vert_to_vert_prism(temp[vert],
                                                       latfacets_struct(latface_ind,7)) == 0)
                                    isconnectedCheck = false;
                            if (isconnectedCheck == 0)
                            {
                                cout << "Error: Both vertices are disconnected" << endl;
                                cout << "tetraind = " << tetraind << ", checking for " <<
                                             latfacets_struct(latface_ind,6) << " and " <<
                                             latfacets_struct(latface_ind,7) << endl;
                                return;
                            }
                            else
                                temp[4] = latfacets_struct(latface_ind,7);
                        }

                        //printInt2D(temp,1,5);

                        // replacing local vertex indices w.r.t to 4d prism to global!
                        temp[0] = elverts_prism[temp[0]];
                        temp[1] = elverts_prism[temp[1]];
                        temp[2] = elverts_prism[temp[2]];
                        temp[3] = elverts_prism[temp[3]];
                        temp[4] = elverts_prism[temp[4]];

                        // sorting the vertex indices
                        std::vector<int> buff (temp, temp+5);
                        std::sort (buff.begin(), buff.begin()+5);

                        // looking whether the current pentatop is new
                        bool isnew = true;
                        for ( int i = 0; i < count_penta; ++i )
                        {
                            std::vector<int> pentatop (simplexes+i*(Dim+1), simplexes+(i+1)*(Dim+1));

                            if ( pentatop == buff )
                                isnew = false;
                        }

                        if ( isnew == true )
                        {
                            for ( int i = 0; i < Dim + 1; ++i )
                                simplexes[count_penta*(Dim+1) + i] = buff[i];
                            //cout << "found a new pentatop from tetraind = " << tetraind << endl;
                            //cout << "now we have " << count_penta << " pentatops" << endl;
                            //printInt2D(pentatops + count_penta*(Dim+1), 1, Dim + 1);

                            ++count_penta;
                        }
                        //cout << "element " << elind << endl;
                        //printInt2D(pentatops, count_penta, Dim + 1);
                    }

                    //cout<< count_penta << " pentatops created" << endl;
                    if ( count_penta != Dim )
                        cout << "Error: Wrong number of simplexes constructed: got " <<
                                count_penta << ", needed " << Dim << endl << flush;
                    //printInt2D(pentatops, count_penta, Dim + 1);

                }

            } //end of if local_method = 0 or 1
            else // local_method == 2
            {
                // The simplest way to generate space-time simplices.
                // But requires to reorder the vertices at first, as done before.
                for ( int count_simplices = 0; count_simplices < Dim; ++count_simplices)
                {
                    for ( int i = 0; i < Dim + 1; ++i )
                    {
                        simplexes[count_simplices*(Dim+1) + i] = count_simplices + i;
                    }

                }
                //cout << "Welcome created pentatops" << endl;
                //printInt2D(pentatops, Dim, Dim + 1);
            }


            // adding boundary elements in local method =  0 or 2
            if (local_method == 0 || local_method == 2)
            {
                //if (local_method == 2)
                    //for ( int i = 0; i < vert_per_base; ++i)
                        //antireordering[ordering[i]] = i;

                if (local_nbdrfaces > 0) //if there is at least one base mesh element face at
                                         // the boundary for a given base element
                {
                    for ( int simplexind = 0; simplexind < Dim; ++simplexind)
                    {
                        //cout << "simplexind = " << simplexind << endl;
                        //printInt2D(pentatops + pentaind*(Dim+1), 1, 5);

                        for ( int faceind = 0; faceind < Dim + 1; ++faceind)
                        {
                            //cout << "faceind = " << faceind << endl;
                            set<int> faceproj;

                            // creating local vertex indices for a simplex face
                            // and projecting the face onto the 3d base
                            if (bnd_method == 0)
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        tempface[cnt] = simplexes[simplexind*(Dim + 1) + j];
                                        if (tempface[cnt] > vert_per_base - 1)
                                            faceproj.insert(tempface[cnt] - vert_per_base);
                                        else
                                            faceproj.insert(tempface[cnt]);
                                        cnt++;
                                    }
                                }

                                //cout << "tempface in local indices" << endl;
                                //printInt2D(tempface,1,4);
                            }
                            else // for bnd_method = 1 we create tempface and projection
                                 // in global indices
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        tempface[cnt] =
                                                elverts_prism[simplexes[simplexind*(Dim + 1) + j]];
                                        faceproj.insert(tempface[cnt] % NumOfBaseVertices );
                                        cnt++;
                                    }
                                }

                                //cout << "tempface in global indices" << endl;
                                //printInt2D(tempface,1,4);
                            }

                            /*
                            cout << "faceproj:" << endl;
                            for ( int temp : faceproj)
                                cout << temp << " ";
                            cout << endl;
                            */

                            // checking whether the projection is at the boundary of base mesh
                            // using the local-to-element LocalBdrs set which has at most Dim elements
                            if ( LocalBdrs.find(faceproj) != LocalBdrs.end())
                            {
                                //cout << "Found a new boundary element" << endl;
                                //cout << "With local indices: " << endl;
                                //printInt2D(tempface, 1, Dim);

                                // converting local indices to global indices and
                                // adding the new boundary element
                                if (bnd_method == 0)
                                {
                                    for ( int facevert = 0; facevert < Dim; ++facevert )
                                        tempface[facevert] = elverts_prism[tempface[facevert]];
                                }

                                //cout << "With global indices: " << endl;
                                //printInt2D(tempface, 1, Dim);

                                if (Dim == 3)
                                    NewBdrEl = new Triangle(tempface);
                                if (Dim == 4)
                                    NewBdrEl = new Tetrahedron(tempface);
                                NewBdrEl->SetAttribute(2);
                                AddBdrElement(NewBdrEl);
                            }


                        } // end of loop over space-time simplex faces
                    } // end of loop over space-time simplices
                } // end of if local_nbdrfaces > 0

                // By this point, for the given base mesh element:
                // space-time elements are constructed, but stored in local array
                // boundary elements are constructed which correspond to the elements in the space-time prism
                // converting local-to-prism indices in simplices to the global indices
                for ( int simplexind = 0; simplexind < Dim; ++simplexind)
                {
                    for ( int j = 0; j < Dim + 1; j++)
                    {
                        simplexes[simplexind*(Dim + 1) + j] =
                                elverts_prism[simplexes[simplexind*(Dim + 1) + j]];
                    }
                }

            } //end of if local_method = 0 or 2

            // printInt2D(pentatops, Dim, Dim + 1);


            // 3.11 adding the constructed space-time simplices to the output mesh
            for ( int simplex_ind = 0; simplex_ind < Dim; ++simplex_ind)
            {
                if (Dim == 3)
                    NewEl = new Tetrahedron(simplexes + simplex_ind*(Dim+1));
                if (Dim == 4)
                    NewEl = new Pentatope(simplexes + simplex_ind*(Dim+1));
                NewEl->SetAttribute(1);
                AddElement(NewEl);
                ++simplex_count;
            }

            //printArr2DInt (&vert_to_vert_prism);

        } // end of loop over time slabs
    } // end of loop over base elements

    if ( NumOfSTElements != GetNE() )
        std::cout << "Error: Wrong number of elements generated: " << GetNE() << " instead of " <<
                        NumOfSTElements << std::endl;
    if ( NumOfSTVertices != GetNV() )
        std::cout << "Error: Wrong number of vertices generated: " << GetNV() << " instead of " <<
                        NumOfSTVertices << std::endl;
    if ( NumOfSTBdrElements!= GetNBE() )
        std::cout << "Error: Wrong number of bdr elements generated: " << GetNBE() << " instead of " <<
                        NumOfSTBdrElements << std::endl;

    delete [] ordering;
    delete [] almostjogglers;
    delete [] temp;
    delete [] tempface;
    delete [] simplexes;
    delete [] elvert_coordprism;

    if (local_method == 1)
    {
        delete [] vert_latface;
        delete [] vert_3Dlatface;
        delete [] facesimplicesAll;
    }
    if (local_method == 0 || local_method == 1)
        delete [] qhull_flags;

    return;
}

#ifdef MFEM_USE_MPI
// parallel version 1 : creating serial space-time mesh from parallel base mesh in parallel
// from a given base mesh produces a space-time mesh for a cylinder
// with the given base and Nsteps * tau height in time
// enumeration of space-time vertices: time slab after time slab
// boundary attributes: 1 for t=0, 2 for lateral boundaries, 3 for t = tau*Nsteps
//void ParMesh3DtoMesh4D (MPI_Comm comm, ParMesh& mesh3d,
//                                       Mesh& mesh4d, double tau, int Nsteps, int bnd_method, int local_method)
Mesh::Mesh (MPI_Comm comm, ParMesh& mesh3d, double tau, int Nsteps,
            int bnd_method, int local_method)
{
    int num_procs, myid;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = 4;
    int dim3 = 3;
    int nvert_per_elem = dim + 1; // PENTATOPE or TETRAHEDRON cases only
    int nvert_per_bdrelem = dim; // PENTATOPE or TETRAHEDRON cases only

    // *************************************************************************
    // step 1 of 3: take the local base mesh for the proc and create a local space-time mesh
    // part as IntermediateMesh
    // *************************************************************************

    // 1.1: create gverts = array with global vertex numbers
    // can be avoided but then a lot of calls to pspace3d->GetGlobalTDofNumber will happen
    int * gvertinds = new int[mesh3d.GetNV()];

    FiniteElementCollection * h1_coll = new H1_FECollection(1, dim3);

    ParFiniteElementSpace * pspace3d = new ParFiniteElementSpace(&mesh3d, h1_coll);

    for ( int lvert = 0; lvert < mesh3d.GetNV(); ++lvert )
        gvertinds[lvert] = pspace3d->GetGlobalTDofNumber(lvert);

    // 1.2: creating local parts of space-time mesh as IntemediateMesh structure

    /*
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << ", myid = " << myid << endl << flush;
            cout << "Creating local parts of 4d mesh" << endl << flush;
            //cout << "Now it is in local indices" << endl;
        }
        MPI_Barrier(comm);
    }
    */

    IntermediateMesh * local_intermesh = mesh3d.MeshSpaceTimeCylinder_toInterMesh( tau, Nsteps, bnd_method, local_method);

    int nv3d_global = pspace3d->GlobalTrueVSize(); // global number of vertices in the 3d mesh

    // 1.3 writing the global vertex numbers inside the local IntermediateMesh(4d)
    int lvert4d;
    int tslab;
    int onslab_lindex; // = lvert3d for the projection of 4d on 3d base
    for ( int lvert4d = 0; lvert4d < local_intermesh->nv; ++lvert4d )
    {
        tslab = lvert4d / mesh3d.GetNV();
        onslab_lindex = lvert4d - tslab * mesh3d.GetNV();

        local_intermesh->vert_gindices[lvert4d] = tslab * nv3d_global + gvertinds[onslab_lindex];
        //local_intermesh->vert_gindices[lvert4d] = tslab * nv3d_global + pspace3d->GetGlobalTDofNumber(onslab_lindex);
    }

    InterMeshPrint (local_intermesh, myid, "local_intermesh");
    MPI_Barrier(comm);

    // 1.4 replacing local vertex indices by global indices from parFEspace
    // converting local to global vertex indices in elements
    for (int elind = 0; elind < local_intermesh->ne; ++elind)
    {
        //cout << "elind = " << elind << endl;
        for ( int j = 0; j < nvert_per_elem; ++j )
        {
            lvert4d = local_intermesh->elements[elind * nvert_per_elem + j];
            tslab = lvert4d / mesh3d.GetNV();
            onslab_lindex = lvert4d - tslab * mesh3d.GetNV();

            //local_intermesh->elements[elind * nvert_per_elem + j] =
                    //tslab * nv3d_global + pspace3d->GetGlobalTDofNumber(onslab_lindex);
            local_intermesh->elements[elind * nvert_per_elem + j] =
                    tslab * nv3d_global + gvertinds[onslab_lindex];
        }
    }

    // converting local to global vertex indices in boundary elements
    for (int bdrelind = 0; bdrelind < local_intermesh->nbe; ++bdrelind)
    {
        //cout << "bdrelind = " << bdrelind << endl;
        for ( int j = 0; j < nvert_per_bdrelem; ++j )
        {
            lvert4d = local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j];
            tslab = lvert4d / mesh3d.GetNV();
            onslab_lindex = lvert4d - tslab * mesh3d.GetNV();

            //local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j] =
                    //tslab * nv3d_global + pspace3d->GetGlobalTDofNumber(onslab_lindex);
            local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j] =
                    tslab * nv3d_global + gvertinds[onslab_lindex];

            //cout << "lindex3d converted to gindex3d = " << pspace3d->GetGlobalTDofNumber(onslab_lindex) << endl;
        }
    }

    delete h1_coll;
    delete pspace3d;

    //InterMeshPrint (local_intermesh, myid, "local_intermesh_newer");
    //MPI_Barrier(comm);

    // *************************************************************************
    // step 2 of 3: exchange the local mesh 4d parts and exchange them
    // *************************************************************************

    // 2.1: exchanging information about local sizes between processors
    // in order to set up mpi exchange parameters and allocate the future 4d mesh;

    // nvdg_global = sum of local number of vertices (without thinking that
    // some vertices are shared between processors)
    int nvdg_global, nv_global, ne_global, nbe_global;

    int *recvcounts_el = new int[num_procs];
    MPI_Allgather( &(local_intermesh->ne), 1, MPI_INT, recvcounts_el, 1, MPI_INT, comm);

    int *rdispls_el = new int[num_procs];
    rdispls_el[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_el[i + 1] = rdispls_el[i] + recvcounts_el[i];

    ne_global = rdispls_el[num_procs - 1] + recvcounts_el[num_procs - 1];

    int *recvcounts_be = new int[num_procs];

    MPI_Allgather( &(local_intermesh->nbe), 1, MPI_INT, recvcounts_be, 1, MPI_INT, comm);

    int *rdispls_be = new int[num_procs];

    rdispls_be[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_be[i + 1] = rdispls_be[i] + recvcounts_be[i];

    nbe_global = rdispls_be[num_procs - 1] + recvcounts_be[num_procs - 1];

    int *recvcounts_v = new int[num_procs];
    MPI_Allgather( &(local_intermesh->nv), 1, MPI_INT, recvcounts_v, 1, MPI_INT, comm);

    int *rdispls_v = new int[num_procs];
    rdispls_v[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_v[i + 1] = rdispls_v[i] + recvcounts_v[i];

    nvdg_global = rdispls_v[num_procs - 1] + recvcounts_v[num_procs - 1];
    nv_global = nv3d_global * (Nsteps + 1);

    MPI_Barrier(comm);

    IntermediateMesh * intermesh_4d = new IntermediateMesh;
    IntermeshInit( intermesh_4d, dim, nvdg_global, ne_global, nbe_global, 1);

    // 2.2: exchanging attributes, elements and vertices between processes using allgatherv

    // exchanging element attributes
    MPI_Allgatherv( local_intermesh->elattrs, local_intermesh->ne, MPI_INT,
                    intermesh_4d->elattrs, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdr element attributes
    MPI_Allgatherv( local_intermesh->bdrattrs, local_intermesh->nbe, MPI_INT,
                    intermesh_4d->bdrattrs, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging elements, changing recvcounts_el!!!
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_el[i] *= nvert_per_elem;
        rdispls_el[i] *= nvert_per_elem;
    }

    MPI_Allgatherv( local_intermesh->elements, (local_intermesh->ne)*nvert_per_elem, MPI_INT,
                    intermesh_4d->elements, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdrelements, changing recvcounts_be!!!

    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_be[i] *= nvert_per_bdrelem;
        rdispls_be[i] *= nvert_per_bdrelem;
    }

    MPI_Allgatherv( local_intermesh->bdrelements, (local_intermesh->nbe)*nvert_per_bdrelem,
              MPI_INT, intermesh_4d->bdrelements, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging global vertex indices
    MPI_Allgatherv( local_intermesh->vert_gindices, local_intermesh->nv, MPI_INT,
                    intermesh_4d->vert_gindices, recvcounts_v, rdispls_v, MPI_INT, comm);

    // exchanging vertices : At the moment dg-type of procedure = without considering
    // presence of shared vertices
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_v[i] *= dim;
        rdispls_v[i] *= dim;
    }

    MPI_Allgatherv( local_intermesh->vertices, (local_intermesh->nv)*dim, MPI_DOUBLE,
                    intermesh_4d->vertices, recvcounts_v, rdispls_v, MPI_DOUBLE, comm);

    IntermeshDelete(local_intermesh);

    // *************************************************************************
    // step 3 of 3: creating serial 4d mesh for each process
    // *************************************************************************

    InitMesh(dim,dim, nv_global, ne_global, nbe_global);

    //InterMeshPrint (intermesh_4d, myid, "intermesh4d");
    //MPI_Barrier(comm);

    // 3.1: creating the correct vertex array where each vertex is met only once
    // 3.1.1: cleaning up the vertices which are at the moment with multiple entries for
    // shared vertices

    int gindex;
    std::map<int, double*> vertices_unique; // map structure for storing only unique vertices

    // loop over all (with multiple entries) vertices, unique are added to the map object
    double * tempvert_map;
    for ( int i = 0; i < nvdg_global; ++i )
    {
        tempvert_map = new double[dim];
        for ( int j = 0; j < dim; j++ )
            tempvert_map[j] = intermesh_4d->vertices[i * dim + j];
        gindex = intermesh_4d->vert_gindices[i];
        vertices_unique[gindex] = tempvert_map;
    }

    // counting the final number of vertices. after that count_vert should be equal to nv_global
    int count_vert = 0;
    //for(auto const& ent : vertices_unique)
    //{
        //count_vert ++;
    //}
    // making compiler happy
    count_vert = vertices_unique.size();

    if ( count_vert != nv_global && myid == 0 )
    {
        cout << "Wrong number of vertices! Smth is probably wrong" << endl << flush;
    }

    // 3.1.2: creating the vertices array with taking care of shared vertices
    // using the map vertices_unique

    // now actual intermesh_4d->vertices is: right unique vertices + some vertices which
    // are still alive after mpi transfer.
    // so we reuse the memory already allocated for vertices array with multiple entries.

    //delete [] intermesh_4d->vertices;
    intermesh_4d->nv = count_vert;
    //intermesh_4d->vertices = new double[count_vert * dim];

    int tmp = 0;
    for(auto const& ent : vertices_unique)
    {
        for ( int j = 0; j < dim; j++)
            intermesh_4d->vertices[tmp*dim + j] = ent.second[j];

        if ( tmp != ent.first )
            cout << "ERROR" << endl;
        tmp++;
    }

    vertices_unique.clear();

    //InterMeshPrint (intermesh_4d, myid, "intermesh4d_reduced");
    //MPI_Barrier(comm);

    // 3.2: loading created intermesh_4d into a mfem mesh object (copying the memory: FIX IT may be)
    BaseGeom = Geometry::PENTATOPE;
    LoadMeshfromArrays( intermesh_4d->nv, intermesh_4d->vertices,
                  intermesh_4d->ne, intermesh_4d->elements, intermesh_4d->elattrs,
                  intermesh_4d->nbe, intermesh_4d->bdrelements, intermesh_4d->bdrattrs, dim );

    // 3.3 create the internal structure for mesh after el-s,, bdel-s and vertices have been loaded
    int refine = 1;
    CreateInternalMeshStructure(refine);

    IntermeshDelete(intermesh_4d);

    MPI_Barrier(comm);


    return;
}
#endif

// Does the same as MeshSpaceTimeCylinder_onlyArrays() but outputs InterMediateMesh structure
// works only in 4d case
Mesh::IntermediateMesh * Mesh::MeshSpaceTimeCylinder_toInterMesh (double tau, int Nsteps, int bnd_method, int local_method)
{
    int Dim3D = Dimension(), NumOf3DElements = GetNE(),
            NumOf3DBdrElements = GetNBE(),
            NumOf3DVertices = GetNV();
    int NumOf4DElements, NumOf4DBdrElements, NumOf4DVertices;

    if ( Dim3D != 3 )
    {
       cerr << "Wrong dimension in MeshSpaceTimeCylinder(): " << Dim3D << endl;
       return NULL;
    }

    const int Dim = Dim3D + 1;
    // for each 3D element and each time slab a 4D-prism with 3D element as a base
    // is decomposed into 4 pentatopes
    NumOf4DElements = NumOf3DElements * 4 * Nsteps;
    // no additional vertices so far
    NumOf4DVertices = NumOf3DVertices * (Nsteps + 1);
    // lateral 4d bdr faces (one for each 3d bdr face) + lower + upper bases
    // of the space-time cylinder
    NumOf4DBdrElements = NumOf3DBdrElements * 3 * Nsteps +
            NumOf3DElements + NumOf3DElements;

    // assuming that the 3D mesh contains elements of the same type
    int vert_per_base = GetElement(0)->GetNVertices();
    int vert_per_prism = 2 * vert_per_base;
    int vert_per_latface = Dim3D * 2;

    IntermediateMesh * intermesh = new IntermediateMesh;
    IntermeshInit( intermesh, Dim, NumOf4DVertices, NumOf4DElements, NumOf4DBdrElements, 1);

    Element * el;

    int * pentatops;
    if (local_method == 1 || local_method == 2)
    {
        pentatops = new int[Dim * (Dim + 1)]; // pentatop's vertices' indices
    }
    else // local_method = 0
    {
        int nsliver = 5; //why 5? how many slivers can b created by qhull? maybe 0 if we don't joggle inside qhull but perturb the coordinates before?
        pentatops = new int[(Dim + nsliver) * (Dim + 1)]; // pentatop's vertices' indices + probably sliver pentatopes
    }

    // stores indices of tetrahedron vertices produced by qhull for all lateral faces
    // (Dim) lateral faces, Dim3D tetrahedrons for each face (which is 3D prism)
    // (Dim3D + 1) vertex indices for each tetrahedron. Used in local_method = 1.
    int * tetrahedronsAll;
    if (local_method == 1 )
        tetrahedronsAll = new int[Dim3D * (Dim3D + 1) * Dim ];


    Array<int> elverts_base;
    Array<int> elverts_prism;

    int temptetra[4]; // temporary array for vertex indices of a pentatope face (used in local_method = 0 and 2)
    int temp[5]; //temp array for pentatops in local_method = 1;
    Array2D<int> vert_to_vert_prism; // for a 4D prism
    // row ~ lateral face of the 4d prism
    // first 6 columns - indices of vertices belonging to the lateral face,
    // last 2 columns - indices of the rest 2 vertices of the prism
    Array2D<int> latfacets_struct;
    // coordinates of vertices of a lateral face of 4D prism
    double * vert_latface;
    // coordinates of vertices of a 3D base (triangle) of a lateral face of 4D prism
    double * vert_3Dlatface;
    if (local_method == 1)
    {
        vert_latface =  new double[Dim * vert_per_latface];
        vert_3Dlatface = new double[Dim3D * vert_per_latface];
        latfacets_struct.SetSize(Dim, vert_per_prism);
        vert_to_vert_prism.SetSize(vert_per_prism, vert_per_prism);
    }


    // coordinates of vertices of 4D prism
    double * elvert_coordprism = new double[Dim * vert_per_prism];

    //char qhull_flags[250];
    char * qhull_flags;
    if (local_method == 0 || local_method == 1)
    {
        qhull_flags = new char[250];
        sprintf(qhull_flags, "qhull d Qbb");
    }


    if (local_method < 0 && local_method > 2)
    {
        cout << "Local method = " << local_method << " is not supported" << endl;
        return NULL;
    }
    //else
        //cout << "Using local_method = " << local_method << " for constructing pentatops" << endl;

    if ( bnd_method != 0 && bnd_method != 1)
    {
        cout << "Illegal value of bnd_method = " << bnd_method << " (must be 0 or 1)" << endl;
        return NULL;
    }
    //else
        //cout << "Using bnd_method = " << bnd_method << " for creating boundary elements" << endl;

    //cout << "Using local_method = 0 in MeshSpaceTimeCylinder_toInterMesh()" << endl;

    int almostjogglers[4];
    //int permutation[4];
    vector<double*> lcoords(Dim);
    vector<vector<double> > lcoordsNew(Dim);

    Vector vert_coord3d(Dim3D * NumOf3DVertices);
    GetVertices(vert_coord3d);
    //printDouble2D(vert_coord3d, 10, Dim3D);


    // adding all the 4d vertices to the mesh
    int vcount = 0;
    for ( int tslab = 0; tslab <= Nsteps; ++tslab)
    {
        // adding the vertices from the slab to the mesh4d
        for ( int vert = 0; vert < NumOf3DVertices; ++vert)
        {
            //tempvert[0] = vert_coord3d[vert + 0 * NumOf3DVertices];
            //tempvert[1] = vert_coord3d[vert + 1 * NumOf3DVertices];
            //tempvert[2] = vert_coord3d[vert + 2 * NumOf3DVertices];
            //tempvert[3] = tau * tslab;
            //mesh4d->AddVertex(tempvert);
            intermesh->vertices[vcount*Dim + 0] = vert_coord3d[vert + 0 * NumOf3DVertices];
            intermesh->vertices[vcount*Dim + 1] = vert_coord3d[vert + 1 * NumOf3DVertices];
            intermesh->vertices[vcount*Dim + 2] = vert_coord3d[vert + 2 * NumOf3DVertices];
            intermesh->vertices[vcount*Dim + 3] = tau * tslab;
            vcount++;
        }
    }

    //delete(tempvert);

    // for each (of Dim) 3d element faces stores 1 if it is at the boundary and 0 else
    int facebdrmarker[Dim];
    // std::set of the 3D boundary elements
    // using set allows to perform a search with O(log N_elem) operations
    std::set< std::vector<int> > BdrTriSet;
    Element * bdrel;

    Array<int> face_bndflags(GetNFaces());

    // if = 0, a search algorithm is used for defining whether faces of a given 3d element
    // are at the boundary.
    // if = 1, instead an array face_bndflags is used, which stores 0 and 1 depending on
    // whether the face is at the boundary, + el_to_face table which is usually already
    // generated for the 3d mesh
    //int bnd_method = 1;

    if (bnd_method == 0)
    {
        // putting 3d boundary elements from mesh3d to the set BdrTriSet
        for ( int boundelem = 0; boundelem < NumOf3DBdrElements; ++boundelem)
        {
            //cout << "boundelem No. " << boundelem << endl;
            bdrel = GetBdrElement(boundelem);
            int * bdrverts = bdrel->GetVertices();

            std::vector<int> buff (bdrverts, bdrverts+3);
            std::sort (buff.begin(), buff.begin()+3);

            BdrTriSet.insert(buff);
        }
        /*
        for (vector<int> temp : BdrTriSet)
        {
            cout << temp[0] << " " <<  temp[1] << " " << temp[2] << endl;
        }
        cout<<endl;
        */
    }
    else // bnd_method = 1
    {
        if (el_to_face == NULL)
        {
            cout << "Have to built el_to_face" << endl;
            GetElementToFaceTable(0);
        }

        //cout << "Special print" << endl;
        //cout << mesh3d.el_to_face(elind, facelind);
        //cout << "be_to_face" << endl;
        //mesh3d.be_to_face.Print();

        //cout << "nfaces = " << mesh3d.GetNFaces();
        //cout << "nbe = " << mesh3d.GetNBE() << endl;
        //cout << "boundary.size = " << mesh3d.boundary.Size() << endl;

        face_bndflags = -1;
        for ( int i = 0; i < NumOf3DBdrElements; ++i )
            face_bndflags[be_to_face[i]] = 1;

        //cout << "face_bndflags" << endl;
        //face_bndflags.Print();
    }

    int * ordering = new int[vert_per_base];
    //int antireordering[vert_per_base]; // used if bnd_method = 0 and local_method = 2
    Array<int> tempelverts(vert_per_base);

    int bdrelcount = 0;
    int elcount = 0;

    // main loop creates 4d elements for all time slabs for all 3d elements
    // loop over 3d elements
    for ( int elind = 0; elind < NumOf3DElements; elind++ )
    //for ( int elind = 0; elind < 1; ++elind )
    {
        //cout << "element " << elind << endl;

        el = GetElement(elind);

        // 1. getting indices of 3d element vertices and their coordinates in the prism
        el->GetVertices(elverts_base);

        // for local_method 2 we need to reorder the local vertices of the prism to preserve the
        // the order in some global sense  = lexicographical order of the vertex coordinates
        if (local_method == 2)
        {
            // setting vertex coordinates for 4d prism, lower base
            for ( int i = 0; i < vert_per_base; ++i)
            {
                //double * temp = vert_coord3d + Dim3D * elverts_base[i];
                //elvert_coordprism[Dim * i + 0] = temp[0];
                //elvert_coordprism[Dim * i + 1] = temp[1];
                //elvert_coordprism[Dim * i + 2] = temp[2];
                elvert_coordprism[Dim * i + 0] = vert_coord3d[elverts_base[i] + 0 * NumOf3DVertices];
                elvert_coordprism[Dim * i + 1] = vert_coord3d[elverts_base[i] + 1 * NumOf3DVertices];
                elvert_coordprism[Dim * i + 2] = vert_coord3d[elverts_base[i] + 2 * NumOf3DVertices];
            }


            /*
            // * old
            for (int vert = 0; vert < Dim; ++vert)
                lcoords[vert] = elvert_coordprism + Dim * vert;

            sortingPermutation(Dim3D, lcoords, ordering);
            */




            for (int vert = 0; vert < Dim; ++vert)
                lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                        elvert_coordprism + Dim * vert + Dim3D);

            sortingPermutationNew(lcoordsNew, ordering);




            // UGLY: Fix it
            for ( int i = 0; i < vert_per_base; ++i)
                tempelverts[i] = elverts_base[ordering[i]];

            for ( int i = 0; i < vert_per_base; ++i)
                elverts_base[i] = tempelverts[i];
        }

        //for ( int k = 0; k < elverts_base.Size(); ++k )
            //cout << "elverts[" << k << "] = " << elverts_base[k] << endl;

        // 2. understanding which of the 3d element faces (triangles) are at the boundary
        int local_nbdrfaces = 0;
        set<set<int> > LocalBdrs;
        if (bnd_method == 0)
        {
            vector<int> face(Dim3D);
            for (int i = 0; i < Dim; ++i )
            {
                // should be consistent with lateral faces ordering in latfacet structure
                // if used with local_method = 1
                if ( i == 0)
                {
                    face[0] = elverts_base[0];
                    face[1] = elverts_base[1];
                    face[2] = elverts_base[2];
                }
                if ( i == 1)
                {
                    face[0] = elverts_base[1];
                    face[1] = elverts_base[2];
                    face[2] = elverts_base[3];
                }
                if ( i == 2)
                {
                    face[0] = elverts_base[2];
                    face[1] = elverts_base[3];
                    face[2] = elverts_base[0];
                }
                if ( i == 3)
                {
                    face[0] = elverts_base[3];
                    face[1] = elverts_base[0];
                    face[2] = elverts_base[1];
                }

                /*
                int cnt = 0;
                for ( int j = 0; j < Dim; ++j)
                    if ( j != i )
                        face[cnt++] = elverts_base[j];
                */
                sort(face.begin(), face.begin()+Dim3D);
                //cout << face[0] << " " <<  face[1] << " " << face[2] << endl;

                if (BdrTriSet.find(face) != BdrTriSet.end() )
                {
                    //cout << "is at the boundary" << endl;
                    local_nbdrfaces++;
                    facebdrmarker[i] = 1;
                    set<int> face_as_set;
                    if ( i == 0)
                    {
                        face_as_set.insert(0);
                        face_as_set.insert(1);
                        face_as_set.insert(2);
                    }
                    if ( i == 1)
                    {
                        face_as_set.insert(1);
                        face_as_set.insert(2);
                        face_as_set.insert(3);
                    }
                    if ( i == 2)
                    {
                        face_as_set.insert(2);
                        face_as_set.insert(3);
                        face_as_set.insert(0);
                    }
                    if ( i == 3)
                    {
                        face_as_set.insert(3);
                        face_as_set.insert(0);
                        face_as_set.insert(1);
                    }
                    LocalBdrs.insert(face_as_set);
                }
                else
                    facebdrmarker[i] = 0;
            }

        } //end of if bnd_method == 0
        else
        //set<set<int>> LocalBdrs2;
        {
            int * faceinds = el_to_face->GetRow(elind);
            for ( int facelind = 0; facelind < Dim; ++facelind)
            {
                int faceind = faceinds[facelind];
                if (face_bndflags[faceind] == 1)
                {
                    Array<int> temp(3);
                    GetFaceVertices(faceind, temp);
                    //set<int> face_as_set(temp, temp+3);
                    set<int> face_as_set;
                    for ( int vert = 0; vert < Dim3D; ++vert )
                        face_as_set.insert(temp[vert]);
                    LocalBdrs.insert(face_as_set);

                    local_nbdrfaces++;
                }

            } // end of loop over element faces

        }

        //cout << "Welcome the facebdrmarker" << endl;
        //printInt2D(facebdrmarker, 1, Dim);

        /*
        cout << "Welcome the LocalBdrs" << endl;
        for ( set<int> tempset: LocalBdrs )
        {
            cout << "element of LocalBdrs for el = " << elind << endl;
            for (int ind: tempset)
                cout << ind << " ";
            cout << endl;
        }
        */


        // 3. loop over all 4D time slabs above a given 3d element
        for ( int tslab = 0; tslab < Nsteps; ++tslab)
        {
            //cout << "tslab " << tslab << endl;

            //3.1 getting vertex indices for the 4d prism
            elverts_prism.SetSize(vert_per_prism);
            for ( int i = 0; i < vert_per_base; ++i)
            {
                elverts_prism[i] = elverts_base[i] + tslab * NumOf3DVertices;
                elverts_prism[i + vert_per_base] = elverts_base[i] + (tslab + 1) * NumOf3DVertices;
            }

            // 3.2 for the first time slab we add the tetrahedrons in the lower base
            // to the bdr elements
            if ( tslab == 0 )
            {
                //cout << "zero slab: adding boundary element:" << endl;
                //NewBdrTri = new Tetrahedron(elverts_prism);
                //NewBdrTri->SetAttribute(1);
                //mesh4d->AddBdrElement(NewBdrTri);

                intermesh->bdrelements[bdrelcount*Dim + 0] = elverts_prism[0];
                intermesh->bdrelements[bdrelcount*Dim + 1] = elverts_prism[1];
                intermesh->bdrelements[bdrelcount*Dim + 2] = elverts_prism[2];
                intermesh->bdrelements[bdrelcount*Dim + 3] = elverts_prism[3];
                intermesh->bdrattrs[bdrelcount] = 1;
                bdrelcount++;

                /*
                const int nv = NewBdrTri->GetNVertices();
                const int *v = NewBdrTri->GetVertices();
                for (int j = 0; j < nv; j++)
                {
                   cout << ' ' << v[j];
                }
                cout << endl;
                */
            }
            // 3.3 for the last time slab we add the tetrahedrons in the upper base
            // to the bdr elements
            if ( tslab == Nsteps - 1 )
            {
                //cout << "last slab: adding boundary element:" << endl;
                //NewBdrTri = new Tetrahedron(elverts_prism + vert_per_base);
                //NewBdrTri->SetAttribute(3);
                //mesh4d->AddBdrElement(NewBdrTri);

                intermesh->bdrelements[bdrelcount*Dim + 0] = elverts_prism[0 + vert_per_base];
                intermesh->bdrelements[bdrelcount*Dim + 1] = elverts_prism[1 + vert_per_base];
                intermesh->bdrelements[bdrelcount*Dim + 2] = elverts_prism[2 + vert_per_base];
                intermesh->bdrelements[bdrelcount*Dim + 3] = elverts_prism[3 + vert_per_base];
                intermesh->bdrattrs[bdrelcount] = 3;
                bdrelcount++;

                /*
                const int nv = NewBdrTri->GetNVertices();
                const int *v = NewBdrTri->GetVertices();
                for (int j = 0; j < nv; j++)
                {
                   cout << ' ' << v[j];
                }
                cout << endl;
                */
            }

            //elverts_prism.Print();
            //return;

            // printInt2D(pentatops, Dim, Dim + 1);

            if (local_method == 0 || local_method == 1)
            {
                // 3.4 setting vertex coordinates for 4d prism, lower base
                for ( int i = 0; i < vert_per_base; ++i)
                {
                    //double * temp = vert_coord3d + Dim3D * elverts_base[i];
                    //elvert_coordprism[Dim * i + 0] = temp[0];
                    //elvert_coordprism[Dim * i + 1] = temp[1];
                    //elvert_coordprism[Dim * i + 2] = temp[2];
                    elvert_coordprism[Dim * i + 0] = vert_coord3d[elverts_base[i] + 0 * NumOf3DVertices];
                    elvert_coordprism[Dim * i + 1] = vert_coord3d[elverts_base[i] + 1 * NumOf3DVertices];
                    elvert_coordprism[Dim * i + 2] = vert_coord3d[elverts_base[i] + 2 * NumOf3DVertices];
                    elvert_coordprism[Dim * i + 3] = tslab * tau;


                    /*
                    std::cout << \
                                 "vert_coord3d[" << elverts_prism[i] + 0 * NumOf3DVertices << "] = " << \
                                 vert_coord3d(elverts_prism[i] + 0 * NumOf3DVertices) << " " << \
                                 "vert_coord3d[" << elverts_prism[i] + 1 * NumOf3DVertices << "] = " << \
                                 vert_coord3d[elverts_prism[i] + 1 * NumOf3DVertices] << " " << \
                                 "vert_coord3d[" << elverts_prism[i] + 2 * NumOf3DVertices << "] = "  << \
                                 vert_coord3d[elverts_prism[i] + 0 * NumOf3DVertices] << std::endl;

                    //std::cout << "indices in coordprism which were set:" << endl;
                    //std::cout << Dim * i + 0 << " " << Dim * i + 1 << " " << Dim * i + 2 << endl;
                    std::cout << "we got:" << endl;
                    std::cout << "elvert_coordprism for vertex " <<  i << ": " << \
                                 elvert_coordprism[Dim * i + 0] << " " << elvert_coordprism[Dim * i + 1] << \
                                 " " << elvert_coordprism[Dim * i + 2] << " " << \
                                 elvert_coordprism[Dim * i + 3] << endl;

                    double temp = vert_coord3d[elverts_prism[i] + 2 * NumOf3DVertices];
                    std::cout << "temp = " << temp << endl;
                    */

                }

                //cout << "Welcome the vertex coordinates for the 4d prism base " << endl;
                //printDouble2D(elvert_coordprism, vert_per_base, Dim);


                /*
                // * old
                for (int vert = 0; vert < Dim; ++vert)
                    lcoords[vert] = elvert_coordprism + Dim * vert;


                //cout << "vector double * lcoords:" << endl;
                //for ( int i = 0; i < Dim; ++i)
                    //cout << "lcoords[" << i << "]: " << lcoords[i][0] << " " << lcoords[i][1] << " " << lcoords[i][2] << endl;

                sortingPermutation(Dim3D, lcoords, permutation);
                */



                for (int vert = 0; vert < Dim; ++vert)
                    lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                            elvert_coordprism + Dim * vert + Dim3D);

                sortingPermutationNew(lcoordsNew, ordering);



                //cout << "Welcome the permutation:" << endl;
                //cout << permutation[0] << " " << permutation[1] << " " << permutation[2] << " " << permutation[3] << endl;

                int joggle_coeff = 0;
                for ( int i = 0; i < Dim; ++i)
                    almostjogglers[ordering[i]] = joggle_coeff++;

                //cout << "Welcome the joggle coeffs:" << endl;
                //cout << almostjogglers[0] << " " << almostjogglers[1] << " " << almostjogglers[2] << " " << almostjogglers[3] << endl;



                // 3.5 setting vertex coordinates for 4d prism, upper layer
                // with joggling of the time coordinate depending on the global vertex indices
                // Joggling is required for getting unique Delaunay tesselation and should be
                // the same for vertices shared between different elements or at least produce
                // the same Delaunay triangulation in the shared faces.
                double joggle;
                for ( int i = 0; i < vert_per_base; ++i)
                {
                    //double * temp = vert_coord3d + Dim3D * elverts_base[i];
                    //elvert_coordprism[Dim * (vert_per_base + i) + 0] = temp[0];
                    //elvert_coordprism[Dim * (vert_per_base + i) + 1] = temp[1];
                    //elvert_coordprism[Dim * (vert_per_base + i) + 2] = temp[2];
                    elvert_coordprism[Dim * (vert_per_base + i) + 0] = elvert_coordprism[Dim * i + 0];
                    elvert_coordprism[Dim * (vert_per_base + i) + 1] = elvert_coordprism[Dim * i + 1];
                    elvert_coordprism[Dim * (vert_per_base + i) + 2] = elvert_coordprism[Dim * i + 2];

                    joggle = 1.0e-2 * (almostjogglers[i]);
                    //joggle = 1.0e-2 * elverts_prism[i + vert_per_base] * 1.0 / NumOf4DVertices;
                    //double joggle = 1.0e-2 * i;
                    elvert_coordprism[Dim * (vert_per_base + i) + 3] = (tslab + 1) * tau * ( 1.0 + joggle );

                }

                //cout << "Welcome the vertex coordinates for the 4d prism" << endl;
                //printDouble2D(elvert_coordprism, 2 * vert_per_base, Dim);

                if (local_method == 0)
                {
                    // ~ 3.6 - 3.10 (in LONGWAY): constructing pentatopes and boundary elements
#ifdef WITH_QHULL
                    qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                    qhT *qh= &qh_qh;
                    int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                    double volumetol = 1.0e-8;
                    qhull_wrapper(pentatops, qh, elvert_coordprism, Dim, volumetol, qhull_flags);
                    qh_freeqhull(qh, !qh_ALL);
                    qh_memfreeshort(qh, &curlong, &totlong);
                    if (curlong || totlong)  /* could also check previous runs */
                    {
                      fprintf(stderr, "qhull internal warning (user_eg, #3): did not free %d bytes \
                            of long memory (%d pieces)\n", totlong, curlong);
                    }
#else
                    cout << "Wrong local method, WITH_QHULL flag was not set" << endl;
#endif

                } // end of if local_method = 0

                if (local_method == 1)
                {
                    // 3.6 - 3.10: constructing pentatopes

                    setzero(&vert_to_vert_prism);

                    // 3.6 creating vert_to_vert for the prism before Delaunay (adding 4d prism edges)
                    for ( int i = 0; i < el->GetNEdges(); i++)
                    {
                        const int * edge = el->GetEdgeVertices(i);
                        //cout << "edge: " << edge[0] << " " << edge[1] << std::endl;
                        vert_to_vert_prism(edge[0], edge[1]) = 1;
                        vert_to_vert_prism(edge[1], edge[0]) = 1;
                        vert_to_vert_prism(edge[0] + vert_per_base, edge[1] + vert_per_base) = 1;
                        vert_to_vert_prism(edge[1] + vert_per_base, edge[0] + vert_per_base) = 1;
                    }

                    for ( int i = 0; i < vert_per_base; i++)
                    {
                        vert_to_vert_prism(i, i) = 1;
                        vert_to_vert_prism(i + vert_per_base, i + vert_per_base) = 1;
                        vert_to_vert_prism(i, i + vert_per_base) = 1;
                        vert_to_vert_prism(i + vert_per_base, i) = 1;
                    }

                    //cout << "vert_to_vert before delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);
                    //cout << endl;

                    // 3.7 creating latfacet structure (brute force), for 4D tetrahedron case
                    // indices are local w.r.t to the 4d prism!!!
                    latfacets_struct(0,0) = 0;
                    latfacets_struct(0,1) = 1;
                    latfacets_struct(0,2) = 2;
                    latfacets_struct(0,6) = 3;

                    latfacets_struct(1,0) = 1;
                    latfacets_struct(1,1) = 2;
                    latfacets_struct(1,2) = 3;
                    latfacets_struct(1,6) = 0;

                    latfacets_struct(2,0) = 2;
                    latfacets_struct(2,1) = 3;
                    latfacets_struct(2,2) = 0;
                    latfacets_struct(2,6) = 1;

                    latfacets_struct(3,0) = 3;
                    latfacets_struct(3,1) = 0;
                    latfacets_struct(3,2) = 1;
                    latfacets_struct(3,6) = 2;

                    for ( int i = 0; i < Dim; ++i)
                    {
                        latfacets_struct(i,3) = latfacets_struct(i,0) + vert_per_base;
                        latfacets_struct(i,4) = latfacets_struct(i,1) + vert_per_base;
                        latfacets_struct(i,5) = latfacets_struct(i,2) + vert_per_base;
                        latfacets_struct(i,7) = latfacets_struct(i,6) + vert_per_base;
                    }

                    //cout << "latfacets_struct (vertex indices)" << endl;
                    //printArr2DInt (&latfacets_struct);

                    //(*)const int * base_face = el->GetFaceVertices(i); // not implemented in MFEM for Tetrahedron ?!

                    int * tetrahedrons;
                    int shift = 0;

                    // 3.8 loop over lateral facets, creating Delaunay triangulations
                    for ( int latfacind = 0; latfacind < Dim; ++latfacind)
                    //for ( int latfacind = 0; latfacind < 1; ++latfacind)
                    {
                        //cout << "latface = " << latfacind << endl;
                        for ( int vert = 0; vert < vert_per_latface ; ++vert )
                        {
                            //cout << "vert index = " << latfacets_struct(latfacind,vert) << endl;
                            for ( int coord = 0; coord < Dim; ++coord)
                            {
                                //cout << "index righthandside " << latfacets_struct(latfacind,vert)* Dim + coord << endl;
                                vert_latface[vert*Dim + coord] =  \
                                        elvert_coordprism[latfacets_struct(latfacind,vert)
                                        * Dim + coord];
                            }

                        }

                        //cout << "Welcome the vertices of a lateral face" << endl;
                        //printDouble2D(vert_latface, vert_per_latface, Dim);

                        // creating from 3Dprism in 4D a true 3D prism in 3D by change of coordinates
                        // = computing input argument vert_3Dlatface for qhull wrapper
                        // we know that the first three coordinated of a lateral face is actually
                        // a triangle, so we set the first vertex to be the origin,
                        // the first-to-second edge to be one of the axis
                        if ( Dim == 4 )
                        {
                            double x1, x2, x3, y1, y2, y3;
                            double dist12, dist13, dist23;
                            double area, h, p;

                            dist12 = dist(vert_latface, vert_latface+Dim , Dim);
                            dist13 = dist(vert_latface, vert_latface+2*Dim , Dim);
                            dist23 = dist(vert_latface+Dim, vert_latface+2*Dim , Dim);

                            p = 0.5 * (dist12 + dist13 + dist23);
                            area = sqrt (p * (p - dist12) * (p - dist13) * (p - dist23));
                            h = 2.0 * area / dist12;

                            x1 = 0.0;
                            y1 = 0.0;
                            x2 = dist12;
                            y2 = 0.0;
                            if ( dist13 - h < 0.0 )
                                if ( fabs(dist13 - h) > 1.0e-10)
                                {
                                    std::cout << "Error: strange: dist13 = " << dist13 << " h = "
                                              << h << std::endl;
                                    return NULL;
                                }
                                else
                                    x3 = 0.0;
                            else
                                x3 = sqrt(dist13*dist13 - h*h);
                            y3 = h;


                            // the time coordinate remains the same
                            for ( int vert = 0; vert < vert_per_latface ; ++vert )
                                vert_3Dlatface[vert*Dim3D + 2] = vert_latface[vert*Dim + 3];


                            // first & fourth vertex
                            vert_3Dlatface[0*Dim3D + 0] = x1;
                            vert_3Dlatface[0*Dim3D + 1] = y1;
                            vert_3Dlatface[3*Dim3D + 0] = x1;
                            vert_3Dlatface[3*Dim3D + 1] = y1;

                            // second & fifth vertex
                            vert_3Dlatface[1*Dim3D + 0] = x2;
                            vert_3Dlatface[1*Dim3D + 1] = y2;
                            vert_3Dlatface[4*Dim3D + 0] = x2;
                            vert_3Dlatface[4*Dim3D + 1] = y2;

                            // third & sixth vertex
                            vert_3Dlatface[2*Dim3D + 0] = x3;
                            vert_3Dlatface[2*Dim3D + 1] = y3;
                            vert_3Dlatface[5*Dim3D + 0] = x3;
                            vert_3Dlatface[5*Dim3D + 1] = y3;
                        } //end of creating a true 3d prism

                        //cout << "Welcome the vertices of a lateral face in 3D" << endl;
                        //printDouble2D(vert_3Dlatface, vert_per_latface, Dim3D);

                        tetrahedrons = tetrahedronsAll + shift;

#ifdef WITH_QHULL
                        qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                        qhT *qh= &qh_qh;
                        int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                        double volumetol = 1.0e-8;
                        qhull_wrapper(tetrahedrons, qh, vert_3Dlatface, Dim3D,  \
                                        volumetol, qhull_flags);

                        qh_freeqhull(qh, !qh_ALL);
                        qh_memfreeshort(qh, &curlong, &totlong);
                        if (curlong || totlong)  /* could also check previous runs */
                          cerr<< "qhull internal warning (user_eg, #3): did not free " << totlong \
                          << "bytes of long memory (" << curlong << " pieces)" << endl;
#else
                        cout << "Wrong local method, WITH_QHULL flag was not set" << endl;
#endif
                        // convert local 3D prism (lateral face) vertex indices back to the 4D prism
                        // indices and adding boundary elements from tetrahedrins for lateral faces
                        // of the 4d prism ...
                        for ( int tetraind = 0; tetraind < Dim3D; ++tetraind)
                        {
                            //cout << "tetraind = " << tetraind << endl;

                            for ( int vert = 0; vert < Dim; ++vert)
                            {
                                int temp = tetrahedrons[tetraind*Dim + vert];
                                tetrahedrons[tetraind*Dim + vert] = latfacets_struct(latfacind, temp);
                            }

                            /*
                            cout << "tetrahedron: " << tetrahedrons[tetraind*Dim + 0] << " " << \
                                    tetrahedrons[tetraind*Dim + 1] << " " << \
                                    tetrahedrons[tetraind*Dim + 2] << " " << \
                                    tetrahedrons[tetraind*Dim + 3] << "\n";

                            cout << "elverts prism " << endl;
                            elverts_prism.Print();
                            */


                            int temptetra[4];
                            if ( bnd_method == 0 )
                            {
                                if ( facebdrmarker[latfacind] == 1 )
                                {
                                    //cout << "lateral facet " << latfacind << " is at the boundary: adding bnd element" << endl;

                                    temptetra[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    temptetra[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    temptetra[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    temptetra[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];
                                    //elverts_prism[i]

                                    // wrong because indices in tetrahedrons are local to 4d prism
                                    //NewBdrTri = new Tetrahedron(tetrahedrons + tetraind*Dim);

                                    intermesh->bdrelements[bdrelcount*Dim + 0] = temptetra[0];
                                    intermesh->bdrelements[bdrelcount*Dim + 1] = temptetra[1];
                                    intermesh->bdrelements[bdrelcount*Dim + 2] = temptetra[2];
                                    intermesh->bdrelements[bdrelcount*Dim + 3] = temptetra[3];
                                    intermesh->bdrattrs[bdrelcount] = 2;
                                    bdrelcount++;
                                }
                            }
                            else // bnd_method = 1
                            {
                                set<int> latface3d_set;
                                for ( int i = 0; i < Dim3D; ++i)
                                    latface3d_set.insert(elverts_prism[latfacets_struct(latfacind,i)] % NumOf3DVertices);

                                // checking whether the face is at the boundary of 3d mesh
                                if ( LocalBdrs.find(latface3d_set) != LocalBdrs.end())
                                {
                                    // converting local indices to global indices and adding the new boundary element
                                    temptetra[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    temptetra[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    temptetra[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    temptetra[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    intermesh->bdrelements[bdrelcount*Dim + 0] = temptetra[0];
                                    intermesh->bdrelements[bdrelcount*Dim + 1] = temptetra[1];
                                    intermesh->bdrelements[bdrelcount*Dim + 2] = temptetra[2];
                                    intermesh->bdrelements[bdrelcount*Dim + 3] = temptetra[3];
                                    intermesh->bdrattrs[bdrelcount] = 2;
                                    bdrelcount++;
                                }
                            }



                         } //end of loop over tetrahedrons for a given lateral face

                        shift += Dim3D * (Dim3D + 1);

                        //return;
                    } // end of loop over lateral faces

                    /*
                    std::cout << "Now final tetrahedrons are:" << endl;
                    for ( int k = 0; k < Dim; ++k )
                        for (int i = 0; i < Dim3D; ++i )
                        {
                            //std::cout << "Tetrahedron " << i << ": ";
                            std::cout << "vert indices: " << endl;
                            for ( int j = 0; j < Dim3D  +1; ++j )
                            {
                                std::cout << tetrahedronsAll[k*Dim3D*(Dim3D+1) +
                                        i*(Dim3D + 1) + j] << " ";
                            }
                            std::cout << endl;
                        }
                    */

                    // 3.9 adding the new edges from created tetrahedrons into the vert_to_vert
                    for ( int k = 0; k < Dim; ++k )
                        for (int i = 0; i < Dim3D; ++i )
                        {
                            int vert0 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 0];
                            int vert1 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 1];
                            int vert2 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 2];
                            int vert3 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 3];

                            vert_to_vert_prism(vert0, vert1) = 1;
                            vert_to_vert_prism(vert1, vert0) = 1;

                            vert_to_vert_prism(vert0, vert2) = 1;
                            vert_to_vert_prism(vert2, vert0) = 1;

                            vert_to_vert_prism(vert0, vert3) = 1;
                            vert_to_vert_prism(vert3, vert0) = 1;

                            vert_to_vert_prism(vert1, vert2) = 1;
                            vert_to_vert_prism(vert2, vert1) = 1;

                            vert_to_vert_prism(vert1, vert3) = 1;
                            vert_to_vert_prism(vert3, vert1) = 1;

                            vert_to_vert_prism(vert2, vert3) = 1;
                            vert_to_vert_prism(vert3, vert2) = 1;
                        }

                    //cout << "vert_to_vert after delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);

                    int count_penta = 0;

                    // 3.10 creating finally 4d pentatopes:
                    // take a tetrahedron related to a lateral face, find out which of the rest
                    // 2 vertices of the 4d prism (one is not) is connected to all vertices of
                    // tetrahedron, and get a pentatope from tetrahedron + this vertex
                    // If pentatope is new, add it to the final structure
                    // To make checking for new pentatopes easy, reoder the pentatope indices
                    // in the default std order

                    for ( int tetraind = 0; tetraind < Dim3D * Dim; ++tetraind)
                    {
                        // creating a pentatop temp
                        int latface_ind = tetraind / Dim3D;
                        for ( int vert = 0; vert < Dim; vert++ )
                            temp[vert] = tetrahedronsAll[tetraind * Dim + vert];

                        //cout << "tetrahedron" << endl;
                        //printInt2D(temp,1,4); // tetrahedron

                        bool isconnected = true;
                        for ( int vert = 0; vert < 4; ++vert)
                            if (vert_to_vert_prism(temp[vert], latfacets_struct(latface_ind,6)) == 0)
                                isconnected = false;

                        if ( isconnected == true)
                            temp[4] = latfacets_struct(latface_ind,6);
                        else
                        {
                            bool isconnectedCheck = true;
                            for ( int vert = 0; vert < 4; ++vert)
                                if (vert_to_vert_prism(temp[vert], latfacets_struct(latface_ind,7)) == 0)
                                    isconnectedCheck = false;
                            if (isconnectedCheck == 0)
                            {
                                cout << "Error: Both vertices are disconnected" << endl;
                                cout << "tetraind = " << tetraind << ", checking for " <<
                                             latfacets_struct(latface_ind,6) << " and " <<
                                             latfacets_struct(latface_ind,7) << endl;
                                return NULL;
                            }
                            else
                                temp[4] = latfacets_struct(latface_ind,7);
                        }

                        //printInt2D(temp,1,5);

                        // replacing local vertex indices w.r.t to 4d prism to global!
                        temp[0] = elverts_prism[temp[0]];
                        temp[1] = elverts_prism[temp[1]];
                        temp[2] = elverts_prism[temp[2]];
                        temp[3] = elverts_prism[temp[3]];
                        temp[4] = elverts_prism[temp[4]];

                        // sorting the vertex indices
                        std::vector<int> buff (temp, temp+5);
                        std::sort (buff.begin(), buff.begin()+5);

                        // looking whether the current pentatop is new
                        bool isnew = true;
                        for ( int i = 0; i < count_penta; ++i )
                        {
                            std::vector<int> pentatop (pentatops+i*(Dim+1), pentatops+(i+1)*(Dim+1));

                            if ( pentatop == buff )
                                isnew = false;
                        }

                        if ( isnew == true )
                        {
                            for ( int i = 0; i < Dim + 1; ++i )
                                pentatops[count_penta*(Dim+1) + i] = buff[i];
                            //cout << "found a new pentatop from tetraind = " << tetraind << endl;
                            //cout << "now we have " << count_penta << " pentatops" << endl;
                            //printInt2D(pentatops + count_penta*(Dim+1), 1, Dim + 1);

                            ++count_penta;
                        }
                        //cout << "element " << elind << endl;
                        //printInt2D(pentatops, count_penta, Dim + 1);
                    }

                    //cout<< count_penta << " pentatops created" << endl;
                    if ( count_penta != Dim )
                        cout << "Error: Wrong number of pentatops constructed: got " << count_penta \
                             << ", needed " << Dim << endl;
                    //printInt2D(pentatops, count_penta, Dim + 1);

                }


            } //end of if local_method = 0 or 1
            else // local_method == 2
            {
                for ( int count_penta = 0; count_penta < Dim; ++count_penta)
                {
                    for ( int i = 0; i < Dim + 1; ++i )
                    {
                        pentatops[count_penta*(Dim+1) + i] = count_penta + i;
                    }

                }
                //cout << "Welcome created pentatops" << endl;
                //printInt2D(pentatops, Dim, Dim + 1);
            }


            // adding boundary elements
            // careful, for now pentatopes give the vertex indices local to the 4D prism above a 3d element!
            if (local_method == 0 || local_method == 2)
            {
                //if (local_method == 2)
                    //for ( int i = 0; i < vert_per_base; ++i)
                        //antireordering[ordering[i]] = i;

                if (local_nbdrfaces > 0) //if there is at least one 3d element face at the boundary for a given base element
                {
                    for ( int pentaind = 0; pentaind < Dim; ++pentaind)
                    {
                        //cout << "pentaind = " << pentaind << endl;
                        //printInt2D(pentatops + pentaind*(Dim+1), 1, 5);

                        for ( int faceind = 0; faceind < Dim + 1; ++faceind)
                        {
                            //cout << "faceind = " << faceind << endl;
                            set<int> tetraproj;

                            // creating local vertex indices for a pentatope face
                            // and projecting the face onto the 3d base
                            if (bnd_method == 0)
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        temptetra[cnt] = pentatops[pentaind*(Dim + 1) + j];
                                        if (temptetra[cnt] > vert_per_base - 1)
                                            tetraproj.insert(temptetra[cnt] - vert_per_base);
                                        else
                                            tetraproj.insert(temptetra[cnt]);
                                        cnt++;
                                    }
                                }

                                //cout << "temptetra in local indices" << endl;
                                //printInt2D(temptetra,1,4);

                                //cout << "temptetra in global indices" << endl;
                            }
                            else // for bnd_method = 1 we create temptetra and projection in global indices
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        temptetra[cnt] = elverts_prism[pentatops[pentaind*(Dim + 1) + j]];
                                        tetraproj.insert(temptetra[cnt] % NumOf3DVertices );
                                        cnt++;
                                    }
                                }

                                //cout << "temptetra in global indices" << endl;
                                //printInt2D(temptetra,1,4);
                            }

                            /*
                            cout << "tetraproj:" << endl;
                            for ( int temp : tetraproj)
                                cout << temp << " ";
                            cout << endl;
                            */


                            // checking whether the projection is at the boundary of 3d mesh
                            if ( LocalBdrs.find(tetraproj) != LocalBdrs.end())
                            {
                                //cout << "Found a new boundary element" << endl;
                                //cout << "With local indices: " << endl;
                                //printInt2D(temptetra, 1, Dim);

                                // converting local indices to global indices and adding the new boundary element
                                if (bnd_method == 0)
                                {
                                    temptetra[0] = elverts_prism[temptetra[0]];
                                    temptetra[1] = elverts_prism[temptetra[1]];
                                    temptetra[2] = elverts_prism[temptetra[2]];
                                    temptetra[3] = elverts_prism[temptetra[3]];
                                }

                                //cout << "With global indices: " << endl;
                                //printInt2D(temptetra, 1, Dim);

                                intermesh->bdrelements[bdrelcount*Dim + 0] = temptetra[0];
                                intermesh->bdrelements[bdrelcount*Dim + 1] = temptetra[1];
                                intermesh->bdrelements[bdrelcount*Dim + 2] = temptetra[2];
                                intermesh->bdrelements[bdrelcount*Dim + 3] = temptetra[3];
                                intermesh->bdrattrs[bdrelcount] = 2;
                                bdrelcount++;

                            }


                        } // end of loop over pentatope faces
                    } // end of loop over pentatopes
                } // end of if local_nbdrfaces > 0

                // converting local indices in pentatopes to the global indices
                // replacing local vertex indices w.r.t to 4d prism to global!
                for ( int pentaind = 0; pentaind < Dim; ++pentaind)
                {
                    for ( int j = 0; j < Dim + 1; j++)
                    {
                        pentatops[pentaind*(Dim + 1) + j] = elverts_prism[pentatops[pentaind*(Dim + 1) + j]];
                    }
                }

            } //end of if local_method = 0 or 2

            // By this point, for the given 3d element:
            // 4d elemnts = pentatops are constructed, but stored in local array
            // boundary elements are constructed which correspond to the elements in the 4D prism


            // 3.11 adding the constructed pentatops to the 4d mesh
            for ( int penta_ind = 0; penta_ind < Dim; ++penta_ind)
            {
                intermesh->elements[elcount*(Dim + 1) + 0] = pentatops[penta_ind*(Dim+1) + 0];
                intermesh->elements[elcount*(Dim + 1) + 1] = pentatops[penta_ind*(Dim+1) + 1];
                intermesh->elements[elcount*(Dim + 1) + 2] = pentatops[penta_ind*(Dim+1) + 2];
                intermesh->elements[elcount*(Dim + 1) + 3] = pentatops[penta_ind*(Dim+1) + 3];
                intermesh->elements[elcount*(Dim + 1) + 4] = pentatops[penta_ind*(Dim+1) + 4];
                intermesh->elattrs[elcount] = 1;
                elcount++;

            }

            //printArr2DInt (&vert_to_vert_prism);


        } // end of loop over base elements
    } // end of loop over time slabs

    delete [] ordering;
    delete [] pentatops;
    delete [] elvert_coordprism;

    if (local_method == 1)
    {
        delete [] vert_latface;
        delete [] vert_3Dlatface;
        delete [] tetrahedronsAll;
    }
    if (local_method == 0 || local_method == 1)
        delete [] qhull_flags;

    return intermesh;
}

// serial space-time mesh constructor
Mesh::Mesh ( Mesh& meshbase, double tau, int Nsteps, int bnd_method, int local_method)
//void MeshSpaceTimeCylinder ( Mesh& mesh3d, Mesh& mesh4d, double tau, int Nsteps, int bnd_method, int local_method)
{
    MeshSpaceTimeCylinder_onlyArrays ( meshbase, tau, Nsteps, bnd_method, local_method);

    int refine = 1;
    CreateInternalMeshStructure(refine);

    return;
}

// M and N are two d-dimensional points 9double * arrays with their coordinates
inline double dist( double * M, double * N , int d)
{
    double res = 0.0;
    for ( int i = 0; i < d; ++i )
        res += (M[i] - N[i])*(M[i] - N[i]);
    return sqrt(res);
}

int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

int setzero(Array2D<int>* arrayint)
{
    for ( int i = 0; i < arrayint->NumRows(); ++i )
        for ( int j = 0; j < arrayint->NumCols(); ++j)
            (*arrayint)(i,j) = 0;
    return 0;
}

// takes coordinates of points and returns a permutation which makes the given vertices
// preserve the geometrical order (based on their coordinates comparison)
void sortingPermutationNew( const std::vector<std::vector<double> >& values, int * permutation)
{
    vector<PairPoint> pairs;
    pairs.reserve(values.size());
    for (unsigned int i = 0; i < values.size(); i++)
    {
        //cout << "i = " << i << endl;
        //for (int j = 0; j < values[i].size(); ++j)
            //cout << values[i][j] << " ";
        //cout << endl;
        pairs.push_back(PairPoint(values[i], i));
    }

    sort(pairs.begin(), pairs.end(), CmpPairPoint());

    typedef std::vector<PairPoint>::const_iterator I;
    int count = 0;
    for (I p = pairs.begin(); p != pairs.end(); ++p)
        permutation[count++] = p->second;

    //cout << "inside sorting permutation is" << endl;
    //for ( int i = 0; i < values.size(); ++i)
        //cout << permutation[i] << " ";
    //cout << endl;
}

// simple algorithm which computes sign of a given permutatation
// for now, this function is applied to permutations of size 3
// so there is no sense in implementing anything more complicated
// the sign is defined so that it is 1 for the loop of length = size
int permutation_sign( int * permutation, int size)
{
    int res = 0;
    const int lsize = size;
    int temp[lsize]; //visited or not
    for ( int i = 0; i < size; ++i)
        temp[i] = -1;

    int pos = 0;
    while ( pos < size )
    {
        if (temp[pos] == -1) // if element is unvisited
        {
            int cycle_len = 1;

            //computing cycle length which starts with unvisited element
            int k = pos;
            while (permutation[k] != pos )
            {
                temp[permutation[k]] = 1;
                k = permutation[k];
                cycle_len++;
            }
            //cout << "pos = " << pos << endl;
            //cout << "cycle of len " << cycle_len << " was found there" << endl;

            res += (cycle_len-1)%2;

            temp[pos] = 1;
        }

        pos++;
    }

    if (res % 2 == 0)
        return 1;
    else
        return -1;
}

#ifdef WITH_QHULL
inline void zero_intinit (int *arr, int size)
{
    for ( int i = 0; i < size; ++i )
        arr[i] = 0;
}

/*-------------------------------------------------
-print_summary(qh)
*/
void print_summary(qhT *qh) {
  facetT *facet;
  int k;

  printf("\n%d vertices and %d facets with normals:\n",
                 qh->num_vertices, qh->num_facets);
  FORALLfacets {
    for (k=0; k < qh->hull_dim; k++)
      printf("%6.2g ", facet->normal[k]);
    printf("\n");
  }
}

void qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... ) {
    va_list args;

    if (!fp) {
        if(!qh){
            qh_fprintf_stderr(6241, "userprintf_r.c: fp and qh not defined for qh_fprintf '%s'", fmt);
            qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
        }
        /* could use qh->qhmem.ferr, but probably better to be cautious */
        qh_fprintf_stderr(6232, "Qhull internal error (userprintf_r.c): fp is 0.  Wrong qh_fprintf called.\n");
        qh_errexit(qh, 6232, NULL, NULL);
    }
    va_start(args, fmt);
    if (qh && qh->ANNOTATEoutput) {
      fprintf(fp, "[QH%.4d]", msgcode);
    }else if (msgcode >= MSG_ERROR && msgcode < MSG_STDERR ) {
      fprintf(fp, "QH%.4d ", msgcode);
    }
    vfprintf(fp, fmt, args);
    va_end(args);

    /* Place debugging traps here. Use with option 'Tn' */

} /* qh_fprintf */

/*--------------------------------------------------
-makePrism- set points for dim Delaunay triangulation of 3D prism
  with 2 x dim points.
notes:
only 3D here!
*/
void makePrism(qhT *qh, coordT *points, int numpoints, int dim) {
  if ( dim != 3 )
  {
      std::cerr << " makePrism() does not work for dim = " << dim << " (only for dim = 3)" << std::endl;
      return;
  }
  if ( numpoints != 6 )
  {
      std::cerr << "Wrong numpoints in makePrism" << endl;
  }
  int j,k;
  coordT *point, realr;

  for (j=0; j<numpoints; j++) {
    point= points + j*dim;
    if (j == 0)
    {
        point[0] = 0.0;
        point[1] = 0.0;
        point[2] = 0.0;
    }
    if (j == 1)
    {
        point[0] = 1.0;
        point[1] = 0.0;
        point[2] = 0.0;
    }
    if (j == 2)
    {
        point[0] = 0.0;
        point[1] = 1.0;
        point[2] = 0.0;
    }
    if (j == 3)
    {
        point[0] = 0.0;
        point[1] = 0.0;
        point[2] = 3.0;
    }
    if (j == 4)
    {
        point[0] = 1.0;
        point[1] = 0.0;
        point[2] = 2.9;
    }
    if (j == 5)
    {
        point[0] = 0.0;
        point[1] = 1.0;
        point[2] = 3.1;
    }
  } // loop over points
} /*.makePrism.*/

/*--------------------------------------------------
-makeOrthotope - set points for dim Delaunay triangulation of dim-dimensional orthotope
  with 2 x (dim + 1) points.
notes:
With joggling the base coordinates
*/
void makeOrthotope(qhT *qh, coordT *points, int numpoints, int dim) {
  if ( numpoints != 1 << dim )
  {
      std::cerr << "Wrong numpoints in makeOrthotope" << endl;
  }

  //cout << "numpoints = " << numpoints << endl;
  int j,k;
  coordT *point, realr;

  if ( dim == 3)
  {
      for (j=0; j<numpoints; j++) {
        point= points + j*dim;
        if (j == 0)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 0.0;
        }
        if (j == 1)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 0.0;
        }
        if (j == 2)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 0.0;
        }
        if (j == 3)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 0.0;
        }
        if (j == 4)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 3.0;
        }
        if (j == 5)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 3.0;
        }
        if (j == 6)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 3.0;
        }
        if (j == 7)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 3.0;
        }

        for ( int coord = 0; coord < dim; ++coord)
            point[coord] += 1.0e-2 * coord * j;
      } // loop over points
  }

  if ( dim == 4)
  {
      for (j=0; j<numpoints; j++) {
        point= points + j*dim;
        if (j == 0)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 1)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 2)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 3)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 4)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 5)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 6)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 7)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 8)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 9)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 10)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 11)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 12)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }
        if (j == 13)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }
        if (j == 14)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }
        if (j == 15)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }

        for ( int coord = 0; coord < dim; ++coord)
            point[coord] += 1.0e-2 * coord * j;

      } // loop over points
  }


} /*.makeOrthotope.*/

// works only for a 3D or 4D prism (see def. of numpoints, etc.)
// volumetol should be large enough to eliminate zero-volume tetrahedrons
// and not too small to keep the proper ones.
// basically it should be about tau * h^n if used for the space-time mesh
// set outfile = NULL to have no output
int qhull_wrapper(int * simplices, qhT * qh, double * points, int dim, double volumetol, char * flags)
{
    if (dim != 4 && dim != 3)
    {
        cout << "Case dim = " << dim << " is not supported by qhull_wrapper" << endl;
        return -1;
    }
    int numpoints = dim * 2;  /* number of points */
    boolT ismalloc= False;    /* True if qhull should free points in qh_freeqhull() or reallocation */
    FILE *outfile= NULL;      /* output from qh_produce_output() \
                                 use NULL to skip qh_produce_output() */
    FILE *errfile= stderr;    /* error messages from qhull code */
    int exitcode;             /* 0 if no error from qhull */
    facetT *facet;            /* set by FORALLfacets */
    int curlong, totlong;     /* memory remaining after qh_memfreeshort */
    int i;

    //QHULL_LIB_CHECK

    qh_zero(qh, errfile);

    //printf( "\ncompute %d-d Delaunay triangulation for my prism \n", dim);
    //sprintf(flags, "qhull QJ s i d Qbb");
    //numpoints = SIZEprism;
    //makePrism(qh, points, numpoints, dim, (int)time(NULL));
    //for (i=numpoints; i--; )
      //rows[i]= points+dim*i;
    //qh_printmatrix(qh, outfile, "input", rows, numpoints, dim);
    exitcode= qh_new_qhull(qh, dim, numpoints, points, ismalloc,
                        flags, outfile, errfile);

    zero_intinit (simplices, dim*(dim+1));

    if (!exitcode) {                  /* if no error */
      /* 'qh->facet_list' contains the convex hull */
      /* If you want a Voronoi diagram ('v') and do not request output (i.e., outfile=NULL),
         call qh_setvoronoi_all() after qh_new_qhull(). */
      //print_summary(qh);
      //qh_printfacet3vertex(qh, stdout, facet, qh_PRINToff);
      //qh_printfacetNvertex_simplicial(qh, qh->fout, qh->facet_list, qh_PRINToff);
      //qh_printfacets(qh, qh->fout, qh->PRINTout[i], qh->facet_list, NULL, !qh_ALL);
      //qh_printsummary(qh, qh->ferr);

      facetT *facet, **facetp;
      setT *vertices;
      vertexT *vertex, **vertexp;

      int temp[dim+1];

      DenseMatrix Volume;
      Volume.SetSize(dim);

      int count = 0;
      FORALLfacet_(qh->facet_list)
      {
          if (facet->good)
          {
              int count2 = 0;

              FOREACHvertexreverse12_(facet->vertices)

              {
                  //qh_fprintf(qh, fp, 9131, "%d ", qh_pointid(qh, vertex->point));

                  //fprintf(qh->fout, "%d ", qh_pointid(qh, vertex->point));
                  //fprintf(qh->fout, "\n ");
                  temp[count2] = qh_pointid(qh, vertex->point);
                  //int pointid = qh_pointid(qh, vertex->point);

                  ++count2;
              }

              double volumesq = 0.0;

              double * pointss[dim + 1];
              for ( int i = 0; i < dim + 1; ++i)
                  pointss[i] = points + temp[i] * dim;

              for ( int i = 0; i < dim; ++i)
              {
                  for ( int j = 0; j < dim; ++j)
                  {
                      Volume.Elem(i,j) = pointss[i + 1][j] - pointss[0][j];
                  }
              }

              double volume = Volume.Det() / factorial(dim);
              //double volume = determinant4x4 ( Volume );

              volumesq = volume * volume;

              if ( fabs(sqrt(volumesq)) > volumetol )
              {
                  for ( int i = 0; i < count2; i++ )
                      simplices[count*(dim + 1) + i] = temp[i];
                  ++count;
              }
              else
              {
                  std::cout << "sliver pentatop rejected" << endl;
                  std::cout << "volume^2 = " << volumesq << endl;
              }
              //std::cout << "volume^2 = " << volumesq << endl;

          }// if facet->good
      } // loop over all facets


      /*
      std::cout << "Now final " << count << " simplices (in qhull wrapper) are:" << endl;
      for (int i = 0; i < dim; i++ ) // or count instead of dim if debugging
      {
          std::cout << "Tetrahedron " << i << ": ";
          std::cout << "vert indices: " << endl;
          for ( int j = 0; j < dim  +1; j++ )
          {
              std::cout << simplices[i*(dim + 1) + j] << " ";
          }
          std::cout << endl;
      }
      */

      qh->NOerrexit= True;
    }

    return 0;
}
#endif


}
