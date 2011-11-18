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

// Implementation of data type mesh

#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>
#include <string.h>
#include <time.h>

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/sort_pairs.hpp"

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
      return pow(fabs(J.Det()), 1./Dim);
   else if (type == 1)
      return J.CalcSingularvalue(Dim-1); // h_min
   else
      return J.CalcSingularvalue(0); // h_max
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

void Mesh::PrintCharacteristics(Vector *Vh, Vector *Vk)
{
   int i, dim;
   DenseMatrix J;
   double h_min, h_max, kappa_min, kappa_max, h, kappa;

   cout << "Mesh Characteristics:" << flush;

   dim = Dimension();
   J.SetSize(dim);

   if (Vh) Vh->SetSize(NumOfElements);
   if (Vk) Vk->SetSize(NumOfElements);

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementJacobian(i, J);
      h = pow(fabs(J.Det()), 1.0/double(dim));
      kappa = J.CalcSingularvalue(0) / J.CalcSingularvalue(dim-1);
      if (Vh) (*Vh)(i) = h;
      if (Vk) (*Vk)(i) = kappa;
      if (i == 0)
      {
         h_min = h_max = h;
         kappa_min = kappa_max = kappa;
      }
      else
      {
         if (h < h_min)  h_min = h;
         if (h > h_max)  h_max = h;
         if (kappa < kappa_min)  kappa_min = kappa;
         if (kappa > kappa_max)  kappa_max = kappa;
      }
   }

   if (dim == 2)
      cout << endl
           << "Number of vertices : " << GetNV() << endl
           << "Number of edges    : " << GetNEdges() << endl
           << "Number of elements : " << GetNE() << endl
           << "Number of bdr elem : " << GetNBE() << endl
           << "Euler Number       : " << EulerNumber2D() << endl
           << "h_min              : " << h_min << endl
           << "h_max              : " << h_max << endl
           << "kappa_min          : " << kappa_min << endl
           << "kappa_max          : " << kappa_max << endl
           << endl;
   else
      cout << endl
           << "Number of vertices : " << GetNV() << endl
           << "Number of edges    : " << GetNEdges() << endl
           << "Number of faces    : " << GetNFaces() << endl
           << "Number of elements : " << GetNE() << endl
           << "Number of bdr elem : " << GetNBE() << endl
           << "Euler Number       : " << EulerNumber() << endl
           << "h_min              : " << h_min << endl
           << "h_max              : " << h_max << endl
           << "kappa_min          : " << kappa_min << endl
           << "kappa_max          : " << kappa_max << endl
           << endl;
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
   }
   mfem_error("Mesh::GetTransformationFEforElement - unknown ElementType");
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
      int n = vdofs.Size()/Dim;
      pm.SetSize(Dim, n);
      for (int k = 0; k < Dim; k++)
         for (int j = 0; j < n; j++)
            pm(k,j) = (*Nodes)(vdofs[n*k+j]);
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
      pm.SetSize(Dim, nv);
      for (int k = 0; k < Dim; k++)
         for (int j = 0; j < nv; j++)
            pm(k, j) = nodes(k*n+v[j]);
      ElTr->SetFE(GetTransformationFEforElementType(GetElementType(i)));
   }
   else
   {
      Array<int> vdofs;
      Nodes->FESpace()->GetElementVDofs(i, vdofs);
      int n = vdofs.Size()/Dim;
      pm.SetSize(Dim, n);
      for (int k = 0; k < Dim; k++)
         for (int j = 0; j < n; j++)
            pm(k,j) = nodes(vdofs[n*k+j]);
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
   FaceTransformation.Attribute = GetBdrAttribute(i);
   FaceTransformation.ElementNo = i;  // boundary element number
   if (Nodes == NULL)
   {
      GetBdrPointMatrix(i, FaceTransformation.GetPointMat());
      FaceTransformation.SetFE(
         GetTransformationFEforElementType(GetBdrElementType(i)));
   }
   else
   {
      DenseMatrix &pm = FaceTransformation.GetPointMat();
      Array<int> vdofs;
      Nodes->FESpace()->GetBdrElementVDofs(i, vdofs);
      int n = vdofs.Size()/Dim;
      pm.SetSize(Dim, n);
      for (int k = 0; k < Dim; k++)
         for (int j = 0; j < n; j++)
            pm(k,j) = (*Nodes)(vdofs[n*k+j]);
      FaceTransformation.SetFE(Nodes->FESpace()->GetBE(i));
   }

   return &FaceTransformation;
}

void Mesh::GetFaceTransformation(int FaceNo, IsoparametricTransformation *FTr)
{
   FTr->Attribute = faces[FaceNo]->GetAttribute();
   FTr->ElementNo = FaceNo;
   const int *v = faces[FaceNo]->GetVertices();
   const int nv = faces[FaceNo]->GetNVertices();
   DenseMatrix &pm = FTr->GetPointMat();
   pm.SetSize(Dim, nv);
   for (int i = 0; i < Dim; i++)
      for (int j = 0; j < nv; j++)
         pm(i, j) = vertices[v[j]](i);
   FTr->SetFE(GetTransformationFEforElementType(faces[FaceNo]->GetType()));
}

ElementTransformation *Mesh::GetFaceTransformation(int FaceNo)
{
   GetFaceTransformation(FaceNo, &FaceTransformation);
   return &FaceTransformation;
}

void Mesh::GetLocalSegToTriTransformation(
   IsoparametricTransformation &Transf, int i)
{
   static const int tri_faces[3][2] = {{1, 0}, {2, 1}, {0, 2}};
   static const int seg_inv_orient[2][2] = {{0, 1}, {1, 0}};
   int j;
   const int *tv, *so;
   const IntegrationRule *TriVert;
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&SegmentFE);
   tv = tri_faces[i/64]; //  (i/64) is the local face no. in the triangle
   so = seg_inv_orient[i%64]; //  (i%64) is the orientation of the segment
   TriVert = Geometries.GetVertices(Geometry::TRIANGLE);
   locpm.SetSize(2, 2);
   for (j = 0; j < 2; j++)
   {
      locpm(0, so[j]) = TriVert->IntPoint(tv[j]).x;
      locpm(1, so[j]) = TriVert->IntPoint(tv[j]).y;
   }
}

void Mesh::GetLocalSegToQuadTransformation(
   IsoparametricTransformation &Transf, int i)
{
   static const int quad_faces[4][2] = {{1, 0}, {2, 1}, {3, 2}, {0, 3}};
   static const int seg_inv_orient[2][2] = {{0, 1}, {1, 0}};
   int j;
   const int *qv, *so;
   const IntegrationRule *QuadVert;
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&SegmentFE);
   qv = quad_faces[i/64]; //  (i/64) is the local face no. in the quad
   so = seg_inv_orient[i%64]; //  (i%64) is the orientation of the segment
   QuadVert = Geometries.GetVertices(Geometry::SQUARE);
   locpm.SetSize(2, 2);
   for (j = 0; j < 2; j++)
   {
      locpm(0, so[j]) = QuadVert->IntPoint(qv[j]).x;
      locpm(1, so[j]) = QuadVert->IntPoint(qv[j]).y;
   }
}

void Mesh::GetLocalTriToTetTransformation(
   IsoparametricTransformation &Transf, int i)
{
   static const int tet_faces[4][3] = {{1, 2, 3}, {0, 3, 2},
                                       {0, 1, 3}, {0, 2, 1}};
   static const int tri_inv_orient[6][3] = {{0, 1, 2}, {1, 0, 2},
                                            {1, 2, 0}, {2, 1, 0},
                                            {2, 0, 1}, {0, 2, 1}};
   int j;
   const int *tv, *to;
   const IntegrationRule *TetVert;
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&TriangleFE);
   tv = tet_faces[i/64];  //  (i/64) is the local face no. in the tet
   //  (i%64) is the orientation of the tetrahedron face
   //         w.r.t. the face element
   to = tri_inv_orient[i%64];
   TetVert = Geometries.GetVertices(Geometry::TETRAHEDRON);
   locpm.SetSize(3, 3);
   for (j = 0; j < 3; j++)
   {
      locpm(0, to[j]) = TetVert->IntPoint(tv[j]).x;
      locpm(1, to[j]) = TetVert->IntPoint(tv[j]).y;
      locpm(2, to[j]) = TetVert->IntPoint(tv[j]).z;
   }
}

void Mesh::GetLocalQuadToHexTransformation(
   IsoparametricTransformation &Transf, int i)
{
   static const int hex_faces[6][4] = {{3, 2, 1, 0}, {0, 1, 5, 4},
                                       {1, 2, 6, 5}, {2, 3, 7, 6},
                                       {3, 0, 4, 7}, {4, 5, 6, 7}};
   // must be  'quad_inv_or' ... fix me
   static const int quad_orient[8][4] = {{0, 1, 2, 3}, {0, 3, 2, 1},
                                         {1, 2, 3, 0}, {1, 0, 3, 2},
                                         {2, 3, 0, 1}, {2, 1, 0, 3},
                                         {3, 0, 1, 2}, {3, 2, 1, 0}};
   int j;
   const int *hv, *qo;
   const IntegrationRule *HexVert;
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&QuadrilateralFE);
   hv = hex_faces[i/64];   //  (i/64) is the local face no. in the hex
   qo = quad_orient[i%64]; //  (i%64) is the orientation of the quad
   HexVert = Geometries.GetVertices(Geometry::CUBE);
   locpm.SetSize(3, 4);
   for (j = 0; j < 4; j++)
   {
      locpm(0, qo[j]) = HexVert->IntPoint(hv[j]).x;
      locpm(1, qo[j]) = HexVert->IntPoint(hv[j]).y;
      locpm(2, qo[j]) = HexVert->IntPoint(hv[j]).z;
   }
}

FaceElementTransformations *Mesh::GetFaceElementTransformations(int FaceNo,
                                                                int mask)
{
   //  setup the transformation for the first element
   FaceElemTr.Elem1No = faces_info[FaceNo].Elem1No;
   if (mask & 1)
   {
      GetElementTransformation(FaceElemTr.Elem1No, &Transformation);
      FaceElemTr.Elem1 = &Transformation;
   }
   else
      FaceElemTr.Elem1 = NULL;

   //  setup the transformation for the second element
   //     return NULL in the Elem2 field if there's no second element, i.e.
   //     the face is on the "boundary"
   if ( (FaceElemTr.Elem2No = faces_info[FaceNo].Elem2No) >= 0 && (mask & 2))
   {
#ifdef MFEM_DEBUG
      if (NURBSext && (mask & 1))
         mfem_error("Mesh::GetFaceElementTransformations :"
                    " NURBS mesh is not supported!");
#endif
      GetElementTransformation(FaceElemTr.Elem2No, &Transformation2);
      FaceElemTr.Elem2 = &Transformation2;
   }
   else
      FaceElemTr.Elem2 = NULL;

   FaceElemTr.FaceGeom = faces[FaceNo]->GetGeometryType();

   // setup the face transformation
   if (mask & 16)
      FaceElemTr.Face = GetFaceTransformation(FaceNo);
   else
      FaceElemTr.Face = NULL;

   // setup Loc1 & Loc2
   switch (faces[FaceNo]->GetType())
   {
   case Element::SEGMENT:
      if (mask & 4)
      {
         if (GetElementType(faces_info[FaceNo].Elem1No) == Element::TRIANGLE)
            GetLocalSegToTriTransformation(FaceElemTr.Loc1.Transf,
                                           faces_info[FaceNo].Elem1Inf);
         else // assume the element is a quad
            GetLocalSegToQuadTransformation(FaceElemTr.Loc1.Transf,
                                            faces_info[FaceNo].Elem1Inf);
      }

      if (FaceElemTr.Elem2No >= 0 && (mask & 8))
      {
         if (GetElementType(faces_info[FaceNo].Elem2No)
             == Element::TRIANGLE)
            GetLocalSegToTriTransformation(FaceElemTr.Loc2.Transf,
                                           faces_info[FaceNo].Elem2Inf);
         else // assume the element is a quad
            GetLocalSegToQuadTransformation(FaceElemTr.Loc2.Transf,
                                            faces_info[FaceNo].Elem2Inf);
      }
      break;
   case Element::TRIANGLE:
      // ---------  assumes the face is a triangle -- face of a tetrahedron
      if (mask & 4)
         GetLocalTriToTetTransformation(FaceElemTr.Loc1.Transf,
                                        faces_info[FaceNo].Elem1Inf);
      if (FaceElemTr.Elem2No >= 0 && (mask & 8))
         GetLocalTriToTetTransformation(FaceElemTr.Loc2.Transf,
                                        faces_info[FaceNo].Elem2Inf);
      break;
   case Element::QUADRILATERAL:
      // ---------  assumes the face is a quad -- face of a hexahedron
      if (mask & 4)
         GetLocalQuadToHexTransformation(FaceElemTr.Loc1.Transf,
                                         faces_info[FaceNo].Elem1Inf);
      if (FaceElemTr.Elem2No >= 0 && (mask & 8))
         GetLocalQuadToHexTransformation(FaceElemTr.Loc2.Transf,
                                         faces_info[FaceNo].Elem2Inf);
      break;
   }

   return &FaceElemTr;
}

FaceElementTransformations *Mesh::GetBdrFaceTransformations(int BdrElemNo)
{
   FaceElementTransformations *tr;
   int fn;
   if (Dim == 3)
      fn = be_to_face[BdrElemNo];
   else
      fn = be_to_edge[BdrElemNo];
   if (faces_info[fn].Elem2No >= 0)
      return NULL;
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

void Mesh::Init()
{
   NumOfVertices = NumOfElements = NumOfBdrElements = NumOfEdges = -1;
   WantTwoLevelState = 0;
   State = Mesh::NORMAL;
   Nodes = NULL;
   own_nodes = 1;
   NURBSext = NULL;
}

void Mesh::InitTables()
{
   el_to_edge = el_to_face = el_to_el =
      bel_to_edge = face_edge = edge_vertex = NULL;
}

void Mesh::DeleteTables()
{
   if (el_to_edge != NULL)
      delete el_to_edge;

   if (el_to_face != NULL)
      delete el_to_face;

   if (el_to_el != NULL)
      delete el_to_el;

   if (Dim == 3 && bel_to_edge != NULL)
      delete bel_to_edge;

   if (face_edge != NULL)
      delete face_edge;

   if (edge_vertex != NULL)
      delete edge_vertex;

   InitTables();
}

void Mesh::DeleteCoarseTables()
{
   delete el_to_el;
   delete face_edge;
   delete edge_vertex;

   el_to_el = face_edge = edge_vertex = NULL;
}

void Mesh::SetAttributes()
{
   int i, j, nattr;
   Array<int> attribs;

   attribs.SetSize(GetNBE());
   for (i = 0; i < attribs.Size(); i++)
      attribs[i] = GetBdrAttribute(i);
   attribs.Sort();

   if (attribs.Size() > 0)
      nattr = 1;
   else
      nattr = 0;
   for (i = 1; i < attribs.Size(); i++)
      if (attribs[i] != attribs[i-1])
         nattr++;

   bdr_attributes.SetSize(nattr);
   if (nattr > 0)
   {
      bdr_attributes[0] = attribs[0];
      for (i = j = 1; i < attribs.Size(); i++)
         if (attribs[i] != attribs[i-1])
            bdr_attributes[j++] = attribs[i];
      if (attribs[0] <= 0)
         cout << "Mesh::SetAttributes(): "
            "Non-positive attributes on the boundary!"
              << endl;
   }


   attribs.SetSize(GetNE());
   for (i = 0; i < attribs.Size(); i++)
      attribs[i] = GetAttribute(i);
   attribs.Sort();

   if (attribs.Size() > 0)
      nattr = 1;
   else
      nattr = 0;
   for (i = 1; i < attribs.Size(); i++)
      if (attribs[i] != attribs[i-1])
         nattr++;

   attributes.SetSize(nattr);
   if (nattr > 0)
   {
      attributes[0] = attribs[0];
      for (i = j = 1; i < attribs.Size(); i++)
         if (attribs[i] != attribs[i-1])
            attributes[j++] = attribs[i];
      if (attribs[0] <= 0)
         cout << "Mesh::SetAttributes(): "
            "Non-positive attributes in the domain!"
              << endl;
   }
}

Mesh::Mesh(int _Dim, int NVert, int NElem, int NBdrElem)
{
   Dim = _Dim;

   Init();
   InitTables();

   NumOfVertices = 0;
   vertices.SetSize(NVert);  // just allocate space for vertices

   NumOfElements = 0;
   elements.SetSize(NElem);  // just allocate space for Element *

   NumOfBdrElements = 0;
   boundary.SetSize(NBdrElem);  // just allocate space for Element *
}

void Mesh::AddVertex(double *x)
{
   double *y = vertices[NumOfVertices]();

   for (int i = 0; i < Dim; i++)
      y[i] = x[i];
   NumOfVertices++;
}

void Mesh::AddTri(int *vi, int attr)
{
   elements[NumOfElements++] = new Triangle(vi, attr);
}

void Mesh::AddTriangle(int *vi, int attr)
{
   elements[NumOfElements++] = new Triangle(vi, attr);
}

void Mesh::AddQuad(int *vi, int attr)
{
   elements[NumOfElements++] = new Quadrilateral(vi, attr);
}

void Mesh::AddTet(int *vi, int attr)
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

void Mesh::AddHex(int *vi, int attr)
{
   elements[NumOfElements++] = new Hexahedron(vi, attr);
}

void Mesh::AddBdrSegment(int *vi, int attr)
{
   boundary[NumOfBdrElements++] = new Segment(vi, attr);
}

void Mesh::AddBdrTriangle(int *vi, int attr)
{
   boundary[NumOfBdrElements++] = new Triangle(vi, attr);
}

void Mesh::AddBdrQuad(int *vi, int attr)
{
   boundary[NumOfBdrElements++] = new Quadrilateral(vi, attr);
}

void Mesh::GenerateBoundaryElements()
{
   int i, j;
   Array<int> &be2face = (Dim == 2) ? be_to_edge : be_to_face;

   // GenerateFaces();

   for (i = 0; i < boundary.Size(); i++)
      FreeElement(boundary[i]);

   if (Dim == 3)
   {
      delete bel_to_edge;
      bel_to_edge = NULL;
   }

   // count the 'NumOfBdrElements'
   NumOfBdrElements = 0;
   for (i = 0; i < faces_info.Size(); i++)
      if (faces_info[i].Elem2No == -1)
         NumOfBdrElements++;

   boundary.SetSize(NumOfBdrElements);
   be2face.SetSize(NumOfBdrElements);
   for (j = i = 0; i < faces_info.Size(); i++)
      if (faces_info[i].Elem2No == -1)
      {
         boundary[j] = faces[i]->Duplicate(this);
         be2face[j++] = i;
      }
   // In 3D, 'bel_to_edge' is destroyed but it's not updated.
}

typedef struct {
   int edge;
   double length;
} edge_length;

// Used by qsort to sort edges in increasing (according their length) order.
static int edge_compare(const void *ii, const void *jj)
{
   edge_length *i = (edge_length *)ii, *j = (edge_length *)jj;
   if (i->length > j->length) return (1);
   if (i->length < j->length) return (-1);
   return (0);
}

void Mesh::FinalizeTriMesh(int generate_edges, int refine)
{
   CheckElementOrientation();

   if (refine)
      MarkTriMeshForRefinement();

   if (generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      GenerateFaces();
      CheckBdrElementOrientation();
   }
   else
      NumOfEdges = 0;

   NumOfFaces = 0;

   SetAttributes();

   meshgen = 1;
}

void Mesh::FinalizeQuadMesh(int generate_edges, int refine)
{
   CheckElementOrientation();

   if (generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      GenerateFaces();
      CheckBdrElementOrientation();
   }
   else
      NumOfEdges = 0;

   NumOfFaces = 0;

   SetAttributes();

   meshgen = 2;
}

void Mesh::MarkForRefinement()
{
   if (meshgen & 1)
   {
      if (Dim == 2)
         MarkTriMeshForRefinement();
      else if (Dim == 3)
         MarkTetMeshForRefinement();
   }
}

void Mesh::MarkTriMeshForRefinement()
{
   // Mark the longest triangle edge by rotating the indeces so that
   // vertex 0 - vertex 1 to be the longest element's edge.
   DenseMatrix pmat;
   for (int i = 0; i < NumOfElements; i++)
      if (elements[i]->GetType() == Element::TRIANGLE)
      {
         GetPointMatrix(i, pmat);
         elements[i]->MarkEdge(pmat);
      }
}

void Mesh::MarkTetMeshForRefinement()
{
   // Mark the longest tetrahedral edge by rotating the indices so that
   // vertex 0 - vertex 1 is the longest edge in the element.
   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);
   NumOfEdges = v_to_v.NumberOfEntries();
   edge_length *length = new edge_length[NumOfEdges];
   for (int i = 0; i < NumOfVertices; i++)
   {
      for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
      {
         int j = it.Index();
         length[j].length = GetLength(i, it.Column());
         length[j].edge = j;
      }
   }

   // sort in increasing order
   qsort(length, NumOfEdges, sizeof(edge_length), edge_compare);

   int *order = new int [NumOfEdges];
   for (int i = 0; i < NumOfEdges; i++)
      order[length[i].edge] = i;

   for (int i = 0; i < NumOfElements; i++)
      if (elements[i]->GetType() == Element::TETRAHEDRON)
         elements[i]->MarkEdge(v_to_v, order);

   for (int i = 0; i < NumOfBdrElements; i++)
      if (boundary[i]->GetType() == Element::TRIANGLE)
         boundary[i]->MarkEdge(v_to_v, order);

   delete [] order;
   delete [] length;
}

void Mesh::FinalizeTetMesh(int generate_edges, int refine)
{
   CheckElementOrientation();

   if (NumOfBdrElements == 0)
   {
      GetElementToFaceTable();
      GenerateFaces();
      GenerateBoundaryElements();
   }

   if (refine)
   {
      MarkTetMeshForRefinement();
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

   meshgen = 1;
}

void Mesh::FinalizeHexMesh(int generate_edges, int refine)
{
   CheckElementOrientation();

   GetElementToFaceTable();
   GenerateFaces();

   if (NumOfBdrElements == 0)
      GenerateBoundaryElements();

   CheckBdrElementOrientation();

   if (generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   else
      NumOfEdges = 0;

   SetAttributes();

   meshgen = 2;
}

Mesh::Mesh(int nx, int ny, Element::Type type, int generate_edges,
           double sx, double sy)
{
   int i, j, k;

   Dim = 2;

   Init();
   InitTables();

   // Creates quadrilateral mesh
   if (type == Element::QUADRILATERAL)
   {
      meshgen = 2;
      NumOfVertices = (nx+1) * (ny+1);
      NumOfElements = nx * ny;
      NumOfBdrElements = 2 * nx + 2 * ny;
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
         boundary[nx+i] = new Segment(m+i, m+i+1, 3);
      }
      m = nx+1;
      for (j = 0; j < ny; j++)
      {
         boundary[2*nx+j] = new Segment(j*m, (j+1)*m,  4);
         boundary[2*nx+ny+j] = new Segment(j*m+nx, (j+1)*m+nx, 2);
      }
   }

   // Creates triangular mesh
   if (type == Element::TRIANGLE)
   {
      meshgen = 1;
      NumOfVertices = (nx+1) * (ny+1);
      NumOfElements = 2 * nx * ny;
      NumOfBdrElements = 2 * nx + 2 * ny;
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
         boundary[nx+i] = new Segment(m+i, m+i+1, 3);
      }
      m = nx+1;
      for (j = 0; j < ny; j++)
      {
         boundary[2*nx+j] = new Segment(j*m, (j+1)*m,  4);
         boundary[2*nx+ny+j] = new Segment(j*m+nx, (j+1)*m+nx, 2);
      }

      MarkTriMeshForRefinement();
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
      NumOfEdges = 0;

   NumOfFaces = 0;

   attributes.Append(1);
   bdr_attributes.Append(1); bdr_attributes.Append(2);
   bdr_attributes.Append(3); bdr_attributes.Append(4);
}

Mesh::Mesh(int n)
{
   int j, ind[1];

   Dim = 1;

   Init();
   InitTables();

   meshgen = 1;

   NumOfVertices = n + 1;
   NumOfElements = n;
   NumOfBdrElements = 2;
   vertices.SetSize(NumOfVertices);
   elements.SetSize(NumOfElements);
   boundary.SetSize(NumOfBdrElements);

   // Sets vertices and the corresponding coordinates
   for (j = 0; j < n+1; j++)
      vertices[j](0) = (double) j / n;

   // Sets elements and the corresponding indices of vertices
   for (j = 0; j < n; j++)
      elements[j] = new Segment(j, j+1, 1);

   // Sets the boundary elements
   ind[0] = 0;
   boundary[0] = new Point(ind, 1);
   ind[0] = n;
   boundary[1] = new Point(ind, 2);

   NumOfEdges = 0;
   NumOfFaces = 0;
}

Mesh::Mesh(istream &input, int generate_edges, int refine)
{
   Init();
   InitTables();
   Load(input, generate_edges, refine);
}

Element *Mesh::NewElement(int geom)
{
   switch (geom)
   {
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
   }

   return NULL;
}

Element *Mesh::ReadElement(istream &input)
{
   int attr, geom, nv, *v;
   Element *el;

   input >> attr >> geom;
   el = NewElement(geom);
   el->SetAttribute(attr);
   nv = el->GetNVertices();
   v  = el->GetVertices();
   for (int i = 0; i < nv; i++)
      input >> v[i];

   return el;
}

void Mesh::PrintElement(Element *el, ostream &out)
{
   out << el->GetAttribute() << ' ' << el->GetGeometryType();
   const int nv = el->GetNVertices();
   const int *v = el->GetVertices();
   for (int j = 0; j < nv; j++)
      out << ' ' << v[j];
   out << '\n';
}

// see Tetrahedron::edges
static const int vtk_quadratic_tet[10] =
{ 0, 1, 2, 3, 4, 7, 5, 6, 8, 9 };

// see Hexahedron::edges & Mesh::GenerateFaces
static const int vtk_quadratic_hex[27] =
{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
  24, 22, 21, 23, 20, 25, 26 };

void skip_comment_lines(istream &is, const char comment_char)
{
   while(1)
   {
      is >> ws;
      if (is.peek() != comment_char)
         break;
      is.ignore(numeric_limits<streamsize>::max(), '\n');
   }
}

void Mesh::Load(istream &input, int generate_edges, int refine)
{
   int i, j, ints[32], n, attr, curved = 0, read_gf = 1;
   const int buflen = 1024;
   char buf[buflen];

#ifdef MFEM_DEBUG
   if (!input)
      mfem_error("Input file stream not opened : Mesh::Load");
#endif

   if (NumOfVertices != -1)
   {
      // Delete the elements.
      for (i = 0; i < NumOfElements; i++)
         // delete elements[i];
         FreeElement(elements[i]);
      elements.DeleteAll();

      // Delete the vertices.
      vertices.DeleteAll();

      // Delete the boundary elements.
      for (i = 0; i < NumOfBdrElements; i++)
         // delete boundary[i];
         FreeElement(boundary[i]);
      boundary.DeleteAll();

      // Delete interior faces (if generated)
      for (i = 0; i < faces.Size(); i++)
         FreeElement(faces[i]);
      faces.DeleteAll();

      faces_info.DeleteAll();

      // Delete the edges (if generated).
      DeleteTables();
      be_to_edge.DeleteAll();
      be_to_face.DeleteAll();

   }

   InitTables();
   if (own_nodes) delete Nodes;
   Nodes = NULL;

   string mesh_type;
   input >> ws;
   getline(input, mesh_type);

   if (mesh_type == "MFEM mesh v1.0")
   {
      // Read MFEM mesh v1.0 format
      string ident;

      // read lines begining with '#' (comments)
      skip_comment_lines(input, '#');

      input >> ident; // 'dimension'
      input >> Dim;

      skip_comment_lines(input, '#');

      input >> ident; // 'elements'
      input >> NumOfElements;
      elements.SetSize(NumOfElements);
      for (j = 0; j < NumOfElements; j++)
         elements[j] = ReadElement(input);

      skip_comment_lines(input, '#');

      input >> ident; // 'boundary'
      input >> NumOfBdrElements;
      boundary.SetSize(NumOfBdrElements);
      for (j = 0; j < NumOfBdrElements; j++)
         boundary[j] = ReadElement(input);

      skip_comment_lines(input, '#');

      input >> ident; // 'vertices'
      input >> NumOfVertices;
      vertices.SetSize(NumOfVertices);

      input >> ws >> ident;
      if (ident != "nodes")
      {
         // read the vertices
         int vdim = atoi(ident.c_str());
         for (j = 0; j < NumOfVertices; j++)
            for (i = 0; i < vdim; i++)
               input >> vertices[j](i);
      }
      else
      {
         // prepare to read the nodes
         input >> ws;
         curved = 1;
      }
   }
   else if (mesh_type == "linemesh")
   {
      int j,p1,p2,a;

      Dim = 1;

      input >> NumOfVertices;
      vertices.SetSize(NumOfVertices);
      // Sets vertices and the corresponding coordinates
      for (j = 0; j < NumOfVertices; j++)
         input >> vertices[j](0);

      input >> NumOfElements;
      elements.SetSize(NumOfElements);
      // Sets elements and the corresponding indices of vertices
      for (j = 0; j < NumOfElements; j++)
      {
         input >> a >> p1 >> p2;
         elements[j] = new Segment(p1-1, p2-1, a);
      }

      int ind[1];
      input >> NumOfBdrElements;
      boundary.SetSize(NumOfBdrElements);
      for (j = 0; j < NumOfBdrElements; j++)
      {
         input >> a >> ind[0];
         ind[0]--;
         boundary[j] = new Point(ind,a);
      }
   }
   else if (mesh_type == "areamesh2" || mesh_type == "curved_areamesh2")
   {
      // Read planar mesh in Netgen format.
      Dim = 2;

      if (mesh_type == "curved_areamesh2")
         curved = 1;

      // Read the boundary elements.
      input >> NumOfBdrElements;
      boundary.SetSize(NumOfBdrElements);
      for (i = 0; i < NumOfBdrElements; i++)
      {
         input >> attr
               >> ints[0] >> ints[1];
         ints[0]--; ints[1]--;
         boundary[i] = new Segment(ints, attr);
      }

      // Read the elements.
      input >> NumOfElements;
      elements.SetSize(NumOfElements);
      for (i = 0; i < NumOfElements; i++)
      {
         input >> attr >> n;
         for (j = 0; j < n; j++)
         {
            input >> ints[j];
            ints[j]--;
         }
         switch (n)
         {
         case 2:
            elements[i] = new Segment(ints, attr);
            break;
         case 3:
            elements[i] = new Triangle(ints, attr);
            break;
         case 4:
            elements[i] = new Quadrilateral(ints, attr);
            break;
         }
      }

      if (!curved)
      {
         // Read the vertices.
         input >> NumOfVertices;
         vertices.SetSize(NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
               input >> vertices[i](j);
      }
      else
      {
         input >> NumOfVertices;
         vertices.SetSize(NumOfVertices);
         input >> ws;
      }
   }
   else if (mesh_type == "NETGEN" || mesh_type == "NETGEN_Neutral_Format")
   {
      // Read a netgen format mesh of tetrahedra.
      Dim = 3;

      // Read the vertices
      input >> NumOfVertices;

      vertices.SetSize(NumOfVertices);
      for (i = 0; i < NumOfVertices; i++)
         for (j = 0; j < Dim; j++)
            input >> vertices[i](j);

      // Read the elements
      input >> NumOfElements;
      elements.SetSize(NumOfElements);
      for (i = 0; i < NumOfElements; i++)
      {
         input >> attr;
         for (j = 0; j < 4; j++)
         {
            input >> ints[j];
            ints[j]--;
         }
#ifdef MFEM_USE_MEMALLOC
         Tetrahedron *tet;
         tet = TetMemory.Alloc();
         tet->SetVertices(ints);
         tet->SetAttribute(attr);
         elements[i] = tet;
#else
         elements[i] = new Tetrahedron(ints, attr);
#endif
      }

      // Read the boundary information.
      input >> NumOfBdrElements;
      boundary.SetSize(NumOfBdrElements);
      for (i = 0; i < NumOfBdrElements; i++)
      {
         input >> attr;
         for (j = 0; j < 3; j++)
         {
            input >> ints[j];
            ints[j]--;
         }
         boundary[i] = new Triangle(ints, attr);
      }
   }
   else if (mesh_type == "TrueGrid")
   {
      // Reading TrueGrid mesh.

      // TODO: find the actual dimension
      Dim = 3;

      if (Dim == 2)
      {
         int vari;
         double varf;

         input >> vari >> NumOfVertices >> vari >> vari >> NumOfElements;
         input.getline(buf, buflen);
         input.getline(buf, buflen);
         input >> vari;
         input.getline(buf, buflen);
         input.getline(buf, buflen);
         input.getline(buf, buflen);

         // Read the vertices.
         vertices.SetSize(NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
         {
            input >> vari >> varf >> vertices[i](0) >> vertices[i](1);
            input.getline(buf, buflen);
         }

         // Read the elements.
         elements.SetSize(NumOfElements);
         for (i = 0; i < NumOfElements; i++)
         {
            input >> vari >> attr;
            for (j = 0; j < 4; j++)
            {
               input >> ints[j];
               ints[j]--;
            }
            input.getline(buf, buflen);
            input.getline(buf, buflen);
            elements[i] = new Quadrilateral(ints, attr);
         }
      }
      else if (Dim == 3)
      {
         int vari;
         double varf;
         input >> vari >> NumOfVertices >> NumOfElements;
         input.getline(buf, buflen);
         input.getline(buf, buflen);
         input >> vari >> vari >> NumOfBdrElements;
         input.getline(buf, buflen);
         input.getline(buf, buflen);
         input.getline(buf, buflen);
         // Read the vertices.
         vertices.SetSize(NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
         {
            input >> vari >> varf >> vertices[i](0) >> vertices[i](1)
                  >> vertices[i](2);
            input.getline(buf, buflen);
         }
         // Read the elements.
         elements.SetSize(NumOfElements);
         for (i = 0; i < NumOfElements; i++)
         {
            input >> vari >> attr;
            for (j = 0; j < 8; j++)
            {
               input >> ints[j];
               ints[j]--;
            }
            input.getline(buf, buflen);
            elements[i] = new Hexahedron(ints, attr);
         }
         // Read the boundary elements.
         boundary.SetSize(NumOfBdrElements);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            input >> attr;
            for (j = 0; j < 4; j++)
            {
               input >> ints[j];
               ints[j]--;
            }
            input.getline(buf, buflen);
            boundary[i] = new Quadrilateral(ints, attr);
         }
      }
   }
   else if (mesh_type == "# vtk DataFile Version 3.0" ||
            mesh_type == "# vtk DataFile Version 2.0")
   {
      // Reading VTK mesh

      string buff;
      getline(input, buff); // comment line
      getline(input, buff);
      if (buff != "ASCII")
      {
         mfem_error("Mesh::Load : VTK mesh is not in ASCII format!");
         return;
      }
      getline(input, buff);
      if (buff != "DATASET UNSTRUCTURED_GRID")
      {
         mfem_error("Mesh::Load : VTK mesh is not UNSTRUCTURED_GRID!");
         return;
      }

      // Read the points, skipping optional sections such as the FIELD data from
      // VisIt's VTK export (or from Mesh::PrintVTK with field_data==1).
      do
      {
         input >> buff;
         if (!input.good())
            mfem_error("Mesh::Load : VTK mesh does not have POINTS data!");
      }
      while (buff != "POINTS");
      int np = 0;
      Vector points;
      {
         input >> np >> ws;
         points.SetSize(3*np);
         getline(input, buff); // "double"
         for (i = 0; i < points.Size(); i++)
            input >> points(i);
      }

      // Read the cells
      NumOfElements = n = 0;
      Array<int> cells_data;
      input >> ws >> buff;
      if (buff == "CELLS")
      {
         input >> NumOfElements >> n >> ws;
         cells_data.SetSize(n);
         for (i = 0; i < n; i++)
            input >> cells_data[i];
      }

      // Read the cell types
      Dim = 0;
      int order = 1;
      input >> ws >> buff;
      if (buff == "CELL_TYPES")
      {
         input >> NumOfElements;
         elements.SetSize(NumOfElements);
         for (j = i = 0; i < NumOfElements; i++)
         {
            int ct;
            input >> ct;
            switch (ct)
            {
            case 5:   // triangle
               Dim = 2;
               elements[i] = new Triangle(&cells_data[j+1]);
               break;
            case 9:   // quadrilateral
               Dim = 2;
               elements[i] = new Quadrilateral(&cells_data[j+1]);
               break;
            case 10:  // tetrahedron
               Dim = 3;
#ifdef MFEM_USE_MEMALLOC
               elements[i] = TetMemory.Alloc();
               elements[i]->SetVertices(&cells_data[j+1]);
#else
               elements[i] = new Tetrahedron(&cells_data[j+1]);
#endif
               break;
            case 12:  // hexahedron
               Dim = 3;
               elements[i] = new Hexahedron(&cells_data[j+1]);
               break;

            case 22:  // quadratic triangle
               Dim = 2;
               order = 2;
               elements[i] = new Triangle(&cells_data[j+1]);
               break;
            case 28:  // biquadratic quadrilateral
               Dim = 2;
               order = 2;
               elements[i] = new Quadrilateral(&cells_data[j+1]);
               break;
            case 24:  // quadratic tetrahedron
               Dim = 3;
               order = 2;
#ifdef MFEM_USE_MEMALLOC
               elements[i] = TetMemory.Alloc();
               elements[i]->SetVertices(&cells_data[j+1]);
#else
               elements[i] = new Tetrahedron(&cells_data[j+1]);
#endif
               break;
            case 29:  // triquadratic hexahedron
               Dim = 3;
               order = 2;
               elements[i] = new Hexahedron(&cells_data[j+1]);
               break;
            default:
               cerr << "Mesh::Load : VTK mesh : cell type " << ct
                    << " is not supported!" << endl;
               mfem_error();
               return;
            }
            j += cells_data[j] + 1;
         }
      }

      // Read attributes
      streampos sp = input.tellg();
      input >> ws >> buff;
      if (buff == "CELL_DATA")
      {
         input >> n >> ws;
         getline(input, buff);
         if (buff == "SCALARS material int" || buff == "SCALARS material float")
         {
            getline(input, buff); // "LOOKUP_TABLE default"
            for (i = 0; i < NumOfElements; i++)
            {
               input >> attr;
               elements[i]->SetAttribute(attr);
            }
         }
         else
            input.seekg(sp);
      }
      else
         input.seekg(sp);

      if (order == 1)
      {
         cells_data.DeleteAll();
         NumOfVertices = np;
         vertices.SetSize(np);
         for (i = 0; i < np; i++)
         {
            vertices[i](0) = points(3*i+0);
            vertices[i](1) = points(3*i+1);
            vertices[i](2) = points(3*i+2);
         }
         points.Destroy();

         // No boundary is defined in a VTK mesh
         NumOfBdrElements = 0;
      }
      else if (order == 2)
      {
         curved = 1;

         // generate new enumeration for the vertices
         Array<int> pts_dof(np);
         pts_dof = -1;
         for (n = i = 0; i < NumOfElements; i++)
         {
            int *v = elements[i]->GetVertices();
            int nv = elements[i]->GetNVertices();
            for (j = 0; j < nv; j++)
               if (pts_dof[v[j]] == -1)
                  pts_dof[v[j]] = n++;
         }
         // keep the original ordering of the vertices
         for (n = i = 0; i < np; i++)
            if (pts_dof[i] != -1)
               pts_dof[i] = n++;
         // update the element vertices
         for (i = 0; i < NumOfElements; i++)
         {
            int *v = elements[i]->GetVertices();
            int nv = elements[i]->GetNVertices();
            for (j = 0; j < nv; j++)
               v[j] = pts_dof[v[j]];
         }
         // Define the 'vertices' from the 'points' through the 'pts_dof' map
         NumOfVertices = n;
         vertices.SetSize(n);
         for (i = 0; i < np; i++)
         {
            if ((j = pts_dof[i]) != -1)
            {
               vertices[j](0) = points(3*i+0);
               vertices[j](1) = points(3*i+1);
               vertices[j](2) = points(3*i+2);
            }
         }

         // No boundary is defined in a VTK mesh
         NumOfBdrElements = 0;

         // Generate faces and edges so that we can define quadratic
         // FE space on the mesh

         // Generate faces
         if (Dim > 2)
         {
            GetElementToFaceTable();
            GenerateFaces();
         }
         else
            NumOfFaces = 0;

         // Generate edges
         el_to_edge = new Table;
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         if (Dim == 2)
            GenerateFaces(); // 'Faces' in 2D refers to the edges

         // Define quadratic FE space
         FiniteElementCollection *fec = new QuadraticFECollection;
         FiniteElementSpace *fes = new FiniteElementSpace(this, fec, Dim);
         Nodes = new GridFunction(fes);
         Nodes->MakeOwner(fec); // Nodes will destroy 'fec' and 'fes'
         own_nodes = 1;

         // Map vtk points to edge/face/element dofs
         Array<int> dofs;
         for (n = i = 0; i < NumOfElements; i++)
         {
            fes->GetElementDofs(i, dofs);
            const int *vtk_mfem;
            switch (elements[i]->GetGeometryType())
            {
            case Geometry::TRIANGLE:
            case Geometry::SQUARE:
               vtk_mfem = vtk_quadratic_hex; break; // identity map
            case Geometry::TETRAHEDRON:
               vtk_mfem = vtk_quadratic_tet; break;
            case Geometry::CUBE:
               vtk_mfem = vtk_quadratic_hex; break;
            }

            for (n++, j = 0; j < dofs.Size(); j++, n++)
            {
               if (pts_dof[cells_data[n]] == -1)
               {
                  pts_dof[cells_data[n]] = dofs[vtk_mfem[j]];
               }
               else
               {
                  if (pts_dof[cells_data[n]] != dofs[vtk_mfem[j]])
                     mfem_error("Mesh::Load : VTK mesh : "
                                "inconsistent quadratic mesh!");
               }
            }
         }

         // Define the 'Nodes' from the 'points' through the 'pts_dof' map
         for (i = 0; i < np; i++)
         {
            dofs.SetSize(1);
            if ((dofs[0] = pts_dof[i]) != -1)
            {
               fes->DofsToVDofs(dofs);
               for (j = 0; j < dofs.Size(); j++)
                  (*Nodes)(dofs[j]) = points(3*i+j);
            }
         }

         read_gf = 0;
      }
   }
   else if (mesh_type == "MFEM NURBS mesh v1.0")
   {
      NURBSext = new NURBSExtension(input);

      Dim              = NURBSext->Dimension();
      NumOfVertices    = NURBSext->GetNV();
      NumOfElements    = NURBSext->GetNE();
      NumOfBdrElements = NURBSext->GetNBE();

      NURBSext->GetElementTopo(elements);
      NURBSext->GetBdrElementTopo(boundary);

      vertices.SetSize(NumOfVertices);
      curved = 1;
      if (NURBSext->HavePatches())
      {
         NURBSFECollection  *fec = new NURBSFECollection(NURBSext->GetOrder());
         FiniteElementSpace *fes = new FiniteElementSpace(this, fec, Dim,
                                                          Ordering::byVDIM);
         Nodes = new GridFunction(fes);
         Nodes->MakeOwner(fec);
         NURBSext->SetCoordsFromPatches(*Nodes);
         own_nodes = 1;
         read_gf = 0;
         int vd = Nodes->VectorDim();
         for (i = 0; i < vd; i++)
         {
            Vector vert_val;
            Nodes->GetNodalValues(vert_val, i+1);
            for (j = 0; j < NumOfVertices; j++)
               vertices[j](i) = vert_val(j);
         }
      }
      else
         read_gf = 1;
   }
   else
   {
      mfem_error("Mesh::Load : Unknown input mesh format!");
      return;
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

   // set the mesh type ('meshgen')
   meshgen = 0;
   for (i = 0; i < NumOfElements; i++)
   {
      switch (elements[i]->GetType())
      {
      case Element::SEGMENT:
      case Element::TRIANGLE:
      case Element::TETRAHEDRON:
         meshgen |= 1; break;

      case Element::QUADRILATERAL:
      case Element::HEXAHEDRON:
         meshgen |= 2;
      }
   }

   if (NumOfBdrElements == 0 && Dim > 2)
   {
      // in 3D, generate boundary elements before we 'MarkForRefinement'
      GetElementToFaceTable();
      GenerateFaces();
      GenerateBoundaryElements();
   }

   if (!curved)
   {
      // check and fix element orientation
      CheckElementOrientation();

      if (refine)
         MarkForRefinement();
   }

   // generate the faces
   if (Dim > 2)
   {
      GetElementToFaceTable();
      GenerateFaces();
      // check and fix boundary element orientation
      if ( !(curved && (meshgen & 1)) )
         CheckBdrElementOrientation();
   }
   else
      NumOfFaces = 0;

   // generate edges if requested
   if (Dim > 1 && generate_edges == 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
      {
         GenerateFaces(); // 'Faces' in 2D refers to the edges
         if (NumOfBdrElements == 0)
            GenerateBoundaryElements();
         // check and fix boundary element orientation
         if ( !(curved && (meshgen & 1)) )
            CheckBdrElementOrientation();
      }
      c_el_to_edge = NULL;
   }
   else
      NumOfEdges = 0;

   // generate the arrays 'attributes' and ' bdr_attributes'
   SetAttributes();

   if (curved)
   {
      if (read_gf)
      {
         Nodes = new GridFunction(this, input);
         own_nodes = 1;
         int vd = Nodes->VectorDim();
         for (i = 0; i < vd; i++)
         {
            Vector vert_val;
            Nodes->GetNodalValues(vert_val, i+1);
            for (j = 0; j < NumOfVertices; j++)
               vertices[j](i) = vert_val(j);
         }
      }

      // Check orientation and mark edges; only for triangles / tets
      if (meshgen & 1)
      {
         FiniteElementSpace *fes = Nodes->FESpace();
         const FiniteElementCollection *fec = fes->FEColl();
         int num_edge_dofs = fec->DofForGeometry(Geometry::SEGMENT);
         DSTable *old_v_to_v = NULL;
         if (num_edge_dofs)
         {
            old_v_to_v = new DSTable(NumOfVertices);
            GetVertexToVertexTable(*old_v_to_v);
         }
         // assuming all faces have the same geometry
         int num_face_dofs =
            (Dim < 3) ? 0 : fec->DofForGeometry(GetFaceBaseGeometry(0));
         // assuming all elements have the same geometry
         int num_elem_dofs = fec->DofForGeometry(GetElementBaseGeometry(0));

         // check orientation and mark for refinement using just vertices
         // (i.e. higher order curvature is not used)
         CheckElementOrientation();
         if (refine)
            MarkForRefinement(); // changes topology!

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
         // edge enumeration may be different but edge orientation is
         // the same
         if (num_edge_dofs > 0)
         {
            DSTable new_v_to_v(NumOfVertices);
            GetVertexToVertexTable(new_v_to_v);

            for (i = 0; i < NumOfVertices; i++)
            {
               for (DSTable::RowIterator it(new_v_to_v, i); !it; ++it)
               {
                  int old_i = (*old_v_to_v)(i, it.Column());
                  int new_i = it.Index();
#ifdef MFEM_DEBUG
                  if (old_i != new_i)
                     redges++;
#endif
                  old_dofs.SetSize(num_edge_dofs);
                  new_dofs.SetSize(num_edge_dofs);
                  for (j = 0; j < num_edge_dofs; j++)
                  {
                     old_dofs[j] = offset + old_i * num_edge_dofs + j;
                     new_dofs[j] = offset + new_i * num_edge_dofs + j;
                  }
                  fes->DofsToVDofs(old_dofs);
                  fes->DofsToVDofs(new_dofs);
                  for (j = 0; j < old_dofs.Size(); j++)
                     (*Nodes)(new_dofs[j]) = onodes(old_dofs[j]);
               }
            }
            offset += NumOfEdges * num_edge_dofs;
            delete old_v_to_v;
         }
#ifdef MFEM_DEBUG
         cout << "Mesh::Load : redges = " << redges << endl;
#endif

         // face dofs:
         // both enumeration and orientation of the faces
         // may be different
         if (num_face_dofs > 0)
         {
            // generate the old face-vertex table
            Table old_face_vertex;
            old_face_vertex.MakeI(NumOfFaces);
            for (i = 0; i < NumOfFaces; i++)
               old_face_vertex.AddColumnsInRow(i, faces[i]->GetNVertices());
            old_face_vertex.MakeJ();
            for (i = 0; i < NumOfFaces; i++)
               old_face_vertex.AddConnections(i, faces[i]->GetVertices(),
                                              faces[i]->GetNVertices());
            old_face_vertex.ShiftUpI();

            // update 'el_to_face', 'be_to_face', 'faces', and 'faces_info'
            STable3D *faces_tbl = GetElementToFaceTable(1);
            GenerateFaces();

            // loop over the old face numbers
            for (i = 0; i < NumOfFaces; i++)
            {
               int *old_v = old_face_vertex.GetRow(i), *new_v;
               int new_i, new_or, *dof_ord;
               switch (old_face_vertex.RowSize(i))
               {
               case 3:
                  new_i = (*faces_tbl)(old_v[0], old_v[1], old_v[2]);
                  new_v = faces[new_i]->GetVertices();
                  new_or = GetTriOrientation(old_v, new_v);
                  dof_ord = fec->DofOrderForOrientation(Geometry::TRIANGLE,
                                                        new_or);
                  break;
               case 4:
                  new_i = (*faces_tbl)(old_v[0], old_v[1], old_v[2], old_v[3]);
                  new_v = faces[new_i]->GetVertices();
                  new_or = GetQuadOrientation(old_v, new_v);
                  dof_ord = fec->DofOrderForOrientation(Geometry::SQUARE,
                                                        new_or);
                  break;
               }

               old_dofs.SetSize(num_face_dofs);
               new_dofs.SetSize(num_face_dofs);
               for (j = 0; j < num_face_dofs; j++)
               {
                  old_dofs[j] = offset +     i * num_face_dofs + j;
                  new_dofs[j] = offset + new_i * num_face_dofs + dof_ord[j];
                  // we assumed the dofs are non-directional
                  // i.e. dof_ord[j] is >= 0
               }
               fes->DofsToVDofs(old_dofs);
               fes->DofsToVDofs(new_dofs);
               for (j = 0; j < old_dofs.Size(); j++)
                  (*Nodes)(new_dofs[j]) = onodes(old_dofs[j]);
            }

            offset += NumOfFaces * num_face_dofs;
            delete faces_tbl;
         }

         // element dofs:
         // element orientation may be different
         if (num_elem_dofs > 0)
         {
            // matters when the 'fec' is
            // (this code is executed only for triangles/tets)
            // - Pk on triangles, k >= 4
            // - Qk on quads,     k >= 3
            // - Pk on tets,      k >= 5
            // - Qk on hexes,     k >= 3
            // - ...
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
      }
   }
}

Mesh::Mesh(Mesh *mesh_array[], int num_pieces)
{
   int      i, j, ie, ib, iv, *v, nv;
   Element *el;
   Mesh    *m;

   Init();
   InitTables();

   Dim = mesh_array[0]->Dimension();

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

      NumOfBdrElements = 0;
      for (i = 0; i < num_pieces; i++)
         NumOfBdrElements += mesh_array[i]->GetNBE();
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
            elements[lelem_elem[j]]->SetAttribute(m->GetAttribute(j));
         // copy the boundary
         for (j = 0; j < m->GetNBE(); j++)
         {
            el = m->GetBdrElement(j)->Duplicate(this);
            v  = el->GetVertices();
            nv = el->GetNVertices();
            for (int k = 0; k < nv; k++)
               v[k] = lvert_vert[v[k]];
            boundary[ib++] = el;
         }
         // copy the vertices
         for (j = 0; j < m->GetNV(); j++)
            vertices[lvert_vert[j]].SetCoords(m->GetVertex(j));
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
               v[k] += iv;
            elements[ie++] = el;
         }
         // copy the boundary elements
         for (j = 0; j < m->GetNBE(); j++)
         {
            el = m->GetBdrElement(j)->Duplicate(this);
            v  = el->GetVertices();
            nv = el->GetNVertices();
            for (int k = 0; k < nv; k++)
               v[k] += iv;
            boundary[ib++] = el;
         }
         // copy the vertices
         for (j = 0; j < m->GetNV(); j++)
            vertices[iv++].SetCoords(m->GetVertex(j));
      }
   }

   // set the mesh type ('meshgen')
   meshgen = 0;
   for (i = 0; i < num_pieces; i++)
      meshgen |= mesh_array[i]->MeshGenerator();

   // generate faces
   if (Dim > 2)
   {
      GetElementToFaceTable();
      GenerateFaces();
   }
   else
      NumOfFaces = 0;

   // generate edges
   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
         GenerateFaces(); // 'Faces' in 2D refers to the edges
   }
   else
      NumOfEdges = 0;

   // generate the arrays 'attributes' and ' bdr_attributes'
   SetAttributes();

   // copy the nodes (curvilinear meshes)
   GridFunction *g = mesh_array[0]->GetNodes();
   if (g)
   {
      Array<GridFunction *> gf_array(num_pieces);
      for (i = 0; i < num_pieces; i++)
         gf_array[i] = mesh_array[i]->GetNodes();
      Nodes = new GridFunction(this, gf_array, num_pieces);
      own_nodes = 1;
   }
}

void Mesh::KnotInsert(Array<KnotVector *> &kv)
{
   if (NURBSext == NULL)
      mfem_error("Mesh::KnotInsert : Not a NURBS mesh!");

   if (kv.Size() != NURBSext->GetNKV())
      mfem_error("Mesh::KnotInsert : KnotVector array size mismatch!");

   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->KnotInsert(kv);

   UpdateNURBS();
}

void Mesh::NURBSUniformRefinement()
{
   // do not check for NURBSext since this method is protected
   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->UniformRefinement();

   UpdateNURBS();
}

void Mesh::DegreeElevate(int t)
{
   if (NURBSext == NULL)
      mfem_error("Mesh::DegreeElevate : Not a NURBS mesh!");

   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->DegreeElevate(t);

   NURBSFECollection *nurbs_fec =
      dynamic_cast<NURBSFECollection *>(Nodes->OwnFEC());
   if (!nurbs_fec)
      mfem_error("Mesh::DegreeElevate");
   nurbs_fec->UpdateOrder(nurbs_fec->GetOrder() + t);

   UpdateNURBS();
}

void Mesh::UpdateNURBS()
{
   NURBSext->SetKnotsFromPatches();

   Dim = NURBSext->Dimension();

   if (NumOfElements != NURBSext->GetNE())
   {
      for (int i = 0; i < elements.Size(); i++)
         FreeElement(elements[i]);
      NumOfElements = NURBSext->GetNE();
      NURBSext->GetElementTopo(elements);
   }

   if (NumOfBdrElements != NURBSext->GetNBE())
   {
      for (int i = 0; i < boundary.Size(); i++)
         FreeElement(boundary[i]);
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
            vertices[j](i) = vert_val(j);
      }
   }

   if (el_to_edge)
   {
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim == 2)
         GenerateFaces();
   }

   if (el_to_face)
   {
      GetElementToFaceTable();
      GenerateFaces();
   }
}

void Mesh::LoadPatchTopo(istream &input, Array<int> &edge_to_knot)
{
   Init();
   InitTables();

   int j;

   // Read MFEM NURBS mesh v1.0 format
   string ident;

   skip_comment_lines(input, '#');

   input >> ident; // 'dimension'
   input >> Dim;

   skip_comment_lines(input, '#');

   input >> ident; // 'elements'
   input >> NumOfElements;
   elements.SetSize(NumOfElements);
   for (j = 0; j < NumOfElements; j++)
      elements[j] = ReadElement(input);

   skip_comment_lines(input, '#');

   input >> ident; // 'boundary'
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (j = 0; j < NumOfBdrElements; j++)
      boundary[j] = ReadElement(input);

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
         edge_to_knot[j] = -1 - edge_to_knot[j];
   }

   skip_comment_lines(input, '#');

   input >> ident; // 'vertices'
   input >> NumOfVertices;
   vertices.SetSize(0);

   meshgen = 2;

   // generate the faces
   if (Dim > 2)
   {
      GetElementToFaceTable();
      GenerateFaces();
      if (NumOfBdrElements == 0)
         GenerateBoundaryElements();
      CheckBdrElementOrientation();
   }
   else
      NumOfFaces = 0;

   // generate edges
   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      if (Dim < 3)
      {
         GenerateFaces();
         if (NumOfBdrElements == 0)
            GenerateBoundaryElements();
         CheckBdrElementOrientation();
      }
   }
   else
      NumOfEdges = 0;

   // generate the arrays 'attributes' and ' bdr_attributes'
   SetAttributes();
}

void XYZ_VectorFunction(const Vector &p, Vector &v)
{
   v = p;
}

void Mesh::SetNodalFESpace(FiniteElementSpace *nfes)
{
   GridFunction *nodes = new GridFunction(nfes);
   VectorFunctionCoefficient xyz(Dim, XYZ_VectorFunction);
   nodes->ProjectCoefficient(xyz);

   if (own_nodes) delete Nodes;
   Nodes = nodes;
   own_nodes = 1;
}

void Mesh::SetNodalGridFunction(GridFunction *nodes)
{
   if (Nodes == NULL || Nodes->FESpace() != nodes->FESpace())
   {
      VectorFunctionCoefficient xyz(Dim, XYZ_VectorFunction);
      nodes->ProjectCoefficient(xyz);
   }
   else
      *nodes = *Nodes;

   if (own_nodes) delete Nodes;
   Nodes = nodes;
   own_nodes = 0;
}

const FiniteElementSpace *Mesh::GetNodalFESpace()
{
   return ((Nodes) ? Nodes->FESpace() : NULL);
}

void Mesh::CheckElementOrientation()
{
   int i, j, k, wo = 0, *vi;
   double *v[4];

   if (Dim == 2)
   {
      DenseMatrix tri(2, 2);

      for (i = 0; i < NumOfElements; i++)
      {
         vi = elements[i]->GetVertices();
         for (j = 0; j < 3; j++)
            v[j] = vertices[vi[j]]();
         for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
               tri(j, k) = v[j+1][k] - v[0][k];
         if (tri.Det() < 0.0)
            switch (GetElementType(i))
            {
            case Element::TRIANGLE:
               k = vi[0], vi[0] = vi[1], vi[1] = k, wo++;
               break;
            case Element::QUADRILATERAL:
               k = vi[1], vi[1] = vi[3], vi[3] = k, wo++;
               break;
            }
      }
   }

   if (Dim == 3)
   {
      DenseMatrix tet(3, 3);

      for (i = 0; i < NumOfElements; i++)
      {
         vi = elements[i]->GetVertices();
         switch (GetElementType(i))
         {
         case Element::TETRAHEDRON:
            for (j = 0; j < 4; j++)
               v[j] = vertices[vi[j]]();
            for (j = 0; j < 3; j++)
               for (k = 0; k < 3; k++)
                  tet(j, k) = v[j+1][k] - v[0][k];
            if (tet.Det() < 0.0)
               k = vi[0], vi[0] = vi[1], vi[1] = k, wo++;
            break;
         case Element::HEXAHEDRON:
            // to do ...
            break;
         }
      }
   }
#if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
   if (wo > 0)
      cout << "Orientation fixed in " << wo << " of "<< NumOfElements
           << " elements" << endl;
#endif
}

int Mesh::GetTriOrientation(const int *base, const int *test)
{
   int orient;

   if (test[0] == base[0])
      if (test[1] == base[1])
         orient = 0;         //  (0, 1, 2)
      else
         orient = 5;         //  (0, 2, 1)
   else if (test[0] == base[1])
      if (test[1] == base[0])
         orient = 1;         //  (1, 0, 2)
      else
         orient = 2;         //  (1, 2, 0)
   else // test[0] == base[2]
      if (test[1] == base[0])
         orient = 4;         //  (2, 0, 1)
      else
         orient = 3;         //  (2, 1, 0)

#ifdef MFEM_DEBUG
   static const int tri_orient[6][3] = {{0, 1, 2}, {1, 0, 2},
                                        {2, 0, 1}, {2, 1, 0},
                                        {1, 2, 0}, {0, 2, 1}};
   const int *aor = tri_orient[orient];
   for (int j = 0; j < 3; j++)
      if (test[aor[j]] != base[j])
         mfem_error("Mesh::GetTriOrientation(...)");
#endif

   return orient;
}

int Mesh::GetQuadOrientation(const int *base, const int *test)
{
   int i;

   for (i = 0; i < 4; i++)
      if (test[i] == base[0])
         break;

#ifdef MFEM_DEBUG
   static const int quad_orient[8][4] = {{0, 1, 2, 3}, {0, 3, 2, 1},
                                         {1, 2, 3, 0}, {1, 0, 3, 2},
                                         {2, 3, 0, 1}, {2, 1, 0, 3},
                                         {3, 0, 1, 2}, {3, 2, 1, 0}};
   int orient;
   if (test[(i+1)%4] == base[1])
      orient = 2*i;
   else
      orient = 2*i+1;
   const int *aor = quad_orient[orient];
   for (int j = 0; j < 4; j++)
      if (test[aor[j]] != base[j])
      {
         cerr << "Mesh::GetQuadOrientation(...)" << endl;
         cerr << " base = [";
         for (int k = 0; k < 4; k++)
            cerr << " " << base[k];
         cerr << " ]\n test = [";
         for (int k = 0; k < 4; k++)
            cerr << " " << test[k];
         cerr << " ]" << endl;
         mfem_error();
      }
#endif

   if (test[(i+1)%4] == base[1])
      return 2*i;

   return 2*i+1;
}

void Mesh::CheckBdrElementOrientation()
{
   int i, j, wo = 0;

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
               j = bv[0]; bv[0] = bv[1]; bv[1] = j;
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
         { // boundary face
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
                  Swap<int>(bv[0], bv[1]);
                  wo++;
               }
            }
            break;

            case Element::HEXAHEDRON:
               switch (faces_info[be_to_face[i]].Elem1Inf/64)
               {
               case 0:
                  v[0] = ev[3]; v[1] = ev[2]; v[2] = ev[1]; v[3] = ev[0];
                  break;
               case 1:
                  v[0] = ev[0]; v[1] = ev[1]; v[2] = ev[5]; v[3] = ev[4];
                  break;
               case 2:
                  v[0] = ev[1]; v[1] = ev[2]; v[2] = ev[6]; v[3] = ev[5];
                  break;
               case 3:
                  v[0] = ev[2]; v[1] = ev[3]; v[2] = ev[7]; v[3] = ev[6];
                  break;
               case 4:
                  v[0] = ev[3]; v[1] = ev[0]; v[2] = ev[4]; v[3] = ev[7];
                  break;
               case 5:
                  v[0] = ev[4]; v[1] = ev[5]; v[2] = ev[6]; v[3] = ev[7];
                  break;
               }
               if (GetQuadOrientation(v, bv) % 2)
               {
                  j = bv[0]; bv[0] = bv[2]; bv[2] = j;
                  wo++;
               }
               break;
            }
         }
      }
   }
// #if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
#ifdef MFEM_DEBUG
   if (wo > 0)
      cout << "Orientation fixed in " << wo << " of "<< NumOfBdrElements
           << " boundary elements" << endl;
#endif
}

void Mesh::GetElementEdges(int i, Array<int> &edges, Array<int> &cor)
   const
{
   if (el_to_edge)
      el_to_edge->GetRow(i, edges);
   else
      mfem_error("Mesh::GetElementEdges(...) element to edge table "
                 "is not generated.");

   const int *v = elements[i]->GetVertices();
   const int ne = elements[i]->GetNEdges();
   cor.SetSize(ne);
   for (int j = 0; j < ne; j++)
   {
      const int *e = elements[i]->GetEdgeVertices(j);
      cor[j] = (v[e[0]] < v[e[1]]) ? (1) : (-1);
   }
}

void Mesh::GetBdrElementEdges(int i, Array<int> &edges, Array<int> &cor)
   const
{
   if (Dim == 2)
   {
      edges.SetSize(1);
      cor.SetSize(1);
      edges[0] = be_to_edge[i];
      const int *v = boundary[i]->GetVertices();
      cor[0] = (v[0] < v[1]) ? (1) : (-1);
   }
   else if (Dim == 3)
   {
      if (bel_to_edge)
         bel_to_edge->GetRow(i, edges);
      else
         mfem_error("Mesh::GetBdrElementEdges(...)");

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

void Mesh::GetFaceEdges(int i, Array<int> &edges, Array<int> &o) const
{
   if (Dim != 3)
      return;

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
   if (Dim == 2 && faces.Size() == NumOfEdges)
   {
      faces[i]->GetVertices(vert);
   }
   else
   {
      GetEdgeVertexTable(); // generate edge_vertex Table (if not generated)

      edge_vertex->GetRow(i, vert);
   }
}

Table *Mesh::GetFaceEdgeTable() const
{
   if (face_edge)
      return face_edge;

   if (Dim != 3)
      return NULL;

#ifdef MFEM_DEBUG
   if (faces.Size() != NumOfFaces)
      mfem_error("Mesh::GetFaceEdgeTable : faces were not generated!");
#endif

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);

   face_edge = new Table;
   GetElementArrayEdgeTable(faces, v_to_v, *face_edge);

   return(face_edge);
}

Table *Mesh::GetEdgeVertexTable() const
{
   if (edge_vertex)
      return edge_vertex;

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
         vert_elem->AddAColumnInRow(v[j]);
   }

   vert_elem->MakeJ();

   for (i = 0; i < NumOfElements; i++)
   {
      nv = elements[i]->GetNVertices();
      v  = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
         vert_elem->AddConnection(v[j], i);
   }

   vert_elem->ShiftUpI();

   return vert_elem;
}

void Mesh::GetElementFaces(int i, Array<int> &fcs, Array<int> &cor)
   const
{
   int n, j;

   if (el_to_face)
      el_to_face->GetRow(i, fcs);
   else
      mfem_error("Mesh::GetElementFaces(...) : el_to_face not generated.");

   n = fcs.Size();
   cor.SetSize(n);
   for (j = 0; j < n; j++)
      if (faces_info[fcs[j]].Elem1No == i)
         cor[j] = faces_info[fcs[j]].Elem1Inf % 64;
#ifdef MFEM_DEBUG
      else if (faces_info[fcs[j]].Elem2No == i)
         cor[j] = faces_info[fcs[j]].Elem2Inf % 64;
      else
         mfem_error("Mesh::GetElementFaces(...) : 2");
#else
   else
      cor[j] = faces_info[fcs[j]].Elem2Inf % 64;
#endif
}

void Mesh::GetBdrElementFace(int i, int *f, int *o) const
{
   const int *bv, *fv;

   if (State == Mesh::TWO_LEVEL_COARSE)
   {
      // the coarse level 'be_to_face' and 'faces' are destroyed
      mfem_error("Mesh::GetBdrElementFace (...)");
   }

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
   default:
      mfem_error("Mesh::GetBdrElementFace(...) 2");
   }
}

int Mesh::GetFaceBaseGeometry(int i) const
{
   // Here, we assume all faces are of the same type
   switch (GetElementType(0))
   {
   case Element::TRIANGLE:
   case Element::QUADRILATERAL:
      return Geometry::SEGMENT; // in 2D 'face' is an edge

   case Element::TETRAHEDRON:
      return Geometry::TRIANGLE;
   case Element::HEXAHEDRON:
      return Geometry::SQUARE;
   default:
      mfem_error("Mesh::GetFaceBaseGeometry(...) #1");
   }
   return(-1);
#if 0
   if (faces[i] == NULL)
      switch (GetElementType(faces_info[i].Elem1No))
      {
      case Element::TETRAHEDRON:
         return Geometry::TRIANGLE;
      case Element::HEXAHEDRON:
         return Geometry::SQUARE;
      default:
         mfem_error("Mesh::GetFaceBaseGeometry(...) #2");
      }
   else
      return faces[i]->GetGeometryType();
#endif
}

int Mesh::GetBdrElementEdgeIndex(int i) const
{
   if (Dim == 2)
      return be_to_edge[i];
   return be_to_face[i];
}

int Mesh::GetElementType(int i) const
{
   Element *El = elements[i];
   int t = El->GetType();

   while (1)
      if (t == Element::BISECTED     ||
          t == Element::QUADRISECTED ||
          t == Element::OCTASECTED)
         t = (El = ((RefinedElement *) El)->IAm())->GetType();
      else
         break;
   return t;
}

int Mesh::GetBdrElementType(int i) const
{
   Element *El = boundary[i];
   int t = El->GetType();

   while (1)
      if (t == Element::BISECTED || t == Element::QUADRISECTED)
         t = (El = ((RefinedElement *) El)->IAm())->GetType();
      else
         break;
   return t;
}

void Mesh::GetPointMatrix(int i, DenseMatrix &pointmat) const
{
   int k, j, nv;
   const int *v;

   v  = elements[i]->GetVertices();
   nv = elements[i]->GetNVertices();

   pointmat.SetSize(Dim, nv);
   for (k = 0; k < Dim; k++)
      for (j = 0; j < nv; j++)
         pointmat(k, j) = vertices[v[j]](k);
}

void Mesh::GetBdrPointMatrix(int i,DenseMatrix &pointmat) const
{
   int k, j, nv;
   const int *v;

   v  = boundary[i]->GetVertices();
   nv = boundary[i]->GetNVertices();

   pointmat.SetSize(Dim, nv);
   for (k = 0; k < Dim; k++)
      for (j = 0; j < nv; j++)
         pointmat(k, j) = vertices[v[j]](k);
}

double Mesh::GetLength(int i, int j) const
{
   const double *vi = vertices[i]();
   const double *vj = vertices[j]();
   double length = 0.;

   for (int k = 0; k < Dim; k++)
      length += (vi[k]-vj[k])*(vi[k]-vj[k]);

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
      // Initialize the indeces for the boundary elements.
      be_to_f.SetSize(NumOfBdrElements);
      for (i = 0; i < NumOfBdrElements; i++)
      {
         const int *v = boundary[i]->GetVertices();
         be_to_f[i] = v_to_v(v[0], v[1]);
      }
   }
   else if (Dim == 3)
   {
      if (bel_to_edge == NULL)
         bel_to_edge = new Table;
      GetElementArrayEdgeTable(boundary, v_to_v, *bel_to_edge);
   }
   else
      mfem_error("1D GetElementToEdgeTable is not yet implemented.");

   // Return the number of edges
   return NumberOfEdges;
}

const Table & Mesh::ElementToElementTable()
{
   if (el_to_el)
      return *el_to_el;

   if (Dim == 2)
   {
      Table edge_el;

      Transpose(ElementToEdgeTable(), edge_el);
      el_to_el = new Table(NumOfElements, 4); // 4 is the max. # of edges

      for (int i = 0; i < edge_el.Size(); i++)
         if (edge_el.RowSize(i) > 1)
         {
            const int *el = edge_el.GetRow(i);
            el_to_el->Push(el[0], el[1]);
            el_to_el->Push(el[1], el[0]);
         }

      el_to_el->Finalize();
   }
   else if (Dim == 3)
   {
      el_to_el = new Table(NumOfElements, 6); // 6 is the max. # of faces

#ifdef MFEM_DEBUG
      if (faces_info.Size() != NumOfFaces)
         mfem_error("Mesh::ElementToElementTable : faces were not generated!");
#endif

      for (int i = 0; i < faces_info.Size(); i++)
         if (faces_info[i].Elem2No >= 0)
         {
            el_to_el->Push(faces_info[i].Elem1No, faces_info[i].Elem2No);
            el_to_el->Push(faces_info[i].Elem2No, faces_info[i].Elem1No);
         }

      el_to_el->Finalize();
   }
   else
      mfem_error("Mesh::ElementToElementTable() in 1D is not implemented!");

   return *el_to_el;
}

const Table & Mesh::ElementToFaceTable() const
{
   if (el_to_face == NULL)
      mfem_error("Mesh::ElementToFaceTable()");
   return *el_to_face;
}

const Table & Mesh::ElementToEdgeTable() const
{
   if (el_to_edge == NULL)
      mfem_error("Mesh::ElementToEdgeTable()");
   return *el_to_edge;
}

void Mesh::AddSegmentFaceElement(int lf, int gf, int el, int v0, int v1)
{
   if (faces[gf] == NULL)  // this will be elem1
   {
      faces[gf] = new Segment(v0, v1);
      faces_info[gf].Elem1No  = el;
      faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
      faces_info[gf].Elem2No  = -1; // in case there's no other side
   }
   else  //  this will be elem2
   {
#ifdef MFEM_DEBUG
      int *v = faces[gf]->GetVertices();
      if (v[1] != v0 || v[0] != v1)
         mfem_error("Mesh::AddSegmentFaceElement(...)");
#endif
      faces_info[gf].Elem2No  = el;
      faces_info[gf].Elem2Inf = 64 * lf + 1;
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
   }
   else  //  this will be elem2
   {
      int orientation, vv[3] = { v0, v1, v2 };
      orientation = GetTriOrientation(faces[gf]->GetVertices(), vv);
#ifdef MFEM_DEBUG
      if (orientation % 2 == 0)
         mfem_error("Mesh::AddTriangleFaceElement(...)");
#endif
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
   }
   else  //  this will be elem2
   {
      int vv[4] = { v0, v1, v2, v3 };
      int oo = GetQuadOrientation(faces[gf]->GetVertices(), vv);
#ifdef MFEM_DEBUG
      if (oo % 2 == 0)
         mfem_error("Mesh::AddQuadFaceElement(...)");
#endif
      faces_info[gf].Elem2No  = el;
      faces_info[gf].Elem2Inf = 64 * lf + oo;
   }
}

void Mesh::GenerateFaces()
{
   int i, nfaces;

   nfaces = (Dim == 2) ? NumOfEdges : NumOfFaces;

   for (i = 0; i < faces.Size(); i++)
      FreeElement(faces[i]);

   // (re)generate the interior faces and the info for them
   faces.SetSize(nfaces);
   faces_info.SetSize(nfaces);
   for (i = 0; i < nfaces; i++)
   {
      faces[i] = NULL;
      faces_info[i].Elem1No = -1;
   }
   for (i = 0; i < NumOfElements; i++)
   {
      const int *v = elements[i]->GetVertices();
      const int *ef;
      if (Dim == 2)
      {
         ef = el_to_edge->GetRow(i);
         const int ne = elements[i]->GetNEdges();
         for (int j = 0; j < ne; j++)
         {
            const int *e = elements[i]->GetEdgeVertices(j);
            AddSegmentFaceElement(j, ef[j], i, v[e[0]], v[e[1]]);
         }
      }
      else
      {
         ef = el_to_face->GetRow(i);
         switch (GetElementType(i))
         {
         case Element::TETRAHEDRON:
            AddTriangleFaceElement(0, ef[0], i, v[1], v[2], v[3]);
            AddTriangleFaceElement(1, ef[1], i, v[0], v[3], v[2]);
            AddTriangleFaceElement(2, ef[2], i, v[0], v[1], v[3]);
            AddTriangleFaceElement(3, ef[3], i, v[0], v[2], v[1]);
            break;
         case Element::HEXAHEDRON:
            AddQuadFaceElement(0, ef[0], i, v[3], v[2], v[1], v[0]);
            AddQuadFaceElement(1, ef[1], i, v[0], v[1], v[5], v[4]);
            AddQuadFaceElement(2, ef[2], i, v[1], v[2], v[6], v[5]);
            AddQuadFaceElement(3, ef[3], i, v[2], v[3], v[7], v[6]);
            AddQuadFaceElement(4, ef[4], i, v[3], v[0], v[4], v[7]);
            AddQuadFaceElement(5, ef[5], i, v[4], v[5], v[6], v[7]);
            break;
         }
      }
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
         faces_tbl->Push(v[1], v[2], v[3]);
         faces_tbl->Push(v[0], v[3], v[2]);
         faces_tbl->Push(v[0], v[1], v[3]);
         faces_tbl->Push(v[0], v[2], v[1]);
         break;
      case Element::HEXAHEDRON:
         // find the face by the vertices with the smallest 3 numbers
         // z = 0, y = 0, x = 1, y = 1, x = 0, z = 1
         faces_tbl->Push4(v[3], v[2], v[1], v[0]);
         faces_tbl->Push4(v[0], v[1], v[5], v[4]);
         faces_tbl->Push4(v[1], v[2], v[6], v[5]);
         faces_tbl->Push4(v[2], v[3], v[7], v[6]);
         faces_tbl->Push4(v[3], v[0], v[4], v[7]);
         faces_tbl->Push4(v[4], v[5], v[6], v[7]);
         break;
      }
   }
   return faces_tbl;
}

STable3D *Mesh::GetElementToFaceTable(int ret_ftbl)
{
   int i, *v;
   STable3D *faces_tbl;

   if (el_to_face != NULL)
      delete el_to_face;
   el_to_face = new Table(NumOfElements, 6);  // must be 6 for hexahedra
   faces_tbl = new STable3D(NumOfVertices);
   for (i = 0; i < NumOfElements; i++)
   {
      v = elements[i]->GetVertices();
      switch (GetElementType(i))
      {
      case Element::TETRAHEDRON:
         el_to_face->Push(i, faces_tbl->Push(v[1], v[2], v[3]));
         el_to_face->Push(i, faces_tbl->Push(v[0], v[3], v[2]));
         el_to_face->Push(i, faces_tbl->Push(v[0], v[1], v[3]));
         el_to_face->Push(i, faces_tbl->Push(v[0], v[2], v[1]));
         break;
      case Element::HEXAHEDRON:
         // find the face by the vertices with the smallest 3 numbers
         // z = 0, y = 0, x = 1, y = 1, x = 0, z = 1
         el_to_face->Push(i, faces_tbl->Push4(v[3], v[2], v[1], v[0]));
         el_to_face->Push(i, faces_tbl->Push4(v[0], v[1], v[5], v[4]));
         el_to_face->Push(i, faces_tbl->Push4(v[1], v[2], v[6], v[5]));
         el_to_face->Push(i, faces_tbl->Push4(v[2], v[3], v[7], v[6]));
         el_to_face->Push(i, faces_tbl->Push4(v[3], v[0], v[4], v[7]));
         el_to_face->Push(i, faces_tbl->Push4(v[4], v[5], v[6], v[7]));
         break;
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
         be_to_face[i] = (*faces_tbl)(v[0], v[1], v[2]);
         break;
      case Element::QUADRILATERAL:
         be_to_face[i] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
         break;
      }
   }

   if (ret_ftbl)
      return faces_tbl;
   delete faces_tbl;
   return NULL;
}

void Mesh::ReorientTetMesh()
{
   int *v;

   if (Dim != 3 || !(meshgen & 1))
      return;

   DeleteCoarseTables();

   for (int i = 0; i < NumOfElements; i++)
      if (GetElementType(i) == Element::TETRAHEDRON)
      {
         v = elements[i]->GetVertices();

         Rotate3(v[0], v[1], v[2]);
         if (v[0] < v[3])
            Rotate3(v[1], v[2], v[3]);
         else
            ShiftL2R(v[0], v[1], v[3]);
      }

   for (int i = 0; i < NumOfBdrElements; i++)
      if (GetBdrElementType(i) == Element::TRIANGLE)
      {
         v = boundary[i]->GetVertices();

         Rotate3(v[0], v[1], v[2]);
      }

   GetElementToFaceTable();
   GenerateFaces();
   if (el_to_edge)
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
}

#ifdef MFEM_USE_MPI
// auxiliary function for qsort
static int mfem_less(const void *x, const void *y)
{
   if (*(int*)x < *(int*)y)
      return 1;
   if (*(int*)x > *(int*)y)
      return -1;
   return 0;
}
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
// METIS 5 prototypes
typedef int idx_t;
typedef float real_t;
extern "C" {
   int METIS_PartGraphRecursive(
      idx_t *nvtxs, idx_t *ncon, idx_t *xadj,
      idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,
      idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,
      idx_t *edgecut, idx_t *part);
   int METIS_PartGraphKway(
      idx_t *nvtxs, idx_t *ncon, idx_t *xadj,
      idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,
      idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,
      idx_t *edgecut, idx_t *part);
   int METIS_SetDefaultOptions(idx_t *options);
}
#endif
#endif

int *Mesh::CartesianPartitioning(int nxyz[])
{
   int *partitioning;
   double pmin[3], pmax[3];
   for (int i = 0; i < Dim; i++)
   {
      pmin[i] = numeric_limits<double>::infinity();
      pmax[i] = -pmin[i];
   }
   // find a bounding box using the vertices
   for (int vi = 0; vi < NumOfVertices; vi++)
   {
      const double *p = vertices[vi]();
      for (int i = 0; i < Dim; i++)
      {
         if (p[i] < pmin[i]) pmin[i] = p[i];
         if (p[i] > pmax[i]) pmax[i] = p[i];
      }
   }

   partitioning = new int[NumOfElements];

   // determine the partitioning using the centers of the elements
   double ppt[3];
   Vector pt(ppt, Dim);
   for (int el = 0; el < NumOfElements; el++)
   {
      GetElementTransformation(el)->Transform(
         Geometries.GetCenter(GetElementBaseGeometry(el)), pt);
      int part = 0;
      for (int i = Dim-1; i >= 0; i--)
      {
         int idx = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
         if (idx < 0) idx = 0;
         if (idx >= nxyz[i]) idx = nxyz[i]-1;
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
         partitioning[i] = 0;
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
      options[10] = 1; // set METIS_OPTION_CONTIG
#endif

      // Sort the neighbor lists
      if (part_method >= 0 && part_method <= 2)
         for (i = 0; i < n; i++)
            qsort(&J[I[i]], I[i+1]-I[i], sizeof(int), &mfem_less);

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
         options[1] = 1; // set METIS_OPTION_OBJTYPE to METIS_OBJTYPE_VOL
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
      delete el_to_el;
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
         psize[partitioning[i]].one++;

      int empty_parts = 0;
      for (i = 0; i < nparts; i++)
         if (psize[i].one == 0)
            empty_parts++;

      // This code just split the largest partitionings in two.
      // Do we need to replace it with something better?
      if (empty_parts)
      {
         cerr << "Mesh::GeneratePartitioning returned " << empty_parts
              << " empty parts!" << endl;

         SortPairs<int,int>(psize, nparts);

         for (i = nparts-1; i > nparts-1-empty_parts; i--)
            psize[i].one /= 2;

         for (int j = 0; j < NumOfElements; j++)
            for (i = nparts-1; i > nparts-1-empty_parts; i--)
               if (psize[i].one == 0 || partitioning[j] != psize[i].two)
                  continue;
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
         num_part = partitioning[i];
      component[i] = -1;
   }
   num_part++;

   num_comp.SetSize(num_part);
   for (i = 0; i < num_part; i++)
      num_comp[i] = 0;

   stack_p = 0;
   stack_top_p = 0;  // points to the first unused element in the stack
   for (elem = 0; elem < num_elem; elem++)
   {
      if (component[elem] >= 0)
         continue;

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
         n_empty++;
      else if (num_comp[i] > 1)
         n_mcomp++;

   if (n_empty > 0)
   {
      cout << "Mesh::CheckPartitioning(...) :\n"
           << "The following subdomains are empty :\n";
      for (i = 0; i < num_comp.Size(); i++)
         if (num_comp[i] == 0)
            cout << ' ' << i;
      cout << endl;
   }
   if (n_mcomp > 0)
   {
      cout << "Mesh::CheckPartitioning(...) :\n"
           << "The following subdomains are NOT connected :\n";
      for (i = 0; i < num_comp.Size(); i++)
         if (num_comp[i] > 1)
            cout << ' ' << i;
      cout << endl;
   }
   if (n_empty == 0 && n_mcomp == 0)
      cout << "Mesh::CheckPartitioning(...) : "
         "All subdomains are connected." << endl;

   if (el_to_el)
      delete el_to_el;
   el_to_el = NULL;
}

// compute the coefficients of the polynomial in t:
//   c(0)+c(1)*t+...+c(d)*t^d = det(A+t*B)
// where A, B are (d x d), d=2,3
void DetOfLinComb(const DenseMatrix &A, const DenseMatrix &B, Vector &c)
{
   const double *a = A.Data();
   const double *b = B.Data();

   c.SetSize(A.Size()+1);
   switch (A.Size())
   {
   case 2:
   {
      // det(A+t*B) = |a0 a2|   / |a0 b2| + |b0 a2| \
      //              |a1 a3| + \ |a1 b3|   |b1 a3| / * t +
      //              |b0 b2|
      //              |b1 b3| * t^2
      c(0) = a[0]*a[3]-a[1]*a[2];
      c(1) = a[0]*b[3]-a[1]*b[2]+b[0]*a[3]-b[1]*a[2];
      c(2) = b[0]*b[3]-b[1]*b[2];
   }
   break;

   case 3:
   {
      //              |a0 a3 a6|
      // det(A+t*B) = |a1 a4 a7| +
      //              |a2 a5 a8|

      //  /  |b0 a3 a6|   |a0 b3 a6|   |a0 a3 b6| \
      //  |  |b1 a4 a7| + |a1 b4 a7| + |a1 a4 b7| | * t +
      //  \  |b2 a5 a8|   |a2 b5 a8|   |a2 a5 b8| /

      //  /  |a0 b3 b6|   |b0 a3 b6|   |b0 b3 a6| \
      //  |  |a1 b4 b7| + |b1 a4 b7| + |b1 b4 a7| | * t^2 +
      //  \  |a2 b5 b8|   |b2 a5 b8|   |b2 b5 a8| /

      //  |b0 b3 b6|
      //  |b1 b4 b7| * t^3
      //  |b2 b5 b8|
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
      mfem_error("FindRoots(...)");

   while (z(d) == 0.0)
   {
      if (d == 0)
         return(-1);
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
            t = -0.5 * (b + sqrt(D));
         else
            t = -0.5 * (b - sqrt(D));
         x(0) = t / a;
         x(1) = c / t;
         if (x(0) > x(1))
            Swap<double>(x(0), x(1));
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
            Swap<double>(x0, x1);
         if (x1 > x2)
         {
            Swap<double>(x1, x2);
            if (x0 > x1)
               Swap<double>(x0, x1);
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
            A = -pow(sqrt(R2 - Q3) + R, 1.0/3.0);
         else
            A =  pow(sqrt(R2 - Q3) - R, 1.0/3.0);
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
         break;
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
         break;
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
   DenseMatrix P, V, DS, PDS(Dim), VDS(Dim);
   Vector c(Dim+1), x(Dim);
   const double factor = 2.0;

   // check for tangling assuming constant speed
   if (tmax < 1.0)
      tmax = 1.0;
   for (int i = 0; i < NumOfElements; i++)
   {
      Element *el = elements[i];
      int nv = el->GetNVertices();
      int *v = el->GetVertices();
      P.SetSize(Dim, nv);
      V.SetSize(Dim, nv);
      for (int j = 0; j < Dim; j++)
         for (int k = 0; k < nv; k++)
         {
            P(j, k) = vertices[v[k]](j);
            V(j, k) = displacements(v[k]+j*nvs);
         }
      DS.SetSize(nv, Dim);
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
            tmax = 0.0;
         else
            FindTMax(c, x, tmax, factor, Dim);
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
               tmax = 0.0;
            else
               FindTMax(c, x, tmax, factor, Dim);
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
      for (int j = 0; j < Dim; j++)
         vertices[i](j) += displacements(j*nv+i);
}

void Mesh::GetVertices(Vector &vert_coord) const
{
   int nv = vertices.Size();
   vert_coord.SetSize(nv*Dim);
   for (int i = 0; i < nv; i++)
      for (int j = 0; j < Dim; j++)
         vert_coord(j*nv+i) = vertices[i](j);
}

void Mesh::SetVertices(const Vector &vert_coord)
{
   for (int i = 0, nv = vertices.Size(); i < nv; i++)
      for (int j = 0; j < Dim; j++)
         vertices[i](j) = vert_coord(j*nv+i);
}

void Mesh::MoveNodes(const Vector &displacements)
{
   if (Nodes)
      (*Nodes) += displacements;
   else
      MoveVertices(displacements);
}

void Mesh::GetNodes(Vector &node_coord) const
{
   if (Nodes)
      node_coord = (*Nodes);
   else
      GetVertices(node_coord);
}

void Mesh::SetNodes(const Vector &node_coord)
{
   if (Nodes)
      (*Nodes) = node_coord;
   else
      SetVertices(node_coord);
}

void Mesh::NewNodes(GridFunction &nodes)
{
   if (own_nodes) delete Nodes;
   Nodes = &nodes;
   own_nodes = 0;
}

void Mesh::AverageVertices(int * indexes, int n, int result)
{
   int j, k;

   for (k = 0; k < Dim; k++)
      vertices[result](k) = vertices[indexes[0]](k);

   for (j = 1; j < n; j++)
      for (k = 0; k < Dim; k++)
         vertices[result](k) += vertices[indexes[j]](k);

   for (k = 0; k < Dim; k++)
      vertices[result](k) *= (1.0 / n);
}

void Mesh::UpdateNodes()
{
   FiniteElementSpace *cfes = Nodes->FESpace()->SaveUpdate();
   SparseMatrix *R =
      Nodes->FESpace()->GlobalRestrictionMatrix(cfes, 0);
   delete cfes;
   {
      Vector cNodes = *Nodes;
      Nodes->Update();
      R->MultTranspose(cNodes, *Nodes);
   }
   delete R;

   SetState(Mesh::TWO_LEVEL_FINE);

   // update the vertices?
}

void Mesh::QuadUniformRefinement()
{
   int i, j, *v, vv[2], attr, wtls = WantTwoLevelState;
   const int *e;

   if (Nodes)  // curved mesh
   {
      UseTwoLevelState(1);
   }

   SetState(Mesh::NORMAL);

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }

   int oedge = NumOfVertices;
   int oelem = oedge + NumOfEdges;

   DeleteCoarseTables();

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

      if (WantTwoLevelState)
      {
         QuadrisectedElement *qe;

         qe = new QuadrisectedElement(elements[i]->Duplicate(this));
         qe->FirstChild = elements[i];
         qe->Child2 = j;
         qe->Child3 = j+1;
         qe->Child4 = j+2;
         elements[i] = qe;
      }

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

      if (WantTwoLevelState)
      {
#ifdef MFEM_USE_MEMALLOC
         BisectedElement *be = BEMemory.Alloc();
#else
         BisectedElement *be = new BisectedElement;
#endif
         be->SetCoarseElem(boundary[i]->Duplicate(this));
         be->FirstChild = boundary[i];
         be->SecondChild = j;
         boundary[i] = be;
      }

      v[1] = oedge+be_to_edge[i];
   }

   if (WantTwoLevelState)
   {
      c_NumOfVertices    = NumOfVertices;
      c_NumOfEdges       = NumOfEdges;
      c_NumOfElements    = NumOfElements;
      c_NumOfBdrElements = NumOfBdrElements;

      RefinedElement::State = RefinedElement::FINE;
      State = Mesh::TWO_LEVEL_FINE;
   }

   NumOfVertices    = oelem + NumOfElements;
   NumOfElements    = 4 * NumOfElements;
   NumOfBdrElements = 2 * NumOfBdrElements;
   NumOfFaces       = 0;

   if (WantTwoLevelState)
   {
      f_NumOfVertices    = NumOfVertices;
      f_NumOfElements    = NumOfElements;
      f_NumOfBdrElements = NumOfBdrElements;
   }

   if (el_to_edge != NULL)
   {
      if (WantTwoLevelState)
      {
         c_el_to_edge = el_to_edge;
         Swap(be_to_edge, fc_be_to_edge); // save coarse be_to_edge
         f_el_to_edge = new Table;
         NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
         el_to_edge = f_el_to_edge;
         f_NumOfEdges = NumOfEdges;
      }
      else
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      GenerateFaces();
   }

#ifdef MFEM_DEBUG
   CheckElementOrientation();
   CheckBdrElementOrientation();
#endif

   if (Nodes)  // curved mesh
   {
      UpdateNodes();
      UseTwoLevelState(wtls);
   }
}

void Mesh::HexUniformRefinement()
{
   int i, wtls = WantTwoLevelState;
   int * v;
   const int *e, *f;
   int vv[4];

   if (Nodes)  // curved mesh
   {
      UseTwoLevelState(1);
   }

   SetState(Mesh::NORMAL);

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   if (el_to_face == NULL)
      GetElementToFaceTable();

   int oedge = NumOfVertices;
   int oface = oedge + NumOfEdges;
   int oelem = oface + NumOfFaces;

   DeleteCoarseTables();

   vertices.SetSize(oelem + NumOfElements);
   for (i = 0; i < NumOfElements; i++)
   {
      v = elements[i]->GetVertices();

      AverageVertices(v, 8, oelem+i);

      f = el_to_face->GetRow(i);

      vv[0] = v[3], vv[1] = v[2], vv[2] = v[1], vv[3] = v[0];
      AverageVertices(vv, 4, oface+f[0]);
      vv[0] = v[0], vv[1] = v[1], vv[2] = v[5], vv[3] = v[4];
      AverageVertices(vv, 4, oface+f[1]);
      vv[0] = v[1], vv[1] = v[2], vv[2] = v[6], vv[3] = v[5];
      AverageVertices(vv, 4, oface+f[2]);
      vv[0] = v[2], vv[1] = v[3], vv[2] = v[7], vv[3] = v[6];
      AverageVertices(vv, 4, oface+f[3]);
      vv[0] = v[3], vv[1] = v[0], vv[2] = v[4], vv[3] = v[7];
      AverageVertices(vv, 4, oface+f[4]);
      vv[0] = v[4], vv[1] = v[5], vv[2] = v[6], vv[3] = v[7];
      AverageVertices(vv, 4, oface+f[5]);

      e = el_to_edge->GetRow(i);

      vv[0] = v[0], vv[1] = v[1]; AverageVertices(vv, 2, oedge+e[0]);
      vv[0] = v[1], vv[1] = v[2]; AverageVertices(vv, 2, oedge+e[1]);
      vv[0] = v[2], vv[1] = v[3]; AverageVertices(vv, 2, oedge+e[2]);
      vv[0] = v[3], vv[1] = v[0]; AverageVertices(vv, 2, oedge+e[3]);
      vv[0] = v[4], vv[1] = v[5]; AverageVertices(vv, 2, oedge+e[4]);
      vv[0] = v[5], vv[1] = v[6]; AverageVertices(vv, 2, oedge+e[5]);
      vv[0] = v[6], vv[1] = v[7]; AverageVertices(vv, 2, oedge+e[6]);
      vv[0] = v[7], vv[1] = v[4]; AverageVertices(vv, 2, oedge+e[7]);
      vv[0] = v[0], vv[1] = v[4]; AverageVertices(vv, 2, oedge+e[8]);
      vv[0] = v[1], vv[1] = v[5]; AverageVertices(vv, 2, oedge+e[9]);
      vv[0] = v[2], vv[1] = v[6]; AverageVertices(vv, 2, oedge+e[10]);
      vv[0] = v[3], vv[1] = v[7]; AverageVertices(vv, 2, oedge+e[11]);
   }

   int attr, j, k;
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

      if (WantTwoLevelState)
      {
         OctasectedElement *oe;

         oe = new OctasectedElement(elements[i]->Duplicate(this));
         oe->FirstChild = elements[i];
         for (k = 0; k < 7; k++)
            oe->Child[k] = j + k;
         elements[i] = oe;
      }

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

      if (WantTwoLevelState)
      {
         QuadrisectedElement *qe;

         qe = new QuadrisectedElement(boundary[i]->Duplicate(this));
         qe->FirstChild = boundary[i];
         qe->Child2 = j;
         qe->Child3 = j+1;
         qe->Child4 = j+2;
         boundary[i] = qe;
      }

      v[1] = oedge+e[0];
      v[2] = oface+f[0];
      v[3] = oedge+e[3];
   }

   if (WantTwoLevelState)
   {
      c_NumOfVertices    = NumOfVertices;
      c_NumOfEdges       = NumOfEdges;
      c_NumOfFaces       = NumOfFaces;
      c_NumOfElements    = NumOfElements;
      c_NumOfBdrElements = NumOfBdrElements;

      RefinedElement::State = RefinedElement::FINE;
      State = Mesh::TWO_LEVEL_FINE;
   }

   NumOfVertices    = oelem + NumOfElements;
   NumOfElements    = 8 * NumOfElements;
   NumOfBdrElements = 4 * NumOfBdrElements;

   if (WantTwoLevelState)
   {
      f_NumOfVertices    = NumOfVertices;
      f_NumOfElements    = NumOfElements;
      f_NumOfBdrElements = NumOfBdrElements;
   }

   if (el_to_edge != NULL)
   {
      if (WantTwoLevelState)
      {
         c_el_to_edge = el_to_edge;
         f_el_to_edge = new Table;
         c_bel_to_edge = bel_to_edge;
         bel_to_edge = NULL;
         NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
         el_to_edge = f_el_to_edge;
         f_bel_to_edge = bel_to_edge;
         f_NumOfEdges = NumOfEdges;
      }
      else
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   if (el_to_face != NULL)
   {
      if (WantTwoLevelState)
      {
         c_el_to_face = el_to_face;
         el_to_face = NULL;
         Swap(faces_info, fc_faces_info);
      }
      GetElementToFaceTable();
      GenerateFaces();
      if (WantTwoLevelState)
      {
         f_el_to_face = el_to_face;
         f_NumOfFaces = NumOfFaces;
      }
   }

#ifdef MFEM_DEBUG
   CheckBdrElementOrientation();
#endif

   if (Nodes)  // curved mesh
   {
      UpdateNodes();
      UseTwoLevelState(wtls);
   }

   //  When 'WantTwoLevelState' is true the coarse level
   //  'be_to_face' and 'faces'
   //  are destroyed !!!
}

void Mesh::LocalRefinement(const Array<int> &marked_el, int type)
{
   int i, j, ind, nedges, wtls = WantTwoLevelState;
   Array<int> v;

   if (Nodes)  // curved mesh
   {
      UseTwoLevelState(1);
   }

   SetState(Mesh::NORMAL);
   DeleteCoarseTables();

   if (Dim == 1) // --------------------------------------------------------
   {
      int cne = NumOfElements, cnv = NumOfVertices;
      NumOfVertices += marked_el.Size();
      NumOfElements += marked_el.Size();
      vertices.SetSize(NumOfVertices);
      elements.SetSize(NumOfElements);
      for (j = 0; j < marked_el.Size(); j++)
      {
         i = marked_el[j];
         int *vert = elements[i]->GetVertices();
         vertices[cnv+j](0) = 0.5 * ( vertices[vert[0]](0) +
                                      vertices[vert[1]](0) );
         elements[cne+j] = new Segment(cnv+j, vert[1],
                                       elements[i]->GetAttribute());
         vert[1] = cnv+j;
      }
   } // end of 'if (Dim == 1)'
   else if (Dim == 2) // ---------------------------------------------------
   {
      if (WantTwoLevelState)
      {
         c_NumOfVertices    = NumOfVertices;
         c_NumOfEdges       = NumOfEdges;
         c_NumOfElements    = NumOfElements;
         c_NumOfBdrElements = NumOfBdrElements;
      }

      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      nedges = v_to_v.NumberOfEntries();
      int *edge1  = new int[nedges];
      int *edge2  = new int[nedges];
      int *middle = new int[nedges];

      for (i = 0; i < nedges; i++)
         edge1[i] = edge2[i] = middle[i] = -1;

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
         RedRefinement(marked_el[i], v_to_v, edge1, edge2, middle);

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      do
      {
         need_refinement = 0;
         for (i = 0; i < nedges; i++)
            if (middle[i] != -1 && edge1[i] != -1)
            {
               need_refinement = 1;
               GreenRefinement(edge1[i], v_to_v, edge1, edge2, middle);
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

               if (WantTwoLevelState)
               {
                  boundary.Append(new Segment(v2, boundary[i]->GetAttribute()));
#ifdef MFEM_USE_MEMALLOC
                  BisectedElement *aux = BEMemory.Alloc();
                  aux->SetCoarseElem(boundary[i]);
#else
                  BisectedElement *aux = new BisectedElement(boundary[i]);
#endif
                  aux->FirstChild =
                     new Segment(v1, boundary[i]->GetAttribute());
                  aux->SecondChild = NumOfBdrElements;
                  boundary[i] = aux;
                  NumOfBdrElements++;
               }
               else
               {
                  boundary[i]->SetVertices(v1);
                  boundary.Append(new Segment(v2, boundary[i]->GetAttribute()));
               }
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

#ifdef MFEM_DEBUG
      CheckElementOrientation();
#endif

      if (WantTwoLevelState)
      {
         f_NumOfVertices    = NumOfVertices;
         f_NumOfElements    = NumOfElements;
         f_NumOfBdrElements = NumOfBdrElements;
         RefinedElement::State = RefinedElement::FINE;
         State = Mesh::TWO_LEVEL_FINE;
      }

      if (el_to_edge != NULL)
      {
         if (WantTwoLevelState)
         {
            c_el_to_edge = el_to_edge;
            Swap(be_to_edge, fc_be_to_edge); // save coarse be_to_edge
            f_el_to_edge = new Table;
            NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
            el_to_edge = f_el_to_edge;
            f_NumOfEdges = NumOfEdges;
         }
         else
            NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         GenerateFaces();
      }

   }
   else if (Dim == 3) // ---------------------------------------------------
   {
      if (WantTwoLevelState)
      {
         c_NumOfVertices    = NumOfVertices;
         c_NumOfEdges       = NumOfEdges;
         c_NumOfFaces       = NumOfFaces;
         c_NumOfElements    = NumOfElements;
         c_NumOfBdrElements = NumOfBdrElements;
      }

      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      nedges = v_to_v.NumberOfEntries();
      int *middle = new int[nedges];

      for (i = 0; i < nedges; i++)
         middle[i] = -1;

      // 3. Do the red refinement.
      int ii;
      switch (type)
      {
      case 1:
         for (i = 0; i < marked_el.Size(); i++)
            Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
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

      if (WantTwoLevelState)
      {
         RefinedElement::State = RefinedElement::FINE;
         State = Mesh::TWO_LEVEL_FINE;
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
         Element *El = elements[i];
         while (El->GetType() == Element::BISECTED)
            El = ((BisectedElement *) El)->FirstChild;
         ((Tetrahedron *)El)->ParseRefinementFlag(refinement_edges, type,
                                                  flag);
         if (type == Tetrahedron::TYPE_PF)
            ((Tetrahedron *)El)->CreateRefinementFlag(refinement_edges,
                                                      Tetrahedron::TYPE_PU,
                                                      flag);
      }

      NumOfBdrElements = boundary.Size();

      // 7. Free the allocated memory.
      delete [] middle;

#ifdef MFEM_DEBUG
      CheckElementOrientation();
#endif

      if (el_to_edge != NULL)
      {
         if (WantTwoLevelState)
         {
            c_el_to_edge = el_to_edge;
            f_el_to_edge = new Table;
            c_bel_to_edge = bel_to_edge;
            bel_to_edge = NULL;
            NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
            el_to_edge = f_el_to_edge;
            f_bel_to_edge = bel_to_edge;
         }
         else
            NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }
      if (el_to_face != NULL)
      {
         if (WantTwoLevelState)
         {
            c_el_to_face = el_to_face;
            el_to_face = NULL;
            Swap(faces_info, fc_faces_info);
         }
         GetElementToFaceTable();
         GenerateFaces();
         if (WantTwoLevelState)
         {
            f_el_to_face = el_to_face;
         }
      }

      if (WantTwoLevelState)
      {
         f_NumOfVertices    = NumOfVertices;
         f_NumOfEdges       = NumOfEdges;
         f_NumOfFaces       = NumOfFaces;
         f_NumOfElements    = NumOfElements;
         f_NumOfBdrElements = NumOfBdrElements;
      }

   } //  end 'if (Dim == 3)'

   if (Nodes)  // curved mesh
   {
      UpdateNodes();
      UseTwoLevelState(wtls);
   }
}

void Mesh::UniformRefinement()
{
   if (NURBSext)
      NURBSUniformRefinement();
   else if (meshgen == 1)
   {
      Array<int> elem_to_refine(GetNE());

      for (int i = 0; i < elem_to_refine.Size(); i++)
         elem_to_refine[i] = i;
      LocalRefinement(elem_to_refine);
   }
   else if (Dim == 2)
      QuadUniformRefinement();
   else if (Dim == 3)
      HexUniformRefinement();
   else
      mfem_error("Mesh::UniformRefinement()");
}

void Mesh::Bisection(int i, const DSTable &v_to_v,
                     int *edge1, int *edge2, int *middle)
{
   int *vert;
   int v[2][4], v_new, bisect, t;
   Element **pce;
   Vertex V;

   if (WantTwoLevelState)
   {
      pce = &(elements[i]);
      while (1)
      {
         t = pce[0]->GetType();
         if (t == Element::BISECTED)
            pce = & ( ((BisectedElement *) pce[0])->FirstChild );
         else if (t == Element::QUADRISECTED)
            pce = & ( ((QuadrisectedElement *) pce[0])->FirstChild );
         else
            break;
      }
   }
   else
      t = elements[i]->GetType();


   if (t == Element::TRIANGLE)
   {
      Triangle *tri;

      if (WantTwoLevelState)
         tri = (Triangle *) pce[0];
      else
         tri = (Triangle *) elements[i];

      vert = tri->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v(vert[0], vert[1]);
#ifdef MFEM_DEBUG
      if (bisect < 0)
         mfem_error("Mesh::Bisection(...): ERROR");
#endif
      if (middle[bisect] == -1)
      {
         v_new = NumOfVertices++;
         V(0) = 0.5 * (vertices[vert[0]](0) + vertices[vert[1]](0));
         V(1) = 0.5 * (vertices[vert[0]](1) + vertices[vert[1]](1));
         V(2) = 0.0;
         vertices.Append(V);

         // Put the element that may need refinement (because of this
         // bisection) in edge1, or -1 if no more refinement is needed.
         if (edge1[bisect] == i)
            edge1[bisect] = edge2[bisect];

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

      if (WantTwoLevelState)
      {
#ifdef MFEM_USE_MEMALLOC
         BisectedElement *aux = BEMemory.Alloc();
         aux->SetCoarseElem(tri);
#else
         BisectedElement *aux = new BisectedElement(tri);
#endif
         aux->FirstChild = tri = new Triangle(v[0], tri->GetAttribute());
         aux->SecondChild = NumOfElements;
         pce[0] = aux;
      }
      else
         tri->SetVertices(v[0]); // changes vert[0..2] !!!
      elements.Append(new Triangle(v[1], tri->GetAttribute()));

      // 3. edge1 and edge2 may have to be changed for the second triangle.
      if (v[1][0] < v_to_v.NumberOfRows() && v[1][1] < v_to_v.NumberOfRows())
      {
         bisect = v_to_v(v[1][0], v[1][1]);
#ifdef MFEM_DEBUG
         if (bisect < 0)
            mfem_error("Mesh::Bisection(...): ERROR 2");
#endif
         if (edge1[bisect] == i)
            edge1[bisect] = NumOfElements;
         else if (edge2[bisect] == i)
            edge2[bisect] = NumOfElements;
      }
      NumOfElements++;
   }
   else if (t == Element::TETRAHEDRON)
   {
      int j, type, new_type, old_redges[2], new_redges[2][2], flag;
      Tetrahedron *tet;

      if (WantTwoLevelState)
         tet = (Tetrahedron *) pce[0];
      else
         tet = (Tetrahedron *) elements[i];

      if (tet->GetRefinementFlag() == 0)
         mfem_error("Mesh::Bisection : TETRAHEDRON element is not marked for "
                    "refinement.");

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
            V(j) = 0.5 * (vertices[vert[0]](j) + vertices[vert[1]](j));
         vertices.Append(V);

         middle[bisect] = v_new;
      }
      else
         v_new = middle[bisect];

      // 2. Set the node indices for the new elements in v[2][4] so that
      //    the edge marked for refinement is between the first two nodes.
      tet->ParseRefinementFlag(old_redges, type, flag);

      v[0][3] = v_new;
      v[1][3] = v_new;
      new_redges[0][0] = 2;
      new_redges[0][1] = 1;
      new_redges[1][0] = 2;
      new_redges[1][1] = 1;
      switch (old_redges[0])
      {
      case 2:
         v[0][0] = vert[0]; v[0][1] = vert[2]; v[0][2] = vert[3];
         if (type == Tetrahedron::TYPE_PF) new_redges[0][1] = 4;
         break;
      case 3:
         v[0][0] = vert[3]; v[0][1] = vert[0]; v[0][2] = vert[2];
         break;
      case 5:
         v[0][0] = vert[2]; v[0][1] = vert[3]; v[0][2] = vert[0];
      }
      switch (old_redges[1])
      {
      case 1:
         v[1][0] = vert[2]; v[1][1] = vert[1]; v[1][2] = vert[3];
         if (type == Tetrahedron::TYPE_PF) new_redges[1][0] = 3;
         break;
      case 4:
         v[1][0] = vert[1]; v[1][1] = vert[3]; v[1][2] = vert[2];
         break;
      case 5:
         v[1][0] = vert[3]; v[1][1] = vert[2]; v[1][2] = vert[1];
      }

      int attr = tet->GetAttribute();
      if (WantTwoLevelState)
      {
#ifdef MFEM_USE_MEMALLOC
         BisectedElement *aux = BEMemory.Alloc();
         aux->SetCoarseElem(tet);
         tet = TetMemory.Alloc();
         tet->SetVertices(v[0]);
         tet->SetAttribute(attr);
#else
         BisectedElement *aux = new BisectedElement(tet);
         tet = new Tetrahedron(v[0], attr);
#endif
         aux->FirstChild = tet;
         aux->SecondChild = NumOfElements;
         pce[0] = aux;
      }
      else
         tet->SetVertices(v[0]);
      //  'tet' now points to the first child
      {
#ifdef MFEM_USE_MEMALLOC
         Tetrahedron *tet2 = TetMemory.Alloc();
         tet2->SetVertices(v[1]);
         tet2->SetAttribute(attr);
         elements.Append(tet2);
#else
         elements.Append(new Tetrahedron(v[1], attr));
#endif
      }

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
      ((Tetrahedron *)elements[NumOfElements])->
         CreateRefinementFlag(new_redges[1], new_type, flag+1);

      NumOfElements++;
   }
   else
      mfem_error("Bisection for now works only for triangles & tetrahedra.");
}

void Mesh::Bisection(int i, const DSTable &v_to_v, int *middle)
{
   int *vert;
   int v[2][3], v_new, bisect, t;
   Element **pce;

   if (WantTwoLevelState)
   {
      pce = &(boundary[i]);
      while (1)
      {
         t = pce[0]->GetType();
         if (t == Element::BISECTED)
            pce = & ( ((BisectedElement *) pce[0])->FirstChild );
         else if (t == Element::QUADRISECTED)
            pce = & ( ((QuadrisectedElement *) pce[0])->FirstChild );
         else
            break;
      }
   }
   else
      t = boundary[i]->GetType();

   if (t == Element::TRIANGLE)
   {
      Triangle *tri;
      if (WantTwoLevelState)
         tri = (Triangle *) pce[0];
      else
         tri = (Triangle *) boundary[i];

      vert = tri->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v(vert[0], vert[1]);
      if (middle[bisect] == -1)
         mfem_error("Error in Bisection(...) of boundary triangle!");
      else
         v_new = middle[bisect];

      // 2. Set the node indices for the new elements in v[0] and v[1] so that
      //    the  edge marked for refinement is between the first two nodes.
      v[0][0] = vert[2]; v[0][1] = vert[0]; v[0][2] = v_new;
      v[1][0] = vert[1]; v[1][1] = vert[2]; v[1][2] = v_new;
      if (WantTwoLevelState)
      {
#ifdef MFEM_USE_MEMALLOC
         BisectedElement *aux = BEMemory.Alloc();
         aux->SetCoarseElem(tri);
#else
         BisectedElement *aux = new BisectedElement(tri);
#endif
         aux->FirstChild = tri = new Triangle(v[0], tri->GetAttribute());
         aux->SecondChild = NumOfBdrElements;
         pce[0] = aux;
      }
      else
         boundary[i]->SetVertices(v[0]);
      //  'tri' now points to the first child
      boundary.Append(new Triangle(v[1], tri->GetAttribute()));

      NumOfBdrElements++;
   }
   else
      mfem_error("Bisection of boundary elements works only for triangles!");
}

void Mesh::UniformRefinement(int i, const DSTable &v_to_v,
                             int *edge1, int *edge2, int *middle)
{
   Array<int> v;
   int j, v1[3], v2[3], v3[3], v4[3], v_new[3], bisect[3];
   double coord[2];

   if (elements[i]->GetType() == Element::TRIANGLE)
   {
      elements[i]->GetVertices(v);

      // 1. Get the indeces for the new vertices in array v_new
      bisect[0] = v_to_v(v[0],v[1]);
      bisect[1] = v_to_v(v[1],v[2]);
      bisect[2] = v_to_v(v[0],v[2]);
#ifdef MFEM_DEBUG
      if (bisect[0] < 0 || bisect[1] < 0 || bisect[2] < 0)
         mfem_error("Mesh::UniformRefinement(...): ERROR");
#endif

      for (j = 0; j < 3; j++)                // for the 3 edges fix v_new
         if (middle[bisect[j]] == -1)
         {
            v_new[j] = NumOfVertices++;
            coord[0] = (vertices[v[j]](0) + vertices[v[(j+1)%3]](0))/2.;
            coord[1] = (vertices[v[j]](1) + vertices[v[(j+1)%3]](1))/2.;
            Vertex V(coord[0], coord[1]);
            vertices.Append(V);

            // Put the element that may need refinement (because of this
            // bisection) in edge1, or -1 if no more refinement is needed.
            if (edge1[bisect[j]] == i)
               edge1[bisect[j]] = edge2[bisect[j]];

            middle[bisect[j]] = v_new[j];
         }
         else
         {
            v_new[j] = middle[bisect[j]];

            // This edge will require no more refinement.
            edge1[bisect[j]] = -1;
         }

      // 2. Set the node indeces for the new elements in v1, v2, v3 & v4 so that
      //    the edges marked for refinement be between the first two nodes.
      v1[0] =     v[0]; v1[1] = v_new[0]; v1[2] = v_new[2];
      v2[0] = v_new[0]; v2[1] =     v[1]; v2[2] = v_new[1];
      v3[0] = v_new[2]; v3[1] = v_new[1]; v3[2] =     v[2];
      v4[0] = v_new[1]; v4[1] = v_new[2]; v4[2] = v_new[0];

      elements.Append(new Triangle(v1, elements[i]->GetAttribute()));
      elements.Append(new Triangle(v2, elements[i]->GetAttribute()));
      elements.Append(new Triangle(v3, elements[i]->GetAttribute()));
      if (WantTwoLevelState)
      {
         QuadrisectedElement *aux = new QuadrisectedElement(elements[i]);
         aux->FirstChild = new Triangle(v4, elements[i]->GetAttribute());
         aux->Child2 = NumOfElements;
         aux->Child3 = NumOfElements+1;
         aux->Child4 = NumOfElements+2;
         elements[i] = aux;
      }
      else
      {
         elements[i]->SetVertices(v4);
      }

      NumOfElements += 3;
   }
   else
      mfem_error("Uniform refinement for now works only for triangles.");
}

void Mesh::SetState(int s)
{
   if (State != Mesh::NORMAL && s == Mesh::NORMAL)
   {
      // two level state  --->>  normal state
      int i, t;

      for (i = 0; i < f_NumOfElements; )
      {
         t = elements[i]->GetType();
         if (t == Element::BISECTED     ||
             t == Element::QUADRISECTED ||
             t == Element::OCTASECTED)
         {
            RefinedElement *aux = (RefinedElement *) elements[i];
            elements[i] = aux->FirstChild;
            FreeElement(aux->CoarseElem);
            FreeElement(aux);
         }
         else
            i++;
      }

      for (i = 0; i < f_NumOfBdrElements; )
      {
         t = boundary[i]->GetType();
         if (t == Element::BISECTED     ||
             t == Element::QUADRISECTED ||
             t == Element::OCTASECTED)
         {
            RefinedElement *aux = (RefinedElement *) boundary[i];
            boundary[i] = aux->FirstChild;
            FreeElement(aux->CoarseElem);
            FreeElement(aux);
         }
         else
            i++;
      }

      if (el_to_edge != NULL)
      {
         delete c_el_to_edge;
         el_to_edge = f_el_to_edge;
         if (Dim == 2)
         {
            if (State == Mesh::TWO_LEVEL_COARSE)
               Swap(be_to_edge, fc_be_to_edge);
            fc_be_to_edge.DeleteAll();
         }
         if (Dim == 3)
         {
            delete c_bel_to_edge;
            bel_to_edge = f_bel_to_edge;
         }
      }
      if (el_to_face != NULL)
      {
         delete c_el_to_face;
         el_to_face = f_el_to_face;
         if (State == Mesh::TWO_LEVEL_COARSE)
            Swap(faces_info, fc_faces_info);
         fc_faces_info.DeleteAll();
      }

      NumOfVertices    = f_NumOfVertices;
      NumOfEdges       = f_NumOfEdges;
      if (Dim == 3)
         NumOfFaces    = f_NumOfFaces;
      NumOfElements    = f_NumOfElements;
      NumOfBdrElements = f_NumOfBdrElements;
      RefinedElement::State = RefinedElement::FINE;
      State = s;
   }
   else if (State == Mesh::TWO_LEVEL_COARSE && s == Mesh::TWO_LEVEL_FINE)
   {
      if (el_to_edge != NULL)
      {
         el_to_edge = f_el_to_edge;
         if (Dim == 2)
            Swap(be_to_edge, fc_be_to_edge);
         if (Dim == 3)
            bel_to_edge = f_bel_to_edge;
      }
      if (el_to_face != NULL)
      {
         el_to_face = f_el_to_face;
         Swap(faces_info, fc_faces_info);
      }
      NumOfVertices    = f_NumOfVertices;
      NumOfEdges       = f_NumOfEdges;
      if (Dim == 3)
         NumOfFaces    = f_NumOfFaces;
      NumOfElements    = f_NumOfElements;
      NumOfBdrElements = f_NumOfBdrElements;
      RefinedElement::State = RefinedElement::FINE;
      State = s;
   }
   else if (State == Mesh::TWO_LEVEL_FINE && s == Mesh::TWO_LEVEL_COARSE)
   {
      if (el_to_edge != NULL)
      {
         el_to_edge = c_el_to_edge;
         if (Dim == 2)
            Swap(be_to_edge, fc_be_to_edge);
         if (Dim == 3)
            bel_to_edge = c_bel_to_edge;
      }
      if (el_to_face != NULL)
      {
         el_to_face = c_el_to_face;
         Swap(faces_info, fc_faces_info);
      }
      NumOfVertices    = c_NumOfVertices;
      NumOfEdges       = c_NumOfEdges;
      if (Dim == 3)
         NumOfFaces    = c_NumOfFaces;
      NumOfElements    = c_NumOfElements;
      NumOfBdrElements = c_NumOfBdrElements;
      RefinedElement::State = RefinedElement::COARSE;
      State = s;
   }
   else if (State != s)
      mfem_error("Oops! Mesh::SetState");
}

int Mesh::GetNumFineElems(int i)
{
   int t;

   if (Dim == 2)
   {
      t = elements[i]->GetType();
      if (t == Element::QUADRISECTED)
         return 4;
      else if (t == Element::BISECTED)
      {
         // assuming that the elements are either BisectedElements or
         // regular elements
         int n = 1;
         BisectedElement *aux = (BisectedElement *) elements[i];
         do
         {
            n += GetNumFineElems(aux->SecondChild);
            if (aux->FirstChild->GetType() != Element::BISECTED)
               break;
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         return n;
      }
   }
   else if (Dim == 3)
   {
      // assuming that the element is a BisectedElement,
      // OctasectedElement (with children that are regular elements) or
      // regular element
      t = elements[i]->GetType();
      if (t == Element::BISECTED)
      {
         int n = 1;
         BisectedElement *aux = (BisectedElement *) elements[i];
         do
         {
            n += GetNumFineElems (aux->SecondChild);
            if (aux->FirstChild->GetType() != Element::BISECTED)
               break;
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         return n;
      }
      else if (t == Element::OCTASECTED)
         return 8;
      return 1; // regular element (i.e. it is not refined)
   }

   return 1;  // the element is not refined
}

int Mesh::GetBisectionHierarchy(Element *E)
{
   if (E->GetType() == Element::BISECTED)
   {
      int L, R, n, s, lb, rb;

      L = GetBisectionHierarchy(((BisectedElement *)E)->FirstChild);
      n = ((BisectedElement *)E)->SecondChild;
      R = GetBisectionHierarchy(elements[n]);
      n = 1; s = 1;
      lb = rb = 1;
      do
      {
         int nlb, nrb;
         nlb = nrb = 0;
         while (lb > 0)
         {
            n |= ((L & 1) << s);
            s++;
            nlb += (L & 1);
            L = (L >> 1);
            lb--;
         }
         while (rb > 0)
         {
            n |= ((R & 1) << s);
            s++;
            nrb += (R & 1);
            R = (R >> 1);
            rb--;
         }
         lb = 2 * nlb; rb = 2 * nrb;
      }
      while (lb > 0 || rb > 0);
      return n;
   }
   return 0;
}

int Mesh::GetRefinementType(int i)
{
   int t;

   if (Dim == 2)
   {
      t = elements[i]->GetType();
      if (t == Element::QUADRISECTED)
      {
         t = ((QuadrisectedElement *)elements[i])->CoarseElem->GetType();
         if (t == Element::QUADRILATERAL)
            return 1;  //  refinement type for quadrisected QUADRILATERAL
         else
            return 2;  //  refinement type for quadrisected TRIANGLE
      }
      else if (t == Element::BISECTED)
      {
         int type;
         type = GetBisectionHierarchy(elements[i]);
         if (type == 0)
            mfem_error("Mesh::GetRefinementType(...)");
         return type+2;
      }
   }
   else if (Dim == 3)
   {
      int redges[2], type, flag;
      Element *E = elements[i];
      Tetrahedron *tet;

      t = E->GetType();
      if (t != Element::BISECTED)
      {
         if (t == Element::OCTASECTED)
            return 1;  //  refinement type for octasected CUBE
         else
            return 0;
      }
      // Bisected TETRAHEDRON
      tet = (Tetrahedron *) (((BisectedElement *) E)->CoarseElem);
      tet->ParseRefinementFlag(redges, type, flag);
      if (type == Tetrahedron::TYPE_A && redges[0] == 2)
         type = 5;
      else if (type == Tetrahedron::TYPE_M && redges[0] == 2)
         type = 6;
      type++;
      type |= ( GetBisectionHierarchy(E) << 3 );
      if (type < 8) type = 0;

      return type;
   }

   return 0;  // no refinement
}

int Mesh::GetFineElem(int i, int j)
{
   int t;

   if (Dim == 2)
   {
      t = elements[i]->GetType();
      if (t == Element::QUADRISECTED)
      {
         QuadrisectedElement *aux = (QuadrisectedElement *) elements[i];
         if (aux->CoarseElem->GetType() == Element::QUADRILATERAL)
            switch (j)
            {
            case 0:   return i;
            case 1:   return aux->Child2;
            case 2:   return aux->Child3;
            case 3:   return aux->Child4;
            default:  *((int *)NULL) = -1; // crash it
            }
         else // quadrisected TRIANGLE
            switch (j)
            {
            case 0:   return aux->Child2;
            case 1:   return aux->Child3;
            case 2:   return aux->Child4;
            case 3:   return i;
            default:  *((int *)NULL) = -1; // crash it
            }
      }
      else if (t == Element::BISECTED)
      {
         int n = 0;
         BisectedElement *aux = (BisectedElement *) elements[i];
         do
         {
            int k = GetFineElem(aux->SecondChild, j-n);
            if (k >= 0)
               return k;
            n -= k;  // (-k) is the number of the leaves in this SecondChild
                     //   n  is the number of the leaves in
                     //      the SecondChild-ren so far
            if (aux->FirstChild->GetType() != Element::BISECTED)
               break;
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         if (j > n)  //  i.e. if (j >= n+1)
            return -(n+1);
         return i;  //  j == n, i.e. j is the index of the last leaf
      }
   }
   else if (Dim == 3)
   {
      t = elements[i]->GetType();
      if (t == Element::BISECTED)
      {
         int n = 0;
         BisectedElement *aux = (BisectedElement *) elements[i];
         do
         {
            int k = GetFineElem(aux->SecondChild, j-n);
            if (k >= 0)
               return k;
            n -= k;  // (-k) is the number of the leaves in this SecondChild
                     //   n  is the number of the leaves in
                     //      the SecondChild-ren so far
            if (aux->FirstChild->GetType() != Element::BISECTED)
               break;
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         if (j > n)  //  i.e. if (j >= n+1)
            return -(n+1);
         return i;  //  j == n, i.e. j is the index of the last leaf
      }
      else if (t == Element::OCTASECTED)
      {
         if (j == 0)  return i;
         return ((OctasectedElement *) elements[i])->Child[j-1];
      }
   }

   if (j > 0)
      return -1;

   return i;  // no refinement
}

void Mesh::BisectTriTrans(DenseMatrix &pointmat, Triangle *tri, int child)
{
   double np[2];

   if (child == 0)  // left triangle
   {
      // Set the new coordinates of the vertices
      np[0] = 0.5 * ( pointmat(0,0) + pointmat(0,1) );
      np[1] = 0.5 * ( pointmat(1,0) + pointmat(1,1) );
      pointmat(0,1) = pointmat(0,0); pointmat(1,1) = pointmat(1,0);
      pointmat(0,0) = pointmat(0,2); pointmat(1,0) = pointmat(1,2);
      pointmat(0,2) = np[0]; pointmat(1,2) = np[1];
   }
   else  // right triangle
   {
      // Set the new coordinates of the vertices
      np[0] = 0.5 * ( pointmat(0,0) + pointmat(0,1) );
      np[1] = 0.5 * ( pointmat(1,0) + pointmat(1,1) );
      pointmat(0,0) = pointmat(0,1); pointmat(1,0) = pointmat(1,1);
      pointmat(0,1) = pointmat(0,2); pointmat(1,1) = pointmat(1,2);
      pointmat(0,2) = np[0]; pointmat(1,2) = np[1];
   }
}

void Mesh::BisectTetTrans(DenseMatrix &pointmat, Tetrahedron *tet, int child)
{
   int i, j, redges[2], type, flag, ind[4];
   double t[4];

   tet->ParseRefinementFlag(redges, type, flag);

   if (child == 0)  // left tetrahedron
   {
      // Set the new coordinates of the vertices
      pointmat(0,1) = 0.5 * ( pointmat(0,0) + pointmat(0,1) );
      pointmat(1,1) = 0.5 * ( pointmat(1,0) + pointmat(1,1) );
      pointmat(2,1) = 0.5 * ( pointmat(2,0) + pointmat(2,1) );
      // Permute the vertices according to the type & redges
      switch (redges[0])
      {
      case 2:  ind[0] = 0; ind[1] = 3; ind[2] = 1; ind[3] = 2;  break;
      case 3:  ind[0] = 1; ind[1] = 3; ind[2] = 2; ind[3] = 0;  break;
      case 5:  ind[0] = 2; ind[1] = 3; ind[2] = 0; ind[3] = 1;
      }
   }
   else  // right tetrahedron
   {
      // Set the new coordinates of the vertices
      pointmat(0,0) = 0.5 * ( pointmat(0,0) + pointmat(0,1) );
      pointmat(1,0) = 0.5 * ( pointmat(1,0) + pointmat(1,1) );
      pointmat(2,0) = 0.5 * ( pointmat(2,0) + pointmat(2,1) );
      // Permute the vertices according to the type & redges
      switch (redges[1])
      {
      case 1:  ind[0] = 3; ind[1] = 1; ind[2] = 0; ind[3] = 2;  break;
      case 4:  ind[0] = 3; ind[1] = 0; ind[2] = 2; ind[3] = 1;  break;
      case 5:  ind[0] = 3; ind[1] = 2; ind[2] = 1; ind[3] = 0;
      }
   }
   // Do the permutation
   for (i = 0; i < 3; i++)
   {
      for (j = 0; j < 4; j++)
         t[j] = pointmat(i,j);
      for (j = 0; j < 4; j++)
         pointmat(i,ind[j]) = t[j];
   }
}

int Mesh::GetFineElemPath(int i, int j)
{
   // if (Dim == 3)
   {
      if (elements[i]->GetType() == Element::BISECTED)
      {
         int n = 0, l = 0;
         BisectedElement *aux = (BisectedElement *) elements[i];
         do
         {
            int k = GetFineElemPath(aux->SecondChild, j-n);
            if (k >= 0)
               return ((k << 1)+1) << l;
            n -= k;  // (-k) is the number of the leaves in this SecondChild
                     //   n  is the number of the leaves in
                     //      the SecondChild-ren so far
            l++;
            if (aux->FirstChild->GetType() != Element::BISECTED)
               break;
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         if (j > n)  //  i.e. if (j >= n+1)
            return -(n+1);
         return 0;  //  j == n, i.e. j is the index of the last leaf
      }
      if (j > 0)
         return -1;
   }

   return 0;
}

ElementTransformation * Mesh::GetFineElemTrans(int i, int j)
{
   int t;

   if (Dim == 2)
   {
      DenseMatrix &pm = Transformation.GetPointMat();
      Transformation.Attribute = 0;
      Transformation.ElementNo = 0;
      t = elements[i]->GetType();
      if (t == Element::QUADRISECTED)
      {
         t = ((QuadrisectedElement *)elements[i])->CoarseElem->GetType();
         if (t == Element::QUADRILATERAL)
         {
            // quadrisected QUADRILATERAL
            Transformation.SetFE(&QuadrilateralFE);
            pm.SetSize(2, 4);
            switch (j)
            {
            case 0:
               pm(0,0) = 0.0; pm(1,0) = 0.0;  //  x; y;
               pm(0,1) = 0.5; pm(1,1) = 0.0;
               pm(0,2) = 0.5; pm(1,2) = 0.5;
               pm(0,3) = 0.0; pm(1,3) = 0.5;
               break;
            case 1:
               pm(0,0) = 0.5; pm(1,0) = 0.0;
               pm(0,1) = 1.0; pm(1,1) = 0.0;
               pm(0,2) = 1.0; pm(1,2) = 0.5;
               pm(0,3) = 0.5; pm(1,3) = 0.5;
               break;
            case 2:
               pm(0,0) = 0.5; pm(1,0) = 0.5;
               pm(0,1) = 1.0; pm(1,1) = 0.5;
               pm(0,2) = 1.0; pm(1,2) = 1.0;
               pm(0,3) = 0.5; pm(1,3) = 1.0;
               break;
            case 3:
               pm(0,0) = 0.0; pm(1,0) = 0.5;
               pm(0,1) = 0.5; pm(1,1) = 0.5;
               pm(0,2) = 0.5; pm(1,2) = 1.0;
               pm(0,3) = 0.0; pm(1,3) = 1.0;
               break;
            default:
               mfem_error("Mesh::GetFineElemTrans(...) 1");
            }
         }
         else
         {
            // quadrisected TRIANGLE
            Transformation.SetFE(&TriangleFE);
            pm.SetSize(2, 3);
            switch (j)
            {
            case 0:
               pm(0,0) = 0.0;  pm(0,1) = 0.5;  pm(0,2) = 0.0;  // x
               pm(1,0) = 0.0;  pm(1,1) = 0.0;  pm(1,2) = 0.5;  // y
               break;
            case 1:
               pm(0,0) = 0.5;  pm(0,1) = 1.0;  pm(0,2) = 0.5;
               pm(1,0) = 0.0;  pm(1,1) = 0.0;  pm(1,2) = 0.5;
               break;
            case 2:
               pm(0,0) = 0.0;  pm(0,1) = 0.5;  pm(0,2) = 0.0;
               pm(1,0) = 0.5;  pm(1,1) = 0.5;  pm(1,2) = 1.0;
               break;
            case 3:
               pm(0,0) = 0.5;  pm(0,1) = 0.0;  pm(0,2) = 0.5;
               pm(1,0) = 0.5;  pm(1,1) = 0.5;  pm(1,2) = 0.0;
               break;
            default:
               mfem_error("Mesh::GetFineElemTrans(...) 2");
            }
         }
      }
      else if (t == Element::BISECTED)
      {
         // bisected TRIANGLE
         Transformation.SetFE(&TriangleFE);
         pm.SetSize(2, 3);

         int path;
         Element *E;

         // pm is initialzed with the coordinates of the vertices of the
         //     reference triangle
         pm(0,0) = 0.0;  pm(0,1) = 1.0;  pm(0,2) = 0.0;
         pm(1,0) = 0.0;  pm(1,1) = 0.0;  pm(1,2) = 1.0;

         path = GetFineElemPath(i, j);

         E = elements[i];
         while (E->GetType() == Element::BISECTED)
         {
            BisectedElement *aux = (BisectedElement *) E;

            BisectTriTrans(pm, (Triangle *) aux->CoarseElem, path & 1);
            E = (path & 1) ? elements[aux->SecondChild] : aux->FirstChild;
            path = path >> 1;
         }
      }
      else
      {
         //  identity transformation
         Transformation.SetFE(&TriangleFE);
         pm.SetSize(2, 3);
         pm(0,0) = 0.0;  pm(0,1) = 1.0;  pm(0,2) = 0.0;
         pm(1,0) = 0.0;  pm(1,1) = 0.0;  pm(1,2) = 1.0;
      }
      return &Transformation;
   }
   else if (Dim == 3)
   {
      if (elements[i]->GetType() == Element::OCTASECTED)
      {
         int jj;
         double dx, dy, dz;
         DenseMatrix &pm = Transformation.GetPointMat();
         Transformation.SetFE(&HexahedronFE);
         Transformation.Attribute = 0;
         Transformation.ElementNo = 0;
         pm.SetSize(3, 8);
         if (j < 4)  dz = 0.0;
         else        dz = 0.5;
         jj = j % 4;
         if (jj < 2)  dy = 0.0;
         else         dy = 0.5;
         if (jj == 0 || jj == 3)  dx = 0.0;
         else                     dx = 0.5;
         pm(0,0) =       dx;  pm(1,0) =       dy;  pm(2,0) =       dz;
         pm(0,1) = 0.5 + dx;  pm(1,1) =       dy;  pm(2,1) =       dz;
         pm(0,2) = 0.5 + dx;  pm(1,2) = 0.5 + dy;  pm(2,2) =       dz;
         pm(0,3) =       dx;  pm(1,3) = 0.5 + dy;  pm(2,3) =       dz;
         pm(0,4) =       dx;  pm(1,4) =       dy;  pm(2,4) = 0.5 + dz;
         pm(0,5) = 0.5 + dx;  pm(1,5) =       dy;  pm(2,5) = 0.5 + dz;
         pm(0,6) = 0.5 + dx;  pm(1,6) = 0.5 + dy;  pm(2,6) = 0.5 + dz;
         pm(0,7) =       dx;  pm(1,7) = 0.5 + dy;  pm(2,7) = 0.5 + dz;
         return &Transformation;
      }
      int path;
      Element *E;
      DenseMatrix &pm = Transformation.GetPointMat();
      Transformation.SetFE(&TetrahedronFE);
      Transformation.Attribute = 0;
      Transformation.ElementNo = 0;
      pm.SetSize(3, 4);

      // pm is initialzed with the coordinates of the vertices of the
      //     reference tetrahedron
      pm(0,0) = 0.0;  pm(0,1) = 1.0;  pm(0,2) = 0.0;  pm(0,3) = 0.0;
      pm(1,0) = 0.0;  pm(1,1) = 0.0;  pm(1,2) = 1.0;  pm(1,3) = 0.0;
      pm(2,0) = 0.0;  pm(2,1) = 0.0;  pm(2,2) = 0.0;  pm(2,3) = 1.0;

      path = GetFineElemPath(i, j);

      E = elements[i];
      while (E->GetType() == Element::BISECTED)
      {
         BisectedElement *aux = (BisectedElement *) E;

         BisectTetTrans(pm, (Tetrahedron *) aux->CoarseElem, path & 1);
         E = (path & 1) ? elements[aux->SecondChild] : aux->FirstChild;
         path = path >> 1;
      }
   }

   return &Transformation;  // no refinement
}

void Mesh::PrintXG(ostream &out) const
{
   int i, j;
   Array<int> v;

   if (Dim == 2)
   {
      // Print the type of the mesh.
      if (Nodes == NULL)
         out << "areamesh2\n\n";
      else
         out << "curved_areamesh2\n\n";

      // Print the boundary elements.
      out << NumOfBdrElements << '\n';
      for (i = 0; i < NumOfBdrElements; i++)
      {
         boundary[i]->GetVertices(v);

         out << boundary[i]->GetAttribute();
         for (j = 0; j < v.Size(); j++)
            out << ' ' << v[j] + 1;
         out << '\n';
      }

      // Print the elements.
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         elements[i]->GetVertices(v);

         out << elements[i]->GetAttribute() << ' ' << v.Size();
         for (j = 0; j < v.Size(); j++)
            out << ' ' << v[j] + 1;
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
               out << ' ' << vertices[i](j);
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
               out << ' ' << vertices[i](j);
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
               out << ' ' << ind[j]+1;
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
               out << ' ' << ind[j]+1;
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
               out << ' ' << ind[j]+1;
            out << '\n';
         }

         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            out << boundary[i]->GetAttribute();
            for (j = 0; j < nv; j++)
               out << ' ' << ind[j]+1;
            out << " 1.0 1.0 1.0 1.0\n";
         }
      }
   }

   out << flush;
}

void Mesh::Print(ostream &out) const
{
   int i, j;

   if (NURBSext)
   {
      NURBSext->Print(out);
      out << '\n';
      Nodes->Save(out);
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
      PrintElement(elements[i], out);

   out << "\nboundary\n" << NumOfBdrElements << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
      PrintElement(boundary[i], out);

   out << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      out << Dim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         for (j = 1; j < Dim; j++)
            out << ' ' << vertices[i](j);
         out << '\n';
      }
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
   }
}

void Mesh::PrintTopo(ostream &out,const Array<int> &e_to_k) const
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
      PrintElement(elements[i], out);

   out << "\nboundary\n" << NumOfBdrElements << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
      PrintElement(boundary[i], out);

   out << "\nedges\n" << NumOfEdges << '\n';
   for (i = 0; i < NumOfEdges; i++)
   {
      edge_vertex->GetRow(i, vert);
      int ki = e_to_k[i];
      if (ki < 0)
         ki = -1 - ki;
      out << ki << ' ' << vert[0] << ' ' << vert[1] << '\n';
   }
   out << "\nvertices\n" << NumOfVertices << '\n';
}

void Mesh::PrintVTK(ostream &out)
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
         for (j = 1; j < Dim; j++)
            out << ' ' << vertices[i](j);
         for ( ; j < 3; j++)
            out << ' ' << 0.0;
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
         for (j = 1; j < Dim; j++)
            out << ' ' << (*Nodes)(vdofs[j]);
         for ( ; j < 3; j++)
            out << ' ' << 0.0;
         out << '\n';
      }
   }

   int order = -1;
   if (Nodes == NULL)
   {
      int size = 0;
      for (int i = 0; i < NumOfElements; i++)
         size += elements[i]->GetNVertices() + 1;
      out << "CELLS " << NumOfElements << ' ' << size << '\n';
      for (int i = 0; i < NumOfElements; i++)
      {
         const int *v = elements[i]->GetVertices();
         const int nv = elements[i]->GetNVertices();
         out << nv;
         for (int j = 0; j < nv; j++)
            out << ' ' << v[j];
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
      if (!strcmp(fec_name, "Linear"))
         order = 1;
      else if (!strcmp(fec_name, "Quadratic"))
         order = 2;
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
               out << ' ' << dofs[j];
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
               vtk_mfem = vtk_quadratic_hex; break;
            }
            for (int j = 0; j < dofs.Size(); j++)
               out << ' ' << dofs[vtk_mfem[j]];
         }
         out << '\n';
      }
   }

   out << "CELL_TYPES " << NumOfElements << '\n';
   for (int i = 0; i < NumOfElements; i++)
   {
      int vtk_cell_type;
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
}

void Mesh::PrintVTK(ostream &out, int ref, int field_data)
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
      out << "FIELD FieldData 1" << endl
          << "MaterialIds " << 1 << " " << attributes.Size() << " int" << endl;
      for (int i = 0; i < attributes.Size(); i++)
         out << attributes[i] << " ";
      out << endl;
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
         out << pmat(0, j) << ' ' << pmat(1, j) << ' ';
         if (pmat.Height() == 2)
            out << 0.0;
         else
            out << pmat(2, j);
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
            out << ' ' << np + RG[j];
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
   srandom(time(0));
   double a = double(random()) / (double(RAND_MAX) + 1.);
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
   out << "POINT_DATA " << np << '\n';
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
         continue;

      colors[el] = -1;
      el_stack[stack_top_p++] = el;

      for ( ; stack_p < stack_top_p; stack_p++)
      {
         int i = el_stack[stack_p];
         int num_nb = i_el_el[i+1] - i_el_el[i];
         if (max_num_col < num_nb + 1)
            max_num_col = num_nb + 1;
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
            col_marker[col] = 1;
      }

      for (col = 0; col < max_num_col; col++)
         if (col_marker[col] == 0)
            break;

      colors[i] = col;
   }

   if (delete_el_to_el)
   {
      delete el_to_el;
      el_to_el = NULL;
   }
}

void Mesh::PrintWithPartitioning(int *partitioning, ostream &out,
                                 int elem_attr) const
{
   if (Dim != 3 && Dim != 2) return;

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
      out << int((elem_attr) ? partitioning[i] : elements[i]->GetAttribute())
          << ' ' << elements[i]->GetGeometryType();
      nv = elements[i]->GetNVertices();
      v  = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
         out << ' ' << v[j];
      out << '\n';
   }
   nbe = 0;
   for (i = 0; i < NumOfFaces; i++)
   {
      if ((l = faces_info[i].Elem2No) >= 0)
      {
         k = partitioning[faces_info[i].Elem1No];
         l = partitioning[l];
         if (k != l)
            nbe += 2;
      }
      else
         nbe++;
   }
   out << "\nboundary\n" << nbe << '\n';
   for (i = 0; i < NumOfFaces; i++)
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
               out << ' ' << v[j];
            out << '\n';
            out << l+1 << ' ' << faces[i]->GetGeometryType();
            for (j = nv-1; j >= 0; j--)
               out << ' ' << v[j];
            out << '\n';
         }
      }
      else
      {
         k = partitioning[faces_info[i].Elem1No];
         nv = faces[i]->GetNVertices();
         v  = faces[i]->GetVertices();
         out << k+1 << ' ' << faces[i]->GetGeometryType();
         for (j = 0; j < nv; j++)
            out << ' ' << v[j];
         out << '\n';
      }
   }
   out << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      out << Dim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         for (j = 1; j < Dim; j++)
            out << ' ' << vertices[i](j);
         out << '\n';
      }
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
   }
}

void Mesh::PrintElementsWithPartitioning(int *partitioning,
                                         ostream &out,
                                         int interior_faces)
{
   if (Dim != 3 && Dim != 2) return;

   int i, j, k, l, s;

   int nv;
   const int *ind;

   int *vcount = new int[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
      vcount[i] = 0;
   for (i = 0; i < NumOfElements; i++)
   {
      nv = elements[i]->GetNVertices();
      ind = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
         vcount[ind[j]]++;
   }

   int *voff = new int[NumOfVertices+1];
   voff[0] = 0;
   for (i = 1; i <= NumOfVertices; i++)
      voff[i] = vcount[i-1] + voff[i-1];

   int **vown = new int*[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
      vown[i] = new int[vcount[i]];

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
         vcount[i] = voff[i+1] - voff[i];

      nbe = 0;
      for (i = 0; i < edge_el.Size(); i++)
      {
         const int *el = edge_el.GetRow(i);
         if (edge_el.RowSize(i) > 1)
         {
            k = partitioning[el[0]];
            l = partitioning[el[1]];
            if (interior_faces || k != l)
               nbe += 2;
         }
         else
            nbe++;
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
                        out << ' ' << voff[ev[j]]+s+1;
               out << '\n';
               out << l+1; // attribute
               for (j = 1; j >= 0; j--)
                  for (s = 0; s < vcount[ev[j]]; s++)
                     if (vown[ev[j]][s] == el[1])
                        out << ' ' << voff[ev[j]]+s+1;
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
                     out << ' ' << voff[ev[j]]+s+1;
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
         vcount[i] = voff[i+1] - voff[i];

      // Print the vertices.
      out << voff[NumOfVertices] << '\n';
      for (i = 0; i < NumOfVertices; i++)
         for (k = 0; k < vcount[i]; k++)
         {
            for (j = 0; j < Dim; j++)
               out << vertices[i](j) << ' ';
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
               out << ' ' << vertices[i](j);
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
         vcount[i] = voff[i+1] - voff[i];

      // print the boundary information.
      int k, l, nbe;
      nbe = 0;
      for (i = 0; i < NumOfFaces; i++)
         if ((l = faces_info[i].Elem2No) >= 0)
         {
            k = partitioning[faces_info[i].Elem1No];
            l = partitioning[l];
            if (interior_faces || k != l)
               nbe += 2;
         }
         else
            nbe++;

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
                        out << ' ' << voff[ind[j]]+s+1;
               out << '\n';
               out << l+1; // attribute
               for (j = nv-1; j >= 0; j--)
                  for (s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem2No)
                        out << ' ' << voff[ind[j]]+s+1;
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
                     out << ' ' << voff[ind[j]]+s+1;
            out << '\n';
         }

      for (i = 0; i < NumOfVertices; i++)
         delete [] vown[i];
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
               nbe += 2;
         }
         else
            nbe++;


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
         vcount[i] = voff[i+1] - voff[i];

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
                        out << ' ' << voff[ind[j]]+s+1;
               out << " 1.0 1.0 1.0 1.0\n";
               out << l+1; // attribute
               for (j = nv-1; j >= 0; j--)
                  for (s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem2No)
                        out << ' ' << voff[ind[j]]+s+1;
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
                     out << ' ' << voff[ind[j]]+s+1;
            out << " 1.0 1.0 1.0 1.0\n";
         }
   }

   out << flush;

   delete [] vcount;
   delete [] voff;
   delete [] vown;
}

void Mesh::ScaleSubdomains(double sf)
{
   int i,j,k;
   Array<int> vert;
   DenseMatrix pointmat;
   int na = attributes.Size();
   double *cg = new double[na*Dim];
   int *nbea = new int[na];

   int *vn = new int[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
      vn[i] = 0;
   for (i = 0; i < na; i++)
   {
      for (j = 0; j < Dim; j++)
         cg[i*Dim+j] = 0.0;
      nbea[i] = 0;
   }

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementVertices(i, vert);
      for (k = 0; k < vert.Size(); k++)
         vn[vert[k]] = 1;
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
            for (j = 0; j < Dim; j++)
               cg[bea*Dim+j] += pointmat(j,k);
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
            for (j = 0; j < Dim; j++)
               vertices[vert[k]](j) = sf*vertices[vert[k]](j) +
                  (1-sf)*cg[bea*Dim+j]/nbea[bea];
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
   double *cg = new double[na*Dim];
   int *nbea = new int[na];

   int *vn = new int[NumOfVertices];
   for (i = 0; i < NumOfVertices; i++)
      vn[i] = 0;
   for (i = 0; i < na; i++)
   {
      for (j = 0; j < Dim; j++)
         cg[i*Dim+j] = 0.0;
      nbea[i] = 0;
   }

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementVertices(i, vert);
      for (k = 0; k < vert.Size(); k++)
         vn[vert[k]] = 1;
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
            for (j = 0; j < Dim; j++)
               cg[bea*Dim+j] += pointmat(j,k);
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
            for (j = 0; j < Dim; j++)
               vertices[vert[k]](j) = sf*vertices[vert[k]](j) +
                  (1-sf)*cg[bea*Dim+j]/nbea[bea];
            vn[vert[k]] = 0;
         }
   }

   delete [] cg;
   delete [] nbea;
   delete [] vn;
}

void Mesh::Transform(void (*f)(const Vector&, Vector&))
{
   if (Nodes == NULL)
   {
      Vector vold(Dim), vnew(NULL, Dim);
      for (int i = 0; i < vertices.Size(); i++)
      {
         for (int j = 0; j < Dim; j++)
            vold(j) = vertices[i](j);
         vnew.SetData(vertices[i]());
         (*f)(vold, vnew);
      }
   }
   else
   {
      GridFunction xnew(Nodes->FESpace());
      VectorFunctionCoefficient f_pert(Dim, f);
      xnew.ProjectCoefficient(f_pert);
      *Nodes = xnew;
   }
}

void Mesh::FreeElement(Element *E)
{
#ifdef MFEM_USE_MEMALLOC
   if (E)
      switch (E->GetType())
      {
      case Element::TETRAHEDRON: TetMemory.Free((Tetrahedron *)E); break;
      case Element::BISECTED: BEMemory.Free((BisectedElement *)E); break;
      default: delete E; break;
      }
#else
   delete E;
#endif
}

Mesh::~Mesh()
{
   int i;

   if (own_nodes) delete Nodes;

   delete NURBSext;

   for (i = 0; i < NumOfElements; i++)
      FreeElement(elements[i]);

   for (i = 0; i < NumOfBdrElements; i++)
      FreeElement(boundary[i]);

   for (i = 0; i < faces.Size(); i++)
      FreeElement(faces[i]);

   DeleteTables();
}
