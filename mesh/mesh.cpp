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

namespace mfem
{

using namespace std;

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

void Mesh::PrintCharacteristics(Vector *Vh, Vector *Vk, std::ostream &out)
{
   int i, dim, sdim;
   DenseMatrix J;
   double h_min, h_max, kappa_min, kappa_max, h, kappa;

   out << "Mesh Characteristics:";

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

   if (dim == 1)
   {
      out << '\n'
          << "Number of vertices : " << GetNV() << '\n'
          << "Number of elements : " << GetNE() << '\n'
          << "Number of bdr elem : " << GetNBE() << '\n'
          << "h_min              : " << h_min << '\n'
          << "h_max              : " << h_max << '\n';
   }
   else if (dim == 2)
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
   else
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
      FTr->SetFE(GetTransformationFEforElementType(
                    (Dim == 1) ? Element::POINT : faces[FaceNo]->GetType()));
   }
   else
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
      else
      {
         int face_geom =
            (Dim == 1) ? Geometry::POINT : faces[FaceNo]->GetGeometryType();
         FaceInfo &face_info = faces_info[FaceNo];

         face_el = Nodes->FESpace()->GetTraceElement(face_info.Elem1No,
                                                     (Geometry::Type)face_geom);

         switch (face_geom)
         {
            case Geometry::POINT:
               GetLocalPtToSegTransformation(FaceElemTr.Loc1.Transf,
                                             face_info.Elem1Inf);
               break;
            case Geometry::SEGMENT:
               if (GetElementType(face_info.Elem1No) == Element::TRIANGLE)
                  GetLocalSegToTriTransformation(FaceElemTr.Loc1.Transf,
                                                 face_info.Elem1Inf);
               else // assume the element is a quad
                  GetLocalSegToQuadTransformation(FaceElemTr.Loc1.Transf,
                                                  face_info.Elem1Inf);
               break;
            case Geometry::TRIANGLE:
               // --- assume the face is a triangle -- face of a tetrahedron
               GetLocalTriToTetTransformation(FaceElemTr.Loc1.Transf,
                                              face_info.Elem1Inf);
               break;
            case Geometry::SQUARE:
               // ---  assume the face is a quad -- face of a hexahedron
               GetLocalQuadToHexTransformation(FaceElemTr.Loc1.Transf,
                                               face_info.Elem1Inf);
               break;
         }

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
      EdTr->SetFE(GetTransformationFEforElementType(Element::SEGMENT));
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
   // tri_faces is the same as Triangle::edges
   static const int tri_faces[3][2] = {{0, 1}, {1, 2}, {2, 0}};
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
   // quad_faces is the same as Quadrilateral::edges
   static const int quad_faces[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
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

const int Mesh::tet_faces[4][3] =
{
   {1, 2, 3}, {0, 3, 2},
   {0, 1, 3}, {0, 2, 1}
};

// same as Hexahedron::faces
const int Mesh::hex_faces[6][4] =
{
   {3, 2, 1, 0}, {0, 1, 5, 4},
   {1, 2, 6, 5}, {2, 3, 7, 6},
   {3, 0, 4, 7}, {4, 5, 6, 7}
};

const int Mesh::tri_orientations[6][3] =
{
   {0, 1, 2}, {1, 0, 2},
   {2, 0, 1}, {2, 1, 0},
   {1, 2, 0}, {0, 2, 1}
};

const int Mesh::quad_orientations[8][4] =
{
   {0, 1, 2, 3}, {0, 3, 2, 1},
   {1, 2, 3, 0}, {1, 0, 3, 2},
   {2, 3, 0, 1}, {2, 1, 0, 3},
   {3, 0, 1, 2}, {3, 2, 1, 0}
};

void Mesh::GetLocalTriToTetTransformation(
   IsoparametricTransformation &Transf, int i)
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&TriangleFE);
   //  (i/64) is the local face no. in the tet
   const int *tv = tet_faces[i/64];
   //  (i%64) is the orientation of the tetrahedron face
   //         w.r.t. the face element
   const int *to = tri_orientations[i%64];
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

void Mesh::GetLocalQuadToHexTransformation(
   IsoparametricTransformation &Transf, int i)
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&QuadrilateralFE);
   //  (i/64) is the local face no. in the hex
   const int *hv = hex_faces[i/64];
   //  (i%64) is the orientation of the quad
   const int *qo = quad_orientations[i%64];
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

FaceElementTransformations *Mesh::GetFaceElementTransformations(int FaceNo,
                                                                int mask)
{
   FaceInfo &face_info = faces_info[FaceNo];

   FaceElemTr.Elem1 = NULL;
   FaceElemTr.Elem2 = NULL;

   //  setup the transformation for the first element
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

   FaceElemTr.FaceGeom = (Dim == 1) ? Geometry::POINT
                         : faces[FaceNo]->GetGeometryType();

   // setup the face transformation
   FaceElemTr.Face = (mask & 16) ? GetFaceTransformation(FaceNo) : NULL;

   // setup Loc1 & Loc2
   int face_type = (Dim == 1) ? Element::POINT : faces[FaceNo]->GetType();
   switch (face_type)
   {
      case Element::POINT:
         if (mask & 4)
         {
            GetLocalPtToSegTransformation(FaceElemTr.Loc1.Transf,
                                          face_info.Elem1Inf);
         }
         if (FaceElemTr.Elem2No >= 0 && (mask & 8))
         {
            GetLocalPtToSegTransformation(FaceElemTr.Loc2.Transf,
                                          face_info.Elem2Inf);
         }
         break;

      case Element::SEGMENT:
         if (mask & 4)
         {
            if (GetElementType(face_info.Elem1No) == Element::TRIANGLE)
            {
               GetLocalSegToTriTransformation(FaceElemTr.Loc1.Transf,
                                              face_info.Elem1Inf);
            }
            else // assume the element is a quad
            {
               GetLocalSegToQuadTransformation(FaceElemTr.Loc1.Transf,
                                               face_info.Elem1Inf);
            }
         }

         if (FaceElemTr.Elem2No >= 0 && (mask & 8))
         {
            if (GetElementType(face_info.Elem2No) == Element::TRIANGLE)
            {
               GetLocalSegToTriTransformation(FaceElemTr.Loc2.Transf,
                                              face_info.Elem2Inf);
            }
            else // assume the element is a quad
            {
               GetLocalSegToQuadTransformation(FaceElemTr.Loc2.Transf,
                                               face_info.Elem2Inf);
            }
            if (IsSlaveFace(face_info))
            {
               ApplySlaveTransformation(FaceElemTr.Loc2.Transf, face_info);
               const int *fv = faces[FaceNo]->GetVertices();
               if (fv[0] > fv[1])
               {
                  DenseMatrix &pm = FaceElemTr.Loc2.Transf.GetPointMat();
                  mfem::Swap<double>(pm(0,0), pm(0,1));
                  mfem::Swap<double>(pm(1,0), pm(1,1));
               }
            }
         }
         break;

      case Element::TRIANGLE:
         // ---------  assumes the face is a triangle -- face of a tetrahedron
         if (mask & 4)
         {
            GetLocalTriToTetTransformation(FaceElemTr.Loc1.Transf,
                                           face_info.Elem1Inf);
         }
         if (FaceElemTr.Elem2No >= 0 && (mask & 8))
         {
            GetLocalTriToTetTransformation(FaceElemTr.Loc2.Transf,
                                           face_info.Elem2Inf);
            if (IsSlaveFace(face_info))
            {
               ApplySlaveTransformation(FaceElemTr.Loc2.Transf, face_info);
            }
         }
         break;

      case Element::QUADRILATERAL:
         // ---------  assumes the face is a quad -- face of a hexahedron
         if (mask & 4)
         {
            GetLocalQuadToHexTransformation(FaceElemTr.Loc1.Transf,
                                            face_info.Elem1Inf);
         }
         if (FaceElemTr.Elem2No >= 0 && (mask & 8))
         {
            GetLocalQuadToHexTransformation(FaceElemTr.Loc2.Transf,
                                            face_info.Elem2Inf);
            if (IsSlaveFace(face_info))
            {
               ApplySlaveTransformation(FaceElemTr.Loc2.Transf, face_info);
            }
         }
         break;
   }

   return &FaceElemTr;
}

bool Mesh::IsSlaveFace(const FaceInfo &fi)
{
   return fi.NCFace >= 0 && nc_faces_info[fi.NCFace].Slave;
}

void Mesh::ApplySlaveTransformation(IsoparametricTransformation &transf,
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

void Mesh::Init()
{
   NumOfVertices = NumOfElements = NumOfBdrElements = NumOfEdges = -1;
   WantTwoLevelState = 0;
   State = Mesh::NORMAL;
   Nodes = NULL;
   own_nodes = 1;
   NURBSext = NULL;
   ncmesh = NULL;
   nc_coarse_level = NULL;
}

void Mesh::InitTables()
{
   el_to_edge =
      el_to_face = el_to_el = bel_to_edge = face_edge = edge_vertex = NULL;
}

void Mesh::DeleteTables()
{
   delete el_to_edge;
   delete el_to_face;
   delete el_to_el;

   if (Dim == 3)
   {
      delete bel_to_edge;
   }

   delete face_edge;
   delete edge_vertex;

   InitTables();

   delete nc_coarse_level;
   nc_coarse_level = NULL;
}

void Mesh::DeleteCoarseTables()
{
   delete el_to_el;
   delete face_edge;
   delete edge_vertex;

   el_to_el = face_edge = edge_vertex = NULL;

   delete nc_coarse_level;
   nc_coarse_level = NULL;
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
   Dim = _Dim;
   spaceDim = _spaceDim;

   Init();
   InitTables();

   NumOfVertices = 0;
   vertices.SetSize(NVert);  // just allocate space for vertices

   NumOfElements = 0;
   elements.SetSize(NElem);  // just allocate space for Element *

   NumOfBdrElements = 0;
   boundary.SetSize(NBdrElem);  // just allocate space for Element *
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

void Mesh::GenerateBoundaryElements()
{
   int i, j;
   Array<int> &be2face = (Dim == 2) ? be_to_edge : be_to_face;

   // GenerateFaces();

   for (i = 0; i < boundary.Size(); i++)
   {
      FreeElement(boundary[i]);
   }

   if (Dim == 3)
   {
      delete bel_to_edge;
      bel_to_edge = NULL;
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

typedef struct
{
   int edge;
   double length;
}
edge_length;

// Used by qsort to sort edges in increasing (according their length) order.
static int edge_compare(const void *ii, const void *jj)
{
   edge_length *i = (edge_length *)ii, *j = (edge_length *)jj;
   if (i->length > j->length) { return (1); }
   if (i->length < j->length) { return (-1); }
   return (0);
}

void Mesh::FinalizeTriMesh(int generate_edges, int refine, bool fix_orientation)
{
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

   meshgen = 1;
}

void Mesh::FinalizeQuadMesh(int generate_edges, int refine,
                            bool fix_orientation)
{
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

   meshgen = 2;
}


#ifdef MFEM_USE_GECKO
void Mesh::GetGeckoElementReordering(Array<int> &ordering)
{
   Gecko::Graph graph;

   //We will put some accesors in for these later
   Gecko::Functional *functional =
      new Gecko::FunctionalGeometric(); // ordering functional
   unsigned int iterations = 1;         // number of V cycles
   unsigned int window = 2;             // initial window size
   unsigned int period = 1;             // iterations between window increment
   unsigned int seed = 0;               // random number seed

   //Run through all the elements and insert the nodes in the graph for them
   for (int elemid = 0; elemid < GetNE(); ++elemid)
   {
      graph.insert();
   }

   //Run through all the elems and insert arcs to the graph for each element face
   //Indices in Gecko are 1 based hence the +1 on the insertion
   const Table &my_el_to_el = ElementToElementTable();
   for (int elemid = 0; elemid < GetNE(); ++elemid)
   {
      const int *neighid = my_el_to_el.GetRow(elemid);
      for (int i = 0; i < my_el_to_el.RowSize(elemid); ++i)
      {
         graph.insert(elemid + 1,  neighid[i] + 1);
      }
   }

   //Get the reordering from Gecko and copy it into the ordering Array<int>
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
   SetState(Mesh::NORMAL);

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

   //Save the locations of the Nodes so we can rebuild them later
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

   //Get the newly ordered elements
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
      //Get the new vertex ordering permutation vectors and fill the new vertices
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

      //Replace the vertex ids in the elements with the reordered vertex numbers
      for (int new_elid = 0; new_elid < GetNE(); ++new_elid)
      {
         int *elem_vert = elements[new_elid]->GetVertices();
         int nv = elements[new_elid]->GetNVertices();
         for (int vi = 0; vi < nv; ++vi)
         {
            elem_vert[vi] = vertex_ordering[elem_vert[vi]];
         }
      }

      //Replace the vertex ids in the boundary with reordered vertex numbers
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

   //Build the nodes from the saved locations if they were around before
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
         MarkTetMeshForRefinement();
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

   order.SetSize(NumOfEdges);
   for (int i = 0; i < NumOfEdges; i++)
   {
      order[length[i].edge] = i;
   }

   delete [] length;
}

void Mesh::MarkTetMeshForRefinement()
{
   // Mark the longest tetrahedral edge by rotating the indices so that
   // vertex 0 - vertex 1 is the longest edge in the element.
   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);
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
}

void Mesh::FinalizeTetMesh(int generate_edges, int refine, bool fix_orientation)
{
   CheckElementOrientation(fix_orientation);

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

void Mesh::FinalizeHexMesh(int generate_edges, int refine, bool fix_orientation)
{
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

   meshgen = 2;
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

   Dim = spaceDim = 2;

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

   Dim = 1;
   spaceDim = 1;

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

   meshgen = mesh.meshgen;

   // Only allow copy of meshes in NORMAL (not TWO_LEVEL_*) state.
   MFEM_VERIFY(mesh.State == NORMAL, "source mesh is not in a NORMAL state");
   State = NORMAL;
   WantTwoLevelState = mesh.WantTwoLevelState;

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

   // Do not copy any of the coarse (c_*), fine (f_*) or fine/coarse (fc_*)
   // data members.

   // Do NOT copy the coarse non-conforming Mesh, 'nc_coarse_level'
   nc_coarse_level = NULL;

   // Copy the attributes and bdr_attributes
   mesh.attributes.Copy(attributes);
   mesh.bdr_attributes.Copy(bdr_attributes);

   // No support for NURBS meshes, yet. Need deep copy for NURBSExtension.
   MFEM_VERIFY(mesh.NURBSext == NULL,
               "copying NURBS meshes is not implemented");
   NURBSext = NULL;

   // No support for non-conforming meshes, yet. Need deep copy for NCMesh.
   MFEM_VERIFY(mesh.ncmesh == NULL,
               "copying non-conforming meshes is not implemented");
   ncmesh = NULL;

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

Mesh::Mesh(std::istream &input, int generate_edges, int refine,
           bool fix_orientation)
{
   Init();
   InitTables();
   Load(input, generate_edges, refine, fix_orientation);
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
            meshgen |= 1; break;

         case Element::QUADRILATERAL:
         case Element::HEXAHEDRON:
            meshgen |= 2;
      }
   }
}

// see Tetrahedron::edges
static const int vtk_quadratic_tet[10] =
{ 0, 1, 2, 3, 4, 7, 5, 6, 8, 9 };

// see Hexahedron::edges & Mesh::GenerateFaces
static const int vtk_quadratic_hex[27] =
{
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
   24, 22, 21, 23, 20, 25, 26
};

void skip_comment_lines(std::istream &is, const char comment_char)
{
   while (1)
   {
      is >> ws;
      if (is.peek() != comment_char)
      {
         break;
      }
      is.ignore(numeric_limits<streamsize>::max(), '\n');
   }
}

void Mesh::Load(std::istream &input, int generate_edges, int refine,
                bool fix_orientation)
{
   int i, j, ints[32], n, attr, curved = 0, read_gf = 1;
   const int buflen = 1024;
   char buf[buflen];

   if (!input)
   {
      MFEM_ABORT("Input stream is not open");
   }

   if (NumOfVertices != -1)
   {
      // Delete the elements.
      for (i = 0; i < NumOfElements; i++)
      {
         FreeElement(elements[i]);
      }
      elements.DeleteAll();
      NumOfElements = 0;

      // Delete the vertices.
      vertices.DeleteAll();
      NumOfVertices = 0;

      // Delete the boundary elements.
      for (i = 0; i < NumOfBdrElements; i++)
      {
         FreeElement(boundary[i]);
      }
      boundary.DeleteAll();
      NumOfBdrElements = 0;

      // Delete interior faces (if generated)
      for (i = 0; i < faces.Size(); i++)
      {
         FreeElement(faces[i]);
      }
      faces.DeleteAll();
      NumOfFaces = 0;

      faces_info.DeleteAll();

      // Delete the edges (if generated).
      DeleteTables();
      be_to_edge.DeleteAll();
      be_to_face.DeleteAll();
      NumOfEdges = 0;

      // TODO: make this a Destroy function
   }

   delete ncmesh;
   ncmesh = NULL;

   if (own_nodes) { delete Nodes; }
   Nodes = NULL;

   InitTables();
   spaceDim = 0;

   string mesh_type;
   input >> ws;
   getline(input, mesh_type);

   bool mfem_v10 = (mesh_type == "MFEM mesh v1.0");
   bool mfem_v11 = (mesh_type == "MFEM mesh v1.1");

   if (mfem_v10 || mfem_v11)
   {
      // Read MFEM mesh v1.0 format
      string ident;

      // read lines beginning with '#' (comments)
      skip_comment_lines(input, '#');
      input >> ident; // 'dimension'

      MFEM_VERIFY(ident == "dimension", "invalid mesh file");
      input >> Dim;

      skip_comment_lines(input, '#');
      input >> ident; // 'elements'

      MFEM_VERIFY(ident == "elements", "invalid mesh file");
      input >> NumOfElements;
      elements.SetSize(NumOfElements);
      for (j = 0; j < NumOfElements; j++)
      {
         elements[j] = ReadElement(input);
      }

      skip_comment_lines(input, '#');
      input >> ident; // 'boundary'

      MFEM_VERIFY(ident == "boundary", "invalid mesh file");
      input >> NumOfBdrElements;
      boundary.SetSize(NumOfBdrElements);
      for (j = 0; j < NumOfBdrElements; j++)
      {
         boundary[j] = ReadElement(input);
      }

      skip_comment_lines(input, '#');
      input >> ident;

      if (ident == "vertex_parents" && mfem_v11)
      {
         ncmesh = new NCMesh(this, &input);
         // NOTE: the constructor above will call LoadVertexParents

         skip_comment_lines(input, '#');
         input >> ident;

         if (ident == "coarse_elements")
         {
            ncmesh->LoadCoarseElements(input);

            skip_comment_lines(input, '#');
            input >> ident;
         }
      }

      MFEM_VERIFY(ident == "vertices", "invalid mesh file");
      input >> NumOfVertices;
      vertices.SetSize(NumOfVertices);

      input >> ws >> ident;
      if (ident != "nodes")
      {
         // read the vertices
         spaceDim = atoi(ident.c_str());
         for (j = 0; j < NumOfVertices; j++)
         {
            for (i = 0; i < spaceDim; i++)
            {
               input >> vertices[j](i);
            }
         }

         // initialize vertex positions in NCMesh
         if (ncmesh) { ncmesh->SetVertexPositions(vertices); }
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
      {
         input >> vertices[j](0);
      }

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
      {
         curved = 1;
      }

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
            {
               input >> vertices[i](j);
            }
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
      // Read a Netgen format mesh of tetrahedra.
      Dim = 3;

      // Read the vertices
      input >> NumOfVertices;

      vertices.SetSize(NumOfVertices);
      for (i = 0; i < NumOfVertices; i++)
         for (j = 0; j < Dim; j++)
         {
            input >> vertices[i](j);
         }

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
         {
            mfem_error("Mesh::Load : VTK mesh does not have POINTS data!");
         }
      }
      while (buff != "POINTS");
      int np = 0;
      Vector points;
      {
         input >> np >> ws;
         points.SetSize(3*np);
         getline(input, buff); // "double"
         for (i = 0; i < points.Size(); i++)
         {
            input >> points(i);
         }
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
         {
            input >> cells_data[i];
         }
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
         // "SCALARS material dataType numComp"
         if (!strncmp(buff.c_str(), "SCALARS material", 16))
         {
            getline(input, buff); // "LOOKUP_TABLE default"
            for (i = 0; i < NumOfElements; i++)
            {
               input >> attr;
               elements[i]->SetAttribute(attr);
            }
         }
         else
         {
            input.seekg(sp);
         }
      }
      else
      {
         input.seekg(sp);
      }

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
               {
                  pts_dof[v[j]] = n++;
               }
         }
         // keep the original ordering of the vertices
         for (n = i = 0; i < np; i++)
            if (pts_dof[i] != -1)
            {
               pts_dof[i] = n++;
            }
         // update the element vertices
         for (i = 0; i < NumOfElements; i++)
         {
            int *v = elements[i]->GetVertices();
            int nv = elements[i]->GetNVertices();
            for (j = 0; j < nv; j++)
            {
               v[j] = pts_dof[v[j]];
            }
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
         {
            NumOfFaces = 0;
         }

         // Generate edges
         el_to_edge = new Table;
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         if (Dim == 2)
         {
            GenerateFaces();   // 'Faces' in 2D refers to the edges
         }

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
               default:
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
               {
                  (*Nodes)(dofs[j]) = points(3*i+j);
               }
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
            {
               vertices[j](i) = vert_val(j);
            }
         }
      }
      else
      {
         read_gf = 1;
      }
   }
   else if (mesh_type == "MFEM INLINE mesh v1.0")
   {
      // Initialize to negative numbers so that we know if they've
      // been set.  We're using Element::POINT as our flag, since
      // we're not going to make a 0D mesh, ever.
      int nx = -1;
      int ny = -1;
      int nz = -1;
      double sx = -1.0;
      double sy = -1.0;
      double sz = -1.0;
      Element::Type type = Element::POINT;

      while (true)
      {
         skip_comment_lines(input, '#');
         // Break out if we reached the end of the file after
         // gobbling up the whitespace and comments after the last keyword.
         if (!input.good())
         {
            break;
         }

         // Read the next keyword
         std::string name;
         input >> name;
         input >> std::ws;
         // Make sure there's an equal sign
         MFEM_VERIFY(input.get() == '=',
                     "Inline mesh expected '=' after keyword " << name);
         input >> std::ws;

         if (name == "nx")
         {
            input >> nx;
         }
         else if (name == "ny")
         {
            input >> ny;
         }
         else if (name == "nz")
         {
            input >> nz;
         }
         else if (name == "sx")
         {
            input >> sx;
         }
         else if (name == "sy")
         {
            input >> sy;
         }
         else if (name == "sz")
         {
            input >> sz;
         }
         else if (name == "type")
         {
            std::string eltype;
            input >> eltype;
            if (eltype == "segment")
            {
               type = Element::SEGMENT;
            }
            else if (eltype == "quad")
            {
               type = Element::QUADRILATERAL;
            }
            else if (eltype == "tri")
            {
               type = Element::TRIANGLE;
            }
            else if (eltype == "hex")
            {
               type = Element::HEXAHEDRON;
            }
            else if (eltype == "tet")
            {
               type = Element::TETRAHEDRON;
            }
            else
            {
               MFEM_ABORT("unrecognized element type (read '" << eltype
                          << "') in inline mesh format.  "
                          "Allowed: segment, tri, tet, quad, hex");
            }
         }
         else
         {
            MFEM_ABORT("unrecognized keyword (" << name
                       << ") in inline mesh format.  "
                       "Allowed: nx, ny, nz, type, sx, sy, sz");
         }

         input >> std::ws;
         // Allow an optional semi-colon at the end of each line.
         if (input.peek() == ';')
         {
            input.get();
         }

         // Done reading file
         if (!input)
         {
            break;
         }
      }

      // Now make the mesh.
      if (type == Element::SEGMENT)
      {
         MFEM_VERIFY(nx > 0 && sx > 0.0,
                     "invalid 1D inline mesh format, all values must be "
                     "positive\n"
                     << "   nx = " << nx << "\n"
                     << "   sx = " << sx << "\n");
         Make1D(nx, sx);
      }
      else if (type == Element::TRIANGLE || type == Element::QUADRILATERAL)
      {
         MFEM_VERIFY(nx > 0 && ny > 0 && sx > 0.0 && sy > 0.0,
                     "invalid 2D inline mesh format, all values must be "
                     "positive\n"
                     << "   nx = " << nx << "\n"
                     << "   ny = " << ny << "\n"
                     << "   sx = " << sx << "\n"
                     << "   sy = " << sy << "\n");
         Make2D(nx, ny, type, generate_edges, sx, sy);
      }
      else if (type == Element::TETRAHEDRON || type == Element::HEXAHEDRON)
      {
         MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0 &&
                     sx > 0.0 && sy > 0.0 && sz > 0.0,
                     "invalid 3D inline mesh format, all values must be "
                     "positive\n"
                     << "   nx = " << nx << "\n"
                     << "   ny = " << ny << "\n"
                     << "   nz = " << nz << "\n"
                     << "   sx = " << sx << "\n"
                     << "   sy = " << sy << "\n"
                     << "   sz = " << sz << "\n");
         Make3D(nx, ny, nz, type, generate_edges, sx, sy, sz);
      }
      else
      {
         mfem_error("Mesh::Load : For inline mesh, must specify an "
                    "element type = [segment, tri, quad, tet, hex]");
      }
      return; // done with inline mesh construction
   }
   else
   {
      MFEM_ABORT("Unknown input mesh format");
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

   if (spaceDim == 0)
   {
      spaceDim = Dim;
   }

   // set the mesh type ('meshgen')
   SetMeshGen();

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
      GetElementToFaceTable();
      GenerateFaces();
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
      c_el_to_edge = NULL;
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

   // generate the arrays 'attributes' and ' bdr_attributes'
   SetAttributes();

   if (curved)
   {
      if (read_gf)
      {
         Nodes = new GridFunction(this, input);
         own_nodes = 1;
         spaceDim = Nodes->VectorDim();
         // Set the 'vertices' from the 'Nodes'
         for (i = 0; i < spaceDim; i++)
         {
            Vector vert_val;
            Nodes->GetNodalValues(vert_val, i+1);
            for (j = 0; j < NumOfVertices; j++)
            {
               vertices[j](i) = vert_val(j);
            }
         }
      }

      // Check orientation and mark edges; only for triangles / tets
      if (meshgen & 1)
      {
         DSTable *old_v_to_v = NULL;
         Table *old_elem_vert = NULL;
         if (fix_orientation || refine)
         {
            PrepareNodeReorder(&old_v_to_v, &old_elem_vert);
         }

         // check orientation and mark for refinement using just vertices
         // (i.e. higher order curvature is not used)
         CheckElementOrientation(fix_orientation);
         if (refine)
         {
            MarkForRefinement();   // changes topology!
         }

         if (fix_orientation || refine)
         {
            DoNodeReorder(old_v_to_v, old_elem_vert);
            delete old_elem_vert;
            delete old_v_to_v;
         }
      }
   }

   if (ncmesh) { ncmesh->spaceDim = spaceDim; }
}

Mesh::Mesh(Mesh *mesh_array[], int num_pieces)
{
   int      i, j, ie, ib, iv, *v, nv;
   Element *el;
   Mesh    *m;

   Init();
   InitTables();

   Dim = mesh_array[0]->Dimension();
   spaceDim = mesh_array[0]->SpaceDimension();

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
      // the individual pieces which include the interior boundaries.
      // This creates 'boundary' array that is different from the one generated
      // by the NURBSExtension which, in particular, makes the boundary-dof
      // table invalid. This, in turn, causes GetBdrElementTransformation to
      // not function properly.
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
            vertices[lvert_vert[j]].SetCoords(m->GetVertex(j));
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
            vertices[iv++].SetCoords(m->GetVertex(j));
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
   Init();
   InitTables();

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

const FiniteElementSpace *Mesh::GetNodalFESpace()
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
   }
   return 0;
}

#if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
static const char *fixed_or_not[] = { "fixed", "NOT FIXED" };
#endif

void Mesh::CheckElementOrientation(bool fix_it)
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
}

int Mesh::GetTriOrientation(const int *base, const int *test)
{
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
   const int *aor = tri_orientations[orient];
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
   const int *aor = quad_orientations[orient];
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

void Mesh::CheckBdrElementOrientation(bool fix_it)
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
                     v[j] = ev[hex_faces[lf][j]];
                  }
                  if (GetQuadOrientation(v, bv) % 2)
                  {
                     if (fix_it)
                     {
                        mfem::Swap<int>(bv[0], bv[2]);
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
   else if (Dim == 3)
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

   if (Dim != 3)
   {
      return;
   }

   MFEM_ASSERT(State != TWO_LEVEL_COARSE, "internal MFEM error!");

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
   MFEM_ASSERT(State != TWO_LEVEL_COARSE, "internal MFEM error!");

   // the two vertices are sorted: vert[0] < vert[1]
   // this is consistent with the global edge orientation
   GetEdgeVertexTable(); // generate edge_vertex Table (if not generated)
   edge_vertex->GetRow(i, vert);
}

Table *Mesh::GetFaceEdgeTable() const
{
   if (face_edge)
   {
      return face_edge;
   }

   if (Dim != 3)
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
      case Element::SEGMENT:
         return Geometry::POINT;

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
   return (-1);
}

int Mesh::GetBdrElementEdgeIndex(int i) const
{
   switch (Dim)
   {
      case 1: return boundary[i]->GetVertices()[0];
      case 2: return be_to_edge[i];
      case 3: return be_to_face[i];
      default: mfem_error("Mesh::GetBdrElementEdgeIndex: invalid dimension!");
   }
   return -1;
}

int Mesh::GetElementType(int i) const
{
   Element *El = elements[i];
   int t = El->GetType();

   while (1)
      if (t == Element::BISECTED     ||
          t == Element::QUADRISECTED ||
          t == Element::OCTASECTED)
      {
         t = (El = ((RefinedElement *) El)->IAm())->GetType();
      }
      else
      {
         break;
      }
   return t;
}

int Mesh::GetBdrElementType(int i) const
{
   Element *El = boundary[i];
   int t = El->GetType();

   while (1)
      if (t == Element::BISECTED || t == Element::QUADRISECTED)
      {
         t = (El = ((RefinedElement *) El)->IAm())->GetType();
      }
      else
      {
         break;
      }
   return t;
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
   else if (Dim == 3)
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
   MFEM_ASSERT(faces_info.Size() == num_faces, "faces were not generated!");

   Array<Connection> conn;
   conn.Reserve(2*num_faces);

   for (int i = 0; i < faces_info.Size(); i++)
   {
      const FaceInfo &fi = faces_info[i];
      if (fi.Elem2No >= 0)
      {
         conn.Append(Connection(fi.Elem1No, fi.Elem2No));
         conn.Append(Connection(fi.Elem2No, fi.Elem1No));
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

void Mesh::GenerateFaces()
{
   int i, nfaces = GetNumFaces();

   for (i = 0; i < faces.Size(); i++)
   {
      FreeElement(faces[i]);
   }

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
      else
      {
         ef = el_to_face->GetRow(i);
         switch (GetElementType(i))
         {
            case Element::TETRAHEDRON:
            {
               for (int j = 0; j < 4; j++)
               {
                  const int *fv = tet_faces[j];
                  AddTriangleFaceElement(j, ef[j], i,
                                         v[fv[0]], v[fv[1]], v[fv[2]]);
               }
               break;
            }
            case Element::HEXAHEDRON:
            {
               for (int j = 0; j < 6; j++)
               {
                  const int *fv = hex_faces[j];
                  AddQuadFaceElement(j, ef[j], i,
                                     v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
               }
               break;
            }
            default:
               MFEM_ABORT("Unexpected type of Element.");
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

   // add records for master faces
   for (unsigned i = 0; i < list.masters.size(); i++)
   {
      const NCMesh::Master &master = list.masters[i];
      faces_info[master.index].NCFace = nc_faces_info.Size();
      nc_faces_info.Append(NCFaceInfo(false, master.local, NULL));
      // NOTE: one of the unused members stores local face no. to be used below
   }

   // add records for slave faces
   for (unsigned i = 0; i < list.slaves.size(); i++)
   {
      const NCMesh::Slave &slave = list.slaves[i];
      FaceInfo &slave_fi = faces_info[slave.index];
      FaceInfo &master_fi = faces_info[slave.master];
      NCFaceInfo &master_nc = nc_faces_info[master_fi.NCFace];

      slave_fi.NCFace = nc_faces_info.Size();
      nc_faces_info.Append(NCFaceInfo(true, slave.master, &slave.point_matrix));

      slave_fi.Elem2No = master_fi.Elem1No;
      slave_fi.Elem2Inf = 64 * master_nc.MasterFace; // get lf no. stored above
      // NOTE: orientation part of Elem2Inf is encoded in the point matrix
      if (Dim == 2)
      {
         const int *fv = faces[slave.master]->GetVertices();
         if (fv[0] > fv[1])
         {
            slave_fi.Elem2Inf++;
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
         {
            for (int j = 0; j < 4; j++)
            {
               const int *fv = tet_faces[j];
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
               const int *fv = hex_faces[j];
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
               const int *fv = tet_faces[j];
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
               const int *fv = hex_faces[j];
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

   DeleteCoarseTables();

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

#ifdef MFEM_USE_MPI
// auxiliary function for qsort
static int mfem_less(const void *x, const void *y)
{
   if (*(int*)x < *(int*)y)
   {
      return 1;
   }
   if (*(int*)x > *(int*)y)
   {
      return -1;
   }
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
      options[10] = 1; // set METIS_OPTION_CONTIG
#endif

      // Sort the neighbor lists
      if (part_method >= 0 && part_method <= 2)
         for (i = 0; i < n; i++)
         {
            qsort(&J[I[i]], I[i+1]-I[i], sizeof(int), &mfem_less);
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
   Nodes->FESpace()->UpdateAndInterpolate(Nodes);
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
         mfem::Swap(be_to_edge, fc_be_to_edge); // save coarse be_to_edge
         f_el_to_edge = new Table;
         NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
         el_to_edge = f_el_to_edge;
         f_NumOfEdges = NumOfEdges;
      }
      else
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }
      GenerateFaces();
   }

   if (Nodes)  // curved mesh
   {
      UpdateNodes();
      UseTwoLevelState(wtls);
   }

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
   CheckBdrElementOrientation(false);
#endif
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
   {
      GetElementToFaceTable();
   }

   int oedge = NumOfVertices;
   int oface = oedge + NumOfEdges;
   int oelem = oface + NumOfFaces;

   DeleteCoarseTables();

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
            vv[k] = v[hex_faces[j][k]];
         }
         AverageVertices(vv, 4, oface+f[j]);
      }

      e = el_to_edge->GetRow(i);

      for (int j = 0; j < 12; j++)
      {
         for (int k = 0; k < 2; k++)
         {
            vv[k] = v[Hexahedron::edges[j][k]];
         }
         AverageVertices(vv, 2, oedge+e[j]);
      }
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
         {
            oe->Child[k] = j + k;
         }
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

   if (el_to_face != NULL)
   {
      if (WantTwoLevelState)
      {
         c_el_to_face = el_to_face;
         el_to_face = NULL;
         mfem::Swap(faces_info, fc_faces_info);
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
   CheckBdrElementOrientation(false);
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
         f_NumOfEdges = NumOfEdges;
      }
      else
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }
   }

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

   if (ncmesh)
   {
      mfem_error("Local and nonconforming refinements cannot be mixed.");
   }

   if (Nodes)  // curved mesh
   {
      UseTwoLevelState(1);
   }

   SetState(Mesh::NORMAL);
   DeleteCoarseTables();

   if (Dim == 1) // --------------------------------------------------------
   {
      if (WantTwoLevelState)
      {
         c_NumOfVertices    = NumOfVertices;
         c_NumOfElements    = NumOfElements;
         c_NumOfBdrElements = NumOfBdrElements;
         c_NumOfEdges = 0;
      }
      int cne = NumOfElements, cnv = NumOfVertices;
      NumOfVertices += marked_el.Size();
      NumOfElements += marked_el.Size();
      vertices.SetSize(NumOfVertices);
      elements.SetSize(NumOfElements);
      for (j = 0; j < marked_el.Size(); j++)
      {
         i = marked_el[j];
         Segment *c_seg = (Segment *)elements[i];
         int *vert = c_seg->GetVertices(), attr = c_seg->GetAttribute();
         int new_v = cnv + j, new_e = cne + j;
         AverageVertices(vert, 2, new_v);
         elements[new_e] = new Segment(new_v, vert[1], attr);
         if (WantTwoLevelState)
         {
#ifdef MFEM_USE_MEMALLOC
            BisectedElement *aux = BEMemory.Alloc();
            aux->SetCoarseElem(c_seg);
#else
            BisectedElement *aux = new BisectedElement(c_seg);
#endif
            aux->FirstChild = new Segment(vert[0], new_v, attr);
            aux->SecondChild = new_e;
            elements[i] = aux;
         }
         else
         {
            vert[1] = new_v;
         }
      }
      if (WantTwoLevelState)
      {
         f_NumOfVertices    = NumOfVertices;
         f_NumOfElements    = NumOfElements;
         f_NumOfBdrElements = NumOfBdrElements;
         f_NumOfEdges = 0;

         RefinedElement::State = RefinedElement::FINE;
         State = Mesh::TWO_LEVEL_FINE;
      }
      GenerateFaces();
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
            mfem::Swap(be_to_edge, fc_be_to_edge); // save coarse be_to_edge
            f_el_to_edge = new Table;
            NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
            el_to_edge = f_el_to_edge;
            f_NumOfEdges = NumOfEdges;
         }
         else
         {
            NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         }
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
         {
            El = ((BisectedElement *) El)->FirstChild;
         }
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
         {
            NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         }
      }
      if (el_to_face != NULL)
      {
         if (WantTwoLevelState)
         {
            c_el_to_face = el_to_face;
            el_to_face = NULL;
            mfem::Swap(faces_info, fc_faces_info);
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

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
#endif
}

void Mesh::NonconformingRefinement(const Array<Refinement> &refinements,
                                   int nc_limit)
{
   if (NURBSext)
   {
      MFEM_ABORT("Mesh::NonconformingRefinement: NURBS meshes are not supported."
                 " Project the NURBS to Nodes first.");
   }

   int wtls = WantTwoLevelState;

   if (Nodes) // curved mesh
   {
      UseTwoLevelState(1);
   }

   SetState(Mesh::NORMAL);
   DeleteCoarseTables();

   if (!ncmesh)
   {
      // start tracking refinement hierarchy
      ncmesh = new NCMesh(this);
   }

   if (WantTwoLevelState)
   {
      ncmesh->MarkCoarseLevel();
   }

   // do the refinements
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

   // retain the coarse mesh if two-level state was requested, delete otherwise
   if (WantTwoLevelState)
   {
      nc_coarse_level = mesh2;
      State = TWO_LEVEL_FINE;
   }
   else
   {
      delete mesh2;
   }

   GenerateNCFaceInfo();

   if (Nodes) // curved mesh
   {
      UpdateNodes();
      UseTwoLevelState(wtls);
   }
}

void Mesh::InitFromNCMesh(const NCMesh &ncmesh)
{
   Dim = ncmesh.Dimension();
   spaceDim = ncmesh.SpaceDimension();

   DeleteTables();

   ncmesh.GetMeshComponents(vertices, elements, boundary);

   NumOfVertices = vertices.Size();
   NumOfElements = elements.Size();
   NumOfBdrElements = boundary.Size();

   SetMeshGen(); // set the mesh type ('meshgen')

   NumOfEdges = NumOfFaces = 0;
   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      c_el_to_edge = NULL;
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

   // NOTE: two-level-state related members are ignored here
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

      if (!ncmesh)
      {
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
   if (Dim == 1)
   {
      nonconforming = 0;
   }
   else if (nonconforming < 0)
   {
      // determine if nonconforming refinement is suitable
      int type = elements[0]->GetType();
      if (type == Element::HEXAHEDRON || type == Element::QUADRILATERAL)
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
      Array<int> el_to_refine;
      for (int i = 0; i < refinements.Size(); i++)
      {
         el_to_refine.Append(refinements[i].index);
      }

      // infer 'type' of local refinement from first element's 'ref_type'
      int type, rt = (refinements.Size() ? refinements[0].ref_type : 7);
      if (rt == 1 || rt == 2 || rt == 4)
      {
         type = 1;
      }
      else if (rt == 3 || rt == 5 || rt == 6)
      {
         type = 2;
      }
      else
      {
         type = 3;
      }

      // red-green refinement, no hanging nodes
      LocalRefinement(el_to_refine, type);
   }
}

void Mesh::GeneralRefinement(const Array<int> &el_to_refine, int nonconforming,
                             int nc_limit)
{
   Array<Refinement> refinements;
   for (int i = 0; i < el_to_refine.Size(); i++)
   {
      refinements.Append(Refinement(el_to_refine[i], 7));
   }
   GeneralRefinement(refinements, nonconforming, nc_limit);
}

void Mesh::EnsureNCMesh()
{
   if (meshgen & 2)
   {
      Array<int> empty;
      GeneralRefinement(empty, 1);
   }
}

void Mesh::RandomRefinement(int levels, int frac, bool aniso,
                            int nonconforming, int nc_limit, int seed)
{
   srand(seed);
   for (int i = 0; i < levels; i++)
   {
      Array<Refinement> refs;
      for (int j = 0; j < GetNE(); j++)
      {
         if (!(rand() % frac))
         {
            int type = 7;
            if (aniso)
            {
               type = (Dim == 3) ? (rand() % 7 + 1) : (rand() % 3 + 1);
            }
            refs.Append(Refinement(j, type));
         }
      }
      GeneralRefinement(refs, nonconforming, nc_limit);
   }
}

void Mesh::RefineAtVertex(const Vertex& vert, int levels, double eps,
                          int nonconforming)
{
   Array<int> v;
   for (int k = 0; k < levels; k++)
   {
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
}

void Mesh::Bisection(int i, const DSTable &v_to_v,
                     int *edge1, int *edge2, int *middle)
{
   int *vert;
   int v[2][4], v_new, bisect, t;
   Element **pce = &elements[i];
   Vertex V;

   t = pce[0]->GetType();
   if (WantTwoLevelState)
   {
      while (1)
      {
         if (t == Element::BISECTED)
         {
            pce = & ( ((BisectedElement *) pce[0])->FirstChild );
         }
         else if (t == Element::QUADRISECTED)
         {
            pce = & ( ((QuadrisectedElement *) pce[0])->FirstChild );
         }
         else
         {
            break;
         }
         t = pce[0]->GetType();
      }
   }

   if (t == Element::TRIANGLE)
   {
      Triangle *tri = (Triangle *) pce[0];

      vert = tri->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v(vert[0], vert[1]);
#ifdef MFEM_DEBUG
      if (bisect < 0)
      {
         mfem_error("Mesh::Bisection(...) of triangle! #1");
      }
#endif
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
      {
         tri->SetVertices(v[0]);   // changes vert[0..2] !!!
      }
      elements.Append(new Triangle(v[1], tri->GetAttribute()));

      // 3. edge1 and edge2 may have to be changed for the second triangle.
      if (v[1][0] < v_to_v.NumberOfRows() && v[1][1] < v_to_v.NumberOfRows())
      {
         bisect = v_to_v(v[1][0], v[1][1]);
#ifdef MFEM_DEBUG
         if (bisect < 0)
         {
            mfem_error("Mesh::Bisection(...) of triangle! #2");
         }
#endif
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
      Tetrahedron *tet = (Tetrahedron *) pce[0];

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
      switch (old_redges[0])
      {
         case 2:
            v[0][0] = vert[0]; v[0][1] = vert[2]; v[0][2] = vert[3];
            if (type == Tetrahedron::TYPE_PF) { new_redges[0][1] = 4; }
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
            if (type == Tetrahedron::TYPE_PF) { new_redges[1][0] = 3; }
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
         // 'tet' now points to the first child
      }
      else
      {
         tet->SetVertices(v[0]);
      }

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
   {
      mfem_error("Bisection for now works only for triangles & tetrahedra.");
   }
}

void Mesh::Bisection(int i, const DSTable &v_to_v, int *middle)
{
   int *vert;
   int v[2][3], v_new, bisect, t;
   Element **pce = &boundary[i];

   t = pce[0]->GetType();
   if (WantTwoLevelState)
   {
      while (1)
      {
         if (t == Element::BISECTED)
         {
            pce = & ( ((BisectedElement *) pce[0])->FirstChild );
         }
         else if (t == Element::QUADRISECTED)
         {
            pce = & ( ((QuadrisectedElement *) pce[0])->FirstChild );
         }
         else
         {
            break;
         }
         t = pce[0]->GetType();
      }
   }

   if (t == Element::TRIANGLE)
   {
      Triangle *tri = (Triangle *) pce[0];

      vert = tri->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v(vert[0], vert[1]);
#ifdef MFEM_DEBUG
      if (bisect < 0)
      {
         mfem_error("Mesh::Bisection(...) of boundary triangle! #1");
      }
#endif
      v_new = middle[bisect];
#ifdef MFEM_DEBUG
      if (v_new == -1)
      {
         mfem_error("Mesh::Bisection(...) of boundary triangle! #2");
      }
#endif

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
         // 'tri' now points to the first child
      }
      else
      {
         tri->SetVertices(v[0]);
      }
      boundary.Append(new Triangle(v[1], tri->GetAttribute()));

      NumOfBdrElements++;
   }
   else
   {
      mfem_error("Bisection of boundary elements works only for triangles!");
   }
}

void Mesh::UniformRefinement(int i, const DSTable &v_to_v,
                             int *edge1, int *edge2, int *middle)
{
   Array<int> v;
   int j, v1[3], v2[3], v3[3], v4[3], v_new[3], bisect[3];
   Vertex V;

   if (elements[i]->GetType() == Element::TRIANGLE)
   {
      elements[i]->GetVertices(v);

      // 1. Get the indeces for the new vertices in array v_new
      bisect[0] = v_to_v(v[0],v[1]);
      bisect[1] = v_to_v(v[1],v[2]);
      bisect[2] = v_to_v(v[0],v[2]);
#ifdef MFEM_DEBUG
      if (bisect[0] < 0 || bisect[1] < 0 || bisect[2] < 0)
      {
         mfem_error("Mesh::UniformRefinement(...): ERROR");
      }
#endif

      for (j = 0; j < 3; j++)                // for the 3 edges fix v_new
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
   {
      mfem_error("Uniform refinement for now works only for triangles.");
   }
}

void Mesh::SetState(int s)
{
   if (ncmesh)
   {
      if (State != Mesh::NORMAL && s == Mesh::NORMAL)
      {
         delete nc_coarse_level;
         nc_coarse_level = NULL;
         ncmesh->ClearCoarseLevel();
         State = s;
      }
      else if ((State == Mesh::TWO_LEVEL_COARSE && s == Mesh::TWO_LEVEL_FINE) ||
               (State == Mesh::TWO_LEVEL_FINE && s == Mesh::TWO_LEVEL_COARSE))
      {
         this->Swap(*nc_coarse_level, false);
         State = s;
      }
      else if (State != s)
      {
         mfem_error("Oops! Mesh::SetState");
      }
   }
   else
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
            {
               i++;
            }
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
            {
               i++;
            }
         }

         if (el_to_edge != NULL)
         {
            delete c_el_to_edge;
            el_to_edge = f_el_to_edge;
            if (Dim == 2)
            {
               if (State == Mesh::TWO_LEVEL_COARSE)
               {
                  mfem::Swap(be_to_edge, fc_be_to_edge);
               }
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
            {
               mfem::Swap(faces_info, fc_faces_info);
            }
            fc_faces_info.DeleteAll();
         }

         NumOfVertices    = f_NumOfVertices;
         NumOfEdges       = f_NumOfEdges;
         if (Dim == 3)
         {
            NumOfFaces    = f_NumOfFaces;
         }
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
            {
               mfem::Swap(be_to_edge, fc_be_to_edge);
            }
            if (Dim == 3)
            {
               bel_to_edge = f_bel_to_edge;
            }
         }
         if (el_to_face != NULL)
         {
            el_to_face = f_el_to_face;
            mfem::Swap(faces_info, fc_faces_info);
         }
         NumOfVertices    = f_NumOfVertices;
         NumOfEdges       = f_NumOfEdges;
         if (Dim == 3)
         {
            NumOfFaces    = f_NumOfFaces;
         }
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
            {
               mfem::Swap(be_to_edge, fc_be_to_edge);
            }
            if (Dim == 3)
            {
               bel_to_edge = c_bel_to_edge;
            }
         }
         if (el_to_face != NULL)
         {
            el_to_face = c_el_to_face;
            mfem::Swap(faces_info, fc_faces_info);
         }
         NumOfVertices    = c_NumOfVertices;
         NumOfEdges       = c_NumOfEdges;
         if (Dim == 3)
         {
            NumOfFaces    = c_NumOfFaces;
         }
         NumOfElements    = c_NumOfElements;
         NumOfBdrElements = c_NumOfBdrElements;
         RefinedElement::State = RefinedElement::COARSE;
         State = s;
      }
      else if (State != s)
      {
         mfem_error("Oops! Mesh::SetState");
      }
   }
}

int Mesh::GetNumFineElems(int i)
{
   int t = elements[i]->GetType();

   if (Dim == 1)
   {
      if (t == Element::BISECTED)
      {
         return 2;
      }
   }
   else if (Dim == 2)
   {
      if (t == Element::QUADRISECTED)
      {
         return 4;
      }
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
            {
               break;
            }
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
      if (t == Element::BISECTED)
      {
         int n = 1;
         BisectedElement *aux = (BisectedElement *) elements[i];
         do
         {
            n += GetNumFineElems (aux->SecondChild);
            if (aux->FirstChild->GetType() != Element::BISECTED)
            {
               break;
            }
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         return n;
      }
      else if (t == Element::OCTASECTED)
      {
         return 8;
      }
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

   if (Dim == 1)
   {
      if (elements[i]->GetType() == Element::BISECTED)
      {
         return 1;   // refinement type for bisected SEGMENT
      }
   }
   else if (Dim == 2)
   {
      t = elements[i]->GetType();
      if (t == Element::QUADRISECTED)
      {
         t = ((QuadrisectedElement *)elements[i])->CoarseElem->GetType();
         if (t == Element::QUADRILATERAL)
         {
            return 1;   //  refinement type for quadrisected QUADRILATERAL
         }
         else
         {
            return 2;   //  refinement type for quadrisected TRIANGLE
         }
      }
      else if (t == Element::BISECTED)
      {
         int type;
         type = GetBisectionHierarchy(elements[i]);
         if (type == 0)
         {
            mfem_error("Mesh::GetRefinementType(...)");
         }
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
         {
            return 1;   //  refinement type for octasected CUBE
         }
         else
         {
            return 0;
         }
      }
      // Bisected TETRAHEDRON
      tet = (Tetrahedron *) (((BisectedElement *) E)->CoarseElem);
      tet->ParseRefinementFlag(redges, type, flag);
      if (type == Tetrahedron::TYPE_A && redges[0] == 2)
      {
         type = 5;
      }
      else if (type == Tetrahedron::TYPE_M && redges[0] == 2)
      {
         type = 6;
      }
      type++;
      type |= ( GetBisectionHierarchy(E) << 3 );
      if (type < 8) { type = 0; }

      return type;
   }

   return 0;  // no refinement
}

int Mesh::GetFineElem(int i, int j)
{
   int t = elements[i]->GetType();

   if (Dim == 1)
   {
      if (t == Element::BISECTED)
      {
         switch (j)
         {
            case 0:  return i;
            default: return ((BisectedElement *)elements[i])->SecondChild;
         }
      }
   }
   else if (Dim == 2)
   {
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
               default:  mfem_error("Mesh::GetFineElem #1");
            }
         else // quadrisected TRIANGLE
            switch (j)
            {
               case 0:   return aux->Child2;
               case 1:   return aux->Child3;
               case 2:   return aux->Child4;
               case 3:   return i;
               default:  mfem_error("Mesh::GetFineElem #2");
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
            {
               return k;
            }
            n -= k;  // (-k) is the number of the leaves in this SecondChild
            //   n  is the number of the leaves in
            //      the SecondChild-ren so far
            if (aux->FirstChild->GetType() != Element::BISECTED)
            {
               break;
            }
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         if (j > n)  //  i.e. if (j >= n+1)
         {
            return -(n+1);
         }
         return i;  //  j == n, i.e. j is the index of the last leaf
      }
   }
   else if (Dim == 3)
   {
      if (t == Element::BISECTED)
      {
         int n = 0;
         BisectedElement *aux = (BisectedElement *) elements[i];
         do
         {
            int k = GetFineElem(aux->SecondChild, j-n);
            if (k >= 0)
            {
               return k;
            }
            n -= k;  // (-k) is the number of the leaves in this SecondChild
            //   n  is the number of the leaves in
            //      the SecondChild-ren so far
            if (aux->FirstChild->GetType() != Element::BISECTED)
            {
               break;
            }
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         if (j > n)  //  i.e. if (j >= n+1)
         {
            return -(n+1);
         }
         return i;  //  j == n, i.e. j is the index of the last leaf
      }
      else if (t == Element::OCTASECTED)
      {
         if (j == 0) { return i; }
         return ((OctasectedElement *) elements[i])->Child[j-1];
      }
   }

   if (j > 0)
   {
      return -1;
   }

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
         case 5:
         default: ind[0] = 2; ind[1] = 3; ind[2] = 0; ind[3] = 1;
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
         case 5:
         default: ind[0] = 3; ind[1] = 2; ind[2] = 1; ind[3] = 0;
      }
   }
   // Do the permutation
   for (i = 0; i < 3; i++)
   {
      for (j = 0; j < 4; j++)
      {
         t[j] = pointmat(i,j);
      }
      for (j = 0; j < 4; j++)
      {
         pointmat(i,ind[j]) = t[j];
      }
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
            {
               return ((k << 1)+1) << l;
            }
            n -= k;  // (-k) is the number of the leaves in this SecondChild
            //   n  is the number of the leaves in
            //      the SecondChild-ren so far
            l++;
            if (aux->FirstChild->GetType() != Element::BISECTED)
            {
               break;
            }
            aux = (BisectedElement *) (aux->FirstChild);
         }
         while (1);
         if (j > n)  //  i.e. if (j >= n+1)
         {
            return -(n+1);
         }
         return 0;  //  j == n, i.e. j is the index of the last leaf
      }
      if (j > 0)
      {
         return -1;
      }
   }

   return 0;
}

ElementTransformation * Mesh::GetFineElemTrans(int i, int j)
{
   int t;

   if (Dim == 1)
   {
      DenseMatrix &pm = Transformation.GetPointMat();
      Transformation.SetFE(&SegmentFE);
      Transformation.Attribute = 0;
      Transformation.ElementNo = 0;
      pm.SetSize(1, 2);
      if (elements[i]->GetType() == Element::BISECTED)
      {
         switch (j)
         {
            case 0:  pm(0,0) = 0.0;  pm(0,1) = 0.5;  break;
            default: pm(0,0) = 0.5;  pm(0,1) = 1.0;  break;
         }
      }
      else
      {
         pm(0,0) = 0.0;  pm(0,1) = 1.0;
      }
   }
   else if (Dim == 2)
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
         // identity transformation
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
         if (j < 4) { dz = 0.0; }
         else { dz = 0.5; }
         jj = j % 4;
         if (jj < 2) { dy = 0.0; }
         else { dy = 0.5; }
         if (jj == 0 || jj == 3) { dx = 0.0; }
         else { dx = 0.5; }
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

void Mesh::Print(std::ostream &out) const
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

   out << (ncmesh ? "MFEM mesh v1.1\n" : "MFEM mesh v1.0\n");

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
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
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
      out << "FIELD FieldData 1" << endl
          << "MaterialIds " << 1 << " " << attributes.Size() << " int" << endl;
      for (int i = 0; i < attributes.Size(); i++)
      {
         out << attributes[i] << " ";
      }
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
            nbe += 2;
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
            out << l+1 << ' ' << faces[i]->GetGeometryType();
            for (j = nv-1; j >= 0; j--)
            {
               out << ' ' << v[j];
            }
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

   SetState(Mesh::NORMAL);

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

   SetState(Mesh::NORMAL);

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

   SetState(Mesh::NORMAL);

   if (own_nodes) { delete Nodes; }

   delete ncmesh;

   delete NURBSext;

   for (i = 0; i < NumOfElements; i++)
   {
      FreeElement(elements[i]);
   }

   for (i = 0; i < NumOfBdrElements; i++)
   {
      FreeElement(boundary[i]);
   }

   for (i = 0; i < faces.Size(); i++)
   {
      FreeElement(faces[i]);
   }

   DeleteTables();
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

}
