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
#include <functional>

// Include the METIS header, if using version 5. If using METIS 4, the needed
// declarations are inlined below, i.e. no header is needed.
#if defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
#include "metis.h"
#endif

// METIS 4 prototypes
#if defined(MFEM_USE_METIS) && !defined(MFEM_USE_METIS_5)
typedef int idx_t;
typedef int idxtype;
extern "C" {
   void METIS_PartGraphRecursive(int*, idxtype*, idxtype*, idxtype*, idxtype*,
                                 int*, int*, int*, int*, int*, idxtype*);
   void METIS_PartGraphKway(int*, idxtype*, idxtype*, idxtype*, idxtype*,
                            int*, int*, int*, int*, int*, idxtype*);
   void METIS_PartGraphVKway(int*, idxtype*, idxtype*, idxtype*, idxtype*,
                             int*, int*, int*, int*, int*, idxtype*);
}
#endif

#ifdef MFEM_USE_GECKO
#include "graph.h"
#endif

using namespace std;

namespace mfem
{

void Mesh::GetElementJacobian(int i, DenseMatrix &J)
{
   Geometry::Type geom = GetElementBaseGeometry(i);
   ElementTransformation *eltransf = GetElementTransformation(i);
   eltransf->SetIntPoint(&Geometries.GetCenter(geom));
   Geometries.JacToPerfJac(geom, eltransf->Jacobian(), J);
}

void Mesh::GetElementCenter(int i, Vector &cent)
{
   cent.SetSize(spaceDim);
   int geom = GetElementBaseGeometry(i);
   ElementTransformation *eltransf = GetElementTransformation(i);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
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
      min(d) = infinity();
      max(d) = -infinity();
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

   if (Vh) { Vh->SetSize(NumOfElements); }
   if (Vk) { Vk->SetSize(NumOfElements); }

   h_min = kappa_min = infinity();
   h_max = kappa_max = -h_min;
   if (dim == 0) { if (Vh) { *Vh = 1.0; } if (Vk) {*Vk = 1.0; } return; }
   J.SetSize(sdim, dim);
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

// static method
void Mesh::PrintElementsByGeometry(int dim,
                                   const Array<int> &num_elems_by_geom,
                                   std::ostream &out)
{
   for (int g = Geometry::DimStart[dim], first = 1;
        g < Geometry::DimStart[dim+1]; g++)
   {
      if (!num_elems_by_geom[g]) { continue; }
      if (!first) { out << " + "; }
      else { first = 0; }
      out << num_elems_by_geom[g] << ' ' << Geometry::Name[g] << "(s)";
   }
}

void Mesh::PrintCharacteristics(Vector *Vh, Vector *Vk, std::ostream &out)
{
   double h_min, h_max, kappa_min, kappa_max;

   out << "Mesh Characteristics:";

   this->GetCharacteristics(h_min, h_max, kappa_min, kappa_max, Vh, Vk);

   Array<int> num_elems_by_geom(Geometry::NumGeom);
   num_elems_by_geom = 0;
   for (int i = 0; i < GetNE(); i++)
   {
      num_elems_by_geom[GetElementBaseGeometry(i)]++;
   }

   out << '\n'
       << "Dimension          : " << Dimension() << '\n'
       << "Space dimension    : " << SpaceDimension();
   if (Dim == 0)
   {
      out << '\n'
          << "Number of vertices : " << GetNV() << '\n'
          << "Number of elements : " << GetNE() << '\n'
          << "Number of bdr elem : " << GetNBE() << '\n';
   }
   else if (Dim == 1)
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
          << "Number of elements : " << GetNE() << "  --  ";
      PrintElementsByGeometry(2, num_elems_by_geom, out);
      out << '\n'
          << "Number of bdr elem : " << GetNBE() << '\n'
          << "Euler Number       : " << EulerNumber2D() << '\n'
          << "h_min              : " << h_min << '\n'
          << "h_max              : " << h_max << '\n'
          << "kappa_min          : " << kappa_min << '\n'
          << "kappa_max          : " << kappa_max << '\n';
   }
   else
   {
      Array<int> num_bdr_elems_by_geom(Geometry::NumGeom);
      num_bdr_elems_by_geom = 0;
      for (int i = 0; i < GetNBE(); i++)
      {
         num_bdr_elems_by_geom[GetBdrElementBaseGeometry(i)]++;
      }
      Array<int> num_faces_by_geom(Geometry::NumGeom);
      num_faces_by_geom = 0;
      for (int i = 0; i < GetNFaces(); i++)
      {
         num_faces_by_geom[GetFaceBaseGeometry(i)]++;
      }

      out << '\n'
          << "Number of vertices : " << GetNV() << '\n'
          << "Number of edges    : " << GetNEdges() << '\n'
          << "Number of faces    : " << GetNFaces() << "  --  ";
      PrintElementsByGeometry(Dim-1, num_faces_by_geom, out);
      out << '\n'
          << "Number of elements : " << GetNE() << "  --  ";
      PrintElementsByGeometry(Dim, num_elems_by_geom, out);
      out << '\n'
          << "Number of bdr elem : " << GetNBE() << "  --  ";
      PrintElementsByGeometry(Dim-1, num_bdr_elems_by_geom, out);
      out << '\n'
          << "Euler Number       : " << EulerNumber() << '\n'
          << "h_min              : " << h_min << '\n'
          << "h_max              : " << h_max << '\n'
          << "kappa_min          : " << kappa_min << '\n'
          << "kappa_max          : " << kappa_max << '\n';
   }
   out << '\n' << std::flush;
}

FiniteElement *Mesh::GetTransformationFEforElementType(Element::Type ElemType)
{
   switch (ElemType)
   {
      case Element::POINT :          return &PointFE;
      case Element::SEGMENT :        return &SegmentFE;
      case Element::TRIANGLE :       return &TriangleFE;
      case Element::QUADRILATERAL :  return &QuadrilateralFE;
      case Element::TETRAHEDRON :    return &TetrahedronFE;
      case Element::HEXAHEDRON :     return &HexahedronFE;
      case Element::WEDGE :          return &WedgeFE;
      default:
         MFEM_ABORT("Unknown element type \"" << ElemType << "\"");
         break;
   }
   MFEM_ABORT("Unknown element type");
   return NULL;
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
   ElTr->FinalizeTransformation();
}

void Mesh::GetElementTransformation(int i, const Vector &nodes,
                                    IsoparametricTransformation *ElTr)
{
   ElTr->Attribute = GetAttribute(i);
   ElTr->ElementNo = i;
   DenseMatrix &pm = ElTr->GetPointMat();
   if (Nodes == NULL)
   {
      MFEM_ASSERT(nodes.Size() == spaceDim*GetNV(), "");
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
      MFEM_ASSERT(nodes.Size() == Nodes->Size(), "");
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
   ElTr->FinalizeTransformation();
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
   ElTr->FinalizeTransformation();
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

         Geometry::Type face_geom = GetFaceGeometryType(FaceNo);
         Element::Type  face_type = GetFaceElementType(FaceNo);

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
   FTr->FinalizeTransformation();
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
   EdTr->FinalizeTransformation();
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
   Transf.FinalizeTransformation();
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
   Transf.FinalizeTransformation();
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
   Transf.FinalizeTransformation();
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
   Transf.FinalizeTransformation();
}

void Mesh::GetLocalTriToWdgTransformation(
   IsoparametricTransformation &Transf, int i)
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&TriangleFE);
   //  (i/64) is the local face no. in the pri
   MFEM_VERIFY(i < 128, "Local face index " << i/64
               << " is not a triangular face of a wedge.");
   const int *pv = pri_t::FaceVert[i/64];
   //  (i%64) is the orientation of the wedge face
   //         w.r.t. the face element
   const int *to = tri_t::Orient[i%64];
   const IntegrationRule *PriVert =
      Geometries.GetVertices(Geometry::PRISM);
   locpm.SetSize(3, 3);
   for (int j = 0; j < 3; j++)
   {
      const IntegrationPoint &vert = PriVert->IntPoint(pv[to[j]]);
      locpm(0, j) = vert.x;
      locpm(1, j) = vert.y;
      locpm(2, j) = vert.z;
   }
   Transf.FinalizeTransformation();
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
   Transf.FinalizeTransformation();
}

void Mesh::GetLocalQuadToWdgTransformation(
   IsoparametricTransformation &Transf, int i)
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&QuadrilateralFE);
   //  (i/64) is the local face no. in the pri
   MFEM_VERIFY(i >= 128, "Local face index " << i/64
               << " is not a quadrilateral face of a wedge.");
   const int *pv = pri_t::FaceVert[i/64];
   //  (i%64) is the orientation of the quad
   const int *qo = quad_t::Orient[i%64];
   const IntegrationRule *PriVert = Geometries.GetVertices(Geometry::PRISM);
   locpm.SetSize(3, 4);
   for (int j = 0; j < 4; j++)
   {
      const IntegrationPoint &vert = PriVert->IntPoint(pv[qo[j]]);
      locpm(0, j) = vert.x;
      locpm(1, j) = vert.y;
      locpm(2, j) = vert.z;
   }
   Transf.FinalizeTransformation();
}

void Mesh::GetLocalFaceTransformation(
   int face_type, int elem_type, IsoparametricTransformation &Transf, int info)
{
   switch (face_type)
   {
      case Element::POINT:
         GetLocalPtToSegTransformation(Transf, info);
         break;

      case Element::SEGMENT:
         if (elem_type == Element::TRIANGLE)
         {
            GetLocalSegToTriTransformation(Transf, info);
         }
         else
         {
            MFEM_ASSERT(elem_type == Element::QUADRILATERAL, "");
            GetLocalSegToQuadTransformation(Transf, info);
         }
         break;

      case Element::TRIANGLE:
         if (elem_type == Element::TETRAHEDRON)
         {
            GetLocalTriToTetTransformation(Transf, info);
         }
         else
         {
            MFEM_ASSERT(elem_type == Element::WEDGE, "");
            GetLocalTriToWdgTransformation(Transf, info);
         }
         break;

      case Element::QUADRILATERAL:
         if (elem_type == Element::HEXAHEDRON)
         {
            GetLocalQuadToHexTransformation(Transf, info);
         }
         else
         {
            MFEM_ASSERT(elem_type == Element::WEDGE, "");
            GetLocalQuadToWdgTransformation(Transf, info);
         }
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
   transf.FinalizeTransformation();
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

void Mesh::GetFaceElements(int Face, int *Elem1, int *Elem2) const
{
   *Elem1 = faces_info[Face].Elem1No;
   *Elem2 = faces_info[Face].Elem2No;
}

void Mesh::GetFaceInfos(int Face, int *Inf1, int *Inf2) const
{
   *Inf1 = faces_info[Face].Elem1Inf;
   *Inf2 = faces_info[Face].Elem2Inf;
}

Geometry::Type Mesh::GetFaceGeometryType(int Face) const
{
   return (Dim == 1) ? Geometry::POINT : faces[Face]->GetGeometryType();
}

Element::Type Mesh::GetFaceElementType(int Face) const
{
   return (Dim == 1) ? Element::POINT : faces[Face]->GetType();
}

void Mesh::Init()
{
   // in order of declaration:
   Dim = spaceDim = 0;
   NumOfVertices = -1;
   NumOfElements = NumOfBdrElements = 0;
   NumOfEdges = NumOfFaces = 0;
   meshgen = mesh_geoms = 0;
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
      el_to_face = el_to_el = bel_to_edge = face_edge = edge_vertex = NULL;
}

void Mesh::SetEmpty()
{
   Init();
   InitTables();
}

void Mesh::DestroyTables()
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

void Mesh::DeleteLazyTables()
{
   delete el_to_el;     el_to_el = NULL;
   delete face_edge;    face_edge = NULL;
   delete edge_vertex;  edge_vertex = NULL;
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

void Mesh::AddWedge(const int *vi, int attr)
{
   elements[NumOfElements++] = new Wedge(vi, attr);
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

void Mesh::AddHexAsWedges(const int *vi, int attr)
{
   static const int hex_to_wdg[2][6] =
   {
      { 0, 1, 2, 4, 5, 6 }, { 0, 2, 3, 4, 6, 7 }
   };
   int ti[6];

   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 6; j++)
      {
         ti[j] = vi[hex_to_wdg[i][j]];
      }
      AddWedge(ti, attr);
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

void Mesh::FinalizeCheck()
{
   MFEM_VERIFY(vertices.Size() == NumOfVertices ||
               vertices.Size() == 0,
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

   SetMeshGen();
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

   SetMeshGen();
}


#ifdef MFEM_USE_GECKO
void Mesh::GetGeckoElementReordering(Array<int> &ordering,
                                     int iterations, int window,
                                     int period, int seed)
{
   Gecko::Graph graph;

   Gecko::Functional *functional =
      new Gecko::FunctionalGeometric(); // ordering functional

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
      // To force FE space update, we need to increase 'sequence':
      sequence++;
      last_operation = Mesh::NONE;
      nodes_fes->Update(false); // want_transform = false
      Nodes->Update(); // just needed to update Nodes->sequence
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
   // vertex 0 - vertex 1 is the longest edge in the triangle.
   DenseMatrix pmat;
   for (int i = 0; i < NumOfElements; i++)
   {
      if (elements[i]->GetType() == Element::TRIANGLE)
      {
         GetPointMatrix(i, pmat);
         static_cast<Triangle*>(elements[i])->MarkEdge(pmat);
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

   if (*old_v_to_v == NULL)
   {
      bool need_v_to_v = false;
      Array<int> dofs;
      for (int i = 0; i < GetNEdges(); i++)
      {
         // Since edge indices may change, we need to permute edge interior dofs
         // any time an edge index changes and there is at least one dof on that
         // edge.
         fes->GetEdgeInteriorDofs(i, dofs);
         if (dofs.Size() > 0)
         {
            need_v_to_v = true;
            break;
         }
      }
      if (need_v_to_v)
      {
         *old_v_to_v = new DSTable(NumOfVertices);
         GetVertexToVertexTable(*(*old_v_to_v));
      }
   }
   if (*old_elem_vert == NULL)
   {
      bool need_elem_vert = false;
      Array<int> dofs;
      for (int i = 0; i < GetNE(); i++)
      {
         // Since element indices do not change, we need to permute element
         // interior dofs only when there are at least 2 interior dofs in an
         // element (assuming the nodal dofs are non-directional).
         fes->GetElementInteriorDofs(i, dofs);
         if (dofs.Size() > 1)
         {
            need_elem_vert = true;
            break;
         }
      }
      if (need_elem_vert)
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
   Array<int> old_dofs, new_dofs;

   // assuming that all edges have the same number of dofs
   if (NumOfEdges) { fes->GetEdgeInteriorDofs(0, old_dofs); }
   const int num_edge_dofs = old_dofs.Size();

   // Save the original nodes
   const Vector onodes = *Nodes;

   // vertex dofs do not need to be moved
   fes->GetVertexDofs(0, old_dofs);
   int offset = NumOfVertices * old_dofs.Size();

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
            const int old_i = (*old_v_to_v)(i, it.Column());
            const int new_i = it.Index();
            if (new_i == old_i) { continue; }

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

   // face dofs:
   // both enumeration and orientation of the faces may be different
   if (fes->GetNFDofs() > 0)
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

      // compute the new face dof offsets
      Array<int> new_fdofs(NumOfFaces+1);
      new_fdofs[0] = 0;
      for (int i = 0; i < NumOfFaces; i++) // i = old face index
      {
         const int *old_v = old_face_vertex.GetRow(i);
         int new_i; // new face index
         switch (old_face_vertex.RowSize(i))
         {
            case 3:
               new_i = (*faces_tbl)(old_v[0], old_v[1], old_v[2]);
               break;
            case 4:
            default:
               new_i = (*faces_tbl)(old_v[0], old_v[1], old_v[2], old_v[3]);
               break;
         }
         fes->GetFaceInteriorDofs(i, old_dofs);
         new_fdofs[new_i+1] = old_dofs.Size();
      }
      new_fdofs.PartialSum();

      // loop over the old face numbers
      for (int i = 0; i < NumOfFaces; i++)
      {
         const int *old_v = old_face_vertex.GetRow(i), *new_v;
         const int *dof_ord;
         int new_i, new_or;
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

         fes->GetFaceInteriorDofs(i, old_dofs);
         new_dofs.SetSize(old_dofs.Size());
         for (int j = 0; j < old_dofs.Size(); j++)
         {
            // we assume the dofs are non-directional, i.e. dof_ord[j] is >= 0
            const int old_j = dof_ord[j];
            new_dofs[old_j] = offset + new_fdofs[new_i] + j;
         }
         fes->DofsToVDofs(old_dofs);
         fes->DofsToVDofs(new_dofs);
         for (int j = 0; j < old_dofs.Size(); j++)
         {
            (*Nodes)(new_dofs[j]) = onodes(old_dofs[j]);
         }
      }

      offset += fes->GetNFDofs();
      delete faces_tbl;
   }

   // element dofs:
   // element orientation may be different
   if (old_elem_vert) // have elements with 2 or more dofs
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
         const int *old_v = old_elem_vert->GetRow(i);
         const int *new_v = elements[i]->GetVertices();
         const int *dof_ord;
         int new_or;
         const Geometry::Type geom = elements[i]->GetGeometryType();
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
               MFEM_ABORT(Geometry::Name[geom] << " elements (" << fec->Name()
                          << " FE collection) are not supported yet!");
               break;
         }
         dof_ord = fec->DofOrderForOrientation(geom, new_or);
         MFEM_VERIFY(dof_ord != NULL,
                     "FE collection '" << fec->Name()
                     << "' does not define reordering for "
                     << Geometry::Name[geom] << " elements!");
         fes->GetElementInteriorDofs(i, old_dofs);
         new_dofs.SetSize(old_dofs.Size());
         for (int j = 0; j < new_dofs.Size(); j++)
         {
            // we assume the dofs are non-directional, i.e. dof_ord[j] is >= 0
            const int old_j = dof_ord[j];
            new_dofs[old_j] = offset + j;
         }
         offset += new_dofs.Size();
         fes->DofsToVDofs(old_dofs);
         fes->DofsToVDofs(new_dofs);
         for (int j = 0; j < old_dofs.Size(); j++)
         {
            (*Nodes)(new_dofs[j]) = onodes(old_dofs[j]);
         }
      }
   }

   // Update Tables, faces, etc
   if (Dim > 2)
   {
      if (fes->GetNFDofs() == 0)
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
   // To force FE space update, we need to increase 'sequence':
   sequence++;
   last_operation = Mesh::NONE;
   fes->Update(false); // want_transform = false
   Nodes->Update(); // just needed to update Nodes->sequence
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

   SetMeshGen();
}

void Mesh::FinalizeWedgeMesh(int generate_edges, int refine,
                             bool fix_orientation)
{
   FinalizeCheck();
   CheckElementOrientation(fix_orientation);

   if (NumOfBdrElements == 0)
   {
      GetElementToFaceTable();
      GenerateFaces();
      GenerateBoundaryElements();
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

   SetMeshGen();
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

   SetMeshGen();
}

void Mesh::FinalizeMesh(int refine, bool fix_orientation)
{
   FinalizeTopology();

   Finalize(refine, fix_orientation);
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

   // set the mesh type: 'meshgen', ...
   SetMeshGen();

   // generate the faces
   if (Dim > 2)
   {
      GetElementToFaceTable();
      GenerateFaces();
      if (NumOfBdrElements == 0)
      {
         GenerateBoundaryElements();
         GetElementToFaceTable(); // update be_to_face
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

   if (Dim == 1)
   {
      GenerateFaces();
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
   //  1) FinalizeTopology() or equivalent was called
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

#ifdef MFEM_DEBUG
   // For non-orientable surfaces/manifolds, the check below will fail, so we
   // only perform it when Dim == spaceDim.
   if (Dim >= 2 && Dim == spaceDim)
   {
      const int num_faces = GetNumFaces();
      for (int i = 0; i < num_faces; i++)
      {
         MFEM_VERIFY(faces_info[i].Elem2No < 0 ||
                     faces_info[i].Elem2Inf%2 != 0, "invalid mesh topology");
      }
   }
#endif
}

void Mesh::Make3D(int nx, int ny, int nz, Element::Type type,
                  double sx, double sy, double sz,
                  bool generate_edges, bool sfc_ordering)
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
   else if (type == Element::WEDGE)
   {
      NElem *= 2;
      NBdrElem += 2*nx*ny;
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
   if (sfc_ordering && type == Element::HEXAHEDRON)
   {
      Array<int> sfc;
      NCMesh::GridSfcOrdering3D(nx, ny, nz, sfc);
      MFEM_VERIFY(sfc.Size() == 3*nx*ny*nz, "");

      for (int k = 0; k < nx*ny*nz; k++)
      {
         x = sfc[3*k + 0];
         y = sfc[3*k + 1];
         z = sfc[3*k + 2];

         ind[0] = VTX(x  , y  , z  );
         ind[1] = VTX(x+1, y  , z  );
         ind[2] = VTX(x+1, y+1, z  );
         ind[3] = VTX(x  , y+1, z  );
         ind[4] = VTX(x  , y  , z+1);
         ind[5] = VTX(x+1, y  , z+1);
         ind[6] = VTX(x+1, y+1, z+1);
         ind[7] = VTX(x  , y+1, z+1);

         AddHex(ind, 1);
      }
   }
   else
   {
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
               else if (type == Element::WEDGE)
               {
                  AddHexAsWedges(ind, 1);
               }
               else
               {
                  AddHex(ind, 1);
               }
            }
         }
      }
   }

   // Sets boundary elements and the corresponding indices of vertices
   // bottom, bdr. attribute 1
   for (y = 0; y < ny; y++)
   {
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
         else if (type == Element::WEDGE)
         {
            AddBdrQuadAsTriangles(ind, 1);
         }
         else
         {
            AddBdrQuad(ind, 1);
         }
      }
   }
   // top, bdr. attribute 6
   for (y = 0; y < ny; y++)
   {
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
         else if (type == Element::WEDGE)
         {
            AddBdrQuadAsTriangles(ind, 1);
         }
         else
         {
            AddBdrQuad(ind, 6);
         }
      }
   }
   // left, bdr. attribute 5
   for (z = 0; z < nz; z++)
   {
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
   }
   // right, bdr. attribute 3
   for (z = 0; z < nz; z++)
   {
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
   }
   // front, bdr. attribute 2
   for (x = 0; x < nx; x++)
   {
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
   }
   // back, bdr. attribute 4
   for (x = 0; x < nx; x++)
   {
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
   }

#undef VTX

#if 0
   ofstream test_stream("debug.mesh");
   Print(test_stream);
   test_stream.close();
#endif

   FinalizeTopology();

   // Finalize(...) can be called after this method, if needed
}

void Mesh::Make2D(int nx, int ny, Element::Type type,
                  double sx, double sy,
                  bool generate_edges, bool sfc_ordering)
{
   int i, j, k;

   SetEmpty();

   Dim = spaceDim = 2;

   // Creates quadrilateral mesh
   if (type == Element::QUADRILATERAL)
   {
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
      if (sfc_ordering)
      {
         Array<int> sfc;
         NCMesh::GridSfcOrdering2D(nx, ny, sfc);
         MFEM_VERIFY(sfc.Size() == 2*nx*ny, "");

         for (k = 0; k < nx*ny; k++)
         {
            i = sfc[2*k + 0];
            j = sfc[2*k + 1];
            ind[0] = i + j*(nx+1);
            ind[1] = i + 1 +j*(nx+1);
            ind[2] = i + 1 + (j+1)*(nx+1);
            ind[3] = i + (j+1)*(nx+1);
            elements[k] = new Quadrilateral(ind);
         }
      }
      else
      {
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

      // MarkTriMeshForRefinement(); // done in Finalize(...)
   }
   else
   {
      MFEM_ABORT("Unsupported element type.");
   }

   SetMeshGen();
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

   // Finalize(...) can be called after this method, if needed
}

void Mesh::Make1D(int n, double sx)
{
   int j, ind[1];

   SetEmpty();

   Dim = 1;
   spaceDim = 1;

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

   SetMeshGen();
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
   mesh_geoms = mesh.mesh_geoms;

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
   mesh.nc_faces_info.Copy(nc_faces_info);

   // Do NOT copy the element-to-element Table, el_to_el
   el_to_el = NULL;

   // Do NOT copy the face-to-edge Table, face_edge
   face_edge = NULL;

   // Copy the edge-to-vertex Table, edge_vertex
   edge_vertex = (mesh.edge_vertex) ? new Table(*mesh.edge_vertex) : NULL;

   // Copy the attributes and bdr_attributes
   mesh.attributes.Copy(attributes);
   mesh.bdr_attributes.Copy(bdr_attributes);

   // Deep copy the NURBSExtension.
#ifdef MFEM_USE_MPI
   ParNURBSExtension *pNURBSext =
      dynamic_cast<ParNURBSExtension *>(mesh.NURBSext);
   if (pNURBSext)
   {
      NURBSext = new ParNURBSExtension(*pNURBSext);
   }
   else
#endif
   {
      NURBSext = mesh.NURBSext ? new NURBSExtension(*mesh.NURBSext) : NULL;
   }

   // Deep copy the NCMesh.
#ifdef MFEM_USE_MPI
   if (dynamic_cast<const ParMesh*>(&mesh))
   {
      ncmesh = NULL; // skip; will be done in ParMesh copy ctor
   }
   else
#endif
   {
      ncmesh = mesh.ncmesh ? new NCMesh(*mesh.ncmesh) : NULL;
   }

   // Duplicate the Nodes, including the FiniteElementCollection and the
   // FiniteElementSpace
   if (mesh.Nodes && copy_nodes)
   {
      FiniteElementSpace *fes = mesh.Nodes->FESpace();
      const FiniteElementCollection *fec = fes->FEColl();
      FiniteElementCollection *fec_copy =
         FiniteElementCollection::New(fec->Name());
      FiniteElementSpace *fes_copy =
         new FiniteElementSpace(*fes, this, fec_copy);
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
   int boundary_index_stride = num_boundary_elements > 0 ?
                               Geometry::NumVerts[boundary_type] : 0;

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
      case Geometry::TETRAHEDRON:
#ifdef MFEM_USE_MEMALLOC
         return TetMemory.Alloc();
#else
         return (new Tetrahedron);
#endif
      case Geometry::CUBE:      return (new Hexahedron);
      case Geometry::PRISM:     return (new Wedge);
      default:
         MFEM_ABORT("invalid Geometry::Type, geom = " << geom);
   }

   return NULL;
}

Element *Mesh::ReadElementWithoutAttr(std::istream &input)
{
   int geom, nv, *v;
   Element *el;

   input >> geom;
   el = NewElement(geom);
   MFEM_VERIFY(el, "Unsupported element type: " << geom);
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
   meshgen = mesh_geoms = 0;
   for (int i = 0; i < NumOfElements; i++)
   {
      const Element::Type type = GetElement(i)->GetType();
      switch (type)
      {
         case Element::TETRAHEDRON:
            mesh_geoms |= (1 << Geometry::TETRAHEDRON);
         case Element::TRIANGLE:
            mesh_geoms |= (1 << Geometry::TRIANGLE);
         case Element::SEGMENT:
            mesh_geoms |= (1 << Geometry::SEGMENT);
         case Element::POINT:
            mesh_geoms |= (1 << Geometry::POINT);
            meshgen |= 1;
            break;

         case Element::HEXAHEDRON:
            mesh_geoms |= (1 << Geometry::CUBE);
         case Element::QUADRILATERAL:
            mesh_geoms |= (1 << Geometry::SQUARE);
            mesh_geoms |= (1 << Geometry::SEGMENT);
            mesh_geoms |= (1 << Geometry::POINT);
            meshgen |= 2;
            break;

         case Element::WEDGE:
            mesh_geoms |= (1 << Geometry::PRISM);
            mesh_geoms |= (1 << Geometry::SQUARE);
            mesh_geoms |= (1 << Geometry::TRIANGLE);
            mesh_geoms |= (1 << Geometry::SEGMENT);
            mesh_geoms |= (1 << Geometry::POINT);
            meshgen |= 4;
            break;

         default:
            MFEM_ABORT("invalid element type: " << type);
            break;
      }
   }
}

void Mesh::Loader(std::istream &input, int generate_edges,
                  std::string parse_tag)
{
   int curved = 0, read_gf = 1;
   bool finalize_topo = true;

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
      ReadVTKMesh(input, curved, read_gf, finalize_topo);
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
   if (finalize_topo)
   {
      FinalizeTopology();
   }

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

   FinalizeTopology();

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
   MFEM_VERIFY(orig_mesh->GetNumGeometries(Dim) <= 1,
               "meshes with mixed elements are not supported");

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
      Geometry::Type geom = orig_mesh->GetElementBaseGeometry(el);
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
      Geometry::Type geom = orig_mesh->GetBdrElementBaseGeometry(el);
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
   CoarseFineTr.embeddings.SetSize(GetNE());
   if (orig_mesh->GetNE() > 0)
   {
      const int el = 0;
      Geometry::Type geom = orig_mesh->GetElementBaseGeometry(el);
      CoarseFineTr.point_matrices[geom].SetSize(Dim, max_nv, r_elem_factor);
      int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, ref_factor);
      const int *c2h_map = rfec.GetDofMap(geom);
      const IntegrationRule &r_nodes = rfes.GetFE(el)->GetNodes();
      for (int j = 0; j < RG.RefGeoms.Size()/nvert; j++)
      {
         DenseMatrix &Pj = CoarseFineTr.point_matrices[geom](j);
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

   last_operation = Mesh::NONE; // FiniteElementSpace::Update is not supported
   sequence++;

   UpdateNURBS();
}

void Mesh::NURBSUniformRefinement()
{
   // do not check for NURBSext since this method is protected
   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->UniformRefinement();

   last_operation = Mesh::NONE; // FiniteElementSpace::Update is not supported
   sequence++;

   UpdateNURBS();
}

void Mesh::DegreeElevate(int rel_degree, int degree)
{
   if (NURBSext == NULL)
   {
      mfem_error("Mesh::DegreeElevate : Not a NURBS mesh!");
   }

   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->DegreeElevate(rel_degree, degree);

   last_operation = Mesh::NONE; // FiniteElementSpace::Update is not supported
   sequence++;

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
   for (int j = 0; j < NumOfElements; j++)
   {
      elements[j] = ReadElement(input);
   }

   skip_comment_lines(input, '#');

   input >> ident; // 'boundary'
   input >> NumOfBdrElements;
   boundary.SetSize(NumOfBdrElements);
   for (int j = 0; j < NumOfBdrElements; j++)
   {
      boundary[j] = ReadElement(input);
   }

   skip_comment_lines(input, '#');

   input >> ident; // 'edges'
   input >> NumOfEdges;
   edge_vertex = new Table(NumOfEdges, 2);
   edge_to_knot.SetSize(NumOfEdges);
   for (int j = 0; j < NumOfEdges; j++)
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

   FinalizeTopology();
   CheckBdrElementOrientation(); // check and fix boundary element orientation
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
                  default:
                     MFEM_ABORT("Invalid 2D element type \""
                                << GetElementType(i) << "\"");
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

            case Element::WEDGE:
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

            default:
               MFEM_ABORT("Invalid 3D element type \""
                          << GetElementType(i) << "\"");
               break;
         }
      }
   }
#if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
   if (wo > 0)
      mfem::out << "Elements with wrong orientation: " << wo << " / "
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
         mfem::err << "Mesh::GetQuadOrientation(...)" << endl;
         mfem::err << " base = [";
         for (int k = 0; k < 4; k++)
         {
            mfem::err << " " << base[k];
         }
         mfem::err << " ]\n test = [";
         for (int k = 0; k < 4; k++)
         {
            mfem::err << " " << test[k];
         }
         mfem::err << " ]" << endl;
         mfem_error();
      }
#endif

   if (test[(i+1)%4] == base[1])
   {
      return 2*i;
   }

   return 2*i+1;
}

int Mesh::CheckBdrElementOrientation(bool fix_it)
{
   int wo = 0; // count wrong orientations

   if (Dim == 2)
   {
      if (el_to_edge == NULL) // edges were not generated
      {
         el_to_edge = new Table;
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         GenerateFaces(); // 'Faces' in 2D refers to the edges
      }
      for (int i = 0; i < NumOfBdrElements; i++)
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
      for (int i = 0; i < NumOfBdrElements; i++)
      {
         const int fi = be_to_face[i];

         if (faces_info[fi].Elem2No >= 0) { continue; }

         // boundary face
         int *bv = boundary[i]->GetVertices();
         // Make sure the 'faces' are generated:
         MFEM_ASSERT(fi < faces.Size(), "internal error");
         const int *fv = faces[fi]->GetVertices();
         int orientation; // orientation of the bdr. elem. w.r.t. the
         // corresponding face element (that's the base)
         const Element::Type bdr_type = GetBdrElementType(i);
         switch (bdr_type)
         {
            case Element::TRIANGLE:
            {
               orientation = GetTriOrientation(fv, bv);
               break;
            }
            case Element::QUADRILATERAL:
            {
               orientation = GetQuadOrientation(fv, bv);
               break;
            }
            default:
               MFEM_ABORT("Invalid 2D boundary element type \""
                          << bdr_type << "\"");
               orientation = 0; // suppress a warning
               break;
         }

         if (orientation % 2 == 0) { continue; }
         wo++;
         if (!fix_it) { continue; }

         switch (bdr_type)
         {
            case Element::TRIANGLE:
            {
               // swap vertices 0 and 1 so that we don't change the marked edge:
               // (0,1,2) -> (1,0,2)
               mfem::Swap<int>(bv[0], bv[1]);
               if (bel_to_edge)
               {
                  int *be = bel_to_edge->GetRow(i);
                  mfem::Swap<int>(be[1], be[2]);
               }
               break;
            }
            case Element::QUADRILATERAL:
            {
               mfem::Swap<int>(bv[0], bv[2]);
               if (bel_to_edge)
               {
                  int *be = bel_to_edge->GetRow(i);
                  mfem::Swap<int>(be[0], be[1]);
                  mfem::Swap<int>(be[2], be[3]);
               }
               break;
            }
            default: // unreachable
               break;
         }
      }
   }
   // #if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
#ifdef MFEM_DEBUG
   if (wo > 0)
   {
      mfem::out << "Boundary elements with wrong orientation: " << wo << " / "
                << NumOfBdrElements << " (" << fixed_or_not[fix_it ? 0 : 1]
                << ")" << endl;
   }
#endif
   return wo;
}

int Mesh::GetNumGeometries(int dim) const
{
   MFEM_ASSERT(0 <= dim && dim <= Dim, "invalid dim: " << dim);
   int num_geoms = 0;
   for (int g = Geometry::DimStart[dim]; g < Geometry::DimStart[dim+1]; g++)
   {
      if (HasGeometry(Geometry::Type(g))) { num_geoms++; }
   }
   return num_geoms;
}

void Mesh::GetGeometries(int dim, Array<Geometry::Type> &el_geoms) const
{
   MFEM_ASSERT(0 <= dim && dim <= Dim, "invalid dim: " << dim);
   el_geoms.SetSize(0);
   for (int g = Geometry::DimStart[dim]; g < Geometry::DimStart[dim+1]; g++)
   {
      if (HasGeometry(Geometry::Type(g)))
      {
         el_geoms.Append(Geometry::Type(g));
      }
   }
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

Element::Type Mesh::GetElementType(int i) const
{
   return elements[i]->GetType();
}

Element::Type Mesh::GetBdrElementType(int i) const
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

   // Note that, for ParNCMeshes, faces_info will contain also the ghost faces
   MFEM_ASSERT(faces_info.Size() >= GetNumFaces(), "faces were not generated!");

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
         // Temporarily allow even edge orientations: see the remark in
         // AddTriangleFaceElement().
         // Also, in a non-orientable surface mesh, the orientation will be even
         // for edges that connect elements with opposite orientations.
         faces_info[gf].Elem2Inf = 64 * lf;
      }
      else
      {
         MFEM_ABORT("internal error");
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
      // In a valid mesh, we should have (orientation % 2 != 0), however, if
      // one of the adjacent elements has wrong orientation, both face
      // orientations can be even, until the element orientations are fixed.
      // MFEM_ASSERT(orientation % 2 != 0, "");
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
      // Temporarily allow even face orientations: see the remark in
      // AddTriangleFaceElement().
      // MFEM_ASSERT(oo % 2 != 0, "");
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
                  const int *fv = tet_t::FaceVert[j];
                  AddTriangleFaceElement(j, ef[j], i,
                                         v[fv[0]], v[fv[1]], v[fv[2]]);
               }
               break;
            }
            case Element::WEDGE:
            {
               for (int j = 0; j < 2; j++)
               {
                  const int *fv = pri_t::FaceVert[j];
                  AddTriangleFaceElement(j, ef[j], i,
                                         v[fv[0]], v[fv[1]], v[fv[2]]);
               }
               for (int j = 2; j < 5; j++)
               {
                  const int *fv = pri_t::FaceVert[j];
                  AddQuadFaceElement(j, ef[j], i,
                                     v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
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
         case Element::WEDGE:
         {
            for (int j = 0; j < 2; j++)
            {
               const int *fv = pri_t::FaceVert[j];
               faces_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]);
            }
            for (int j = 2; j < 5; j++)
            {
               const int *fv = pri_t::FaceVert[j];
               faces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
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
         case Element::WEDGE:
         {
            for (int j = 0; j < 2; j++)
            {
               const int *fv = pri_t::FaceVert[j];
               el_to_face->Push(
                  i, faces_tbl->Push(v[fv[0]], v[fv[1]], v[fv[2]]));
            }
            for (int j = 2; j < 5; j++)
            {
               const int *fv = pri_t::FaceVert[j];
               el_to_face->Push(
                  i, faces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]));
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

int *Mesh::CartesianPartitioning(int nxyz[])
{
   int *partitioning;
   double pmin[3] = { infinity(), infinity(), infinity() };
   double pmax[3] = { -infinity(), -infinity(), -infinity() };
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
#ifdef MFEM_USE_METIS
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
   else if (NumOfElements <= nparts)
   {
      for (i = 0; i < NumOfElements; i++)
      {
         partitioning[i] = i;
      }
   }
   else
   {
      idx_t *I, *J, n;
#ifndef MFEM_USE_METIS_5
      idx_t wgtflag = 0;
      idx_t numflag = 0;
      idx_t options[5];
#else
      idx_t ncon = 1;
      idx_t err;
      idx_t options[40];
#endif
      idx_t edgecut;

      // In case METIS have been compiled with 64bit indices
      bool freedata = false;
      idx_t mparts = (idx_t) nparts;
      idx_t *mpartitioning;

      n = NumOfElements;
      if (sizeof(idx_t) == sizeof(int))
      {
         I = (idx_t*) el_to_el->GetI();
         J = (idx_t*) el_to_el->GetJ();
         mpartitioning = (idx_t*) partitioning;
      }
      else
      {
         int *iI = el_to_el->GetI();
         int *iJ = el_to_el->GetJ();
         int m = iI[n];
         I = new idx_t[n+1];
         J = new idx_t[m];
         for (int k = 0; k < n+1; k++) { I[k] = iI[k]; }
         for (int k = 0; k < m; k++) { J[k] = iJ[k]; }
         mpartitioning = new idx_t[n];
         freedata = true;
      }
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
            std::sort(J+I[i], J+I[i+1], std::greater<idx_t>());
         }
      }

      // This function should be used to partition a graph into a small
      // number of partitions (less than 8).
      if (part_method == 0 || part_method == 3)
      {
#ifndef MFEM_USE_METIS_5
         METIS_PartGraphRecursive(&n,
                                  I,
                                  J,
                                  NULL,
                                  NULL,
                                  &wgtflag,
                                  &numflag,
                                  &mparts,
                                  options,
                                  &edgecut,
                                  mpartitioning);
#else
         err = METIS_PartGraphRecursive(&n,
                                        &ncon,
                                        I,
                                        J,
                                        NULL,
                                        NULL,
                                        NULL,
                                        &mparts,
                                        NULL,
                                        NULL,
                                        options,
                                        &edgecut,
                                        mpartitioning);
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
                             I,
                             J,
                             NULL,
                             NULL,
                             &wgtflag,
                             &numflag,
                             &mparts,
                             options,
                             &edgecut,
                             mpartitioning);
#else
         err = METIS_PartGraphKway(&n,
                                   &ncon,
                                   I,
                                   J,
                                   NULL,
                                   NULL,
                                   NULL,
                                   &mparts,
                                   NULL,
                                   NULL,
                                   options,
                                   &edgecut,
                                   mpartitioning);
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
                              I,
                              J,
                              NULL,
                              NULL,
                              &wgtflag,
                              &numflag,
                              &mparts,
                              options,
                              &edgecut,
                              mpartitioning);
#else
         options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
         err = METIS_PartGraphKway(&n,
                                   &ncon,
                                   I,
                                   J,
                                   NULL,
                                   NULL,
                                   NULL,
                                   &mparts,
                                   NULL,
                                   NULL,
                                   options,
                                   &edgecut,
                                   mpartitioning);
         if (err != 1)
            mfem_error("Mesh::GeneratePartitioning: "
                       " error in METIS_PartGraphKway!");
#endif
      }

#ifdef MFEM_DEBUG
      mfem::out << "Mesh::GeneratePartitioning(...): edgecut = "
                << edgecut << endl;
#endif
      nparts = (int) mparts;
      if (mpartitioning != (idx_t*)partitioning)
      {
         for (int k = 0; k<NumOfElements; k++) { partitioning[k] = mpartitioning[k]; }
      }
      if (freedata)
      {
         delete[] I;
         delete[] J;
         delete[] mpartitioning;
      }
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
         mfem::err << "Mesh::GeneratePartitioning returned " << empty_parts
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
      mfem::out << "Mesh::CheckPartitioning(...) :\n"
                << "The following subdomains are empty :\n";
      for (i = 0; i < num_comp.Size(); i++)
         if (num_comp[i] == 0)
         {
            mfem::out << ' ' << i;
         }
      mfem::out << endl;
   }
   if (n_mcomp > 0)
   {
      mfem::out << "Mesh::CheckPartitioning(...) :\n"
                << "The following subdomains are NOT connected :\n";
      for (i = 0; i < num_comp.Size(); i++)
         if (num_comp[i] > 1)
         {
            mfem::out << ' ' << i;
         }
      mfem::out << endl;
   }
   if (n_empty == 0 && n_mcomp == 0)
      mfem::out << "Mesh::CheckPartitioning(...) : "
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

void Mesh::AverageVertices(const int *indexes, int n, int result)
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

void Mesh::UniformRefinement2D()
{
   DeleteLazyTables();

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }

   int quad_counter = 0;
   for (int i = 0; i < NumOfElements; i++)
   {
      if (elements[i]->GetType() == Element::QUADRILATERAL)
      {
         quad_counter++;
      }
   }

   const int oedge = NumOfVertices;
   const int oelem = oedge + NumOfEdges;

   vertices.SetSize(oelem + quad_counter);
   elements.SetSize(4 * NumOfElements);
   quad_counter = 0;
   for (int i = 0; i < NumOfElements; i++)
   {
      const Element::Type el_type = elements[i]->GetType();
      const int attr = elements[i]->GetAttribute();
      int *v = elements[i]->GetVertices();
      const int *e = el_to_edge->GetRow(i);
      const int j = NumOfElements + 3 * i;
      int vv[2];

      if (el_type == Element::TRIANGLE)
      {
         for (int ei = 0; ei < 3; ei++)
         {
            for (int k = 0; k < 2; k++)
            {
               vv[k] = v[tri_t::Edges[ei][k]];
            }
            AverageVertices(vv, 2, oedge+e[ei]);
         }

         elements[j+0] = new Triangle(oedge+e[1], oedge+e[2], oedge+e[0], attr);
         elements[j+1] = new Triangle(oedge+e[0], v[1], oedge+e[1], attr);
         elements[j+2] = new Triangle(oedge+e[2], oedge+e[1], v[2], attr);

         v[1] = oedge+e[0];
         v[2] = oedge+e[2];
      }
      else if (el_type == Element::QUADRILATERAL)
      {
         const int qe = quad_counter;
         quad_counter++;
         AverageVertices(v, 4, oelem+qe);

         for (int ei = 0; ei < 4; ei++)
         {
            for (int k = 0; k < 2; k++)
            {
               vv[k] = v[quad_t::Edges[ei][k]];
            }
            AverageVertices(vv, 2, oedge+e[ei]);
         }

         elements[j+0] = new Quadrilateral(oedge+e[0], v[1], oedge+e[1],
                                           oelem+qe, attr);
         elements[j+1] = new Quadrilateral(oelem+qe, oedge+e[1],
                                           v[2], oedge+e[2], attr);
         elements[j+2] = new Quadrilateral(oedge+e[3], oelem+qe,
                                           oedge+e[2], v[3], attr);

         v[1] = oedge+e[0];
         v[2] = oelem+qe;
         v[3] = oedge+e[3];
      }
      else
      {
         MFEM_ABORT("unknown element type: " << el_type);
      }
   }

   boundary.SetSize(2 * NumOfBdrElements);
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      const int attr = boundary[i]->GetAttribute();
      int *v = boundary[i]->GetVertices();
      const int j = NumOfBdrElements + i;

      boundary[j] = new Segment(oedge+be_to_edge[i], v[1], attr);

      v[1] = oedge+be_to_edge[i];
   }

   static const double A = 0.0, B = 0.5, C = 1.0;
   static double tri_children[2*3*4] =
   {
      A,A, B,A, A,B,
      B,B, A,B, B,A,
      B,A, C,A, B,B,
      A,B, B,B, A,C
   };
   static double quad_children[2*4*4] =
   {
      A,A, B,A, B,B, A,B, // lower-left
      B,A, C,A, C,B, B,B, // lower-right
      B,B, C,B, C,C, B,C, // upper-right
      A,B, B,B, B,C, A,C  // upper-left
   };

   CoarseFineTr.point_matrices[Geometry::TRIANGLE].
   UseExternalData(tri_children, 2, 3, 4);
   CoarseFineTr.point_matrices[Geometry::SQUARE].
   UseExternalData(quad_children, 2, 4, 4);
   CoarseFineTr.embeddings.SetSize(elements.Size());

   for (int i = 0; i < elements.Size(); i++)
   {
      Embedding &emb = CoarseFineTr.embeddings[i];
      emb.parent = (i < NumOfElements) ? i : (i - NumOfElements) / 3;
      emb.matrix = (i < NumOfElements) ? 0 : (i - NumOfElements) % 3 + 1;
   }

   NumOfVertices    = vertices.Size();
   NumOfElements    = 4 * NumOfElements;
   NumOfBdrElements = 2 * NumOfBdrElements;
   NumOfFaces       = 0;

   NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   GenerateFaces();

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
   CheckBdrElementOrientation(false);
#endif
}

static inline double sqr(const double &x)
{
   return x*x;
}

void Mesh::UniformRefinement3D_base(Array<int> *f2qf_ptr, DSTable *v_to_v_p)
{
   DeleteLazyTables();

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }

   if (el_to_face == NULL)
   {
      GetElementToFaceTable();
   }

   Array<int> f2qf_loc;
   Array<int> &f2qf = f2qf_ptr ? *f2qf_ptr : f2qf_loc;
   f2qf.SetSize(0);

   int NumOfQuadFaces = 0;
   if (HasGeometry(Geometry::SQUARE))
   {
      if (HasGeometry(Geometry::TRIANGLE))
      {
         f2qf.SetSize(faces.Size());
         for (int i = 0; i < faces.Size(); i++)
         {
            if (faces[i]->GetType() == Element::QUADRILATERAL)
            {
               f2qf[i] = NumOfQuadFaces;
               NumOfQuadFaces++;
            }
         }
      }
      else
      {
         NumOfQuadFaces = faces.Size();
      }
   }

   int hex_counter = 0;
   if (HasGeometry(Geometry::CUBE))
   {
      for (int i = 0; i < elements.Size(); i++)
      {
         if (elements[i]->GetType() == Element::HEXAHEDRON)
         {
            hex_counter++;
         }
      }
   }

   // Map from edge-index to vertex-index, needed for ReorientTetMesh() for
   // parallel meshes.
   Array<int> e2v;
   if (HasGeometry(Geometry::TETRAHEDRON))
   {
      e2v.SetSize(NumOfEdges);

      DSTable *v_to_v_ptr = v_to_v_p;
      if (!v_to_v_p)
      {
         v_to_v_ptr = new DSTable(NumOfVertices);
         GetVertexToVertexTable(*v_to_v_ptr);
      }

      Array<Pair<int,int> > J_v2v(NumOfEdges); // (second vertex id, edge id)
      J_v2v.SetSize(0);
      for (int i = 0; i < NumOfVertices; i++)
      {
         Pair<int,int> *row_start = J_v2v.end();
         for (DSTable::RowIterator it(*v_to_v_ptr, i); !it; ++it)
         {
            J_v2v.Append(Pair<int,int>(it.Column(), it.Index()));
         }
         std::sort(row_start, J_v2v.end());
      }

      for (int i = 0; i < J_v2v.Size(); i++)
      {
         e2v[J_v2v[i].two] = i;
      }

      if (!v_to_v_p)
      {
         delete v_to_v_ptr;
      }
      else
      {
         for (int i = 0; i < NumOfVertices; i++)
         {
            for (DSTable::RowIterator it(*v_to_v_ptr, i); !it; ++it)
            {
               it.SetIndex(e2v[it.Index()]);
            }
         }
      }
   }

   // Offsets for new vertices from edges, faces (quads only), and elements
   // (hexes only); each of these entities generates one new vertex.
   const int oedge = NumOfVertices;
   const int oface = oedge + NumOfEdges;
   const int oelem = oface + NumOfQuadFaces;

   vertices.SetSize(oelem + hex_counter);
   elements.SetSize(8 * NumOfElements);
   CoarseFineTr.embeddings.SetSize(elements.Size());
   hex_counter = 0;
   for (int i = 0; i < NumOfElements; i++)
   {
      const Element::Type el_type = elements[i]->GetType();
      const int attr = elements[i]->GetAttribute();
      int *v = elements[i]->GetVertices();
      const int *e = el_to_edge->GetRow(i);
      const int j = NumOfElements + 7 * i;
      int vv[4], ev[12];

      if (e2v.Size())
      {
         const int ne = el_to_edge->RowSize(i);
         for (int k = 0; k < ne; k++) { ev[k] = e2v[e[k]]; }
         e = ev;
      }

      switch (el_type)
      {
         case Element::TETRAHEDRON:
         {
            for (int ei = 0; ei < 6; ei++)
            {
               for (int k = 0; k < 2; k++)
               {
                  vv[k] = v[tet_t::Edges[ei][k]];
               }
               AverageVertices(vv, 2, oedge+e[ei]);
            }

            // Algorithm for choosing refinement type:
            // 0: smallest octahedron diagonal
            // 1: best aspect ratio
            const int rt_algo = 1;
            // Refinement type:
            // 0: (v0,v1)-(v2,v3), 1: (v0,v2)-(v1,v3), 2: (v0,v3)-(v1,v2)
            // 0:      e0-e5,      1:      e1-e4,      2:      e2-e3
            int rt;
            ElementTransformation *T = GetElementTransformation(i);
            T->SetIntPoint(&Geometries.GetCenter(Geometry::TETRAHEDRON));
            const DenseMatrix &J = T->Jacobian();
            if (rt_algo == 0)
            {
               // smallest octahedron diagonal
               double len_sqr, min_len;

               min_len = sqr(J(0,0)-J(0,1)-J(0,2)) +
                         sqr(J(1,0)-J(1,1)-J(1,2)) +
                         sqr(J(2,0)-J(2,1)-J(2,2));
               rt = 0;

               len_sqr = sqr(J(0,1)-J(0,0)-J(0,2)) +
                         sqr(J(1,1)-J(1,0)-J(1,2)) +
                         sqr(J(2,1)-J(2,0)-J(2,2));
               if (len_sqr < min_len) { min_len = len_sqr; rt = 1; }

               len_sqr = sqr(J(0,2)-J(0,0)-J(0,1)) +
                         sqr(J(1,2)-J(1,0)-J(1,1)) +
                         sqr(J(2,2)-J(2,0)-J(2,1));
               if (len_sqr < min_len) { rt = 2; }
            }
            else
            {
               // best aspect ratio
               double Em_data[18], Js_data[9], Jp_data[9];
               DenseMatrix Em(Em_data, 3, 6);
               DenseMatrix Js(Js_data, 3, 3), Jp(Jp_data, 3, 3);
               double ar1, ar2, kappa, kappa_min;

               for (int s = 0; s < 3; s++)
               {
                  for (int t = 0; t < 3; t++)
                  {
                     Em(t,s) = 0.5*J(t,s);
                  }
               }
               for (int t = 0; t < 3; t++)
               {
                  Em(t,3) = 0.5*(J(t,0)+J(t,1));
                  Em(t,4) = 0.5*(J(t,0)+J(t,2));
                  Em(t,5) = 0.5*(J(t,1)+J(t,2));
               }

               // rt = 0; Em: {0,5,1,2}, {0,5,2,4}
               for (int t = 0; t < 3; t++)
               {
                  Js(t,0) = Em(t,5)-Em(t,0);
                  Js(t,1) = Em(t,1)-Em(t,0);
                  Js(t,2) = Em(t,2)-Em(t,0);
               }
               Geometries.JacToPerfJac(Geometry::TETRAHEDRON, Js, Jp);
               ar1 = Jp.CalcSingularvalue(0)/Jp.CalcSingularvalue(2);
               for (int t = 0; t < 3; t++)
               {
                  Js(t,0) = Em(t,5)-Em(t,0);
                  Js(t,1) = Em(t,2)-Em(t,0);
                  Js(t,2) = Em(t,4)-Em(t,0);
               }
               Geometries.JacToPerfJac(Geometry::TETRAHEDRON, Js, Jp);
               ar2 = Jp.CalcSingularvalue(0)/Jp.CalcSingularvalue(2);
               kappa_min = std::max(ar1, ar2);
               rt = 0;

               // rt = 1; Em: {1,0,4,2}, {1,2,4,5}
               for (int t = 0; t < 3; t++)
               {
                  Js(t,0) = Em(t,0)-Em(t,1);
                  Js(t,1) = Em(t,4)-Em(t,1);
                  Js(t,2) = Em(t,2)-Em(t,1);
               }
               Geometries.JacToPerfJac(Geometry::TETRAHEDRON, Js, Jp);
               ar1 = Jp.CalcSingularvalue(0)/Jp.CalcSingularvalue(2);
               for (int t = 0; t < 3; t++)
               {
                  Js(t,0) = Em(t,2)-Em(t,1);
                  Js(t,1) = Em(t,4)-Em(t,1);
                  Js(t,2) = Em(t,5)-Em(t,1);
               }
               Geometries.JacToPerfJac(Geometry::TETRAHEDRON, Js, Jp);
               ar2 = Jp.CalcSingularvalue(0)/Jp.CalcSingularvalue(2);
               kappa = std::max(ar1, ar2);
               if (kappa < kappa_min) { kappa_min = kappa; rt = 1; }

               // rt = 2; Em: {2,0,1,3}, {2,1,5,3}
               for (int t = 0; t < 3; t++)
               {
                  Js(t,0) = Em(t,0)-Em(t,2);
                  Js(t,1) = Em(t,1)-Em(t,2);
                  Js(t,2) = Em(t,3)-Em(t,2);
               }
               Geometries.JacToPerfJac(Geometry::TETRAHEDRON, Js, Jp);
               ar1 = Jp.CalcSingularvalue(0)/Jp.CalcSingularvalue(2);
               for (int t = 0; t < 3; t++)
               {
                  Js(t,0) = Em(t,1)-Em(t,2);
                  Js(t,1) = Em(t,5)-Em(t,2);
                  Js(t,2) = Em(t,3)-Em(t,2);
               }
               Geometries.JacToPerfJac(Geometry::TETRAHEDRON, Js, Jp);
               ar2 = Jp.CalcSingularvalue(0)/Jp.CalcSingularvalue(2);
               kappa = std::max(ar1, ar2);
               if (kappa < kappa_min) { rt = 2; }
            }

            static const int mv_all[3][4][4] =
            {
               { {0,5,1,2}, {0,5,2,4}, {0,5,4,3}, {0,5,3,1} }, // rt = 0
               { {1,0,4,2}, {1,2,4,5}, {1,5,4,3}, {1,3,4,0} }, // rt = 1
               { {2,0,1,3}, {2,1,5,3}, {2,5,4,3}, {2,4,0,3} }  // rt = 2
            };
            const int (&mv)[4][4] = mv_all[rt];

#ifndef MFEM_USE_MEMALLOC
            elements[j+0] = new Tetrahedron(oedge+e[0], v[1],
                                            oedge+e[3], oedge+e[4], attr);
            elements[j+1] = new Tetrahedron(oedge+e[1], oedge+e[3],
                                            v[2], oedge+e[5], attr);
            elements[j+2] = new Tetrahedron(oedge+e[2], oedge+e[4],
                                            oedge+e[5], v[3], attr);
            for (int k = 0; k < 4; k++)
            {
               elements[j+k+3] =
                  new Tetrahedron(oedge+e[mv[k][0]], oedge+e[mv[k][1]],
                                  oedge+e[mv[k][2]], oedge+e[mv[k][3]], attr);
            }
#else
            Tetrahedron *tet;
            elements[j+0] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[0], v[1], oedge+e[3], oedge+e[4], attr);
            elements[j+1] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[1], oedge+e[3], v[2], oedge+e[5], attr);
            elements[j+2] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[2], oedge+e[4], oedge+e[5], v[3], attr);
            for (int k = 0; k < 4; k++)
            {
               elements[j+k+3] = tet = TetMemory.Alloc();
               tet->Init(oedge+e[mv[k][0]], oedge+e[mv[k][1]],
                         oedge+e[mv[k][2]], oedge+e[mv[k][3]], attr);
            }
#endif

            v[1] = oedge+e[0];
            v[2] = oedge+e[1];
            v[3] = oedge+e[2];
            ((Tetrahedron*)elements[i])->SetRefinementFlag(0);

            CoarseFineTr.embeddings[i].parent = i;
            CoarseFineTr.embeddings[i].matrix = 0;
            for (int k = 0; k < 3; k++)
            {
               CoarseFineTr.embeddings[j+k].parent = i;
               CoarseFineTr.embeddings[j+k].matrix = k+1;
            }
            for (int k = 0; k < 4; k++)
            {
               CoarseFineTr.embeddings[j+k+3].parent = i;
               CoarseFineTr.embeddings[j+k+3].matrix = 4*(rt+1)+k;
            }
         }
         break;

         case Element::WEDGE:
         {
            const int *f = el_to_face->GetRow(i);

            for (int fi = 2; fi < 5; fi++)
            {
               for (int k = 0; k < 4; k++)
               {
                  vv[k] = v[pri_t::FaceVert[fi][k]];
               }
               AverageVertices(vv, 4, oface + f2qf[f[fi]]);
            }

            for (int ei = 0; ei < 9; ei++)
            {
               for (int k = 0; k < 2; k++)
               {
                  vv[k] = v[pri_t::Edges[ei][k]];
               }
               AverageVertices(vv, 2, oedge+e[ei]);
            }

            const int qf2 = f2qf[f[2]];
            const int qf3 = f2qf[f[3]];
            const int qf4 = f2qf[f[4]];

            elements[j+0] = new Wedge(oedge+e[1], oedge+e[2], oedge+e[0],
                                      oface+qf3, oface+qf4, oface+qf2,
                                      attr);
            elements[j+1] = new Wedge(oedge+e[0], v[1], oedge+e[1],
                                      oface+qf2, oedge+e[7], oface+qf3,
                                      attr);
            elements[j+2] = new Wedge(oedge+e[2], oedge+e[1], v[2],
                                      oface+qf4, oface+qf3, oedge+e[8],
                                      attr);
            elements[j+3] = new Wedge(oedge+e[6], oface+qf2, oface+qf4,
                                      v[3], oedge+e[3], oedge+e[5],
                                      attr);
            elements[j+4] = new Wedge(oface+qf3, oface+qf4, oface+qf2,
                                      oedge+e[4], oedge+e[5], oedge+e[3],
                                      attr);
            elements[j+5] = new Wedge(oface+qf2, oedge+e[7], oface+qf3,
                                      oedge+e[3], v[4], oedge+e[4],
                                      attr);
            elements[j+6] = new Wedge(oface+qf4, oface+qf3, oedge+e[8],
                                      oedge+e[5], oedge+e[4], v[5],
                                      attr);

            v[1] = oedge+e[0];
            v[2] = oedge+e[2];
            v[3] = oedge+e[6];
            v[4] = oface+qf2;
            v[5] = oface+qf4;
         }
         break;

         case Element::HEXAHEDRON:
         {
            const int *f = el_to_face->GetRow(i);
            const int he = hex_counter;
            hex_counter++;

            const int *qf;
            int qf_data[6];
            if (f2qf.Size() == 0)
            {
               qf = f;
            }
            else
            {
               for (int k = 0; k < 6; k++) { qf_data[k] = f2qf[f[k]]; }
               qf = qf_data;
            }

            AverageVertices(v, 8, oelem+he);

            for (int fi = 0; fi < 6; fi++)
            {
               for (int k = 0; k < 4; k++)
               {
                  vv[k] = v[hex_t::FaceVert[fi][k]];
               }
               AverageVertices(vv, 4, oface + qf[fi]);
            }

            for (int ei = 0; ei < 12; ei++)
            {
               for (int k = 0; k < 2; k++)
               {
                  vv[k] = v[hex_t::Edges[ei][k]];
               }
               AverageVertices(vv, 2, oedge+e[ei]);
            }

            elements[j+0] = new Hexahedron(oedge+e[0], v[1], oedge+e[1],
                                           oface+qf[0], oface+qf[1], oedge+e[9],
                                           oface+qf[2], oelem+he, attr);
            elements[j+1] = new Hexahedron(oface+qf[0], oedge+e[1], v[2],
                                           oedge+e[2], oelem+he, oface+qf[2],
                                           oedge+e[10], oface+qf[3], attr);
            elements[j+2] = new Hexahedron(oedge+e[3], oface+qf[0], oedge+e[2],
                                           v[3], oface+qf[4], oelem+he,
                                           oface+qf[3], oedge+e[11], attr);
            elements[j+3] = new Hexahedron(oedge+e[8], oface+qf[1], oelem+he,
                                           oface+qf[4], v[4], oedge+e[4],
                                           oface+qf[5], oedge+e[7], attr);
            elements[j+4] = new Hexahedron(oface+qf[1], oedge+e[9], oface+qf[2],
                                           oelem+he, oedge+e[4], v[5],
                                           oedge+e[5], oface+qf[5], attr);
            elements[j+5] = new Hexahedron(oelem+he, oface+qf[2], oedge+e[10],
                                           oface+qf[3], oface+qf[5], oedge+e[5],
                                           v[6], oedge+e[6], attr);
            elements[j+6] = new Hexahedron(oface+qf[4], oelem+he, oface+qf[3],
                                           oedge+e[11], oedge+e[7], oface+qf[5],
                                           oedge+e[6], v[7], attr);

            v[1] = oedge+e[0];
            v[2] = oface+qf[0];
            v[3] = oedge+e[3];
            v[4] = oedge+e[8];
            v[5] = oface+qf[1];
            v[6] = oelem+he;
            v[7] = oface+qf[4];
         }
         break;

         default:
            MFEM_ABORT("Unknown 3D element type \"" << el_type << "\"");
            break;
      }
   }

   boundary.SetSize(4 * NumOfBdrElements);
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      const Element::Type bdr_el_type = boundary[i]->GetType();
      const int attr = boundary[i]->GetAttribute();
      int *v = boundary[i]->GetVertices();
      const int *e = bel_to_edge->GetRow(i);
      const int j = NumOfBdrElements + 3 * i;
      int ev[4];

      if (e2v.Size())
      {
         const int ne = bel_to_edge->RowSize(i);
         for (int k = 0; k < ne; k++) { ev[k] = e2v[e[k]]; }
         e = ev;
      }

      if (bdr_el_type == Element::TRIANGLE)
      {
         boundary[j+0] = new Triangle(oedge+e[1], oedge+e[2], oedge+e[0], attr);
         boundary[j+1] = new Triangle(oedge+e[0], v[1], oedge+e[1], attr);
         boundary[j+2] = new Triangle(oedge+e[2], oedge+e[1], v[2], attr);

         v[1] = oedge+e[0];
         v[2] = oedge+e[2];
      }
      else if (bdr_el_type == Element::QUADRILATERAL)
      {
         const int qf =
            (f2qf.Size() == 0) ? be_to_face[i] : f2qf[be_to_face[i]];

         boundary[j+0] = new Quadrilateral(oedge+e[0], v[1], oedge+e[1],
                                           oface+qf, attr);
         boundary[j+1] = new Quadrilateral(oface+qf, oedge+e[1], v[2],
                                           oedge+e[2], attr);
         boundary[j+2] = new Quadrilateral(oedge+e[3], oface+qf,
                                           oedge+e[2], v[3], attr);

         v[1] = oedge+e[0];
         v[2] = oface+qf;
         v[3] = oedge+e[3];
      }
      else
      {
         MFEM_ABORT("boundary Element is not a triangle or a quad!");
      }
   }

   static const double A = 0.0, B = 0.5, C = 1.0;
   static double tet_children[3*4*16] =
   {
      A,A,A, B,A,A, A,B,A, A,A,B,
      B,A,A, C,A,A, B,B,A, B,A,B,
      A,B,A, B,B,A, A,C,A, A,B,B,
      A,A,B, B,A,B, A,B,B, A,A,C,
      // edge coordinates:
      //    0 -> B,A,A  1 -> A,B,A  2 -> A,A,B
      //    3 -> B,B,A  4 -> B,A,B  5 -> A,B,B
      // rt = 0: {0,5,1,2}, {0,5,2,4}, {0,5,4,3}, {0,5,3,1}
      B,A,A, A,B,B, A,B,A, A,A,B,
      B,A,A, A,B,B, A,A,B, B,A,B,
      B,A,A, A,B,B, B,A,B, B,B,A,
      B,A,A, A,B,B, B,B,A, A,B,A,
      // rt = 1: {1,0,4,2}, {1,2,4,5}, {1,5,4,3}, {1,3,4,0}
      A,B,A, B,A,A, B,A,B, A,A,B,
      A,B,A, A,A,B, B,A,B, A,B,B,
      A,B,A, A,B,B, B,A,B, B,B,A,
      A,B,A, B,B,A, B,A,B, B,A,A,
      // rt = 2: {2,0,1,3}, {2,1,5,3}, {2,5,4,3}, {2,4,0,3}
      A,A,B, B,A,A, A,B,A, B,B,A,
      A,A,B, A,B,A, A,B,B, B,B,A,
      A,A,B, A,B,B, B,A,B, B,B,A,
      A,A,B, B,A,B, B,A,A, B,B,A
   };
   static double pri_children[3*6*8] =
   {
      A,A,A, B,A,A, A,B,A, A,A,B, B,A,B, A,B,B,
      B,B,A, A,B,A, B,A,A, B,B,B, A,B,B, B,A,B,
      B,A,A, C,A,A, B,B,A, B,A,B, C,A,B, B,B,B,
      A,B,A, B,B,A, A,C,A, A,B,B, B,B,B, A,C,B,
      A,A,B, B,A,B, A,B,B, A,A,C, B,A,C, A,B,C,
      B,B,B, A,B,B, B,A,B, B,B,C, A,B,C, B,A,C,
      B,A,B, C,A,B, B,B,B, B,A,C, C,A,C, B,B,C,
      A,B,B, B,B,B, A,C,B, A,B,C, B,B,C, A,C,C
   };
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

   CoarseFineTr.point_matrices[Geometry::TETRAHEDRON].
   UseExternalData(tet_children, 3, 4, 16);
   CoarseFineTr.point_matrices[Geometry::PRISM].
   UseExternalData(pri_children, 3, 6, 8);
   CoarseFineTr.point_matrices[Geometry::CUBE].
   UseExternalData(hex_children, 3, 8, 8);

   for (int i = 0; i < elements.Size(); i++)
   {
      // Tetrahedron elements are handled above:
      if (elements[i]->GetType() == Element::TETRAHEDRON) { continue; }
      Embedding &emb = CoarseFineTr.embeddings[i];
      emb.parent = (i < NumOfElements) ? i : (i - NumOfElements) / 7;
      emb.matrix = (i < NumOfElements) ? 0 : (i - NumOfElements) % 7 + 1;
   }

   NumOfVertices    = vertices.Size();
   NumOfElements    = 8 * NumOfElements;
   NumOfBdrElements = 4 * NumOfBdrElements;

   GetElementToFaceTable();
   GenerateFaces();

#ifdef MFEM_DEBUG
   CheckBdrElementOrientation(false);
#endif

   NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();
}

void Mesh::LocalRefinement(const Array<int> &marked_el, int type)
{
   int i, j, ind, nedges;
   Array<int> v;

   DeleteLazyTables();

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
      CoarseFineTr.point_matrices[Geometry::SEGMENT].
      UseExternalData(seg_children, 1, 2, 3);

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
      // 1. Hash table of vertex to vertex connections corresponding to refined
      //    edges.
      HashTable<Hashed2> v_to_v;

      MFEM_VERIFY(GetNE() == 0 ||
                  ((Tetrahedron*)elements[0])->GetRefinementFlag() != 0,
                  "tetrahedral mesh is not marked for refinement:"
                  " call Finalize(true)");

      // 2. Do the red refinement.
      int ii;
      switch (type)
      {
         case 1:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v);
            }
            break;
         case 2:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v);

               Bisection(NumOfElements - 1, v_to_v);
               Bisection(marked_el[i], v_to_v);
            }
            break;
         case 3:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v);

               ii = NumOfElements - 1;
               Bisection(ii, v_to_v);
               Bisection(NumOfElements - 1, v_to_v);
               Bisection(ii, v_to_v);

               Bisection(marked_el[i], v_to_v);
               Bisection(NumOfElements-1, v_to_v);
               Bisection(marked_el[i], v_to_v);
            }
            break;
      }

      // 3. Do the green refinement (to get conforming mesh).
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
            // ((Tetrahedron *)elements[i])->
            // ParseRefinementFlag(redges, type, flag);
            // if (flag > max_gen)  max_gen = flag;
            if (elements[i]->NeedRefinement(v_to_v))
            {
               need_refinement = 1;
               Bisection(i, v_to_v);
            }
         }
      }
      while (need_refinement == 1);

      // mfem::out << "Maximum generation: " << max_gen << endl;

      // 4. Update the boundary elements.
      do
      {
         need_refinement = 0;
         for (i = 0; i < NumOfBdrElements; i++)
            if (boundary[i]->NeedRefinement(v_to_v))
            {
               need_refinement = 1;
               BdrBisection(i, v_to_v);
            }
      }
      while (need_refinement == 1);

      NumOfVertices = vertices.Size();
      NumOfBdrElements = boundary.Size();

      // 5. Update element-to-edge and element-to-face relations.
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

   DeleteLazyTables();

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

double Mesh::AggregateError(const Array<double> &elem_error,
                            const int *fine, int nfine, int op)
{
   double error = 0.0;
   for (int i = 0; i < nfine; i++)
   {
      MFEM_VERIFY(fine[i] < elem_error.Size(), "");

      double err_fine = elem_error[fine[i]];
      switch (op)
      {
         case 0: error = std::min(error, err_fine); break;
         case 1: error += err_fine; break;
         case 2: error = std::max(error, err_fine); break;
      }
   }
   return error;
}

bool Mesh::NonconformingDerefinement(Array<double> &elem_error,
                                     double threshold, int nc_limit, int op)
{
   MFEM_VERIFY(ncmesh, "Only supported for non-conforming meshes.");
   MFEM_VERIFY(!NURBSext, "Derefinement of NURBS meshes is not supported. "
               "Project the NURBS to Nodes first.");

   DeleteLazyTables();

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

      double error =
         AggregateError(elem_error, dt.GetRow(i), dt.RowSize(i), op);

      if (error < threshold) { derefs.Append(i); }
   }

   if (!derefs.Size()) { return false; }

   ncmesh->Derefine(derefs);

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

   return true;
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

   DeleteTables();

   ncmesh.GetMeshComponents(vertices, elements, boundary);

   NumOfVertices = vertices.Size();
   NumOfElements = elements.Size();
   NumOfBdrElements = boundary.Size();

   SetMeshGen(); // set the mesh type: 'meshgen', ...

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
   mfem::Swap(mesh_geoms, other.mesh_geoms);

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

void Mesh::UniformRefinement(int ref_algo)
{
   if (NURBSext)
   {
      NURBSUniformRefinement();
   }
   else if (ref_algo == 0 && Dim == 3 && meshgen == 1)
   {
      UniformRefinement3D();
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
   else
   {
      switch (Dim)
      {
         case 2: UniformRefinement2D(); break;
         case 3: UniformRefinement3D(); break;
         default: MFEM_ABORT("internal error");
      }
   }
}

void Mesh::GeneralRefinement(const Array<Refinement> &refinements,
                             int nonconforming, int nc_limit)
{
   if (ncmesh)
   {
      nonconforming = 1;
   }
   else if (Dim == 1 || (Dim == 3 && (meshgen & 1)))
   {
      nonconforming = 0;
   }
   else if (nonconforming < 0)
   {
      // determine if nonconforming refinement is suitable
      if (meshgen & 2)
      {
         nonconforming = 1;
      }
      else
      {
         nonconforming = 0;
      }
   }

   if (nonconforming)
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
          (triangles_nonconforming && Dim == 2 && (meshgen & 1)))
      {
         MFEM_VERIFY(GetNumGeometries(Dim) <= 1,
                     "mixed meshes are not supported");
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
   else
   {
      MFEM_ABORT("Bisection for now works only for triangles.");
   }
}

void Mesh::Bisection(int i, HashTable<Hashed2> &v_to_v)
{
   int *vert;
   int v[2][4], v_new, bisect, t;
   Element *el = elements[i];
   Vertex V;

   t = el->GetType();
   if (t == Element::TETRAHEDRON)
   {
      int j, type, new_type, old_redges[2], new_redges[2][2], flag;
      Tetrahedron *tet = (Tetrahedron *) el;

      MFEM_VERIFY(tet->GetRefinementFlag() != 0,
                  "TETRAHEDRON element is not marked for refinement.");

      vert = tet->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v.FindId(vert[0], vert[1]);
      if (bisect == -1)
      {
         v_new = NumOfVertices + v_to_v.GetId(vert[0],vert[1]);
         for (j = 0; j < 3; j++)
         {
            V(j) = 0.5 * (vertices[vert[0]](j) + vertices[vert[1]](j));
         }
         vertices.Append(V);
      }
      else
      {
         v_new = NumOfVertices + bisect;
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
      MFEM_ABORT("Bisection with HashTable for now works only for tetrahedra.");
   }
}

void Mesh::BdrBisection(int i, const HashTable<Hashed2> &v_to_v)
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
      bisect = v_to_v.FindId(vert[0], vert[1]);
      MFEM_ASSERT(bisect >= 0, "");
      v_new = NumOfVertices + bisect;
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
      MFEM_ABORT("Bisection of boundary elements with HashTable works only for"
                 " triangles!");
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
   map<Geometry::Type,DenseTensor> &pms = CoarseFineTr.point_matrices;
   map<Geometry::Type,DenseTensor>::iterator pms_iter;
   for (pms_iter = pms.begin(); pms_iter != pms.end(); ++pms_iter)
   {
      pms_iter->second.SetSize(0, 0, 0);
   }
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

   Mesh::GeometryList elem_geoms(*this);
   for (int i = 0; i < elem_geoms.Size(); i++)
   {
      const Geometry::Type geom = elem_geoms[i];
      if (CoarseFineTr.point_matrices[geom].SizeK()) { continue; }

      if (geom == Geometry::TRIANGLE ||
          geom == Geometry::TETRAHEDRON)
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

         DenseTensor &pmats = CoarseFineTr.point_matrices[geom];
         pmats.SetSize(Dim, Dim+1, mat_no.size());

         // calculate the point matrices used
         std::map<unsigned, int>::iterator it;
         for (it = mat_no.begin(); it != mat_no.end(); ++it)
         {
            if (geom == Geometry::TRIANGLE)
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
         MFEM_ABORT("Don't know how to construct CoarseFineTransformations for"
                    " geom = " << geom);
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
       "# PRISM       = 6\n"
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
         MFEM_ASSERT(Dim != 0 || dofs.Size() == 1,
                     "Point meshes should have a single dof per element");
         size += dofs.Size() + 1;
      }
      out << "CELLS " << NumOfElements << ' ' << size << '\n';
      const char *fec_name = Nodes->FESpace()->FEColl()->Name();

      if (!strcmp(fec_name, "Linear") ||
          !strcmp(fec_name, "H1_0D_P1") ||
          !strcmp(fec_name, "H1_1D_P1") ||
          !strcmp(fec_name, "H1_2D_P1") ||
          !strcmp(fec_name, "H1_3D_P1"))
      {
         order = 1;
      }
      else if (!strcmp(fec_name, "Quadratic") ||
               !strcmp(fec_name, "H1_1D_P2") ||
               !strcmp(fec_name, "H1_2D_P2") ||
               !strcmp(fec_name, "H1_3D_P2"))
      {
         order = 2;
      }
      if (order == -1)
      {
         mfem::err << "Mesh::PrintVTK : can not save '"
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
               case Geometry::SEGMENT:
               case Geometry::TRIANGLE:
               case Geometry::SQUARE:
                  vtk_mfem = vtk_quadratic_hex; break; // identity map
               case Geometry::TETRAHEDRON:
                  vtk_mfem = vtk_quadratic_tet; break;
               case Geometry::PRISM:
                  vtk_mfem = vtk_quadratic_wedge; break;
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
      Geometry::Type geom_type = GetElement(i)->GetGeometryType();
      if (order == 1)
      {
         switch (geom_type)
         {
            case Geometry::POINT:        vtk_cell_type = 1;   break;
            case Geometry::SEGMENT:      vtk_cell_type = 3;   break;
            case Geometry::TRIANGLE:     vtk_cell_type = 5;   break;
            case Geometry::SQUARE:       vtk_cell_type = 9;   break;
            case Geometry::TETRAHEDRON:  vtk_cell_type = 10;  break;
            case Geometry::CUBE:         vtk_cell_type = 12;  break;
            case Geometry::PRISM:        vtk_cell_type = 13;  break;
            default: break;
         }
      }
      else if (order == 2)
      {
         switch (geom_type)
         {
            case Geometry::SEGMENT:      vtk_cell_type = 21;  break;
            case Geometry::TRIANGLE:     vtk_cell_type = 22;  break;
            case Geometry::SQUARE:       vtk_cell_type = 28;  break;
            case Geometry::TETRAHEDRON:  vtk_cell_type = 24;  break;
            case Geometry::CUBE:         vtk_cell_type = 29;  break;
            case Geometry::PRISM:        vtk_cell_type = 32;  break;
            default: break;
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
      Geometry::Type geom = GetElementBaseGeometry(i);
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
      Geometry::Type geom = GetElementBaseGeometry(i);
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
      Geometry::Type geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      Array<int> &RG = RefG->RefGeoms;
      int vtk_cell_type = 5;

      switch (geom)
      {
         case Geometry::POINT:        vtk_cell_type = 1;   break;
         case Geometry::SEGMENT:      vtk_cell_type = 3;   break;
         case Geometry::TRIANGLE:     vtk_cell_type = 5;   break;
         case Geometry::SQUARE:       vtk_cell_type = 9;   break;
         case Geometry::TETRAHEDRON:  vtk_cell_type = 10;  break;
         case Geometry::CUBE:         vtk_cell_type = 12;  break;
         case Geometry::PRISM:        vtk_cell_type = 13;  break;
         default:
            MFEM_ABORT("Unrecognized VTK element type \"" << geom << "\"");
            break;
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
      Geometry::Type geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      int attr = GetAttribute(i);
      for (int j = 0; j < RefG->RefGeoms.Size(); j += nv)
      {
         out << attr << '\n';
      }
   }

   if (Dim > 1)
   {
      Array<int> coloring;
      srand((unsigned)time(0));
      double a = double(rand()) / (double(RAND_MAX) + 1.);
      int el0 = (int)floor(a * GetNE());
      GetElementColoring(coloring, el0);
      out << "SCALARS element_coloring int\n"
          << "LOOKUP_TABLE default\n";
      for (int i = 0; i < GetNE(); i++)
      {
         Geometry::Type geom = GetElementBaseGeometry(i);
         int nv = Geometries.GetVertices(geom)->GetNPoints();
         RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
         for (int j = 0; j < RefG->RefGeoms.Size(); j += nv)
         {
            out << coloring[i] + 1 << '\n';
         }
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
       "# PRISM       = 6\n"
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
   }
   //  Dim is 3
   else if (meshgen == 1)
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
   }
   //  Dim is 3
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

   for (i = 0; i < NumOfVertices; i++)
   {
      delete [] vown[i];
   }

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
       "# PRISM       = 6\n"
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
      for (const int * iface = j_AF_f + i_AF_f[iAF];
           iface < j_AF_f + i_AF_f[iAF+1];
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

int Mesh::FindPoints(DenseMatrix &point_mat, Array<int>& elem_ids,
                     Array<IntegrationPoint>& ips, bool warn,
                     InverseElementTransformation *inv_trans)
{
   const int npts = point_mat.Width();
   if (!npts) { return 0; }
   MFEM_VERIFY(point_mat.Height() == spaceDim,"Invalid points matrix");
   elem_ids.SetSize(npts);
   ips.SetSize(npts);
   elem_ids = -1;
   if (!GetNE()) { return 0; }

   double *data = point_mat.GetData();
   InverseElementTransformation *inv_tr = inv_trans;
   inv_tr = inv_tr ? inv_tr : new InverseElementTransformation;

   // For each point in 'point_mat', find the element whose center is closest.
   Vector min_dist(npts);
   Array<int> e_idx(npts);
   min_dist = std::numeric_limits<double>::max();
   e_idx = -1;

   Vector pt(spaceDim);
   for (int i = 0; i < GetNE(); i++)
   {
      GetElementTransformation(i)->Transform(
         Geometries.GetCenter(GetElementBaseGeometry(i)), pt);
      for (int k = 0; k < npts; k++)
      {
         double dist = pt.DistanceTo(data+k*spaceDim);
         if (dist < min_dist(k))
         {
            min_dist(k) = dist;
            e_idx[k] = i;
         }
      }
   }

   // Checks if the points lie in the closest element
   int pts_found = 0;
   pt.NewDataAndSize(NULL, spaceDim);
   for (int k = 0; k < npts; k++)
   {
      pt.SetData(data+k*spaceDim);
      inv_tr->SetTransformation(*GetElementTransformation(e_idx[k]));
      int res = inv_tr->Transform(pt, ips[k]);
      if (res == InverseElementTransformation::Inside)
      {
         elem_ids[k] = e_idx[k];
         pts_found++;
      }
   }
   if (pts_found != npts)
   {
      Array<int> vertices;
      Table *vtoel = GetVertexToElementTable();
      for (int k = 0; k < npts; k++)
      {
         if (elem_ids[k] != -1) { continue; }
         // Try all vertex-neighbors of element e_idx[k]
         pt.SetData(data+k*spaceDim);
         GetElementVertices(e_idx[k], vertices);
         for (int v = 0; v < vertices.Size(); v++)
         {
            int vv = vertices[v];
            int ne = vtoel->RowSize(vv);
            const int* els = vtoel->GetRow(vv);
            for (int e = 0; e < ne; e++)
            {
               if (els[e] == e_idx[k]) { continue; }
               inv_tr->SetTransformation(*GetElementTransformation(els[e]));
               int res = inv_tr->Transform(pt, ips[k]);
               if (res == InverseElementTransformation::Inside)
               {
                  elem_ids[k] = els[e];
                  pts_found++;
                  goto next_point;
               }
            }
         }
      next_point: ;
      }
      delete vtoel;
   }
   if (inv_trans == NULL) { delete inv_tr; }

   if (warn && pts_found != npts)
   {
      MFEM_WARNING((npts-pts_found) << " points were not found");
   }
   return pts_found;
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
      mfem::err << "Extrude1D : Not a 1D mesh!" << endl;
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
         mfem::err << "Extrude1D : The mesh uses unknown FE collection : "
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

Mesh *Extrude2D(Mesh *mesh, const int nz, const double sz)
{
   if (mesh->Dimension() != 2)
   {
      mfem::err << "Extrude2D : Not a 2D mesh!" << endl;
      mfem_error();
   }

   int nvz = nz + 1;
   int nvt = mesh->GetNV() * nvz;

   Mesh *mesh3d = new Mesh(3, nvt, mesh->GetNE()*nz,
                           mesh->GetNBE()*nz+2*mesh->GetNE());

   bool wdgMesh = false;
   bool hexMesh = false;

   // vertices
   double vc[3];
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      vc[0] = mesh->GetVertex(i)[0];
      vc[1] = mesh->GetVertex(i)[1];
      for (int j = 0; j < nvz; j++)
      {
         vc[2] = sz * (double(j) / nz);
         mesh3d->AddVertex(vc);
      }
   }
   // elements
   Array<int> vert;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const Element *elem = mesh->GetElement(i);
      elem->GetVertices(vert);
      const int attr = elem->GetAttribute();
      Geometry::Type geom = elem->GetGeometryType();
      switch (geom)
      {
         case Geometry::TRIANGLE:
            wdgMesh = true;
            for (int j = 0; j < nz; j++)
            {
               int pv[6];
               pv[0] = vert[0] * nvz + j;
               pv[1] = vert[1] * nvz + j;
               pv[2] = vert[2] * nvz + j;
               pv[3] = vert[0] * nvz + (j + 1) % nvz;
               pv[4] = vert[1] * nvz + (j + 1) % nvz;
               pv[5] = vert[2] * nvz + (j + 1) % nvz;

               mesh3d->AddWedge(pv, attr);
            }
            break;
         case Geometry::SQUARE:
            hexMesh = true;
            for (int j = 0; j < nz; j++)
            {
               int hv[8];
               hv[0] = vert[0] * nvz + j;
               hv[1] = vert[1] * nvz + j;
               hv[2] = vert[2] * nvz + j;
               hv[3] = vert[3] * nvz + j;
               hv[4] = vert[0] * nvz + (j + 1) % nvz;
               hv[5] = vert[1] * nvz + (j + 1) % nvz;
               hv[6] = vert[2] * nvz + (j + 1) % nvz;
               hv[7] = vert[3] * nvz + (j + 1) % nvz;

               mesh3d->AddHex(hv, attr);
            }
            break;
         default:
            mfem::err << "Extrude2D : Invalid 2D element type \'"
                      << geom << "\'" << endl;
            mfem_error();
            break;
      }
   }
   // 3D boundary from the 2D boundary
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const Element *elem = mesh->GetBdrElement(i);
      elem->GetVertices(vert);
      const int attr = elem->GetAttribute();
      for (int j = 0; j < nz; j++)
      {
         int qv[4];
         qv[0] = vert[0] * nvz + j;
         qv[1] = vert[1] * nvz + j;
         qv[2] = vert[1] * nvz + (j + 1) % nvz;
         qv[3] = vert[0] * nvz + (j + 1) % nvz;

         mesh3d->AddBdrQuad(qv, attr);
      }
   }

   // 3D boundary from the 2D elements (bottom + top)
   int nba = (mesh->bdr_attributes.Size() > 0 ?
              mesh->bdr_attributes.Max() : 0);
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const Element *elem = mesh->GetElement(i);
      elem->GetVertices(vert);
      const int attr = nba + elem->GetAttribute();
      Geometry::Type geom = elem->GetGeometryType();
      switch (geom)
      {
         case Geometry::TRIANGLE:
         {
            int tv[3];
            tv[0] = vert[0] * nvz;
            tv[1] = vert[2] * nvz;
            tv[2] = vert[1] * nvz;

            mesh3d->AddBdrTriangle(tv, attr);

            tv[0] = vert[0] * nvz + nz;
            tv[1] = vert[1] * nvz + nz;
            tv[2] = vert[2] * nvz + nz;

            mesh3d->AddBdrTriangle(tv, attr);
         }
         break;
         case Geometry::SQUARE:
         {
            int qv[4];
            qv[0] = vert[0] * nvz;
            qv[1] = vert[3] * nvz;
            qv[2] = vert[2] * nvz;
            qv[3] = vert[1] * nvz;

            mesh3d->AddBdrQuad(qv, attr);

            qv[0] = vert[0] * nvz + nz;
            qv[1] = vert[1] * nvz + nz;
            qv[2] = vert[2] * nvz + nz;
            qv[3] = vert[3] * nvz + nz;

            mesh3d->AddBdrQuad(qv, attr);
         }
         break;
         default:
            mfem::err << "Extrude2D : Invalid 2D element type \'"
                      << geom << "\'" << endl;
            mfem_error();
            break;
      }
   }

   if ( hexMesh && wdgMesh )
   {
      mesh3d->FinalizeMesh(0, false);
   }
   else if ( hexMesh )
   {
      mesh3d->FinalizeHexMesh(1, 0, false);
   }
   else if ( wdgMesh )
   {
      mesh3d->FinalizeWedgeMesh(1, 0, false);
   }

   GridFunction *nodes = mesh->GetNodes();
   if (nodes)
   {
      // duplicate the fec of the 2D mesh so that it can be deleted safely
      // along with its nodes, fes and fec
      FiniteElementCollection *fec3d = NULL;
      FiniteElementSpace *fes3d;
      const char *name = nodes->FESpace()->FEColl()->Name();
      string cname = name;
      if (cname == "Linear")
      {
         fec3d = new LinearFECollection;
      }
      else if (cname == "Quadratic")
      {
         fec3d = new QuadraticFECollection;
      }
      else if (cname == "Cubic")
      {
         fec3d = new CubicFECollection;
      }
      else if (!strncmp(name, "H1_", 3))
      {
         fec3d = new H1_FECollection(atoi(name + 7), 3);
      }
      else if (!strncmp(name, "L2_T", 4))
      {
         fec3d = new L2_FECollection(atoi(name + 10), 3, atoi(name + 4));
      }
      else if (!strncmp(name, "L2_", 3))
      {
         fec3d = new L2_FECollection(atoi(name + 7), 3);
      }
      else
      {
         delete mesh3d;
         mfem::err << "Extrude3D : The mesh uses unknown FE collection : "
                   << cname << endl;
         mfem_error();
      }
      fes3d = new FiniteElementSpace(mesh3d, fec3d, 3);
      mesh3d->SetNodalFESpace(fes3d);
      GridFunction *nodes3d = mesh3d->GetNodes();
      nodes3d->MakeOwner(fec3d);

      NodeExtrudeCoefficient ecoeff(3, nz, sz);
      Vector lnodes;
      Array<int> vdofs3d;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         ElementTransformation &T = *mesh->GetElementTransformation(i);
         for (int j = nz-1; j >= 0; j--)
         {
            fes3d->GetElementVDofs(i*nz+j, vdofs3d);
            lnodes.SetSize(vdofs3d.Size());
            ecoeff.SetLayer(j);
            fes3d->GetFE(i*nz+j)->Project(ecoeff, T, lnodes);
            nodes3d->SetSubVector(vdofs3d, lnodes);
         }
      }
   }
   return mesh3d;
}

}
