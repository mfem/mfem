// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of data type mesh

#include "mesh_headers.hpp"
#include "vtkhdf.hpp"
#include "../fem/fem.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/binaryio.hpp"
#include "../general/text.hpp"
#include "../general/device.hpp"
#include "../general/tic_toc.hpp"
#include "../general/gecko.hpp"
#include "../general/kdtree.hpp"
#include "../general/sets.hpp"
#include "../fem/quadinterpolator.hpp"

// headers already included by mesh.hpp: <iostream>, <array>, <map>, <memory>
#include <sstream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstring>
#include <ctime>
#include <functional>
#include <set>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <list>

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

using namespace std;

namespace mfem
{

void Mesh::GetElementJacobian(int i, DenseMatrix &J, const IntegrationPoint *ip)
{
   Geometry::Type geom = GetElementBaseGeometry(i);
   ElementTransformation *eltransf = GetElementTransformation(i);
   if (ip == NULL)
   {
      eltransf->SetIntPoint(&Geometries.GetCenter(geom));
   }
   else
   {
      eltransf->SetIntPoint(ip);
   }
   Geometries.JacToPerfJac(geom, eltransf->Jacobian(), J);
}

void Mesh::GetElementCenter(int i, Vector &center)
{
   center.SetSize(spaceDim);
   int geom = GetElementBaseGeometry(i);
   ElementTransformation *eltransf = GetElementTransformation(i);
   eltransf->Transform(Geometries.GetCenter(geom), center);
}

real_t Mesh::GetElementSize(ElementTransformation *T, int type) const
{
   DenseMatrix J(spaceDim, Dim);

   Geometry::Type geom = T->GetGeometryType();
   T->SetIntPoint(&Geometries.GetCenter(geom));
   Geometries.JacToPerfJac(geom, T->Jacobian(), J);

   if (type == 0)
   {
      return pow(fabs(J.Weight()), 1./Dim);
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

real_t Mesh::GetElementSize(int i, int type)
{
   return GetElementSize(GetElementTransformation(i), type);
}

real_t Mesh::GetElementSize(int i, const Vector &dir)
{
   DenseMatrix J(spaceDim, Dim);
   Vector d_hat(Dim);
   GetElementJacobian(i, J);
   J.MultTranspose(dir, d_hat);
   return sqrt((d_hat * d_hat) / (dir * dir));
}

real_t Mesh::GetElementVolume(int i)
{
   ElementTransformation *et = GetElementTransformation(i);
   const IntegrationRule &ir = IntRules.Get(GetElementBaseGeometry(i),
                                            et->OrderJ());
   real_t volume = 0.0;
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
      real_t *coord;
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
            RefG = GlobGeometryRefiner.Refine(GetFaceGeometry(fn), ref);
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

void Mesh::GetCharacteristics(real_t &h_min, real_t &h_max,
                              real_t &kappa_min, real_t &kappa_max,
                              Vector *Vh, Vector *Vk)
{
   int i, dim, sdim;
   DenseMatrix J;
   real_t h, kappa;

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
      h = pow(fabs(J.Weight()), 1.0/real_t(dim));
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
                                   std::ostream &os)
{
   for (int g = Geometry::DimStart[dim], first = 1;
        g < Geometry::DimStart[dim+1]; g++)
   {
      if (!num_elems_by_geom[g]) { continue; }
      if (!first) { os << " + "; }
      else { first = 0; }
      os << num_elems_by_geom[g] << ' ' << Geometry::Name[g] << "(s)";
   }
}

void Mesh::PrintCharacteristics(Vector *Vh, Vector *Vk, std::ostream &os)
{
   real_t h_min, h_max, kappa_min, kappa_max;

   os << "Mesh Characteristics:";

   this->GetCharacteristics(h_min, h_max, kappa_min, kappa_max, Vh, Vk);

   Array<int> num_elems_by_geom(Geometry::NumGeom);
   num_elems_by_geom = 0;
   for (int i = 0; i < GetNE(); i++)
   {
      num_elems_by_geom[GetElementBaseGeometry(i)]++;
   }

   os << '\n'
      << "Dimension          : " << Dimension() << '\n'
      << "Space dimension    : " << SpaceDimension();
   if (Dim == 0)
   {
      os << '\n'
         << "Number of vertices : " << GetNV() << '\n'
         << "Number of elements : " << GetNE() << '\n'
         << "Number of bdr elem : " << GetNBE() << '\n';
   }
   else if (Dim == 1)
   {
      os << '\n'
         << "Number of vertices : " << GetNV() << '\n'
         << "Number of elements : " << GetNE() << '\n'
         << "Number of bdr elem : " << GetNBE() << '\n'
         << "h_min              : " << h_min << '\n'
         << "h_max              : " << h_max << '\n';
   }
   else if (Dim == 2)
   {
      os << '\n'
         << "Number of vertices : " << GetNV() << '\n'
         << "Number of edges    : " << GetNEdges() << '\n'
         << "Number of elements : " << GetNE() << "  --  ";
      PrintElementsByGeometry(2, num_elems_by_geom, os);
      os << '\n'
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
         num_bdr_elems_by_geom[GetBdrElementGeometry(i)]++;
      }
      Array<int> num_faces_by_geom(Geometry::NumGeom);
      num_faces_by_geom = 0;
      for (int i = 0; i < GetNFaces(); i++)
      {
         num_faces_by_geom[GetFaceGeometry(i)]++;
      }

      os << '\n'
         << "Number of vertices : " << GetNV() << '\n'
         << "Number of edges    : " << GetNEdges() << '\n'
         << "Number of faces    : " << GetNFaces() << "  --  ";
      PrintElementsByGeometry(Dim-1, num_faces_by_geom, os);
      os << '\n'
         << "Number of elements : " << GetNE() << "  --  ";
      PrintElementsByGeometry(Dim, num_elems_by_geom, os);
      os << '\n'
         << "Number of bdr elem : " << GetNBE() << "  --  ";
      PrintElementsByGeometry(Dim-1, num_bdr_elems_by_geom, os);
      os << '\n'
         << "Euler Number       : " << EulerNumber() << '\n'
         << "h_min              : " << h_min << '\n'
         << "h_max              : " << h_max << '\n'
         << "kappa_min          : " << kappa_min << '\n'
         << "kappa_max          : " << kappa_max << '\n';
   }
   os << '\n' << std::flush;
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
      case Element::PYRAMID :        return &PyramidFE;
      default:
         MFEM_ABORT("Unknown element type \"" << ElemType << "\"");
         break;
   }
   MFEM_ABORT("Unknown element type");
   return NULL;
}


void Mesh::GetElementTransformation(int i,
                                    IsoparametricTransformation *ElTr) const
{
   ElTr->Attribute = GetAttribute(i);
   ElTr->ElementNo = i;
   ElTr->ElementType = ElementTransformation::ELEMENT;
   ElTr->mesh = this;
   ElTr->Reset();
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
      Nodes->HostRead();
      const GridFunction &nodes = *Nodes;
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

ElementTransformation *Mesh::GetTypicalElementTransformation()
{
   if (GetNE() > 0) { return GetElementTransformation(0); }

   Transformation.Attribute = -1;
   Transformation.ElementNo = -1;
   Transformation.ElementType = ElementTransformation::ELEMENT;
   Transformation.mesh = this;
   Transformation.Reset();
   const Geometry::Type geom = GetTypicalElementGeometry();
   if (Nodes == NULL)
   {
      Element::Type el_type = Element::TypeFromGeometry(geom);
      Transformation.SetFE(GetTransformationFEforElementType(el_type));
   }
   else
   {
      Transformation.SetFE(GetNodalFESpace()->GetTypicalFE());
   }
   Transformation.SetIdentityTransformation(geom);
   return &Transformation;
}

ElementTransformation *Mesh::GetElementTransformation(int i)
{
   GetElementTransformation(i, &Transformation);
   return &Transformation;
}

void Mesh::GetElementTransformation(int i, const Vector &nodes,
                                    IsoparametricTransformation *ElTr) const
{
   ElTr->Attribute = GetAttribute(i);
   ElTr->ElementNo = i;
   ElTr->ElementType = ElementTransformation::ELEMENT;
   ElTr->mesh = this;
   DenseMatrix &pm = ElTr->GetPointMat();
   ElTr->Reset();
   nodes.HostRead();
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
}

void Mesh::GetBdrElementTransformation(int i,
                                       IsoparametricTransformation* ElTr) const
{
   ElTr->Attribute = GetBdrAttribute(i);
   ElTr->ElementNo = i; // boundary element number
   ElTr->ElementType = ElementTransformation::BDR_ELEMENT;
   ElTr->mesh = this;
   DenseMatrix &pm = ElTr->GetPointMat();
   ElTr->Reset();
   if (Nodes == NULL)
   {
      GetBdrPointMatrix(i, pm);
      ElTr->SetFE(GetTransformationFEforElementType(GetBdrElementType(i)));
   }
   else
   {
      const FiniteElement *bdr_el = Nodes->FESpace()->GetBE(i);
      Nodes->HostRead();
      const GridFunction &nodes = *Nodes;
      if (bdr_el)
      {
         Array<int> vdofs;
         Nodes->FESpace()->GetBdrElementVDofs(i, vdofs);
         int n = vdofs.Size()/spaceDim;
         pm.SetSize(spaceDim, n);
         for (int k = 0; k < spaceDim; k++)
         {
            for (int j = 0; j < n; j++)
            {
               int idx = vdofs[n*k+j];
               pm(k,j) = nodes((idx<0)? -1-idx:idx);
            }
         }
         ElTr->SetFE(bdr_el);
      }
      else // L2 Nodes (e.g., periodic mesh)
      {
         int elem_id, face_info;
         GetBdrElementAdjacentElement(i, elem_id, face_info);
         Geometry::Type face_geom = GetBdrElementGeometry(i);
         face_info = EncodeFaceInfo(
                        DecodeFaceInfoLocalIndex(face_info),
                        Geometry::GetInverseOrientation(
                           face_geom, DecodeFaceInfoOrientation(face_info))
                     );

         IntegrationPointTransformation Loc1;
         GetLocalFaceTransformation(GetBdrElementType(i),
                                    GetElementType(elem_id),
                                    Loc1.Transf, face_info);
         const FiniteElement *face_el =
            Nodes->FESpace()->GetTraceElement(elem_id, face_geom);
         MFEM_VERIFY(dynamic_cast<const NodalFiniteElement*>(face_el),
                     "Mesh requires nodal Finite Element.");

         IntegrationRule eir(face_el->GetDof());
         Loc1.Transf.ElementNo = elem_id;
         Loc1.Transf.mesh = this;
         Loc1.Transf.ElementType = ElementTransformation::ELEMENT;
         Loc1.Transform(face_el->GetNodes(), eir);
         Nodes->GetVectorValues(Loc1.Transf, eir, pm);

         ElTr->SetFE(face_el);
      }
   }
}

ElementTransformation *Mesh::GetBdrElementTransformation(int i)
{
   GetBdrElementTransformation(i, &BdrTransformation);
   return &BdrTransformation;
}

void Mesh::GetFaceTransformation(int FaceNo,
                                 IsoparametricTransformation *FTr) const
{
   FTr->Attribute = (Dim == 1) ? 1 : faces[FaceNo]->GetAttribute();
   FTr->ElementNo = FaceNo;
   FTr->ElementType = ElementTransformation::FACE;
   FTr->mesh = this;
   DenseMatrix &pm = FTr->GetPointMat();
   FTr->Reset();
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
      Nodes->HostRead();
      const GridFunction &nodes = *Nodes;
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
               pm(i, j) = nodes(vdofs[n*i+j]);
            }
         }
         FTr->SetFE(face_el);
      }
      else // L2 Nodes (e.g., periodic mesh), go through the volume of Elem1
      {
         const FaceInfo &face_info = faces_info[FaceNo];
         Geometry::Type face_geom = GetFaceGeometry(FaceNo);
         Element::Type  face_type = GetFaceElementType(FaceNo);

         IntegrationPointTransformation Loc1;
         GetLocalFaceTransformation(face_type,
                                    GetElementType(face_info.Elem1No),
                                    Loc1.Transf, face_info.Elem1Inf);

         face_el = Nodes->FESpace()->GetTraceElement(face_info.Elem1No,
                                                     face_geom);
         MFEM_VERIFY(dynamic_cast<const NodalFiniteElement*>(face_el),
                     "Mesh requires nodal Finite Element.");

         IntegrationRule eir(face_el->GetDof());
         Loc1.Transf.ElementNo = face_info.Elem1No;
         Loc1.Transf.ElementType = ElementTransformation::ELEMENT;
         Loc1.Transf.mesh = this;
         Loc1.Transform(face_el->GetNodes(), eir);
         Nodes->GetVectorValues(Loc1.Transf, eir, pm);

         FTr->SetFE(face_el);
      }
   }
}

ElementTransformation *Mesh::GetFaceTransformation(int FaceNo)
{
   GetFaceTransformation(FaceNo, &FaceTransformation);
   return &FaceTransformation;
}

void Mesh::GetEdgeTransformation(int EdgeNo,
                                 IsoparametricTransformation *EdTr) const
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
   EdTr->ElementType = ElementTransformation::EDGE;
   EdTr->mesh = this;
   DenseMatrix &pm = EdTr->GetPointMat();
   EdTr->Reset();
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
      Nodes->HostRead();
      const GridFunction &nodes = *Nodes;
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
               pm(i, j) = nodes(vdofs[n*i+j]);
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
   IsoparametricTransformation &Transf, int i) const
{
   const IntegrationRule *SegVert;
   DenseMatrix &locpm = Transf.GetPointMat();
   Transf.Reset();

   Transf.SetFE(&PointFE);
   SegVert = Geometries.GetVertices(Geometry::SEGMENT);
   locpm.SetSize(1, 1);
   locpm(0, 0) = SegVert->IntPoint(i/64).x;
   //  (i/64) is the local face no. in the segment
   //  (i%64) is the orientation of the point (not used)
}

void Mesh::GetLocalSegToTriTransformation(
   IsoparametricTransformation &Transf, int i) const
{
   const int *tv, *so;
   const IntegrationRule *TriVert;
   DenseMatrix &locpm = Transf.GetPointMat();
   Transf.Reset();

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
   IsoparametricTransformation &Transf, int i) const
{
   const int *qv, *so;
   const IntegrationRule *QuadVert;
   DenseMatrix &locpm = Transf.GetPointMat();
   Transf.Reset();

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
   IsoparametricTransformation &Transf, int i) const
{
   DenseMatrix &locpm = Transf.GetPointMat();
   Transf.Reset();

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

void Mesh::GetLocalTriToWdgTransformation(
   IsoparametricTransformation &Transf, int i) const
{
   DenseMatrix &locpm = Transf.GetPointMat();
   Transf.Reset();

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
}

void Mesh::GetLocalTriToPyrTransformation(
   IsoparametricTransformation &Transf, int i) const
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&TriangleFE);
   //  (i/64) is the local face no. in the pyr
   MFEM_VERIFY(i >= 64, "Local face index " << i/64
               << " is not a triangular face of a pyramid.");
   const int *pv = pyr_t::FaceVert[i/64];
   //  (i%64) is the orientation of the pyramid face
   //         w.r.t. the face element
   const int *to = tri_t::Orient[i%64];
   const IntegrationRule *PyrVert =
      Geometries.GetVertices(Geometry::PYRAMID);
   locpm.SetSize(3, 3);
   for (int j = 0; j < 3; j++)
   {
      const IntegrationPoint &vert = PyrVert->IntPoint(pv[to[j]]);
      locpm(0, j) = vert.x;
      locpm(1, j) = vert.y;
      locpm(2, j) = vert.z;
   }
}

void Mesh::GetLocalQuadToHexTransformation(
   IsoparametricTransformation &Transf, int i) const
{
   DenseMatrix &locpm = Transf.GetPointMat();
   Transf.Reset();

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

void Mesh::GetLocalQuadToWdgTransformation(
   IsoparametricTransformation &Transf, int i) const
{
   DenseMatrix &locpm = Transf.GetPointMat();
   Transf.Reset();

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
}

void Mesh::GetLocalQuadToPyrTransformation(
   IsoparametricTransformation &Transf, int i) const
{
   DenseMatrix &locpm = Transf.GetPointMat();

   Transf.SetFE(&QuadrilateralFE);
   //  (i/64) is the local face no. in the pyr
   MFEM_VERIFY(i < 64, "Local face index " << i/64
               << " is not a quadrilateral face of a pyramid.");
   const int *pv = pyr_t::FaceVert[i/64];
   //  (i%64) is the orientation of the quad
   const int *qo = quad_t::Orient[i%64];
   const IntegrationRule *PyrVert = Geometries.GetVertices(Geometry::PYRAMID);
   locpm.SetSize(3, 4);
   for (int j = 0; j < 4; j++)
   {
      const IntegrationPoint &vert = PyrVert->IntPoint(pv[qo[j]]);
      locpm(0, j) = vert.x;
      locpm(1, j) = vert.y;
      locpm(2, j) = vert.z;
   }
}

const GeometricFactors* Mesh::GetGeometricFactors(const IntegrationRule& ir,
                                                  const int flags,
                                                  MemoryType d_mt)
{
   for (int i = 0; i < geom_factors.Size(); i++)
   {
      GeometricFactors *gf = geom_factors[i];
      if (gf->IntRule == &ir && (gf->computed_factors & flags) == flags)
      {
         return gf;
      }
   }

   this->EnsureNodes();

   GeometricFactors *gf = new GeometricFactors(this, ir, flags, d_mt);
   geom_factors.Append(gf);
   return gf;
}

const FaceGeometricFactors* Mesh::GetFaceGeometricFactors(
   const IntegrationRule& ir,
   const int flags, FaceType type, MemoryType d_mt)
{
   for (int i = 0; i < face_geom_factors.Size(); i++)
   {
      FaceGeometricFactors *gf = face_geom_factors[i];
      if (gf->IntRule == &ir && (gf->computed_factors & flags) == flags &&
          gf->type==type)
      {
         return gf;
      }
   }

   this->EnsureNodes();

   FaceGeometricFactors *gf = new FaceGeometricFactors(this, ir, flags, type,
                                                       d_mt);
   face_geom_factors.Append(gf);
   return gf;
}

const Array<int>& Mesh::GetBdrFaceAttributes() const
{
   if (bdr_face_attrs_cache.Size() == 0)
   {
      std::unordered_map<int, int> f_to_be;
      for (int i = 0; i < GetNBE(); ++i)
      {
         const int f = GetBdrElementFaceIndex(i);
         f_to_be[f] = i;
      }
      const int nf_bdr = GetNFbyType(FaceType::Boundary);
      // MFEM_VERIFY(size_t(nf_bdr) == f_to_be.size(), "Incompatible sizes");
      bdr_face_attrs_cache.SetSize(nf_bdr);
      int f_ind = 0;
      const int nf = GetNumFaces();
      for (int f = 0; f < nf; ++f)
      {
         if (!GetFaceInformation(f).IsOfFaceType(FaceType::Boundary))
         {
            continue;
         }
         int attribute = -1; // default value
         auto iter = f_to_be.find(f);
         if (iter != f_to_be.end())
         {
            const int be = iter->second;
            attribute = GetBdrAttribute(be);
         }
         else
         {
            // If a boundary face does not correspond to the a boundary element,
            // we assign it the default attribute of -1.
         }
         bdr_face_attrs_cache[f_ind] = attribute;
         ++f_ind;
      }
   }
   return bdr_face_attrs_cache;
}

const Array<int>& Mesh::GetElementAttributes() const
{
   if (elem_attrs_cache.Size() == 0)
   {
      // re-compute cache
      elem_attrs_cache.SetSize(GetNE());
      elem_attrs_cache.HostWrite();
      for (int i = 0; i < GetNE(); ++i)
      {
         elem_attrs_cache[i] = GetAttribute(i);
         MFEM_ASSERT(elem_attrs_cache[i] > 0,
                     "Negative attribute on element " << i);
      }
   }
   return elem_attrs_cache;
}

void Mesh::ComputeFaceInfo(FaceType ftype) const
{
   auto &fidcs = face_indices[static_cast<int>(ftype)];
   auto &ifidcs = inv_face_indices[static_cast<int>(ftype)];
   fidcs.SetSize(GetNFbyType(ftype));
   fidcs.HostWrite();
   ifidcs.reserve(fidcs.Size());
   int f_idx = 0;
   for (int i = 0; i < GetNumFacesWithGhost(); ++i)
   {
      const FaceInformation face = GetFaceInformation(i);
      if (face.IsNonconformingCoarse() || !face.IsOfFaceType(ftype))
      {
         continue;
      }
      fidcs[f_idx] = i;
      ifidcs[i] = f_idx;
      ++f_idx;
   }
}

const Array<int> &Mesh::GetFaceIndices(FaceType ftype) const
{
   if (face_indices[static_cast<int>(ftype)].Size() == 0)
   {
      ComputeFaceInfo(ftype);
   }
   return face_indices[static_cast<int>(ftype)];
}

const std::unordered_map<int, int> &
Mesh::GetInvFaceIndices(FaceType ftype) const
{
   if (inv_face_indices[static_cast<int>(ftype)].empty())
   {
      ComputeFaceInfo(ftype);
   }
   return inv_face_indices[static_cast<int>(ftype)];
}

void Mesh::DeleteGeometricFactors()
{
   for (int i = 0; i < geom_factors.Size(); i++)
   {
      delete geom_factors[i];
   }
   geom_factors.SetSize(0);
   for (int i = 0; i < face_geom_factors.Size(); i++)
   {
      delete face_geom_factors[i];
   }
   face_geom_factors.SetSize(0);

   ++nodes_sequence;
}

void Mesh::GetLocalFaceTransformation(int face_type, int elem_type,
                                      IsoparametricTransformation &Transf,
                                      int info) const
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
         else if (elem_type == Element::WEDGE)
         {
            GetLocalTriToWdgTransformation(Transf, info);
         }
         else if (elem_type == Element::PYRAMID)
         {
            GetLocalTriToPyrTransformation(Transf, info);
         }
         else
         {
            MFEM_ABORT("Mesh::GetLocalFaceTransformation not defined for "
                       "face type " << face_type
                       << " and element type " << elem_type << "\n");
         }
         break;

      case Element::QUADRILATERAL:
         if (elem_type == Element::HEXAHEDRON)
         {
            GetLocalQuadToHexTransformation(Transf, info);
         }
         else if (elem_type == Element::WEDGE)
         {
            GetLocalQuadToWdgTransformation(Transf, info);
         }
         else if (elem_type == Element::PYRAMID)
         {
            GetLocalQuadToPyrTransformation(Transf, info);
         }
         else
         {
            MFEM_ABORT("Mesh::GetLocalFaceTransformation not defined for "
                       "face type " << face_type
                       << " and element type " << elem_type << "\n");
         }
         break;
   }
}

FaceElementTransformations *Mesh::GetFaceElementTransformations(int FaceNo,
                                                                int mask)
{
   GetFaceElementTransformations(FaceNo, FaceElemTr, Transformation,
                                 Transformation2, mask);
   return &FaceElemTr;
}

void Mesh::GetFaceElementTransformations(int FaceNo,
                                         FaceElementTransformations &FElTr,
                                         IsoparametricTransformation &ElTr1,
                                         IsoparametricTransformation &ElTr2,
                                         int mask) const
{
   const FaceInfo &face_info = faces_info[FaceNo];

   int cmask = 0;
   FElTr.SetConfigurationMask(cmask);
   FElTr.Elem1 = NULL;
   FElTr.Elem2 = NULL;

   // setup the transformation for the first element
   FElTr.Elem1No = face_info.Elem1No;
   if (mask & FaceElementTransformations::HAVE_ELEM1)
   {
      GetElementTransformation(FElTr.Elem1No, &ElTr1);
      FElTr.Elem1 = &ElTr1;
      cmask |= 1;
   }

   //  setup the transformation for the second element
   //     return NULL in the Elem2 field if there's no second element, i.e.
   //     the face is on the "boundary"
   FElTr.Elem2No = face_info.Elem2No;
   if ((mask & FaceElementTransformations::HAVE_ELEM2) &&
       FElTr.Elem2No >= 0)
   {
#ifdef MFEM_DEBUG
      if (NURBSext && (mask & FaceElementTransformations::HAVE_ELEM1))
      { MFEM_ABORT("NURBS mesh not supported!"); }
#endif
      GetElementTransformation(FElTr.Elem2No, &ElTr2);
      FElTr.Elem2 = &ElTr2;
      cmask |= 2;
   }

   // setup the face transformation
   if (mask & FaceElementTransformations::HAVE_FACE)
   {
      GetFaceTransformation(FaceNo, &FElTr);
      cmask |= 16;
   }
   else
   {
      FElTr.SetGeometryType(GetFaceGeometry(FaceNo));
   }

   // setup Loc1 & Loc2
   int face_type = GetFaceElementType(FaceNo);
   if (mask & FaceElementTransformations::HAVE_LOC1)
   {
      int elem_type = GetElementType(face_info.Elem1No);
      GetLocalFaceTransformation(face_type, elem_type,
                                 FElTr.Loc1.Transf, face_info.Elem1Inf);
      cmask |= 4;
   }
   if ((mask & FaceElementTransformations::HAVE_LOC2) &&
       FElTr.Elem2No >= 0)
   {
      int elem_type = GetElementType(face_info.Elem2No);
      GetLocalFaceTransformation(face_type, elem_type,
                                 FElTr.Loc2.Transf, face_info.Elem2Inf);

      // NC meshes: prepend slave edge/face transformation to Loc2
      if (Nonconforming() && IsSlaveFace(face_info))
      {
         ApplyLocalSlaveTransformation(FElTr, face_info, false);
      }
      cmask |= 8;
   }

   FElTr.SetConfigurationMask(cmask);

   // This check can be useful for internal debugging, however it will fail on
   // periodic boundary faces, so we keep it disabled in general.
#if 0
#ifdef MFEM_DEBUG
   real_t dist = FElTr.CheckConsistency();
   if (dist >= 1e-12)
   {
      mfem::out << "\nInternal error: face id = " << FaceNo
                << ", dist = " << dist << '\n';
      FElTr.CheckConsistency(1); // print coordinates
      MFEM_ABORT("internal error");
   }
#endif
#endif
}

FaceElementTransformations *Mesh::GetInteriorFaceTransformations(int FaceNo)
{
   GetInteriorFaceTransformations(FaceNo, FaceElemTr, Transformation,
                                  Transformation2);
   return (FaceElemTr.geom == Geometry::INVALID) ? nullptr : &FaceElemTr;
}

void Mesh::GetInteriorFaceTransformations(int FaceNo,
                                          FaceElementTransformations &FElTr,
                                          IsoparametricTransformation &ElTr1,
                                          IsoparametricTransformation &ElTr2) const
{
   if (faces_info[FaceNo].Elem2No < 0)
   {
      FElTr.SetGeometryType(Geometry::INVALID);
      return;
   }
   GetFaceElementTransformations(FaceNo, FElTr, ElTr1, ElTr2);
}

FaceElementTransformations *Mesh::GetBdrFaceTransformations(int BdrElemNo)
{
   GetBdrFaceTransformations(BdrElemNo, FaceElemTr, Transformation,
                             Transformation2);
   return (FaceElemTr.geom == Geometry::INVALID) ? nullptr : &FaceElemTr;
}

void Mesh::GetBdrFaceTransformations(int BdrElemNo,
                                     FaceElementTransformations &FElTr,
                                     IsoparametricTransformation &ElTr1,
                                     IsoparametricTransformation &ElTr2) const
{
   // Check if the face is interior, shared, or nonconforming.
   int fn = GetBdrElementFaceIndex(BdrElemNo);
   if (FaceIsTrueInterior(fn) || faces_info[fn].NCFace >= 0)
   {
      FElTr.SetGeometryType(Geometry::INVALID);
      return;
   }
   GetFaceElementTransformations(fn, FElTr, ElTr1, ElTr2, 21);
   FElTr.Attribute = boundary[BdrElemNo]->GetAttribute();
   FElTr.ElementNo = BdrElemNo;
   FElTr.ElementType = ElementTransformation::BDR_FACE;
   FElTr.mesh = this;
}

bool Mesh::IsSlaveFace(const FaceInfo &fi) const
{
   return fi.NCFace >= 0 && nc_faces_info[fi.NCFace].Slave;
}

void Mesh::ApplyLocalSlaveTransformation(FaceElementTransformations &FT,
                                         const FaceInfo &fi, bool is_ghost) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix composition;
#else
   static DenseMatrix composition;
#endif
   MFEM_ASSERT(fi.NCFace >= 0, "");
   MFEM_ASSERT(nc_faces_info[fi.NCFace].Slave, "internal error");
   if (!is_ghost)
   {
      // side 1 -> child side, side 2 -> parent side
      IsoparametricTransformation &LT = FT.Loc2.Transf;
      LT.Transform(*nc_faces_info[fi.NCFace].PointMatrix, composition);
      // In 2D, we need to flip the point matrix since it is aligned with the
      // parent side.
      if (Dim == 2)
      {
         // swap points (columns) 0 and 1
         std::swap(composition(0,0), composition(0,1));
         std::swap(composition(1,0), composition(1,1));
      }
      LT.SetPointMat(composition);
   }
   else // is_ghost == true
   {
      // side 1 -> parent side, side 2 -> child side
      IsoparametricTransformation &LT = FT.Loc1.Transf;
      LT.Transform(*nc_faces_info[fi.NCFace].PointMatrix, composition);
      // In 2D, there is no need to flip the point matrix since it is already
      // aligned with the parent side, see also ParNCMesh::GetFaceNeighbors.
      // In 3D the point matrix was flipped during construction in
      // ParNCMesh::GetFaceNeighbors and due to that it is already aligned with
      // the parent side.
      LT.SetPointMat(composition);
   }
}

Mesh::FaceInformation Mesh::GetFaceInformation(int f) const
{
   FaceInformation face;
   int e1, e2;
   int inf1, inf2;
   int ncface;
   GetFaceElements(f, &e1, &e2);
   GetFaceInfos(f, &inf1, &inf2, &ncface);
   face.element[0].index = e1;
   face.element[0].location = ElementLocation::Local;
   face.element[0].orientation = inf1%64;
   face.element[0].local_face_id = inf1/64;
   face.element[1].local_face_id = inf2/64;
   face.ncface = ncface;
   face.point_matrix = nullptr;
   // The following figures out face.location, face.conformity,
   // face.element[1].index, and face.element[1].orientation.
   if (f < GetNumFaces()) // Non-ghost face
   {
      if (e2>=0)
      {
         if (ncface==-1)
         {
            face.tag = FaceInfoTag::LocalConforming;
            face.topology = FaceTopology::Conforming;
            face.element[1].location = ElementLocation::Local;
            face.element[0].conformity = ElementConformity::Coincident;
            face.element[1].conformity = ElementConformity::Coincident;
            face.element[1].index = e2;
            face.element[1].orientation = inf2%64;
         }
         else // ncface >= 0
         {
            face.tag = FaceInfoTag::LocalSlaveNonconforming;
            face.topology = FaceTopology::Nonconforming;
            face.element[1].location = ElementLocation::Local;
            face.element[0].conformity = ElementConformity::Coincident;
            face.element[1].conformity = ElementConformity::Superset;
            face.element[1].index = e2;
            MFEM_ASSERT(inf2%64==0, "unexpected slave face orientation.");
            face.element[1].orientation = inf2%64;
            face.point_matrix = nc_faces_info[ncface].PointMatrix;
         }
      }
      else // e2<0
      {
         if (ncface==-1)
         {
            if (inf2<0)
            {
               face.tag = FaceInfoTag::Boundary;
               face.topology = FaceTopology::Boundary;
               face.element[1].location = ElementLocation::NA;
               face.element[0].conformity = ElementConformity::Coincident;
               face.element[1].conformity = ElementConformity::NA;
               face.element[1].index = -1;
               face.element[1].orientation = -1;
            }
            else // inf2 >= 0
            {
               face.tag = FaceInfoTag::SharedConforming;
               face.topology = FaceTopology::Conforming;
               face.element[0].conformity = ElementConformity::Coincident;
               face.element[1].conformity = ElementConformity::Coincident;
               face.element[1].location = ElementLocation::FaceNbr;
               face.element[1].index = -1 - e2;
               face.element[1].orientation = inf2%64;
            }
         }
         else // ncface >= 0
         {
            if (inf2 < 0)
            {
               face.tag = FaceInfoTag::MasterNonconforming;
               face.topology = FaceTopology::Nonconforming;
               face.element[1].location = ElementLocation::NA;
               face.element[0].conformity = ElementConformity::Coincident;
               face.element[1].conformity = ElementConformity::Subset;
               face.element[1].index = -1;
               face.element[1].orientation = -1;
            }
            else
            {
               face.tag = FaceInfoTag::SharedSlaveNonconforming;
               face.topology = FaceTopology::Nonconforming;
               face.element[1].location = ElementLocation::FaceNbr;
               face.element[0].conformity = ElementConformity::Coincident;
               face.element[1].conformity = ElementConformity::Superset;
               face.element[1].index = -1 - e2;
               face.element[1].orientation = inf2%64;
            }
            face.point_matrix = nc_faces_info[ncface].PointMatrix;
         }
      }
   }
   else // Ghost face
   {
      if (e1==-1)
      {
         face.tag = FaceInfoTag::GhostMaster;
         face.topology = FaceTopology::NA;
         face.element[1].location = ElementLocation::NA;
         face.element[0].conformity = ElementConformity::NA;
         face.element[1].conformity = ElementConformity::NA;
         face.element[1].index = -1;
         face.element[1].orientation = -1;
      }
      else
      {
         face.tag = FaceInfoTag::GhostSlave;
         face.topology = FaceTopology::Nonconforming;
         face.element[1].location = ElementLocation::FaceNbr;
         face.element[0].conformity = ElementConformity::Superset;
         face.element[1].conformity = ElementConformity::Coincident;
         face.element[1].index = -1 - e2;
         face.element[1].orientation = inf2%64;
         face.point_matrix = nc_faces_info[ncface].PointMatrix;
      }
   }
   return face;
}

Mesh::FaceInformation::operator Mesh::FaceInfo() const
{
   FaceInfo res {-1, -1, -1, -1, -1};
   switch (tag)
   {
      case FaceInfoTag::LocalConforming:
         res.Elem1No = element[0].index;
         res.Elem2No = element[1].index;
         res.Elem1Inf = element[0].orientation + element[0].local_face_id*64;
         res.Elem2Inf = element[1].orientation + element[1].local_face_id*64;
         res.NCFace = ncface;
         break;
      case FaceInfoTag::LocalSlaveNonconforming:
         res.Elem1No = element[0].index;
         res.Elem2No = element[1].index;
         res.Elem1Inf = element[0].orientation + element[0].local_face_id*64;
         res.Elem2Inf = element[1].orientation + element[1].local_face_id*64;
         res.NCFace = ncface;
         break;
      case FaceInfoTag::Boundary:
         res.Elem1No = element[0].index;
         res.Elem1Inf = element[0].orientation + element[0].local_face_id*64;
         break;
      case FaceInfoTag::SharedConforming:
         res.Elem1No = element[0].index;
         res.Elem2No = -1 - element[1].index;
         res.Elem1Inf = element[0].orientation + element[0].local_face_id*64;
         res.Elem2Inf = element[1].orientation + element[1].local_face_id*64;
         break;
      case FaceInfoTag::MasterNonconforming:
         res.Elem1No = element[0].index;
         res.Elem1Inf = element[0].orientation + element[0].local_face_id*64;
         break;
      case FaceInfoTag::SharedSlaveNonconforming:
         res.Elem1No = element[0].index;
         res.Elem2No = -1 - element[1].index;
         res.Elem1Inf = element[0].orientation + element[0].local_face_id*64;
         res.Elem2Inf = element[1].orientation + element[1].local_face_id*64;
         break;
      case FaceInfoTag::GhostMaster:
         break;
      case FaceInfoTag::GhostSlave:
         res.Elem1No = element[0].index;
         res.Elem2No = -1 - element[1].index;
         res.Elem1Inf = element[0].orientation + element[0].local_face_id*64;
         res.Elem2Inf = element[1].orientation + element[1].local_face_id*64;
         break;
   }
   return res;
}

std::ostream &operator<<(std::ostream &os, const Mesh::FaceInformation& info)
{
   os << "face topology=";
   switch (info.topology)
   {
      case Mesh::FaceTopology::Boundary:
         os << "Boundary";
         break;
      case Mesh::FaceTopology::Conforming:
         os << "Conforming";
         break;
      case Mesh::FaceTopology::Nonconforming:
         os << "Non-conforming";
         break;
      case Mesh::FaceTopology::NA:
         os << "NA";
         break;
   }
   os << '\n';
   os << "element[0].location=";
   switch (info.element[0].location)
   {
      case Mesh::ElementLocation::Local:
         os << "Local";
         break;
      case Mesh::ElementLocation::FaceNbr:
         os << "FaceNbr";
         break;
      case Mesh::ElementLocation::NA:
         os << "NA";
         break;
   }
   os << '\n';
   os << "element[1].location=";
   switch (info.element[1].location)
   {
      case Mesh::ElementLocation::Local:
         os << "Local";
         break;
      case Mesh::ElementLocation::FaceNbr:
         os << "FaceNbr";
         break;
      case Mesh::ElementLocation::NA:
         os << "NA";
         break;
   }
   os << '\n';
   os << "element[0].conformity=";
   switch (info.element[0].conformity)
   {
      case Mesh::ElementConformity::Coincident:
         os << "Coincident";
         break;
      case Mesh::ElementConformity::Superset:
         os << "Superset";
         break;
      case Mesh::ElementConformity::Subset:
         os << "Subset";
         break;
      case Mesh::ElementConformity::NA:
         os << "NA";
         break;
   }
   os << '\n';
   os << "element[1].conformity=";
   switch (info.element[1].conformity)
   {
      case Mesh::ElementConformity::Coincident:
         os << "Coincident";
         break;
      case Mesh::ElementConformity::Superset:
         os << "Superset";
         break;
      case Mesh::ElementConformity::Subset:
         os << "Subset";
         break;
      case Mesh::ElementConformity::NA:
         os << "NA";
         break;
   }
   os << '\n';
   os << "element[0].index=" << info.element[0].index << '\n'
      << "element[1].index=" << info.element[1].index << '\n'
      << "element[0].local_face_id=" << info.element[0].local_face_id << '\n'
      << "element[1].local_face_id=" << info.element[1].local_face_id << '\n'
      << "element[0].orientation=" << info.element[0].orientation << '\n'
      << "element[1].orientation=" << info.element[1].orientation << '\n'
      << "ncface=" << info.ncface << std::endl;
   return os;
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

void Mesh::GetFaceInfos(int Face, int *Inf1, int *Inf2, int *NCFace) const
{
   *Inf1   = faces_info[Face].Elem1Inf;
   *Inf2   = faces_info[Face].Elem2Inf;
   *NCFace = faces_info[Face].NCFace;
}

Geometry::Type Mesh::GetFaceGeometry(int Face) const
{
   switch (Dim)
   {
      case 1: return Geometry::POINT;
      case 2: return Geometry::SEGMENT;
      case 3:
         if (Face < NumOfFaces) // local (non-ghost) face
         {
            return faces[Face]->GetGeometryType();
         }
         // ghost face
         const int nc_face_id = faces_info[Face].NCFace;

         MFEM_ASSERT(nc_face_id >= 0, "parent ghost faces are not supported");
         return faces[nc_faces_info[nc_face_id].MasterFace]->GetGeometryType();
   }
   return Geometry::INVALID;
}

Geometry::Type Mesh::GetTypicalFaceGeometry() const
{
   Geometry::Type elem_geom = GetTypicalElementGeometry();
   switch (elem_geom)
   {
      case Geometry::SEGMENT: return Geometry::POINT;
      case Geometry::TRIANGLE: return Geometry::SEGMENT;
      case Geometry::SQUARE: return Geometry::SEGMENT;
      case Geometry::TETRAHEDRON: return Geometry::TRIANGLE;
      case Geometry::CUBE: return Geometry::SQUARE;
      case Geometry::PRISM: return Geometry::TRIANGLE;
      case Geometry::PYRAMID: return Geometry::TRIANGLE;
      default: return Geometry::INVALID;
   }
}

Element::Type Mesh::GetFaceElementType(int Face) const
{
   return (Dim == 1) ? Element::POINT : faces[Face]->GetType();
}

Array<int> Mesh::GetFaceToBdrElMap() const
{
   Array<int> face_to_be(Dim == 2 ? NumOfEdges : NumOfFaces);
   face_to_be = -1;
   for (int i = 0; i < NumOfBdrElements; i++)
   {
      face_to_be[GetBdrElementFaceIndex(i)] = i;
   }
   return face_to_be;
}

Geometry::Type Mesh::GetTypicalElementGeometry() const
{
   if (GetNE() > 0) { return GetElementGeometry(0); }

   const int dim = Dimension();
   if (dim == 1)
   {
      return Geometry::SEGMENT;
   }
   Geometry::Type geom = Geometry::INVALID;
   if (dim == 2)
   {
      geom = ((meshgen & 1) ? Geometry::TRIANGLE :
              ((meshgen & 2) ? Geometry::SQUARE : Geometry::INVALID));
   }
   else if (dim == 3)
   {
      geom = ((meshgen & 1) ? Geometry::TETRAHEDRON :
              ((meshgen & 2) ? Geometry::CUBE :
               ((meshgen & 4) ? Geometry::PRISM :
                ((meshgen & 8) ? Geometry::PYRAMID : Geometry::INVALID))));
   }
   MFEM_VERIFY(geom != Geometry::INVALID,
               "Could not determine a typical element Geometry!");
   return geom;
}


void Mesh::GetExteriorFaceMarker(Array<int> & face_marker) const
{
   const int num_faces = GetNumFaces();

   face_marker.SetSize(num_faces);

   for (int f = 0; f < num_faces; f++)
   {
      if (FaceIsTrueInterior(f))
      {
         face_marker[f] = 0;
      }
      else
      {
         face_marker[f] = 1;
      }
   }
}

void Mesh::UnmarkInternalBoundaries(Array<int> &bdr_marker, bool excl) const
{
   const int max_bdr_attr = bdr_attributes.Max();

   MFEM_VERIFY(bdr_marker.Size() >= max_bdr_attr,
               "bdr_marker must be at least bdr_attriburtes.Max() in length");

   Array<bool> interior_bdr(max_bdr_attr); interior_bdr = false;
   Array<bool> exterior_bdr(max_bdr_attr); exterior_bdr = false;

   // Identify attributes which contain interior faces and those which
   // contain exterior faces.
   for (int be = 0; be < boundary.Size(); be++)
   {
      const int bea = boundary[be]->GetAttribute();

      if (bdr_marker[bea-1] != 0)
      {
         const int f = be_to_face[be];

         if (FaceIsTrueInterior(f))
         {
            interior_bdr[bea-1] = true;
         }
         else
         {
            exterior_bdr[bea-1] = true;
         }
      }
   }

   // Unmark attributes which are currently marked, contain interior faces,
   // and satisfy the appropriate exclusivity requirement.
   for (int b = 0; b < max_bdr_attr; b++)
   {
      if (bdr_marker[b] != 0 && interior_bdr[b])
      {
         if (!excl || !exterior_bdr[b])
         {
            bdr_marker[b] = 0;
         }
      }
   }
}

void Mesh::UnmarkNamedBoundaries(const std::string &set_name,
                                 Array<int> &bdr_marker) const
{
   const int max_bdr_attr = bdr_attributes.Max();

   MFEM_VERIFY(bdr_attribute_sets.AttributeSetExists(set_name),
               "Named set is not defined in this mesh!");
   MFEM_VERIFY(bdr_marker.Size() >= bdr_attributes.Max(),
               "bdr_marker must be at least bdr_attriburtes.Max() in length");

   Array<int> set_marker = bdr_attribute_sets.GetAttributeSetMarker(set_name);

   for (int b = 0; b < max_bdr_attr; b++)
   {
      if (set_marker[b])
      {
         bdr_marker[b] = 0;
      }
   }
}

void Mesh::MarkExternalBoundaries(Array<int> &bdr_marker, bool excl) const
{
   const int max_bdr_attr = bdr_attributes.Max();

   MFEM_VERIFY(bdr_marker.Size() >= max_bdr_attr,
               "bdr_marker must be at least bdr_attriburtes.Max() in length");

   Array<bool> interior_bdr(max_bdr_attr); interior_bdr = false;
   Array<bool> exterior_bdr(max_bdr_attr); exterior_bdr = false;

   // Mark boundary attributes containing exterior faces while keeping track of
   // those which also contain interior faces.
   for (int be = 0; be < boundary.Size(); be++)
   {
      const int bea = boundary[be]->GetAttribute();

      const int f = be_to_face[be];

      if (FaceIsTrueInterior(f))
      {
         interior_bdr[bea-1] = true;
      }
      else
      {
         exterior_bdr[bea-1] = true;
      }
   }

   // Mark attributes which were found to contain exterior faces and satisfy
   // the appropriate exclusivity requirement.
   for (int b = 0; b < max_bdr_attr; b++)
   {
      if (bdr_marker[b] == 0 && exterior_bdr[b])
      {
         if (!excl || !interior_bdr[b])
         {
            bdr_marker[b] = 1;
         }
      }
   }
}

void Mesh::MarkNamedBoundaries(const std::string &set_name,
                               Array<int> &bdr_marker) const
{
   const int max_bdr_attr = bdr_attributes.Max();

   MFEM_VERIFY(bdr_attribute_sets.AttributeSetExists(set_name),
               "Named set is not defined in this mesh!");
   MFEM_VERIFY(bdr_marker.Size() >= max_bdr_attr,
               "bdr_marker must be at least bdr_attriburtes.Max() in length");

   Array<int> set_marker = bdr_attribute_sets.GetAttributeSetMarker(set_name);

   for (int b = 0; b < max_bdr_attr; b++)
   {
      if (set_marker[b])
      {
         bdr_marker[b] = 1;
      }
   }
}

void Mesh::Init()
{
   // in order of declaration:
   Dim = spaceDim = 0;
   NumOfVertices = -1;
   NumOfElements = NumOfBdrElements = 0;
   NumOfEdges = NumOfFaces = 0;
   nbInteriorFaces = -1;
   nbBoundaryFaces = -1;
   meshgen = mesh_geoms = 0;
   sequence = 0;
   nodes_sequence = 0;
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
   face_to_elem = NULL;
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
   DeleteGeometricFactors();

   if (Dim == 3)
   {
      delete bel_to_edge;
   }

   delete face_edge;
   delete edge_vertex;

   delete face_to_elem;
   face_to_elem = NULL;
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
   be_to_face.DeleteAll();

   // TODO:
   // IsoparametricTransformations
   // Transformation, Transformation2, BdrTransformation, FaceTransformation,
   // EdgeTransformation;
   // FaceElementTransformations FaceElemTr;

   CoarseFineTr.Clear();

#ifdef MFEM_USE_MEMALLOC
   TetMemory.Clear();
#endif

   elem_attrs_cache.DeleteAll();
   bdr_face_attrs_cache.DeleteAll();
   attributes.DeleteAll();
   bdr_attributes.DeleteAll();

   face_indices[0].DeleteAll();
   face_indices[1].DeleteAll();
   // force de-allocation so after this mesh has the smallest memory footprint
   // possible
   inv_face_indices[0] = std::unordered_map<int, int>();
   inv_face_indices[1] = std::unordered_map<int, int>();
}

void Mesh::ResetLazyData()
{
   delete el_to_el;     el_to_el = NULL;
   delete face_edge;    face_edge = NULL;
   delete face_to_elem;    face_to_elem = NULL;
   delete edge_vertex;  edge_vertex = NULL;
   DeleteGeometricFactors();
   nbInteriorFaces = -1;
   nbBoundaryFaces = -1;
   // set size to 0 so re-computations can potentially avoid a new allocation
   bdr_face_attrs_cache.SetSize(0);
   elem_attrs_cache.SetSize(0);

   face_indices[0].SetSize(0);
   face_indices[1].SetSize(0);
   inv_face_indices[0].clear();
   inv_face_indices[1].clear();
}

void Mesh::SetAttributes(bool elem_attrs_changed, bool bdr_face_attrs_changed)
{
   if (bdr_face_attrs_changed)
   {
      bdr_face_attrs_cache.SetSize(0); // Invalidate the cache

      // Get sorted list of unique boundary element attributes
      std::set<int> attribs;
      for (int i = 0; i < GetNBE(); i++)
      {
         attribs.emplace(GetBdrAttribute(i));
      }

      bdr_attributes.SetSize(attribs.size());
      bdr_attributes.HostWrite();
      std::copy(attribs.begin(), attribs.end(), bdr_attributes.begin());
      if (bdr_attributes.Size() > 0 && bdr_attributes[0] <= 0)
      {
         MFEM_WARNING("Non-positive attributes on the boundary!");
      }
   }

   if (elem_attrs_changed)
   {
      // Re-compute the attributes cache
      elem_attrs_cache.SetSize(0);
      GetElementAttributes();
      // Get sorted list of unique element attributes
      std::set<int> attribs(elem_attrs_cache.begin(), elem_attrs_cache.end());
      attributes.SetSize(attribs.size());
      attributes.HostWrite();
      std::copy(attribs.begin(), attribs.end(), attributes.begin());

      if (attributes.Size() > 0 && attributes[0] <= 0)
      {
         MFEM_WARNING("Non-positive attributes in the domain!");
      }
   }
}

void Mesh::InitMesh(int Dim_, int spaceDim_, int NVert, int NElem, int NBdrElem)
{
   SetEmpty();

   Dim = Dim_;
   spaceDim = spaceDim_;

   NumOfVertices = 0;
   vertices.SetSize(NVert);  // just allocate space for vertices

   NumOfElements = 0;
   elements.SetSize(NElem);  // just allocate space for Element *

   NumOfBdrElements = 0;
   boundary.SetSize(NBdrElem);  // just allocate space for Element *
}

template<typename T>
static void CheckEnlarge(Array<T> &array, int size)
{
   if (size >= array.Size()) { array.SetSize(size + 1); }
}

int Mesh::AddVertex(real_t x, real_t y, real_t z)
{
   CheckEnlarge(vertices, NumOfVertices);
   real_t *v = vertices[NumOfVertices]();
   v[0] = x;
   v[1] = y;
   v[2] = z;
   return NumOfVertices++;
}

int Mesh::AddVertex(const real_t *coords)
{
   CheckEnlarge(vertices, NumOfVertices);
   vertices[NumOfVertices].SetCoords(spaceDim, coords);
   return NumOfVertices++;
}

int Mesh::AddVertex(const Vector &coords)
{
   MFEM_ASSERT(coords.Size() >= spaceDim,
               "invalid 'coords' size: " << coords.Size());
   return AddVertex(coords.GetData());
}

void Mesh::AddVertexParents(int i, int p1, int p2)
{
   tmp_vertex_parents.Append(Triple<int, int, int>(i, p1, p2));

   // if vertex coordinates are defined, make sure the hanging vertex has the
   // correct position
   if (i < vertices.Size())
   {
      real_t *vi = vertices[i](), *vp1 = vertices[p1](), *vp2 = vertices[p2]();
      for (int j = 0; j < 3; j++)
      {
         vi[j] = (vp1[j] + vp2[j]) * 0.5;
      }
   }
}

int Mesh::AddVertexAtMeanCenter(const int *vi, int nverts, int dim)
{
   Vector vii(dim);
   vii = 0.0;
   for (int i = 0; i < nverts; i++)
   {
      real_t *vp = vertices[vi[i]]();
      for (int j = 0; j < dim; j++)
      {
         vii(j) += vp[j];
      }
   }
   vii /= nverts;
   AddVertex(vii);
   return NumOfVertices;
}

int Mesh::AddSegment(int v1, int v2, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Segment(v1, v2, attr);
   return NumOfElements++;
}

int Mesh::AddSegment(const int *vi, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Segment(vi, attr);
   return NumOfElements++;
}

int Mesh::AddTriangle(int v1, int v2, int v3, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Triangle(v1, v2, v3, attr);
   return NumOfElements++;
}

int Mesh::AddTriangle(const int *vi, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Triangle(vi, attr);
   return NumOfElements++;
}

int Mesh::AddQuad(int v1, int v2, int v3, int v4, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Quadrilateral(v1, v2, v3, v4, attr);
   return NumOfElements++;
}

int Mesh::AddQuad(const int *vi, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Quadrilateral(vi, attr);
   return NumOfElements++;
}

int Mesh::AddTet(int v1, int v2, int v3, int v4, int attr)
{
   int vi[4] = {v1, v2, v3, v4};
   return AddTet(vi, attr);
}

int Mesh::AddTet(const int *vi, int attr)
{
   CheckEnlarge(elements, NumOfElements);
#ifdef MFEM_USE_MEMALLOC
   Tetrahedron *tet;
   tet = TetMemory.Alloc();
   tet->SetVertices(vi);
   tet->SetAttribute(attr);
   elements[NumOfElements] = tet;
#else
   elements[NumOfElements] = new Tetrahedron(vi, attr);
#endif
   return NumOfElements++;
}

int Mesh::AddWedge(int v1, int v2, int v3, int v4, int v5, int v6, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Wedge(v1, v2, v3, v4, v5, v6, attr);
   return NumOfElements++;
}

int Mesh::AddWedge(const int *vi, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Wedge(vi, attr);
   return NumOfElements++;
}

int Mesh::AddPyramid(int v1, int v2, int v3, int v4, int v5, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Pyramid(v1, v2, v3, v4, v5, attr);
   return NumOfElements++;
}

int Mesh::AddPyramid(const int *vi, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Pyramid(vi, attr);
   return NumOfElements++;
}

int Mesh::AddHex(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8,
                 int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] =
      new Hexahedron(v1, v2, v3, v4, v5, v6, v7, v8, attr);
   return NumOfElements++;
}

int Mesh::AddHex(const int *vi, int attr)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = new Hexahedron(vi, attr);
   return NumOfElements++;
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

void Mesh::AddHexAsPyramids(const int *vi, int attr)
{
   static const int hex_to_pyr[6][5] =
   {
      { 0, 1, 2, 3, 8 }, { 0, 4, 5, 1, 8 }, { 1, 5, 6, 2, 8 },
      { 2, 6, 7, 3, 8 }, { 3, 7, 4, 0, 8 }, { 7, 6, 5, 4, 8 }
   };
   int ti[5];

   for (int i = 0; i < 6; i++)
   {
      for (int j = 0; j < 5; j++)
      {
         ti[j] = vi[hex_to_pyr[i][j]];
      }
      AddPyramid(ti, attr);
   }
}

void Mesh::AddQuadAs4TrisWithPoints(int *vi, int attr)
{
   int num_faces = 4;
   static const int quad_to_tri[4][2] =
   {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}
   };

   int elem_center_index = AddVertexAtMeanCenter(vi, 4, 2) - 1;

   int ti[3];
   ti[2] = elem_center_index;
   for (int i = 0; i < num_faces; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         ti[j] = vi[quad_to_tri[i][j]];
      }
      AddTri(ti, attr);
   }
}

void Mesh::AddQuadAs5QuadsWithPoints(int *vi, int attr)
{
   int num_faces = 4;
   static const int quad_faces[4][2] =
   {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}
   };

   Vector px(4), py(4);
   for (int i = 0; i < 4; i++)
   {
      real_t *vp = vertices[vi[i]]();
      px(i) = vp[0];
      py(i) = vp[1];
   }

   int vnew_index[4];
   real_t vnew[2];
   real_t r = 0.25, s = 0.25;
   vnew[0] = px(0)*(1-r)*(1-s) + px(1)*(r)*(1-s) + px(2)*r*s + px(3)*(1-r)*s;
   vnew[1] = py(0)*(1-r)*(1-s) + py(1)*(r)*(1-s) + py(2)*r*s + py(3)*(1-r)*s;
   AddVertex(vnew);
   vnew_index[0] = NumOfVertices-1;

   r = 0.75, s = 0.25;
   vnew[0] = px(0)*(1-r)*(1-s) + px(1)*(r)*(1-s) + px(2)*r*s + px(3)*(1-r)*s;
   vnew[1] = py(0)*(1-r)*(1-s) + py(1)*(r)*(1-s) + py(2)*r*s + py(3)*(1-r)*s;
   AddVertex(vnew);
   vnew_index[1] = NumOfVertices-1;

   r = 0.75, s = 0.75;
   vnew[0] = px(0)*(1-r)*(1-s) + px(1)*(r)*(1-s) + px(2)*r*s + px(3)*(1-r)*s;
   vnew[1] = py(0)*(1-r)*(1-s) + py(1)*(r)*(1-s) + py(2)*r*s + py(3)*(1-r)*s;
   AddVertex(vnew);
   vnew_index[2] = NumOfVertices-1;

   r = 0.25, s = 0.75;
   vnew[0] = px(0)*(1-r)*(1-s) + px(1)*(r)*(1-s) + px(2)*r*s + px(3)*(1-r)*s;
   vnew[1] = py(0)*(1-r)*(1-s) + py(1)*(r)*(1-s) + py(2)*r*s + py(3)*(1-r)*s;
   AddVertex(vnew);
   vnew_index[3] = NumOfVertices-1;

   static const int quad_faces_new[4][2] =
   {
      { 1, 0}, { 2, 1}, { 3, 2}, { 0, 3}
   };

   int ti[4];
   for (int i = 0; i < num_faces; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         ti[j] = vi[quad_faces[i][j]];
         ti[j+2] = vnew_index[quad_faces_new[i][j]];
      }
      AddQuad(ti, attr);
   }
   AddQuad(vnew_index, attr);
}

void Mesh::AddHexAs24TetsWithPoints(int *vi,
                                    std::map<std::array<int, 4>, int> &hex_face_verts,
                                    int attr)
{
   auto get4arraysorted = [](Array<int> v)
   {
      v.Sort();
      return std::array<int, 4> {v[0], v[1], v[2], v[3]};
   };

   int num_faces = 6;
   static const int hex_to_tet[6][4] =
   {
      { 0, 1, 2, 3 }, { 1, 2, 6, 5 }, { 5, 4, 7, 6},
      { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 0,3, 7, 4}
   };

   int elem_center_index = AddVertexAtMeanCenter(vi, 8, 3) - 1;

   Array<int> flist(4);

   // local vertex indices for each of the 4 edges of the face
   static const int tet_face[4][2] =
   {
      {0, 1}, {1, 2}, {3, 2}, {3, 0}
   };

   for (int i = 0; i < num_faces; i++)
   {
      for (int j = 0; j < 4; j++)
      {
         flist[j] = vi[hex_to_tet[i][j]];
      }
      int face_center_index;

      auto t = get4arraysorted(flist);
      auto it = hex_face_verts.find(t);
      if (it == hex_face_verts.end())
      {
         face_center_index = AddVertexAtMeanCenter(flist.GetData(),
                                                   flist.Size(), 3) - 1;
         hex_face_verts.insert({t, face_center_index});
      }
      else
      {
         face_center_index = it->second;
      }
      int fti[4];
      fti[2] = face_center_index;
      fti[3] = elem_center_index;
      for (int j = 0; j < 4; j++)
      {
         for (int k = 0; k < 2; k++)
         {
            fti[k] = flist[tet_face[j][k]];
         }
         AddTet(fti, attr);
      }
   }
}

int Mesh::AddElement(Element *elem)
{
   CheckEnlarge(elements, NumOfElements);
   elements[NumOfElements] = elem;
   return NumOfElements++;
}

int Mesh::AddBdrElement(Element *elem)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = elem;
   return NumOfBdrElements++;
}

void Mesh::AddBdrElements(Array<Element *> &bdr_elems,
                          const Array<int> &new_be_to_face)
{
   boundary.Reserve(boundary.Size() + bdr_elems.Size());
   MFEM_ASSERT(bdr_elems.Size() == new_be_to_face.Size(), "wrong size");
   for (int i = 0; i < bdr_elems.Size(); i++)
   {
      AddBdrElement(bdr_elems[i]);
   }
   be_to_face.Append(new_be_to_face);
}

int Mesh::AddBdrSegment(int v1, int v2, int attr)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = new Segment(v1, v2, attr);
   return NumOfBdrElements++;
}

int Mesh::AddBdrSegment(const int *vi, int attr)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = new Segment(vi, attr);
   return NumOfBdrElements++;
}

int Mesh::AddBdrTriangle(int v1, int v2, int v3, int attr)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = new Triangle(v1, v2, v3, attr);
   return NumOfBdrElements++;
}

int Mesh::AddBdrTriangle(const int *vi, int attr)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = new Triangle(vi, attr);
   return NumOfBdrElements++;
}

int Mesh::AddBdrQuad(int v1, int v2, int v3, int v4, int attr)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = new Quadrilateral(v1, v2, v3, v4, attr);
   return NumOfBdrElements++;
}

int Mesh::AddBdrQuad(const int *vi, int attr)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = new Quadrilateral(vi, attr);
   return NumOfBdrElements++;
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

int Mesh::AddBdrPoint(int v, int attr)
{
   CheckEnlarge(boundary, NumOfBdrElements);
   boundary[NumOfBdrElements] = new Point(&v, attr);
   return NumOfBdrElements++;
}

void Mesh::GenerateBoundaryElements()
{
   for (auto &b : boundary)
   {
      FreeElement(b);
   }

   if (Dim == 3)
   {
      delete bel_to_edge;
      bel_to_edge = NULL;
   }

   // count the 'NumOfBdrElements'
   NumOfBdrElements = 0;
   for (const auto &fi : faces_info)
   {
      if (fi.Elem2No < 0) { ++NumOfBdrElements; }
   }

   // Add the boundary elements
   boundary.SetSize(NumOfBdrElements);
   be_to_face.SetSize(NumOfBdrElements);
   for (int i = 0, j = 0; i < faces_info.Size(); i++)
   {
      if (faces_info[i].Elem2No < 0)
      {
         boundary[j] = faces[i]->Duplicate(this);
         be_to_face[j++] = i;
      }
   }

   // Note: in 3D, 'bel_to_edge' is destroyed but it's not updated.
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
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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


class GeckoProgress : public Gecko::Progress
{
   real_t limit;
   mutable StopWatch sw;
public:
   GeckoProgress(real_t limit) : limit(limit) { sw.Start(); }
   bool quit() const override { return limit > 0 && sw.UserTime() > limit; }
};

class GeckoVerboseProgress : public GeckoProgress
{
   using Float = Gecko::Float;
   using Graph = Gecko::Graph;
   using uint = Gecko::uint;
public:
   GeckoVerboseProgress(real_t limit) : GeckoProgress(limit) {}

   void beginorder(const Graph* graph, Float cost) const override
   { mfem::out << "Begin Gecko ordering, cost = " << cost << std::endl; }
   void endorder(const Graph* graph, Float cost) const override
   { mfem::out << "End ordering, cost = " << cost << std::endl; }

   void beginiter(const Graph* graph,
                  uint iter, uint maxiter, uint window) const override
   {
      mfem::out << "Iteration " << iter << "/" << maxiter << ", window "
                << window << std::flush;
   }
   void enditer(const Graph* graph, Float mincost, Float cost) const override
   { mfem::out << ", cost = " << cost << endl; }
};


real_t Mesh::GetGeckoElementOrdering(Array<int> &ordering,
                                     int iterations, int window,
                                     int period, int seed, bool verbose,
                                     real_t time_limit)
{
   Gecko::Graph graph;
   Gecko::FunctionalGeometric functional; // edge product cost

   GeckoProgress progress(time_limit);
   GeckoVerboseProgress vprogress(time_limit);

   // insert elements as nodes in the graph
   for (int elemid = 0; elemid < GetNE(); ++elemid)
   {
      graph.insert_node();
   }

   // insert graph edges for element neighbors
   // NOTE: indices in Gecko are 1 based hence the +1 on insertion
   const Table &my_el_to_el = ElementToElementTable();
   for (int elemid = 0; elemid < GetNE(); ++elemid)
   {
      const int *neighid = my_el_to_el.GetRow(elemid);
      for (int i = 0; i < my_el_to_el.RowSize(elemid); ++i)
      {
         graph.insert_arc(elemid + 1,  neighid[i] + 1);
      }
   }

   // get the ordering from Gecko and copy it into the Array<int>
   graph.order(&functional, iterations, window, period, seed,
               verbose ? &vprogress : &progress);

   ordering.SetSize(GetNE());
   Gecko::Node::Index NE = GetNE();
   for (Gecko::Node::Index gnodeid = 1; gnodeid <= NE; ++gnodeid)
   {
      ordering[gnodeid - 1] = graph.rank(gnodeid);
   }

   return graph.cost();
}


struct HilbertCmp
{
   int coord;
   bool dir;
   const Array<real_t> &points;
   real_t mid;

   HilbertCmp(int coord, bool dir, const Array<real_t> &points, real_t mid)
      : coord(coord), dir(dir), points(points), mid(mid) {}

   bool operator()(int i) const
   {
      return (points[3*i + coord] < mid) != dir;
   }
};

static void HilbertSort2D(int coord1, // major coordinate to sort points by
                          bool dir1,  // sort coord1 ascending/descending?
                          bool dir2,  // sort coord2 ascending/descending?
                          const Array<real_t> &points, int *beg, int *end,
                          real_t xmin, real_t ymin, real_t xmax, real_t ymax)
{
   if (end - beg <= 1) { return; }

   real_t xmid = (xmin + xmax)*0.5;
   real_t ymid = (ymin + ymax)*0.5;

   int coord2 = (coord1 + 1) % 2; // the 'other' coordinate

   // sort (partition) points into four quadrants
   int *p0 = beg, *p4 = end;
   int *p2 = std::partition(p0, p4, HilbertCmp(coord1,  dir1, points, xmid));
   int *p1 = std::partition(p0, p2, HilbertCmp(coord2,  dir2, points, ymid));
   int *p3 = std::partition(p2, p4, HilbertCmp(coord2, !dir2, points, ymid));

   if (p1 != p4)
   {
      HilbertSort2D(coord2, dir2, dir1, points, p0, p1,
                    ymin, xmin, ymid, xmid);
   }
   if (p1 != p0 || p2 != p4)
   {
      HilbertSort2D(coord1, dir1, dir2, points, p1, p2,
                    xmin, ymid, xmid, ymax);
   }
   if (p2 != p0 || p3 != p4)
   {
      HilbertSort2D(coord1, dir1, dir2, points, p2, p3,
                    xmid, ymid, xmax, ymax);
   }
   if (p3 != p0)
   {
      HilbertSort2D(coord2, !dir2, !dir1, points, p3, p4,
                    ymid, xmax, ymin, xmid);
   }
}

static void HilbertSort3D(int coord1, bool dir1, bool dir2, bool dir3,
                          const Array<real_t> &points, int *beg, int *end,
                          real_t xmin, real_t ymin, real_t zmin,
                          real_t xmax, real_t ymax, real_t zmax)
{
   if (end - beg <= 1) { return; }

   real_t xmid = (xmin + xmax)*0.5;
   real_t ymid = (ymin + ymax)*0.5;
   real_t zmid = (zmin + zmax)*0.5;

   int coord2 = (coord1 + 1) % 3;
   int coord3 = (coord1 + 2) % 3;

   // sort (partition) points into eight octants
   int *p0 = beg, *p8 = end;
   int *p4 = std::partition(p0, p8, HilbertCmp(coord1,  dir1, points, xmid));
   int *p2 = std::partition(p0, p4, HilbertCmp(coord2,  dir2, points, ymid));
   int *p6 = std::partition(p4, p8, HilbertCmp(coord2, !dir2, points, ymid));
   int *p1 = std::partition(p0, p2, HilbertCmp(coord3,  dir3, points, zmid));
   int *p3 = std::partition(p2, p4, HilbertCmp(coord3, !dir3, points, zmid));
   int *p5 = std::partition(p4, p6, HilbertCmp(coord3,  dir3, points, zmid));
   int *p7 = std::partition(p6, p8, HilbertCmp(coord3, !dir3, points, zmid));

   if (p1 != p8)
   {
      HilbertSort3D(coord3, dir3, dir1, dir2, points, p0, p1,
                    zmin, xmin, ymin, zmid, xmid, ymid);
   }
   if (p1 != p0 || p2 != p8)
   {
      HilbertSort3D(coord2, dir2, dir3, dir1, points, p1, p2,
                    ymin, zmid, xmin, ymid, zmax, xmid);
   }
   if (p2 != p0 || p3 != p8)
   {
      HilbertSort3D(coord2, dir2, dir3, dir1, points, p2, p3,
                    ymid, zmid, xmin, ymax, zmax, xmid);
   }
   if (p3 != p0 || p4 != p8)
   {
      HilbertSort3D(coord1, dir1, !dir2, !dir3, points, p3, p4,
                    xmin, ymax, zmid, xmid, ymid, zmin);
   }
   if (p4 != p0 || p5 != p8)
   {
      HilbertSort3D(coord1, dir1, !dir2, !dir3, points, p4, p5,
                    xmid, ymax, zmid, xmax, ymid, zmin);
   }
   if (p5 != p0 || p6 != p8)
   {
      HilbertSort3D(coord2, !dir2, dir3, !dir1, points, p5, p6,
                    ymax, zmid, xmax, ymid, zmax, xmid);
   }
   if (p6 != p0 || p7 != p8)
   {
      HilbertSort3D(coord2, !dir2, dir3, !dir1, points, p6, p7,
                    ymid, zmid, xmax, ymin, zmax, xmid);
   }
   if (p7 != p0)
   {
      HilbertSort3D(coord3, !dir3, !dir1, dir2, points, p7, p8,
                    zmid, xmax, ymin, zmin, xmid, ymid);
   }
}

void Mesh::GetHilbertElementOrdering(Array<int> &ordering)
{
   MFEM_VERIFY(spaceDim <= 3, "");

   Vector min, max, center;
   GetBoundingBox(min, max);

   Array<int> indices(GetNE());
   Array<real_t> points(3*GetNE());

   if (spaceDim < 3) { points = 0.0; }

   // calculate element centers
   for (int i = 0; i < GetNE(); i++)
   {
      GetElementCenter(i, center);
      for (int j = 0; j < spaceDim; j++)
      {
         points[3*i + j] = center(j);
      }
      indices[i] = i;
   }

   if (spaceDim == 1)
   {
      indices.Sort([&](int a, int b)
      { return points[3*a] < points[3*b]; });
   }
   else if (spaceDim == 2)
   {
      // recursively partition the points in 2D
      HilbertSort2D(0, false, false,
                    points, indices.begin(), indices.end(),
                    min(0), min(1), max(0), max(1));
   }
   else
   {
      // recursively partition the points in 3D
      HilbertSort3D(0, false, false, false,
                    points, indices.begin(), indices.end(),
                    min(0), min(1), min(2), max(0), max(1), max(2));
   }

   // return ordering in the format required by ReorderElements
   ordering.SetSize(GetNE());
   for (int i = 0; i < GetNE(); i++)
   {
      ordering[indices[i]] = i;
   }
}


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
   // - geom_factors - no need to rebuild

   // - be_to_face

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
      // generate el_to_edge, be_to_face (2D), bel_to_edge (3D)
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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
      nodes_sequence++;
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
   // Mark the longest triangle edge by rotating the indices so that
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

void Mesh::GetEdgeOrdering(const DSTable &v_to_v, Array<int> &order)
{
   NumOfEdges = v_to_v.NumberOfEntries();
   order.SetSize(NumOfEdges);
   Array<Pair<real_t, int> > length_idx(NumOfEdges);

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

void Mesh::MarkTetMeshForRefinement(const DSTable &v_to_v)
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
   Nodes->HostReadWrite(); // for "(*Nodes)() = "
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
            case Geometry::TETRAHEDRON:
               new_or = GetTetOrientation(old_v, new_v);
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
      // update 'el_to_edge', 'be_to_face' (2D), 'bel_to_edge' (3D)
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
      if (Dim == 2)
      {
         // update 'faces' and 'faces_info'
         GenerateFaces();
         CheckBdrElementOrientation();
      }
   }
   // To force FE space update, we need to increase 'sequence':
   sequence++;
   nodes_sequence++;
   last_operation = Mesh::NONE;
   fes->Update(false); // want_transform = false
   Nodes->Update(); // just needed to update Nodes->sequence
}

void Mesh::SetPatchAttribute(int i, int attr)
{
   MFEM_ASSERT(NURBSext, "SetPatchAttribute is only for NURBS meshes");
   NURBSext->SetPatchAttribute(i, attr);
   const Array<int>& elems = NURBSext->GetPatchElements(i);
   for (auto e : elems)
   {
      SetAttribute(e, attr);
   }
}

int Mesh::GetPatchAttribute(int i) const
{
   MFEM_ASSERT(NURBSext, "GetPatchAttribute is only for NURBS meshes");
   return NURBSext->GetPatchAttribute(i);
}

void Mesh::SetPatchBdrAttribute(int i, int attr)
{
   MFEM_ASSERT(NURBSext, "SetPatchBdrAttribute is only for NURBS meshes");
   NURBSext->SetPatchBdrAttribute(i, attr);

   const Array<int>& bdryelems = NURBSext->GetPatchBdrElements(i);
   for (auto be : bdryelems)
   {
      SetBdrAttribute(be, attr);
   }
}

int Mesh::GetPatchBdrAttribute(int i) const
{
   MFEM_ASSERT(NURBSext, "GetBdrPatchBdrAttribute is only for NURBS meshes");
   return NURBSext->GetPatchBdrAttribute(i);
}

void Mesh::GetNURBSPatches(Array<NURBSPatch*> &patches)
{
   MFEM_VERIFY(NURBSext, "Must be a NURBS mesh");
   // This sets the data in NURBSPatch(es) from the control points (Nodes)
   NURBSext->ConvertToPatches(*Nodes);

   // Deep copy patches
   NURBSext->GetPatches(patches);

   // Among other things, this deletes patches in NURBSext
   UpdateNURBS();
}

void Mesh::FinalizeTetMesh(int generate_edges, int refine, bool fix_orientation)
{
   FinalizeCheck();
   CheckElementOrientation(fix_orientation);

   if (!HasBoundaryElements())
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
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

   if (!HasBoundaryElements())
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
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

   if (!HasBoundaryElements())
   {
      GenerateBoundaryElements();
   }

   CheckBdrElementOrientation();

   if (generate_edges)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

void Mesh::FinalizeTopology(bool generate_bdr)
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

   // if the user defined any hanging nodes (see AddVertexParent),
   // we're initializing a non-conforming mesh
   if (tmp_vertex_parents.Size())
   {
      MFEM_VERIFY(ncmesh == NULL, "");
      ncmesh = new NCMesh(this);

      // we need to recreate the Mesh because NCMesh reorders the vertices
      // (see NCMesh::UpdateVertices())
      InitFromNCMesh(*ncmesh);
      ncmesh->OnMeshUpdated(this);
      GenerateNCFaceInfo();

      SetAttributes();

      tmp_vertex_parents.DeleteAll();
      return;
   }

   // set the mesh type: 'meshgen', ...
   SetMeshGen();

   // generate the faces
   if (Dim > 2)
   {
      GetElementToFaceTable();
      GenerateFaces();
      if (!HasBoundaryElements() && generate_bdr)
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
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
      if (Dim == 2)
      {
         GenerateFaces(); // 'Faces' in 2D refers to the edges
         if (!HasBoundaryElements() && generate_bdr)
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
      if (!HasBoundaryElements() && generate_bdr)
      {
         // be_to_face will be set inside GenerateBoundaryElements
         GenerateBoundaryElements();
      }
      else
      {
         be_to_face.SetSize(NumOfBdrElements);
         for (int i = 0; i < NumOfBdrElements; ++i)
         {
            be_to_face[i] = boundary[i]->GetVertices()[0];
         }
      }
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
                     faces_info[i].Elem2Inf%2 != 0, "Invalid mesh topology."
                     " Interior face with incompatible orientations.");
      }
   }
#endif
}

void Mesh::Make3D(int nx, int ny, int nz, Element::Type type,
                  real_t sx, real_t sy, real_t sz, bool sfc_ordering)
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
   else if (type == Element::PYRAMID)
   {
      NElem *= 6;
      NVert += nx * ny * nz;
   }

   InitMesh(3, 3, NVert, NElem, NBdrElem);

   real_t coord[3];
   int ind[9];

   // Sets vertices and the corresponding coordinates
   for (z = 0; z <= nz; z++)
   {
      coord[2] = ((real_t) z / nz) * sz;
      for (y = 0; y <= ny; y++)
      {
         coord[1] = ((real_t) y / ny) * sy;
         for (x = 0; x <= nx; x++)
         {
            coord[0] = ((real_t) x / nx) * sx;
            AddVertex(coord);
         }
      }
   }
   if (type == Element::PYRAMID)
   {
      for (z = 0; z < nz; z++)
      {
         coord[2] = (((real_t) z + 0.5) / nz) * sz;
         for (y = 0; y < ny; y++)
         {
            coord[1] = (((real_t) y + 0.5) / ny) * sy;
            for (x = 0; x < nx; x++)
            {
               coord[0] = (((real_t) x + 0.5) / nx) * sx;
               AddVertex(coord);
            }
         }
      }
   }

#define VTX(XC, YC, ZC) ((XC)+((YC)+(ZC)*(ny+1))*(nx+1))
#define VTXP(XC, YC, ZC) ((nx+1)*(ny+1)*(nz+1)+(XC)+((YC)+(ZC)*ny)*nx)

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

         // *INDENT-OFF*
         ind[0] = VTX(x  , y  , z  );
         ind[1] = VTX(x+1, y  , z  );
         ind[2] = VTX(x+1, y+1, z  );
         ind[3] = VTX(x  , y+1, z  );
         ind[4] = VTX(x  , y  , z+1);
         ind[5] = VTX(x+1, y  , z+1);
         ind[6] = VTX(x+1, y+1, z+1);
         ind[7] = VTX(x  , y+1, z+1);
         // *INDENT-ON*

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
               // *INDENT-OFF*
               ind[0] = VTX(x  , y  , z  );
               ind[1] = VTX(x+1, y  , z  );
               ind[2] = VTX(x+1, y+1, z  );
               ind[3] = VTX(x  , y+1, z  );
               ind[4] = VTX(x  , y  , z+1);
               ind[5] = VTX(x+1, y  , z+1);
               ind[6] = VTX(x+1, y+1, z+1);
               ind[7] = VTX(  x, y+1, z+1);
               // *INDENT-ON*
               if (type == Element::TETRAHEDRON)
               {
                  AddHexAsTets(ind, 1);
               }
               else if (type == Element::WEDGE)
               {
                  AddHexAsWedges(ind, 1);
               }
               else if (type == Element::PYRAMID)
               {
                  ind[8] = VTXP(x, y, z);
                  AddHexAsPyramids(ind, 1);
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
         // *INDENT-OFF*
         ind[0] = VTX(x  , y  , 0);
         ind[1] = VTX(x  , y+1, 0);
         ind[2] = VTX(x+1, y+1, 0);
         ind[3] = VTX(x+1, y  , 0);
         // *INDENT-ON*
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
         // *INDENT-OFF*
         ind[0] = VTX(x  , y  , nz);
         ind[1] = VTX(x+1, y  , nz);
         ind[2] = VTX(x+1, y+1, nz);
         ind[3] = VTX(x  , y+1, nz);
         // *INDENT-ON*
         if (type == Element::TETRAHEDRON)
         {
            AddBdrQuadAsTriangles(ind, 6);
         }
         else if (type == Element::WEDGE)
         {
            AddBdrQuadAsTriangles(ind, 6);
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
         // *INDENT-OFF*
         ind[0] = VTX(0  , y  , z  );
         ind[1] = VTX(0  , y  , z+1);
         ind[2] = VTX(0  , y+1, z+1);
         ind[3] = VTX(0  , y+1, z  );
         // *INDENT-ON*
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
         // *INDENT-OFF*
         ind[0] = VTX(nx, y  , z  );
         ind[1] = VTX(nx, y+1, z  );
         ind[2] = VTX(nx, y+1, z+1);
         ind[3] = VTX(nx, y  , z+1);
         // *INDENT-ON*
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
         // *INDENT-OFF*
         ind[0] = VTX(x  , 0, z  );
         ind[1] = VTX(x+1, 0, z  );
         ind[2] = VTX(x+1, 0, z+1);
         ind[3] = VTX(x  , 0, z+1);
         // *INDENT-ON*
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
         // *INDENT-OFF*
         ind[0] = VTX(x  , ny, z  );
         ind[1] = VTX(x  , ny, z+1);
         ind[2] = VTX(x+1, ny, z+1);
         ind[3] = VTX(x+1, ny, z  );
         // *INDENT-ON*
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
#undef VTXP

#if 0
   ofstream test_stream("debug.mesh");
   Print(test_stream);
   test_stream.close();
#endif

   FinalizeTopology();

   // Finalize(...) can be called after this method, if needed
}


void Mesh::Make2D4TrisFromQuad(int nx, int ny, real_t sx, real_t sy)
{
   SetEmpty();

   Dim = 2;
   spaceDim = 2;

   NumOfVertices = (nx+1) * (ny+1);
   NumOfElements = nx * ny * 4;
   NumOfBdrElements =  (2 * nx + 2 * ny);
   vertices.SetSize(NumOfVertices);
   elements.SetSize(NumOfElements);
   boundary.SetSize(NumOfBdrElements);
   NumOfElements = 0;

   int ind[4];

   // Sets vertices and the corresponding coordinates
   int k = 0;
   for (real_t j = 0; j < ny+1; j++)
   {
      real_t cy = (j / ny) * sy;
      for (real_t i = 0; i < nx+1; i++)
      {
         real_t cx = (i / nx) * sx;
         vertices[k](0) = cx;
         vertices[k](1) = cy;
         k++;
      }
   }

   for (int y = 0; y < ny; y++)
   {
      for (int x = 0; x < nx; x++)
      {
         ind[0] = x + y*(nx+1);
         ind[1] = x + 1 +y*(nx+1);
         ind[2] = x + 1 + (y+1)*(nx+1);
         ind[3] = x + (y+1)*(nx+1);
         AddQuadAs4TrisWithPoints(ind, 1);
      }
   }

   int m = (nx+1)*ny;
   for (int i = 0; i < nx; i++)
   {
      boundary[i] = new Segment(i, i+1, 1);
      boundary[nx+i] = new Segment(m+i+1, m+i, 3);
   }
   m = nx+1;
   for (int j = 0; j < ny; j++)
   {
      boundary[2*nx+j] = new Segment((j+1)*m, j*m, 4);
      boundary[2*nx+ny+j] = new Segment(j*m+nx, (j+1)*m+nx, 2);
   }

   SetMeshGen();
   CheckElementOrientation(true);

   el_to_edge = new Table;
   NumOfEdges = GetElementToEdgeTable(*el_to_edge);
   GenerateFaces();
   CheckBdrElementOrientation();

   NumOfFaces = 0;

   attributes.Append(1);
   bdr_attributes.Append(1); bdr_attributes.Append(2);
   bdr_attributes.Append(3); bdr_attributes.Append(4);

   FinalizeTopology();
}

void Mesh::Make2D5QuadsFromQuad(int nx, int ny,
                                real_t sx, real_t sy)
{
   SetEmpty();

   Dim = 2;
   spaceDim = 2;

   NumOfElements = nx * ny * 5;
   NumOfVertices = (nx+1) * (ny+1); //it will be enlarged later on
   NumOfBdrElements =  (2 * nx + 2 * ny);
   vertices.SetSize(NumOfVertices);
   elements.SetSize(NumOfElements);
   boundary.SetSize(NumOfBdrElements);
   NumOfElements = 0;

   int ind[4];

   // Sets vertices and the corresponding coordinates
   int k = 0;
   for (real_t j = 0; j < ny+1; j++)
   {
      real_t cy = (j / ny) * sy;
      for (real_t i = 0; i < nx+1; i++)
      {
         real_t cx = (i / nx) * sx;
         vertices[k](0) = cx;
         vertices[k](1) = cy;
         k++;
      }
   }

   for (int y = 0; y < ny; y++)
   {
      for (int x = 0; x < nx; x++)
      {
         ind[0] = x + y*(nx+1);
         ind[1] = x + 1 +y*(nx+1);
         ind[2] = x + 1 + (y+1)*(nx+1);
         ind[3] = x + (y+1)*(nx+1);
         AddQuadAs5QuadsWithPoints(ind, 1);
      }
   }

   int m = (nx+1)*ny;
   for (int i = 0; i < nx; i++)
   {
      boundary[i] = new Segment(i, i+1, 1);
      boundary[nx+i] = new Segment(m+i+1, m+i, 3);
   }
   m = nx+1;
   for (int j = 0; j < ny; j++)
   {
      boundary[2*nx+j] = new Segment((j+1)*m, j*m, 4);
      boundary[2*nx+ny+j] = new Segment(j*m+nx, (j+1)*m+nx, 2);
   }

   SetMeshGen();
   CheckElementOrientation(true);

   el_to_edge = new Table;
   NumOfEdges = GetElementToEdgeTable(*el_to_edge);
   GenerateFaces();
   CheckBdrElementOrientation();

   NumOfFaces = 0;

   attributes.Append(1);
   bdr_attributes.Append(1); bdr_attributes.Append(2);
   bdr_attributes.Append(3); bdr_attributes.Append(4);

   FinalizeTopology();
}

void Mesh::Make3D24TetsFromHex(int nx, int ny, int nz,
                               real_t sx, real_t sy, real_t sz)
{
   const int NVert = (nx+1) * (ny+1) * (nz+1);
   const int NElem = nx * ny * nz * 24;
   const int NBdrElem = 2*(nx*ny+nx*nz+ny*nz)*4;

   InitMesh(3, 3, NVert, NElem, NBdrElem);

   real_t coord[3];

   // Sets vertices and the corresponding coordinates
   for (real_t z = 0; z <= nz; z++)
   {
      coord[2] = ( z / nz) * sz;
      for (real_t y = 0; y <= ny; y++)
      {
         coord[1] = (y / ny) * sy;
         for (real_t x = 0; x <= nx; x++)
         {
            coord[0] = (x / nx) * sx;
            AddVertex(coord);
         }
      }
   }

   std::map<std::array<int, 4>, int> hex_face_verts;
   auto VertexIndex = [nx, ny](int xc, int yc, int zc)
   {
      return xc + (yc + zc*(ny+1))*(nx+1);
   };

   int ind[9];
   for (int z = 0; z < nz; z++)
   {
      for (int y = 0; y < ny; y++)
      {
         for (int x = 0; x < nx; x++)
         {
            // *INDENT-OFF*
            ind[0] = VertexIndex(x  , y  , z  );
            ind[1] = VertexIndex(x+1, y  , z  );
            ind[2] = VertexIndex(x+1, y+1, z  );
            ind[3] = VertexIndex(x  , y+1, z  );
            ind[4] = VertexIndex(x  , y  , z+1);
            ind[5] = VertexIndex(x+1, y  , z+1);
            ind[6] = VertexIndex(x+1, y+1, z+1);
            ind[7] = VertexIndex(  x, y+1, z+1);
            // *INDENT-ON*

            AddHexAs24TetsWithPoints(ind, hex_face_verts, 1);
         }
      }
   }

   hex_face_verts.clear();
   CheckElementOrientation(true);

   // Done adding Tets
   // Now figure out elements that are on the boundary
   GetElementToFaceTable(false);
   GenerateFaces();

   // Map to count number of tets sharing a face
   std::map<std::array<int, 3>, int> tet_face_count;
   // Map from tet face defined by three vertices to the local face number
   std::map<std::array<int, 3>, int> face_count_map;

   auto get3array = [](Array<int> v)
   {
      v.Sort();
      return std::array<int, 3> {v[0], v[1], v[2]};
   };

   Array<int> el_faces;
   Array<int> ori;
   Array<int> vertidxs;
   for (int i = 0; i < el_to_face->Size(); i++)
   {
      el_to_face->GetRow(i, el_faces);
      for (int j = 0; j < el_faces.Size(); j++)
      {
         GetFaceVertices(el_faces[j], vertidxs);
         auto t = get3array(vertidxs);
         auto it = tet_face_count.find(t);
         if (it == tet_face_count.end())  //edge does not already exist
         {
            tet_face_count.insert({t, 1});
            face_count_map.insert({t, el_faces[j]});
         }
         else
         {
            it->second++; // increase edge count value by 1.
         }
      }
   }

   for (const auto &edge : tet_face_count)
   {
      if (edge.second == 1)  //if this only appears once, it is a boundary edge
      {
         int facenum = (face_count_map.find(edge.first))->second;
         GetFaceVertices(facenum, vertidxs);
         AddBdrTriangle(vertidxs, 1);
      }
   }

#if 0
   ofstream test_stream("debug.mesh");
   Print(test_stream);
   test_stream.close();
#endif

   FinalizeTopology();
   // Finalize(...) can be called after this method, if needed
}

void Mesh::Make2D(int nx, int ny, Element::Type type,
                  real_t sx, real_t sy,
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

      real_t cx, cy;
      int ind[4];

      // Sets vertices and the corresponding coordinates
      k = 0;
      for (j = 0; j < ny+1; j++)
      {
         cy = ((real_t) j / ny) * sy;
         for (i = 0; i < nx+1; i++)
         {
            cx = ((real_t) i / nx) * sx;
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

      real_t cx, cy;
      int ind[3];

      // Sets vertices and the corresponding coordinates
      k = 0;
      for (j = 0; j < ny+1; j++)
      {
         cy = ((real_t) j / ny) * sy;
         for (i = 0; i < nx+1; i++)
         {
            cx = ((real_t) i / nx) * sx;
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
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

void Mesh::Make1D(int n, real_t sx)
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
      vertices[j](0) = ((real_t) j / n) * sx;
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

   // Set be_to_face
   be_to_face.SetSize(2);
   be_to_face[0] = 0;
   be_to_face[1] = n;

   attributes.Append(1);
   bdr_attributes.Append(1); bdr_attributes.Append(2);
}

Mesh::Mesh(const Mesh &mesh, bool copy_nodes)
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
{
   Dim = mesh.Dim;
   spaceDim = mesh.spaceDim;

   NumOfVertices = mesh.NumOfVertices;
   NumOfElements = mesh.NumOfElements;
   NumOfBdrElements = mesh.NumOfBdrElements;
   NumOfEdges = mesh.NumOfEdges;
   NumOfFaces = mesh.NumOfFaces;
   nbInteriorFaces = mesh.nbInteriorFaces;
   nbBoundaryFaces = mesh.nbBoundaryFaces;

   meshgen = mesh.meshgen;
   mesh_geoms = mesh.mesh_geoms;

   // Create the new Mesh instance without a record of its refinement history
   sequence = 0;
   nodes_sequence = 0;
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

   // Copy the boundary-to-edge Table, bel_to_edge (3D)
   bel_to_edge = (mesh.bel_to_edge) ? new Table(*mesh.bel_to_edge) : NULL;

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
   face_to_elem = NULL;

   // Copy the edge-to-vertex Table, edge_vertex
   edge_vertex = (mesh.edge_vertex) ? new Table(*mesh.edge_vertex) : NULL;

   // Copy the attributes and bdr_attributes
   mesh.attributes.Copy(attributes);
   mesh.bdr_attributes.Copy(bdr_attributes);

   // Copy attribute and bdr_attribute names
   mesh.attribute_sets.Copy(attribute_sets);
   mesh.bdr_attribute_sets.Copy(bdr_attribute_sets);

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

   // copy attribute caches
   elem_attrs_cache = mesh.elem_attrs_cache;
   bdr_face_attrs_cache = mesh.bdr_face_attrs_cache;
}

Mesh::Mesh(Mesh &&mesh) : Mesh()
{
   Swap(mesh, true);
}

Mesh& Mesh::operator=(Mesh &&mesh)
{
   Swap(mesh, true);
   return *this;
}

Mesh Mesh::LoadFromFile(const std::string &filename, int generate_edges,
                        int refine, bool fix_orientation)
{
   Mesh mesh;
   named_ifgzstream imesh(filename);
   if (!imesh) { MFEM_ABORT("Mesh file not found: " << filename << '\n'); }
   else { mesh.Load(imesh, generate_edges, refine, fix_orientation); }
   return mesh;
}

Mesh Mesh::MakeCartesian1D(int n, real_t sx)
{
   Mesh mesh;
   mesh.Make1D(n, sx);
   // mesh.Finalize(); not needed in this case
   return mesh;
}

Mesh Mesh::MakeCartesian2D(
   int nx, int ny, Element::Type type, bool generate_edges,
   real_t sx, real_t sy, bool sfc_ordering)
{
   Mesh mesh;
   mesh.Make2D(nx, ny, type, sx, sy, generate_edges, sfc_ordering);
   mesh.Finalize(true); // refine = true
   return mesh;
}

Mesh Mesh::MakeCartesian3D(
   int nx, int ny, int nz, Element::Type type,
   real_t sx, real_t sy, real_t sz, bool sfc_ordering)
{
   Mesh mesh;
   mesh.Make3D(nx, ny, nz, type, sx, sy, sz, sfc_ordering);
   mesh.Finalize(true); // refine = true
   return mesh;
}

Mesh Mesh::MakeCartesian3DWith24TetsPerHex(int nx, int ny, int nz,
                                           real_t sx, real_t sy, real_t sz)
{
   Mesh mesh;
   mesh.Make3D24TetsFromHex(nx, ny, nz, sx, sy, sz);
   mesh.Finalize(false, false);
   return mesh;
}

Mesh Mesh::MakeCartesian2DWith4TrisPerQuad(int nx, int ny,
                                           real_t sx, real_t sy)
{
   Mesh mesh;
   mesh.Make2D4TrisFromQuad(nx, ny, sx, sy);
   mesh.Finalize(false, false);
   return mesh;
}

Mesh Mesh::MakeCartesian2DWith5QuadsPerQuad(int nx, int ny,
                                            real_t sx, real_t sy)
{
   Mesh mesh;
   mesh.Make2D5QuadsFromQuad(nx, ny, sx, sy);
   mesh.Finalize(false, false);
   return mesh;
}

Mesh Mesh::MakeRefined(Mesh &orig_mesh, int ref_factor, int ref_type)
{
   Mesh mesh;
   Array<int> ref_factors(orig_mesh.GetNE());
   ref_factors = ref_factor;
   mesh.MakeRefined_(orig_mesh, ref_factors, ref_type);
   return mesh;
}

Mesh Mesh::MakeRefined(Mesh &orig_mesh, const Array<int> &ref_factors,
                       int ref_type)
{
   Mesh mesh;
   mesh.MakeRefined_(orig_mesh, ref_factors, ref_type);
   return mesh;
}

Mesh::Mesh(const std::string &filename, int generate_edges, int refine,
           bool fix_orientation)
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
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
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
{
   SetEmpty();
   Load(input, generate_edges, refine, fix_orientation);
}

void Mesh::ChangeVertexDataOwnership(real_t *vertex_data, int len_vertex_data,
                                     bool zerocopy)
{
   // A dimension of 3 is now required since we use mfem::Vertex objects as PODs
   // and these object have a hardcoded double[3] entry
   MFEM_VERIFY(len_vertex_data >= NumOfVertices * 3,
               "Not enough vertices in external array : "
               "len_vertex_data = "<< len_vertex_data << ", "
               "NumOfVertices * 3 = " << NumOfVertices * 3);
   // Allow multiple calls to this method with the same vertex_data
   if (vertex_data == (real_t *)(vertices.GetData()))
   {
      MFEM_ASSERT(!vertices.OwnsData(), "invalid ownership");
      return;
   }
   if (!zerocopy)
   {
      memcpy(vertex_data, vertices.GetData(),
             NumOfVertices * 3 * sizeof(real_t));
   }
   // Vertex is POD double[3]
   vertices.MakeRef(reinterpret_cast<Vertex*>(vertex_data), NumOfVertices);
}

Mesh::Mesh(real_t *vertices_, int num_vertices,
           int *element_indices, Geometry::Type element_type,
           int *element_attributes, int num_elements,
           int *boundary_indices, Geometry::Type boundary_type,
           int *boundary_attributes, int num_boundary_elements,
           int dimension, int space_dimension)
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
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
   vertices.MakeRef(reinterpret_cast<Vertex*>(vertices_), num_vertices);
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

Mesh::Mesh(const NURBSExtension& ext)
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
{
   SetEmpty();
   /// make an internal copy of the NURBSExtension
   NURBSext = new NURBSExtension(ext);

   Dim              = NURBSext->Dimension();
   NumOfVertices    = NURBSext->GetNV();
   NumOfElements    = NURBSext->GetNE();
   NumOfBdrElements = NURBSext->GetNBE();

   NURBSext->GetElementTopo(elements);
   NURBSext->GetBdrElementTopo(boundary);

   vertices.SetSize(NumOfVertices);
   if (NURBSext->HavePatches())
   {
      NURBSFECollection  *fec = new NURBSFECollection(NURBSext->GetOrder());
      const int vdim = NURBSext->GetPatchSpaceDimension();
      FiniteElementSpace *fes = new FiniteElementSpace(this, fec, vdim,
                                                       Ordering::byVDIM);
      Nodes = new GridFunction(fes);
      Nodes->MakeOwner(fec);
      NURBSext->SetCoordsFromPatches(*Nodes, vdim);
      own_nodes = 1;
      spaceDim = Nodes->VectorDim();
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
   else
   {
      MFEM_ABORT("NURBS mesh has no patches.");
   }
   FinalizeMesh();
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
      case Geometry::PYRAMID:   return (new Pyramid);
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

void Mesh::PrintElementWithoutAttr(const Element *el, std::ostream &os)
{
   os << el->GetGeometryType();
   const int nv = el->GetNVertices();
   const int *v = el->GetVertices();
   for (int j = 0; j < nv; j++)
   {
      os << ' ' << v[j];
   }
   os << '\n';
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

void Mesh::PrintElement(const Element *el, std::ostream &os)
{
   os << el->GetAttribute() << ' ';
   PrintElementWithoutAttr(el, os);
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

         case Element::PYRAMID:
            mesh_geoms |= (1 << Geometry::PYRAMID);
            mesh_geoms |= (1 << Geometry::SQUARE);
            mesh_geoms |= (1 << Geometry::TRIANGLE);
            mesh_geoms |= (1 << Geometry::SEGMENT);
            mesh_geoms |= (1 << Geometry::POINT);
            meshgen |= 8;
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

   // MFEM's conforming mesh formats
   int mfem_version = 0;
   if (mesh_type == "MFEM mesh v1.0") { mfem_version = 10; } // serial
   else if (mesh_type == "MFEM mesh v1.2") { mfem_version = 12; } // parallel
   else if (mesh_type == "MFEM mesh v1.3") { mfem_version = 13; } // attr sets

   // MFEM nonconforming mesh format
   // (NOTE: previous v1.1 is now under this branch for backward compatibility)
   int mfem_nc_version = 0;
   if (mesh_type == "MFEM NC mesh v1.0") { mfem_nc_version = 10; }
   else if (mesh_type == "MFEM NC mesh v1.1") { mfem_nc_version = 11; }
   else if (mesh_type == "MFEM mesh v1.1") { mfem_nc_version = 1 /*legacy*/; }

   if (mfem_version)
   {
      // Formats mfem_v12 and newer have a tag indicating the end of the mesh
      // section in the stream. A user provided parse tag can also be provided
      // via the arguments. For example, if this is called from parallel mesh
      // object, it can indicate to read until parallel mesh section begins.
      if (mfem_version >= 12 && parse_tag.empty())
      {
         parse_tag = "mfem_mesh_end";
      }
      ReadMFEMMesh(input, mfem_version, curved);
   }
   else if (mfem_nc_version)
   {
      MFEM_ASSERT(ncmesh == NULL, "internal error");
      int is_nc = 1;

#ifdef MFEM_USE_MPI
      ParMesh *pmesh = dynamic_cast<ParMesh*>(this);
      if (pmesh)
      {
         MFEM_VERIFY(mfem_nc_version >= 10,
                     "Legacy nonconforming format (MFEM mesh v1.1) cannot be "
                     "used to load a parallel nonconforming mesh, sorry.");

         ncmesh = new ParNCMesh(pmesh->GetComm(),
                                input, mfem_nc_version, curved, is_nc);
      }
      else
#endif
      {
         ncmesh = new NCMesh(input, mfem_nc_version, curved, is_nc);
      }
      InitFromNCMesh(*ncmesh);

      if (!is_nc)
      {
         // special case for backward compatibility with MFEM <=4.2:
         // if the "vertex_parents" section is missing in the v1.1 format,
         // the mesh is treated as conforming
         delete ncmesh;
         ncmesh = NULL;
      }
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
   else if (mesh_type.rfind("# vtk DataFile Version") == 0)
   {
      int major_vtk_version = mesh_type[mesh_type.length()-3] - '0';
      // int minor_vtk_version = mesh_type[mesh_type.length()-1] - '0';
      MFEM_VERIFY(major_vtk_version >= 2 && major_vtk_version <= 4,
                  "Unsupported VTK format");
      ReadVTKMesh(input, curved, read_gf, finalize_topo);
   }
   else if (mesh_type.rfind("<VTKFile ") == 0 || mesh_type.rfind("<?xml") == 0)
   {
      ReadXML_VTKMesh(input, curved, read_gf, finalize_topo, mesh_type);
   }
   else if (mesh_type == "MFEM NURBS mesh v1.0")
   {
      ReadNURBSMesh(input, curved, read_gf);
   }
   else if (mesh_type == "MFEM NURBS NC-patch mesh v1.0")
   {
      ReadNURBSMesh(input, curved, read_gf, true, true);  // Spacing is required
   }
   else if (mesh_type == "MFEM NURBS mesh v1.1")
   {
      ReadNURBSMesh(input, curved, read_gf, true);
   }
   else if (mesh_type == "MFEM INLINE mesh v1.0")
   {
      ReadInlineMesh(input, generate_edges);
      return; // done with inline mesh construction
   }
   else if (mesh_type == "$MeshFormat") // Gmsh
   {
      ReadGmshMesh(input, curved, read_gf);
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
      // don't generate any boundary elements, especially in parallel
      bool generate_bdr = false;

      FinalizeTopology(generate_bdr);
   }

   if (curved && read_gf)
   {
      Nodes = new GridFunction(this, input);

      own_nodes = 1;
      spaceDim = Nodes->VectorDim();
      if (ncmesh) { ncmesh->spaceDim = spaceDim; }

      // Set vertex coordinates from the 'Nodes'
      SetVerticesFromNodes(Nodes);
   }

   // If a parse tag was supplied, keep reading the stream until the tag is
   // encountered.
   if (mfem_version >= 12)
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
   else if (mfem_nc_version >= 10)
   {
      string ident;
      skip_comment_lines(input, '#');
      input >> ident;
      MFEM_VERIFY(ident == "mfem_mesh_end",
                  "invalid mesh: end of file tag not found");
   }

   if (NURBSext && NURBSext->NonconformingPatches())
   {
      string ident;
      skip_comment_lines(input, '#');
      // Check for the optional section "patch_cp"
      if (input.peek() == 'p')
      {
         input >> ident;
         MFEM_VERIFY(ident == "patch_cp", "Invalid mesh format");
         NURBSext->ReadCoarsePatchCP(input);
      }
   }

   // Finalize(...) should be called after this, if needed.
}

Mesh::Mesh(Mesh *mesh_array[], int num_pieces)
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
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
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
{
   Array<int> ref_factors(orig_mesh->GetNE());
   ref_factors = ref_factor;
   MakeRefined_(*orig_mesh, ref_factors, ref_type);
}

void Mesh::MakeRefined_(Mesh &orig_mesh, const Array<int> &ref_factors,
                        int ref_type)
{
   SetEmpty();
   Dim = orig_mesh.Dimension();
   spaceDim = orig_mesh.SpaceDimension();

   int orig_ne = orig_mesh.GetNE();
   MFEM_VERIFY(ref_factors.Size() == orig_ne,
               "Number of refinement factors must equal number of elements")
   MFEM_VERIFY(orig_ne == 0 ||
               ref_factors.Min() >= 1, "Refinement factor must be >= 1");
   const int q_type = BasisType::GetQuadrature1D(ref_type);
   MFEM_VERIFY(Quadrature1D::CheckClosed(q_type) != Quadrature1D::Invalid,
               "Invalid refinement type. Must use closed basis type.");

   int min_ref = orig_ne > 0 ? ref_factors.Min() : 1;
   int max_ref = orig_ne > 0 ? ref_factors.Max() : 1;

   bool var_order = (min_ref != max_ref);

   // variable order space can only be constructed over an NC mesh
   if (var_order) { orig_mesh.EnsureNCMesh(true); }

   // Construct a scalar H1 FE space of order ref_factor and use its dofs as
   // the indices of the new, refined vertices.
   H1_FECollection rfec(min_ref, Dim, ref_type);
   FiniteElementSpace rfes(&orig_mesh, &rfec);

   if (var_order)
   {
      rfes.SetRelaxedHpConformity(false);
      for (int i = 0; i < orig_ne; i++)
      {
         rfes.SetElementOrder(i, ref_factors[i]);
      }
      rfes.Update(false);
   }

   // Set the number of vertices, set the actual coordinates later
   NumOfVertices = rfes.GetNDofs();
   vertices.SetSize(NumOfVertices);

   Array<int> rdofs;
   DenseMatrix phys_pts;
   GeometryRefiner refiner(q_type);

   // Add refined elements and set vertex coordinates
   for (int el = 0; el < orig_ne; el++)
   {
      Geometry::Type geom = orig_mesh.GetElementGeometry(el);
      int attrib = orig_mesh.GetAttribute(el);
      int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *refiner.Refine(geom, ref_factors[el]);

      rfes.GetElementDofs(el, rdofs);
      MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
      const FiniteElement *rfe = rfes.GetFE(el);
      orig_mesh.GetElementTransformation(el)->Transform(rfe->GetNodes(),
                                                        phys_pts);
      const int *c2h_map = rfec.GetDofMap(geom, ref_factors[el]);
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

   if (Dim > 2)
   {
      GetElementToFaceTable(false);
      GenerateFaces();
   }

   // Add refined boundary elements
   for (int el = 0; el < orig_mesh.GetNBE(); el++)
   {
      int i, info;
      orig_mesh.GetBdrElementAdjacentElement(el, i, info);
      Geometry::Type geom = orig_mesh.GetBdrElementGeometry(el);
      int attrib = orig_mesh.GetBdrAttribute(el);
      int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *refiner.Refine(geom, ref_factors[i]);

      rfes.GetBdrElementDofs(el, rdofs);
      MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
      const int *c2h_map = rfec.GetDofMap(geom, ref_factors[i]);
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
   FinalizeTopology(false);
   sequence = orig_mesh.GetSequence() + 1;
   last_operation = Mesh::REFINE;

   // Set up the nodes of the new mesh (if the original mesh has nodes). The new
   // mesh is always straight-sided (i.e. degree 1 finite element space), but
   // the nodes are required for e.g. periodic meshes.
   if (orig_mesh.GetNodes())
   {
      bool discont = orig_mesh.GetNodalFESpace()->IsDGSpace();
      Ordering::Type dof_ordering = orig_mesh.GetNodalFESpace()->GetOrdering();
      Mesh::SetCurvature(1, discont, spaceDim, dof_ordering);
      FiniteElementSpace *nodal_fes = Nodes->FESpace();
      const FiniteElementCollection *nodal_fec = nodal_fes->FEColl();
      H1_FECollection vertex_fec(1, Dim);
      Array<int> dofs;
      int el_counter = 0;
      for (int iel = 0; iel < orig_ne; iel++)
      {
         Geometry::Type geom = orig_mesh.GetElementBaseGeometry(iel);
         int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG = *refiner.Refine(geom, ref_factors[iel]);
         rfes.GetElementDofs(iel, rdofs);
         const FiniteElement *rfe = rfes.GetFE(iel);
         orig_mesh.GetElementTransformation(iel)->Transform(rfe->GetNodes(),
                                                            phys_pts);
         const int *node_map = NULL;
         const H1_FECollection *h1_fec =
            dynamic_cast<const H1_FECollection *>(nodal_fec);
         if (h1_fec != NULL) { node_map = h1_fec->GetDofMap(geom); }
         const int *vertex_map = vertex_fec.GetDofMap(geom);
         const int *c2h_map = rfec.GetDofMap(geom, ref_factors[iel]);
         for (int jel = 0; jel < RG.RefGeoms.Size()/nvert; jel++)
         {
            nodal_fes->GetElementVDofs(el_counter++, dofs);
            for (int iv_lex=0; iv_lex<nvert; ++iv_lex)
            {
               // convert from lexicographic to vertex index
               int iv = vertex_map[iv_lex];
               // index of vertex of current element in phys_pts matrix
               int pt_idx = c2h_map[RG.RefGeoms[iv+nvert*jel]];
               // index of current vertex into DOF array
               int node_idx = node_map ? node_map[iv_lex] : iv_lex;
               for (int d=0; d<spaceDim; ++d)
               {
                  (*Nodes)[dofs[node_idx + d*nvert]] = phys_pts(d,pt_idx);
               }
            }
         }
      }
   }

   // Setup the data for the coarse-fine refinement transformations
   CoarseFineTr.embeddings.SetSize(GetNE());
   // First, compute total number of point matrices that we need per geometry
   // and the offsets into that array
   using GeomRef = std::pair<Geometry::Type, int>;
   std::map<GeomRef, int> point_matrices_offsets;
   int n_point_matrices[Geometry::NumGeom] = {}; // initialize to zero
   for (int el_coarse = 0; el_coarse < orig_ne; ++el_coarse)
   {
      Geometry::Type geom = orig_mesh.GetElementBaseGeometry(el_coarse);
      // Have we seen this pair of (goemetry, refinement level) before?
      GeomRef id(geom, ref_factors[el_coarse]);
      if (point_matrices_offsets.find(id) == point_matrices_offsets.end())
      {
         RefinedGeometry &RG = *refiner.Refine(geom, ref_factors[el_coarse]);
         int nvert = Geometry::NumVerts[geom];
         int nref_el = RG.RefGeoms.Size()/nvert;
         // If not, then store the offset and add to the size required
         point_matrices_offsets[id] = n_point_matrices[geom];
         n_point_matrices[geom] += nref_el;
      }
   }

   // Set up the sizes
   for (int geom = 0; geom < Geometry::NumGeom; ++geom)
   {
      int nmatrices = n_point_matrices[geom];
      int nvert = Geometry::NumVerts[geom];
      CoarseFineTr.point_matrices[geom].SetSize(Dim, nvert, nmatrices);
   }

   // Compute the point matrices and embeddings
   int el_fine = 0;
   for (int el_coarse = 0; el_coarse < orig_ne; ++el_coarse)
   {
      Geometry::Type geom = orig_mesh.GetElementBaseGeometry(el_coarse);
      int ref = ref_factors[el_coarse];
      int offset = point_matrices_offsets[GeomRef(geom, ref)];
      int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *refiner.Refine(geom, ref);
      for (int j = 0; j < RG.RefGeoms.Size()/nvert; j++)
      {
         DenseMatrix &Pj = CoarseFineTr.point_matrices[geom](offset + j);
         for (int k = 0; k < nvert; k++)
         {
            int cid = RG.RefGeoms[k+nvert*j]; // local Cartesian index
            const IntegrationPoint &ip = RG.RefPts[cid];
            ip.Get(Pj.GetColumn(k), Dim);
         }

         Embedding &emb = CoarseFineTr.embeddings[el_fine];
         emb.geom = geom;
         emb.parent = el_coarse;
         emb.matrix = offset + j;
         ++el_fine;
      }
   }

   MFEM_ASSERT(CheckElementOrientation(false) == 0, "");

   // The check below is disabled because is fails for parallel meshes with
   // interior "boundary" element that, when such "boundary" element is between
   // two elements on different processors.
   // MFEM_ASSERT(CheckBdrElementOrientation(false) == 0, "");
}

Mesh Mesh::MakeSimplicial(const Mesh &orig_mesh)
{
   Mesh mesh;
   auto parent_elements = mesh.MakeSimplicial_(orig_mesh, NULL);
   if (orig_mesh.GetNodes() != nullptr)
   {
      mesh.MakeHigherOrderSimplicial_(orig_mesh, parent_elements);
   }
   return mesh;
}

Array<int> Mesh::MakeSimplicial_(const Mesh &orig_mesh, int *vglobal)
{
   MFEM_VERIFY(const_cast<Mesh&>(orig_mesh).CheckElementOrientation(false) == 0,
               "Mesh::MakeSimplicial requires a properly oriented input mesh");
   MFEM_VERIFY(orig_mesh.Conforming(),
               "Mesh::MakeSimplicial does not support non-conforming meshes.")

   int dim = orig_mesh.Dimension();
   int sdim = orig_mesh.SpaceDimension();

   if (dim == 1)
   {
      Mesh copy(orig_mesh);
      Swap(copy, true);
      Array<int> parent_elements(GetNE());
      std::iota(parent_elements.begin(), parent_elements.end(), 0);
      return parent_elements;
   }

   int nv = orig_mesh.GetNV();
   int ne = orig_mesh.GetNE();
   int nbe = orig_mesh.GetNBE();

   static int num_subdivisions[Geometry::NUM_GEOMETRIES];
   num_subdivisions[Geometry::POINT] = 1;
   num_subdivisions[Geometry::SEGMENT] = 1;
   num_subdivisions[Geometry::TRIANGLE] = 1;
   num_subdivisions[Geometry::TETRAHEDRON] = 1;
   num_subdivisions[Geometry::SQUARE] = 2;
   num_subdivisions[Geometry::PRISM] = 3;
   num_subdivisions[Geometry::CUBE] = 6;
   // NOTE: some hexes may be subdivided into only 5 tets, so this is an
   // estimate only. The actual number of created tets may be less, so the
   // elements array will need to be shrunk after mesh creation.
   int new_ne = 0, new_nbe = 0;
   for (int i=0; i<ne; ++i)
   {
      new_ne += num_subdivisions[orig_mesh.GetElementBaseGeometry(i)];
   }
   for (int i=0; i<nbe; ++i)
   {
      new_nbe += num_subdivisions[orig_mesh.GetBdrElementGeometry(i)];
   }

   InitMesh(dim, sdim, nv, new_ne, new_nbe);

   // Vertices of the new mesh are same as the original mesh
   NumOfVertices = nv;
   for (int i=0; i<nv; ++i)
   {
      vertices[i].SetCoords(sdim, orig_mesh.vertices[i]());
   }

   // We need a global vertex numbering to identify which diagonals to split
   // (quad faces are split using the diagonal originating from the smallest
   // global vertex number). Use the supplied global numbering, if it is
   // non-NULL, otherwise use the local numbering.
   Array<int> vglobal_id;
   if (vglobal == nullptr)
   {
      vglobal_id.SetSize(nv);
      std::iota(vglobal_id.begin(), vglobal_id.end(), 0);
      vglobal = vglobal_id.GetData();
   }

   // Number of vertices per element
   constexpr int nv_tri = 3, nv_quad = 4, nv_tet = 4, nv_prism = 6, nv_hex = 8;
   constexpr int quad_ntris = 2; // NTriangles per quad
   constexpr int prism_ntets = 3; // NTets per prism
   // Map verts of quad to verts of tri, in two possible configurations.
   // quad_trimap[i][0,2,4] is the first triangle, and quad_trimap[i][1,3,5] is
   // the second, for each configuration.
   static const int quad_trimap[2][nv_tri*quad_ntris] =
   {
      {
         0, 0,
         1, 2,
         2, 3
      },{
         0, 1,
         1, 2,
         3, 3
      }
   };
   static const int prism_rot[nv_prism*nv_prism] =
   {
      0, 1, 2, 3, 4, 5,
      1, 2, 0, 4, 5, 3,
      2, 0, 1, 5, 3, 4,
      3, 5, 4, 0, 2, 1,
      4, 3, 5, 1, 0, 2,
      5, 4, 3, 2, 1, 0
   };
   static const int prism_f[nv_quad] = {1, 2, 5, 4};
   static const int prism_tetmaps[2][nv_prism*prism_ntets] =
   {
      {
         0, 0, 0,
         1, 1, 4,
         2, 5, 5,
         5, 4, 3
      },{
         0, 0, 0,
         1, 4, 4,
         2, 2, 5,
         4, 5, 3
      }
   };
   static const int hex_rot[nv_hex*nv_hex] =
   {
      0, 1, 2, 3, 4, 5, 6, 7,
      1, 0, 4, 5, 2, 3, 7, 6,
      2, 1, 5, 6, 3, 0, 4, 7,
      3, 0, 1, 2, 7, 4, 5, 6,
      4, 0, 3, 7, 5, 1, 2, 6,
      5, 1, 0, 4, 6, 2, 3, 7,
      6, 2, 1, 5, 7, 3, 0, 4,
      7, 3, 2, 6, 4, 0, 1, 5
   };
   static const int hex_f0[nv_quad] = {1, 2, 6, 5};
   static const int hex_f1[nv_quad] = {2, 3, 7, 6};
   static const int hex_f2[nv_quad] = {4, 5, 6, 7};
   static const int num_rot[8] = {0, 1, 2, 0, 0, 2, 1, 0};
   static const int hex_tetmap0[nv_tet*5] =
   {
      0, 0, 0, 0, 2,
      1, 2, 2, 5, 7,
      2, 7, 3, 7, 5,
      5, 5, 7, 4, 6
   };
   static const int hex_tetmap1[nv_tet*6] =
   {
      0, 0, 1, 0, 0, 1,
      5, 1, 6, 7, 7, 7,
      7, 7, 7, 2, 1, 6,
      4, 5, 5, 3, 2, 2
   };
   static const int hex_tetmap2[nv_tet*6] =
   {
      0, 0, 0, 0, 0, 0,
      4, 3, 7, 1, 3, 6,
      5, 7, 4, 2, 6, 5,
      6, 6, 6, 5, 2, 2
   };
   static const int hex_tetmap3[nv_tet*6] =
   {
      0, 0, 0, 0, 1, 1,
      2, 3, 7, 5, 5, 6,
      3, 7, 4, 6, 6, 2,
      6, 6, 6, 4, 0, 0
   };
   static const int *hex_tetmaps[4] =
   {
      hex_tetmap0, hex_tetmap1, hex_tetmap2, hex_tetmap3
   };

   auto find_min = [](const int *a, int n) { return std::min_element(a,a+n)-a; };

   Array<int> parent_elems;
   for (int i=0; i<ne; ++i)
   {
      const int *v = orig_mesh.elements[i]->GetVertices();
      const int attrib = orig_mesh.GetAttribute(i);
      const Geometry::Type orig_geom = orig_mesh.GetElementBaseGeometry(i);

      if (num_subdivisions[orig_geom] == 1)
      {
         // (num_subdivisions[orig_geom] == 1) implies that the element does not
         // need to be further split (it is either a segment, triangle, or
         // tetrahedron), and so it is left unchanged.
         Element *e = NewElement(orig_geom);
         e->SetAttribute(attrib);
         e->SetVertices(v);
         AddElement(e);
         parent_elems.Append(i);
      }
      else if (orig_geom == Geometry::SQUARE)
      {
         for (int itri=0; itri<quad_ntris; ++itri)
         {
            Element *e = NewElement(Geometry::TRIANGLE);
            e->SetAttribute(attrib);
            int *v2 = e->GetVertices();
            for (int iv=0; iv<nv_tri; ++iv)
            {
               v2[iv] = v[quad_trimap[0][itri + iv*quad_ntris]];
            }
            AddElement(e);
            parent_elems.Append(i);
         }
      }
      else if (orig_geom == Geometry::PRISM)
      {
         int vg[nv_prism];
         for (int iv=0; iv<nv_prism; ++iv) { vg[iv] = vglobal[v[iv]]; }
         // Rotate the vertices of the prism so that the smallest vertex index
         // is in the first place
         int irot = find_min(vg, nv_prism);
         for (int iv=0; iv<nv_prism; ++iv)
         {
            int jv = prism_rot[iv + irot*nv_prism];
            vg[iv] = v[jv];
         }
         // Two cases according to which diagonal splits third quad face
         int q[nv_quad];
         for (int iv=0; iv<nv_quad; ++iv) { q[iv] = vglobal[vg[prism_f[iv]]]; }
         int j = find_min(q, nv_quad);
         const int *tetmap = (j == 0 || j == 2) ? prism_tetmaps[0] : prism_tetmaps[1];
         for (int itet=0; itet<prism_ntets; ++itet)
         {
            Element *e = NewElement(Geometry::TETRAHEDRON);
            e->SetAttribute(attrib);
            int *v2 = e->GetVertices();
            for (int iv=0; iv<nv_tet; ++iv)
            {
               v2[iv] = vg[tetmap[itet + iv*prism_ntets]];
            }
            AddElement(e);
            parent_elems.Append(i);
         }
      }
      else if (orig_geom == Geometry::CUBE)
      {
         int vg[nv_hex];
         for (int iv=0; iv<nv_hex; ++iv) { vg[iv] = vglobal[v[iv]]; }

         // Rotate the vertices of the hex so that the smallest vertex index is
         // in the first place
         int irot = find_min(vg, nv_hex);
         for (int iv=0; iv<nv_hex; ++iv)
         {
            int jv = hex_rot[iv + irot*nv_hex];
            vg[iv] = v[jv];
         }

         int q[nv_quad];
         // Bitmask is three binary digits, each digit is 1 if the diagonal of
         // the corresponding face goes through the 7th vertex, and 0 if not.
         int bitmask = 0;
         int j;
         // First quad face
         for (int iv=0; iv<nv_quad; ++iv) { q[iv] = vglobal[vg[hex_f0[iv]]]; }
         j = find_min(q, nv_quad);
         if (j == 0 || j == 2) { bitmask += 4; }
         // Second quad face
         for (int iv=0; iv<nv_quad; ++iv) { q[iv] = vglobal[vg[hex_f1[iv]]]; }
         j = find_min(q, nv_quad);
         if (j == 1 || j == 3) { bitmask += 2; }
         // Third quad face
         for (int iv=0; iv<nv_quad; ++iv) { q[iv] = vglobal[vg[hex_f2[iv]]]; }
         j = find_min(q, nv_quad);
         if (j == 0 || j == 2) { bitmask += 1; }

         // Apply rotations
         int nrot = num_rot[bitmask];
         for (int k=0; k<nrot; ++k)
         {
            int vtemp;
            vtemp = vg[1];
            vg[1] = vg[4];
            vg[4] = vg[3];
            vg[3] = vtemp;
            vtemp = vg[5];
            vg[5] = vg[7];
            vg[7] = vg[2];
            vg[2] = vtemp;
         }

         // Sum up nonzero bits in bitmask
         int ndiags = ((bitmask&4) >> 2) + ((bitmask&2) >> 1) + (bitmask&1);
         int ntets = (ndiags == 0) ? 5 : 6;
         const int *tetmap = hex_tetmaps[ndiags];
         for (int itet=0; itet<ntets; ++itet)
         {
            Element *e = NewElement(Geometry::TETRAHEDRON);
            e->SetAttribute(attrib);
            int *v2 = e->GetVertices();
            for (int iv=0; iv<nv_tet; ++iv)
            {
               v2[iv] = vg[tetmap[itet + iv*ntets]];
            }
            AddElement(e);
            parent_elems.Append(i);
         }
      }
   }
   // In 3D, shrink the element array because some hexes have only 5 tets
   if (dim == 3) { elements.SetSize(NumOfElements); }

   for (int i=0; i<nbe; ++i)
   {
      const int *v = orig_mesh.boundary[i]->GetVertices();
      const int attrib = orig_mesh.GetBdrAttribute(i);
      const Geometry::Type orig_geom = orig_mesh.GetBdrElementGeometry(i);
      if (num_subdivisions[orig_geom] == 1)
      {
         Element *be = NewElement(orig_geom);
         be->SetAttribute(attrib);
         be->SetVertices(v);
         AddBdrElement(be);
      }
      else if (orig_geom == Geometry::SQUARE)
      {
         int vg[nv_quad];
         for (int iv=0; iv<nv_quad; ++iv) { vg[iv] = vglobal[v[iv]]; }
         // Split quad according the smallest (global) vertex
         int iv_min = find_min(vg, nv_quad);
         int isplit = (iv_min == 0 || iv_min == 2) ? 0 : 1;
         for (int itri=0; itri<quad_ntris; ++itri)
         {
            Element *be = NewElement(Geometry::TRIANGLE);
            be->SetAttribute(attrib);
            int *v2 = be->GetVertices();
            for (int iv=0; iv<nv_tri; ++iv)
            {
               v2[iv] = v[quad_trimap[isplit][itri + iv*quad_ntris]];
            }
            AddBdrElement(be);
         }
      }
      else
      {
         MFEM_ABORT("Unreachable");
      }
   }

   FinalizeTopology(false);
   sequence = orig_mesh.GetSequence();
   last_operation = orig_mesh.last_operation;

   MFEM_ASSERT(CheckElementOrientation(false) == 0, "");
   MFEM_ASSERT(CheckBdrElementOrientation(false) == 0, "");

   return parent_elems;
}


void Mesh::MakeHigherOrderSimplicial_(const Mesh &orig_mesh,
                                      const Array<int> &parent_elements)
{
   // Higher order associated to vertices are unchanged, and those for
   // previously existing edges. DOFs associated to new elements need to be set.
   const int sdim = orig_mesh.SpaceDimension();
   auto *orig_fespace = orig_mesh.GetNodes()->FESpace();
   SetCurvature(orig_fespace->GetMaxElementOrder(), orig_fespace->IsDGSpace(),
                orig_mesh.SpaceDimension(), orig_fespace->GetOrdering());

   // The dofs associated with vertices are unchanged, but there can be new dofs
   // associated to edges, faces and volumes. Additionally, because we know that
   // the set of vertices is unchanged by the splitting operation, we can use
   // the vertices to map local coordinates of the "child" elements (the new
   // simplices introduced), from the "parent" element (the quad, prism, hex
   // that was split).

   // For segment, triangle and tetrahedron, the dof values are copied directly.
   // For the others, we have to construct a map from the Node locations in the
   // new simplex to the parent non-simplex element. This could be sped up by
   // not repeatedly access the original FE as the accesses will be coherent
   // (i.e. all child elems are consecutive).

   Array<int> edofs; // element dofs in new element
   Array<int> parent_vertices, child_vertices; // vertices of parent and child.
   Array<int> node_map; // node indices of parent from child.
   Vector edofvals; // values of elements dofs in original element
   // Storage for evaluating node function on parent element, at node locations
   // of child element
   DenseMatrix shape; // ndof_coarse x nnode_refined.
   DenseMatrix point_matrix; // sdim x nnode_refined
   IntegrationRule
   child_nodes_in_parent; // The parent nodes that correspond to the child nodes
   for (int i = 0; i < parent_elements.Size(); i++)
   {
      const int ip = parent_elements[i];
      const Geometry::Type orig_geom = orig_mesh.GetElementBaseGeometry(ip);
      orig_mesh.GetNodes()->GetElementDofValues(ip, edofvals);
      switch (orig_geom)
      {
         case Geometry::Type::SEGMENT : // fall through
         case Geometry::Type::TRIANGLE : // fall through
         case Geometry::Type::TETRAHEDRON :
            GetNodes()->FESpace()->GetElementVDofs(i, edofs);
            GetNodes()->SetSubVector(edofs, edofvals);
            break;
         case Geometry::Type::CUBE : // fall through
         case Geometry::Type::PRISM : // fall through
         case Geometry::Type::PYRAMID : // fall through
         case Geometry::Type::SQUARE :
         {
            // Extract the vertices of parent and child, can then form the
            // map from child reference coordinates to parent reference
            // coordinates. Exploit the fact that for Nodes, the vertex
            // entries come first, and their indexing matches the vertex
            // numbering. Thus we have already have an inverse index map.
            orig_mesh.GetElementVertices(ip, parent_vertices);
            GetElementVertices(i, child_vertices);
            node_map.SetSize(0);
            for (auto cv : child_vertices)
               for (int ipv = 0; ipv < parent_vertices.Size(); ipv++)
                  if (cv == parent_vertices[ipv])
                  {
                     node_map.Append(ipv);
                     break;
                  }
            MFEM_ASSERT(node_map.Size() == Geometry::NumVerts[GetElementBaseGeometry(i)],
                        "!");
            // node_map now says which of the parent vertex nodes map to each
            // of the child vertex nodes. Using this can build a basis in the
            // parent element from child Node values, exploit the linearity
            // to then transform all nodes.
            child_nodes_in_parent.SetSize(0);
            const auto *orig_FE = orig_mesh.GetNodes()->FESpace()->GetFE(ip);
            for (auto pn : node_map)
            {
               child_nodes_in_parent.Append(orig_FE->GetNodes()[pn]);
            }
            const auto *simplex_FE = GetNodes()->FESpace()->GetFE(i);
            shape.SetSize(orig_FE->GetDof(),
                          simplex_FE->GetDof()); // One set of evaluations per simplex dof.
            Vector col;
            for (int j = 0; j < simplex_FE->GetNodes().Size(); j++)
            {
               const auto &simplex_node = simplex_FE->GetNodes()[j];
               IntegrationPoint simplex_node_in_orig;
               // Handle the 2D vs 3D case by multiplying .z by zero.
               simplex_node_in_orig.Set3(
                  child_nodes_in_parent[0].x +
                  simplex_node.x * (child_nodes_in_parent[1].x - child_nodes_in_parent[0].x)
                  + simplex_node.y * (child_nodes_in_parent[2].x - child_nodes_in_parent[0].x)
                  + simplex_node.z * (child_nodes_in_parent[(Dim > 2) ? 3 : 0].x -
                                      child_nodes_in_parent[0].x),
                  child_nodes_in_parent[0].y +
                  simplex_node.x * (child_nodes_in_parent[1].y - child_nodes_in_parent[0].y)
                  + simplex_node.y * (child_nodes_in_parent[2].y - child_nodes_in_parent[0].y)
                  + simplex_node.z * (child_nodes_in_parent[(Dim > 2) ? 3 : 0].y -
                                      child_nodes_in_parent[0].y),
                  child_nodes_in_parent[0].z +
                  simplex_node.x * (child_nodes_in_parent[1].z - child_nodes_in_parent[0].z)
                  + simplex_node.y * (child_nodes_in_parent[2].z - child_nodes_in_parent[0].z)
                  + simplex_node.z * (child_nodes_in_parent[(Dim > 2) ? 3 : 0].z -
                                      child_nodes_in_parent[0].z));
               shape.GetColumnReference(j, col);
               orig_FE->CalcShape(simplex_node_in_orig, col);
            }
            // All the non-simplex basis functions have now been evaluated at
            // all the simplex basis function node locations. Now evaluate
            // the summations and place back into the Nodes vector.
            orig_mesh.GetNodes()->GetElementDofValues(ip, edofvals);
            // Dof values are always returned as
            // [[x_1,x_2,x_3,...],
            //  [y_1,y_2,y_3,...],
            //  [z_1,z_2,z_3,...]]
            DenseMatrix edofvals_mat(edofvals.GetData(), orig_FE->GetDof(), sdim);
            point_matrix.SetSize(simplex_FE->GetDof(), sdim);
            MultAtB(shape, edofvals_mat, point_matrix);
            GetNodes()->FESpace()->GetElementVDofs(i, edofs);
            GetNodes()->SetSubVector(edofs, point_matrix.GetData());
         }
         break;
         case Geometry::Type::POINT :  // fall through
         case Geometry::Type::INVALID :
         case Geometry::Type::NUM_GEOMETRIES :
            MFEM_ABORT("Internal Error!");
      }
   }
}


Mesh Mesh::MakePeriodic(const Mesh &orig_mesh, const std::vector<int> &v2v)
{
   Mesh periodic_mesh(orig_mesh, true); // Make a copy of the original mesh
   const FiniteElementSpace *nodal_fes = orig_mesh.GetNodalFESpace();
   int nodal_order = nodal_fes ? nodal_fes->GetMaxElementOrder() : 1;
   periodic_mesh.SetCurvature(nodal_order, true);

   // renumber element vertices
   for (int i = 0; i < periodic_mesh.GetNE(); i++)
   {
      Element *el = periodic_mesh.GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary element vertices
   for (int i = 0; i < periodic_mesh.GetNBE(); i++)
   {
      Element *el = periodic_mesh.GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   periodic_mesh.RemoveUnusedVertices();
   return periodic_mesh;
}

std::vector<int> Mesh::CreatePeriodicVertexMapping(
   const std::vector<Vector> &translations, real_t tol) const
{
   const int sdim = SpaceDimension();

   Vector coord(sdim), at(sdim), dx(sdim);
   Vector xMax(sdim), xMin(sdim), xDiff(sdim);
   xMax = xMin = xDiff = 0.0;

   // Get a list of all vertices on the boundary
   unordered_set<int> bdr_v;
   for (int be = 0; be < GetNBE(); be++)
   {
      Array<int> dofs;
      GetBdrElementVertices(be,dofs);

      for (int i = 0; i < dofs.Size(); i++)
      {
         bdr_v.insert(dofs[i]);

         coord = GetVertex(dofs[i]);
         for (int j = 0; j < sdim; j++)
         {
            xMax[j] = max(xMax[j], coord[j]);
            xMin[j] = min(xMin[j], coord[j]);
         }
      }
   }
   add(xMax, -1.0, xMin, xDiff);
   real_t dia = xDiff.Norml2(); // compute mesh diameter

   // We now identify coincident vertices. Several originally distinct vertices
   // may become coincident under the periodic mapping. One of these vertices
   // will be identified as the "primary" vertex, and all other coincident
   // vertices will be considered as "replicas".

   // replica2primary[v] is the index of the primary vertex of replica `v`
   unordered_map<int, int> replica2primary;
   // primary2replicas[v] is a set of indices of replicas of primary vertex `v`
   unordered_map<int, unordered_set<int>> primary2replicas;

   // Create a KD-tree containing all the boundary vertices
   std::unique_ptr<KDTreeBase<int,real_t>> kdtree;
   if (sdim == 1) { kdtree.reset(new KDTree1D); }
   else if (sdim == 2) { kdtree.reset(new KDTree2D); }
   else if (sdim == 3) { kdtree.reset(new KDTree3D); }
   else { MFEM_ABORT("Invalid space dimension."); }

   // We begin with the assumption that all vertices are primary, and that there
   // are no replicas.
   for (const int v : bdr_v)
   {
      primary2replicas[v];
      kdtree->AddPoint(GetVertex(v), v);
   }

   kdtree->Sort();

   // Make `r` and all of `r`'s replicas be replicas of `p`. Delete `r` from the
   // list of primary vertices.
   auto make_replica = [&replica2primary, &primary2replicas](int r, int p)
   {
      if (r == p) { return; }
      primary2replicas[p].insert(r);
      replica2primary[r] = p;
      for (const int s : primary2replicas[r])
      {
         primary2replicas[p].insert(s);
         replica2primary[s] = p;
      }
      primary2replicas.erase(r);
   };

   for (unsigned int i = 0; i < translations.size(); i++)
   {
      for (int vi : bdr_v)
      {
         coord = GetVertex(vi);
         add(coord, translations[i], at);

         const int vj = kdtree->FindClosestPoint(at.GetData());
         coord = GetVertex(vj);
         add(at, -1.0, coord, dx);

         if (dx.Norml2() > dia*tol) { continue; }

         // The two vertices vi and vj are coincident.

         // Are vertices `vi` and `vj` already primary?
         const bool pi = primary2replicas.find(vi) != primary2replicas.end();
         const bool pj = primary2replicas.find(vj) != primary2replicas.end();

         if (pi && pj)
         {
            // Both vertices are currently primary
            // Demote `vj` to be a replica of `vi`
            make_replica(vj, vi);
         }
         else if (pi && !pj)
         {
            // `vi` is primary and `vj` is a replica
            const int owner_of_vj = replica2primary[vj];
            // Make `vi` and its replicas be replicas of `vj`'s owner
            make_replica(vi, owner_of_vj);
         }
         else if (!pi && pj)
         {
            // `vi` is currently a replica and `vj` is currently primary
            // Make `vj` and its replicas be replicas of `vi`'s owner
            const int owner_of_vi = replica2primary[vi];
            make_replica(vj, owner_of_vi);
         }
         else
         {
            // Both vertices are currently replicas
            // Make `vj`'s owner and all of its owner's replicas be replicas
            // of `vi`'s owner
            const int owner_of_vi = replica2primary[vi];
            const int owner_of_vj = replica2primary[vj];
            make_replica(owner_of_vj, owner_of_vi);
         }
      }
   }

   std::vector<int> v2v(GetNV());
   for (size_t i = 0; i < v2v.size(); i++)
   {
      v2v[i] = static_cast<int>(i);
   }
   for (const auto &r2p : replica2primary)
   {
      v2v[r2p.first] = r2p.second;
   }
   return v2v;
}

void Mesh::RefineNURBSFromFile(std::string ref_file)
{
   MFEM_VERIFY(NURBSext,"Mesh::RefineNURBSFromFile: Not a NURBS mesh!");
   mfem::out<<"Refining NURBS from refinement file: "<<ref_file<<endl;

   int nkv;
   ifstream input(ref_file);
   input >> nkv;

   // Check if the number of knot vectors in the refinement file and mesh match
   if ( nkv != NURBSext->GetNKV())
   {
      mfem::out<<endl;
      mfem::out<<"Knot vectors in ref_file: "<<nkv<<endl;
      mfem::out<<"Knot vectors in NURBSExt: "<<NURBSext->GetNKV()<<endl;
      MFEM_ABORT("Refine file does not have the correct number of knot vectors");
   }

   // Read knot vectors from file
   Array<Vector *> knotVec(nkv);
   for (int kv = 0; kv < nkv; kv++)
   {
      knotVec[kv] = new Vector();
      knotVec[kv]-> Load(input);
   }
   input.close();

   // Insert knots
   KnotInsert(knotVec);

   // Delete knots
   for (int kv = 0; kv < nkv; kv++)
   {
      delete knotVec[kv];
   }
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

void Mesh::KnotInsert(Array<Vector *> &kv)
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

void Mesh::KnotRemove(Array<Vector *> &kv)
{
   if (NURBSext == NULL)
   {
      mfem_error("Mesh::KnotRemove : Not a NURBS mesh!");
   }

   if (kv.Size() != NURBSext->GetNKV())
   {
      mfem_error("Mesh::KnotRemove : KnotVector array size mismatch!");
   }

   NURBSext->ConvertToPatches(*Nodes);

   NURBSext->KnotRemove(kv);

   last_operation = Mesh::NONE; // FiniteElementSpace::Update is not supported
   sequence++;

   UpdateNURBS();
}

void Mesh::RefineNURBSWithKVFactors(int rf, const std::string &kvf)
{
   RefineNURBS(true, 0.0, Array<int>(&rf, 1), kvf);
}

void Mesh::NURBSUniformRefinement(int rf, real_t tol)
{
   Array<int> rf_array(Dim);
   rf_array = rf;
   NURBSUniformRefinement(rf_array, tol);
}

void Mesh::NURBSUniformRefinement(Array<int> const& rf, real_t tol)
{
   MFEM_VERIFY(rf.Size() == Dim,
               "Refinement factors must be defined for each dimension");

   RefineNURBS(false, tol, rf, "");
}

void Mesh::RefineNURBS(bool usingKVF, real_t tol, const Array<int> &rf,
                       const std::string &kvf)
{
   MFEM_VERIFY(NURBSext, "This type of refinement is only for NURBS meshes");
   NURBSext->ConvertToPatches(*Nodes);

   Array<int> cf;
   NURBSext->GetCoarseningFactors(cf);

   bool cf1 = true;
   for (auto f : cf)
   {
      cf1 = (cf1 && f == 1);
   }

   if (!cf1 && NURBSext->NonconformingPatches())
   {
      NURBSext->FullyCoarsen();
      last_operation = Mesh::NONE; // FiniteElementSpace::Update is not supported
   }
   else if (!cf1 && !NURBSext->NonconformingPatches())
   {
      MFEM_VERIFY(!usingKVF, "This refinement type is not supported for this"
                  " NURBS mesh type");
      NURBSext->Coarsen(cf, tol);

      last_operation = Mesh::NONE; // FiniteElementSpace::Update is not supported
      sequence++;
      UpdateNURBS();

      NURBSext->ConvertToPatches(*Nodes);
      for (int i=0; i<cf.Size(); ++i) { cf[i] *= rf[i]; }
      NURBSext->UniformRefinement(cf);
   }

   if (cf1 || NURBSext->NonconformingPatches())
   {
      if (usingKVF || NURBSext->NonconformingPatches())
      {
         NURBSext->RefineWithKVFactors(rf[0], kvf, !cf1);
      }
      else
      {
         NURBSext->UniformRefinement(rf);
      }
   }

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
   ResetLazyData();

   NURBSext->SetKnotsFromPatches();

   Dim = NURBSext->Dimension();
   spaceDim = Nodes->FESpace()->GetVDim();

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
   NodesUpdated();
   const int vdim = Nodes->FESpace()->GetVDim();
   NURBSext->SetCoordsFromPatches(*Nodes, vdim);

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
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
   }

   if (el_to_face)
   {
      GetElementToFaceTable();
   }
   GenerateFaces();
}

void Mesh::LoadPatchTopo(std::istream &input, Array<int> &edge_to_ukv)
{
   SetEmpty();

   // Read MFEM NURBS mesh v1.0 or 1.1 format
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
   if (NumOfEdges > 0)
   {
      edge_vertex = new Table(NumOfEdges, 2);
      edge_to_ukv.SetSize(NumOfEdges);
      for (int j = 0; j < NumOfEdges; j++)
      {
         int *v = edge_vertex->GetRow(j);
         input >> edge_to_ukv[j] >> v[0] >> v[1];
         if (v[0] > v[1])
         {
            edge_to_ukv[j] = -1 - edge_to_ukv[j];
         }
      }
   }
   else
   {
      edge_to_ukv.SetSize(0);
   }

   skip_comment_lines(input, '#');

   input >> ident; // 'vertices'
   input >> NumOfVertices;
   vertices.SetSize(0);

   FinalizeTopology();
   CheckBdrElementOrientation(); // check and fix boundary element orientation

   /* Generate edge to knotvector mapping if edges are not specified in the
      mesh file. See miniapps/nurbs/meshes/two-squares-nurbs-autoedge.mesh
      for an example */
   if (edge_to_ukv.Size() == 0)
   {
      Array<int> ukv_to_rpkv;
      GetEdgeToUniqueKnotvector(edge_to_ukv, ukv_to_rpkv);
   }

   CorrectPatchTopoOrientations(edge_to_ukv);
}

void Mesh::GetEdgeToUniqueKnotvector(Array<int> &edge_to_ukv,
                                     Array<int> &ukv_to_rpkv) const
{
   const int dim = Dimension();  // topological (not physical) dimension
   const int NP = NumOfElements; // number of patches
   const int NPKV = NP * dim;    // number of patch knotvectors
   constexpr int notset = -9999999;
   // Sign convention
   auto flipSign = [](int i) { return -1 - i; };
   auto unSign = [](int i) { return (i < 0) ? -1 - i : i; };
   // Local edge index -> dimension convention
   auto edge_to_dim = [](int i) { return (i < 8) ? ((i & 1) ? 1 : 0) : 2; };

   Array<int> v(2); // vertices of an edge

   // 1D case is special: edge index = signed element index
   // ukv_to_rpkv = Identity
   if (dim == 1)
   {
      edge_to_ukv.SetSize(NP);
      ukv_to_rpkv.SetSize(NP);
      for (int i = 0; i < NP; i++)
      {
         GetElementVertices(i, v);
         // Sign is based on the edge's vertex indices
         edge_to_ukv[i] = (v[1] > v[0]) ? i : flipSign(i);
         ukv_to_rpkv[i] = i;
      }
      return;
   }

   // Local (per-patch) variables
   Array<int> edges, oedges;
   // Edge index -> signed patch knotvector index (p*dim + d)
   Array<int> edge_to_pkv(NumOfEdges);
   edge_to_pkv.SetSize(NumOfEdges);
   edge_to_pkv = notset;

   // Initialize pkv_map as identity - this is the storage for the
   // disjoint-set/union-find algorithm which will later be used
   // to get the map pkv_to_rpkv
   Array<int> pkv_map(NPKV);
   for (int i = 0; i < NPKV; i++)
   {
      pkv_map[i] = i;
   }
   std::function<int(int)> get_root;
   get_root = [&pkv_map, &get_root](int i) -> int
   {
      return (pkv_map[i] == i) ? i : get_root(pkv_map[i]);
   };
   auto unite = [&pkv_map, &get_root](int i, int j)
   {
      const int ri = get_root(i);
      const int rj = get_root(j);
      if (ri == rj) { return; }
      // keep the lowest index
      (ri < rj) ? pkv_map[rj] = ri : pkv_map[ri] = rj;
   };

   // Get edge_to_pkv (one edge can link to multiple pkv) and pkv_map
   for (int p = 0; p < NP; p++)
   {
      GetElementEdges(p, edges, oedges);

      // First loop checks for if edge has already been set
      for (int i = 0; i < edges.Size(); i++)
      {
         const int edge = edges[i];
         const int d = edge_to_dim(i);
         const int pkv = p*dim+d;

         // We've set this edge already - link this index to it
         if (edge_to_pkv[edge] != notset)
         {
            const int pkv_other = unSign(edge_to_pkv[edge]);
            unite(pkv, pkv_other);
         }
         else
         {
            GetEdgeVertices(edge, v);
            // Sign is based on the edge's vertex indices
            edge_to_pkv[edge] = (v[1] > v[0]) ? pkv : flipSign(pkv);
         }
      }
   }

   // Construct the pkv_to_rpkv map by finding the lowest/root index
   Array<int> pkv_to_rpkv(NPKV);
   ukv_to_rpkv.SetSize(NPKV);
   for (int i = 0; i < NPKV; i++)
   {
      pkv_to_rpkv[i] = get_root(pkv_map[i]);
      ukv_to_rpkv[i] = pkv_to_rpkv[i];
   }
   ukv_to_rpkv.Sort(); // ukv is just a renumbering of rpkv
   ukv_to_rpkv.Unique();

   // Create inverse map
   std::map<int, int> rpkv_to_ukv;
   for (int i = 0; i < ukv_to_rpkv.Size(); i++)
   {
      rpkv_to_ukv[ukv_to_rpkv[i]] = i;
   }

   // Get edge_to_ukv = edge_to_pkv -> pkv_to_rpkv -> rpkv_to_ukv
   edge_to_ukv.SetSize(NumOfEdges);
   for (int i = 0; i < NumOfEdges; i++)
   {
      const int pkv = unSign(edge_to_pkv[i]);
      const int rpkv = pkv_to_rpkv[pkv];
      const int ukv = rpkv_to_ukv[rpkv];
      edge_to_ukv[i] = (edge_to_pkv[i] < 0) ? flipSign(ukv) : ukv;
   }

   CorrectPatchTopoOrientations(edge_to_ukv);
}

void Mesh::CorrectPatchTopoOrientations(Array<int> &edge_to_ukv) const
{
   const int dim = Dimension(); // Topological (not physical) dimension
   if (dim == 1) { return; }

   // Sign convention
   auto flipSign = [](int i) { return -1 - i; };

   const Table *face2elem = GetFaceToElementTable();
   Array<int> pfaces, orient;
   Array<int> fe, feo;

   // Finds elements sharing a face containing knotvector kv.
   auto faceNeighbors = [&](int p, int kv, std::unordered_set<int> &nghb)
   {
      if (dim == 2) { GetElementEdges(p, pfaces, orient); }
      else { GetElementFaces(p, pfaces, orient); }

      for (auto face : pfaces)
      {
         // Check whether this face contains kv.
         GetFaceEdges(face, fe, feo);
         bool hasKV = false;
         for (auto e : fe)
         {
            const int skv = edge_to_ukv[e];
            if (skv == kv || flipSign(skv) == kv) { hasKV = true; }
         }
         if (hasKV)
         {
            Array<int> row;
            face2elem->GetRow(face, row);
            for (auto elem : row) { nghb.insert(elem); }
         }
      }
   };

   std::vector<std::vector<int>> dir_edges;
   if (dim == 2)
   {
      dir_edges =
      {
         {0,2},
         {1,3}
      };
   }
   else
   {
      dir_edges =
      {
         {0,2,4,6},
         {1,3,5,7},
         {8,9,10,11}
      };
   }

   Array<int> ukvs((dim==2) ? 4 : 12);
   Array<int> pe, oe;
   bool initKV = false;

   auto setPatchDirections = [&](int p, int kv, Array<bool> &edgeSet,
                                 std::unordered_set<int> &visited)
   {
      // Edges and orientations for this patch
      GetElementEdges(p, pe, oe);

      // Get the signed unique knot vector indices
      for (int i = 0; i < pe.Size(); i++)
      {
         ukvs[i] = edge_to_ukv[pe[i]];
         ukvs[i] = (oe[i] < 0) ? flipSign(ukvs[i]) : ukvs[i];
      }

      // Find the direction with this kv.
      int thisDir = -1;
      for (int d=0; d<dim; ++d) // Loop over directions.
      {
         const int skv = edge_to_ukv[pe[dir_edges[d][0]]];
         if (skv == kv || flipSign(skv) == kv)
         {
            thisDir = d;
         }
      }
      MFEM_VERIFY(thisDir >= 0, "");

      // For this direction, find any edge already set. If no edge is set, we
      // arbitrarily take the first.
      int ref_edge0 = dir_edges[thisDir][0];
      for (auto ref_edge : dir_edges[thisDir])
      {
         const int edge = pe[ref_edge];
         if (edgeSet[edge])
         {
            ref_edge0 = ref_edge;
         }
      }

      if (initKV && !edgeSet[pe[ref_edge0]])
      {
         visited.erase(p);
         return false; // There is no set edge in this direction on this patch.
      }

      initKV = true;

      // Use ref_edge0 to set other edges in this direction.
      edgeSet[pe[ref_edge0]] = true;
      for (auto i : dir_edges[thisDir])
      {
         if (i == ref_edge0)
         {
            continue;
         }

         const int edge = pe[i];
         if ((dim == 2 && ukvs[i] != flipSign(ukvs[ref_edge0])) ||
             (dim == 3 && ukvs[i] == flipSign(ukvs[ref_edge0])))
         {
            // Flip the sign of this edge
            MFEM_VERIFY(!edgeSet[edge], "");
            edge_to_ukv[edge] = flipSign(edge_to_ukv[edge]);
         }

         edgeSet[edge] = true;
      }

      return true;
   };

   Array<bool> edgeSet(NumOfEdges); // Whether edge has orientation set
   edgeSet = false;

   std::unordered_set<int> unset; // Patches with an unset edge
   for (int i=0; i<NumOfElements; ++i) { unset.insert(i); }

   const int max_iter = 3 * NumOfElements;
   for (int iter=0; iter<max_iter; ++iter)
   {
      // Iteratively choose an unset patch (meaning not all edges have
      // orientation set), choose a knotvector index for which the corresponding
      // edges on this patch are not set, and sweep over all patches containing
      // this knotvector. The patch sweep is ordered, by maintaining an ordered
      // list `nextPatches` set by finding face-neighbor patches of visited
      // patches, where the common face contains the knotvector. When each patch
      // is visited, the edge orientations are set consistently. This iteration
      // terminates when all edges have been set on all patches.

      std::list<int> nextPatches; // Next patches to visit, ordered
      std::unordered_set<int> nextSet; // nextPatches as a set
      std::unordered_set<int> visited; // Visit each patch only once

      if (unset.size() == 0)
      {
         break;
      }

      const int p0 = *unset.begin();
      nextPatches.push_back(p0); // Start from arbitrary unset patch
      nextSet.insert(p0);

      // Choose an arbitrary unset direction for the first patch.
      GetElementEdges(p0, pe, oe);
      int unsetDim = -1;
      for (int d=0; d<dim; ++d) // Loop over dimensions.
      {
         if (!edgeSet[pe[dir_edges[d][0]]])
         {
            unsetDim = d;
         }
      }

      if (unsetDim == -1)
      {
         unset.erase(p0);
         continue;
      }

      const int kv_signed = edge_to_ukv[pe[dir_edges[unsetDim][0]]];
      const int kv = kv_signed < 0 ? flipSign(kv_signed) : kv_signed;
      MFEM_VERIFY(!edgeSet[pe[dir_edges[unsetDim][0]]], "");

      initKV = false;

      while (nextPatches.size() > 0)
      {
         const int p = nextPatches.front();
         nextPatches.pop_front();
         nextSet.erase(p);
         visited.insert(p);

         const bool somethingSet = setPatchDirections(p, kv, edgeSet, visited);
         if (!somethingSet)
         {
            continue;
         }

         // Find neighbors of patch p sharing a conforming face, via face2elem.
         std::unordered_set<int> neighbors;
         faceNeighbors(p, kv, neighbors);

         bool allSet = true;
         GetElementEdges(p, pe, oe);
         for (auto edge : pe)
         {
            if (!edgeSet[edge])
            {
               allSet = false;
            }
         }
         if (allSet)
         {
            unset.erase(p);
         }

         // Add neighbors not done to nextPatches.
         for (auto n : neighbors)
         {
            if (n != p && visited.count(n) == 0 && unset.count(n) > 0)
            {
               if (nextSet.count(n) == 0)
               {
                  nextPatches.push_back(n);
                  nextSet.insert(n);
               }
            }
         }
      }
   }

   bool allSet = true;
   for (auto eset : edgeSet)
   {
      if (!eset)
      {
         allSet = false;
      }
   }
   MFEM_VERIFY(allSet && unset.size() == 0, "Some edge is not set");

   delete face2elem;
}

void Mesh::LoadNonconformingPatchTopo(std::istream &input,
                                      Array<int> &edge_to_ukv)
{
   SetEmpty();

   // Read MFEM NURBS NC-patch mesh v1.0 format
   int curved = 0;
   int is_nc = 1;

   ncmesh = new NCMesh(input, 10, curved, is_nc);

   InitFromNCMesh(*ncmesh);

   skip_comment_lines(input, '#');

   string ident;
   int inputNumOfEdges = -1;

   input >> ident; // 'edges'
   input >> inputNumOfEdges;

   MFEM_VERIFY(NumOfEdges == inputNumOfEdges, "");

   edge_to_ukv.SetSize(NumOfEdges);
   for (int j = 0; j < NumOfEdges; j++)
   {
      int v[2]; // Vertex indices
      int ukv; // Unique KnotVector index
      input >> ukv >> v[0] >> v[1];

      for (int i=0; i<2; ++i)
      {
         v[i] = ncmesh->vertex_nodeId[v[i]];
      }

      if (v[0] > v[1])
      {
         ukv = -1 - ukv;
      }
      edge_to_ukv[j] = ukv;
   }

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

void Mesh::EnsureNodes()
{
   if (Nodes)
   {
      const FiniteElementCollection *fec = GetNodalFESpace()->FEColl();
      if (dynamic_cast<const H1_FECollection*>(fec)
          || dynamic_cast<const L2_FECollection*>(fec))
      {
         return;
      }
      else // Mesh using a legacy FE_Collection
      {
         const int order = GetNodalFESpace()->GetMaxElementOrder();
         if (NURBSext)
         {
#ifndef MFEM_USE_MPI
            const bool warn = true;
#else
            ParMesh *pmesh = dynamic_cast<ParMesh*>(this);
            const bool warn = !pmesh || pmesh->GetMyRank() == 0;
#endif
            if (warn)
            {
               MFEM_WARNING("converting NURBS mesh to order " << order <<
                            " H1-continuous mesh!\n   "
                            "If this is the desired behavior, you can silence"
                            " this warning by converting\n   "
                            "the NURBS mesh to high-order mesh in advance by"
                            " calling the method\n   "
                            "Mesh::SetCurvature().");
            }
         }
         SetCurvature(order, false, -1, Ordering::byVDIM);
      }
   }
   else // First order H1 mesh
   {
      SetCurvature(1, false, -1, Ordering::byVDIM);
   }
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
   if (order <= 0)
   {
      delete Nodes;
      Nodes = nullptr;
      return;
   }
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

   const int old_space_dim = spaceDim;
   SetNodalFESpace(nfes);
   Nodes->MakeOwner(nfec);

   if (spaceDim != old_space_dim)
   {
      // Fix dimension of the vertices if the space dimension changes
      SetVerticesFromNodes(Nodes);
   }
}

void Mesh::SetVerticesFromNodes(const GridFunction *nodes)
{
   MFEM_ASSERT(nodes != NULL, "");
   for (int i = 0; i < spaceDim; i++)
   {
      Vector vert_val;
      nodes->GetNodalValues(vert_val, i+1);
      for (int j = 0; j < NumOfVertices; j++)
      {
         vertices[j](i) = vert_val(j);
      }
   }
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

int Mesh::GetNumFacesWithGhost() const
{
   return faces_info.Size();
}

int Mesh::GetNFbyType(FaceType type) const
{
   const bool isInt = type==FaceType::Interior;
   int &nf = isInt ? nbInteriorFaces : nbBoundaryFaces;
   if (nf<0)
   {
      nf = 0;
      for (int f = 0; f < GetNumFacesWithGhost(); ++f)
      {
         FaceInformation face = GetFaceInformation(f);
         if (face.IsOfFaceType(type))
         {
            if (face.IsNonconformingCoarse())
            {
               // We don't count nonconforming coarse faces.
               continue;
            }
            nf++;
         }
      }
   }
   return nf;
}

#if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
static const char *fixed_or_not[] = { "fixed", "NOT FIXED" };
#endif

int Mesh::CheckElementOrientation(bool fix_it)
{
   int i, j, k, wo = 0, fo = 0;
   real_t *v[4];

   if (Dim == 2 && spaceDim == 2)
   {
      DenseMatrix J(2, 2);

      for (i = 0; i < NumOfElements; i++)
      {
         int *vi = elements[i]->GetVertices();
         if (Nodes == NULL)
         {
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
         int *vi = elements[i]->GetVertices();
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

            case Element::PYRAMID:
               // only check the Jacobian at the center of the element
               GetElementJacobian(i, J);
               if (J.Det() < 0.0)
               {
                  wo++;
                  if (fix_it)
                  {
                     mfem::Swap(vi[1], vi[3]);
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

            default:
               MFEM_ABORT("Invalid 3D element type \""
                          << GetElementType(i) << "\"");
               break;
         }
      }
   }
#if (!defined(MFEM_USE_MPI) || defined(MFEM_DEBUG))
   if (wo > 0)
   {
      mfem::out << "Elements with wrong orientation: " << wo << " / "
                << NumOfElements << " (" << fixed_or_not[(wo == fo) ? 0 : 1]
                << ")" << endl;
   }
#else
   MFEM_CONTRACT_VAR(fo);
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
         mfem::err << "Mesh::GetTriOrientation(...)" << endl;
         mfem::err << " base = [";
         for (int k = 0; k < 3; k++)
         {
            mfem::err << " " << base[k];
         }
         mfem::err << " ]\n test = [";
         for (int k = 0; k < 3; k++)
         {
            mfem::err << " " << test[k];
         }
         mfem::err << " ]" << endl;
         mfem_error();
      }
#endif

   return orient;
}

int Mesh::ComposeTriOrientations(int ori_a_b, int ori_b_c)
{
   // Static method.
   // Given three, possibly different, configurations of triangular face
   // vertices: va, vb, and vc.  This function returns the relative orientation
   // GetTriOrientation(va, vc) by composing previously computed orientations
   // ori_a_b = GetTriOrientation(va, vb) and
   // ori_b_c = GetTriOrientation(vb, vc) without accessing the vertices.

   const int oo[6][6] =
   {
      {0, 1, 2, 3, 4, 5},
      {1, 0, 5, 4, 3, 2},
      {2, 3, 4, 5, 0, 1},
      {3, 2, 1, 0, 5, 4},
      {4, 5, 0, 1, 2, 3},
      {5, 4, 3, 2, 1, 0}
   };

   int ori_a_c = oo[ori_a_b][ori_b_c];
   return ori_a_c;
}

int Mesh::InvertTriOrientation(int ori)
{
   const int inv_ori[6] = {0, 1, 4, 3, 2, 5};
   return inv_ori[ori];
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

int Mesh::ComposeQuadOrientations(int ori_a_b, int ori_b_c)
{
   // Static method.
   // Given three, possibly different, configurations of quadrilateral face
   // vertices: va, vb, and vc.  This function returns the relative orientation
   // GetQuadOrientation(va, vc) by composing previously computed orientations
   // ori_a_b = GetQuadOrientation(va, vb) and
   // ori_b_c = GetQuadOrientation(vb, vc) without accessing the vertices.

   const int oo[8][8] =
   {
      {0, 1, 2, 3, 4, 5, 6, 7},
      {1, 0, 3, 2, 5, 4, 7, 6},
      {2, 7, 4, 1, 6, 3, 0, 5},
      {3, 6, 5, 0, 7, 2, 1, 4},
      {4, 5, 6, 7, 0, 1, 2, 3},
      {5, 4, 7, 6, 1, 0, 3, 2},
      {6, 3, 0, 5, 2, 7, 4, 1},
      {7, 2, 1, 4, 3, 6, 5, 0}
   };

   int ori_a_c = oo[ori_a_b][ori_b_c];
   return ori_a_c;
}

int Mesh::InvertQuadOrientation(int ori)
{
   const int inv_ori[8] = {0, 1, 6, 3, 4, 5, 2, 7};
   return inv_ori[ori];
}

int Mesh::GetTetOrientation(const int *base, const int *test)
{
   // Static method.
   // This function computes the index 'j' of the permutation that transforms
   // test into base: test[tet_orientation[j][i]]=base[i].
   // tet_orientation = Geometry::Constants<Geometry::TETRAHEDRON>::Orient
   int orient;

   if (test[0] == base[0])
      if (test[1] == base[1])
         if (test[2] == base[2])
         {
            orient = 0;   //  (0, 1, 2, 3)
         }
         else
         {
            orient = 1;   //  (0, 1, 3, 2)
         }
      else if (test[2] == base[1])
         if (test[3] == base[2])
         {
            orient = 2;   //  (0, 2, 3, 1)
         }
         else
         {
            orient = 3;   //  (0, 2, 1, 3)
         }
      else // test[3] == base[1]
         if (test[1] == base[2])
         {
            orient = 4;   //  (0, 3, 1, 2)
         }
         else
         {
            orient = 5;   //  (0, 3, 2, 1)
         }
   else if (test[1] == base[0])
      if (test[2] == base[1])
         if (test[0] == base[2])
         {
            orient = 6;   //  (1, 2, 0, 3)
         }
         else
         {
            orient = 7;   //  (1, 2, 3, 0)
         }
      else if (test[3] == base[1])
         if (test[2] == base[2])
         {
            orient = 8;   //  (1, 3, 2, 0)
         }
         else
         {
            orient = 9;   //  (1, 3, 0, 2)
         }
      else // test[0] == base[1]
         if (test[3] == base[2])
         {
            orient = 10;   //  (1, 0, 3, 2)
         }
         else
         {
            orient = 11;   //  (1, 0, 2, 3)
         }
   else if (test[2] == base[0])
      if (test[3] == base[1])
         if (test[0] == base[2])
         {
            orient = 12;   //  (2, 3, 0, 1)
         }
         else
         {
            orient = 13;   //  (2, 3, 1, 0)
         }
      else if (test[0] == base[1])
         if (test[1] == base[2])
         {
            orient = 14;   //  (2, 0, 1, 3)
         }
         else
         {
            orient = 15;   //  (2, 0, 3, 1)
         }
      else // test[1] == base[1]
         if (test[3] == base[2])
         {
            orient = 16;   //  (2, 1, 3, 0)
         }
         else
         {
            orient = 17;   //  (2, 1, 0, 3)
         }
   else // (test[3] == base[0])
      if (test[0] == base[1])
         if (test[2] == base[2])
         {
            orient = 18;   //  (3, 0, 2, 1)
         }
         else
         {
            orient = 19;   //  (3, 0, 1, 2)
         }
      else if (test[1] == base[1])
         if (test[0] == base[2])
         {
            orient = 20;   //  (3, 1, 0, 2)
         }
         else
         {
            orient = 21;   //  (3, 1, 2, 0)
         }
      else // test[2] == base[1]
         if (test[1] == base[2])
         {
            orient = 22;   //  (3, 2, 1, 0)
         }
         else
         {
            orient = 23;   //  (3, 2, 0, 1)
         }

#ifdef MFEM_DEBUG
   const int *aor = tet_t::Orient[orient];
   for (int j = 0; j < 4; j++)
      if (test[aor[j]] != base[j])
      {
         mfem_error("Mesh::GetTetOrientation(...)");
      }
#endif

   return orient;
}

int Mesh::CheckBdrElementOrientation(bool fix_it)
{
   int wo = 0; // count wrong orientations

   if (Dim == 2)
   {
      if (el_to_edge == NULL) // edges were not generated
      {
         el_to_edge = new Table;
         NumOfEdges = GetElementToEdgeTable(*el_to_edge);
         GenerateFaces(); // 'Faces' in 2D refers to the edges
      }
      for (int i = 0; i < NumOfBdrElements; i++)
      {
         if (faces_info[be_to_face[i]].Elem2No < 0) // boundary face
         {
            int *bv = boundary[i]->GetVertices();
            int *fv = faces[be_to_face[i]]->GetVertices();
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
               mfem::Swap(bv[0], bv[1]);
               if (bel_to_edge)
               {
                  int *be = bel_to_edge->GetRow(i);
                  mfem::Swap(be[1], be[2]);
               }
               break;
            }
            case Element::QUADRILATERAL:
            {
               mfem::Swap(bv[0], bv[2]);
               if (bel_to_edge)
               {
                  int *be = bel_to_edge->GetRow(i);
                  mfem::Swap(be[0], be[1]);
                  mfem::Swap(be[2], be[3]);
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

IntegrationPoint Mesh::TransformBdrElementToFace(Geometry::Type geom, int o,
                                                 const IntegrationPoint &ip)
{
   IntegrationPoint fip = ip;
   if (geom == Geometry::POINT)
   {
      return fip;
   }
   else if (geom == Geometry::SEGMENT)
   {
      MFEM_ASSERT(o >= 0 && o < 2, "Invalid orientation for Geometry::SEGMENT!");
      if (o == 0)
      {
         fip.x = ip.x;
      }
      else if (o == 1)
      {
         fip.x = 1.0 - ip.x;
      }
   }
   else if (geom == Geometry::TRIANGLE)
   {
      MFEM_ASSERT(o >= 0 && o < 6, "Invalid orientation for Geometry::TRIANGLE!");
      if (o == 0)  // 0, 1, 2
      {
         fip.x = ip.x;
         fip.y = ip.y;
      }
      else if (o == 5)  // 0, 2, 1
      {
         fip.x = ip.y;
         fip.y = ip.x;
      }
      else if (o == 2)  // 1, 2, 0
      {
         fip.x = 1.0 - ip.x - ip.y;
         fip.y = ip.x;
      }
      else if (o == 1)  // 1, 0, 2
      {
         fip.x = 1.0 - ip.x - ip.y;
         fip.y = ip.y;
      }
      else if (o == 4)  // 2, 0, 1
      {
         fip.x = ip.y;
         fip.y = 1.0 - ip.x - ip.y;
      }
      else if (o == 3)  // 2, 1, 0
      {
         fip.x = ip.x;
         fip.y = 1.0 - ip.x - ip.y;
      }
   }
   else if (geom == Geometry::SQUARE)
   {
      MFEM_ASSERT(o >= 0 && o < 8, "Invalid orientation for Geometry::SQUARE!");
      if (o == 0)  // 0, 1, 2, 3
      {
         fip.x = ip.x;
         fip.y = ip.y;
      }
      else if (o == 1)  // 0, 3, 2, 1
      {
         fip.x = ip.y;
         fip.y = ip.x;
      }
      else if (o == 2)  // 1, 2, 3, 0
      {
         fip.x = ip.y;
         fip.y = 1.0 - ip.x;
      }
      else if (o == 3)  // 1, 0, 3, 2
      {
         fip.x = 1.0 - ip.x;
         fip.y = ip.y;
      }
      else if (o == 4)  // 2, 3, 0, 1
      {
         fip.x = 1.0 - ip.x;
         fip.y = 1.0 - ip.y;
      }
      else if (o == 5)  // 2, 1, 0, 3
      {
         fip.x = 1.0 - ip.y;
         fip.y = 1.0 - ip.x;
      }
      else if (o == 6)  // 3, 0, 1, 2
      {
         fip.x = 1.0 - ip.y;
         fip.y = ip.x;
      }
      else if (o == 7)  // 3, 2, 1, 0
      {
         fip.x = ip.x;
         fip.y = 1.0 - ip.y;
      }
   }
   else
   {
      MFEM_ABORT("Unsupported face geometry for TransformBdrElementToFace!");
   }
   return fip;
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

bool Mesh::IsMixedMesh() const
{
   // Return true if meshgen has more than one bit set, zero otherwise
   return meshgen & (meshgen - 1);
}

void Mesh::GetElementEdges(int i, Array<int> &edges, Array<int> &cor) const
{
   if (Dim == 1)
   {
      // In 1D, elements are segments and can be treated as edges.
      edges.SetSize(1);
      cor.SetSize(1);
      edges[0] = i;
      const int *v = elements[i]->GetVertices();
      cor[0] = (v[0] < v[1]) ? (1) : (-1);
      return;
   }

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
      edges[0] = be_to_face[i];
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
   Table *vert_elem = new Table;

   vert_elem->MakeI(NumOfVertices);

   for (int i = 0; i < NumOfElements; i++)
   {
      const int nv = elements[i]->GetNVertices();
      const int *v = elements[i]->GetVertices();
      for (int j = 0; j < nv; j++)
      {
         vert_elem->AddAColumnInRow(v[j]);
      }
   }

   vert_elem->MakeJ();

   for (int i = 0; i < NumOfElements; i++)
   {
      const int nv = elements[i]->GetNVertices();
      const int *v = elements[i]->GetVertices();
      for (int j = 0; j < nv; j++)
      {
         vert_elem->AddConnection(v[j], i);
      }
   }

   vert_elem->ShiftUpI();

   return vert_elem;
}

Table *Mesh::GetVertexToBdrElementTable()
{
   Table *vert_bdr_elem = new Table;

   vert_bdr_elem->MakeI(NumOfVertices);

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      const int nv = boundary[i]->GetNVertices();
      const int *v = boundary[i]->GetVertices();
      for (int j = 0; j < nv; j++)
      {
         vert_bdr_elem->AddAColumnInRow(v[j]);
      }
   }

   vert_bdr_elem->MakeJ();

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      const int nv = boundary[i]->GetNVertices();
      const int *v = boundary[i]->GetVertices();
      for (int j = 0; j < nv; j++)
      {
         vert_bdr_elem->AddConnection(v[j], i);
      }
   }

   vert_bdr_elem->ShiftUpI();

   return vert_bdr_elem;
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

void Mesh::GetElementFaces(int i, Array<int> &el_faces, Array<int> &ori) const
{
   MFEM_VERIFY(el_to_face != NULL, "el_to_face not generated");

   el_to_face->GetRow(i, el_faces);

   int n = el_faces.Size();
   ori.SetSize(n);

   for (int j = 0; j < n; j++)
   {
      if (faces_info[el_faces[j]].Elem1No == i)
      {
         ori[j] = faces_info[el_faces[j]].Elem1Inf % 64;
      }
      else
      {
         MFEM_ASSERT(faces_info[el_faces[j]].Elem2No == i, "internal error");
         ori[j] = faces_info[el_faces[j]].Elem2Inf % 64;
      }
   }
}

Array<int> Mesh::FindFaceNeighbors(const int elem) const
{
   if (face_to_elem == NULL)
   {
      face_to_elem = GetFaceToElementTable();
   }

   Array<int> elem_faces;
   Array<int> ori;
   GetElementFaces(elem, elem_faces, ori);

   Array<int> nghb;
   for (auto f : elem_faces)
   {
      Array<int> row;
      face_to_elem->GetRow(f, row);
      for (auto r : row)
      {
         nghb.Append(r);
      }
   }

   nghb.Sort();
   nghb.Unique();

   return nghb;
}

void Mesh::GetBdrElementFace(int i, int *f, int *o) const
{
   *f = GetBdrElementFaceIndex(i);

   const int *fv = (Dim > 1) ? faces[*f]->GetVertices() : NULL;
   const int *bv = boundary[i]->GetVertices();

   // find the orientation of the bdr. elem. w.r.t.
   // the corresponding face element (that's the base)
   switch (GetBdrElementGeometry(i))
   {
      case Geometry::POINT:    *o = 0; break;
      case Geometry::SEGMENT:  *o = (fv[0] == bv[0]) ? 0 : 1; break;
      case Geometry::TRIANGLE: *o = GetTriOrientation(fv, bv); break;
      case Geometry::SQUARE:   *o = GetQuadOrientation(fv, bv); break;
      default: MFEM_ABORT("invalid geometry");
   }
}

void Mesh::GetBdrElementAdjacentElement(int bdr_el, int &el, int &info) const
{
   int fid = GetBdrElementFaceIndex(bdr_el);

   const FaceInfo &fi = faces_info[fid];
   MFEM_ASSERT(fi.Elem1Inf % 64 == 0, "internal error"); // orientation == 0

   const int *fv = (Dim > 1) ? faces[fid]->GetVertices() : NULL;
   const int *bv = boundary[bdr_el]->GetVertices();
   int ori;
   switch (GetBdrElementGeometry(bdr_el))
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

void Mesh::GetBdrElementAdjacentElement2(
   int bdr_el, int &el, int &info) const
{
   int fid = GetBdrElementFaceIndex(bdr_el);

   const FaceInfo &fi = faces_info[fid];
   MFEM_ASSERT(fi.Elem1Inf % 64 == 0, "internal error"); // orientation == 0

   const int *fv = (Dim > 1) ? faces[fid]->GetVertices() : NULL;
   const int *bv = boundary[bdr_el]->GetVertices();
   int ori;
   switch (GetBdrElementGeometry(bdr_el))
   {
      case Geometry::POINT:    ori = 0; break;
      case Geometry::SEGMENT:  ori = (fv[0] == bv[0]) ? 0 : 1; break;
      case Geometry::TRIANGLE: ori = GetTriOrientation(bv, fv); break;
      case Geometry::SQUARE:   ori = GetQuadOrientation(bv, fv); break;
      default: MFEM_ABORT("boundary element type not implemented"); ori = 0;
   }
   el   = fi.Elem1No;
   info = fi.Elem1Inf + ori;
}

void Mesh::SetAttribute(int i, int attr)
{
   elements[i]->SetAttribute(attr);
   if (elem_attrs_cache.Size() == GetNE())
   {
      // update the existing cache instead of deleting it
      elem_attrs_cache.HostReadWrite();
      elem_attrs_cache[i] = attr;
   }
   if (ncmesh) { ncmesh->SetAttribute(i, attr); }
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
   {
      for (j = 0; j < nv; j++)
      {
         pointmat(k, j) = vertices[v[j]](k);
      }
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

real_t Mesh::GetLength(int i, int j) const
{
   const real_t *vi = vertices[i]();
   const real_t *vj = vertices[j]();
   real_t length = 0.;

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

int Mesh::GetElementToEdgeTable(Table &e_to_f)
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
      be_to_face.SetSize(NumOfBdrElements);
      for (i = 0; i < NumOfBdrElements; i++)
      {
         const int *v = boundary[i]->GetVertices();
         be_to_face[i] = v_to_v(v[0], v[1]);
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
   if (faces[gf] == NULL)  // this will be elem1
   {
      faces[gf] = new Point(&gf);
      faces_info[gf].Elem1No  = el;
      faces_info[gf].Elem1Inf = 64 * lf; // face lf with orientation 0
      faces_info[gf].Elem2No  = -1; // in case there's no other side
      faces_info[gf].Elem2Inf = -1; // face is not shared
   }
   else  //  this will be elem2
   {
      /* WARNING: Without the following check the mesh faces_info data structure
         may contain unreliable data. Normally, the order in which elements are
         processed could swap which elements appear as Elem1No and Elem2No. In
         branched meshes, where more than two elements can meet at a given node,
         the indices stored in Elem1No and Elem2No will be the first and last,
         respectively, elements found which touch a given node. This can lead to
         inconsistencies in any algorithms which rely on this data structure. To
         properly support branched meshes this data structure should be extended
         to support multiple elements per face. */
      /*
      MFEM_VERIFY(faces_info[gf].Elem2No < 0, "Invalid mesh topology. "
                  "Interior point found connecting 1D elements "
                  << faces_info[gf].Elem1No << ", " << faces_info[gf].Elem2No
                  << " and " << el << ".");
      */
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
      MFEM_VERIFY(faces_info[gf].Elem2No < 0, "Invalid mesh topology.  "
                  "Interior edge found between 2D elements "
                  << faces_info[gf].Elem1No << ", " << faces_info[gf].Elem2No
                  << " and " << el << ".");
      int *v = faces[gf]->GetVertices();
      faces_info[gf].Elem2No  = el;
      if (v[1] == v0 && v[0] == v1)
      {
         faces_info[gf].Elem2Inf = 64 * lf + 1;
      }
      else if (v[0] == v0 && v[1] == v1)
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
      MFEM_VERIFY(faces_info[gf].Elem2No < 0, "Invalid mesh topology.  "
                  "Interior triangular face found connecting elements "
                  << faces_info[gf].Elem1No << ", " << faces_info[gf].Elem2No
                  << " and " << el << ".");
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
      MFEM_VERIFY(faces_info[gf].Elem2No < 0, "Invalid mesh topology.  "
                  "Interior quadrilateral face found connecting elements "
                  << faces_info[gf].Elem1No << ", " << faces_info[gf].Elem2No
                  << " and " << el << ".");
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
   int nfaces = GetNumFaces();
   for (auto &f : faces)
   {
      FreeElement(f);
   }

   // delete caches
   face_indices[0].SetSize(0);
   face_indices[1].SetSize(0);
   inv_face_indices[0].clear();
   inv_face_indices[1].clear();

   // (re)generate the interior faces and the info for them
   faces.SetSize(nfaces);
   faces_info.SetSize(nfaces);
   for (int i = 0; i < nfaces; ++i)
   {
      faces[i] = NULL;
      faces_info[i].Elem1No = -1;
      faces_info[i].NCFace = -1;
   }

   Array<int> v;
   for (int i = 0; i < NumOfElements; ++i)
   {
      elements[i]->GetVertices(v);
      if (Dim == 1)
      {
         AddPointFaceElement(0, v[0], i);
         AddPointFaceElement(1, v[1], i);
      }
      else if (Dim == 2)
      {
         const int * const ef = el_to_edge->GetRow(i);
         const int ne = elements[i]->GetNEdges();
         for (int j = 0; j < ne; j++)
         {
            const int *e = elements[i]->GetEdgeVertices(j);
            AddSegmentFaceElement(j, ef[j], i, v[e[0]], v[e[1]]);
         }
      }
      else
      {
         const int * const ef = el_to_face->GetRow(i);
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
            case Element::PYRAMID:
            {
               for (int j = 0; j < 1; j++)
               {
                  const int *fv = pyr_t::FaceVert[j];
                  AddQuadFaceElement(j, ef[j], i,
                                     v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
               }
               for (int j = 1; j < 5; j++)
               {
                  const int *fv = pyr_t::FaceVert[j];
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
   }
}

void Mesh::GenerateNCFaceInfo()
{
   MFEM_VERIFY(ncmesh, "missing NCMesh.");

   for (auto &x : faces_info)
   {
      x.NCFace = -1;
   }

   const NCMesh::NCList &list =
      (Dim == 2) ? ncmesh->GetEdgeList() : ncmesh->GetFaceList();

   nc_faces_info.SetSize(0);
   nc_faces_info.Reserve(list.masters.Size() + list.slaves.Size());

   int nfaces = GetNumFaces();

   // add records for master faces
   for (const NCMesh::Master &master : list.masters)
   {
      if (master.index >= nfaces) { continue; }

      FaceInfo &master_fi = faces_info[master.index];
      master_fi.NCFace = nc_faces_info.Size();
      nc_faces_info.Append(NCFaceInfo(false, master.local, NULL));
      // NOTE: one of the unused members stores local face no. to be used below
      MFEM_ASSERT(master_fi.Elem2No == -1, "internal error");
      MFEM_ASSERT(master_fi.Elem2Inf == -1, "internal error");
   }

   // add records for slave faces
   for (const NCMesh::Slave &slave : list.slaves)
   {
      if (slave.index < 0 || // degenerate slave face
          slave.index >= nfaces || // ghost slave
          slave.master >= nfaces) // has ghost master
      {
         continue;
      }

      FaceInfo &slave_fi = faces_info[slave.index];
      FaceInfo &master_fi = faces_info[slave.master];
      NCFaceInfo &master_nc = nc_faces_info[master_fi.NCFace];

      slave_fi.NCFace = nc_faces_info.Size();
      slave_fi.Elem2No = master_fi.Elem1No;
      slave_fi.Elem2Inf = 64 * master_nc.MasterFace; // get lf no. stored above
      // NOTE: In 3D, the orientation part of Elem2Inf is encoded in the point
      //       matrix. In 2D, the point matrix has the orientation of the parent
      //       edge, so its columns need to be flipped when applying it, see
      //       ApplyLocalSlaveTransformation.

      nc_faces_info.Append(
         NCFaceInfo(true, slave.master,
                    list.point_matrices[slave.geom][slave.matrix]));
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
         case Element::PYRAMID:
         {
            for (int j = 0; j < 1; j++)
            {
               const int *fv = pyr_t::FaceVert[j];
               faces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]);
            }
            for (int j = 1; j < 5; j++)
            {
               const int *fv = pyr_t::FaceVert[j];
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
            MFEM_ABORT("Unexpected type of Element: " << GetElementType(i));
      }
   }
   return faces_tbl;
}

STable3D *Mesh::GetElementToFaceTable(int ret_ftbl)
{
   Array<int> v;
   STable3D *faces_tbl;

   if (el_to_face != NULL)
   {
      delete el_to_face;
   }
   el_to_face = new Table(NumOfElements, 6);  // must be 6 for hexahedra
   faces_tbl = new STable3D(NumOfVertices);
   for (int i = 0; i < NumOfElements; i++)
   {
      elements[i]->GetVertices(v);
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
         case Element::PYRAMID:
         {
            for (int j = 0; j < 1; j++)
            {
               const int *fv = pyr_t::FaceVert[j];
               el_to_face->Push(
                  i, faces_tbl->Push4(v[fv[0]], v[fv[1]], v[fv[2]], v[fv[3]]));
            }
            for (int j = 1; j < 5; j++)
            {
               const int *fv = pyr_t::FaceVert[j];
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

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      boundary[i]->GetVertices(v);
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

// shift cyclically 3 integers so that the smallest is first
static inline
void Rotate3(int &a, int &b, int &c)
{
   if (a < b)
   {
      if (a > c)
      {
         ShiftRight(a, b, c);
      }
   }
   else
   {
      if (b < c)
      {
         ShiftRight(c, b, a);
      }
      else
      {
         ShiftRight(a, b, c);
      }
   }
}

void Mesh::ReorientTetMesh()
{
   if (Dim != 3 || !(meshgen & 1))
   {
      return;
   }

   ResetLazyData();

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
         int *v = elements[i]->GetVertices();

         Rotate3(v[0], v[1], v[2]);
         if (v[0] < v[3])
         {
            Rotate3(v[1], v[2], v[3]);
         }
         else
         {
            ShiftRight(v[0], v[1], v[3]);
         }
      }
   }

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      if (GetBdrElementType(i) == Element::TRIANGLE)
      {
         int *v = boundary[i]->GetVertices();

         Rotate3(v[0], v[1], v[2]);
      }
   }

   if (!Nodes)
   {
      GetElementToFaceTable();
      GenerateFaces();
      if (el_to_edge)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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
   real_t pmin[3] = { infinity(), infinity(), infinity() };
   real_t pmax[3] = { -infinity(), -infinity(), -infinity() };
   // find a bounding box using the vertices
   for (int vi = 0; vi < NumOfVertices; vi++)
   {
      const real_t *p = vertices[vi]();
      for (int i = 0; i < spaceDim; i++)
      {
         if (p[i] < pmin[i]) { pmin[i] = p[i]; }
         if (p[i] > pmax[i]) { pmax[i] = p[i]; }
      }
   }

   partitioning = new int[NumOfElements];

   // determine the partitioning using the centers of the elements
   real_t ppt[3];
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

void FindPartitioningComponents(Table &elem_elem,
                                const Array<int> &partitioning,
                                Array<int> &component,
                                Array<int> &num_comp);

int *Mesh::GeneratePartitioning(int nparts, int part_method)
{
#ifdef MFEM_USE_METIS

   int print_messages = 1;
   // If running in parallel, print messages only from rank 0.
#ifdef MFEM_USE_MPI
   int init_flag, fin_flag;
   MPI_Initialized(&init_flag);
   MPI_Finalized(&fin_flag);
   if (init_flag && !fin_flag)
   {
      int rank;
      MPI_Comm_rank(GetGlobalMPI_Comm(), &rank);
      if (rank != 0) { print_messages = 0; }
   }
#endif

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
      idx_t errflag;
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
      // If the mesh is disconnected, disable METIS_OPTION_CONTIG.
      {
         Array<int> part(partitioning, NumOfElements);
         part = 0; // single part for the whole mesh
         Array<int> component; // size will be set to num. elem.
         Array<int> num_comp;  // size will be set to num. parts (1)
         FindPartitioningComponents(*el_to_el, part, component, num_comp);
         if (num_comp[0] > 1) { options[METIS_OPTION_CONTIG] = 0; }
      }
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
         errflag = METIS_PartGraphRecursive(&n,
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
         if (errflag != 1)
         {
            mfem_error("Mesh::GeneratePartitioning: "
                       " error in METIS_PartGraphRecursive!");
         }
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
         errflag = METIS_PartGraphKway(&n,
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
         if (errflag != 1)
         {
            mfem_error("Mesh::GeneratePartitioning: "
                       " error in METIS_PartGraphKway!");
         }
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
         errflag = METIS_PartGraphKway(&n,
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
         if (errflag != 1)
         {
            mfem_error("Mesh::GeneratePartitioning: "
                       " error in METIS_PartGraphKway!");
         }
#endif
      }

#ifdef MFEM_DEBUG
      if (print_messages)
      {
         mfem::out << "Mesh::GeneratePartitioning(...): edgecut = "
                   << edgecut << endl;
      }
#endif
      nparts = (int) mparts;
      if (mpartitioning != (idx_t*)partitioning)
      {
         for (int k = 0; k<NumOfElements; k++)
         {
            partitioning[k] = mpartitioning[k];
         }
      }
      if (freedata)
      {
         delete[] I;
         delete[] J;
         delete[] mpartitioning;
      }
   }

   delete el_to_el;
   el_to_el = NULL;

   // Check for empty partitionings (a "feature" in METIS)
   if (nparts > 1 && NumOfElements > nparts)
   {
      Array< Pair<int,int> > psize(nparts);
      int empty_parts;

      // Count how many elements are in each partition, and store the result in
      // psize, where psize[i].one is the number of elements, and psize[i].two
      // is partition index. Keep track of the number of empty parts.
      auto count_partition_elements = [&]()
      {
         for (i = 0; i < nparts; i++)
         {
            psize[i].one = 0;
            psize[i].two = i;
         }

         for (i = 0; i < NumOfElements; i++)
         {
            psize[partitioning[i]].one++;
         }

         empty_parts = 0;
         for (i = 0; i < nparts; i++)
         {
            if (psize[i].one == 0) { empty_parts++; }
         }
      };

      count_partition_elements();

      // This code just split the largest partitionings in two.
      // Do we need to replace it with something better?
      while (empty_parts)
      {
         if (print_messages)
         {
            mfem::err << "Mesh::GeneratePartitioning(...): METIS returned "
                      << empty_parts << " empty parts!"
                      << " Applying a simple fix ..." << endl;
         }

         SortPairs<int,int>(psize, nparts);

         for (i = nparts-1; i > nparts-1-empty_parts; i--)
         {
            psize[i].one /= 2;
         }

         for (int j = 0; j < NumOfElements; j++)
         {
            for (i = nparts-1; i > nparts-1-empty_parts; i--)
            {
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

         // Check for empty partitionings again
         count_partition_elements();
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

void Mesh::CheckPartitioning(int *partitioning_)
{
   int i, n_empty, n_mcomp;
   Array<int> component, num_comp;
   const Array<int> partitioning(partitioning_, GetNE());

   ElementToElementTable();

   FindPartitioningComponents(*el_to_el, partitioning, component, num_comp);

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
   const real_t *a = A.Data();
   const real_t *b = B.Data();

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
         real_t a = z(2), b = z(1), c = z(0);
         real_t D = b*b-4*a*c;
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
            real_t t;
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
               Swap<real_t>(x(0), x(1));
            }
            return 2;
         }
      }

      case 3:
      {
         real_t a = z(2)/z(3), b = z(1)/z(3), c = z(0)/z(3);

         // find the real roots of x^3 + a x^2 + b x + c = 0
         real_t Q = (a * a - 3 * b) / 9;
         real_t R = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
         real_t Q3 = Q * Q * Q;
         real_t R2 = R * R;

         if (R2 == Q3)
         {
            if (Q == 0)
            {
               x(0) = x(1) = x(2) = - a / 3;
            }
            else
            {
               real_t sqrtQ = sqrt(Q);

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
            real_t theta = acos(R / sqrt(Q3));
            real_t A = -2 * sqrt(Q);
            real_t x0, x1, x2;
            x0 = A * cos(theta / 3) - a / 3;
            x1 = A * cos((theta + 2.0 * M_PI) / 3) - a / 3;
            x2 = A * cos((theta - 2.0 * M_PI) / 3) - a / 3;

            /* Sort x0, x1, x2 */
            if (x0 > x1)
            {
               Swap<real_t>(x0, x1);
            }
            if (x1 > x2)
            {
               Swap<real_t>(x1, x2);
               if (x0 > x1)
               {
                  Swap<real_t>(x0, x1);
               }
            }
            x(0) = x0;
            x(1) = x1;
            x(2) = x2;
            return 3;
         }
         else
         {
            real_t A;
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

void FindTMax(Vector &c, Vector &x, real_t &tmax,
              const real_t factor, const int Dim)
{
   const real_t c0 = c(0);
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

void Mesh::CheckDisplacements(const Vector &displacements, real_t &tmax)
{
   int nvs = vertices.Size();
   DenseMatrix P, V, DS, PDS(spaceDim), VDS(spaceDim);
   Vector c(spaceDim+1), x(spaceDim);
   const real_t factor = 2.0;

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
   MFEM_VERIFY(vert_coord.Size() == spaceDim * NumOfVertices, "");
   vertices.SetSize(NumOfVertices);
   for (int i = 0, nv = vertices.Size(); i < nv; i++)
      for (int j = 0; j < spaceDim; j++)
      {
         vertices[i](j) = vert_coord(j*nv+i);
      }
}

void Mesh::GetNode(int i, real_t *coord) const
{
   if (Nodes)
   {
      FiniteElementSpace *fes = Nodes->FESpace();
      for (int j = 0; j < spaceDim; j++)
      {
         coord[j] = AsConst(*Nodes)(fes->DofToVDof(i, j));
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

void Mesh::SetNode(int i, const real_t *coord)
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

   // Invalidate the old geometric factors
   NodesUpdated();
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

   // Invalidate the old geometric factors
   NodesUpdated();
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

   if (ncmesh)
   {
      ncmesh->MakeTopologyOnly();
   }

   // Invalidate the old geometric factors
   NodesUpdated();
}

void Mesh::SwapNodes(GridFunction *&nodes, int &own_nodes_)
{
   // If this is a nonconforming mesh without nodes, ncmesh->coordinates will
   // be non-empty; so if the 'nodes' argument is not NULL, we will create an
   // inconsistent state where the Mesh has nodes and ncmesh->coordinates is not
   // empty. This was creating an issue for Mesh::Print() since both the
   // "coordinates" and "nodes" sections were written, leading to crashes during
   // loading. This issue is now fixed in Mesh::Printer() by temporarily
   // swapping ncmesh->coordinates with an empty array when the Mesh has nodes.

   mfem::Swap<GridFunction*>(Nodes, nodes);
   mfem::Swap<int>(own_nodes, own_nodes_);
   // TODO:
   // if (nodes)
   //    nodes->FESpace()->MakeNURBSextOwner();
   // NURBSext = (Nodes) ? Nodes->FESpace()->StealNURBSext() : NULL;

   // Invalidate the old geometric factors
   NodesUpdated();
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

      // update vertex coordinates for compatibility (e.g., GetVertex())
      SetVerticesFromNodes(Nodes);

      // Invalidate the old geometric factors
      NodesUpdated();
   }
}

void Mesh::UniformRefinement2D_base(bool update_nodes)
{
   ResetLazyData();

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

   Array<Element*> new_elements;
   Array<Element*> new_boundary;

   vertices.SetSize(oelem + quad_counter);
   new_elements.SetSize(4 * NumOfElements);
   quad_counter = 0;

   for (int i = 0, j = 0; i < NumOfElements; i++)
   {
      const Element::Type el_type = elements[i]->GetType();
      const int attr = elements[i]->GetAttribute();
      int *v = elements[i]->GetVertices();
      const int *e = el_to_edge->GetRow(i);
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

         new_elements[j++] =
            new Triangle(v[0], oedge+e[0], oedge+e[2], attr);
         new_elements[j++] =
            new Triangle(oedge+e[1], oedge+e[2], oedge+e[0], attr);
         new_elements[j++] =
            new Triangle(oedge+e[0], v[1], oedge+e[1], attr);
         new_elements[j++] =
            new Triangle(oedge+e[2], oedge+e[1], v[2], attr);
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

         new_elements[j++] =
            new Quadrilateral(v[0], oedge+e[0], oelem+qe, oedge+e[3], attr);
         new_elements[j++] =
            new Quadrilateral(oedge+e[0], v[1], oedge+e[1], oelem+qe, attr);
         new_elements[j++] =
            new Quadrilateral(oelem+qe, oedge+e[1], v[2], oedge+e[2], attr);
         new_elements[j++] =
            new Quadrilateral(oedge+e[3], oelem+qe, oedge+e[2], v[3], attr);
      }
      else
      {
         MFEM_ABORT("unknown element type: " << el_type);
      }
      FreeElement(elements[i]);
   }
   mfem::Swap(elements, new_elements);

   // refine boundary elements
   new_boundary.SetSize(2 * NumOfBdrElements);
   for (int i = 0, j = 0; i < NumOfBdrElements; i++)
   {
      const int attr = boundary[i]->GetAttribute();
      int *v = boundary[i]->GetVertices();

      new_boundary[j++] = new Segment(v[0], oedge+be_to_face[i], attr);
      new_boundary[j++] = new Segment(oedge+be_to_face[i], v[1], attr);

      FreeElement(boundary[i]);
   }
   mfem::Swap(boundary, new_boundary);

   static const real_t A = 0.0, B = 0.5, C = 1.0;
   static real_t tri_children[2*3*4] =
   {
      A,A, B,A, A,B,
      B,B, A,B, B,A,
      B,A, C,A, B,B,
      A,B, B,B, A,C
   };
   static real_t quad_children[2*4*4] =
   {
      A,A, B,A, B,B, A,B, // lower-left
      B,A, C,A, C,B, B,B, // lower-right
      B,B, C,B, C,C, B,C, // upper-right
      A,B, B,B, B,C, A,C  // upper-left
   };

   CoarseFineTr.point_matrices[Geometry::TRIANGLE]
   .UseExternalData(tri_children, 2, 3, 4);
   CoarseFineTr.point_matrices[Geometry::SQUARE]
   .UseExternalData(quad_children, 2, 4, 4);
   CoarseFineTr.embeddings.SetSize(elements.Size());

   for (int i = 0; i < elements.Size(); i++)
   {
      Embedding &emb = CoarseFineTr.embeddings[i];
      emb.parent = i / 4;
      emb.matrix = i % 4;
   }

   NumOfVertices    = vertices.Size();
   NumOfElements    = 4 * NumOfElements;
   NumOfBdrElements = 2 * NumOfBdrElements;
   NumOfFaces       = 0;

   NumOfEdges = GetElementToEdgeTable(*el_to_edge);
   GenerateFaces();

   last_operation = Mesh::REFINE;
   sequence++;

   if (update_nodes) { UpdateNodes(); }

#ifdef MFEM_DEBUG
   if (!Nodes || update_nodes)
   {
      CheckElementOrientation(false);
   }
   CheckBdrElementOrientation(false);
#endif
}

static inline real_t sqr(const real_t &x)
{
   return x*x;
}

void Mesh::UniformRefinement3D_base(Array<int> *f2qf_ptr, DSTable *v_to_v_p,
                                    bool update_nodes)
{
   ResetLazyData();

   if (el_to_edge == NULL)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

   int pyr_counter = 0;
   if (HasGeometry(Geometry::PYRAMID))
   {
      for (int i = 0; i < elements.Size(); i++)
      {
         if (elements[i]->GetType() == Element::PYRAMID)
         {
            pyr_counter++;
         }
      }
   }

   // Map from edge-index to vertex-index, needed for ReorientTetMesh() for
   // parallel meshes.
   // Note: with the removal of ReorientTetMesh() this may no longer
   // be needed.  Unfortunately, it's hard to be sure.
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

   Array<Element*> new_elements;
   Array<Element*> new_boundary;

   vertices.SetSize(oelem + hex_counter);
   new_elements.SetSize(8 * NumOfElements + 2 * pyr_counter);
   CoarseFineTr.embeddings.SetSize(new_elements.Size());

   hex_counter = 0;
   for (int i = 0, j = 0; i < NumOfElements; i++)
   {
      const Element::Type el_type = elements[i]->GetType();
      const int attr = elements[i]->GetAttribute();
      int *v = elements[i]->GetVertices();
      const int *e = el_to_edge->GetRow(i);
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
               real_t len_sqr, min_len;

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
               real_t Em_data[18], Js_data[9], Jp_data[9];
               DenseMatrix Em(Em_data, 3, 6);
               DenseMatrix Js(Js_data, 3, 3), Jp(Jp_data, 3, 3);
               real_t ar1, ar2, kappa, kappa_min;

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
            new_elements[j+0] =
               new Tetrahedron(v[0], oedge+e[0], oedge+e[1], oedge+e[2], attr);
            new_elements[j+1] =
               new Tetrahedron(oedge+e[0], v[1], oedge+e[3], oedge+e[4], attr);
            new_elements[j+2] =
               new Tetrahedron(oedge+e[1], oedge+e[3], v[2], oedge+e[5], attr);
            new_elements[j+3] =
               new Tetrahedron(oedge+e[2], oedge+e[4], oedge+e[5], v[3], attr);

            for (int k = 0; k < 4; k++)
            {
               new_elements[j+4+k] =
                  new Tetrahedron(oedge+e[mv[k][0]], oedge+e[mv[k][1]],
                                  oedge+e[mv[k][2]], oedge+e[mv[k][3]], attr);
            }
#else
            Tetrahedron *tet;
            new_elements[j+0] = tet = TetMemory.Alloc();
            tet->Init(v[0], oedge+e[0], oedge+e[1], oedge+e[2], attr);

            new_elements[j+1] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[0], v[1], oedge+e[3], oedge+e[4], attr);

            new_elements[j+2] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[1], oedge+e[3], v[2], oedge+e[5], attr);

            new_elements[j+3] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[2], oedge+e[4], oedge+e[5], v[3], attr);

            for (int k = 0; k < 4; k++)
            {
               new_elements[j+4+k] = tet = TetMemory.Alloc();
               tet->Init(oedge+e[mv[k][0]], oedge+e[mv[k][1]],
                         oedge+e[mv[k][2]], oedge+e[mv[k][3]], attr);
            }
#endif
            for (int k = 0; k < 4; k++)
            {
               CoarseFineTr.embeddings[j+k].parent = i;
               CoarseFineTr.embeddings[j+k].matrix = k;
            }
            for (int k = 0; k < 4; k++)
            {
               CoarseFineTr.embeddings[j+4+k].parent = i;
               CoarseFineTr.embeddings[j+4+k].matrix = 4*(rt+1)+k;
            }

            j += 8;
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

            new_elements[j++] =
               new Wedge(v[0], oedge+e[0], oedge+e[2],
                         oedge+e[6], oface+qf2, oface+qf4, attr);

            new_elements[j++] =
               new Wedge(oedge+e[1], oedge+e[2], oedge+e[0],
                         oface+qf3, oface+qf4, oface+qf2, attr);

            new_elements[j++] =
               new Wedge(oedge+e[0], v[1], oedge+e[1],
                         oface+qf2, oedge+e[7], oface+qf3, attr);

            new_elements[j++] =
               new Wedge(oedge+e[2], oedge+e[1], v[2],
                         oface+qf4, oface+qf3, oedge+e[8], attr);

            new_elements[j++] =
               new Wedge(oedge+e[6], oface+qf2, oface+qf4,
                         v[3], oedge+e[3], oedge+e[5], attr);

            new_elements[j++] =
               new Wedge(oface+qf3, oface+qf4, oface+qf2,
                         oedge+e[4], oedge+e[5], oedge+e[3], attr);

            new_elements[j++] =
               new Wedge(oface+qf2, oedge+e[7], oface+qf3,
                         oedge+e[3], v[4], oedge+e[4], attr);

            new_elements[j++] =
               new Wedge(oface+qf4, oface+qf3, oedge+e[8],
                         oedge+e[5], oedge+e[4], v[5], attr);
         }
         break;

         case Element::PYRAMID:
         {
            const int *f = el_to_face->GetRow(i);
            // pyr_counter++;

            for (int fi = 0; fi < 1; fi++)
            {
               for (int k = 0; k < 4; k++)
               {
                  vv[k] = v[pyr_t::FaceVert[fi][k]];
               }
               AverageVertices(vv, 4, oface + f2qf[f[fi]]);
            }

            for (int ei = 0; ei < 8; ei++)
            {
               for (int k = 0; k < 2; k++)
               {
                  vv[k] = v[pyr_t::Edges[ei][k]];
               }
               AverageVertices(vv, 2, oedge+e[ei]);
            }

            const int qf0 = f2qf[f[0]];

            new_elements[j++] =
               new Pyramid(v[0], oedge+e[0], oface+qf0,
                           oedge+e[3], oedge+e[4], attr);

            new_elements[j++] =
               new Pyramid(oedge+e[0], v[1], oedge+e[1],
                           oface+qf0, oedge+e[5], attr);

            new_elements[j++] =
               new Pyramid(oface+qf0, oedge+e[1], v[2],
                           oedge+e[2], oedge+e[6], attr);

            new_elements[j++] =
               new Pyramid(oedge+e[3], oface+qf0, oedge+e[2],
                           v[3], oedge+e[7], attr);

            new_elements[j++] =
               new Pyramid(oedge+e[4], oedge+e[5], oedge+e[6],
                           oedge+e[7], v[4], attr);

            new_elements[j++] =
               new Pyramid(oedge+e[7], oedge+e[6], oedge+e[5],
                           oedge+e[4], oface+qf0, attr);

#ifndef MFEM_USE_MEMALLOC
            new_elements[j++] =
               new Tetrahedron(oedge+e[0], oedge+e[4], oedge+e[5],
                               oface+qf0, attr);

            new_elements[j++] =
               new Tetrahedron(oedge+e[1], oedge+e[5], oedge+e[6],
                               oface+qf0, attr);

            new_elements[j++] =
               new Tetrahedron(oedge+e[2], oedge+e[6], oedge+e[7],
                               oface+qf0, attr);

            new_elements[j++] =
               new Tetrahedron(oedge+e[3], oedge+e[7], oedge+e[4],
                               oface+qf0, attr);
#else
            Tetrahedron *tet;
            new_elements[j++] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[0], oedge+e[4], oedge+e[5],
                      oface+qf0, attr);

            new_elements[j++] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[1], oedge+e[5], oedge+e[6],
                      oface+qf0, attr);

            new_elements[j++] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[2], oedge+e[6], oedge+e[7],
                      oface+qf0, attr);

            new_elements[j++] = tet = TetMemory.Alloc();
            tet->Init(oedge+e[3], oedge+e[7], oedge+e[4],
                      oface+qf0, attr);
#endif
            // Tetrahedral elements may be new to this mesh so ensure that
            // the relevant flags are switched on
            mesh_geoms |= (1 << Geometry::TETRAHEDRON);
            meshgen |= 1;
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

            new_elements[j++] =
               new Hexahedron(v[0], oedge+e[0], oface+qf[0],
                              oedge+e[3], oedge+e[8], oface+qf[1],
                              oelem+he, oface+qf[4], attr);
            new_elements[j++] =
               new Hexahedron(oedge+e[0], v[1], oedge+e[1],
                              oface+qf[0], oface+qf[1], oedge+e[9],
                              oface+qf[2], oelem+he, attr);
            new_elements[j++] =
               new Hexahedron(oface+qf[0], oedge+e[1], v[2],
                              oedge+e[2], oelem+he, oface+qf[2],
                              oedge+e[10], oface+qf[3], attr);
            new_elements[j++] =
               new Hexahedron(oedge+e[3], oface+qf[0], oedge+e[2],
                              v[3], oface+qf[4], oelem+he,
                              oface+qf[3], oedge+e[11], attr);
            new_elements[j++] =
               new Hexahedron(oedge+e[8], oface+qf[1], oelem+he,
                              oface+qf[4], v[4], oedge+e[4],
                              oface+qf[5], oedge+e[7], attr);
            new_elements[j++] =
               new Hexahedron(oface+qf[1], oedge+e[9], oface+qf[2],
                              oelem+he, oedge+e[4], v[5],
                              oedge+e[5], oface+qf[5], attr);
            new_elements[j++] =
               new Hexahedron(oelem+he, oface+qf[2], oedge+e[10],
                              oface+qf[3], oface+qf[5], oedge+e[5],
                              v[6], oedge+e[6], attr);
            new_elements[j++] =
               new Hexahedron(oface+qf[4], oelem+he, oface+qf[3],
                              oedge+e[11], oedge+e[7], oface+qf[5],
                              oedge+e[6], v[7], attr);
         }
         break;

         default:
            MFEM_ABORT("Unknown 3D element type \"" << el_type << "\"");
            break;
      }
      FreeElement(elements[i]);
   }
   mfem::Swap(elements, new_elements);

   // refine boundary elements
   new_boundary.SetSize(4 * NumOfBdrElements);
   for (int i = 0, j = 0; i < NumOfBdrElements; i++)
   {
      const Element::Type bdr_el_type = boundary[i]->GetType();
      const int attr = boundary[i]->GetAttribute();
      int *v = boundary[i]->GetVertices();
      const int *e = bel_to_edge->GetRow(i);
      int ev[4];

      if (e2v.Size())
      {
         const int ne = bel_to_edge->RowSize(i);
         for (int k = 0; k < ne; k++) { ev[k] = e2v[e[k]]; }
         e = ev;
      }

      if (bdr_el_type == Element::TRIANGLE)
      {
         new_boundary[j++] =
            new Triangle(v[0], oedge+e[0], oedge+e[2], attr);
         new_boundary[j++] =
            new Triangle(oedge+e[1], oedge+e[2], oedge+e[0], attr);
         new_boundary[j++] =
            new Triangle(oedge+e[0], v[1], oedge+e[1], attr);
         new_boundary[j++] =
            new Triangle(oedge+e[2], oedge+e[1], v[2], attr);
      }
      else if (bdr_el_type == Element::QUADRILATERAL)
      {
         const int qf =
            (f2qf.Size() == 0) ? be_to_face[i] : f2qf[be_to_face[i]];

         new_boundary[j++] =
            new Quadrilateral(v[0], oedge+e[0], oface+qf, oedge+e[3], attr);
         new_boundary[j++] =
            new Quadrilateral(oedge+e[0], v[1], oedge+e[1], oface+qf, attr);
         new_boundary[j++] =
            new Quadrilateral(oface+qf, oedge+e[1], v[2], oedge+e[2], attr);
         new_boundary[j++] =
            new Quadrilateral(oedge+e[3], oface+qf, oedge+e[2], v[3], attr);
      }
      else
      {
         MFEM_ABORT("boundary Element is not a triangle or a quad!");
      }
      FreeElement(boundary[i]);
   }
   mfem::Swap(boundary, new_boundary);

   static const real_t A = 0.0, B = 0.5, C = 1.0, D = -1.0;
   static real_t tet_children[3*4*16] =
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
   static real_t pyr_children[3*5*10] =
   {
      A,A,A, B,A,A, B,B,A, A,B,A, A,A,B,
      B,A,A, C,A,A, C,B,A, B,B,A, B,A,B,
      B,B,A, C,B,A, C,C,A, B,C,A, B,B,B,
      A,B,A, B,B,A, B,C,A, A,C,A, A,B,B,
      A,A,B, B,A,B, B,B,B, A,B,B, A,A,C,
      A,B,B, B,B,B, B,A,B, A,A,B, B,B,A,
      B,A,A, A,A,B, B,A,B, B,B,A, D,D,D,
      C,B,A, B,A,B, B,B,B, B,B,A, D,D,D,
      B,C,A, B,B,B, A,B,B, B,B,A, D,D,D,
      A,B,A, A,B,B, A,A,B, B,B,A, D,D,D
   };
   static real_t pri_children[3*6*8] =
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
   static real_t hex_children[3*8*8] =
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

   CoarseFineTr.point_matrices[Geometry::TETRAHEDRON]
   .UseExternalData(tet_children, 3, 4, 16);
   CoarseFineTr.point_matrices[Geometry::PYRAMID]
   .UseExternalData(pyr_children, 3, 5, 10);
   CoarseFineTr.point_matrices[Geometry::PRISM]
   .UseExternalData(pri_children, 3, 6, 8);
   CoarseFineTr.point_matrices[Geometry::CUBE]
   .UseExternalData(hex_children, 3, 8, 8);

   for (int i = 0; i < elements.Size(); i++)
   {
      // tetrahedron elements are handled above:
      if (elements[i]->GetType() == Element::TETRAHEDRON) { continue; }

      Embedding &emb = CoarseFineTr.embeddings[i];
      emb.parent = i / 8;
      emb.matrix = i % 8;
   }

   NumOfVertices    = vertices.Size();
   NumOfElements    = 8 * NumOfElements + 2 * pyr_counter;
   NumOfBdrElements = 4 * NumOfBdrElements;

   GetElementToFaceTable();
   GenerateFaces();

#ifdef MFEM_DEBUG
   CheckBdrElementOrientation(false);
#endif

   NumOfEdges = GetElementToEdgeTable(*el_to_edge);

   last_operation = Mesh::REFINE;
   sequence++;

   if (update_nodes) { UpdateNodes(); }
}

void Mesh::LocalRefinement(const Array<int> &marked_el, int type)
{
   int i, j, ind, nedges;
   Array<int> v;

   ResetLazyData();

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

         CoarseFineTr.embeddings[i] = Embedding(i, Geometry::SEGMENT, 1);
         CoarseFineTr.embeddings[new_e] = Embedding(i, Geometry::SEGMENT, 2);
      }

      static real_t seg_children[3*2] = { 0.0,1.0, 0.0,0.5, 0.5,1.0 };
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
         NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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
         NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

   ResetLazyData();

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

   UpdateNodes();
}

real_t Mesh::AggregateError(const Array<real_t> &elem_error,
                            const int *fine, int nfine, int op)
{
   real_t error = (op == 3) ? std::pow(elem_error[fine[0]],
                                       2.0) : elem_error[fine[0]];

   for (int i = 1; i < nfine; i++)
   {
      MFEM_VERIFY(fine[i] < elem_error.Size(), "");

      real_t err_fine = elem_error[fine[i]];
      switch (op)
      {
         case 0: error = std::min(error, err_fine); break;
         case 1: error += err_fine; break;
         case 2: error = std::max(error, err_fine); break;
         case 3: error += std::pow(err_fine, 2.0); break;
         default: MFEM_ABORT("Invalid operation.");
      }
   }
   return (op == 3) ? std::sqrt(error) : error;
}

bool Mesh::NonconformingDerefinement(Array<real_t> &elem_error,
                                     real_t threshold, int nc_limit, int op)
{
   MFEM_VERIFY(ncmesh, "Only supported for non-conforming meshes.");
   MFEM_VERIFY(!NURBSext, "Derefinement of NURBS meshes is not supported. "
               "Project the NURBS to Nodes first.");

   ResetLazyData();

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

      real_t error =
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

   UpdateNodes();

   return true;
}

bool Mesh::DerefineByError(Array<real_t> &elem_error, real_t threshold,
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

bool Mesh::DerefineByError(const Vector &elem_error, real_t threshold,
                           int nc_limit, int op)
{
   Array<real_t> tmp(elem_error.Size());
   for (int i = 0; i < tmp.Size(); i++)
   {
      tmp[i] = elem_error(i);
   }
   return DerefineByError(tmp, threshold, nc_limit, op);
}


void Mesh::InitFromNCMesh(const NCMesh &ncmesh_)
{
   Dim = ncmesh_.Dimension();
   spaceDim = ncmesh_.SpaceDimension();

   DeleteTables();

   ncmesh_.GetMeshComponents(*this);

   NumOfVertices = vertices.Size();
   NumOfElements = elements.Size();
   NumOfBdrElements = boundary.Size();

   SetMeshGen(); // set the mesh type: 'meshgen', ...

   NumOfEdges = NumOfFaces = 0;
   nbInteriorFaces = nbBoundaryFaces = -1;

   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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

Mesh::Mesh(const NCMesh &ncmesh_)
   : attribute_sets(attributes), bdr_attribute_sets(bdr_attributes)
{
   Init();
   InitTables();
   InitFromNCMesh(ncmesh_);
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
   mfem::Swap(nbInteriorFaces, other.nbInteriorFaces);
   mfem::Swap(nbBoundaryFaces, other.nbBoundaryFaces);

   mfem::Swap(el_to_edge, other.el_to_edge);
   mfem::Swap(el_to_face, other.el_to_face);
   mfem::Swap(el_to_el, other.el_to_el);
   mfem::Swap(bel_to_edge, other.bel_to_edge);
   mfem::Swap(be_to_face, other.be_to_face);
   mfem::Swap(face_edge, other.face_edge);
   mfem::Swap(face_to_elem, other.face_to_elem);
   mfem::Swap(edge_vertex, other.edge_vertex);

   mfem::Swap(attributes, other.attributes);
   mfem::Swap(bdr_attributes, other.bdr_attributes);

   mfem::Swap(geom_factors, other.geom_factors);
   mfem::Swap(face_geom_factors, other.face_geom_factors);

#ifdef MFEM_USE_MEMALLOC
   TetMemory.Swap(other.TetMemory);
#endif

   if (non_geometry)
   {
      mfem::Swap(NURBSext, other.NURBSext);
      mfem::Swap(ncmesh, other.ncmesh);

      mfem::Swap(Nodes, other.Nodes);
      if (Nodes) { Nodes->FESpace()->UpdateMeshPointer(this); }
      if (other.Nodes) { other.Nodes->FESpace()->UpdateMeshPointer(&other); }
      mfem::Swap(own_nodes, other.own_nodes);

      mfem::Swap(CoarseFineTr, other.CoarseFineTr);

      mfem::Swap(sequence, other.sequence);
      mfem::Swap(nodes_sequence, other.nodes_sequence);
      mfem::Swap(last_operation, other.last_operation);
   }

   // copy attribute caches
   mfem::Swap(elem_attrs_cache, other.elem_attrs_cache);
   mfem::Swap(bdr_face_attrs_cache, other.bdr_face_attrs_cache);

   mfem::Swap(face_indices[0], other.face_indices[0]);
   mfem::Swap(face_indices[1], other.face_indices[1]);
   inv_face_indices[0].swap(other.inv_face_indices[0]);
   inv_face_indices[1].swap(other.inv_face_indices[1]);
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

static Array<int>& AllElements(Array<int> &list, int nelem)
{
   list.SetSize(nelem);
   for (int i = 0; i < nelem; i++) { list[i] = i; }
   return list;
}

void Mesh::UniformRefinement(int ref_algo)
{
   Array<int> list;

   if (NURBSext)
   {
      NURBSUniformRefinement();
   }
   else if (ncmesh)
   {
      GeneralRefinement(AllElements(list, GetNE()));
   }
   else if (ref_algo == 1 && meshgen == 1 && Dim == 3)
   {
      // algorithm "B" for an all-tet mesh
      LocalRefinement(AllElements(list, GetNE()));
   }
   else
   {
      switch (Dim)
      {
         case 1: LocalRefinement(AllElements(list, GetNE())); break;
         case 2: UniformRefinement2D(); break;
         case 3: UniformRefinement3D(); break;
         default: MFEM_ABORT("internal error");
      }
   }
}

void Mesh::NURBSCoarsening(int cf, real_t tol)
{
   if (NURBSext && cf > 1)
   {
      NURBSext->ConvertToPatches(*Nodes);
      Array<int> initialCoarsening;  // Initial coarsening factors
      NURBSext->GetCoarseningFactors(initialCoarsening);

      // If refinement formulas are nested, then initial coarsening is skipped.
      bool noInitialCoarsening = true;
      for (auto f : initialCoarsening)
      {
         noInitialCoarsening = (noInitialCoarsening && f == 1);
      }

      if (noInitialCoarsening)
      {
         NURBSext->Coarsen(cf, tol);
      }
      else
      {
         // Perform an initial full coarsening, and then refine. This is
         // necessary only for non-nested refinement formulas.
         NURBSext->Coarsen(initialCoarsening, tol);

         // FiniteElementSpace::Update is not supported
         last_operation = Mesh::NONE;
         sequence++;

         UpdateNURBS();

         // Prepare for refinement by factors.
         NURBSext->ConvertToPatches(*Nodes);

         Array<int> rf(initialCoarsening);
         bool divisible = true;
         for (int i=0; i<rf.Size(); ++i)
         {
            rf[i] /= cf;
            divisible = divisible && cf * rf[i] == initialCoarsening[i];
         }

         MFEM_VERIFY(divisible, "Invalid coarsening");

         // Refine from the fully coarsened mesh to the mesh coarsened by the
         // factor cf.
         NURBSext->UniformRefinement(rf);
      }

      last_operation = Mesh::NONE; // FiniteElementSpace::Update is not supported
      sequence++;

      UpdateNURBS();
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
      if ((meshgen & 2) || (meshgen & 4) || (meshgen & 8))
      {
         nonconforming = 1; // tensor product elements and wedges
      }
      else
      {
         nonconforming = 0; // simplices
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
      int type, rt = (refinements.Size() ? refinements[0].GetType() : 7);
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

void Mesh::EnsureNCMesh(bool simplices_nonconforming)
{
   MFEM_VERIFY(!NURBSext, "Cannot convert a NURBS mesh to an NC mesh. "
               "Please project the NURBS to Nodes first, with SetCurvature().");

#ifdef MFEM_USE_MPI
   MFEM_VERIFY(ncmesh != NULL || dynamic_cast<const ParMesh*>(this) == NULL,
               "Sorry, converting a conforming ParMesh to an NC mesh is "
               "not possible.");
#endif

   if (!ncmesh)
   {
      if ((meshgen & 0x2) /* quads/hexes */ ||
          (meshgen & 0x4) /* wedges */ ||
          (simplices_nonconforming && (meshgen & 0x1)) /* simplices */)
      {
         ncmesh = new NCMesh(this);
         ncmesh->OnMeshUpdated(this);
         GenerateNCFaceInfo();
      }
   }
}

void Mesh::RandomRefinement(real_t prob, bool aniso, int nonconforming,
                            int nc_limit)
{
   Array<Refinement> refs;
   for (int i = 0; i < GetNE(); i++)
   {
      if ((real_t) rand() / real_t(RAND_MAX) < prob)
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

void Mesh::RefineAtVertex(const Vertex& vert, real_t eps, int nonconforming)
{
   Array<int> v;
   Array<Refinement> refs;
   for (int i = 0; i < GetNE(); i++)
   {
      GetElementVertices(i, v);
      bool refine = false;
      for (int j = 0; j < v.Size(); j++)
      {
         real_t dist = 0.0;
         for (int l = 0; l < spaceDim; l++)
         {
            real_t d = vert(l) - vertices[v[j]](l);
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

bool Mesh::RefineByError(const Array<real_t> &elem_error, real_t threshold,
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

bool Mesh::RefineByError(const Vector &elem_error, real_t threshold,
                         int nonconforming, int nc_limit)
{
   Array<real_t> tmp(const_cast<real_t*>(elem_error.GetData()),
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
      CoarseFineTr.embeddings.Append(Embedding(coarse, Geometry::TRIANGLE));

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
      Tetrahedron *tet = (Tetrahedron *) el;

      MFEM_VERIFY(tet->GetRefinementFlag() != 0,
                  "TETRAHEDRON element is not marked for refinement.");

      vert = tet->GetVertices();

      // 1. Get the index for the new vertex in v_new.
      bisect = v_to_v.FindId(vert[0], vert[1]);
      if (bisect == -1)
      {
         v_new = NumOfVertices + v_to_v.GetId(vert[0],vert[1]);
         for (int j = 0; j < 3; j++)
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
      int type, old_redges[2], flag;
      tet->ParseRefinementFlag(old_redges, type, flag);

      int new_type, new_redges[2][2];
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
      CoarseFineTr.embeddings.Append(Embedding(coarse, Geometry::TETRAHEDRON));

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

      // 1. Get the indices for the new vertices in array v_new
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

      // 2. Set the node indices for the new elements in v1, v2, v3 & v4 so that
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
      CoarseFineTr.embeddings[i] = Embedding(coarse, Geometry::TRIANGLE);
      CoarseFineTr.embeddings.Append(Embedding(coarse, Geometry::TRIANGLE));
      CoarseFineTr.embeddings.Append(Embedding(coarse, Geometry::TRIANGLE));
      CoarseFineTr.embeddings.Append(Embedding(coarse, Geometry::TRIANGLE));

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
   CoarseFineTr.Clear();
   CoarseFineTr.embeddings.SetSize(NumOfElements);
   for (int i = 0; i < NumOfElements; i++)
   {
      elements[i]->ResetTransform(0);
      CoarseFineTr.embeddings[i] = Embedding(i, GetElementGeometry(i));
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

const CoarseFineTransformations &Mesh::GetRefinementTransforms() const
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
         for (int j = 0; j < elements.Size(); j++)
         {
            int index = 0;
            unsigned code = elements[j]->GetTransform();
            if (code)
            {
               int &matrix = mat_no[code];
               if (!matrix) { matrix = static_cast<int>(mat_no.size()); }
               index = matrix-1;
            }
            CoarseFineTr.embeddings[j].matrix = index;
         }

         DenseTensor &pmats = CoarseFineTr.point_matrices[geom];
         pmats.SetSize(Dim, Dim+1, static_cast<int>((mat_no.size())));

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

void Mesh::PrintXG(std::ostream &os) const
{
   MFEM_ASSERT(Dim==spaceDim, "2D Manifold meshes not supported");
   int i, j;
   Array<int> v;

   if (Dim == 2)
   {
      // Print the type of the mesh.
      if (Nodes == NULL)
      {
         os << "areamesh2\n\n";
      }
      else
      {
         os << "curved_areamesh2\n\n";
      }

      // Print the boundary elements.
      os << NumOfBdrElements << '\n';
      for (i = 0; i < NumOfBdrElements; i++)
      {
         boundary[i]->GetVertices(v);

         os << boundary[i]->GetAttribute();
         for (j = 0; j < v.Size(); j++)
         {
            os << ' ' << v[j] + 1;
         }
         os << '\n';
      }

      // Print the elements.
      os << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         elements[i]->GetVertices(v);

         os << elements[i]->GetAttribute() << ' ' << v.Size();
         for (j = 0; j < v.Size(); j++)
         {
            os << ' ' << v[j] + 1;
         }
         os << '\n';
      }

      if (Nodes == NULL)
      {
         // Print the vertices.
         os << NumOfVertices << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            os << vertices[i](0);
            for (j = 1; j < Dim; j++)
            {
               os << ' ' << vertices[i](j);
            }
            os << '\n';
         }
      }
      else
      {
         os << NumOfVertices << '\n';
         Nodes->Save(os);
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

         os << "NETGEN_Neutral_Format\n";
         // print the vertices
         os << NumOfVertices << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               os << ' ' << vertices[i](j);
            }
            os << '\n';
         }

         // print the elements
         os << NumOfElements << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            os << elements[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               os << ' ' << ind[j]+1;
            }
            os << '\n';
         }

         // print the boundary information.
         os << NumOfBdrElements << '\n';
         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            os << boundary[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               os << ' ' << ind[j]+1;
            }
            os << '\n';
         }
      }
      else if (meshgen == 2)  // TrueGrid
      {
         int nv;
         const int *ind;

         os << "TrueGrid\n"
            << "1 " << NumOfVertices << " " << NumOfElements
            << " 0 0 0 0 0 0 0\n"
            << "0 0 0 1 0 0 0 0 0 0 0\n"
            << "0 0 " << NumOfBdrElements << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
            << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
            << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

         for (i = 0; i < NumOfVertices; i++)
            os << i+1 << " 0.0 " << vertices[i](0) << ' ' << vertices[i](1)
               << ' ' << vertices[i](2) << " 0.0\n";

         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            os << i+1 << ' ' << elements[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               os << ' ' << ind[j]+1;
            }
            os << '\n';
         }

         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            os << boundary[i]->GetAttribute();
            for (j = 0; j < nv; j++)
            {
               os << ' ' << ind[j]+1;
            }
            os << " 1.0 1.0 1.0 1.0\n";
         }
      }
   }

   os << flush;
}

void Mesh::Printer(std::ostream &os, std::string section_delimiter,
                   const std::string &comments) const
{
   int i, j;

   if (NURBSext)
   {
      // general format
      NURBSext->Print(os, comments);
      os << '\n';
      Nodes->Save(os);

      NURBSext->PrintCoarsePatches(os);
      // patch-wise format
      // NURBSext->ConvertToPatches(*Nodes);
      // NURBSext->Print(os);

      return;
   }

   if (Nonconforming())
   {
      // Workaround for inconsistent Mesh state where the Mesh has nodes and
      // ncmesh->coodrinates is not empty. Such state can be created with the
      // method Mesh::SwapNodes(), see the comment at the beginning of its
      // implementation.
      Array<real_t> coords_save;
      if (Nodes) { mfem::Swap(coords_save, ncmesh->coordinates); }

      // nonconforming mesh format
      ncmesh->Print(os, comments);

      if (Nodes)
      {
         mfem::Swap(coords_save, ncmesh->coordinates);

         os << "\n# mesh curvature GridFunction";
         os << "\nnodes\n";
         Nodes->Save(os);
      }

      os << "\nmfem_mesh_end" << endl;
      return;
   }

   // serial/parallel conforming mesh format
   const bool set_names = attribute_sets.SetsExist() ||
                          bdr_attribute_sets.SetsExist();
   os << (!set_names && section_delimiter.empty()
          ? "MFEM mesh v1.0\n" :
          (!set_names ? "MFEM mesh v1.2\n" : "MFEM mesh v1.3\n"));

   if (set_names && section_delimiter.empty())
   {
      section_delimiter = "mfem_mesh_end";
   }

   // optional
   if (!comments.empty()) { os << '\n' << comments << '\n'; }

   os <<
      "\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n"
      "# POINT       = 0\n"
      "# SEGMENT     = 1\n"
      "# TRIANGLE    = 2\n"
      "# SQUARE      = 3\n"
      "# TETRAHEDRON = 4\n"
      "# CUBE        = 5\n"
      "# PRISM       = 6\n"
      "# PYRAMID     = 7\n"
      "#\n";

   os << "\ndimension\n" << Dim;

   os << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], os);
   }

   if (set_names)
   {
      os << "\nattribute_sets\n";
      attribute_sets.Print(os);
   }

   os << "\nboundary\n" << NumOfBdrElements << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
   {
      PrintElement(boundary[i], os);
   }

   if (set_names)
   {
      os << "\nbdr_attribute_sets\n";
      bdr_attribute_sets.Print(os);
   }

   os << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      os << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         os << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            os << ' ' << vertices[i](j);
         }
         os << '\n';
      }
      os.flush();
   }
   else
   {
      os << "\nnodes\n";
      Nodes->Save(os);
   }

   if (!section_delimiter.empty())
   {
      os << '\n'
         << section_delimiter << endl; // only with formats v1.2 and above
   }
}

void Mesh::PrintTopo(std::ostream &os, const Array<int> &e_to_k,
                     const int version, const std::string &comments) const
{
   MFEM_VERIFY(version == 10 || version == 11, "Invalid NURBS mesh version");

   int i;
   Array<int> vert;

   os << "MFEM NURBS mesh v" << int(version / 10) << "." << version % 10 << "\n";

   // optional
   if (!comments.empty()) { os << '\n' << comments << '\n'; }

   os <<
      "\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n"
      "# SEGMENT     = 1\n"
      "# SQUARE      = 3\n"
      "# CUBE        = 5\n"
      "#\n";

   os << "\ndimension\n" << Dim
      << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], os);
   }

   os << "\nboundary\n" << NumOfBdrElements << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
   {
      PrintElement(boundary[i], os);
   }

   PrintTopoEdges(os, e_to_k);
}

void Mesh::PrintTopoEdges(std::ostream &os, const Array<int> &e_to_k,
                          bool vmap) const
{
   Array<int> vert;

   // In 1D patch-topology NURBS meshes, knotvector orientation is stored in the
   // file's `edges` section, but the topological 1D mesh has NumOfEdges == 0
   // (its "faces" are vertices). When a valid edge->knotvector map is provided,
   // print a pseudo-edge list derived from the 1D elements so external tools
   // (e.g. VisIt) can consume the mapping.
   if (Dim == 1 && NumOfEdges == 0 && e_to_k.Size() == NumOfElements)
   {
      const int ne = NumOfElements;
      os << "\nedges\n" << ne << '\n';
      for (int i = 0; i < ne; i++)
      {
         const int *v = elements[i]->GetVertices();
         int v0 = v[0], v1 = v[1];

         int ki = e_to_k[i];
         const bool flip = (ki < 0); // desired output vertex order: descending
         if (flip) { ki = -1 - ki; } // print the unsigned knotvector index

         // Encode the sign of e_to_k in the vertex ordering, consistent with
         // Mesh::LoadPatchTopo(): v0 > v1 => negative sign.
         if ((v0 > v1) != flip) { std::swap(v0, v1); }

         os << ki << ' ' << v0 << ' ' << v1 << '\n';
      }

      if (!vmap)
      {
         os << "\nvertices\n" << NumOfVertices << '\n';
      }
      return;
   }

   os << "\nedges\n" << NumOfEdges << '\n';
   for (int i = 0; i < NumOfEdges; i++)
   {
      edge_vertex->GetRow(i, vert);
      int ki = e_to_k[i];
      if (ki < 0)
      {
         ki = -1 - ki;
      }

      if (vmap)
      {
         for (int j=0; j<2; ++j)
         {
            vert[j] = ncmesh->vertex_nodeId[vert[j]];
         }

         if (e_to_k[i] < 0)
         {
            // Swap the entries of vert
            const int s = vert[0];
            vert[0] = vert[1];
            vert[1] = s;
         }
      }

      os << ki << ' ' << vert[0] << ' ' << vert[1] << '\n';
   }

   if (!vmap)
   {
      os << "\nvertices\n" << NumOfVertices << '\n';
   }
}

void Mesh::Save(const std::string &fname, int precision) const
{
   ofstream ofs(fname);
   ofs.precision(precision);
   Print(ofs);
}

#ifdef MFEM_USE_ADIOS2
void Mesh::Print(adios2stream &os) const
{
   os.Print(*this);
}
#endif

void Mesh::PrintVTK(std::ostream &os)
{
   os <<
      "# vtk DataFile Version 3.0\n"
      "Generated by MFEM\n"
      "ASCII\n"
      "DATASET UNSTRUCTURED_GRID\n";

   if (Nodes == NULL)
   {
      os << "POINTS " << NumOfVertices << " double\n";
      for (int i = 0; i < NumOfVertices; i++)
      {
         os << vertices[i](0);
         int j;
         for (j = 1; j < spaceDim; j++)
         {
            os << ' ' << vertices[i](j);
         }
         for ( ; j < 3; j++)
         {
            os << ' ' << 0.0;
         }
         os << '\n';
      }
   }
   else
   {
      Array<int> vdofs(3);
      os << "POINTS " << Nodes->FESpace()->GetNDofs() << " double\n";
      for (int i = 0; i < Nodes->FESpace()->GetNDofs(); i++)
      {
         vdofs.SetSize(1);
         vdofs[0] = i;
         Nodes->FESpace()->DofsToVDofs(vdofs);
         os << (*Nodes)(vdofs[0]);
         int j;
         for (j = 1; j < spaceDim; j++)
         {
            os << ' ' << (*Nodes)(vdofs[j]);
         }
         for ( ; j < 3; j++)
         {
            os << ' ' << 0.0;
         }
         os << '\n';
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
      os << "CELLS " << NumOfElements << ' ' << size << '\n';
      for (int i = 0; i < NumOfElements; i++)
      {
         const int *v = elements[i]->GetVertices();
         const int nv = elements[i]->GetNVertices();
         os << nv;
         Geometry::Type geom = elements[i]->GetGeometryType();
         const int *perm = VTKGeometry::VertexPermutation[geom];
         for (int j = 0; j < nv; j++)
         {
            os << ' ' << v[perm ? perm[j] : j];
         }
         os << '\n';
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
      os << "CELLS " << NumOfElements << ' ' << size << '\n';
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
         os << dofs.Size();
         if (order == 1)
         {
            for (int j = 0; j < dofs.Size(); j++)
            {
               os << ' ' << dofs[j];
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
               os << ' ' << dofs[vtk_mfem[j]];
            }
         }
         os << '\n';
      }
   }

   os << "CELL_TYPES " << NumOfElements << '\n';
   for (int i = 0; i < NumOfElements; i++)
   {
      int vtk_cell_type = 5;
      Geometry::Type geom = GetElement(i)->GetGeometryType();
      if (order == 1) { vtk_cell_type = VTKGeometry::Map[geom]; }
      else if (order == 2) { vtk_cell_type = VTKGeometry::QuadraticMap[geom]; }
      os << vtk_cell_type << '\n';
   }

   // write attributes
   os << "CELL_DATA " << NumOfElements << '\n'
      << "SCALARS material int\n"
      << "LOOKUP_TABLE default\n";
   for (int i = 0; i < NumOfElements; i++)
   {
      os << elements[i]->GetAttribute() << '\n';
   }
   os.flush();
}

void Mesh::PrintVTU(std::string fname,
                    VTKFormat format,
                    bool high_order_output,
                    int compression_level,
                    bool bdr_elements)
{
   int ref = (high_order_output && Nodes)
             ? Nodes->FESpace()->GetMaxElementOrder() : 1;

   fname = fname + ".vtu";
   std::fstream os(fname.c_str(),std::ios::out);
   os << "<VTKFile type=\"UnstructuredGrid\" version=\"2.2\"";
   if (compression_level != 0)
   {
      os << " compressor=\"vtkZLibDataCompressor\"";
   }
   os << " byte_order=\"" << VTKByteOrder() << "\">\n";
   os << "<UnstructuredGrid>\n";
   PrintVTU(os, ref, format, high_order_output, compression_level, bdr_elements);
   os << "</Piece>\n"; // need to close the piece open in the PrintVTU method
   os << "</UnstructuredGrid>\n";
   os << "</VTKFile>" << std::endl;

   os.close();
}

void Mesh::PrintBdrVTU(std::string fname,
                       VTKFormat format,
                       bool high_order_output,
                       int compression_level)
{
   PrintVTU(fname, format, high_order_output, compression_level, true);
}

void Mesh::PrintVTU(std::ostream &os, int ref, VTKFormat format,
                    bool high_order_output, int compression_level,
                    bool bdr_elements)
{
   RefinedGeometry *RefG;
   DenseMatrix pmat;

   const char *fmt_str = (format == VTKFormat::ASCII) ? "ascii" : "binary";
   const char *type_str = (format != VTKFormat::BINARY32) ? "Float64" : "Float32";
   std::vector<char> buf;

   auto get_geom = [&](int i)
   {
      if (bdr_elements) { return GetBdrElementGeometry(i); }
      else { return GetElementBaseGeometry(i); }
   };

   int ne = bdr_elements ? GetNBE() : GetNE();
   // count the number of points and cells
   int np = 0, nc_ref = 0;
   for (int i = 0; i < ne; i++)
   {
      Geometry::Type geom = get_geom(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      np += RefG->RefPts.GetNPoints();
      nc_ref += RefG->RefGeoms.Size() / nv;
   }

   os << "<Piece NumberOfPoints=\"" << np << "\" NumberOfCells=\""
      << (high_order_output ? ne : nc_ref) << "\">\n";

   // print out the points
   os << "<Points>\n";
   os << "<DataArray type=\"" << type_str
      << "\" NumberOfComponents=\"3\" format=\"" << fmt_str << "\">\n";
   for (int i = 0; i < ne; i++)
   {
      RefG = GlobGeometryRefiner.Refine(get_geom(i), ref, 1);

      if (bdr_elements)
      {
         GetBdrElementTransformation(i)->Transform(RefG->RefPts, pmat);
      }
      else
      {
         GetElementTransformation(i)->Transform(RefG->RefPts, pmat);
      }

      for (int j = 0; j < pmat.Width(); j++)
      {
         WriteBinaryOrASCII(os, buf, pmat(0,j), " ", format);
         if (pmat.Height() > 1)
         {
            WriteBinaryOrASCII(os, buf, pmat(1,j), " ", format);
         }
         else
         {
            WriteBinaryOrASCII(os, buf, 0.0, " ", format);
         }
         if (pmat.Height() > 2)
         {
            WriteBinaryOrASCII(os, buf, pmat(2,j), "", format);
         }
         else
         {
            WriteBinaryOrASCII(os, buf, 0.0, "", format);
         }
         if (format == VTKFormat::ASCII) { os << '\n'; }
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>" << std::endl;
   os << "</Points>" << std::endl;

   os << "<Cells>" << std::endl;
   os << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\""
      << fmt_str << "\">" << std::endl;
   // connectivity
   std::vector<int> offset;

   np = 0;
   if (high_order_output)
   {
      Array<int> local_connectivity;
      for (int iel = 0; iel < ne; iel++)
      {
         Geometry::Type geom = get_geom(iel);
         CreateVTKElementConnectivity(local_connectivity, geom, ref);
         int nnodes = local_connectivity.Size();
         for (int i=0; i<nnodes; ++i)
         {
            WriteBinaryOrASCII(os, buf, np+local_connectivity[i], " ",
                               format);
         }
         if (format == VTKFormat::ASCII) { os << '\n'; }
         np += nnodes;
         offset.push_back(np);
      }
   }
   else
   {
      int coff = 0;
      for (int i = 0; i < ne; i++)
      {
         Geometry::Type geom = get_geom(i);
         int nv = Geometries.GetVertices(geom)->GetNPoints();
         RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
         Array<int> &RG = RefG->RefGeoms;
         for (int j = 0; j < RG.Size(); )
         {
            coff = coff+nv;
            offset.push_back(coff);
            const int *p = VTKGeometry::VertexPermutation[geom];
            for (int k = 0; k < nv; k++, j++)
            {
               WriteBinaryOrASCII(os, buf, np + RG[p ? (j - k + p[k]) : j], " ",
                                  format);
            }
            if (format == VTKFormat::ASCII) { os << '\n'; }
         }
         np += RefG->RefPts.GetNPoints();
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>" << std::endl;

   os << "<DataArray type=\"Int32\" Name=\"offsets\" format=\""
      << fmt_str << "\">" << std::endl;
   // offsets
   for (size_t ii=0; ii<offset.size(); ii++)
   {
      WriteBinaryOrASCII(os, buf, offset[ii], "\n", format);
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>" << std::endl;
   os << "<DataArray type=\"UInt8\" Name=\"types\" format=\""
      << fmt_str << "\">" << std::endl;
   // cell types
   const int *vtk_geom_map =
      high_order_output ? VTKGeometry::HighOrderMap : VTKGeometry::Map;
   for (int i = 0; i < ne; i++)
   {
      Geometry::Type geom = get_geom(i);
      uint8_t vtk_cell_type = 5;

      vtk_cell_type = vtk_geom_map[geom];

      if (high_order_output)
      {
         WriteBinaryOrASCII(os, buf, vtk_cell_type, "\n", format);
      }
      else
      {
         int nv = Geometries.GetVertices(geom)->GetNPoints();
         RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
         Array<int> &RG = RefG->RefGeoms;
         for (int j = 0; j < RG.Size(); j += nv)
         {
            WriteBinaryOrASCII(os, buf, vtk_cell_type, "\n", format);
         }
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>" << std::endl;
   os << "</Cells>" << std::endl;

   os << "<CellData Scalars=\"attribute\">" << std::endl;
   os << "<DataArray type=\"Int32\" Name=\"attribute\" format=\""
      << fmt_str << "\">" << std::endl;
   for (int i = 0; i < ne; i++)
   {
      int attr = bdr_elements ? GetBdrAttribute(i) : GetAttribute(i);
      if (high_order_output)
      {
         WriteBinaryOrASCII(os, buf, attr, "\n", format);
      }
      else
      {
         Geometry::Type geom = get_geom(i);
         int nv = Geometries.GetVertices(geom)->GetNPoints();
         RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
         for (int j = 0; j < RefG->RefGeoms.Size(); j += nv)
         {
            WriteBinaryOrASCII(os, buf, attr, "\n", format);
         }
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>" << std::endl;
   os << "</CellData>" << std::endl;
}


void Mesh::PrintVTK(std::ostream &os, int ref, int field_data)
{
   int np, nc, size;
   RefinedGeometry *RefG;
   DenseMatrix pmat;

   os <<
      "# vtk DataFile Version 3.0\n"
      "Generated by MFEM\n"
      "ASCII\n"
      "DATASET UNSTRUCTURED_GRID\n";

   // additional dataset information
   if (field_data)
   {
      os << "FIELD FieldData 1\n"
         << "MaterialIds " << 1 << " " << attributes.Size() << " int\n";
      for (int i = 0; i < attributes.Size(); i++)
      {
         os << ' ' << attributes[i];
      }
      os << '\n';
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
   os << "POINTS " << np << " double\n";
   // write the points
   for (int i = 0; i < GetNE(); i++)
   {
      RefG = GlobGeometryRefiner.Refine(
                GetElementBaseGeometry(i), ref, 1);

      GetElementTransformation(i)->Transform(RefG->RefPts, pmat);

      for (int j = 0; j < pmat.Width(); j++)
      {
         os << pmat(0, j) << ' ';
         if (pmat.Height() > 1)
         {
            os << pmat(1, j) << ' ';
            if (pmat.Height() > 2)
            {
               os << pmat(2, j);
            }
            else
            {
               os << 0.0;
            }
         }
         else
         {
            os << 0.0 << ' ' << 0.0;
         }
         os << '\n';
      }
   }

   // write the cells
   os << "CELLS " << nc << ' ' << size << '\n';
   np = 0;
   for (int i = 0; i < GetNE(); i++)
   {
      Geometry::Type geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      Array<int> &RG = RefG->RefGeoms;

      for (int j = 0; j < RG.Size(); )
      {
         os << nv;
         for (int k = 0; k < nv; k++, j++)
         {
            os << ' ' << np + RG[j];
         }
         os << '\n';
      }
      np += RefG->RefPts.GetNPoints();
   }
   os << "CELL_TYPES " << nc << '\n';
   for (int i = 0; i < GetNE(); i++)
   {
      Geometry::Type geom = GetElementBaseGeometry(i);
      int nv = Geometries.GetVertices(geom)->GetNPoints();
      RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
      Array<int> &RG = RefG->RefGeoms;
      int vtk_cell_type = VTKGeometry::Map[geom];

      for (int j = 0; j < RG.Size(); j += nv)
      {
         os << vtk_cell_type << '\n';
      }
   }
   // write attributes (materials)
   os << "CELL_DATA " << nc << '\n'
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
         os << attr << '\n';
      }
   }

   if (Dim > 1)
   {
      Array<int> coloring;
      srand((unsigned)time(0));
      real_t a = rand_real();
      int el0 = (int)floor(a * GetNE());
      GetElementColoring(coloring, el0);
      os << "SCALARS element_coloring int\n"
         << "LOOKUP_TABLE default\n";
      for (int i = 0; i < GetNE(); i++)
      {
         Geometry::Type geom = GetElementBaseGeometry(i);
         int nv = Geometries.GetVertices(geom)->GetNPoints();
         RefG = GlobGeometryRefiner.Refine(geom, ref, 1);
         for (int j = 0; j < RefG->RefGeoms.Size(); j += nv)
         {
            os << coloring[i] + 1 << '\n';
         }
      }
   }

   // prepare to write data
   os << "POINT_DATA " << np << '\n' << flush;
}

#ifdef MFEM_USE_HDF5

void Mesh::SaveVTKHDF(const std::string &fname, bool high_order)
{
#ifdef MFEM_USE_MPI
   if (ParMesh *pmesh = dynamic_cast<ParMesh*>(this))
   {
#ifdef MFEM_PARALLEL_HDF5
      VTKHDF vtkhdf(fname, pmesh->GetComm());
      vtkhdf.SaveMesh(*this, high_order);
      return;
#else
      MFEM_ABORT("Requires HDF5 library with parallel support enabled");
#endif
   }
#endif
   VTKHDF vtkhdf(fname);
   vtkhdf.SaveMesh(*this, high_order);
}

#endif

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

void Mesh::PrintWithPartitioning(int *partitioning, std::ostream &os,
                                 int elem_attr) const
{
   if (Dim != 3 && Dim != 2) { return; }

   int i, j, k, l, nv, nbe, *v;

   os << "MFEM mesh v1.0\n";

   // optional
   os <<
      "\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n"
      "# POINT       = 0\n"
      "# SEGMENT     = 1\n"
      "# TRIANGLE    = 2\n"
      "# SQUARE      = 3\n"
      "# TETRAHEDRON = 4\n"
      "# CUBE        = 5\n"
      "# PRISM       = 6\n"
      "#\n";

   os << "\ndimension\n" << Dim
      << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      os << int((elem_attr) ? partitioning[i]+1 : elements[i]->GetAttribute())
         << ' ' << elements[i]->GetGeometryType();
      nv = elements[i]->GetNVertices();
      v  = elements[i]->GetVertices();
      for (j = 0; j < nv; j++)
      {
         os << ' ' << v[j];
      }
      os << '\n';
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
   os << "\nboundary\n" << nbe << '\n';
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
            os << k+1 << ' ' << faces[i]->GetGeometryType();
            for (j = 0; j < nv; j++)
            {
               os << ' ' << v[j];
            }
            os << '\n';
            if (!Nonconforming() || !IsSlaveFace(faces_info[i]))
            {
               os << l+1 << ' ' << faces[i]->GetGeometryType();
               for (j = nv-1; j >= 0; j--)
               {
                  os << ' ' << v[j];
               }
               os << '\n';
            }
         }
      }
      else
      {
         k = partitioning[faces_info[i].Elem1No];
         nv = faces[i]->GetNVertices();
         v  = faces[i]->GetVertices();
         os << k+1 << ' ' << faces[i]->GetGeometryType();
         for (j = 0; j < nv; j++)
         {
            os << ' ' << v[j];
         }
         os << '\n';
      }
   }
   os << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      os << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         os << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            os << ' ' << vertices[i](j);
         }
         os << '\n';
      }
      os.flush();
   }
   else
   {
      os << "\nnodes\n";
      Nodes->Save(os);
   }
}

void Mesh::PrintElementsWithPartitioning(int *partitioning,
                                         std::ostream &os,
                                         int interior_faces)
{
   MFEM_ASSERT(Dim == spaceDim, "2D Manifolds not supported\n");
   if (Dim != 3 && Dim != 2) { return; }

   int *vcount = new int[NumOfVertices];
   for (int i = 0; i < NumOfVertices; i++)
   {
      vcount[i] = 0;
   }
   for (int i = 0; i < NumOfElements; i++)
   {
      int nv = elements[i]->GetNVertices();
      const int *ind = elements[i]->GetVertices();
      for (int j = 0; j < nv; j++)
      {
         vcount[ind[j]]++;
      }
   }

   int *voff = new int[NumOfVertices+1];
   voff[0] = 0;
   for (int i = 1; i <= NumOfVertices; i++)
   {
      voff[i] = vcount[i-1] + voff[i-1];
   }

   int **vown = new int*[NumOfVertices];
   for (int i = 0; i < NumOfVertices; i++)
   {
      vown[i] = new int[vcount[i]];
   }

   // 2D
   if (Dim == 2)
   {
      Table edge_el;
      Transpose(ElementToEdgeTable(), edge_el);

      // Fake printing of the elements.
      for (int i = 0; i < NumOfElements; i++)
      {
         int nv  = elements[i]->GetNVertices();
         const int *ind = elements[i]->GetVertices();
         for (int j = 0; j < nv; j++)
         {
            vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
      }

      for (int i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      int nbe = 0;
      for (int i = 0; i < edge_el.Size(); i++)
      {
         const int *el = edge_el.GetRow(i);
         if (edge_el.RowSize(i) > 1)
         {
            int k = partitioning[el[0]];
            int l = partitioning[el[1]];
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
      os << "areamesh2\n\n" << nbe << '\n';

      for (int i = 0; i < edge_el.Size(); i++)
      {
         const int *el = edge_el.GetRow(i);
         if (edge_el.RowSize(i) > 1)
         {
            int k = partitioning[el[0]];
            int l = partitioning[el[1]];
            if (interior_faces || k != l)
            {
               Array<int> ev;
               GetEdgeVertices(i,ev);
               os << k+1; // attribute
               for (int j = 0; j < 2; j++)
                  for (int s = 0; s < vcount[ev[j]]; s++)
                     if (vown[ev[j]][s] == el[0])
                     {
                        os << ' ' << voff[ev[j]]+s+1;
                     }
               os << '\n';
               os << l+1; // attribute
               for (int j = 1; j >= 0; j--)
                  for (int s = 0; s < vcount[ev[j]]; s++)
                     if (vown[ev[j]][s] == el[1])
                     {
                        os << ' ' << voff[ev[j]]+s+1;
                     }
               os << '\n';
            }
         }
         else
         {
            int k = partitioning[el[0]];
            Array<int> ev;
            GetEdgeVertices(i,ev);
            os << k+1; // attribute
            for (int j = 0; j < 2; j++)
               for (int s = 0; s < vcount[ev[j]]; s++)
                  if (vown[ev[j]][s] == el[0])
                  {
                     os << ' ' << voff[ev[j]]+s+1;
                  }
            os << '\n';
         }
      }

      // Print the elements.
      os << NumOfElements << '\n';
      for (int i = 0; i < NumOfElements; i++)
      {
         int nv  = elements[i]->GetNVertices();
         const int *ind = elements[i]->GetVertices();
         os << partitioning[i]+1 << ' '; // use subdomain number as attribute
         os << nv << ' ';
         for (int j = 0; j < nv; j++)
         {
            os << ' ' << voff[ind[j]]+vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
         os << '\n';
      }

      for (int i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      // Print the vertices.
      os << voff[NumOfVertices] << '\n';
      for (int i = 0; i < NumOfVertices; i++)
         for (int k = 0; k < vcount[i]; k++)
         {
            for (int j = 0; j < Dim; j++)
            {
               os << vertices[i](j) << ' ';
            }
            os << '\n';
         }
   }
   //  Dim is 3
   else if (meshgen == 1)
   {
      os << "NETGEN_Neutral_Format\n";
      // print the vertices
      os << voff[NumOfVertices] << '\n';
      for (int i = 0; i < NumOfVertices; i++)
         for (int k = 0; k < vcount[i]; k++)
         {
            for (int j = 0; j < Dim; j++)
            {
               os << ' ' << vertices[i](j);
            }
            os << '\n';
         }

      // print the elements
      os << NumOfElements << '\n';
      for (int i = 0; i < NumOfElements; i++)
      {
         int nv = elements[i]->GetNVertices();
         const int *ind = elements[i]->GetVertices();
         os << partitioning[i]+1; // use subdomain number as attribute
         for (int j = 0; j < nv; j++)
         {
            os << ' ' << voff[ind[j]]+vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
         os << '\n';
      }

      for (int i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      // print the boundary information.
      int nbe = 0;
      for (int i = 0; i < NumOfFaces; i++)
      {
         int l = faces_info[i].Elem2No;
         if (l >= 0)
         {
            int k = partitioning[faces_info[i].Elem1No];
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
      }

      os << nbe << '\n';
      for (int i = 0; i < NumOfFaces; i++)
      {
         int l = faces_info[i].Elem2No;
         if (l >= 0)
         {
            int k = partitioning[faces_info[i].Elem1No];
            l = partitioning[l];
            if (interior_faces || k != l)
            {
               int nv = faces[i]->GetNVertices();
               const int *ind = faces[i]->GetVertices();
               os << k+1; // attribute
               for (int j = 0; j < nv; j++)
                  for (int s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem1No)
                     {
                        os << ' ' << voff[ind[j]]+s+1;
                     }
               os << '\n';
               os << l+1; // attribute
               for (int j = nv-1; j >= 0; j--)
                  for (int s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem2No)
                     {
                        os << ' ' << voff[ind[j]]+s+1;
                     }
               os << '\n';
            }
         }
         else
         {
            int k = partitioning[faces_info[i].Elem1No];
            int nv = faces[i]->GetNVertices();
            const int *ind = faces[i]->GetVertices();
            os << k+1; // attribute
            for (int j = 0; j < nv; j++)
               for (int s = 0; s < vcount[ind[j]]; s++)
                  if (vown[ind[j]][s] == faces_info[i].Elem1No)
                  {
                     os << ' ' << voff[ind[j]]+s+1;
                  }
            os << '\n';
         }
      }
   }
   //  Dim is 3
   else if (meshgen == 2) // TrueGrid
   {
      // count the number of the boundary elements.
      int nbe = 0;
      for (int i = 0; i < NumOfFaces; i++)
      {
         int l = faces_info[i].Elem2No;
         if (l >= 0)
         {
            int k = partitioning[faces_info[i].Elem1No];
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
      }

      os << "TrueGrid\n"
         << "1 " << voff[NumOfVertices] << " " << NumOfElements
         << " 0 0 0 0 0 0 0\n"
         << "0 0 0 1 0 0 0 0 0 0 0\n"
         << "0 0 " << nbe << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
         << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
         << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

      for (int i = 0; i < NumOfVertices; i++)
         for (int k = 0; k < vcount[i]; k++)
            os << voff[i]+k << " 0.0 " << vertices[i](0) << ' '
               << vertices[i](1) << ' ' << vertices[i](2) << " 0.0\n";

      for (int i = 0; i < NumOfElements; i++)
      {
         int nv = elements[i]->GetNVertices();
         const int *ind = elements[i]->GetVertices();
         os << i+1 << ' ' << partitioning[i]+1; // partitioning as attribute
         for (int j = 0; j < nv; j++)
         {
            os << ' ' << voff[ind[j]]+vcount[ind[j]]--;
            vown[ind[j]][vcount[ind[j]]] = i;
         }
         os << '\n';
      }

      for (int i = 0; i < NumOfVertices; i++)
      {
         vcount[i] = voff[i+1] - voff[i];
      }

      // boundary elements
      for (int i = 0; i < NumOfFaces; i++)
      {
         int l = faces_info[i].Elem2No;
         if (l >= 0)
         {
            int k = partitioning[faces_info[i].Elem1No];
            l = partitioning[l];
            if (interior_faces || k != l)
            {
               int nv = faces[i]->GetNVertices();
               const int *ind = faces[i]->GetVertices();
               os << k+1; // attribute
               for (int j = 0; j < nv; j++)
                  for (int s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem1No)
                     {
                        os << ' ' << voff[ind[j]]+s+1;
                     }
               os << " 1.0 1.0 1.0 1.0\n";
               os << l+1; // attribute
               for (int j = nv-1; j >= 0; j--)
                  for (int s = 0; s < vcount[ind[j]]; s++)
                     if (vown[ind[j]][s] == faces_info[i].Elem2No)
                     {
                        os << ' ' << voff[ind[j]]+s+1;
                     }
               os << " 1.0 1.0 1.0 1.0\n";
            }
         }
         else
         {
            int k = partitioning[faces_info[i].Elem1No];
            int nv = faces[i]->GetNVertices();
            const int *ind = faces[i]->GetVertices();
            os << k+1; // attribute
            for (int j = 0; j < nv; j++)
               for (int s = 0; s < vcount[ind[j]]; s++)
                  if (vown[ind[j]][s] == faces_info[i].Elem1No)
                  {
                     os << ' ' << voff[ind[j]]+s+1;
                  }
            os << " 1.0 1.0 1.0 1.0\n";
         }
      }
   }

   os << flush;

   for (int i = 0; i < NumOfVertices; i++)
   {
      delete [] vown[i];
   }

   delete [] vcount;
   delete [] voff;
   delete [] vown;
}

void Mesh::PrintSurfaces(const Table & Aface_face, std::ostream &os) const
{
   int i, j;

   if (NURBSext)
   {
      mfem_error("Mesh::PrintSurfaces"
                 " NURBS mesh is not supported!");
      return;
   }

   os << "MFEM mesh v1.0\n";

   // optional
   os <<
      "\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n"
      "# POINT       = 0\n"
      "# SEGMENT     = 1\n"
      "# TRIANGLE    = 2\n"
      "# SQUARE      = 3\n"
      "# TETRAHEDRON = 4\n"
      "# CUBE        = 5\n"
      "# PRISM       = 6\n"
      "#\n";

   os << "\ndimension\n" << Dim
      << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], os);
   }

   os << "\nboundary\n" << Aface_face.Size_of_connections() << '\n';
   const int * const i_AF_f = Aface_face.GetI();
   const int * const j_AF_f = Aface_face.GetJ();

   for (int iAF=0; iAF < Aface_face.Size(); ++iAF)
      for (const int * iface = j_AF_f + i_AF_f[iAF];
           iface < j_AF_f + i_AF_f[iAF+1];
           ++iface)
      {
         os << iAF+1 << ' ';
         PrintElementWithoutAttr(faces[*iface],os);
      }

   os << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      os << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         os << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            os << ' ' << vertices[i](j);
         }
         os << '\n';
      }
      os.flush();
   }
   else
   {
      os << "\nnodes\n";
      Nodes->Save(os);
   }
}

void Mesh::ScaleSubdomains(real_t sf)
{
   int i,j,k;
   Array<int> vert;
   DenseMatrix pointmat;
   int na = attributes.Size();
   real_t *cg = new real_t[na*spaceDim];
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

void Mesh::ScaleElements(real_t sf)
{
   int i,j,k;
   Array<int> vert;
   DenseMatrix pointmat;
   int na = NumOfElements;
   real_t *cg = new real_t[na*spaceDim];
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

void Mesh::Transform(std::function<void(const Vector &, Vector&)> f)
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
         f(vold, vnew);
      }
   }
   else
   {
      GridFunction xnew(Nodes->FESpace());
      VectorFunctionCoefficient f_pert(spaceDim, f);
      xnew.ProjectCoefficient(f_pert);
      *Nodes = xnew;
   }
   NodesUpdated();
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
   NodesUpdated();
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
      // generate el_to_edge, be_to_face (2D), bel_to_edge (3D)
      el_to_edge = new Table;
      NumOfEdges = GetElementToEdgeTable(*el_to_edge);
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
      if (FaceIsInterior(GetBdrElementFaceIndex(i)))
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
   Array<int> new_be_to_face;
   Table *new_bel_to_edge = NULL;
   new_boundary.SetSize(0);
   new_be_to_face.Reserve(num_bdr_elem);
   if (Dim == 3)
   {
      new_bel_to_edge = new Table;
      new_bel_to_edge->SetDims(num_bdr_elem, new_bel_to_edge_nnz);
   }
   for (int i = 0; i < GetNBE(); i++)
   {
      if (!FaceIsInterior(GetBdrElementFaceIndex(i)))
      {
         new_boundary.Append(boundary[i]);
         int row = new_be_to_face.Size();
         new_be_to_face.Append(be_to_face[i]);
         if (Dim == 3)
         {
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

   mfem::Swap(be_to_face, new_be_to_face);

   if (Dim == 3)
   {
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

std::ostream &operator<<(std::ostream &os, const Mesh &mesh)
{
   mesh.Print(os);
   return os;
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

   real_t *data = point_mat.GetData();
   InverseElementTransformation *inv_tr = inv_trans;
   inv_tr = inv_tr ? inv_tr : new InverseElementTransformation;

   // For each point in 'point_mat', find the element whose center is closest.
   Vector min_dist(npts);
   Array<int> e_idx(npts);
   min_dist = std::numeric_limits<real_t>::max();
   e_idx = -1;

   Vector pt(spaceDim);
   for (int i = 0; i < GetNE(); i++)
   {
      GetElementTransformation(i)->Transform(
         Geometries.GetCenter(GetElementBaseGeometry(i)), pt);
      for (int k = 0; k < npts; k++)
      {
         real_t dist = pt.DistanceTo(data+k*spaceDim);
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
      Array<int> elvertices;
      Table *vtoel = GetVertexToElementTable();
      for (int k = 0; k < npts; k++)
      {
         if (elem_ids[k] != -1) { continue; }
         // Try all vertex-neighbors of element e_idx[k]
         pt.SetData(data+k*spaceDim);
         GetElementVertices(e_idx[k], elvertices);
         for (int v = 0; v < elvertices.Size(); v++)
         {
            int vv = elvertices[v];
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
         // Try neighbors for non-conforming meshes
         if (ncmesh)
         {
            Array<int> neigh;
            int le = ncmesh->leaf_elements[e_idx[k]];
            ncmesh->FindNeighbors(le,neigh);
            for (int e = 0; e < neigh.Size(); e++)
            {
               int nn = neigh[e];
               if (ncmesh->IsGhost(ncmesh->elements[nn])) { continue; }
               int el = ncmesh->elements[nn].index;
               inv_tr->SetTransformation(*GetElementTransformation(el));
               int res = inv_tr->Transform(pt, ips[k]);
               if (res == InverseElementTransformation::Inside)
               {
                  elem_ids[k] = el;
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

void Mesh::GetGeometricParametersFromJacobian(const DenseMatrix &J,
                                              real_t &volume,
                                              Vector &aspr,
                                              Vector &skew,
                                              Vector &ori) const
{
   J.HostRead();
   aspr.HostWrite();
   skew.HostWrite();
   ori.HostWrite();
   MFEM_VERIFY(Dim == 2 || Dim == 3, "Only 2D/3D meshes supported right now.");
   MFEM_VERIFY(Dim == spaceDim, "Surface meshes not currently supported.");
   if (Dim == 2)
   {
      aspr.SetSize(1);
      skew.SetSize(1);
      ori.SetSize(1);
      Vector col1, col2;
      J.GetColumn(0, col1);
      J.GetColumn(1, col2);

      // Area/Volume
      volume = J.Det();

      // Aspect-ratio
      aspr(0) = col2.Norml2()/col1.Norml2();

      // Skewness
      skew(0) = std::atan2(J.Det(), col1 * col2);

      // Orientation
      ori(0) = std::atan2(J(1,0), J(0,0));
   }
   else if (Dim == 3)
   {
      aspr.SetSize(4);
      skew.SetSize(3);
      ori.SetSize(4);
      Vector col1, col2, col3;
      J.GetColumn(0, col1);
      J.GetColumn(1, col2);
      J.GetColumn(2, col3);
      real_t len1 = col1.Norml2(),
             len2 = col2.Norml2(),
             len3 = col3.Norml2();

      Vector col1unit = col1,
             col2unit = col2,
             col3unit = col3;
      col1unit *= 1.0/len1;
      col2unit *= 1.0/len2;
      col3unit *= 1.0/len3;

      // Area/Volume
      volume = J.Det();

      // Aspect-ratio - non-dimensional
      aspr(0) = len1/std::sqrt(len2*len3),
      aspr(1) = len2/std::sqrt(len1*len3);

      // Aspect-ratio - dimensional - needed for TMOP
      aspr(2) = std::sqrt(len1/(len2*len3)),
      aspr(3) = std::sqrt(len2/(len1*len3));

      // Skewness
      Vector crosscol12, crosscol13;
      col1.cross3D(col2, crosscol12);
      col1.cross3D(col3, crosscol13);
      skew(0) = std::acos(col1unit*col2unit);
      skew(1) = std::acos(col1unit*col3unit);
      skew(2) = std::atan(len1*volume/(crosscol12*crosscol13));

      // Orientation
      // First we define the rotation matrix
      DenseMatrix rot(Dim);
      // First column
      for (int d=0; d<Dim; d++) { rot(d, 0) = col1unit(d); }
      // Second column
      Vector rot2 = col2unit;
      Vector rot1 = col1unit;
      rot1 *= col1unit*col2unit;
      rot2 -= rot1;
      col1unit.cross3D(col2unit, rot1);
      rot2 /= rot1.Norml2();
      for (int d=0; d < Dim; d++) { rot(d, 1) = rot2(d); }
      // Third column
      rot1 /= rot1.Norml2();
      for (int d=0; d < Dim; d++) { rot(d, 2) = rot1(d); }
      real_t delta = sqrt(pow(rot(2,1)-rot(1,2), 2.0) +
                          pow(rot(0,2)-rot(2,0), 2.0) +
                          pow(rot(1,0)-rot(0,1), 2.0));
      ori = 0.0;
      if (delta == 0.0)   // Matrix is symmetric. Check if it is Identity.
      {
         DenseMatrix Iden(Dim);
         for (int d = 0; d < Dim; d++) { Iden(d, d) = 1.0; };
         Iden -= rot;
         if (Iden.FNorm2() != 0)
         {
            // TODO: Handling of these cases.
            rot.Print();
            MFEM_ABORT("Invalid rotation matrix. Contact TMOP Developers.");
         }
      }
      else
      {
         ori(0) = (1./delta)*(rot(2,1)-rot(1,2));
         ori(1) = (1./delta)*(rot(0,2)-rot(2,0));
         ori(2) = (1./delta)*(rot(1,0)-rot(0,1));
         ori(3) = std::acos(0.5*(rot.Trace()-1.0));
      }
   }
}


MeshPart::EntityHelper::EntityHelper(
   int dim_, const Array<int> (&entity_to_vertex_)[Geometry::NumGeom])
   : dim(dim_),
     entity_to_vertex(entity_to_vertex_)
{
   int geom_offset = 0;
   for (int g = Geometry::DimStart[dim]; g < Geometry::DimStart[dim+1]; g++)
   {
      geom_offsets[g] = geom_offset;
      geom_offset += entity_to_vertex[g].Size()/Geometry::NumVerts[g];
   }
   geom_offsets[Geometry::DimStart[dim+1]] = geom_offset;
   num_entities = geom_offset;
}

MeshPart::Entity MeshPart::EntityHelper::FindEntity(int bytype_entity_id)
{
   // Find the 'geom' that corresponds to 'bytype_entity_id'
   int geom = Geometry::DimStart[dim];
   while (geom_offsets[geom+1] <= bytype_entity_id) { geom++; }
   MFEM_ASSERT(geom < Geometry::NumGeom, "internal error");
   MFEM_ASSERT(Geometry::Dimension[geom] == dim, "internal error");
   const int nv = Geometry::NumVerts[geom];
   const int geom_elem_id = bytype_entity_id - geom_offsets[geom];
   const int *v = &entity_to_vertex[geom][nv*geom_elem_id];
   return { geom, nv, v };
}

void MeshPart::Print(std::ostream &os) const
{
   os << "MFEM mesh v1.2\n";

   // optional
   os <<
      "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
      "# POINT       = 0\n"
      "# SEGMENT     = 1\n"
      "# TRIANGLE    = 2\n"
      "# SQUARE      = 3\n"
      "# TETRAHEDRON = 4\n"
      "# CUBE        = 5\n"
      "# PRISM       = 6\n"
      "# PYRAMID     = 7\n"
      "#\n";

   const int dim = dimension;
   os << "\ndimension\n" << dim;

   os << "\n\nelements\n" << num_elements << '\n';
   {
      const bool have_element_map = (element_map.Size() == num_elements);
      MFEM_ASSERT(have_element_map || element_map.Size() == 0,
                  "invalid MeshPart state");
      EntityHelper elem_helper(dim, entity_to_vertex);
      MFEM_ASSERT(elem_helper.num_entities == num_elements,
                  "invalid MeshPart state");
      for (int nat_elem_id = 0; nat_elem_id < num_elements; nat_elem_id++)
      {
         const int bytype_elem_id = have_element_map ?
                                    element_map[nat_elem_id] : nat_elem_id;
         const Entity ent = elem_helper.FindEntity(bytype_elem_id);
         // Print the element
         os << attributes[nat_elem_id] << ' ' << ent.geom;
         for (int i = 0; i < ent.num_verts; i++)
         {
            os << ' ' << ent.verts[i];
         }
         os << '\n';
      }
   }

   os << "\nboundary\n" << num_bdr_elements << '\n';
   {
      const bool have_boundary_map = (boundary_map.Size() == num_bdr_elements);
      MFEM_ASSERT(have_boundary_map || boundary_map.Size() == 0,
                  "invalid MeshPart state");
      EntityHelper bdr_helper(dim-1, entity_to_vertex);
      MFEM_ASSERT(bdr_helper.num_entities == num_bdr_elements,
                  "invalid MeshPart state");
      for (int nat_bdr_id = 0; nat_bdr_id < num_bdr_elements; nat_bdr_id++)
      {
         const int bytype_bdr_id = have_boundary_map ?
                                   boundary_map[nat_bdr_id] : nat_bdr_id;
         const Entity ent = bdr_helper.FindEntity(bytype_bdr_id);
         // Print the boundary element
         os << bdr_attributes[nat_bdr_id] << ' ' << ent.geom;
         for (int i = 0; i < ent.num_verts; i++)
         {
            os << ' ' << ent.verts[i];
         }
         os << '\n';
      }
   }

   os << "\nvertices\n" << num_vertices << '\n';
   if (!nodes)
   {
      const int sdim = space_dimension;
      os << sdim << '\n';
      for (int i = 0; i < num_vertices; i++)
      {
         os << vertex_coordinates[i*sdim];
         for (int d = 1; d < sdim; d++)
         {
            os << ' ' << vertex_coordinates[i*sdim+d];
         }
         os << '\n';
      }
   }
   else
   {
      os << "\nnodes\n";
      nodes->Save(os);
   }

   os << "\nmfem_serial_mesh_end\n";

   // Start: GroupTopology::Save
   const int num_groups = my_groups.Size();
   os << "\ncommunication_groups\n";
   os << "number_of_groups " << num_groups << "\n\n";

   os << "# number of entities in each group, followed by ranks in group\n";
   for (int group_id = 0; group_id < num_groups; ++group_id)
   {
      const int group_size = my_groups.RowSize(group_id);
      const int *group_ptr = my_groups.GetRow(group_id);
      os << group_size;
      for (int group_member_index = 0; group_member_index < group_size;
           ++group_member_index)
      {
         os << ' ' << group_ptr[group_member_index];
      }
      os << '\n';
   }
   // End: GroupTopology::Save

   const Table &g2v  = group_shared_entity_to_vertex[Geometry::POINT];
   const Table &g2ev = group_shared_entity_to_vertex[Geometry::SEGMENT];
   const Table &g2tv = group_shared_entity_to_vertex[Geometry::TRIANGLE];
   const Table &g2qv = group_shared_entity_to_vertex[Geometry::SQUARE];

   MFEM_VERIFY(g2v.RowSize(0) == 0, "internal erroor");
   os << "\ntotal_shared_vertices " << g2v.Size_of_connections() << '\n';
   if (dimension >= 2)
   {
      MFEM_VERIFY(g2ev.RowSize(0) == 0, "internal erroor");
      os << "total_shared_edges " << g2ev.Size_of_connections()/2 << '\n';
   }
   if (dimension >= 3)
   {
      MFEM_VERIFY(g2tv.RowSize(0) == 0, "internal erroor");
      MFEM_VERIFY(g2qv.RowSize(0) == 0, "internal erroor");
      const int total_shared_faces =
         g2tv.Size_of_connections()/3 + g2qv.Size_of_connections()/4;
      os << "total_shared_faces " << total_shared_faces << '\n';
   }
   os << "\n# group 0 has no shared entities\n";
   for (int gr = 1; gr < num_groups; gr++)
   {
      {
         const int  nv = g2v.RowSize(gr);
         const int *sv = g2v.GetRow(gr);
         os << "\n# group " << gr << "\nshared_vertices " << nv << '\n';
         for (int i = 0; i < nv; i++)
         {
            os << sv[i] << '\n';
         }
      }
      if (dimension >= 2)
      {
         const int  ne = g2ev.RowSize(gr)/2;
         const int *se = g2ev.GetRow(gr);
         os << "\nshared_edges " << ne << '\n';
         for (int i = 0; i < ne; i++)
         {
            const int *v = se + 2*i;
            os << v[0] << ' ' << v[1] << '\n';
         }
      }
      if (dimension >= 3)
      {
         const int  nt = g2tv.RowSize(gr)/3;
         const int *st = g2tv.GetRow(gr);
         const int  nq = g2qv.RowSize(gr)/4;
         const int *sq = g2qv.GetRow(gr);
         os << "\nshared_faces " << nt+nq << '\n';
         for (int i = 0; i < nt; i++)
         {
            os << Geometry::TRIANGLE;
            const int *v = st + 3*i;
            for (int j = 0; j < 3; j++) { os << ' ' << v[j]; }
            os << '\n';
         }
         for (int i = 0; i < nq; i++)
         {
            os << Geometry::SQUARE;
            const int *v = sq + 4*i;
            for (int j = 0; j < 4; j++) { os << ' ' << v[j]; }
            os << '\n';
         }
      }
   }

   // Write out section end tag for mesh.
   os << "\nmfem_mesh_end" << endl;
}

Mesh &MeshPart::GetMesh()
{
   if (mesh) { return *mesh; }

   mesh.reset(new Mesh(dimension,
                       num_vertices,
                       num_elements,
                       num_bdr_elements,
                       space_dimension));

   // Add elements
   {
      const bool have_element_map = (element_map.Size() == num_elements);
      MFEM_ASSERT(have_element_map || element_map.Size() == 0,
                  "invalid MeshPart state");
      EntityHelper elem_helper(dimension, entity_to_vertex);
      MFEM_ASSERT(elem_helper.num_entities == num_elements,
                  "invalid MeshPart state");
      const bool have_tet_refine_flags = (tet_refine_flags.Size() > 0);
      for (int nat_elem_id = 0; nat_elem_id < num_elements; nat_elem_id++)
      {
         const int bytype_elem_id = have_element_map ?
                                    element_map[nat_elem_id] : nat_elem_id;
         const Entity ent = elem_helper.FindEntity(bytype_elem_id);
         Element *el = mesh->NewElement(ent.geom);
         el->SetVertices(ent.verts);
         el->SetAttribute(attributes[nat_elem_id]);
         if (ent.geom == Geometry::TETRAHEDRON && have_tet_refine_flags)
         {
            constexpr int geom_tet = Geometry::TETRAHEDRON;
            const int tet_id = (ent.verts - entity_to_vertex[geom_tet])/4;
            const int ref_flag = tet_refine_flags[tet_id];
            static_cast<Tetrahedron*>(el)->SetRefinementFlag(ref_flag);
         }
         mesh->AddElement(el);
      }
   }

   // Add boundary elements
   {
      const bool have_boundary_map = (boundary_map.Size() == num_bdr_elements);
      MFEM_ASSERT(have_boundary_map || boundary_map.Size() == 0,
                  "invalid MeshPart state");
      EntityHelper bdr_helper(dimension-1, entity_to_vertex);
      MFEM_ASSERT(bdr_helper.num_entities == num_bdr_elements,
                  "invalid MeshPart state");
      for (int nat_bdr_id = 0; nat_bdr_id < num_bdr_elements; nat_bdr_id++)
      {
         const int bytype_bdr_id = have_boundary_map ?
                                   boundary_map[nat_bdr_id] : nat_bdr_id;
         const Entity ent = bdr_helper.FindEntity(bytype_bdr_id);
         Element *bdr = mesh->NewElement(ent.geom);
         bdr->SetVertices(ent.verts);
         bdr->SetAttribute(bdr_attributes[nat_bdr_id]);
         mesh->AddBdrElement(bdr);
      }
   }

   // Add vertices
   if (vertex_coordinates.Size() == space_dimension*num_vertices)
   {
      MFEM_ASSERT(!nodes, "invalid MeshPart state");
      for (int vert_id = 0; vert_id < num_vertices; vert_id++)
      {
         mesh->AddVertex(vertex_coordinates + space_dimension*vert_id);
      }
   }
   else
   {
      MFEM_ASSERT(vertex_coordinates.Size() == 0, "invalid MeshPart state");
      for (int vert_id = 0; vert_id < num_vertices; vert_id++)
      {
         mesh->AddVertex(0., 0., 0.);
      }
      // 'mesh.Nodes' cannot be set here -- they can be set later, if needed
   }

   mesh->FinalizeTopology(/* generate_bdr: */ false);

   return *mesh;
}


MeshPartitioner::MeshPartitioner(Mesh &mesh_,
                                 int num_parts_,
                                 const int *partitioning_,
                                 int part_method)
   : mesh(mesh_)
{
   if (partitioning_)
   {
      partitioning.MakeRef(const_cast<int *>(partitioning_), mesh.GetNE(),
                           false);
   }
   else
   {
      // Mesh::GeneratePartitioning always uses new[] to allocate the,
      // partitioning, so we need to tell the memory manager to free it with
      // delete[] (even if a different host memory type has been selected).
      constexpr MemoryType mt = MemoryType::HOST;
      partitioning.MakeRef(mesh.GeneratePartitioning(num_parts_, part_method),
                           mesh.GetNE(), mt, true);
   }

   Transpose(partitioning, part_to_element, num_parts_);
   // Note: the element ids in each row of 'part_to_element' are sorted.

   const int dim = mesh.Dimension();
   if (dim >= 2)
   {
      Transpose(mesh.ElementToEdgeTable(), edge_to_element, mesh.GetNEdges());
   }

   Array<int> boundary_to_part(mesh.GetNBE());
   // Same logic as in ParMesh::BuildLocalBoundary
   if (dim >= 3)
   {
      for (int i = 0; i < boundary_to_part.Size(); i++)
      {
         int face, o, el1, el2;
         mesh.GetBdrElementFace(i, &face, &o);
         mesh.GetFaceElements(face, &el1, &el2);
         boundary_to_part[i] =
            partitioning[(o % 2 == 0 || el2 < 0) ? el1 : el2];
      }
   }
   else if (dim == 2)
   {
      for (int i = 0; i < boundary_to_part.Size(); i++)
      {
         int edge = mesh.GetBdrElementFaceIndex(i);
         int el1 = edge_to_element.GetRow(edge)[0];
         boundary_to_part[i] = partitioning[el1];
      }
   }
   else if (dim == 1)
   {
      for (int i = 0; i < boundary_to_part.Size(); i++)
      {
         int vert = mesh.GetBdrElementFaceIndex(i);
         int el1, el2;
         mesh.GetFaceElements(vert, &el1, &el2);
         boundary_to_part[i] = partitioning[el1];
      }
   }
   Transpose(boundary_to_part, part_to_boundary, num_parts_);
   // Note: the boundary element ids in each row of 'part_to_boundary' are
   // sorted.
   boundary_to_part.DeleteAll();

   Table *vert_element = mesh.GetVertexToElementTable(); // we must delete this
   vertex_to_element.Swap(*vert_element);
   delete vert_element;
}

void MeshPartitioner::ExtractPart(int part_id, MeshPart &mesh_part) const
{
   const int num_parts = part_to_element.Size();

   MFEM_VERIFY(0 <= part_id && part_id < num_parts,
               "invalid part_id = " << part_id
               << ", num_parts = " << num_parts);

   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   const int num_elems = part_to_element.RowSize(part_id);
   const int *elem_list = part_to_element.GetRow(part_id); // sorted
   const int num_bdr_elems = part_to_boundary.RowSize(part_id);
   const int *bdr_elem_list = part_to_boundary.GetRow(part_id); // sorted

   // Initialize 'mesh_part'
   mesh_part.dimension = dim;
   mesh_part.space_dimension = sdim;
   mesh_part.num_vertices = 0;
   mesh_part.num_elements = num_elems;
   mesh_part.num_bdr_elements = num_bdr_elems;
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      mesh_part.entity_to_vertex[g].SetSize(0); // can reuse Array allocation
   }
   mesh_part.tet_refine_flags.SetSize(0);
   mesh_part.element_map.SetSize(0); // 0 or 'num_elements', if needed
   mesh_part.boundary_map.SetSize(0); // 0 or 'num_bdr_elements', if needed
   mesh_part.attributes.SetSize(num_elems);
   mesh_part.bdr_attributes.SetSize(num_bdr_elems);
   mesh_part.vertex_coordinates.SetSize(0);

   mesh_part.num_parts = num_parts;
   mesh_part.my_part_id = part_id;
   mesh_part.my_groups.Clear();
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      mesh_part.group_shared_entity_to_vertex[g].Clear();
   }
   mesh_part.nodes.reset(nullptr);
   mesh_part.nodal_fes.reset(nullptr);
   mesh_part.mesh.reset(nullptr);

   // Initialize:
   // - 'mesh_part.entity_to_vertex' for the elements (boundary elements are
   //   set later); vertex ids are global at this point - they will be mapped to
   //   local ids later
   // - 'mesh_part.attributes'
   // - 'mesh_part.tet_refine_flags' if needed
   int geom_marker = 0, num_geom = 0;
   for (int i = 0; i < num_elems; i++)
   {
      const Element *elem = mesh.GetElement(elem_list[i]);
      const int geom = elem->GetGeometryType();
      const int nv = Geometry::NumVerts[geom];
      const int *v = elem->GetVertices();
      MFEM_VERIFY(numeric_limits<int>::max() - nv >=
                  mesh_part.entity_to_vertex[geom].Size(),
                  "overflow in 'entity_to_vertex[geom]', geom: "
                  << Geometry::Name[geom]);
      mesh_part.entity_to_vertex[geom].Append(v, nv);
      mesh_part.attributes[i] = elem->GetAttribute();
      if (geom == Geometry::TETRAHEDRON)
      {
         // Create 'mesh_part.tet_refine_flags' but only if we find at least one
         // non-zero flag in a tetrahedron.
         const Tetrahedron *tet = static_cast<const Tetrahedron*>(elem);
         const int ref_flag = tet->GetRefinementFlag();
         if (mesh_part.tet_refine_flags.Size() == 0)
         {
            if (ref_flag)
            {
               // This is the first time we encounter non-zero 'ref_flag'
               const int num_tets = mesh_part.entity_to_vertex[geom].Size()/nv;
               mesh_part.tet_refine_flags.SetSize(num_tets, 0);
               mesh_part.tet_refine_flags.Last() = ref_flag;
            }
         }
         else
         {
            mesh_part.tet_refine_flags.Append(ref_flag);
         }
      }
      if ((geom_marker & (1 << geom)) == 0)
      {
         geom_marker |= (1 << geom);
         num_geom++;
      }
   }
   MFEM_ASSERT(mesh_part.tet_refine_flags.Size() == 0 ||
               mesh_part.tet_refine_flags.Size() ==
               mesh_part.entity_to_vertex[Geometry::TETRAHEDRON].Size()/4,
               "internal error");
   // Initialize 'mesh_part.element_map' if needed
   if (num_geom > 1)
   {
      int offsets[Geometry::NumGeom];
      int offset = 0;
      for (int g = Geometry::DimStart[dim]; g < Geometry::DimStart[dim+1]; g++)
      {
         offsets[g] = offset;
         offset += mesh_part.entity_to_vertex[g].Size()/Geometry::NumVerts[g];
      }
      mesh_part.element_map.SetSize(num_elems);
      for (int i = 0; i < num_elems; i++)
      {
         const int geom = mesh.GetElementGeometry(elem_list[i]);
         mesh_part.element_map[i] = offsets[geom]++;
      }
   }

   // Initialize:
   // - 'mesh_part.entity_to_vertex' for the boundary elements; vertex ids are
   //   global at this point - they will be mapped to local ids later
   // - 'mesh_part.bdr_attributes'
   geom_marker = 0; num_geom = 0;
   for (int i = 0; i < num_bdr_elems; i++)
   {
      const Element *bdr_elem = mesh.GetBdrElement(bdr_elem_list[i]);
      const int geom = bdr_elem->GetGeometryType();
      const int nv = Geometry::NumVerts[geom];
      const int *v = bdr_elem->GetVertices();
      MFEM_VERIFY(numeric_limits<int>::max() - nv >=
                  mesh_part.entity_to_vertex[geom].Size(),
                  "overflow in 'entity_to_vertex[geom]', geom: "
                  << Geometry::Name[geom]);
      mesh_part.entity_to_vertex[geom].Append(v, nv);
      mesh_part.bdr_attributes[i] = bdr_elem->GetAttribute();
      if ((geom_marker & (1 << geom)) == 0)
      {
         geom_marker |= (1 << geom);
         num_geom++;
      }
   }
   // Initialize 'mesh_part.boundary_map' if needed
   if (num_geom > 1)
   {
      int offsets[Geometry::NumGeom];
      int offset = 0;
      for (int g = Geometry::DimStart[dim-1]; g < Geometry::DimStart[dim]; g++)
      {
         offsets[g] = offset;
         offset += mesh_part.entity_to_vertex[g].Size()/Geometry::NumVerts[g];
      }
      mesh_part.boundary_map.SetSize(num_bdr_elems);
      for (int i = 0; i < num_bdr_elems; i++)
      {
         const int geom = mesh.GetBdrElementGeometry(bdr_elem_list[i]);
         mesh_part.boundary_map[i] = offsets[geom]++;
      }
   }

   // Create the vertex id map, 'vertex_loc_to_glob', which maps local ids to
   // global ones; the map is sorted, preserving the global ordering.
   Array<int> vertex_loc_to_glob;
   {
      std::unordered_set<int> vertex_set;
      for (int i = 0; i < num_elems; i++)
      {
         const Element *elem = mesh.GetElement(elem_list[i]);
         const int geom = elem->GetGeometryType();
         const int nv = Geometry::NumVerts[geom];
         const int *v = elem->GetVertices();
         vertex_set.insert(v, v + nv);
      }
      vertex_loc_to_glob.SetSize(static_cast<int>(vertex_set.size()));
      std::copy(vertex_set.begin(), vertex_set.end(), // src
                vertex_loc_to_glob.begin());          // dest
   }
   vertex_loc_to_glob.Sort();

   // Initialize 'mesh_part.num_vertices'
   mesh_part.num_vertices = vertex_loc_to_glob.Size();

   // Update the vertex ids in the arrays 'mesh_part.entity_to_vertex' from
   // global to local.
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      Array<int> &vert_array = mesh_part.entity_to_vertex[g];
      for (int i = 0; i < vert_array.Size(); i++)
      {
         const int glob_id = vert_array[i];
         const int loc_id = vertex_loc_to_glob.FindSorted(glob_id);
         MFEM_ASSERT(loc_id >= 0, "internal error: global vertex id not found");
         vert_array[i] = loc_id;
      }
   }

   // Initialize one of 'mesh_part.vertex_coordinates' or 'mesh_part.nodes'
   if (!mesh.GetNodes())
   {
      MFEM_VERIFY(numeric_limits<int>::max()/sdim >= vertex_loc_to_glob.Size(),
                  "overflow in 'vertex_coordinates', num_vertices = "
                  << vertex_loc_to_glob.Size() << ", sdim = " << sdim);
      mesh_part.vertex_coordinates.SetSize(sdim*vertex_loc_to_glob.Size());
      for (int i = 0; i < vertex_loc_to_glob.Size(); i++)
      {
         const real_t *coord = mesh.GetVertex(vertex_loc_to_glob[i]);
         for (int d = 0; d < sdim; d++)
         {
            mesh_part.vertex_coordinates[i*sdim+d] = coord[d];
         }
      }
   }
   else
   {
      const GridFunction &glob_nodes = *mesh.GetNodes();
      mesh_part.nodal_fes = ExtractFESpace(mesh_part, *glob_nodes.FESpace());
      // Initialized 'mesh_part.mesh'.
      // Note: the nodes of 'mesh_part.mesh' are not set.

      mesh_part.nodes = ExtractGridFunction(mesh_part, glob_nodes,
                                            *mesh_part.nodal_fes);

      // Attach the 'mesh_part.nodes' to the 'mesh_part.mesh'.
      mesh_part.mesh->NewNodes(*mesh_part.nodes, /* make_owner: */ false);
      // Note: the vertices of 'mesh_part.mesh' are not set.
   }

   // Begin constructing the "neighbor" groups, i.e. the groups that contain
   // 'part_id'.
   ListOfIntegerSets groups;
   {
      // the first group is the local one
      IntegerSet group;
      group.Recreate(1, &part_id);
      groups.Insert(group);
   }

   // 'shared_faces' : shared face id -> (global_face_id, group_id)
   // Note: 'shared_faces' will be sorted by 'global_face_id'.
   Array<Pair<int,int>> shared_faces;

   // Add "neighbor" groups defined by faces
   // Construct 'shared_faces'.
   if (dim >= 3)
   {
      std::unordered_set<int> face_set;
      // Construct 'face_set'
      const Table &elem_to_face = mesh.ElementToFaceTable();
      for (int loc_elem_id = 0; loc_elem_id < num_elems; loc_elem_id++)
      {
         const int glob_elem_id = elem_list[loc_elem_id];
         const int nfaces = elem_to_face.RowSize(glob_elem_id);
         const int *faces = elem_to_face.GetRow(glob_elem_id);
         face_set.insert(faces, faces + nfaces);
      }
      // Construct 'shared_faces'; add "neighbor" groups defined by faces.
      IntegerSet group;
      for (int glob_face_id : face_set)
      {
         int el[2];
         mesh.GetFaceElements(glob_face_id, &el[0], &el[1]);
         if (el[1] < 0) { continue; }
         el[0] = partitioning[el[0]];
         el[1] = partitioning[el[1]];
         MFEM_ASSERT(el[0] == part_id || el[1] == part_id, "internal error");
         if (el[0] != part_id || el[1] != part_id)
         {
            group.Recreate(2, el);
            const int group_id = groups.Insert(group);
            shared_faces.Append(Pair<int,int>(glob_face_id, group_id));
         }
      }
      shared_faces.Sort(); // sort the shared faces by 'glob_face_id'
   }

   // 'shared_edges' : shared edge id -> (global_edge_id, group_id)
   // Note: 'shared_edges' will be sorted by 'global_edge_id'.
   Array<Pair<int,int>> shared_edges;

   // Add "neighbor" groups defined by edges.
   // Construct 'shared_edges'.
   if (dim >= 2)
   {
      std::unordered_set<int> edge_set;
      // Construct 'edge_set'
      const Table &elem_to_edge = mesh.ElementToEdgeTable();
      for (int loc_elem_id = 0; loc_elem_id < num_elems; loc_elem_id++)
      {
         const int glob_elem_id = elem_list[loc_elem_id];
         const int nedges = elem_to_edge.RowSize(glob_elem_id);
         const int *edges = elem_to_edge.GetRow(glob_elem_id);
         edge_set.insert(edges, edges + nedges);
      }
      // Construct 'shared_edges'; add "neighbor" groups defined by edges.
      IntegerSet group;
      for (int glob_edge_id : edge_set)
      {
         const int nelem = edge_to_element.RowSize(glob_edge_id);
         const int *elem = edge_to_element.GetRow(glob_edge_id);
         Array<int> &gr = group; // reference to the 'group' internal Array
         gr.SetSize(nelem);
         for (int j = 0; j < nelem; j++)
         {
            gr[j] = partitioning[elem[j]];
         }
         gr.Sort();
         gr.Unique();
         MFEM_ASSERT(gr.FindSorted(part_id) >= 0, "internal error");
         if (group.Size() > 1)
         {
            const int group_id = groups.Insert(group);
            shared_edges.Append(Pair<int,int>(glob_edge_id, group_id));
         }
      }
      shared_edges.Sort(); // sort the shared edges by 'glob_edge_id'
   }

   // 'shared_verts' : shared vertex id -> (global_vertex_id, group_id)
   // Note: 'shared_verts' will be sorted by 'global_vertex_id'.
   Array<Pair<int,int>> shared_verts;

   // Add "neighbor" groups defined by vertices.
   // Construct 'shared_verts'.
   {
      IntegerSet group;
      for (int i = 0; i < vertex_loc_to_glob.Size(); i++)
      {
         // 'vertex_to_element' maps global vertex ids to global element ids
         const int glob_vertex_id = vertex_loc_to_glob[i];
         const int nelem = vertex_to_element.RowSize(glob_vertex_id);
         const int *elem = vertex_to_element.GetRow(glob_vertex_id);
         Array<int> &gr = group; // reference to the 'group' internal Array
         gr.SetSize(nelem);
         for (int j = 0; j < nelem; j++)
         {
            gr[j] = partitioning[elem[j]];
         }
         gr.Sort();
         gr.Unique();
         MFEM_ASSERT(gr.FindSorted(part_id) >= 0, "internal error");
         if (group.Size() > 1)
         {
            const int group_id = groups.Insert(group);
            shared_verts.Append(Pair<int,int>(glob_vertex_id, group_id));
         }
      }
   }

   // Done constructing the "neighbor" groups in 'groups'.
   const int num_groups = groups.Size();

   // Define 'mesh_part.my_groups'
   groups.AsTable(mesh_part.my_groups);

   // Construct 'mesh_part.group_shared_entity_to_vertex[Geometry::POINT]'
   Table &group__shared_vertex_to_vertex =
      mesh_part.group_shared_entity_to_vertex[Geometry::POINT];
   group__shared_vertex_to_vertex.MakeI(num_groups);
   for (int sv = 0; sv < shared_verts.Size(); sv++)
   {
      const int group_id = shared_verts[sv].two;
      group__shared_vertex_to_vertex.AddAColumnInRow(group_id);
   }
   group__shared_vertex_to_vertex.MakeJ();
   for (int sv = 0; sv < shared_verts.Size(); sv++)
   {
      const int glob_vertex_id = shared_verts[sv].one;
      const int group_id       = shared_verts[sv].two;
      const int loc_vertex_id = vertex_loc_to_glob.FindSorted(glob_vertex_id);
      MFEM_ASSERT(loc_vertex_id >= 0, "internal error");
      group__shared_vertex_to_vertex.AddConnection(group_id, loc_vertex_id);
   }
   group__shared_vertex_to_vertex.ShiftUpI();

   // Construct 'mesh_part.group_shared_entity_to_vertex[Geometry::SEGMENT]'
   if (dim >= 2)
   {
      Table &group__shared_edge_to_vertex =
         mesh_part.group_shared_entity_to_vertex[Geometry::SEGMENT];
      group__shared_edge_to_vertex.MakeI(num_groups);
      for (int se = 0; se < shared_edges.Size(); se++)
      {
         const int group_id = shared_edges[se].two;
         group__shared_edge_to_vertex.AddColumnsInRow(group_id, 2);
      }
      group__shared_edge_to_vertex.MakeJ();
      const Table &edge_to_vertex = *mesh.GetEdgeVertexTable();
      for (int se = 0; se < shared_edges.Size(); se++)
      {
         const int glob_edge_id = shared_edges[se].one;
         const int group_id     = shared_edges[se].two;
         const int *v = edge_to_vertex.GetRow(glob_edge_id);
         for (int i = 0; i < 2; i++)
         {
            const int loc_vertex_id = vertex_loc_to_glob.FindSorted(v[i]);
            MFEM_ASSERT(loc_vertex_id >= 0, "internal error");
            group__shared_edge_to_vertex.AddConnection(group_id, loc_vertex_id);
         }
      }
      group__shared_edge_to_vertex.ShiftUpI();
   }

   // Construct 'mesh_part.group_shared_entity_to_vertex[Geometry::TRIANGLE]'
   // and 'mesh_part.group_shared_entity_to_vertex[Geometry::SQUARE]'.
   if (dim >= 3)
   {
      Table &group__shared_tria_to_vertex =
         mesh_part.group_shared_entity_to_vertex[Geometry::TRIANGLE];
      Table &group__shared_quad_to_vertex =
         mesh_part.group_shared_entity_to_vertex[Geometry::SQUARE];
      Array<int> vertex_ids;
      group__shared_tria_to_vertex.MakeI(num_groups);
      group__shared_quad_to_vertex.MakeI(num_groups);
      for (int sf = 0; sf < shared_faces.Size(); sf++)
      {
         const int glob_face_id = shared_faces[sf].one;
         const int group_id     = shared_faces[sf].two;
         const int geom         = mesh.GetFaceGeometry(glob_face_id);
         mesh_part.group_shared_entity_to_vertex[geom].
         AddColumnsInRow(group_id, Geometry::NumVerts[geom]);
      }
      group__shared_tria_to_vertex.MakeJ();
      group__shared_quad_to_vertex.MakeJ();
      for (int sf = 0; sf < shared_faces.Size(); sf++)
      {
         const int glob_face_id = shared_faces[sf].one;
         const int group_id     = shared_faces[sf].two;
         const int geom         = mesh.GetFaceGeometry(glob_face_id);
         mesh.GetFaceVertices(glob_face_id, vertex_ids);
         // Rotate shared triangles that have an adjacent tetrahedron with a
         // nonzero refinement flag.
         // See also ParMesh::BuildSharedFaceElems.
         if (geom == Geometry::TRIANGLE)
         {
            int glob_el_id[2];
            mesh.GetFaceElements(glob_face_id, &glob_el_id[0], &glob_el_id[1]);
            int side = 0;
            const Element *el = mesh.GetElement(glob_el_id[0]);
            const Tetrahedron *tet = nullptr;
            if (el->GetGeometryType() == Geometry::TETRAHEDRON)
            {
               tet = static_cast<const Tetrahedron*>(el);
            }
            else
            {
               side = 1;
               el = mesh.GetElement(glob_el_id[1]);
               if (el->GetGeometryType() == Geometry::TETRAHEDRON)
               {
                  tet = static_cast<const Tetrahedron*>(el);
               }
            }
            if (tet && tet->GetRefinementFlag())
            {
               // mark the shared face for refinement by reorienting
               // it according to the refinement flag in the tetrahedron
               // to which this shared face belongs to.
               int info[2];
               mesh.GetFaceInfos(glob_face_id, &info[0], &info[1]);
               tet->GetMarkedFace(info[side]/64, &vertex_ids[0]);
            }
         }
         for (int i = 0; i < vertex_ids.Size(); i++)
         {
            const int glob_id = vertex_ids[i];
            const int loc_id = vertex_loc_to_glob.FindSorted(glob_id);
            MFEM_ASSERT(loc_id >= 0, "internal error");
            vertex_ids[i] = loc_id;
         }
         mesh_part.group_shared_entity_to_vertex[geom].
         AddConnections(group_id, vertex_ids, vertex_ids.Size());
      }
      group__shared_tria_to_vertex.ShiftUpI();
      group__shared_quad_to_vertex.ShiftUpI();
   }
}

std::unique_ptr<FiniteElementSpace>
MeshPartitioner::ExtractFESpace(MeshPart &mesh_part,
                                const FiniteElementSpace &global_fespace) const
{
   mesh_part.GetMesh(); // initialize 'mesh_part.mesh'
   // Note: the nodes of 'mesh_part.mesh' are not set by GetMesh() unless they
   // were already constructed, e.g. by ExtractPart().

   return std::unique_ptr<FiniteElementSpace>(
             new FiniteElementSpace(mesh_part.mesh.get(),
                                    global_fespace.FEColl(),
                                    global_fespace.GetVDim(),
                                    global_fespace.GetOrdering()));
}

std::unique_ptr<GridFunction>
MeshPartitioner::ExtractGridFunction(const MeshPart &mesh_part,
                                     const GridFunction &global_gf,
                                     FiniteElementSpace &local_fespace) const
{
   std::unique_ptr<GridFunction> local_gf(new GridFunction(&local_fespace));

   // Transfer data from 'global_gf' to 'local_gf'.
   Array<int> gvdofs, lvdofs;
   Vector loc_vals;
   const int part_id = mesh_part.my_part_id;
   const int num_elems = part_to_element.RowSize(part_id);
   const int *elem_list = part_to_element.GetRow(part_id); // sorted
   for (int loc_elem_id = 0; loc_elem_id < num_elems; loc_elem_id++)
   {
      const int glob_elem_id = elem_list[loc_elem_id];
      DofTransformation glob_dt, local_dt;
      global_gf.FESpace()->GetElementVDofs(glob_elem_id, gvdofs, glob_dt);
      global_gf.GetSubVector(gvdofs, loc_vals);
      glob_dt.InvTransformPrimal(loc_vals);
      local_fespace.GetElementVDofs(loc_elem_id, lvdofs, local_dt);
      local_dt.TransformPrimal(loc_vals);
      local_gf->SetSubVector(lvdofs, loc_vals);
   }
   return local_gf;
}


GeometricFactors::GeometricFactors(const Mesh *mesh, const IntegrationRule &ir,
                                   int flags, MemoryType d_mt)
{
   this->mesh = mesh;
   IntRule = &ir;
   computed_factors = flags;

   MFEM_ASSERT(mesh->GetNumGeometries(mesh->Dimension()) <= 1,
               "mixed meshes are not supported!");
   MFEM_ASSERT(mesh->GetNodes(), "meshes without nodes are not supported!");

   Compute(*mesh->GetNodes(), d_mt);
}

GeometricFactors::GeometricFactors(const GridFunction &nodes,
                                   const IntegrationRule &ir,
                                   int flags, MemoryType d_mt)
{
   this->mesh = nodes.FESpace()->GetMesh();
   IntRule = &ir;
   computed_factors = flags;

   Compute(nodes, d_mt);
}

void GeometricFactors::Compute(const GridFunction &nodes,
                               MemoryType d_mt)
{

   const FiniteElementSpace *fespace = nodes.FESpace();
   const FiniteElement *fe = fespace->GetTypicalFE();
   const int dim  = fe->GetDim();
   const int vdim = fespace->GetVDim();
   const int NE   = fespace->GetNE();
   const int ND   = fe->GetDof();
   const int NQ   = IntRule->GetNPoints();

   unsigned eval_flags = 0;
   MemoryType my_d_mt = (d_mt != MemoryType::DEFAULT) ? d_mt :
                        Device::GetDeviceMemoryType();
   if (computed_factors & GeometricFactors::COORDINATES)
   {
      X.SetSize(vdim*NQ*NE, my_d_mt); // NQ x SDIM x NE
      eval_flags |= QuadratureInterpolator::VALUES;
   }
   if (computed_factors & GeometricFactors::JACOBIANS)
   {
      J.SetSize(dim*vdim*NQ*NE, my_d_mt); // NQ x SDIM x DIM x NE
      eval_flags |= QuadratureInterpolator::DERIVATIVES;
   }
   if (computed_factors & GeometricFactors::DETERMINANTS)
   {
      detJ.SetSize(NQ*NE, my_d_mt); // NQ x NE
      eval_flags |= QuadratureInterpolator::DETERMINANTS;
   }

   const QuadratureInterpolator *qi = fespace->GetQuadratureInterpolator(*IntRule);
   // All X, J, and detJ use this layout:
   qi->SetOutputLayout(QVectorLayout::byNODES);

   const bool use_tensor_products = UsesTensorBasis(*fespace);

   qi->DisableTensorProducts(!use_tensor_products);
   const ElementDofOrdering e_ordering = use_tensor_products ?
                                         ElementDofOrdering::LEXICOGRAPHIC :
                                         ElementDofOrdering::NATIVE;
   const Operator *elem_restr = fespace->GetElementRestriction(e_ordering);

   if (elem_restr) // Always true as of 2021-04-27
   {
      Vector Enodes(vdim*ND*NE, my_d_mt);
      elem_restr->Mult(nodes, Enodes);
      qi->Mult(Enodes, eval_flags, X, J, detJ);
   }
   else
   {
      qi->Mult(nodes, eval_flags, X, J, detJ);
   }
}

FaceGeometricFactors::FaceGeometricFactors(const Mesh *mesh,
                                           const IntegrationRule &ir,
                                           int flags, FaceType type,
                                           MemoryType d_mt)
   : type(type)
{
   this->mesh = mesh;
   IntRule = &ir;
   computed_factors = flags;

   const GridFunction *nodes = mesh->GetNodes();
   const FiniteElementSpace *fespace = nodes->FESpace();
   const int vdim = fespace->GetVDim();
   const int NF   = fespace->GetNFbyType(type);
   const int NQ   = ir.GetNPoints();

   const FaceRestriction *face_restr = fespace->GetFaceRestriction(
                                          ElementDofOrdering::LEXICOGRAPHIC,
                                          type,
                                          L2FaceValues::SingleValued);


   MemoryType my_d_mt = (d_mt != MemoryType::DEFAULT) ? d_mt :
                        Device::GetDeviceMemoryType();

   Vector Fnodes(face_restr->Height(), my_d_mt);
   face_restr->Mult(*nodes, Fnodes);

   unsigned eval_flags = 0;

   if (flags & FaceGeometricFactors::COORDINATES)
   {
      X.SetSize(vdim*NQ*NF, my_d_mt);
      eval_flags |= FaceQuadratureInterpolator::VALUES;
   }
   if (flags & FaceGeometricFactors::JACOBIANS)
   {
      J.SetSize(vdim*(mesh->Dimension() - 1)*NQ*NF, my_d_mt);
      eval_flags |= FaceQuadratureInterpolator::DERIVATIVES;
   }
   if (flags & FaceGeometricFactors::DETERMINANTS)
   {
      detJ.SetSize(NQ*NF, my_d_mt);
      eval_flags |= FaceQuadratureInterpolator::DETERMINANTS;
   }
   if (flags & FaceGeometricFactors::NORMALS)
   {
      normal.SetSize(vdim*NQ*NF, my_d_mt);
      eval_flags |= FaceQuadratureInterpolator::NORMALS;
   }

   const FaceQuadratureInterpolator *qi =
      fespace->GetFaceQuadratureInterpolator(ir, type);
   // All face data vectors assume layout byNODES.
   qi->SetOutputLayout(QVectorLayout::byNODES);
   const bool use_tensor_products = UsesTensorBasis(*fespace);
   qi->DisableTensorProducts(!use_tensor_products);

   qi->Mult(Fnodes, eval_flags, X, J, detJ, normal);
}

NodeExtrudeCoefficient::NodeExtrudeCoefficient(const int dim, const int n_,
                                               const real_t s_)
   : VectorCoefficient(dim), n(n_), s(s_), tip(p, dim-1)
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


Mesh *Extrude1D(Mesh *mesh, const int ny, const real_t sy, const bool closed)
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
   real_t vc[2];
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      vc[0] = mesh->GetVertex(i)[0];
      for (int j = 0; j < nvy; j++)
      {
         vc[1] = sy * (real_t(j) / ny);
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

Mesh *Extrude2D(Mesh *mesh, const int nz, const real_t sz)
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
   real_t vc[3];
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      vc[0] = mesh->GetVertex(i)[0];
      vc[1] = mesh->GetVertex(i)[1];
      for (int j = 0; j < nvz; j++)
      {
         vc[2] = sz * (real_t(j) / nz);
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

Mesh PartitionMPI(int dim, int mpi_cnt, int elem_per_mpi, bool print,
                  int &par_ref, Array<int> &partitioning)
{
   MFEM_VERIFY(dim > 1, "Not implemented for 1D meshes.");

   auto factor = [&](int N)
   {
      for (int i = static_cast<int>(sqrt(N)); i > 0; i--)
      { if (N % i == 0) { return i; } }
      return 1;
   };

   par_ref = 0;
   const int ref_factor = (dim == 2) ? 4 : 8;

   // Elements per task before performing parallel refinements.
   // This will be used to form the serial mesh.
   int el0 = elem_per_mpi;
   while (el0 % ref_factor == 0)
   {
      el0 /= ref_factor;
      par_ref++;
   }

   // In the serial mesh we have:
   // The number of MPI blocks is mpi_cnt = mp_x.mpy_y.mpy_z.
   // The size of each MPI block is el0 = el0_x.el0_y.el0_z.
   int mpi_x, mpi_y, mpi_z;
   int el0_x, el0_y, el0_z;
   if (dim == 2)
   {
      mpi_x = factor(mpi_cnt);
      mpi_y = mpi_cnt / mpi_x;

      // Switch order for better balance.
      el0_y = factor(el0);
      el0_x = el0 / el0_y;
   }
   else
   {
      mpi_x = factor(mpi_cnt);
      mpi_y = factor(mpi_cnt / mpi_x);
      mpi_z = mpi_cnt / mpi_x / mpi_y;

      // Switch order for better balance.
      el0_z = factor(el0);
      el0_y = factor(el0 / el0_z);
      el0_x = el0 / el0_y / el0_z;
   }

   if (print && dim == 2)
   {
      int elem_par_x = mpi_x * el0_x * pow(2, par_ref),
          elem_par_y = mpi_y * el0_y * pow(2, par_ref);

      mfem::out << "--- Mesh generation: \n";
      mfem::out << "Par mesh:    " << elem_par_x << " x " << elem_par_y
                << " (" << elem_par_x * elem_par_y << " elements)\n"
                << "Elem / task: "
                << el0_x * pow(2, par_ref) << " x "
                << el0_y * pow(2, par_ref)
                << " (" << el0_x * pow(2, 2*par_ref) * el0_y << " elements)\n"
                << "MPI blocks:  " << mpi_x << " x " << mpi_y
                << " (" << mpi_x * mpi_y << " mpi tasks)\n" << "-\n"
                << "Serial mesh: "
                << mpi_x * el0_x << " x " << mpi_y * el0_y
                << " (" << mpi_x * el0_x * mpi_y * el0_y << " elements)\n"
                << "Elem / task: " << el0_x << " x " << el0_y << std::endl
                << "Par refine:  " << par_ref << std::endl;
      mfem::out << "--- \n";
   }

   if (print && dim == 3)
   {
      int elem_par_x = mpi_x * el0_x * pow(2, par_ref),
          elem_par_y = mpi_y * el0_y * pow(2, par_ref),
          elem_par_z = mpi_z * el0_z * pow(2, par_ref);

      mfem::out << "--- Mesh generation: \n";
      mfem::out << "Par mesh:    "
                << elem_par_x << " x " << elem_par_y << " x " << elem_par_z
                << " (" << elem_par_x*elem_par_y*elem_par_z << " elements)\n"
                << "Elem / task: "
                << el0_x * pow(2, par_ref) << " x "
                << el0_y * pow(2, par_ref) << " x "
                << el0_z * pow(2, par_ref)
                << " (" << el0_x*pow(2, 3*par_ref)*el0_y*el0_z << " elements)\n"
                << "MPI blocks:  " << mpi_x << " x " << mpi_y << " x " << mpi_z
                << " (" << mpi_x * mpi_y * mpi_z << " mpi tasks)\n" << "-\n"
                << "Serial mesh: "
                << mpi_x*el0_x << " x " << mpi_y*el0_y << " x " << mpi_z*el0_z
                << " (" << mpi_x*el0_x*mpi_y*el0_y*mpi_z*el0_z << " elements)\n"
                << "Elem / task: "
                << el0_x << " x " << el0_y << " x " << el0_z  << std::endl
                << "Par refine:  " << par_ref << std::endl;
      mfem::out << "--- \n";
   }

   Mesh mesh;
   int nxyz[3];
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(mpi_x * el0_x,
                                   mpi_y * el0_y, Element::QUADRILATERAL, true);
      nxyz[0] = mpi_x; nxyz[1] = mpi_y;
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(mpi_x * el0_x,
                                   mpi_y * el0_y,
                                   mpi_z * el0_z, Element::HEXAHEDRON, true);
      nxyz[0] = mpi_x; nxyz[1] = mpi_y; nxyz[2] = mpi_z;
   }

   const int NE = mesh.GetNE();
   partitioning.SetSize(NE);
   std::unique_ptr<int[]> p_raw(mesh.CartesianPartitioning(nxyz));
   std::copy(p_raw.get(), p_raw.get() + NE, partitioning.GetData());

   return mesh;
}

bool Mesh::Conforming() const
{
   if (NURBSext)
   {
      // NURBS meshes are always conforming (element-wise). NURBS patch
      // conformity is indicated by NURBSExtension::NonconformingPatches.
      return true;
   }
   else
   {
      return ncmesh == NULL;
   }
}

#ifdef MFEM_DEBUG
void Mesh::DebugDump(std::ostream &os) const
{
   // dump vertices and edges (NCMesh "nodes")
   os << NumOfVertices + NumOfEdges << "\n";
   for (int i = 0; i < NumOfVertices; i++)
   {
      const real_t *v = GetVertex(i);
      os << i << " " << v[0] << " " << v[1] << " " << v[2]
         << " 0 0 " << i << " -1 0\n";
   }

   Array<int> ev;
   for (int i = 0; i < NumOfEdges; i++)
   {
      GetEdgeVertices(i, ev);
      real_t mid[3] = {0, 0, 0};
      for (int j = 0; j < 2; j++)
      {
         for (int k = 0; k < spaceDim; k++)
         {
            mid[k] += GetVertex(ev[j])[k];
         }
      }
      os << NumOfVertices+i << " "
         << mid[0]/2 << " " << mid[1]/2 << " " << mid[2]/2 << " "
         << ev[0] << " " << ev[1] << " -1 " << i << " 0\n";
   }

   // dump elements
   os << NumOfElements << "\n";
   for (int i = 0; i < NumOfElements; i++)
   {
      const Element* e = elements[i];
      os << e->GetNVertices() << " ";
      for (int j = 0; j < e->GetNVertices(); j++)
      {
         os << e->GetVertices()[j] << " ";
      }
      os << e->GetAttribute() << " 0 " << i << "\n";
   }

   // dump faces
   os << "0\n";
}
#endif

}
